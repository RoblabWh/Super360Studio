#!/usr/bin/env python3
"""Schneidet die Methoden eines Zielmoduls aus ui/main_window.py.

Welche Namen in welches Modul gehören, steht in ``aufteilung.json`` (nach
Namen, nicht nach Zeilen). Geschnitten wird nach den ast-Zeilen jeder Methode,
samt Dekoratoren und der Kommentarzeilen direkt davor (Abschnittsköpfe wie
``# ===== Mesh``); Klassenattribute und die Modulnamen, deren Ziel dieses
Modul ist, kommen auf dieselbe Weise mit. Nichts wird abgetippt: der Text
ist zeichengleich mit der Quelle, nur der Dateikopf ist ein Platzhalter mit
der Liste der Namen, die die Datei aus dem alten Modulkopf braucht.

Nach dem Schneiden wird das Ergebnis geparst und jede Methode per ``ast.dump``
gegen die Quelle verglichen; jede Abweichung bricht ab.

Aufrufe (alle zusätzlich mit ``--wurzel DIR``)::

    zerlege.py --liste
    zerlege.py --probe grundgeruest              # Mixin-Text nach stdout
    zerlege.py --schreibe grundgeruest [--ersetzen] [--ziel DATEI]

``--schreibe`` legt die Datei aus ``aufteilung.json`` an (Vorgabe
``ui/fenster/<modul>.py`` unter der Wurzel) und verweigert eine vorhandene
ohne ``--ersetzen``.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys

sys.dont_write_bytecode = True

import basis  # noqa: E402 — liegt im selben Ordner
import vergleiche_methoden as vm  # noqa: E402


def _kommentar_davor(zeilen: list, start: int, einzug: int) -> int:
    """Erste Zeile des Blocks aus Kommentaren (und Leerzeilen) direkt vor ``start``.

    Nur Kommentare auf der Ebene des Knotens zählen (``einzug`` Leerzeichen);
    führende Leerzeilen gehören nicht dazu.
    """
    j = start - 1
    while j >= 1:
        z = zeilen[j - 1]
        s = z.strip()
        if not s:
            j -= 1
            continue
        if s.startswith("#") and len(z) - len(z.lstrip(" ")) == einzug:
            j -= 1
            continue
        break
    erste = j + 1
    while erste < start and not zeilen[erste - 1].strip():
        erste += 1
    return erste


def _block(zeilen: list, knoten: ast.AST, einzug: int) -> tuple:
    """(erste Zeile samt Kommentaren, letzte Zeile, Text)."""
    start = vm.anfang(knoten)
    von = _kommentar_davor(zeilen, start, einzug)
    return von, knoten.end_lineno, vm.ausschnitt(zeilen, von, knoten.end_lineno)


def _modulkopf_namen(baum: ast.Module) -> dict:
    """Namen, die der alte Modulkopf bindet -> Herkunft (für den Platzhalter)."""
    herkunft: dict = {}

    def importe(knoten_liste):
        for k in knoten_liste:
            if isinstance(k, ast.Import):
                for a in k.names:
                    herkunft[(a.asname or a.name).split(".")[0]] = \
                        ("import", f"{a.name} as {a.asname}" if a.asname else a.name)
            elif isinstance(k, ast.ImportFrom):
                for a in k.names:
                    herkunft[a.asname or a.name] = (
                        f"from {'.' * k.level}{k.module or ''}",
                        f"{a.name} as {a.asname}" if a.asname else a.name)
            elif isinstance(k, ast.Try):
                importe(k.body)

    importe(baum.body)
    return herkunft


def _benutzte_namen(text_teile: list) -> set:
    namen = set()
    for t in text_teile:
        knoten = ast.parse("class _K:\n" + t if t.startswith(" ") else t)
        namen |= {k.id for k in ast.walk(knoten) if isinstance(k, ast.Name)}
    return namen


def zerlege(wurzel: str, modul: str, auft: dict) -> tuple:
    """(Dateitext, Zusammenfassung) für ein Zielmodul."""
    if modul not in auft["module"] or modul == vm.BLEIBT:
        raise vm.Fehler(f"'{modul}' ist kein Mixin-Modul der Aufteilung "
                        f"({', '.join(m for m in auft['module'] if m != vm.BLEIBT)}).")
    ziel = auft["module"][modul]
    quelle = os.path.join(wurzel, vm.QUELLE)
    text, baum, zeilen = vm.lies(quelle)
    klasse = next((k for k in baum.body if isinstance(k, ast.ClassDef)
                   and k.name == vm.KLASSE), None)
    if klasse is None:
        raise vm.Fehler(f"{vm.QUELLE}: keine Klasse {vm.KLASSE}.")
    inhalt = vm.klassen_inhalt(klasse)
    oben = vm.oberste(baum)

    namen = [n for n, m in auft["methoden"].items() if m == modul]
    namen += [n for n, m in auft["klassenattribute"].items() if m == modul]
    fehlt = [n for n in namen if n not in inhalt]
    if fehlt:
        raise vm.Fehler(f"{vm.QUELLE}: {', '.join(fehlt)} nicht mehr in {vm.KLASSE} "
                        f"(schon umgeschaltet?).")
    glieder = sorted(((inhalt[n], n) for n in namen), key=lambda x: x[0].lineno)

    modulnamen = [n for n, z in auft["modulnamen"].items() if z["ziel"] == ziel["datei"]]
    fehlt = [n for n in modulnamen if n not in oben]
    if fehlt:
        raise vm.Fehler(f"{vm.QUELLE}: Modulnamen {', '.join(fehlt)} fehlen.")
    konstanten = sorted(((oben[n], n) for n in modulnamen), key=lambda x: x[0].lineno)

    k_teile = [_block(zeilen, k, 0) for k, _ in konstanten]
    m_teile = [_block(zeilen, k, 4) for k, _ in glieder]
    bereiche = [f"{a}-{b}" for a, b, _ in sorted(k_teile + m_teile)]

    # Platzhalter: was die Datei aus dem alten Modulkopf braucht
    herkunft = _modulkopf_namen(baum)
    benutzt = _benutzte_namen([t for _, _, t in k_teile + m_teile])
    eigene = set(modulnamen)
    gruppen: dict = {}
    for n in sorted(benutzt - eigene, key=str.lower):
        if n in herkunft:
            gruppen.setdefault(herkunft[n][0], []).append(herkunft[n][1])
        elif n in auft["modulnamen"]:
            z = auft["modulnamen"][n]
            neu = f" (dort {z['name']})" if z["name"] != n else ""
            gruppen.setdefault(f"jetzt in {z['ziel']}", []).append(n + neu)
    kopf_liste = []
    for quelle_ in sorted(gruppen, key=lambda q: (q.startswith("jetzt"), q)):
        kopf_liste.append(f"#   {quelle_}: {', '.join(gruppen[quelle_])}")

    titel = ziel["titel"]
    teile = [
        f'"""{titel} (Mixin des Hauptfensters).\n'
        f"\n"
        f"PLATZHALTER: hier die Attribute nennen, die diese Datei schreibt.\n"
        f'"""\n'
        f"from __future__ import annotations\n"
        f"\n"
        f"# PLATZHALTER Importe. Aus dem alten Modulkopf von {vm.QUELLE} benutzt:\n"
        + ("\n".join(kopf_liste) + "\n" if kopf_liste else "#   (nichts)\n")
    ]
    vorige = None
    for von, bis, t in k_teile:
        # im Original direkt aufeinander folgende Zeilen bleiben beisammen
        teile.append(("" if vorige is not None and von == vorige + 1 else "\n") + t)
        vorige = bis
    teile.append(f"\n\nclass {ziel['klasse']}:\n")
    for i, (_, _, t) in enumerate(m_teile):
        teile.append(("\n" if i else "") + t)
    neu_text = "".join(teile)

    # Gegenprobe: jede Methode, jedes Attribut, jeder Modulname AST-gleich
    neu_baum = ast.parse(neu_text)
    neu_oben = vm.oberste(neu_baum)
    neu_inhalt = vm.klassen_inhalt(neu_oben[ziel["klasse"]])
    abw = [n for k, n in glieder if ast.dump(k) != ast.dump(neu_inhalt.get(n, ast.Pass()))]
    abw += [n for k, n in konstanten if ast.dump(k) != ast.dump(neu_oben.get(n, ast.Pass()))]
    zusaetzlich = set(neu_inhalt) - set(namen)
    if abw or zusaetzlich:
        raise vm.Fehler(f"Gegenprobe gescheitert: abweichend {abw}, zusätzlich "
                        f"{sorted(zusaetzlich)}")
    n_m = sum(1 for _, n in glieder if n in auft["methoden"])
    info = (f"{modul}: {n_m} Methoden, {len(glieder) - n_m} Klassenattribute, "
            f"{len(konstanten)} Modulnamen aus {vm.QUELLE} {', '.join(bereiche)}; "
            f"Gegenprobe AST-gleich")
    return neu_text, info


def main(argv=None) -> int:
    parser = basis.argumente(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    gruppe = parser.add_mutually_exclusive_group(required=True)
    gruppe.add_argument("--liste", action="store_true")
    gruppe.add_argument("--probe", metavar="MODUL")
    gruppe.add_argument("--schreibe", metavar="MODUL")
    parser.add_argument("--ziel", metavar="DATEI",
                        help="--schreibe: andere Zieldatei (relativ zur Wurzel)")
    parser.add_argument("--ersetzen", action="store_true")
    args = parser.parse_args(argv)
    wurzel = os.path.abspath(args.wurzel or basis.ARBEITSBAUM)
    try:
        auft = vm.aufteilung_laden()
        if args.liste:
            for m, z in auft["module"].items():
                n = sum(1 for x in auft["methoden"].values() if x == m)
                b = ", ".join(f"{a}-{e}" for a, e in z["bereiche"])
                print(f"{m:18s} {z['klasse']:22s} {n:3d} Methoden  {z['datei']}  ({b})")
            return 0
        modul = args.probe or args.schreibe
        text, info = zerlege(wurzel, modul, auft)
        if args.probe:
            sys.stdout.write(text)
            print(info, file=sys.stderr)
            return 0
        voll, rel = vm.datei_pfad(wurzel, args.ziel or auft["module"][modul]["datei"])
        if os.path.exists(voll) and not args.ersetzen:
            print(f"{rel} gibt es schon (--ersetzen zum Überschreiben).", file=sys.stderr)
            return 2
        os.makedirs(os.path.dirname(voll), exist_ok=True)
        with open(voll, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"geschrieben: {rel}")
        print(info)
        return 0
    except vm.Fehler as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

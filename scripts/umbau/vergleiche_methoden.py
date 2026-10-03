#!/usr/bin/env python3
"""Methodenvergleich für die Zerlegung von ui/main_window.py.

Grundlage ist ``vorher/methoden.json``: je Methode von ``MainWindow``, je
Klassenattribut und je Modulname von ``ui/main_window.py`` der Quelltext,
sein SHA-256 und ``ast.dump`` (ohne Zeilennummern), aufgenommen aus dem Tag
``vor-umbau``. ``aufteilung.json`` sagt, wohin jeder Name geht.

"gleich" heißt AST-gleich: Kommentare und Zeilenumbrüche zählen nicht,
Dekoratoren, Docstrings und jede Konstante schon. Ob der Quelltext samt
Kommentaren gleich blieb, steht zusätzlich als "textgleich" daneben.
Umbenennungen gelten nur dort, wo ``aufteilung.json`` sie für die Zieldatei
benennt (``umbenennungen``); ein Schlüssel ``modul.`` steht für das
Weglassen dieses Modulpräfixes.

Aufrufe (alle zusätzlich mit ``--wurzel DIR``)::

    vergleiche_methoden.py --selbsttest
    vergleiche_methoden.py --schreibe [--ersetzen]   # vorher/methoden.json
    vergleiche_methoden.py --modul ui/fenster/splat.py [--neu-erlaubt]
    vergleiche_methoden.py --klasse Worker ui/jobs.py
    vergleiche_methoden.py --modulnamen core/ebenen.py
    vergleiche_methoden.py --gesamt [--entfallen name,…]
    vergleiche_methoden.py --fenster [--entfallen name,…]

``--modul``, ``--klasse`` und ``--modulnamen`` lassen sich wiederholen und
mischen. Relative Dateipfade gelten ab der Wurzel. ``--fenster`` importiert
``ui.main_window`` und folgt der echten MRO von ``MainWindow``; alle anderen
Modi lesen nur Quelltext. Exit 0: alles gleich und vollständig, 1: Abweichung,
2: Aufruf- oder Lesefehler.
"""
from __future__ import annotations

import argparse
import ast
import collections
import difflib
import glob
import hashlib
import json
import os
import subprocess
import sys
import textwrap

# Kein __pycache__ in den geprüften Bäumen hinterlassen.
sys.dont_write_bytecode = True

import basis  # noqa: E402 — liegt im selben Ordner

QUELLE = "ui/main_window.py"
KLASSE = "MainWindow"
AUFTEILUNG = os.path.join(basis.WERKZEUG_ORDNER, "aufteilung.json")
METHODEN_JSON = os.path.join(basis.VORHER_ORDNER, "methoden.json")
PYTHON = "%d.%d" % sys.version_info[:2]
#: Modulschlüssel in aufteilung.json für das, was in MainWindow bleibt
BLEIBT = "main_window"


class Fehler(Exception):
    """Aufruf- oder Lesefehler (Exit 2)."""


# ------------------------------------------------------------------ Erfassen

def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def anfang(knoten: ast.AST) -> int:
    """Erste Zeile eines Knotens samt Dekoratoren."""
    return min([knoten.lineno] + [d.lineno for d in getattr(knoten, "decorator_list", [])])


def definierte_namen(knoten: ast.AST) -> list:
    """Namen, die eine Anweisung auf ihrer Ebene bindet (def, class, Zuweisung)."""
    if isinstance(knoten, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [knoten.name]
    if isinstance(knoten, ast.Assign):
        return [z.id for z in knoten.targets if isinstance(z, ast.Name)]
    if isinstance(knoten, ast.AnnAssign) and isinstance(knoten.target, ast.Name):
        return [knoten.target.id]
    return []


def ist_funktion(knoten: ast.AST) -> bool:
    return isinstance(knoten, (ast.FunctionDef, ast.AsyncFunctionDef))


def _art(knoten: ast.AST, in_klasse: bool) -> str:
    if ist_funktion(knoten):
        return "methode" if in_klasse else "funktion"
    if isinstance(knoten, ast.ClassDef):
        return "klasse"
    return "klassenattribut" if in_klasse else "konstante"


def ausschnitt(zeilen: list, von: int, bis: int) -> str:
    return "\n".join(zeilen[von - 1:bis]) + "\n"


def klassen_kopf(klasse: ast.ClassDef) -> str:
    """ast.dump der Klassenzeile (Name, Basen, Dekoratoren) ohne Rumpf."""
    return ast.dump(ast.ClassDef(name=klasse.name, bases=klasse.bases,
                                 keywords=klasse.keywords, body=[],
                                 decorator_list=klasse.decorator_list))


def klassen_glieder(klasse: ast.ClassDef) -> dict:
    """Rumpf einer Klasse nach Schlüsseln: Name, '__doc__' oder '#<i>'."""
    glieder: dict = {}
    for i, k in enumerate(klasse.body):
        namen = definierte_namen(k)
        if namen:
            schluessel = namen[0]
        elif (i == 0 and isinstance(k, ast.Expr) and isinstance(k.value, ast.Constant)
              and isinstance(k.value.value, str)):
            schluessel = "__doc__"
        else:
            schluessel = f"#{i}"
        if schluessel in glieder:
            schluessel = f"{schluessel}#{i}"
        glieder[schluessel] = k
    return glieder


def eintrag(knoten: ast.AST, zeilen: list, in_klasse: bool) -> dict:
    von, bis = anfang(knoten), knoten.end_lineno
    text = ausschnitt(zeilen, von, bis)
    e = {"art": _art(knoten, in_klasse), "zeilen": [von, bis],
         "sha256": sha(text), "dump": ast.dump(knoten), "quelltext": text}
    if ist_funktion(knoten) and knoten.decorator_list:
        e["dekoratoren"] = [ast.unparse(d) for d in knoten.decorator_list]
    if isinstance(knoten, ast.ClassDef):
        e["kopf"] = klassen_kopf(knoten)
        e["glieder"] = {s: ast.dump(g) for s, g in klassen_glieder(knoten).items()}
    return e


def erfasse(text: str) -> dict:
    """Methoden, Klassenattribute und Modulnamen eines main_window.py-Texts."""
    baum = ast.parse(text)
    zeilen = text.splitlines()
    erg = {"quelle": QUELLE, "stand": basis.VORHER_TAG, "python": PYTHON,
           "sha256": sha(text), "basen": [], "methoden": {}, "klassenattribute": {},
           "modulnamen": {}, "doppelt": []}
    for k in baum.body:
        if isinstance(k, ast.ClassDef) and k.name == KLASSE:
            erg["basen"] = [ast.unparse(b) for b in k.bases]
            for g in k.body:
                for name in definierte_namen(g):
                    ziel = erg["methoden"] if ist_funktion(g) else erg["klassenattribute"]
                    if name in erg["methoden"] or name in erg["klassenattribute"]:
                        erg["doppelt"].append(name)
                    ziel[name] = eintrag(g, zeilen, True)
            continue
        for name in definierte_namen(k):
            if name in erg["modulnamen"]:
                erg["doppelt"].append(name)
            erg["modulnamen"][name] = eintrag(k, zeilen, False)
    return erg


def git_text(tag: str = basis.VORHER_TAG, pfad: str = QUELLE) -> str:
    try:
        return subprocess.run(["git", "-C", basis.ARBEITSBAUM, "show", f"{tag}:{pfad}"],
                              check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Fehler(f"git show {tag}:{pfad} geht nicht: {exc}") from exc


def vorher_laden() -> dict:
    """vorher/methoden.json; unter einer anderen Python-Version neu aus dem Tag.

    ast.dump ist nur innerhalb derselben Python-Version vergleichbar.
    """
    try:
        with open(METHODEN_JSON, encoding="utf-8") as fh:
            vorher = json.load(fh)
    except (OSError, ValueError) as exc:
        raise Fehler(f"{os.path.relpath(METHODEN_JSON, basis.ARBEITSBAUM)} nicht lesbar: "
                     f"{exc} (erst --schreibe)") from exc
    if vorher.get("python") != PYTHON:
        print(f"Hinweis: methoden.json stammt aus Python {vorher.get('python')}, hier "
              f"läuft {PYTHON}; der Vorher-Stand wird aus dem Tag {basis.VORHER_TAG} "
              f"neu erfasst.")
        neu = erfasse(git_text())
        if neu["sha256"] != vorher.get("sha256"):
            raise Fehler("Der Tag liefert einen anderen main_window.py-Stand als "
                         "methoden.json.")
        vorher = neu
    return vorher


def _ohne_doppelte(paare):
    d: dict = {}
    for schluessel, wert in paare:
        if schluessel in d:
            raise Fehler(f"aufteilung.json: Schlüssel '{schluessel}' doppelt.")
        d[schluessel] = wert
    return d


def aufteilung_laden() -> dict:
    try:
        with open(AUFTEILUNG, encoding="utf-8") as fh:
            return json.load(fh, object_pairs_hook=_ohne_doppelte)
    except (OSError, ValueError) as exc:
        raise Fehler(f"aufteilung.json nicht lesbar: {exc}") from exc


def methode_an_zeile(vorher: dict, zeile: int) -> str | None:
    """Name des Methoden- oder Modulnamens, in dessen Zeilen ``zeile`` liegt."""
    for gruppe in ("methoden", "klassenattribute", "modulnamen"):
        for name, e in vorher[gruppe].items():
            if e["zeilen"][0] <= zeile <= e["zeilen"][1]:
                return name
    return None


# ----------------------------------------------------------------- Vergleich

class _Umbenenner(ast.NodeTransformer):
    """Wendet die benannten Umbenennungen einer Zieldatei auf den alten Stand an."""

    def __init__(self, karte: dict):
        self.namen = {a: n for a, n in karte.items() if not a.endswith(".")}
        self.praefixe = {a[:-1]: n for a, n in karte.items() if a.endswith(".")}

    def visit_Name(self, knoten):
        knoten.id = self.namen.get(knoten.id, knoten.id)
        return knoten

    def visit_Attribute(self, knoten):
        if isinstance(knoten.value, ast.Name) and knoten.value.id in self.praefixe:
            neu = self.praefixe[knoten.value.id]
            if not neu:
                return ast.copy_location(ast.Name(id=knoten.attr, ctx=knoten.ctx), knoten)
            knoten.value.id = neu
            return knoten
        self.generic_visit(knoten)
        return knoten

    def _def(self, knoten):
        knoten.name = self.namen.get(knoten.name, knoten.name)
        self.generic_visit(knoten)
        return knoten

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _def


def alter_knoten(e: dict) -> ast.AST:
    """Den gespeicherten Quelltext wieder zum Knoten machen (Methoden eingerückt)."""
    text = e["quelltext"]
    einzug = len(text) - len(text.lstrip(" "))
    if einzug == 0:
        return ast.parse(text).body[0]
    if einzug != 4:
        text = textwrap.indent(textwrap.dedent(text), "    ")
    return ast.parse("class _K:\n" + text).body[0].body[0]


def _ohne_docstrings(knoten: ast.AST) -> str:
    knoten = ast.parse(ast.unparse(knoten))
    for k in ast.walk(knoten):
        if isinstance(k, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (k.body and isinstance(k.body[0], ast.Expr)
                    and isinstance(k.body[0].value, ast.Constant)
                    and isinstance(k.body[0].value.value, str)):
                k.body = k.body[1:] or [ast.Pass()]
    return ast.dump(knoten)


def normiert(e: dict, karte: dict | None = None, statisch_weg: bool = False) -> ast.AST:
    """Alter Knoten mit Umbenennungen; ``statisch_weg`` streicht @staticmethod
    (Methode wird Modulfunktion)."""
    alt = alter_knoten(e)
    if karte:
        alt = _Umbenenner(karte).visit(alt)
    if statisch_weg and ist_funktion(alt):
        alt.decorator_list = [d for d in alt.decorator_list
                              if not (isinstance(d, ast.Name) and d.id == "staticmethod")]
    return alt


def vergleiche_knoten(e: dict, neu: ast.AST, zeilen: list, karte: dict | None = None,
                      statisch_weg: bool = False) -> dict:
    """{'gleich', 'textgleich', 'hinweis', 'diff'} für einen alten Eintrag."""
    if karte or statisch_weg:
        alt = normiert(e, karte, statisch_weg)
        gleich = ast.dump(alt) == ast.dump(neu)
    else:
        alt = None
        gleich = e["dump"] == ast.dump(neu)
    neu_text = ausschnitt(zeilen, anfang(neu), neu.end_lineno)
    erg = {"gleich": gleich, "textgleich": sha(neu_text) == e["sha256"], "hinweis": "",
           "diff": []}
    if not gleich:
        alt = alt if alt is not None else alter_knoten(e)
        if _ohne_docstrings(alt) == _ohne_docstrings(neu):
            erg["hinweis"] = "nur im Docstring abweichend"
        erg["diff"] = list(difflib.unified_diff(
            e["quelltext"].splitlines(), neu_text.splitlines(), "vorher", "jetzt",
            lineterm="", n=1))[:40]
    return erg


def vergleiche_klasse(e: dict, neu: ast.ClassDef, zeilen: list, karte: dict | None = None,
                      entfallen: tuple = ()) -> dict:
    """Klassenkopf und jedes Glied einzeln; benannt entfallene Glieder sind erlaubt."""
    alt = normiert(e, karte)
    alt_glieder = klassen_glieder(alt)
    neu_glieder = klassen_glieder(neu)
    abw, fehlt, weg = [], [], []
    if klassen_kopf(alt) != klassen_kopf(neu):
        abw.append("(Klassenkopf)")
    for s, g in alt_glieder.items():
        if s not in neu_glieder:
            (weg if s in entfallen else fehlt).append(s)
        elif ast.dump(g) != ast.dump(neu_glieder[s]):
            abw.append(s)
    zusaetzlich = [s for s in neu_glieder if s not in alt_glieder]
    gleich = not (abw or fehlt or zusaetzlich)
    neu_text = ausschnitt(zeilen, anfang(neu), neu.end_lineno)
    return {"gleich": gleich, "textgleich": sha(neu_text) == e["sha256"],
            "abweichend": abw, "fehlt": fehlt, "entfallen": weg,
            "zusaetzlich": zusaetzlich}


# ------------------------------------------------------------- Dateien lesen

def datei_pfad(wurzel: str, pfad: str) -> tuple:
    """(absoluter Pfad, Pfad relativ zur Wurzel)."""
    voll = pfad if os.path.isabs(pfad) else os.path.join(wurzel, pfad)
    voll = os.path.normpath(voll)
    return voll, os.path.relpath(voll, wurzel).replace(os.sep, "/")


def lies(voll: str) -> tuple:
    try:
        with open(voll, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise Fehler(f"{voll}: {exc}") from exc
    try:
        return text, ast.parse(text), text.splitlines()
    except SyntaxError as exc:
        raise Fehler(f"{voll}: Syntaxfehler {exc}") from exc


def oberste(baum: ast.Module) -> dict:
    """Namen der Modulebene -> Knoten (letzte Bindung gewinnt, wie zur Laufzeit)."""
    d: dict = {}
    for k in baum.body:
        for name in definierte_namen(k):
            d[name] = k
    return d


def klassen_inhalt(klasse: ast.ClassDef) -> dict:
    """Methoden und Klassenattribute einer Klasse -> Knoten."""
    d: dict = {}
    for g in klasse.body:
        for name in definierte_namen(g):
            d[name] = g
    return d


def ziel_namen(auft: dict, datei: str) -> list:
    """[(alter Name, neuer Name, ist_kopie)] aller Modulnamen und Kopien mit Ziel ``datei``."""
    karte = auft["umbenennungen"].get(datei, {})
    erg = []
    for alt, z in auft["modulnamen"].items():
        if z["ziel"] == datei:
            erg.append((alt, z.get("name", karte.get(alt, alt)), False))
    for alt, kopien in auft.get("kopien", {}).items():
        for kop in kopien:
            if kop["ziel"] == datei:
                erg.append((alt, kop["name"], True))
    return erg


def _alt_eintrag(vorher: dict, alt: str) -> dict | None:
    for gruppe in ("modulnamen", "methoden", "klassenattribute"):
        if alt in vorher[gruppe]:
            return vorher[gruppe][alt]
    return None


# --------------------------------------------------------------- Ausgaben

def _zeige(name: str, erg: dict) -> None:
    zusatz = f" ({erg['hinweis']})" if erg.get("hinweis") else ""
    print(f"  ABWEICHEND {name}{zusatz}")
    for z in erg.get("diff", []):
        print(f"      {z}")


def _textinfo(gleich: int, textgleich: int) -> str:
    return f" (davon textgleich samt Kommentaren: {textgleich})" if gleich else ""


# ------------------------------------------------------------- --modul

def pruefe_modul(wurzel: str, pfad: str, auft: dict, vorher: dict, neu_erlaubt: bool) -> bool:
    voll, rel = datei_pfad(wurzel, pfad)
    modul = next((m for m, z in auft["module"].items() if z["datei"] == rel), None)
    if modul is None:
        raise Fehler(f"--modul {rel}: kein Zielmodul in aufteilung.json.")
    _, baum, zeilen = lies(voll)
    kname = auft["module"][modul]["klasse"]
    klassen = {k.name: k for k in baum.body if isinstance(k, ast.ClassDef)}
    print(f"{rel} ({kname}):")
    if kname not in klassen:
        print(f"  Klasse {kname} fehlt.")
        return False
    inhalt = klassen_inhalt(klassen[kname])
    ok = True

    soll_m = [m for m, z in auft["methoden"].items() if z == modul]
    soll_a = [a for a, z in auft["klassenattribute"].items() if z == modul]
    gleich = textgleich = 0
    abw, fehlt = [], []
    for name in soll_m + soll_a:
        e = vorher["methoden"].get(name) or vorher["klassenattribute"][name]
        if name not in inhalt:
            fehlt.append(name)
            continue
        erg = vergleiche_knoten(e, inhalt[name], zeilen)
        if erg["gleich"]:
            gleich += 1
            textgleich += erg["textgleich"]
        else:
            abw.append(name)
            _zeige(name, erg)
    falsch = sorted(n for n in inhalt if n not in soll_m and n not in soll_a
                    and (n in auft["methoden"] or n in auft["klassenattribute"]))
    neu = sorted(n for n in inhalt if n not in auft["methoden"]
                 and n not in auft["klassenattribute"])
    for n in fehlt:
        print(f"  FEHLT {n}")
    for n in falsch:
        ziel = auft["methoden"].get(n) or auft["klassenattribute"].get(n)
        print(f"  FALSCH ZUGEORDNET {n} (gehört nach {ziel})")
    for n in neu:
        print(f"  ZUSÄTZLICH {n}{' (erlaubt)' if neu_erlaubt else ''}")
    if "__init__" in inhalt and modul != BLEIBT:
        print("  __init__ im Mixin: der Sitzungszustand gehört nach MainWindow.__init__.")
        ok = False
    print(f"  Methoden und Klassenattribute: {gleich} gleich{_textinfo(gleich, textgleich)}, "
          f"{len(abw)} abweichend, {len(fehlt)} fehlend, {len(falsch)} falsch zugeordnet, "
          f"{len(neu)} zusätzlich (von {len(soll_m)} Methoden, {len(soll_a)} Klassenattributen)")
    ok = ok and not (abw or fehlt or falsch or (neu and not neu_erlaubt))

    namen = ziel_namen(auft, rel)
    if namen:
        ok = _pruefe_namen(rel, baum, zeilen, namen, auft, vorher) and ok
    print("  gleich und vollständig" if ok else "  NICHT gleich oder unvollständig")
    return ok


# ---------------------------------------------------- --modulnamen, --klasse

def _pruefe_namen(rel: str, baum: ast.Module, zeilen: list, namen: list, auft: dict,
                  vorher: dict) -> bool:
    karte = auft["umbenennungen"].get(rel, {})
    oben = oberste(baum)
    gleich = textgleich = 0
    probleme = 0
    for alt, neu_name, kopie in namen:
        e = _alt_eintrag(vorher, alt)
        anzeige = alt if alt == neu_name else f"{alt} -> {neu_name}"
        if kopie:
            anzeige += " (Kopie)"
        if e is None:
            print(f"  UNBEKANNT {alt}: nicht im Vorher-Stand")
            probleme += 1
            continue
        k = oben.get(neu_name)
        if k is None:
            print(f"  FEHLT {anzeige}")
            probleme += 1
            continue
        if e["art"] == "klasse":
            entf = tuple(auft["modulnamen"].get(alt, {}).get("entfallen", ()))
            erg = vergleiche_klasse(e, k, zeilen, karte, entf)
            if erg["entfallen"]:
                print(f"  {anzeige}: entfallen wie benannt: {', '.join(erg['entfallen'])}")
            if not erg["gleich"]:
                print(f"  ABWEICHEND {anzeige}: abweichend {erg['abweichend']}, fehlend "
                      f"{erg['fehlt']}, zusätzlich {erg['zusaetzlich']}")
        else:
            erg = vergleiche_knoten(e, k, zeilen, karte, statisch_weg=e["art"] == "methode")
            if not erg["gleich"]:
                _zeige(anzeige, erg)
        if erg["gleich"]:
            gleich += 1
            textgleich += erg["textgleich"]
            if alt != neu_name:
                print(f"  gleich {anzeige}")
        else:
            probleme += 1
    print(f"  Modulnamen: {gleich} gleich{_textinfo(gleich, textgleich)}, "
          f"{probleme} abweichend oder fehlend (von {len(namen)})")
    return probleme == 0


def pruefe_modulnamen(wurzel: str, pfad: str, auft: dict, vorher: dict) -> bool:
    voll, rel = datei_pfad(wurzel, pfad)
    namen = ziel_namen(auft, rel)
    if not namen:
        raise Fehler(f"--modulnamen {rel}: aufteilung.json schickt keinen Modulnamen dorthin.")
    _, baum, zeilen = lies(voll)
    print(f"{rel}: {len(namen)} Modulnamen ({', '.join(a for a, _, _ in namen)})")
    ok = _pruefe_namen(rel, baum, zeilen, namen, auft, vorher)
    print("  gleich" if ok else "  NICHT gleich")
    return ok


def pruefe_klasse(wurzel: str, name: str, pfad: str, auft: dict, vorher: dict) -> bool:
    voll, rel = datei_pfad(wurzel, pfad)
    e = vorher["modulnamen"].get(name)
    if e is None or e["art"] != "klasse":
        raise Fehler(f"--klasse {name}: keine Klasse dieses Namens im Vorher-Stand.")
    z = auft["modulnamen"].get(name, {})
    karte = auft["umbenennungen"].get(rel, {}) if z.get("ziel") == rel else {}
    neu_name = karte.get(name, name)
    _, baum, zeilen = lies(voll)
    k = next((k for k in baum.body if isinstance(k, ast.ClassDef) and k.name == neu_name), None)
    if k is None:
        print(f"{rel}: Klasse {neu_name} fehlt.")
        return False
    erg = vergleiche_klasse(e, k, zeilen, karte, tuple(z.get("entfallen", ())))
    print(f"{rel}: Klasse {name}" + (f" als {neu_name}" if neu_name != name else ""))
    if erg["entfallen"]:
        print(f"  entfallen wie benannt: {', '.join(erg['entfallen'])}")
    for teil in ("abweichend", "fehlt", "zusaetzlich"):
        if erg[teil]:
            print(f"  {teil}: {', '.join(erg[teil])}")
    if erg["gleich"]:
        print("  gleich" + (" (textgleich samt Kommentaren)" if erg["textgleich"] else "")
              + (" bis auf die benannten Auslassungen" if erg["entfallen"] else ""))
    else:
        print("  NICHT gleich")
    return erg["gleich"]


# ---------------------------------------------------------------- --gesamt

def mixin_dateien(wurzel: str, auft: dict) -> list:
    rel = {z["datei"] for m, z in auft["module"].items() if m != BLEIBT}
    for p in glob.glob(os.path.join(wurzel, "ui", "fenster", "*.py")):
        if os.path.basename(p) != "__init__.py":
            rel.add(os.path.relpath(p, wurzel).replace(os.sep, "/"))
    return sorted(r for r in rel if os.path.isfile(os.path.join(wurzel, r)))


def gesamt_auswerten(texte: dict, auft: dict, vorher: dict, entfallen: set,
                     mixins: set | None = None) -> dict:
    """Kern von --gesamt auf ``{relpfad: quelltext}`` (der Selbsttest füttert ihn direkt).

    Klassen zählen nur aus ``ui/main_window.py`` (MainWindow) und den
    Mixin-Dateien ``mixins`` (Vorgabe: alle Dateien außer den Zielen der
    Modulnamen in core/ und ui/).
    """
    if mixins is None:
        mixins = {z["datei"] for m, z in auft["module"].items() if m != BLEIBT}
    orte = collections.defaultdict(list)       # name -> [(datei, klasse)]
    oben = {}                                    # datei -> Modulnamen
    mw_basen: list = []
    mixin_klassen = set()
    for rel, text in texte.items():
        baum = ast.parse(text)
        oben[rel] = oberste(baum)
        for k in baum.body:
            if not isinstance(k, ast.ClassDef) or (rel != QUELLE and rel not in mixins):
                continue
            if rel == QUELLE and k.name != KLASSE:
                continue
            if rel == QUELLE:
                mw_basen = [ast.unparse(b) for b in k.bases]
            else:
                mixin_klassen.add(k.name)
            for name in klassen_inhalt(k):
                orte[name].append((rel, k.name))
    umgeschaltet = any(b.split(".")[-1] in mixin_klassen for b in mw_basen)
    mw = (QUELLE, KLASSE)

    erg = {"umgeschaltet": umgeschaltet, "in_mw": 0, "in_mixins": 0, "zugeordnet": 0,
           "attr_zugeordnet": 0,
           "noch_in_mw": [], "doppelt": [], "fehlend": [], "falsch": [], "entfallen": [],
           "namen_am_ziel": [], "namen_in_mw": [], "namen_fehlend": [], "zusaetzlich": []}
    soll = dict(auft["methoden"])
    soll.update(auft["klassenattribute"])
    for name, modul in soll.items():
        ziel = (auft["module"][modul]["datei"], auft["module"][modul]["klasse"])
        da = orte.get(name, [])
        in_mw = mw in da
        mixins = [o for o in da if o != mw]
        if name in auft["methoden"]:
            erg["in_mw"] += in_mw
            erg["in_mixins"] += bool(mixins)
        if ziel in da:
            erg["zugeordnet" if name in auft["methoden"] else "attr_zugeordnet"] += 1
        elif name in entfallen:
            erg["entfallen"].append(name)
        elif not da:
            erg["fehlend"].append(name)
        falsch = [o for o in mixins if o != ziel]
        if falsch:
            erg["falsch"].append((name, falsch))
        if len(mixins) > 1:
            erg["doppelt"].append((name, mixins))
        elif modul != BLEIBT and in_mw and mixins:
            if umgeschaltet:
                erg["doppelt"].append((name, da))
            else:
                erg["noch_in_mw"].append(name)
    # neue Namen: auch sie nur einmal über alle Mixins (und nicht zusätzlich in MainWindow)
    for name, da in orte.items():
        if name in soll:
            continue
        mixins = [o for o in da if o != mw]
        if mixins:
            erg["zusaetzlich"].append(name)
        if len(mixins) > 1 or (umgeschaltet and mixins and mw in da):
            erg["doppelt"].append((name, da))
    # Modulnamen: am Ziel (unter dem neuen Namen) oder noch in main_window.py
    for alt, z in auft["modulnamen"].items():
        mw_hat = alt in oben.get(QUELLE, {})
        ziel_hat = z["name"] in oben.get(z["ziel"], {}) if z["ziel"] in texte else False
        if ziel_hat:
            erg["namen_am_ziel"].append(alt)
        if mw_hat:
            erg["namen_in_mw"].append(alt)
        if not (ziel_hat or mw_hat) and alt not in entfallen:
            erg["namen_fehlend"].append(alt)
    return erg


def pruefe_gesamt(wurzel: str, auft: dict, vorher: dict, entfallen: set) -> bool:
    dateien = [QUELLE] + mixin_dateien(wurzel, auft)
    ziele = {z["ziel"] for z in auft["modulnamen"].values()}
    ziele |= {k["ziel"] for ks in auft.get("kopien", {}).values() for k in ks}
    texte = {}
    for rel in sorted(set(dateien) | ziele):
        voll = os.path.join(wurzel, rel)
        if os.path.isfile(voll):
            texte[rel] = lies(voll)[0]
    if QUELLE not in texte:
        raise Fehler(f"{QUELLE} fehlt unter {wurzel}.")
    erg = gesamt_auswerten(texte, auft, vorher, entfallen, set(dateien) - {QUELLE})
    n = len(auft["methoden"])
    print(f"Wurzel {wurzel}; Mixin-Dateien: {len(dateien) - 1}; MainWindow erbt "
          f"{'von den Mixins' if erg['umgeschaltet'] else 'noch nicht von den Mixins'}.")
    print(f"Methoden: {n} im Vorher-Stand; {erg['in_mw']} in {QUELLE}, "
          f"{erg['in_mixins']} in Mixins; {erg['zugeordnet']} zugeordnet (am Ziel laut "
          f"aufteilung.json)")
    print(f"Klassenattribute: {erg['attr_zugeordnet']} von {len(auft['klassenattribute'])} "
          f"am Ziel")
    if erg["noch_in_mw"]:
        print(f"  noch zusätzlich in {QUELLE} (bis zum Umschalten erlaubt): "
              f"{len(erg['noch_in_mw'])}")
    for name, orte in erg["falsch"]:
        print(f"  FALSCH ZUGEORDNET {name}: {', '.join(f'{d}::{k}' for d, k in orte)}")
    for name, orte in erg["doppelt"]:
        print(f"  DOPPELT {name}: {', '.join(f'{d}::{k}' for d, k in orte)}")
    for name in erg["fehlend"]:
        print(f"  FEHLT {name}")
    if erg["entfallen"]:
        print(f"  entfallen wie benannt: {', '.join(erg['entfallen'])}")
    if erg["zusaetzlich"]:
        print(f"  neue Namen in Mixins: {', '.join(sorted(erg['zusaetzlich']))}")
    print(f"{len(erg['doppelt'])} doppelt, {len(erg['fehlend'])} fehlend, "
          f"{len(erg['falsch'])} falsch zugeordnet")
    m = len(auft["modulnamen"])
    print(f"Modulnamen: {m} erfasst; {len(erg['namen_am_ziel'])} am Ziel, "
          f"{len(erg['namen_in_mw'])} noch in {QUELLE}, {len(erg['namen_fehlend'])} fehlend"
          + ("; Modulnamen vollständig am Ziel" if len(erg["namen_am_ziel"]) == m else ""))
    for name in erg["namen_fehlend"]:
        print(f"  FEHLT Modulname {name} (Ziel {auft['modulnamen'][name]['ziel']})")
    ok = not (erg["doppelt"] or erg["fehlend"] or erg["falsch"] or erg["namen_fehlend"])
    print("gesamt: in Ordnung" if ok else "gesamt: NICHT in Ordnung")
    return ok


# ---------------------------------------------------------------- --fenster

def pruefe_fenster(wurzel: str, auft: dict, vorher: dict, entfallen: set) -> bool:
    import importlib
    import inspect
    basis.wurzel_setzen(wurzel)
    try:
        mw_mod = importlib.import_module("ui.main_window")
    except Exception as exc:  # noqa: BLE001 — jeder Importfehler ist ein Befund
        print(f"ui.main_window lässt sich nicht importieren: {exc.__class__.__name__}: {exc}")
        return False
    klasse = getattr(mw_mod, KLASSE)
    mro = klasse.__mro__
    eigene = []          # (Klasse, relpfad, ClassDef, zeilen) aus der Wurzel
    gelesen: dict = {}
    for c in mro:
        try:
            datei = inspect.getsourcefile(c)
        except TypeError:
            datei = None
        if not datei or not os.path.abspath(datei).startswith(wurzel + os.sep):
            continue
        if datei not in gelesen:
            gelesen[datei] = lies(datei)
        _, baum, zeilen = gelesen[datei]
        k = next((k for k in baum.body if isinstance(k, ast.ClassDef)
                  and k.name == c.__name__), None)
        if k is not None:
            eigene.append((c, os.path.relpath(datei, wurzel), k, zeilen))
    print(f"MRO: {' -> '.join(c.__name__ for c in mro)}")
    ok = True
    gleich = textgleich = 0
    abw, fehlt, weg = [], [], []
    for name, e in list(vorher["methoden"].items()) + list(vorher["klassenattribute"].items()):
        besitzer = next((c for c in mro if name in c.__dict__), None)
        eig = next((x for x in eigene if x[0] is besitzer), None)
        if eig is None:
            (weg if name in entfallen else fehlt).append(name)
            continue
        knoten = klassen_inhalt(eig[2]).get(name)
        if knoten is None:
            fehlt.append(name)
            continue
        erg = vergleiche_knoten(e, knoten, eig[3])
        if erg["gleich"]:
            gleich += 1
            textgleich += erg["textgleich"]
        else:
            abw.append(name)
            _zeige(f"{name} ({eig[1]}::{eig[0].__name__})", erg)
    bekannt = set(vorher["methoden"]) | set(vorher["klassenattribute"])
    zusaetzlich = sorted({n for _, _, k, _ in eigene for n in klassen_inhalt(k)} - bekannt)
    # ein Name in zwei Klassen der MRO: die spätere wird still überdeckt
    besitzer_je_name = collections.defaultdict(list)
    for c, rel, k, _ in eigene:
        for n in klassen_inhalt(k):
            besitzer_je_name[n].append(f"{rel}::{c.__name__}")
    mehrfach = {n: b for n, b in besitzer_je_name.items() if len(b) > 1}
    for n, b in sorted(mehrfach.items()):
        print(f"  MEHRFACH {n}: {', '.join(b)} (es gilt die erste)")
    ok = ok and not mehrfach
    for n in fehlt:
        print(f"  FEHLT {n}")
    if weg:
        print(f"  entfallen wie benannt: {', '.join(weg)}")
    if zusaetzlich:
        print(f"  zusätzlich: {', '.join(zusaetzlich)}")
    # Rahmen: __init__ und closeEvent gehören MainWindow, kein Mixin hat __init__
    for name in ("__init__", "closeEvent"):
        besitzer = next((c for c in mro if name in c.__dict__), None)
        if besitzer is not klasse:
            print(f"  {name} kommt aus {getattr(besitzer, '__name__', None)}, nicht aus {KLASSE}")
            ok = False
    for c, rel, k, _ in eigene:
        if c is not klasse and "__init__" in klassen_inhalt(k):
            print(f"  {rel}::{c.__name__} definiert __init__")
            ok = False
    n_m = len(vorher["methoden"])
    n_a = len(vorher["klassenattribute"])
    print(f"{KLASSE}: {gleich} gleich{_textinfo(gleich, textgleich)}, {len(abw)} abweichend, "
          f"{len(fehlt)} fehlend, {len(zusaetzlich)} zusätzlich (Vorher-Stand: {n_m} Methoden "
          f"und {n_a} Klassenattribute)")
    n_meth_gleich = sum(1 for n in vorher["methoden"] if n not in abw and n not in fehlt
                        and n not in weg)
    print(f"{n_meth_gleich} Methoden gleich, {len([n for n in fehlt if n in vorher['methoden']])}"
          f" fehlend, {len(zusaetzlich)} zusätzlich")
    return ok and not (abw or fehlt or zusaetzlich)


# ------------------------------------------------------------- --schreibe

def schreibe(ersetzen: bool) -> int:
    if os.path.exists(METHODEN_JSON) and not ersetzen:
        print(f"{os.path.relpath(METHODEN_JSON, basis.ARBEITSBAUM)} gibt es schon "
              f"(--ersetzen zum Überschreiben).", file=sys.stderr)
        return 2
    erg = erfasse(git_text())
    os.makedirs(os.path.dirname(METHODEN_JSON), exist_ok=True)
    tmp = METHODEN_JSON + ".neu"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(erg, fh, ensure_ascii=False, indent=1, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, METHODEN_JSON)
    print(f"geschrieben: {os.path.relpath(METHODEN_JSON, basis.ARBEITSBAUM)} aus "
          f"{basis.VORHER_TAG}:{QUELLE} — {len(erg['methoden'])} Methoden, "
          f"{len(erg['klassenattribute'])} Klassenattribute, {len(erg['modulnamen'])} Modulnamen")
    return 0


# ------------------------------------------------------------ Selbsttest

def pruefe_aufteilung(auft: dict, vorher: dict) -> list:
    """Widersprüche zwischen aufteilung.json und dem Vorher-Stand (leer = gut)."""
    fehler = []
    module = auft["module"]
    for gruppe in ("methoden", "klassenattribute"):
        soll, ist = set(vorher[gruppe]), set(auft[gruppe])
        for n in sorted(soll - ist):
            fehler.append(f"{gruppe}: {n} ist keinem Zielmodul zugeordnet")
        for n in sorted(ist - soll):
            fehler.append(f"{gruppe}: {n} gibt es im Vorher-Stand nicht")
        for n, m in auft[gruppe].items():
            if m not in module:
                fehler.append(f"{gruppe}: {n} -> unbekanntes Modul {m}")
    for n in sorted(set(vorher["modulnamen"]) ^ set(auft["modulnamen"])):
        fehler.append(f"modulnamen: {n} nicht in beiden Dateien")
    for n, z in auft["modulnamen"].items():
        karte = auft["umbenennungen"].get(z["ziel"], {})
        if z["name"] != karte.get(n, n):
            fehler.append(f"modulnamen: {n} heißt am Ziel {z['name']}, die Umbenennung "
                          f"sagt {karte.get(n, n)}")
        if z.get("modul") and module[z["modul"]]["datei"] != z["ziel"]:
            fehler.append(f"modulnamen: {n}: Modul {z['modul']} und Ziel {z['ziel']} passen "
                          f"nicht zusammen")
    for alt, kopien in auft.get("kopien", {}).items():
        if _alt_eintrag(vorher, alt) is None:
            fehler.append(f"kopien: {alt} gibt es im Vorher-Stand nicht")
        for k in kopien:
            karte = auft["umbenennungen"].get(k["ziel"], {})
            if k["name"] != karte.get(alt, alt):
                fehler.append(f"kopien: {alt} -> {k['name']} fehlt in den Umbenennungen")
    # Jede Methode (samt Dekorator) ganz in genau einem Zeilenbereich, und zwar
    # in dem ihres Zielmoduls; bekannte Abweichungen sind benannt.
    bekannt = {a["name"]: a for a in auft.get("bekannte_abweichungen", [])}
    soll = dict(auft["methoden"])
    soll.update(auft["klassenattribute"])
    for name, modul in soll.items():
        e = vorher["methoden"].get(name) or vorher["klassenattribute"][name]
        von, bis = e["zeilen"]
        ganz = [m for m, z in module.items() for a, b in z["bereiche"] if a <= von and bis <= b]
        if len(ganz) == 1 and ganz[0] == modul:
            if name in bekannt:
                fehler.append(f"bekannte Abweichung {name} trifft nicht mehr zu")
            continue
        if name in bekannt:
            a = bekannt[name]
            def_zeile = von + len(e.get("dekoratoren", ()))
            treffer = [m for m, z in module.items() for x, y in z["bereiche"]
                       if x <= def_zeile and bis <= y]
            if treffer == [modul] and a.get("zeilen") == [von, bis]:
                continue
        fehler.append(f"{name} ({von}-{bis}, Ziel {modul}) liegt ganz in: {ganz or 'keinem'}")
    # Vertragsstellen: Zeile -> Methode -> Zieldatei muss stimmen
    for v in auft.get("vertragsstellen", []):
        for s in v["stellen"]:
            name = methode_an_zeile(vorher, s["zeile"])
            if name != s["name"]:
                fehler.append(f"Vertragsstelle {v['was']}: Zeile {s['zeile']} liegt in {name}, "
                              f"nicht in {s['name']}")
                continue
            if name in soll:
                datei = module[soll[name]]["datei"]
            else:
                datei = auft["modulnamen"][name]["ziel"]
            if s["datei"] != datei:
                fehler.append(f"Vertragsstelle {v['was']}: {name} geht nach {datei}, "
                              f"nicht nach {s['datei']}")
    return fehler


def _mixin_text(vorher: dict, namen: list, klasse: str, aendern=None) -> str:
    teile = [f"class {klasse}:\n"]
    for n in namen:
        text = vorher["methoden"][n]["quelltext"]
        teile.append("\n" + (aendern(n, text) if aendern else text))
    return "".join(teile)


def selbsttest(wurzel: str) -> int:
    fehler: list = []

    def soll(bedingung: bool, text: str) -> None:
        print(("  ok   " if bedingung else "  FEHL ") + text)
        if not bedingung:
            fehler.append(text)

    vorher = vorher_laden()
    tag = git_text()
    neu = erfasse(tag)
    print(f"Vorher-Stand {basis.VORHER_TAG}:{QUELLE} ({neu['sha256'][:12]}):")
    soll(neu["sha256"] == vorher["sha256"], "methoden.json gehört zum Tag-Stand")
    gleich = sum(1 for n, e in neu["methoden"].items()
                 if n in vorher["methoden"] and vorher["methoden"][n]["dump"] == e["dump"])
    fehlend = len(set(vorher["methoden"]) - set(neu["methoden"]))
    print(f"{gleich} Methoden gleich, {fehlend} fehlend")
    soll(gleich == len(vorher["methoden"]) == 184 and fehlend == 0,
         "184 Methoden von MainWindow erfasst und gleich")
    soll(not neu["doppelt"], f"kein Name doppelt definiert ({neu['doppelt']})")
    quelle = os.path.join(wurzel, QUELLE)
    if os.path.isfile(quelle):
        ist = erfasse(lies(quelle)[0])
        gleich_w = sum(1 for n, e in ist["methoden"].items()
                       if vorher["methoden"].get(n, {}).get("dump") == e["dump"])
        print(f"  {QUELLE} unter der Wurzel: {'unverändert' if ist['sha256'] == vorher['sha256'] else 'verändert'}"
              f", {len(ist['methoden'])} Methoden in {KLASSE}, davon {gleich_w} gleich")
    # gespeicherter Quelltext ergibt wieder denselben Dump
    roh = [n for g in ("methoden", "klassenattribute", "modulnamen")
           for n, e in vorher[g].items() if ast.dump(alter_knoten(e)) != e["dump"]]
    soll(not roh, f"Quelltext und Dump in methoden.json passen zusammen ({roh[:3]})")

    arten = collections.defaultdict(list)
    for n, e in vorher["modulnamen"].items():
        arten[e["art"]].append(n)
    print(f"Erfasste Modulnamen: {len(arten['klasse'])} Klassen, {len(arten['funktion'])} "
          f"Funktionen, {len(arten['konstante'])} Konstanten")
    for a in ("klasse", "funktion", "konstante"):
        print(f"  {a}: {', '.join(arten[a])}")
    print(f"Klassenattribute: {', '.join(vorher['klassenattribute'])}")

    print("aufteilung.json:")
    auft = aufteilung_laden()
    probleme = pruefe_aufteilung(auft, vorher)
    for p in probleme:
        print(f"    {p}")
    soll(not probleme, "jede Methode, jedes Klassenattribut, jeder Modulname genau einem "
                       "Zielmodul zugeordnet; jede Methode ganz in einem Bereich ihres "
                       "Zielmoduls (bis auf die benannten Abweichungen)")
    zaehl = collections.Counter(auft["methoden"].values())
    print("  " + ", ".join(f"{m} {zaehl[m]}" for m in auft["module"]))
    for a in auft.get("bekannte_abweichungen", []):
        print(f"  bekannte Abweichung: {a['name']}: {a['grund']}")
    try:
        json.loads('{"a": 1, "a": 2}', object_pairs_hook=_ohne_doppelte)
        soll(False, "doppelter Schlüssel in aufteilung.json wird erkannt")
    except Fehler:
        soll(True, "doppelter Schlüssel in aufteilung.json wird erkannt")

    print("Vergleich an gezielten Änderungen:")
    namen = ["_log", "_meander_build", "_on_menu_colormode", "_ebene_passt"]
    sauber = ast.parse(_mixin_text(vorher, namen, "AMixin")).body[0]
    zeilen = _mixin_text(vorher, namen, "AMixin").splitlines()
    inh = klassen_inhalt(sauber)
    soll(all(vergleiche_knoten(vorher["methoden"][n], inh[n], zeilen)["gleich"] for n in namen),
         "ausgeschnittene Methoden sind gleich")

    def mutiert(name, alt, neu_, erwartet_gleich, text):
        t = _mixin_text(vorher, namen, "AMixin",
                        lambda n, q: q.replace(alt, neu_, 1) if n == name else q)
        k = klassen_inhalt(ast.parse(t).body[0])[name]
        erg = vergleiche_knoten(vorher["methoden"][name], k, t.splitlines())
        soll(erg["gleich"] == erwartet_gleich and (erwartet_gleich or erg["diff"]), text)
        return erg

    erg = mutiert("_log", "def _log", "# Kommentar\n    def _log", True,
                  "Kommentar vor der Methode: gleich")
    soll(erg["textgleich"], "Kommentar vor der Methode: textgleich (zählt zur Lücke davor)")
    erg = mutiert("_on_menu_colormode", "if idx >= 0:", "if idx >= 0:  # angehängt", True,
                  "Kommentar im Rumpf: gleich")
    soll(not erg["textgleich"], "Kommentar im Rumpf: nicht textgleich")
    mutiert("_on_menu_colormode", "idx >= 0", "idx > 0", False, "'>=' zu '>': abweichend")
    mutiert("_meander_build", "@staticmethod\n    ", "", False, "fehlender @staticmethod: abweichend")
    mutiert("_log", '"%H:%M:%S"', '"%H:%M"', False, "geänderte Zeichenkette: abweichend")
    erg = mutiert("_ebene_passt", "Gehoert eine", "Gehört eine", False,
                  "geänderter Docstring: abweichend")
    soll(erg["hinweis"] == "nur im Docstring abweichend", "… mit Hinweis 'nur im Docstring'")

    # Umbenennungen: _exploration_holen in core/exploration.py als holen
    e = vorher["modulnamen"]["_exploration_holen"]
    karte = auft["umbenennungen"].get("core/exploration.py", {})
    t = e["quelltext"].replace("_exploration_rechnen(", "rechnen_und_ablegen(") \
        .replace("exploration.Explorationsgrad", "Explorationsgrad") \
        .replace("def _exploration_holen", "def holen")
    k = ast.parse(t).body[0]
    soll(vergleiche_knoten(e, k, t.splitlines(), karte)["gleich"],
         "benannte Umbenennung samt weggelassenem Modulpräfix: gleich")
    soll(not vergleiche_knoten(e, k, t.splitlines())["gleich"],
         "dieselbe Umbenennung ohne Nennung: abweichend")
    e = vorher["methoden"]["_ebene_passt"]
    t = textwrap.dedent(e["quelltext"]).replace("@staticmethod\n", "") \
        .replace("def _ebene_passt", "def passt")
    k = ast.parse(t).body[0]
    soll(vergleiche_knoten(e, k, t.splitlines(), {"_ebene_passt": "passt"},
                           statisch_weg=True)["gleich"],
         "staticmethod als Modulfunktion (Kopie): gleich")
    e = vorher["modulnamen"]["ThreadLocalBag"]
    t = e["quelltext"]
    kl = ast.parse(t).body[0]
    kl.body = [g for g in kl.body if getattr(g, "name", "") != "iter_camera"]
    t2 = ast.unparse(kl)
    erg = vergleiche_klasse(e, ast.parse(t2).body[0], t2.splitlines(), None, ("iter_camera",))
    soll(erg["gleich"] and erg["entfallen"] == ["iter_camera"],
         "Klasse ohne benannt entfallenes Glied: gleich")
    erg = vergleiche_klasse(e, ast.parse(t2).body[0], t2.splitlines(), None, ())
    soll(not erg["gleich"] and erg["fehlt"] == ["iter_camera"],
         "Klasse ohne unbenannt entfallenes Glied: abweichend")

    # --gesamt an erfundenen Ständen
    texte = {QUELLE: tag}
    erg = gesamt_auswerten(texte, auft, vorher, set())
    soll(erg["in_mw"] == 184 and not erg["doppelt"] and not erg["fehlend"]
         and len(erg["namen_in_mw"]) == len(auft["modulnamen"]),
         "gesamt vor der Zerlegung: 184 in main_window.py, nichts doppelt oder fehlend")
    ziel = auft["module"][auft["methoden"]["_log"]]
    andere = next(z for m, z in auft["module"].items()
                  if m not in (BLEIBT, auft["methoden"]["_log"]))
    texte[ziel["datei"]] = _mixin_text(vorher, ["_log"], ziel["klasse"])
    erg = gesamt_auswerten(texte, auft, vorher, set())
    soll(not erg["doppelt"] and erg["noch_in_mw"] == ["_log"],
         "verschoben, main_window.py noch unverändert: kein doppelt")
    texte[andere["datei"]] = _mixin_text(vorher, ["_log"], andere["klasse"])
    erg = gesamt_auswerten(texte, auft, vorher, set())
    soll(any(n == "_log" for n, _ in erg["doppelt"]) and erg["falsch"],
         "dieselbe Methode in zwei Mixins: doppelt und falsch zugeordnet")
    del texte[andere["datei"]]
    texte[QUELLE] = tag.replace("class MainWindow(QMainWindow):",
                                f"class MainWindow({ziel['klasse']}, QMainWindow):")
    erg = gesamt_auswerten(texte, auft, vorher, set())
    soll(any(n == "_log" for n, _ in erg["doppelt"]),
         "nach dem Umschalten überdeckt MainWindow die Mixin-Methode: doppelt")

    print("vergleiche_methoden SELFTEST " + ("OK" if not fehler else
                                             f"FEHLGESCHLAGEN ({len(fehler)})"))
    return 0 if not fehler else 1


# -------------------------------------------------------------------- main

def main(argv=None) -> int:
    parser = basis.argumente(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    parser.add_argument("--selbsttest", action="store_true")
    parser.add_argument("--schreibe", action="store_true",
                        help="vorher/methoden.json aus dem Tag vor-umbau aufnehmen")
    parser.add_argument("--ersetzen", action="store_true")
    parser.add_argument("--modul", action="append", default=[], metavar="DATEI")
    parser.add_argument("--neu-erlaubt", action="store_true",
                        help="--modul: zusätzliche (neue) Methoden nicht als Fehler werten")
    parser.add_argument("--klasse", action="append", nargs=2, default=[],
                        metavar=("NAME", "DATEI"))
    parser.add_argument("--modulnamen", action="append", default=[], metavar="DATEI")
    parser.add_argument("--gesamt", action="store_true")
    parser.add_argument("--fenster", action="store_true")
    parser.add_argument("--entfallen", default="", metavar="NAME,…",
                        help="benannt entfernte Namen für --gesamt und --fenster")
    args = parser.parse_args(argv)
    wurzel = os.path.abspath(args.wurzel or basis.ARBEITSBAUM)
    entfallen = {n for n in args.entfallen.split(",") if n}
    if not (args.selbsttest or args.schreibe or args.modul or args.klasse
            or args.modulnamen or args.gesamt or args.fenster):
        parser.print_help()
        return 2
    try:
        if args.schreibe:
            return schreibe(args.ersetzen)
        if args.selbsttest:
            return selbsttest(wurzel)
        auft = aufteilung_laden()
        vorher = vorher_laden()
        ok = True
        for pfad in args.modul:
            ok = pruefe_modul(wurzel, pfad, auft, vorher, args.neu_erlaubt) and ok
        for name, pfad in args.klasse:
            ok = pruefe_klasse(wurzel, name, pfad, auft, vorher) and ok
        for pfad in args.modulnamen:
            ok = pruefe_modulnamen(wurzel, pfad, auft, vorher) and ok
        if args.gesamt:
            ok = pruefe_gesamt(wurzel, auft, vorher, entfallen) and ok
        if args.fenster:
            ok = pruefe_fenster(wurzel, auft, vorher, entfallen) and ok
        return 0 if ok else 1
    except Fehler as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

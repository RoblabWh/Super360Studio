#!/usr/bin/env python3
"""Numerik-Proben: feste Eingaben, Ausgaben als Hash gegen den Vorher-Stand.

Starter für alle ``scripts/umbau/proben_<familie>.py``. Vertrag je Modul::

    PROBEN = {name: funktion}      # funktion() -> {teilname: ndarray | bytes | zahl}
    UNSTET = {name: toleranz}      # nicht bitgleich wiederholbare Proben

Der Name einer Probe beginnt mit dem geprüften Modul (``meander.thin``); danach
wird in der Zusammenfassung gezählt. Die Proben laden ``core`` aus der Wurzel
(``--wurzel``) und rechnen auf festen, geseedeten Eingaben.

Aufrufe::

    python3 scripts/umbau/numerik_probe.py --schreibe [--ersetzen] [--nur familie,…]
    python3 scripts/umbau/numerik_probe.py --vergleiche [--nur familie,… | familie.probe,…]
                                           [--erwartet <json im Arbeitsordner>]
                                           [--erwartet-vorlage <json im Arbeitsordner>]
    python3 scripts/umbau/numerik_probe.py --selbsttest

``--schreibe`` legt je Familie ``vorher/numerik_<familie>.json`` (Hashes und
Umgebungsstempel) und ``.npz`` (Ausgaben bis 2 MB je Probe) ab und verweigert
ein vorhandenes Ziel ohne ``--ersetzen``. ``--vergleiche`` rechnet neu und
vergleicht die Hashes; unstete Proben gelten innerhalb ihrer Toleranz als
gleich. Passt der Umgebungsstempel nicht (Bibliotheken, PointCloudMerger),
wird der Vorher-Stand im selben Lauf aus ``git archive vor-umbau`` neu
gerechnet (Bibliotheken, die ein älterer Vorher-Stand noch nicht nennt, zählen
dabei nicht; es gibt einen Hinweis). Eine Abweichung nennt Probe, Teil und
größte Differenz; fehlt der alte Wert in der .npz, wird er dafür ebenfalls aus
dem Tag nachgerechnet.

``--erwartet`` nennt gewollte Abweichungen je Teil und bindet sie an den neuen
Wert, damit eine weitere Änderung an derselben Stelle ein Fehler bleibt::

    {"familie.probe/teil": {"grund": "Text", "hash": "<Hash des neuen Werts>"}}

Anstelle des Hashs steht ``"fehlt"`` für eine entfallene Probe oder einen
entfallenen Teil, ``"neu"`` für eine neue Probe (Teil ``-``) und ``"unstet"``
für den Teil einer unsteten Probe außerhalb der Toleranz. Der Schlüssel darf
``*`` enthalten. Eine Probe, die mit einem Fehler endet, lässt sich nicht
freigeben. ``--erwartet-vorlage`` schreibt für alle Abweichungen eine solche
Datei mit leerem ``grund`` — ohne Grund wird ein Eintrag nicht angenommen.

Jede Familie rechnet in einem eigenen Kindprozess. Endet er durch ein Signal
(open3d stürzt unter Last vereinzelt ab), wird die Familie einmal wiederholt;
seine Arbeitsordner liegen im Ordner des Starters und verschwinden mit ihm.

Für Selbsttest und Gegenprüfung: ``--proben-ordner`` (wo die proben_*.py
liegen), ``--ziel`` (Ordner des Vorher-Stands), ``--vorher-wurzel`` (ein
schon entpackter Vorher-Stand anstelle von ``git archive``), ``--ohne-grenze``
(alle Ausgaben ablegen).

Ende: 0 = alle Proben gleich, 1 = Abweichung, 2 = Aufruf- oder Laufzeitfehler.
"""
from __future__ import annotations

import argparse
import collections
import fnmatch
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import basis  # noqa: E402

GRENZE_BYTES = 2 * 1024 * 1024


# ------------------------------------------------------------------- Laden

def familien_finden(ordner: str) -> dict:
    out = {}
    for datei in sorted(os.listdir(ordner)):
        if datei.startswith("proben_") and datei.endswith(".py"):
            out[datei[len("proben_"):-3]] = os.path.join(ordner, datei)
    return out


def familie_laden(familie: str, pfad: str):
    name = f"proben_{familie}"
    spec = importlib.util.spec_from_file_location(name, pfad)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    proben = getattr(mod, "PROBEN", None)
    unstet = getattr(mod, "UNSTET", {})
    if not isinstance(proben, dict) or not proben:
        raise RuntimeError(f"{pfad}: PROBEN fehlt oder ist leer.")
    fremd = sorted(set(unstet) - set(proben))
    if fremd:
        raise RuntimeError(f"{pfad}: UNSTET nennt unbekannte Proben: {fremd}")
    return proben, dict(unstet)


def _wahl(nur: str | None, familien: dict):
    """``--nur`` aufteilen: gewählte Familien und je Familie die Probennamen."""
    if not nur:
        return {f: None for f in familien}
    wahl: dict = {}
    for teil in (t.strip() for t in nur.split(",")):
        if not teil:
            continue
        if teil in familien:
            wahl[teil] = None
            continue
        fam, _, probe = teil.partition(".")
        if fam in familien and probe:
            if wahl.get(fam, ()) is not None:
                wahl.setdefault(fam, set()).add(probe)
            continue
        print(f"--nur {teil}: keine Familie dieses Namens "
              f"(vorhanden: {', '.join(familien) or 'keine'}).")
        raise SystemExit(2)
    return wahl


# ----------------------------------------------------------------- Rechnen

def _als_array(wert):
    import numpy as np
    if isinstance(wert, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(wert), dtype=np.uint8), "bytes"
    arr = np.asarray(wert)
    if arr.dtype.hasobject:
        raise TypeError(f"{type(wert).__name__} ist weder Array noch Bytes noch Zahl")
    return arr, ("ndarray" if isinstance(wert, np.ndarray) else "zahl")


def rechne(proben: dict, namen=None) -> dict:
    """Fährt die Proben: ``{name: {teil: wert}}`` bzw. ``{name: Exception}``."""
    out: dict = {}
    for name in sorted(proben):
        if namen is not None and name not in namen:
            continue
        try:
            teile = proben[name]()
            if not isinstance(teile, dict) or not teile:
                raise TypeError("die Probe liefert kein dict {teilname: wert}")
            for teil, wert in teile.items():
                _als_array(wert)
            out[name] = teile
        except Exception as exc:  # noqa: BLE001 — der Lauf nennt sie als Fehler
            exc.spur = traceback.format_exc()
            out[name] = exc
    return out


#: Ein Teil einer Probe, wie ihn der Kindprozess meldet.
Wert = collections.namedtuple("Wert", "arr art hash")


class ProbenFehler(Exception):
    """Eine im Kindprozess gescheiterte Probe: ``Klasse: Text`` und die Spur."""

    def __init__(self, text: str, spur: str):
        super().__init__(text)
        self.spur = spur


def _kind(args) -> int:
    """Der Kindprozess von :func:`rechne_familie`."""
    import numpy as np

    with open(args.kind, encoding="utf-8") as fh:
        auftrag = json.load(fh)
    basis.wurzel_setzen(args.wurzel)
    familie = auftrag["familie"]
    try:
        proben, unstet = familie_laden(familie, familien_finden(args.proben_ordner)[familie])
    except RuntimeError as exc:
        print(f"FEHLER: {exc}")
        return 2
    namen = auftrag["namen"]
    werte = rechne(proben, None if namen is None else set(namen))
    bericht: dict = {}
    ablage: dict = {}
    for name, teile in werte.items():
        if isinstance(teile, Exception):
            bericht[name] = {"fehler": f"{teile.__class__.__name__}: {teile}",
                             "spur": teile.spur}
            continue
        bericht[name] = {"teile": {}}
        for teil, wert in teile.items():
            arr, art = _als_array(wert)
            ablage[f"{name}/{teil}"] = arr
            bericht[name]["teile"][teil] = {"art": art, "hash": basis.hash_wert(wert)}
    ergebnis = auftrag["ergebnis"]
    with open(ergebnis + ".npz", "wb") as fh:
        np.savez(fh, **ablage)
    with open(ergebnis + ".tmp", "w", encoding="utf-8") as fh:
        json.dump({"proben": sorted(proben), "unstet": unstet, "werte": bericht}, fh,
                  ensure_ascii=False)
    os.replace(ergebnis + ".tmp", ergebnis)
    return 0


def rechne_familie(args, familie: str, namen=None) -> tuple:
    """Rechnet eine Familie in einem Kindprozess.

    Zurück kommen die Namen aller Proben der Familie, ``UNSTET`` und
    ``{name: {teil: Wert} | ProbenFehler}``. Endet das Kind durch ein Signal,
    wird es einmal wiederholt. Seine Arbeitsordner entstehen im Ordner dieses
    Aufrufs und werden mit ihm geräumt, auch die eines abgestürzten Laufs.
    """
    import numpy as np

    ordner = basis.arbeitsordner("numerik")
    auftrag = os.path.join(ordner, "auftrag.json")
    ergebnis = os.path.join(ordner, "ergebnis.json")
    with open(auftrag, "w", encoding="utf-8") as fh:
        json.dump({"familie": familie, "ergebnis": ergebnis,
                   "namen": None if namen is None else sorted(namen)}, fh)
    befehl = [sys.executable, os.path.abspath(__file__), "--wurzel", basis.wurzel(),
              "--proben-ordner", args.proben_ordner, "--kind", auftrag]
    umgebung = dict(os.environ, **{basis.ELTERN_VARIABLE: ordner})
    try:
        for versuch in (1, 2):
            sys.stdout.flush()
            code = subprocess.run(befehl, env=umgebung).returncode
            if code == 0 and os.path.isfile(ergebnis):
                with open(ergebnis, encoding="utf-8") as fh:
                    stand = json.load(fh)
                with np.load(ergebnis + ".npz", allow_pickle=False) as daten:
                    arrays = {k: daten[k] for k in daten.files}
                werte: dict = {}
                for name, eintrag in stand["werte"].items():
                    if "fehler" in eintrag:
                        werte[name] = ProbenFehler(eintrag["fehler"], eintrag["spur"])
                        continue
                    werte[name] = {teil: Wert(arrays[f"{name}/{teil}"], t["art"], t["hash"])
                                   for teil, t in eintrag["teile"].items()}
                return stand["proben"], stand["unstet"], werte
            if code >= 0 or versuch == 2:
                raise RuntimeError(
                    f"Familie {familie}: der Kindprozess endete mit "
                    f"{f'Signal {-code}' if code < 0 else f'Ende {code}'}"
                    f"{' (auch beim zweiten Versuch)' if versuch == 2 else ''}.")
            print(f"Hinweis: Familie {familie}: der Kindprozess endete mit Signal "
                  f"{-code}; die Familie wird einmal wiederholt.")
            for rest in os.listdir(ordner):
                if rest != os.path.basename(auftrag):
                    basis.raeume(os.path.join(ordner, rest))
    finally:
        if ordner in basis._raeumen:
            basis.raeume(ordner)
    raise AssertionError("nicht erreichbar")


def _gruppen(namen) -> str:
    zahl: dict = {}
    for name in namen:
        kopf = name.split(".", 1)[0]
        zahl[kopf] = zahl.get(kopf, 0) + 1
    return ", ".join(f"{k} {v}" for k, v in sorted(zahl.items()))


def _zusammenfassung(familie: str, namen, unstet: dict) -> str:
    text = f"Familie {familie}: {len(namen)} Proben ({_gruppen(namen)})"
    da = {n: t for n, t in sorted(unstet.items()) if n in namen}
    if da:
        text += "; unstet: " + ", ".join(f"{n} (Toleranz {t:g})" for n, t in da.items())
    else:
        text += "; keine unstet"
    return text


# --------------------------------------------------------------- Schreiben

def _ziele(ziel: str, familie: str):
    return (os.path.join(ziel, f"numerik_{familie}.json"),
            os.path.join(ziel, f"numerik_{familie}.npz"))


def schreibe(args, familie: str, grenze) -> int:
    import numpy as np

    ziel = args.ziel
    js, npz = _ziele(ziel, familie)
    da = [p for p in (js, npz) if os.path.exists(p)]
    if da and not args.ersetzen:
        print(f"{familie}: Ziel vorhanden ({', '.join(os.path.basename(p) for p in da)}) "
              f"— nur mit --ersetzen.")
        return 2
    _proben, unstet, werte = rechne_familie(args, familie)
    fehler = {n: w for n, w in werte.items() if isinstance(w, Exception)}
    for name, exc in fehler.items():
        print(f"{familie}: Probe {name} scheitert:\n{exc.spur}")
    if fehler:
        return 2
    eintraege: dict = {}
    ablage: dict = {}
    for name, teile in werte.items():
        rest = grenze
        eintrag = {"unstet": unstet.get(name), "teile": {}}
        for teil in sorted(teile):
            arr, art = teile[teil].arr, teile[teil].art
            passt = rest is None or arr.nbytes <= rest
            if passt:
                ablage[f"{name}/{teil}"] = arr
                if rest is not None:
                    rest -= arr.nbytes
            eintrag["teile"][teil] = {
                "hash": teile[teil].hash, "art": art,
                "dtype": arr.dtype.str, "shape": list(arr.shape),
                "abgelegt": bool(passt)}
        if name in unstet and not all(t["abgelegt"] for t in eintrag["teile"].values()):
            print(f"{familie}: unstete Probe {name} ist größer als die Ablagegrenze — "
                  f"ohne abgelegten Wert lässt sich keine Toleranz prüfen. "
                  f"Bitte die Probe verkleinern.")
            return 2
        eintraege[name] = eintrag
    os.makedirs(ziel, exist_ok=True)
    stand = {"familie": familie, "umgebung": basis.umgebungsstempel(),
             "grenze_bytes": grenze, "proben": eintraege}
    with open(js + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(stand, fh, indent=1, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    os.replace(js + ".tmp", js)
    with open(npz + ".tmp", "wb") as fh:
        np.savez_compressed(fh, **ablage)
    os.replace(npz + ".tmp", npz)
    print(_zusammenfassung(familie, list(eintraege), unstet))
    print(f"{familie}: geschrieben nach {basis.pfad_neutral(js)} "
          f"({len(ablage)} von {sum(len(e['teile']) for e in eintraege.values())} "
          f"Teilen als Wert abgelegt)")
    return 0


# -------------------------------------------------------------- Vergleichen

def _lade_stand(ziel: str, familie: str):
    import numpy as np

    js, npz = _ziele(ziel, familie)
    if not os.path.isfile(js):
        return None, {}
    with open(js, encoding="utf-8") as fh:
        stand = json.load(fh)
    werte = {}
    if os.path.isfile(npz):
        with np.load(npz, allow_pickle=False) as daten:
            werte = {k: daten[k] for k in daten.files}
    return stand, werte


def groesste_differenz(alt, neu) -> tuple:
    """(Zahl oder inf, Beschreibung) für zwei Arrays."""
    import numpy as np

    if alt.shape != neu.shape:
        return float("inf"), f"shape {tuple(alt.shape)} gegen {tuple(neu.shape)}"
    zusatz = "" if alt.dtype == neu.dtype else f", dtype {alt.dtype} gegen {neu.dtype}"
    if alt.dtype.kind in "SUV" or neu.dtype.kind in "SUV":
        n = int(np.count_nonzero(alt != neu))
        return (float("inf") if n else 0.0), f"{n} von {alt.size} Elementen verschieden{zusatz}"
    if alt.size == 0:
        return 0.0, f"leer{zusatz}"
    a = alt.astype(np.float64).ravel()
    b = neu.astype(np.float64).ravel()
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    d = np.abs(a - b)
    d[nan_a & nan_b] = 0.0
    d[nan_a ^ nan_b] = np.inf
    d[(a == b)] = 0.0                      # auch inf gegen inf
    i = int(np.argmax(d))
    n = int(np.count_nonzero(d > 0))
    ort = tuple(int(v) for v in np.unravel_index(i, alt.shape)) if alt.ndim else ()
    return float(d[i]), (f"größte Differenz {d[i]:.6g} bei Index {ort} "
                         f"(alt {float(a[i])!r}, neu {float(b[i])!r}), {n} von {a.size} "
                         f"Elementen verschieden{zusatz}")


def _neu_rechnen(args, familien: list) -> str:
    """Rechnet den Vorher-Stand in einem Kindprozess neu; gibt dessen Ordner."""
    vorher = args.vorher_wurzel or basis.tag_entpacken()
    ordner = basis.arbeitsordner("vorher")
    ziel = os.path.join(ordner, "vorher")
    befehl = [sys.executable, os.path.abspath(__file__), "--wurzel", vorher,
              "--schreibe", "--ohne-grenze", "--ziel", ziel,
              "--proben-ordner", args.proben_ordner, "--nur", ",".join(familien)]
    lauf = subprocess.run(befehl, capture_output=True, text=True,
                          env=dict(os.environ, **{basis.ELTERN_VARIABLE: ordner}))
    if lauf.returncode != 0:
        print(lauf.stdout + lauf.stderr)
        raise RuntimeError("Der Vorher-Stand ließ sich nicht neu rechnen.")
    return ziel


def erwartet_laden(pfad: str) -> dict:
    """Liest die Datei für ``--erwartet`` und prüft ihre Form."""
    with open(pfad, encoding="utf-8") as fh:
        erwartet = json.load(fh)
    if not isinstance(erwartet, dict):
        raise RuntimeError("--erwartet: die Datei muss ein Objekt "
                           "{familie.probe/teil: eintrag} enthalten.")
    for key, eintrag in erwartet.items():
        if not isinstance(eintrag, dict) or set(eintrag) != {"grund", "hash"} \
                or not isinstance(eintrag["hash"], str):
            raise RuntimeError(f"--erwartet '{key}': verlangt ist "
                               f"{{\"grund\": …, \"hash\": …}}.")
        if not isinstance(eintrag["grund"], str) or not eintrag["grund"].strip():
            raise RuntimeError(f"--erwartet '{key}': der Grund fehlt.")
    return erwartet


def _eintrag_fuer(erwartet: dict, key: str):
    """Schlüssel des Eintrags für eine Stelle: der wörtliche vor einem Muster."""
    if key in erwartet:
        return key
    return next((k for k in erwartet if fnmatch.fnmatchcase(key, k)), None)


def _stempel_vergleich(alt: dict, stempel: dict) -> tuple:
    """(abweichende, im Vorher-Stand noch nicht genannte) Schlüssel des Stempels."""
    anders = sorted(k for k in alt if alt.get(k) != stempel.get(k))
    unbekannt = sorted(k for k in stempel if k not in alt)
    return anders, unbekannt


def vergleiche(args, familien: dict, wahl: dict) -> int:
    erwartet = erwartet_laden(args.erwartet) if args.erwartet else {}
    staende: dict = {}
    for familie in wahl:
        stand, werte = _lade_stand(args.ziel, familie)
        if stand is None:
            print(f"{familie}: kein Vorher-Stand in {basis.pfad_neutral(args.ziel)} "
                  f"— erst --schreibe.")
            return 2
        staende[familie] = (stand, werte)

    # Umgebung: passt sie nicht, gilt der abgelegte Stand nicht mehr
    stempel = basis.umgebungsstempel()
    pruefung = {f: _stempel_vergleich(stand.get("umgebung") or {}, stempel)
                for f, (stand, _) in staende.items()}
    fremd = [f for f, (anders, _) in pruefung.items()
             if anders or not staende[f][0].get("umgebung")]
    for familie, (anders, unbekannt) in pruefung.items():
        if unbekannt and familie not in fremd:
            print(f"Hinweis: {familie}: der Vorher-Stand nennt {', '.join(unbekannt)} "
                  f"noch nicht im Umgebungsstempel; eine andere Fassung davon fiele "
                  f"erst nach dem nächsten --schreibe auf.")
    neu_ordner = None
    if fremd:
        for familie in fremd:
            anders = pruefung[familie][0] or ["kein Stempel"]
            print(f"{familie}: Umgebungsstempel weicht ab ({', '.join(anders)}) — "
                  f"der Vorher-Stand wird aus '{basis.VORHER_TAG}' neu gerechnet.")
        neu_ordner = _neu_rechnen(args, fremd)
        for familie in fremd:
            staende[familie] = _lade_stand(neu_ordner, familie)

    abweichungen: list = []      # (familie, probe, teil, text)
    offen: list = []             # Abweichungen ohne abgelegten alten Wert
    neue_werte: dict = {}
    n_gesamt = 0
    for familie, namen in wahl.items():
        stand, alt_werte = staende[familie]
        gewaehlt = None if namen is None else set(namen)
        proben, unstet, werte = rechne_familie(args, familie, gewaehlt)
        unbekannt = sorted((gewaehlt or set()) - set(proben) - set(stand["proben"]))
        if unbekannt:
            print(f"--nur: in {familie} gibt es keine Probe {unbekannt}.")
            return 2
        alle = sorted((set(proben) | set(stand["proben"])) if gewaehlt is None else gewaehlt)
        n_gesamt += len(alle)
        print(_zusammenfassung(familie, alle, unstet))
        for name in alle:
            alt = stand["proben"].get(name)
            neu = werte.get(name)
            if alt is None:
                abweichungen.append((familie, name, "-", "neu, steht nicht im Vorher-Stand",
                                     "neu"))
                continue
            if neu is None:
                abweichungen.append((familie, name, "-", "fehlt in proben_"
                                     f"{familie}.py, steht aber im Vorher-Stand", "fehlt"))
                continue
            if isinstance(neu, Exception):
                abweichungen.append((familie, name, "-",
                                     f"Fehler: {neu}", None))
                continue
            tol = unstet.get(name)
            for teil in sorted(set(alt["teile"]) | set(neu)):
                if teil not in neu:
                    abweichungen.append((familie, name, teil, "Teil fehlt", "fehlt"))
                    continue
                marke = "unstet" if tol is not None else neu[teil].hash
                if teil not in alt["teile"]:
                    abweichungen.append((familie, name, teil, "Teil ist neu", marke))
                    continue
                arr = neu[teil].arr
                if tol is None and marke == alt["teile"][teil]["hash"]:
                    continue
                schluessel = f"{name}/{teil}"
                if schluessel not in alt_werte:
                    neue_werte[(familie, schluessel)] = arr
                    offen.append((familie, name, teil, marke))
                    continue
                diff, text = groesste_differenz(alt_werte[schluessel], arr)
                if tol is not None:
                    if diff <= tol:
                        continue
                    text += f", Toleranz {tol:g}"
                abweichungen.append((familie, name, teil, text, marke))

    if offen:
        # der alte Wert liegt nicht in der .npz: aus dem Tag nachrechnen
        fams = sorted({f for f, _, _, _ in offen})
        nach_ordner = _neu_rechnen(args, fams)
        nach = {f: _lade_stand(nach_ordner, f) for f in fams}
        for familie, name, teil, marke in offen:
            stand_n, werte_n = nach[familie]
            schluessel = f"{name}/{teil}"
            if schluessel not in werte_n:
                abweichungen.append((familie, name, teil,
                                     "Hash verschieden; die Probe gibt es im "
                                     "Vorher-Stand des Tags nicht", marke))
                continue
            _diff, text = groesste_differenz(werte_n[schluessel],
                                             neue_werte[(familie, schluessel)])
            alt_hash = staende[familie][0]["proben"][name]["teile"][teil]["hash"]
            if stand_n["proben"][name]["teile"][teil]["hash"] != alt_hash:
                text += ("; Achtung: der aus dem Tag nachgerechnete Wert trägt "
                         "nicht den abgelegten Hash")
            abweichungen.append((familie, name, teil, text + " (alter Wert nachgerechnet)",
                                 marke))

    benutzt: set = set()
    unerklaert: list = []
    vorlage: dict = {}
    n_erwartet = 0
    for familie, name, teil, text, marke in sorted(abweichungen, key=lambda a: a[:4]):
        stelle = f"{familie}.{name}/{teil}"
        key = _eintrag_fuer(erwartet, stelle)
        eintrag = erwartet.get(key, {})
        if marke is not None:
            vorlage[stelle] = {"grund": eintrag.get("grund", ""), "hash": marke}
        if key is not None:
            benutzt.add(key)
            if marke is not None and eintrag["hash"] == marke:
                n_erwartet += 1
                print(f"ERWARTET ({eintrag['grund']}) {familie}: Probe {name}, "
                      f"Teil {teil}: {text}")
                continue
            text += (f"; --erwartet '{key}' nennt einen anderen Stand "
                     f"({eintrag['hash'][:12]}, der neue Wert trägt {str(marke)[:12]})"
                     if marke is not None else
                     f"; --erwartet '{key}' kann einen Fehler nicht freigeben")
        unerklaert.append((familie, name))
        print(f"ABWEICHUNG {familie}: Probe {name}, Teil {teil}: {text}")
    for key in sorted(set(erwartet) - benutzt):
        print(f"Hinweis: erwartete Abweichung '{key}' ist nicht eingetreten.")
    if args.erwartet_vorlage:
        with open(args.erwartet_vorlage, "w", encoding="utf-8") as fh:
            json.dump(vorlage, fh, indent=1, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
        print(f"Vorlage für --erwartet geschrieben: {len(vorlage)} Teile "
              f"({basis.pfad_neutral(args.erwartet_vorlage)}); der Grund ist je "
              f"Eintrag nachzutragen.")
    if unerklaert:
        print(f"{len(set(unerklaert))} von {n_gesamt} Proben weichen ab "
              f"({len(unerklaert)} Teile"
              f"{f', {n_erwartet} weitere wie erwartet' if n_erwartet else ''}).")
        return 1
    print(f"alle Proben gleich ({n_gesamt} Proben in {len(wahl)} "
          f"{'Familie' if len(wahl) == 1 else 'Familien'}"
          f"{', Vorher-Stand neu gerechnet' if neu_ordner else ''}"
          f"{f', davon {n_erwartet} Teile mit erwarteter Abweichung' if n_erwartet else ''})")
    return 0


# -------------------------------------------------------------- Selbsttest

_DEMO_RECHNUNG = '''\
"""Demo-Rechnung für den Selbsttest der Numerik-Probe."""
import numpy as np

FAKTOR = 2.0


def kurve(n):
    return np.sin(FAKTOR * np.linspace(0.0, 1.0, n))


def feld():
    rng = np.random.default_rng(7)
    return (rng.uniform(0.0, 1.0, size=(400, 1000)) * FAKTOR).astype(np.float64)
'''

_DEMO_PROBEN = '''\
"""Demo-Proben für den Selbsttest der Numerik-Probe."""
import numpy as np

from core import rechnung


def _kurve():
    werte = rechnung.kurve(1000)
    return {"werte": werte, "summe": float(werte.sum()),
            "roh": rechnung.kurve(8).astype(np.float32).tobytes()}


def _feld():
    return {"feld": rechnung.feld()}          # 3,2 MB: wird nicht abgelegt


def _rauschen():
    return {"wert": rechnung.kurve(5) + np.random.default_rng().normal(0, 1e-9, 5)}


PROBEN = {"rechnung.kurve": _kurve, "rechnung.feld": _feld,
          "rauschen.kurve": _rauschen}
UNSTET = {"rauschen.kurve": 1e-6}
'''

#: Eine Probe, deren Prozess durch ein Signal endet — beim ersten Mal oder immer.
_DEMO_ABSTURZ = '''\
"""Demo-Probe für den Selbsttest: der Prozess endet durch ein Signal."""
import os
import signal

import basis


def _absturz():
    ordner = basis.arbeitsordner("liegengeblieben")
    marke = os.environ["NUMERIK_DEMO_MARKE"]
    if os.environ.get("NUMERIK_DEMO_IMMER") or not os.path.exists(marke):
        open(marke, "w").close()
        os.kill(os.getpid(), signal.SIGKILL)
    return {"wert": 1.5, "ordner_da": int(os.path.isdir(ordner))}


PROBEN = {"absturz.probe": _absturz}
'''


def _selbsttest() -> int:
    ordner = basis.arbeitsordner("numerik_selbst")
    a = os.path.join(ordner, "stand_a")
    for paket in ("core", "ui"):
        os.makedirs(os.path.join(a, paket))
        open(os.path.join(a, paket, "__init__.py"), "w").close()
    with open(os.path.join(a, "core", "rechnung.py"), "w", encoding="utf-8") as fh:
        fh.write(_DEMO_RECHNUNG)
    proben = os.path.join(ordner, "proben")
    os.makedirs(proben)
    with open(os.path.join(proben, "proben_demo.py"), "w", encoding="utf-8") as fh:
        fh.write(_DEMO_PROBEN)
    # die veränderte Kopie: ein Faktor in der achten Stelle
    b = os.path.join(ordner, "stand_b")
    shutil.copytree(a, b)
    with open(os.path.join(b, "core", "rechnung.py"), encoding="utf-8") as fh:
        text = fh.read()
    assert "FAKTOR = 2.0\n" in text
    with open(os.path.join(b, "core", "rechnung.py"), "w", encoding="utf-8") as fh:
        fh.write(text.replace("FAKTOR = 2.0\n", "FAKTOR = 2.0000001\n"))
    ziel = os.path.join(ordner, "vorher")

    # Arbeitsordner der Läufe und ihrer Kinder entstehen in diesem Ordner
    umgebung = dict(os.environ, **{basis.ELTERN_VARIABLE: ordner,
                                   "NUMERIK_DEMO_MARKE": os.path.join(ordner, "marke")})
    umgebung.pop("NUMERIK_DEMO_IMMER", None)
    umgebung.pop("SUPER360_UMBAU_BEHALTEN", None)

    def lauf(*argumente, wurzel=a, proben=proben, **zusatz):
        befehl = [sys.executable, os.path.abspath(__file__), "--wurzel", wurzel,
                  "--proben-ordner", proben, "--ziel", ziel, "--vorher-wurzel", a,
                  *argumente]
        p = subprocess.run(befehl, capture_output=True, text=True,
                           env=dict(umgebung, **zusatz))
        return p.returncode, p.stdout + p.stderr

    def reste():
        return sorted(n for n in os.listdir(ordner)
                      if n not in ("stand_a", "stand_b", "proben", "absturz", "vorher",
                                   "marke")
                      and not n.endswith(".json"))

    code, aus = lauf("--schreibe")
    assert code == 0 and "3 Proben (rauschen 1, rechnung 2)" in aus, aus
    assert "unstet: rauschen.kurve (Toleranz 1e-06)" in aus, aus
    with open(os.path.join(ziel, "numerik_demo.json"), encoding="utf-8") as fh:
        stand = json.load(fh)
    teile = stand["proben"]["rechnung.kurve"]["teile"]
    assert sorted(teile) == ["roh", "summe", "werte"] and teile["roh"]["art"] == "bytes"
    assert teile["werte"]["abgelegt"] and teile["werte"]["shape"] == [1000]
    assert not stand["proben"]["rechnung.feld"]["teile"]["feld"]["abgelegt"]
    assert stand["umgebung"] == basis.umgebungsstempel()
    print("schreibe: Hashes, Umgebungsstempel und Ausgaben bis 2 MB abgelegt")

    code, aus = lauf("--schreibe")
    assert code == 2 and "--ersetzen" in aus, aus
    code, aus = lauf("--schreibe", "--ersetzen")
    assert code == 0, aus
    print("schreibe: vorhandenes Ziel verweigert, mit --ersetzen angenommen")

    for _ in range(2):
        code, aus = lauf("--vergleiche")
        assert code == 0 and "alle Proben gleich" in aus, aus
    print("vergleiche: zweimal gleich (unstete Probe innerhalb der Toleranz)")

    code, aus = lauf("--vergleiche", wurzel=b)
    assert code == 1 and "alle Proben gleich" not in aus, aus
    zeilen = [z for z in aus.splitlines() if z.startswith("ABWEICHUNG")]
    assert len(zeilen) == 4, aus
    assert any("Probe rechnung.kurve, Teil werte: größte Differenz" in z for z in zeilen), aus
    assert any("Probe rechnung.feld, Teil feld: größte Differenz" in z
               and "nachgerechnet" in z for z in zeilen), aus
    assert not any("rauschen" in z for z in zeilen), aus
    print("vergleiche: veränderte Kopie gemeldet —")
    for z in zeilen:
        print("   " + z[:150])

    code, aus = lauf("--vergleiche", "--nur", "demo.rechnung.feld", wurzel=b)
    assert code == 1 and "1 Proben" in aus and "rechnung.kurve" not in aus, aus
    code, aus = lauf("--vergleiche", "--nur", "gibtsnicht")
    assert code == 2, aus
    print("vergleiche: --nur wählt Familie und einzelne Probe")

    # --erwartet: an den neuen Wert gebunden
    vorlage = os.path.join(ordner, "vorlage.json")
    erwartet = os.path.join(ordner, "erwartet.json")

    def mit_erwartet(eintraege, wurzel=b):
        with open(erwartet, "w", encoding="utf-8") as fh:
            json.dump(eintraege, fh)
        return lauf("--vergleiche", "--erwartet", erwartet, wurzel=wurzel)

    code, aus = lauf("--vergleiche", "--erwartet-vorlage", vorlage, wurzel=b)
    assert code == 1, aus
    with open(vorlage, encoding="utf-8") as fh:
        vorl = json.load(fh)
    assert sorted(vorl) == ["demo.rechnung.feld/feld", "demo.rechnung.kurve/roh",
                            "demo.rechnung.kurve/summe", "demo.rechnung.kurve/werte"], vorl
    code, aus = mit_erwartet(vorl)
    assert code == 2 and "der Grund fehlt" in aus, aus
    code, aus = mit_erwartet({"demo.rechnung.kurve/werte": "Faktor geändert"})
    assert code == 2 and "verlangt ist" in aus, aus
    gebunden = {k: {"grund": "Faktor geändert", "hash": v["hash"]} for k, v in vorl.items()}
    gebunden["demo.rechnung.gibtsnicht/teil"] = {"grund": "tritt nicht ein", "hash": "fehlt"}
    code, aus = mit_erwartet(gebunden)
    assert code == 0 and aus.count("ERWARTET (Faktor geändert) demo: Probe") == 4 \
        and "ABWEICHUNG" not in aus and "davon 4 Teile mit erwarteter Abweichung" in aus \
        and "'demo.rechnung.gibtsnicht/teil' ist nicht eingetreten" in aus, aus
    code, aus = mit_erwartet(gebunden, wurzel=a)
    assert code == 0 and aus.count("ist nicht eingetreten") == 5, aus
    # ein anderer neuer Wert als der erwartete bleibt ein Fehler
    falsch = dict(gebunden)
    falsch["demo.rechnung.kurve/summe"] = {"grund": "alter Stand", "hash": "f" * 64}
    del falsch["demo.rechnung.kurve/roh"]
    code, aus = mit_erwartet(falsch)
    zeilen = [z for z in aus.splitlines() if z.startswith("ABWEICHUNG")]
    assert code == 1 and len(zeilen) == 2 and "nennt einen anderen Stand" in aus \
        and "1 von 3 Proben weichen ab (2 Teile, 2 weitere wie erwartet)" in aus, aus
    print("vergleiche: --erwartet lässt an den neuen Wert gebundene Abweichungen gelten; "
          "ein anderer Wert an derselben Stelle bleibt ein Fehler")

    js = os.path.join(ziel, "numerik_demo.json")
    del stand["umgebung"]["pyproj"]
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(stand, fh)
    code, aus = lauf("--vergleiche")
    assert code == 0 and "nennt pyproj noch nicht im Umgebungsstempel" in aus \
        and "neu gerechnet" not in aus, aus
    stand["umgebung"]["numpy"] = "0.0"
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(stand, fh)
    code, aus = lauf("--vergleiche")
    assert code == 0 and "Umgebungsstempel weicht ab (numpy)" in aus \
        and "Vorher-Stand neu gerechnet" in aus, aus
    code, aus = lauf("--vergleiche", wurzel=b)
    assert code == 1 and "Umgebungsstempel weicht ab" in aus, aus
    print("vergleiche: fremder Umgebungsstempel — Vorher-Stand im selben Lauf neu gerechnet; "
          "ein dort noch nicht genannter Schlüssel gibt nur einen Hinweis")
    assert not reste(), reste()

    # Kindprozess endet durch ein Signal: einmal wiederholen, nichts bleibt liegen
    absturz = os.path.join(ordner, "absturz")
    os.makedirs(absturz)
    with open(os.path.join(absturz, "proben_absturz.py"), "w", encoding="utf-8") as fh:
        fh.write(_DEMO_ABSTURZ)
    code, aus = lauf("--schreibe", proben=absturz)
    assert code == 0 and "Signal 9; die Familie wird einmal wiederholt" in aus \
        and "Familie absturz: 1 Proben" in aus, aus
    assert not reste(), reste()
    code, aus = lauf("--vergleiche", proben=absturz)
    assert code == 0 and "alle Proben gleich (1 Proben in 1 Familie)" in aus \
        and "Signal" not in aus, aus
    code, aus = lauf("--vergleiche", proben=absturz, NUMERIK_DEMO_IMMER="1")
    assert code == 2 and "auch beim zweiten Versuch" in aus \
        and "alle Proben gleich" not in aus, aus
    assert not reste(), reste()
    print("Kindprozess: Ende durch Signal einmal wiederholt, beim zweiten Mal Fehler; "
          "kein Arbeitsordner bleibt liegen")
    print("numerik_probe SELFTEST OK")
    return 0


# ------------------------------------------------------------------ Aufruf

def main(argv=None) -> int:
    parser = basis.argumente(argparse.ArgumentParser(
        description="Numerik-Proben gegen den Vorher-Stand"))
    modus = parser.add_mutually_exclusive_group()
    modus.add_argument("--schreibe", action="store_true")
    modus.add_argument("--vergleiche", action="store_true")
    modus.add_argument("--selbsttest", action="store_true")
    parser.add_argument("--ersetzen", action="store_true")
    parser.add_argument("--nur", default=None, metavar="FAMILIE,…")
    parser.add_argument("--erwartet", default=None, metavar="JSON")
    parser.add_argument("--erwartet-vorlage", default=None, metavar="JSON")
    parser.add_argument("--kind", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--proben-ordner", default=basis.WERKZEUG_ORDNER)
    parser.add_argument("--ziel", default=basis.VORHER_ORDNER)
    parser.add_argument("--vorher-wurzel", default=None)
    parser.add_argument("--ohne-grenze", action="store_true")
    args = parser.parse_args(argv)
    args.proben_ordner = os.path.abspath(args.proben_ordner)
    args.ziel = os.path.abspath(args.ziel)
    for name in ("vorher_wurzel", "erwartet", "erwartet_vorlage"):
        if getattr(args, name):
            setattr(args, name, os.path.abspath(getattr(args, name)))
    if args.kind:
        return _kind(args)
    if not (args.schreibe or args.vergleiche or args.selbsttest):
        parser.error("einer der Modi --schreibe, --vergleiche, --selbsttest ist verlangt")
    if args.selbsttest:
        return _selbsttest()

    basis.wurzel_setzen(args.wurzel)
    familien = familien_finden(args.proben_ordner)
    if not familien:
        print(f"Keine proben_*.py in {basis.pfad_neutral(args.proben_ordner)}.")
        return 2
    wahl = _wahl(args.nur, familien)
    try:
        if args.schreibe:
            if any(n is not None for n in wahl.values()):
                print("--schreibe nimmt nur ganze Familien.")
                return 2
            grenze = None if args.ohne_grenze else GRENZE_BYTES
            codes = [schreibe(args, f, grenze) for f in wahl]
            return max(codes)
        return vergleiche(args, familien, wahl)
    except RuntimeError as exc:
        print(f"FEHLER: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())

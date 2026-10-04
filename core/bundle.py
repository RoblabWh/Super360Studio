"""Ein Projekt als Ordner aus- und wieder einpacken.

Ein Projekt von Super360 Studio besteht aus dem Rosbag als Eingang und einem
Cache-Ordner mit allem, was daraus berechnet wurde: der FAST-LIO-Aufzeichnung,
den Farbebenen, den gestitchten Panoramen, dem Arbeitsordner der
Maeander-Pipeline und den kleinen JSON-Dateien fuer GPS, Extrinsik und
Einstellungen. Der Cache liegt ausserhalb des Repos und traegt einen Namen aus
einem Pfad-Hash — von aussen ist er nicht zu finden und nicht mitzunehmen.

:func:`export_project` legt daraus einen Ordner an, den man weiterreichen kann,
:func:`import_project` macht daraus wieder ein Projekt. Das Bag bleibt dabei
optional: es ist der Eingang, nicht das Ergebnis, und mit 24 GB der grosse
Brocken. Ohne Bag bleiben die 3D-Karte, die Farben, Messen, Hoehenschnitt und
der Export erhalten; das 360-Video und ein erneutes Einfaerben brauchen es.

Layout des Ordners::

    <ziel>/
      super360.json      Manifest: was drin ist, woher es kommt, Kennzahlen
      projekt/           1:1 der Projektordner (recording/, colors*/, pano_*/, …)
      bags/              nur wenn mitgenommen
      kalibrierung/      die verwendete calibration.json

Qt-frei.
"""

from __future__ import annotations

import json
import os
import shutil
import time

from core.gemeinsam import write_json_atomic

MANIFEST = "super360.json"
FORMAT = "super360studio-projekt"
VERSION = 1

#: Teile eines Projekts: Schluessel -> (Anzeigename, Pflicht)
TEILE = {
    "recording": ("Punktwolke (FAST-LIO-Aufzeichnung)", True),
    "colors": ("Farbebenen", False),
    "panos": ("gestitchte Panoramen", False),
    "meander": ("Mäander-Arbeitsordner (COLMAP, Bilder, Lage)", False),
    "bags": ("Rosbags (Rohdaten)", False),
}

_KLEINKRAM = ("gps.json", "extrinsic.json", "settings.json", "exploration.json")


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def fmt_size(n: int) -> str:
    """Bytes menschenlesbar, deutsche Schreibweise."""
    for einheit, teiler in (("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20),
                            ("kB", 1 << 10)):
        if n >= teiler:
            return f"{n / teiler:.1f} {einheit}".replace(".", ",")
    return f"{n} B"


def describe(project, bag_paths=None) -> dict:
    """Was ist da und wie gross? Grundlage fuer die Auswahl vor dem Export."""
    d = project.dir
    teile: dict = {}
    rec = os.path.join(d, "recording")
    teile["recording"] = {"da": os.path.isdir(rec), "bytes": _dir_size(rec)}

    farben = 0
    ebenen = []
    for key, name in project.LAYERS.items():
        p = os.path.join(d, name)
        if os.path.isdir(p):
            farben += _dir_size(p)
            if project.has_layer(key):
                ebenen.append(key)
    teile["colors"] = {"da": bool(ebenen), "bytes": farben, "ebenen": ebenen}

    panos = 0
    breiten = []
    for name in os.listdir(d) if os.path.isdir(d) else []:
        if name.startswith("pano_") and os.path.isdir(os.path.join(d, name)):
            panos += _dir_size(os.path.join(d, name))
            try:
                breiten.append(int(name.split("_", 1)[1]))
            except ValueError:
                pass
    teile["panos"] = {"da": bool(breiten), "bytes": panos, "breiten": sorted(breiten)}

    m = os.path.join(d, "meander")
    teile["meander"] = {"da": os.path.isdir(m), "bytes": _dir_size(m)}

    bags = [b for b in (bag_paths or []) if b and os.path.exists(b)]
    teile["bags"] = {"da": bool(bags), "bytes": sum(_dir_size(b) for b in bags),
                     "pfade": bags}
    return teile


def _copy_tree(src: str, dst: str, zaehler: dict, progress=None, cancel=None) -> None:
    """Ordner kopieren und dabei die Bytes mitzaehlen (fuer den Fortschritt)."""
    for root, dirs, files in os.walk(src):
        if cancel is not None and cancel():
            raise RuntimeError("Abgebrochen")
        rel = os.path.relpath(root, src)
        ziel = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(ziel, exist_ok=True)
        for name in files:
            q = os.path.join(root, name)
            try:
                shutil.copy2(q, os.path.join(ziel, name))
                zaehler["ist"] += os.path.getsize(q)
            except OSError as exc:
                raise RuntimeError(f"{q} nicht kopierbar: {exc}") from exc
            if progress is not None and zaehler["soll"] > 0:
                progress(min(zaehler["ist"] / zaehler["soll"], 1.0),
                         f"Kopiere {name} — {fmt_size(zaehler['ist'])} von "
                         f"{fmt_size(zaehler['soll'])}")
        dirs.sort()


def pruefe_ziel(dest: str) -> tuple:
    """Taugt ``dest`` als Zielordner fuer den Export? -> (ok, zustand).

    ``zustand``: "neu" (wird angelegt), "leer", "projekt" (enthaelt schon ein
    exportiertes Projekt, wird ueberschrieben), "fremd" (nicht leer und ohne
    Manifest) oder "datei" (kein Ordner). Abgelehnt werden die beiden letzten.
    """
    if not os.path.exists(dest):
        return True, "neu"
    if not os.path.isdir(dest):
        return False, "datei"
    if not os.listdir(dest):
        return True, "leer"
    if os.path.isfile(os.path.join(dest, MANIFEST)):
        return True, "projekt"
    return False, "fremd"


def export_project(project, dest: str, teile: dict, bag_paths=None,
                   calib_path: str | None = None, meta_extra: dict | None = None,
                   progress=None, cancel=None) -> dict:
    """Projekt nach ``dest`` schreiben. ``teile`` ist {schluessel: bool}.

    Der Zielordner darf leer sein oder neu; ist er nicht leer und enthaelt kein
    Manifest, wird abgelehnt — sonst schuettet der Export fremde Daten zu.
    """
    dest = os.path.abspath(dest)
    ok, _zustand = pruefe_ziel(dest)
    if not ok:
        raise RuntimeError(
            f"'{os.path.basename(dest)}' ist nicht leer und enthält kein "
            f"Super360-Projekt. Bitte einen leeren oder neuen Ordner wählen.")
    os.makedirs(dest, exist_ok=True)

    info = describe(project, bag_paths)
    gewaehlt = {k: bool(teile.get(k, False)) and info[k]["da"] for k in TEILE}
    gewaehlt["recording"] = info["recording"]["da"]      # ohne die geht nichts
    if not gewaehlt["recording"]:
        raise RuntimeError("Ohne berechnete Karte gibt es nichts zu exportieren.")
    soll = sum(info[k]["bytes"] for k, an in gewaehlt.items() if an)
    zaehler = {"ist": 0, "soll": max(soll, 1)}

    ziel_projekt = os.path.join(dest, "projekt")
    os.makedirs(ziel_projekt, exist_ok=True)
    _copy_tree(os.path.join(project.dir, "recording"),
               os.path.join(ziel_projekt, "recording"), zaehler, progress, cancel)
    if gewaehlt["colors"]:
        for key in info["colors"]["ebenen"]:
            name = project.LAYERS[key]
            _copy_tree(os.path.join(project.dir, name),
                       os.path.join(ziel_projekt, name), zaehler, progress, cancel)
    if gewaehlt["panos"]:
        for w in info["panos"]["breiten"]:
            _copy_tree(os.path.join(project.dir, f"pano_{w}"),
                       os.path.join(ziel_projekt, f"pano_{w}"), zaehler,
                       progress, cancel)
    if gewaehlt["meander"]:
        _copy_tree(os.path.join(project.dir, "meander"),
                   os.path.join(ziel_projekt, "meander"), zaehler, progress, cancel)
    for name in _KLEINKRAM:
        q = os.path.join(project.dir, name)
        if os.path.isfile(q):
            shutil.copy2(q, os.path.join(ziel_projekt, name))

    if calib_path and os.path.isfile(calib_path):
        os.makedirs(os.path.join(dest, "kalibrierung"), exist_ok=True)
        shutil.copy2(calib_path, os.path.join(dest, "kalibrierung",
                                              os.path.basename(calib_path)))

    quellen = []
    for b in (bag_paths or []):
        eintrag = {"bag": str(b), "enthalten": False}
        if gewaehlt["bags"] and os.path.exists(b):
            if progress is not None:
                progress(zaehler["ist"] / zaehler["soll"],
                         f"Kopiere Rosbag {os.path.basename(b)} …")
            ziel = os.path.join(dest, "bags", os.path.basename(b))
            if os.path.isdir(b):
                _copy_tree(b, ziel, zaehler, progress, cancel)
            else:
                os.makedirs(os.path.dirname(ziel), exist_ok=True)
                shutil.copy2(b, ziel)
                zaehler["ist"] += os.path.getsize(b)
            eintrag["enthalten"] = True
            eintrag["im_ordner"] = os.path.join("bags", os.path.basename(b))
        quellen.append(eintrag)

    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "erstellt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "projekt": {"name": project.bag_name, "bag": project.bag_path},
        "quellen": quellen,
        "inhalt": {k: bool(v) for k, v in gewaehlt.items()},
        "ebenen": info["colors"]["ebenen"] if gewaehlt["colors"] else [],
        "panos": info["panos"]["breiten"] if gewaehlt["panos"] else [],
        "bytes": zaehler["ist"],
        **(meta_extra or {}),
    }
    with open(os.path.join(dest, MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    if progress is not None:
        progress(1.0, f"Export fertig: {fmt_size(zaehler['ist'])}")
    return manifest


def read_manifest(src: str) -> dict:
    """Manifest lesen und pruefen; klare Meldung, wenn der Ordner keiner ist."""
    p = os.path.join(src, MANIFEST)
    if not os.path.isfile(p):
        raise RuntimeError(
            f"In '{os.path.basename(src)}' liegt keine {MANIFEST} — das ist kein "
            f"exportiertes Super360-Projekt.")
    try:
        with open(p, encoding="utf-8") as fh:
            m = json.load(fh)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{MANIFEST} nicht lesbar: {exc}") from exc
    if m.get("format") != FORMAT:
        raise RuntimeError(f"Unbekanntes Format: {m.get('format')!r}")
    if int(m.get("version", 0)) > VERSION:
        raise RuntimeError(
            f"Das Projekt wurde mit einer neueren Fassung geschrieben "
            f"(Version {m.get('version')}, hier bis {VERSION}).")
    if not os.path.isdir(os.path.join(src, "projekt", "recording")):
        raise RuntimeError("Im Ordner fehlt projekt/recording — unvollständig.")
    return m


def bag_paths_after_import(src: str, manifest: dict) -> list:
    """Wo die Bags nach dem Import liegen.

    Mitgenommene Bags bleiben im Exportordner liegen und werden von dort
    referenziert — sie ein zweites Mal zu kopieren waere bei 24 GB je Bag
    Verschwendung. Wandert der Ordner weg, faellt nur das 360-Video und ein
    erneutes Einfaerben aus, die Karte bleibt.
    """
    out = []
    for q in manifest.get("quellen") or []:
        if q.get("enthalten") and q.get("im_ordner"):
            out.append(os.path.join(os.path.abspath(src), q["im_ordner"]))
        else:
            out.append(q.get("bag"))
    return out


def oeffnen(src: str) -> dict:
    """Exportordner an Ort und Stelle oeffnen — ohne Kopie, ohne Rueckfrage.

    Gearbeitet wird direkt in ``src/projekt``: jede Aenderung (Einstellungen,
    Ausrichtung, Einfaerbung) landet sofort dort. Frueher wurde der Ordner in
    den Cache kopiert und bei einem vorhandenen Projekt gefragt, ob es
    ersetzt werden soll; die Kopie ist entfallen, damit auch die Frage.

    Die Bagpfade in der ``meta.json`` werden auf das gezogen, was hier liegt —
    mitgenommene Bags im Ordner, sonst die urspruenglichen Pfade. Das geht
    auch mehrfach und auf einem anderen Rechner, gesucht wird ueber den
    Ordnernamen des Bags.
    """
    from core.project import Project, lies_aufzeichnungs_meta  # noqa: PLC0415

    src = os.path.abspath(src)
    manifest = read_manifest(src)
    projekt_dir = os.path.join(src, "projekt")
    if not os.path.isdir(os.path.join(projekt_dir, "recording")):
        raise RuntimeError(f"In '{src}' liegt kein Projekt (projekt/recording fehlt).")
    neue = bag_paths_after_import(src, manifest)
    alte = [q.get("bag") for q in (manifest.get("quellen") or [])]
    nach_name = {os.path.basename(str(a).rstrip("/")): n
                 for a, n in zip(alte, neue) if a and n}
    rec_dir = os.path.join(projekt_dir, "recording")
    try:
        meta = lies_aufzeichnungs_meta(rec_dir)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"meta.json der Aufzeichnung nicht lesbar: {exc}") from exc

    def neu(pfad):
        if not pfad:
            return pfad
        ziel = nach_name.get(os.path.basename(str(pfad).rstrip("/")))
        return ziel if ziel and ziel != pfad and os.path.exists(ziel) else pfad

    geaendert = False
    if neu(meta.get("bag")) != meta.get("bag"):
        meta["bag"] = neu(meta["bag"])
        geaendert = True
    for s in meta.get("sources") or []:
        if neu(s.get("bag")) != s.get("bag"):
            s["bag"] = neu(s["bag"])
            geaendert = True
    if geaendert:
        write_json_atomic(os.path.join(rec_dir, "meta.json"), meta)
    project = Project.from_dir(projekt_dir)
    name = (manifest.get("projekt") or {}).get("name")
    if name:
        project.bag_name = str(name)
    bags = [neu(s.get("bag")) for s in (meta.get("sources") or [])] or [meta.get("bag")]
    return {"manifest": manifest, "project": project, "bags": bags,
            "fehlende_bags": [b for b in bags if b and not os.path.exists(b)],
            "zusammengefuehrt": len(meta.get("sources") or []) >= 2}


def import_project(src: str, project, progress=None, cancel=None) -> dict:
    """Ordner ``src`` in ``project`` einspielen und die Bagpfade nachziehen.

    ``project`` ist ein :class:`core.project.Project` fuer den Zielort; sein
    Verzeichnis wird angelegt und ueberschrieben.
    """
    src = os.path.abspath(src)
    manifest = read_manifest(src)
    quelle = os.path.join(src, "projekt")
    soll = _dir_size(quelle)
    zaehler = {"ist": 0, "soll": max(soll, 1)}
    os.makedirs(project.dir, exist_ok=True)
    _copy_tree(quelle, project.dir, zaehler, progress, cancel)

    # Bagpfade in der meta.json auf den Ort zeigen lassen, an dem sie jetzt
    # wirklich liegen — sonst zeigt der Import ins Leere des fremden Rechners.
    neue = bag_paths_after_import(src, manifest)
    meta_p = os.path.join(project.dir, "recording", "meta.json")
    try:
        with open(meta_p, encoding="utf-8") as fh:
            meta = json.load(fh)
        alte = [q.get("bag") for q in (manifest.get("quellen") or [])]
        zuordnung = {a: n for a, n in zip(alte, neue) if a and n}
        if meta.get("bag") in zuordnung:
            meta["bag"] = zuordnung[meta["bag"]]
        for s in meta.get("sources") or []:
            if s.get("bag") in zuordnung:
                s["bag"] = zuordnung[s["bag"]]
        tmp = meta_p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        os.replace(tmp, meta_p)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"meta.json der Aufzeichnung nicht anpassbar: {exc}") from exc

    fehlend = [b for b in neue if b and not os.path.exists(b)]
    if progress is not None:
        progress(1.0, f"Import fertig: {fmt_size(zaehler['ist'])}")
    return {"manifest": manifest, "bags": neue, "fehlende_bags": fehlend,
            "bytes": zaehler["ist"], "project": project}


if __name__ == "__main__":
    import tempfile
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import numpy as np
    from core.project import Project

    wurzel = tempfile.mkdtemp(prefix="bundletest_")
    bag_a = os.path.join(wurzel, "rosbag_2026-01-01_10-00-00")
    os.makedirs(bag_a)
    open(os.path.join(bag_a, "metadata.yaml"), "w").close()
    with open(os.path.join(bag_a, "daten_0.db3"), "wb") as fh:
        fh.write(b"x" * 4096)

    cache = os.path.join(wurzel, "cache")
    proj = Project(bag_a, cache_root=cache)
    rec = os.path.join(proj.recording_dir())
    np.zeros((30, 3), np.float32).tofile(os.path.join(rec, "points.bin"))
    np.zeros(30, np.float32).tofile(os.path.join(rec, "intensity.bin"))
    np.save(os.path.join(rec, "offsets.npy"), np.arange(0, 31, 10, dtype=np.int64))
    np.save(os.path.join(rec, "stamps.npy"), np.arange(3, dtype=np.float64))
    p = np.zeros((3, 7)); p[:, 6] = 1.0
    np.save(os.path.join(rec, "poses.npy"), p)
    json.dump({"bag": bag_a, "n_scans": 3, "n_points": 30},
              open(os.path.join(rec, "meta.json"), "w"))
    for key in ("onboard", "meander_rgb"):
        d = proj.layer_dir(key)
        np.zeros((30, 3), np.uint8).tofile(os.path.join(d, "colors.bin"))
        np.ones(30, np.uint8).tofile(os.path.join(d, "valid.bin"))
        json.dump({"quelle": key}, open(os.path.join(d, "meta.json"), "w"))
    json.dump({"a": 1}, open(proj.settings_json(), "w"))
    os.makedirs(os.path.join(proj.dir, "pano_1920"), exist_ok=True)
    open(os.path.join(proj.dir, "pano_1920", "index.json"), "w").write("{}")

    print("== Test 1: describe ==")
    info = describe(proj, [bag_a])
    for k, v in info.items():
        print(f"  {k:10s} da={v['da']!s:5s} {fmt_size(v['bytes'])}")
    assert info["recording"]["da"] and info["colors"]["ebenen"] == ["onboard", "meander_rgb"]
    assert info["panos"]["breiten"] == [1920] and info["bags"]["da"]

    print("== Test 2: Export ohne Bag, ohne Panos ==")
    ziel = os.path.join(wurzel, "export")
    m = export_project(proj, ziel, {"colors": True, "panos": False,
                                    "meander": False, "bags": False},
                       bag_paths=[bag_a], progress=lambda f, s: None)
    assert os.path.isfile(os.path.join(ziel, MANIFEST))
    assert os.path.isdir(os.path.join(ziel, "projekt", "recording"))
    assert os.path.isdir(os.path.join(ziel, "projekt", "colors_meander_rgb"))
    assert not os.path.exists(os.path.join(ziel, "projekt", "pano_1920"))
    assert not os.path.exists(os.path.join(ziel, "bags"))
    assert os.path.isfile(os.path.join(ziel, "projekt", "settings.json"))
    print(f"  {fmt_size(m['bytes'])}, Ebenen {m['ebenen']}, Bags enthalten="
          f"{m['quellen'][0]['enthalten']}")

    print("== Test 3: nicht leerer fremder Ordner wird abgelehnt ==")
    fremd = os.path.join(wurzel, "fremd")
    os.makedirs(fremd)
    open(os.path.join(fremd, "wichtig.txt"), "w").write("nicht loeschen")
    try:
        export_project(proj, fremd, {}, bag_paths=[bag_a])
    except RuntimeError as exc:
        print(f"  abgelehnt: {str(exc)[:60]}…")
    else:
        raise AssertionError("fremder Ordner wurde nicht abgelehnt")
    assert os.path.isfile(os.path.join(fremd, "wichtig.txt"))

    print("== Test 3b: pruefe_ziel, eine Datei als Ziel ==")
    leer = os.path.join(wurzel, "leer")
    os.makedirs(leer)
    datei = os.path.join(fremd, "wichtig.txt")
    zustaende = {p: pruefe_ziel(p) for p in (os.path.join(wurzel, "neu"), leer, ziel,
                                             fremd, datei)}
    assert list(zustaende.values()) == [(True, "neu"), (True, "leer"), (True, "projekt"),
                                        (False, "fremd"), (False, "datei")], zustaende
    try:
        export_project(proj, datei, {}, bag_paths=[bag_a])
    except RuntimeError as exc:
        print(f"  Datei abgelehnt: {str(exc)[:60]}…")
    else:
        raise AssertionError("Datei als Ziel wurde nicht abgelehnt")
    assert open(datei).read() == "nicht loeschen"

    print("== Test 4: Import in einen anderen Cache ==")
    cache2 = os.path.join(wurzel, "cache2")
    proj2 = Project(bag_a, cache_root=cache2)
    res = import_project(ziel, proj2, progress=lambda f, s: None)
    assert proj2.has_recording() and proj2.has_layer("onboard")
    assert proj2.available_layers() == ["onboard", "meander_rgb"]
    meta = json.load(open(os.path.join(proj2.recording_dir(), "meta.json")))
    print(f"  {fmt_size(res['bytes'])}, Bag zeigt auf {meta['bag']}")
    assert meta["bag"] == bag_a, meta["bag"]
    assert not res["fehlende_bags"]

    print("== Test 5: Export MIT Bag, Pfad zeigt danach in den Ordner ==")
    ziel2 = os.path.join(wurzel, "export2")
    export_project(proj, ziel2, {"colors": True, "bags": True},
                   bag_paths=[bag_a], progress=lambda f, s: None)
    assert os.path.isfile(os.path.join(ziel2, "bags",
                                       os.path.basename(bag_a), "daten_0.db3"))
    cache3 = os.path.join(wurzel, "cache3")
    proj3 = Project(os.path.join(ziel2, "bags", os.path.basename(bag_a)),
                    cache_root=cache3)
    res3 = import_project(ziel2, proj3, progress=lambda f, s: None)
    meta3 = json.load(open(os.path.join(proj3.recording_dir(), "meta.json")))
    print(f"  Bag zeigt danach auf {os.path.relpath(meta3['bag'], wurzel)}")
    assert meta3["bag"].startswith(ziel2), meta3["bag"]
    assert os.path.exists(meta3["bag"]) and not res3["fehlende_bags"]

    print("== Test 5b: Exportordner an Ort und Stelle oeffnen ==")
    auf = oeffnen(ziel2)
    meta_p = os.path.join(ziel2, "projekt", "recording", "meta.json")
    meta5 = json.load(open(meta_p, encoding="utf-8"))
    assert meta5["bag"] == os.path.join(ziel2, "bags", os.path.basename(bag_a)), meta5
    assert open(meta_p, encoding="utf-8").read() == json.dumps(meta5, indent=2)
    assert auf["bags"] == [meta5["bag"]] and not auf["fehlende_bags"]
    assert auf["project"].dir == os.path.join(ziel2, "projekt")
    assert auf["project"].bag_name == os.path.basename(bag_a)
    print(f"  Bag zeigt danach auf {os.path.relpath(meta5['bag'], wurzel)}")

    print("== Test 6: kaputter Ordner ==")
    for pfad, erwartet in ((wurzel, "keine super360.json"),
                           (os.path.join(wurzel, "gibtsnicht"), "keine super360.json")):
        try:
            read_manifest(pfad)
        except RuntimeError as exc:
            print(f"  {os.path.basename(pfad) or '<wurzel>'}: {str(exc)[:50]}…")
        else:
            raise AssertionError(f"{pfad} haette abgelehnt werden muessen")

    shutil.rmtree(wurzel, ignore_errors=True)
    print("bundle SELFTEST OK")

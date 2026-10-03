#!/usr/bin/env python3
"""Welche Aufbereitung und welcher Matcher verbinden die Nachtbilder am besten?

Laeuft im Interpreter mit pycolmap (``core.meander.find_colmap_python``). Ein
Aufruf rechnet eine Variante: Merkmale, Paarvergleich, Rekonstruktion — mit
denselben Vorgaben wie ``colorize_pipeline/_colmap_worker.py`` (eine Kamera,
Optik aus dem EXIF festgehalten, Geotags als Positions-Prior), damit nur das
verglichen wird, was die Variante aendert.

    verbinden_versuch.py <bilder> <arbeitsordner> <gps_enu.json> <brennweite_px> <variante.json>

Die Variante ist ein JSON-Objekt:

    {"sift": {...}}        Felder von SiftExtractionOptions
    {"abgleich": {...}}    Felder von SiftMatchingOptions, dazu "guided_matching"
    {"merkmale": "ALIKED_N16ROT", "matcher": "ALIKED_LIGHTGLUE"}
    {"max_bild": 800}      Bilder vor der Merkmalssuche verkleinern
    {"mapper": "global"}   GLOMAP statt der schrittweisen Rekonstruktion
    {"fremd": "x.npz"}     Punkte und Treffer aus verbinden_lightglue.py,
                           dazu "min_inlier" fuer die geometrische Pruefung

Ergebnis: ``<arbeitsordner>/ergebnis.json`` und eine Zeile ``ERGEBNIS …``.
"""

import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, os.path.expanduser("~/PointCloudMerger/colorize_pipeline"))
import _colmap_worker as worker  # noqa: E402


def _gps_fehler(rec, positionen: dict) -> float:
    """RMS der Kamerazentren gegen die Geotags nach Aehnlichkeitsabbildung.

    Viele registrierte Bilder sind nur dann ein Gewinn, wenn sie auch am
    richtigen Ort stehen; falsch verbundene Bilder fallen hier auf.
    """
    a, b = [], []
    for im in rec.images.values():
        if im.name in positionen and im.has_pose:
            a.append(im.projection_center())
            b.append(positionen[im.name])
    if len(a) < 4:
        return float("nan")
    a, b = np.asarray(a, float), np.asarray(b, float)
    ma, mb = a.mean(0), b.mean(0)
    U, S, Vt = np.linalg.svd((b - mb).T @ (a - ma) / len(a))
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    s = (S * [1, 1, d]).sum() / ((a - ma) ** 2).sum(1).mean()
    r = (s * (R @ (a - ma).T).T + mb) - b
    return float(np.sqrt((r ** 2).sum(1).mean()))


def _sift(db, bilder, ro, v, faeden):
    eo = pycolmap.FeatureExtractionOptions()
    eo.num_threads = faeden
    if v.get("merkmale"):
        eo.type = getattr(pycolmap.FeatureExtractorType, v["merkmale"])
    if v.get("max_bild"):
        eo.max_image_size = int(v["max_bild"])
    for k, w in v.get("sift", {}).items():
        setattr(eo.sift, k, w)
    pycolmap.extract_features(database_path=db, image_path=bilder,
                              camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=ro, extraction_options=eo)

    mo = pycolmap.FeatureMatchingOptions()
    mo.num_threads = faeden
    if v.get("matcher"):
        mo.type = getattr(pycolmap.FeatureMatcherType, v["matcher"])
    for k, w in v.get("abgleich", {}).items():
        if k == "guided_matching":
            mo.guided_matching = bool(w)
        else:
            setattr(mo.sift, k, w)
    pycolmap.match_exhaustive(db, matching_options=mo)


def _fremde_treffer(db, bilder, work, ro, npz, min_inlier=15):
    """Punkte und Treffer aus ``verbinden_lightglue.py`` in die Datenbank legen
    und von COLMAP geometrisch pruefen lassen."""
    f = np.load(npz)
    pycolmap.Database.open(db).close()   # legt die leere Datenbank an
    pycolmap.import_images(database_path=db, image_path=bilder,
                           camera_mode=pycolmap.CameraMode.SINGLE, options=ro)
    con = sqlite3.connect(str(db))
    ids = dict(con.execute("SELECT name, image_id FROM images").fetchall())
    namen = [str(n) for n in f["namen"]]
    g = f["grenzen"]
    for i, n in enumerate(namen):
        p = np.ascontiguousarray(f["punkte"][g[i]:g[i + 1]], np.float32)
        con.execute("INSERT INTO keypoints (image_id, rows, cols, data) VALUES (?,?,?,?)",
                    (ids[n], len(p), 2, p.tobytes()))
    zeilen, a = [], 0
    for i, j, anz in f["paare"]:
        m = np.array(f["treffer"][a:a + anz], np.uint32)
        a += anz
        ia, ib = ids[namen[i]], ids[namen[j]]
        if ia > ib:
            ia, ib, m = ib, ia, m[:, ::-1]
        con.execute("INSERT INTO matches (pair_id, rows, cols, data) VALUES (?,?,?,?)",
                    (ia * 2147483647 + ib, len(m), 2, np.ascontiguousarray(m).tobytes()))
        zeilen.append(f"{namen[i]} {namen[j]}")
    con.commit()
    con.close()
    liste = work / "paare.txt"
    liste.write_text("\n".join(zeilen) + "\n")
    pruefung = pycolmap.TwoViewGeometryOptions()
    pruefung.min_num_inliers = int(min_inlier)
    pycolmap.verify_matches(db, liste, options=pruefung)


def main() -> int:
    bilder, work, gps, focal, variante = sys.argv[1:6]
    v = json.loads(Path(variante).read_text()) if os.path.exists(variante) else json.loads(variante)
    bilder, work = Path(bilder), Path(work)
    db = work / "database.db"
    sparse = work / "sparse"
    if work.exists():
        shutil.rmtree(work)
    sparse.mkdir(parents=True)
    faeden = int(v.get("faeden", 8))
    t0 = time.time()

    namen = sorted(p.name for p in bilder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    from PIL import Image
    with Image.open(bilder / namen[0]) as im:
        W, H = im.size
    ro = pycolmap.ImageReaderOptions()
    ro.camera_model = "SIMPLE_RADIAL"
    ro.camera_params = f"{float(focal):.4f},{W / 2.0:.4f},{H / 2.0:.4f},0.0"

    if v.get("fremd"):
        _fremde_treffer(db, bilder, work, ro, v["fremd"], v.get("min_inlier", 15))
    else:
        _sift(db, bilder, ro, v, faeden)

    d = json.loads(Path(gps).read_text())
    worker._prioren_schreiben(db, d.get("positionen", {}), float(d.get("sigma", 0.15)))

    con = sqlite3.connect(str(db))
    merkmale = con.execute("SELECT SUM(rows) FROM keypoints").fetchone()[0] or 0
    paare = con.execute("SELECT rows FROM two_view_geometries WHERE rows > 0").fetchall()
    con.close()

    if v.get("mapper") == "global":
        go = pycolmap.GlobalPipelineOptions()
        go.mapper.num_threads = faeden
        go.mapper.bundle_adjustment.refine_focal_length = False
        go.mapper.bundle_adjustment.refine_extra_params = False
        maps = pycolmap.global_mapping(db, bilder, sparse, options=go)
    else:
        opt = pycolmap.IncrementalPipelineOptions()
        opt.num_threads = faeden
        opt.use_prior_position = True
        opt.use_robust_loss_on_prior_position = False
        opt.ba_refine_focal_length = False
        opt.ba_refine_extra_params = False
        maps = pycolmap.incremental_mapping(db, bilder, sparse, options=opt)

    erg = {"variante": v, "bilder": len(namen), "merkmale_je_bild": merkmale / len(namen),
           "paare": len(paare), "inlier": int(sum(r[0] for r in paare)),
           "modelle": len(maps), "registriert": 0, "punkte": 0}
    if maps:
        best = max(maps.values(), key=lambda m: m.num_reg_images())
        (sparse / "bestes").mkdir(exist_ok=True)
        best.write(sparse / "bestes")
        erg.update(registriert=best.num_reg_images(), punkte=best.num_points3D(),
                   spurlaenge=best.compute_mean_track_length(),
                   rueckprojektion=best.compute_mean_reprojection_error(),
                   gps_rms=_gps_fehler(best, d.get("positionen", {})),
                   fehlend=sorted(set(namen) - {i.name for i in best.images.values() if i.has_pose}))
    erg["sekunden"] = time.time() - t0
    (work / "ergebnis.json").write_text(json.dumps(erg, indent=1))
    print("ERGEBNIS " + json.dumps({k: w for k, w in erg.items() if k not in ("fehlend", "variante")}),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

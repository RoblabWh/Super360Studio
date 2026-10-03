#!/usr/bin/env python3
"""Numerik-Proben der Familie geometrie: mesh, merge, recording, georef.

Vertrag wie in ``numerik_probe.py``: ``PROBEN`` und ``UNSTET``. Gestartet wird
über den Starter::

    python3 scripts/umbau/numerik_probe.py --nur geometrie --schreibe
    python3 scripts/umbau/numerik_probe.py --nur geometrie --vergleiche

Jede Probe rechnet zweimal auf denselben festen Eingaben (:func:`_zweimal`).
Eine stete Probe muss dabei bitgleich bleiben, sonst scheitert sie mit dem
Hinweis, sie als unstet zu führen; eine unstete muss innerhalb ihrer Toleranz
bleiben. Abgelegt wird der erste Lauf.

Was sich als unstet erwiesen hat (CPU, open3d):

* Poisson in ``mesh.build_geometry``: Ecken und Dreiecke kommen von Lauf zu
  Lauf in anderer Reihenfolge und leicht verschoben heraus. Geprüft werden
  deshalb reihenfolgefreie Kennwerte (Quantile der Radien und Achsen, Zahl
  der Ecken und Dreiecke). Was vor Poisson liegt — Voxelraster, Einteilung in Fläche
  und Rest — ist bitgleich und steht in einer eigenen, steten Probe.
* ``merge.register``: die globale Suche (FGR) liefert jedes Mal andere
  Kandidaten, und auch ICP allein ist nur auf etwa 1e-14 wiederholbar.

Die open3d-Normalen (``mesh._normals_cpu``) sind dagegen bitgleich
wiederholbar und laufen als stete Probe.

Dateien entstehen nur im Arbeitsordner (``basis.arbeitsordner``). In den
geschriebenen Dateien werden zwei Zeitangaben neutral gemacht: ``created`` in
der meta.json der zusammengeführten Aufzeichnung und das Erstellungsdatum im
LAS-Kopf.
"""
from __future__ import annotations

import functools
import json
import os
import re
import shutil
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

import basis
from core import georef, merge, mesh, recording

_ordner: str | None = None


def _arbeit(name: str, lauf: int) -> str:
    """Leerer Unterordner je Probe und Lauf im Arbeitsordner dieses Prozesses."""
    global _ordner
    if _ordner is None:
        _ordner = basis.arbeitsordner("geometrie")
    pfad = os.path.join(_ordner, f"{name}_{lauf}")
    shutil.rmtree(pfad, ignore_errors=True)
    os.makedirs(pfad)
    return pfad


def _lies(pfad: str) -> bytes:
    with open(pfad, "rb") as fh:
        return fh.read()


# ------------------------------------------------------------------ zweimal

def _als_array(wert) -> np.ndarray:
    if isinstance(wert, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(wert), dtype=np.uint8)
    return np.asarray(wert)


def _abstand(a, b) -> float:
    """Größte Differenz zweier Werte; inf bei anderer Form oder anderem Typ."""
    a, b = _als_array(a), _als_array(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return float("inf")
    if a.size == 0:
        return 0.0
    if a.dtype.kind in "SUV":
        return 0.0 if bool(np.all(a == b)) else float("inf")
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    d = np.abs(x - y)
    d[x == y] = 0.0
    d[np.isnan(x) & np.isnan(y)] = 0.0
    d[np.isnan(d)] = np.inf
    return float(d.max())


def _zweimal(funktion, toleranz: float | None = None):
    """Probe, die ``funktion(lauf)`` zweimal fährt und beide Läufe vergleicht."""
    def probe() -> dict:
        erst, zweit = funktion(0), funktion(1)
        if sorted(erst) != sorted(zweit):
            raise RuntimeError(f"zweiter Lauf liefert andere Teile: "
                               f"{sorted(erst)} gegen {sorted(zweit)}")
        for teil in sorted(erst):
            if toleranz is None:
                if basis.hash_wert(erst[teil]) != basis.hash_wert(zweit[teil]):
                    raise RuntimeError(
                        f"Teil {teil} ist nicht bitgleich wiederholbar (größte "
                        f"Differenz {_abstand(erst[teil], zweit[teil]):.6g}) — "
                        f"als UNSTET mit Toleranz führen.")
            else:
                d = _abstand(erst[teil], zweit[teil])
                if not d <= toleranz:
                    raise RuntimeError(
                        f"Teil {teil} weicht im zweiten Lauf um {d:.6g} ab, "
                        f"Toleranz {toleranz:g}.")
        return erst
    return probe


# ----------------------------------------------------------------- Eingaben

def _aufzeichnung_daten(seed: int, n_scans: int, t0: float) -> dict:
    """Felder einer synthetischen Aufzeichnung (ein leerer Scan, ein doppelter Stempel)."""
    rng = np.random.default_rng(seed)
    je_scan = rng.integers(20, 90, size=n_scans)
    je_scan[3] = 0
    offsets = np.concatenate([[0], np.cumsum(je_scan)]).astype(np.int64)
    n = int(offsets[-1])
    points = rng.uniform(-6.0, 6.0, size=(n, 3)).astype(np.float32)
    intensity = rng.uniform(0.0, 255.0, size=n).astype(np.float32)
    stamps = t0 + np.arange(n_scans, dtype=np.float64) * 0.1
    stamps[7] = stamps[6]
    quat = rng.normal(size=(n_scans, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    pos = np.cumsum(rng.normal(0.0, 0.05, size=(n_scans, 3)), axis=0)
    poses = np.hstack([pos, quat]).astype(np.float64)
    return {"points": points, "intensity": intensity, "offsets": offsets,
            "stamps": stamps, "poses": poses}


def _aufzeichnung(daten: dict):
    """Aufzeichnung im Speicher, ohne Kippkorrektur."""
    return recording.Recording(
        points=daten["points"], intensity=daten["intensity"],
        offsets=daten["offsets"], stamps=daten["stamps"], poses=daten["poses"],
        meta={"bag": "synthetisch", "config": "whs_dense.yaml"})


def _aufzeichnung_schreiben(ordner: str, daten: dict, gravity_level=None) -> str:
    """Legt die Aufzeichnung im Format von ``core/recording.py`` ab."""
    os.makedirs(ordner)
    daten["points"].tofile(os.path.join(ordner, "points.bin"))
    daten["intensity"].tofile(os.path.join(ordner, "intensity.bin"))
    for name in ("offsets", "stamps", "poses"):
        np.save(os.path.join(ordner, name + ".npy"), daten[name])
    meta = {"bag": "synthetisch", "config": "whs_dense.yaml",
            "n_scans": int(len(daten["stamps"])), "n_points": int(len(daten["points"]))}
    if gravity_level is not None:
        meta["gravity_level"] = gravity_level
    with open(os.path.join(ordner, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return ordner


def _lage(gier_grad: float, kipp_grad: float, versatz) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("zx", [gier_grad, kipp_grad], degrees=True).as_matrix()
    T[:3, 3] = versatz
    return T


# --------------------------------------------------------------------- mesh

def _mesh_voxelize(lauf: int) -> dict:
    rng = np.random.default_rng(101)
    punkte = rng.uniform([-3.0, -2.0, -1.0], [3.0, 2.0, 1.5], size=(6000, 3)).astype(np.float32)
    out: dict = {}
    for name, voxel in (("grob", 0.25), ("fein", 0.04)):
        point_voxel, cells, keys, cen, cnt, dims = mesh._voxelize(punkte, voxel)
        out.update({f"{name}_point_voxel": point_voxel, f"{name}_zellen": cells,
                    f"{name}_schluessel": keys, f"{name}_mitten": cen,
                    f"{name}_anzahl": cnt, f"{name}_dims": np.asarray(dims)})
    return out


def _kunst_geom():
    """Geometrie-dict ohne Poisson: Raster aus ``_voxelize``, Ecken zufällig zugeordnet."""
    rng = np.random.default_rng(102)
    punkte = rng.uniform(-3.0, 3.0, size=(8000, 3)).astype(np.float32)
    point_voxel, _cells, keys, _cen, _cnt, _dims = mesh._voxelize(punkte, 0.3)
    m = len(keys)
    geom = {"point_voxel": point_voxel,
            "vertex_voxel": rng.integers(0, m, size=(1500, 4)).astype(np.int32),
            "zelle_rest": rng.random(m) < 0.3,
            "stats": {"zellen": int(m)}}
    return geom, punkte, rng


def _mesh_vertex_colors(lauf: int) -> dict:
    geom, punkte, rng = _kunst_geom()
    farben = rng.integers(0, 256, size=(len(punkte), 3), dtype=np.uint8)
    gueltig = rng.random(len(punkte)) < 0.02
    rgb, ok = mesh.vertex_colors(geom, farben, gueltig)
    rgb_alle, ok_alle = mesh.vertex_colors(geom, farben, None)
    return {"rgb": rgb, "gueltig": ok, "rgb_ohne_maske": rgb_alle,
            "gueltig_ohne_maske": ok_alle}


def _mesh_vertex_scalar(lauf: int) -> dict:
    geom, punkte, rng = _kunst_geom()
    werte = rng.uniform(0.0, 255.0, size=len(punkte)).astype(np.float32)
    return {"intensitaet": mesh.vertex_scalar(geom, werte),
            "hoehe": mesh.vertex_scalar(geom, punkte[:, 2])}


def _mesh_rest_maske(lauf: int) -> dict:
    geom, _punkte, _rng = _kunst_geom()
    ohne = dict(geom, zelle_rest=None)
    return {"maske": mesh.rest_maske(geom),
            "ohne_hybrid_none": int(mesh.rest_maske(ohne) is None)}


def _kugel() -> np.ndarray:
    """Oben offene Kugel (Radius 2 m), ein Laubballen, eine Platte und ein Mast.

    Die Platte wird entlang x von glatt bis 12 cm rau, der Mast ist linienhaft:
    so liegen Zellen beiderseits der Schwellen für Streuung und Flachheit.
    """
    rng = np.random.default_rng(103)
    v = rng.normal(size=(24_000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v = v[v[:, 2] < 0.8] * 2.0
    laub = rng.normal(size=(4_000, 3)) * 0.3 + [4.0, 0.0, 0.0]
    platte = rng.uniform([-2.0, -1.0, 0.0], [2.0, 1.0, 0.0], size=(9_000, 3))
    platte[:, 2] = -3.0 + rng.normal(size=len(platte)) * 0.03 * (platte[:, 0] + 2.0)
    mast = np.stack([-4.0 + rng.normal(size=1_500) * 0.02,
                     rng.normal(size=1_500) * 0.02,
                     rng.uniform(-2.0, 2.0, size=1_500)], axis=1)
    return np.concatenate([v, laub, platte, mast]).astype(np.float32)


def _mesh_normals_cpu(lauf: int) -> dict:
    _pv, _cells, _keys, cen, _cnt, _dims = mesh._voxelize(_kugel(), 0.08)
    normalen, _, _ = mesh._normals_cpu(cen, 0.08)
    _, eigenwerte, nachbarn = mesh._normals_cpu(cen, 0.08, mesh._FORM_R)
    return {"normalen": normalen, "eigenwerte": eigenwerte, "nachbarn": nachbarn}


@functools.lru_cache(maxsize=None)
def _geometrie(lauf: int) -> dict:
    """Ein Lauf ``build_geometry`` je Laufnummer; die drei Proben dazu teilen ihn."""
    return mesh.build_geometry(_kugel(), np.zeros((1, 3)), voxel=0.08, depth=6,
                               backend="cpu")


def _mesh_build_geometry(lauf: int) -> dict:
    """Der Teil vor Poisson: Raster und Einteilung in Fläche und Rest."""
    geom = _geometrie(lauf)
    stats = geom["stats"]
    return {"point_voxel": geom["point_voxel"], "zelle_rest": geom["zelle_rest"],
            "rest_maske": mesh.rest_maske(geom),
            "zellen": np.array([stats["zellen"], stats["flaechenzellen"],
                                stats["restzellen"], stats["punkte"]], dtype=np.int64)}


def _mesh_build_geometry_form(lauf: int) -> dict:
    """Poisson-Netz als reihenfolgefreie Kennwerte in Metern."""
    geom = _geometrie(lauf)
    ecken = np.asarray(geom["vertices"], dtype=np.float64)
    radius = np.linalg.norm(ecken, axis=1)
    innen = np.einsum("ij,ij->i", np.asarray(geom["normals"], np.float64), -ecken) > 0
    # ohne Minimum und Maximum: am offenen Rand hängt die äußerste Ecke am Beschnitt
    return {"radius_quantile": np.quantile(radius, [0.01, 0.25, 0.5, 0.75, 0.99]),
            "radius_mittel": float(radius.mean()),
            "achsen_quantile": np.quantile(ecken, [0.01, 0.5, 0.99], axis=0),
            "normalen_innen": float(innen.mean())}


def _mesh_build_geometry_netz(lauf: int) -> dict:
    """Größe des Poisson-Netzes: Ecken, Dreiecke, Dreiecke vor dem Beschnitt."""
    geom = _geometrie(lauf)
    stats = geom["stats"]
    assert geom["vertex_voxel"].shape == (len(geom["vertices"]), 4)
    return {"groesse": np.array([stats["ecken"], stats["dreiecke"],
                                 stats["dreiecke_roh"]], dtype=np.int64)}


# -------------------------------------------------------------------- merge

def _merge_cloud_for_registration(lauf: int) -> dict:
    rec = _aufzeichnung(_aufzeichnung_daten(201, 120, 1000.0))
    return {"schritt_8": merge.cloud_for_registration(rec, max_points=800),
            "alle": merge.cloud_for_registration(rec),
            "ein_punkt": merge.cloud_for_registration(rec, max_points=0)}


def _merge_transform_poses(lauf: int) -> dict:
    daten = _aufzeichnung_daten(202, 60, 500.0)
    return {"posen": merge.transform_poses(daten["poses"], _lage(35.0, 4.0, [3.0, -2.0, 0.4])),
            "identitaet": merge.transform_poses(daten["poses"], np.eye(4))}


def _meta_ohne_zeit(pfad: str) -> bytes:
    daten, n = re.subn(rb'("created": )"[^"]*"', rb'\1""', _lies(pfad))
    if n != 1:
        raise RuntimeError(f"meta.json: 'created' {n}-mal gefunden, erwartet einmal.")
    return daten


def _merge_merge_recordings(lauf: int) -> dict:
    ordner = _arbeit("merge", lauf)
    rec_a = recording.Recording.load(_aufzeichnung_schreiben(
        os.path.join(ordner, "a"), _aufzeichnung_daten(203, 40, 1000.0)))
    rec_b = recording.Recording.load(_aufzeichnung_schreiben(
        os.path.join(ordner, "b"), _aufzeichnung_daten(204, 30, 2000.0)))
    T_ab = _lage(90.0, 2.0, [10.0, 0.5, -0.2])
    out: dict = {}
    # hin: A ist der frühere Flug; zurueck: Argumente vertauscht, mit info
    for name, (erst, zweit, bags, info) in (
            ("hin", (rec_a, rec_b, ("flug_a", "flug_b"), None)),
            ("zurueck", (rec_b, rec_a, ("flug_b", "flug_a"),
                         {"fitness": 0.75, "rmse": 0.031, "kandidat": "Handjustage"}))):
        ziel = os.path.join(ordner, name, "recording")
        merge.merge_recordings(erst, zweit, T_ab, ziel, bags[0], bags[1], info=info)
        for datei in ("points.bin", "intensity.bin", "offsets.npy", "stamps.npy",
                      "poses.npy"):
            out[f"{name}/{datei}"] = _lies(os.path.join(ziel, datei))
        out[f"{name}/meta.json"] = _meta_ohne_zeit(os.path.join(ziel, "meta.json"))
        if sorted(os.listdir(ziel)) != ["intensity.bin", "meta.json", "offsets.npy",
                                        "points.bin", "poses.npy", "stamps.npy"]:
            raise RuntimeError(f"unerwartete Dateien: {sorted(os.listdir(ziel))}")
    return out


def _raum():
    """Raum mit Boden, zwei Wänden und Säule, dazu dieselbe Wolke bekannt verdreht."""
    rng = np.random.default_rng(205)
    n = 3000
    boden = rng.uniform([-8, -8, 0], [8, 8, 0.02], size=(3 * n, 3))
    wand1 = np.stack([np.full(n, -8.0), rng.uniform(-8, 8, n), rng.uniform(0, 4, n)], axis=1)
    wand2 = np.stack([rng.uniform(-8, 8, n), np.full(n, 8.0), rng.uniform(0, 4, n)], axis=1)
    saeule = np.stack([2 + 0.4 * np.cos(rng.uniform(0, 6.28, n)),
                       -3 + 0.4 * np.sin(rng.uniform(0, 6.28, n)),
                       rng.uniform(0, 4, n)], axis=1)
    a = np.vstack([boden, wand1, wand2, saeule])
    T_wahr = np.eye(4)
    T_wahr[:3, :3] = Rotation.from_euler("z", 35.0, degrees=True).as_matrix()
    T_wahr[:3, 3] = [3.0, -2.0, 0.4]
    b = (np.linalg.inv(T_wahr)[:3, :3] @ (a - T_wahr[:3, 3]).T).T
    return a, b, T_wahr


def _register_teile(res: dict) -> dict:
    return {"T": res["T"], "fitness": float(res["fitness"]), "rmse": float(res["rmse"]),
            "voxel": np.asarray(res["voxel"], dtype=np.float64)}


def _merge_register_auto(lauf: int) -> dict:
    a, b, _T_wahr = _raum()
    return _register_teile(merge.register(a, b))


def _merge_register_icp(lauf: int) -> dict:
    a, b, T_wahr = _raum()
    T_start = T_wahr.copy()
    T_start[:3, 3] += [0.2, -0.1, 0.05]
    return _register_teile(merge.register(a, b, T_init=T_start, mode="icp"))


# ---------------------------------------------------------------- recording

def _recording_world_points(lauf: int) -> dict:
    rec = _aufzeichnung(_aufzeichnung_daten(301, 120, 1000.0))
    fortschritt: list = []
    welt = rec.world_points(progress_cb=lambda f, m: fortschritt.append(f))
    return {"welt": welt, "fortschritt": np.asarray(fortschritt, dtype=np.float64)}


def _recording_interpolate_pose(lauf: int) -> dict:
    rec = _aufzeichnung(_aufzeichnung_daten(302, 40, 1000.0))
    s = rec.stamps
    # vor dem Anfang (gehalten / zu weit), auf Stempeln, dazwischen, am
    # doppelten Stempel, hinter dem Ende (gehalten / zu weit)
    zeiten = [s[0] - 0.2, s[0] - 0.1, s[0], s[0] + 0.033, s[5], s[5] + 0.05,
              s[6], s[6] + 0.025, s[8] - 0.01, s[20] + 0.0777, s[-1] - 0.001,
              s[-1], s[-1] + 0.1, s[-1] + 0.2]
    out = np.full((len(zeiten), 4, 4), np.nan)
    for i, t in enumerate(zeiten):
        T = rec.interpolate_pose(float(t))
        if T is not None:
            out[i] = T
    return {"posen": out, "bahn": rec.path_positions()}


def _recording_rotate_world(lauf: int) -> dict:
    daten = _aufzeichnung_daten(303, 60, 0.0)
    rot = Rotation.from_rotvec([0.31, -0.12, 0.05])
    return {"posen": recording._rotate_world(daten["poses"], rot),
            "identitaet": recording._rotate_world(daten["poses"], Rotation.identity())}


def _recording_level_rotation(lauf: int) -> dict:
    rng = np.random.default_rng(304)
    hoch = [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 9.81],
            [1.0, 0.0, 0.0], [0.3, -0.2, 0.9]]
    for grad, achse in ((40.0, [0, 1, 0]), (7.0, [1, 0, 0]), (10.0, [1, 1, 0])):
        achse = np.asarray(achse, dtype=np.float64)
        hoch.append(Rotation.from_rotvec(np.radians(grad) * achse / np.linalg.norm(achse))
                    .apply([0.0, 0.0, 1.0]).tolist())
    hoch.extend(rng.normal(size=(8, 3)).tolist())
    return {"quat": np.stack([recording.level_rotation(np.asarray(v)).as_quat()
                              for v in hoch])}


def _recording_load(lauf: int) -> dict:
    ordner = _arbeit("recording", lauf)
    daten = _aufzeichnung_daten(305, 50, 1000.0)
    kipp = recording.level_rotation(
        Rotation.from_rotvec(np.radians(25.0) * np.array([0.0, 1.0, 0.0]))
        .apply([0.0, 0.0, 1.0]))
    stufe = {"quat": [float(x) for x in kipp.as_quat()], "tilt_deg": 25.0,
             "threshold_deg": 10.0, "applied": True}
    flach = dict(stufe, tilt_deg=4.0, applied=False)
    out: dict = {}
    for name, gravity_level, level in (("gekippt", stufe, True),
                                       ("unter_schwelle", flach, True),
                                       ("ohne_messung", None, True),
                                       ("level_aus", stufe, False)):
        pfad = _aufzeichnung_schreiben(os.path.join(ordner, name), daten, gravity_level)
        meta_vorher = _lies(os.path.join(pfad, "meta.json"))
        rec = recording.Recording.load(pfad, level=level)
        if _lies(os.path.join(pfad, "meta.json")) != meta_vorher:
            raise RuntimeError(f"{name}: Recording.load hat die meta.json verändert.")
        out[f"{name}_posen"] = np.array(rec.poses)
        out[f"{name}_stufe"] = np.array(
            [-1 if rec.gravity_level is None else int(bool(rec.gravity_level["applied"]))])
    out.update({"punkte": np.array(rec.points), "intensitaet": np.array(rec.intensity),
                "offsets": np.array(rec.offsets), "stempel": np.array(rec.stamps),
                "welt_gekippt": recording.Recording.load(
                    os.path.join(ordner, "gekippt")).world_points()})
    return out


# ------------------------------------------------------------------- georef

def _fix(i: int, lat: float, lon: float, **kw):
    werte = dict(stamp=1000.0 + 0.5 * i, lat=lat, lon=lon, alt=60.0 + 0.01 * i,
                 status=0, service=1, cov_east_m=0.8, cov_north_m=0.8, cov_up_m=1.5,
                 cov_type=2, fix_type=4, eph_cm=80 + (i % 7), epv_cm=120,
                 satellites=10 + (i % 6))
    werte.update(kw)
    return SimpleNamespace(**werte)


def _fixe() -> dict:
    """Vier Fälle: gemischt, zu kurze Strecke, totes GPS, keine Fixe."""
    gemischt = []
    for i in range(80):
        kw: dict = {}
        if i % 11 == 3:
            kw["eph_cm"] = 9999
        if i % 13 == 5:
            kw["satellites"] = 4
        if i % 17 == 7:
            kw["satellites"] = 255
        if i % 19 == 9:
            kw["fix_type"] = 2
        if i % 23 == 11:
            kw["status"] = -1
        if i % 29 == 13:
            kw["cov_east_m"] = 4294967.295
        if i % 31 == 15:
            kw.update(fix_type=None, eph_cm=None, satellites=None)
        if i == 40:
            gemischt.append(_fix(i, 0.0, 0.0, **kw))
            continue
        gemischt.append(_fix(i, 51.5740 + 4e-6 * i, 7.0270 + 9e-6 * i, **kw))
    kurz = [_fix(i, 51.5740 + 1e-7 * i, 7.0270, eph_cm=650 if i < 3 else 90)
            for i in range(30)]
    tot = [_fix(i, 0.0, 0.0, status=-1, fix_type=0, eph_cm=9999, satellites=0,
                cov_east_m=4294967.295, cov_north_m=4294967.295) for i in range(25)]
    return {"gemischt": gemischt, "kurz": kurz, "tot": tot, "leer": []}


def _georef_assess(lauf: int) -> dict:
    out: dict = {}
    for name, fixe in _fixe().items():
        q = georef.assess(fixe)
        nan = float("nan")
        out[f"{name}_gut"] = np.array(q.per_fix_good)
        out[f"{name}_zahlen"] = np.array([int(q.usable), q.n_total, q.n_good], dtype=np.int64)
        out[f"{name}_werte"] = np.array(
            [q.baseline_m,
             nan if q.median_eph_cm is None else q.median_eph_cm,
             nan if q.median_sats is None else q.median_sats], dtype=np.float64)
        out[f"{name}_fix_typen"] = np.array(sorted(q.fix_type_hist.items()),
                                            dtype=np.int64).reshape(-1, 2)
        out[f"{name}_gruende"] = "\n".join(q.reasons).encode("utf-8")
    return out


def _wolke():
    rng = np.random.default_rng(402)
    punkte = rng.uniform(-20.0, 20.0, size=(1000, 3))
    farben = rng.integers(0, 256, size=(1000, 3), dtype=np.uint8)
    return punkte, farben, rng.uniform(0.0, 1.0, size=(1000, 3))


def _georef_export_ply_pcd(lauf: int) -> dict:
    ordner = _arbeit("ply_pcd", lauf)
    punkte, farben, farben_01 = _wolke()
    out: dict = {}
    for name, c in (("farbe.ply", farben), ("farbe.pcd", farben),
                    ("farbe_01.ply", farben_01), ("ohne_farbe.ply", None),
                    ("ohne_farbe.pcd", None)):
        pfad = os.path.join(ordner, name)
        georef.export_ply_pcd(punkte, c, pfad)
        out[name] = _lies(pfad)
    return out


def _las_ohne_datum(pfad: str) -> bytes:
    """Dateibytes; Tag und Jahr der Erstellung (Kopf, Byte 90 bis 93) auf null."""
    daten = bytearray(_lies(pfad))
    if daten[:4] != b"LASF" or bytes(daten[24:26]) != b"\x01\x02":
        raise RuntimeError("kein LAS-1.2-Kopf — das Erstellungsdatum liegt woanders.")
    daten[90:94] = b"\x00\x00\x00\x00"
    return bytes(daten)


def _georef_export_las(lauf: int) -> dict:
    ordner = _arbeit("las", lauf)
    punkte, farben, farben_01 = _wolke()
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", 37.0, degrees=True).as_matrix()
    T[:3, 3] = [12.0, -5.0, 3.0]
    lage = georef.GeorefResult(T_enu_world=T, origin_llh=(51.5740, 7.0270, 60.0),
                               utm_epsg=32632, rms_m=0.0, n_used=0,
                               utm_offset=(0.0, 0.0))
    out: dict = {}
    for name, c, g in (("lokal.las", farben, None), ("utm.las", farben, lage),
                       ("lokal_farbe_01.las", farben_01, None),
                       ("utm_ohne_farbe.las", None, lage)):
        pfad = os.path.join(ordner, name)
        georef.export_las(punkte, c, g, pfad)
        out[name] = _las_ohne_datum(pfad)
    return out


# ------------------------------------------------------------------ Vertrag

# Toleranzen: Poisson-Kennwerte in Metern (gemessene Streuung bis 5e-4 bei
# 8 cm Raster), Größe des Netzes in Ecken bzw. Dreiecken (rund ein halbes
# Prozent; gemessen schwankt sie um ein Dreieck), register in Metern bzw. als
# Matrixeintrag (gemessene Streuung um 1e-14).
UNSTET = {
    "mesh.build_geometry_form": 5e-3,
    "mesh.build_geometry_netz": 150.0,
    "merge.register_auto": 1e-6,
    "merge.register_icp": 1e-6,
}

_FUNKTIONEN = {
    "mesh._voxelize": _mesh_voxelize,
    "mesh.vertex_colors": _mesh_vertex_colors,
    "mesh.vertex_scalar": _mesh_vertex_scalar,
    "mesh.rest_maske": _mesh_rest_maske,
    "mesh._normals_cpu": _mesh_normals_cpu,
    "mesh.build_geometry": _mesh_build_geometry,
    "mesh.build_geometry_form": _mesh_build_geometry_form,
    "mesh.build_geometry_netz": _mesh_build_geometry_netz,
    "merge.cloud_for_registration": _merge_cloud_for_registration,
    "merge.transform_poses": _merge_transform_poses,
    "merge.merge_recordings": _merge_merge_recordings,
    "merge.register_auto": _merge_register_auto,
    "merge.register_icp": _merge_register_icp,
    "recording.world_points": _recording_world_points,
    "recording.interpolate_pose": _recording_interpolate_pose,
    "recording._rotate_world": _recording_rotate_world,
    "recording.level_rotation": _recording_level_rotation,
    "recording.Recording.load": _recording_load,
    "georef.assess": _georef_assess,
    "georef.export_ply_pcd": _georef_export_ply_pcd,
    "georef.export_las": _georef_export_las,
}

PROBEN = {name: _zweimal(funktion, UNSTET.get(name))
          for name, funktion in _FUNKTIONEN.items()}

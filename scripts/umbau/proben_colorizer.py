#!/usr/bin/env python3
"""Numerik-Proben für ``core.colorizer`` und ``core.stitcher``.

Familie ``colorizer`` der Numerik-Probe (Vertrag s. ``numerik_probe.py``)::

    python3 scripts/umbau/numerik_probe.py --nur colorizer --schreibe
    python3 scripts/umbau/numerik_probe.py --nur colorizer --vergleiche

Drei Gruppen:

* die kleinen Rechenhelfer auf festen, geseedeten Eingaben,
* ``check_extrinsic`` und ``auto_calibrate`` mit einem Stellvertreter für
  ``_PhotoScoreContext``. Dessen Score ist gestuft (Plateaus): Nachbarn mit
  gleichem Wert gibt es bei jeder Schrittweite, damit hängt das Ergebnis am
  Strikt-größer-Vergleich und an der Reihenfolge der Kandidaten. Die Folge der
  bewerteten Drehungen wird mit abgelegt,
* die Einfärbung auf dem seg0-Bag mit ``backend="cpu"``: einmal voll mit den
  Vorgaben, dazu zwei kurze Läufe über 40 Scans (Blaulichtfilter mit zwei
  Abschnitten; Linsenwahl, Himmelssaum und Rangfolge abgeschaltet). Neben
  ``colors.bin`` und ``valid.bin`` wird je Lauf die ``meta.json`` abgelegt
  (ohne Laufzeit und Datum, Pfade neutral) — an ihrem ``rec_fingerprint``
  hängt, ob gespeicherte Farben zur Wolke passen.

Die Aufzeichnung kommt nur aus ``basis.cache_kopie()``, die Ausgabe landet im
Arbeitsordner. ``Recording.load`` schreibt ``gravity_level`` in die meta.json —
jede seg0-Probe vergleicht deshalb die Änderungszeiten im echten Cache vor und
nach dem Lauf und scheitert, wenn sich eine bewegt hat.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.dont_write_bytecode = True
import basis  # noqa: E402

from core import colorizer, stitcher  # noqa: E402


def _calib() -> str:
    return os.path.join(basis.wurzel(), "calib", "calib_result_new2.json")


def _oder_leer(wert, dtype=bool) -> np.ndarray:
    """``None`` als leeres Array, damit es sich als Teil ablegen lässt."""
    return np.zeros(0, dtype=dtype) if wert is None else wert


def _listen(listen) -> np.ndarray:
    """Folge von Indexlisten als ein Array, je Liste mit -1 abgeschlossen."""
    out: list = []
    for eintrag in listen:
        out.extend(int(v) for v in eintrag)
        out.append(-1)
    return np.asarray(out, dtype=np.int64)


# ------------------------------------------------------------ Rechenhelfer

def _farbproben(rng, n: int) -> np.ndarray:
    """BGR float32: Zufall plus die Ränder (grau, schwarz, weiß, reine Farben)."""
    feste = np.array([[0, 0, 0], [255, 255, 255], [128, 128, 128], [255, 0, 0],
                      [0, 255, 0], [0, 0, 255], [230, 90, 20], [140, 120, 110],
                      [20, 20, 200], [50, 8, 4], [255, 255, 0], [255, 0, 255],
                      [200, 199, 198], [10, 10, 11]], dtype=np.float32)
    zufall = rng.uniform(0.0, 255.0, size=(n, 3)).astype(np.float32)
    blass = np.clip(rng.uniform(40.0, 250.0, size=(n, 1))
                    + rng.normal(0.0, 6.0, size=(n, 3)), 0.0, 255.0).astype(np.float32)
    return np.concatenate([feste, zufall, blass])


_BLAU_BEREICHE = {"vorgabe": (150.0, 290.0, 0.05, 10.0),
                  "eng": (200.0, 240.0, 0.4, 60.0),
                  "umlauf": (340.0, 20.0, 0.4, 60.0)}


def _is_blue():
    bgr = _farbproben(np.random.default_rng(101), 6000)
    out = {name: colorizer._is_blue(bgr, *bereich)
           for name, bereich in _BLAU_BEREICHE.items()}
    out["leer"] = colorizer._is_blue(np.zeros((0, 3), np.float32), *_BLAU_BEREICHE["eng"])
    return out


def _neutralize_blue():
    rng = np.random.default_rng(102)
    rgb = np.ascontiguousarray(_farbproben(rng, 12000)[:, ::-1]).astype(np.uint8)
    valid = (rng.uniform(size=len(rgb)) < 0.8).astype(np.uint8)
    out = {}
    for name, bereich, staerke, chunk in (
            ("voll", _BLAU_BEREICHE["vorgabe"], 1.0, None),
            ("teilweise", _BLAU_BEREICHE["vorgabe"], 0.35, 7000),
            ("eng", _BLAU_BEREICHE["eng"], 1.0, 1000),
            ("umlauf", _BLAU_BEREICHE["umlauf"], 0.6, None),
            ("ueber_eins", _BLAU_BEREICHE["eng"], 1.7, None),
            ("aus", _BLAU_BEREICHE["vorgabe"], 0.0, None)):
        farben = rgb.copy()
        zusatz = {} if chunk is None else {"chunk": chunk}
        n = colorizer.neutralize_blue(farben, valid, *bereich, staerke, **zusatz)
        out[f"{name}_farben"] = farben
        out[f"{name}_n"] = int(n)
    return out


def _bild(rng, hoehe: int, breite: int) -> np.ndarray:
    """Weiches Zufallsbild (BGR uint8) mit zwei ausgebrannten Flächen samt Saum."""
    import cv2

    grob = rng.uniform(0.0, 255.0, size=(hoehe // 8 + 1, breite // 8 + 1, 3)).astype(np.float32)
    img = cv2.resize(grob, (breite, hoehe), interpolation=cv2.INTER_CUBIC)
    img += rng.normal(0.0, 4.0, size=img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 249.0)
    yy, xx = np.mgrid[0:hoehe, 0:breite]
    for cy, cx, r in ((hoehe * 0.3, breite * 0.25, min(hoehe, breite) * 0.12),
                      (hoehe * 0.7, breite * 0.8, min(hoehe, breite) * 0.07)):
        d = np.hypot(yy - cy, xx - cx)
        img[d <= r] = 255.0
        saum = (d > r) & (d <= r + 3.0)
        img[saum] = np.maximum(img[saum], 222.0)
    return img.astype(np.uint8)


def _blue_preview():
    rng = np.random.default_rng(103)
    img = _bild(rng, 120, 168)
    out = {}
    for name, bereich in _BLAU_BEREICHE.items():
        bild, anteil = colorizer.blue_preview(img, *bereich)
        out[f"{name}_bild"] = bild
        out[f"{name}_anteil"] = float(anteil)
    return out


def _blown_mask():
    rng = np.random.default_rng(104)
    img = _bild(rng, 200, 304)
    dunkel = np.clip(img, 0, 180).astype(np.uint8)
    out = {}
    for name, bild, clip, grow in (("saum4", img, 250, 4), ("saum0", img, 250, 0),
                                   ("clip240_saum7", img, 240, 7),
                                   ("saum_negativ", img, 250, -1),
                                   ("clip256", img, 256, 4),
                                   ("nichts_ausgebrannt", dunkel, 250, 4)):
        maske = colorizer._blown_mask(bild, clip, grow)
        out[name] = _oder_leer(maske)
        out[f"{name}_ist_none"] = int(maske is None)
    return out


def _not_blown():
    rng = np.random.default_rng(105)
    maske = colorizer._blown_mask(_bild(rng, 200, 304), 250, 4)
    uv = rng.uniform(-20.0, 330.0, size=(5000, 2)).astype(np.float32)
    uv[:, 1] *= 0.7
    return {"mit_maske": colorizer._not_blown(maske, uv),
            "ohne_maske": colorizer._not_blown(None, uv)}


def _is_skyish():
    bgr = _farbproben(np.random.default_rng(106), 6000)
    luma = 0.299 * bgr[:, 2] + 0.587 * bgr[:, 1] + 0.114 * bgr[:, 0]
    return {"vorgabe": colorizer._is_skyish(bgr, luma, 200.0, 0.15),
            "streng": colorizer._is_skyish(bgr, luma, 120.0, 0.4),
            "leer": colorizer._is_skyish(np.zeros((0, 3), np.float32),
                                         np.zeros(0, np.float32), 200.0, 0.15)}


def _sample_bgr():
    rng = np.random.default_rng(107)
    img = _bild(rng, 400, 496)
    out = {}
    # 40000 > 16384: mehrere Zeilen mit Auffüllen; 16384 und 32768: ohne Rest
    for n in (0, 5, 16384, 32768, 40000):
        uv = np.empty((n, 2), dtype=np.float32)
        uv[:, 0] = rng.uniform(-8.0, 504.0, size=n)
        uv[:, 1] = rng.uniform(-8.0, 408.0, size=n)
        out[f"n{n}"] = colorizer._sample_bgr(img, uv)
    ganz = np.array([[0.0, 0.0], [495.0, 399.0], [17.0, 23.0], [17.5, 23.5]], np.float32)
    out["ganzzahlig"] = colorizer._sample_bgr(img, ganz)
    return out


def _in_circle():
    rng = np.random.default_rng(108)
    uv = rng.uniform(-50.0, 1570.0, size=(20000, 2)).astype(np.float32)
    winkel = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    rand = [np.stack([760.0 + r * np.cos(winkel), 760.0 + r * np.sin(winkel)], 1)
            for r in (699.0, 700.0, 700.001, 740.0)]
    uv = np.concatenate([uv] + [r.astype(np.float32) for r in rand])
    return {"float32": colorizer._in_circle(uv),
            "float64": colorizer._in_circle(uv.astype(np.float64))}


def _equirect_uv():
    rng = np.random.default_rng(109)
    feste = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                      [0, 0, 1], [0, 0, -1], [1e-9, -3.0, 0.0]], dtype=np.float32)
    dirs = np.concatenate([feste, (rng.normal(size=(12000, 3)) * 6.0).astype(np.float32)])
    norms = np.linalg.norm(dirs, axis=1)
    out = {}
    for breite, hoehe in ((1920, 960), (640, 320)):
        u, v = colorizer._equirect_uv(dirs, norms, breite, hoehe)
        out[f"u_{breite}"] = u
        out[f"v_{breite}"] = v
    u, v = colorizer._equirect_uv(dirs.astype(np.float64), norms.astype(np.float64), 640, 320)
    out["u_640_float64"] = u
    out["v_640_float64"] = v
    return out


def _kamera_stempel(rng, n: int = 600) -> np.ndarray:
    """Kamera-Stempel mit ~11 Bildern je Sekunde, Zittern und zwei Lücken."""
    schritt = np.full(n, 1.0 / 11.0) + rng.normal(0.0, 0.004, size=n)
    schritt[200] += 0.9
    schritt[410] += 0.35
    return 5000.0 + np.cumsum(schritt)


def _candidate_frames():
    rng = np.random.default_rng(110)
    stempel = _kamera_stempel(rng)
    zeiten = np.concatenate([rng.uniform(stempel[0] - 0.3, stempel[-1] + 0.3, size=300),
                             stempel[[0, 7, 200, 201, 599]],
                             (stempel[10:14] + stempel[11:15]) / 2.0])
    out = {}
    for name, max_dt, k in (("k3", 0.08, 3), ("k1", 0.08, 1), ("k5_weit", 0.25, 5),
                            ("k3_eng", 0.03, 3)):
        out[name] = _listen(colorizer._candidate_frames(stempel, float(t), max_dt, k)
                            for t in zeiten)
    out["ohne_stempel"] = _listen([colorizer._candidate_frames(np.zeros(0), 1.0, 0.08, 3)])
    return out


def _scan_candidates():
    rng = np.random.default_rng(111)
    stempel = _kamera_stempel(rng)
    zeiten = np.concatenate([rng.uniform(stempel[0] - 1.5, stempel[-1] + 1.5, size=300),
                             stempel[[0, 3, 199, 200, 201, 410, 599]],
                             (stempel[10:14] + stempel[11:15]) / 2.0])
    out = {}
    for name, k, max_dt in (("k3", 3, 0.08), ("k1", 1, 0.08), ("k5", 5, 0.2),
                            ("k3_eng", 3, 0.03)):
        for blau in (False, True):
            out[f"{name}_{'blau' if blau else 'ohne'}"] = _listen(
                colorizer._scan_candidates(stempel, float(t), k, max_dt, blau)
                for t in zeiten)
    return out


# ---------------------------------------------------------------- stitcher

def _kamerapunkte(rng, n: int) -> np.ndarray:
    feste = np.array([[0, 0, 0], [0, 0, 1], [0, 0, -1], [1, 0, 0], [0, -1, 0],
                      [1e-7, 0, 1e-7], [3, 4, -0.2], [-3, 4, -6]], dtype=np.float64)
    return np.concatenate([feste, rng.normal(size=(n, 3)) * 5.0])


def _project():
    rng = np.random.default_rng(201)
    cam0, cam1, T01 = stitcher.DoubleSphereCamera.from_calib(_calib())
    # alpha <= 0.5: der andere Zweig der Gültigkeitsgrenze
    cam2 = stitcher.DoubleSphereCamera(310.0, 305.0, 640.0, 512.0, -0.2, 0.45)
    pts = _kamerapunkte(rng, 20000)
    out = {"T_cam0_cam1": T01}
    for name, cam in (("cam0", cam0), ("cam1", cam1), ("alpha_klein", cam2)):
        uv, ok = cam.project(pts)
        out[f"{name}_uv"] = uv
        out[f"{name}_gueltig"] = ok
    uv, ok = cam0.project(pts.astype(np.float32))
    out["cam0_float32_uv"] = uv
    out["cam0_float32_gueltig"] = ok
    uv, ok = cam1.project(pts[:6000].reshape(60, 100, 3))
    out["cam1_raster_uv"] = uv
    out["cam1_raster_gueltig"] = ok
    return out


def _unproject():
    rng = np.random.default_rng(202)
    cam0, cam1, _ = stitcher.DoubleSphereCamera.from_calib(_calib())
    cam2 = stitcher.DoubleSphereCamera(310.0, 305.0, 640.0, 512.0, -0.2, 0.45)
    uv = rng.uniform(-100.0, 1620.0, size=(20000, 2))
    out = {}
    for name, cam in (("cam0", cam0), ("cam1", cam1), ("alpha_klein", cam2)):
        d, ok = cam.unproject(uv)
        out[f"{name}_richtung"] = d
        out[f"{name}_gueltig"] = ok
    return out


def _pano_rays():
    return {"b256": stitcher._pano_rays(256, 128),
            "b640": stitcher._pano_rays(640, 320),
            "b6_h5": stitcher._pano_rays(6, 5)}


# ------------------------------------------- check_extrinsic, auto_calibrate

_STUFE_GRAD = 4.0      # Breite eines Plateaus
_STUFE_HOEHE = 0.02    # Score-Abfall je Plateau
# Gipfel (Yaw, Pitch, Roll in Grad; Höhe). Krumme Winkel, damit kein
# Kandidat des Gitters genau auf einer Stufenkante liegt.
_GIPFEL = (((-92.3, 1.7, 88.9), 0.9), ((63.1, -21.4, -170.6), 0.74),
           ((151.9, 33.2, 12.3), 0.58))


class _StufenScore:
    """Stellvertreter für ``_PhotoScoreContext`` mit gestuftem Score.

    Der Score fällt mit dem Winkelabstand zum nächsten Gipfel in Stufen von
    ``_STUFE_GRAD``; innerhalb einer Stufe sind alle Drehungen gleich gut. Die
    Gipfel hängen an den Anker-Frames (der Validierungssatz bewertet also
    etwas anders als der Optimierungssatz). Wie das Original scheitert der Bau
    mit weniger als drei Ankern. ``protokoll`` sammelt je Kontext die Anker
    und alle bewerteten Drehungen in Aufruffolge.
    """

    protokoll: list = []

    def __init__(self, rec, bag, calib_json, anchor_frames, cancel=None):
        self.anker = [int(f) for f in anchor_frames]
        if len(self.anker) < 3:
            raise RuntimeError("zu wenige Frame-Paare (Stellvertreter)")
        versatz = float(sum(self.anker) % 7) - 3.0
        self._gipfel = [
            (Rotation.from_euler("ZYX", [y + versatz, p, r], degrees=True).as_matrix(), h)
            for (y, p, r), h in _GIPFEL]
        self.aufrufe: list = []
        type(self).protokoll.append(self)

    def score(self, R_imu_cam0) -> float:
        R = np.array(R_imu_cam0, dtype=np.float64)
        self.aufrufe.append(R)
        bester = -1.0
        for ziel, hoehe in self._gipfel:
            c = (float(np.trace(ziel.T @ R)) - 1.0) / 2.0
            grad = float(np.degrees(np.arccos(min(1.0, max(-1.0, c)))))
            bester = max(bester, hoehe - _STUFE_HOEHE * np.floor(grad / _STUFE_GRAD))
        return float(round(bester, 6))


class _StubRec:
    """Nur was ``_default_score_frames`` braucht: Scan-Stempel."""

    def __init__(self):
        self.stamps = 1000.0 + np.arange(400, dtype=np.float64) * 0.1


class _StubBag:
    """Kamera-Stempel und ein leeres Bild in der erwarteten Auflösung."""

    bag_path = None

    def __init__(self):
        self._stempel = 999.5 + np.arange(480, dtype=np.float64) / 11.0
        self._bild = np.zeros((1520, 3040, 3), dtype=np.uint8)

    def camera_stamps(self):
        return self._stempel

    def read_camera(self, idx):
        return self._bild


def _mit_stufenscore(lauf) -> dict:
    """Fährt ``lauf()`` mit dem Stellvertreter und hängt das Protokoll an."""
    original = colorizer._PhotoScoreContext
    _StufenScore.protokoll = []
    basis.ueberall_ersetzen(original, _StufenScore)
    try:
        out = lauf()
    finally:
        basis.ueberall_ersetzen(_StufenScore, original)
        kontexte, _StufenScore.protokoll = _StufenScore.protokoll, []
    if not kontexte:
        raise RuntimeError("Der Stellvertreter wurde nicht benutzt — "
                           "_PhotoScoreContext wird woanders gebaut.")
    out["anker"] = _listen(k.anker for k in kontexte)
    out["aufrufe_je_kontext"] = np.asarray([len(k.aufrufe) for k in kontexte], np.int64)
    alle = [R for k in kontexte for R in k.aufrufe]
    out["aufrufe"] = np.stack(alle) if alle else np.zeros((0, 3, 3))
    return out


def _lage(yaw: float, pitch: float, roll: float, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler("ZYX", [yaw, pitch, roll], degrees=True).as_matrix()
    T[:3, 3] = t
    return T


def _check_extrinsic(T: np.ndarray, frames=None):
    def probe():
        def lauf():
            res = colorizer.check_extrinsic(_StubRec(), _StubBag(), _calib(), T,
                                            frames=frames)
            fremd = sorted(set(res) - {"score", "best_score", "best_T", "dist_deg",
                                       "suspect"})
            if fremd:
                raise RuntimeError(f"check_extrinsic liefert unbekannte Schlüssel {fremd}")
            return {"score": float(res["score"]), "best_score": float(res["best_score"]),
                    "best_T": np.asarray(res["best_T"]),
                    "dist_deg": float(res["dist_deg"]), "suspect": int(res["suspect"])}
        return _mit_stufenscore(lauf)
    return probe


def _auto_calibrate(T_init=None, frames=None):
    def probe():
        def lauf():
            anteile: list = []
            texte: list = []

            def fortschritt(anteil, text):
                anteile.append(float(anteil))
                texte.append(str(text))

            T, score = colorizer.auto_calibrate(_StubRec(), _StubBag(), _calib(),
                                                T_init=T_init, frames=frames,
                                                progress_cb=fortschritt)
            return {"T": np.asarray(T), "score": float(score),
                    "fortschritt": np.asarray(anteile, dtype=np.float64),
                    "meldungen": "\n".join(texte).encode("utf-8")}
        return _mit_stufenscore(lauf)
    return probe


# ------------------------------------------------------------ seg0-Einfärbung

# Extrinsik des Selbsttests von core.colorizer (Yaw -92, Roll 90)
_T_SEG0 = _lage(-92.0, 0.0, 90.0)


def _seg0_lauf(rechne) -> dict:
    """Frische Cache-Kopie, Aufzeichnung und Bag für ``rechne(rec, bag, ordner)``.

    Scheitert, wenn sich danach eine meta.json des echten Caches bewegt hat.
    """
    from core.bag_reader import BagReader
    from core.project import Project
    from core.recording import Recording

    echt_vorher = basis.meta_zeiten()
    kopie = basis.cache_kopie()
    projekt = Project(kopie.bag)
    if os.path.realpath(projekt.dir) != os.path.realpath(kopie.projekt):
        raise RuntimeError("Das Projekt zeigt nicht auf die Cache-Kopie.")
    rec = Recording.load(projekt.recording_dir(), bag_path=kopie.bag)
    ordner = basis.arbeitsordner("colorizer")
    bag = BagReader(kopie.bag)
    try:
        out = rechne(rec, bag, ordner)
    finally:
        bag.close()
        del rec
    if basis.meta_zeiten() != echt_vorher:
        raise RuntimeError("Eine recording/meta.json im echten Cache hat sich geändert.")
    return out


_ZAEHLER = ("n_valid", "n_sky_blocked", "n_sky_outvoted", "n_edge_outvoted",
            "n_blue_outvoted", "n_blue_kept", "n_blue_neutral")


# Schlüssel der meta.json, die von Lauf zu Lauf wechseln
_META_UNSTET = ("runtime_s", "created")


def _neutral(pfad):
    """Pfad ohne den Ort auf dieser Maschine (Repo-Wurzel, Bag-Ordner)."""
    if pfad is None:
        return None
    pfad = str(pfad)
    if os.path.abspath(pfad) == os.path.abspath(basis.SEG0_BAG):
        return "<seg0>"
    rel = os.path.relpath(os.path.abspath(pfad), basis.wurzel())
    if not rel.startswith(os.pardir):
        return rel.replace(os.sep, "/")
    return "<fremd>/" + os.path.basename(pfad)


def _meta(ziel: str) -> bytes:
    """Die meta.json der Einfärbung, vergleichbar gemacht.

    Schlüsselfolge und Werte bleiben wie geschrieben; es fehlen nur Laufzeit
    und Datum, die Pfade stehen neutral. Fehlt einer der erwarteten Schlüssel,
    scheitert die Probe, statt ihn still zu übergehen.
    """
    if os.path.exists(os.path.join(ziel, "meta.json.tmp")):
        raise RuntimeError("meta.json.tmp ist liegen geblieben.")
    with open(os.path.join(ziel, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    fehlt = sorted(set(_META_UNSTET + ("calib_json", "bag", "bags")) - set(meta))
    if fehlt:
        raise RuntimeError(f"meta.json der Einfärbung ohne {fehlt}")
    for k in _META_UNSTET:
        del meta[k]
    meta["calib_json"] = _neutral(meta["calib_json"])
    meta["bag"] = _neutral(meta["bag"])
    meta["bags"] = [_neutral(b) for b in meta["bags"]]
    return json.dumps(meta, indent=2, ensure_ascii=False).encode("utf-8")


def _einfaerben(rec, bag, ordner: str, name: str, params, parts=None) -> dict:
    ziel = os.path.join(ordner, name)
    anteile: list = []
    res = colorizer.colorize(rec, bag, _calib(), params, ziel,
                             progress_cb=lambda a, _t: anteile.append(float(a)),
                             parts=parts, backend="cpu")
    if res.get("gpu") is not None:
        raise RuntimeError("colorize lief nicht auf der CPU.")
    out = {}
    for datei in ("colors.bin", "valid.bin"):
        with open(os.path.join(ziel, datei), "rb") as fh:
            out[datei] = fh.read()
    out["meta.json"] = _meta(ziel)
    for z in _ZAEHLER:
        out[z] = int(res[z])
    out["frac_valid"] = float(res["frac_valid"])
    out["fortschritt"] = np.asarray(anteile, dtype=np.float64)
    return out


def _colorize_seg0():
    def rechne(rec, bag, ordner):
        params = colorizer.ColorizeParams(T_imu_cam0=_T_SEG0)
        out = _einfaerben(rec, bag, ordner, "voll", params)
        out["n_points"] = int(rec.n_points)
        out["n_scans"] = int(rec.n_scans)
        return out
    return _seg0_lauf(rechne)


_TEIL_VON, _TEIL_BIS = 200, 240


def _ausschnitt(rec):
    """Die Scans ``_TEIL_VON`` bis ``_TEIL_BIS`` als eigene Aufzeichnung."""
    from core.recording import Recording

    a, b = int(rec.offsets[_TEIL_VON]), int(rec.offsets[_TEIL_BIS])
    return Recording(points=np.asarray(rec.points[a:b]),
                     intensity=np.asarray(rec.intensity[a:b]),
                     offsets=rec.offsets[_TEIL_VON:_TEIL_BIS + 1] - a,
                     stamps=rec.stamps[_TEIL_VON:_TEIL_BIS],
                     poses=rec.poses[_TEIL_VON:_TEIL_BIS],
                     meta=rec.meta, gravity_level=rec.gravity_level)


def _colorize_teil():
    def rechne(rec, bag, ordner):
        teil = _ausschnitt(rec)
        halb = teil.n_scans // 2
        out = {}
        # Blaulichtfilter samt Restblau, zwei Abschnitte auf demselben Bag
        blau = colorizer.ColorizeParams(T_imu_cam0=_T_SEG0, blue_filter=True)
        for k, v in _einfaerben(teil, bag, ordner, "blau", blau,
                                parts=[(bag, 0, halb), (bag, halb, teil.n_scans)]).items():
            out[f"blau_{k}"] = v
        # ohne Linsenwahl, Himmelssaum und Rangfolge; ein Frame je Scan
        schlicht = colorizer.ColorizeParams(
            T_imu_cam0=_T_SEG0, lens_best=False, sky_grow=0, sky_prefer=False,
            edge_r=700.0, k_frames=1, blue_filter=True, blue_least=False,
            blue_neutral=0.5, blue_hue_lo=200.0, blue_hue_hi=240.0, blue_sat=0.3,
            blue_val=40.0)
        for k, v in _einfaerben(teil, bag, ordner, "schlicht", schlicht).items():
            out[f"schlicht_{k}"] = v
        out["n_points"] = int(teil.n_points)
        return out
    return _seg0_lauf(rechne)


PROBEN = {
    "colorizer._is_blue": _is_blue,
    "colorizer.neutralize_blue": _neutralize_blue,
    "colorizer.blue_preview": _blue_preview,
    "colorizer._blown_mask": _blown_mask,
    "colorizer._not_blown": _not_blown,
    "colorizer._is_skyish": _is_skyish,
    "colorizer._sample_bgr": _sample_bgr,
    "colorizer._in_circle": _in_circle,
    "colorizer._equirect_uv": _equirect_uv,
    "colorizer._candidate_frames": _candidate_frames,
    "colorizer._scan_candidates": _scan_candidates,
    # check_extrinsic: auf dem Gipfel (mit Translation), 12 Grad daneben,
    # weit weg, auf einem Nebengipfel mit eigenen Frames
    "colorizer.check_extrinsic_gipfel": _check_extrinsic(
        _lage(-92.3, 1.7, 88.9, t=(0.03, -0.01, 0.12))),
    "colorizer.check_extrinsic_verdreht": _check_extrinsic(_lage(-80.3, 1.7, 88.9)),
    "colorizer.check_extrinsic_fern": _check_extrinsic(np.eye(4)),
    "colorizer.check_extrinsic_nebengipfel": _check_extrinsic(
        _lage(58.0, -17.0, -166.0), frames=[5, 50, 120, 200]),
    # auto_calibrate: Vorgaben; mit Startwert und eigenen Frames; mit so wenigen
    # Frames, dass es keinen Validierungssatz gibt
    "colorizer.auto_calibrate": _auto_calibrate(),
    "colorizer.auto_calibrate_startwert": _auto_calibrate(
        T_init=_lage(150.0, 30.0, 10.0, t=(0.1, 0.2, 0.3)),
        frames=[400, 12, 96, 180, 250, 330]),
    "colorizer.auto_calibrate_ohne_validierung": _auto_calibrate(frames=[10, 11, 12]),
    "colorizer.colorize_seg0": _colorize_seg0,
    "colorizer.colorize_seg0_teil": _colorize_teil,
    "stitcher.project": _project,
    "stitcher.unproject": _unproject,
    "stitcher._pano_rays": _pano_rays,
}

UNSTET: dict = {}

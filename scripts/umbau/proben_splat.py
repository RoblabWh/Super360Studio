"""Numerik-Proben für ``core.splat`` und ``core.fusion``.

Alles rechnet auf einer synthetischen, geseedeten Szene: eine Karte mit
Relief und einem Dach, sieben Kameras darüber (eine davon sieht die Karte
nicht), Bilder und Temperaturfelder aus festen Formeln. Es läuft kein
Training und kein fremder Interpreter; ``ergebnis.npz`` des Trainers wird für
``punkt_farben`` aus festen Werten geschrieben.

Die Datensätze entstehen in einem Arbeitsordner unter
``/tmp/super360_modtests/umbau``. .npz-Dateien werden je Feld gehasht, nicht
als Datei: das Zip trägt die Uhrzeit des Schreibens.

Die Normalen aus open3d (``anker``) stehen in einer eigenen Probe und sind
über wiederholte Läufe bitgleich; ``UNSTET`` bleibt deshalb leer.
"""
from __future__ import annotations

import functools
import os

import numpy as np

import basis
from core import fusion, splat


# ------------------------------------------------------------------ Helfer

def _text(wert) -> bytes:
    return ("<None>" if wert is None else str(wert)).encode("utf-8")


class _Mitschnitt:
    """Fortschritt und Logzeilen eines Aufrufs."""

    def __init__(self):
        self.anteile: list = []
        self.meldungen: list = []
        self.zeilen: list = []

    def progress(self, f, m):
        self.anteile.append(float(f))
        self.meldungen.append(str(m))

    def log(self, zeile):
        self.zeilen.append(str(zeile))

    def teile(self) -> dict:
        return {"fortschritt": np.asarray(self.anteile, np.float64),
                "meldungen": _text("\n".join(self.meldungen)),
                "log": _text("\n".join(self.zeilen))}


def _fehler(fn, *a, **k) -> bytes:
    try:
        fn(*a, **k)
    except Exception as exc:  # noqa: BLE001 — die Meldung ist der Wert
        return _text(f"{type(exc).__name__}: {exc}")
    return _text("kein Fehler")


def _drehung(gier, nick, roll) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("zyx", [gier, nick, roll], degrees=True).as_matrix()


# ------------------------------------------------------------------- Szene

N_PUNKTE = 60_000
N_KAMERAS = 7            # die letzte steht weit neben der Karte
BILD = (400, 300)        # abgelegte Bilder: halbe Größe der Kameras
THERMAL = (320, 240)


@functools.lru_cache(maxsize=None)
def _karte() -> np.ndarray:
    rng = np.random.default_rng(20260711)
    xy = rng.uniform(-10.0, 10.0, (N_PUNKTE, 2))
    z = 0.4 * np.sin(0.5 * xy[:, 0]) * np.cos(0.4 * xy[:, 1]) + rng.normal(0, 0.01, N_PUNKTE)
    dach = (xy[:, 0] > 2.0) & (xy[:, 0] < 5.0) & (xy[:, 1] > -3.0) & (xy[:, 1] < 0.0)
    z[dach] += 2.5
    return np.c_[xy, z].astype(np.float32)


@functools.lru_cache(maxsize=None)
def _anker() -> dict:
    return splat.anker(_karte(), voxel=0.1)


def _lage(aehnlich: bool):
    Q = _drehung(80.0, 3.0, -2.0)
    A = 1.3 * Q
    if not aehnlich:
        A = A @ (np.eye(3) + 3e-3 * np.array([[0.0, 1.0, 0.5], [1.0, 0.0, -0.7],
                                              [0.5, -0.7, 0.0]]))
    return A, np.array([12.0, -4.0, 3.0]), Q


def _kameras(aehnlich: bool) -> tuple:
    """Kameras im COLMAP-Rahmen, so dass sie im Kartenrahmen senkrecht blicken."""
    A, b, Q = _lage(aehnlich)
    Ai = np.linalg.inv(A)
    orte = [(-4.0, -3.0), (0.0, -3.0), (4.0, -3.0), (-4.0, 3.0), (0.0, 3.0), (4.0, 3.0),
            (200.0, 200.0)]
    Rcw, tcw, params = [], [], []
    for i, (x, y) in enumerate(orte):
        R_karte = np.diag([1.0, -1.0, -1.0]) @ _drehung(15.0 * i, 2.0 - i, 1.5 * i - 3.0)
        R = R_karte @ Q
        Cc = Ai @ (np.array([x, y, 25.0 + 0.5 * i]) - b)
        Rcw.append(R)
        tcw.append(-R @ Cc)
        params.append([700.0, 400.0, 300.0, -0.05] if i % 2 == 0
                      else [690.0, 404.0, 297.0, -0.03])
    cams = {"names": np.array([f"b{i:02d}.jpg" for i in range(N_KAMERAS)]),
            "Rcw": np.asarray(Rcw), "tcw": np.asarray(tcw),
            "size": np.tile([800.0, 600.0], (N_KAMERAS, 1)),
            "params": np.asarray(params), "model": np.array("SIMPLE_RADIAL")}
    return cams, A, b


def _bild(i: int) -> np.ndarray:
    w, h = BILD
    jj, ii = np.mgrid[0:h, 0:w].astype(np.float64)
    rng = np.random.default_rng(100 + i)
    bild = np.stack([128 + 80 * np.sin(0.05 * ii + i), 128 + 80 * np.cos(0.04 * jj - i),
                     100 + 60 * np.sin(0.03 * (ii + jj))], -1)
    return np.clip(bild + rng.normal(0, 4.0, bild.shape), 0, 255).astype(np.uint8)


def _temperatur(i: int, name: str):
    if i == 2:
        return None              # ein Bild ohne Rohwerte
    w, h = THERMAL
    jj, ii = np.mgrid[0:h, 0:w].astype(np.float64)
    rng = np.random.default_rng(200 + i)
    feld = 27.0 + 9.0 * np.sin(0.03 * ii + 0.5 * i) * np.cos(0.05 * jj) + rng.normal(0, 0.2, ii.shape)
    return feld.astype(np.float32)


@functools.lru_cache(maxsize=None)
def _ordner() -> str:
    return basis.arbeitsordner("proben_splat")


@functools.lru_cache(maxsize=None)
def _datensatz(art: str) -> tuple:
    """Baut den Datensatz ``rgb`` oder ``temperatur``: (ordner, info, mitschnitt)."""
    from PIL import Image

    ziel = os.path.join(_ordner(), art)
    mit = _Mitschnitt()
    if art == "rgb":
        cams, A, b = _kameras(aehnlich=False)
        bilder = os.path.join(_ordner(), "quelle")
        os.makedirs(bilder, exist_ok=True)
        for i, name in enumerate(cams["names"]):
            Image.fromarray(_bild(i)).save(os.path.join(bilder, str(name)), quality=95)
        info = splat.datensatz_maeander(ziel, _anker(), _karte(), cams, bilder, A, b,
                                        halte_jedes=3, skala=0.5, param={"schritte": 7},
                                        progress=mit.progress, log=mit.log)
    else:
        cams, A, b = _kameras(aehnlich=True)
        info = splat.datensatz_maeander(ziel, _anker(), _karte(), cams, "", A, b,
                                        temperatur=_temperatur, halte_jedes=2, skala=0.25,
                                        progress=mit.progress, log=mit.log)
    return ziel, info, mit


def _datensatz_teile(art: str) -> dict:
    ziel, info, mit = _datensatz(art)
    out = {f"info_{k}": v for k, v in info.items()}
    out.update(mit.teile())
    with np.load(os.path.join(ziel, "ansichten.npz")) as z:
        out.update({f"ansichten_{k}": z[k] for k in z.files})
    with np.load(os.path.join(ziel, "anker.npz")) as z:
        out.update({f"anker_{k}": z[k] for k in z.files})
    for name in ("param.json", "punkte.npy"):
        with open(os.path.join(ziel, name), "rb") as fh:
            out[name] = fh.read()
    namen = sorted(os.listdir(os.path.join(ziel, "bilder")))
    out["bilder_namen"] = _text("\n".join(namen))
    for name in namen:
        with open(os.path.join(ziel, "bilder", name), "rb") as fh:
            out[f"bild_{name}"] = fh.read()
    return out


# ------------------------------------------------------------------- splat

def _p_anker():
    mit = _Mitschnitt()
    P = _karte()
    ak = splat.anker(P, voxel=0.1, progress=mit.progress, log=mit.log)
    out = {"pos": ak["pos"], "index": ak["index"], "anzahl": ak["anzahl"],
           "voxel": ak["voxel"]}
    # zu viele Zellen: das Raster wächst
    eng = splat.anker(P, voxel=0.1, max_anker=5_000, progress=mit.progress, log=mit.log)
    out.update({"eng_pos": eng["pos"], "eng_index": eng["index"],
                "eng_anzahl": eng["anzahl"], "eng_voxel": eng["voxel"]})
    # Raster so fein, dass der Zellschlüssel überläuft: erst verdoppeln
    fein = splat.anker(P[:5_000], voxel=4e-6, max_anker=5_000,
                       progress=mit.progress, log=mit.log)
    # … und mit zu kleiner Obergrenze findet es in acht Anläufen kein Raster
    out["keine_zellgroesse"] = _fehler(splat.anker, P[:5_000], voxel=1e-6,
                                       max_anker=2_000, log=mit.log)
    out.update({"fein_pos": fein["pos"], "fein_index": fein["index"],
                "fein_anzahl": fein["anzahl"], "fein_voxel": fein["voxel"],
                "fein_normal": fein["normal"]})
    out["leer"] = _fehler(splat.anker, np.zeros((0, 3), np.float32))
    out.update(mit.teile())
    return out


def _p_anker_normalen():
    eng = splat.anker(_karte(), voxel=0.1, max_anker=5_000)
    return {"normal": _anker()["normal"], "eng_normal": eng["normal"]}


def _p_ansichten():
    out = {}
    for name, aehnlich in (("aehnlich", True), ("verzogen", False)):
        cams, A, b = _kameras(aehnlich)
        vm, info = splat.ansichten_aus_colmap(cams, A, b)
        out[f"{name}_viewmat"] = vm
        out[f"{name}_massstab"] = info["massstab"]
        out[f"{name}_abweichung"] = info["abweichung"]
    cams, A, b = _kameras(True)
    out["gespiegelt"] = _fehler(splat.ansichten_aus_colmap, cams,
                                A @ np.diag([1.0, 1.0, -1.0]), b)
    return out


def _p_entzerrung():
    out = {}
    faelle = (("simple_radial", "SIMPLE_RADIAL", [700.0, 400.0, 300.0, -0.05], 0.5),
              ("tonne", "SIMPLE_RADIAL", [350.0, 400.0, 300.0, -0.6], 0.25),
              ("pinhole", "PINHOLE", [710.0, 695.0, 402.0, 296.0], 0.5),
              ("opencv", "OPENCV", [710.0, 695.0, 402.0, 296.0, -0.08, 0.02, 1e-3, -5e-4],
               0.37))
    for name, model, params, skala in faelle:
        mx, my, K, gueltig = splat.entzerrung(model, params, (800.0, 600.0), BILD, skala)
        out.update({f"{name}_mx": mx, f"{name}_my": my, f"{name}_K": K,
                    f"{name}_gueltig": gueltig})
    out["lochkamera"] = np.asarray(
        [splat._lochkamera(m, [1.0, 2.0, 3.0, 4.0, 5.0])
         for m in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "PINHOLE", "OPENCV")])
    out["fremdes_modell"] = _fehler(splat._lochkamera, "FISHEYE", [1.0, 2.0, 3.0])
    out["modell"] = _text(splat._modell({"model": np.array("OPENCV")}) + "|"
                          + splat._modell({"model": "PINHOLE"}))
    return out


def _ansicht(i: int, skala: float = 0.5):
    cams, A, b = _kameras(True)
    vm, _ = splat.ansichten_aus_colmap(cams, A, b)
    _, _, K, _ = splat.entzerrung("SIMPLE_RADIAL", cams["params"][i], cams["size"][i],
                                  BILD, skala)
    return vm[i], K, int(round(800 * skala)), int(round(600 * skala))


def _p_abdeckung():
    ak = _anker()
    out = {}
    for i in (0, 4, 6):
        vm, K, W, H = _ansicht(i)
        out[f"kamera{i}"] = splat.abdeckung(ak["pos"], ak["voxel"], vm, K, W, H)
    # nah an der Karte: große Fußabdrücke und die Sperre unter min_weite
    vm = np.eye(4)
    vm[:3, :3] = np.diag([1.0, -1.0, -1.0]) @ _drehung(20.0, 25.0, -10.0)
    vm[:3, 3] = -vm[:3, :3] @ np.array([3.5, -1.5, 4.0])
    K = np.array([[160.0, 0.0, 161.0], [0.0, 160.0, 119.5], [0.0, 0.0, 1.0]])
    out["nah"] = splat.abdeckung(ak["pos"], ak["voxel"], vm, K, 322, 239)
    out["nah_min_weite"] = splat.abdeckung(ak["pos"], ak["voxel"], vm, K, 322, 239,
                                           min_weite=2.0)
    return out


def _p_zpuffer():
    ak = _anker()
    out = {}
    for name, i, skala in (("kamera1", 1, 0.25), ("kamera6", 6, 0.25), ("halb", 3, 0.5)):
        vm, K, W, H = _ansicht(i)
        puffer, w, h = splat._zpuffer(ak["pos"], vm.astype(np.float32),
                                      K.astype(np.float32), W, H + 1, skala)
        out.update({f"{name}_puffer": puffer, f"{name}_w": w, f"{name}_h": h})
    return out


def _p_probe():
    ak = _anker()
    return {"vorgabe": splat.probe(ak), "seed3": splat.probe(ak, zellen=8_000, seed=3),
            "seed4": splat.probe(ak, zellen=500, seed=4),
            "pruefbild": np.asarray([[splat._pruefbild(i, h) for i in range(24)]
                                     for h in (0, 1, 2, 3, 8, 10)])}


def _p_temperatur_farben():
    rng = np.random.default_rng(31)
    temp = rng.normal(28.0, 6.0, 40_000).astype(np.float32)
    maske = rng.uniform(0, 1, len(temp)) > 0.2
    temp[~maske] = np.nan
    flach = np.full(100, 21.5, np.float32)
    return {"rgb": splat.temperatur_farben(temp, maske),
            "ohne_maske": splat.temperatur_farben(temp[:50], np.zeros(50, bool)),
            "flach": splat.temperatur_farben(flach, np.ones(100, bool))}


def _ergebnis(ziel: str, kanaele: int) -> None:
    rng = np.random.default_rng(41 + kanaele)
    wert = rng.uniform(-0.1, 1.2, (N_PUNKTE, kanaele)).astype(np.float16)
    guete = np.where(rng.uniform(0, 1, N_PUNKTE) > 0.3,
                     rng.uniform(0.1, 1.0, N_PUNKTE), 0.0).astype(np.float16)
    np.savez(os.path.join(ziel, "ergebnis.npz"), punkt_wert=wert, punkt_guete=guete)


def _p_punkt_farben():
    out = {}
    ziel, _, _ = _datensatz("rgb")
    _ergebnis(ziel, 3)
    pf = splat.punkt_farben(ziel, N_PUNKTE)
    out.update({"rgb": pf["rgb"], "maske": pf["maske"],
                "schluessel": _text(",".join(sorted(pf)))})
    out["falsche_karte"] = _fehler(splat.punkt_farben, ziel, N_PUNKTE + 1)
    np.savez(os.path.join(ziel, "ergebnis.npz"), farbe=np.zeros(3))
    out["alte_version"] = _fehler(splat.punkt_farben, ziel, N_PUNKTE)
    os.remove(os.path.join(ziel, "ergebnis.npz"))

    ziel, _, _ = _datensatz("temperatur")
    _ergebnis(ziel, 1)
    pf = splat.punkt_farben(ziel, N_PUNKTE)
    out.update({"t_rgb": pf["rgb"], "t_maske": pf["maske"], "t_temperatur": pf["temperatur"],
                "t_schluessel": _text(",".join(sorted(pf)))})
    os.remove(os.path.join(ziel, "ergebnis.npz"))
    return out


def _vergleich_teile(st: dict, vorsatz: str) -> dict:
    out = {f"{vorsatz}bilder": st["bilder"], f"{vorsatz}punkte": st["punkte"],
           f"{vorsatz}methoden": _text(",".join(st["methoden"]))}
    for nm, werte in st["methoden"].items():
        out[f"{vorsatz}{nm}_roh"] = werte["roh"]
        out[f"{vorsatz}{nm}_angepasst"] = werte["angepasst"]
    return out


def _p_vergleich():
    ak, P = _anker(), _karte()
    idx = splat.probe(ak, zellen=8_000, seed=3)
    punkte = P[idx]
    rng = np.random.default_rng(51)
    nrm = np.c_[rng.normal(0, 0.25, (len(idx), 2)), np.ones(len(idx))]
    nrm[::7] = (1.0, 0.0, 0.05)                  # streifend: fällt über min_cos heraus
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    out = {}

    ziel, _, _ = _datensatz("rgb")
    wahr = rng.uniform(20, 235, (len(idx), 3)).astype(np.float32)
    werte = {"splat": (wahr + rng.normal(0, 3, wahr.shape).astype(np.float32),
                       rng.uniform(0, 1, len(idx)) > 0.05),
             "direkt": (np.clip(wahr + rng.normal(0, 25, wahr.shape), 0, 255),
                        rng.uniform(0, 1, len(idx)) > 0.10)}
    mit = _Mitschnitt()
    st = splat.vergleich(ziel, ak, punkte, nrm, werte, progress=mit.progress)
    out.update(_vergleich_teile(st, ""))
    out["urteil"] = _text(splat.urteil(st, "splat", "direkt", "(0–255)"))
    out["fortschritt"] = mit.teile()["fortschritt"]
    out["meldungen"] = mit.teile()["meldungen"]
    st = splat.vergleich(ziel, ak, punkte, nrm, werte, min_cos=0.9)
    out.update(_vergleich_teile(st, "steil_"))
    out["pruef_namen"] = _text("\n".join(splat.pruef_namen(ziel)))

    ziel, _, _ = _datensatz("temperatur")
    grad = rng.normal(28.0, 5.0, len(idx)).astype(np.float32)
    werte = {"splat": (grad, np.ones(len(idx), bool)),
             "direkt": (grad + rng.normal(0, 1.5, len(idx)).astype(np.float32),
                        rng.uniform(0, 1, len(idx)) > 0.2)}
    st = splat.vergleich(ziel, ak, punkte, nrm, werte)
    out.update(_vergleich_teile(st, "t_"))
    out["t_urteil"] = _text(splat.urteil(st, "splat", "direkt", "K"))
    out["t_pruef_namen"] = _text("\n".join(splat.pruef_namen(ziel)))
    return out


def _p_urteil():
    def stats(punkte, s, d):
        return {"bilder": 4, "punkte": punkte,
                "methoden": {"a": {"roh": s * 1.5, "angepasst": s},
                             "b": {"roh": d * 1.25, "angepasst": d}}}
    return {"splat_besser": _text(splat.urteil(stats(1_234_567, 4.25, 6.5), "a", "b", "(0–255)")),
            "direkt_besser": _text(splat.urteil(stats(812, 0.75, 0.5), "a", "b", "K")),
            "gleich": _text(splat.urteil(stats(100, 2.0, 2.0), "a", "b", "K")),
            "null": _text(splat.urteil(stats(5_000, 0.0, 0.0), "a", "b", "K")),
            "zu_wenig": _text(splat.urteil(stats(99, 1.0, 2.0), "a", "b", "K")),
            "ohne_punkte": _text(splat.urteil({"bilder": 0}, "a", "b", "K"))}


def _p_cams_ohne():
    cams, _, _ = _kameras(True)
    out = splat.cams_ohne(cams, ["b01.jpg", "b04.jpg", "fehlt.jpg"])
    return {"names": out["names"], "Rcw": out["Rcw"], "tcw": out["tcw"],
            "size": out["size"], "params": out["params"], "model": out["model"],
            "schluessel": _text(",".join(sorted(out)))}


def _p_starre_pixel():
    rng = np.random.default_rng(61)
    h, w = 60, 80
    yy, xx = np.mgrid[0:h, 0:w]
    kreis = (xx - 40) ** 2 + (yy - 30) ** 2 < 28 ** 2
    proben = []
    for _ in range(16):
        g = rng.uniform(40, 200, (h, w))
        g[20:26, 30:44] = 3.0 + rng.normal(0, 1.0, (6, 14))     # Arm: dunkel, fest
        g[5:10, 35:45] = 250.0                                  # Himmel: hell, fest
        proben.append(np.clip(g, 0, 255).astype(np.uint8))
    starr = splat._starre_pixel(proben, kreis)
    stehend = splat._starre_pixel([proben[0]] * 12, kreis)
    return {"starr": starr, "zu_wenige": _text(splat._starre_pixel(proben[:9], kreis)),
            "stehend": _text(stehend)}


def _p_hinweis():
    faelle = {
        "ohne_torch": {"fehler": "kein Interpreter"},
        "ohne_gsplat": {"torch": "2.7.0", "gsplat": False},
        "gsplat_falsch_gebaut": {"torch": "2.7.0", "gsplat": True, "cuda": True,
                                 "gsplat_rechnet": False, "faehigkeit": "8.6",
                                 "gpu": "Karte A", "gsplat_fehler": "no kernel image",
                                 "gsplat_version": "1.5.3"},
        "bereit": {"torch": "2.7.0", "gsplat": True, "cuda": True, "gsplat_rechnet": True},
        "rechnet_nicht": {"torch": "2.4.0", "gsplat": True, "cuda": False, "rechnet": False,
                          "gpu": "Karte B", "faehigkeit": "12.0",
                          "architekturen": ["sm_80", "sm_86"], "cuda_fehler": "no kernel image"},
    }
    return {name: _text(splat.hinweis(info)) for name, info in faelle.items()}


# ------------------------------------------------------------------ fusion

def _farbpaar(n: int, seed: int):
    """Quelle und Ziel 0..1 mit bekannter Abbildung, Rauschen und Fehlgriffen."""
    rng = np.random.default_rng(seed)
    M0 = np.array([[0.9, 0.05, 0.0], [0.02, 1.1, -0.03], [0.0, 0.08, 0.8]])
    t0 = np.array([0.03, -0.02, 0.05])
    q = rng.uniform(0.1, 0.8, (n, 3))
    z = np.clip(q @ M0.T + t0 + rng.normal(0, 0.01, q.shape), 0, 1)
    z[: n // 5] = rng.uniform(0, 1, (n // 5, 3))
    return rng, q, z, M0, t0


def _p_schaetze_abbildung():
    rng, q, z, _, _ = _farbpaar(30_000, 71)
    M, t = fusion.schaetze_abbildung(q, z)
    g = rng.uniform(0.1, 1.0, len(q))
    Mg, tg = fusion.schaetze_abbildung(q, z, gewicht=g)
    M32, t32 = fusion.schaetze_abbildung(q.astype(np.float32), z.astype(np.float32),
                                         gewicht=g.astype(np.float32))
    Me, te = fusion.schaetze_abbildung(q[:500], q[:500])
    return {"M": M, "t": t, "M_gewicht": Mg, "t_gewicht": tg, "M_float32": M32,
            "t_float32": t32, "M_identisch": Me, "t_identisch": te}


def _p_hilfen():
    rng = np.random.default_rng(81)
    rgb = rng.integers(0, 256, (20_000, 3), dtype=np.uint8)
    M = np.array([[0.9, 0.05, 0.0], [0.02, 1.1, -0.03], [0.0, 0.08, 0.8]])
    t = np.array([0.03, -0.02, 0.05])
    nrm = rng.normal(0, 1, (20_000, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    a = rng.uniform(0, 1, (5_001, 3)).astype(np.float32)
    b = rng.uniform(0, 1, (5_001, 3)).astype(np.float32)
    return {"anwenden": fusion.anwenden(rgb, M, t),
            "anwenden_float32": fusion.anwenden(rgb, M.astype(np.float32), t.astype(np.float32)),
            "gewicht_maeander": fusion.gewicht_maeander(nrm),
            "gewicht_maeander_float32": fusion.gewicht_maeander(nrm.astype(np.float32)),
            "gewicht_stufen": fusion.gewicht_maeander(
                np.c_[np.zeros((11, 2)), np.linspace(-1.0, 1.0, 11)]),
            "abstand": fusion._abstand(a, b),
            "konstanten": np.asarray([fusion.MAX_PROBEN, fusion.MIN_UEBERLAPPUNG,
                                      fusion.HUBER_K, fusion.ITERATIONEN, fusion.LAGE_VON,
                                      fusion.LAGE_BIS, fusion.GRUND, fusion._BLOCK])}


def _ebenen(n: int, seed: int):
    rng, q, z, M0, t0 = _farbpaar(n, seed)
    mea = np.rint(z * 255).astype(np.uint8)
    onb = np.rint(q * 255).astype(np.uint8)
    nrm = rng.normal(0, 1, (n, 3)).astype(np.float32)
    nrm[: n // 3] = (0.0, 0.0, 1.0)
    nrm[n // 3: n // 2] = (1.0, 0.0, 0.0)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    m_o = rng.uniform(0, 1, n) > 0.25
    m_m = rng.uniform(0, 1, n) > 0.30
    return (onb, m_o), (mea, m_m), nrm


def _fusion_teile(erg: dict, vorsatz: str = "") -> dict:
    out = {f"{vorsatz}rgb": erg["rgb"], f"{vorsatz}maske": erg["maske"],
           f"{vorsatz}M": erg["M"], f"{vorsatz}t": erg["t"],
           f"{vorsatz}schluessel": _text(",".join(sorted(erg)) + "|"
                                         + ",".join(erg["bericht"]))}
    for k, v in erg["bericht"].items():
        out[f"{vorsatz}bericht_{k}"] = _text(v) if (v is None or isinstance(v, str)) else v
    return out


def _p_fusioniere():
    onboard, maeander, nrm = _ebenen(90_000, 91)
    mit = _Mitschnitt()
    out = _fusion_teile(fusion.fusioniere(onboard, maeander, nrm, progress=mit.progress))
    out["fortschritt"] = mit.teile()["fortschritt"]
    out["meldungen"] = mit.teile()["meldungen"]
    out.update(_fusion_teile(fusion.fusioniere(onboard, maeander, nrm, seed=5), "seed5_"))
    return out


def _p_fusioniere_grenzen():
    out = {}
    # mehr Überlappung als MAX_PROBEN je Hälfte und mehrere Blöcke
    alt = fusion.MAX_PROBEN, fusion._BLOCK
    fusion.MAX_PROBEN, fusion._BLOCK = 4_000, 25_000
    try:
        onboard, maeander, nrm = _ebenen(60_000, 92)
        mit = _Mitschnitt()
        erg = fusion.fusioniere(onboard, maeander, nrm, progress=mit.progress)
        out.update(_fusion_teile(erg, "klein_"))
        out["klein_fortschritt"] = mit.teile()["fortschritt"]
    finally:
        fusion.MAX_PROBEN, fusion._BLOCK = alt
    # verschiedene Bildinhalte: die Abbildung gilt als unplausibel
    rng = np.random.default_rng(93)
    n = 20_000
    onb = rng.integers(0, 256, (n, 3), dtype=np.uint8)
    mea = rng.integers(0, 256, (n, 3), dtype=np.uint8)
    nrm = np.tile(np.float32([0.0, 0.0, 1.0]), (n, 1))
    alle = np.ones(n, bool)
    out.update(_fusion_teile(fusion.fusioniere((onb, alle), (mea, alle), nrm), "fremd_"))
    out["zu_wenig"] = _fehler(fusion.fusioniere, (onb, np.arange(n) < 100),
                              (mea, np.arange(n) >= 50), nrm)
    out["falsche_laenge"] = _fehler(fusion.fusioniere, (onb, alle), (mea, alle[:-1]), nrm)
    out["abbruch"] = _fehler(fusion.fusioniere, (onb, alle), (mea, alle), nrm,
                             cancel=lambda: True)
    return out


def _p_unplausibel():
    gut = {"abstand_vorher": 40.0, "abstand_nachher": 6.0}
    faelle = {
        "gut": (np.diag([0.9, 1.1, 0.8]), gut),
        "is7": ([[0.422, -0.325, 0.142], [0.128, -0.238, 0.322], [0.108, -0.575, 0.766]],
                {"abstand_vorher": 110.6, "abstand_nachher": 65.5}),
        "diagonale_klein": (np.diag([0.39, 1.0, 1.0]), gut),
        "nebenwerte": ([[1.0, 0.4, 0.21], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], gut),
        "grenze_nebenwerte": ([[1.0, 0.25, 0.25], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], gut),
        "senkt_kaum": (np.eye(3), {"abstand_vorher": 50.0, "abstand_nachher": 31.0}),
        "ohne_bericht": (np.eye(3), {}),
    }
    return {name: _text(fusion.unplausibel(M, [0.0, 0.0, 0.0], bericht))
            for name, (M, bericht) in faelle.items()}


PROBEN = {
    "splat.anker": _p_anker,
    "splat.anker_normalen": _p_anker_normalen,
    "splat.ansichten_aus_colmap": _p_ansichten,
    "splat.entzerrung": _p_entzerrung,
    "splat.abdeckung": _p_abdeckung,
    "splat.zpuffer": _p_zpuffer,
    "splat.probe": _p_probe,
    "splat.temperatur_farben": _p_temperatur_farben,
    "splat.datensatz_maeander": lambda: _datensatz_teile("rgb"),
    "splat.datensatz_maeander_temperatur": lambda: _datensatz_teile("temperatur"),
    "splat.punkt_farben": _p_punkt_farben,
    "splat.vergleich": _p_vergleich,
    "splat.urteil": _p_urteil,
    "splat.cams_ohne": _p_cams_ohne,
    "splat.starre_pixel": _p_starre_pixel,
    "splat.hinweis": _p_hinweis,
    "fusion.schaetze_abbildung": _p_schaetze_abbildung,
    "fusion.hilfen": _p_hilfen,
    "fusion.fusioniere": _p_fusioniere,
    "fusion.fusioniere_grenzen": _p_fusioniere_grenzen,
    "fusion.unplausibel": _p_unplausibel,
}

#: Über wiederholte Läufe war jede Probe bitgleich, auch die open3d-Normalen.
UNSTET: dict = {}

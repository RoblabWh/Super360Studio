"""Optik der Maeanderkameras einmessen: Hoehe, Brennweite, Thermalkamera.

Warum das noetig ist, am DRZ-Flug gemessen (DJI M30T, 57 m):

* **RGB-Brennweite.** Die Pipeline haelt die Optik bei COLMAP bewusst fest,
  damit sich keine Kuppel in die Rekonstruktion schleicht — mit dem Nennwert aus
  dem EXIF (24 mm KB). Der ist aber rund 11 % zu kurz. Jedes Bild wird dadurch
  zu klein auf die Karte projiziert; am Bildrand sind das Meter, und zwar in
  jedem Bild in eine andere Richtung. Genau das sieht aus wie „die Bilder sind
  zueinander verzerrt“. Die Fotopunkte liegen aus demselben Grund knapp 4 m
  ueber der Lidar-Oberflaeche (die Tiefe skaliert mit der Brennweite).
* **Thermal.** Die Pipeline nimmt fuer die Thermalbilder die Pose der RGB-Kamera
  und die Brennweite aus dem EXIF, ohne Verzeichnung. Tatsaechlich schielt die
  Thermalkamera um rund 2° gegen die RGB-Kamera (2 m am Boden), die Brennweite
  weicht ab, und das Objektiv hat eine kraeftige Tonnenverzeichnung.

Was hier passiert:

1. **Hoehe** ueber den Laser-Entfernungsmesser der Drohne: jedes Bild traegt im
   XMP den Abstand zum Boden entlang der Blickachse (``LRFTargetDistance``).
   Derselbe Abstand laesst sich in der Karte messen; der Unterschied ist die
   Hoehenkorrektur. Das muss zuerst kommen — auf ebenem Boden lassen sich
   Brennweite und Flughoehe gegeneinander tauschen, erst die Hoehe legt die
   Brennweite fest.
2. **RGB-Brennweite** aus der Tiefe der Fotopunkte: bei fester Brennweite und
   vom GPS gehaltenen Kameras skaliert ihre Tiefe mit der Brennweite, das
   Verhaeltnis Lidar-Tiefe zu Fototiefe ist der Faktor. Gegenprobe ueber die
   Farbkonsistenz (jeder Kartenpunkt in alle Bilder projiziert, die ihn sehen;
   stimmt die Brennweite, widersprechen sie sich weniger) — die allein legt
   den Faktor nicht fest, ihr Minimum ist zwischen 1,08 und 1,16 flach. Am
   DRZ-Flug stimmt das Ergebnis (1,084) mit der Physik der Thermalkamera
   ueberein (9,1 mm bei 12 µm Pixeln ergibt ueber die Einmessung 1,090).
3. **Thermaloptik** gegen das Weitwinkelbild desselben Ausloesers: beide
   Objektive sitzen Zentimeter auseinander, die Szene liegt 57 m entfernt, die
   Zuordnung ist also eine reine Drehung plus Brennweite, Hauptpunkt und
   Verzeichnung der Thermalkamera. Gemessen wird die Transinformation zwischen
   den Grauwerten — sie funktioniert zwischen Modalitaeten, wo Korrelation
   versagt. Kalibriert wird an einem Teil der Paare, geprueft an den anderen.

Qt-frei.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

DATEI = "optik_kalibrierung.json"

_LRF = re.compile(rb'LRFTargetDistance="?\s*([-+0-9.]+)')
_LRF_STATUS = re.compile(rb'LRFStatus="?\s*([A-Za-z]+)')


# ------------------------------------------------------------- Kameras

def _brennweiten_spalten(model: str) -> tuple:
    return (0, 1) if str(model) in ("PINHOLE", "OPENCV", "FULL_OPENCV") else (0,)


def _modell(cams) -> str:
    m = cams["model"]
    return m.item() if getattr(m, "shape", None) == () else str(m)


def cams_dict(cams) -> dict:
    """Die Schluessel, die das Einfaerben braucht, als veraenderbares dict."""
    return {"names": cams["names"], "Rcw": np.asarray(cams["Rcw"], float),
            "tcw": np.asarray(cams["tcw"], float),
            "size": np.asarray(cams["size"], float),
            "params": np.array(cams["params"], float),
            "model": np.array(_modell(cams))}


def rgb_cams(pipe, faktor: float = 1.0) -> dict:
    """RGB-Kameras der Pipeline (mit Hauptpunkt-Versatz), Brennweite mal ``faktor``."""
    d = cams_dict(pipe.rgb_cams())
    if faktor != 1.0:
        for s in _brennweiten_spalten(_modell(d)):
            d["params"][:, s] *= float(faktor)
    return d


def thermal_umrechnen(kal: dict, s: float) -> dict:
    """Thermalkalibrierung auf eine um ``s`` geaenderte RGB-Brennweite umrechnen.

    Eingemessen wird Thermal gegen RGB; beobachtet ist die Zuordnung der
    Pixel, nicht die Winkel. Wird die RGB-Brennweite spaeter um ``s``
    verlaengert, werden alle Winkel um ``s`` kleiner — die Thermalbrennweite
    waechst mit, die Verzeichnung (in normierten Koordinaten) mit ``s^2`` bzw.
    ``s^4``, die Schielwinkel quer zur Achse schrumpfen mit ``1/s``.
    """
    s = float(s)
    if s == 1.0:
        return dict(kal)
    rx, ry, rz = kal["drehung_grad"]
    return dict(kal, f=kal["f"] * s, k1=kal["k1"] * s ** 2, k2=kal["k2"] * s ** 4,
                drehung_grad=[rx / s, ry / s, rz])


def drehung(grad) -> np.ndarray:
    from scipy.spatial.transform import Rotation  # noqa: PLC0415
    return Rotation.from_rotvec(np.radians(np.asarray(grad, float))).as_matrix()


def thermal_cams(pipe, kal: dict | None, faktor: float = 1.0,
                 rgb_faktor: float = 1.0) -> dict | None:
    """Thermalkameras: RGB-Posen, gedreht um den Schielwinkel, eigene Optik.

    Ohne Kalibrierung das, was die Pipeline liefert (EXIF-Brennweite, keine
    Verzeichnung) — dann nur mit ``faktor`` auf der Brennweite.
    """
    if kal is None:
        c = pipe.thermal_cams()
        if c is None:
            return None
        d = cams_dict(c)
        d["params"][:, 0] *= float(faktor)
        return d
    k = thermal_umrechnen(kal, rgb_faktor / float(kal.get("rgb_faktor", 1.0)))
    R = drehung(k["drehung_grad"])
    Rcw = np.einsum("ij,njk->nik", R, np.asarray(pipe.cams["Rcw"], float))
    tcw = np.asarray(pipe.cams["tcw"], float) @ R.T
    n = len(Rcw)
    du, dv = (float(v) for v in getattr(pipe, "thermal_versatz", (0.0, 0.0)))
    return {"names": pipe.cams["names"], "Rcw": Rcw, "tcw": tcw,
            "size": np.tile([float(k["breite"]), float(k["hoehe"])], (n, 1)),
            "params": np.tile([k["f"] * float(faktor), k["cx"] + du, k["cy"] + dv,
                               k["k1"], k["k2"]], (n, 1)),
            "model": np.array("RADIAL")}


def foto_hoehe(P: np.ndarray, C: np.ndarray, faktor: float) -> np.ndarray:
    """Fotopunkte auf eine andere Brennweite umrechnen (fuer die Anzeige).

    Die Rekonstruktion lief mit fester Brennweite. Ist die wahre um ``faktor``
    laenger, liegen die Punkte um denselben Faktor zu nah an den Kameras — bei
    Nadirblick heisst das: zu hoch. Die Lage in der Ebene bleibt, die schneiden
    sich die Strahlen richtig.
    """
    if faktor == 1.0 or not len(P) or not len(C):
        return P
    P = np.array(P, float)
    cz = float(np.mean(C[:, 2]))
    P[:, 2] = cz - float(faktor) * (cz - P[:, 2])
    return P


# ------------------------------------------------------------ Datei

def laden(work_dir: str) -> dict:
    """Gespeicherte Kalibrierung; ohne Datei die neutrale."""
    d = {"rgb_faktor": 1.0, "thermal_faktor": 1.0, "thermal": None}
    try:
        with open(os.path.join(work_dir, DATEI), encoding="utf-8") as fh:
            gel = json.load(fh)
        d.update({k: gel[k] for k in gel})
        d["rgb_faktor"] = float(d["rgb_faktor"])
        d["thermal_faktor"] = float(d["thermal_faktor"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return d


def vorhanden(work_dir: str) -> bool:
    return os.path.isfile(os.path.join(work_dir, DATEI))


def speichern(work_dir: str, d: dict) -> None:
    os.makedirs(work_dir, exist_ok=True)
    tmp = os.path.join(work_dir, DATEI + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2)
    os.replace(tmp, os.path.join(work_dir, DATEI))


# ------------------------------------------------------------- Hoehe

def lies_lrf(pfad: str) -> float | None:
    """Laserabstand zum Boden aus dem DJI-XMP, oder None."""
    try:
        with open(pfad, "rb") as fh:
            kopf = fh.read(256 * 1024)
    except OSError:
        return None
    st = _LRF_STATUS.search(kopf)
    if st and st.group(1).lower() not in (b"normal",):
        return None
    m = _LRF.search(kopf)
    if not m:
        return None
    try:
        d = float(m.group(1))
    except ValueError:
        return None
    return d if 1.0 < d < 2000.0 else None


def _dsm(punkte: np.ndarray, res: float = 0.5):
    x0, y0 = float(punkte[:, 0].min()), float(punkte[:, 1].min())
    ix = ((punkte[:, 0] - x0) / res).astype(np.int64)
    iy = ((punkte[:, 1] - y0) / res).astype(np.int64)
    dsm = np.full((ix.max() + 1, iy.max() + 1), -np.inf)
    np.maximum.at(dsm, (ix, iy), punkte[:, 2])
    return dsm, x0, y0, res


def hoehe_aus_lrf(pipe, punkte: np.ndarray, A, b, foto_ordner: str) -> dict | None:
    """Hoehenkorrektur der Kameras aus dem Laser-Entfernungsmesser.

    Fuer jedes Bild den Abstand von der Kamera entlang der Blickachse bis zur
    Karte messen und mit dem Laserwert vergleichen. Rueckgabe: ``dz`` (um so
    viel muessen die Kameras hoch), Anzahl der Bilder, Streuung — oder None,
    wenn zu wenige Bilder einen Laserwert tragen.
    """
    dsm, x0, y0, res = _dsm(np.asarray(punkte, float))
    A, b = np.asarray(A, float), np.asarray(b, float)
    s = np.arange(5.0, 400.0, 0.05)
    diffs = []
    for i, name in enumerate(pipe.cams["names"]):
        lrf = lies_lrf(os.path.join(foto_ordner, str(name)))
        if lrf is None:
            continue
        C = A @ np.asarray(pipe.cams["C"][i], float) + b
        achse = A @ np.asarray(pipe.cams["Rcw"][i], float)[2]
        achse /= np.linalg.norm(achse)
        if achse[2] > -0.5:          # nicht steil nach unten: taugt nicht
            continue
        P = C[None, :] + s[:, None] * achse[None, :]
        jx = ((P[:, 0] - x0) / res).astype(np.int64)
        jy = ((P[:, 1] - y0) / res).astype(np.int64)
        drin = (jx >= 0) & (jx < dsm.shape[0]) & (jy >= 0) & (jy < dsm.shape[1])
        z = np.full(len(s), -np.inf)
        z[drin] = dsm[jx[drin], jy[drin]]
        unten = np.nonzero(P[:, 2] <= z)[0]
        # Nur Treffer in der Naehe des Laserwerts: ein Strahl durch eine
        # Luecke der Karte trifft sonst irgendwo tief darunter.
        if len(unten) and abs(s[unten[0]] - lrf) < 15.0:
            diffs.append(s[unten[0]] - lrf)
    if len(diffs) < 5:
        return None
    d = np.asarray(diffs)
    # Haeufigster Wert statt Median: sitzt die Lage in der Ebene noch nicht,
    # trifft ein Teil der Strahlen neben der Dachkante den Boden — das ergibt
    # eine zweite Haeufung Meter daneben, die den Median verschiebt.
    h, kanten = np.histogram(d, bins=120, range=(-15.0, 15.0))
    h = np.convolve(h, [1, 2, 3, 2, 1], mode="same")
    spitze = 0.5 * (kanten[np.argmax(h)] + kanten[np.argmax(h) + 1])
    nah = d[np.abs(d - spitze) < 1.0]
    wert = float(np.median(nah)) if len(nah) >= 3 else float(np.median(d))
    return {"dz": -wert, "n": int(len(d)), "n_spitze": int(len(nah)),
            "iqr": [float(np.percentile(d, 25)), float(np.percentile(d, 75))]}


# ----------------------------------------------------- RGB-Brennweite

class Farbkonsistenz:
    """Wie sehr widersprechen sich die Bilder? Streuung der Farben je Punkt.

    Die Bilder werden einmal verkleinert als Grauwerte geladen; ein Durchlauf
    ueber 60.000 Punkte und 60 Bilder dauert dann etwa eine Sekunde.
    """

    def __init__(self, cams, bild_ordner: str, breite: int = 800, cancel=None):
        from PIL import Image  # noqa: PLC0415
        self.names = [str(n) for n in cams["names"]]
        self.bilder = []
        self.skala = []
        for n in self.names:
            if cancel is not None and cancel():
                raise RuntimeError("Abgebrochen")
            with Image.open(os.path.join(bild_ordner, n)) as im:
                sk = breite / float(im.width)
                klein = im.convert("L").resize((breite, max(1, int(im.height * sk))),
                                                Image.BILINEAR)
            self.bilder.append(np.asarray(klein, np.float32))
            self.skala.append(sk)

    def streuung(self, punkte: np.ndarray, cams: dict, A, b) -> float:
        from colorize_pipeline import colorize as cz  # noqa: PLC0415
        P = np.asarray(punkte, float)
        PT = np.linalg.inv(np.asarray(A, float)) @ (P - np.asarray(b, float)).T
        s1 = np.zeros(len(P))
        s2 = np.zeros(len(P))
        n = np.zeros(len(P))
        model = _modell(cams)
        for i, img in enumerate(self.bilder):
            pc = (cams["Rcw"][i] @ PT).T + cams["tcw"][i]
            u, v, vorn, _ = cz._project(pc, cams["size"][i], cams["params"][i], model)
            W, H = cams["size"][i]
            ok = vorn & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            sk = self.skala[i]
            ih, iw = img.shape
            g = img[np.clip((v[ok] * sk).astype(np.int64), 0, ih - 1),
                    np.clip((u[ok] * sk).astype(np.int64), 0, iw - 1)]
            s1[ok] += g
            s2[ok] += g * g
            n[ok] += 1
        m = n >= 3
        if not m.any():
            return float("inf")
        var = s2[m] / n[m] - (s1[m] / n[m]) ** 2
        return float(np.sqrt(np.maximum(var, 0.0)).mean())


def schaetze_rgb_faktor_tiefe(pipe, punkte: np.ndarray, A, b) -> dict | None:
    """Brennweitenfaktor aus der Tiefe der Fotopunkte.

    Die Rekonstruktion lief mit fester Brennweite, die Kameras haelt das GPS.
    Dann skaliert die Tiefe der Fotopunkte unter den Kameras mit der
    Brennweite: ist die wahre um ``s`` laenger, liegen die Punkte um ``s`` zu
    nah. Das Verhaeltnis der Lidar-Tiefe zur Fototiefe ist also ``s`` — ein
    scharfes Mass, anders als die Farbkonsistenz, die zwischen 1,08 und 1,16
    kaum unterscheidet. Setzt die richtige Hoehe voraus (``hoehe_aus_lrf``).
    """
    A, b = np.asarray(A, float), np.asarray(b, float)
    try:
        C = (A @ np.asarray(pipe.cams["C"], float).T).T + b
        P = (A @ np.asarray(pipe.cams["xyz"], float).T).T + b
    except (KeyError, TypeError):
        return None
    dsm, x0, y0, res = _dsm(np.asarray(punkte, float))
    jx = ((P[:, 0] - x0) / res).astype(np.int64)
    jy = ((P[:, 1] - y0) / res).astype(np.int64)
    ok = (jx >= 0) & (jx < dsm.shape[0]) & (jy >= 0) & (jy < dsm.shape[1])
    z = np.full(len(P), -np.inf)
    z[ok] = dsm[jx[ok], jy[ok]]
    ok &= np.isfinite(z)
    cz = float(np.mean(C[:, 2]))
    tiefe_foto = cz - P[ok, 2]
    tiefe_lidar = cz - z[ok]
    gut = (tiefe_foto > 10.0) & (tiefe_lidar > 10.0)
    if gut.sum() < 200:
        return None
    r = tiefe_lidar[gut] / tiefe_foto[gut]
    r = r[(r > 0.7) & (r < 1.5)]
    if len(r) < 200:
        return None
    # Haeufigster Wert: Punkte an Dachkanten landen neben dem Dach auf dem
    # Boden, solange die Lage in der Ebene noch nicht sitzt.
    h, kanten = np.histogram(r, bins=160, range=(0.7, 1.5))
    h = np.convolve(h, [1, 2, 3, 2, 1], mode="same")
    spitze = 0.5 * (kanten[np.argmax(h)] + kanten[np.argmax(h) + 1])
    nah = r[np.abs(r - spitze) < 0.02]
    return {"faktor": float(np.median(nah)), "n": int(len(r)), "n_spitze": int(len(nah))}


def anteil_auf_flaeche(pipe, punkte: np.ndarray, A, b, faktor: float,
                       toleranz: float = 0.5) -> float | None:
    """Anteil der Fotopunkte, die hoechstens ``toleranz`` von der Karte liegen."""
    A, b = np.asarray(A, float), np.asarray(b, float)
    try:
        C = (A @ np.asarray(pipe.cams["C"], float).T).T + b
        P = (A @ np.asarray(pipe.cams["xyz"], float).T).T + b
    except (KeyError, TypeError):
        return None
    P = foto_hoehe(P, C, faktor)
    dsm, x0, y0, res = _dsm(np.asarray(punkte, float))
    jx = ((P[:, 0] - x0) / res).astype(np.int64)
    jy = ((P[:, 1] - y0) / res).astype(np.int64)
    ok = (jx >= 0) & (jx < dsm.shape[0]) & (jy >= 0) & (jy < dsm.shape[1])
    z = np.full(len(P), -np.inf)
    z[ok] = dsm[jx[ok], jy[ok]]
    ok &= np.isfinite(z)
    if ok.sum() < 50:
        return None
    return float((np.abs(P[ok, 2] - z[ok]) <= toleranz).mean())


def schaetze_rgb_faktor(pipe, punkte: np.ndarray, A, b, progress=None,
                        cancel=None) -> dict:
    """Brennweitenfaktor der RGB-Kamera, der die Bilder am besten eint.

    Rueckfall, wenn die Fotopunkte fehlen: das Minimum ist flach (s. oben).
    """
    if len(punkte) > 60_000:
        punkte = punkte[:: len(punkte) // 60_000 + 1]
    fk = Farbkonsistenz(pipe.rgb_cams(), pipe._p("images"), cancel=cancel)

    def wert(f):
        if cancel is not None and cancel():
            raise RuntimeError("Abgebrochen")
        return fk.streuung(punkte, rgb_cams(pipe, f), A, b)

    grob = np.round(np.arange(0.85, 1.2501, 0.01), 4)
    werte = []
    for i, f in enumerate(grob):
        werte.append(wert(f))
        if progress is not None:
            progress((i + 1) / (len(grob) + 12), f"RGB-Brennweite: Faktor {f:.2f} …")
    j = int(np.argmin(werte))
    fein = np.round(np.arange(grob[j] - 0.012, grob[j] + 0.0121, 0.002), 4)
    fw = []
    for i, f in enumerate(fein):
        fw.append(wert(f))
        if progress is not None:
            progress((len(grob) + i + 1) / (len(grob) + len(fein)),
                     f"RGB-Brennweite fein: Faktor {f:.3f} …")
    k = int(np.argmin(fw))
    return {"faktor": float(fein[k]), "streuung": float(fw[k]),
            "streuung_bei_1": float(werte[int(np.argmin(np.abs(grob - 1.0)))]),
            "am_rand": bool(j in (0, len(grob) - 1))}


# ----------------------------------------------------------- Thermal

def _mi(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    h, _, _ = np.histogram2d(a, b, bins=bins, range=[[0, 256], [0, 256]])
    p = h / max(h.sum(), 1.0)
    px, py = p.sum(1), p.sum(0)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (px[:, None] * py[None, :])[nz])).sum())


class _ThermalPaare:
    """RGB- und Thermalbild desselben Ausloesers, als Grauwerte."""

    def __init__(self, names, w_ordner, t_ordner, w_cam, w_breite=800, cancel=None):
        from PIL import Image, ImageFilter  # noqa: PLC0415
        self.w, self.t = [], []
        W0, H0 = (float(v) for v in w_cam["size"])
        sk = w_breite / W0
        for n in names:
            if cancel is not None and cancel():
                raise RuntimeError("Abgebrochen")
            with Image.open(os.path.join(w_ordner, str(n))) as im:
                self.w.append(np.asarray(im.convert("L").resize(
                    (w_breite, int(round(H0 * sk))), Image.BILINEAR), np.float32))
            with Image.open(os.path.join(t_ordner, str(n))) as im:
                g = im.convert("L").filter(ImageFilter.GaussianBlur(1.5))
                self.t.append(np.asarray(g, np.float32))
        self.t_groesse = (self.t[0].shape[1], self.t[0].shape[0])
        # Raster der W-Pixel (jeder zweite der verkleinerten) als Strahlen
        h, w = self.w[0].shape
        gy, gx = np.mgrid[0:h:2, 0:w:2]
        self.gx, self.gy = gx.ravel(), gy.ravel()
        f, cx, cy = (float(v) for v in w_cam["params"][:3])
        k = float(w_cam["params"][3]) if len(w_cam["params"]) > 3 else 0.0
        x = (self.gx / sk - cx) / f
        y = (self.gy / sk - cy) / f
        if k:
            xu, yu = x.copy(), y.copy()
            for _ in range(8):       # SIMPLE_RADIAL umkehren
                d = 1.0 + k * (xu * xu + yu * yu)
                xu, yu = x / d, y / d
            x, y = xu, yu
        self.strahl = np.stack([x, y, np.ones_like(x)])

    def nach_t(self, p):
        f, cx, cy, k1, k2, rx, ry, rz = p
        q = drehung([rx, ry, rz]) @ self.strahl
        x, y = q[0] / q[2], q[1] / q[2]
        r2 = x * x + y * y
        d = 1.0 + k1 * r2 + k2 * r2 * r2
        return f * x * d + cx, f * y * d + cy

    def mi(self, p, welche=None) -> float:
        TW, TH = self.t_groesse
        u, v = self.nach_t(p)
        drin = (u >= 0) & (u < TW - 1.001) & (v >= 0) & (v < TH - 1.001)
        if drin.mean() < 0.05:
            return 0.0
        ud, vd = u[drin], v[drin]
        x0, y0 = ud.astype(np.int64), vd.astype(np.int64)
        ax, ay = (ud - x0).astype(np.float32), (vd - y0).astype(np.float32)
        gx, gy = self.gx[drin], self.gy[drin]
        idx = range(len(self.w)) if welche is None else welche
        ges = 0.0
        for i in idx:
            t = self.t[i]
            tw = ((t[y0, x0] * (1 - ax) + t[y0, x0 + 1] * ax) * (1 - ay)
                  + (t[y0 + 1, x0] * (1 - ax) + t[y0 + 1, x0 + 1] * ax) * ay)
            ges += _mi(self.w[i][gy, gx], tw)
        return ges / len(idx)


def kalibriere_thermal(pipe, rgb_faktor: float = 1.0, n_kalib: int = 12,
                       n_pruef: int = 10, progress=None, cancel=None) -> dict:
    """Thermaloptik gegen das RGB-Bild desselben Ausloesers einmessen.

    Rueckgabe: Brennweite, Hauptpunkt, radiale Verzeichnung (k1, k2), der
    Schielwinkel gegen die RGB-Kamera (Drehvektor in Grad) — und die
    Transinformation vorher/nachher an Paaren, die nicht zum Einmessen dienten.
    """
    from scipy.optimize import minimize  # noqa: PLC0415

    def p_(f, m):
        if progress is not None:
            progress(f, m)
        if cancel is not None and cancel():
            raise RuntimeError("Abgebrochen")

    w_cam = rgb_cams(pipe, rgb_faktor)
    w_cam1 = {"size": w_cam["size"][0], "params": w_cam["params"][0]}
    names = [str(n) for n in pipe.cams["names"]]
    schritt = max(1, len(names) // (n_kalib + n_pruef))
    kalib_namen = names[::schritt][:n_kalib]
    pruef_namen = [n for n in names[schritt // 2::schritt] if n not in kalib_namen][:n_pruef]
    p_(0.02, "Thermal: lade Bildpaare …")
    paare = _ThermalPaare(kalib_namen, pipe._p("images"), pipe._p("thermal"), w_cam1,
                          cancel=cancel)
    TW, TH = paare.t_groesse
    tg = getattr(pipe, "thermal_groesse", None)
    f_start = float(tg[2]) if tg and tg[2] else TW * 40.0 / 36.0
    if tg and tg[0] and float(tg[0]) != TW:
        f_start *= TW / float(tg[0])
    p0 = np.array([f_start, TW / 2.0, TH / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mi_start = paare.mi(p0)

    # Grob: Brennweite und Schielwinkel quer zur Achse im Raster
    best_c, best_p = -mi_start, p0
    raster = [(f, rx, ry) for f in np.linspace(0.88, 1.12, 9) * f_start
              for rx in np.linspace(-3, 3, 7) for ry in np.linspace(-3, 3, 7)]
    for i, (f, rx, ry) in enumerate(raster):
        if i % 40 == 0:
            p_(0.05 + 0.35 * i / len(raster), "Thermal: grobe Suche …")
        p = p0.copy()
        p[0], p[5], p[6] = f, rx, ry
        c = -paare.mi(p)
        if c < best_c:
            best_c, best_p = c, p
    # Fein: alle acht Parameter
    skala = np.array([0.02 * f_start, 20.0, 20.0, 0.1, 0.1, 0.5, 0.5, 0.5])
    zaehler = [0]

    def ziel(z):
        zaehler[0] += 1
        if zaehler[0] % 50 == 0:
            p_(min(0.4 + 0.5 * zaehler[0] / 1500.0, 0.9), "Thermal: Feinabgleich …")
        return -paare.mi(best_p + z * skala)

    res = minimize(ziel, np.zeros(8), method="Powell",
                   options={"maxiter": 3000, "xtol": 1e-3, "ftol": 1e-5})
    p = best_p + res.x * skala
    p_(0.92, "Thermal: Gegenprobe an anderen Bildpaaren …")
    pruef = _ThermalPaare(pruef_namen or kalib_namen, pipe._p("images"),
                          pipe._p("thermal"), w_cam1, cancel=cancel)
    return {"f": float(p[0]), "cx": float(p[1]), "cy": float(p[2]),
            "k1": float(p[3]), "k2": float(p[4]),
            "drehung_grad": [float(p[5]), float(p[6]), float(p[7])],
            "breite": int(TW), "hoehe": int(TH), "rgb_faktor": float(rgb_faktor),
            "f_exif": float(f_start),
            "mi_vorher": float(pruef.mi(p0)), "mi_nachher": float(pruef.mi(p)),
            "mi_kalib_vorher": float(mi_start), "mi_kalib_nachher": float(-res.fun),
            "paare_kalib": len(kalib_namen), "paare_pruef": len(pruef_namen)}


# ---------------------------------------------------------------- alles

def einmessen(pipe, punkte: np.ndarray, yaw_deg: float, t, foto_ordner: str,
              thermal: bool = True, progress=None, cancel=None, log=None) -> dict:
    """Hoehe, RGB-Brennweite und Thermaloptik nacheinander einmessen.

    ``yaw_deg``/``t`` ist die aktuelle RGB-Lage. Rueckgabe enthaelt die
    Hoehenkorrektur ``dz`` (auf die Lage anzuwenden), den RGB-Faktor und die
    Thermalkalibrierung samt Kennzahlen.
    """
    from core import meander as meander_mod  # noqa: PLC0415

    def p_(a, b_, f, m):
        if progress is not None:
            progress(a + (b_ - a) * f, m)

    def l_(m):
        if log is not None:
            log(m)

    t = meander_mod.as_t3(t)
    A, b = meander_mod.lage_affine(pipe, yaw_deg, t)
    p_(0, 0.05, 0.0, "Höhe über den Laser-Entfernungsmesser …")
    h = hoehe_aus_lrf(pipe, punkte, A, b, foto_ordner)
    dz = 0.0
    if h is None:
        l_("Kein Laserabstand in den Bildern — die Höhe bleibt, wie sie ist.")
    else:
        dz = h["dz"]
        l_(f"Höhe über den Laser: Kameras {dz:+.2f} m — {h['n_spitze']} von "
           f"{h['n']} Bildern stimmen darin auf einen Meter überein.")
        t = t + np.array([0.0, 0.0, dz])
        A, b = meander_mod.lage_affine(pipe, yaw_deg, t)
    p_(0.05, 0.1, 0.0, "RGB-Brennweite aus der Tiefe der Fotopunkte …")
    tiefe = schaetze_rgb_faktor_tiefe(pipe, punkte, A, b)
    if tiefe is not None:
        # Gegenprobe ueber die Farbkonsistenz: zwei Durchlaeufe genuegen
        p_(0.1, 0.45, 0.2, "Gegenprobe über die Farbkonsistenz …")
        sub = punkte[:: max(1, len(punkte) // 60_000)]
        fk = Farbkonsistenz(pipe.rgb_cams(), pipe._p("images"), cancel=cancel)
        s1 = fk.streuung(sub, rgb_cams(pipe, 1.0), A, b)
        sf = fk.streuung(sub, rgb_cams(pipe, tiefe["faktor"]), A, b)
        rgb = {"faktor": tiefe["faktor"], "verfahren": "tiefe", "streuung": sf,
               "streuung_bei_1": s1, "n": tiefe["n"], "n_spitze": tiefe["n_spitze"]}
        l_(f"RGB-Brennweite aus der Tiefe von {tiefe['n_spitze']} Fotopunkten: "
           f"Faktor {rgb['faktor']:.3f}. Gegenprobe: die Bilder streuen "
           f"{sf:.1f} statt {s1:.1f} Grauwerte.")
        if sf > s1:
            l_("WARNUNG: Mit diesem Maßstab widersprechen sich die Bilder mehr als "
               "ohne — Lage in der Ebene prüfen und neu einmessen.")
    else:
        rgb = schaetze_rgb_faktor(pipe, punkte, A, b, cancel=cancel,
                                  progress=lambda f, m: p_(0.05, 0.45, f, m))
        rgb["verfahren"] = "farbkonsistenz"
        l_(f"RGB-Brennweite über die Farbkonsistenz (zu wenige Fotopunkte über "
           f"der Karte): Faktor {rgb['faktor']:.3f} — die Bilder streuen "
           f"{rgb['streuung']:.1f} statt {rgb['streuung_bei_1']:.1f} Grauwerte. "
           f"Das Minimum ist flach, im Ausrichtfenster in der Seitenansicht prüfen."
           + (" Es liegt am Rand des Suchbereichs." if rgb["am_rand"] else ""))
    A0, b0 = meander_mod.lage_affine(pipe, yaw_deg, t - np.array([0.0, 0.0, dz]))
    vorher = anteil_auf_flaeche(pipe, punkte, A0, b0, 1.0)
    nachher = anteil_auf_flaeche(pipe, punkte, A, b, rgb["faktor"])
    if vorher is not None and nachher is not None:
        rgb["auf_flaeche_vorher"], rgb["auf_flaeche_nachher"] = vorher, nachher
        l_(f"Fotopunkte auf der Lidar-Oberfläche (±0,5 m): vorher {vorher * 100:.1f} %, "
           f"jetzt {nachher * 100:.1f} %.")
    th = None
    if thermal and getattr(pipe, "thermal_groesse", None) and \
            os.path.isdir(pipe._p("thermal")):
        th = kalibriere_thermal(pipe, rgb["faktor"], cancel=cancel,
                                progress=lambda f, m: p_(0.45, 1.0, f, m))
        gewinn = th["mi_nachher"] / max(th["mi_vorher"], 1e-9) - 1.0
        l_(f"Thermal eingemessen: Brennweite {th['f']:.0f} px (EXIF "
           f"{th['f_exif']:.0f}), Hauptpunkt {th['cx']:.0f}/{th['cy']:.0f}, "
           f"Verzeichnung k1 {th['k1']:+.3f} k2 {th['k2']:+.3f}, Schielwinkel "
           f"{th['drehung_grad'][0]:+.2f}°/{th['drehung_grad'][1]:+.2f}°/"
           f"{th['drehung_grad'][2]:+.2f}°. Übereinstimmung mit RGB an "
           f"{th['paare_pruef']} anderen Bildpaaren {gewinn * 100:+.0f} %.")
        if gewinn < 0.03:
            l_("WARNUNG: Die Thermal-Kalibrierung verbessert kaum etwas — sie "
               "wird trotzdem gespeichert, im Ausrichtfenster prüfen.")
    return {"dz": dz, "hoehe": h, "rgb": rgb, "thermal": th}


if __name__ == "__main__":
    import tempfile

    print("== Test 1: LRF aus dem XMP ==")
    tmp = tempfile.mkdtemp(prefix="optiktest_")
    pfad = os.path.join(tmp, "a.jpg")
    with open(pfad, "wb") as fh:
        fh.write(b"\xff\xd8junk<x drone-dji:LRFStatus=\"Normal\" "
                 b"drone-dji:LRFTargetDistance=\"57.116\"/>")
    assert lies_lrf(pfad) == 57.116, lies_lrf(pfad)
    with open(pfad, "wb") as fh:
        fh.write(b"drone-dji:LRFStatus=\"TooFar\" drone-dji:LRFTargetDistance=\"0\"")
    assert lies_lrf(pfad) is None
    assert lies_lrf(os.path.join(tmp, "fehlt.jpg")) is None
    print("  57.116 gelesen, ungueltig und fehlend ergeben None")

    print("== Test 2: Thermal-Umrechnung ist konsistent ==")
    # Eine Pixelzuordnung W->T haengt nicht davon ab, welche RGB-Brennweite
    # man annimmt, wenn die Kalibrierung mit umgerechnet wird.
    kal = {"f": 1340.0, "cx": 624.0, "cy": 509.0, "k1": -0.01, "k2": -0.39,
           "drehung_grad": [2.0, -0.7, 0.6], "breite": 1280, "hoehe": 1024,
           "rgb_faktor": 1.0}

    def zuordnung(k, fw):
        uw = np.array([[100.0, 800.0, 1500.0], [100.0, 600.0, 1100.0]])
        x = np.stack([(uw[0] - 800) / fw, (uw[1] - 600) / fw, np.ones(3)])
        q = drehung(k["drehung_grad"]) @ x
        xx, yy = q[0] / q[2], q[1] / q[2]
        r2 = xx * xx + yy * yy
        d = 1 + k["k1"] * r2 + k["k2"] * r2 * r2
        return k["f"] * xx * d + k["cx"], k["f"] * yy * d + k["cy"]
    u1, v1 = zuordnung(kal, 1066.67)
    u2, v2 = zuordnung(thermal_umrechnen(kal, 1.1), 1066.67 * 1.1)
    abw = float(np.max(np.abs(np.r_[u1 - u2, v1 - v2])))
    print(f"  groesste Abweichung der Zuordnung nach Umrechnung: {abw:.2f} px")
    assert abw < 1.5, abw

    print("== Test 3: Fotopunkte-Hoehe ==")
    C = np.array([[0.0, 0.0, 57.0], [10.0, 0.0, 57.0]])
    P = np.array([[1.0, 2.0, 3.9], [5.0, 5.0, 0.0]])
    Q = foto_hoehe(P, C, 57.0 / 53.1)
    assert abs(Q[0, 2]) < 1e-9 and np.allclose(Q[:, :2], P[:, :2]), Q
    print("  3,9 m zu hoch bei 7 % zu kurzer Brennweite -> auf null")

    print("== Test 4: Datei ==")
    assert laden(tmp)["rgb_faktor"] == 1.0 and not vorhanden(tmp)
    speichern(tmp, {"rgb_faktor": 1.11, "thermal_faktor": 0.99, "thermal": kal})
    d = laden(tmp)
    assert d["rgb_faktor"] == 1.11 and d["thermal"]["k2"] == -0.39 and vorhanden(tmp)
    print("  hin und zurueck")

    print("== Test 5: Transinformation erkennt die richtige Zuordnung ==")
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, 5000).astype(float)
    assert _mi(a, 255 - a) > _mi(a, rng.permutation(a)) + 1.0
    print("  invertierte Abbildung (Thermalpalette) klar ueber Zufall")
    print("optik SELFTEST OK")

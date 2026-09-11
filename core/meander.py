"""Punktwolke aus den Bildern eines DJI-Maeanderfluges einfaerben.

Huelle um ``colorize_pipeline`` aus dem Repo PointCloudMerger. Der Weg dort:
COLMAP rekonstruiert die Kameras in einem willkuerlichen Rahmen, die RTK-Geotags
machen daraus per Umeyama einen metrischen ENU-Rahmen, und zwischen ENU und dem
Lidar-Rahmen bleibt nur noch eine Drehung um die Hochachse plus Verschiebung —
beide Rahmen sind lotrecht, das ENU per Definition und die Karte seit der
Kippkorrektur (s. core/recording.py) ueber die IMU. Vier Freiheitsgrade statt
sieben, und die lassen sich suchen.

Diese Datei aendert an dem Verfahren nichts. Sie tut drei Dinge:

* die Arbeitswolke von Super360 Studio hineinreichen, ohne sie erst als Datei
  zu schreiben (:func:`build_pipeline` legt sie als ``cloud.npy`` in den
  Arbeitsordner, wo die Pipeline ihren eigenen Zwischenstand erwartet);
* die Ausrichtung anstossen und ihre Kennwerte zurueckgeben;
* die **volle** Wolke einfaerben statt der ausgeduennten, die die Pipeline fuer
  ihren PLY-Export nimmt, und dabei eine Gueltigkeitsmaske mitfuehren
  (:func:`colorize_points`) — Super360 Studio braucht je Punkt eine Farbe und
  die Auskunft, ob sie echt ist.

Qt-frei. ``colorize_pipeline`` wird erst beim Gebrauch importiert, damit die
Anwendung ohne das Nachbarrepo startet.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

#: Orte, an denen das Nachbarrepo liegen kann. Der erste Treffer gewinnt.
PIPELINE_CANDIDATES = (
    os.path.expanduser("~/PointCloudMerger"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "..", "PointCloudMerger"),
)

_ALIGN_TARGET_PTS = 400_000   # so viel Wolke sieht die Ausrichtung
_FALLBACK = (107, 107, 107)   # Grau fuer nicht getroffene Punkte

#: Unter diesem Anteil sitzt die Ausrichtung falsch. Bei einer Nadir-
#: befliegung liegen die Fotopunkte auf genau der Oberflaeche, die das
#: Lidar von oben sieht — stimmt der Winkel, sind es 70 % und mehr
#: innerhalb eines halben Meters. Ein gemessener Fehlgriff lag bei 2,5 %.
MIN_AUF_FLAECHE = 0.40


def find_pipeline():
    """``colorize_pipeline`` importieren; klare Meldung, wenn es fehlt."""
    for root in PIPELINE_CANDIDATES:
        root = os.path.abspath(root)
        if os.path.isdir(os.path.join(root, "colorize_pipeline")):
            if root not in sys.path:
                sys.path.insert(0, root)
            from colorize_pipeline import pipeline as pl  # noqa: PLC0415
            return pl
    raise RuntimeError(
        "Die Mäander-Pipeline wurde nicht gefunden. Erwartet wird das Repo "
        "PointCloudMerger mit dem Paket 'colorize_pipeline' unter einem von: "
        + ", ".join(PIPELINE_CANDIDATES))


def find_colmap_python() -> str | None:
    """Interpreter mit pycolmap, oder None. Ohne ihn geht nur ein fertiges Modell."""
    try:
        find_pipeline()
        from colorize_pipeline import sfm  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    return sfm.find_python()


def thin(points: np.ndarray, max_points: int = _ALIGN_TARGET_PTS) -> np.ndarray:
    """Gleichmaessig ausduennen — die Ausrichtung braucht keine volle Dichte."""
    n = len(points)
    if n <= max_points:
        return np.asarray(points, dtype=np.float64)
    return np.asarray(points[:: n // max_points + 1], dtype=np.float64)


def build_pipeline(points: np.ndarray, photo_dir: str, work_dir: str,
                   thermal: bool = False, rgb_versatz=None, thermal_versatz=None,
                   log=None, cancel=None):
    """Pipeline mit der Arbeitswolke als Eingang aufsetzen.

    Die Wolke wird als ``cloud.npy`` in den Arbeitsordner gelegt. Genau von
    dort liest ``Pipeline.load_cloud`` sie beim naechsten Lauf wieder, ohne
    eine Datei zu lesen — der Umweg ueber PCD oder PLY entfaellt damit.
    """
    pl = find_pipeline()
    os.makedirs(work_dir, exist_ok=True)
    cache = os.path.join(work_dir, "cloud.npy")
    duenn = thin(points)
    if not os.path.exists(cache) or len(np.load(cache, mmap_mode="r")) != len(duenn):
        np.save(cache, duenn)
    return pl.Pipeline(
        cloud_path=cache, photo_dir=photo_dir, work_dir=work_dir,
        out_path=os.path.join(work_dir, "maeander.ply"),
        thermal=bool(thermal), rgb_versatz=rgb_versatz,
        thermal_versatz=thermal_versatz,
        python_exe=find_colmap_python(),
        log=log or (lambda s: None),
        cancel=cancel or (lambda: False))


def prepare(pipe, progress=None) -> dict:
    """Fotos, Rekonstruktion, Georeferenzierung — alles vor der Ausrichtung.

    Ein vorhandenes COLMAP-Modell im Arbeitsordner wird wiederverwendet; sonst
    laeuft die Rekonstruktion, und die dauert bei 255 Bildern eine halbe Stunde.
    """
    def p(f, m):
        if progress is not None:
            progress(f, m)

    p(0.02, "Lade Punktwolke …")
    pipe.load_cloud()
    p(0.10, "Bereite Fotos auf …")
    pipe.prepare_photos()
    p(0.30, "COLMAP-Modell …")
    pipe.reconstruct()
    p(0.80, "Georeferenzierung über die Geotags …")
    pipe.georeference()
    p(0.90, "bereit zum Ausrichten")
    return {"kameras": int(len(pipe.cams["names"])),
            "massstab": float(pipe.enu[0]),
            "gps_residuum": float(pipe.enu[5])}


def align(pipe, progress=None) -> dict:
    """Gierwinkel und Verschiebung suchen (grob per FFT, fein am Rastermodell).

    ``anteil_auf_flaeche`` in der Rueckgabe ist das Guetemass, nicht die
    spaetere Trefferquote beim Einfaerben: die liegt auch bei einer voellig
    verdrehten Lage nahe 100 %, weil fast jeder Punkt in IRGENDEIN Bild
    faellt. Wer den Erfolg daran misst, merkt den Fehlgriff nie.
    """
    if progress is not None:
        progress(0.05, "Suche Gierwinkel und Verschiebung …")
    aus_cache = os.path.exists(pipe._p("align.json"))
    pipe.align()
    # Eine align.json aus der Zeit vor as_t3 traegt nur x und y. Die Pipeline
    # liest sie unveraendert ein, und jedes spaetere affine() bricht dann ab.
    pipe.t = as_t3(pipe.t)
    k = dict(pipe.kennwerte or {})
    k.update({"yaw_deg": float(np.degrees(pipe.yaw)),
              "t": [float(x) for x in np.asarray(pipe.t).ravel()],
              # Frisch gesucht heisst: Hoehenkorrektur und Feinausrichtung
              # gehoerten zu einer anderen Lage und gelten nicht mehr.
              "aus_cache": bool(aus_cache)})
    if progress is not None:
        progress(1.0, f"Ausgerichtet: {k['yaw_deg']:.2f}°")
    return k


def pruefe_ausrichtung(kennwerte: dict) -> str | None:
    """Meldung, wenn die Ausrichtung nicht zu trauen ist; sonst None."""
    anteil = kennwerte.get("anteil_auf_flaeche")
    if anteil is None or anteil >= MIN_AUF_FLAECHE:
        return None
    med = kennwerte.get("median_abweichung")
    return (f"Die Ausrichtung sitzt nicht: nur {anteil * 100:.1f} % der "
            f"Fotopunkte liegen auf der Oberfläche der Wolke"
            + (f" (Median {med:.2f} m)" if med is not None else "")
            + f", erwartet sind über {MIN_AUF_FLAECHE * 100:.0f} %. Der "
            f"gefundene Gierwinkel {kennwerte.get('yaw_deg', 0.0):.2f}° ist "
            f"vermutlich falsch — die Bewertung der Kandidaten liegt eng "
            f"beieinander. Gier von Hand nachziehen und erneut ausrichten.")


def as_t3(t) -> np.ndarray:
    """Verschiebung als 3er-Vektor, wie ``register.affine`` sie braucht.

    ``fit_to_dsm`` liefert drei Komponenten (x, y **und z**). Wer daraus einen
    2er macht, bekommt in ``affine`` einen Broadcast-Fehler — und wenn der in
    einem Qt-Slot passiert, verschluckt Qt ihn und es sieht so aus, als taete
    der Regler einfach nichts.
    """
    v = np.asarray(t, dtype=float).ravel()
    if v.size >= 3:
        return v[:3].copy()
    out = np.zeros(3)
    out[:v.size] = v
    return out


def set_manual(pipe, yaw_deg: float, t) -> None:
    """Handjustage uebernehmen (Gier in Grad, Versatz als 3er in Metern)."""
    pipe.yaw = float(np.radians(yaw_deg))
    pipe.t = as_t3(t)
    pipe.save_align()


def lage_affine(pipe, yaw_deg: float, t) -> tuple:
    """Affin fuer eine beliebige Lage, ohne die Lage der Pipeline anzufassen.

    ``affine()`` liest Gier und Verschiebung aus der Pipeline. Fuer eine
    Vorschau oder die Thermallage werden sie kurz gesetzt und danach
    zurueckgelegt — die Basis bleibt, was ``align`` gefunden hat.
    """
    alt_yaw, alt_t = pipe.yaw, pipe.t
    try:
        pipe.yaw = float(np.radians(yaw_deg))
        pipe.t = as_t3(t)
        A, b = pipe.affine()
    finally:
        pipe.yaw, pipe.t = alt_yaw, alt_t
    return korrektur_anwenden(A, b, getattr(pipe, "s360_korrektur", None))


def korrektur_anwenden(A, b, korrektur) -> tuple:
    """Feinausrichtung auf ein Affin legen: p -> M p + v nach der Lage.

    Die Pipeline kennt nur Gier und Verschiebung. Die Feinausrichtung
    (``core.optik.feinausrichten``) findet dazu eine kleine Neigung und einen
    Versatz; sie haengt an der Pipeline, damit jede Stelle, die eine Lage in
    ein Affin uebersetzt, sie mitnimmt — Vorschau, Fenster, Einfaerben.
    """
    if not korrektur:
        return A, b
    M = np.asarray(korrektur["M"], dtype=float)
    v = np.asarray(korrektur["v"], dtype=float)
    return M @ np.asarray(A, float), M @ np.asarray(b, float) + v


# ------------------------------------------------------------ Thermallage
#
# Die Thermalbilder haben eine eigene Handlage, aber als Zuschlag auf die
# RGB-Lage, nicht als zweite Lage daneben: beide Optiken haengen an derselben
# Gimbal. Wird RGB neu ausgerichtet oder nachgezogen, zieht Thermal mit, und im
# Zuschlag steht nur, was zwischen den beiden Optiken nicht passt.

_THERMAL_LAGE = "thermal_lage.json"


def thermal_lage(rgb_yaw_deg: float, rgb_t, zuschlag) -> tuple:
    """Lage der Thermalbilder: RGB-Lage plus Zuschlag (Gier Grad, X m, Y m)."""
    dgier, dx, dy = (float(v) for v in zuschlag)
    t = as_t3(rgb_t)
    t[0] += dx
    t[1] += dy
    return float(rgb_yaw_deg) + dgier, t


def load_thermal_zuschlag(work_dir: str) -> tuple:
    """Gespeicherten Thermal-Zuschlag lesen; (0, 0, 0), wenn keiner da ist."""
    pfad = os.path.join(work_dir, _THERMAL_LAGE)
    try:
        with open(pfad, encoding="utf-8") as fh:
            d = json.load(fh)
        return (float(d["gier_grad"]), float(d["x"]), float(d["y"]))
    except (OSError, ValueError, KeyError, TypeError):
        return (0.0, 0.0, 0.0)


def save_thermal_zuschlag(work_dir: str, zuschlag) -> None:
    dgier, dx, dy = (float(v) for v in zuschlag)
    os.makedirs(work_dir, exist_ok=True)
    tmp = os.path.join(work_dir, _THERMAL_LAGE + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"gier_grad": dgier, "x": dx, "y": dy,
                   "bezug": "Zuschlag auf die RGB-Lage, X und Y in Metern"},
                  fh, indent=2)
    os.replace(tmp, os.path.join(work_dir, _THERMAL_LAGE))


def colorize_points(points: np.ndarray, cams, image_dir: str, A, b,
                    progress=None, cancel=None, temperatur=None) -> tuple:
    """Volle Wolke einfaerben; gibt (rgb uint8 (N,3), maske bool (N,)).

    Mit ``temperatur`` (s. ``core.temperatur.quelle``) kommt als drittes
    Element die Temperatur je Punkt dazu, aus demselben Bild wie die Farbe.

    Wie ``colorize_pipeline.colorize.colorize``, aber ueber alle Punkte der
    Arbeitswolke statt der ausgeduennten, und mit Maske statt nur einer Quote.
    Je Punkt gewinnt die Kamera, in deren Bild er am dichtesten am Nadir liegt
    — bei einem Maeanderflug ist das die mit dem steilsten Blick auf ihn.

    Die Kameras werden nacheinander abgearbeitet und ihr Bild dabei je einmal
    geladen; die Punkte bleiben durchgehend im Speicher, damit kein Bild
    mehrfach von der Platte kommt.
    """
    find_pipeline()
    from colorize_pipeline import colorize as cz  # noqa: PLC0415

    names = cams["names"]
    Rcw = np.asarray(cams["Rcw"], dtype=np.float64)
    tcw = np.asarray(cams["tcw"], dtype=np.float64)
    size = np.asarray(cams["size"])
    params = np.asarray(cams["params"])
    model = cams["model"].item() if getattr(cams["model"], "shape", None) == () \
        else str(cams["model"])
    from PIL import Image  # noqa: PLC0415

    P = np.asarray(points, dtype=np.float64)
    N = len(P)
    Ainv = np.linalg.inv(np.asarray(A, dtype=np.float64))
    P_col = (Ainv @ (P - np.asarray(b, dtype=np.float64)).T).T
    PT = np.ascontiguousarray(P_col.T)
    del P_col

    best = np.full(N, np.inf, dtype=np.float32)
    col = np.empty((N, 3), dtype=np.uint8)
    col[:] = _FALLBACK
    temp = np.full(N, np.nan, dtype=np.float32) if temperatur is not None else None
    n_cams = len(names)
    for i, n in enumerate(names):
        if cancel is not None and cancel():
            raise RuntimeError("Abgebrochen")
        pc = (Rcw[i] @ PT).T + tcw[i]
        u, v, front, rad = cz._project(pc, size[i], params[i], model)
        W, H = size[i]
        gilt = front & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (rad < best)
        if gilt.any():
            img = np.asarray(Image.open(os.path.join(image_dir, str(n))).convert("RGB"))
            ih, iw = img.shape[:2]
            ui = np.clip(u[gilt].astype(np.int32), 0, iw - 1)
            vi = np.clip(v[gilt].astype(np.int32), 0, ih - 1)
            col[gilt] = img[vi, ui]
            best[gilt] = rad[gilt].astype(np.float32)
            if temp is not None:
                from core.temperatur import abtasten  # noqa: PLC0415
                t_bild = temperatur(i, n)
                temp[gilt] = np.nan if t_bild is None else \
                    abtasten(t_bild, u[gilt], v[gilt], W, H)
        if progress is not None and (i % 10 == 0 or i == n_cams - 1):
            progress((i + 1) / n_cams,
                     f"Färbe aus Bild {i + 1}/{n_cams} — "
                     f"{np.isfinite(best).mean() * 100:.1f} % getroffen")
    maske = np.isfinite(best)
    if temp is not None:
        return col, maske, temp
    return col, maske


class LivePreview:
    """Einfaerbung fuer die Handjustage: klein, aber sofort.

    Der volle Lauf laedt fuer jede der 255 Kameras ihr Bild von der Platte —
    Sekunden, und bei jedem Reglerzug von vorn. Fuer eine Vorschau reicht viel
    weniger: die Bilder einmal stark verkleinert in den Speicher (255 Stueck bei
    Faktor 1/6 sind rund 20 MB) und eine Stichprobe der Wolke statt aller
    Punkte. Danach ist ein Durchlauf reine Rechnung und dauert Millisekunden,
    die Wolke folgt dem Regler also ohne Verzoegerung.

    Die Projektion bleibt dieselbe wie beim vollen Lauf; nur die Bildkoordinaten
    werden am Ende mit dem Verkleinerungsfaktor multipliziert. Intrinsik und
    Verzeichnung gelten weiter fuer das Originalbild, damit die Vorschau nicht
    an einer anderen Stelle sitzt als das Ergebnis.
    """

    def __init__(self, cams, image_dir: str, scale: float = 1.0 / 6.0,
                 progress=None, cancel=None):
        find_pipeline()
        from colorize_pipeline import colorize as cz  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        self._cz = cz
        self.scale = float(scale)
        self.names = list(cams["names"])
        self.Rcw = np.asarray(cams["Rcw"], dtype=np.float64)
        self.tcw = np.asarray(cams["tcw"], dtype=np.float64)
        self.size = np.asarray(cams["size"])
        self.params = np.asarray(cams["params"])
        self.model = (cams["model"].item()
                      if getattr(cams["model"], "shape", None) == ()
                      else str(cams["model"]))
        self.bilder: list = []
        n = len(self.names)
        for i, name in enumerate(self.names):
            if cancel is not None and cancel():
                raise RuntimeError("Abgebrochen")
            with Image.open(os.path.join(image_dir, str(name))) as im:
                im = im.convert("RGB")
                klein = im.resize((max(int(im.width * self.scale), 1),
                                   max(int(im.height * self.scale), 1)),
                                  Image.BILINEAR)
                self.bilder.append(np.asarray(klein))
            if progress is not None and (i % 20 == 0 or i == n - 1):
                progress((i + 1) / n, f"Lade Vorschaubild {i + 1}/{n} …")

    def colorize(self, points: np.ndarray, A, b, cams=None) -> tuple[np.ndarray, np.ndarray]:
        """Wie :func:`colorize_points`, nur auf den verkleinerten Bildern.

        ``cams`` ersetzt Posen und Optik fuer diesen einen Durchlauf (gleiche
        Bilder, gleiche Reihenfolge) — so wirken Massstab und Thermaloptik
        sofort, ohne dass ein Bild neu geladen wird.
        """
        Rcw, tcw, size, params, model = self.Rcw, self.tcw, self.size, self.params, self.model
        if cams is not None:
            Rcw = np.asarray(cams["Rcw"], dtype=np.float64)
            tcw = np.asarray(cams["tcw"], dtype=np.float64)
            size = np.asarray(cams["size"])
            params = np.asarray(cams["params"])
            model = (cams["model"].item() if getattr(cams["model"], "shape", None) == ()
                     else str(cams["model"]))
        P = np.asarray(points, dtype=np.float64)
        N = len(P)
        Ainv = np.linalg.inv(np.asarray(A, dtype=np.float64))
        PT = np.ascontiguousarray(((Ainv @ (P - np.asarray(b, float)).T).T).T)
        best = np.full(N, np.inf, dtype=np.float32)
        col = np.empty((N, 3), dtype=np.uint8)
        col[:] = _FALLBACK
        for i, img in enumerate(self.bilder):
            pc = (Rcw[i] @ PT).T + tcw[i]
            u, v, front, rad = self._cz._project(pc, size[i], params[i], model)
            W, H = size[i]
            gilt = front & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (rad < best)
            if not gilt.any():
                continue
            ih, iw = img.shape[:2]
            ui = np.clip((u[gilt] * self.scale).astype(np.int32), 0, iw - 1)
            vi = np.clip((v[gilt] * self.scale).astype(np.int32), 0, ih - 1)
            col[gilt] = img[vi, ui]
            best[gilt] = rad[gilt].astype(np.float32)
        return col, np.isfinite(best)


# ------------------------------------------------------------------ Ebenen

def save_layer(out_dir: str, rgb: np.ndarray, maske: np.ndarray, meta: dict,
               temperatur: np.ndarray | None = None) -> None:
    """Farbebene ablegen — dasselbe Format wie die Einfaerbung aus der 360-Kamera.

    ``temperatur`` (float32 je Punkt, NaN wo keine) landet als
    ``temperatur.bin`` daneben; ohne wird eine alte entfernt, damit nie eine
    Temperatur zu einer anderen Einfaerbung passt als ihrer eigenen.
    """
    os.makedirs(out_dir, exist_ok=True)
    tbin = os.path.join(out_dir, "temperatur.bin")
    if temperatur is not None:
        np.ascontiguousarray(temperatur, dtype=np.float32).tofile(tbin)
    elif os.path.exists(tbin):
        os.remove(tbin)
    np.ascontiguousarray(rgb, dtype=np.uint8).tofile(os.path.join(out_dir, "colors.bin"))
    np.ascontiguousarray(maske.astype(np.uint8)).tofile(
        os.path.join(out_dir, "valid.bin"))
    tmp = os.path.join(out_dir, "meta.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    os.replace(tmp, os.path.join(out_dir, "meta.json"))


def load_temperatur(out_dir: str, n_points: int) -> np.ndarray | None:
    """Temperatur je Punkt (float32, NaN wo keine) oder None."""
    tbin = os.path.join(out_dir, "temperatur.bin")
    if not os.path.isfile(tbin):
        return None
    t = np.fromfile(tbin, dtype=np.float32)
    return t if t.size == n_points else None


def load_layer(out_dir: str, n_points: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Farbebene lesen, oder None wenn sie fehlt bzw. nicht zur Wolke passt."""
    cbin = os.path.join(out_dir, "colors.bin")
    vbin = os.path.join(out_dir, "valid.bin")
    if not (os.path.isfile(cbin) and os.path.isfile(vbin)):
        return None
    rgb = np.fromfile(cbin, dtype=np.uint8)
    val = np.fromfile(vbin, dtype=np.uint8)
    if rgb.size != n_points * 3 or val.size != n_points:
        return None
    return rgb.reshape(-1, 3), val.astype(bool)


if __name__ == "__main__":
    # Selbsttest ohne COLMAP: Projektion und Ebenen-Format pruefen.
    import tempfile
    import shutil

    print("== Test 1: Pipeline gefunden ==")
    pl = find_pipeline()
    print(f"  colorize_pipeline aus {os.path.dirname(pl.__file__)}")
    print(f"  Interpreter mit pycolmap: {find_colmap_python()}")

    print("== Test 2: Ausduennen ==")
    P = np.random.default_rng(0).uniform(-10, 10, size=(1_000_000, 3))
    d = thin(P, 100_000)
    assert 90_000 <= len(d) <= 100_000, len(d)
    assert thin(P[:50], 100_000).shape == (50, 3)
    print(f"  1 Mio. -> {len(d)} Punkte, kleine Wolken bleiben unveraendert")

    print("== Test 3: Einfaerben mit einer synthetischen Kamera ==")
    # Kamera 20 m ueber dem Ursprung, schaut nach unten (+z der Kamera = Blick)
    R = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    C = np.array([0.0, 0.0, 20.0])
    cams = {"names": np.array(["a.png"]), "Rcw": R[None], "tcw": (-R @ C)[None],
            "size": np.array([[64.0, 64.0]]),
            "params": np.array([[64.0, 32.0, 32.0, 0.0]]),
            "model": np.array("SIMPLE_RADIAL")}
    tmp = tempfile.mkdtemp(prefix="meandertest_")
    from PIL import Image
    bild = np.zeros((64, 64, 3), np.uint8)
    bild[:, :32] = (200, 30, 30)      # linke Bildhaelfte rot
    bild[:, 32:] = (30, 30, 200)      # rechte blau
    Image.fromarray(bild).save(os.path.join(tmp, "a.png"))
    # Punkte am Boden, links und rechts der Bildmitte
    pts = np.array([[-5.0, 0, 0], [5.0, 0, 0], [0, 0, 0], [500.0, 0, 0]])
    rgb, maske = colorize_points(pts, cams, tmp, np.eye(3), np.zeros(3))
    print(f"  Farben: {rgb.tolist()}  Maske: {maske.tolist()}")
    assert maske[:3].all() and not maske[3], maske
    assert rgb[0][0] > rgb[0][2], "linker Punkt ist nicht rot"
    assert rgb[1][2] > rgb[1][0], "rechter Punkt ist nicht blau"
    assert tuple(rgb[3]) == _FALLBACK, "Punkt ausserhalb wurde eingefaerbt"

    print("== Test 4: Guetepruefung der Ausrichtung ==")
    assert pruefe_ausrichtung({"anteil_auf_flaeche": 0.73}) is None
    assert pruefe_ausrichtung({}) is None            # ohne Kennwert kein Urteil
    schlecht = pruefe_ausrichtung({"anteil_auf_flaeche": 0.025,
                                   "median_abweichung": 5.84,
                                   "yaw_deg": 154.53})
    assert schlecht and "2.5 %" in schlecht and "154.53" in schlecht, schlecht
    print(f"  {schlecht[:78]}…")

    print("== Test 5: LivePreview liefert dasselbe wie der volle Lauf ==")
    lp = LivePreview(cams, tmp, scale=1.0, progress=None)   # Faktor 1 = kein Verlust
    rgb_l, maske_l = lp.colorize(pts, np.eye(3), np.zeros(3))
    assert np.array_equal(maske_l, maske), (maske_l, maske)
    assert np.array_equal(rgb_l, rgb), (rgb_l, rgb)
    lp2 = LivePreview(cams, tmp, scale=0.5)
    rgb2, maske2 = lp2.colorize(pts, np.eye(3), np.zeros(3))
    assert np.array_equal(maske2, maske)
    print(f"  Faktor 1 bitgleich, Faktor 0.5 gleiche Maske, "
          f"Bildgroesse {lp2.bilder[0].shape[:2]}")
    # eine verschobene Lage muss andere Farben ergeben
    b_weg = np.array([100.0, 0.0, 0.0])
    _, maske_weg = lp.colorize(pts, np.eye(3), b_weg)
    assert not maske_weg.any(), "verschobene Lage trifft immer noch"
    print("  verschobene Lage trifft nichts mehr — die Vorschau reagiert")

    print("== Test 6: Verschiebung bleibt ein 3er-Vektor ==")
    assert as_t3([1.0, 2.0, 3.0]).tolist() == [1.0, 2.0, 3.0]
    assert as_t3([1.0, 2.0]).tolist() == [1.0, 2.0, 0.0], "2er nicht aufgefuellt"
    assert as_t3(np.array([[4.0, 5.0, 6.0, 7.0]])).tolist() == [4.0, 5.0, 6.0]
    assert as_t3([]).tolist() == [0.0, 0.0, 0.0]
    # und genau so muss register.affine sie nehmen — ein 2er bricht dort ab
    find_pipeline()
    from colorize_pipeline import register as reg
    A_, b_ = reg.affine(0.5, as_t3([1.0, 2.0]), 1.0, np.eye(3), np.zeros(3))
    assert b_.shape == (3,), b_.shape
    try:
        reg.affine(0.5, np.array([1.0, 2.0]), 1.0, np.eye(3), np.zeros(3))
    except ValueError:
        print("  2er-Vektor bricht in register.affine ab — genau deshalb as_t3")
    else:
        raise AssertionError("2er-Vektor haette abbrechen muessen")

    print("== Test 6b: Thermallage ist ein Zuschlag auf RGB ==")
    yaw_t, t_t = thermal_lage(80.0, [-18.0, 9.0, -2.0], (1.5, 0.5, -0.25))
    assert abs(yaw_t - 81.5) < 1e-12 and np.allclose(t_t, [-17.5, 8.75, -2.0]), t_t
    yaw_t, t_t = thermal_lage(80.0, [-18.0, 9.0], (0.0, 0.0, 0.0))
    assert t_t.shape == (3,), "Thermallage muss ein 3er sein"
    assert load_thermal_zuschlag(tmp) == (0.0, 0.0, 0.0), "ohne Datei nicht 0"
    save_thermal_zuschlag(tmp, (1.5, 0.5, -0.25))
    assert load_thermal_zuschlag(tmp) == (1.5, 0.5, -0.25)
    with open(os.path.join(tmp, _THERMAL_LAGE), "w") as fh:
        fh.write("{kaputt")
    assert load_thermal_zuschlag(tmp) == (0.0, 0.0, 0.0), "kaputte Datei nicht 0"

    class _P:
        yaw, t, enu = 0.3, np.array([1.0, 2.0]), (1.0, np.eye(3), np.zeros(3))
        skala, zentrum = 1.0, None

        def affine(self):
            return reg.affine(self.yaw, self.t, *self.enu, skala=self.skala,
                              zentrum=self.zentrum)
    p_ = _P()
    A_, b_ = lage_affine(p_, 90.0, [5.0, 6.0])
    assert np.allclose(b_, [5.0, 6.0, 0.0]) and np.allclose(A_ @ [1, 0, 0], [0, 1, 0])
    assert p_.yaw == 0.3 and p_.t.tolist() == [1.0, 2.0], "Basislage veraendert"
    # Feinausrichtung haengt an der Pipeline und wirkt in jedem lage_affine
    kipp = np.array([[1.0, 0, 0], [0, np.cos(0.01), -np.sin(0.01)],
                     [0, np.sin(0.01), np.cos(0.01)]])
    p_.s360_korrektur = {"M": kipp.tolist(), "v": [0.4, 0.35, 0.0]}
    A2, b2 = lage_affine(p_, 90.0, [5.0, 6.0])
    assert np.allclose(A2, kipp @ A_) and np.allclose(b2, kipp @ b_ + [0.4, 0.35, 0.0])
    p_.s360_korrektur = None
    print("  Zuschlag addiert, Datei hin und zurueck, Basislage unberuehrt, "
          "Feinausrichtung wirkt")

    print("== Test 7: Ebene schreiben und lesen ==")
    lay = os.path.join(tmp, "ebene")
    save_layer(lay, rgb, maske, {"quelle": "selbsttest"})
    zurueck = load_layer(lay, len(pts))
    assert zurueck is not None
    assert np.array_equal(zurueck[0], rgb) and np.array_equal(zurueck[1], maske)
    assert load_layer(lay, len(pts) + 1) is None, "falsche Punktzahl nicht erkannt"
    assert load_layer(os.path.join(tmp, "gibtsnicht"), 4) is None
    assert load_temperatur(lay, len(pts)) is None
    save_layer(lay, rgb, maske, {"quelle": "selbsttest"},
               temperatur=np.array([20.5, 21.0, np.nan, 30.0], np.float32))
    t = load_temperatur(lay, len(pts))
    assert t is not None and t[0] == 20.5 and np.isnan(t[2])
    save_layer(lay, rgb, maske, {"quelle": "selbsttest"})
    assert load_temperatur(lay, len(pts)) is None, "alte Temperatur blieb liegen"
    print("  Ebene passt, falsche Punktzahl wird abgelehnt, Temperatur hin und zurueck")
    shutil.rmtree(tmp, ignore_errors=True)
    print("meander SELFTEST OK")

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
    pipe.align()
    k = dict(pipe.kennwerte or {})
    k.update({"yaw_deg": float(np.degrees(pipe.yaw)),
              "t": [float(x) for x in np.asarray(pipe.t).ravel()]})
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


def set_manual(pipe, yaw_deg: float, tx: float, ty: float) -> None:
    """Handjustage uebernehmen (Gier in Grad, Versatz in Metern)."""
    pipe.yaw = float(np.radians(yaw_deg))
    t = np.asarray(pipe.t, dtype=float).ravel()
    if t.size < 2:
        t = np.zeros(2)
    pipe.t = np.array([float(tx), float(ty)])
    pipe.save_align()


def colorize_points(points: np.ndarray, cams, image_dir: str, A, b,
                    progress=None, cancel=None) -> tuple[np.ndarray, np.ndarray]:
    """Volle Wolke einfaerben; gibt (rgb uint8 (N,3), maske bool (N,)).

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
        if progress is not None and (i % 10 == 0 or i == n_cams - 1):
            progress((i + 1) / n_cams,
                     f"Färbe aus Bild {i + 1}/{n_cams} — "
                     f"{np.isfinite(best).mean() * 100:.1f} % getroffen")
    maske = np.isfinite(best)
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

    def colorize(self, points: np.ndarray, A, b) -> tuple[np.ndarray, np.ndarray]:
        """Wie :func:`colorize_points`, nur auf den verkleinerten Bildern."""
        P = np.asarray(points, dtype=np.float64)
        N = len(P)
        Ainv = np.linalg.inv(np.asarray(A, dtype=np.float64))
        PT = np.ascontiguousarray(((Ainv @ (P - np.asarray(b, float)).T).T).T)
        best = np.full(N, np.inf, dtype=np.float32)
        col = np.empty((N, 3), dtype=np.uint8)
        col[:] = _FALLBACK
        for i, img in enumerate(self.bilder):
            pc = (self.Rcw[i] @ PT).T + self.tcw[i]
            u, v, front, rad = self._cz._project(pc, self.size[i],
                                                 self.params[i], self.model)
            W, H = self.size[i]
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

def save_layer(out_dir: str, rgb: np.ndarray, maske: np.ndarray, meta: dict) -> None:
    """Farbebene ablegen — dasselbe Format wie die Einfaerbung aus der 360-Kamera."""
    os.makedirs(out_dir, exist_ok=True)
    np.ascontiguousarray(rgb, dtype=np.uint8).tofile(os.path.join(out_dir, "colors.bin"))
    np.ascontiguousarray(maske.astype(np.uint8)).tofile(
        os.path.join(out_dir, "valid.bin"))
    tmp = os.path.join(out_dir, "meta.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    os.replace(tmp, os.path.join(out_dir, "meta.json"))


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

    print("== Test 6: Ebene schreiben und lesen ==")
    lay = os.path.join(tmp, "ebene")
    save_layer(lay, rgb, maske, {"quelle": "selbsttest"})
    zurueck = load_layer(lay, len(pts))
    assert zurueck is not None
    assert np.array_equal(zurueck[0], rgb) and np.array_equal(zurueck[1], maske)
    assert load_layer(lay, len(pts) + 1) is None, "falsche Punktzahl nicht erkannt"
    assert load_layer(os.path.join(tmp, "gibtsnicht"), 4) is None
    print("  Ebene passt, falsche Punktzahl wird abgelehnt")
    shutil.rmtree(tmp, ignore_errors=True)
    print("meander SELFTEST OK")

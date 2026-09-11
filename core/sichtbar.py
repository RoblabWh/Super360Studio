"""Einfaerben mit Sichtbarkeit: jeder Punkt nur aus Bildern, die ihn sehen.

Warum: der einfache Weg projiziert jeden Kartenpunkt in das Bild, in dem er am
naechsten zur Bildmitte liegt. Bei einem Nadirflug sieht dieses Bild von einer
Wand aber nichts — davor liegt das Dach. Der Wandpunkt bekommt die Farbe des
Daches, das an dieser Bildstelle liegt, und das Dachmuster laeuft die Wand
hinunter. Genau das sieht „an Waenden verzogen“ aus. Ebenso bekommt der Boden
hinter einer Dachkante die Farbe der Kante.

Hier zwei Aenderungen:

1. **Verdeckung.** Je Kamera wird aus der Karte selbst eine Tiefenkarte
   gerechnet (verkleinert, der naechste Punkt je Zelle). Ein Punkt gilt als
   sichtbar, wenn er nicht merklich hinter dieser Tiefe liegt.
2. **Blickwinkel.** Unter den Bildern, die ihn sehen, gewinnt das, das am
   frontalsten auf seine Flaeche blickt — Normale aus der Karte. Fuer Dach und
   Boden ist das die Kamera darueber, wie bisher; fuer eine Wand eine Kamera
   seitlich davon, an deren Bildrand die Wand liegt. Ein leichter Abschlag fuer
   den Bildrand bleibt, dort ist die Optik am unsichersten.

Punkte, die kein Bild brauchbar sieht — verdeckt, oder nur streifend unter
``MIN_COS`` —, bleiben ungefaerbt statt falsch gefaerbt. Am DRZ-Flug sind das
rund 30 %: Unterholz, Hallen-Innenraum, eine zweite Dachschicht. Von oben
betrachtet fehlt dadurch nichts (0,2 % der sichtbaren Punkte).

Qt-frei.
"""

from __future__ import annotations

import os

import numpy as np

MIN_COS = 0.12            # streifender als ~83° zur Normalen: nicht verwenden
TIEFE_SKALA = 0.5         # Tiefenkarte in halber Bildaufloesung
_FALLBACK = (107, 107, 107)


def normalen(punkte: np.ndarray, raster: float = 0.15, progress=None) -> np.ndarray:
    """Oberflaechennormalen je Punkt (Vorzeichen beliebig), float32 (N, 3).

    Geschaetzt auf einem Raster der Karte und von dort auf alle Punkte
    uebertragen — 24 Millionen Punkte einzeln waeren Minuten, das Raster
    Sekunden, und eine Wand ist auf 15 cm immer noch eine Wand.
    """
    import open3d as o3d  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415

    P = np.asarray(punkte, dtype=np.float64)
    if progress is not None:
        progress(0.0, "Normalen: Raster …")
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[:: max(1, len(P) // 8_000_000)]))
    pc = pc.voxel_down_sample(raster)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=raster * 4, max_nn=30))
    Q = np.asarray(pc.points)
    N = np.asarray(pc.normals, dtype=np.float32)
    if progress is not None:
        progress(0.4, f"Normalen: {len(Q):,} Rasterpunkte, übertrage auf alle …")
    _, idx = cKDTree(Q).query(P, k=1, workers=-1)
    return N[idx]


def _tiefenkarte(u, v, z, W, H, skala):
    """Naechste Tiefe je Zelle.

    Bewusst ohne Minimumfilter ueber die Nachbarzellen: eine schraeg gesehene
    Wand spannt je Zelle ueber einen halben Meter Hoehe, ein Filter zoege die
    Tiefe ihrer Oberkante ueber mehrere Zellen und die Wand hielte sich selbst
    fuer verdeckt — so verlor ein erster Versuch die Haelfte der Karte.
    """
    w, h = max(1, int(W * skala)), max(1, int(H * skala))
    iu = np.clip((u * skala).astype(np.int64), 0, w - 1)
    iv = np.clip((v * skala).astype(np.int64), 0, h - 1)
    lin = iv * w + iu
    karte = np.full(w * h, np.inf, np.float32)
    np.minimum.at(karte, lin, z.astype(np.float32))
    return karte, lin


def colorize_sichtbar(points: np.ndarray, cams, image_dir: str, A, b,
                      normalen_welt: np.ndarray | None = None, progress=None,
                      cancel=None, log=None, temperatur=None) -> tuple:
    """Wie ``meander.colorize_points``, aber mit Verdeckung und Blickwinkel.

    ``normalen_welt`` (N, 3) im Rahmen der Karte; fehlen sie, werden sie
    geschaetzt. Rueckgabe: (rgb uint8 (N, 3), maske bool (N,)).

    ``temperatur`` ist optional eine Funktion ``(index, name) -> Bild in °C``
    (s. ``core.temperatur.quelle``); dann kommt als drittes Element je Punkt
    die Temperatur aus demselben Bild und Pixel wie seine Farbe (NaN, wo keine).
    """
    from PIL import Image  # noqa: PLC0415
    from core.meander import find_pipeline  # noqa: PLC0415
    find_pipeline()
    from colorize_pipeline import colorize as cz  # noqa: PLC0415

    def p_(f, m):
        if progress is not None:
            progress(f, m)
        if cancel is not None and cancel():
            raise RuntimeError("Abgebrochen")

    P = np.asarray(points, dtype=np.float64)
    N_ = len(P)
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if normalen_welt is None:
        normalen_welt = normalen(P, progress=lambda f, m: p_(0.1 * f, m))
    names = cams["names"]
    Rcw = np.asarray(cams["Rcw"], dtype=np.float64)
    tcw = np.asarray(cams["tcw"], dtype=np.float64)
    size = np.asarray(cams["size"], dtype=np.float64)
    params = np.asarray(cams["params"], dtype=np.float64)
    model = cams["model"].item() if getattr(cams["model"], "shape", None) == () \
        else str(cams["model"])
    Ainv = np.linalg.inv(A)
    massstab = float(np.cbrt(abs(np.linalg.det(A))))   # COLMAP-Einheit -> Meter
    # Kamerazentren im Rahmen der Karte (fuer den Blickwinkel)
    C_col = -np.einsum("nji,nj->ni", Rcw, tcw)
    C_welt = C_col @ A.T + b

    # Punkte einmal nach x sortieren: je Kamera nur den Streifen anfassen,
    # den sie ueberhaupt sehen kann.
    ordnung = np.argsort(P[:, 0], kind="stable")
    xs = P[ordnung, 0]
    boden = float(np.percentile(P[:, 2], 5))

    beste = np.zeros(N_, dtype=np.float32)
    farbe = np.empty((N_, 3), dtype=np.uint8)
    farbe[:] = _FALLBACK
    temp = np.full(N_, np.nan, dtype=np.float32) if temperatur is not None else None
    n_cams = len(names)
    for i, name in enumerate(names):
        p_(0.1 + 0.9 * i / n_cams, f"Färbe mit Sichtbarkeit: Bild {i + 1}/{n_cams} — "
                                    f"{(beste > 0).mean() * 100:.1f} % gefärbt")
        W, H = size[i]
        # Reichweite am Boden grob: halbe Bilddiagonale mal Tiefe durch f
        f = float(params[i][0])
        tiefe = max(C_welt[i, 2] - boden, 5.0)
        reich = tiefe * np.hypot(W, H) / (2.0 * f) * 1.6 + 10.0
        lo, hi = np.searchsorted(xs, [C_welt[i, 0] - reich, C_welt[i, 0] + reich])
        idx = ordnung[lo:hi]
        idx = idx[np.abs(P[idx, 1] - C_welt[i, 1]) < reich]
        if not len(idx):
            continue
        Pc = (P[idx] - b) @ Ainv.T                    # in den COLMAP-Rahmen
        pc = Pc @ Rcw[i].T + tcw[i]
        u, v, vorn, rad = cz._project(pc, size[i], params[i], model)
        drin = vorn & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not drin.any():
            continue
        idx, u, v, z, rad = idx[drin], u[drin], v[drin], pc[drin, 2], rad[drin]
        karte, lin = _tiefenkarte(u, v, z, W, H, TIEFE_SKALA)
        # sichtbar: nicht merklich hinter der naechsten Flaeche in dieser Zelle
        # (30 cm plus 1 % der Tiefe: auf einer schraeg gesehenen Wand liegen
        # die Punkte einer Zelle bis dahin hintereinander)
        sichtbar = z <= karte[lin] + (0.30 / massstab + 0.01 * z)
        if not sichtbar.any():
            continue
        idx, u, v, z, rad = idx[sichtbar], u[sichtbar], v[sichtbar], z[sichtbar], rad[sichtbar]
        blick = C_welt[i] - P[idx]
        blick /= np.linalg.norm(blick, axis=1, keepdims=True)
        cos = np.abs((blick * normalen_welt[idx]).sum(1))
        # leichter Abschlag zum Bildrand hin
        r_max = np.hypot(W, H) / (2.0 * f)
        guete = (cos * (1.0 - 0.25 * (rad / r_max) ** 2)).astype(np.float32)
        besser = (cos >= MIN_COS) & (guete > beste[idx])
        if not besser.any():
            continue
        with Image.open(os.path.join(image_dir, str(name))) as im:
            img = np.asarray(im.convert("RGB"))
        ih, iw = img.shape[:2]
        sk_u, sk_v = iw / W, ih / H
        j = idx[besser]
        ui = np.clip((u[besser] * sk_u).astype(np.int64), 0, iw - 1)
        vi = np.clip((v[besser] * sk_v).astype(np.int64), 0, ih - 1)
        farbe[j] = img[vi, ui]
        beste[j] = guete[besser]
        if temp is not None:
            t_bild = temperatur(i, name)
            # Wer die Farbe aus diesem Bild bekommt, bekommt auch die
            # Temperatur daraus — sonst NaN statt einer fremden
            temp[j] = np.nan if t_bild is None else \
                _tabtasten(t_bild, u[besser], v[besser], W, H)
    maske = beste > 0
    if log is not None:
        log(f"Mit Sichtbarkeit gefärbt: {maske.mean() * 100:.1f} % der Punkte. Der "
            f"Rest ist von keiner Kamera zu sehen (unter Baumkronen und Vordächern, "
            f"im Inneren von Hallen, zweite Dachschicht) und bleibt ungefärbt, "
            f"statt die Farbe dessen zu bekommen, was davor liegt.")
    if temp is not None:
        return farbe, maske, temp
    return farbe, maske


def _tabtasten(bild, u, v, W, H):
    from core.temperatur import abtasten  # noqa: PLC0415
    return abtasten(bild, u, v, W, H)


# Bewusst kein Auffuellen verdeckter Punkte vom Nachbarn: ausprobiert, und im
# Inneren einer Halle (durch Tor und Oberlichter gescannt) verteilte es die
# Farbe weniger zufaellig sichtbarer Punkte zu Klecksen. Ungefaerbt und mit
# „Nur eingefaerbte Punkte“ ausgeblendet ist das ehrlichere Bild.


if __name__ == "__main__":
    # Selbsttest: ein Dach ueber einem Boden, eine Kamera schraeg darueber.
    # Der Boden unter dem Dach darf nicht die Dachfarbe bekommen.
    import tempfile
    from PIL import Image

    tmp = tempfile.mkdtemp(prefix="sichtbartest_")
    bild = np.zeros((200, 200, 3), np.uint8)
    bild[:, :] = (200, 30, 30)                  # alles rot ...
    Image.fromarray(bild).save(os.path.join(tmp, "a.png"))
    R = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])   # blickt nach unten
    C = np.array([0.0, 0.0, 30.0])
    cams = {"names": np.array(["a.png"]), "Rcw": R[None], "tcw": (-R @ C)[None],
            "size": np.array([[200.0, 200.0]]),
            "params": np.array([[200.0, 100.0, 100.0, 0.0]]),
            "model": np.array("SIMPLE_RADIAL")}
    g = np.mgrid[-4:4:0.1, -4:4:0.1].reshape(2, -1).T
    dach = np.c_[g, np.full(len(g), 5.0)]           # Dach 5 m hoch, 8 x 8 m
    boden = np.c_[g * 0.5, np.zeros(len(g))]        # Boden darunter, 4 x 4 m
    frei = np.c_[g * 0.5 + [9.0, 0.0], np.zeros(len(g))]  # Boden daneben, frei
    P = np.vstack([dach, boden, frei])
    n = np.tile([0.0, 0.0, 1.0], (len(P), 1)).astype(np.float32)
    rgb, maske = colorize_sichtbar(P, cams, tmp, np.eye(3), np.zeros(3), normalen_welt=n)
    a, bo, fr = len(dach), len(boden), len(frei)
    print(f"Dach sichtbar {maske[:a].mean() * 100:.0f} %, Boden unter dem Dach "
          f"{maske[a:a + bo].mean() * 100:.0f} %, freier Boden {maske[a + bo:].mean() * 100:.0f} %")
    assert maske[:a].mean() > 0.95, "Dach nicht gefaerbt"
    assert maske[a:a + bo].mean() < 0.05, "verdeckter Boden bekam Farbe"
    assert maske[a + bo:].mean() > 0.95, "freier Boden nicht gefaerbt"
    # Wand: Normale waagerecht, Kamera fast senkrecht darueber -> streifend
    wand = np.c_[np.full(50, 0.0), np.linspace(-1, 1, 50), np.linspace(0, 3, 50)]
    nw = np.tile([1.0, 0.0, 0.0], (50, 1)).astype(np.float32)
    _, mw = colorize_sichtbar(wand + [0.3, 0, 0], cams, tmp, np.eye(3), np.zeros(3),
                              normalen_welt=nw)
    assert mw.mean() < 0.05, "streifend gesehene Wand wurde gefaerbt"
    print("Wand unter streifendem Blick bleibt ungefärbt")
    print("sichtbar SELFTEST OK")

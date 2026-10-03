"""Onboard- und Maeander-Farben zu einer Ebene verschmelzen.

Beide Quellen faerben dieselben Kartenpunkte, aber mit verschiedenen Kameras:
die 360°-Kamera an Bord sieht Fassaden und Unterseiten, der DJI-Maeanderflug
Daecher und Boden. Nebeneinander passen ihre Farben nicht — anderer Sensor,
anderer Weissabgleich, andere Belichtung. Hier zwei Schritte:

1. **Farbabbildung.** Auf den Punkten, die *beide* Ebenen gefaerbt haben, wird
   eine affine Abbildung Onboard → Maeander geschaetzt, ``ziel ≈ M·quelle + t``
   mit einer 3×3-Matrix und einem Versatz — dasselbe Modell, das der
   Splat-Trainer je Bild lernt (``belichtung_D``/``belichtung_e``), hier einmal
   fuer den ganzen Flug. Bezug ist der Maeander: fester Weissabgleich, RTK,
   senkrechter Blick. Geschaetzt wird robust (Huber, iterativ neu gewichtet),
   denn in der Ueberlappung stecken Schatten, die zwischen den Fluegen
   gewandert sind, und Punkte, die eine der Quellen falsch getroffen hat.
2. **Mischung nach Flaechenlage.** Wo beide Farben da sind, entscheidet die
   Normale: waagerechte Flaechen nehmen ueberwiegend den Maeander, senkrechte
   ueberwiegend die angeglichene Onboard-Farbe, dazwischen ein weicher
   Uebergang. Wo nur eine Quelle faerbt, bleibt deren Farbe (Onboard
   angeglichen).

Die Abbildung wird auf einem Teil der Ueberlappung geschaetzt und auf dem Rest
geprueft; ``bericht`` nennt den mittleren Farbabstand vorher und nachher.

Qt-frei.
"""

from __future__ import annotations

import numpy as np

MAX_PROBEN = 300_000      # so viele Ueberlappungspunkte gehen in die Schaetzung
MIN_UEBERLAPPUNG = 2_000  # darunter ist keine Abbildung zu trauen
HUBER_K = 1.345           # Huber-Schwelle in Einheiten der robusten Streuung
ITERATIONEN = 12
#: Flaechenlage |n_z|, ab der der Maeander uebernimmt (weicher Uebergang)
LAGE_VON, LAGE_BIS = 0.40, 0.80
#: Mindestanteil jeder Quelle, wo beide faerben — kein harter Schnitt
GRUND = 0.10
_BLOCK = 2_000_000


def _abbruch(cancel) -> None:
    if cancel is not None and cancel():
        raise RuntimeError("Abgebrochen")


def schaetze_abbildung(quelle: np.ndarray, ziel: np.ndarray,
                       gewicht: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Robuste affine Abbildung ``ziel ≈ quelle @ M.T + t`` (Werte 0..1).

    Iterativ neu gewichtete kleinste Quadrate mit Huber-Gewichten auf dem
    Abstand je Punkt (alle drei Kanaele zusammen). ``gewicht`` ist ein
    optionales Vorgewicht je Punkt.
    """
    X = np.hstack([np.asarray(quelle, np.float64), np.ones((len(quelle), 1))])
    Y = np.asarray(ziel, np.float64)
    g0 = np.ones(len(X)) if gewicht is None else np.asarray(gewicht, np.float64)
    w = g0.copy()
    B = np.vstack([np.eye(3), np.zeros((1, 3))])
    for _ in range(ITERATIONEN):
        sw = np.sqrt(w)[:, None]
        B, *_ = np.linalg.lstsq(X * sw, Y * sw, rcond=None)
        r = np.linalg.norm(X @ B - Y, axis=1)
        s = 1.4826 * np.median(r) + 1e-6
        grenze = HUBER_K * s
        w = g0 * np.minimum(1.0, grenze / np.maximum(r, 1e-12))
    return B[:3].T.copy(), B[3].copy()


def anwenden(rgb: np.ndarray, M: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Abbildung auf uint8-Farben anwenden, Ergebnis float32 0..1."""
    x = np.asarray(rgb, np.float32) / 255.0
    return np.clip(x @ M.T.astype(np.float32) + t.astype(np.float32), 0.0, 1.0)


def _abstand(a: np.ndarray, b: np.ndarray) -> float:
    """Mittlerer Farbabstand in 0..255 (Median der euklidischen Abstaende)."""
    return float(np.median(np.linalg.norm(a - b, axis=1)) * 255.0)


def gewicht_maeander(normalen: np.ndarray) -> np.ndarray:
    """Anteil des Maeanders je Punkt aus der Flaechenlage, float32 0..1."""
    nz = np.abs(np.asarray(normalen, np.float32)[:, 2])
    s = np.clip((nz - LAGE_VON) / (LAGE_BIS - LAGE_VON), 0.0, 1.0)
    s = s * s * (3.0 - 2.0 * s)
    return GRUND + (1.0 - 2.0 * GRUND) * s


def unplausibel(M, t, bericht: dict) -> str | None:
    """Warum die Abbildung nicht als Kamera-Unterschied taugt — oder None.

    Zwei Kameras vor derselben Szene unterscheiden sich um Verstaerkung,
    Weissabgleich und etwas Uebersprechen: die Diagonale bleibt deutlich
    positiv und traegt die Matrix. Am is7-Flug lagen zwischen Onboard (08.09.)
    und Maeander (09.09.) umgeparkte Autos und Daecher, die Onboard aus 3 m Hoehe
    nur von innen sieht; die Abbildung bekam negative Spalten und senkte den
    Abstand nur von 111 auf 66. So eine Matrix ist kein Kameramodell, sondern
    ein Ausgleich zwischen verschiedenen Bildinhalten.
    """
    M = np.asarray(M, float)
    d = np.diag(M)
    neben = np.abs(M - np.diag(d)).sum(1)
    if (d < 0.4).any() or (neben > 0.6 * d).any():
        return ("die Farbmatrix ist kein Kameraunterschied (Diagonale "
                + " ".join(f"{x:.2f}" for x in d) + ")")
    if bericht.get("abstand_nachher", 0.0) > 0.6 * bericht.get("abstand_vorher", 1.0):
        return (f"die Angleichung senkt den Farbabstand kaum "
                f"({bericht['abstand_vorher']:.0f} → {bericht['abstand_nachher']:.0f})")
    return None


def fusioniere(onboard: tuple, maeander: tuple, normalen: np.ndarray,
               seed: int = 0, progress=None, cancel=None) -> dict:
    """Zwei Farbebenen (je ``(rgb uint8 N×3, maske bool N)``) zu einer.

    Rueckgabe ``{"rgb", "maske", "M", "t", "bericht"}``. Wirft, wenn die
    Ueberlappung fuer eine Abbildung zu klein ist.
    """
    rgb_o, m_o = onboard
    rgb_m, m_m = maeander
    n = len(m_o)
    if len(m_m) != n or len(normalen) != n:
        raise ValueError("Ebenen und Normalen passen nicht zur selben Wolke")

    def p_(f, m):
        if progress is not None:
            progress(f, m)

    beide = np.flatnonzero(m_o & m_m)
    if len(beide) < MIN_UEBERLAPPUNG:
        raise RuntimeError(
            f"Nur {len(beide):,} Punkte haben beide Farben — zu wenig, um die "
            f"Onboard-Farben an den Mäander anzugleichen (mindestens "
            f"{MIN_UEBERLAPPUNG:,}). Liegen beide Ebenen auf derselben Karte?")
    p_(0.05, f"Überlappung: {len(beide):,} Punkte mit beiden Farben")

    rng = np.random.default_rng(seed)
    auswahl = rng.permutation(beide)[: 2 * MAX_PROBEN]
    lern, pruef = auswahl[0::2], auswahl[1::2]
    q = rgb_o[lern].astype(np.float64) / 255.0
    z = rgb_m[lern].astype(np.float64) / 255.0
    # Waagerechte Flaechen sind die, auf denen der Maeander verlaesslich ist
    M, t = schaetze_abbildung(q, z, gewicht=gewicht_maeander(normalen[lern]))
    _abbruch(cancel)

    qp = rgb_o[pruef].astype(np.float32) / 255.0
    zp = rgb_m[pruef].astype(np.float32) / 255.0
    vorher = _abstand(qp, zp)
    nachher = _abstand(anwenden(rgb_o[pruef], M, t), zp)
    p_(0.15, f"Farbabstand Onboard↔Mäander: {vorher:.1f} → {nachher:.1f}")

    rgb = np.zeros((n, 3), np.uint8)
    for a in range(0, n, _BLOCK):
        _abbruch(cancel)
        e = min(n, a + _BLOCK)
        o = anwenden(rgb_o[a:e], M, t)
        m = rgb_m[a:e].astype(np.float32) / 255.0
        wm = gewicht_maeander(normalen[a:e])
        mo, mm = m_o[a:e], m_m[a:e]
        wm = np.where(mo & mm, wm, np.where(mm, 1.0, 0.0)).astype(np.float32)[:, None]
        rgb[a:e] = np.rint((wm * m + (1.0 - wm) * o) * 255.0).astype(np.uint8)
        p_(0.15 + 0.85 * e / n, "Mische Farben …")
    maske = m_o | m_m
    rgb[~maske] = 0

    bericht = {
        "ueberlappung": int(len(beide)),
        "nur_onboard": int((m_o & ~m_m).sum()),
        "nur_maeander": int((m_m & ~m_o).sum()),
        "abstand_vorher": vorher,
        "abstand_nachher": nachher,
        "anteil": float(maske.mean()),
    }
    bericht["unplausibel"] = unplausibel(M, t, bericht)
    return {"rgb": rgb, "maske": maske, "M": M, "t": t, "bericht": bericht}


if __name__ == "__main__":
    print("== Test 1: Abbildung wird wiedergefunden, trotz Ausreissern ==")
    rng = np.random.default_rng(1)
    M0 = np.array([[0.9, 0.05, 0.0], [0.02, 1.1, -0.03], [0.0, 0.08, 0.8]])
    t0 = np.array([0.03, -0.02, 0.05])
    q = rng.uniform(0.1, 0.8, (50_000, 3))
    z = np.clip(q @ M0.T + t0 + rng.normal(0, 0.01, q.shape), 0, 1)
    z[:10_000] = rng.uniform(0, 1, (10_000, 3))           # 20 % Fehlgriffe
    M, t = schaetze_abbildung(q, z)
    assert np.abs(M - M0).max() < 0.02, M
    assert np.abs(t - t0).max() < 0.01, t
    print(f"  groesster Fehler: M {np.abs(M - M0).max():.4f}, t {np.abs(t - t0).max():.4f}")

    print("== Test 2: Fusion mischt nach Flaechenlage ==")
    n = 40_000
    wahr = rng.uniform(0.1, 0.8, (n, 3))
    mea = np.rint(wahr * 255).astype(np.uint8)
    # Onboard = verzerrte Kamera: umgekehrte Abbildung der wahren Farbe
    onb_f = np.clip((wahr - t0) @ np.linalg.inv(M0).T, 0, 1)
    onb = np.rint(onb_f * 255).astype(np.uint8)
    nrm = np.zeros((n, 3), np.float32)
    nrm[: n // 2, 2] = 1.0                                  # Boden
    nrm[n // 2:, 0] = 1.0                                   # Wand
    m_o = np.ones(n, bool)
    m_m = np.ones(n, bool)
    m_m[-5_000:] = False                                    # Wand nur von Onboard
    m_o[:5_000] = False                                     # Boden nur vom Maeander
    mea[-5_000:] = 0
    erg = fusioniere((onb, m_o), (mea, m_m), nrm)
    b = erg["bericht"]
    print(f"  Abstand {b['abstand_vorher']:.1f} -> {b['abstand_nachher']:.1f}")
    assert b["abstand_nachher"] < 2.0 and b["abstand_vorher"] > 10.0, b
    fehler = np.abs(erg["rgb"].astype(int) - np.rint(wahr * 255).astype(int)).max()
    assert fehler <= 4, fehler
    assert erg["maske"].all()
    assert np.array_equal(erg["rgb"][:5_000], mea[:5_000])  # nur Maeander: unveraendert
    print(f"  groesster Fehler gegen die wahre Farbe: {fehler} von 255")

    assert b["unplausibel"] is None, b["unplausibel"]

    print("== Test 3: die Matrix vom is7-Flug gilt als unplausibel ==")
    M_is7 = [[0.422, -0.325, 0.142], [0.128, -0.238, 0.322], [0.108, -0.575, 0.766]]
    grund = unplausibel(M_is7, [0.46, 0.49, 0.46], {"abstand_vorher": 110.6,
                                                     "abstand_nachher": 65.5})
    assert grund is not None
    print(f"  {grund}")

    print("== Test 4: zu wenig Ueberlappung wird abgelehnt ==")
    try:
        fusioniere((onb, np.arange(n) < 100), (mea, np.arange(n) >= 100), nrm)
    except RuntimeError as e:
        print(f"  ok: {str(e)[:60]} …")
    else:
        raise AssertionError("haette ablehnen muessen")
    print("alle Tests bestanden")

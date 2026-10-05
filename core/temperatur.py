"""Temperaturen aus den radiometrischen JPEGs (R-JPEG) der DJI-Thermalkamera.

Das sichtbare Thermalbild ist nur eine Palette — Farbe, keine Grad. Die
Messwerte stecken daneben in der Datei:

* **APP3** traegt die Rohwerte des Sensors, 16 Bit je Pixel, in der vollen
  Sensoraufloesung (M30T: 640 x 512, das JPEG daneben ist hochgerechnet auf
  1280 x 1024), ueber mehrere Segmente verteilt.
* **APP5** traegt eine Umrechnungstabelle Rohwert -> Temperatur in Zehntelgrad
  (16.384 Eintraege int16, monoton). Das ist die kameraeigene Kalibrierung,
  mit den in der Kamera eingestellten Parametern (Emissionsgrad, Abstand) —
  also die Temperatur, die auch die DJI-App anzeigt.

Das DJI Thermal SDK wird dafuer nicht gebraucht. Am DRZ-Flug (M30T, 10:52 Uhr,
sonnig) ergibt die Tabelle fuer ein Dachbild 14,8 bis 40,4 °C, Median 34 °C.

Qt-frei.
"""

from __future__ import annotations

import os
import struct

import numpy as np

_TABELLE = 16384


def _segmente(daten: bytes) -> dict:
    """JPEG-Marker bis zum Bildbeginn: {Marker: [Nutzdaten, ...]}."""
    out: dict = {}
    i = 2
    n = len(daten)
    while i < n - 4 and daten[i] == 0xFF:
        m = daten[i + 1]
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        laenge = struct.unpack(">H", daten[i + 2:i + 4])[0]
        out.setdefault(m, []).append(daten[i + 4:i + 2 + laenge])
        if m == 0xDA:              # Bilddaten beginnen, danach keine Kopfdaten
            break
        i += 2 + laenge
    return out


def lies_rjpeg(pfad: str) -> np.ndarray | None:
    """Temperaturbild in °C (float32, Sensoraufloesung) oder None.

    None, wenn die Datei keine lesbaren Rohwerte oder keine plausible
    Umrechnungstabelle traegt — dann gibt es eben keine Temperatur, und das
    Einfaerben laeuft ohne.
    """
    try:
        with open(pfad, "rb") as fh:
            daten = fh.read()
    except OSError:
        return None
    seg = _segmente(daten)
    if 0xE3 not in seg or 0xE5 not in seg or 0xC0 not in seg:
        return None
    roh = np.frombuffer(b"".join(seg[0xE3]), dtype="<u2")
    tab = seg[0xE5][0]
    if len(tab) < 2 * _TABELLE:
        return None
    lut = np.frombuffer(tab[:2 * _TABELLE], dtype="<i2").astype(np.float32) / 10.0
    if np.any(np.diff(lut) < 0) or not (-100.0 <= lut.min() and lut.max() <= 1000.0):
        return None
    # Seitenverhaeltnis aus dem sichtbaren Bild (SOF0: Hoehe, Breite)
    hoehe, breite = struct.unpack(">HH", seg[0xC0][0][1:5])
    n = len(roh)
    h = int(round(np.sqrt(n * hoehe / max(breite, 1))))
    w = n // max(h, 1)
    if h * w != n or h < 16 or w < 16:
        return None
    roh = roh.reshape(h, w)
    if roh.max() >= _TABELLE:
        return None
    return lut[roh]


def quelle(pipe):
    """Funktion ``(index, rgb_name) -> Temperaturbild | None`` fuer das Einfaerben.

    Die Pipeline legt die Thermalbilder unter dem Namen ihres RGB-Partners ab
    und merkt sich in ``thermal_paare``, woher sie kamen — dort liegen die
    Originale mit den Rohwerten. Gelesene Bilder werden behalten; 59 Bilder
    zu 640 x 512 sind 77 MB.
    """
    paare = dict(getattr(pipe, "thermal_paare", None) or {})
    cache: dict = {}

    def hole(_i, name):
        name = os.path.basename(str(name))
        if name not in cache:
            pfad = paare.get(name)
            cache[name] = lies_rjpeg(pfad) if pfad else None
        return cache[name]

    return hole if paare else None


def abtasten(bild: np.ndarray, u: np.ndarray, v: np.ndarray, W: float, H: float) -> np.ndarray:
    """Temperatur (oder Farbe, bei einem RGB-Bild) an den Bildkoordinaten (u, v)
    eines W x H grossen Bildes."""
    h, w = bild.shape[:2]
    ui = np.clip((u * (w / W)).astype(np.int64), 0, w - 1)
    vi = np.clip((v * (h / H)).astype(np.int64), 0, h - 1)
    return bild[vi, ui]


if __name__ == "__main__":
    import shutil
    import sys
    import tempfile

    print("== Test 1: synthetisches R-JPEG ==")
    # Kopf wie bei DJI: SOF0 mit 1280 x 1024, APP3 mit Rohwerten 640 x 512,
    # APP5 mit Tabelle Rohwert -> Zehntelgrad
    roh = np.full((512, 640), 4000, np.uint16)
    roh[100:200, 100:200] = 4400
    lut = np.clip((np.arange(_TABELLE) - 3000) * 0.4 - 300, -300, 1600).astype(np.int16)

    def seg(marker, nutz):
        return bytes([0xFF, marker]) + struct.pack(">H", len(nutz) + 2) + nutz

    teile = [b"\xff\xd8", seg(0xE0, b"JFIF\x00")]
    rohb = roh.tobytes()
    for k in range(0, len(rohb), 65530):
        teile.append(seg(0xE3, rohb[k:k + 65530]))
    teile.append(seg(0xE5, lut.tobytes() + b"\x00\x00"))
    teile.append(seg(0xC0, b"\x08" + struct.pack(">HH", 1024, 1280) + b"\x03"))
    teile.append(seg(0xDA, b"\x00"))
    tmp = tempfile.mkdtemp(prefix="rjpeg_")
    pfad = os.path.join(tmp, "x_T.JPG")
    with open(pfad, "wb") as fh:
        fh.write(b"".join(teile))
    t = lies_rjpeg(pfad)
    assert t is not None and t.shape == (512, 640), None if t is None else t.shape
    erwartet_kalt = lut[4000] / 10.0
    erwartet_warm = lut[4400] / 10.0
    assert abs(t[0, 0] - erwartet_kalt) < 1e-4 and abs(t[150, 150] - erwartet_warm) < 1e-4
    print(f"  {t.shape[1]}x{t.shape[0]}, kalt {t[0, 0]:.1f} °C, warm {t[150, 150]:.1f} °C")
    # Abtasten in Koordinaten des 1280er-Bildes
    wert = abtasten(t, np.array([300.0, 10.0]), np.array([300.0, 10.0]), 1280, 1024)
    assert abs(wert[0] - erwartet_warm) < 1e-4 and abs(wert[1] - erwartet_kalt) < 1e-4
    print("  Abtasten im hochgerechneten Bild trifft das Sensorpixel")
    with open(pfad, "wb") as fh:
        fh.write(b"\xff\xd8" + seg(0xE0, b"JFIF\x00") + seg(0xDA, b"\x00"))
    assert lies_rjpeg(pfad) is None, "Datei ohne Rohwerte ergab Temperaturen"
    print("  ohne Rohwerte: None")
    shutil.rmtree(tmp, ignore_errors=True)

    echt = sys.argv[1] if len(sys.argv) > 1 else None
    if echt:
        print("== Test 2: echtes Bild ==")
        t = lies_rjpeg(echt)
        assert t is not None
        print(f"  {os.path.basename(echt)}: {t.min():.1f} .. {t.max():.1f} °C, "
              f"Median {np.median(t):.1f} °C")
    print("temperatur SELFTEST OK")

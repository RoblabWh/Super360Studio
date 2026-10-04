"""Kamera-Kalibrierung des 360°-Kopfes: welche Kalibrierdatei gilt."""
from __future__ import annotations

import os

REPO_WURZEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2026-07-25: neue Basalt-Kalibrierung (RoblabWh-Fork, pro Kamera headless)
# + photometrisch verfeinerte Extrinsik (rx 4.6°, ry 182.4°) + Vignette-Profil.
# Die Kopie in calib/ macht das Repo eigenstaendig.
CALIB_CANDIDATES = (
    os.path.join(REPO_WURZEL, "calib", "calib_result_new2.json"),
)


def default_calib() -> str:
    for p in CALIB_CANDIDATES:
        if os.path.isfile(p):
            return p
    raise RuntimeError("Keine Kamera-Kalibrierung gefunden "
                       f"(gesucht: {', '.join(CALIB_CANDIDATES)})")

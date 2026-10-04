"""Farbebenen eines Projekts: Dateien lesen, prüfen und vergleichen.

Eine Ebene ist ein Ordner im Projekt mit ``colors.bin`` (RGB uint8 N×3),
``valid.bin`` (uint8 N) und ``meta.json``; Thermal-Ebenen haben zusätzlich die
Temperatur je Punkt.
"""
from __future__ import annotations

import json
import os
from typing import Callable

import numpy as np

#: Ebenen mit Temperatur je Punkt
THERMAL = ("meander_thermal", "meander_thermal_splat")


def lade_farbdateien(colors_dir: str, n_points: int,
                      expected_fingerprint: str | None = None,
                      log_cb: Callable[[str], None] | None = None
                      ) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """colors.bin (RGB uint8 N×3) + valid.bin (uint8 N) laden und prüfen.

    ``expected_fingerprint``: Fingerprint der aktuell geladenen Aufzeichnung
    (``core.colorizer.rec_fingerprint``). Steht in colors/meta.json ein anderer
    ``rec_fingerprint``, stammt der Farb-Cache von einer früheren Aufzeichnung
    und wird ignoriert. Fehlt der Schlüssel (älterer Cache), wird er aus
    Kompatibilität akzeptiert. Die Farben werden unverändert zurückgegeben
    (ungefärbte Punkte bleiben 0,0,0 — Grau-Ersatz ist Sache der Anzeige).
    """
    try:
        colors = np.fromfile(os.path.join(colors_dir, "colors.bin"), dtype=np.uint8)
        valid = np.fromfile(os.path.join(colors_dir, "valid.bin"), dtype=np.uint8)
    except OSError as exc:
        return None, None, f"Farben nicht lesbar: {exc}"
    if colors.size != 3 * n_points or valid.size != n_points:
        return None, None, (f"Farb-Cache passt nicht zur Punktwolke "
                            f"({colors.size // 3} Farben, {n_points} Punkte) — ignoriert.")
    if expected_fingerprint is not None:
        stored = None
        try:
            with open(os.path.join(colors_dir, "meta.json"), encoding="utf-8") as fh:
                stored = json.load(fh).get("rec_fingerprint")
        except (OSError, ValueError):
            stored = None
        if stored is not None and stored != expected_fingerprint:
            return None, None, ("Farb-Cache stammt von einer anderen Aufzeichnung "
                                "— ignoriert.")
        if stored is None and log_cb is not None:
            log_cb("Farb-Cache ohne Aufzeichnungs-Fingerprint (älterer Stand) — "
                   "wird übernommen.")
    return colors.reshape(-1, 3), valid.astype(bool), None


def meta_lesen(ordner: str) -> dict | None:
    """``meta.json`` eines Ebenen-Ordners; None, wenn sie fehlt oder unlesbar ist."""
    try:
        with open(os.path.join(ordner, "meta.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def passt(ordner: str, **soll) -> bool:
    """Gehoert eine gespeicherte Ebene zu dieser Lage? Vergleicht Werte der meta.json."""
    try:
        with open(os.path.join(ordner, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return False
    for key, wert in soll.items():
        ist = meta.get(key)
        if wert is None or ist is None:
            if wert is not ist:
                return False
            continue
        if isinstance(wert, dict):
            if not all(np.allclose(np.asarray(ist.get(k), float), np.asarray(wert[k], float),
                                   atol=1e-6) for k in ("M", "v")):
                return False
        elif not np.allclose(np.asarray(ist, float), np.asarray(wert, float), atol=1e-6):
            return False
    return True

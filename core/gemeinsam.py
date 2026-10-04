"""Kleine Helfer, die mehrere core-Module brauchen.

Nur Standardbibliothek; dieses Modul importiert nichts aus ``core``.

Importform für alle core-Module: absolut am Modulkopf
(``from core.gemeinsam import write_json_atomic``), kein ``sys.path``-Eingriff
am Modulkopf und keine try/except-Rückfälle beim Import. Selbsttests laufen
nur als ``python3 -m core.<modul>`` aus der Repo-Wurzel.
"""
from __future__ import annotations

import json
import os

#: Grau für ungefärbte Punkte, wenn es in eine Farbebene geschrieben wird
GRAU_EBENE = 107
#: Grau für ungefärbte Punkte in der Anzeige (3D-Ansicht, Export)
GRAU_ANZEIGE = 90


def write_json_atomic(path: str, obj, indent: int = 2) -> None:
    """JSON über eine .tmp-Datei schreiben, dann umbenennen."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent)
    os.replace(tmp, path)


def pruefe_abbruch(cancel) -> None:
    """Wirft RuntimeError('Abgebrochen'), wenn ``cancel`` gesetzt ist.

    ``cancel`` darf ein ``threading.Event``, ein Callable ohne Argumente oder
    None sein.
    """
    if cancel is not None and (cancel() if callable(cancel) else cancel.is_set()):
        raise RuntimeError("Abgebrochen")


def melde(progress, f, m) -> None:
    """Fortschritt melden, sofern es einen Empfänger gibt."""
    if progress is not None:
        progress(float(f), m)


def kamera_modell(cams) -> str:
    """Kameramodell aus einem npz-/dict-Eintrag (0-d-Array oder str)."""
    m = cams["model"]
    return m.item() if getattr(m, "shape", None) == () else str(m)


def de(x: float, n: int = 1) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{n}f}".replace(".", ",")


def fmt_int(n: int) -> str:
    """Ganze Zahl mit Tausenderpunkten: 1234567 -> '1.234.567'."""
    return f"{n:,}".replace(",", ".")


if __name__ == "__main__":
    import tempfile
    import threading

    ereignis = threading.Event()
    pruefe_abbruch(ereignis)
    pruefe_abbruch(None)
    pruefe_abbruch(lambda: False)
    ereignis.set()
    for c in (ereignis, lambda: True):
        try:
            pruefe_abbruch(c)
        except RuntimeError as exc:
            assert str(exc) == "Abgebrochen"
        else:
            raise AssertionError("pruefe_abbruch hat nicht abgebrochen")

    gemeldet = []
    melde(lambda f, m: gemeldet.append((f, m)), 1, "x")
    melde(None, 0.5, "y")
    assert gemeldet == [(1.0, "x")] and isinstance(gemeldet[0][0], float)

    assert fmt_int(1234567) == "1.234.567" and fmt_int(12) == "12"
    from core.exploration import _de
    for x in (1.25, 0.05, -3.0, 1234.5):
        for n in (0, 1, 2):
            assert de(x, n) == _de(x, n), (x, n)

    import numpy as np
    assert kamera_modell({"model": np.array("PINHOLE")}) == "PINHOLE"
    assert kamera_modell({"model": "OPENCV"}) == "OPENCV"

    from core.project import _write_json_atomic
    with tempfile.TemporaryDirectory() as tmp:
        obj = {"a": [1, 2.5, None], "ä": {"b": True}}
        write_json_atomic(os.path.join(tmp, "neu.json"), obj)
        _write_json_atomic(os.path.join(tmp, "alt.json"), obj)
        with open(os.path.join(tmp, "neu.json"), "rb") as f1, \
                open(os.path.join(tmp, "alt.json"), "rb") as f2:
            assert f1.read() == f2.read()
        assert not os.path.exists(os.path.join(tmp, "neu.json.tmp"))
    print("gemeinsam SELFTEST OK")

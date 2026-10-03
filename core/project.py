"""Session/Cache-Verwaltung pro Bag (cache/<bag_dir_name>-<md5(abspath)[:8]>/...). Qt-frei.

Der Cache-Schluessel enthaelt neben dem Verzeichnisnamen einen kurzen Hash des
absoluten Bag-Pfads, damit zwei verschiedene Bags mit gleichem Ordnernamen
(z. B. /data/siteA/rosbag2_x und /data/siteB/rosbag2_x) sich nicht denselben
Cache teilen bzw. gegenseitig ueberschreiben. Ein Alt-Verzeichnis mit reinem
Basename wird beim ersten Zugriff automatisch migriert (umbenannt).

Layout (s. ARCHITECTURE.md; <key> = <bag_dir_name>-<md5(abspath)[:8]>):
    cache/<key>/
      recording/          FAST-LIO-Aufzeichnung (recording.py)
      colors/             colors.bin / valid.bin / meta.json
      pano_<W>/           index.json + %06d.jpg
      gps.json            Fix-Liste inkl. GPSRAW-Merge
      extrinsic.json      {"T_imu_cam0": [[4x4]]}
      settings.json       zuletzt genutzte Einstellungen
      exploration.json    Explorationsgrad des Bags (exploration.Explorationsgrad)
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

# Der Cache haelt abgeleitete Daten (mehrere GB) und liegt bewusst ausserhalb
# des Repos, weiterhin am historischen Ort. Per Env-Variable umhaengbar.
DEFAULT_CACHE_ROOT = os.environ.get(
    "SUPER360_CACHE_ROOT",
    os.path.expanduser("~/RosBagSuper_Gui/rosbag_suite/cache"),
)

_RECORDING_FILES = ("points.bin", "intensity.bin", "offsets.npy", "stamps.npy", "poses.npy", "meta.json")


def _write_json_atomic(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


class Project:
    """Cache-Verzeichnis eines Bags: Pfad-Helfer + Settings/Extrinsik-Persistenz."""

    def __init__(self, bag_path: str, cache_root: str = DEFAULT_CACHE_ROOT):
        self.bag_path = str(bag_path)
        # abspath normalisiert auch Trailing-Slashes → stabiler Schluessel.
        abs_path = os.path.abspath(self.bag_path)
        self.bag_name = os.path.basename(abs_path)
        if not self.bag_name:
            raise RuntimeError(f"Ungueltiger Bag-Pfad: {bag_path}")
        self.cache_root = str(cache_root)
        # Schluessel: Basename + kurzer Hash des absoluten Pfads, damit
        # namensgleiche Bags an verschiedenen Orten getrennte Caches bekommen.
        digest = hashlib.md5(abs_path.encode("utf-8")).hexdigest()[:8]
        self.dir = os.path.join(self.cache_root, f"{self.bag_name}-{digest}")
        # Migration: Alt-Verzeichnis (reiner Basename) auf den neuen
        # gehashten Namen umbenennen, sofern das Ziel noch nicht existiert.
        legacy = os.path.join(self.cache_root, self.bag_name)
        if legacy != self.dir and os.path.isdir(legacy) and not os.path.exists(self.dir):
            try:
                os.rename(legacy, self.dir)
            except OSError:
                pass  # z. B. Rennen mit zweiter Instanz — dann frisches Verzeichnis
        try:
            os.makedirs(self.dir, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"Cache-Verzeichnis nicht anlegbar: {self.dir} ({exc})") from exc

    @classmethod
    def from_dir(cls, dir_path: str) -> "Project":
        """Vorhandenes Projektverzeichnis oeffnen, ohne den Schluessel zu kennen.

        Der normale Weg leitet das Verzeichnis aus dem Bagpfad ab. Bei einer
        zusammengefuehrten Aufzeichnung ist dieser Pfad erfunden ("A+B") und
        steht nirgends — ohne diesen Weg liesse sich so ein Projekt nach dem
        Schliessen nie wieder oeffnen.
        """
        dir_path = os.path.abspath(dir_path)
        if not os.path.isdir(dir_path):
            raise RuntimeError(f"Kein Verzeichnis: {dir_path}")
        obj = cls.__new__(cls)
        obj.dir = dir_path
        obj.cache_root = os.path.dirname(dir_path)
        obj.bag_name = os.path.basename(dir_path).rsplit("-", 1)[0]
        obj.bag_path = ""
        meta_p = os.path.join(dir_path, "recording", "meta.json")
        if os.path.isfile(meta_p):
            try:
                with open(meta_p, encoding="utf-8") as fh:
                    obj.bag_path = str(json.load(fh).get("bag") or "")
            except (OSError, ValueError):
                pass
        return obj

    @staticmethod
    def list_projects(cache_root: str = DEFAULT_CACHE_ROOT) -> list:
        """Alle Projekte im Cache mit ihren Kennzahlen, neueste zuerst."""
        out = []
        if not os.path.isdir(cache_root):
            return out
        for name in sorted(os.listdir(cache_root)):
            d = os.path.join(cache_root, name)
            meta_p = os.path.join(d, "recording", "meta.json")
            if not os.path.isfile(meta_p):
                continue
            try:
                with open(meta_p, encoding="utf-8") as fh:
                    meta = json.load(fh)
            except (OSError, ValueError):
                continue
            quellen = meta.get("sources") or []
            out.append({
                "dir": d,
                "name": name.rsplit("-", 1)[0],
                "bag": meta.get("bag") or "",
                "n_scans": int(meta.get("n_scans") or 0),
                "n_points": int(meta.get("n_points") or 0),
                "created": meta.get("created") or "",
                "quellen": [q.get("bag", "") for q in quellen],
                "zusammengefuehrt": len(quellen) >= 2,
            })
        out.sort(key=lambda e: e["created"], reverse=True)
        return out

    # ------------------------------------------------------------ dir layout

    def _subdir(self, name: str) -> str:
        d = os.path.join(self.dir, name)
        os.makedirs(d, exist_ok=True)
        return d

    def recording_dir(self) -> str:
        return self._subdir("recording")

    #: Farbebenen: Schluessel -> Unterordner. "onboard" ist die Einfaerbung
    #: aus der 360-Kamera und heisst aus Kompatibilitaet weiter "colors".
    #: Die "_splat"-Ebenen kommen aus einem Gaussian Splat auf der Karte
    #: (core/splat.py) und liegen neben der direkten Projektion derselben
    #: Bilder, damit beide sich vergleichen lassen.
    LAYERS = {
        "onboard": "colors",
        "onboard_splat": "colors_onboard_splat",
        "meander_rgb": "colors_meander_rgb",
        "meander_splat": "colors_meander_splat",
        "meander_thermal": "colors_meander_thermal",
        "meander_thermal_splat": "colors_meander_thermal_splat",
        "fusion": "colors_fusion",
        "fusion_splat": "colors_fusion_splat",
    }

    def colors_dir(self) -> str:
        return self._subdir("colors")

    def layer_dir(self, key: str) -> str:
        """Ordner einer Farbebene; legt ihn an."""
        name = self.LAYERS.get(str(key))
        if name is None:
            raise RuntimeError(f"Unbekannte Farbebene: {key}")
        return self._subdir(name)

    def has_layer(self, key: str) -> bool:
        name = self.LAYERS.get(str(key))
        if name is None:
            return False
        d = os.path.join(self.dir, name)
        return all(os.path.isfile(os.path.join(d, f))
                   for f in ("colors.bin", "valid.bin", "meta.json"))

    def available_layers(self) -> list:
        """Vorhandene Farbebenen in fester Reihenfolge."""
        return [k for k in self.LAYERS if self.has_layer(k)]

    def meander_work_dir(self) -> str:
        """Arbeitsordner der Maeander-Pipeline (COLMAP-Modell, Bilder, Lage)."""
        return self._subdir("meander")

    def pano_dir(self, width: int) -> str:
        return self._subdir(f"pano_{int(width)}")

    def gps_json(self) -> str:
        return os.path.join(self.dir, "gps.json")

    def extrinsic_json(self) -> str:
        return os.path.join(self.dir, "extrinsic.json")

    def settings_json(self) -> str:
        return os.path.join(self.dir, "settings.json")

    def exploration_json(self) -> str:
        return os.path.join(self.dir, "exploration.json")

    # ------------------------------------------------------------ existence

    def has_recording(self) -> bool:
        d = os.path.join(self.dir, "recording")
        return all(os.path.isfile(os.path.join(d, f)) for f in _RECORDING_FILES)

    def has_colors(self) -> bool:
        d = os.path.join(self.dir, "colors")
        return all(os.path.isfile(os.path.join(d, f)) for f in ("colors.bin", "valid.bin", "meta.json"))

    def has_pano(self, width: int) -> bool:
        return os.path.isfile(os.path.join(self.dir, f"pano_{int(width)}", "index.json"))

    def has_gps(self) -> bool:
        return os.path.isfile(self.gps_json())

    def has_extrinsic(self) -> bool:
        return os.path.isfile(self.extrinsic_json())

    def has_exploration(self) -> bool:
        return os.path.isfile(self.exploration_json())

    # -------------------------------------------------------------- settings

    def load_settings(self) -> dict:
        path = self.settings_json()
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"Einstellungen nicht lesbar ({path}): {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"Einstellungen beschaedigt ({path}): kein JSON-Objekt.")
        return data

    def save_settings(self, d: dict) -> None:
        _write_json_atomic(self.settings_json(), dict(d))

    # ----------------------------------------------------------- Explorationsgrad

    def load_exploration(self) -> dict | None:
        """Gespeicherter Explorationsgrad, oder None.

        Anders als bei den Einstellungen wirft eine beschaedigte Datei hier
        nicht: der Wert ist abgeleitet und in Sekunden neu gerechnet — ein
        kaputter Zwischenstand darf das Oeffnen des Bags nicht aufhalten.
        """
        path = self.exploration_json()
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None
        return data if isinstance(data, dict) else None

    def save_exploration(self, d: dict) -> None:
        _write_json_atomic(self.exploration_json(), dict(d))

    # ------------------------------------------------------------- extrinsic

    def load_extrinsic(self) -> np.ndarray | None:
        path = self.extrinsic_json()
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            T = np.asarray(data["T_imu_cam0"], dtype=np.float64)
        except (json.JSONDecodeError, OSError, KeyError, ValueError) as exc:
            raise RuntimeError(f"Extrinsik nicht lesbar ({path}): {exc}") from exc
        if T.shape != (4, 4):
            raise RuntimeError(f"Extrinsik beschaedigt ({path}): Form {T.shape} statt (4, 4).")
        return T

    def save_extrinsic(self, T) -> None:
        arr = np.asarray(T, dtype=np.float64)
        if arr.shape != (4, 4):
            raise RuntimeError(f"Extrinsik muss 4x4 sein, erhalten {arr.shape}.")
        _write_json_atomic(self.extrinsic_json(), {"T_imu_cam0": arr.tolist()})


if __name__ == "__main__":
    BAG = "/home/lena/RosBagSuper_Gui/rosbag_2026-07-11_15-37-07_seg0"
    OUT = (
        "/tmp/super360_modtests/"
        "project"
    )
    os.makedirs(OUT, exist_ok=True)
    # Test-Cache-Root im Scratchpad: der echte cache/<seg0>/recording-Ordner wird
    # parallel von einem anderen Agenten beschrieben und bleibt hier unberuehrt.
    root = os.path.join(OUT, "cache_root")

    import hashlib as _hl
    import shutil as _sh

    if os.path.isdir(root):
        _sh.rmtree(root)  # frischer Testlauf

    _digest = _hl.md5(os.path.abspath(BAG).encode("utf-8")).hexdigest()[:8]
    prj = Project(BAG, cache_root=root)
    print(f"bag_name = {prj.bag_name}")
    print(f"dir      = {prj.dir}")
    assert prj.bag_name == "rosbag_2026-07-11_15-37-07_seg0"
    assert prj.dir == os.path.join(root, f"rosbag_2026-07-11_15-37-07_seg0-{_digest}")
    assert os.path.isdir(prj.dir)

    # Pfad-Helfer + Layout
    assert prj.recording_dir() == os.path.join(prj.dir, "recording")
    assert prj.colors_dir() == os.path.join(prj.dir, "colors")
    assert prj.pano_dir(1920) == os.path.join(prj.dir, "pano_1920")
    assert os.path.isdir(prj.pano_dir(1920))
    assert prj.gps_json() == os.path.join(prj.dir, "gps.json")
    assert prj.extrinsic_json() == os.path.join(prj.dir, "extrinsic.json")
    assert prj.settings_json() == os.path.join(prj.dir, "settings.json")
    print("dir layout helpers OK")

    # has_* vor/nach Anlegen
    assert not prj.has_recording() and not prj.has_colors() and not prj.has_pano(1920)
    assert not prj.has_gps() and not prj.has_extrinsic()
    for f in _RECORDING_FILES:
        open(os.path.join(prj.recording_dir(), f), "wb").close()
    assert prj.has_recording()
    print("has_recording False->True OK")

    # Settings-Roundtrip
    assert prj.load_settings() == {}
    settings = {"brightness_min": 20, "brightness_max": 235, "pano_width": 1920, "rate": 1.0}
    prj.save_settings(settings)
    back = prj.load_settings()
    assert back == settings, f"Settings-Roundtrip: {back} != {settings}"
    print(f"settings roundtrip OK: {back}")

    # Extrinsik-Roundtrip
    assert prj.load_extrinsic() is None
    rng = np.random.default_rng(7)
    T = np.eye(4)
    T[:3, :3] = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    T[:3, 3] = [0.1, -0.02, 0.05]
    prj.save_extrinsic(T)
    T2 = prj.load_extrinsic()
    err = np.abs(T2 - T).max()
    assert T2 is not None and T2.shape == (4, 4) and err < 1e-12
    print(f"extrinsic roundtrip OK: max|err|={err:.2e}")

    # Explorationsgrad-Rundlauf; beschaedigte Datei ergibt None statt Fehler
    assert not prj.has_exploration() and prj.load_exploration() is None
    grad = {"prozent": 59.7, "phasen": [[18.7, 101.7]], "voxel_m": 0.5}
    prj.save_exploration(grad)
    assert prj.has_exploration() and prj.load_exploration() == grad
    with open(prj.exploration_json(), "w", encoding="utf-8") as fh:
        fh.write("{kaputt")
    assert prj.load_exploration() is None, "beschaedigte Datei muss None ergeben"
    os.remove(prj.exploration_json())
    print("exploration roundtrip OK (inkl. beschaedigter Datei)")

    # Trailing-Slash-Pfad ergibt gleichen Namen UND gleiches Cache-Verzeichnis
    prj_slash = Project(BAG + "/", cache_root=root)
    assert prj_slash.bag_name == prj.bag_name and prj_slash.dir == prj.dir

    # Namensgleiche Bags an verschiedenen Orten -> verschiedene Cache-Verzeichnisse
    other_bag = os.path.join(OUT, "elsewhere", prj.bag_name)
    os.makedirs(other_bag, exist_ok=True)
    prj_other = Project(other_bag, cache_root=root)
    assert prj_other.bag_name == prj.bag_name
    assert prj_other.dir != prj.dir, "Namenskollision: gleicher Cache fuer verschiedene Bags"
    print(f"kollisionsfrei: {os.path.basename(prj.dir)} != {os.path.basename(prj_other.dir)}")

    # Migration: Alt-Verzeichnis (reiner Basename) wird auf den Hash-Namen umbenannt
    root2 = os.path.join(OUT, "cache_root_migration")
    if os.path.isdir(root2):
        _sh.rmtree(root2)
    legacy = os.path.join(root2, prj.bag_name)
    os.makedirs(os.path.join(legacy, "recording"))
    with open(os.path.join(legacy, "recording", "meta.json"), "w", encoding="utf-8") as fh:
        fh.write("{}")
    prj_mig = Project(BAG, cache_root=root2)
    assert not os.path.exists(legacy), "Alt-Verzeichnis nicht migriert"
    assert os.path.isfile(os.path.join(prj_mig.dir, "recording", "meta.json"))
    print(f"migration OK: {legacy} -> {prj_mig.dir}")
    # existiert das Ziel bereits, bleibt ein (neu angelegtes) Alt-Verzeichnis stehen
    os.makedirs(legacy)
    prj_mig2 = Project(BAG, cache_root=root2)
    assert os.path.isdir(legacy) and prj_mig2.dir == prj_mig.dir
    print("migration idempotent OK")

    # Default-Root nur als Pfad geprueft (kein Anlegen erzwingen noetig — s. fixtests)
    prj_default = Project(BAG)
    print(f"default cache dir = {prj_default.dir}")
    assert prj_default.dir == os.path.join(DEFAULT_CACHE_ROOT, f"{prj.bag_name}-{_digest}")

    with open(os.path.join(OUT, "selftest_metrics.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"dir={prj.dir}\nsettings_roundtrip=OK\nextrinsic_max_err={err:.3e}\n")
    print("project SELFTEST OK")

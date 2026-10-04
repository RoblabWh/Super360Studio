"""RViz-Wiedergabe eines Rosbags: Start / Stopp / Wiederholen.

Qt-frei (s. ARCHITECTURE.md). Die GUI laeuft im Host-Python OHNE ROS, daher
laufen rviz2, `ros2 bag play` und das Leeren als Subprozesse in einer per
`bash -c 'source ... && exec ...'` gesourcten ROS-Umgebung.

Alle Prozesse laufen in eigenen Prozessgruppen (start_new_session=True), damit
sie samt Kindern zuverlaessig beendet werden koennen — `ros2 bag play` startet
Unterprozesse, die ein einfaches kill() sonst ueberleben.

Benutzung (aus einem Qt-Worker/Thread):
    p = RvizPlayer(log_cb=print)
    p.start("/pfad/zum/bag")     # rviz2 + bag play
    p.replay()                   # Anzeige leeren + von vorn
    p.stop()                     # alles beenden
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from typing import Callable, Optional

from core import ros_umgebung

__all__ = ["RvizPlayer", "RvizPlayerError"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)

DEFAULT_RVIZ_CONFIG = os.path.join(_PKG, "config", "playback.rviz")
_CLEAR_SCRIPT = os.path.join(_PKG, "scripts", "rviz_clear.py")
_PANO_SCRIPT = os.path.join(_PKG, "scripts", "pano_publisher.py")

# Nur /opt/ros reicht nicht: der Bag enthaelt livox_ros_driver2/CustomMsg sowie
# traj_utils/quadrotor_msgs aus EPIC. Ohne diese Typen bricht `ros2 bag play` ab.
# Die ROS-Distribution (Humble oder Jazzy) erkennt core.ros_umgebung.
DEFAULT_SETUPS = (
    ros_umgebung.setup_bash() or "",
    os.path.join(ros_umgebung.workspace("ws_livox"), "install", "setup.bash"),
    os.path.join(ros_umgebung.workspace("EPIC_ros2"), "install", "setup.bash"),
)


class RvizPlayerError(RuntimeError):
    """Fehler der RViz-Wiedergabe (deutsche Meldung)."""


class RvizPlayer:
    def __init__(self, rviz_config: str = DEFAULT_RVIZ_CONFIG,
                 setups: tuple[str, ...] = DEFAULT_SETUPS,
                 log_cb: Optional[Callable[[str], None]] = None,
                 pano: bool = True) -> None:
        self.pano = bool(pano)
        self.rviz_config = str(rviz_config)
        self.setups = tuple(s for s in setups if os.path.isfile(s))
        self._log_cb = log_cb
        self._rviz: Optional[subprocess.Popen] = None
        self._play: Optional[subprocess.Popen] = None
        self._pano: Optional[subprocess.Popen] = None
        self._bag: Optional[str] = None
        self._lock = threading.RLock()
        self._readers: list[threading.Thread] = []

    # ------------------------------------------------------------- Hilfsmittel
    def _log(self, line: str) -> None:
        if self._log_cb:
            try:
                self._log_cb(str(line).rstrip())
            except Exception:
                pass

    def _cmd(self, inner: str) -> list[str]:
        src = " && ".join(f"source {s}" for s in self.setups)
        return ["bash", "-c", f"{src} && exec {inner}" if src else f"exec {inner}"]

    @staticmethod
    def child_env() -> dict:
        """Umgebung fuer Kindprozesse — OHNE die Qt-Variablen von cv2.

        `import cv2` setzt prozessweit QT_QPA_PLATFORM_PLUGIN_PATH/QT_QPA_FONTDIR
        auf cv2s mitgelieferte Qt-Plugins und stellt cv2s lib64 vor
        LD_LIBRARY_PATH. Die GUI importiert cv2, wuerde das vererben — und rviz2
        ist selbst eine Qt-App: es laedt dann cv2s Plugins, kann das
        xcb-Plattform-Plugin nicht initialisieren und stirbt sofort beim Start.
        """
        env = dict(os.environ)
        env.setdefault("DISPLAY", ":0")
        for var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR",
                    "QT_PLUGIN_PATH"):
            env.pop(var, None)
        ld = env.get("LD_LIBRARY_PATH", "")
        if ld:
            keep = [x for x in ld.split(":") if x and "site-packages/cv2" not in x]
            env["LD_LIBRARY_PATH"] = ":".join(keep)
        return env

    def _spawn(self, inner: str, tag: str) -> subprocess.Popen:
        p = subprocess.Popen(
            self._cmd(inner), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, bufsize=1,
            env=self.child_env(), start_new_session=True)
        t = threading.Thread(target=self._pump, args=(p, tag), daemon=True)
        t.start()
        self._readers.append(t)
        return p

    def _pump(self, p: subprocess.Popen, tag: str) -> None:
        try:
            for line in p.stdout:            # type: ignore[union-attr]
                line = line.rstrip()
                if line and not _noise(line):
                    self._log(f"[{tag}] {line}")
        except (ValueError, OSError):
            pass

    @staticmethod
    def _kill(p: Optional[subprocess.Popen], timeout: float = 4.0) -> None:
        """Ganze Prozessgruppe beenden (SIGINT -> SIGTERM -> SIGKILL)."""
        if p is None or p.poll() is not None:
            return
        try:
            pgid = os.getpgid(p.pid)
        except OSError:
            return
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except OSError:
                return
            end = time.time() + timeout
            while time.time() < end:
                if p.poll() is not None:
                    return
                time.sleep(0.05)
            timeout = 1.5

    # ------------------------------------------------------------------ Status
    @staticmethod
    def _alive(p: Optional[subprocess.Popen]) -> bool:
        return p is not None and p.poll() is None

    def rviz_running(self) -> bool:
        with self._lock:
            return self._alive(self._rviz)

    def is_playing(self) -> bool:
        with self._lock:
            return self._alive(self._play)

    # ------------------------------------------------------------------ Aktion
    def ensure_metadata(self, bag_path: str) -> None:
        """metadata.yaml rekonstruieren, falls sie fehlt oder leer ist.

        Abgebrochene Aufnahmen haben eine 0-Byte-metadata.yaml; `ros2 bag play`
        weigert sich dann. Die .db3-Dateien selbst sind intakt.
        """
        meta = os.path.join(bag_path, "metadata.yaml")
        if os.path.isfile(meta) and os.path.getsize(meta) > 0:
            return
        if not any(f.endswith(".db3") for f in os.listdir(bag_path)):
            raise RvizPlayerError(
                f"Kein Rosbag2-Verzeichnis (keine .db3-Datei): {bag_path}")
        self._log("metadata.yaml fehlt oder ist leer — rekonstruiere (reindex) …")
        try:
            os.remove(meta)
        except OSError:
            pass
        r = subprocess.run(
            self._cmd(f"ros2 bag reindex {_q(bag_path)} -s sqlite3"),
            capture_output=True, text=True, timeout=900, env=self.child_env())
        if r.returncode != 0 or not os.path.getsize(meta):
            raise RvizPlayerError(
                "metadata.yaml konnte nicht rekonstruiert werden:\n"
                + (r.stderr or r.stdout)[-800:])
        self._log("metadata.yaml rekonstruiert.")

    def start(self, bag_path: str, rate: float = 1.0) -> None:
        """RViz oeffnen (falls noetig) und den Bag von vorn abspielen."""
        bag_path = os.path.abspath(bag_path)
        if not os.path.isdir(bag_path):
            raise RvizPlayerError(f"Bag-Verzeichnis nicht gefunden: {bag_path}")
        if not os.path.isfile(self.rviz_config):
            raise RvizPlayerError(f"RViz-Konfiguration fehlt: {self.rviz_config}")
        if not self.setups:
            raise RvizPlayerError(
                "Keine ROS-Umgebung gefunden (erwartet "
                f"{ros_umgebung.erwartet()}).")
        self.ensure_metadata(bag_path)

        with self._lock:
            self._bag = bag_path
            if self.is_playing():
                self._log("Wiedergabe laeuft bereits.")
                return
            if not self.rviz_running():
                self._log("Starte RViz …")
                self._rviz = self._spawn(
                    f"rviz2 -d {_q(self.rviz_config)}", "rviz")
                time.sleep(3.0)          # RViz soll die Topics abonniert haben
            if self.pano and not self._alive(self._pano):
                self._log("Starte 360°-Pano-Stitcher …")
                self._pano = self._spawn(f"python3 {_q(_PANO_SCRIPT)}", "pano")
            self._log(f"Spiele Bag ab: {os.path.basename(bag_path)}")
            self._play = self._spawn(
                f"ros2 bag play {_q(bag_path)} --rate {float(rate)}", "play")

    def stop(self, close_rviz: bool = True) -> None:
        """Wiedergabe (und optional RViz) beenden."""
        with self._lock:
            if self.is_playing():
                self._log("Stoppe Wiedergabe …")
            self._kill(self._play)
            self._play = None
            if close_rviz:
                if self.rviz_running():
                    self._log("Schliesse RViz …")
                self._kill(self._rviz)
                self._rviz = None
                self._kill(self._pano)   # Stitcher gehoert zur Anzeige
                self._pano = None

    def clear(self) -> None:
        """Alles aus der RViz-Anzeige loeschen (leere Wolken + DELETEALL)."""
        if not self.rviz_running():
            return
        self._log("Leere RViz-Anzeige …")
        r = subprocess.run(self._cmd(f"python3 {_q(_CLEAR_SCRIPT)}"),
                           capture_output=True, text=True, timeout=60,
                           env=self.child_env())
        out = (r.stdout or "").strip()
        if out:
            self._log(f"[clear] {out}")
        if r.returncode != 0:
            self._log(f"Leeren fehlgeschlagen: {(r.stderr or '')[-300:]}")

    def replay(self, rate: float = 1.0) -> None:
        """Wiederholen: stoppen, Anzeige leeren, von vorn abspielen."""
        with self._lock:
            bag = self._bag
        if not bag:
            raise RvizPlayerError("Kein Bag geladen — bitte zuerst „Starten“.")
        self.stop(close_rviz=False)
        # Der Player muss weg sein, BEVOR geleert wird, sonst schreibt er
        # sofort wieder Punkte in die gerade geleerte Anzeige.
        self.clear()
        self.start(bag, rate=rate)


def _q(s: str) -> str:
    import shlex
    return shlex.quote(str(s))


# Beim Wiederholen sendet der Bag dieselben (aelteren) Zeitstempel erneut; RViz
# meldet dann je TF-Nachricht TF_OLD_DATA und flutet das Protokoll. Fuer die
# Anzeige ist das folgenlos: alle Displays liegen im Fixed Frame `world`, und
# tf2 liefert fuer Ziel==Quelle die Identitaet — es wird gar nichts abgefragt.
_NOISE = (
    "TF_OLD_DATA",
    "Possible reasons are listed at",
    "buffer_core.cpp",
    'not found: "/home/jan/',      # fremder Pfad aus einem gesourcten Workspace
    "stdin is not a terminal device",
    "Press SPACE for Pause/Resume",
    "Press CURSOR_",
    "Adding keyboard callbacks",
    "Stereo is NOT SUPPORTED",
)


def _noise(line: str) -> bool:
    return any(n in line for n in _NOISE)

"""FAST-LIO2 subprocess orchestration: launch + ros2 bag play + recorder.

Qt-free (see ARCHITECTURE.md). All callbacks (progress_cb, log_cb) are invoked
on the thread that called run(); subprocess output is funneled through an
internal queue and drained there.

Flow of run():
  0. exclusive instance lock (fcntl.flock on /tmp/rosbag_suite_fastlio.lock) —
     a second concurrent run() fails fast with a German RuntimeError.
  1. kill_stale()  — only ONE fast_lio instance may exist (shared /laser_mapping).
  2. spawn recorder (scripts/record_fastlio.py) inside the sourced ROS env.
     The recorder writes into <out_dir>.tmp and NEVER touches out_dir, so a
     pre-existing recording survives any failed/cancelled run.
  3. spawn `ros2 launch fast_lio mapping.launch.py config_file:=<cfg> rviz:=false`,
     wait for "Node init finished." (timeout 30 s).
  4. wait for recorder READY (20 s), then `ros2 bag play <bag> --rate <rate> --topics <lid> <imu>`
     (only the config's sensor topics — never the bag's own /Odometry, /tf);
     first SCAN line must appear within max(20, 20/rate) s.
  5. after bag play exits: >=3 s grace, SIGINT recorder, SIGINT fast_lio group,
     escalate TERM/KILL; guaranteed cleanup in finally.
  6. success only if the recorder printed DONE and n_scans >= MIN_SCANS; then
     the old recording dir is removed and <out_dir>.tmp promoted via
     os.replace(). On any failure the tmp dir is deleted instead.
"""
from __future__ import annotations

import fcntl
import json
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from core import ros_umgebung

__all__ = ["FastLioResult", "FastLioRunner"]

# Coarse pgrep pre-filter; _match_stale_cmdline() decides what really belongs
# to fast_lio (finding: plain substring matching killed innocent processes).
_STALE_PATTERNS = ("fastlio_mapping", "rviz2", "ros2 launch fast_lio")

# Global single-instance lock (one FAST-LIO run per machine — shared /laser_mapping).
_LOCK_FILE = "/tmp/rosbag_suite_fastlio.lock"


@dataclass
class FastLioResult:
    recording_dir: str
    n_scans: int
    n_points: int
    expected_scans: int
    dropped_scans: int
    duration_s: float
    log_tail: str


class _Cancelled(Exception):
    pass


class _RunState:
    """Mutable per-run state; queue is fed by reader threads, drained by run()."""

    def __init__(self, expected: int, progress_cb, cancel, log_cb):
        self.expected = expected
        self.progress_cb = progress_cb
        self.cancel = cancel
        self.log_cb = log_cb
        self.q: queue.Queue[tuple[str, str]] = queue.Queue()
        self.tail: deque[str] = deque(maxlen=400)
        self.readers: list[threading.Thread] = []
        self.init_seen = False
        self.ready = False
        self.n_scans = 0
        self.n_points = 0
        self.last_scan_mono = 0.0
        self.done: tuple[int, int] | None = None


class FastLioRunner:
    INIT_TIMEOUT_S = 30.0
    READY_TIMEOUT_S = 20.0
    FIRST_SCAN_TIMEOUT_S = 20.0   # base; scaled with 1/rate in run()
    MIN_SCANS = 5                 # below this a run does not count as success

    def __init__(self, fastlio_ws: str | None = None, livox_ws: str | None = None,
                 ros_setup: str | None = None):
        # Humble (22.04) oder Jazzy (24.04) und die Workspaces (~ oder
        # $SUPER360_ROS_WS): erkannt in core.ros_umgebung.
        if ros_setup is None:
            ros_setup = ros_umgebung.setup_bash() or ""
        fastlio_ws = fastlio_ws or ros_umgebung.workspace("fastlio2_ws")
        livox_ws = livox_ws or ros_umgebung.workspace("ws_livox")
        self.fastlio_ws = fastlio_ws
        self.livox_ws = livox_ws
        self.ros_setup = ros_setup
        self._source = (
            "export PYTHONUNBUFFERED=1 RCUTILS_LOGGING_BUFFERED_STREAM=0 && "
            f"source {shlex.quote(ros_setup)} && "
            f"source {shlex.quote(os.path.join(livox_ws, 'install', 'setup.bash'))} && "
            f"source {shlex.quote(os.path.join(fastlio_ws, 'install', 'setup.bash'))}")
        self._recorder_script = str(
            Path(__file__).resolve().parent.parent / "scripts" / "record_fastlio.py")

    # ------------------------------------------------------------------ stale

    def kill_stale(self, log_cb=None) -> list[str]:
        """Kill foreign fast_lio processes (SIGINT, wait, escalate SIGKILL).

        Only processes whose cmdline CLEARLY belongs to fast_lio are touched
        (see _match_stale_cmdline): fastlio_mapping, 'ros2 launch fast_lio',
        and rviz2 only when its cmdline references fast_lio/fastlio.
        Zombies and processes we may not signal (PermissionError) are logged
        via log_cb and ignored; only a live, killable fastlio_mapping/launch
        process that survives SIGKILL raises a (German) RuntimeError.
        Returns list of "PID cmdline" strings that were signalled.
        """
        def _warn(msg: str) -> None:
            if log_cb is not None:
                log_cb(msg)

        victims = self._find_stale()
        if not victims:
            return []
        denied: set[int] = set()
        for pid in victims:
            if self._kill_pid(pid, signal.SIGINT) == "denied":
                denied.add(pid)

        def _pending() -> dict[int, str]:
            """Survivors we can and must still kill (no zombies, no denied)."""
            return {pid: cmd for pid, cmd in self._find_stale().items()
                    if pid not in denied and self._proc_state(pid) != "Z"}

        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline and _pending():
            time.sleep(0.25)
        survivors = _pending()
        if survivors:
            for pid in survivors:
                if self._kill_pid(pid, signal.SIGKILL) == "denied":
                    denied.add(pid)
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline and _pending():
                time.sleep(0.25)

        blocking: dict[int, str] = {}
        for pid, cmd in self._find_stale().items():
            if pid in denied:
                _warn(f"[runner] WARNUNG: keine Berechtigung, PID {pid} zu "
                      f"beenden ({cmd}) — wird ignoriert.")
            elif self._proc_state(pid) == "Z":
                _warn(f"[runner] WARNUNG: Zombie-Prozess PID {pid} ({cmd}) — "
                      "wird ignoriert.")
            elif self._match_stale_cmdline(cmd) in ("fastlio_mapping", "launch"):
                blocking[pid] = cmd
            else:
                _warn(f"[runner] WARNUNG: Prozess PID {pid} ({cmd}) ließ sich "
                      "nicht beenden — wird ignoriert.")
        if blocking:
            raise RuntimeError(
                "Konnte alte FAST-LIO-Prozesse nicht beenden: "
                + ", ".join(f"PID {p} ({c})" for p, c in blocking.items()))
        return [f"{pid} {cmd}" for pid, cmd in victims.items() if pid not in denied]

    @staticmethod
    def _match_stale_cmdline(cmd: str) -> str | None:
        """Classify a full cmdline: does it CLEARLY belong to fast_lio?

        Returns "fastlio_mapping" | "launch" | "rviz2" | None. Deliberately
        narrow: an innocent process whose arguments merely contain the pattern
        (e.g. 'vim fastlio_mapping_params.md', 'tail -f rviz2.log',
        'grep -r fastlio_mapping ~/ws') must NOT match.
        """
        toks = cmd.split()
        if not toks:
            return None
        base0 = os.path.basename(toks[0])
        if base0 == "fastlio_mapping":
            return "fastlio_mapping"
        # 'ros2 launch fast_lio …' — directly or via the interpreter that the
        # shebang prepends ('/usr/bin/python3 /opt/ros/…/bin/ros2 launch …').
        for i in (0, 1):
            if (len(toks) >= i + 3 and os.path.basename(toks[i]) == "ros2"
                    and toks[i + 1] == "launch" and toks[i + 2] == "fast_lio"):
                return "launch"
        # rviz2 only with a fast_lio reference (the launch file starts
        # 'rviz2 -d .../fast_lio/rviz/fastlio.rviz').
        if base0 == "rviz2" and ("fast_lio" in cmd or "fastlio" in cmd):
            return "rviz2"
        return None

    @classmethod
    def _find_stale(cls) -> dict[int, str]:
        me, parent = os.getpid(), os.getppid()
        found: dict[int, str] = {}
        for pat in _STALE_PATTERNS:
            # No pkill: collect PIDs explicitly, signal individually.
            r = subprocess.run(["pgrep", "-fa", pat], capture_output=True, text=True)
            for line in r.stdout.splitlines():
                pid_s, _, cmd = line.partition(" ")
                try:
                    pid = int(pid_s)
                except ValueError:
                    continue
                if pid in (me, parent):
                    continue
                if cls._match_stale_cmdline(cmd) is None:
                    continue
                found[pid] = cmd
        return found

    @staticmethod
    def _proc_state(pid: int) -> str:
        """Process state from /proc/<pid>/stat ('Z' = zombie, '?' = gone)."""
        try:
            with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as f:
                stat = f.read()
        except OSError:
            return "?"
        try:
            return stat.rsplit(")", 1)[1].split()[0]
        except IndexError:
            return "?"

    @staticmethod
    def _kill_pid(pid: int, sig: int) -> str:
        """Send signal; returns 'ok' | 'gone' | 'denied'."""
        try:
            os.kill(pid, sig)
            return "ok"
        except ProcessLookupError:
            return "gone"
        except PermissionError:
            return "denied"

    # -------------------------------------------------------------------- run

    def run(self, bag_path: str, out_dir: str, config: str = "whs_dense.yaml",
            rate: float = 1.0, expected_scans: int | None = None,
            progress_cb=None, cancel=None, log_cb=None) -> FastLioResult:
        t0 = time.monotonic()
        bag_path = os.path.abspath(bag_path)
        if not os.path.exists(bag_path):
            raise RuntimeError(f"Bag nicht gefunden: {bag_path}")
        self._pruefe_umgebung()
        out_dir = os.path.abspath(out_dir)
        # Recorder writes into <out_dir>.tmp (same convention in
        # scripts/record_fastlio.py); promoted to out_dir only on success.
        tmp_dir = out_dir + ".tmp"
        parent = os.path.dirname(out_dir)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if expected_scans is None:
            expected_scans = self._count_lidar_scans(bag_path)

        st = _RunState(expected_scans, progress_cb, cancel, log_cb)
        lock_fd = self._acquire_lock()
        promote = False
        try:
            self._progress(st, "Beende alte FAST-LIO-Prozesse…", frac=0.0)
            for k in self.kill_stale(log_cb=lambda line: self._log(st, line)):
                self._log(st, f"[runner] Altprozess beendet: {k}")

            procs: dict[str, subprocess.Popen] = {}
            try:
                rec_cmd = (f"exec python3 -u {shlex.quote(self._recorder_script)}"
                           f" --out {shlex.quote(out_dir)} --expected {expected_scans}"
                           f" --bag {shlex.quote(bag_path)} --config {shlex.quote(config)}"
                           f" --rate {rate:g}")
                procs["recorder"] = self._spawn(rec_cmd, stdin=subprocess.PIPE)
                self._start_reader(procs["recorder"], "recorder", st)

                launch_cmd = ("exec stdbuf -oL -eL ros2 launch fast_lio mapping.launch.py "
                              f"config_file:={shlex.quote(config)} rviz:=false")
                procs["fast_lio"] = self._spawn(launch_cmd)
                self._start_reader(procs["fast_lio"], "fast_lio", st)

                self._progress(st, f"Starte FAST-LIO2 ({config})…", frac=0.0)
                self._wait(st, lambda: st.init_seen, self.INIT_TIMEOUT_S,
                           {n: procs[n] for n in ("fast_lio", "recorder")},
                           "FAST-LIO2-Start fehlgeschlagen: 'Node init finished' nicht "
                           f"innerhalb von {self.INIT_TIMEOUT_S:.0f} s im Log.")
                self._wait(st, lambda: st.ready, self.READY_TIMEOUT_S,
                           {n: procs[n] for n in ("fast_lio", "recorder")},
                           f"Recorder nicht bereit (READY) innerhalb von "
                           f"{self.READY_TIMEOUT_S:.0f} s.")

                self._progress(st, f"Spiele Bag ab (Rate {rate:g})…", frac=0.0)
                # Only the sensor topics FAST-LIO consumes: bags from the drone
                # also carry the onboard /Odometry (same stamps, other world
                # frame) and /tf; replayed, the recorder would pair some scans
                # with those foreign poses -> scans skewed through the cloud.
                topics = self._input_topics(config)
                self._log(st, f"[runner] Spiele nur ab: {' '.join(topics)}")
                play_cmd = (f"exec ros2 bag play {shlex.quote(bag_path)} --rate {rate:g}"
                            " --topics " + " ".join(shlex.quote(t) for t in topics))
                procs["bag_play"] = self._spawn(play_cmd)
                self._start_reader(procs["bag_play"], "bag_play", st)

                # Scale with playback rate: at rate 0.25 the lidar lead-in of a
                # bag takes 4x as long in wall time before the first scan.
                first_scan_timeout = max(
                    self.FIRST_SCAN_TIMEOUT_S,
                    self.FIRST_SCAN_TIMEOUT_S / max(float(rate), 1e-3))
                self._wait(st, lambda: st.n_scans > 0, first_scan_timeout, dict(procs),
                           "ros2 bag play liefert keine Scans (keine SCAN-Zeile innerhalb "
                           f"von {first_scan_timeout:.0f} s bei Rate {rate:g}) — Abbruch.")

                stall_warned = False
                while procs["bag_play"].poll() is None:
                    self._drain(st)
                    self._check_cancel(st)
                    for name in ("fast_lio", "recorder"):
                        if procs[name].poll() is not None:
                            self._drain(st)
                            raise RuntimeError(
                                f"{name} unerwartet beendet (Exit-Code "
                                f"{procs[name].returncode}).\n" + self._tail(st, 15))
                    if (not stall_warned and st.n_scans > 0
                            and time.monotonic() - st.last_scan_mono > 20.0):
                        stall_warned = True
                        self._log(st, "[runner] WARNUNG: seit 20 s keine neuen Scans.")
                    time.sleep(0.1)
                self._drain(st)
                if procs["bag_play"].returncode != 0:
                    raise RuntimeError(
                        "ros2 bag play fehlgeschlagen (Exit-Code "
                        f"{procs['bag_play'].returncode}).\n" + self._tail(st, 15))

                # Grace period: >=3 s after bag end, until 1.5 s without new scans.
                self._progress(st, "Warte auf letzte Scans…")
                t_end = time.monotonic()
                while True:
                    self._drain(st)
                    self._check_cancel(st)
                    now = time.monotonic()
                    if now - t_end >= 3.0 and now - max(st.last_scan_mono, t_end) >= 1.5:
                        break
                    if now - t_end >= 15.0:
                        break
                    time.sleep(0.1)

                self._progress(st, "Finalisiere Aufzeichnung…")
                self._teardown(procs, st)
            except _Cancelled:
                self._teardown(procs, st, fast=True)
                raise RuntimeError("Abgebrochen") from None
            except BaseException:
                try:
                    self._teardown(procs, st, fast=True)
                except Exception:
                    pass
                raise
            finally:
                self._hard_cleanup(procs, st)

            # ---- success evaluation + promotion --------------------------------
            # meta.json is read from the TMP dir only, and only after DONE was
            # parsed: NEVER fall back to a stale meta.json of a previous run.
            meta_path = os.path.join(tmp_dir, "meta.json")
            if st.done is None or not os.path.isfile(meta_path):
                raise RuntimeError(
                    "Recorder hat die Aufzeichnung nicht finalisiert (kein DONE) — "
                    "Lauf verworfen; eine vorhandene Aufzeichnung bleibt unverändert.\n"
                    + self._tail(st, 15))
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            n_scans, n_points = int(meta["n_scans"]), int(meta["n_points"])
            if n_scans < self.MIN_SCANS:
                raise RuntimeError(
                    f"FAST-LIO hat keine/zu wenige Scans geliefert ({n_scans}) — "
                    "Lauf verworfen; eine vorhandene Aufzeichnung bleibt unverändert.\n"
                    + self._tail(st, 20))
            promote = True                    # from here on never delete tmp_dir
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            os.replace(tmp_dir, out_dir)
        finally:
            if not promote:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            self._release_lock(lock_fd)

        self._progress(st, f"Fertig: {n_scans} Scans, {n_points} Punkte", frac=1.0)
        return FastLioResult(
            recording_dir=out_dir, n_scans=n_scans, n_points=n_points,
            expected_scans=expected_scans,
            dropped_scans=max(0, expected_scans - n_scans),
            duration_s=time.monotonic() - t0, log_tail=self._tail(st, 40))

    def _pruefe_umgebung(self) -> None:
        """Fehlt ROS oder ein Workspace, sofort abbrechen statt nach 30 s
        Wartezeit auf 'Node init finished' mit einem fremden Log."""
        fehlt = [p for p in (self.ros_setup,
                             os.path.join(self.livox_ws, "install", "setup.bash"),
                             os.path.join(self.fastlio_ws, "install", "setup.bash"))
                 if not p or not os.path.isfile(p)]
        if fehlt:
            raise RuntimeError(
                "ROS-Umgebung für FAST-LIO2 unvollständig, es fehlt: "
                + ", ".join(p or f"ROS 2 ({ros_umgebung.erwartet()})" for p in fehlt)
                + " — s. README „Karte berechnen“.")

    # ------------------------------------------------------------------- lock

    @staticmethod
    def _acquire_lock() -> int:
        """Exclusive non-blocking instance lock; RuntimeError (German) if held."""
        try:
            fd = os.open(_LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o666)
        except OSError as exc:
            raise RuntimeError(
                f"Sperrdatei {_LOCK_FILE} nicht zugreifbar: {exc}") from exc
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            try:
                os.close(fd)
            except OSError:
                pass
            raise RuntimeError(
                "Eine andere FAST-LIO-Ausführung läuft bereits (Instanz-Sperre "
                f"{_LOCK_FILE} ist belegt) — bitte den anderen Lauf abwarten "
                "oder abbrechen.") from None
        return fd

    @staticmethod
    def _release_lock(fd: int) -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass

    # -------------------------------------------------------------- internals

    def _spawn(self, cmd: str, stdin=subprocess.DEVNULL) -> subprocess.Popen:
        return subprocess.Popen(
            ["bash", "-lc", f"{self._source} && {cmd}"],
            stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True)

    @staticmethod
    def _start_reader(proc: subprocess.Popen, name: str, st: _RunState) -> None:
        def pump() -> None:
            try:
                for raw in proc.stdout:
                    st.q.put((name, raw.rstrip("\n")))
            except (ValueError, OSError):
                pass

        t = threading.Thread(target=pump, name=f"pump-{name}", daemon=True)
        t.start()
        st.readers.append(t)

    @staticmethod
    def _log(st: _RunState, line: str) -> None:
        st.tail.append(line)
        if st.log_cb is not None:
            st.log_cb(line)

    @staticmethod
    def _progress(st: _RunState, msg: str, frac: float | None = None) -> None:
        if st.progress_cb is not None:
            if frac is None:
                frac = min(1.0, st.n_scans / max(1, st.expected))
            st.progress_cb(frac, msg)

    @staticmethod
    def _tail(st: _RunState, n: int) -> str:
        return "\n".join(list(st.tail)[-n:])

    @staticmethod
    def _check_cancel(st: _RunState) -> None:
        if st.cancel is not None and st.cancel.is_set():
            raise _Cancelled()

    def _drain(self, st: _RunState) -> None:
        while True:
            try:
                name, line = st.q.get_nowait()
            except queue.Empty:
                return
            if name == "recorder":
                if line == "READY":
                    st.ready = True
                elif line.startswith("SCAN "):
                    parts = line.split()
                    try:
                        st.n_scans, st.n_points = int(parts[1]), int(parts[2])
                    except (IndexError, ValueError):
                        pass
                    st.last_scan_mono = time.monotonic()
                    self._progress(st, f"Scan {st.n_scans}/{st.expected}")
                    continue                      # progress only, keep log lean
                elif line.startswith("DONE "):
                    parts = line.split()
                    try:
                        st.done = (int(parts[1]), int(parts[2]))
                    except (IndexError, ValueError):
                        pass
            elif name == "fast_lio" and "Node init finished" in line:
                st.init_seen = True
            self._log(st, f"[{name}] {line}")

    def _wait(self, st: _RunState, pred, timeout: float,
              procs: dict[str, subprocess.Popen], err_msg: str) -> None:
        deadline = time.monotonic() + timeout
        while True:
            self._drain(st)
            if pred():
                return
            self._check_cancel(st)
            for name, p in procs.items():
                if p.poll() is not None:
                    self._drain(st)
                    raise RuntimeError(f"{name} unerwartet beendet (Exit-Code "
                                       f"{p.returncode}).\n" + self._tail(st, 15))
            if time.monotonic() > deadline:
                raise RuntimeError(err_msg + "\n" + self._tail(st, 15))
            time.sleep(0.05)

    @staticmethod
    def _signal_group(proc: subprocess.Popen, sig: int) -> None:
        try:
            os.killpg(proc.pid, sig)              # pgid == pid (start_new_session)
        except (ProcessLookupError, PermissionError):
            try:
                proc.send_signal(sig)
            except Exception:
                pass

    def _await_exit(self, proc: subprocess.Popen, st: _RunState, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._drain(st)
            if proc.poll() is not None:
                return True
            time.sleep(0.05)
        return proc.poll() is not None

    def _teardown(self, procs: dict[str, subprocess.Popen], st: _RunState,
                  fast: bool = False) -> None:
        """Graceful stop: bag play, then recorder (finalizes files), then fast_lio."""
        play = procs.get("bag_play")
        if play is not None and play.poll() is None:
            self._signal_group(play, signal.SIGINT)
            self._await_exit(play, st, 5.0)
        rec = procs.get("recorder")
        if rec is not None:
            if rec.poll() is None:
                try:
                    rec.stdin.write("STOP\n")
                    rec.stdin.flush()
                except Exception:
                    pass
                self._signal_group(rec, signal.SIGINT)
            if not self._await_exit(rec, st, 4.0 if fast else 10.0):
                self._signal_group(rec, signal.SIGTERM)
                if not self._await_exit(rec, st, 3.0):
                    self._signal_group(rec, signal.SIGKILL)
                    self._await_exit(rec, st, 2.0)
        fl = procs.get("fast_lio")
        if fl is not None and fl.poll() is None:
            self._signal_group(fl, signal.SIGINT)
            if not self._await_exit(fl, st, 4.0 if fast else 10.0):
                self._signal_group(fl, signal.SIGTERM)
                if not self._await_exit(fl, st, 5.0):
                    self._signal_group(fl, signal.SIGKILL)
                    self._await_exit(fl, st, 2.0)
        self._drain(st)

    def _hard_cleanup(self, procs: dict[str, subprocess.Popen], st: _RunState) -> None:
        """finally-block safety net: no orphan processes, all pipes closed."""
        for p in procs.values():
            if p.poll() is None:
                self._signal_group(p, signal.SIGKILL)
                try:
                    p.wait(3)
                except Exception:
                    pass
        for t in st.readers:
            t.join(timeout=2.0)
        self._drain(st)
        for p in procs.values():
            for h in (p.stdin, p.stdout):
                try:
                    if h:
                        h.close()
                except Exception:
                    pass
        # Belt and braces: sweep only processes that clearly belong to fast_lio
        # (narrowed _find_stale). The instance lock guarantees no second run is
        # active, so this cannot kill another GUI instance's fresh run.
        for pid in self._find_stale():
            self._kill_pid(pid, signal.SIGKILL)

    DEFAULT_TOPICS = ("/livox/lidar", "/livox/imu")

    def _input_topics(self, config: str) -> list[str]:
        """lid_topic and imu_topic of the fast_lio config (installed copy
        first, then config/fastlio/ in the repo); defaults if not found."""
        candidates = [
            Path(self.fastlio_ws) / "install" / "fast_lio" / "share" / "fast_lio"
            / "config" / config,
            Path(__file__).resolve().parent.parent / "config" / "fastlio" / config,
        ]
        for path in candidates:
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            found = {}
            for key in ("lid_topic", "imu_topic"):
                m = re.search(rf"^\s*{key}\s*:\s*[\"']?([^\"'#\s]+)", text, re.MULTILINE)
                if m:
                    found[key] = m.group(1)
            if len(found) == 2:
                return [found["lid_topic"], found["imu_topic"]]
        return list(self.DEFAULT_TOPICS)

    @staticmethod
    def _count_lidar_scans(bag_path: str) -> int:
        from rosbags.highlevel import AnyReader
        try:
            with AnyReader([Path(bag_path)]) as reader:
                for conn in reader.connections:
                    if conn.topic == "/livox/lidar" or conn.msgtype.endswith("CustomMsg"):
                        return int(conn.msgcount)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Bag konnte nicht gelesen werden: {exc}") from exc
        raise RuntimeError(
            "Lidar-Topic (/livox/lidar) im Bag nicht gefunden — bitte expected_scans angeben.")


# ---------------------------------------------------------------------- test

if __name__ == "__main__":
    import numpy as np

    BAG = "/home/lena/RosBagSuper_Gui/rosbag_2026-07-11_15-37-07_seg0"
    ROOT = str(Path(__file__).resolve().parent.parent)
    OUT = os.path.join(ROOT, "cache", "rosbag_2026-07-11_15-37-07_seg0", "recording")
    MOD = ("/tmp/super360_modtests/"
           "fastlio_runner")
    os.makedirs(MOD, exist_ok=True)

    logf = open(os.path.join(MOD, "run.log"), "w", encoding="utf-8")

    def log_cb(line: str) -> None:
        logf.write(line + "\n")
        logf.flush()

    _last = [-1.0]

    def progress_cb(frac: float, msg: str) -> None:
        if frac - _last[0] >= 0.10 or frac in (0.0, 1.0):
            print(f"  [{frac*100:5.1f}%] {msg}", flush=True)
            _last[0] = frac

    runner = FastLioRunner()
    killed = runner.kill_stale()
    print("kill_stale():")
    if killed:
        for k in killed:
            print(f"  getötet: {k}")
    else:
        print("  keine Altprozesse gefunden")

    def one_run(rate: float) -> FastLioResult:
        _last[0] = -1.0
        print(f"\n=== Lauf mit Rate {rate} ===")
        return runner.run(BAG, OUT, config="whs_dense.yaml", rate=rate,
                          expected_scans=463, progress_cb=progress_cb, log_cb=log_cb)

    t_wall0 = time.time()
    attempts: list[tuple[float, FastLioResult]] = []
    res = one_run(1.0)
    attempts.append((1.0, res))
    if res.n_scans < 440:
        print(f"Nur {res.n_scans} Scans bei Rate 1.0 — Wiederholung mit Rate 0.5 …")
        res = one_run(0.5)
        attempts.append((0.5, res))
    wall = time.time() - t_wall0

    for rate, r in attempts:
        print(f"Rate {rate}: n_scans={r.n_scans} n_points={r.n_points} "
              f"expected={r.expected_scans} dropped={r.dropped_scans} "
              f"dauer={r.duration_s:.1f}s")

    # --- verify files on disk against ARCHITECTURE format ---
    offsets = np.load(os.path.join(OUT, "offsets.npy"))
    stamps = np.load(os.path.join(OUT, "stamps.npy"))
    poses = np.load(os.path.join(OUT, "poses.npy"))
    pts = np.fromfile(os.path.join(OUT, "points.bin"), dtype=np.float32).reshape(-1, 3)
    inten = np.fromfile(os.path.join(OUT, "intensity.bin"), dtype=np.float32)
    with open(os.path.join(OUT, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    print(f"\nDateien: points{pts.shape} intensity{inten.shape} offsets{offsets.shape} "
          f"stamps{stamps.shape} poses{poses.shape}")
    print("meta.json:", json.dumps(meta))

    assert res.n_scans >= 440, f"zu wenige Scans: {res.n_scans}"
    assert offsets.dtype == np.int64 and offsets.shape == (res.n_scans + 1,)
    assert offsets[0] == 0 and offsets[-1] == res.n_points == len(pts) == len(inten)
    assert stamps.shape == (res.n_scans,) and np.all(np.diff(stamps) > 0), \
        "Stempel nicht streng monoton"
    assert poses.shape == (res.n_scans, 7) and np.isfinite(poses).all(), \
        "Posen nicht endlich"
    per_scan = np.diff(offsets)
    assert 5000 <= per_scan.mean() <= 25000, f"Punkte/Scan unplausibel: {per_scan.mean():.0f}"

    traj = poses[:, :3]
    traj_len = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
    print(f"\nScans: {res.n_scans}/463  Punkte: {res.n_points}")
    print(f"Punkte/Scan: min={per_scan.min()} mean={per_scan.mean():.0f} "
          f"median={np.median(per_scan):.0f} max={per_scan.max()}")
    print(f"Stempelspanne: {stamps[-1]-stamps[0]:.2f} s, "
          f"dt median={np.median(np.diff(stamps))*1000:.1f} ms")
    print(f"Trajektorienlänge: {traj_len:.2f} m, "
          f"bbox={np.ptp(traj, axis=0).round(2).tolist()} m")
    print(f"Gesamt-Wandzeit Selbsttest: {wall:.1f} s")

    # --- evidence PNG: top-down trajectory ---
    import cv2
    xy = traj[:, :2]
    mn, span = xy.min(0), max(float(np.ptp(xy, axis=0).max()), 1e-6)
    pix = ((xy - mn) / span * 820 + 40).astype(np.int32)
    pix[:, 1] = 899 - pix[:, 1]
    img = np.full((900, 900, 3), 255, np.uint8)
    cv2.polylines(img, [pix], False, (180, 60, 0), 2)
    cv2.circle(img, tuple(pix[0]), 7, (0, 160, 0), -1)
    cv2.circle(img, tuple(pix[-1]), 7, (0, 0, 220), -1)
    cv2.putText(img, f"seg0 Trajektorie ({traj_len:.1f} m, {res.n_scans} Scans)",
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imwrite(os.path.join(MOD, "trajectory_topdown.png"), img)
    print(f"Beweis-PNG: {os.path.join(MOD, 'trajectory_topdown.png')}")

    # --- no stray processes ---
    print("\nProzess-Check (pgrep -fa):")
    stray = False
    for pat in ("fastlio_mapping", "rviz2", "ros2 launch", "ros2 bag", "record_fastlio"):
        r = subprocess.run(["pgrep", "-fa", pat], capture_output=True, text=True)
        lines = [ln for ln in r.stdout.splitlines() if "fastlio_runner.py" not in ln]
        for ln in lines:
            stray = True
            print(f"  STREUNER [{pat}]: {ln}")
    if not stray:
        print("  keine Streuner-Prozesse (fastlio/rviz/ros2 launch/bag/recorder)")
    r = subprocess.run(["pgrep", "-fa", "ros2-daemon"], capture_output=True, text=True)
    if r.stdout.strip():
        print(f"  Hinweis: ros2-Daemon läuft (normal): {r.stdout.strip()}")
    logf.close()
    print("\nSELBSTTEST BESTANDEN")

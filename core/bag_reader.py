"""Rosbag-Lesezugriff (rosbags-basiert, kein ROS-Sourcing noetig).

Liest ROS2-Bags (sqlite3) mit rosbags.highlevel.AnyReader. Der Nicht-Standard-Typ
mavros_msgs/GPSRAW wird aus der .msg-Definition registriert. Nur lesen, nie schreiben.
Qt-frei.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

GPSRAW_MSG_PATH = "/opt/ros/humble/share/mavros_msgs/msg/GPSRAW.msg"
GPSRAW_TYPENAME = "mavros_msgs/msg/GPSRAW"

_CAMERA_TYPES = ("sensor_msgs/msg/CompressedImage",)
_LIDAR_TYPES = ("livox_ros_driver2/msg/CustomMsg", "sensor_msgs/msg/PointCloud2")
_NAVSATFIX_TYPE = "sensor_msgs/msg/NavSatFix"
_IMU_TYPE = "sensor_msgs/msg/Imu"

_GPS_MERGE_MAX_DT = 0.3  # s: NavSatFix <-> GPSRAW Zuordnung

# Lotrechte aus dem Bag-Anfang. Ein Beschleunigungsmesser misst im Stand UND im
# Schwebeflug die Gegenkraft zur Schwerkraft, der Vektor zeigt also nach oben —
# nur waehrend echter Drehungen taugt er nicht. Rotorvibration mittelt sich raus,
# deshalb wird nicht auf Stille gefiltert, sondern nur auf grobe Drehbewegung.
_UP_GYRO_QUIET = 0.10   # rad/s, echter Stand am Anfang: wenn lang genug, gewinnt er
_UP_GYRO_MAX = 0.50     # rad/s, darueber dreht die Drohne wirklich
_UP_MIN_S = 0.30        # s brauchbare Samples, sonst keine Aussage
_UP_MAX_S = 3.00        # s Auswertefenster ab Bag-Anfang
_UP_MAX_SPREAD_DEG = 10.0  # Streuung der Richtungen, darueber unbrauchbar


@dataclass
class BagInfo:
    path: str
    start: float
    end: float
    duration: float
    topics: dict[str, tuple[str, int]]  # name -> (typ, anzahl)
    camera_topic: str | None
    lidar_topic: str | None
    gps_fix_topic: str | None
    gps_raw_topic: str | None
    imu_topic: str | None = None


@dataclass
class GpsFix:
    """NavSatFix + zeitlich naechster GPSRAW (|dt| <= 0.3 s) gemerged.

    Sentinel-Werte werden ROH durchgereicht (cov 4294967.295 m = UINT32_MAX mm,
    eph/epv 9999); die Bewertung uebernimmt georef.assess().
    """

    stamp: float
    lat: float
    lon: float
    alt: float
    status: int
    service: int
    cov_east_m: float
    cov_north_m: float
    cov_up_m: float
    cov_type: int
    fix_type: int | None = None
    eph_cm: int | None = None
    epv_cm: int | None = None
    satellites: int | None = None  # None wenn GPSRAW fehlt


@dataclass
class ImuUp:
    """Lotrechte im Sensorsystem, gemessen am Bag-Anfang.

    ``up_body`` ist der Einheitsvektor, der im Sensorsystem nach OBEN zeigt,
    ``tilt_deg`` sein Winkel gegen die Sensor-z-Achse und damit die Schraeglage
    des Einbaus. Einheiten sind egal, nur die Richtung zaehlt (der Livox meldet
    die Beschleunigung in g, nicht in m/s^2). ``spread_deg`` ist die Streuung
    der Einzelrichtungen und damit das Guetemass der Messung.
    """

    up_body: np.ndarray  # (3,) Einheitsvektor
    tilt_deg: float
    window_s: float  # Zeitspanne der verwendeten Samples
    n_samples: int
    spread_deg: float  # mittlere Winkelabweichung der Samples
    t_end: float  # Ende des Fensters (Bag-Zeit, s)


def baue_typestore(zusatz=()):
    """ROS2-Humble-Typestore plus Nicht-Standard-Typen aus .msg-Dateien.

    ``zusatz`` ist eine Folge von ``(msg_pfad, typname, ersatz)``. Liegt die
    .msg-Datei vor, wird ihre Definition registriert, sonst die Kurzfassung
    ``ersatz``. Laesst sich die Datei nicht verarbeiten, springt ebenfalls
    ``ersatz`` ein. Ist ``ersatz`` None, fehlt der Typ ohne Datei, und eine
    beschaedigte Datei wirft.
    """
    store = get_typestore(Stores.ROS2_HUMBLE)
    for msg_pfad, typname, ersatz in zusatz:
        pfad = Path(msg_pfad)
        text = pfad.read_text() if pfad.is_file() else ersatz
        if text is None:
            continue
        try:
            store.register(get_types_from_msg(text, typname))
        except Exception:  # noqa: BLE001 — beschaedigte .msg: Kurzfassung genuegt
            if ersatz is None:
                raise
            store.register(get_types_from_msg(ersatz, typname))
    return store


class BagReader:
    """Lesezugriff auf ein ROS2-Bag: Info, Kamera-Frames (JPEG/BGR), GPS-Fixe."""

    def __init__(self, bag_path: str):
        self.bag_path = str(bag_path)
        p = Path(self.bag_path)
        if not p.exists():
            raise FileNotFoundError(f"Bag nicht gefunden: {self.bag_path}")
        if p.is_dir():
            self._validate_bag_dir(p)
        self._typestore = baue_typestore([(GPSRAW_MSG_PATH, GPSRAW_TYPENAME, None)])
        try:
            self._reader = AnyReader([p], default_typestore=self._typestore)
            self._reader.open()
        except Exception as exc:
            raise RuntimeError(f"Bag konnte nicht geoeffnet werden: {exc}") from exc

        self._info: BagInfo | None = None
        # Kamera-Index (lazy): Header-Stamps sortiert + zugehoerige Log-Zeitstempel (ns)
        self._cam_conn = None
        self._cam_stamps: np.ndarray | None = None
        self._cam_log_ns: np.ndarray | None = None
        # Dekodierte Frames: LRU auf Instanzebene (OrderedDict) statt
        # functools.lru_cache ueber die gebundene Methode — letzteres haelt
        # self in einem Referenzzyklus fest (Reader + ~440 MB Cache wuerden
        # erst von der zyklischen GC freigegeben). close() leert den Cache.
        self._frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._frame_cache_size = 32

    @staticmethod
    def _validate_bag_dir(p: Path) -> None:
        """Klare deutsche Diagnose, bevor rosbags den Ordner als ROS1-Datei anfasst.

        Ein ROS2-Bag-Ordner braucht metadata.yaml. Ohne sie versucht AnyReader
        den ORDNER als ROS1-.bag-Datei zu öffnen und scheitert mit dem
        nichtssagenden 'Is a directory'.
        """
        if (p / "metadata.yaml").is_file():
            return
        db3 = sorted(p.glob("*.db3"))
        if db3:
            msg = (f"'{p.name}' ist ein unvollständiges ROS2-Bag: {len(db3)} "
                   ".db3-Datei(en), aber keine metadata.yaml — die Aufnahme "
                   "wurde vermutlich nicht sauber beendet (korrupt).")
            recovered = sorted(x.name for x in p.parent.glob(p.name + "_*")
                               if (x / "metadata.yaml").is_file())
            if recovered:
                msg += ("\nWiederhergestellte Varianten gefunden — bitte eine "
                        "davon öffnen: " + ", ".join(recovered))
            raise RuntimeError(msg)
        sub_bags = sorted(x.name for x in p.iterdir()
                          if x.is_dir() and (x / "metadata.yaml").is_file())
        if sub_bags:
            shown = ", ".join(sub_bags[:6]) + (" …" if len(sub_bags) > 6 else "")
            raise RuntimeError(
                f"'{p.name}' ist selbst kein Bag, enthält aber Bag-Ordner — "
                f"bitte einen davon öffnen: {shown}")
        raise RuntimeError(
            f"'{p.name}' ist kein ROS2-Bag (weder metadata.yaml noch "
            ".db3-Dateien gefunden).")

    # ------------------------------------------------------------------ info

    def info(self) -> BagInfo:
        if self._info is not None:
            return self._info
        topics: dict[str, tuple[str, int]] = {}
        for conn in self._reader.connections:
            typ, count = conn.msgtype, conn.msgcount
            if conn.topic in topics:
                typ, count = topics[conn.topic][0], topics[conn.topic][1] + count
            topics[conn.topic] = (typ, count)

        def pick(candidates: list[str], prefer_substr: str | None = None) -> str | None:
            if not candidates:
                return None
            if prefer_substr is not None:
                preferred = [t for t in candidates if prefer_substr in t]
                if preferred:
                    candidates = preferred
            return max(candidates, key=lambda t: topics[t][1])

        cam = pick([t for t, (typ, n) in topics.items() if typ in _CAMERA_TYPES and n > 0])
        lidar = pick([t for t, (typ, n) in topics.items() if typ in _LIDAR_TYPES and n > 0])
        gps_fix = pick(
            [t for t, (typ, n) in topics.items() if typ == _NAVSATFIX_TYPE and n > 0],
            prefer_substr="raw/fix",
        )
        gps_raw = pick([t for t, (typ, n) in topics.items() if typ == GPSRAW_TYPENAME and n > 0])
        imu = pick([t for t, (typ, n) in topics.items() if typ == _IMU_TYPE and n > 0],
                   prefer_substr="livox")

        start = self._reader.start_time * 1e-9
        end = self._reader.end_time * 1e-9
        self._info = BagInfo(
            path=self.bag_path,
            start=start,
            end=end,
            duration=end - start,
            topics=topics,
            camera_topic=cam,
            lidar_topic=lidar,
            gps_fix_topic=gps_fix,
            gps_raw_topic=gps_raw,
            imu_topic=imu,
        )
        return self._info

    # ---------------------------------------------------------------- camera

    def _ensure_camera_index(self) -> None:
        if self._cam_stamps is not None:
            return
        info = self.info()
        if info.camera_topic is None:
            raise RuntimeError("Kein Kamera-Topic (CompressedImage) im Bag gefunden.")
        conns = [c for c in self._reader.connections if c.topic == info.camera_topic]
        self._cam_conn = conns
        stamps: list[float] = []
        log_ns: list[int] = []
        for conn, timestamp, rawdata in self._reader.messages(connections=conns):
            msg = self._reader.deserialize(rawdata, conn.msgtype)
            stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            log_ns.append(timestamp)
        order = np.argsort(np.asarray(stamps), kind="stable")
        self._cam_stamps = np.asarray(stamps, dtype=np.float64)[order]
        self._cam_log_ns = np.asarray(log_ns, dtype=np.int64)[order]

    def camera_stamps(self) -> np.ndarray:
        """Header-Stamps aller Kamera-Frames, float64 (F,), sortiert."""
        self._ensure_camera_index()
        assert self._cam_stamps is not None
        return self._cam_stamps

    def _fetch_camera_raw(self, idx: int):
        self._ensure_camera_index()
        assert self._cam_log_ns is not None
        n = len(self._cam_log_ns)
        if not 0 <= idx < n:
            raise IndexError(f"Kamera-Frame {idx} ausserhalb des Bereichs 0..{n - 1}.")
        target = int(self._cam_log_ns[idx])
        for conn, timestamp, rawdata in self._reader.messages(
            connections=self._cam_conn, start=target, stop=target + 1
        ):
            if timestamp == target:
                return self._reader.deserialize(rawdata, conn.msgtype)
        raise RuntimeError(f"Kamera-Frame {idx} konnte nicht aus dem Bag gelesen werden.")

    def read_camera_jpeg(self, idx: int) -> bytes:
        """Rohe JPEG-Bytes des Frames idx."""
        return bytes(self._fetch_camera_raw(idx).data.tobytes())

    def _read_camera_uncached(self, idx: int) -> np.ndarray:
        msg = self._fetch_camera_raw(idx)
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"JPEG-Dekodierung fehlgeschlagen (Frame {idx}).")
        return img

    def read_camera(self, idx: int) -> np.ndarray:
        """Dekodiertes BGR-Bild (H,W,3) uint8; LRU-Cache 32 Frames (nicht mutieren!)."""
        idx = int(idx)
        cache = self._frame_cache
        img = cache.get(idx)
        if img is not None:
            cache.move_to_end(idx)
            return img
        img = self._read_camera_uncached(idx)
        cache[idx] = img
        while len(cache) > self._frame_cache_size:
            cache.popitem(last=False)
        return img

    # ------------------------------------------------------------------- imu

    def read_imu_accel(self, t_from: float | None = None,
                       t_to: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Header-Stempel (N,) und Beschleunigung (N,3) des IMU-Topics.

        Optional auf [t_from, t_to] begrenzt (Header-Zeit, dieselbe Uhr wie die
        Scan-Stempel der Aufzeichnung). Ohne IMU-Topic zwei leere Arrays.
        """
        info = self.info()
        leer = (np.empty(0, dtype=np.float64), np.empty((0, 3), dtype=np.float64))
        if info.imu_topic is None:
            return leer
        conns = [c for c in self._reader.connections if c.topic == info.imu_topic]
        if not conns:
            return leer
        stamps: list[float] = []
        accel: list[tuple[float, float, float]] = []
        for conn, timestamp, rawdata in self._reader.messages(connections=conns):
            msg = self._reader.deserialize(rawdata, conn.msgtype)
            hdr = getattr(msg, "header", None)
            t = (hdr.stamp.sec + hdr.stamp.nanosec * 1e-9) if hdr is not None else timestamp * 1e-9
            if (t_from is not None and t < t_from) or (t_to is not None and t > t_to):
                continue
            a = msg.linear_acceleration
            stamps.append(t)
            accel.append((a.x, a.y, a.z))
        if not stamps:
            return leer
        return np.asarray(stamps, dtype=np.float64), np.asarray(accel, dtype=np.float64)

    def read_imu_up(self, max_window_s: float = _UP_MAX_S) -> "ImuUp | None":
        """Lotrechte im Sensorsystem aus dem Anfang des Bags.

        Gemittelt werden die Richtungen der Beschleunigungsvektoren der ersten
        ``max_window_s``, ohne die Samples mit einer Drehrate ueber
        _UP_GYRO_MAX — waehrend einer Drehung misst der Sensor nicht mehr nur
        die Schwerkraft. Vibration der laufenden Rotoren stoert nicht, sie
        mittelt sich heraus.

        Liefert None, wenn es kein IMU-Topic gibt, zu wenig brauchbare Samples
        zusammenkommen oder die Richtungen zu stark streuen (dann war die
        Drohne von Anfang an in Bewegung und die Einbaulage ist aus diesem Bag
        nicht bestimmbar).
        """
        info = self.info()
        if info.imu_topic is None:
            return None
        conns = [c for c in self._reader.connections if c.topic == info.imu_topic]
        if not conns:
            return None

        dirs: list[np.ndarray] = []
        times: list[float] = []
        gyros: list[float] = []
        t0: float | None = None
        for conn, timestamp, rawdata in self._reader.messages(connections=conns):
            msg = self._reader.deserialize(rawdata, conn.msgtype)
            # Header-Stempel, damit t_end in derselben Uhr liegt wie die
            # Scan-Stempel der Aufzeichnung; Log-Zeit nur als Rueckfall.
            hdr = getattr(msg, "header", None)
            t = (hdr.stamp.sec + hdr.stamp.nanosec * 1e-9) if hdr is not None else timestamp * 1e-9
            if t0 is None:
                t0 = t
            if t - t0 > max_window_s:
                break
            w = msg.angular_velocity
            if float(np.sqrt(w.x * w.x + w.y * w.y + w.z * w.z)) > _UP_GYRO_MAX:
                continue  # dreht gerade: dieses Sample nicht verwenden
            a = msg.linear_acceleration
            v = np.array([a.x, a.y, a.z], dtype=np.float64)
            n = float(np.linalg.norm(v))
            if n < 1e-9:
                continue
            dirs.append(v / n)
            times.append(t)
            gyros.append(float(np.sqrt(w.x * w.x + w.y * w.y + w.z * w.z)))

        if len(dirs) < 2 or (times[-1] - times[0]) < _UP_MIN_S:
            return None
        arr = np.asarray(dirs)
        ts = np.asarray(times)
        # Steht die Drohne am Anfang wirklich still, ist genau dieser Abschnitt
        # die beste Messung — er endet beim ersten Sample daraus. Nur der
        # ZUSAMMENHAENGENDE Anfang zaehlt: wer bloss nach Drehrate filtert, nimmt
        # auch ruhige Momente aus dem Flug mit, in denen der Sensor beschleunigt
        # und der Vektor eben nicht mehr die Lotrechte ist. Reicht der Anfang
        # nicht (Rotoren liefen beim Aufnahmestart schon), bleibt es beim ganzen
        # Fenster: im Schwebeflug zeigt der Vektor im Mittel weiter nach oben.
        quiet = np.asarray(gyros) < _UP_GYRO_QUIET
        n_prefix = int(np.argmin(quiet)) if not quiet.all() else len(quiet)
        if n_prefix >= 2 and (ts[n_prefix - 1] - ts[0]) >= _UP_MIN_S:
            arr = arr[:n_prefix]
            ts = ts[:n_prefix]
        mean = arr.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm < 1e-9:
            return None
        up = mean / norm
        spread = float(np.degrees(np.arccos(np.clip(arr @ up, -1.0, 1.0))).mean())
        if spread > _UP_MAX_SPREAD_DEG:
            return None
        return ImuUp(
            up_body=up,
            tilt_deg=float(np.degrees(np.arccos(float(np.clip(up[2], -1.0, 1.0))))),
            window_s=float(ts[-1] - ts[0]),
            n_samples=int(len(arr)),
            spread_deg=spread,
            t_end=float(ts[-1]),
        )

    # ------------------------------------------------------------------- gps

    def read_gps(self) -> list[GpsFix]:
        """Alle NavSatFix-Fixe, je mit zeitlich naechstem GPSRAW gemerged."""
        info = self.info()
        if info.gps_fix_topic is None:
            return []

        raw_stamps: np.ndarray | None = None
        raw_msgs: list = []
        if info.gps_raw_topic is not None:
            conns = [c for c in self._reader.connections if c.topic == info.gps_raw_topic]
            stamps_list: list[float] = []
            for conn, _timestamp, rawdata in self._reader.messages(connections=conns):
                msg = self._reader.deserialize(rawdata, conn.msgtype)
                stamps_list.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
                raw_msgs.append(msg)
            if raw_msgs:
                order = np.argsort(np.asarray(stamps_list), kind="stable")
                raw_msgs = [raw_msgs[i] for i in order]
                raw_stamps = np.asarray(stamps_list, dtype=np.float64)[order]

        fixes: list[GpsFix] = []
        conns = [c for c in self._reader.connections if c.topic == info.gps_fix_topic]
        for conn, _timestamp, rawdata in self._reader.messages(connections=conns):
            msg = self._reader.deserialize(rawdata, conn.msgtype)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            cov = np.asarray(msg.position_covariance, dtype=np.float64)
            fix = GpsFix(
                stamp=stamp,
                lat=float(msg.latitude),
                lon=float(msg.longitude),
                alt=float(msg.altitude),
                status=int(msg.status.status),
                service=int(msg.status.service),
                cov_east_m=float(np.sqrt(max(cov[0], 0.0))),
                cov_north_m=float(np.sqrt(max(cov[4], 0.0))),
                cov_up_m=float(np.sqrt(max(cov[8], 0.0))),
                cov_type=int(msg.position_covariance_type),
            )
            if raw_stamps is not None and len(raw_msgs) > 0:
                j = int(np.searchsorted(raw_stamps, stamp))
                best, best_dt = None, _GPS_MERGE_MAX_DT
                for k in (j - 1, j):
                    if 0 <= k < len(raw_msgs):
                        dt = abs(raw_stamps[k] - stamp)
                        if dt <= best_dt:
                            best, best_dt = raw_msgs[k], dt
                if best is not None:
                    fix.fix_type = int(best.fix_type)
                    fix.eph_cm = int(best.eph)
                    fix.epv_cm = int(best.epv)
                    fix.satellites = int(best.satellites_visible)
            fixes.append(fix)
        fixes.sort(key=lambda f: f.stamp)
        return fixes

    # --------------------------------------------------------------- cleanup

    def close(self) -> None:
        # Cache sofort freigeben (~440 MB bei vollem Cache), nicht erst per GC.
        self._frame_cache.clear()
        if self._reader is not None and self._reader.isopen:
            self._reader.close()

    def __enter__(self) -> "BagReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()



class ThreadLocalBag:
    """BagReader-Fassade mit einem echten Reader je Thread.

    rosbags öffnet sqlite3-Verbindungen ohne ``check_same_thread=False`` —
    ein Reader darf daher nur in seinem Erzeuger-Thread lesen. Worker- und
    Prefetch-Threads bekommen hier je einen eigenen BagReader (Aufbau des
    Kamera-Index dauert < 0,2 s).
    """

    def __init__(self, bag_path: str):
        self.bag_path = str(bag_path)
        self._local = threading.local()

    def _bag(self) -> BagReader:
        bag = getattr(self._local, "bag", None)
        if bag is None:
            bag = BagReader(self.bag_path)
            self._local.bag = bag
        return bag

    def info(self) -> BagInfo:
        return self._bag().info()

    def camera_stamps(self) -> np.ndarray:
        return self._bag().camera_stamps()

    def read_camera(self, idx: int) -> np.ndarray:
        return self._bag().read_camera(idx)

    def read_camera_jpeg(self, idx: int) -> bytes:
        return self._bag().read_camera_jpeg(idx)

    def read_gps(self):
        return self._bag().read_gps()


if __name__ == "__main__":
    import os
    import time

    BAG = "/home/lena/RosBagSuper_Gui/rosbag_2026-07-11_15-37-07_seg0"
    OUT = (
        "/tmp/super360_modtests/"
        "bag_reader"
    )
    os.makedirs(OUT, exist_ok=True)

    reader = BagReader(BAG)
    info = reader.info()
    print("BagInfo:")
    print(f"  path         = {info.path}")
    print(f"  start/end    = {info.start:.3f} / {info.end:.3f}  (dauer {info.duration:.2f} s)")
    for name, (typ, n) in sorted(info.topics.items()):
        print(f"  topic {name}: {typ} x{n}")
    print(f"  camera_topic = {info.camera_topic}")
    print(f"  lidar_topic  = {info.lidar_topic}")
    print(f"  gps_fix_topic= {info.gps_fix_topic}")
    print(f"  gps_raw_topic= {info.gps_raw_topic}")

    t0 = time.perf_counter()
    stamps = reader.camera_stamps()
    t_index = time.perf_counter() - t0
    print(f"camera frames = {len(stamps)} (index in {t_index:.2f} s), "
          f"span {stamps[-1] - stamps[0]:.2f} s, monoton={bool(np.all(np.diff(stamps) >= 0))}")
    assert len(stamps) == 944, f"erwartet 944 Frames, erhalten {len(stamps)}"

    for idx in (0, 943):
        t0 = time.perf_counter()
        img = reader.read_camera(idx)
        dt = time.perf_counter() - t0
        print(f"frame {idx}: shape={img.shape} dtype={img.dtype} decode={dt * 1e3:.1f} ms")
        assert img.shape == (1520, 3040, 3) and img.dtype == np.uint8

    t0 = time.perf_counter()
    _ = reader.read_camera(0)
    print(f"frame 0 (LRU-Cache): {1e3 * (time.perf_counter() - t0):.3f} ms")

    jpeg = reader.read_camera_jpeg(0)
    assert jpeg[:2] == b"\xff\xd8", "kein JPEG-Header"
    print(f"frame 0 jpeg bytes = {len(jpeg)}")

    cv2.imwrite(os.path.join(OUT, "frame_000000.png"), reader.read_camera(0))
    print(f"Beweis-PNG: {OUT}/frame_000000.png")

    fixes = reader.read_gps()
    print(f"gps fixes = {len(fixes)}")
    assert len(fixes) == 226, f"erwartet 226 Fixe, erhalten {len(fixes)}"
    assert all(f.fix_type == 0 for f in fixes), "fix_type != 0 gefunden"
    assert all(f.satellites == 0 for f in fixes), "satellites != 0 gefunden"
    assert all(f.eph_cm is not None for f in fixes), "GPSRAW-Merge fehlt"
    f0 = fixes[0]
    print(f"fix[0]: stamp={f0.stamp:.3f} lat={f0.lat} lon={f0.lon} status={f0.status} "
          f"fix_type={f0.fix_type} sats={f0.satellites} eph_cm={f0.eph_cm} "
          f"cov_e={f0.cov_east_m:.3f} m cov_u={f0.cov_up_m:.3f} m")
    sentinel = sum(1 for f in fixes if f.cov_east_m > 4.0e6)
    print(f"fixe mit Sentinel-Kovarianz (~4294967.295 m): {sentinel}/{len(fixes)}")
    print(f"fixe mit eph_cm==9999: {sum(1 for f in fixes if f.eph_cm == 9999)}/{len(fixes)}")

    up = reader.read_imu_up()
    assert up is not None, "read_imu_up() liefert nichts"
    assert abs(np.linalg.norm(up.up_body) - 1.0) < 1e-9, "up_body nicht normiert"
    assert 0.0 <= up.tilt_deg <= 180.0 and up.n_samples >= 2
    assert up.spread_deg <= _UP_MAX_SPREAD_DEG
    assert up.window_s >= _UP_MIN_S
    print(f"imu_up: Einbau {up.tilt_deg:.1f} Grad, up_body={np.round(up.up_body, 3)}, "
          f"Fenster {up.window_s:.2f} s, n={up.n_samples}, Streuung {up.spread_deg:.1f} Grad")
    assert up.tilt_deg < 10.0, (
        f"dieser Flug war flach montiert, gemessen {up.tilt_deg:.1f} Grad")

    reader.close()
    print("bag_reader SELFTEST OK")

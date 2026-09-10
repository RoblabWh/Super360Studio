# Super360 Studio — Architektur (VERBINDLICH)

PyQt5-GUI: Rosbag → FAST-LIO2 (dichteste Punktwolke) → 360°-Pano-Player → Einfärbung
der Punktwolke aus den 360°-Bildern (mit Helligkeitsfilter) → GPS-Qualitätsprüfung +
Georeferenzierung → Export. Alle UI-Texte DEUTSCH.

Dieses Dokument friert die Interfaces ein. Module werden parallel implementiert —
**Signaturen, Dateiformate und Konventionen hier sind bindend.** Wer abweichen muss,
dokumentiert es in seinem Abschlussbericht.

## Verzeichnis

```
Super360Studio/
  app.py                 # Einstieg: QApplication, qdarktheme.setup_theme("dark"), MainWindow
  ARCHITECTURE.md
  core/                  # KOMPLETT Qt-frei (import PyQt5 verboten). Fortschritt via Callbacks.
    __init__.py
    bag_reader.py        # rosbags-basiert (kein ROS-Sourcing nötig)
    recording.py         # Datenformat der FAST-LIO-Aufzeichnung
    fastlio_runner.py    # Subprozess-Orchestrierung fast_lio + bag play + Recorder
    stitcher.py          # Double-Sphere-Kamera + Equirect-Stitcher (cv2.remap)
    colorizer.py         # Punktwolken-Einfärbung aus Dual-Fisheye, Helligkeits- und Himmelsfilter
    georef.py            # GPS-Qualitätsprüfung + Ausrichtung LIO-Trajektorie ↔ ENU
    project.py           # Session/Cache-Verwaltung pro Bag
  scripts/
    record_fastlio.py    # Standalone rclpy-Recorder (läuft in ROS-Umgebung als Subprozess)
  ui/                    # PyQt5
    __init__.py
    main_window.py       # MainWindow: Sidebar (Pipeline+Einstellungen) + Tabs
    cloud_view.py        # CloudView(QWidget): VTK-Punktwolken-Viewer
    pano_view.py         # PanoView(QWidget): 360°-Player mit Zoom
    gps_panel.py         # GpsPanel(QWidget): Qualitätsbericht + Georeferenzierung
  calib/                 # Double-Sphere-Kalibrierungen (Kopien aus dem Stitcher-Projekt)
  config/                # playback.rviz
  assets/                # Anwendungs-Icon
  run_gui.sh             # Launcher

<cache_root>/<bag_name>-<hash>/   # ausserhalb des Repos, s.u.
                         # Default: /home/lena/RosBagSuper_Gui/rosbag_suite/cache
                         # ueberschreibbar via SUPER360_CACHE_ROOT
```

## Fakten (verifiziert, nicht neu recherchieren)

- Bag (ROS2 Humble, sqlite3): Topics
  `/paycam/image_raw/compressed` sensor_msgs/CompressedImage, JPEG 3040×1520, ~20.4 fps.
  Layout: 2 kreisförmige Fisheyes nebeneinander, SPLIT_X=1520, links=cam0, rechts=cam1,
  je 1520×1520, Kreis Ø≈1520 zentriert bei x≈760/2280.
  `/mavros/global_position/raw/fix` NavSatFix 5 Hz; `/mavros/gpsstatus/gps1/raw`
  mavros_msgs/GPSRAW 5 Hz; `/livox/lidar` livox CustomMsg 10 Hz; `/livox/imu` 200 Hz.
- Referenz-Bag: `/home/lena/RosBagSuper_Gui/rosbag_2026-07-11_15-37-07_seg0`
  (46 s, 944 Kamera-Frames, 463 Scans). GPS darin TOT (fix_type=0, 0 Sats, lat/lon=0).
- Kalibrierung (Double-Sphere "ds", Basalt-JSON):
  `/home/lena/RosBagSuper_Gui/Super360_Stitcher_rosbag/work/calib_new_vign/calibration.json` (Default)
  `.../work/calib_new_refined/calibration.json` (Fallback ohne Vignette).
  JSON: `value0.T_imu_cam` (Liste, [0]=Identität⇒"imu"-Frame==cam0, [1]=Pose cam0→cam1,
  px..qw), `value0.intrinsics[i].intrinsics` = {fx,fy,cx,cy,xi,alpha}. Referenz-Numpy-Code:
  `Super360_Stitcher_rosbag/work/refine_extrinsic.py` (ds_project, rays) und `work/panoviewer.py`.
- FAST-LIO2: ws `/home/lena/fastlio2_ws`, Livox-Typen `/home/lena/ws_livox`.
  Start: `ros2 launch fast_lio mapping.launch.py config_file:=whs_dense.yaml rviz:=false`.
  Dichteste Daten: `/cloud_registered_body` (PointCloud2 XYZI, IMU-Frame, volle
  unverdünnte Scans, unabhängig von dense_pub_en) + `/Odometry` (nav_msgs, Pose des
  IMU-Body in `camera_init`); Stempel beider Topics sind IDENTISCH (lidar_end_time) →
  1:1-Matching über exakte (sec,nanosec). Erste(r) Scan(s) publizieren nicht (Init).
  PCD-Save im Code ist tot — NICHT verwenden. Publisher-QoS: RELIABLE depth 20.
  Lidar-IMU-Extrinsik (whs_dense.yaml): extrinsic_T=[-0.011,-0.02329,0.04412], R=I.
- Host-Python: numpy 2.2.6, cv2 4.12, PyQt5 5.15.6, vtk 9.1 (+QVTKRenderWindowInteractor
  funktioniert, getestet), pyqtgraph 0.12.4, open3d 0.19, scipy 1.14, rosbags 0.11.2,
  qdarktheme (pyqtdarktheme), pyproj, laspy. KEIN pyvista/PySide6.
  mavros_msgs .msg-Definitionen: `/opt/ros/humble/share/mavros_msgs/msg/GPSRAW.msg`
  (rosbags braucht register_types aus dieser Definition; Abhängigkeit
  `sensor_msgs/NavSatStatus` ggf. mitregistrieren).
- Hardware: 12 Kerne, 124 GB RAM, RTX 3060 Ti, X11 DISPLAY=:0.

## Konventionen

- Koordinaten: FAST-LIO-Welt = `camera_init` (IMU-Frame beim Start). Pose = T_world_imu.
  Punkt Welt: `p_w = R(q) @ p_body + t`. Quaternion-Reihenfolge IMMER (qx,qy,qz,qw).
  `scipy.spatial.transform.Rotation` verwenden.
- Bilder: BGR uint8 (cv2-Konvention) überall in core/. Erst die UI konvertiert zu RGB/QImage.
- Fortschritt: `progress_cb(frac: float, msg: str)` (frac 0..1, msg deutsch),
  Abbruch: `cancel: threading.Event` — jede lange Schleife prüft `cancel.is_set()`
  und bricht mit `RuntimeError("Abgebrochen")` ab. Beide Parameter optional (None-safe).
- Fehler: Exceptions mit deutscher Message werfen; UI fängt und zeigt QMessageBox.
- Kein `print` in core/ (außer scripts/record_fastlio.py, dessen stdout Protokoll ist).

## Cache-Layout (`project.py` verwaltet)

```
cache/<bag_dir_name>/
  recording/            # s. recording.py
  colors/colors.bin     # uint8, N×3, RGB (LEDIGLICH RGB, nicht BGR!)
  colors/valid.bin      # uint8 (0/1), N
  colors/meta.json      # {extrinsic, brightness_min, brightness_max, k_frames, ...}
  pano_<W>/index.json   # {"stamps": [...], "width": W}
  pano_<W>/%06d.jpg     # gestitchte Panos (Cache, lazy)
  gps.json              # Liste der Fixe + Merge GPSRAW (s. bag_reader)
  extrinsic.json        # {"T_imu_cam0": [[4x4]]} Kamera-Extrinsik (cam0 im IMU/Body-Frame)
  settings.json         # zuletzt genutzte Einstellungen
```

## core/bag_reader.py

```python
@dataclass
class BagInfo:
    path: str; start: float; end: float; duration: float
    topics: dict[str, tuple[str, int]]        # name -> (typ, anzahl)
    camera_topic: str | None; lidar_topic: str | None
    gps_fix_topic: str | None; gps_raw_topic: str | None

@dataclass
class GpsFix:      # NavSatFix + zeitlich nächster GPSRAW (|dt|<=0.3 s) gemerged
    stamp: float; lat: float; lon: float; alt: float
    status: int; service: int
    cov_east_m: float; cov_north_m: float; cov_up_m: float   # sqrt(diag), METER
    cov_type: int
    fix_type: int | None; eph_cm: int | None; epv_cm: int | None
    satellites: int | None                     # None wenn GPSRAW fehlt

class BagReader:
    def __init__(self, bag_path: str): ...     # AnyReader öffnen, GPSRAW-Typ registrieren
    def info(self) -> BagInfo: ...
    def camera_stamps(self) -> np.ndarray: ... # float64 (F,), Header-Stamps, sortiert
    def read_camera(self, idx: int) -> np.ndarray: ...       # BGR (1520,3040,3); LRU-Cache 32
    def read_camera_jpeg(self, idx: int) -> bytes: ...
    def iter_camera(self, start=0, stop=None): ...           # yield (idx, stamp, bgr)
    def read_gps(self) -> list[GpsFix]: ...
    def close(self): ...
```
Implementierung mit `rosbags.highlevel.AnyReader` + `rosbags.typesys` (Store ros2_humble,
GPSRAW aus .msg-Text registrieren). Kamera-Index (Verbindung→Offsets) beim ersten Zugriff
aufbauen. Nur lesen, nie schreiben.

## core/recording.py

Format auf Platte (`recording/`):
```
points.bin       float32, N×3 konkateniert   (Body/IMU-Frame, Reihenfolge der Scans)
intensity.bin    float32, N
offsets.npy      int64, (S+1,)               # Scan i = points[offsets[i]:offsets[i+1]]
stamps.npy       float64, (S,)               # lidar_end_time je Scan
poses.npy        float64, (S,7)              # x y z qx qy qz qw  (T_world_imu)
meta.json        {"bag": ..., "config": "whs_dense.yaml", "n_scans": S, "n_points": N,
                  "expected_scans": ..., "rate": ..., "created": ...,
                  "gravity_level": {...}}   # s.u., wird beim ersten Laden ergaenzt
```

### Kippkorrektur (gravity_level)

FAST-LIO verankert sein Weltsystem in der IMU-Lage des ersten Scans und richtet
es **nicht** an der Schwerkraft aus. Ein schräg montierter Livox kippt damit die
ganze Karte, nicht nur den ersten Scan — der legt die Lage nur fest. Gemessen ab
2026-09-08: 40,1° / 39,7° / 41,5°, davor durchgehend 0,3°–5,6°.

`BagReader.read_imu_up()` bestimmt die Lotrechte im Sensorsystem aus den ersten 3 s
des Bags. Der Beschleunigungsmesser misst im Stand **und** im Schwebeflug die
Gegenkraft zur Schwerkraft, der Vektor zeigt also nach oben; unbrauchbar wird er
nur während echter Drehungen (Samples über 0,50 rad/s fallen raus). Steht die
Drohne am Anfang still, zählt nur dieser zusammenhängende Abschnitt (Drehrate
unter 0,10 rad/s, mindestens 0,30 s) — nach Drehrate allein zu filtern würde auch
ruhige Momente aus dem Flug mitnehmen, in denen der Sensor beschleunigt. Liefen
die Rotoren beim Aufnahmestart schon, bleibt es beim ganzen Fenster; Vibration
mittelt sich heraus. `spread_deg` (mittlere Winkelabweichung der Einzelrichtungen)
ist das Gütemaß, über 10° gilt die Messung als unbrauchbar.

`Recording.load()` dreht damit das **Weltsystem** lotrecht: `T_neu = R_lot · T_alt`.
Nur die Posen ändern sich. Punkte im Body-Frame, Kamera-Extrinsik und
`rec_fingerprint` bleiben unberührt, der Farb-Cache gilt weiter. Gedreht wird die
kürzeste Drehung, der Gierwinkel bleibt also stehen.

Angewendet erst ab `LEVEL_MIN_TILT_DEG = 10.0`; darunter bleibt die Karte
bitgleich. Das Ergebnis steht als `gravity_level` in der meta.json:
`{quat, tilt_deg, threshold_deg, applied, up_body, mount_tilt_deg, window_s,
samples, spread_deg, scans_in_window}`. Schlägt die Messung fehl (Bag verschoben,
kein IMU-Topic, Drohne von Anfang an in Bewegung), wird nichts geschrieben und die
Karte bleibt, wie sie ist. `load(bag_path=...)` übersteuert den Pfad aus der
meta.json, der nach einem Umbenennen des Bags ins Leere zeigt.

Nachgemessen an den drei September-Flügen: die größte Ebene der ersten acht Scans
(der Boden) steht roh 37,1° / 39,3° / 40,4° schief und nach der Korrektur 5,1° /
2,2° / 0,9°. Die Drohnenlage aus der lotrechten Karte stimmt dann im Median auf
1,8°–2,3° mit `/mavros/local_position/odom` überein, genauso gut wie bei den
sauber montierten Flügen (1,7°–2,4°).
```python
class Recording:
    points: np.ndarray; intensity: np.ndarray; offsets: np.ndarray
    stamps: np.ndarray; poses: np.ndarray; meta: dict
    gravity_level: dict | None
    @staticmethod
    def load(dir_path: str) -> "Recording": ...      # np.memmap für points/intensity
    def world_points(self, progress_cb=None, cancel=None) -> np.ndarray: ...  # float32 N×3
    def interpolate_pose(self, t: float) -> np.ndarray | None:
        # 4×4 T_world_imu; SLERP(rot)+linear(trans) zwischen Nachbar-Scans,
        # None wenn t außerhalb (Toleranz 0.15 s an den Rändern: Randpose halten)
    def path_positions(self) -> np.ndarray: ...       # (S,3) Trajektorie
```

## core/fastlio_runner.py + scripts/record_fastlio.py

```python
@dataclass
class FastLioResult:
    recording_dir: str; n_scans: int; n_points: int
    expected_scans: int; dropped_scans: int; duration_s: float; log_tail: str

class FastLioRunner:
    def __init__(self, fastlio_ws="/home/lena/fastlio2_ws",
                 livox_ws="/home/lena/ws_livox", ros_setup="/opt/ros/humble/setup.bash"): ...
    def kill_stale(self) -> list[str]: ...  # tötet fremde fastlio_mapping/rviz per PID (SIGINT→KILL)
    def run(self, bag_path: str, out_dir: str, config: str = "whs_dense.yaml",
            rate: float = 1.0, expected_scans: int | None = None,
            progress_cb=None, cancel=None, log_cb=None) -> FastLioResult: ...
```
Ablauf von `run` (alles `bash -c 'source …humble… && source …livox… && source …fastlio…
&& exec …'`, Prozessgruppen + `start_new_session=True`, Aufräumen garantiert in finally):
1. `kill_stale()` — es darf nur EINE fast_lio-Instanz laufen (geteiltes /laser_mapping).
2. Recorder-Subprozess: `python3 scripts/record_fastlio.py --out <dir> --expected N`.
   Recorder: rclpy-Node, RELIABLE/VOLATILE depth 200, abonniert `/cloud_registered_body`
   + `/Odometry`; Matching über exakten Stamp (sec,nanosec); schreibt points/intensity
   inkrementell (offene .bin-Dateien), sammelt offsets/stamps/poses im RAM; druckt
   `READY`, dann pro Scan `SCAN <n>` auf stdout (line-buffered); auf SIGINT/`STOP`-Zeile:
   Dateien finalisieren, meta.json schreiben, `DONE <n_scans> <n_points>` drucken, Exit 0.
3. fast_lio-Launch (rviz:=false), auf "Node init finished" im Log warten (Timeout 30 s).
4. Auf READY des Recorders warten; dann `ros2 bag play <bag> --rate <rate>`.
5. bag play-Exit abwarten → 3 s Nachlauf → Recorder SIGINT, fast_lio SIGINT (dann
   TERM/KILL-Eskalation). progress = n_scans/expected_scans.
`fastlio_runner` parst Recorder-stdout für progress_cb und reicht fast_lio-stderr/stdout
zeilenweise an log_cb. seg0: expected_scans=463 (ggf. −2 Init-Scans tolerieren).

## core/stitcher.py

```python
class DoubleSphereCamera:
    def __init__(self, fx, fy, cx, cy, xi, alpha): ...
    @staticmethod
    def from_calib(calib_json_path: str) -> tuple["DoubleSphereCamera","DoubleSphereCamera", np.ndarray]:
        # (cam0, cam1, T_cam0_cam1 4×4)  — T aus value0.T_imu_cam[1] (== T_cam0_cam1!)
    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # pts (N,3) im Kameraframe → uv float32 (N,2), valid bool (N,)
        # DS-Modell: d1=|p|; zeta=xi*d1+z; d2=sqrt(x²+y²+zeta²);
        # den=alpha*d2+(1-alpha)*zeta; u=fx*x/den+cx; v=fy*y/den+cy
        # Gültigkeit: w1 = alpha/(1-alpha) wenn alpha<=0.5 sonst (1-alpha)/alpha;
        # w2=(w1+xi)/sqrt(2*w1*xi+xi²+1); gültig ⟺ z > -w2*d1 und den>eps
    def unproject(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...  # Richtungen+valid

class EquirectStitcher:
    def __init__(self, calib_json_path: str, width=1920,
                 depth_m=10.0, vignette_softness=0.15, expo_norm=True,
                 lut_cache_dir: str | None = None): ...
    # height = width//2. Beim Bau: für jedes Pano-Pixel Richtung (rays), Punkt = ray*depth,
    # in cam0- und cam1-Frame projizieren → 2 remap-Mappaare + Feather-Gewichte
    # (Distanz zum Bildkreisrand / FOV-Grenze, Kosinus-Feather, Breite vignette_softness).
    # LUT-Cache: npz unter lut_cache_dir, Key md5(calib+width+depth).
    def stitch(self, raw_3040: np.ndarray) -> np.ndarray: ...  # BGR (H,W,3)
    # expo_norm: Mittelwerte beider Fisheyes im Überlappband angleichen (Gain je Kamera).
```
Pano-Konvention: Spalte 0..W ⇒ Azimut −π..π (Längengrad), Zeile 0..H ⇒ Polar 0..π.
Ray im cam0-Frame ("imu"==cam0): x=sin(θ)sin(φ)... — exakt wie `refine_extrinsic.py::rays`
(dort nachlesen und übernehmen, damit kompatibel zur bestehenden Kalibrierung).

## core/colorizer.py

```python
@dataclass
class ColorizeParams:
    brightness_min: int = 20      # Graustufen-Schwelle (0..255): dunkler ⇒ Pixel ungültig
    brightness_max: int = 235     # heller ⇒ ungültig (überbelichtet)
    k_frames: int = 3             # bis zu K zeitnächste Kamera-Frames je Scan probieren
    max_dt: float = 0.08          # s; Frames weiter weg werden ignoriert
    min_range: float = 0.5        # m; Punkte näher an der Kamera nicht einfärben
    sky_clip: int = 250           # alle Kanäle ≥ ⇒ Pixel ausgebrannt (Himmel)
    sky_grow: int = 4             # px Saum um die ausgebrannte Fläche; 0 schaltet ab
    sky_prefer: bool = True       # himmelsartige Proben nur, wenn keine andere da ist
    sky_luma: int = 200           # ab hier gilt eine Probe als himmelsartig hell …
    sky_sat: float = 0.15         # … und muss zugleich so flau sein
    T_imu_cam0: np.ndarray = ...  # 4×4, Default aus extrinsic.json bzw. Identität

def colorize(rec: Recording, bag: BagReader, calib_json: str, params: ColorizeParams,
             out_dir: str, progress_cb=None, cancel=None) -> dict:
    # je Scan: Kandidaten-Frames nach |dt| sortiert (<=max_dt, max K);
    # p_cam = T_cam0_imu @ inv(T_world_imu(t_frame)) @ p_world   (Posen-Interpolation!)
    # cam0 projizieren, wo invalid → in cam1 (T_cam1_cam0) projizieren;
    # Bilinear sampeln (cv2.remap auf Punktlisten oder Gather), Grauwert prüfen
    # (brightness_min<=g<=brightness_max UND im Fisheye-Kreis, Radius<=740 px um Zentrum
    # UND ausserhalb des Himmelssaums, s.u.);
    # je Punkt gewinnt der MEDIAN aller gültigen Proben. colors.bin (RGB!), valid.bin,
    # meta.json schreiben.
    # return {"n_valid": ..., "frac_valid": ..., "n_sky_blocked": ..., "n_sky_outvoted": ...}

def _blown_mask(img_half, clip: int, grow: int) -> np.ndarray | None:
    # Pixel mit min(B,G,R) >= clip sind ausgebrannt, um `grow` px geweitet.
```

### Himmelssaum-Sperre

Der Himmel liefert keine Lidar-Punkte, jede Farbe von dort ist ein Fehlgriff. Reines
Weiß fängt schon `brightness_max` ab — das eigentliche Problem ist der **Saum**: an
der Silhouette mischen Unschärfe, JPEG-Ringing und der cyanfarbene Farbsaum Himmel
und Objekt zu Grauweiß, und die bilineare Abtastung zieht direkt an der Kante
zusätzlich Himmel mit hinein. Gemessen an Flug3 (Bäume gegen ausgebrannten Himmel,
19–26 % der Pano-Fläche voll geklippt):

| Abstand zur ausgebrannten Fläche | Median-Luma | davon unter `brightness_max` |
|---|---|---|
| 1 px | 252 | 6–12 % |
| 2 px | 248 | 24–26 % |
| 3 px | 226–241 | 44–59 % |
| 5 px | 187–222 | 58–74 % |
| 8–12 px | 124–201 | 65–81 % |
| fern | 128–190 | 79–90 % |

Deshalb zwei Mechanismen:

1. **`sky_grow`** weitet die geklippte Fläche um n px; Proben darin werden verworfen.
2. **`sky_prefer`** stellt himmelsartige Proben (hell **und** flau) zurück: existiert
   für einen Punkt mindestens eine nicht-himmelsartige Probe, bestimmen nur diese den
   Median. Gibt es ausschließlich himmelsartige (echte weiße Wand), bleibt alles gültig.

Der globale HSV-Filter der früheren WHS-Lösung (V ≥ 0,92 **und** S ≤ 0,12) ist hier
nicht übertragbar: bei Flug3 ist auch der Boden überbelichtet, der Filter würde
8–43 % der Pixel unterhalb des Horizonts mitnehmen.

### Wie genau die Extrinsik sein muss

Gemessen an Flug3 als mittlere Abweichung derselben Punkte zwischen den Frames
(Skala 0–255; das ist das, was die Einfärbung verdirbt):

| Auslenkung ab dem Optimum | alle Punkte | näher als 12 m |
|---|---|---|
| 0 | **10,36** | **12,85** |
| Rotation ±0,5° (je Achse) | 10,35–10,50 | 12,86–13,04 |
| Rotation ±2° | 10,42–11,18 | 13,01–13,76 |
| Hebelarm ±0,15 m | 10,46–10,75 | 13,11–13,54 |
| Hebelarm ±0,30 m | 10,59–11,14 | 13,47–14,20 |

Daraus folgt, was **nicht** zu bauen ist:

* **Kein Hebelarm in der Suche.** Das Optimum liegt in allen drei Achsen bei 0 —
  Kamera und Lidar sitzen praktisch aufeinander. Die WHS-Lösung schätzte 0,4 m mit,
  bei diesem Rig würde das nur Rauschen einbauen.
* **Kein Feinschliff unter 1°.** Zwei zusätzliche Stufen (0,3°/0,1°) heben den ZNCC
  um 0,001–0,005, bewegen die Ausrichtung um 0,3–0,5° und ändern die Farbstreuung um
  +0,3 % bzw. −0,1 % — sie jagen Rauschen. Zum Vergleich: Trainings- und
  Validierungssatz unterscheiden sich am selben Punkt um 0,026.
* **Kein Zeitversatz Kamera↔Lidar.** Das Optimum liegt bei −10 ms, Gewinn 0,5 %.
* **Kein Belichtungsausgleich zwischen Frames.** Die Frames schwanken um 1,5 %.

Was dagegen viel bringt: **eine schiefe gespeicherte Extrinsik überhaupt zu
bemerken**. Der absolute ZNCC ist zwischen Flügen nicht vergleichbar — Flug0 stand
mit 0,708 gespeichert da, während 0,835 möglich waren, und die Farbstreuung war
dadurch 11,96 statt 8,85 (**26 % schlechter**). Genau dafür ist `check_extrinsic`.

Die Schwellen sind gemessen, nicht geraten:

| Fall | Vorsprung | Abstand | Farbstreuung |
|---|---|---|---|
| 07-25 Flug3 | 0,000 | 0,0° | — |
| 09-08 12-53-52 | +0,007 | 1,0° | −2,3 % |
| 09-08 12-56-28 | +0,044 | 2,5° | −3,2 % |
| **07-17 Flug0** | **+0,143** | **15,7°** | **−26 %** |

Nur der letzte Fall rechtfertigt es, einen laufenden Einfärbe-Lauf abzubrechen.
Darunter wird die Güte protokolliert und nicht angemahnt — sonst warnt das
Programm für 3 %, und die Warnung wird wertlos.

```python
def overlay_preview(rec, bag, calib_json, T_imu_cam0, frame_idx: int,
                    stride=50) -> np.ndarray:
    # Pano-großes BGR-Bild: gestitchtes Pano (separater Stitcher) + projizierte
    # Lidar-Punkte (Tiefe→Turbo-Colormap) übergezeichnet → für Extrinsik-Justage.

def check_extrinsic(rec, bag, calib_json, T, frames=None, cancel=None) -> dict:
    # Kurzer Hillclimb (10/3/1°) ab T. Bleibt er stehen, sitzt T auf einem Gipfel
    # der Foto-Konsistenz; läuft er weg, ist die gespeicherte Extrinsik verdreht.
    # {"score", "best_score", "best_T", "dist_deg", "suspect"};
    # suspect = Vorsprung >= 0.08 UND Abstand >= 5°. Läuft vor jeder Einfärbung (3-6 s).

def auto_calibrate(rec, bag, calib_json, T_init=None, frames: list[int] = None,
                   progress_cb=None, cancel=None) -> tuple[np.ndarray, float]:
    # Grobe Rotationssuche (Yaw/Pitch/Roll-Gitter 10° → Verfeinerung 3°/1°, Translation 0):
    # Score = Korrelation Kantenbild(Pano) ↔ Kantenbild(projizierte Punktdichte/-tiefe)
    # über ~5 gleichverteilte Frames. Rückgabe (T_imu_cam0, score). Ehrlich bleiben:
    # score mitliefern, UI zeigt Warnung bei schwachem Score.
```

## core/georef.py

```python
@dataclass
class GpsQuality:
    usable: bool
    reasons: list[str]            # DEUTSCH, z.B. "fix_type=0 (kein GPS-Gerät erkannt)"
    n_total: int; n_good: int
    fix_type_hist: dict[int,int]; median_eph_cm: float | None
    median_sats: float | None; baseline_m: float
    per_fix_good: np.ndarray      # bool (n_total,)

def assess(fixes: list[GpsFix]) -> GpsQuality:
    # Kriterien je Fix: NavSatFix.status>=0, lat/lon != 0, fix_type>=3 (3D),
    # satellites>=6, eph_cm<=500, cov_east/north<=10 m (Sentinels 4294967.295 m
    # u. 9999 cm erkennen!). usable ⟺ n_good>=20 UND baseline_m>=5.

@dataclass
class GeorefResult:
    T_enu_world: np.ndarray       # 4×4: LIO-Welt → lokales ENU (Ursprung=erster guter Fix)
    origin_llh: tuple[float,float,float]
    utm_epsg: int; rms_m: float; n_used: int
    utm_offset: tuple[float,float]  # ENU-Ursprung in UTM

def align(rec: Recording, fixes, quality, progress_cb=None) -> GeorefResult:
    # 1) gute Fixe → ENU (WGS84→ECEF→ENU um ersten guten Fix; pyproj)
    # 2) LIO-Pose zu jedem Fix-Stempel interpolieren
    # 3) 4-DOF gewichtete Ausrichtung (Yaw+xyz; Gewichte 1/eph²): Horizontal-Umeyama
    #    ohne Skalierung + z-Offset  → T_enu_world
    # 4) RMS der Residuen; RuntimeError (deutsch) wenn RMS > max(3 m, 2×median_eph).

def export_las(points_xyz, colors_rgb, georef: GeorefResult | None, path: str): ...
def export_ply_pcd(points_xyz, colors_rgb, path: str): ...   # open3d, Endung entscheidet
```

## ui/cloud_view.py — `class CloudView(QWidget)`

VTK (QVTKRenderWindowInteractor). API (alles Slots-tauglich, Aufruf aus GUI-Thread):
```python
def set_cloud(self, points: np.ndarray, colors: np.ndarray | None = None,
              intensity: np.ndarray | None = None, valid: np.ndarray | None = None): ...
def set_path(self, positions: np.ndarray | None): ...   # Polyline der Trajektorie
# Optionen (Setter, von main_window-Sidebar bedient):
set_point_size(int 1..8); set_color_mode(str in {"rgb","hoehe","intensitaet","uniform"});
set_only_colored(bool)    # valid-Maske anwenden ("nur eingefärbte Punkte")
set_voxel_display(float)  # 0=aus, sonst Anzeige-Downsample in m (numpy-Grid-Hash)
set_background(str in {"dunkel","hell"}); set_eyedome(bool)  # EDL, Fallback ohne
def reset_camera(self); def screenshot(self, path: str)
```
vtkPolyData + vtkVertexGlyphFilter vermeiden bei 9 Mio Punkten — stattdessen
`vtkPolyData` mit direktem `vtkCellArray` aus arange (oder `vtkGlyph3DMapper`); Farben
als `vtkUnsignedCharArray` (RGB). Höhe: Turbo/Viridis-Colormap über z-Perzentile 2..98.
EDL: `vtkRenderStepsPass`+`vtkEDLShading` (try/except → Checkbox deaktivieren).
`if __name__=="__main__":` Selbsttest 2 Mio Zufallspunkte + screenshot.

## ui/pano_view.py — `class PanoView(QWidget)`

```python
class PanoSource:   # abstrakt, threadsicher
    count: int; stamps: np.ndarray; fps: float
    def get_pano(self, idx: int) -> np.ndarray: ...  # BGR

class StitchingPanoSource(PanoSource):
    # BagReader + EquirectStitcher + JPEG-Disk-Cache (pano_<W>/, quality 92) + RAM-LRU(8);
    def __init__(self, bag, calib_json, width, cache_dir): ...

class PanoView(QWidget):
    def set_source(self, src: PanoSource): ...
    frameChanged = pyqtSignal(int, float)   # (idx, stamp) — für Kopplung an 3D später
```
Player: Play/Pause (Space), Frame ±1 (←/→), Speed-Combo 0.25/0.5/1/2/4×, Zeit-Slider,
Label "Frame i/N — t=…s". Anzeige: QGraphicsView, Mausrad-Zoom 10 %–1600 % auf
Mauszeiger (AnchorUnderMouse), Drag-Pan (ScrollHandDrag), Doppelklick/Knopf "Einpassen",
Screenshot-Knopf. Dekodier-/Stitch-Arbeit in QThread-Prefetcher (aktuellen + nächste 4
Frames vorladen), UI nie blockieren; Drop-Frames wenn zu langsam (Stamps-basiert).
`if __name__=="__main__":` Selbsttest mit synthetischer Quelle.

## ui/gps_panel.py — `class GpsPanel(QWidget)`

Zeigt GpsQuality: Ampel (rot/gelb/grün), Kennzahlen-Tabelle (fix_type-Histogramm,
Satelliten, eph, Baseline), Gründe-Liste, Knopf "Georeferenzierung ausführen"
(nur aktiv wenn usable) → zeigt GeorefResult (EPSG, RMS, Ursprung) + Hinweistext.
Signal `georefReady = pyqtSignal(object)`.

## ui/main_window.py — `class MainWindow(QMainWindow)`

Layout: links Sidebar (QScrollArea, feste Breite ~360 px) mit Gruppen:
1. **Rosbag** — "Bag öffnen…" (Ordner-Dialog), Info-Tabelle (Topics/Dauer/Frames).
2. **Punktwolke (FAST-LIO2)** — Config-Combo (whs_dense.yaml=„Maximal dicht“ Default,
   mid360.yaml=„Schnell“), Rate-Spin (0.25–2.0, Default 1.0), Knopf "Karte berechnen",
   QProgressBar, danach Kennzahlen (Scans, Punkte, Drops).
3. **Einfärbung** — Helligkeit-min/max (2 Slider 0..255 mit Zahlenanzeige),
   K-Frames-Spin, Extrinsik: 6 DoubleSpin (Yaw/Pitch/Roll °, x/y/z m) + Knöpfe
   "Auto-Kalibrierung (grob)", "Overlay-Vorschau", "Einfärben".
4. **Anzeige** — Punktgröße, Farbmodus-Combo, "Nur eingefärbte Punkte",
   Anzeige-Voxel-Combo (Aus/0.05/0.10/0.20 m), Hintergrund, EDL, "Trajektorie zeigen".
5. **Export** — "PLY/PCD speichern…", "LAS speichern…" (mit Georef falls vorhanden).
Zentral: QTabWidget "3D-Karte" (CloudView) / "360°-Video" (PanoView) / "GPS" (GpsPanel)
/ "Protokoll" (QPlainTextEdit, alle log_cb-Zeilen).
Statusleiste: aktueller Schritt + unbestimmte Busy-Anzeige.
Worker: EIN generisches `class Worker(QThread)` mit fn/args, Signale
progress(float,str)/finished(object)/failed(str); pro Schritt ein Worker; Knöpfe
gegenseitig sperren solange busy; Abbrechen-Knopf setzt cancel-Event.
Beim Bag-Öffnen: project.py-Cache prüfen und Vorhandenes (Recording/Farben/Panos)
sofort laden. Einstellungen in cache/<bag>/settings.json persistieren.
Fenster: 1600×950, Titel "Super360 Studio". qdarktheme dark + Akzent #4FC3F7.

## core/project.py

```python
class Project:
    def __init__(self, bag_path: str, cache_root=DEFAULT_CACHE_ROOT): ...
    dir: str                       # cache/<bag_name>
    def recording_dir(self); def colors_dir(self); def pano_dir(self, width)
    def gps_json(self); def extrinsic_json(self); def settings_json(self)
    def has_recording(self) -> bool; ...
    def load_settings(self) -> dict; def save_settings(self, d: dict)
    def load_extrinsic(self) -> np.ndarray | None; def save_extrinsic(self, T)
```

## Teststrategie

Jedes Modul hat `if __name__ == "__main__":`-Selbsttest (synthetisch oder gegen seg0)
und legt Beweis-PNGs/Ausgaben unter
`/tmp/super360_modtests/<modul>/` ab.
UI-Selbsttests: `QT_QPA_PLATFORM=offscreen` wenn möglich (VTK braucht evtl. :0 — dann :0
nehmen und Fenster sofort wieder schließen).

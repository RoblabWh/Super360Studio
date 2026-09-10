# Super360 Studio

Bis 2026-09 hieß das Programm „RosBag Suite 360".

PyQt5-GUI für die Auswertung der Drohnen-Rosbags (ROS2 Humble, Livox Mid-360 +
Dual-Fisheye-360°-Kamera "paycam" + mavros-GPS):

1. **Punktwolke (FAST-LIO2)** — berechnet aus dem Bag die dichteste Karte
   (`whs_dense.yaml`: alle gültigen Punkte, volle Scans via
   `/cloud_registered_body` + `/Odometry`) und zeigt sie im 3D-Viewer (VTK).
2. **360°-Video** — stitcht die Dual-Fisheye-Frames Docker-frei (OpenCV-Remap,
   Double-Sphere-Kalibrierung aus `Super360_Stitcher_rosbag`) und spielt sie ab
   (Zoom aufs Mausrad, Pan, Frame-genaues Springen, 180°-Drehung).
3. **Einfärbung** — färbt die Punktwolke aus den 360°-Bildern ein; zu dunkle /
   zu helle Pixel (Sliders „Helligkeit min/max") werden ausgefiltert, nicht
   eingefärbte Punkte lassen sich ausblenden („Nur eingefärbte Punkte").
   Kamera-Extrinsik: grobe Auto-Kalibrierung + manuelle Feinjustage
   (Yaw/Pitch/Roll/x/y/z) mit Overlay-Vorschau.
4. **GPS** — prüft IMMER die Signalqualität (fix_type, Satelliten, eph/HDOP,
   Kovarianz, Bewegungs-Baseline; Sentinel-Werte werden erkannt). Nur bei
   brauchbarem Signal wird die LIO-Trajektorie per gewichteter
   4-DOF-Ausrichtung (Yaw + Translation, Gewichte 1/eph²) auf ENU/UTM
   georeferenziert; Residuen (RMS) werden ausgewiesen.
5. **Export** — PLY/PCD (lokal) und LAS (georeferenziert in UTM, falls GPS
   brauchbar; exakte pyproj-Projektion).

## Start

```bash
./run_gui.sh          # oder: python3 app.py
```

Kein ROS-Sourcing nötig — die GUI liest Bags über die `rosbags`-Bibliothek;
nur der FAST-LIO-Schritt startet intern Subprozesse mit ROS-Umgebung
(`/opt/ros/humble`, `~/ws_livox`, `~/fastlio2_ws`).

## Bedienung (Pipeline in der Sidebar)

1. **Bag öffnen…** — Bag-Ordner wählen (z. B.
   `rosbag_2026-07-11_15-37-07_seg0`). Vorhandene Cache-Artefakte
   (Karte/Farben/Panoramen) werden automatisch geladen.
2. **Karte berechnen** — FAST-LIO2-Lauf (Dauer ≈ Bag-Länge / Abspielrate).
   Es darf nur eine Instanz laufen (Instanz-Sperre); alte FAST-LIO-Prozesse
   werden vorher beendet.
3. **Einfärben** — erst ggf. „Auto-Kalibrierung (grob)", Overlay-Vorschau
   prüfen, dann „Einfärben".
4. **GPS-Tab** — Ampel + Gründe; Georeferenzierung nur bei grün/gelb möglich.
5. **Export** — berücksichtigt „Nur eingefärbte Punkte".

## Kalibrierung

Die Double-Sphere-Kalibrierungen liegen als Kopie in `calib/`
(`calib_result_new2.json`, `calib_new_vign.json`, `calib_new_refined.json`),
damit das Repo ohne das Stitcher-Projekt lauffähig ist. Gefunden wird die
erste existierende Datei der Kandidatenliste; danach folgen die Originale in
`Super360_Stitcher_rosbag/work/` als Fallback.

## Cache

Der Cache liegt **außerhalb** des Repos, per Default unter
`/home/lena/RosBagSuper_Gui/rosbag_suite/cache`. Ein anderer Ort geht über
die Umgebungsvariable `SUPER360_CACHE_ROOT`.

`<cache_root>/<bagname>-<pfad-hash>/` enthält `recording/` (Punkte/Posen),
`colors/`, `pano_<Breite>/` (gestitchte JPEGs), `extrinsic.json`,
`settings.json`. Löschen ist jederzeit erlaubt (wird neu berechnet).

Ein fehlgeschlagener/abgebrochener FAST-LIO-Lauf lässt eine vorhandene
Aufzeichnung unangetastet (Schreiben in `recording.tmp`, Promotion nur bei
Erfolg).

## Bekannte Grenzen

- GPS in den Juli-Bags ist tot (fix_type=0, 0 Satelliten) — die GUI zeigt das
  als „GPS unbrauchbar" mit Gründen; Georeferenzierung ist dann deaktiviert.
- Kamera↔Lidar-Extrinsik ist nicht werksseitig kalibriert; die
  Auto-Kalibrierung ist grob (Rotation, ±wenige Grad) — Feinjustage über die
  Spinboxen, Ergebnis wird pro Bag gespeichert.
- Colorization hat keine Occlusion-Behandlung; bewegte Objekte können
  Farbschlieren bekommen.
- Der obere Polbereich (~5 %) des Panos ist physikalisch von keiner Linse
  abgedeckt (Sensor beschneidet die Fisheye-Kreise).

## Architektur

Siehe `ARCHITECTURE.md` (Module, Datenformate, Konventionen). Kern Qt-frei
(`core/`), UI in `ui/`, FAST-LIO-Recorder in `scripts/record_fastlio.py`.

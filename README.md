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
4. **Zusammenführen** — zweiten Flug dazuladen und beide Karten zu einer machen.
5. **GPS-Tab** — Ampel + Gründe; Georeferenzierung nur bei grün/gelb möglich.
6. **Export** — berücksichtigt „Nur eingefärbte Punkte".

## Höhenschnitt

Rechts am Viewer liegt eine Leiste mit zwei Griffen, die die sichtbare
Höhenschicht aufspannen — damit lässt sich das Dach abnehmen und in ein Gebäude
hineinschauen. Die Höhen stehen in Metern an den Griffen.

| Eingabe | Wirkung |
|---|---|
| Griff ziehen | obere oder untere Schnittebene setzen |
| auf die Leiste klicken | den näher liegenden Griff dorthin holen |
| Mausrad über der Leiste | die ganze Schicht nach oben oder unten schieben |
| Doppelklick oder „alles" | Schnitt aufheben |

Geschnitten wird über Clipping-Ebenen auf der Grafikkarte, die Geometrie bleibt
also unberührt und das Ziehen ist auch bei Millionen Punkten flüssig. Die
Trajektorie hängt an einem eigenen Mapper und bleibt sichtbar.

## Zusammenführen

Jeder Flug bekommt von FAST-LIO ein eigenes Weltsystem, verankert im ersten
Scan. Zwei Karten liegen darum beliebig zueinander, auch wenn sie dasselbe
Gebäude zeigen. Der Ablauf in der Sidebar:

1. **Zweiten Flug wählen** — dessen Karte muss berechnet sein; ist sie es nicht,
   wird mit der zu erwartenden Dauer gefragt und FAST-LIO läuft direkt hier.
   Die zweite Wolke erscheint orange im Viewer.
2. **Auto-Ausrichten** — globale Suche (FGR über FPFH) plus ICP von grob nach
   fein. Dauert Sekunden bis Minuten. **Nur ICP** verfeinert stattdessen die
   aktuelle Lage, was nach einer Handjustage reicht.
3. **X/Y/Z/Gier** schieben und drehen die zweite Wolke von Hand, mit sofortiger
   Vorschau. Gedreht wird um ihren eigenen Schwerpunkt.
4. **Übernehmen** schreibt eine gemeinsame Aufzeichnung und öffnet sie als
   Arbeitswolke. Sie lässt sich danach als Ganzes einfärben und exportieren;
   jeder Abschnitt wird mit der Kamera seines eigenen Bags eingefärbt.

Nach dem Ausrichten stehen Trefferquote und Restfehler im Protokoll. Unter 0,3
Trefferquote überlappen die Wolken zu wenig — dann von Hand grob zusammenschieben
und „Nur ICP" nachlaufen lassen. Ein Restfehler über 0,3 m heißt: die Lage stimmt
grob, sitzt aber nicht sauber.

Grenzen: das 360°-Video und die GPS-Prüfung zeigen weiter den zeitlich ersten
Flug. Zeitlich überlappende Aufnahmen werden abgelehnt, weil die Scan-Reihenfolge
dann nicht mehr eindeutig wäre.

## Kippkorrektur

FAST-LIO richtet sein Weltsystem an der Sensorlage des ersten Scans aus, nicht an
der Schwerkraft. Steht der Livox schräg auf der Drohne, kippt die ganze Karte mit.
Seit dem 08.09.2026 ist er rund 40° gekippt montiert (gemessen 40,1° / 39,7° /
41,5°; davor durchgehend 0,3°–5,6°).

Die GUI misst die Lotrechte deshalb aus dem IMU-Ruhefenster am Bag-Anfang und
dreht die Karte beim Laden gerade — aber erst ab 10° Schräglage, damit die sauber
montierten Flüge unverändert bleiben. Im Protokoll steht dann eine Zeile wie
„Karte lotrecht gedreht: Livox war 40.1° schräg montiert". Gedreht wird nur das
Weltsystem, Einfärbung und Farb-Cache sind davon nicht betroffen.

## Kalibrierung

Die Double-Sphere-Kalibrierungen liegen als Kopie in `calib/`
(`calib_result_new2.json`, `calib_new_vign.json`, `calib_new_refined.json`),
damit das Repo ohne das Stitcher-Projekt lauffähig ist. Gefunden wird die
erste existierende Datei der Kandidatenliste; danach folgen die Originale in
`Super360_Stitcher_rosbag/work/` als Fallback.

## Extrinsik prüfen

Vor jeder Einfärbung wird die gespeicherte Kamera-Extrinsik geprüft (3–6 s): ein
kurzer Hillclimb startet dort und schaut, ob sie auf einem Gipfel der
Foto-Konsistenz sitzt. Die Güte steht danach im Protokoll, zusammen mit dem besten
erreichbaren Wert und dem Abstand dorthin.

Das ist nötig, weil der absolute Wert zwischen Flügen nicht vergleichbar ist: bei
`rosbag_2026-07-17_13-16-18_Flug0` stand 0,71 gespeichert und sah unauffällig aus,
während 0,84 möglich waren — die Extrinsik war 15,7° verdreht und kostete 26 %
Farbqualität. Erst ab 5° Abstand und 0,08 Vorsprung wird gewarnt; darunter lohnt
der Abbruch nicht (bei 2,5° sind es 3 %).

Genauer als das braucht die Kalibrierung nicht zu sein. Unterhalb von etwa 0,5°
ändert eine Verschiebung die Farbqualität nur noch um Bruchteile eines Prozents,
und der Hebelarm zwischen Kamera und Lidar ist bei diesem Aufbau messbar null.

## Einfärbung und der weiße Himmel

Je Scan werden bis zu `K` zeitnächste Kamera-Frames plus zwei Konsens-Frames aus
±0,8 s gesampelt; je Punkt gewinnt der Farb-Median. Dazu kommen zwei Filter gegen
den Himmel, denn der Himmel liefert keine Lidar-Punkte — jede Farbe von dort ist
ein Fehlgriff:

* **Himmelssaum** (Standard 4 px) sperrt den Rand um ausgebrannte Flächen. Reines
  Weiß fängt „Helligkeit max" schon ab; was Baumkronen weiß überzieht, ist der Saum
  daneben, wo Unschärfe und Farbsaum Himmel und Blattwerk zu Grauweiß mischen. Der
  Wert steht in der Sidebar, 0 schaltet die Sperre ab.
* **Vorrang für echte Oberflächen**: hat ein Punkt neben hellen, flauen Proben auch
  eine normale, bestimmen nur die normalen den Median. Punkte, für die es nur helle
  flaue Proben gibt (weiße Wand), bleiben unberührt.

Was das nicht repariert: die 360°-Kamera belichtet im Gegenlicht die ganze Szene
über, nicht nur den Himmel. Kronen bleiben dadurch blasser als in Wirklichkeit.

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
- Beim Zusammenführen wird nur eine starre Transformation gesucht. Driftet eine
  der beiden LIO-Karten in sich, lässt sich das damit nicht geradebiegen.

## Architektur

Siehe `ARCHITECTURE.md` (Module, Datenformate, Konventionen). Kern Qt-frei
(`core/`), UI in `ui/`, FAST-LIO-Recorder in `scripts/record_fastlio.py`.

# Super360 Studio

![Punktwolke des DRZ-Geländes in fünf Ansichten, nacheinander überblendet](assets/modelle.gif)

## Vor dem ersten Start

Getestet auf **Ubuntu 22.04** mit dem System-Python **3.10** und einem
Grafiktreiber mit OpenGL ≥ 3.2 (die 3D-Ansicht ist VTK). Was davon da ist,
prüft ein Skript und sagt zu jedem fehlenden Teil den Befehl:

```bash
python3 scripts/pruefe_installation.py
```

Was man braucht, hängt davon ab, was man machen will:

| Teil | wofür | Pflicht? |
|---|---|---|
| PyQt5, VTK (apt) + Python-Pakete (pip) | App starten, Projekte öffnen, ansehen, messen, exportieren | **ja** |
| PointCloudMerger (Nachbar-Repo) | Mäander-Einfärbung aus dem DJI-Flug | für Mäander |
| Interpreter mit `pycolmap` | neue COLMAP-Rekonstruktion eines Mäanderfluges | nur ohne fertiges Modell |
| ROS 2 Humble + Livox-Treiber + FAST_LIO_ROS2 | **Karte berechnen** aus einem Rosbag | nur dafür |
| EPIC_ros2 | RViz-Wiedergabe (`traj_utils`/`quadrotor_msgs`) | nur dafür |

Ein fertiges Projekt (z. B. ein exportierter Ordner) lässt sich also mit den
Pflichtteilen allein öffnen, ansehen, messen, einfärben aus der Mäander-Ebene
und exportieren — ROS braucht es nur zum Berechnen der Karte.

### 1. Pflicht: GUI und Python-Pakete

```bash
sudo apt install python3-pip python3-pyqt5 python3-pyqt5.qtopengl python3-vtk9
pip install --user -r requirements.txt
```

PyQt5 und VTK bewusst aus **apt** (getestet: PyQt5 5.15.6, VTK 9.1.0) — die
pip-Wheels von VTK bringen eigene Qt-Bindungen mit und vertragen sich damit
nicht zuverlässig. `requirements.txt` enthält den Rest (numpy, scipy,
opencv-python, open3d, rosbags 0.10, Pillow, laspy, pyproj, pyqtdarktheme)
mit den getesteten Versionen als Kommentar. Fehlt `pyqtdarktheme`, startet die
App im Standard-Look.

### 2. Mäander-Einfärbung

```bash
git clone git@github.com:LenaKremer98/PointCloudMerger.git ~/PointCloudMerger
sudo apt install libimage-exiftool-perl        # optional: Geotags schneller
```

Gesucht wird unter `~/PointCloudMerger` und neben diesem Repo. Temperaturen aus
den Thermalbildern brauchen **kein** DJI-SDK (s. „Temperaturen“).

Liegt im Projekt noch kein COLMAP-Modell des Fluges, rechnet die App eins — dafür
braucht es einen Interpreter mit `pycolmap`, getrennt vom System-Python:

```bash
python3 -m venv ~/.venvs/colmap
~/.venvs/colmap/bin/pip install pycolmap        # getestet: 4.0.4
```

### 3. Karte berechnen (FAST-LIO2)

Nur dafür braucht es ROS 2 Humble. Die App erwartet die Workspaces unter
`~/ws_livox` und `~/fastlio2_ws` und sourct sie selbst — vor dem Start der App
muss nichts gesourct werden. Kurzfassung (Details in den READMEs der Repos):

```bash
# ROS 2 Humble (https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
sudo apt install ros-humble-desktop ros-humble-pcl-ros libpcl-dev libeigen3-dev \
                 python3-colcon-common-extensions

# Livox-SDK2
git clone https://github.com/Livox-SDK/Livox-SDK2.git ~/Livox-SDK2
cd ~/Livox-SDK2 && mkdir -p build && cd build && cmake .. && make -j && sudo make install

# livox_ros_driver2 (liefert den Nachrichtentyp CustomMsg der Mid-360)
mkdir -p ~/ws_livox/src
git clone https://github.com/Livox-SDK/livox_ros_driver2.git ~/ws_livox/src/livox_ros_driver2
cd ~/ws_livox/src/livox_ros_driver2 && source /opt/ros/humble/setup.bash && ./build.sh humble

# FAST_LIO_ROS2 mit der Konfiguration aus diesem Repo
mkdir -p ~/fastlio2_ws/src
git clone --recursive https://github.com/Ericsii/FAST_LIO_ROS2.git ~/fastlio2_ws/src/FAST_LIO_ROS2
cp config/fastlio/whs_dense.yaml ~/fastlio2_ws/src/FAST_LIO_ROS2/config/
cd ~/fastlio2_ws && source /opt/ros/humble/setup.bash \
    && source ~/ws_livox/install/setup.bash && colcon build
```

**`whs_dense.yaml` gehört nicht zu FAST_LIO_ROS2** — es ist die eigene
Konfiguration für die Super-Drohne (jeder Rohpunkt, volle Scans, Lidar-IMU-
Extrinsik der Mid-360) und liegt deshalb unter `config/fastlio/` in diesem Repo.
Ohne sie schlägt „Karte berechnen“ fehl. Nach einer Änderung neu bauen, der
Workspace ist ohne `--symlink-install` gebaut.

Optional für die RViz-Wiedergabe: [EPIC](https://github.com/Robotics-STAR-Lab/EPIC)
als `~/EPIC_ros2` bauen — ohne dessen Nachrichtentypen bricht `ros2 bag play`
bei den Drohnen-Bags ab.

### 4. Tests (optional)

```bash
sudo apt install xvfb
xvfb-run -a python3 scripts/test_cloud_view_steuerung.py
python3 -m core.meander        # jedes Modul in core/ und ui/ hat einen Selbsttest
```

## Start

```bash
./run_gui.sh          # oder: python3 app.py
```

Kein ROS-Sourcing nötig — die GUI liest Bags über die `rosbags`-Bibliothek;
nur der FAST-LIO-Schritt startet intern Subprozesse mit ROS-Umgebung
(`/opt/ros/humble`, `~/ws_livox`, `~/fastlio2_ws`).

Projekte liegen im Cache unter `~/RosBagSuper_Gui/rosbag_suite/cache`, umzuhängen
mit der Umgebungsvariable `SUPER360_CACHE_ROOT`.

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

## 3D-Ansicht: Maus und Leiste

Die Maus steuert die Wolke **genau wie der VS-Code-Punktwolken-Viewer**
(`PointCloudMerger/vscode-pointcloud-viewer`):

| Eingabe | Wirkung |
|---|---|
| links ziehen | um den Zielpunkt drehen (Z oben, 0,006 rad je Pixel) |
| rechts ziehen, oder Umschalt/Strg und ziehen | in der Bildebene verschieben |
| Mausrad | zoomen (Abstand × exp(0,0012 · Δ), eine Raste ≈ 12 %) |
| `R` oder „Ansicht zurücksetzen“ | auf den Schwerpunkt, schräg von oben |
| `M` oder „Messen“ | Messen ein/aus; ein Klick ist ein Klick, solange die Maus unter 5 px bleibt — gedreht wird auch beim Messen |
| `Esc` | Messung verwerfen |

Oben liegt die **Leiste** wie dort:

* **Farbe** — Einheitsfarbe, Intensität, Höhe, RGB Onboard, RGB Mäander,
  Thermal Mäander. Farbmodus und Farbquelle in einem; was das Projekt nicht hat,
  ist ausgegraut. Seitenleiste und Menü ziehen mit.
* **Punkte** — 0,5 bis 10 px in Viertelschritten. Gebrochene Größen wirken
  wirklich (bei 1 / 1,5 / 2 px gemessen 18,6 / 22,2 / 23,4 % bedeckte Pixel).
* **Messen** und **Ansicht zurücksetzen**.
* **Temperatur anzeigen** (Standard an) — über einem Punkt zeigt die Maus
  seine Temperatur, **in jedem Farbmodus**, sobald es eine Thermal-Mäander-Ebene
  mit Temperaturen gibt.

### Temperaturen

Die Thermalbilder der M30T sind radiometrische JPEGs: neben dem Palettenbild
tragen sie die Rohwerte des Sensors (APP3, 640 × 512 × 16 Bit) und die
kameraeigene Umrechnungstabelle Rohwert → Zehntelgrad (APP5). `core/temperatur.py`
liest beides, ohne DJI-SDK. Beim Einfärben bekommt jeder Punkt die Temperatur aus
demselben Bild und Pixel wie seine Farbe; sie liegt als
`colors_meander_thermal/temperatur.bin` neben der Ebene. Am DRZ-Flug: 64 % der
Punkte, 21,9 bis 37,3 °C (1.–99. Perzentil). Es ist die Temperatur, die auch
die DJI-App zeigt — mit dem Emissionsgrad, der in der Kamera eingestellt war.

Ältere Thermal-Ebenen haben noch keine Temperaturen; einmal neu einfärben.

## Messen

`M`, **Werkzeuge → Messen** oder **Messen** in der Leiste schaltet um. Zwei Klicks in die Wolke setzen die
Marken, dazwischen liegt eine Linie mit dem Abstand in Metern. Unter der
3D-Ansicht stehen beide Punkte in Originalkoordinaten, dazu Abstand, waagerechter
Anteil, Höhenunterschied und die Differenz je Achse. Ein dritter Klick fängt neu
an.

Genommen wird der Punkt, welcher dem Klick am nächsten liegt und dabei der
Kamera am nächsten steht — sonst greift man durch eine Wand hindurch. Gesucht
wird nur unter den **sichtbaren** Punkten: ein aktiver Höhenschnitt schließt
alles Weggeschnittene aus.

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

## Mäander-Einfärbung

Zweite Art, die Wolke einzufärben: mit den Nadirbildern eines DJI-Kartierungs-
fluges statt aus der 360°-Kamera an Bord. Der Weg dahinter kommt aus dem Repo
[PointCloudMerger](https://github.com/LenaKremer98/PointCloudMerger) und läuft
hier ohne Eingriff durch:

```
Bilder ──► COLMAP ──► Kameraposen im willkürlichen Rahmen
                          │
   RTK-Geotags ───────────┤ Umeyama          → metrisch in ENU
                          ▼
   LiDAR-Karte ───────────┤ Drehung um die Hochachse + Verschiebung
                          ▼
   jeder Punkt ──► in das nadirnächste Bild ──► RGB
```

Zwischen ENU und der Karte bleibt nur eine Drehung um die Hochachse, weil beide
Rahmen lotrecht sind — das ENU per Definition, die Karte seit der Kippkorrektur
über die IMU. Vier Freiheitsgrade statt sieben, und die lassen sich suchen: grob
per Kreuzkorrelation über den Gierwinkel, fein über den Höhenunterschied zum
Rastermodell. Kein ICP über sechs Freiheitsgrade — das verkippt an Gebäudekanten
und zerstört die Lotrechte, die man geschenkt bekommt.

Ablauf: **Mäanderflug wählen** (Ordner mit den `_V.JPG`), **Ausrichten**,
Ergebnis im Viewer prüfen, dann **Einfärben**. Ist noch kein COLMAP-Modell da,
wird vorher gefragt — bei 255 Bildern dauert die Rekonstruktion etwa eine halbe
Stunde und liegt danach im Arbeitsordner des Projekts.

**Auf die Gütezahl achten, nicht auf die Trefferquote.** Nach dem Ausrichten
steht im Protokoll, wie viele Fotopunkte auf der Oberfläche der Wolke liegen.
Bei einer Nadirbefliegung gehören sie dorthin, stimmt der Winkel sind es 70 %
und mehr. Liegt der Wert unter 40 %, sitzt die Ausrichtung falsch und das
Einfärben bricht mit einer Meldung ab, statt Minuten in eine verdrehte Lage zu
stecken. Die Trefferquote beim Einfärben taugt dafür nicht: sie liegt auch bei
einer um 60° verdrehten Lage bei 99,8 %, weil fast jeder Punkt in *irgendein*
Bild fällt.

Die Bewertung der Kandidaten liegt eng beieinander — in einem gemessenen Fall
gewann 154,53° mit 0,291 gegen die richtigen 92,35° mit 0,289. Wenn das
passiert: den Gierwinkel von Hand auf den Wert der Grobsuche ziehen (der steht
mit im Protokoll) und erneut ausrichten.

**Thermal** braucht keine zweite Rekonstruktion: beide Optiken sitzen auf
derselben Gimbal und lösen zusammen aus, nur die Brennweite ist eine andere.

**Die Handjustage wirkt erst nach dem Ausrichten.** Die Regler sind ein Zuschlag
auf die gefundene Lage — solange es keine gibt, sind sie ausgegraut, und unter
ihnen steht, was fehlt. Reihenfolge also: Flug wählen → **Ausrichten** →
justieren → **Einfärben**.

**Die Handjustage wirkt live.** Nach dem Ausrichten lädt das Programm die
Bilder einmal stark verkleinert in den Speicher (255 Stück in gut drei
Sekunden, rund 40 MB) und färbt damit eine Stichprobe von 50.000 Punkten. Jeder
Zug an Gier, X oder Y färbt diese Stichprobe neu ein und legt sie über die
Karte — ein Durchlauf dauert etwa 90 ms, die Wolke folgt dem Regler also ohne
Verzögerung. In der Statuszeile stehen dabei Winkel, Versatz und wie viel
Prozent der Stichprobe getroffen wurde. Beim Einfärben wird die eingestellte
Lage übernommen und die Vorschau geräumt.

**Während der Justage ist die volle Karte ausgeblendet** und nur die Stichprobe
zu sehen. Das ist Absicht: 50.000 Punkte sind bei einer Karte aus 24 Millionen
zwei Promille — als Staub darüber gestreut wäre von einer Farbänderung nichts zu
erkennen. Allein gezeigt ist die Stichprobe die ganze Ansicht, und jeder
Reglerzug ist sofort sichtbar.

Der Haken **„Während der Justage nur die Vorschau zeigen"** schaltet das ab;
dann liegen beide übereinander.

### Überlagern und justieren (eigenes Fenster)

**„Überlagern und justieren …"** öffnet ein eigenes Fenster, in dem beide Wolken
übereinander liegen — so wie in der Pipeline. Das Hauptfenster bleibt dabei
unberührt.

| Ansicht | was zu sehen ist |
|---|---|
| **Überlagerung** | Karte in Grau (hell = hoch), die Fotopunkte des Fluges in Magenta, mit eigener Thermallage zusätzlich in Orange |
| **Farbvorschau RGB** | die Karte, eingefärbt aus den RGB-Bildern mit der aktuellen Lage |
| **Farbvorschau Thermal** | dasselbe aus den Thermalbildern, mit der Thermallage |

**Navigation mit der Maus:** Mausrad zoomt zum Mauszeiger hin, links ziehen
dreht, rechts oder mittel ziehen (oder Umschalt + links) verschiebt,
Doppelklick passt die ganze Karte ein. Dazu Blickrichtungen (oben, vorn,
Seite, schräg), eine Punktgröße und oben links ein Maßstabsbalken in Metern.
Beim Justieren bleibt die Ansicht stehen — wo man hingezoomt hat, sieht man
die Wirkung des Reglers.

Gier, X und Y gibt es **zweimal**: eine Zeile für RGB, eine für Thermal. Alle
wirken sofort. Unter dem Bild stehen Winkel, Versatz und — je nach Ansicht — der
Abstand der Schwerpunkte oder die Trefferquote. Wer an einem Thermalregler
dreht, bekommt die Thermal-Farbvorschau. **Lage übernehmen** schreibt beide
Lagen zurück ins Hauptfenster, **zurücksetzen** stellt die Lage beim Öffnen
wieder her. Ein Handzuschlag aus dem Hauptfenster wird beim Öffnen verrechnet,
das Fenster setzt auf der Lage auf, die man gerade sieht.

Gezeichnet wird als **Bild**, nicht mit einem zweiten 3D-Fenster. Zwei
OpenGL-Kontexte in einer Anwendung sind je nach Grafiktreiber und Sitzung eine
Quelle schwarzer Fenster. Das Bild rechnet numpy mit Tiefenpuffer, ein Bild aus
400.000 Punkten kostet einige zehn Millisekunden — genug, damit Drehen und
Zoomen der Maus folgen, und zuverlässig, geprüft sogar ganz ohne X-Server.

### Optik einmessen

Die Bilder eines Mäanderfluges passen erst zueinander, wenn die Optik stimmt.
Am DRZ-Flug (DJI M30T, 57 m) stimmte sie nicht:

* **RGB-Brennweite ~11 % zu kurz.** COLMAP hält die Optik bewusst fest (sonst
  wölbt sich die Rekonstruktion zur Kuppel) — aber mit dem Nennwert aus dem
  EXIF. Jedes Bild landet dadurch zu klein auf der Karte, am Rand um Meter und
  in jedem Bild in eine andere Richtung: die Bilder wirken **zueinander
  verzerrt**. Die Fotopunkte schweben aus demselben Grund knapp 4 m über der
  Lidar-Oberfläche.
* **Thermal schielt, ist verzeichnet und hat eine andere Brennweite.** Die
  Thermalkamera ist gegen die RGB-Kamera um rund 2° verdreht (2 m am Boden),
  ihr Objektiv hat eine kräftige Tonnenverzeichnung, und die Brennweite weicht
  vom EXIF ab. Bisher wurde mit der RGB-Pose, EXIF-Brennweite und ohne
  Verzeichnung gerechnet.

**„Optik einmessen“** (läuft nach dem ersten Ausrichten von selbst, rund eine
Minute) misst das nacheinander ein, siehe `core/optik.py`:

1. **Höhe** über den Laser-Entfernungsmesser der Drohne — jedes Bild trägt im
   XMP den Abstand zum Boden. Die Höhe muss zuerst feststehen: auf ebenem Boden
   lassen sich Brennweite und Flughöhe gegeneinander tauschen.
2. **RGB-Brennweite** über die Farbkonsistenz: jeder Kartenpunkt wird in alle
   Bilder projiziert, die ihn sehen; stimmt die Brennweite, widersprechen sich
   die Bilder am wenigsten.
3. **Thermaloptik** gegen das RGB-Bild desselben Auslösers — Brennweite,
   Hauptpunkt, Verzeichnung und Schielwinkel, über die Transinformation der
   Grauwerte. Geprüft an Bildpaaren, die nicht zum Einmessen dienten.

Das Ergebnis liegt in `meander/optik_kalibrierung.json`, die Maßstab-Regler
zeigen es, Einfärben und Vorschau benutzen es. ODM/WebODM wurde erwogen und
verworfen: die M30T ist dort nicht unterstützt, eine gekoppelte Verarbeitung von
Weitwinkel und Thermal gibt es nicht, und SfM auf reinen Thermalbildern scheitert
an texturarmen Dächern.

### Automatisch bis zur Farbe

**„Automatisch: ausrichten bis zur Farbe“** macht alles hintereinander, rund
sechs Minuten: Ausrichten → Optik einmessen → Feinausrichten → Einfärben mit
Sichtprüfung. Jeder Schritt gibt es auch als eigenen Knopf.

**Feinausrichten** (`core/optik.py`, `feinausrichten`) legt das Fotomodell auf
die Karte — mit den Maßen, die den jeweiligen Freiheitsgrad wirklich festlegen:

| Freiheitsgrad | woran gemessen | DRZ-Flug |
|---|---|---|
| Neigung, Höhe | Fotopunkte auf der Lidar-Oberfläche (Punkt-zu-Ebene, robust) | −0,04°/−0,18° |
| Versatz in der Ebene | Kantenmaß (Farbkanten gegen Höhenkanten), zwei Stichproben gemittelt | +0,25/+0,45 m |
| Maßstab | Brennweite aus „Optik einmessen“ | 1,086 |
| Gier | bleibt aus dem Ausrichten | — |

Gier und Blockmaßstab werden bewusst **nicht** automatisch gesucht: gemessen
ist das Kantenmaß über ±0,5° Gier flach (zwei Stichproben fanden +0,15° und
+0,45°), und ein Blockmaßstab tauscht gegen Brennweite und Höhe — er zog die
Fotopunkte von der Oberfläche (50 % → 34 %). Übernommen wird eine Korrektur nur,
wenn die Farbkonsistenz der Bilder dabei nicht schlechter wird. Eine neue
Ausrichtung verwirft Feinausrichtung und Höhe und misst sie neu; Brennweite und
Thermaloptik bleiben, die gehören zur Kamera.

### Sichtprüfung beim Einfärben

Bisher bekam jeder Punkt die Farbe aus dem Bild, in dem er am nächsten zur
Bildmitte lag. Von einer Wand sieht dieses Bild aber nichts — davor liegt das
Dach. Das Dachmuster lief die Wände hinunter.

Mit **„Beim Einfärben Sichtbarkeit prüfen“** (Standard, `core/sichtbar.py`):

1. Je Kamera wird aus der Karte eine Tiefenkarte gerechnet; ein Punkt bekommt
   nur Farbe aus Bildern, in denen er nicht verdeckt ist.
2. Unter diesen gewinnt das Bild, das am frontalsten auf seine Fläche blickt
   (Normale aus der Karte) — für Dächer die Kamera darüber, für Wände eine von
   der Seite.

Was keine Kamera sieht, bleibt **ungefärbt**: Unterholz, das Innere einer
Halle, das der Lidar durch Tor und Oberlichter gescannt hat, eine zweite
Dachschicht. Am DRZ-Flug rund 30 % der Punkte; von oben fehlt dadurch nichts
(0,2 %). Mit „Nur eingefärbte Punkte“ sind sie ausgeblendet. Ein Auffüllen vom
Nachbarn wurde ausprobiert und verworfen — es verteilte die Farbe weniger
zufällig sichtbarer Punkte zu Klecksen. Die Einfärbung dauert etwa dreimal so
lang; Live-Vorschau und Ausrichtfenster rechnen weiter ohne Sichtprüfung.

### Schieber statt Zahlenfelder

Jeder Versatz — Lage des Mäanderfluges (RGB und Thermal), Maßstab, Hauptpunkt
der Optiken, Zusammenführen, Extrinsik — hat **zwei Schieber**: einen groben für
den Weg und einen feinen für das letzte Stück (Meter bis auf den Zentimeter,
Winkel bis 0,005°, Maßstab bis 0,01 %). Der Wert ist die Summe; das Feld daneben
zeigt sie und nimmt einen getippten Wert an, „0“ setzt zurück.

### Eigene Lage für Thermal

Die Thermalbilder haben eigene Regler für Gier, X und Y — im Hauptfenster unter
**Lage von Hand** in der Zeile **Thermal**, im Ausrichtfenster ebenso. Sie sind
ein **Zuschlag auf die RGB-Lage**, keine zweite Lage daneben: beide Optiken
hängen an derselben Gimbal. Wird RGB neu ausgerichtet oder nachgezogen, zieht
Thermal mit, und in den Thermalreglern steht nur, was zwischen den Optiken
nicht passt. Der Wert bleibt im Projekt (`meander/thermal_lage.json`) und wird
beim Einfärben für die Thermalebene verwendet; ihre `meta.json` hält ihn fest.

Das ist der Weg, wenn die automatische Ausrichtung danebenliegt: erst in der
Überlagerung grob schieben, bis Magenta auf Grau liegt, dann in der Farbvorschau
feinjustieren.

In der Vorschau bedeutet **Grau: von keinem Bild getroffen**, und **Magenta** sind
die Kamerastandorte des Fluges. Sieht man nur Grau, beantwortet ein Blick auf die
magentafarbenen Punkte die erste Frage sofort — liegen sie weit neben der Wolke,
deckt der Mäanderflug dieses Gebiet nicht ab; liegen sie darüber, stimmt die
Ausrichtung nicht. Unter 5 % Trefferquote schreibt das Protokoll beides mit
Zahlen hin: Abstand der Kameras zur Wolkenmitte gegen die Ausdehnung der Wolke.

Die Vorschau erscheint **erst beim ersten Zug an einem Regler**, nicht schon
nach dem Ausrichten — wer nichts justiert, soll seine Karte sehen. Und sie
verschwindet wieder bei jedem Fehlschlag, jedem Abbruch, jedem Wechsel der
Farbquelle und nach dem Einfärben. Eine ausgeblendete Karte bleibt nie zurück.

Die Regler sind ein **Zuschlag** auf die gefundene Lage, nicht die Lage selbst —
sonst würde jeder Zug auf dem vorigen aufbauen und man käme nie zurück. X, Y
und Z sind Meter; der grobe Schieber geht in Dezimetern, der feine in
Zentimetern. Beim DRZ-Datensatz deckt ein RGB-Pixel rund 5 cm Boden ab — der
feine Schieber bewegt also um Bruchteile eines Pixels. Die Maßstab-Regler sind
kein Zuschlag, sondern der Wert selbst; sie bleiben im Projekt gespeichert.

Die beiden Kästen **RGB-Optik** und **Thermal-Optik** verschieben den
Bildhauptpunkt in Pixeln. Das wirkt wie eine Verkippung der Kamera gegen die
Achse, die COLMAP angenommen hat, und die Verschiebung am Boden wächst mit dem
Abstand — anders als die Regler für Gier, X und Y darüber, die starr schieben. Getrennt
je Optik, weil es zwei Objektive sind. Bewusst ohne Automatik: eine
Kennzahl dafür ist nicht zu finden, ein sonnenwarmes Dach ist thermisch
gleichmäßig und optisch strukturiert, ein Schatten umgekehrt.

## Farbquellen

Drei Einfärbungen liegen nebeneinander im Projekt und lassen sich unter
**Ansicht → Farbquelle** oder in der Seitenleiste sofort umschalten:

| Quelle | woher |
|---|---|
| **Onboard RGB** | 360°-Kamera an der Super-Drohne, `colors/` |
| **Mäander RGB** | `_V.JPG` des DJI-Fluges, `colors_meander_rgb/` |
| **Mäander Thermal** | `_T.JPG` desselben Fluges, `colors_meander_thermal/` |

Nach dem Einfärben einer zusammengeführten Karte steht im Protokoll die Quote je
Abschnitt, nicht nur eine Gesamtzahl — sonst merkt man nicht, wenn ein ganzer
Flug leer geblieben ist. Bleibt alles leer, löst das Programm „Nur eingefärbte
Punkte" von selbst, weil die Ansicht sonst komplett verschwindet.

Angeboten wird nur, was berechnet ist. Der Export schreibt die angezeigte Ebene.

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

Die zweite Wolke ist **orange und halbdurchsichtig** dargestellt. Orange heißt:
Vorschau, noch nicht übernommen. Sie gehört erst nach **Übernehmen** zur Karte,
und bis dahin ändert kein anderer Schritt etwas an ihr — färbst du in diesem
Zustand ein, wird nur der offene Flug eingefärbt, und das Programm fragt vorher
nach. Über **Ansicht → Zweiten Flug anzeigen** lässt sie sich ausblenden, ohne
sie zu verwerfen.
4. **Übernehmen** schreibt eine gemeinsame Aufzeichnung und öffnet sie als
   Arbeitswolke. Sie lässt sich danach als Ganzes einfärben und exportieren;
   jeder Abschnitt wird mit der Kamera seines eigenen Bags eingefärbt.

Nach dem Ausrichten stehen Trefferquote und Restfehler im Protokoll. Unter 0,3
Trefferquote überlappen die Wolken zu wenig — dann von Hand grob zusammenschieben
und „Nur ICP" nachlaufen lassen. Ein Restfehler über 0,3 m heißt: die Lage stimmt
grob, sitzt aber nicht sauber.

Ein zusammengeführtes Projekt hat keinen einzelnen Bagpfad, unter dem man es
wiederfände — sein Schlüssel ist ein erfundenes „A+B". Es lässt sich deshalb nur
über **Datei → Berechnetes Projekt öffnen …** (`Strg+Umschalt+P`) wieder öffnen;
die Liste zeigt alle Projekte des Caches und markiert die zusammengeführten.

Beim Laden liest das Programm die Abschnitte aus der `meta.json` der
Aufzeichnung, nicht aus dem Sitzungsgedächtnis. Nur so färbt jeder Abschnitt aus
der Kamera seines eigenen Bags — sonst bliebe der zweite Flug ungefärbt, weil es
in den Bildern des ersten keine Frames in seinem Zeitfenster gibt.

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

## Projekt mitnehmen

**Datei → Projekt exportieren …** (`Strg+Umschalt+E`) kopiert alles Berechnete in
einen Ordner, den du weiterreichen kannst. Der Dialog zeigt je Teil die Größe,
weil der Unterschied zwischen mit und ohne Rosbag schnell 24 GB ausmacht:

```
<ziel>/
  super360.json      Manifest: was drin ist, woher es kommt, Kennzahlen
  projekt/           recording/, colors*/, pano_*/, meander/, gps/extrinsic/settings
  bags/              nur wenn angehakt
  kalibrierung/      die verwendete calibration.json
```

Die Punktwolke ist immer dabei, ohne sie gibt es kein Projekt. Das **Rosbag ist
per Vorgabe nicht dabei** — es ist der Eingang, nicht das Ergebnis. Ohne Bag
bleiben Karte, Farben, Messen, Höhenschnitt und Export erhalten; das 360°-Video
und ein erneutes Einfärben brauchen es.

**Datei → Projektordner öffnen …** (`Strg+I`) öffnet so einen Ordner **an Ort
und Stelle** — ohne Kopie in den Cache und ohne Rückfrage. Gearbeitet wird im
Ordner selbst, jede Änderung landet sofort dort: Einstellungen, Ausrichtung,
Handzuschlag, Optik, Einfärbungen. Mitgenommene Bags werden aus dem Ordner
benutzt, die Bagpfade in der `meta.json` darauf gezogen (auch auf einem anderen
Rechner, gesucht wird über den Ordnernamen). Fehlt ein Bag, sagt das Protokoll
welches, und das Projekt öffnet trotzdem — nur eben ohne die Schritte, die es
braucht.

Ein Zielordner, der nicht leer ist und kein Projekt enthält, wird abgelehnt statt
zugeschüttet.

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

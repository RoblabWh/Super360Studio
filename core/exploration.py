"""Explorationsgrad: welchen Anteil des Zielgebiets ein Flug beobachtet hat.

Waehrend der autonomen Exploration steuert EPIC die Drohne selbst. Die Frage
danach lautet: **wie viel des vorgegebenen Gebiets hat sie in dieser Zeit
wirklich gesehen?** Genau das rechnet dieses Modul — nach demselben Verfahren
wie die Auswertung der Bachelorarbeit (``BA_Evaluation/auswertung_exploration.py``),
damit die Zahlen in der GUI und in der Arbeit dieselben sind.

Verfahren
---------

1. **Zielgebiet** aus ``/exploration/box`` (CUBE-Marker: Mitte + Kantenlaengen)
   als achsenparallele Box; sie wird in Zellen der Kantenlaenge ``voxel``
   zerlegt (Vorgabe 0,5 m, die Kartenaufloesung von EPIC).
2. **Beobachteter Raum** aus den im Weltsystem registrierten LiDAR-Scans
   (``/quad0_pcl_render_node/cloud``) und der Flugbahn
   (``/quad_0/lidar_slam/odom``): Jeder Strahl vom Sensor zum Messpunkt wird in
   Schritten einer halben Zellenkante abgetastet. Jede Zelle, in die eine Probe
   oder ein Messpunkt faellt, gilt als beobachtet — bekannter Raum ist der freie
   Raum entlang des Strahls plus die getroffene Oberflaeche. Die halbe Kante als
   Schrittweite ueberspringt praktisch keine durchquerte Zelle; feiner abtasten
   aendert das Ergebnis nicht mehr messbar.
3. **Explorationsmodus**: die Zeitraeume, in denen EPIC tatsaechlich flog.
   Erste Wahl ist der Zustand ``FEEDING`` aus ``/epic/bridge_status`` — das ist
   die Bruecke, die EPICs Sollwerte an PX4 weiterreicht, also der
   Explorationsmodus aus Sicht des Systems selbst. Fehlt das Topic, gilt
   ersatzweise der PX4-Modus ``OFFBOARD`` aus ``/mavros/state``. In den
   Testfluegen fallen beide zusammen.

Gezaehlt wird in **zwei** Rastern zugleich, in einem Durchlauf:

* ``prozent`` — nur die Scans **im Explorationsmodus**. Das ist der Wert, den
  die GUI oben rechts zeigt: was die autonome Exploration in ihrer Zeit
  gesehen hat, ohne den manuellen An- und Abflug davor und danach.
* ``prozent_gesamt`` — alle Scans des Bags, also der ganze Flug. Daraus kommen
  auch ``stand_beginn``/``stand_ende``/``zuwachs`` wie im Bericht der Arbeit.

Fliegt niemand autonom (die Bruecke bleibt in ``HOLD``, die EPIC-FSM in
``WAIT_TRIGGER``), gibt es keine Phase: ``prozent`` ist dann ``None`` und nur
der Wert des ganzen Fluges steht zur Verfuegung. Das ist kein Fehler, sondern
die ehrliche Auskunft ueber einen handgeflogenen Bag.

Der Boden schneidet das Ergebnis: liegt die Box teilweise unter dem Boden, sind
100 % im Volumen prinzipiell nicht erreichbar. Deshalb steht neben dem Volumen
immer die Abdeckung der Grundflaeche (Draufsicht).

Gelesen wird das Bag direkt mit ``rosbags`` (kein ROS noetig), unabhaengig von
der FAST-LIO-Aufzeichnung des Studios: Box, Scans und Flugbahn stammen damit
alle aus dem Weltsystem der Onboard-Kartierung und passen ohne Umrechnung
zusammen. Die lotrecht gedrehte Karte des Studios (``gravity_level``) wuerde
gegen die Box verkippen — deshalb wird sie hier bewusst nicht verwendet.

Qt-frei.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

# --------------------------------------------------------------------- Topics
TOPIC_BOX = "/exploration/box"                  # Zielgebiet (MarkerArray)
TOPIC_SCAN = "/quad0_pcl_render_node/cloud"     # LiDAR-Scan im Weltsystem
TOPIC_ODOM = "/quad_0/lidar_slam/odom"          # Flugbahn im Weltsystem
TOPIC_STATUS = "/epic/bridge_status"            # Zustand der PX4-Bruecke (JSON)
TOPIC_MODE = "/mavros/state"                    # PX4-Flugmodus (Ersatzquelle)

#: Kantenlaenge der Rasterzellen in m — die Kartenaufloesung von EPIC.
VOXEL_M = 0.5

#: Zustand der Bruecke, waehrend EPIC die Drohne steuert.
STATE_FEEDING = "FEEDING"
#: PX4-Modus, in dem die Sollwerte von EPIC kommen.
MODE_OFFBOARD = "OFFBOARD"

STATE_MSG_PATH = "/opt/ros/humble/share/mavros_msgs/msg/State.msg"
STATE_TYPENAME = "mavros_msgs/msg/State"
# mavros_msgs/State ist kein Standardtyp. Liegt ROS nicht auf dem Rechner,
# genuegt diese Kurzfassung — gebraucht wird nur das Feld "mode".
_STATE_FALLBACK = """
std_msgs/Header header
bool connected
bool armed
bool guided
bool manual_input
string mode
uint8 system_status
"""

#: Hoechstzahl der Proben je Block — begrenzt den Speicher beim Abtasten.
_BLOCK_PROBEN = 2_000_000


class KeineExplorationsdaten(RuntimeError):
    """Das Bag traegt keine EPIC-Explorationsdaten (Topics fehlen).

    Kein Fehler im Sinne eines Defekts: ein reiner Kamera-/LiDAR-Flug ohne
    laufendes EPIC hat schlicht kein Zielgebiet, zu dem sich ein Grad rechnen
    liesse. Die GUI faengt das ab und zeigt statt einer Zahl einen Strich.
    """


def _typestore():
    store = get_typestore(Stores.ROS2_HUMBLE)
    pfad = Path(STATE_MSG_PATH)
    text = pfad.read_text() if pfad.is_file() else _STATE_FALLBACK
    try:
        store.register(get_types_from_msg(text, STATE_TYPENAME))
    except Exception:  # noqa: BLE001 — beschaedigte .msg: Kurzfassung genuegt
        store.register(get_types_from_msg(_STATE_FALLBACK, STATE_TYPENAME))
    return store


def _nachrichten(reader, topic: str):
    """(Zeit ab Bag-Beginn in s, Nachricht) je Nachricht eines Topics."""
    conns = [c for c in reader.connections if c.topic == topic]
    if not conns:
        return
    for conn, stempel, roh in reader.messages(connections=conns):
        yield (stempel - reader.start_time) / 1e9, reader.deserialize(roh, conn.msgtype)


def _cloud_xyz(msg) -> np.ndarray:
    """x/y/z einer PointCloud2 als (N,3) float64, ohne NaN/Inf."""
    felder = {f.name: f for f in msg.fields}
    if any(a not in felder or felder[a].datatype != 7 for a in ("x", "y", "z")):
        raise RuntimeError(
            f"Punktwolke in {TOPIC_SCAN!r}: x/y/z fehlen oder sind nicht FLOAT32.")
    n = msg.width * msg.height
    daten = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
    xyz = np.column_stack([
        daten[:, felder[a].offset:felder[a].offset + 4].copy().view(np.float32).ravel()
        for a in ("x", "y", "z")
    ]).astype(np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]


# --------------------------------------------------- Zielgebiet, Bahn, Phasen

def lies_box(reader) -> tuple[np.ndarray, np.ndarray] | None:
    """Zielgebiet als (min-Ecke, max-Ecke) in m, oder None."""
    for _, msg in _nachrichten(reader, TOPIC_BOX):
        for marker in msg.markers:
            if marker.type == 1:  # CUBE: Mittelpunkt + Kantenlaengen
                mitte = np.array([marker.pose.position.x,
                                  marker.pose.position.y,
                                  marker.pose.position.z], dtype=np.float64)
                groesse = np.array([marker.scale.x, marker.scale.y, marker.scale.z],
                                   dtype=np.float64)
                if np.all(groesse > 1e-6):
                    return mitte - groesse / 2.0, mitte + groesse / 2.0
            if len(marker.points) > 0:  # Ersatz: Eckpunkte eines Linienmarkers
                pts = np.array([[p.x, p.y, p.z] for p in marker.points], dtype=np.float64)
                return pts.min(axis=0), pts.max(axis=0)
    return None


def lies_flugbahn(reader) -> tuple[np.ndarray, np.ndarray]:
    """Zeiten (s ab Bag-Beginn) und Positionen (N,3) der Flugbahn."""
    zeiten: list[float] = []
    positionen: list[list[float]] = []
    for t, msg in _nachrichten(reader, TOPIC_ODOM):
        p = msg.pose.pose.position
        zeiten.append(t)
        positionen.append([p.x, p.y, p.z])
    z = np.asarray(zeiten, dtype=np.float64)
    p = np.asarray(positionen, dtype=np.float64).reshape(-1, 3)
    if len(p):
        gut = np.isfinite(p).all(axis=1)
        z, p = z[gut], p[gut]
    return z, p


def _zeitraeume(zeiten, aktiv, ende) -> list[tuple[float, float]]:
    """Folge von (Zeit, an/aus) zu Zeitraeumen [(Start, Ende), ...]."""
    raeume: list[tuple[float, float]] = []
    start = None
    for t, an in zip(zeiten, aktiv):
        if an and start is None:
            start = t
        elif not an and start is not None:
            raeume.append((float(start), float(t)))
            start = None
    if start is not None:
        raeume.append((float(start), float(ende)))
    return raeume


def lies_phasen(reader) -> tuple[list[tuple[float, float]], str]:
    """Zeitraeume im Explorationsmodus und die Quelle, aus der sie stammen.

    Reihenfolge: ``FEEDING`` aus der EPIC-Bruecke, sonst ``OFFBOARD`` aus dem
    PX4-Status. Gibt es beides nicht, ist die Liste leer — dann flog niemand
    autonom.
    """
    ende = reader.duration / 1e9
    zeiten, aktiv, gesehen = [], [], False
    for t, msg in _nachrichten(reader, TOPIC_STATUS):
        gesehen = True
        try:
            status = json.loads(msg.data)
        except (ValueError, TypeError):
            continue
        zeiten.append(t)
        aktiv.append(status.get("state") == STATE_FEEDING)
    phasen = _zeitraeume(zeiten, aktiv, ende)
    if phasen:
        return phasen, f"{STATE_FEEDING} aus {TOPIC_STATUS}"

    zeiten, aktiv = [], []
    for t, msg in _nachrichten(reader, TOPIC_MODE):
        zeiten.append(t)
        aktiv.append(getattr(msg, "mode", "") == MODE_OFFBOARD)
    phasen = _zeitraeume(zeiten, aktiv, ende)
    if phasen:
        return phasen, f"{MODE_OFFBOARD} aus {TOPIC_MODE}"
    if gesehen:
        return [], f"{TOPIC_STATUS} kennt keinen {STATE_FEEDING}-Abschnitt"
    return [], "kein Statustopic im Bag"


# ------------------------------------------------------- beobachteten Raum rechnen

def _setze(gitter, box_min, box_max, voxel, proben) -> None:
    """Rasterzellen markieren, in die die Proben fallen."""
    if len(proben) == 0:
        return
    # "< box_max", nicht "<=": eine Probe genau auf der Maximalflaeche ergaebe
    # sonst einen Index ausserhalb des Rasters.
    drin = np.all((proben >= box_min) & (proben < box_max), axis=1)
    if not drin.any():
        return
    idx = ((proben[drin] - box_min) / voxel).astype(np.int64)
    np.minimum(idx, np.asarray(gitter.shape) - 1, out=idx)   # Rundungsschutz am Rand
    gitter[idx[:, 0], idx[:, 1], idx[:, 2]] = True


def _markiere_scan(gitter_liste, box_min, box_max, voxel, ursprung, punkte,
                   max_reichweite) -> None:
    """Strahlen Sensor -> Messpunkt abtasten und die Zellen markieren.

    ``gitter_liste`` sind die Raster, die denselben Scan bekommen (ganzer Flug
    und, wenn der Scan in eine Phase faellt, zusaetzlich das Phasenraster) —
    die Proben werden nur einmal gerechnet.
    """
    if len(punkte) == 0:
        return
    richtung = punkte - ursprung
    laenge = np.linalg.norm(richtung, axis=1)
    gut = laenge > 1e-6
    richtung, laenge, punkte = richtung[gut], laenge[gut], punkte[gut]
    if len(punkte) == 0:
        return
    einheit = richtung / laenge[:, np.newaxis]
    # Weiter als bis zur entferntesten Box-Ecke muss kein Strahl abgetastet werden.
    laenge = np.minimum(laenge, max_reichweite)
    for g in gitter_liste:
        _setze(g, box_min, box_max, voxel, punkte)          # Messpunkte selbst

    schritt = voxel / 2.0
    n_max = int(np.ceil(float(laenge.max()) / schritt))
    if n_max <= 0:
        return
    block = max(1, _BLOCK_PROBEN // n_max)
    for i in range(0, len(einheit), block):
        e, l = einheit[i:i + block], laenge[i:i + block]
        abstaende = np.arange(schritt, float(l.max()) + schritt, schritt)
        # (m,1,1) * (1,n,3) -> (m,n,3): m Abstaende entlang jedes der n Strahlen
        proben = ursprung + e[np.newaxis] * abstaende[:, np.newaxis, np.newaxis]
        proben = proben[abstaende[:, np.newaxis] <= l]      # hinter dem Strahlende weg
        for g in gitter_liste:
            _setze(g, box_min, box_max, voxel, proben)


# -------------------------------------------------------------------- Ergebnis

def _de(x: float, n: int = 1) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{n}f}".replace(".", ",")


@dataclass
class Explorationsgrad:
    """Ergebnis einer Auswertung. Alle Prozentwerte 0..100, Zeiten in s."""

    #: beobachteter Anteil des Zielvolumens im Explorationsmodus (None: keine Phase)
    prozent: float | None
    #: dasselbe fuer die Grundflaeche (Draufsicht)
    prozent_flaeche: float | None
    #: beobachteter Anteil ueber den ganzen Flug
    prozent_gesamt: float
    prozent_flaeche_gesamt: float
    #: Stand des ganzen Fluges bei Beginn/Ende der Exploration und der Zuwachs
    #: waehrend der Phasen (Prozentpunkte) — wie im Bericht der Arbeit
    stand_beginn: float | None
    stand_ende: float | None
    zuwachs: float | None
    #: Zielgebiet
    box_min: list[float]
    box_max: list[float]
    volumen_m3: float
    flaeche_m2: float
    voxel_m: float
    #: Explorationsmodus
    phasen: list[tuple[float, float]]
    quelle: str
    dauer_s: float
    strecke_m: float
    #: Umfang der Auswertung
    scans: int
    scans_phase: int
    bag_dauer_s: float
    bag: str = ""
    #: Verlauf des ganzen Fluges: Zeiten (s) und Prozentwerte
    verlauf_t: list[float] = field(default_factory=list)
    verlauf_p: list[float] = field(default_factory=list)

    # ------------------------------------------------------------- Kennzahlen

    @property
    def hat_phase(self) -> bool:
        """True, wenn im Bag ueberhaupt autonom exploriert wurde."""
        return bool(self.phasen)

    @property
    def wert(self) -> float:
        """Der anzuzeigende Grad: Explorationsmodus, sonst ganzer Flug."""
        return self.prozent_gesamt if self.prozent is None else self.prozent

    @property
    def wert_flaeche(self) -> float:
        return (self.prozent_flaeche_gesamt if self.prozent_flaeche is None
                else self.prozent_flaeche)

    @property
    def bezug(self) -> str:
        """Worauf sich :attr:`wert` bezieht — gehoert an jede Anzeige der Zahl."""
        return "Explorationsmodus" if self.hat_phase else "ganzer Flug"

    def kurz(self) -> str:
        """Eine Zeile fuer die Anzeige: '96,8 % (Explorationsmodus)'."""
        return f"{_de(self.wert)} % ({self.bezug})"

    # -------------------------------------------------------------- Persistenz

    def als_dict(self) -> dict:
        d = dict(self.__dict__)
        d["phasen"] = [[float(s), float(e)] for s, e in self.phasen]
        return d

    @classmethod
    def aus_dict(cls, d: dict) -> "Explorationsgrad":
        d = dict(d)
        d["phasen"] = [(float(s), float(e)) for s, e in d.get("phasen", [])]
        bekannt = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in bekannt})

    # ----------------------------------------------------------------- Bericht

    def text(self) -> str:
        """Ausfuehrlicher Bericht, gleiche Groessen wie in der BA-Auswertung."""
        b_min, b_max = np.asarray(self.box_min), np.asarray(self.box_max)
        g = b_max - b_min
        z = [
            "Explorationsgrad",
            f"Bag: {self.bag}" if self.bag else "",
            "",
            "--- Zielgebiet (Explorationsbox) ---",
            f"Abmessungen (x/y/z):     {_de(g[0])} m x {_de(g[1])} m x {_de(g[2])} m",
            f"Bereich:                 x [{_de(b_min[0])}, {_de(b_max[0])}], "
            f"y [{_de(b_min[1])}, {_de(b_max[1])}], z [{_de(b_min[2])}, {_de(b_max[2])}] m",
            f"Grundfläche:             {_de(self.flaeche_m2, 0)} m²",
            f"Volumen:                 {_de(self.volumen_m3, 0)} m³",
            "",
            "--- Explorationsmodus ---",
            f"Quelle der Phasen:       {self.quelle}",
        ]
        if self.hat_phase:
            z += [
                f"Dauer:                   {_de(self.dauer_s)} s "
                + " ".join(f"[{_de(s)}–{_de(e)} s]" for s, e in self.phasen),
                f"Flugstrecke dabei:       {_de(self.strecke_m)} m",
                f"Ausgewertete Scans:      {self.scans_phase} von {self.scans}",
                "",
                "--- Erreichte Exploration (nur im Explorationsmodus) ---",
                f"Beobachtetes Volumen:    {_de(self.wert)} % des Zielvolumens",
                f"Abgedeckte Grundfläche:  {_de(self.wert_flaeche)} %  (Draufsicht)",
                "",
                "--- Zum Vergleich: ganzer Flug ---",
                f"Beobachtetes Volumen:    {_de(self.prozent_gesamt)} % "
                f"({_de(self.prozent_gesamt / 100.0 * self.volumen_m3, 0)} m³ "
                f"von {_de(self.volumen_m3, 0)} m³)",
                f"Abgedeckte Grundfläche:  {_de(self.prozent_flaeche_gesamt)} %",
                f"Stand bei Explorationsbeginn: {_de(self.stand_beginn or 0.0)} %, "
                f"bei Explorationsende: {_de(self.stand_ende or 0.0)} %",
                f"Zuwachs durch autonome Exploration: "
                f"{_de(self.zuwachs or 0.0)} Prozentpunkte",
            ]
        else:
            z += [
                "Es gab keinen autonomen Abschnitt — der Flug wurde von Hand",
                "geflogen. Der Grad unten gilt deshalb für den ganzen Flug.",
                "",
                "--- Erreichte Exploration (ganzer Flug) ---",
                f"Beobachtetes Volumen:    {_de(self.prozent_gesamt)} % "
                f"({_de(self.prozent_gesamt / 100.0 * self.volumen_m3, 0)} m³ "
                f"von {_de(self.volumen_m3, 0)} m³)",
                f"Abgedeckte Grundfläche:  {_de(self.prozent_flaeche_gesamt)} %  (Draufsicht)",
                f"Ausgewertete Scans:      {self.scans}",
            ]
        z += [
            "",
            f"(Rasterauflösung {_de(self.voxel_m, 2)} m; liegt die Box teilweise unter",
            " dem Boden, sind 100 % im Volumen prinzipiell nicht erreichbar —",
            " deshalb steht die Grundfläche daneben.)",
            f"Dauer der Aufnahme:      {_de(self.bag_dauer_s)} s",
        ]
        return "\n".join(x for x in z if x is not None)


# ----------------------------------------------------------------- Hauptweg

def berechne(bag_path: str, voxel: float = VOXEL_M,
             progress=None, cancel=None) -> Explorationsgrad:
    """Explorationsgrad eines Bags rechnen.

    :param bag_path: Ordner des ROS2-Bags (mit metadata.yaml)
    :param voxel: Kantenlaenge der Rasterzellen in m
    :param progress: ``progress(frac, msg)`` — Fortschritt 0..1, Meldung deutsch
    :param cancel: ``threading.Event``; gesetzt ⇒ ``RuntimeError("Abgebrochen")``
    :raises KeineExplorationsdaten: das Bag traegt keine EPIC-Topics
    """
    def melde(frac: float, msg: str) -> None:
        if progress is not None:
            progress(float(frac), msg)

    def pruefe() -> None:
        if cancel is not None and cancel.is_set():
            raise RuntimeError("Abgebrochen")

    if voxel <= 0:
        raise RuntimeError("Rasterauflösung muss größer als 0 sein.")
    pfad = Path(bag_path)
    if not pfad.exists():
        raise RuntimeError(f"Bag nicht gefunden: {bag_path}")

    melde(0.02, "Öffne Bag …")
    with AnyReader([pfad], default_typestore=_typestore()) as reader:
        vorhanden = {c.topic for c in reader.connections}
        fehlt = [t for t in (TOPIC_BOX, TOPIC_SCAN, TOPIC_ODOM) if t not in vorhanden]
        if fehlt:
            raise KeineExplorationsdaten(
                "Das Bag enthält keine EPIC-Explorationsdaten — es fehlen: "
                + ", ".join(fehlt))

        melde(0.05, "Lese Zielgebiet, Flugbahn und Phasen …")
        box = lies_box(reader)
        if box is None:
            raise KeineExplorationsdaten(
                f"In {TOPIC_BOX!r} steht kein Zielgebiet — ohne Box gibt es "
                "keinen Bezug, auf den sich ein Grad rechnen ließe.")
        box_min, box_max = box
        odom_t, odom_p = lies_flugbahn(reader)
        if len(odom_t) == 0:
            raise KeineExplorationsdaten(
                f"Keine Flugbahn in {TOPIC_ODOM!r} — ohne Sensorposition lässt "
                "sich kein Strahl verfolgen.")
        phasen, quelle = lies_phasen(reader)
        bag_dauer = reader.duration / 1e9
        pruefe()

        # Mindestens eine Zelle je Achse: eine entartete Box (flacher Marker)
        # ergaebe sonst ein leeres Raster und eine Division durch null.
        form = np.maximum(np.ceil((box_max - box_min) / voxel).astype(int), 1)
        gitter_alle = np.zeros(form, dtype=bool)
        gitter_phase = np.zeros(form, dtype=bool)
        ecken = np.array([[x, y, z]
                          for x in (box_min[0], box_max[0])
                          for y in (box_min[1], box_max[1])
                          for z in (box_min[2], box_max[2])], dtype=np.float64)

        n_scans = sum(c.msgcount for c in reader.connections if c.topic == TOPIC_SCAN)
        melde(0.08, f"Rekonstruiere beobachteten Raum ({n_scans} Scans) …")
        zellen = int(gitter_alle.size)
        verlauf_t: list[float] = []
        verlauf_p: list[float] = []
        scans = scans_phase = 0
        for t, msg in _nachrichten(reader, TOPIC_SCAN):
            if scans % 10 == 0:
                pruefe()
                melde(0.08 + 0.88 * (scans / max(n_scans, 1)),
                      f"Beobachteten Raum rekonstruieren … Scan {scans}/{n_scans}")
            punkte = _cloud_xyz(msg)
            scans += 1
            if len(punkte) == 0:
                continue
            # Sensorposition: zeitlich naechste Odometrie. Scans und Odometrie
            # kommen beide mit ~10 Hz fast zeitgleich; bei den geflogenen
            # Geschwindigkeiten liegt der Versatz weit unter einer Zellenkante.
            ursprung = odom_p[int(np.argmin(np.abs(odom_t - t)))]
            in_phase = any(s <= t <= e for s, e in phasen)
            ziele = [gitter_alle, gitter_phase] if in_phase else [gitter_alle]
            reichweite = float(np.linalg.norm(ecken - ursprung, axis=1).max())
            _markiere_scan(ziele, box_min, box_max, voxel, ursprung, punkte, reichweite)
            if in_phase:
                scans_phase += 1
            verlauf_t.append(float(t))
            verlauf_p.append(100.0 * int(gitter_alle.sum()) / zellen)

    if not verlauf_t:
        raise KeineExplorationsdaten(
            f"Keine auswertbaren Scans in {TOPIC_SCAN!r} — der Explorationsgrad "
            "lässt sich nicht rechnen.")

    melde(0.98, "Kennzahlen …")
    groesse = box_max - box_min
    flaeche = float(groesse[0] * groesse[1])
    volumen = flaeche * float(groesse[2])
    vt, vp = np.asarray(verlauf_t), np.asarray(verlauf_p)

    prozent_gesamt = 100.0 * int(gitter_alle.sum()) / zellen
    flaeche_gesamt = 100.0 * float(gitter_alle.any(axis=2).mean())
    hat_phase = bool(phasen)
    prozent = 100.0 * int(gitter_phase.sum()) / zellen if hat_phase else None
    prozent_flaeche = (100.0 * float(gitter_phase.any(axis=2).mean())
                       if hat_phase else None)

    stand_beginn = stand_ende = zuwachs = None
    if hat_phase:
        stand_beginn = float(np.interp(phasen[0][0], vt, vp))
        stand_ende = float(np.interp(phasen[-1][1], vt, vp))
        zuwachs = float(sum(np.interp(e, vt, vp) - np.interp(s, vt, vp)
                            for s, e in phasen))

    dauer = float(sum(e - s for s, e in phasen))
    strecke = 0.0
    for s, e in phasen:
        maske = (odom_t >= s) & (odom_t <= e)
        if maske.sum() >= 2:
            strecke += float(np.linalg.norm(np.diff(odom_p[maske], axis=0), axis=1).sum())

    melde(1.0, "Explorationsgrad berechnet")
    return Explorationsgrad(
        prozent=prozent,
        prozent_flaeche=prozent_flaeche,
        prozent_gesamt=prozent_gesamt,
        prozent_flaeche_gesamt=flaeche_gesamt,
        stand_beginn=stand_beginn,
        stand_ende=stand_ende,
        zuwachs=zuwachs,
        box_min=[float(x) for x in box_min],
        box_max=[float(x) for x in box_max],
        volumen_m3=volumen,
        flaeche_m2=flaeche,
        voxel_m=float(voxel),
        phasen=[(float(s), float(e)) for s, e in phasen],
        quelle=quelle,
        dauer_s=dauer,
        strecke_m=strecke,
        scans=scans,
        scans_phase=scans_phase,
        bag_dauer_s=float(bag_dauer),
        bag=os.path.basename(os.path.abspath(str(bag_path))),
        verlauf_t=[round(float(x), 2) for x in vt],
        verlauf_p=[round(float(x), 2) for x in vp],
    )


if __name__ == "__main__":
    import sys

    print("== Test 1: Zeitraeume aus Zustandsfolgen ==")
    assert _zeitraeume([0, 1, 2, 3], [False, True, True, False], 9) == [(1.0, 3.0)]
    assert _zeitraeume([0, 1], [False, True], 9) == [(1.0, 9.0)], "offene Phase bis Bag-Ende"
    assert _zeitraeume([0, 1], [False, False], 9) == []
    print("  Start/Ende, offene Phase am Bag-Ende, gar keine Phase — OK")

    print("== Test 2: Strahlen fuellen das Raster ==")
    # Wuerfel 4 x 4 x 4 m bei 1 m Raster, Sensor in einer Ecke ausserhalb,
    # ein Messpunkt in der gegenueberliegenden Ecke: die Diagonale muss
    # durchgehend markiert sein.
    b_min, b_max, vx = np.zeros(3), np.full(3, 4.0), 1.0
    g = np.zeros((4, 4, 4), dtype=bool)
    ursprung = np.array([-1.0, -1.0, -1.0])
    punkte = np.array([[4.5, 4.5, 4.5]])
    _markiere_scan([g], b_min, b_max, vx, ursprung, punkte,
                   float(np.linalg.norm(b_max - ursprung)))
    diagonale = [g[i, i, i] for i in range(4)]
    assert all(diagonale), f"Diagonale nicht durchgehend: {diagonale}"
    assert g.sum() == 4, f"{g.sum()} Zellen statt der 4 auf der Diagonalen"
    print(f"  Diagonale vollstaendig, {g.sum()} von {g.size} Zellen — OK")

    print("== Test 3: Punkt genau auf der Maximalflaeche ==")
    g = np.zeros((4, 4, 4), dtype=bool)
    _setze(g, b_min, b_max, vx, np.array([[4.0, 4.0, 4.0], [3.9, 0.1, 0.1]]))
    assert g[3, 0, 0] and not g[3, 3, 3], "Punkt auf box_max darf nicht zaehlen"
    print("  liegt ausserhalb (< box_max) und sprengt das Raster nicht — OK")

    print("== Test 4: Ergebnis-Objekt, Rundlauf ueber JSON ==")
    grad = Explorationsgrad(
        prozent=59.7, prozent_flaeche=93.3, prozent_gesamt=69.9,
        prozent_flaeche_gesamt=99.2, stand_beginn=21.9, stand_ende=60.6,
        zuwachs=38.7, box_min=[0, 0, 0], box_max=[10, 10, 5],
        volumen_m3=500.0, flaeche_m2=100.0, voxel_m=0.5,
        phasen=[(18.7, 101.7)], quelle="FEEDING aus /epic/bridge_status",
        dauer_s=83.0, strecke_m=28.4, scans=1635, scans_phase=830,
        bag_dauer_s=163.6, bag="testbag")
    assert grad.hat_phase and abs(grad.wert - 59.7) < 1e-9
    assert grad.bezug == "Explorationsmodus" and grad.kurz() == "59,7 % (Explorationsmodus)"
    zurueck = Explorationsgrad.aus_dict(json.loads(json.dumps(grad.als_dict())))
    assert zurueck.als_dict() == grad.als_dict(), "JSON-Rundlauf veraendert das Ergebnis"
    assert "Zuwachs durch autonome Exploration: 38,7" in grad.text()
    ohne = Explorationsgrad.aus_dict({**grad.als_dict(), "prozent": None,
                                     "prozent_flaeche": None, "phasen": []})
    assert not ohne.hat_phase and ohne.bezug == "ganzer Flug"
    assert abs(ohne.wert - 69.9) < 1e-9, "ohne Phase muss der ganze Flug gelten"
    print(f"  {grad.kurz()} / ohne Phase: {ohne.kurz()} — OK")

    bag = sys.argv[1] if len(sys.argv) > 1 else None
    if bag:
        print("== Test 5: echtes Bag ==")
        try:
            g2 = berechne(bag, progress=lambda f, m: print(f"    {f * 100:5.1f} % {m}",
                                                           end="\r"))
        except KeineExplorationsdaten as exc:
            print(f"\n  keine Explorationsdaten: {exc}")
        else:
            print("\n" + g2.text())
            assert 0.0 <= g2.wert <= 100.0
            assert g2.prozent is None or g2.prozent <= g2.prozent_gesamt + 1e-9, \
                "die Phase allein kann nicht mehr sehen als der ganze Flug"
    print("exploration SELFTEST OK")

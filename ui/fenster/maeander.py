"""Mäander (Mixin des Hauptfensters).

Schreibt am Hauptfenster: _auto_kette, _btn_meander_align, _btn_meander_auto,
_btn_meander_fein, _btn_meander_fenster, _btn_meander_optik, _btn_meander_pick,
_btn_meander_run, _chk_sichtbar, _chk_solo, _chk_thermal, _lbl_meander,
_lbl_meander_lage, _massstab, _meander_dir, _meander_pipe, _optik,
_optik_neu_messen, _spin_meander, _spin_meander_th, _spin_optik.
"""
from __future__ import annotations

import os

import numpy as np

from PyQt5.QtWidgets import (
    QCheckBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QWidget,
)

from ui.bausteine import _wrappable, still_setzen


class MaeanderMixin:
    def _abschnitt_maeander(self) -> QWidget:
        box = QWidget()
        form = _wrappable(QFormLayout(box))
        self._btn_meander_pick = QPushButton("Mäanderflug wählen …")
        self._btn_meander_pick.setToolTip(
            "Ordner mit den Bildern eines DJI-Kartierungsfluges.\n"
            "Gesucht werden die _V.JPG, die _T.JPG sind die Thermalbilder.")
        self._btn_meander_pick.clicked.connect(self._on_meander_pick)
        form.addRow(self._btn_meander_pick)
        self._lbl_meander = QLabel("Kein Mäanderflug geladen.")
        self._lbl_meander.setWordWrap(True)
        form.addRow(self._lbl_meander)

        self._chk_thermal = QCheckBox("Thermalbilder mitrechnen")
        self._chk_thermal.setToolTip(
            "Färbt ein zweites Mal mit den _T.JPG und legt eine eigene Ebene an.\n"
            "Eine zweite Rekonstruktion braucht es nicht — beide Optiken sitzen\n"
            "auf derselben Gimbal und lösen zusammen aus.")
        self._chk_thermal.stateChanged.connect(self._on_setting_changed)
        form.addRow(self._chk_thermal)

        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        self._btn_meander_align = QPushButton("Ausrichten")
        self._btn_meander_align.setToolTip(
            "Grob per Kreuzkorrelation über den Gierwinkel, fein über den\n"
            "Höhenunterschied zum Rastermodell der Wolke. Kein ICP — das würde\n"
            "an Gebäudekanten verkippen und die Lotrechte zerstören.")
        self._btn_meander_align.clicked.connect(self._on_meander_align)
        self._btn_meander_run = QPushButton("Einfärben")
        self._btn_meander_run.clicked.connect(self._on_meander_run)
        hl.addWidget(self._btn_meander_align)
        hl.addWidget(self._btn_meander_run)
        form.addRow(row)
        self._btn_meander_fenster = QPushButton("Überlagern und justieren …")
        self._btn_meander_fenster.setToolTip(
            "Eigenes Fenster: Karte und Flug übereinander, live verschieben,\n"
            "mit Farbvorschau. Das Hauptfenster bleibt unberührt.")
        self._btn_meander_fenster.clicked.connect(self._on_meander_fenster)
        form.addRow(self._btn_meander_fenster)
        self._btn_meander_optik = QPushButton("Optik einmessen")
        self._btn_meander_optik.setToolTip(
            "Höhe über den Laser-Entfernungsmesser der Drohne, RGB-Brennweite\n"
            "über die Farbkonsistenz der Bilder, Thermalkamera (Brennweite,\n"
            "Verzeichnung, Schielwinkel) gegen das RGB-Bild desselben Auslösers.\n"
            "Rund eine Minute. Läuft nach dem ersten Ausrichten von selbst.")
        self._btn_meander_optik.clicked.connect(lambda: self._on_meander_einmessen())
        self._btn_meander_fein = QPushButton("Feinausrichten")
        self._btn_meander_fein.setToolTip(
            "Fotomodell auf die Karte legen: Neigung und Höhe über die\n"
            "Oberfläche, Versatz in der Ebene über die Kanten. Gegengeprüft über\n"
            "die Farbkonsistenz — was nicht hilft, wird nicht übernommen.")
        self._btn_meander_fein.clicked.connect(lambda: self._on_meander_fein())
        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self._btn_meander_optik)
        hl.addWidget(self._btn_meander_fein)
        form.addRow(row)
        self._btn_meander_auto = QPushButton("Automatisch: ausrichten bis zur Farbe")
        self._btn_meander_auto.setToolTip(
            "Alles hintereinander: Ausrichten, Optik einmessen, Feinausrichten,\n"
            "Einfärben mit Sichtprüfung. Rund sechs Minuten.")
        self._btn_meander_auto.setStyleSheet("font-weight: bold;")
        self._btn_meander_auto.clicked.connect(lambda: self._on_meander_auto())
        form.addRow(self._btn_meander_auto)
        self._chk_sichtbar = QCheckBox("Beim Einfärben Sichtbarkeit prüfen (Wände)")
        self._chk_sichtbar.setChecked(True)
        self._chk_sichtbar.setToolTip(
            "Jeder Punkt nur aus Bildern, die ihn wirklich sehen, und aus dem,\n"
            "das am frontalsten auf seine Fläche blickt. Sonst läuft das Dach-\n"
            "muster die Wände hinunter. Verdeckte Punkte bleiben ungefärbt.\n"
            "Dauert etwa dreimal so lang.")
        self._chk_sichtbar.stateChanged.connect(self._on_setting_changed)
        form.addRow(self._chk_sichtbar)

        # Handjustage: verschiebt die Fotopunkte starr gegen die Wolke. RGB
        # ist ein Zuschlag auf die gefundene Lage, Thermal ein Zuschlag auf
        # die RGB-Lage — beide Optiken haengen an derselben Gimbal, wird RGB
        # nachgezogen, zieht Thermal mit. Jeder Wert hat einen groben und einen
        # feinen Schieber, der feine bis auf den Zentimeter.
        from ui.feinregler import regler_grad, regler_meter, regler_pixel, regler_prozent
        self._spin_meander = {}
        self._spin_meander_th = {}
        self._massstab = {}
        for optik, titel, ziel, felder in (
                ("rgb", "RGB — Zuschlag auf die gefundene Lage", self._spin_meander,
                 (("yaw", "Gier", regler_grad()), ("x", "X", regler_meter()),
                  ("y", "Y", regler_meter()), ("z", "Z (Höhe)", regler_meter(50.0)))),
                ("thermal", "Thermal — Zuschlag auf die RGB-Lage", self._spin_meander_th,
                 (("yaw", "Gier", regler_grad()), ("x", "X", regler_meter()),
                  ("y", "Y", regler_meter())))):
            form.addRow(QLabel(f"<b>{titel}</b>"))
            for key, label, regler in felder:
                regler.valueChanged.connect(
                    lambda _v, o=optik: self._on_meander_manual(o))
                form.addRow(label, regler)
                ziel[key] = regler
            m = regler_prozent()
            m.setToolTip(
                "Maßstab = Brennweite gegenüber der Rekonstruktion, in Prozent.\n"
                "Zu kurz, und jedes Bild landet zu klein auf der Karte — am Rand\n"
                "um Meter, in jedem Bild anders: die Bilder wirken zueinander\n"
                "verzerrt. „Optik einmessen“ findet ihn selbst."
                if optik == "rgb" else
                "Maßstab der Thermalkamera gegenüber ihrer Einmessung, in Prozent.")
            m.valueChanged.connect(lambda _v, o=optik: self._on_meander_massstab(o))
            form.addRow("Maßstab", m)
            self._massstab[optik] = m
        self._chk_solo = QCheckBox("Während der Justage nur die Vorschau zeigen")
        self._chk_solo.setChecked(False)
        self._chk_solo.setToolTip(
            "Blendet die volle Karte aus, solange die Vorschau läuft.\n"
            "50.000 Stichprobenpunkte gehen in 24 Millionen sonst unter.\n"
            "Abschalten zeigt beides übereinander.")
        self._chk_solo.stateChanged.connect(self._on_solo_changed)
        form.addRow(self._chk_solo)
        self._lbl_meander_lage = QLabel("")
        self._lbl_meander_lage.setWordWrap(True)
        form.addRow(self._lbl_meander_lage)

        # Hauptpunkt-Versatz je Optik: wirkt wie eine Verkippung der Kamera
        # gegen die Achse, die COLMAP angenommen hat, und waechst mit dem
        # Abstand — anders als die Regler darueber, die starr schieben.
        self._spin_optik = {}
        for optik, titel, tip in (
                ("rgb", "RGB-Optik", "Versatz des Bildhauptpunkts in Pixeln des RGB-Bildes."),
                ("thermal", "Thermal-Optik", "Dasselbe für die Thermaloptik — eigener Wert, "
                                             "es ist ein zweites Objektiv.")):
            form.addRow(QLabel(f"<b>{titel} — Hauptpunkt</b>"))
            for achse, label in (("u", "rechts"), ("v", "unten")):
                sp = regler_pixel()
                sp.setToolTip(tip)
                sp.valueChanged.connect(self._on_setting_changed)
                form.addRow(label, sp)
                self._spin_optik[(optik, achse)] = sp
        return box

    # ================================================= Mäander-Einfärbung

    def _meander_versatz(self, optik: str) -> list:
        return [float(self._spin_optik[(optik, "u")].value()),
                float(self._spin_optik[(optik, "v")].value())]

    def _on_meander_pick(self) -> None:
        if self._rec is None or self._project is None:
            QMessageBox.information(
                self, "Mäander-Einfärbung",
                "Erst einen Flug öffnen und seine Karte berechnen — sie ist die "
                "Wolke, die eingefärbt wird.")
            return
        start = self._meander_dir or os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(
            self, "Ordner mit den Bildern des Mäanderfluges", start)
        if not path:
            return
        n_v = len([f for f in os.listdir(path) if f.upper().endswith("_V.JPG")])
        n_t = len([f for f in os.listdir(path) if f.upper().endswith("_T.JPG")])
        if n_v == 0:
            from core import meander as meander_mod
            n_v = meander_mod.zaehle_rgb_jpeg(path)
        if n_v == 0:
            self._show_error("Mäander-Einfärbung",
                             f"In '{os.path.basename(path)}' liegen keine JPEGs.")
            return
        self._meander_dir = path
        self._meander_pipe = None      # Pipeline wird beim nächsten Lauf neu gebaut
        self._live_clear()
        self._chk_thermal.setEnabled(n_t > 0)
        if n_t == 0:
            self._chk_thermal.setChecked(False)
        self._lbl_meander.setText(
            f"{os.path.basename(path)}: {n_v} RGB-Bilder"
            + (f", {n_t} Thermalbilder" if n_t else ", keine Thermalbilder"))
        self._log(f"Mäanderflug gewählt: {path} — {n_v} RGB, {n_t} Thermal.")
        self._save_settings()
        self._update_enabled()

    def _meander_args(self) -> dict:
        """Alles, was der Arbeitsthread braucht — im GUI-Thread eingesammelt.

        Widgets duerfen nur hier gelesen werden. Der Worker laeuft in einem
        eigenen Thread, und Qt-Widgets von dort anzufassen ist ein Fehler, der
        sich erst spaeter und schlecht reproduzierbar zeigt.
        """
        return {
            "points": self._world,
            "photo_dir": self._meander_dir,
            "work_dir": self._project.meander_work_dir(),
            "thermal": bool(self._chk_thermal.isChecked()),
            "rgb_versatz": self._meander_versatz("rgb"),
            "thermal_versatz": self._meander_versatz("thermal"),
        }

    @staticmethod
    def _meander_build(args: dict, log_cb):
        """Pipeline aufsetzen; die Arbeitswolke geht als cloud.npy hinein."""
        from core import meander as meander_mod
        return meander_mod.bauen(args, log_cb)

    @staticmethod
    def _meander_bereit(pipe, args: dict, bauen, th_zuschlag, optik_jetzt, progress_cb,
                        cancel, log_cb, von: float, bis: float) -> tuple:
        """Pipeline mit Lage fuer einen Arbeitsthread: (pipe, thermal_zuschlag, optik).

        Siehe ``core.meander.bereit_machen``; ``bauen`` wird nicht mehr
        gebraucht, die Pipeline baut ``core.meander.bauen``.
        """
        from core import meander as meander_mod
        return meander_mod.bereit_machen(pipe, args, th_zuschlag, optik_jetzt, progress_cb,
                                         cancel, log_cb, von, bis)

    def _maeander_kontext(self, titel: str, text: str, zusatz_ok: bool = True) -> dict | None:
        """Vorspann der Mäander-Jobs im GUI-Thread: Flug, Lage und Optik.

        Ohne Mäanderflug, Karte oder Projekt — oder wenn der Aufrufer mit
        ``zusatz_ok`` eine eigene Bedingung verneint — kommt ``text`` als
        Hinweis und None zurück, ebenso, wenn die Rekonstruktion abgelehnt
        wird. Sonst ist die Handjustage in die Pipeline geschrieben und das
        dict traegt pipe, args, th_zuschlag und optik_jetzt.
        """
        if not self._meander_dir or self._world is None or self._project is None \
                or not zusatz_ok:
            QMessageBox.information(self, titel, text)
            return None
        if not self._meander_ask_colmap():
            return None
        self._meander_apply_manual()
        pipe = self._meander_pipe
        args = self._meander_args()
        th_zuschlag = self._thermal_zuschlag()
        optik_jetzt = dict(self._optik, rgb_faktor=self._faktor("rgb"),
                           thermal_faktor=self._faktor("thermal"))
        return {"pipe": pipe, "args": args, "th_zuschlag": th_zuschlag,
                "optik_jetzt": optik_jetzt}

    def _meander_ask_colmap(self) -> bool:
        """Vor einer Rekonstruktion fragen — die dauert eine halbe Stunde."""
        from core import meander as meander_mod
        work = self._project.meander_work_dir()
        if meander_mod.hat_modell(work):
            return True
        if meander_mod.find_colmap_python() is None:
            self._show_error(
                "Mäander-Einfärbung",
                "Für die Rekonstruktion wird ein Interpreter mit pycolmap "
                "gebraucht, es ist keiner gefunden worden. Entweder pycolmap "
                "installieren oder ein fertiges COLMAP-Modell als sparse/0 in "
                f"'{work}' ablegen.")
            return False
        n = meander_mod.zaehle_rgb_jpeg(self._meander_dir)
        return QMessageBox.question(
            self, "Rekonstruktion nötig",
            f"Für diesen Flug gibt es noch kein COLMAP-Modell.\n\n"
            f"Die Rekonstruktion von {n} Bildern dauert etwa eine halbe Stunde. "
            f"Danach liegt sie im Arbeitsordner und wird wiederverwendet.\n\n"
            f"Jetzt rechnen?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes

    def _on_meander_align(self) -> None:
        if not self._meander_dir or self._world is None:
            QMessageBox.information(self, "Mäander-Einfärbung",
                                    "Erst einen Mäanderflug wählen.")
            return
        if not self._meander_ask_colmap():
            return
        args = self._meander_args()

        def job(progress_cb, cancel, log_cb):
            from core import meander as meander_mod
            pipe = meander_mod.bauen(args, log_cb)
            pipe._cancel = lambda: cancel.is_set()
            vor = meander_mod.prepare(
                pipe, progress=lambda f, m: progress_cb(0.05 + 0.7 * f, m))
            k = meander_mod.align(
                pipe, progress=lambda f, m: progress_cb(0.75 + 0.25 * f, m))
            return {"pipe": pipe, "vor": vor, "kennwerte": k}

        self._start_worker("Richte den Mäanderflug aus …", job,
                           self._on_meander_aligned)

    def _on_meander_aligned(self, res: dict) -> None:
        self._meander_pipe = res["pipe"]
        v, k = res["vor"], res["kennwerte"]
        anteil = k.get("anteil_auf_flaeche")
        med = k.get("median_abweichung")
        self._lbl_meander.setText(
            f"Ausgerichtet: {k['yaw_deg']:.2f}°"
            + (f", {anteil * 100:.0f} % der Fotopunkte auf der Oberfläche "
               f"(Median {med:.2f} m)" if anteil is not None else ""))
        self._log(f"Mäander: {v['kameras']} Kameras, Maßstab {v['massstab']:.3f}, "
                  f"GPS-Residuum {v['gps_residuum']:.2f} m.")
        self._log(f"Ausrichtung: {k['yaw_deg']:.2f}°, Versatz "
                  f"{np.round(k['t'], 2).tolist()} m.")
        from core import meander as meander_mod
        schlecht = meander_mod.pruefe_ausrichtung(k)
        if schlecht:
            self._log("WARNUNG: " + schlecht)
            self._lbl_meander.setText(
                f"Ausrichtung fraglich: {k['yaw_deg']:.2f}°, nur "
                f"{anteil * 100:.1f} % auf der Oberfläche.")
        for key in ("yaw", "x", "y", "z"):
            still_setzen(self._spin_meander[key], 0.0)
        if k.get("aus_cache", True):
            self._meander_lade_zuschlag()     # Handzuschlag von zuletzt
        self._meander_lade_thermal()
        self._meander_lade_optik()
        from core import optik as optik_mod
        if not k.get("aus_cache", True) and self._project is not None and \
                optik_mod.vorhanden(self._project.meander_work_dir()):
            # Neue Lage: Hoehenkorrektur und Feinausrichtung gehoerten zur
            # alten. Brennweite und Thermaloptik bleiben, die sind Kamera.
            self._optik["korrektur"] = None
            self._meander_pipe.s360_korrektur = None
            self._meander_speichere_optik()
            self._optik_neu_messen = True
            self._log("Neu ausgerichtet — Höhe und Feinausrichtung werden gleich "
                      "neu gemessen, die Optik bleibt.")
        self._update_enabled()
        self._start_live_preview()

    def _on_meander_einmessen(self) -> None:
        """Hoehe, RGB-Brennweite und Thermaloptik einmessen (s. core.optik)."""
        pipe = self._meander_pipe
        if not self._hat_lage or self._world is None:
            QMessageBox.information(self, "Optik einmessen",
                                    "Erst „Ausrichten“ laufen lassen.")
            return
        self._meander_apply_manual()          # auf der Lage aufsetzen, die man sieht
        self._optik_neu_messen = False
        yaw, t = self._meander_lage()
        welt = self._world
        thermal = self._hat_thermal()

        def job(progress_cb, cancel, log_cb):
            from core import optik as optik_mod
            punkte = welt[:: max(1, len(welt) // 400_000)]
            return optik_mod.einmessen(
                pipe, np.asarray(punkte, dtype=np.float64), yaw, t, pipe.photo_dir,
                thermal=thermal, progress=progress_cb,
                cancel=lambda: cancel.is_set(), log=log_cb)

        def fertig(res: dict) -> None:
            from core import meander as meander_mod
            if res["dz"]:
                y, tt = self._meander_lage()
                tt = tt + np.array([0.0, 0.0, res["dz"]])
                meander_mod.set_manual(self._meander_pipe, y, tt)
            neu = {"rgb_faktor": res["rgb"]["faktor"], "thermal_faktor": 1.0,
                   "thermal": res["thermal"], "hoehe": res["hoehe"],
                   "rgb": res["rgb"], "korrektur": self._optik.get("korrektur")}
            self._meander_setze_optik(neu)
            self._meander_speichere_optik()
            auf = res["rgb"].get("auf_flaeche_nachher")
            if auf is not None:
                self._lbl_meander.setText(
                    f"Eingemessen: {auf * 100:.0f} % der Fotopunkte auf der "
                    f"Oberfläche (vorher {res['rgb']['auf_flaeche_vorher'] * 100:.0f} %), "
                    f"RGB-Maßstab {res['rgb']['faktor']:.3f}"
                    + (", Thermal eingemessen." if res["thermal"] else "."))
            self._log("Optik eingemessen und gespeichert. Einfärben und Vorschau "
                      "benutzen sie ab jetzt; die Maßstab-Regler zeigen das "
                      "Ergebnis und lassen sich weiter von Hand nachziehen.")
            if self._cloud_view.has_color_preview():
                self._live_timer.start()
            self._update_enabled()
            if self._auto_kette:
                self._auto_weiter("fein")

        self._start_worker("Messe die Optik ein …", job, fertig)

    def _on_meander_fein(self) -> None:
        """Fotomodell fein auf die Karte legen (s. core.optik.feinausrichten)."""
        pipe = self._meander_pipe
        if not self._hat_lage or self._world is None:
            QMessageBox.information(self, "Feinausrichten",
                                    "Erst „Ausrichten“ laufen lassen.")
            return
        self._meander_apply_manual()
        yaw, t = self._meander_lage()
        welt = self._world
        faktor = self._faktor("rgb")

        def job(progress_cb, cancel, log_cb):
            from core import optik as optik_mod
            punkte = np.asarray(welt[:: max(1, len(welt) // 400_000)], dtype=np.float64)
            return optik_mod.feinausrichten(pipe, punkte, yaw, t, faktor,
                                            progress=progress_cb,
                                            cancel=lambda: cancel.is_set(), log=log_cb)

        def fertig(k: dict) -> None:
            self._optik["korrektur"] = k if k["stufe"] != "nichts" else None
            self._meander_setze_optik(self._optik)
            self._meander_speichere_optik()
            self._lbl_meander.setText(
                f"Feinausgerichtet ({k['stufe']}): {k['auf_flaeche_nachher'] * 100:.0f} % "
                f"der Fotopunkte auf der Oberfläche, Neigung "
                f"{k['neigung_grad'][0]:+.2f}°/{k['neigung_grad'][1]:+.2f}°, Versatz "
                f"{k['versatz_m'][0]:+.2f}/{k['versatz_m'][1]:+.2f} m.")
            if self._cloud_view.has_color_preview():
                self._live_timer.start()
            self._update_enabled()
            if self._auto_kette:
                self._auto_weiter("einfaerben")

        self._start_worker("Feinausrichtung läuft …", job, fertig)

    def _on_meander_auto(self) -> None:
        """Alles hintereinander: ausrichten, einmessen, feinausrichten, einfaerben.

        Jeder Schritt stoesst den naechsten aus seinem Fertig-Zweig an; schlaegt
        einer fehl, haelt ``_worker_failed`` die Kette an.
        """
        if not self._meander_dir or self._world is None:
            QMessageBox.information(self, "Automatisch",
                                    "Erst einen Flug öffnen und einen Mäanderflug wählen.")
            return
        if not self._meander_ask_colmap():
            return
        self._auto_kette = True
        self._log("Automatik: ausrichten → Optik einmessen → feinausrichten → "
                  "einfärben mit Sichtprüfung.")
        if not self._hat_lage or self._live is None:
            self._on_meander_align()      # weiter ueber die Vorschaubilder
        else:
            self._auto_weiter("einmessen")

    def _auto_weiter(self, schritt: str) -> None:
        if not self._auto_kette:
            return
        if schritt == "einmessen":
            self._on_meander_einmessen()
        elif schritt == "fein":
            self._on_meander_fein()
        elif schritt == "einfaerben":
            self._on_meander_run()
        else:
            self._auto_kette = False
            self._log("Automatik fertig.")

    def _on_meander_run(self) -> None:
        ktx = self._maeander_kontext("Mäander-Einfärbung", "Erst einen Mäanderflug wählen.")
        if ktx is None:
            return
        pipe, args = ktx["pipe"], ktx["args"]
        th_zuschlag, optik_jetzt = ktx["th_zuschlag"], ktx["optik_jetzt"]
        welt = self._world
        thermal = bool(self._chk_thermal.isChecked())
        proj = self._project
        sichtbar = bool(self._chk_sichtbar.isChecked())

        def job(progress_cb, cancel, log_cb):
            from core import meander as meander_mod
            from core import optik as optik_mod
            p, th, opt = meander_mod.bereit_machen(pipe, args, th_zuschlag, optik_jetzt,
                                                   progress_cb, cancel, log_cb, 0.02, 0.50)
            ergebnis = {"pipe": p, "ebenen": {}, "sichtbar": sichtbar}
            rf = float(opt["rgb_faktor"])
            if rf != 1.0:
                log_cb(f"RGB mit Maßstab {rf:.4f} (Brennweite).")
            if getattr(p, "s360_korrektur", None):
                k = p.s360_korrektur
                log_cb(f"Mit Feinausrichtung: Neigung {k['neigung_grad'][0]:+.2f}°/"
                       f"{k['neigung_grad'][1]:+.2f}°, Versatz {k['versatz_m'][0]:+.2f}/"
                       f"{k['versatz_m'][1]:+.2f} m.")
            normalen = None
            if sichtbar:
                from core import sichtbar as sichtbar_mod
                progress_cb(0.5, "Normalen der Karte für die Sichtprüfung …")
                normalen = sichtbar_mod.normalen(welt)

            def faerben(cams, ordner, A_, b_, von, bis, temperatur=None):
                prog = lambda f, m: progress_cb(von + (bis - von) * f, m)  # noqa: E731
                if sichtbar:
                    return sichtbar_mod.colorize_sichtbar(
                        welt, cams, ordner, A_, b_, normalen_welt=normalen,
                        progress=prog, cancel=lambda: cancel.is_set(), log=log_cb,
                        temperatur=temperatur)
                return meander_mod.colorize_points(
                    welt, cams, ordner, A_, b_, progress=prog,
                    cancel=lambda: cancel.is_set(), temperatur=temperatur)

            progress_cb(0.52, "Färbe die volle Wolke aus den RGB-Bildern …")
            # Thermal folgt unten, erst nach dem RGB-Färben
            kam = meander_mod.kameras(p, opt, th, thermal=False)
            rgb, maske = faerben(kam["rgb"], p._p("images"), kam["A"], kam["b"],
                                 0.52, 0.80)
            meander_mod.save_layer(
                proj.layer_dir("meander_rgb"), rgb, maske,
                {"quelle": "meander_rgb", "flug": p.photo_dir,
                 "yaw_deg": kam["yaw_deg"],
                 "t": [float(x) for x in meander_mod.as_t3(p.t)],
                 "rgb_faktor": rf,
                 "anteil": float(maske.mean()),
                 "rgb_versatz": [float(x) for x in p.rgb_versatz],
                 "sichtpruefung": sichtbar,
                 "feinausrichtung": getattr(p, "s360_korrektur", None)})
            ergebnis["ebenen"]["meander_rgb"] = float(maske.mean())
            tf = kam["thermal_faktor"]
            th_cams = optik_mod.thermal_cams(p, opt.get("thermal"), tf, rf) \
                if thermal else None
            if th_cams is not None:
                progress_cb(0.82, "Färbe aus den Thermalbildern …")
                yaw_th, t_th = meander_mod.thermal_lage(kam["yaw_deg"], p.t, th)
                A_th, b_th = meander_mod.lage_affine(p, yaw_th, t_th)
                if any(th):
                    log_cb(f"Thermal mit eigener Lage: {th[0]:+.3f}°, "
                           f"{th[1]:+.2f}/{th[2]:+.2f} m auf die RGB-Lage.")
                log_cb("Thermal mit eingemessener Optik (Brennweite, Verzeichnung, "
                       "Schielwinkel)." if opt.get("thermal") else
                       "Thermal ohne Einmessung — nur EXIF-Brennweite. „Optik "
                       "einmessen“ macht das deutlich besser.")
                from core import temperatur as temperatur_mod
                tquelle = temperatur_mod.quelle(p)
                erg = faerben(th_cams, p._p("thermal"), A_th, b_th, 0.82, 0.98,
                              temperatur=tquelle)
                trgb, tmaske = erg[0], erg[1]
                ttemp = erg[2] if len(erg) > 2 else None
                if ttemp is not None and np.isfinite(ttemp).any():
                    tt = ttemp[np.isfinite(ttemp)]
                    log_cb(f"Temperaturen aus den R-JPEG: {len(tt) / len(ttemp) * 100:.0f} % "
                           f"der Punkte, Median {np.median(tt):.1f} °C.")
                elif tquelle is None:
                    log_cb("Keine Temperaturen: die Thermalbilder tragen keine "
                           "Rohwerte (kein radiometrisches JPEG).")
                meander_mod.save_layer(
                    proj.layer_dir("meander_thermal"), trgb, tmaske,
                    {"quelle": "meander_thermal", "flug": p.photo_dir,
                     "yaw_deg": float(yaw_th),
                     "thermal_zuschlag": [float(x) for x in th],
                     "thermal_faktor": tf,
                     "thermal_eingemessen": bool(opt.get("thermal")),
                     "sichtpruefung": sichtbar,
                     "feinausrichtung": getattr(p, "s360_korrektur", None),
                     "anteil": float(tmaske.mean()),
                     "thermal_versatz": [float(x) for x in p.thermal_versatz],
                     "temperatur": ttemp is not None},
                    temperatur=ttemp)
                ergebnis["ebenen"]["meander_thermal"] = float(tmaske.mean())
            elif thermal:
                log_cb("Thermal übersprungen: die Optik fehlt (keine Brennweite "
                       "im EXIF der _T.JPG).")
            progress_cb(1.0, "Mäander-Einfärbung fertig")
            return ergebnis

        self._start_worker("Mäander-Einfärbung läuft …", job, self._on_meander_done)

    def _on_meander_done(self, res: dict) -> None:
        self._meander_pipe = res["pipe"]
        self._meander_lade_thermal()     # Lief ohne Pipeline: Wert aus dem Projekt
        self._meander_lade_optik()
        self._ebenen_melden(res["ebenen"])
        if res.get("sichtbar"):
            self._log("Ungefärbt ist, was keine Kamera sieht — mit „Nur eingefärbte "
                      "Punkte“ ausgeblendet, von oben fehlt dadurch nichts.")
        elif res["ebenen"].get("meander_thermal", 1.0) < 0.9:
            self._log("Der Rest liegt außerhalb der Thermalbilder — die sehen "
                      "einen schmaleren Ausschnitt als die RGB-Kamera.")
        self._live_hide()
        self._reload_layers()
        if self._live is None and self._meander_pipe is not None:
            self._start_live_preview()   # Nachjustieren soll sofort wirken
        self._ebene_waehlen("meander_rgb")
        self._update_enabled()
        if self._auto_kette:
            self._auto_weiter("fertig")

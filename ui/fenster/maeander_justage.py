"""Mäander-Lage, Optik und Ausrichtfenster (Mixin des Hauptfensters).

Von Hand justiert wird nur im Ausrichtfenster (``ui.meander_align_window``).
Hier liegen Lage und Optik des Mäanderfluges als Zustand: der Thermal-Zuschlag
auf die RGB-Lage und die Maßstäbe, beide auf dem Raster der Regler im Fenster
(``ui.feinregler``), dazu die Dateien im Arbeitsordner und die verkleinerten
Bilder für die Farbvorschau des Fensters.

Schreibt am Hauptfenster: _live, _live_th, _meander_fenster, _meander_pipe,
_optik, _optik_neu_messen, _th_zuschlag.
"""
from __future__ import annotations

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from ui.feinregler import raster_grad, raster_meter, raster_prozent


class MaeanderJustageMixin:
    def _start_live_preview(self) -> None:
        """Verkleinerte Bilder für die Farbvorschau des Ausrichtfensters laden."""
        pipe = self._meander_pipe
        if pipe is None or pipe.cams is None or self._world is None:
            return
        cams = pipe.rgb_cams()
        bilder = pipe._p("images")
        th_cams = pipe.thermal_cams() if self._hat_thermal() else None
        th_bilder = pipe._p("thermal")

        def job(progress_cb, cancel, log_cb):
            from core import meander as meander_mod
            anteil = 0.7 if th_cams is not None else 1.0
            live = meander_mod.LivePreview(
                cams, bilder, progress=lambda f, m: progress_cb(anteil * f, m),
                cancel=lambda: cancel.is_set())
            live_th = None
            if th_cams is not None:
                live_th = meander_mod.LivePreview(
                    th_cams, th_bilder,
                    progress=lambda f, m: progress_cb(0.7 + 0.3 * f,
                                                      "Thermal: " + m),
                    cancel=lambda: cancel.is_set())
            return {"live": live, "live_th": live_th}

        def fertig(res: dict) -> None:
            self._live = res["live"]
            self._live_th = res["live_th"]
            th = (f" und {len(res['live_th'].bilder)} Thermalbilder"
                  if res["live_th"] is not None else "")
            self._log(f"Vorschaubilder geladen: {len(res['live'].bilder)} verkleinerte "
                      f"RGB-Bilder{th} — für die Farbvorschau unter „Im Fenster "
                      f"justieren …“.")
            self._update_enabled()
            # Die Optik einmessen, wenn das fuer diesen Flug noch nie
            # geschehen ist — ohne sie sind die Bilder zueinander verzerrt.
            from core import optik as optik_mod
            if self._auto_kette:
                self._auto_weiter("einmessen")
            elif self._project is not None and (self._optik_neu_messen or not
                                                optik_mod.vorhanden(
                                                    self._project.meander_work_dir())):
                self._log("Die Optik dieses Fluges ist noch nicht eingemessen — "
                          "das läuft jetzt einmal von selbst."
                          if not self._optik_neu_messen else
                          "Messe Höhe und Brennweite zur neuen Lage …")
                self._optik_neu_messen = False
                self._on_meander_einmessen()

        self._start_worker("Lade Vorschaubilder für die Handjustage …",
                           job, fertig)

    def _freigabe_maeander(self, zustand: dict, busy: bool) -> None:
        """Lagetext der Seitenleiste nachziehen (aufgerufen von _update_enabled)."""
        self._lbl_meander_lage.setText(self._meander_zustand_text())

    def _meander_zustand_text(self) -> str:
        """Welche Lage und Optik gilt — oder was dafür noch fehlt."""
        if not self._meander_dir:
            return "Noch kein Mäanderflug gewählt."
        if not self._hat_lage:
            return ("Noch nicht ausgerichtet — erst „Ausrichten“, danach lässt sich "
                    "die Lage unter „Im Fenster justieren …“ von Hand nachziehen.")
        text = (f"Ausgerichtet auf {np.degrees(self._meander_pipe.yaw):.2f}°, "
                f"RGB-Maßstab {self._faktor('rgb'):.4f}.")
        if self._hat_thermal():
            dg, dx, dy = self._thermal_zuschlag()
            text += (f" Thermal: {dg:+.3f}°, {dx:+.2f}/{dy:+.2f} m auf die RGB-Lage, "
                     f"Maßstab {self._faktor('thermal'):.4f}.")
        if self._live is None:
            text += " Die Vorschaubilder für das Fenster werden noch geladen …"
        return text

    def _meander_lage(self) -> tuple:
        """Ausgerichtete Lage: (yaw_grad, t als 3er-Vektor).

        Die Hoehe muss mitgefuehrt werden: ``register.affine`` rechnet mit drei
        Komponenten, ein 2er-Vektor bricht dort ab.
        """
        from core import meander as meander_mod
        pipe = self._meander_pipe
        return float(np.degrees(pipe.yaw)), meander_mod.as_t3(pipe.t)

    def _faktor(self, optik: str) -> float:
        """Massstab der Optik als Brennweitenfaktor (1.0 = unveraendert)."""
        return float(self._optik.get(f"{optik}_faktor", 1.0))

    def _meander_setze_optik(self, d: dict) -> None:
        """Optik uebernehmen; die Massstaebe auf das Raster der Regler legen."""
        self._optik = dict(d)
        if self._meander_pipe is not None:
            self._meander_pipe.s360_korrektur = d.get("korrektur")
        for optik in ("rgb", "thermal"):
            prozent = (float(d.get(f"{optik}_faktor", 1.0)) - 1.0) * 100.0
            self._optik[f"{optik}_faktor"] = 1.0 + raster_prozent(prozent) / 100.0

    def _meander_lade_optik(self) -> None:
        if self._project is None:
            return
        from core import optik as optik_mod
        d = optik_mod.laden(self._project.meander_work_dir())
        self._meander_setze_optik(d)
        if optik_mod.vorhanden(self._project.meander_work_dir()):
            self._log(f"Optik aus dem Projekt: RGB-Maßstab {d['rgb_faktor']:.4f}, "
                      f"Thermal {'eingemessen' if d.get('thermal') else 'nicht eingemessen'}"
                      f", Maßstab {d['thermal_faktor']:.4f}.")

    def _meander_speichere_optik(self) -> None:
        if self._project is None or self._meander_pipe is None:
            return
        from core import optik as optik_mod
        try:
            optik_mod.speichern(self._project.meander_work_dir(), self._optik)
        except OSError as exc:
            self._log(f"Optik nicht gespeichert: {exc}")

    def _thermal_zuschlag(self) -> tuple:
        """Thermal-Zuschlag (Gier Grad, X m, Y m) auf die RGB-Lage."""
        return self._th_zuschlag

    @property
    def _hat_lage(self) -> bool:
        """Gibt es eine ausgerichtete Lage, auf die sich die Justage bezieht?"""
        pipe = self._meander_pipe
        return pipe is not None and getattr(pipe, "yaw", None) is not None

    def _hat_thermal(self) -> bool:
        pipe = self._meander_pipe
        if pipe is None:
            return False
        try:
            return pipe.thermal_cams() is not None
        except Exception:  # noqa: BLE001
            return False

    def _meander_setze_thermal(self, zuschlag) -> None:
        """Thermal-Zuschlag uebernehmen, auf dem Raster der Regler."""
        dgier, dx, dy = (float(v) for v in zuschlag)
        self._th_zuschlag = (raster_grad(dgier), raster_meter(dx), raster_meter(dy))

    def _meander_lade_thermal(self) -> None:
        """Gespeicherten Thermal-Zuschlag des Projekts uebernehmen."""
        if self._project is None:
            return
        from core import meander as meander_mod
        z = meander_mod.load_thermal_zuschlag(self._project.meander_work_dir())
        self._meander_setze_thermal(z)
        if any(z):
            self._log(f"Thermal-Lage aus dem Projekt: {z[0]:+.2f}°, "
                      f"{z[1]:+.2f}/{z[2]:+.2f} m auf die RGB-Lage.")

    def _meander_speichere_zuschlag(self) -> None:
        """RGB-Zuschlag ins Projekt: null, die ganze Lage steht in align.json.

        Die Datei bleibt, damit ein aelterer Programmstand, der sie als
        Zuschlag auf die Lage liest, nichts doppelt rechnet.
        """
        if self._project is None or self._meander_pipe is None:
            return
        from core import meander as meander_mod
        d = dict.fromkeys(("yaw", "x", "y", "z"), 0.0)
        try:
            meander_mod.save_rgb_zuschlag(self._project.meander_work_dir(), d)
        except OSError as exc:
            self._log(f"Handzuschlag nicht gespeichert: {exc}")

    def _meander_lade_zuschlag(self) -> None:
        """Handzuschlag eines aelteren Programmstands sofort in die Lage rechnen.

        Er kommt auf das Raster der Regler, die ihn damals hielten; danach
        steht in der Datei null.
        """
        if self._project is None or not self._hat_lage:
            return
        from core import meander as meander_mod
        d = meander_mod.load_rgb_zuschlag(self._project.meander_work_dir())
        if d is None:
            return      # Datei fehlt oder unlesbar: die Lage bleibt, wie sie ist
        if not any(float(d.get(k, 0.0)) for k in ("yaw", "x", "y", "z")):
            return
        yaw, t = self._meander_lage()
        t[0] += raster_meter(float(d.get("x", 0.0)))
        t[1] += raster_meter(float(d.get("y", 0.0)))
        t[2] += raster_meter(float(d.get("z", 0.0)), 50.0)
        meander_mod.set_manual(self._meander_pipe,
                               yaw + raster_grad(float(d.get("yaw", 0.0))), t)
        self._log(f"Handzuschlag von zuletzt: Gier {d.get('yaw', 0):+.3f}°, "
                  f"X {d.get('x', 0):+.2f}, Y {d.get('y', 0):+.2f}, "
                  f"Z {d.get('z', 0):+.2f} m.")
        self._meander_speichere_zuschlag()

    def _meander_speichere_thermal(self) -> None:
        if self._project is None or self._meander_pipe is None:
            return
        from core import meander as meander_mod
        try:
            meander_mod.save_thermal_zuschlag(self._project.meander_work_dir(),
                                              self._thermal_zuschlag())
        except OSError as exc:
            self._log(f"Thermal-Lage nicht gespeichert: {exc}")

    def _on_meander_fenster(self) -> None:
        """Ausrichtfenster oeffnen: Karte und Flug uebereinander, von Hand
        justierbar. Es gibt hoechstens eines; ist es offen, kommt es nach vorn."""
        offen = self._meander_fenster
        if offen is not None:
            if self._meander_fenster_gilt(offen):
                offen.show()
                offen.raise_()
                offen.activateWindow()
                return
            # Seit dem Oeffnen wurde neu ausgerichtet, eingemessen oder ein
            # anderer Flug gewaehlt: Das alte Fenster rechnet mit der alten
            # Lage und schriebe sie mit „Lage übernehmen“ zurück.
            offen.close()
            self._meander_fenster = None
            self._log("Das offene Ausrichtfenster zeigte eine ältere Lage — "
                      "es wird durch ein neues ersetzt.")
        pipe = self._meander_pipe
        if not self._hat_lage:
            QMessageBox.information(
                self, "Im Fenster justieren",
                "Erst „Ausrichten“ laufen lassen — das Fenster zeigt die "
                "gefundene Lage und lässt sie von Hand nachziehen.")
            return
        if self._world is None:
            return
        from ui.meander_align_window import MeanderAlignWindow
        # Die Lage erst festschreiben: das Fenster setzt auf der auf, die gilt.
        self._meander_apply_manual()
        fenster = MeanderAlignWindow(
            self._world, pipe, self._live, live_th=self._live_th,
            thermal_zuschlag=self._thermal_zuschlag(),
            thermal_da=self._hat_thermal(), optik=dict(self._optik), parent=self)
        fenster.uebernommen.connect(
            lambda e, f=fenster: self._on_meander_fenster_lage(e, f))
        # Geschlossen ist es vergessen; der naechste Aufruf baut ein neues.
        fenster.finished.connect(lambda *_a, f=fenster: self._meander_fenster_zu(f))
        fenster.destroyed.connect(lambda *_a, f=fenster: self._meander_fenster_zu(f))
        fenster.setAttribute(Qt.WA_DeleteOnClose, True)
        self._meander_fenster = fenster       # Referenz halten, sonst weg
        fenster.show()
        if self._live is None:
            self._log("Das Ausrichtfenster zeigt die Überlagerung. Für die "
                      "Farbvorschau darin werden die Vorschaubilder gebraucht — "
                      "die lädt „Ausrichten“ im Anschluss.")

    def _meander_fenster_zu(self, fenster) -> None:
        if self._meander_fenster is fenster:
            self._meander_fenster = None

    def _meander_fenster_gilt(self, fenster, bilder: bool = True) -> bool:
        """Rechnet das Ausrichtfenster noch mit der Lage, die jetzt gilt?

        Das Fenster friert beim Oeffnen Pipeline, Lage, Massstaebe,
        Thermal-Zuschlag und Vorschaubilder ein. Verglichen wird mit einer
        kleinen Toleranz: Werte auf halbem Reglerraster wechseln beim
        erneuten Rastern um ein Bit. Mit ``bilder=False`` zaehlen nachgeladene
        Vorschaubilder nicht – fuer die Uebernahme der Lage sind sie egal.
        """
        from core import meander as meander_mod
        pipe = self._meander_pipe
        if fenster._pipe is not pipe or not self._hat_lage:
            return False
        if bilder and (fenster._live is not self._live
                       or fenster._live_th is not self._live_th):
            return False
        jetzt = [float(np.degrees(pipe.yaw)), *meander_mod.as_t3(pipe.t),
                 *(float(v) for v in self._thermal_zuschlag()),
                 float(self._optik.get("rgb_faktor", 1.0)),
                 float(self._optik.get("thermal_faktor", 1.0))]
        damals = [fenster._basis_yaw, *fenster._basis_t, *fenster._start["th"],
                  fenster._start["rgb_faktor"], fenster._start["thermal_faktor"]]
        return bool(np.allclose(jetzt, damals, rtol=0.0, atol=1e-9))

    def _on_meander_fenster_lage(self, e: dict, fenster=None) -> None:
        """Lage aus dem Ausrichtfenster uebernehmen: RGB als neue Basis,
        Thermal als Zuschlag darauf, dazu beide Massstaebe."""
        from core import meander as meander_mod
        if fenster is not None and not self._meander_fenster_gilt(fenster, bilder=False):
            # Inzwischen gilt eine andere Lage oder ein anderer Flug: die
            # Werte des Fensters beziehen sich nicht mehr darauf.
            self._log("Lage aus dem Ausrichtfenster nicht übernommen — seit dem "
                      "Öffnen gilt eine andere Lage. Bitte das Fenster neu öffnen.")
            QMessageBox.information(
                self, "Im Fenster justieren",
                "Seit dem Öffnen des Fensters wurde neu ausgerichtet, eingemessen "
                "oder ein anderer Flug gewählt. Die Lage aus dem Fenster wird "
                "deshalb nicht übernommen — bitte „Im Fenster justieren …“ neu "
                "öffnen.")
            return
        yaw_deg, t, th_zuschlag = float(e["yaw"]), e["t"], e["thermal"]
        meander_mod.set_manual(self._meander_pipe, yaw_deg, t)
        self._meander_setze_thermal(th_zuschlag)
        self._meander_speichere_thermal()
        self._meander_setze_optik(dict(self._optik, rgb_faktor=e["rgb_faktor"],
                                       thermal_faktor=e["thermal_faktor"]))
        self._meander_speichere_optik()
        t = np.asarray(t, dtype=float)
        self._log(f"Lage aus dem Ausrichtfenster übernommen: Gier {yaw_deg:.3f}°, "
                  f"Versatz {t[0]:+.2f}/{t[1]:+.2f}/{t[2]:+.2f} m, RGB-Maßstab "
                  f"{e['rgb_faktor']:.4f}.")
        if self._hat_thermal():
            dg, dx, dy = th_zuschlag
            self._log(f"Thermal: {dg:+.3f}°, {dx:+.2f}/{dy:+.2f} m auf die RGB-Lage, "
                      f"Maßstab {e['thermal_faktor']:.4f}.")
        self._lbl_meander_lage.setText(self._meander_zustand_text())

    def _meander_apply_manual(self) -> None:
        """Lage festschreiben (align.json), dazu RGB- und Thermal-Zuschlag ins Projekt."""
        pipe = self._meander_pipe
        if not self._hat_lage:
            return
        from core import meander as meander_mod
        yaw, t = self._meander_lage()
        meander_mod.set_manual(pipe, yaw, t)
        self._meander_speichere_zuschlag()
        # Thermal bleibt ein Zuschlag auf die RGB-Lage
        self._meander_speichere_thermal()

    def _live_clear(self) -> None:
        self._live = None
        self._live_th = None

"""Mäander-Justage und Live-Vorschau (Mixin des Hauptfensters).

Schreibt am Hauptfenster: _live, _live_gemeckert, _live_optik, _live_pts,
_live_th, _meander_fenster, _meander_pipe, _optik, _optik_neu_messen.
"""
from __future__ import annotations

import time
import traceback
from typing import Optional

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from core.gemeinsam import fmt_int as _fmt_int
from ui.bausteine import still_setzen

#: So viele Punkte gehen in die Live-Vorschau der Handjustage. Bei 50.000
#: dauert ein Durchlauf rund 80 ms — schnell genug, um dem Regler zu folgen.
_LIVE_PUNKTE = 50_000


class MaeanderJustageMixin:
    def _start_live_preview(self) -> None:
        """Verkleinerte Bilder laden, damit die Handjustage live wirkt."""
        pipe = self._meander_pipe
        if pipe is None or pipe.cams is None or self._world is None:
            return
        welt = self._world
        n = len(welt)
        schritt = max(1, n // _LIVE_PUNKTE)
        stich = np.ascontiguousarray(welt[::schritt][:_LIVE_PUNKTE],
                                     dtype=np.float64)
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
            return {"live": live, "live_th": live_th, "pts": stich}

        def fertig(res: dict) -> None:
            self._live = res["live"]
            self._live_th = res["live_th"]
            self._live_pts = res["pts"]
            th = (f" und {len(res['live_th'].bilder)} Thermalbilder"
                  if res["live_th"] is not None else "")
            self._log(f"Live-Vorschau bereit: {_fmt_int(len(res['pts']))} "
                      f"Punkte, {len(res['live'].bilder)} verkleinerte RGB-Bilder"
                      f"{th}. Sie erscheint beim ersten Zug an Gier, X oder Y — "
                      f"RGB-Regler zeigen RGB, Thermal-Regler Thermal — und "
                      f"blendet die Karte dabei aus (Haken darüber schaltet "
                      f"das ab). Die Karte bleibt bis dahin stehen.")
            self._update_enabled()
            # Bewusst KEIN _live_update hier: die Karte soll nach dem
            # Ausrichten stehen bleiben. Wer nichts justiert, will sie sehen.
            # Aber die Optik einmessen, wenn das fuer diesen Flug noch nie
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

    def _on_solo_changed(self) -> None:
        self._cloud_view.set_preview_solo(bool(self._chk_solo.isChecked()))
        self._lbl_meander_lage.setText(self._meander_zustand_text())
        self._save_settings()

    def _meander_zustand_text(self) -> str:
        """Was die Handregler gerade koennen — und was fehlt, wenn nicht."""
        if not self._meander_dir:
            return "Noch kein Mäanderflug gewählt."
        pipe = self._meander_pipe
        if not self._hat_lage:
            return ("Noch nicht ausgerichtet — die Regler brauchen eine Lage, "
                    "auf die sie sich beziehen. Erst „Ausrichten“.")
        if self._live is None:
            return (f"Ausgerichtet auf {np.degrees(pipe.yaw):.2f}°. "
                    f"Vorschaubilder werden noch geladen …")
        if self._cloud_view.has_color_preview():
            wo = ("die volle Karte ist solange ausgeblendet"
                  if not self._cloud_view.map_visible()
                  else "über der vollen Karte")
            optik = "Thermal" if self._live_optik == "thermal" else "RGB"
            return (f"Ausgerichtet auf {np.degrees(pipe.yaw):.2f}°. Gezeigt wird "
                    f"die {optik}-Vorschau aus {_fmt_int(len(self._live_pts))} "
                    f"Punkten, {wo}. Magenta sind die Kamerastandorte, Grau ist "
                    f"von keinem Bild getroffen.")
        return (f"Ausgerichtet auf {np.degrees(pipe.yaw):.2f}°, Live-Vorschau "
                f"mit {_fmt_int(len(self._live_pts))} Punkten bereit.")

    def _meander_lage(self) -> tuple:
        """Ausgerichtete Lage plus Handjustage: (yaw_grad, t als 3er-Vektor).

        Die Hoehe bleibt stehen — geregelt werden nur Gier, X und Y. Sie muss
        aber mitgeführt werden: ``register.affine`` rechnet mit drei
        Komponenten, ein 2er-Vektor bricht dort ab.
        """
        from core import meander as meander_mod
        pipe = self._meander_pipe
        t = meander_mod.as_t3(pipe.t)
        t[0] += float(self._spin_meander["x"].value())
        t[1] += float(self._spin_meander["y"].value())
        t[2] += float(self._spin_meander["z"].value())
        return (float(np.degrees(pipe.yaw)) + float(self._spin_meander["yaw"].value()),
                t)

    def _faktor(self, optik: str) -> float:
        """Massstab-Regler als Brennweitenfaktor (0 % = 1.0)."""
        return 1.0 + float(self._massstab[optik].value()) / 100.0

    def _meander_cams(self, thermal: bool) -> Optional[dict]:
        """Kameras mit dem Massstab der Regler und der Thermal-Einmessung."""
        from core import optik as optik_mod
        pipe = self._meander_pipe
        if thermal:
            return optik_mod.thermal_cams(pipe, self._optik.get("thermal"),
                                          self._faktor("thermal"), self._faktor("rgb"))
        return optik_mod.rgb_cams(pipe, self._faktor("rgb"))

    def _meander_setze_optik(self, d: dict) -> None:
        """Optik uebernehmen und die Massstab-Regler setzen, ohne Vorschau."""
        self._optik = dict(d)
        if self._meander_pipe is not None:
            self._meander_pipe.s360_korrektur = d.get("korrektur")
        for optik in ("rgb", "thermal"):
            still_setzen(self._massstab[optik],
                         (float(d.get(f"{optik}_faktor", 1.0)) - 1.0) * 100.0)

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
        self._optik["rgb_faktor"] = self._faktor("rgb")
        self._optik["thermal_faktor"] = self._faktor("thermal")
        try:
            optik_mod.speichern(self._project.meander_work_dir(), self._optik)
        except OSError as exc:
            self._log(f"Optik nicht gespeichert: {exc}")

    def _on_meander_massstab(self, optik: str) -> None:
        """Massstab geaendert: merken und die Vorschau dieser Optik zeigen."""
        self._meander_speichere_optik()
        self._on_meander_manual(optik)

    def _thermal_zuschlag(self) -> tuple:
        """Thermal-Regler: (Gier Grad, X m, Y m) als Zuschlag auf die RGB-Lage."""
        return tuple(float(self._spin_meander_th[k].value()) for k in ("yaw", "x", "y"))

    def _meander_lage_thermal(self) -> tuple:
        """Lage der Thermalbilder: RGB-Lage samt Handjustage plus Thermal-Zuschlag."""
        from core import meander as meander_mod
        return meander_mod.thermal_lage(*self._meander_lage(), self._thermal_zuschlag())

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
        """Thermal-Regler setzen, ohne eine Vorschau auszuloesen."""
        for key, wert in zip(("yaw", "x", "y"), zuschlag):
            still_setzen(self._spin_meander_th[key], float(wert))

    def _meander_lade_thermal(self) -> None:
        """Gespeicherten Thermal-Zuschlag des Projekts in die Regler holen."""
        if self._project is None:
            return
        from core import meander as meander_mod
        z = meander_mod.load_thermal_zuschlag(self._project.meander_work_dir())
        self._meander_setze_thermal(z)
        if any(z):
            self._log(f"Thermal-Lage aus dem Projekt: {z[0]:+.2f}°, "
                      f"{z[1]:+.2f}/{z[2]:+.2f} m auf die RGB-Lage.")

    def _meander_speichere_zuschlag(self) -> None:
        """RGB-Handzuschlag sofort ins Projekt — nichts geht beim Schliessen verloren."""
        if self._project is None or self._meander_pipe is None:
            return
        from core import meander as meander_mod
        d = {k: float(self._spin_meander[k].value()) for k in ("yaw", "x", "y", "z")}
        try:
            meander_mod.save_rgb_zuschlag(self._project.meander_work_dir(), d)
        except OSError as exc:
            self._log(f"Handzuschlag nicht gespeichert: {exc}")

    def _meander_lade_zuschlag(self) -> None:
        if self._project is None:
            return
        from core import meander as meander_mod
        d = meander_mod.load_rgb_zuschlag(self._project.meander_work_dir())
        if d is None:
            return      # Datei fehlt oder unlesbar: Regler bleiben, wie sie sind
        if any(float(d.get(k, 0.0)) for k in ("yaw", "x", "y", "z")):
            for k in ("yaw", "x", "y", "z"):
                still_setzen(self._spin_meander[k], float(d.get(k, 0.0)))
            self._log(f"Handzuschlag von zuletzt: Gier {d.get('yaw', 0):+.3f}°, "
                      f"X {d.get('x', 0):+.2f}, Y {d.get('y', 0):+.2f}, "
                      f"Z {d.get('z', 0):+.2f} m.")

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
        """Ausrichtfenster oeffnen: Karte und Flug uebereinander, live justierbar."""
        pipe = self._meander_pipe
        if not self._hat_lage:
            QMessageBox.information(
                self, "Überlagern",
                "Erst „Ausrichten“ laufen lassen — das Fenster zeigt die "
                "gefundene Lage und lässt sie von Hand nachziehen.")
            return
        if self._world is None:
            return
        from ui.meander_align_window import MeanderAlignWindow
        # Den Handzuschlag aus dem Hauptfenster erst verrechnen: das Fenster
        # setzt auf der Lage auf, die man gerade sieht, nicht auf der alten.
        self._meander_apply_manual()
        optik = dict(self._optik, rgb_faktor=self._faktor("rgb"),
                     thermal_faktor=self._faktor("thermal"))
        fenster = MeanderAlignWindow(
            self._world, pipe, self._live, live_th=self._live_th,
            thermal_zuschlag=self._thermal_zuschlag(),
            thermal_da=self._hat_thermal(), optik=optik, parent=self)
        fenster.uebernommen.connect(self._on_meander_fenster_lage)
        fenster.setAttribute(Qt.WA_DeleteOnClose, True)
        self._meander_fenster = fenster       # Referenz halten, sonst weg
        fenster.show()
        if self._live is None:
            self._log("Das Ausrichtfenster zeigt die Überlagerung. Für die "
                      "Farbvorschau darin werden die Vorschaubilder gebraucht — "
                      "die lädt „Ausrichten“ im Anschluss.")

    def _on_meander_fenster_lage(self, e: dict) -> None:
        """Lage aus dem Ausrichtfenster uebernehmen: RGB als neue Basis,
        Thermal als Zuschlag darauf, dazu beide Massstaebe."""
        from core import meander as meander_mod
        yaw_deg, t, th_zuschlag = float(e["yaw"]), e["t"], e["thermal"]
        meander_mod.set_manual(self._meander_pipe, yaw_deg, t)
        for sp in self._spin_meander.values():
            still_setzen(sp, 0.0)
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
        if self._cloud_view.has_color_preview():
            self._live_timer.start()     # sichtbare Vorschau auf die neue Lage
        self._lbl_meander_lage.setText(self._meander_zustand_text())

    def _on_meander_manual(self, optik: str = "rgb") -> None:
        """Handjustage anwenden und die Vorschau nachziehen.

        Die Basislage bleibt stehen, die Regler sind ein Zuschlag darauf —
        sonst wuerde jeder Reglerzug auf dem vorigen aufbauen und man kaeme nie
        zurueck. Neu gerechnet wird erst nach kurzer Ruhe (Timer), damit ein
        Ziehen nicht Dutzende Durchlaeufe ausloest. Gezeigt wird die Optik,
        an deren Regler zuletzt gedreht wurde.
        """
        if self._meander_pipe is None or self._meander_pipe.yaw is None:
            self._log("Handjustage ohne Wirkung: es gibt noch keine "
                      "Ausrichtung. Erst „Ausrichten“ laufen lassen.")
            return
        if optik == "thermal":
            self._meander_speichere_thermal()
        else:
            self._zuschlag_timer.start()      # kurz nach dem Zug speichern
        self._live_optik = optik
        live = self._live_th if optik == "thermal" else self._live
        if live is None:
            self._log("Die Vorschaubilder sind noch nicht geladen — die "
                      "Regler wirken, sobald sie da sind.")
            return
        self._live_timer.start()

    def _live_update(self) -> None:
        """Stichprobe mit der aktuellen Lage einfaerben und anzeigen."""
        pipe = self._meander_pipe
        if pipe is None or pipe.yaw is None or self._live is None \
                or self._live_pts is None:
            return
        from core import meander as meander_mod
        thermal = self._live_optik == "thermal" and self._live_th is not None
        live = self._live_th if thermal else self._live
        yaw, t = self._meander_lage_thermal() if thermal else self._meander_lage()
        try:
            # Die Basis bleibt, die Regler sind nur ein Zuschlag
            A, b = meander_mod.lage_affine(pipe, yaw, t)
            t0 = time.perf_counter()
            rgb, maske = live.colorize(self._live_pts, A, b,
                                       cams=self._meander_cams(thermal))
        except Exception as exc:  # noqa: BLE001
            # Ohne diesen Fang verschluckt Qt den Fehler im Timer-Slot und der
            # Regler sieht aus, als bewirke er nichts.
            self._log(f"Live-Vorschau fehlgeschlagen: {exc}")
            self._log(traceback.format_exc())
            self._live_timer.stop()
            return
        rgb = rgb.copy()
        rgb[~maske] = 60          # nicht getroffen: dunkel, nicht Fallback-grau
        anteil = float(maske.mean())

        # Kamerastandorte mit einzeichnen. Sieht man nur graue Punkte, ist die
        # erste Frage, ob der Flug ueberhaupt ueber dieser Wolke lag — und das
        # beantwortet ein Blick auf die Standorte sofort.
        punkte, farben = self._live_pts, rgb
        try:
            C = np.asarray(self._meander_pipe.cams["C"], dtype=np.float64)
            C_welt = (np.asarray(A) @ C.T).T + np.asarray(b)
            punkte = np.vstack([self._live_pts, C_welt])
            farben = np.vstack([rgb, np.tile(np.array([255, 0, 200], np.uint8),
                                             (len(C_welt), 1))])
        except Exception:  # noqa: BLE001 — ohne Kameras eben nur die Punkte
            C_welt = None
        self._cloud_view.set_color_preview(
            punkte, farben, solo=bool(self._chk_solo.isChecked()))

        if anteil < 0.05 and not self._live_gemeckert:
            self._live_gemeckert = True
            hinweis = ("Die Vorschau trifft fast nichts: nur "
                       f"{100.0 * anteil:.1f} % der Punkte liegen in einem Bild. "
                       "Alles Graue ist ungetroffen.")
            if C_welt is not None and len(self._live_pts):
                mitte_w = self._live_pts.mean(axis=0)
                mitte_c = C_welt.mean(axis=0)
                abstand = float(np.linalg.norm(mitte_c[:2] - mitte_w[:2]))
                ausdehnung = float(np.linalg.norm(
                    self._live_pts[:, :2].max(0) - self._live_pts[:, :2].min(0)))
                hinweis += (f" Die Kameras (magenta) liegen im Mittel {abstand:.0f} m "
                            f"von der Wolkenmitte entfernt, die Wolke selbst misst "
                            f"{ausdehnung:.0f} m.")
                if abstand > ausdehnung:
                    hinweis += (" Das ist weiter weg als die Wolke breit ist — "
                                "entweder deckt der Mäanderflug dieses Gebiet gar "
                                "nicht ab, oder die Ausrichtung sitzt völlig falsch.")
                else:
                    hinweis += (" Die Kameras liegen über der Wolke; dann fehlt es "
                                "an der Ausrichtung — Gier grob durchdrehen und "
                                "auf die Trefferquote schauen.")
            self._log("WARNUNG: " + hinweis)
        self._lbl_meander_lage.setText(self._meander_zustand_text())
        self._status_lbl.setText(
            f"Vorschau {'Thermal' if thermal else 'RGB'}: Gier {yaw:.2f}°, "
            f"Versatz {t[0]:+.1f}/{t[1]:+.1f} m — "
            f"{100.0 * maske.mean():.0f} % getroffen "
            f"({(time.perf_counter() - t0) * 1000:.0f} ms)")

    def _meander_apply_manual(self) -> None:
        """Handjustage endgueltig in die Pipeline schreiben (vor dem Einfaerben)."""
        pipe = self._meander_pipe
        if not self._hat_lage:
            return
        from core import meander as meander_mod
        yaw, t = self._meander_lage()
        meander_mod.set_manual(pipe, yaw, t)
        for sp in self._spin_meander.values():   # Zuschlag ist verrechnet
            still_setzen(sp, 0.0)
        self._meander_speichere_zuschlag()
        # Thermal bleibt ein Zuschlag auf RGB und damit in seinen Reglern
        self._meander_speichere_thermal()

    def _live_hide(self) -> None:
        """Nur die Anzeige raeumen; die geladenen Bilder bleiben im Speicher,
        damit ein Nachjustieren danach weiter sofort wirkt."""
        self._live_timer.stop()
        self._cloud_view.set_color_preview(None)

    def _live_clear(self) -> None:
        self._live = None
        self._live_th = None
        self._live_optik = "rgb"
        self._live_pts = None
        self._live_gemeckert = False
        self._live_hide()

"""Gaussian Splat und Fusion (Mixin des Hauptfensters).

Schreibt am Hauptfenster: _btn_splat_gemeinsam, _btn_splat_maeander,
_btn_splat_onboard, _btn_splat_pruefen, _chk_splat_posen, _chk_splat_pruefen,
_chk_splat_thermal, _combo_splat_raster, _combo_splat_sh, _lbl_splat,
_meander_pipe, _spin_splat_anker, _spin_splat_frames, _spin_splat_schritte,
_spin_splat_schritte_onboard.
"""
from __future__ import annotations

import json
import os

import numpy as np

from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QSpinBox, QWidget,
)

from core.gemeinsam import fmt_int as _fmt_int
from core.project import Project

from ui.bausteine import _compact_combo, _wrappable


def _splat_anker(welt, cfg, progress_cb, cancel, log_cb, von, bis) -> dict:
    """Anker auf der Karte setzen, Fortschritt von ``von`` bis ``bis``; mit Logzeile."""
    from core import splat as splat_mod
    ak = splat_mod.anker(welt, cfg["raster"], cfg["max_anker"],
                         progress=lambda f, m: progress_cb(von + (bis - von) * f, m),
                         cancel=lambda: cancel.is_set(), log=log_cb)
    log_cb(f"Splat-Anker: {_fmt_int(len(ak['pos']))} Gaussians, Raster "
           f"{ak['voxel'] * 100:.1f} cm.")
    return ak


class SplatMixin:
    def _abschnitt_splat(self) -> QWidget:
        box = QWidget()
        form = _wrappable(QFormLayout(box))
        info = QLabel(
            "Lernt die Farben aus allen Bildern zugleich — auf Gaussians, die auf "
            "der Karte sitzen, mit Belichtung je Bild, Verdeckung und nachgeführten "
            "Posen. Die Ergebnisse sind eigene Ebenen neben der direkten Einfärbung. "
            "Braucht eine NVIDIA-GPU und einen Interpreter mit torch und gsplat.")
        info.setWordWrap(True)
        form.addRow(info)
        self._lbl_splat = QLabel("GPU noch nicht geprüft.")
        self._lbl_splat.setWordWrap(True)
        form.addRow(self._lbl_splat)
        self._btn_splat_pruefen = self._befehlsknopf("splat_pruefen")
        form.addRow(self._btn_splat_pruefen)

        self._combo_splat_raster = _compact_combo(QComboBox())
        for label, wert in (("2 cm", 0.02), ("3 cm", 0.03), ("5 cm", 0.05),
                            ("8 cm", 0.08), ("12 cm", 0.12)):
            self._combo_splat_raster.addItem(label, wert)
        self._combo_splat_raster.setCurrentIndex(2)
        self._combo_splat_raster.setToolTip(
            "Kleinste Zellgröße der Anker. Jede belegte Zelle der Karte wird eine\n"
            "Gaussian; das Raster wächst von selbst, bis ihre Zahl unter die Grenze\n"
            "passt. Feiner als ein Bildpixel am Boden bringt nichts.")
        form.addRow("Ankerraster ab:", self._combo_splat_raster)
        self._spin_splat_anker = QDoubleSpinBox()
        self._spin_splat_anker.setRange(0.2, 8.0)
        self._spin_splat_anker.setSingleStep(0.5)
        self._spin_splat_anker.setDecimals(1)
        self._spin_splat_anker.setSuffix(" Mio.")
        self._spin_splat_anker.setValue(4.0)
        self._spin_splat_anker.setToolTip(
            "Höchstens so viele Gaussians. 4 Mio. passen auf eine 8-GB-Karte\n"
            "(geschätzt). Auf der RTX 5090 gemessen: 3,9 Mio. (7,6 cm) brauchen\n"
            "GPU_SPEICHER_76, 7,2 Mio. (5 cm) GPU_SPEICHER_50. Bei Speichermangel\n"
            "hier herunter, das Raster wird dann gröber.")
        form.addRow("höchstens:", self._spin_splat_anker)
        self._combo_splat_sh = _compact_combo(QComboBox())
        for label, wert in (("keine", 0), ("1. Grades", 1), ("2. Grades", 2)):
            self._combo_splat_sh.addItem(label, wert)
        self._combo_splat_sh.setCurrentIndex(1)
        self._combo_splat_sh.setToolTip(
            "Kugelflächenfunktionen für das, was sich mit dem Blickwinkel ändert\n"
            "(Spiegelungen, Glanz). Jeder Punkt bekommt die Farbe, die das Splat\n"
            "aus der frontalsten Ansicht zeigt. Höher heißt mehr Freiheit und mehr\n"
            "Gefahr, Rauschen einzelner Bilder mitzulernen.")
        form.addRow("Blickabhängigkeit:", self._combo_splat_sh)
        self._chk_splat_posen = QCheckBox("Posen je Bild nachführen")
        self._chk_splat_posen.setChecked(True)
        self._chk_splat_posen.setToolTip(
            "Je Bild eine kleine starre Korrektur gegen die feste Lidar-Geometrie.\n"
            "Wie weit sie gewandert sind, steht danach im Protokoll.")
        self._chk_splat_pruefen = QCheckBox("Gegenprobe mit zurückgehaltenen Bildern")
        self._chk_splat_pruefen.setChecked(True)
        self._chk_splat_pruefen.setToolTip(
            "Erst ohne jedes achte Bild (Onboard: jeden zehnten Frame) trainieren\n"
            "und das Splat und die direkte Projektion an diesen Bildern messen,\n"
            "dann mit allen Bildern für die Ebene. Doppelte Zeit.")
        for chk in (self._chk_splat_posen, self._chk_splat_pruefen):
            chk.stateChanged.connect(self._on_setting_changed)
            form.addRow(chk)

        form.addRow(QLabel("<b>Mäanderflug</b>"))
        self._spin_splat_schritte = QSpinBox()
        self._spin_splat_schritte.setRange(500, 100_000)
        self._spin_splat_schritte.setSingleStep(500)
        self._spin_splat_schritte.setValue(3000)
        form.addRow("Schritte:", self._spin_splat_schritte)
        self._chk_splat_thermal = QCheckBox("Temperaturen mitlernen")
        self._chk_splat_thermal.setChecked(True)
        self._chk_splat_thermal.setToolTip(
            "Ein zweites Splat auf den Temperaturen der R-JPEGs statt auf der\n"
            "Palette — die skaliert die Kamera je Bild selbst.")
        self._chk_splat_thermal.stateChanged.connect(self._on_setting_changed)
        form.addRow(self._chk_splat_thermal)
        self._btn_splat_maeander = self._befehlsknopf("splat_meander")
        form.addRow(self._btn_splat_maeander)

        form.addRow(QLabel("<b>360°-Kamera an Bord</b>"))
        self._spin_splat_frames = QSpinBox()
        self._spin_splat_frames.setRange(20, 3000)
        self._spin_splat_frames.setSingleStep(50)
        self._spin_splat_frames.setValue(300)
        self._spin_splat_frames.setToolTip(
            "Höchstens so viele Frames, gewählt nach Bewegung (0,3 m oder 8°).\n"
            "Je Frame bis zu zehn Würfelseiten, fünf je Fisheye.")
        form.addRow("Frames:", self._spin_splat_frames)
        self._spin_splat_schritte_onboard = QSpinBox()
        self._spin_splat_schritte_onboard.setRange(1000, 300_000)
        self._spin_splat_schritte_onboard.setSingleStep(1000)
        self._spin_splat_schritte_onboard.setValue(15000)
        form.addRow("Schritte:", self._spin_splat_schritte_onboard)
        self._btn_splat_onboard = self._befehlsknopf("splat_onboard")
        form.addRow(self._btn_splat_onboard)
        form.addRow(QLabel("<b>Beide gemeinsam</b>"))
        self._btn_splat_gemeinsam = self._befehlsknopf("splat_gemeinsam")
        form.addRow(self._btn_splat_gemeinsam)
        for sp in (self._spin_splat_anker, self._spin_splat_schritte,
                   self._spin_splat_frames, self._spin_splat_schritte_onboard):
            sp.valueChanged.connect(self._on_setting_changed)
        for cb in (self._combo_splat_raster, self._combo_splat_sh):
            cb.currentIndexChanged.connect(self._on_setting_changed)
        return box

    # ===================================================== Gaussian Splat

    def _splat_cfg(self, schritte: int) -> dict:
        """Einstellungen des Splat-Abschnitts fuer einen Lauf, im GUI-Thread gelesen."""
        return {
            "raster": float(self._combo_splat_raster.currentData()),
            "max_anker": int(round(self._spin_splat_anker.value() * 1e6)),
            "pruefen": bool(self._chk_splat_pruefen.isChecked()),
            "param": {"schritte": int(schritte),
                      "sh_grad": int(self._combo_splat_sh.currentData()),
                      "posen": bool(self._chk_splat_posen.isChecked())},
        }

    def _on_splat_pruefen(self) -> None:
        def job(progress_cb, cancel, log_cb):
            from core import splat as splat_mod
            progress_cb(0.1, "Suche Interpreter mit torch und gsplat …")
            return splat_mod.find_splat_python()

        def fertig(res) -> None:
            from core import splat as splat_mod
            py, info = res
            h = splat_mod.hinweis(info)
            text = h or (f"Bereit: {info.get('gpu')} ({info.get('vram_gb')} GB), torch "
                         f"{info.get('torch')}, gsplat {info.get('gsplat_version')}.")
            self._lbl_splat.setText(text)
            self._log(f"Splat: {text}" + (f" Interpreter: {py}" if py else ""))

        self._start_worker("Prüfe GPU und Splat-Interpreter …", job, fertig)

    def _on_splat_maeander(self) -> None:
        kontext = self._maeander_kontext(
            "Gaussian Splat", "Erst einen Flug öffnen und einen Mäanderflug wählen.")
        if kontext is None:
            return
        welt, proj = self._world, self._project
        thermal = bool(self._chk_splat_thermal.isChecked())
        cfg = self._splat_cfg(int(self._spin_splat_schritte.value()))

        def job(progress_cb, cancel, log_cb):
            from core import ebenen as ebenen_mod
            from core import meander as meander_mod
            from core import sichtbar as sichtbar_mod
            from core import splat as splat_mod
            py, _info = splat_mod.interpreter()
            p, th, opt = meander_mod.bereit_machen(
                kontext["pipe"], kontext["args"], kontext["th_zuschlag"],
                kontext["optik_jetzt"], progress_cb, cancel, log_cb, 0.0, 0.05)
            korr = getattr(p, "s360_korrektur", None)
            ak = _splat_anker(welt, cfg, progress_cb, cancel, log_cb, 0.05, 0.1)
            k = meander_mod.kameras(p, opt, th, thermal=thermal)
            # (Ebene, Kameras, Bildordner, A, b, Temperatur, Lage fuer die meta.json)
            aufgaben = [("meander_splat", k["rgb"], p._p("images"), k["A"], k["b"], None,
                         {"yaw_deg": k["yaw_deg"], "t": list(meander_mod.as_t3(p.t)),
                          "rgb_faktor": k["rgb_faktor"], "feinausrichtung": korr})]
            if thermal:
                if k["thermal"] is None or k["temperatur"] is None:
                    log_cb("Splat Thermal übersprungen: " + (
                        "keine Thermaloptik." if k["thermal"] is None else
                        "die Thermalbilder tragen keine Rohwerte."))
                else:
                    aufgaben.append(("meander_thermal_splat", k["thermal"],
                                     p._p("thermal"), k["A_th"], k["b_th"], k["temperatur"],
                                     {"yaw_deg": k["yaw_th"],
                                      "thermal_faktor": k["thermal_faktor"],
                                      "feinausrichtung": korr}))
            ergebnis = {"pipe": p, "ebenen": {}}
            spanne = 0.9 / len(aufgaben)
            for i, (ziel, cams, bilder, A_, b_, tq, soll) in enumerate(aufgaben):
                von = 0.1 + i * spanne
                temp = tq is not None
                ordner = os.path.join(proj.dir, splat_mod.ORDNER, ziel)
                splat_mod.datensatz_maeander(
                    ordner, ak, welt, cams, bilder, A_, b_, temperatur=tq,
                    halte_jedes=8 if cfg["pruefen"] else 0, param=cfg["param"],
                    progress=lambda f, m: progress_cb(von + 0.1 * spanne * f, m),
                    cancel=lambda: cancel.is_set(), log=log_cb)

                def pruefen(_erg, ordner=ordner, cams=cams, bilder=bilder, A_=A_, b_=b_,
                            tq=tq, temp=temp, von=von):
                    pf = splat_mod.punkt_farben(ordner, len(welt))
                    weg = splat_mod.pruef_namen(ordner)
                    progress_cb(von + 0.45 * spanne, "Gegenprobe: direkte Projektion ohne "
                                                     "die Prüfbilder …")

                    def direkt_w(idx, P, nrm):
                        # Tiefenkarte aus allen Ankern: aus der Probe allein
                        # entstuende keine Oberflaeche, und die direkte
                        # Projektion faerbte auch verdeckte Punkte — die
                        # Gegenprobe fiele zugunsten des Splats aus.
                        e = sichtbar_mod.colorize_sichtbar(
                            P, splat_mod.cams_ohne(cams, weg), bilder, A_, b_,
                            normalen_welt=nrm, cancel=lambda: cancel.is_set(),
                            temperatur=tq, tiefe_punkte=ak["pos"])
                        if temp:
                            return e[2][:, None], e[1] & np.isfinite(e[2])
                        return e[0], e[1]

                    return splat_mod.gegenprobe(ordner, ak, welt, pf, direkt_w, temp,
                                                cancel, log_cb)

                erg, stats = splat_mod.trainingsfolge(
                    py, ordner, cfg, (von + 0.1 * spanne, von + 0.45 * spanne,
                                      von + 0.6 * spanne, von + 0.97 * spanne),
                    pruefen, progress_cb, cancel, log_cb)
                pf = splat_mod.punkt_farben(ordner, len(welt))
                maske = pf["maske"]
                ebenen_mod.speichern(
                    proj.layer_dir(ziel), pf["rgb"], maske,
                    dict(soll, quelle=ziel, flug=p.photo_dir, anteil=float(maske.mean()),
                         temperatur=temp, gegenprobe=stats,
                         splat=splat_mod.splat_meta(ak, cfg, erg["bericht"].get("posen"))),
                    temperatur=pf.get("temperatur"))
                ergebnis["ebenen"][ziel] = float(maske.mean())
            progress_cb(1.0, "Splat-Einfärbung fertig")
            return ergebnis

        self._start_worker("Gaussian Splat des Mäanderfluges läuft …", job,
                           self._on_splat_done)

    def _on_splat_onboard(self) -> None:
        if self._rec is None or self._bag is None or self._project is None \
                or self._world is None or not self._calib:
            return
        kontext = self._onboard_kontext()
        T, teile, masken = kontext["T"], kontext["teile"], kontext["masken"]
        rec, calib, proj, welt = self._rec, self._calib, self._project, self._world
        frames = int(self._spin_splat_frames.value())
        cfg = self._splat_cfg(int(self._spin_splat_schritte_onboard.value()))
        paar = self._layers.get("onboard")

        def job(progress_cb, cancel, log_cb):
            from core import ebenen as ebenen_mod
            from core import splat as splat_mod
            py, _info = splat_mod.interpreter()
            ak = _splat_anker(welt, cfg, progress_cb, cancel, log_cb, 0.0, 0.08)
            direkt = paar if paar is not None and ebenen_mod.passt(
                proj.layer_dir("onboard"), extrinsic=T) else None
            if direkt is None and cfg["pruefen"]:
                log_cb("Keine direkte Onboard-Einfärbung mit dieser Extrinsik — ohne "
                       "Gegenprobe.")
            ordner = os.path.join(proj.dir, splat_mod.ORDNER, "onboard_splat")
            d = splat_mod.datensatz_onboard(
                ordner, ak, welt, rec, teile, calib, T, max_bilder=frames,
                halte_jedes=10 if cfg["pruefen"] else 0, param=cfg["param"],
                progress=lambda f, m: progress_cb(0.08 + 0.17 * f, m),
                cancel=lambda: cancel.is_set(), log=log_cb, **masken)
            log_cb(f"Onboard-Splat-Datensatz: {d['ansichten']} Ansichten aus {d['frames']} "
                   f"Frames, im Mittel {d['abdeckung'] * 100:.0f} % der Pixel brauchbar.")

            def pruefen(_erg):
                if direkt is None:
                    return None
                pf = splat_mod.punkt_farben(ordner, len(welt))
                stats = splat_mod.gegenprobe(
                    ordner, ak, welt, pf,
                    lambda idx, P, nrm: (direkt[0][idx], direkt[1][idx]),
                    False, cancel, log_cb)
                log_cb("Vorsicht beim Onboard-Vergleich: die direkte Einfärbung hat die "
                       "Prüfframes gekannt, das Splat nicht — die Probe begünstigt die "
                       "direkte Projektion.")
                return stats

            erg, stats = splat_mod.trainingsfolge(py, ordner, cfg, (0.25, 0.55, 0.6, 0.97),
                                                  pruefen, progress_cb, cancel, log_cb)
            pf = splat_mod.punkt_farben(ordner, len(welt))
            maske = pf["maske"]
            ebenen_mod.speichern(
                proj.layer_dir("onboard_splat"), pf["rgb"], maske,
                {"quelle": "onboard_splat", "extrinsic": np.asarray(T).tolist(),
                 "calib_json": calib, "anteil": float(maske.mean()), "frames": d["frames"],
                 "ansichten": d["ansichten"], "gegenprobe": stats,
                 "splat": splat_mod.splat_meta(ak, cfg, erg["bericht"].get("posen")),
                 **masken})
            progress_cb(1.0, "Splat-Einfärbung fertig")
            return {"pipe": None, "ebenen": {"onboard_splat": float(maske.mean())}}

        self._start_worker("Gaussian Splat der 360°-Kamera läuft …", job, self._on_splat_done)

    def _on_splat_gemeinsam(self) -> None:
        kontext = self._maeander_kontext(
            "Gaussian Splat", "Erst einen Flug mit Bag öffnen und einen Mäanderflug wählen.",
            zusatz_ok=self._rec is not None and self._bag is not None and bool(self._calib))
        if kontext is None:
            return
        welt, proj, rec, calib = self._world, self._project, self._rec, self._calib
        onboard = self._onboard_kontext()
        T, teile, masken = onboard["T"], onboard["teile"], onboard["masken"]
        frames = int(self._spin_splat_frames.value())
        cfg = self._splat_cfg(int(self._spin_splat_schritte.value())
                              + int(self._spin_splat_schritte_onboard.value()))

        def job(progress_cb, cancel, log_cb):
            from core import ebenen as ebenen_mod
            from core import meander as meander_mod
            from core import splat as splat_mod
            py, _info = splat_mod.interpreter()
            p, th, opt = meander_mod.bereit_machen(
                kontext["pipe"], kontext["args"], kontext["th_zuschlag"],
                kontext["optik_jetzt"], progress_cb, cancel, log_cb, 0.0, 0.05)
            ak = _splat_anker(welt, cfg, progress_cb, cancel, log_cb, 0.05, 0.1)
            # Die Abbildung aus der Fusion ohne Splat ist ein guter Start fuer
            # die Farbmatrix der Onboard-Bilder
            from core import fusion as fusion_mod
            abbildung = None
            try:
                with open(os.path.join(proj.dir, Project.LAYERS["fusion"], "meta.json"),
                          encoding="utf-8") as fh:
                    fm = json.load(fh)
                grund = fusion_mod.unplausibel(fm["M"], fm["t"], fm)
                if grund:
                    log_cb(f"Die Fusion ohne Splat taugt nicht als Start: {grund}. Die "
                           f"Farbmatrix der Onboard-Bilder startet bei der Einheit.")
                else:
                    abbildung = (np.asarray(fm["M"], float), np.asarray(fm["t"], float))
                    log_cb(f"Farbmatrix der Onboard-Bilder startet bei der Fusion "
                           f"(Abstand dort {fm['abstand_vorher']:.1f} → "
                           f"{fm['abstand_nachher']:.1f}).")
            except (OSError, ValueError, KeyError):
                log_cb("Keine Fusion ohne Splat vorhanden — die Farbmatrix der "
                       "Onboard-Bilder startet bei der Einheit.")
            k = meander_mod.kameras(p, opt, th, thermal=False)
            ordner = os.path.join(proj.dir, splat_mod.ORDNER, "fusion_splat")
            d = splat_mod.datensatz_gemeinsam(
                ordner, ak, welt,
                maeander={"cams": k["rgb"], "bild_ordner": p._p("images"),
                          "A": k["A"], "b": k["b"], "halte_jedes": 8 if cfg["pruefen"] else 0},
                onboard=dict({"rec": rec, "teile": teile, "calib_json": calib,
                              "T_imu_cam0": T, "max_bilder": frames,
                              "halte_jedes": 10 if cfg["pruefen"] else 0}, **masken),
                abbildung=abbildung, param=cfg["param"],
                progress=lambda f, m: progress_cb(0.1 + 0.2 * f, m),
                cancel=lambda: cancel.is_set(), log=log_cb)
            log_cb(f"Gemeinsamer Datensatz: {d['maeander']} Mäanderbilder und "
                   f"{d['onboard']} Onboard-Ansichten aus {d['frames']} Frames.")

            def je_herkunft(erg):
                for h in ("maeander", "onboard"):
                    e = [x["psnr_angepasst"] for x in erg["bericht"]["pruefung"]
                         if x.get("herkunft") == h]
                    if e:
                        log_cb(f"Prüfbilder {h}: PSNR {np.mean(e):.1f} dB ({len(e)} Bilder).")

            erg, _ = splat_mod.trainingsfolge(py, ordner, cfg, (0.3, 0.6, 0.62, 0.97),
                                              je_herkunft, progress_cb, cancel, log_cb)
            bf = erg["bericht"].get("belichtung_frei")
            if bf:
                log_cb("Gelernte mittlere Farbmatrix Onboard → Mäander: diag "
                       + " ".join(f"{x:.3f}" for x in np.diag(np.asarray(bf["M"])))
                       + ", Versatz " + " ".join(f"{x * 255:+.1f}" for x in bf["t"])
                       + " (0–255).")
            pf = splat_mod.punkt_farben(ordner, len(welt))
            maske = pf["maske"]
            ebenen_mod.speichern(
                proj.layer_dir("fusion_splat"), pf["rgb"], maske,
                {"quelle": "fusion_splat", "flug": p.photo_dir, "yaw_deg": k["yaw_deg"],
                 "t": list(meander_mod.as_t3(p.t)), "rgb_faktor": k["rgb_faktor"],
                 "extrinsic": np.asarray(T).tolist(), "calib_json": calib,
                 "anteil": float(maske.mean()), "maeander_bilder": d["maeander"],
                 "onboard_ansichten": d["onboard"], "frames": d["frames"],
                 "startabbildung": d["startabbildung"], "belichtung_onboard": bf,
                 "splat": splat_mod.splat_meta(ak, cfg, erg["bericht"].get("posen")),
                 **masken})
            progress_cb(1.0, "Gemeinsames Splat fertig")
            return {"pipe": p, "ebenen": {"fusion_splat": float(maske.mean())}}

        self._start_worker("Gemeinsames Gaussian Splat (Onboard + Mäander) läuft …", job,
                           self._on_splat_done)

    def _fusion_quellen(self) -> tuple | None:
        """(Onboard-Ebene, Maeander-Ebene) fuer die Fusion; Splat vor direkt."""
        from core import fusion as fusion_mod
        return fusion_mod.quellen(self._layers)

    def _on_fusion(self) -> None:
        quellen = self._fusion_quellen()
        if self._project is None or self._world is None or quellen is None:
            return
        k_onb, k_mea = quellen
        onb, mea = self._layers[k_onb], self._layers[k_mea]
        welt, proj = self._world, self._project

        def job(progress_cb, cancel, log_cb):
            from core import ebenen as ebenen_mod
            from core import fusion as fusion_mod
            from core import sichtbar as sichtbar_mod
            log_cb(f"Fusion: '{k_onb}' wird an '{k_mea}' angeglichen.")
            normalen = sichtbar_mod.normalen(
                welt, progress=lambda f, m: progress_cb(0.4 * f, m))
            erg = fusion_mod.fusioniere(
                onb, mea, normalen,
                progress=lambda f, m: progress_cb(0.4 + 0.55 * f, m),
                cancel=lambda: cancel.is_set())
            b = erg["bericht"]
            log_cb(f"Überlappung {_fmt_int(b['ueberlappung'])} Punkte; Farbabstand "
                   f"Onboard↔Mäander {b['abstand_vorher']:.1f} → "
                   f"{b['abstand_nachher']:.1f} (Median, 0–255, Prüfhälfte).")
            if b["unplausibel"]:
                log_cb(f"WARNUNG: {b['unplausibel']}. Meist zeigen beide Flüge an vielen "
                       f"Stellen Verschiedenes — umgeparkte Autos, Dächer, die an Bord nur "
                       f"von innen zu sehen sind. Dann hilft keine Farbabbildung.")
            ebenen_mod.speichern(
                proj.layer_dir("fusion"), erg["rgb"], erg["maske"],
                {"quelle": "fusion", "onboard": k_onb, "maeander": k_mea,
                 "M": erg["M"].tolist(), "t": erg["t"].tolist(), **b})
            progress_cb(1.0, "Fusion fertig")
            return {"pipe": None, "ebenen": {"fusion": b["anteil"]}}

        self._start_worker("Onboard und Mäander werden fusioniert …", job,
                           self._on_splat_done)

    def _on_splat_done(self, res: dict) -> None:
        if res.get("pipe") is not None:
            self._meander_pipe = res["pipe"]
            self._meander_lade_thermal()
            self._meander_lade_optik()
        self._ebenen_melden(res["ebenen"],
                            zusatz=" Ungefärbt bleibt, was kein Trainingsbild sichtbar zeigt.")
        self._reload_layers()
        erste = next(iter(res["ebenen"]), None)
        if erste is not None:
            self._ebene_waehlen(erste)
        self._update_enabled()

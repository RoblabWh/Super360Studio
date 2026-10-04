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
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QMessageBox,
    QPushButton, QSpinBox, QWidget,
)

from core.gemeinsam import fmt_int as _fmt_int
from core.project import Project

from ui.bausteine import _compact_combo, _wrappable


class SplatMixin:
    def _group_splat(self) -> QWidget:
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
        self._btn_splat_pruefen = QPushButton("GPU und Interpreter prüfen")
        self._btn_splat_pruefen.clicked.connect(self._on_splat_pruefen)
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
        self._btn_splat_maeander = QPushButton("Mäander per Splat einfärben")
        self._btn_splat_maeander.clicked.connect(self._on_splat_maeander)
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
        self._btn_splat_onboard = QPushButton("Onboard per Splat einfärben")
        self._btn_splat_onboard.setToolTip(
            "Nimmt Extrinsik, Helligkeitsfenster und Himmelssaum aus Abschnitt 3.")
        self._btn_splat_onboard.clicked.connect(self._on_splat_onboard)
        form.addRow(self._btn_splat_onboard)
        form.addRow(QLabel("<b>Beide gemeinsam</b>"))
        self._btn_splat_gemeinsam = QPushButton("Onboard + Mäander in einem Splat")
        self._btn_splat_gemeinsam.setToolTip(
            "Ein Splat aus beiden Flügen. Die Farbe steht im Mäander, jedes Onboard-Bild\n"
            "bekommt seine eigene Farbmatrix. Schritte: beide Felder zusammen.\n"
            "Liegt eine Fusion (ohne Splat) vor, startet die Farbmatrix dort.")
        self._btn_splat_gemeinsam.clicked.connect(self._on_splat_gemeinsam)
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

    @staticmethod
    def _ebene_passt(ordner: str, **soll) -> bool:
        """Gehoert eine gespeicherte Ebene zu dieser Lage? Vergleicht Werte der meta.json."""
        try:
            with open(os.path.join(ordner, "meta.json"), encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            return False
        for key, wert in soll.items():
            ist = meta.get(key)
            if wert is None or ist is None:
                if wert is not ist:
                    return False
                continue
            if isinstance(wert, dict):
                if not all(np.allclose(np.asarray(ist.get(k), float), np.asarray(wert[k], float),
                                       atol=1e-6) for k in ("M", "v")):
                    return False
            elif not np.allclose(np.asarray(ist, float), np.asarray(wert, float), atol=1e-6):
                return False
        return True

    @staticmethod
    def _splat_lauf(py, ordner, cfg, progress_cb, cancel, log_cb, von, bis, alle) -> dict:
        from core import splat as splat_mod
        erg = splat_mod.trainieren(
            py, ordner, progress=lambda f, m: progress_cb(von + (bis - von) * f, m),
            cancel=lambda: cancel.is_set(), log=log_cb, alle_bilder=alle)
        p = erg["bericht"].get("posen") or {}
        if cfg["param"]["posen"] and p:
            log_cb(f"Splat-Posen nachgeführt: Drehung Median {p['dreh_grad_median']:.3f}° "
                   f"(max {p['dreh_grad_max']:.3f}°), Weg Median "
                   f"{p['weg_m_median'] * 100:.1f} cm (max {p['weg_m_max'] * 100:.1f} cm).")
        if erg["pruefung"]:
            psnr = np.mean([e["psnr_angepasst"] for e in erg["pruefung"]])
            log_cb(f"Splat an {len(erg['pruefung'])} Prüfbildern gerendert: "
                   f"PSNR {psnr:.1f} dB nach Belichtungsangleich.")
        return erg

    @staticmethod
    def _splat_gegenprobe(ordner, ak, welt, pf, direkt_w, temp, cancel, log_cb) -> dict:
        """Splat und direkte Projektion an den Pruefbildern messen."""
        from core import splat as splat_mod
        idx = splat_mod.probe(ak)
        P = welt[idx]
        nrm = ak["normal"][ak["index"][idx]]
        if temp:
            s_w, einheit = pf["temperatur"][idx][:, None], "°C"
        else:
            s_w, einheit = pf["rgb"][idx], "(0–255)"
        werte = {"splat": (s_w, pf["maske"][idx]), "direkt": direkt_w(idx, P, nrm)}
        stats = splat_mod.vergleich(ordner, ak, P, nrm, werte,
                                    cancel=lambda: cancel.is_set())
        log_cb(splat_mod.urteil(stats, "splat", "direkt", einheit))
        return stats

    def _on_splat_maeander(self) -> None:
        if not self._meander_dir or self._world is None or self._project is None:
            QMessageBox.information(self, "Gaussian Splat",
                                    "Erst einen Flug öffnen und einen Mäanderflug wählen.")
            return
        if not self._meander_ask_colmap():
            return
        self._meander_apply_manual()
        pipe = self._meander_pipe
        args = self._meander_args()
        bauen, bereit = self._meander_build, self._meander_bereit
        lauf, gegenprobe = self._splat_lauf, self._splat_gegenprobe
        welt, proj = self._world, self._project
        th_zuschlag = self._thermal_zuschlag()
        optik_jetzt = dict(self._optik, rgb_faktor=self._faktor("rgb"),
                           thermal_faktor=self._faktor("thermal"))
        thermal = bool(self._chk_splat_thermal.isChecked())
        cfg = self._splat_cfg(int(self._spin_splat_schritte.value()))

        def job(progress_cb, cancel, log_cb):
            from core import meander as meander_mod
            from core import optik as optik_mod
            from core import sichtbar as sichtbar_mod
            from core import splat as splat_mod
            from core import temperatur as temperatur_mod
            py, info = splat_mod.find_splat_python()
            if splat_mod.hinweis(info):
                raise RuntimeError(splat_mod.hinweis(info))
            p, th, opt = bereit(pipe, args, bauen, th_zuschlag, optik_jetzt,
                                progress_cb, cancel, log_cb, 0.0, 0.05)
            rf, tf = float(opt["rgb_faktor"]), float(opt["thermal_faktor"])
            korr = getattr(p, "s360_korrektur", None)
            yaw = float(np.degrees(p.yaw))
            A, b = meander_mod.lage_affine(p, yaw, p.t)
            ak = splat_mod.anker(welt, cfg["raster"], cfg["max_anker"],
                                 progress=lambda f, m: progress_cb(0.05 + 0.05 * f, m),
                                 cancel=lambda: cancel.is_set(), log=log_cb)
            log_cb(f"Splat-Anker: {_fmt_int(len(ak['pos']))} Gaussians, Raster "
                   f"{ak['voxel'] * 100:.1f} cm.")
            # (Ebene, Kameras, Bildordner, A, b, Temperatur, Lage fuer die meta.json)
            aufgaben = [("meander_splat", optik_mod.rgb_cams(p, rf),
                         p._p("images"), A, b, None,
                         {"yaw_deg": yaw, "t": list(meander_mod.as_t3(p.t)),
                          "rgb_faktor": rf, "feinausrichtung": korr})]
            if thermal:
                th_cams = optik_mod.thermal_cams(p, opt.get("thermal"), tf, rf)
                tq = temperatur_mod.quelle(p)
                if th_cams is None or tq is None:
                    log_cb("Splat Thermal übersprungen: " + (
                        "keine Thermaloptik." if th_cams is None else
                        "die Thermalbilder tragen keine Rohwerte."))
                else:
                    yaw_th, t_th = meander_mod.thermal_lage(yaw, p.t, th)
                    A_th, b_th = meander_mod.lage_affine(p, yaw_th, t_th)
                    aufgaben.append(("meander_thermal_splat", th_cams,
                                     p._p("thermal"), A_th, b_th, tq,
                                     {"yaw_deg": yaw_th, "thermal_faktor": tf,
                                      "feinausrichtung": korr}))
            ergebnis = {"pipe": p, "ebenen": {}}
            spanne = 0.9 / len(aufgaben)
            for k, (ziel, cams, bilder, A_, b_, tq, soll) in enumerate(aufgaben):
                von = 0.1 + k * spanne
                temp = tq is not None
                ordner = os.path.join(proj.dir, splat_mod.ORDNER, ziel)
                splat_mod.datensatz_maeander(
                    ordner, ak, welt, cams, bilder, A_, b_, temperatur=tq,
                    halte_jedes=8 if cfg["pruefen"] else 0, param=cfg["param"],
                    progress=lambda f, m: progress_cb(von + 0.1 * spanne * f, m),
                    cancel=lambda: cancel.is_set(), log=log_cb)
                stats = None
                if cfg["pruefen"]:
                    lauf(py, ordner, cfg, progress_cb, cancel, log_cb,
                         von + 0.1 * spanne, von + 0.45 * spanne, False)
                    pf = splat_mod.punkt_farben(ordner, len(welt))
                    weg = splat_mod.pruef_namen(ordner)
                    progress_cb(von + 0.45 * spanne, "Gegenprobe: direkte Projektion ohne "
                                                     "die Prüfbilder …")

                    def direkt_w(idx, P, nrm, cams=cams, bilder=bilder, A_=A_, b_=b_,
                                 tq=tq, temp=temp, weg=weg):
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

                    stats = gegenprobe(ordner, ak, welt, pf, direkt_w, temp, cancel, log_cb)
                    lauf(py, ordner, cfg, progress_cb, cancel, log_cb,
                         von + 0.6 * spanne, von + 0.97 * spanne, True)
                else:
                    lauf(py, ordner, cfg, progress_cb, cancel, log_cb,
                         von + 0.1 * spanne, von + 0.97 * spanne, False)
                pf = splat_mod.punkt_farben(ordner, len(welt))
                maske = pf["maske"]
                with open(os.path.join(ordner, "bericht.json"), encoding="utf-8") as fh:
                    bericht = json.load(fh)
                meander_mod.save_layer(
                    proj.layer_dir(ziel), pf["rgb"], maske,
                    dict(soll, quelle=ziel, flug=p.photo_dir, anteil=float(maske.mean()),
                         temperatur=temp, gegenprobe=stats,
                         splat={"anker": int(len(ak["pos"])), "raster_m": ak["voxel"],
                                **cfg["param"], "posen": bericht.get("posen")}),
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
        T = self._extrinsic_from_spins()
        rec, calib, proj, welt = self._rec, self._calib, self._project, self._world
        teile = list(self._parts) if self._parts else [(self._bag, 0, int(rec.n_scans))]
        masken = {"bmin": int(self._sld_bmin.value()), "bmax": int(self._sld_bmax.value()),
                  "sky_grow": int(self._spin_sky.value())}
        frames = int(self._spin_splat_frames.value())
        cfg = self._splat_cfg(int(self._spin_splat_schritte_onboard.value()))
        paar = self._layers.get("onboard")
        lauf, gegenprobe, passt = self._splat_lauf, self._splat_gegenprobe, self._ebene_passt

        def job(progress_cb, cancel, log_cb):
            from core import meander as meander_mod
            from core import splat as splat_mod
            py, info = splat_mod.find_splat_python()
            if splat_mod.hinweis(info):
                raise RuntimeError(splat_mod.hinweis(info))
            ak = splat_mod.anker(welt, cfg["raster"], cfg["max_anker"],
                                 progress=lambda f, m: progress_cb(0.08 * f, m),
                                 cancel=lambda: cancel.is_set(), log=log_cb)
            log_cb(f"Splat-Anker: {_fmt_int(len(ak['pos']))} Gaussians, Raster "
                   f"{ak['voxel'] * 100:.1f} cm.")
            direkt = paar if paar is not None and passt(proj.colors_dir(), extrinsic=T) else None
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
            stats = None
            if cfg["pruefen"]:
                lauf(py, ordner, cfg, progress_cb, cancel, log_cb, 0.25, 0.55, False)
                if direkt is not None:
                    pf = splat_mod.punkt_farben(ordner, len(welt))
                    stats = gegenprobe(ordner, ak, welt, pf,
                                       lambda idx, P, nrm: (direkt[0][idx], direkt[1][idx]),
                                       False, cancel, log_cb)
                    log_cb("Vorsicht beim Onboard-Vergleich: die direkte Einfärbung hat die "
                           "Prüfframes gekannt, das Splat nicht — die Probe begünstigt die "
                           "direkte Projektion.")
                lauf(py, ordner, cfg, progress_cb, cancel, log_cb, 0.6, 0.97, True)
            else:
                lauf(py, ordner, cfg, progress_cb, cancel, log_cb, 0.25, 0.97, False)
            pf = splat_mod.punkt_farben(ordner, len(welt))
            maske = pf["maske"]
            with open(os.path.join(ordner, "bericht.json"), encoding="utf-8") as fh:
                bericht = json.load(fh)
            meander_mod.save_layer(
                proj.layer_dir("onboard_splat"), pf["rgb"], maske,
                {"quelle": "onboard_splat", "extrinsic": np.asarray(T).tolist(),
                 "calib_json": calib, "anteil": float(maske.mean()), "frames": d["frames"],
                 "ansichten": d["ansichten"], "gegenprobe": stats,
                 "splat": {"anker": int(len(ak["pos"])), "raster_m": ak["voxel"],
                           **cfg["param"], "posen": bericht.get("posen")}, **masken})
            progress_cb(1.0, "Splat-Einfärbung fertig")
            return {"pipe": None, "ebenen": {"onboard_splat": float(maske.mean())}}

        self._start_worker("Gaussian Splat der 360°-Kamera läuft …", job, self._on_splat_done)

    def _on_splat_gemeinsam(self) -> None:
        if not self._meander_dir or self._world is None or self._project is None \
                or self._rec is None or self._bag is None or not self._calib:
            QMessageBox.information(self, "Gaussian Splat",
                                    "Erst einen Flug mit Bag öffnen und einen Mäanderflug wählen.")
            return
        if not self._meander_ask_colmap():
            return
        self._meander_apply_manual()
        pipe = self._meander_pipe
        args = self._meander_args()
        bauen, bereit, lauf = self._meander_build, self._meander_bereit, self._splat_lauf
        welt, proj, rec, calib = self._world, self._project, self._rec, self._calib
        th_zuschlag = self._thermal_zuschlag()
        optik_jetzt = dict(self._optik, rgb_faktor=self._faktor("rgb"),
                           thermal_faktor=self._faktor("thermal"))
        T = self._extrinsic_from_spins()
        teile = list(self._parts) if self._parts else [(self._bag, 0, int(rec.n_scans))]
        masken = {"bmin": int(self._sld_bmin.value()), "bmax": int(self._sld_bmax.value()),
                  "sky_grow": int(self._spin_sky.value())}
        frames = int(self._spin_splat_frames.value())
        cfg = self._splat_cfg(int(self._spin_splat_schritte.value())
                              + int(self._spin_splat_schritte_onboard.value()))

        def job(progress_cb, cancel, log_cb):
            from core import meander as meander_mod
            from core import optik as optik_mod
            from core import splat as splat_mod
            py, info = splat_mod.find_splat_python()
            if splat_mod.hinweis(info):
                raise RuntimeError(splat_mod.hinweis(info))
            p, _th, opt = bereit(pipe, args, bauen, th_zuschlag, optik_jetzt,
                                 progress_cb, cancel, log_cb, 0.0, 0.05)
            rf = float(opt["rgb_faktor"])
            yaw = float(np.degrees(p.yaw))
            A, b = meander_mod.lage_affine(p, yaw, p.t)
            ak = splat_mod.anker(welt, cfg["raster"], cfg["max_anker"],
                                 progress=lambda f, m: progress_cb(0.05 + 0.05 * f, m),
                                 cancel=lambda: cancel.is_set(), log=log_cb)
            log_cb(f"Splat-Anker: {_fmt_int(len(ak['pos']))} Gaussians, Raster "
                   f"{ak['voxel'] * 100:.1f} cm.")
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
            ordner = os.path.join(proj.dir, splat_mod.ORDNER, "fusion_splat")
            d = splat_mod.datensatz_gemeinsam(
                ordner, ak, welt,
                maeander={"cams": optik_mod.rgb_cams(p, rf), "bild_ordner": p._p("images"),
                          "A": A, "b": b, "halte_jedes": 8 if cfg["pruefen"] else 0},
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

            if cfg["pruefen"]:
                je_herkunft(lauf(py, ordner, cfg, progress_cb, cancel, log_cb,
                                 0.3, 0.6, False))
                erg = lauf(py, ordner, cfg, progress_cb, cancel, log_cb, 0.62, 0.97, True)
            else:
                erg = lauf(py, ordner, cfg, progress_cb, cancel, log_cb, 0.3, 0.97, False)
            bf = erg["bericht"].get("belichtung_frei")
            if bf:
                log_cb("Gelernte mittlere Farbmatrix Onboard → Mäander: diag "
                       + " ".join(f"{x:.3f}" for x in np.diag(np.asarray(bf["M"])))
                       + ", Versatz " + " ".join(f"{x * 255:+.1f}" for x in bf["t"])
                       + " (0–255).")
            pf = splat_mod.punkt_farben(ordner, len(welt))
            maske = pf["maske"]
            meander_mod.save_layer(
                proj.layer_dir("fusion_splat"), pf["rgb"], maske,
                {"quelle": "fusion_splat", "flug": p.photo_dir, "yaw_deg": yaw,
                 "t": list(meander_mod.as_t3(p.t)), "rgb_faktor": rf,
                 "extrinsic": np.asarray(T).tolist(), "calib_json": calib,
                 "anteil": float(maske.mean()), "maeander_bilder": d["maeander"],
                 "onboard_ansichten": d["onboard"], "frames": d["frames"],
                 "startabbildung": d["startabbildung"], "belichtung_onboard": bf,
                 "splat": {"anker": int(len(ak["pos"])), "raster_m": ak["voxel"],
                           **cfg["param"], "posen": erg["bericht"].get("posen")},
                 **masken})
            progress_cb(1.0, "Gemeinsames Splat fertig")
            return {"pipe": p, "ebenen": {"fusion_splat": float(maske.mean())}}

        self._start_worker("Gemeinsames Gaussian Splat (Onboard + Mäander) läuft …", job,
                           self._on_splat_done)

    def _fusion_quellen(self) -> tuple | None:
        """(Onboard-Ebene, Maeander-Ebene) fuer die Fusion; Splat vor direkt."""
        onb = next((k for k in ("onboard_splat", "onboard") if k in self._layers), None)
        mea = next((k for k in ("meander_splat", "meander_rgb") if k in self._layers), None)
        return (onb, mea) if onb and mea else None

    def _on_fusion(self) -> None:
        quellen = self._fusion_quellen()
        if self._project is None or self._world is None or quellen is None:
            return
        k_onb, k_mea = quellen
        onb, mea = self._layers[k_onb], self._layers[k_mea]
        welt, proj = self._world, self._project

        def job(progress_cb, cancel, log_cb):
            from core import fusion as fusion_mod
            from core import meander as meander_mod
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
            meander_mod.save_layer(
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
        for key, anteil in res["ebenen"].items():
            self._log(f"Farbebene '{key}': {anteil * 100:.1f} % der Punkte eingefärbt. "
                      f"Ungefärbt bleibt, was kein Trainingsbild sichtbar zeigt.")
        self._reload_layers()
        erste = next(iter(res["ebenen"]), None)
        if erste is not None and erste in self._layers:
            idx = self._combo_layer.findData(erste)
            if idx >= 0:
                self._combo_layer.setCurrentIndex(idx)
        self._update_enabled()

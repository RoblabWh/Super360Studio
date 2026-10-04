"""Mesh (Mixin des Hauptfensters).

Schreibt am Hauptfenster: _mesh_farbcache, _mesh_geom, _mesh_geom_dir,
_mesh_laeuft, _mesh_wartet, _settings.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from core.gemeinsam import fmt_int as _fmt_int


class MeshMixin:
    # =============================================================== Mesh

    def _on_leiste_mesh(self, an: bool) -> None:
        self._settings["mesh_an"] = bool(an)
        self._save_settings()
        if an:
            self._mesh_sicherstellen()

    def _mesh_param(self) -> tuple[float, int, float, bool]:
        return (self._spin_mesh_voxel.value() / 100.0, int(self._spin_mesh_depth.value()),
                self._spin_mesh_trim.value() / 100.0, bool(self._chk_mesh_hybrid.isChecked()))

    def _mesh_ziel(self) -> Optional[str]:
        """Ablage der Geometrie fuer die offene Wolke und die Mesh-Parameter."""
        if self._rec is None or self._project is None:
            return None
        from core import mesh as mesh_mod
        from core.colorizer import rec_fingerprint
        vox, tiefe, trim, hybrid = self._mesh_param()
        return mesh_mod.geometry_dir(self._project.dir, rec_fingerprint(self._rec),
                                     vox, tiefe, trim, hybrid)

    def _mesh_vergessen(self) -> None:
        self._mesh_geom = None
        self._mesh_geom_dir = None
        self._mesh_farbcache = {}
        self._cloud_view.set_mesh(None)
        self._cloud_view.set_mesh_schalter(self._cloud_view.mesh_an())

    def _mesh_nachholen(self) -> None:
        if self._busy:
            return
        if self._mesh_laeuft:
            # Mesh-Lauf endete ohne Ergebnis (abgebrochen): Schalter loesen
            self._mesh_laeuft = False
            self._cloud_view.set_mesh_schalter(False)
            self._settings["mesh_an"] = False
            self._save_settings()
            return
        if self._mesh_wartet:
            self._mesh_wartet = False
            self._mesh_sicherstellen()

    def _on_mesh_param_changed(self, *_a) -> None:
        if self._cloud_view.mesh_an() and not self._loading_ui:
            self._mesh_timer.start()      # nicht bei jedem Pfeilklick neu rechnen

    def _mesh_sicherstellen(self) -> None:
        """Ist der Schalter an, gehoert zur offenen Wolke ein Mesh: laden oder rechnen.

        Laeuft gerade ein anderer Schritt, wartet das Mesh und kommt danach.
        """
        if not self._cloud_view.mesh_an() or self._world is None or self._closing:
            return
        ziel = self._mesh_ziel()
        if ziel is None:
            return
        if self._mesh_geom is not None and self._mesh_geom_dir == ziel:
            self._mesh_farben_zeigen()
            return
        if self._busy:
            self._mesh_wartet = True
            self._cloud_view.set_mesh_schalter(True, "Mesh (wartet)")
            return
        from core import mesh as mesh_mod
        world, rec = self._world, self._rec
        vox, tiefe, trim, hybrid = self._mesh_param()

        def job(progress_cb, cancel, log_cb):
            geom = mesh_mod.load_geometry(ziel)
            if geom is not None:
                progress_cb(0.5, "Lade gespeichertes Mesh …")
                log_cb(f"Mesh aus dem Projekt: {_fmt_int(geom['stats']['dreiecke'])} "
                       f"Dreiecke.")
            else:
                geom = mesh_mod.build_geometry(
                    world, rec.path_positions(), voxel=vox, depth=tiefe, trim=trim,
                    hybrid=hybrid, progress=lambda f, m: progress_cb(0.9 * f, m),
                    cancel=lambda: cancel.is_set(), log=log_cb)
                progress_cb(0.9, "Speichere Mesh …")
                mesh_mod.save_geometry(geom, ziel)
                geom = mesh_mod.load_geometry(ziel) or geom
            # Intensitaet je Ecke einmal rechnen und ablegen (liest alle Punkte)
            ipfad = os.path.join(ziel, "vertex_intensity.npy")
            if os.path.isfile(ipfad):
                geom["intensity"] = np.load(ipfad)
            elif rec is not None and getattr(rec, "intensity", None) is not None:
                progress_cb(0.95, "Mesh: Intensität je Ecke …")
                geom["intensity"] = mesh_mod.vertex_scalar(geom, rec.intensity)
                np.save(ipfad, geom["intensity"])
            geom["rest_punkte"] = mesh_mod.rest_maske(geom)
            return geom

        def on_done(geom) -> None:
            self._mesh_laeuft = False
            if self._world is not world:
                return                          # Wolke wurde inzwischen gewechselt
            self._mesh_geom = geom
            self._mesh_geom_dir = ziel
            self._mesh_farbcache = {}
            self._cloud_view.set_mesh(geom["vertices"], geom["triangles"], geom["normals"])
            self._cloud_view.set_mesh_restpunkte(geom.get("rest_punkte"))
            self._cloud_view.set_mesh_schalter(self._cloud_view.mesh_an())
            st = geom["stats"]
            rest = (f", {100.0 * st['restzellen'] / max(st['zellen'], 1):.0f} % der Zellen "
                    f"als Punkte" if st.get("hybrid") else "")
            self._log(f"Mesh bereit: {_fmt_int(st['dreiecke'])} Dreiecke aus "
                      f"{_fmt_int(st.get('flaechenzellen', st['zellen']))} Flächenzellen "
                      f"({st['voxel_m'] * 100:g} cm, Tiefe {st['tiefe']}{rest}; "
                      f"Normalen {str(st.get('normalen', '')).upper()}).")
            self._mesh_farben_zeigen()

        def on_failed(msg: str) -> None:
            self._mesh_laeuft = False
            self._cloud_view.set_mesh_schalter(False)
            self._settings["mesh_an"] = False
            self._save_settings()
            if msg != "Abgebrochen":
                self._show_error("Mesh", msg)

        self._cloud_view.set_mesh_schalter(True, "Mesh …")
        self._mesh_laeuft = True
        self._start_worker("Berechne Mesh der Punktwolke …", job, on_done,
                           on_failed=on_failed)

    def _mesh_farben_zeigen(self) -> None:
        """Farben der angezeigten Ebene auf das Mesh legen (je Ebene gemerkt)."""
        geom = self._mesh_geom
        if geom is None:
            return
        from core import mesh as mesh_mod
        rgb = ok = None
        if self._colors is not None:
            schluessel = (self._layer_key, id(self._colors), id(self._valid))
            paar = self._mesh_farbcache.get(schluessel)
            if paar is None:
                paar = mesh_mod.vertex_colors(geom, self._colors, self._valid)
                self._mesh_farbcache = {schluessel: paar}   # nur die aktuelle halten
            rgb, ok = paar
        self._cloud_view.set_mesh_farben(rgb, ok, geom.get("intensity"))

    def _on_mesh_cloudcompare(self) -> None:
        """Mesh der offenen Wolke mit den Farben der angezeigten Ebene als PLY
        schreiben und in CloudCompare oeffnen. Dieselbe Geometrie wie in der
        Ansicht: ist sie schon gerechnet, kostet das nur das Schreiben."""
        if self._world is None or self._project is None:
            return
        from core import mesh as mesh_mod
        ziel = self._mesh_ziel()
        vox, tiefe, trim, hybrid = self._mesh_param()
        world, rec = self._world, self._rec
        vorhanden = self._mesh_geom if self._mesh_geom_dir == ziel else None
        farben, gueltig = self._colors, self._valid
        path = os.path.join(self._project.dir, "mesh",
                            f"mesh_{self._spin_mesh_voxel.value()}cm_t{tiefe}.ply")

        def job(progress_cb, cancel, log_cb):
            geom = vorhanden or mesh_mod.load_geometry(ziel)
            if geom is None:
                geom = mesh_mod.build_geometry(
                    world, rec.path_positions() if rec is not None else None,
                    voxel=vox, depth=tiefe, trim=trim, hybrid=hybrid,
                    progress=lambda f, m: progress_cb(0.9 * f, m),
                    cancel=lambda: cancel.is_set(), log=log_cb)
                mesh_mod.save_geometry(geom, ziel)
            rgb = None
            if farben is not None:
                progress_cb(0.92, "Mesh: Farben der Ebene …")
                rgb, _ok = mesh_mod.vertex_colors(geom, farben, gueltig)
            progress_cb(0.96, f"Schreibe {os.path.basename(path)} …")
            mesh_mod.write_mesh(mesh_mod.to_open3d(geom, rgb), path, geom["stats"])
            return path, geom["stats"]

        def on_done(res) -> None:
            p, st = res
            self._log(f"Mesh gespeichert: {p} — {_fmt_int(st['dreiecke'])} Dreiecke.")
            self._cc_open([p])

        self._start_worker("Schreibe Mesh für CloudCompare …", job, on_done)

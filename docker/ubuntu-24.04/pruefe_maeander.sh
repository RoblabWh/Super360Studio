#!/usr/bin/env bash
# Mäander-Einfärbung ohne Oberfläche durchspielen, so wie die App sie rechnet,
# auf einer Kopie eines Cache-Projekts — und das Ergebnis gegen das Projekt
# halten, aus dem die Kopie stammt.
#
#   pruefe_maeander.sh [--modus modell|ausrichten|neu] PROJEKT BILDER [ZIEL]
#
#   PROJEKT  Projektordner im Cache (wird nur gelesen), mit recording/ und
#            meander/; dessen Lage und Farbebenen sind der Vergleich.
#   BILDER   Ordner des Mäanderfluges (wird nur gelesen).
#   ZIEL     Arbeitsordner fuer die Kopie und das Ergebnis
#            (Vorgabe: ${TMPDIR:-/tmp}/super360_maeander). SUPER360_CACHE_ROOT
#            zeigt waehrend des Laufs auf ZIEL/cache.
#
# Modi (wie viel vom Arbeitsordner meander/ die Kopie behaelt):
#   modell      alles — COLMAP-Modell, Lage, Optik; nur Einfaerben
#   ausrichten  COLMAP-Modell und Optik bleiben; ausrichten, einmessen,
#               feinausrichten, einfaerben (die Automatik der App)
#   neu         nichts — Fotos, COLMAP, Georeferenz und dann wie „ausrichten“
#
# Thermal folgt der Einstellung des Projekts (meander_thermal), die
# Sichtpruefung ebenso (meander_sichtbar).
#
# Im Container (Repo unter /super360, Nachbarrepo unter ~/PointCloudMerger):
#
#   docker run --rm --user "$(id -u):$(id -g)" \
#       -v "$PWD":/super360:ro -v <PointCloudMerger>:/home/super360/PointCloudMerger:ro \
#       -v <projekt>:/projekt:ro -v <bilder>:/bilder:ro -v <ziel>:/ziel \
#       super360-u2404-colmap:latest \
#       bash docker/ubuntu-24.04/pruefe_maeander.sh --modus neu /projekt /bilder /ziel
#
# Ergebnis: ZIEL/ergebnis_<modus>.json und je Schritt eine Zeile auf stdout.
# Die Lage wird ueber die Kameras verglichen, nicht ueber Gier und
# Verschiebung — die beziehen sich auf den Rahmen des jeweiligen COLMAP-Modells
# und sind zwischen zwei Rekonstruktionen nicht dasselbe. Je Kamera: Abstand
# der Zentren in Metern und Drehwinkel zwischen den Blickrichtungen in Grad,
# beides im Rahmen der Karte.

set -euo pipefail

MODUS=modell
if [[ "${1:-}" == "--modus" ]]; then
    MODUS="$2"
    shift 2
fi
case "$MODUS" in modell|ausrichten|neu) ;; *)
    echo "Unbekannter Modus: $MODUS (modell, ausrichten, neu)" >&2; exit 2;;
esac
if [[ $# -lt 2 ]]; then
    sed -n '2,20p' "$0" >&2
    exit 2
fi
PROJEKT="$(realpath "$1")"
BILDER="$(realpath "$2")"
ZIEL="$(realpath -m "${3:-${TMPDIR:-/tmp}/super360_maeander}")"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAME="$(basename "$PROJEKT")"

[[ -d "$PROJEKT/recording" ]] || { echo "Keine Aufzeichnung in $PROJEKT" >&2; exit 2; }
[[ -d "$BILDER" ]] || { echo "Kein Bilderordner: $BILDER" >&2; exit 2; }

# Zwei Kopien: „lauf“ wird gerechnet, „bezug“ bleibt wie das Projekt und
# liefert die Vergleichslage (die Pipeline schreibt beim Laden gps_enu.json,
# darum nicht das Projekt selbst).
KOPIE="$ZIEL/cache/$NAME"
BEZUG="$ZIEL/bezug/$NAME"
rm -rf "$KOPIE" "$BEZUG"
mkdir -p "$KOPIE" "$BEZUG"
cp -a "$PROJEKT/recording" "$PROJEKT/settings.json" "$KOPIE/"
if [[ -d "$PROJEKT/meander" ]]; then
    cp -a "$PROJEKT/meander" "$KOPIE/"
    cp -a "$PROJEKT/meander" "$BEZUG/"
fi
W="$KOPIE/meander"
case "$MODUS" in
    ausrichten)
        rm -f "$W/align.json" "$W/rgb_zuschlag.json" "$W/thermal_lage.json"
        ;;
    neu)
        rm -rf "$W"
        ;;
esac

echo "Modus $MODUS: $NAME -> $KOPIE"
cd "$REPO"
export SUPER360_CACHE_ROOT="$ZIEL/cache" PYTHONDONTWRITEBYTECODE=1
python3 - "$MODUS" "$KOPIE" "$BEZUG" "$PROJEKT" "$BILDER" \
    "$ZIEL/ergebnis_$MODUS.json" <<'PY'
import json
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.getcwd())
from core import meander as meander_mod  # noqa: E402
from core import optik as optik_mod  # noqa: E402
from core.ebenen import laden as lade_ebene  # noqa: E402
from core.project import Project  # noqa: E402
from core.recording import lade_mit_hinweis  # noqa: E402
from ui.feinregler import raster_prozent  # noqa: E402

modus, kopie, bezug, projekt, bilder, ausgabe = sys.argv[1:7]
t_start = time.time()
erg = {"modus": modus, "python": sys.version.split()[0], "schritte": {}}


def log(m):
    print(f"  {m}", flush=True)


def schritt(name, t0, **werte):
    werte["sekunden"] = round(time.time() - t0, 1)
    erg["schritte"][name] = werte
    print(f"[{name}] " + ", ".join(f"{k}={v}" for k, v in werte.items()), flush=True)


def ohne(f, m):
    pass


cancel = threading.Event()
proj = Project.from_dir(kopie)
settings = proj.load_settings()
thermal = bool(settings.get("meander_thermal", False))
sichtbar = bool(settings.get("meander_sichtbar", False))
work = proj.meander_work_dir()

t0 = time.time()
rec = lade_mit_hinweis(proj.recording_dir(), proj.bag_path, log)
welt = rec.world_points()
alt = os.path.join(bezug, "meander", "cloud.npy")
gleich = None
if os.path.exists(alt):
    gleich = bool(np.array_equal(np.load(alt), meander_mod.thin(welt)))
schritt("wolke", t0, punkte=len(welt), wie_cloud_npy=gleich)

args = {"points": welt, "photo_dir": bilder, "work_dir": work, "thermal": thermal,
        "rgb_versatz": [0.0, 0.0], "thermal_versatz": [0.0, 0.0]}


def setze_optik(pipe, d):
    """Wie _meander_setze_optik: Massstaebe auf das Raster der Regler."""
    d = dict(d)
    pipe.s360_korrektur = d.get("korrektur")
    for o in ("rgb", "thermal"):
        prozent = (float(d.get(f"{o}_faktor", 1.0)) - 1.0) * 100.0
        d[f"{o}_faktor"] = 1.0 + raster_prozent(prozent) / 100.0
    return d


pipe = None
optik = None
if modus in ("ausrichten", "neu"):
    # „Ausrichten“ der App, dann die Automatik: einmessen, feinausrichten
    t0 = time.time()
    pipe = meander_mod.bauen(args, log)
    pipe._cancel = lambda: cancel.is_set()
    vor = meander_mod.prepare(pipe)
    schritt("vorbereiten", t0, colmap_python=meander_mod.find_colmap_python(), **vor)
    t0 = time.time()
    k = meander_mod.align(pipe)
    schritt("ausrichten", t0, yaw_deg=round(k["yaw_deg"], 3),
            t=[round(x, 3) for x in k["t"]],
            anteil_auf_flaeche=k.get("anteil_auf_flaeche"),
            anteil_auf_flaeche_optik=k.get("anteil_auf_flaeche_optik"),
            pruefung=meander_mod.pruefe_ausrichtung(k))
    optik = setze_optik(pipe, optik_mod.laden(work))
    if optik_mod.vorhanden(work):
        optik["korrektur"] = None       # gehoerte zur alten Lage
        pipe.s360_korrektur = None
        optik_mod.speichern(work, optik)

    t0 = time.time()
    yaw, t = float(np.degrees(pipe.yaw)), meander_mod.as_t3(pipe.t)
    punkte = np.asarray(welt[:: max(1, len(welt) // 400_000)], dtype=np.float64)
    res = optik_mod.einmessen(pipe, punkte, yaw, t, pipe.photo_dir,
                              thermal=bool(getattr(pipe, "thermal", False)),
                              progress=ohne, cancel=lambda: cancel.is_set(), log=log)
    if res["dz"]:
        meander_mod.set_manual(pipe, yaw, t + np.array([0.0, 0.0, res["dz"]]))
    optik = setze_optik(pipe, {"rgb_faktor": res["rgb"]["faktor"], "thermal_faktor": 1.0,
                               "thermal": res["thermal"], "hoehe": res["hoehe"],
                               "rgb": res["rgb"], "korrektur": optik.get("korrektur")})
    optik_mod.speichern(work, optik)
    schritt("einmessen", t0, dz=round(float(res["dz"]), 3),
            rgb_faktor=round(optik["rgb_faktor"], 4),
            auf_flaeche_nachher=res["rgb"].get("auf_flaeche_nachher"),
            thermal=res["thermal"] is not None)

    t0 = time.time()
    yaw, t = float(np.degrees(pipe.yaw)), meander_mod.as_t3(pipe.t)
    kf = optik_mod.feinausrichten(pipe, punkte, yaw, t, optik["rgb_faktor"],
                                  progress=ohne, cancel=lambda: cancel.is_set(), log=log)
    optik["korrektur"] = kf if kf["stufe"] != "nichts" else None
    optik = setze_optik(pipe, optik)
    optik_mod.speichern(work, optik)
    schritt("feinausrichten", t0, stufe=kf["stufe"],
            neigung_grad=[round(x, 3) for x in kf["neigung_grad"]],
            versatz_m=[round(x, 3) for x in kf["versatz_m"]],
            auf_flaeche_nachher=kf.get("auf_flaeche_nachher"))

# --- Einfaerben, wie _on_meander_run
t0 = time.time()
th_zuschlag = meander_mod.load_thermal_zuschlag(work)
p, th, opt = meander_mod.bereit_machen(pipe, args, th_zuschlag, optik, ohne, cancel,
                                       log, 0.02, 0.50)
normalen = None
if sichtbar:
    from core import sichtbar as sichtbar_mod
    normalen = sichtbar_mod.normalen(welt)


def faerben(cams, ordner, A_, b_, temperatur=None):
    if sichtbar:
        return sichtbar_mod.colorize_sichtbar(
            welt, cams, ordner, A_, b_, normalen_welt=normalen, progress=ohne,
            cancel=lambda: cancel.is_set(), log=log, temperatur=temperatur)
    return meander_mod.colorize_points(welt, cams, ordner, A_, b_, progress=ohne,
                                       cancel=lambda: cancel.is_set(),
                                       temperatur=temperatur)


kam = meander_mod.kameras(p, opt, th, thermal=False)
rgb, maske = faerben(kam["rgb"], p._p("images"), kam["A"], kam["b"])
meander_mod.save_layer(proj.layer_dir("meander_rgb"), rgb, maske,
                       {"quelle": "meander_rgb", "flug": p.photo_dir,
                        "yaw_deg": kam["yaw_deg"], "anteil": float(maske.mean()),
                        "sichtpruefung": sichtbar,
                        "feinausrichtung": getattr(p, "s360_korrektur", None)})
schritt("einfaerben_rgb", t0, anteil=round(float(maske.mean()), 6),
        rgb_faktor=round(kam["rgb_faktor"], 4), sichtpruefung=sichtbar)

if thermal:
    from core import temperatur as temperatur_mod
    t0 = time.time()
    tf, rf = kam["thermal_faktor"], kam["rgb_faktor"]
    th_cams = optik_mod.thermal_cams(p, opt.get("thermal"), tf, rf)
    if th_cams is None:
        schritt("einfaerben_thermal", t0, uebersprungen="keine Thermaloptik")
    else:
        yaw_th, t_th = meander_mod.thermal_lage(kam["yaw_deg"], p.t, th)
        A_th, b_th = meander_mod.lage_affine(p, yaw_th, t_th)
        e = faerben(th_cams, p._p("thermal"), A_th, b_th,
                    temperatur=temperatur_mod.quelle(p))
        ttemp = e[2] if len(e) > 2 else None
        meander_mod.save_layer(proj.layer_dir("meander_thermal"), e[0], e[1],
                               {"quelle": "meander_thermal", "anteil": float(e[1].mean())},
                               temperatur=ttemp)
        n_temp = None if ttemp is None else float(np.isfinite(ttemp).mean())
        schritt("einfaerben_thermal", t0, anteil=round(float(e[1].mean()), 6),
                anteil_mit_temperatur=n_temp)


# --- Vergleich mit dem Projekt
t0 = time.time()
bw = os.path.join(bezug, "meander")
if os.path.exists(os.path.join(bw, "align.json")) and os.path.exists(
        os.path.join(bw, "cameras.npz")):
    from colorize_pipeline import pipeline as pl
    q = pl.Pipeline(cloud_path=os.path.join(bw, "cloud.npy"), photo_dir=bilder,
                    work_dir=bw, out_path=os.path.join(bw, "x.ply"))
    q.load_cloud()
    q.gps = json.load(open(os.path.join(bw, "gps.json")))
    q.cams = np.load(os.path.join(bw, "cameras.npz"), allow_pickle=True)
    q.georeference()
    q.align()
    q.t = meander_mod.as_t3(q.t)
    q.s360_korrektur = optik_mod.laden(bw).get("korrektur")

    def lage(pp):
        A, b = meander_mod.lage_affine(pp, float(np.degrees(pp.yaw)), pp.t)
        A = np.asarray(A, float)
        R_kc = A / np.cbrt(np.linalg.det(A))
        C = (A @ np.asarray(pp.cams["C"], float).T).T + b
        R = np.asarray(pp.cams["Rcw"], float) @ R_kc.T
        return {str(n): (C[i], R[i]) for i, n in enumerate(pp.cams["names"])}

    ln, lb = lage(p), lage(q)
    gemeinsam = sorted(set(ln) & set(lb))
    dm = np.array([np.linalg.norm(ln[n][0] - lb[n][0]) for n in gemeinsam])
    dg = np.array([np.degrees(np.arccos(np.clip(
        (np.trace(ln[n][1] @ lb[n][1].T) - 1.0) / 2.0, -1.0, 1.0))) for n in gemeinsam])
    schritt("vergleich_lage", t0, kameras=f"{len(gemeinsam)} von {len(lb)}",
            yaw_deg=[round(float(np.degrees(p.yaw)), 3), round(float(np.degrees(q.yaw)), 3)],
            abstand_m_median=round(float(np.median(dm)), 4),
            abstand_m_max=round(float(dm.max()), 4),
            winkel_grad_median=round(float(np.median(dg)), 4),
            winkel_grad_max=round(float(dg.max()), 4))

for key in ("meander_rgb", "meander_thermal") if thermal else ("meander_rgb",):
    t0 = time.time()
    ordner = os.path.join(projekt, Project.LAYERS[key])
    alt = lade_ebene(ordner, len(welt))
    neu = lade_ebene(proj.layer_dir(key), len(welt))
    if alt is None or neu is None:
        continue
    with open(os.path.join(ordner, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    beide = alt[1] & neu[1]
    diff = np.abs(alt[0][beide].astype(np.int16) - neu[0][beide].astype(np.int16))
    schritt(f"vergleich_{key}", t0, anteil_projekt=round(float(alt[1].mean()), 6),
            anteil_jetzt=round(float(neu[1].mean()), 6),
            maske_gleich=round(float((alt[1] == neu[1]).mean()), 6),
            farbe_gleich=round(float((diff.max(1) == 0).mean()), 6) if len(diff) else None,
            farbe_mittl_abw=round(float(diff.mean()), 3) if len(diff) else None,
            yaw_projekt=meta.get("yaw_deg"))

erg["sekunden"] = round(time.time() - t_start, 1)
with open(ausgabe, "w", encoding="utf-8") as fh:
    json.dump(erg, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print(f"FERTIG in {erg['sekunden']} s -> {ausgabe}", flush=True)
PY

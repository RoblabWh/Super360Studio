#!/usr/bin/env python3
"""Gaussian Splat auf der Lidar-Karte trainieren — Farben, nicht Geometrie.

Laeuft im Interpreter mit torch und gsplat (s. ``core/splat.py``,
``find_splat_python``), getrennt vom System-Python der App, so wie COLMAP im
Nachbarrepo. stdout ist Protokoll, eine Zeile je Meldung:

    INFO <text>
    STEP <i> <n> <verlust> <psnr>
    EVAL <name> <l1_roh> <l1_angepasst> <psnr_angepasst> <pixel>
    DONE <pfad>

Aufruf:

    splat_train.py <datensatz>            # GPU, gsplat
    splat_train.py <datensatz> --cpu      # dichter Renderer in reinem torch (nur winzig)
    splat_train.py --selbsttest           # synthetische Szene auf der CPU

Warum kein freies Splat: ein Splat aus COLMAP allein legt seine Gaussians dahin,
wo die Photogrammetrie sie vermutet, nicht auf die Lidar-Oberflaeche. Der
Versuch mit dem Avata-360-Splat ist genau daran gescheitert (PointCloudMerger,
„Was nicht funktioniert hat“). Hier sitzt jede Gaussian auf einem Anker der
Karte — dem Schwerpunkt einer Voxelzelle — und darf nur um eine halbe Zelle
entlang der Normalen gleiten. Gelernt werden Farbe, Deckkraft, Form und Lage
der Scheibe. Damit gehoert jede gelernte Farbe zu genau einem Ort der Karte.

Was das Training gegenueber der direkten Projektion leistet:

* **alle Bilder zugleich.** Jede Farbe erklaert jedes Bild, das den Ort sieht,
  statt aus einem einzigen zu stammen — keine Naehte zwischen Bildern.
* **Belichtung je Bild** (3×3-Matrix plus Versatz, um den Mittelwert zentriert):
  was eine Kamera heller oder waermer abbildet, landet dort und nicht in der
  Farbe des Punktes.
* **Verdeckung beim Rendern.** Eine Gaussian hinter einer anderen bekommt
  keinen Gradienten; ob ein Punkt gesehen wurde, sagt ihr Gewicht.
* **Blickabhaengiges in den hoeheren SH-Baendern**, die Grundfarbe bleibt frei
  von Spiegelungen.
* **Posen nachfuehren** (optional): je Bild eine kleine starre Korrektur gegen
  die feste Lidar-Geometrie — eine photometrische Buendelausgleichung.

**Gemeinsamer Datensatz** (Onboard + Maeander, ``core/splat.py``,
``datensatz_gemeinsam``): ``ansichten.npz`` traegt dann je Ansicht ``bezug``,
``herkunft`` und ``reichweite``, dazu wahlweise ``bel0_D``/``bel0_e``. Die
Bezugsansichten (Maeander) werden **nicht** belichtet — ihre Farbe ist die
Farbe des Splats. Alle anderen bekommen ihre Matrix ohne Zentrierung, gestartet
und leicht gehalten bei ``bel0`` (die Abbildung aus ``core/fusion.py``). So
landet der Unterschied zwischen den Kameras in den Matrizen der Onboard-Bilder,
und die gelernte Farbe steht im Farbraum des Maeanders. Gezogen wird
abwechselnd aus jeder Herkunft, sonst uebertoenen 3000 Wuerfelseiten die 255
Maeanderbilder.

**Startfarbe** (:func:`startfarbe`): vor dem Training bekommt jeder Anker das
gewichtete Mittel der Pixel, die ihn in den Trainingsbildern zeigen. Ab Grau
reichten 3000 Schritte nicht: am Maeanderflug 09-08 lag das Splat dann 18 %
vor der direkten Projektion, mit Startfarbe 26 % — so weit kam es ab Grau erst
nach 9000 Schritten. Aus den Trainingsbildern selbst gerechnet, kennt die
Startfarbe die Pruefbilder nie.

**Farbe je Kartenpunkt** (:func:`abtasten`): nach dem Training wird jede
Trainingsansicht ohne Belichtung gerendert, und jeder Punkt bekommt den Wert
des Renderings an seinem Pixel — nach denselben Regeln wie die direkte
Projektion (``core/sichtbar.py``): sichtbar innerhalb von 30 cm plus 1 % der
Tiefe, nicht streifend, aus der frontalsten Ansicht. Frueher wurde die
gelernte Farbe jeder einzelnen Gaussian genommen und ueber die Summe ihrer
Blendgewichte entschieden, ob sie gesehen ist. Am Maeanderflug war das
schlechter als die direkte Projektion (+4 %, nach 9000 Schritten +13 %): die
Gaussians ueberlappen, und erst ihre Mischung trifft das Bild, die einzelne
Farbe traegt das Rauschen der Zerlegung. Und die Haelfte der sichtbaren Anker
blieb ungefaerbt — eine Scheibe wenige Zentimeter hinter der vordersten bekommt
kaum Gewicht, auf einem Trapezblechdach gab das Streifen.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time

import numpy as np

# Grosse Maeanderbilder (1600x1200) und kleine Wuerfelseiten im Wechsel
# zerstueckeln den Grafikspeicher: am 2026-09-23 fehlten auf 8 GB 400 MB,
# waehrend 1,3 GB reserviert und frei lagen. Muss vor dem Import von torch stehen.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch.nn.functional as F

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
         -1.0925484305920792, 0.5462742152960396)

#: Vorgaben, ``param.json`` im Datensatz ueberschreibt sie einzeln.
VORGABE = {
    "schritte": 3000,
    "sh_grad": 1,              # 0 = keine Blickabhaengigkeit; hoechstens 2
    "posen": True,             # Posen nachfuehren
    "posen_ab": 0.2,           # ab diesem Anteil der Schritte
    "belichtung": True,
    "ssim": 0.2,               # Anteil SSIM am Verlust (nur bei Farbe)
    "deckung": 0.01,           # Strafe fuer Luecken im Lidar-Bereich
    "lr_farbe": 2.5e-3,
    "lr_deckkraft": 2.5e-2,
    "lr_form": 2.5e-3,
    "lr_versatz": 2.5e-3,
    "lr_belichtung": 1e-3,
    "belichtung_gemeinsam": True,
    "ziehen": {},                  # Gewicht je Herkunft, z. B. {"maeander": 2}; sonst 1  # Nicht-Bezugsansichten: gemeinsame Matrix + Abweichung
    "lr_pose_dreh": 2e-5,      # rad je Schritt (Adam)
    "lr_pose_weg": 2e-4,       # m je Schritt
    "vergleichsbilder": 4,
    "startfarbe": True,        # Anker vorab aus den Trainingsbildern faerben
    # Sichtbarkeit beim Abtasten, wie core/sichtbar.py (core/splat.py schreibt
    # die dortigen Werte in param.json)
    "min_cos": 0.12,
    "tiefe_skala": 0.5,
    "toleranz_m": 0.30,
    "toleranz_rel": 0.01,
    "min_deckung": 0.5,        # Deckkraft des Renderings am Pixel, darunter kein Wert
    "reichweite": 0.0,         # m um die Kamera, 0 = alle Punkte
}

# ------------------------------------------------------------ Protokoll

def melde(art: str, *teile) -> None:
    print(art, *teile, flush=True)


# ----------------------------------------------------------- Mathematik

def sh_farbe(grad: int, dirs: torch.Tensor, koeff: torch.Tensor) -> torch.Tensor:
    """Kugelflaechenfunktionen wie 3DGS/gsplat: koeff [M, K, D], dirs [M, 3].

    ``dirs`` zeigt von der Kamera zur Gaussian (normiert). Ergebnis ohne den
    Versatz 0,5 — den legt :func:`farben_fuer` drauf.
    """
    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
    out = SH_C0 * koeff[:, 0]
    if grad >= 1:
        out = out - SH_C1 * y * koeff[:, 1] + SH_C1 * z * koeff[:, 2] - SH_C1 * x * koeff[:, 3]
    if grad >= 2:
        out = (out + SH_C2[0] * (x * y) * koeff[:, 4] + SH_C2[1] * (y * z) * koeff[:, 5]
               + SH_C2[2] * (2 * z * z - x * x - y * y) * koeff[:, 6]
               + SH_C2[3] * (x * z) * koeff[:, 7] + SH_C2[4] * (x * x - y * y) * koeff[:, 8])
    return out


def hut(w: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(w[:, 0])
    return torch.stack([torch.stack([z, -w[:, 2], w[:, 1]], -1),
                        torch.stack([w[:, 2], z, -w[:, 0]], -1),
                        torch.stack([-w[:, 1], w[:, 0], z], -1)], -2)


def exp_so3(w: torch.Tensor) -> torch.Tensor:
    """Drehvektor [V, 3] -> Drehmatrix [V, 3, 3], auch um null differenzierbar."""
    t2 = (w * w).sum(-1)
    t = torch.sqrt(t2.clamp_min(1e-20))
    klein = t2 < 1e-6
    a = torch.where(klein, 1.0 - t2 / 6.0, torch.sin(t) / t)
    b = torch.where(klein, 0.5 - t2 / 24.0, (1.0 - torch.cos(t)) / t2.clamp_min(1e-20))
    W = hut(w)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(W)
    return eye + a[:, None, None] * W + b[:, None, None] * (W @ W)


def quat_matrix(q: torch.Tensor) -> torch.Tensor:
    """Quaternion [M, 4] in gsplat-Reihenfolge (w, x, y, z) -> [M, 3, 3]."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                       -1).reshape(-1, 3, 3)


def quat_aus_normalen(n: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z), das die lokale z-Achse auf die Normale dreht.

    Das Vorzeichen der Normalen ist beliebig (eine Gaussian ist symmetrisch);
    nach oben gewendet bleibt ``1 + n_z >= 1`` und die Formel stabil.
    """
    n = np.asarray(n, np.float64)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    n = np.where(n[:, 2:3] < 0, -n, n)
    q = np.stack([1.0 + n[:, 2], -n[:, 1], n[:, 0], np.zeros(len(n))], 1)
    return (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


# ---------------------------------------------------------- Gaussians

class Splats(torch.nn.Module):
    """Gaussians auf den Ankern der Karte.

    Lage:     Anker + Normale · Voxel/2 · tanh(versatz)
    Groesse:  Voxel · (0,05 + 1,95 · sigmoid(form)), je Achse; lokale z = Normale
    Farbe:    SH-Koeffizienten (D = 3) oder direkt D Kanaele (Temperatur)
    """

    def __init__(self, anker: dict, kanaele: int, sh_grad: int, device):
        super().__init__()
        pos = torch.from_numpy(np.asarray(anker["pos"], np.float32))
        normal = torch.from_numpy(np.asarray(anker["normal"], np.float32))
        M = len(pos)
        self.voxel = float(anker["voxel"])
        self.kanaele = int(kanaele)
        self.sh = self.kanaele == 3
        self.sh_grad = int(sh_grad) if self.sh else 0
        self.register_buffer("pos", pos)
        self.register_buffer("normal", F.normalize(normal, dim=-1))
        farbe0 = np.asarray(anker.get("farbe0", np.full((M, kanaele), 0.5)), np.float32)
        farbe0 = torch.from_numpy(farbe0.reshape(M, kanaele))
        if self.sh:
            self.sh0 = torch.nn.Parameter(((farbe0 - 0.5) / SH_C0)[:, None, :])
            self.shN = torch.nn.Parameter(torch.zeros(M, (self.sh_grad + 1) ** 2 - 1, 3))
        else:
            self.werte = torch.nn.Parameter(farbe0.clone())
        tang, quer = (0.5 - 0.05) / 1.95, (0.1 - 0.05) / 1.95
        form = torch.tensor([logit(tang), logit(tang), logit(quer)]).repeat(M, 1)
        self.form = torch.nn.Parameter(form)
        self.quats = torch.nn.Parameter(torch.from_numpy(
            quat_aus_normalen(anker["normal"])))
        self.deckkraft_roh = torch.nn.Parameter(torch.full((M,), logit(0.8)))
        self.versatz = torch.nn.Parameter(torch.zeros(M))
        self.to(device)

    def means(self) -> torch.Tensor:
        return self.pos + self.normal * (0.5 * self.voxel * torch.tanh(self.versatz))[:, None]

    def scales(self) -> torch.Tensor:
        return self.voxel * (0.05 + 1.95 * torch.sigmoid(self.form))

    def deckkraft(self) -> torch.Tensor:
        return torch.sigmoid(self.deckkraft_roh)

    def farben_fuer(self, means: torch.Tensor, campos: torch.Tensor, grad: int) -> torch.Tensor:
        if not self.sh:
            return self.werte
        koeff = torch.cat([self.sh0, self.shN], 1)
        dirs = F.normalize(means - campos[None], dim=-1)
        return (sh_farbe(min(grad, self.sh_grad), dirs, koeff) + 0.5).clamp_min(0.0)


# ------------------------------------------------------------ Rendern

def rendere_dicht(means, quats, scales, deck, farben, viewmat, K, W, H):
    """Rasterisierung in reinem torch, fuer jede Pixel-Gaussian-Paarung.

    Dasselbe Modell wie gsplat (EWA-Projektion, Tiefpass 0,3 px², Alpha bis
    0,999, unter 1/255 verworfen, von vorn nach hinten gemischt) — aber dicht und
    damit nur fuer winzige Szenen. Der Selbsttest laeuft darauf, ohne CUDA.
    """
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    mc = means @ R.T + t
    idx = torch.nonzero(mc[:, 2] > 0.05).squeeze(1)
    D = farben.shape[1]
    if len(idx) == 0:
        return torch.zeros(H, W, D), torch.zeros(H, W, 1), torch.zeros(H, W, 1)
    ordnung = idx[torch.argsort(mc[idx, 2])]
    mc = mc[ordnung]
    z = mc[:, 2]
    Mq = quat_matrix(quats[ordnung]) * scales[ordnung][:, None, :]
    cov = R @ (Mq @ Mq.transpose(1, 2)) @ R.T
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    null = torch.zeros_like(z)
    J = torch.stack([torch.stack([fx / z, null, -fx * mc[:, 0] / z ** 2], -1),
                     torch.stack([null, fy / z, -fy * mc[:, 1] / z ** 2], -1)], -2)
    c2 = J @ cov @ J.transpose(1, 2) + 0.3 * torch.eye(2)
    det = (c2[:, 0, 0] * c2[:, 1, 1] - c2[:, 0, 1] * c2[:, 1, 0]).clamp_min(1e-10)
    ca, cb, cc = c2[:, 1, 1] / det, -c2[:, 0, 1] / det, c2[:, 0, 0] / det
    mu_x, mu_y = fx * mc[:, 0] / z + cx, fy * mc[:, 1] / z + cy
    ys, xs = torch.meshgrid(torch.arange(H) + 0.5, torch.arange(W) + 0.5, indexing="ij")
    dx = xs.reshape(-1, 1) - mu_x[None]
    dy = ys.reshape(-1, 1) - mu_y[None]
    potenz = -0.5 * (ca * dx * dx + cc * dy * dy) - cb * dx * dy
    alpha = (deck[ordnung][None] * torch.exp(potenz.clamp_max(0.0))).clamp_max(0.999)
    alpha = torch.where(alpha < 1.0 / 255.0, torch.zeros_like(alpha), alpha)
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha[:, :-1]], 1), 1)
    w = alpha * T
    bild = w @ farben[ordnung]
    deckung = w.sum(1)
    tiefe = (w @ z) / deckung.clamp_min(1e-10)
    return bild.reshape(H, W, D), deckung.reshape(H, W, 1), tiefe.reshape(H, W, 1)


def rendern(means, quats, scales, deck, farben, viewmat, K, W, H, cpu: bool,
            mit_tiefe: bool = False):
    """(Bild [H, W, D], Deckung [H, W, 1]) — mit ``mit_tiefe`` dazu die
    erwartete Tiefe [H, W, 1] (Kamera-z, durch die Deckung geteilt)."""
    if cpu:
        erg = rendere_dicht(means, quats, scales, deck, farben, viewmat, K, W, H)
        return erg if mit_tiefe else erg[:2]
    from gsplat.rendering import rasterization  # noqa: PLC0415
    r, a, _ = rasterization(
        means=means, quats=quats, scales=scales, opacities=deck, colors=farben,
        viewmats=viewmat[None], Ks=K[None], width=int(W), height=int(H),
        near_plane=0.05, packed=False, render_mode="RGB+ED" if mit_tiefe else "RGB")
    if mit_tiefe:
        return r[0, ..., :-1], a[0], r[0, ..., -1:]
    return r[0], a[0]


# ------------------------------------------------------------- Verlust

def _fenster(kanaele: int, device) -> torch.Tensor:
    g = torch.exp(-((torch.arange(11, dtype=torch.float32) - 5.0) ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(kanaele, 1, 11, 11).contiguous().to(device)


def ssim_karte(a: torch.Tensor, b: torch.Tensor, fenster: torch.Tensor) -> torch.Tensor:
    """SSIM je Pixel, Eingang [H, W, C] in 0..1."""
    C = a.shape[2]
    a = a.permute(2, 0, 1)[None]
    b = b.permute(2, 0, 1)[None]
    f = lambda x: F.conv2d(x, fenster, padding=5, groups=C)  # noqa: E731
    ma, mb = f(a), f(b)
    saa = f(a * a) - ma * ma
    sbb = f(b * b) - mb * mb
    sab = f(a * b) - ma * mb
    k1, k2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * ma * mb + k1) * (2 * sab + k2)) / ((ma * ma + mb * mb + k1) * (saa + sbb + k2))
    return s[0].mean(0)


def affin_anpassen(pred: torch.Tensor, gt: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Je Kanal Verstaerkung und Versatz nach kleinsten Quadraten — fuer Bilder,
    deren Belichtung das Training nie gesehen hat."""
    out = pred.clone()
    for c in range(pred.shape[2]):
        x, y = pred[..., c][m], gt[..., c][m]
        if x.numel() < 10:
            continue
        vx = x.var()
        g = ((x - x.mean()) * (y - y.mean())).mean() / vx if vx > 1e-8 else torch.tensor(1.0)
        out[..., c] = g * pred[..., c] + (y.mean() - g * x.mean())
    return out


# ------------------------------------------------------------ Datensatz

def speichergrenze() -> int:
    """So viele Bytes Bilder duerfen in den Speicher: ein Drittel des freien.

    Ein Onboard-Datensatz mit 3000 Wuerfelseiten zu 640x640 waere als Tensoren
    rund 5 GB; auf einem Rechner mit 7,5 GB (so war es am 2026-09-16) faende der
    Trainer sein Ende im OOM-Killer. Auf dem Rechner mit 124 GB dagegen kostete
    die feste Grenze von 1,2 GB bei jedem Schritt einen Lesezugriff. Ohne
    ``/proc/meminfo`` bleibt es bei 1,2 GB.
    """
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for zeile in fh:
                if zeile.startswith("MemAvailable:"):
                    return max(1_200_000_000, int(zeile.split()[1]) * 1024 // 3)
    except (OSError, ValueError, IndexError):
        pass
    return 1_200_000_000


class Datenquelle:
    """Bilder und Masken des Datensatzes — im Speicher, solange sie hineinpassen.

    Ab ``grenze`` (s. :func:`speichergrenze`) kommen Bild und Maske je Schritt
    von der Platte; ein kleiner LRU haelt die zuletzt benutzten.
    """

    def __init__(self, ordner, namen, masken, breite, hoehe, kanaele,
                 grenze=None, lru=48):
        self.ordner, self.namen, self.masken = ordner, list(namen), list(masken)
        je = kanaele * (2 if str(namen[0]).endswith(".npy") else 1) + 1
        self.gesamt = int(sum(int(w) * int(h) * je for w, h in zip(breite, hoehe)))
        self.im_speicher = self.gesamt <= (speichergrenze() if grenze is None else grenze)
        self.lru = max(4, int(lru))
        self._cache = {}
        self._alter = []
        if self.im_speicher:
            for i in range(len(self.namen)):
                self._cache[i] = self._lade(i)

    def _lade(self, i):
        from PIL import Image  # noqa: PLC0415
        pfad = os.path.join(self.ordner, str(self.namen[i]))
        if pfad.endswith(".npy"):
            bild = torch.from_numpy(np.load(pfad).astype(np.float16))[..., None]
        else:
            with Image.open(pfad) as im:
                bild = torch.from_numpy(np.array(im.convert("RGB")))
        with Image.open(os.path.join(self.ordner, str(self.masken[i]))) as im:
            maske = torch.from_numpy(np.array(im.convert("L")) > 127)
        return bild, maske

    def hole(self, i):
        i = int(i)
        paar = self._cache.get(i)
        if paar is None:
            paar = self._cache[i] = self._lade(i)
            self._alter.append(i)
            while len(self._alter) > self.lru:
                self._cache.pop(self._alter.pop(0), None)
        return paar

    def __len__(self):
        return len(self.namen)


def lade_datensatz(ordner: str) -> dict:
    ans = np.load(os.path.join(ordner, "ansichten.npz"), allow_pickle=False)
    anker = dict(np.load(os.path.join(ordner, "anker.npz"), allow_pickle=False))
    cfg = dict(VORGABE)
    pfad = os.path.join(ordner, "param.json")
    if os.path.isfile(pfad):
        with open(pfad, encoding="utf-8") as fh:
            cfg.update(json.load(fh))
    kanaele = 1 if str(ans["bild"][0]).endswith(".npy") else 3
    daten = Datenquelle(ordner, ans["bild"], ans["maske"], ans["breite"], ans["hoehe"],
                        kanaele)
    pfad = os.path.join(ordner, "punkte.npy")
    if not os.path.isfile(pfad) or "index" not in anker:
        raise RuntimeError("Datensatz ohne Kartenpunkte (punkte.npy, Zellindex) — "
                           "aus einer älteren Version, bitte neu anlegen.")
    punkte = np.load(pfad)
    if len(punkte) != len(anker["index"]):
        raise RuntimeError("punkte.npy und Zellindex passen nicht zusammen.")
    V = len(ans["bild"])
    hat = set(ans.files)
    bezug = ans["bezug"].astype(bool) if "bezug" in hat else np.zeros(V, bool)
    reich = (ans["reichweite"].astype(np.float32) if "reichweite" in hat
             else np.full(V, float(cfg["reichweite"]), np.float32))
    herkunft = ([str(h) for h in ans["herkunft"]] if "herkunft" in hat
                else [str(ans["art"]) if "art" in hat else ""] * V)
    bel0 = ((ans["bel0_D"].astype(np.float32), ans["bel0_e"].astype(np.float32))
            if "bel0_D" in hat else None)
    return {"bezug": bezug, "reichweite": reich, "herkunft": herkunft, "bel0": bel0,
            "viewmat": torch.from_numpy(ans["viewmat"].astype(np.float32)),
            "K": torch.from_numpy(ans["K"].astype(np.float32)),
            "breite": ans["breite"].astype(int), "hoehe": ans["hoehe"].astype(int),
            "holdout": ans["holdout"].astype(bool), "namen": [str(n) for n in ans["bild"]],
            "daten": daten, "anker": anker, "cfg": cfg, "kanaele": kanaele,
            "punkte": punkte}


def als_float(b: torch.Tensor, device) -> torch.Tensor:
    if b.dtype == torch.uint8:
        return b.to(device).float() / 255.0
    return b.to(device).float()


# ------------------------------------------------ Sichtbarkeit, Abtasten

class Wolke:
    """Punkte auf dem Rechengeraet, nach x sortiert — mit ``reichweite`` wird
    je Ansicht nur der Streifen um die Kamera angefasst.

    ``normale(i)`` liefert die Normalen zu Indizes in der urspruenglichen
    Reihenfolge; ``zurueck`` bringt ein Ergebnis in diese Reihenfolge.
    """

    def __init__(self, punkte, normale, device):
        P = torch.as_tensor(np.asarray(punkte, np.float32), device=device)
        self.ordnung = torch.argsort(P[:, 0])
        self.P = P[self.ordnung].contiguous()
        del P
        self.xs = self.P[:, 0].contiguous()
        self.normale = normale

    def __len__(self):
        return len(self.P)

    def bereich(self, C, reich: float) -> tuple:
        if reich <= 0:
            return 0, len(self.P)
        grenzen = torch.stack([C[0] - reich, C[0] + reich]).contiguous()
        lo, hi = torch.searchsorted(self.xs, grenzen).tolist()
        return lo, hi

    def zurueck(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        out[self.ordnung] = x
        return out


def _projektion(P, R, t, K, W, H):
    pc = P @ R.T + t
    z = pc[:, 2]
    vorn = z > 0.05
    zs = torch.where(vorn, z, torch.ones_like(z))
    u = K[0, 0] * pc[:, 0] / zs + K[0, 2]
    v = K[1, 1] * pc[:, 1] / zs + K[1, 2]
    ok = vorn & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    rad2 = (pc[:, 0] * pc[:, 0] + pc[:, 1] * pc[:, 1]) / (zs * zs)
    return z, u, v, ok, rad2


def _zelle(u, v, w: int, h: int, skala: float):
    iu = (u * skala).long().clamp(0, w - 1)
    iv = (v * skala).long().clamp(0, h - 1)
    return iv * w + iu


def gesehen(ziel: Wolke, tiefe_von: Wolke, vm, K, W: int, H: int, gueltig, cfg,
            tiefe_render=None, block: int = 4_000_000, reich: float | None = None):
    """Punkte von ``ziel``, die die Ansicht brauchbar zeigt — wie
    ``core.sichtbar.colorize_sichtbar``.

    Tiefenkarte in ``tiefe_skala``-facher Aufloesung aus ``tiefe_von`` (die
    naechste Tiefe je Zelle); ist ``tiefe_render`` [H, W] da (inf, wo nichts
    deckt), gilt je Zelle das Kleinere von beidem — die Gaussians schliessen
    die Luecken, die verstreute Punkte aus der Naehe lassen. Sichtbar ist, was
    hoechstens ``toleranz_m`` plus ``toleranz_rel`` der Tiefe dahinter liegt,
    nicht streifender als ``min_cos`` und auf einem Pixel mit ``gueltig``.
    ``reich`` ersetzt ``cfg["reichweite"]`` fuer diese Ansicht.
    Liefert je Block ``(a, j, u, v, guete)``; ``a + j`` sind Positionen in
    ``ziel`` (sortiert), ``guete`` wie dort: cos, zum Bildrand etwas weniger.
    """
    R, t = vm[:3, :3], vm[:3, 3]
    C = -R.T @ t
    reich = float(cfg["reichweite"] if reich is None else reich)
    skala = float(cfg["tiefe_skala"])
    w, h = max(1, int(W * skala)), max(1, int(H * skala))
    karte = torch.full((w * h,), float("inf"), device=ziel.P.device)
    lo, hi = tiefe_von.bereich(C, reich)
    for a in range(lo, hi, block):
        Pb = tiefe_von.P[a:min(a + block, hi)]
        z, u, v, ok, _ = _projektion(Pb, R, t, K, W, H)
        if reich > 0:
            ok &= (Pb[:, 1] - C[1]).abs() < reich
        karte.scatter_reduce_(0, _zelle(u[ok], v[ok], w, h, skala), z[ok], reduce="amin")
    if tiefe_render is not None:
        k = max(1, int(round(1.0 / skala)))
        tr = -F.max_pool2d(-tiefe_render[None, None], k, k)[0, 0]
        tr = F.pad(tr, (0, max(0, w - tr.shape[1]), 0, max(0, h - tr.shape[0])),
                   value=float("inf"))[:h, :w]
        karte = torch.minimum(karte, tr.reshape(-1))
    r_max2 = (W * W + H * H) / (4.0 * float(K[0, 0]) ** 2)
    tol_m, tol_rel = float(cfg["toleranz_m"]), float(cfg["toleranz_rel"])
    lo, hi = ziel.bereich(C, reich)
    for a in range(lo, hi, block):
        Pb = ziel.P[a:min(a + block, hi)]
        z, u, v, ok, rad2 = _projektion(Pb, R, t, K, W, H)
        if reich > 0:
            ok &= (Pb[:, 1] - C[1]).abs() < reich
        ok &= z <= karte[_zelle(u, v, w, h, skala)] + tol_m + tol_rel * z
        j = torch.nonzero(ok).squeeze(1)
        if len(j) == 0:
            continue
        blick = F.normalize(C[None] - Pb[j], dim=-1)
        cos = (blick * ziel.normale(ziel.ordnung[a + j])).sum(1).abs()
        ui = u[j].long().clamp(0, W - 1)
        vi = v[j].long().clamp(0, H - 1)
        k = (cos >= float(cfg["min_cos"])) & gueltig[vi, ui]
        j, cos = j[k], cos[k]
        yield a, j, u[j], v[j], cos * (1.0 - 0.25 * rad2[j] / r_max2)


def _abtasten_bild(bild, u, v, W: int, H: int) -> torch.Tensor:
    """Bilinear an (u, v) mit Pixelmitten bei k + 0,5; bild [H, W, D] -> [n, D]."""
    gitter = torch.stack([u / W * 2 - 1, v / H * 2 - 1], -1)[None, None]
    s = F.grid_sample(bild.permute(2, 0, 1)[None].float(), gitter, mode="bilinear",
                      padding_mode="border", align_corners=False)
    return s[0, :, 0].T


def startfarbe(ds: dict, ansichten, device) -> np.ndarray:
    """Startwert je Anker: Mittel der Pixel, die ihn zeigen, nach Guete gewichtet.

    Tiefe aus den Kartenpunkten, nicht aus den Ankern — die liegen aus der
    Naehe zu weit auseinander, um etwas zu verdecken. Ungesehen: 0,5.
    Mit ``bel0`` werden die Pixel der Nicht-Bezugsansichten vorher durch die
    Umkehrung ihrer Startbelichtung geschickt, sonst mittelt die Startfarbe
    zwei Farbraeume.
    """
    cfg = ds["cfg"]
    anker = ds["anker"]
    normal = torch.from_numpy(np.asarray(anker["normal"], np.float32)).to(device)
    normal = F.normalize(normal, dim=-1)
    ziel = Wolke(anker["pos"], lambda i: normal[i], device)
    punkte = Wolke(ds["punkte"], None, device)
    M, D = len(ziel), ds["kanaele"]
    summe = torch.zeros(M, D, device=device)
    gewicht = torch.zeros(M, device=device)
    for v in ansichten:
        bild_v, maske_v = ds["daten"].hole(v)
        gt = als_float(bild_v, device)
        W, H = int(ds["breite"][v]), int(ds["hoehe"][v])
        vm, K = ds["viewmat"][v].to(device), ds["K"][v].to(device)
        umkehr = None
        if ds.get("bel0") is not None and not ds["bezug"][v]:
            Mv = np.eye(D, dtype=np.float32) + ds["bel0"][0][v]
            umkehr = (torch.from_numpy(np.linalg.inv(Mv).T.copy()).to(device),
                      torch.from_numpy(ds["bel0"][1][v]).to(device))
        for a, j, u, vv, guete in gesehen(ziel, punkte, vm, K, W, H,
                                          maske_v.to(device), cfg,
                                          reich=float(ds["reichweite"][v])):
            s = _abtasten_bild(gt, u, vv, W, H)
            if umkehr is not None:
                s = (s - umkehr[1]) @ umkehr[0]
            summe.index_add_(0, a + j, s * guete[:, None])
            gewicht.index_add_(0, a + j, guete)
    farbe = torch.full((M, D), 0.5, device=device)
    hat = gewicht > 0
    farbe[hat] = summe[hat] / gewicht[hat, None]
    if D == 3:
        farbe.clamp_(0.0, 1.0)
    melde("INFO", f"Startfarbe aus {len(ansichten)} Bildern für "
                  f"{float(hat.float().mean()) * 100:.1f} % der Anker")
    return ziel.zurueck(farbe).cpu().numpy()


def abtasten(ds: dict, g: "Splats", ansichten, viewmats, cpu: bool) -> tuple:
    """Wert je Kartenpunkt aus dem Rendering ohne Belichtung.

    Jede Ansicht in ``ansichten`` wird mit ``viewmats[v]`` gerendert; jeder
    Punkt nimmt den Wert der frontalsten, die ihn zeigt (s. :func:`gesehen`),
    und nur, wo das Rendering mindestens ``min_deckung`` deckt. Rueckgabe
    ``(wert float16 [N, D], guete float16 [N])``, Guete 0 = ungesehen.
    """
    cfg = ds["cfg"]
    device = g.pos.device
    index = torch.from_numpy(np.asarray(ds["anker"]["index"], np.int32)).to(device)
    wolke = Wolke(ds["punkte"], lambda i: g.normal[index[i].long()], device)
    N, D = len(wolke), g.kanaele
    wert = torch.zeros(N, D, dtype=torch.float16, device=device)
    beste = torch.zeros(N, device=device)
    with torch.no_grad():
        means, quats, scales, deck = g.means(), g.quats, g.scales(), g.deckkraft()
        for v in ansichten:
            vm, K = viewmats[v], ds["K"][v].to(device)
            W, H = int(ds["breite"][v]), int(ds["hoehe"][v])
            campos = -vm[:3, :3].T @ vm[:3, 3]
            farben = g.farben_fuer(means, campos, g.sh_grad)
            bild, alpha, tiefe = rendern(means, quats, scales, deck, farben, vm, K, W, H,
                                         cpu, mit_tiefe=True)
            deckt = alpha[..., 0] >= float(cfg["min_deckung"])
            tiefe = torch.where(deckt, tiefe[..., 0], torch.full_like(tiefe[..., 0],
                                                                       float("inf")))
            gueltig = ds["daten"].hole(v)[1].to(device) & deckt
            for a, j, u, vv, guete in gesehen(wolke, wolke, vm, K, W, H, gueltig, cfg,
                                              tiefe_render=tiefe,
                                              reich=float(ds["reichweite"][v])):
                k = guete > beste[a + j]
                if not bool(k.any()):
                    continue
                j, guete = j[k], guete[k]
                wert[a + j] = _abtasten_bild(bild, u[k], vv[k], W, H).to(torch.float16)
                beste[a + j] = guete
    return (wolke.zurueck(wert).cpu().numpy(),
            wolke.zurueck(beste).cpu().numpy().astype(np.float16))


# ------------------------------------------------------------- Training

def trainieren(ds: dict, aus: str, cpu: bool = False, log_every: int = 50,
               alle: bool = False) -> dict:
    """``alle``: die zurueckgehaltenen Bilder mittrainieren — der zweite Lauf
    nach der Gegenprobe braucht dafuer keinen neuen Datensatz."""
    cfg = ds["cfg"]
    device = torch.device("cpu" if cpu else "cuda")
    V = len(ds["daten"])
    holdout = np.zeros(V, bool) if alle else ds["holdout"]
    train = np.flatnonzero(~holdout)
    halt = np.flatnonzero(holdout)
    if len(train) == 0:
        raise RuntimeError("Keine Trainingsbilder im Datensatz.")
    D = ds["kanaele"]
    if not cpu:
        torch.cuda.reset_peak_memory_stats()
    anker = ds["anker"]
    if cfg["startfarbe"]:
        t0 = time.time()
        anker = dict(anker, farbe0=startfarbe(ds, train, device))
        melde("INFO", f"Startfarbe {time.time() - t0:.0f} s")
    g = Splats(anker, D, int(cfg["sh_grad"]), device)
    M = g.pos.shape[0]
    melde("INFO", f"{M} Gaussians auf den Ankern (Raster {g.voxel * 100:.1f} cm), "
                  f"{len(train)} Trainings- und {len(halt)} Prüfbilder, {D} Kanäle, "
                  f"{'CPU' if cpu else torch.cuda.get_device_name(0)}")
    melde("INFO", f"Bilder {ds['daten'].gesamt / 1e9:.2f} GB — "
                  + ("im Speicher" if ds["daten"].im_speicher else
                     "je Schritt von der Platte (zu groß für den Speicher)"))

    viewmat0 = ds["viewmat"].to(device)
    Ks = ds["K"].to(device)
    bezug = np.asarray(ds["bezug"], bool)
    hat_bezug = bool(bezug.any())
    if ds.get("bel0") is not None:
        bel0_D = torch.from_numpy(ds["bel0"][0]).to(device)
        bel0_e = torch.from_numpy(ds["bel0"][1]).to(device)
    else:
        bel0_D = torch.zeros(V, D, D, device=device)
        bel0_e = torch.zeros(V, D, device=device)
    # Mit Farbbezug: eine Matrix fuer alle Nicht-Bezugsansichten, je Ansicht nur
    # die Abweichung davon. 2373 Onboard-Ansichten in 1500 Onboard-Schritten
    # kommen je ein- bis zweimal dran — eine eigene Matrix lernt da nichts (am
    # is7-Flug blieb sie bei 0,997), die gemeinsame lernt bei jedem Schritt.
    gemeinsam = hat_bezug and bool(cfg["belichtung_gemeinsam"])
    frei_idx = torch.from_numpy(np.flatnonzero(~bezug)).to(device)
    if gemeinsam and len(frei_idx):
        gem_D = torch.nn.Parameter(bel0_D[frei_idx].mean(0).clone())
        gem_e = torch.nn.Parameter(bel0_e[frei_idx].mean(0).clone())
        bel0_D = bel0_D - gem_D.detach()
        bel0_e = bel0_e - gem_e.detach()
    else:
        gem_D = torch.nn.Parameter(torch.zeros(D, D, device=device))
        gem_e = torch.nn.Parameter(torch.zeros(D, device=device))
    bel_D = torch.nn.Parameter(bel0_D.clone())
    bel_e = torch.nn.Parameter(bel0_e.clone())
    if hat_bezug:
        melde("INFO", f"Farbbezug: {int(bezug.sum())} Ansichten "
                      f"({ds['herkunft'][int(np.flatnonzero(bezug)[0])]}) unbelichtet, "
                      f"{int((~bezug).sum())} bekommen ihre Matrix"
                      + (" ab der Startabbildung" if ds.get("bel0") is not None else ""))
    pose_w = torch.nn.Parameter(torch.zeros(V, 3, device=device))
    pose_t = torch.nn.Parameter(torch.zeros(V, 3, device=device))

    farb_params = [g.sh0, g.shN] if g.sh else [g.werte]
    gruppen = [
        {"params": farb_params[:1], "lr": cfg["lr_farbe"]},
        {"params": [g.deckkraft_roh], "lr": cfg["lr_deckkraft"]},
        {"params": [g.form, g.quats], "lr": cfg["lr_form"]},
        {"params": [g.versatz], "lr": cfg["lr_versatz"]},
    ]
    if g.sh and g.shN.shape[1]:
        gruppen.append({"params": [g.shN], "lr": cfg["lr_farbe"] / 20.0})
    opt = torch.optim.Adam(gruppen, eps=1e-15)
    opt_bel = torch.optim.Adam([bel_D, bel_e, gem_D, gem_e], lr=cfg["lr_belichtung"])
    opt_pose = torch.optim.Adam([{"params": [pose_w], "lr": cfg["lr_pose_dreh"]},
                                 {"params": [pose_t], "lr": cfg["lr_pose_weg"]}])
    fenster = _fenster(D, device)
    train_t = torch.from_numpy(train).to(device)
    eye = torch.eye(D, device=device)

    def ansicht(v: int, mit_pose: bool) -> torch.Tensor:
        vm = viewmat0[v]
        if not mit_pose:
            return vm
        T = torch.eye(4, device=device)
        T = torch.cat([torch.cat([exp_so3(pose_w[v:v + 1])[0], pose_t[v][:, None]], 1),
                       T[3:]], 0)
        return T @ vm

    def belichten(v: int, bild: torch.Tensor) -> torch.Tensor:
        if not cfg["belichtung"]:
            return bild
        if hat_bezug:
            if bezug[v]:
                return bild
            return bild @ (eye + gem_D + bel_D[v]).T + (gem_e + bel_e[v])
        Dm = bel_D[train_t].mean(0)
        em = bel_e[train_t].mean(0)
        Mv = eye + bel_D[v] - Dm
        return bild @ Mv.T + (bel_e[v] - em)

    n = int(cfg["schritte"])
    posen_ab = int(n * float(cfg["posen_ab"])) if cfg["posen"] else n + 1
    rng = np.random.default_rng(0)
    herkunft = np.asarray(ds["herkunft"])
    gruppen_v = [train[herkunft[train] == h] for h in dict.fromkeys(herkunft[train])]
    reihen = [rng.permutation(gv) for gv in gruppen_v]
    zaehler = [0] * len(gruppen_v)
    gewicht = [max(1, int(dict(cfg["ziehen"]).get(str(herkunft[gv[0]]), 1)))
               for gv in gruppen_v]
    zyklus = [k for k, gw in enumerate(gewicht) for _ in range(gw)]
    if len(gruppen_v) > 1:
        melde("INFO", "Abwechselnd gezogen aus " + ", ".join(
            f"{herkunft[gv[0]]} ({len(gv)}, Gewicht {gw})"
            for gv, gw in zip(gruppen_v, gewicht)))
    t0 = time.time()
    verlauf = []
    for schritt in range(n):
        k = zyklus[schritt % len(zyklus)]
        if zaehler[k] and zaehler[k] % len(gruppen_v[k]) == 0:
            reihen[k] = rng.permutation(gruppen_v[k])
        v = int(reihen[k][zaehler[k] % len(gruppen_v[k])])
        zaehler[k] += 1
        if schritt == int(0.7 * n):
            for grp in opt.param_groups:
                grp["lr"] *= 0.3
        bild_v, maske_v = ds["daten"].hole(v)
        gt = als_float(bild_v, device)
        m = maske_v.to(device)
        mit_pose = schritt >= posen_ab
        vm = ansicht(v, mit_pose)
        campos = -vm[:3, :3].T @ vm[:3, 3]
        means = g.means()
        grad = min(g.sh_grad, (4 * schritt) // max(n, 1))
        farben = g.farben_fuer(means, campos, grad)
        bild, alpha = rendern(means, g.quats, g.scales(), g.deckkraft(), farben,
                              vm, Ks[v], ds["breite"][v], ds["hoehe"][v], cpu)
        pred = belichten(v, bild)
        mf = m[..., None].float()
        anzahl = mf.sum().clamp_min(1.0)
        l1 = ((pred - gt).abs() * mf).sum() / (anzahl * D)
        verlust = l1
        if D == 3 and cfg["ssim"] > 0:
            s = ssim_karte(torch.where(m[..., None], pred, gt).clamp(0, 1), gt, fenster)
            verlust = (1 - cfg["ssim"]) * l1 + cfg["ssim"] * ((1 - s) * m).sum() / anzahl
        verlust = verlust + cfg["deckung"] * ((1 - alpha) * mf).sum() / anzahl
        if cfg["belichtung"]:
            verlust = verlust + 1e-3 * ((bel_D[v] - bel0_D[v]).pow(2).sum()
                                        + (bel_e[v] - bel0_e[v]).pow(2).sum())
        if mit_pose:
            verlust = verlust + 1e-2 * (pose_w[v].pow(2).sum() + pose_t[v].pow(2).sum())
        opt.zero_grad(set_to_none=True)
        opt_bel.zero_grad(set_to_none=True)
        opt_pose.zero_grad(set_to_none=True)
        verlust.backward()
        opt.step()
        if cfg["belichtung"]:
            opt_bel.step()
        if mit_pose:
            # Adam bewegt sonst auch die Posen der Bilder weiter, die gerade
            # nicht dran waren — mit ihrem alten Schwung. Nur diese eine Zeile
            # darf sich aendern.
            with torch.no_grad():
                alt_w, alt_t = pose_w.detach().clone(), pose_t.detach().clone()
            opt_pose.step()
            with torch.no_grad():
                andere = torch.ones(V, dtype=torch.bool, device=device)
                andere[v] = False
                pose_w[andere] = alt_w[andere]
                pose_t[andere] = alt_t[andere]
        if schritt % log_every == 0 or schritt == n - 1:
            mse = (((pred.detach().clamp(0, 1) - gt) ** 2) * mf).sum() / (anzahl * D)
            psnr = float(-10 * torch.log10(mse.clamp_min(1e-10)))
            verlauf.append((schritt, float(verlust.detach()), psnr))
            melde("STEP", schritt + 1, n, f"{float(verlust.detach()):.5f}", f"{psnr:.2f}")
    melde("INFO", f"Training {time.time() - t0:.0f} s")

    # ------------------------------------------------- Farbe je Punkt
    t0 = time.time()
    with torch.no_grad():
        means = g.means().detach()
        quats = g.quats.detach()
        scales = g.scales().detach()
        deck = g.deckkraft().detach()
    punkt_wert, punkt_guete = abtasten(ds, g, train, viewmat0, cpu)
    melde("INFO", f"Abgetastet in {time.time() - t0:.0f} s: "
                  f"{float((punkt_guete > 0).mean()) * 100:.1f} % der Kartenpunkte "
                  f"in mindestens einem Trainingsbild sichtbar")

    # ----------------------------------------------------- Pruefbilder
    bericht = {"schritte": n, "gaussians": int(M), "voxel": g.voxel, "kanaele": D,
               "verlauf": verlauf, "pruefung": []}
    os.makedirs(os.path.join(aus, "vergleich"), exist_ok=True)
    with torch.no_grad():
        for k, v in enumerate(halt):
            vm = viewmat0[v]
            campos = -vm[:3, :3].T @ vm[:3, 3]
            farben = g.farben_fuer(means, campos, g.sh_grad)
            bild, _ = rendern(means, quats, scales, deck, farben, vm, Ks[v],
                              ds["breite"][v], ds["hoehe"][v], cpu)
            bild_v, maske_v = ds["daten"].hole(v)
            gt = als_float(bild_v, device)
            m = maske_v.to(device)
            if int(m.sum()) < 10:
                continue
            roh = float((bild - gt).abs()[m].mean())
            angepasst = affin_anpassen(bild, gt, m)
            fit = float((angepasst - gt).abs()[m].mean())
            mse = float(((angepasst.clamp(0, 1) - gt) ** 2)[m].mean()) if D == 3 else \
                float(((angepasst - gt) ** 2)[m].mean())
            psnr = -10 * math.log10(max(mse, 1e-10))
            name = ds["namen"][v]
            eintrag = {"bild": name, "l1_roh": roh, "l1_angepasst": fit,
                       "psnr_angepasst": psnr, "pixel": int(m.sum()),
                       "herkunft": ds["herkunft"][v]}
            if hat_bezug and not bezug[v] and cfg["belichtung"]:
                # mit der gelernten gemeinsamen Matrix: sagt, ob die Farbe im
                # Bezug die fremde Kamera erklaert, ohne das Pruefbild zu kennen
                gb = bild @ (eye + gem_D).T + gem_e
                eintrag["l1_gemeinsam"] = float((gb - gt).abs()[m].mean())
            bericht["pruefung"].append(eintrag)
            melde("EVAL", os.path.basename(name), f"{roh:.5f}", f"{fit:.5f}", f"{psnr:.2f}",
                  int(m.sum()))
            if k < int(cfg["vergleichsbilder"]) and D == 3:
                _vergleichsbild(gt, angepasst, m, os.path.join(aus, "vergleich", f"{k:02d}.jpg"))

    # ----------------------------------------------------------- Ablage
    np.savez(os.path.join(aus, "ergebnis.npz"),
             punkt_wert=punkt_wert, punkt_guete=punkt_guete,
             deckkraft=deck.cpu().numpy().astype(np.float32),
             versatz_m=(0.5 * g.voxel * torch.tanh(g.versatz.detach())).cpu().numpy()
             .astype(np.float32),
             belichtung_D=bel_D.detach().cpu().numpy(), belichtung_e=bel_e.detach().cpu().numpy(),
             belichtung_gemein_D=gem_D.detach().cpu().numpy(),
             belichtung_gemein_e=gem_e.detach().cpu().numpy(),
             pose_dreh=pose_w.detach().cpu().numpy(), pose_weg=pose_t.detach().cpu().numpy())
    pw = pose_w.detach().cpu().numpy()[train]
    pt = pose_t.detach().cpu().numpy()[train]
    bericht["posen"] = {
        "dreh_grad_median": float(np.degrees(np.median(np.linalg.norm(pw, axis=1)))),
        "dreh_grad_max": float(np.degrees(np.linalg.norm(pw, axis=1).max())),
        "weg_m_median": float(np.median(np.linalg.norm(pt, axis=1))),
        "weg_m_max": float(np.linalg.norm(pt, axis=1).max()),
        "weg_m_mittel": [float(x) for x in pt.mean(0)],
    }
    bericht["punkte_gesehen"] = float((punkt_guete > 0).mean())
    if hat_bezug and cfg["belichtung"] and (~bezug[train]).any():
        frei = train[~bezug[train]]
        Mm = (np.eye(D) + gem_D.detach().cpu().numpy()
              + bel_D.detach().cpu().numpy()[frei]).mean(0)
        em = (gem_e.detach().cpu().numpy() + bel_e.detach().cpu().numpy()[frei]).mean(0)
        bericht["belichtung_frei"] = {"M": Mm.tolist(), "t": em.tolist()}
        melde("INFO", "Mittlere Farbmatrix der Nicht-Bezugsansichten: diag "
                      + " ".join(f"{x:.3f}" for x in np.diag(Mm))
                      + ", Versatz " + " ".join(f"{x:+.3f}" for x in em))
    if not cpu:
        bericht["gpu_speicher_gb"] = torch.cuda.max_memory_allocated() / 1e9
        melde("INFO", f"GPU-Speicher höchstens {bericht['gpu_speicher_gb']:.1f} GB")
    with open(os.path.join(aus, "bericht.json"), "w", encoding="utf-8") as fh:
        json.dump(bericht, fh, indent=2)
    return {"g": g, "punkt_wert": punkt_wert, "punkt_guete": punkt_guete,
            "bericht": bericht}


def _vergleichsbild(gt, pred, m, pfad):
    from PIL import Image  # noqa: PLC0415
    a = (gt.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    b = (pred.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    d = (np.abs(a.astype(np.int16) - b.astype(np.int16)) * 3).clip(0, 255).astype(np.uint8)
    d[~m.cpu().numpy()] = (40, 0, 40)
    bild = np.concatenate([a, b, d], 1)
    h, w = bild.shape[:2]
    if w > 2400:
        s = 2400 / w
        bild = np.asarray(Image.fromarray(bild).resize((2400, int(h * s))))
    Image.fromarray(bild).save(pfad, quality=90)


# ------------------------------------------------------------ Selbsttest

def selbsttest() -> None:
    """Synthetische Szene: Boden mit Muster, ein Dach darueber, acht Kameras.

    Die Bilder werden mit demselben Renderer aus bekannten Farben erzeugt und
    je Bild anders belichtet (Verstaerkung 0,7 bis 1,3). Geprueft wird: die
    Farben der Kartenpunkte kommen zurueck, obwohl kein einziges Bild richtig
    belichtet ist; der Boden unter dem Dach bleibt ungefaerbt, der freie
    nicht; die Ablage ist lesbar.
    """
    torch.manual_seed(0)
    from PIL import Image  # noqa: PLC0415
    rng = np.random.default_rng(1)
    vox = 0.1
    g = np.mgrid[-1.5:1.5:vox, -1.5:1.5:vox].reshape(2, -1).T + vox / 2
    boden = np.c_[g, np.zeros(len(g))]
    farbe_boden = np.stack([(g[:, 0] > 0) * 0.7 + 0.15, (g[:, 1] > 0) * 0.7 + 0.15,
                            0.5 + 0.3 * np.sin(3 * g[:, 0])], 1)
    d = np.mgrid[-0.55:0.55:vox, -0.55:0.55:vox].reshape(2, -1).T + vox / 2
    dach = np.c_[d, np.full(len(d), 1.0)]
    farbe_dach = np.tile([0.9, 0.2, 0.2], (len(d), 1))
    pos = np.vstack([boden, dach]).astype(np.float32)
    wahr = np.vstack([farbe_boden, farbe_dach]).astype(np.float32)
    normal = np.tile([0.0, 0.0, 1.0], (len(pos), 1)).astype(np.float32)
    # Kartenpunkte: je Anker vier, bis 3 cm neben seiner Mitte
    index = np.repeat(np.arange(len(pos)), 4).astype(np.int32)
    punkte = pos[index] + np.c_[rng.uniform(-0.03, 0.03, (len(index), 2)),
                                np.zeros(len(index))].astype(np.float32)
    anker = {"pos": pos, "normal": normal, "voxel": np.float32(vox), "index": index}

    W, H, f = 64, 48, 40.0
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float32)
    views, halt = [], []
    for i in range(10):
        C = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), 4.0])
        R = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])      # blickt nach unten
        vm = np.eye(4, dtype=np.float32)
        vm[:3, :3] = R
        vm[:3, 3] = -R @ C
        views.append(vm)
        halt.append(i >= 8)
    ordner = tempfile.mkdtemp(prefix="splat_selbsttest_")
    os.makedirs(os.path.join(ordner, "bilder"))
    gew = torch.from_numpy(wahr)
    wahr_g = Splats(anker, 3, 0, "cpu")
    with torch.no_grad():
        wahr_g.sh0.copy_(((gew - 0.5) / SH_C0)[:, None, :])
        wahr_g.deckkraft_roh.fill_(logit(0.95))
    gains = np.linspace(0.7, 1.3, 10)
    rng.shuffle(gains)
    namen, masken = [], []
    for i, vm in enumerate(views):
        with torch.no_grad():
            b, a, _ = rendere_dicht(wahr_g.means(), wahr_g.quats, wahr_g.scales(),
                                    wahr_g.deckkraft(), gew, torch.from_numpy(vm),
                                    torch.from_numpy(K), W, H)
        bild = (b.numpy() * gains[i]).clip(0, 1)
        Image.fromarray((bild * 255).round().astype(np.uint8)).save(
            os.path.join(ordner, "bilder", f"{i:02d}.png"))
        Image.fromarray(((a[..., 0].numpy() > 0.5) * 255).astype(np.uint8)).save(
            os.path.join(ordner, "bilder", f"{i:02d}_m.png"))
        namen.append(f"bilder/{i:02d}.png")
        masken.append(f"bilder/{i:02d}_m.png")
    np.savez(os.path.join(ordner, "anker.npz"), **anker)
    np.save(os.path.join(ordner, "punkte.npy"), punkte)
    np.savez(os.path.join(ordner, "ansichten.npz"), viewmat=np.stack(views),
             K=np.stack([K] * 10), breite=np.full(10, W), hoehe=np.full(10, H),
             bild=np.array(namen), maske=np.array(masken), holdout=np.array(halt))
    with open(os.path.join(ordner, "param.json"), "w") as fh:
        json.dump({"schritte": 240, "sh_grad": 0, "posen": True, "ssim": 0.0,
                   "lr_farbe": 3e-2, "vergleichsbilder": 1}, fh)

    ds = lade_datensatz(ordner)
    erg = trainieren(ds, ordner, cpu=True, log_every=100)
    guete, wert = erg["punkt_guete"], erg["punkt_wert"].astype(np.float32)
    nb = len(boden)
    ist_boden = index < nb
    gx, gy = pos[index, 0], pos[index, 1]
    # Kameras bis 0,85 m seitlich, 3 m ueber dem Dach: der Blick reicht
    # hoechstens (0,85 − 0,55) / 3 = 0,1 m unter die Dachkante — innerhalb
    # ±0,15 m ist der Boden fuer alle verdeckt.
    unter = ist_boden & (np.abs(gx) < 0.15) & (np.abs(gy) < 0.15)
    frei = ist_boden & ((np.abs(gx) > 0.9) | (np.abs(gy) > 0.9)) \
        & (np.abs(gx) < 1.1) & (np.abs(gy) < 1.1)
    # Gerendert wird mit Scheiben von einer halben Zelle: an den harten
    # Farbspruengen des Musters (x = 0, y = 0) mischt jedes Rendering die
    # Nachbarzelle hinein. Gemessen wird daneben.
    glatt = (np.abs(gx) > 0.25) & (np.abs(gy) > 0.25)
    mess = (frei | ~ist_boden) & glatt & (guete > 0)
    # Die Belichtung ist nur bis auf den gemeinsamen Mittelwert bestimmt: der
    # liegt bei den Gains ueber die Trainingsbilder nicht exakt bei 1.
    skala = gains[:8].mean()
    fehler = np.abs(wert[mess] / skala - wahr[index][mess]).mean()
    print(f"  mittlerer Farbfehler sichtbar (auf mittlere Belichtung {skala:.2f} "
          f"bezogen): {fehler:.3f}, {mess.sum()} Punkte")
    print(f"  gefärbt: unter dem Dach {(guete[unter] > 0).mean() * 100:.0f} %, "
          f"frei {(guete[frei] > 0).mean() * 100:.0f} %")
    assert fehler < 0.06, "Farben nicht zurueckgewonnen"
    assert (guete[unter] > 0).mean() < 0.02, "verdeckter Boden wurde gefaerbt"
    assert (guete[frei] > 0).mean() > 0.95, "freier Boden blieb ungefaerbt"
    z = np.load(os.path.join(ordner, "ergebnis.npz"))
    assert z["punkt_wert"].shape == (len(punkte), 3) and z["punkt_guete"].shape == (len(punkte),)
    with open(os.path.join(ordner, "bericht.json")) as fh:
        b = json.load(fh)
    assert len(b["pruefung"]) == 2 and b["pruefung"][0]["psnr_angepasst"] > 20, b["pruefung"]
    # ohne Belichtungsausgleich muss der Fehler groesser sein — sonst prueft der
    # Test nichts
    ds["cfg"].update({"belichtung": False, "posen": False})
    ohne = trainieren(ds, ordner, cpu=True, log_every=1000)
    w_ohne = ohne["punkt_wert"].astype(np.float32)
    f_ohne = np.abs(w_ohne[mess] / skala - wahr[index][mess]).mean()
    print(f"  ohne Belichtungsausgleich: {f_ohne:.3f}")
    assert f_ohne > fehler, "Belichtungsausgleich bringt nichts"

    # Gemeinsamer Datensatz: jede zweite Ansicht ist Bezug (richtig belichtet),
    # die anderen zeigt eine fremde Kamera mit eigener Farbmatrix. Die Farbe
    # muss dann im Bezug stehen — ohne Umrechnen auf eine mittlere Belichtung.
    print("  Bezug: halbe Ansichten mit fremder Farbmatrix")
    M_kam = np.array([[0.80, 0.10, 0.0], [0.05, 1.05, 0.0], [0.0, 0.10, 0.75]], np.float32)
    e_kam = np.array([0.05, -0.03, 0.04], np.float32)
    bezug = np.arange(10) % 2 == 0
    for i, vm in enumerate(views):
        with torch.no_grad():
            b, _, _ = rendere_dicht(wahr_g.means(), wahr_g.quats, wahr_g.scales(),
                                    wahr_g.deckkraft(), gew, torch.from_numpy(vm),
                                    torch.from_numpy(K), W, H)
        bild = b.numpy() if bezug[i] else b.numpy() @ M_kam.T + e_kam
        Image.fromarray((bild.clip(0, 1) * 255).round().astype(np.uint8)).save(
            os.path.join(ordner, "bilder", f"{i:02d}.png"))
    np.savez(os.path.join(ordner, "ansichten.npz"), viewmat=np.stack(views),
             K=np.stack([K] * 10), breite=np.full(10, W), hoehe=np.full(10, H),
             bild=np.array(namen), maske=np.array(masken), holdout=np.array(halt),
             bezug=bezug, reichweite=np.zeros(10, np.float32),
             herkunft=np.where(bezug, "maeander", "onboard"))
    ds = lade_datensatz(ordner)
    ds["cfg"].update({"posen": False})
    mit = trainieren(ds, ordner, cpu=True, log_every=1000)
    f_bezug = np.abs(mit["punkt_wert"].astype(np.float32)[mess] - wahr[index][mess]).mean()
    # Matrix und Versatz gleichen sich auf dem schmalen Farbbereich der Szene
    # teilweise aus; gemessen wird darum die Abbildung auf ihren Farben.
    bf = mit["bericht"]["belichtung_frei"]
    Mm, em = np.asarray(bf["M"]), np.asarray(bf["t"])
    f_abb = np.abs((wahr @ Mm.T + em) - (wahr @ M_kam.T + e_kam)).mean()
    print(f"  mit Bezug: Farbfehler {f_bezug:.3f}, Fehler der gelernten "
          f"Kameraabbildung {f_abb:.3f}")
    assert f_bezug < 0.06, "Farbe steht nicht im Bezug"
    assert f_abb < 0.05, "Kameraabbildung nicht gefunden"
    assert {e["herkunft"] for e in mit["bericht"]["pruefung"]} == {"maeander", "onboard"}
    melde("DONE", ordner)
    print("splat_train SELFTEST OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("datensatz", nargs="?")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--alle", action="store_true",
                    help="auch die zurückgehaltenen Prüfbilder trainieren")
    ap.add_argument("--selbsttest", action="store_true")
    a = ap.parse_args()
    if a.selbsttest:
        selbsttest()
        return
    if not a.datensatz:
        ap.error("Datensatz-Ordner fehlt")
    if not a.cpu and not torch.cuda.is_available():
        melde("INFO", "FEHLER PyTorch sieht keine CUDA-GPU.")
        sys.exit(2)
    ds = lade_datensatz(a.datensatz)
    trainieren(ds, a.datensatz, cpu=a.cpu, alle=a.alle)
    melde("DONE", a.datensatz)


if __name__ == "__main__":
    main()

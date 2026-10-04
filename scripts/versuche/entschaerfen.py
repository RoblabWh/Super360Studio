#!/usr/bin/env python3
"""Verwischte Nachtbilder des Meanders mit klassischer Bildverarbeitung schaerfen.

Die Weitwinkelbilder des Nachtflugs sind mit 1/8 s belichtet, waehrend die
Drohne mit rund 4 m/s fliegt: jedes Bild ist mit einer Wischspur gefaltet, dem
Kern (PSF). Dieses Skript probiert der Reihe nach Verfahren aus, die den Kern
schaetzen und wieder herausrechnen, misst jedes und wendet das beste auf alle
Bilder an. Jeder Versuch ist eine *Iteration* mit eigenem Ordner; die Anzeige
``entschaerfen_gui.py`` liest ``stand.json`` und zeigt immer die neueste.

Aufruf:

    entschaerfen.py <bilderordner> <ausgabeordner>
    entschaerfen.py <bilderordner> <ausgabeordner> --alle   # nur noch das beste
                                   # Verfahren aus stand.json auf alle Bilder

Kernschaetzung
    * ``blind``  — freier Kern, grob nach fein geschaetzt mit L0-Gradientenprior
      (Xu 2013 / Pan 2014): abwechselnd ein kantenscharfes Zwischenbild und der
      Kern, der es auf das Foto abbildet.
    * ``linie``  — die gerade Wischspur (Laenge, Winkel) mit derselben Lage wie
      der blinde Kern. Eine Rastersuche ueber Laenge und Winkel nach den
      duennsten Kanten (l1/l2, Krishnan 2011) fand auf diesen Bildern den
      bekannten Kern nicht und ist deshalb nicht mehr dabei.
    * ``kacheln`` — je Bildkachel ein eigener Kern, weil Drehung der Kamera die
      Spur ueber das Bild hinweg aendert.

Entfaltung mit bekanntem Kern
    * ``wiener`` — ein Schritt im Frequenzraum, Gradienten gedaempft.
    * ``rl``     — Richardson-Lucy, passt zum Photonenrauschen, bleibt positiv,
      ueberstrahlte Lampen werden nicht bestraft.
    * ``tv``     — kleinste Quadrate mit Kantenprior (l1 der Gradienten).
    * ``usm``    — Unscharfmaske, der Vergleich ohne Kern.

Bewertung — zwei Masse, weil es kein scharfes Original gibt:
    * **PSNR-Gewinn (synthetisch):** die schaerfsten Bilder werden halbiert (das
      drueckt ihre Restunschaerfe unter ein Pixel), mit bekannten Kernen
      verwischt und verrauscht. Dort ist die Wahrheit bekannt.
    * **SIFT-Inlier (echt):** Treffer zwischen aufeinanderfolgenden Bildern nach
      RANSAC, relativ zum Original. Das ist, was Ausrichtung und Einfaerbung
      spaeter brauchen; erfundene Strukturen zaehlen dort nicht.
    Die Wertung ist ``PSNR-Gewinn + 10*log10(Inlier/Inlier_Original)``.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import scipy.fft as sf

GAMMA = 2.2
#: ab diesem Wert (0..1, wie im JPEG) gilt ein Pixel als ueberstrahlt
SATT = 0.96
#: laengste Wischspur in Pixeln des vollen Bildes, die gesucht wird
KERN_MAX = 61


# --------------------------------------------------------------------------
# Grundlagen: Lesen, Licht, Rand, Faltung
# --------------------------------------------------------------------------

def lies(pfad: str) -> np.ndarray:
    """Graubild als float32 in 0..1, so wie es im JPEG steht."""
    g = cv2.imread(pfad, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise OSError(f"nicht lesbar: {pfad}")
    return g.astype(np.float32) / 255.0


def schreibe(pfad: str, g: np.ndarray) -> None:
    cv2.imwrite(pfad, np.clip(g * 255.0 + 0.5, 0, 255).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 95])


def linear(g: np.ndarray) -> np.ndarray:
    """JPEG-Werte nach Licht. Nur dort ist Wischen eine Faltung."""
    return np.power(np.clip(g, 0, 1), GAMMA).astype(np.float32)


def gamma(x: np.ndarray) -> np.ndarray:
    return np.power(np.clip(x, 0, 1), 1.0 / GAMMA).astype(np.float32)


def otf(k: np.ndarray, form: tuple[int, int]) -> np.ndarray:
    """Kern nach Frequenzraum, Mitte des Kerns auf den Ursprung gelegt."""
    p = np.zeros(form, np.float32)
    kh, kw = k.shape
    p[:kh, :kw] = k
    p = np.roll(p, (-(kh // 2), -(kw // 2)), (0, 1))
    return sf.rfft2(p)


def _d2(form: tuple[int, int]) -> np.ndarray:
    """|Dx|^2 + |Dy|^2 der Vorwaertsdifferenzen."""
    dx = otf(np.array([[0, -1, 1]], np.float32), form)
    dy = otf(np.array([[0], [-1], [1]], np.float32), form)
    return (np.abs(dx) ** 2 + np.abs(dy) ** 2).astype(np.float32)


def _grad(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.roll(x, -1, 1) - x, np.roll(x, -1, 0) - x


def _div(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Transponierte von :func:`_grad` (negative Divergenz)."""
    return (np.roll(gx, 1, 1) - gx) + (np.roll(gy, 1, 0) - gy)


def polstern(y: np.ndarray, p: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Bild spiegelnd erweitern und den aeusseren Rand zyklisch schliessen.

    Die Entfaltung rechnet im Frequenzraum, also auf einem Torus. Ohne Polster
    schlaegt der Sprung zwischen linkem und rechtem Rand als Wellen ins Bild.
    Der Rand laeuft deshalb in eine zyklisch weichgezeichnete Fassung aus; das
    eigentliche Bild bleibt unberuehrt.
    """
    h, w = y.shape
    H = sf.next_fast_len(h + 2 * p, real=True)
    W = sf.next_fast_len(w + 2 * p, real=True)
    o, l = (H - h) // 2, (W - w) // 2
    yp = cv2.copyMakeBorder(y, o, H - h - o, l, W - w - l, cv2.BORDER_REFLECT_101)
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.rfftfreq(W)[None, :]
    weich = np.exp(-2 * (np.pi * p / 2) ** 2 * (fy ** 2 + fx ** 2)).astype(np.float32)
    yb = sf.irfft2(sf.rfft2(yp) * weich, s=yp.shape)

    def rampe(n: int, a: int, b: int) -> np.ndarray:
        i = np.arange(n, dtype=np.float32)
        r = np.minimum(i / max(a, 1), (n - 1 - i) / max(b, 1))
        r = np.clip(r, 0, 1)
        return 0.5 - 0.5 * np.cos(np.pi * r)

    wy = rampe(H, o, H - h - o)[:, None]
    wx = rampe(W, l, W - w - l)[None, :]
    g = wy * wx
    return (g * yp + (1 - g) * yb).astype(np.float32), (o, l)


def linienkern(laenge: float, winkel_grad: float) -> np.ndarray:
    """Gerade Wischspur, geglaettet gezeichnet, Summe 1."""
    laenge = max(float(laenge), 1.0)
    n = int(math.ceil(laenge)) + 3
    n += 1 - n % 2
    u = 8
    bild = np.zeros((n * u, n * u), np.uint8)
    c = (n * u - 1) / 2.0
    dx = math.cos(math.radians(winkel_grad)) * (laenge - 1) / 2 * u
    dy = -math.sin(math.radians(winkel_grad)) * (laenge - 1) / 2 * u
    cv2.line(bild, (int(round(c - dx)), int(round(c - dy))),
             (int(round(c + dx)), int(round(c + dy))), 255, u, cv2.LINE_AA)
    k = cv2.resize(bild.astype(np.float32), (n, n), interpolation=cv2.INTER_AREA)
    return k / k.sum()


def zentriere(k: np.ndarray) -> np.ndarray:
    """Schwerpunkt des Kerns in die Mitte schieben (ganze Pixel)."""
    h, w = k.shape
    s = k.sum()
    if s <= 0:
        return k
    cy = (k.sum(1) * np.arange(h)).sum() / s
    cx = (k.sum(0) * np.arange(w)).sum() / s
    return np.roll(k, (int(round(h // 2 - cy)), int(round(w // 2 - cx))), (0, 1))


# --------------------------------------------------------------------------
# Entfaltung mit bekanntem Kern (alles in linearem Licht)
# --------------------------------------------------------------------------

def wiener(y: np.ndarray, k: np.ndarray, nsr: float = 0.01) -> np.ndarray:
    yp, (o, l) = polstern(y, max(k.shape))
    K = otf(k, yp.shape)
    X = np.conj(K) * sf.rfft2(yp) / (np.abs(K) ** 2 + nsr * _d2(yp.shape))
    x = sf.irfft2(X, s=yp.shape)
    return x[o:o + y.shape[0], l:l + y.shape[1]]


def richardson_lucy(y: np.ndarray, k: np.ndarray, n: int = 30, tv: float = 0.0,
                    satt: np.ndarray | None = None) -> np.ndarray:
    """Richardson-Lucy. ``tv`` daempft Rauschen (Dey 2006), ``satt`` markiert
    ueberstrahlte Pixel: dort darf die Schaetzung beliebig heller sein."""
    yp, (o, l) = polstern(y, max(k.shape))
    grund = 1e-4
    yp = np.maximum(yp, 0) + grund
    K = otf(k, yp.shape)
    Kc = np.conj(K)
    sp = None
    if satt is not None:
        sp = cv2.copyMakeBorder(satt.astype(np.uint8), o, yp.shape[0] - y.shape[0] - o,
                                l, yp.shape[1] - y.shape[1] - l,
                                cv2.BORDER_REFLECT_101).astype(bool)
    x = yp.copy()
    for _ in range(n):
        est = sf.irfft2(sf.rfft2(x) * K, s=yp.shape)
        est = np.maximum(est, grund)
        r = yp / est
        if sp is not None:
            r[sp & (est > yp)] = 1.0
        kor = sf.irfft2(sf.rfft2(r) * Kc, s=yp.shape)
        if tv > 0:
            gx, gy = _grad(x)
            norm = np.sqrt(gx * gx + gy * gy) + 1e-4
            kor = kor / np.maximum(1.0 + tv * _div(gx / norm, gy / norm), 0.5)
        x = np.maximum(x * kor, 0)
    return x[o:o + y.shape[0], l:l + y.shape[1]] - grund


def _hqs(yp: np.ndarray, K: np.ndarray, lam: float, hart: bool,
         d2: np.ndarray, stufen: int = 10) -> np.ndarray:
    """min 1/2|k*x-y|^2 + lam*|grad x| per Halbquadratik; ``hart`` = L0 statt l1."""
    Z = np.conj(K) * sf.rfft2(yp)
    KK = np.abs(K) ** 2
    x = yp
    beta = lam * 10.0
    for _ in range(stufen):
        gx, gy = _grad(x)
        t = lam / beta
        if hart:
            weg = gx * gx + gy * gy < t
            gx = np.where(weg, 0, gx)
            gy = np.where(weg, 0, gy)
        else:
            gx = np.sign(gx) * np.maximum(np.abs(gx) - t, 0)
            gy = np.sign(gy) * np.maximum(np.abs(gy) - t, 0)
        X = (Z + beta * sf.rfft2(_div(gx, gy))) / (KK + beta * d2)
        x = sf.irfft2(X, s=yp.shape)
        beta *= 2.0
    return x


def tv(y: np.ndarray, k: np.ndarray, lam: float = 2e-3) -> np.ndarray:
    yp, (o, l) = polstern(y, max(k.shape))
    x = _hqs(yp, otf(k, yp.shape), lam, False, _d2(yp.shape))
    return x[o:o + y.shape[0], l:l + y.shape[1]]


def unscharfmaske(g: np.ndarray, sigma: float = 3.0, staerke: float = 1.5) -> np.ndarray:
    return g + staerke * (g - cv2.GaussianBlur(g, (0, 0), sigma))


# --------------------------------------------------------------------------
# Kernschaetzung
# --------------------------------------------------------------------------

def l1l2(x: np.ndarray, gewicht: np.ndarray | None = None) -> float:
    """Duennheit der Kanten: l1/l2 der Gradienten. Kleiner = schaerfer."""
    gx, gy = _grad(x)
    if gewicht is not None:
        gx, gy = gx * gewicht, gy * gewicht
    l1 = np.abs(gx).sum() + np.abs(gy).sum()
    l2 = math.sqrt(float((gx * gx).sum() + (gy * gy).sum())) + 1e-12
    return float(l1 / l2)


def linie_aus_kern(k: np.ndarray) -> tuple[np.ndarray, dict]:
    """Einen freien Kern durch die gerade Wischspur gleicher Lage ersetzen.

    Hauptachse und Streuung des Kerns geben Winkel und Laenge (eine Strecke der
    Laenge L streut mit L/sqrt(12)). Die Gerade hat zwei Parameter statt
    einiger hundert und kann deshalb kein Rauschen mitlernen.
    """
    h, w = k.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    s = float(k.sum())
    mx, my = (k * xx).sum() / s, (k * yy).sum() / s
    cxx = (k * (xx - mx) ** 2).sum() / s
    cyy = (k * (yy - my) ** 2).sum() / s
    cxy = (k * (xx - mx) * (yy - my)).sum() / s
    werte, vek = np.linalg.eigh(np.array([[cxx, cxy], [cxy, cyy]]))
    vx, vy = vek[:, 1]
    winkel = math.degrees(math.atan2(-vy, vx)) % 180
    laenge = math.sqrt(12 * max(werte[1], 0)) + 1
    return linienkern(laenge, winkel), {"laenge": laenge, "winkel": winkel}


def _kern_aus_gradienten(gx, gy, gyx, gyy, ks: int, reg: float) -> np.ndarray:
    Fx, Fy = sf.rfft2(gx), sf.rfft2(gy)
    nenner = np.abs(Fx) ** 2 + np.abs(Fy) ** 2
    A = np.conj(Fx) * sf.rfft2(gyx) + np.conj(Fy) * sf.rfft2(gyy)
    kv = sf.irfft2(A / (nenner + reg * nenner.mean()), s=gx.shape)
    k = np.roll(kv, (ks // 2, ks // 2), (0, 1))[:ks, :ks].copy()
    k[k < 0.05 * k.max()] = 0
    # lose Kruemel abseits der Spur sind Rauschen
    n, marken, stat, _ = cv2.connectedComponentsWithStats((k > 0).astype(np.uint8), connectivity=8)
    if n > 2:
        gewicht = np.array([k[marken == i].sum() for i in range(n)])
        gewicht[0] = 0
        for i in range(1, n):
            if gewicht[i] < 0.1 * gewicht.max():
                k[marken == i] = 0
    s = k.sum()
    if s <= 0:
        k = np.zeros((ks, ks), np.float32)
        k[ks // 2, ks // 2] = 1.0
        return k
    return zentriere((k / s).astype(np.float32))


def kern_blind(y: np.ndarray, kmax: int = KERN_MAX, skala: float = 0.5,
               lam: float = 4e-3, reg: float = 1.0, runden: int = 5) -> tuple[np.ndarray, dict]:
    """Freier Kern, grob nach fein, mit L0-Gradientenprior fuer das Zwischenbild.

    ``y`` in linearem Licht, volle Groesse. Geschaetzt wird auf ``skala``, der
    Kern am Ende auf die volle Aufloesung gestreckt.
    """
    ys = cv2.resize(y, None, fx=skala, fy=skala, interpolation=cv2.INTER_AREA)
    ys = ys / max(float(np.percentile(ys, 99.5)), 1e-6)
    kgr = int(kmax * skala) | 1
    stufen = []
    s = 1.0
    while True:
        ks = max(int(math.ceil(kgr * s)) | 1, 3)
        stufen.append((s, ks))
        if ks <= 5:
            break
        s *= 0.7071
    k = None
    for s, ks in reversed(stufen):
        yb = ys if s >= 1 else cv2.resize(ys, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if k is None:
            k = np.zeros((ks, ks), np.float32)
            k[ks // 2, ks // 2 - 1:ks // 2 + 1] = 0.5
        else:
            k = np.maximum(cv2.resize(k, (ks, ks), interpolation=cv2.INTER_LINEAR), 0)
            k = k / k.sum()
        yp, _ = polstern(yb, ks)
        d2 = _d2(yp.shape)
        gyx, gyy = _grad(yp)
        l = lam
        for _ in range(runden):
            x = _hqs(yp, otf(k, yp.shape), l, True, d2, stufen=12)
            gx, gy = _grad(x)
            k = _kern_aus_gradienten(gx, gy, gyx, gyy, ks, reg)
            l = max(l / 1.1, 1e-4)
    n = int(round(k.shape[0] / skala)) | 1
    kv = np.maximum(cv2.resize(k, (n, n), interpolation=cv2.INTER_CUBIC), 0)
    kv = (kv / kv.sum()).astype(np.float32)
    yy, xx = np.nonzero(kv > 0.1 * kv.max())
    return kv, {"ausdehnung": float(math.hypot(np.ptp(yy), np.ptp(xx)))}


# --------------------------------------------------------------------------
# Ein Bild nach einer Vorschrift verarbeiten
# --------------------------------------------------------------------------

def _nlm(g: np.ndarray, h: float) -> np.ndarray:
    u = np.clip(g * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.fastNlMeansDenoising(u, None, float(h), 7, 21).astype(np.float32) / 255.0


def _schaetze(y: np.ndarray, sp: dict) -> np.ndarray:
    k, _ = kern_blind(y, **sp.get("kern_par", {}))
    if sp["kern"] == "linie":
        k, _ = linie_aus_kern(k)
    return k


def _entfalte(y: np.ndarray, k: np.ndarray, satt: np.ndarray, sp: dict) -> np.ndarray:
    par = sp.get("par", {})
    if sp["entf"] == "wiener":
        return wiener(y, k, **par)
    if sp["entf"] == "rl":
        return richardson_lucy(y, k, satt=satt, **par)
    if sp["entf"] == "tv":
        return tv(y, k, **par)
    raise ValueError(sp["entf"])


def _kernbild(kerne: list[list[np.ndarray]]) -> np.ndarray:
    n = max(k.shape[0] for zeile in kerne for k in zeile)
    zeilen = []
    for zeile in kerne:
        teile = []
        for k in zeile:
            f = np.zeros((n, n), np.float32)
            a = (n - k.shape[0]) // 2
            f[a:a + k.shape[0], a:a + k.shape[1]] = k / max(float(k.max()), 1e-9)
            teile.append(cv2.copyMakeBorder(f, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0.35))
        zeilen.append(np.hstack(teile))
    b = np.vstack(zeilen)
    return cv2.resize((b * 255).astype(np.uint8), None, fx=2, fy=2,
                      interpolation=cv2.INTER_NEAREST)


def verarbeite(g: np.ndarray, sp: dict) -> tuple[np.ndarray, np.ndarray | None]:
    """``g`` wie im JPEG (0..1) -> (Ergebnis 0..1, Bild der Kerne oder None)."""
    if sp["entf"] == "usm":
        return np.clip(unscharfmaske(g, **sp.get("par", {})), 0, 1), None
    roh = linear(g)
    y = linear(_nlm(g, sp["vor_nlm"])) if sp.get("vor_nlm") else roh
    satt = g >= SATT
    if sp.get("kacheln"):
        ny, nx = sp["kacheln"]
        H, W = g.shape
        th, tw = int(2 * H / (ny + 1)), int(2 * W / (nx + 1))
        fy = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(th) + 0.5) / th)
        fx = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(tw) + 0.5) / tw)
        fenster = (np.outer(fy, fx) + 0.01).astype(np.float32)
        summe = np.zeros_like(g)
        gewicht = np.zeros_like(g)
        kglobal = None
        kerne = []
        for iy in range(ny):
            zeile = []
            for ix in range(nx):
                a = min(iy * th // 2, H - th)
                b = min(ix * tw // 2, W - tw)
                t = (slice(a, a + th), slice(b, b + tw))
                # ohne Licht gibt es nichts zu schaetzen: dort gilt der Kern des ganzen Bildes
                if float(g[t].mean()) < 0.08:
                    if kglobal is None:
                        kglobal = _schaetze(roh, sp)
                    k = kglobal
                else:
                    k = _schaetze(roh[t], sp)
                zeile.append(k)
                summe[t] += fenster * _entfalte(y[t], k, satt[t], sp)
                gewicht[t] += fenster
            kerne.append(zeile)
        x = summe / gewicht
    else:
        k = _schaetze(roh, sp)
        kerne = [[k]]
        x = _entfalte(y, k, satt, sp)
    r = gamma(x)
    if sp.get("nach_nlm"):
        r = _nlm(r, sp["nach_nlm"])
    if sp.get("dunkel"):
        # im Dunkeln steht nur Rauschen; dort bleibt das (geglaettete) Original
        d = float(sp["dunkel"])
        m = np.clip((cv2.GaussianBlur(g, (0, 0), 12) - 0.5 * d) / d, 0, 1)
        m = m * m * (3 - 2 * m)
        r = m * r + (1 - m) * _nlm(g, 6)
    return np.clip(r, 0, 1), _kernbild(kerne)


# --------------------------------------------------------------------------
# Bewertung
# --------------------------------------------------------------------------

def _acht(g: np.ndarray) -> np.ndarray:
    return np.clip(g * 255.0 + 0.5, 0, 255).astype(np.uint8)


def inlier(a: np.ndarray, b: np.ndarray) -> int:
    """SIFT-Treffer zwischen zwei Bildern, die eine Epipolargeometrie tragen."""
    hell = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.02)
    pa, da = sift.detectAndCompute(hell.apply(_acht(a)), None)
    pb, db = sift.detectAndCompute(hell.apply(_acht(b)), None)
    if da is None or db is None or len(pa) < 8 or len(pb) < 8:
        return 0
    paare = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    gut = [m for m, n in (p for p in paare if len(p) == 2) if m.distance < 0.8 * n.distance]
    if len(gut) < 8:
        return 0
    xa = np.float32([pa[m.queryIdx].pt for m in gut])
    xb = np.float32([pb[m.trainIdx].pt for m in gut])
    _, maske = cv2.findFundamentalMat(xa, xb, cv2.USAC_MAGSAC, 2.0, 0.999, 5000)
    return int(maske.sum()) if maske is not None else 0


def psnr(x: np.ndarray, wahr: np.ndarray, rand: int = 48) -> float:
    """PSNR gegen die Wahrheit. Ein blinder Kern legt die Lage des Bildes nicht
    fest, deshalb wird vorher um ganze Pixel ausgerichtet."""
    (dx, dy), _ = cv2.phaseCorrelate(wahr.astype(np.float32), x.astype(np.float32))
    x = np.roll(x, (-int(round(dy)), -int(round(dx))), (0, 1))
    a = x[rand:-rand, rand:-rand]
    b = wahr[rand:-rand, rand:-rand]
    return float(10 * math.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-12)))


def kurvenkern(keim: int, laenge: float) -> np.ndarray:
    """Gekruemmte Wischspur: Flug plus Schwenk mit wechselnder Geschwindigkeit."""
    rng = np.random.default_rng(keim)
    n = 400
    winkel = rng.uniform(0, 2 * np.pi) + np.cumsum(rng.normal(0, 0.035, n))
    tempo = np.clip(1 + np.cumsum(rng.normal(0, 0.04, n)), 0.3, 2.0)
    px = np.cumsum(np.cos(winkel) * tempo)
    py = np.cumsum(np.sin(winkel) * tempo)
    f = laenge / max(math.hypot(np.ptp(px), np.ptp(py)), 1e-6)
    px, py = (px - px.mean()) * f, (py - py.mean()) * f
    g = int(math.ceil(laenge)) + 5
    g += 1 - g % 2
    k = np.zeros((g, g), np.float32)
    for x, y in zip(px + g // 2, py + g // 2):
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        ax, ay = x - x0, y - y0
        if 0 <= x0 < g - 1 and 0 <= y0 < g - 1:
            k[y0, x0] += (1 - ax) * (1 - ay)
            k[y0, x0 + 1] += ax * (1 - ay)
            k[y0 + 1, x0] += (1 - ax) * ay
            k[y0 + 1, x0 + 1] += ax * ay
    return zentriere(k / k.sum())


def _rauschmass(g: np.ndarray) -> float:
    d = g - cv2.GaussianBlur(g, (0, 0), 1.5)
    return float(np.median(np.abs(d - np.median(d))) / 0.6745)


def baue_synthetik(pfade: list[str], schaerfe: np.ndarray, ordner: str) -> list[dict]:
    """Verwischte Bilder mit bekannter Wahrheit aus den schaerfsten Aufnahmen."""
    os.makedirs(ordner, exist_ok=True)
    wahl: list[int] = []
    for i in np.argsort(-schaerfe):
        if all(abs(int(i) - j) >= 6 for j in wahl):
            wahl.append(int(i))
        if len(wahl) == 3:
            break
    ziel = float(np.median([_rauschmass(lies(p)) for p in pfade[::6]]))
    kerne = [linienkern(14, 100), kurvenkern(1, 20), linienkern(27, 62),
             kurvenkern(2, 30), linienkern(40, 95), kurvenkern(3, 40)]
    rng = np.random.default_rng(7)
    aus = []
    for nr, k in enumerate(kerne):
        w = cv2.resize(linear(lies(pfade[wahl[nr % 3]])), None, fx=0.5, fy=0.5,
                       interpolation=cv2.INTER_AREA)
        yb = cv2.filter2D(w, -1, cv2.flip(k, -1), borderType=cv2.BORDER_REFLECT_101)
        # Photonen- plus Ausleserauschen, so stark, dass es dem der Fotos gleicht
        bestes = None
        for f in (0.5, 1, 2, 4, 8, 16):
            z = rng.normal(0, 1, yb.shape).astype(np.float32)
            v = gamma(np.clip(yb + f * np.sqrt(1e-4 * yb + 1e-6) * z, 0, 1))
            abw = abs(math.log(max(_rauschmass(v), 1e-6) / ziel))
            if bestes is None or abw < bestes[0]:
                bestes = (abw, v)
        ok, puffer = cv2.imencode(".jpg", _acht(bestes[1]), [cv2.IMWRITE_JPEG_QUALITY, 92])
        v = cv2.imdecode(puffer, cv2.IMREAD_GRAYSCALE)
        pv = os.path.join(ordner, f"syn_{nr}.png")
        pw = os.path.join(ordner, f"syn_{nr}_wahr.png")
        cv2.imwrite(pv, v)
        cv2.imwrite(pw, _acht(gamma(w)))
        cv2.imwrite(os.path.join(ordner, f"syn_{nr}_kern.png"), _kernbild([[k]]))
        aus.append({"bild": pv, "wahr": pw,
                    "psnr_vorher": psnr(v.astype(np.float32) / 255, lies(pw))})
    return aus


# --------------------------------------------------------------------------
# Iterationen
# --------------------------------------------------------------------------

def _arbeite(auftrag: dict) -> dict:
    cv2.setNumThreads(1)
    g = lies(auftrag["bild"])
    r, kb = verarbeite(g, auftrag["sp"])
    aus = {"bild": auftrag["bild"]}
    if auftrag.get("wahr"):
        aus["psnr"] = psnr(r, lies(auftrag["wahr"]))
    if auftrag.get("ziel"):
        schreibe(auftrag["ziel"], r)
        if kb is not None:
            cv2.imwrite(os.path.splitext(auftrag["ziel"])[0] + ".kern.png", kb)
    return aus


def _schreibe_stand(ordner: str, stand: dict) -> None:
    tmp = os.path.join(ordner, "stand.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(stand, f, indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(ordner, "stand.json"))


def beschreibe(sp: dict) -> str:
    if sp["entf"] == "usm":
        return "Unscharfmaske " + ", ".join(f"{k}={v}" for k, v in sp.get("par", {}).items())
    t = {"blind": "freier Kern", "linie": "gerade Spur"}[sp["kern"]]
    if sp.get("kern_par"):
        t += " (" + ", ".join(f"{k}={v}" for k, v in sp["kern_par"].items()) + ")"
    if sp.get("kacheln"):
        t += f", {sp['kacheln'][0]}x{sp['kacheln'][1]} Kacheln"
    t += " + " + {"wiener": "Wiener", "rl": "Richardson-Lucy", "tv": "TV"}[sp["entf"]]
    if sp.get("par"):
        t += " (" + ", ".join(f"{k}={v}" for k, v in sp["par"].items()) + ")"
    for k, n in (("vor_nlm", "vorher entrauscht"), ("nach_nlm", "nachher entrauscht"),
                 ("dunkel", "Dunkles bleibt Original")):
        if sp.get(k):
            t += f", {n} ({sp[k]})"
    return t


def main() -> int:
    nur_alle = sys.argv[3:] == ["--alle"]
    if len(sys.argv) != 3 and not nur_alle:
        print(__doc__)
        return 2
    quelle, ordner = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    os.makedirs(ordner, exist_ok=True)
    pfade = sorted(os.path.join(quelle, n) for n in os.listdir(quelle)
                   if n.lower().endswith((".jpg", ".jpeg", ".png")))
    n = len(pfade)
    # vier Bloecke aus je vier aufeinanderfolgenden Bildern: zwoelf Nachbarpaare
    test = [i for a in (int(n * q) for q in (0.08, 0.33, 0.58, 0.81)) for i in range(a, a + 4)]
    paare = [(a, a + 1) for a in range(len(test) - 1) if test[a + 1] == test[a] + 1]

    becken = ProcessPoolExecutor(max_workers=max(2, min(24, (os.cpu_count() or 4) - 4)))
    if nur_alle:
        with open(os.path.join(ordner, "stand.json"), encoding="utf-8") as f:
            stand = json.load(f)
        stand["iterationen"] = [i for i in stand["iterationen"] if "wertung" in i]
        return _abschluss(stand, ordner, pfade, becken)
    stand = {"quelle": quelle, "status": "Vorbereitung", "iterationen": [], "beste": None}
    _schreibe_stand(ordner, stand)

    print("INFO Schaerfe aller Bilder und synthetische Pruefbilder", flush=True)
    schaerfe = np.array(list(becken.map(_lapvar, pfade)))
    syn = baue_synthetik(pfade, schaerfe, os.path.join(ordner, "synthetisch"))
    orig = [lies(pfade[i]) for i in test]
    ref_inlier = [inlier(orig[a], orig[b]) for a, b in paare]
    ref_l1l2 = float(np.mean([l1l2(g) for g in orig]))
    stand["referenz"] = {"inlier": float(np.mean(ref_inlier)), "l1l2": ref_l1l2,
                         "psnr": float(np.mean([s["psnr_vorher"] for s in syn]))}
    print(f"INFO Original: {stand['referenz']}", flush=True)

    def laufe(sp: dict) -> dict:
        nr = len(stand["iterationen"]) + 1
        name = f"iter_{nr:02d}"
        ziel = os.path.join(ordner, name)
        os.makedirs(ziel, exist_ok=True)
        t0 = time.time()
        stand["status"] = f"Iteration {nr}: {beschreibe(sp)}"
        _schreibe_stand(ordner, stand)
        auftraege = [{"bild": pfade[i], "sp": sp, "ziel": _ziel(ziel, pfade[i])} for i in test]
        auftraege += [{"bild": s["bild"], "wahr": s["wahr"], "sp": sp, "ziel": _ziel(ziel, s["bild"])}
                      for s in syn]
        erg = list(becken.map(_arbeite, auftraege))
        it = {"nr": nr, "ordner": name, "beschreibung": beschreibe(sp), "sp": sp,
              "bilder": [os.path.basename(pfade[i]) for i in test]}
        neu = [lies(a["ziel"]) for a in auftraege[:len(test)]]
        inl = [inlier(neu[a], neu[b]) for a, b in paare]
        it["psnr_gewinn"] = float(np.mean([e["psnr"] - s["psnr_vorher"]
                                           for e, s in zip(erg[len(test):], syn)]))
        it["inlier"] = float(np.mean(inl))
        it["inlier_rel"] = float(np.sum(inl) / max(np.sum(ref_inlier), 1))
        it["l1l2"] = float(np.mean([l1l2(g) for g in neu]))
        it["wertung"] = it["psnr_gewinn"] + 10 * math.log10(max(it["inlier_rel"], 1e-3))
        if stand["beste"] is None or it["wertung"] > bestes()["wertung"]:
            stand["beste"] = nr
        print(f"ITER {nr} wertung={it['wertung']:+.2f} psnr={it['psnr_gewinn']:+.2f}dB "
              f"inlier={it['inlier']:.0f} ({it['inlier_rel']:.2f}x) l1l2={it['l1l2']:.0f} "
              f"{time.time() - t0:.0f}s  {it['beschreibung']}", flush=True)
        it["sekunden"] = time.time() - t0
        stand["iterationen"].append(it)
        _schreibe_stand(ordner, stand)
        return it

    def bestes() -> dict:
        return stand["iterationen"][stand["beste"] - 1]

    def probiere(aenderungen: list[dict]) -> None:
        """Jede Aenderung am bisher besten Verfahren einzeln pruefen."""
        basis = json.loads(json.dumps(bestes()["sp"]))
        for a in aenderungen:
            sp = json.loads(json.dumps(basis))
            for k, v in a.items():
                if isinstance(v, dict):
                    sp[k] = {**sp.get(k, {}), **v}
                else:
                    sp[k] = v
            if sp["entf"] != "usm" and not any(sp == i["sp"] for i in stand["iterationen"]):
                laufe(sp)

    # A: Verfahren der Entfaltung, alle mit dem freien Kern
    laufe({"entf": "usm", "par": {"sigma": 3.0, "staerke": 1.5}})
    laufe({"kern": "blind", "entf": "wiener", "par": {"nsr": 0.01}})
    laufe({"kern": "blind", "entf": "rl", "par": {"n": 30, "tv": 0.002}})
    laufe({"kern": "blind", "entf": "tv", "par": {"lam": 2e-3}})
    # B: Staerke des besten Verfahrens
    probiere({"wiener": [{"par": {"nsr": v}} for v in (0.003, 0.03, 0.1)],
              "rl": [{"par": {"n": 15}}, {"par": {"n": 60}}, {"par": {"tv": 0.0}},
                     {"par": {"tv": 0.01}}],
              "tv": [{"par": {"lam": v}} for v in (5e-4, 1e-3, 5e-3, 1e-2)]}[bestes()["sp"]["entf"]])
    # C: Kernschaetzung
    probiere([{"kern": "linie"},
              {"kern_par": {"reg": 0.3}}, {"kern_par": {"reg": 3.0}},
              {"kern_par": {"lam": 2e-3}}, {"kern_par": {"lam": 8e-3}},
              {"kern_par": {"skala": 0.33}}, {"kern_par": {"skala": 1.0}}])
    probiere([{"kacheln": [2, 3]}, {"kacheln": [3, 4]}])
    # D: Rauschen
    probiere([{"vor_nlm": 4}, {"nach_nlm": 5}, {"dunkel": 0.12}, {"dunkel": 0.2}])
    # E: Staerke noch einmal, jetzt mit dem endgueltigen Kern
    probiere({"wiener": [{"par": {"nsr": v}} for v in (0.005, 0.02, 0.05)],
              "rl": [{"par": {"n": 20}}, {"par": {"n": 45}}, {"par": {"tv": 0.005}}],
              "tv": [{"par": {"lam": v}} for v in (1.5e-3, 3e-3, 7e-3)]}[bestes()["sp"]["entf"]])

    return _abschluss(stand, ordner, pfade, becken)


def _ziel(ordner: str, bild: str) -> str:
    return os.path.join(ordner, os.path.splitext(os.path.basename(bild))[0] + ".jpg")


def _abschluss(stand: dict, ordner: str, pfade: list[str], becken: ProcessPoolExecutor) -> int:
    """Das beste Verfahren auf alle Bilder anwenden; das ist die letzte Iteration."""
    beste = stand["iterationen"][stand["beste"] - 1]
    print(f"INFO bestes Verfahren: Iteration {beste['nr']} — {beste['beschreibung']}", flush=True)
    nr = len(stand["iterationen"]) + 1
    name = f"iter_{nr:02d}"
    ziel = os.path.join(ordner, name)
    os.makedirs(ziel, exist_ok=True)
    it = {"nr": nr, "ordner": name, "sp": beste["sp"], "bilder": [],
          "beschreibung": "ALLE BILDER — " + beste["beschreibung"]}
    stand["iterationen"].append(it)
    auftraege = [{"bild": p, "sp": beste["sp"], "ziel": _ziel(ziel, p)} for p in pfade]
    for i, e in enumerate(becken.map(_arbeite, auftraege)):
        # in der Reihenfolge der Bilder fertig melden, damit die Anzeige mitwaechst
        it["bilder"].append(os.path.basename(e["bild"]))
        stand["status"] = f"bestes Verfahren auf alle Bilder: {i + 1}/{len(pfade)}"
        if i % 8 == 7 or i == len(pfade) - 1:
            _schreibe_stand(ordner, stand)
            print(f"INFO {stand['status']}", flush=True)
    stand["status"] = "fertig"
    _schreibe_stand(ordner, stand)
    print(f"DONE {ordner}", flush=True)
    return 0


def _lapvar(pfad: str) -> float:
    return float(cv2.Laplacian(lies(pfad), cv2.CV_32F).var())


if __name__ == "__main__":
    sys.exit(main())

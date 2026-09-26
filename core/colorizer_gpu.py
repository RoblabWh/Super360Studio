"""Einfaerbung auf der Grafikkarte (OpenCL ueber pyopencl).

Gleiches Verfahren wie die CPU-Schleife in :func:`core.colorizer.colorize`,
nur je Punkt ein OpenCL-Work-Item: Transformation in beide Kameras,
Double-Sphere-Projektion, Farbe holen, Helligkeits-, Himmelssaum-, Rand- und
Blautest, Rangfolge und Median ueber die Frames.

OpenCL statt CUDA, damit nichts gebaut werden muss: pyopencl kommt als fertiges
pip-Paket, den OpenCL-Treiber bringt der Grafiktreiber mit (NVIDIA, AMD, Intel).

Ziel ist **bitgleich** zur CPU. Dafuer bildet der Kernel die Rechenwege der
CPU-Fassung nach, nicht nur die Formeln:
  * ``cv2.remap`` INTER_LINEAR auf uint8 rechnet in Festkomma: Koordinate auf
    1/32 px gerundet, Gewichte als 15-Bit-Ganzzahlen (:func:`_remap_tab`).
  * Numpy-``matmul`` (n,3)@(3,3) in float32 rechnet ``fma(z, m2, fma(y, m1,
    x*m0))``, ``einsum`` und elementweise Ausdruecke ohne FMA — daher
    ``FP_CONTRACT OFF`` und die FMA ausdruecklich.
  * Die Projektion rechnet wie Numpy in double.
  * ``cv2.cvtColor`` BGR->HSV in float: Farbton ``fma(n, 60/(diff+eps), off)``.
  * float-Division und -Wurzel korrekt gerundet
    (``-cl-fp32-correctly-rounded-divide-sqrt``).
Nachgemessen gegen die CPU an rosbag_2026-09-19_02-52-40, s. Selbsttest.

Qt-frei, kein print (ausser Selbsttest).
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

try:  # Paket-Import (App) vs. Direktstart des Selbsttests
    from .stitcher import SPLIT_X
except ImportError:  # pragma: no cover - nur "python3 core/colorizer_gpu.py"
    from stitcher import SPLIT_X  # type: ignore

__all__ = ["available", "colorize_gpu"]

_IMH = 1520
_IMW = 2 * SPLIT_X
_SLOTS = 128               # Frames gleichzeitig auf der Karte (~2,4 GB)
_BATCH_POINTS = 4_000_000  # Punkte je Kernel-Aufruf
_DECODE_THREADS = min(12, os.cpu_count() or 4)
# Zahl der Zaehler je Punkt (uchar): Saum gesperrt, Himmel/Blau/Rand
# zurueckgestellt, nur blau, cam1-Proben
_NSTAT = 6

_KERNEL = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF

#define HALF %(HALF)d
#define IMW %(IMW)d
#define IMH %(IMH)d
#define KMAX %(KMAX)d

__kernel void blown(__global const uchar* img, __global uchar* mask,
                    const ulong img_base, const ulong mask_base, const int clip,
                    __global const int2* offs, const int noffs)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if (x >= IMW || y >= IMH) return;
    const int lo = (x / HALF) * HALF, hi = lo + HALF - 1;
    uchar m = 0;
    for (int k = 0; k < noffs; ++k) {
        const int nx = x + offs[k].x, ny = y + offs[k].y;
        if (nx < lo || nx > hi || ny < 0 || ny >= IMH) continue;
        __global const uchar* p = img + img_base + ((ulong)ny * IMW + nx) * 3;
        if (p[0] >= clip && p[1] >= clip && p[2] >= clip) { m = 1; break; }
    }
    mask[mask_base + (ulong)y * IMW + x] = m;
}

inline int ds_project(double x, double y, double z, __global const double* c,
                      float* u, float* v)
{
    /* c: fx fy cx cy xi alpha w2 — Reihenfolge wie DoubleSphereCamera.project */
    const double fx = c[0], fy = c[1], cx = c[2], cy = c[3];
    const double xi = c[4], a = c[5], w2 = c[6];
    const double r2 = x * x + y * y;
    const double d1 = sqrt(r2 + z * z);
    int valid = z > (-w2) * d1;
    const double k = xi * d1 + z;
    const double d2 = sqrt(r2 + k * k);
    double den = a * d2 + (1.0 - a) * k;
    valid = valid && (den > 1e-6);
    if (den == 0.0) den = 1e-9;
    *u = (float)(fx * x / den + cx);
    *v = (float)(fy * y / den + cy);
    return valid;
}

inline int in_circle(float u, float v, float rr)
{
    const float du = u - 760.0f, dv = v - 760.0f;
    return du * du + dv * dv <= rr;
}

inline void sample(__global const uchar* img, ulong base, int xoff, float u, float v,
                   __global const int* tab, float* fb, float* fg, float* fr)
{
    const int X = convert_int_rte(u * 32.0f), Y = convert_int_rte(v * 32.0f);
    const int sx = X >> 5, sy = Y >> 5;
    __global const int* w = tab + ((Y & 31) * 32 + (X & 31)) * 4;
    const int x0 = clamp(sx, 0, HALF - 1) + xoff, x1 = clamp(sx + 1, 0, HALF - 1) + xoff;
    const int y0 = clamp(sy, 0, IMH - 1), y1 = clamp(sy + 1, 0, IMH - 1);
    __global const uchar* p00 = img + base + ((ulong)y0 * IMW + x0) * 3;
    __global const uchar* p01 = img + base + ((ulong)y0 * IMW + x1) * 3;
    __global const uchar* p10 = img + base + ((ulong)y1 * IMW + x0) * 3;
    __global const uchar* p11 = img + base + ((ulong)y1 * IMW + x1) * 3;
    int c[3];
    for (int ch = 0; ch < 3; ++ch) {
        int s = p00[ch] * w[0] + p01[ch] * w[1] + p10[ch] * w[2] + p11[ch] * w[3];
        c[ch] = clamp((s + (1 << 14)) >> 15, 0, 255);
    }
    *fb = (float)c[0]; *fg = (float)c[1]; *fr = (float)c[2];
}

inline int is_blue(float fb, float fg, float fr, __global const float* bp)
{
    /* bp: hue_lo hue_hi sat_min val_min */
    const float b = fb / 255.0f, g = fg / 255.0f, r = fr / 255.0f;
    float v = r, vmin = r;
    if (v < g) v = g;
    if (v < b) v = b;
    if (vmin > g) vmin = g;
    if (vmin > b) vmin = b;
    const float diff = v - vmin;
    const float s = diff / (fabs(v) + FLT_EPSILON);
    const float d = 60.0f / (diff + FLT_EPSILON);
    float h;
    if (v == r) h = fma(g - b, d, 0.0f);
    else if (v == g) h = fma(b - r, d, 120.0f);
    else h = fma(r - g, d, 240.0f);
    if (h < 0.0f) h += 360.0f;
    int in_hue;
    if (bp[0] <= bp[1]) in_hue = (h >= bp[0]) && (h <= bp[1]);
    else in_hue = (h >= bp[0]) || (h <= bp[1]);
    return in_hue && (s >= bp[2]) && (v * 255.0f >= bp[3]);
}

__kernel void colorize(
    __global const float* pw,          /* (n,3) Weltpunkte */
    __global const int* scan_of,       /* (n,) Scan im Stapel */
    __global const int* scan_nc,       /* (nb,) Kandidaten je Scan */
    __global const int* scan_slot,     /* (nb,KMAX) */
    __global const float* scan_M,      /* (nb,KMAX,12) Welt -> cam0, 3x4 */
    __global const uchar* img, __global const uchar* mask,
    __global const int* tab,
    __global const double* cam,        /* 2 x 7 */
    __global const float* R01t01,      /* R01 (3x3 zeilenweise), t01 */
    __global const float* fp,          /* Parameter, s. _fparams */
    __global const float* bp,          /* Blaubereich */
    const int flags,                   /* 1 lens_best, 2 sky_prefer, 4 blue */
    const int n,
    __global uchar* out_rgb, __global uchar* out_valid, __global uchar* out_stat)
{
    const int i = get_global_id(0);
    if (i >= n) return;
    const float min_r2 = fp[0], bmin = fp[1], bmax = fp[2], rr = fp[3];
    const float edge_r = fp[4], sky_luma = fp[5], sky_sat = fp[6];
    const float px = pw[3 * i], py = pw[3 * i + 1], pz = pw[3 * i + 2];
    const int sl = scan_of[i];
    const int nc = scan_nc[sl];

    uchar cb[KMAX], cg[KMAX], cr[KMAX], bad[KMAX], ok[KMAX];
    int n_blocked = 0, n_cam1 = 0;

    for (int c = 0; c < nc; ++c) {
        ok[c] = 0;
        __global const float* M = scan_M + ((ulong)sl * KMAX + c) * 12;
        const float p0 = fma(pz, M[2], fma(py, M[1], px * M[0])) + M[3];
        const float p1 = fma(pz, M[6], fma(py, M[5], px * M[4])) + M[7];
        const float p2 = fma(pz, M[10], fma(py, M[9], px * M[8])) + M[11];
        const float r2 = p0 * p0 + p1 * p1 + p2 * p2;
        const int rng_ok = r2 >= min_r2;
        const int slot = scan_slot[sl * KMAX + c];
        const ulong ib = (ulong)slot * IMH * IMW * 3, mb = (ulong)slot * IMH * IMW;

        float u[2], v[2];
        int geo[2];
        geo[0] = rng_ok && ds_project(p0, p1, p2, cam, &u[0], &v[0]);
        geo[0] = geo[0] && in_circle(u[0], v[0], rr);
        const int need1 = (flags & 1) ? rng_ok : (!geo[0] && rng_ok);
        geo[1] = 0;
        if (need1) {
            const float q0 = p0 - R01t01[9], q1 = p1 - R01t01[10], q2 = p2 - R01t01[11];
            const float c0 = fma(q2, R01t01[6], fma(q1, R01t01[3], q0 * R01t01[0]));
            const float c1 = fma(q2, R01t01[7], fma(q1, R01t01[4], q0 * R01t01[1]));
            const float c2 = fma(q2, R01t01[8], fma(q1, R01t01[5], q0 * R01t01[2]));
            geo[1] = ds_project(c0, c1, c2, cam + 7, &u[1], &v[1]);
            geo[1] = geo[1] && in_circle(u[1], v[1], rr);
        }

        float key = INFINITY;
        for (int lens = 0; lens < 2; ++lens) {
            if (!geo[lens]) continue;
            const int xoff = lens * HALF;
            float fb, fg, fr;
            sample(img, ib, xoff, u[lens], v[lens], tab, &fb, &fg, &fr);
            const float g = 0.299f * fr + 0.587f * fg + 0.114f * fb;
            const int bright = (g >= bmin) && (g <= bmax);
            const int mu = clamp(convert_int(rint(u[lens])), 0, HALF - 1) + xoff;
            const int mv = clamp(convert_int(rint(v[lens])), 0, IMH - 1);
            const int freep = mask[mb + (ulong)mv * IMW + mu] == 0;
            if (bright && !freep) ++n_blocked;
            if (!(bright && freep)) continue;
            const float du = u[lens] - 760.0f, dv = v[lens] - 760.0f;
            const float rad = (float)sqrt((double)du * du + (double)dv * dv);
            uchar b = rad > edge_r ? 1 : 0;
            if (flags & 2) {
                const float mx = fmax(fmax(fb, fg), fr), mn = fmin(fmin(fb, fg), fr);
                const float sat = (mx - mn) / fmax(mx, 1.0f);
                if (g >= sky_luma && sat <= sky_sat) b |= 4;
            }
            if ((flags & 4) && is_blue(fb, fg, fr, bp)) b |= 2;
            const float k = (float)b * 1000.0f + rad;
            if (k < key) {
                key = k;
                cb[c] = (uchar)fb; cg[c] = (uchar)fg; cr[c] = (uchar)fr;
                bad[c] = b; ok[c] = 1;
                if (lens == 1) ++n_cam1;
            }
        }
    }

    uchar best = 255;
    for (int c = 0; c < nc; ++c)
        if (ok[c] && bad[c] < best) best = bad[c];
    int d_sky = 0, d_blue = 0, d_edge = 0;
    uchar vb[KMAX], vg[KMAX], vr[KMAX];
    int m = 0;
    for (int c = 0; c < nc; ++c) {
        if (!ok[c]) continue;
        if (bad[c] == best) {
            vb[m] = cb[c]; vg[m] = cg[c]; vr[m] = cr[c]; ++m;
        } else {
            if (bad[c] & 4) ++d_sky;
            if (bad[c] & 2) ++d_blue;
            if (bad[c] & 1) ++d_edge;
        }
    }
    __global uchar* st = out_stat + (ulong)i * 6;
    st[0] = (uchar)n_blocked; st[1] = (uchar)d_sky; st[2] = (uchar)d_blue;
    st[3] = (uchar)d_edge; st[4] = (uchar)(m > 0 && (best & 2)); st[5] = (uchar)n_cam1;
    if (m == 0) {
        out_valid[i] = 0;
        out_rgb[3 * i] = 0; out_rgb[3 * i + 1] = 0; out_rgb[3 * i + 2] = 0;
        return;
    }
    /* Median je Kanal wie np.nanmedian: gerade Anzahl -> Mittel der beiden
       mittleren, danach astype(uint8) schneidet ab */
    for (int a = 1; a < m; ++a) {
        uchar tb = vb[a], tg = vg[a], tr = vr[a];
        int j = a - 1;
        while (j >= 0 && vb[j] > tb) { vb[j + 1] = vb[j]; --j; }
        vb[j + 1] = tb;
        j = a - 1;
        while (j >= 0 && vg[j] > tg) { vg[j + 1] = vg[j]; --j; }
        vg[j + 1] = tg;
        j = a - 1;
        while (j >= 0 && vr[j] > tr) { vr[j + 1] = vr[j]; --j; }
        vr[j + 1] = tr;
    }
    const int h = m / 2;
    if (m & 1) {
        out_rgb[3 * i] = vr[h]; out_rgb[3 * i + 1] = vg[h]; out_rgb[3 * i + 2] = vb[h];
    } else {
        out_rgb[3 * i] = (uchar)((vr[h - 1] + vr[h]) / 2);
        out_rgb[3 * i + 1] = (uchar)((vg[h - 1] + vg[h]) / 2);
        out_rgb[3 * i + 2] = (uchar)((vb[h - 1] + vb[h]) / 2);
    }
    out_valid[i] = 1;
}
"""


def _remap_tab() -> np.ndarray:
    """Festkomma-Gewichte von cv2.remap INTER_LINEAR (32x32 Teilpixel, je 4).

    Wie OpenCVs initInterTab2D: Produkt der linearen Gewichte, auf 1/32768
    gerundet; weicht die Summe ab, gleicht das groesste (bzw. kleinste)
    Gewicht aus. Gegen cv2.remap geprueft (s. Selbsttest).
    """
    scale = 1 << 15
    tab = np.zeros((32, 32, 4), np.int32)
    for ay in range(32):
        for ax in range(32):
            fx, fy = ax / 32.0, ay / 32.0
            w = np.array([(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx])
            it = np.rint(w * scale).astype(np.int32)
            d = int(it.sum()) - scale
            if d < 0:
                it[np.argmax(it)] -= d
            elif d > 0:
                it[np.argmin(it)] -= d
            tab[ay, ax] = it
    return tab


_lock = threading.Lock()
_state: dict = {}


def _context():
    """(ctx, queue, device) der ersten OpenCL-GPU mit double; sonst None."""
    with _lock:
        if "ctx" in _state:
            return _state["ctx"]
        res = None
        if os.environ.get("SUPER360_GPU", "1") != "0":
            try:
                import pyopencl as cl
                for plat in cl.get_platforms():
                    for dev in plat.get_devices(device_type=cl.device_type.GPU):
                        if "cl_khr_fp64" not in dev.extensions:
                            continue
                        ctx = cl.Context([dev])
                        res = (ctx, cl.CommandQueue(ctx), dev)
                        break
                    if res:
                        break
            except Exception as exc:  # noqa: BLE001 - ohne GPU gilt die CPU
                _state["fehler"] = str(exc)
                res = None
        _state["ctx"] = res
        return res


def available() -> str | None:
    """Name der OpenCL-GPU, wenn die GPU-Einfaerbung laufen kann, sonst None."""
    c = _context()
    return None if c is None else c[2].name.strip()


def _kernels(kmax: int):
    """(blown, colorize) fuer diese Kandidatenzahl, einmal gebaut und gemerkt."""
    import warnings
    import pyopencl as cl
    ctx = _context()[0]
    key = ("prog", kmax)
    with _lock:
        if key not in _state:
            src = _KERNEL % {"HALF": SPLIT_X, "IMW": _IMW, "IMH": _IMH, "KMAX": kmax}
            with warnings.catch_warnings():
                # der NVIDIA-Compiler meldet auch bei Erfolg Text
                warnings.simplefilter("ignore", cl.CompilerWarning)
                prg = cl.Program(ctx, src).build(
                    options=["-cl-fp32-correctly-rounded-divide-sqrt"])
            _state[key] = (cl.Kernel(prg, "blown"), cl.Kernel(prg, "colorize"))
        return _state[key]


def _cam_params(cam) -> list[float]:
    """fx fy cx cy xi alpha w2 — w2 wie in DoubleSphereCamera.project."""
    a, xi = cam.alpha, cam.xi
    w1 = (1.0 - a) / a if a > 0.5 else a / (1.0 - a)
    w2 = (w1 + xi) / np.sqrt(2.0 * w1 * xi + xi * xi + 1.0)
    return [cam.fx, cam.fy, cam.cx, cam.cy, xi, a, float(w2)]


def colorize_gpu(rec, teile, teil_stamps, cam0, cam1, R01, t01, T_cam0_imu,
                 params, candidates, inv_rigid, rmax: float, progress_cb=None, cancel=None,
                 check_cancel=None) -> tuple[np.ndarray, np.ndarray, dict]:
    """GPU-Gegenstueck der Scan-Schleife in colorize().

    ``candidates(cam_stamps, t_scan)`` liefert die Frame-Indizes je Scan
    (dieselbe Funktion wie auf der CPU), ``rmax`` den Radius des nutzbaren
    Bildkreises in Pixeln (wie ``colorizer._in_circle``). Rueckgabe: (colors RGB uint8 (N,3),
    valid uint8 (N,), Zaehler wie in colorize()).
    """
    import pyopencl as cl
    ctx, queue, _dev = _context()
    blue_on = bool(params.blue_filter)
    kmax = int(params.k_frames) + 2 + (4 if blue_on else 0)
    k_blown, k_color = _kernels(kmax)
    mf = cl.mem_flags

    N, S = int(rec.n_points), int(rec.n_scans)
    colors = np.zeros((N, 3), np.uint8)
    valid = np.zeros(N, np.uint8)
    stats = np.zeros(_NSTAT, np.int64)

    frame_bytes = _IMH * _IMW * 3
    d_img = cl.Buffer(ctx, mf.READ_ONLY, _SLOTS * frame_bytes)
    d_mask = cl.Buffer(ctx, mf.READ_WRITE, _SLOTS * _IMH * _IMW)
    d_tab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=_remap_tab())
    d_cam = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array(
        _cam_params(cam0) + _cam_params(cam1), np.float64))
    d_r01 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.concatenate(
        [np.asarray(R01, np.float32).ravel(), np.asarray(t01, np.float32)]))
    fparams = np.array([float(params.min_range) ** 2, params.brightness_min,
                        params.brightness_max, rmax * rmax, params.edge_r,
                        params.sky_luma, params.sky_sat], np.float32)
    d_fp = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fparams)
    d_bp = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array(
        [params.blue_hue_lo, params.blue_hue_hi, params.blue_sat, params.blue_val],
        np.float32))
    flags = (1 if params.lens_best else 0) | (2 if params.sky_prefer else 0) \
        | (4 if blue_on else 0)

    clip, grow = int(params.sky_clip), int(params.sky_grow)
    if grow < 0 or clip > 255:
        offs = np.zeros((0, 2), np.int32)
    else:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        yy, xx = np.nonzero(ker)
        offs = np.stack([xx - grow, yy - grow], axis=1).astype(np.int32)
    d_offs = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                       hostbuf=offs if len(offs) else np.zeros((1, 2), np.int32))

    slots: OrderedDict = OrderedDict()   # (teil, frame) -> Slot, LRU
    free_slots = list(range(_SLOTS))
    pool = ThreadPoolExecutor(_DECODE_THREADS)

    def decode(data: bytes) -> np.ndarray:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("JPEG-Dekodierung fehlgeschlagen.")
        return img

    def load_frames(keys: list, keep: set) -> None:
        todo = [k for k in keys if k not in slots]
        for k in keys:
            if k in slots:
                slots.move_to_end(k)
        while len(free_slots) < len(todo):
            for old in slots:
                if old not in keep:
                    free_slots.append(slots.pop(old))
                    break
            else:
                raise RuntimeError("GPU-Einfärbung: zu viele Frames je Stapel.")
        futs = []
        for ti_, fidx in todo:
            b = teile[ti_][0]
            if hasattr(b, "read_camera_jpeg"):
                futs.append(pool.submit(decode, b.read_camera_jpeg(fidx)))
            else:
                futs.append(pool.submit(lambda im: im, b.read_camera(fidx)))
        for k, fut in zip(todo, futs):
            img = np.ascontiguousarray(fut.result())
            if img.shape != (_IMH, _IMW, 3):
                raise RuntimeError(f"Kamerabild {img.shape} statt ({_IMH}, {_IMW}, 3).")
            s = free_slots.pop()
            slots[k] = s
            cl.enqueue_copy(queue, d_img, img, dst_offset=s * frame_bytes)
            k_blown(queue, (_IMW, _IMH), None, d_img, d_mask,
                      np.uint64(s * frame_bytes), np.uint64(s * _IMH * _IMW),
                      np.int32(clip), d_offs, np.int32(len(offs)))

    R_all = Rotation.from_quat(rec.poses[:, 3:7]).as_matrix()
    ti = 0
    i = 0
    try:
        while i < S:
            if check_cancel is not None:
                check_cancel(cancel)
            # Stapel: Scans bis ~_BATCH_POINTS Punkte bzw. bis die Frames knapp werden
            b_scans, b_nc, b_slotkeys, b_M = [], [], [], []
            keys_needed: list = []
            keyset: set = set()
            n_pts = 0
            while i < S and n_pts < _BATCH_POINTS and len(keyset) < _SLOTS - 24:
                s0, e0 = int(rec.offsets[i]), int(rec.offsets[i + 1])
                while ti + 1 < len(teile) and i >= teile[ti][2]:
                    ti += 1
                cam_stamps = teil_stamps[ti]
                cands = []
                if e0 > s0 and len(cam_stamps):
                    for f in candidates(cam_stamps, float(rec.stamps[i])):
                        T_wi = rec.interpolate_pose(float(cam_stamps[f]))
                        if T_wi is not None:
                            cands.append((f, (T_cam0_imu @ inv_rigid(T_wi)).astype(np.float32)))
                cands = cands[:kmax]
                b_scans.append((i, s0, e0))
                b_nc.append(len(cands))
                b_slotkeys.append([(ti, f) for f, _ in cands])
                b_M.append([M[:3, :].ravel() for _, M in cands])
                for f, _ in cands:
                    if (ti, f) not in keyset:
                        keyset.add((ti, f))
                        keys_needed.append((ti, f))
                n_pts += e0 - s0
                i += 1
            load_frames(keys_needed, keyset)

            nb = len(b_scans)
            nc = np.array(b_nc, np.int32)
            slot_arr = np.zeros((nb, kmax), np.int32)
            M_arr = np.zeros((nb, kmax, 12), np.float32)
            for j in range(nb):
                for c, k in enumerate(b_slotkeys[j]):
                    slot_arr[j, c] = slots[k]
                    M_arr[j, c] = b_M[j][c]
            lo, hi = b_scans[0][1], b_scans[-1][2]
            n = hi - lo
            if n > 0:
                pw = np.empty((n, 3), np.float32)
                scan_of = np.empty(n, np.int32)
                for j, (si, s0, e0) in enumerate(b_scans):
                    if e0 <= s0:
                        continue
                    p_body = np.asarray(rec.points[s0:e0], dtype=np.float32)
                    pw[s0 - lo:e0 - lo] = p_body @ R_all[si].T.astype(np.float32) \
                        + rec.poses[si, 0:3].astype(np.float32)
                    scan_of[s0 - lo:e0 - lo] = j
                d_pw = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pw)
                d_sof = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=scan_of)
                d_nc = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nc)
                d_sl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=slot_arr)
                d_M = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=M_arr)
                out_rgb = np.empty((n, 3), np.uint8)
                out_val = np.empty(n, np.uint8)
                out_st = np.empty((n, _NSTAT), np.uint8)
                d_rgb = cl.Buffer(ctx, mf.WRITE_ONLY, out_rgb.nbytes)
                d_val = cl.Buffer(ctx, mf.WRITE_ONLY, out_val.nbytes)
                d_st = cl.Buffer(ctx, mf.WRITE_ONLY, out_st.nbytes)
                gsz = ((n + 255) // 256) * 256
                k_color(queue, (gsz,), (256,), d_pw, d_sof, d_nc, d_sl, d_M,
                             d_img, d_mask, d_tab, d_cam, d_r01, d_fp, d_bp,
                             np.int32(flags), np.int32(n), d_rgb, d_val, d_st)
                cl.enqueue_copy(queue, out_rgb, d_rgb)
                cl.enqueue_copy(queue, out_val, d_val)
                cl.enqueue_copy(queue, out_st, d_st)
                queue.finish()
                colors[lo:hi] = out_rgb
                valid[lo:hi] = out_val
                stats += out_st.sum(axis=0, dtype=np.int64)
                for d in (d_pw, d_sof, d_nc, d_sl, d_M, d_rgb, d_val, d_st):
                    d.release()
            if progress_cb is not None:
                progress_cb(i / max(S, 1), f"Faerbe Scan {i}/{S} (GPU)")
    finally:
        pool.shutdown(wait=True)
        for d in (d_img, d_mask, d_tab, d_cam, d_r01, d_fp, d_bp, d_offs):
            d.release()

    return colors, valid, {
        "n_sky_blocked": int(stats[0]), "n_sky_outvoted": int(stats[1]),
        "n_blue_outvoted": int(stats[2]), "n_edge_outvoted": int(stats[3]),
        "n_blue_kept": int(stats[4]), "n_cam1": int(stats[5]),
        "n_valid": int(valid.sum(dtype=np.int64))}


# ================================================================ Selbsttest

if __name__ == "__main__":
    import json
    import sys
    import time

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core import colorizer as C
    from core.bag_reader import BagReader
    from core.recording import Recording

    print(f"OpenCL-GPU: {available()}")
    assert available(), "keine OpenCL-GPU"

    # ---------- Test 1: Festkomma-Tabelle == cv2.remap ----------
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (64, 64, 3), np.uint8)
    u = rng.uniform(1, 62, 16384).astype(np.float32)
    v = rng.uniform(1, 62, 16384).astype(np.float32)
    ref = cv2.remap(img, u.reshape(1, -1), v.reshape(1, -1), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)[0]
    X, Y = np.rint(u * 32).astype(np.int64), np.rint(v * 32).astype(np.int64)
    sx, sy, w = X >> 5, Y >> 5, _remap_tab()[Y & 31, X & 31]
    p = img.astype(np.int64)
    val = (p[sy, sx] * w[:, 0, None] + p[sy, sx + 1] * w[:, 1, None]
           + p[sy + 1, sx] * w[:, 2, None] + p[sy + 1, sx + 1] * w[:, 3, None]
           + (1 << 14)) >> 15
    assert np.array_equal(np.clip(val, 0, 255), ref), "Festkomma weicht von cv2.remap ab"
    print("== Test 1: Festkomma-Gewichte gleich cv2.remap (16384 Proben)")

    # ---------- Test 2: CPU gegen GPU, bitgleich ----------
    BAG = "/home/lena/RosBagSuper/rosbag_2026-09-19_02-52-40"
    CACHE = ("/home/lena/RosBagSuper_Gui/rosbag_suite/cache/"
             "rosbag_2026-09-19_02-52-40-6e1e1c7b")
    CALIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "calib", "calib_result_new2.json")
    OUT = "/tmp/super360_modtests/colorizer_gpu"
    rec = Recording.load(os.path.join(CACHE, "recording"), bag_path=BAG)
    bag = BagReader(BAG)
    cs = bag.camera_stamps()
    T = np.asarray(json.load(open(os.path.join(CACHE, "extrinsic.json"),
                                  encoding="utf-8"))["T_imu_cam0"])
    lo = int(np.searchsorted(rec.stamps, cs[1290]))
    hi = int(np.searchsorted(rec.stamps, cs[1350]))
    a, b = int(rec.offsets[lo]), int(rec.offsets[hi])
    sub = Recording(rec.points[a:b], rec.intensity[a:b], rec.offsets[lo:hi + 1] - a,
                    rec.stamps[lo:hi], rec.poses[lo:hi], rec.meta, rec.gravity_level)
    for name, kw in (("standard", {}), ("alt", {"lens_best": False, "edge_r": 700.0}),
                     ("blaulicht", {"blue_filter": True}),
                     ("ueber_rot", {"blue_filter": True, "blue_hue_lo": 330.0,
                                    "blue_hue_hi": 30.0, "sky_grow": 0})):
        prm = C.ColorizeParams(T_imu_cam0=T, **kw)
        res = {}
        for be in ("cpu", "gpu"):
            t0 = time.time()
            C.colorize(sub, bag, CALIB, prm, os.path.join(OUT, be), backend=be)
            res[be] = (np.fromfile(os.path.join(OUT, be, "colors.bin"), np.uint8),
                       np.fromfile(os.path.join(OUT, be, "valid.bin"), np.uint8),
                       json.load(open(os.path.join(OUT, be, "meta.json"), encoding="utf-8")),
                       time.time() - t0)
        (c0, v0, m0, t_cpu), (c1, v1, m1, t_gpu) = res["cpu"], res["gpu"]
        assert np.array_equal(v0, v1), f"{name}: valid weicht ab"
        assert np.array_equal(c0, c1), f"{name}: Farben weichen ab"
        for k in ("n_valid", "n_sky_blocked", "n_sky_outvoted", "n_edge_outvoted",
                  "n_blue_outvoted", "n_blue_kept", "n_cam1_samples"):
            assert m0[k] == m1[k], f"{name}: {k} {m0[k]} != {m1[k]}"
        print(f"== Test 2 {name}: {sub.n_points} Punkte bitgleich, "
              f"CPU {t_cpu:.1f} s, GPU {t_gpu:.1f} s")
    bag.close()
    print("colorizer_gpu SELFTEST OK")

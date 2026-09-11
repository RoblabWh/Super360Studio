"""Test der 3D-Ansicht: Steuerung wie im VS-Code-Viewer, Leiste, Temperatur.

Laeuft unter einem X-Server (auch Xvfb):  xvfb-run -a python3 scripts/test_cloud_view_steuerung.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

app = QtWidgets.QApplication(sys.argv)
from ui.cloud_view import CloudView, _DREH, _ZOOM  # noqa: E402


def maus(widget, art, pos, knopf=QtCore.Qt.NoButton, knoepfe=QtCore.Qt.NoButton,
         mod=QtCore.Qt.NoModifier):
    ev = QtGui.QMouseEvent(art, QtCore.QPointF(pos), knopf, knoepfe, mod)
    QtWidgets.QApplication.sendEvent(widget, ev)
    app.processEvents()


def ziehen(widget, von, nach, knopf, mod=QtCore.Qt.NoModifier):
    maus(widget, QtCore.QEvent.MouseButtonPress, von, knopf, knopf, mod)
    schritte = 5
    for k in range(1, schritte + 1):
        p = von + (nach - von) * k / schritte
        maus(widget, QtCore.QEvent.MouseMove, p, QtCore.Qt.NoButton, knopf, mod)
    maus(widget, QtCore.QEvent.MouseButtonRelease, nach, knopf, QtCore.Qt.NoButton, mod)


view = CloudView()
view.resize(1100, 750)
view.show()
app.processEvents()

rng = np.random.default_rng(3)
N = 2_000_000
xy = rng.uniform(-30, 30, (N, 2)).astype(np.float32)
z = (np.sin(xy[:, 0] * 0.2) * 2).astype(np.float32)
pts = np.c_[xy, z]
temp = (20.0 + xy[:, 0] / 3.0).astype(np.float32)        # 10 .. 30 °C von West nach Ost
temp[xy[:, 1] > 25] = np.nan                               # ein Streifen ohne Temperatur
view.set_cloud(pts, None)
view.set_temperatur(temp)
view.reset_camera()
app.processEvents()
vw = view._vtkw
k = view._kam
print(f"Start: Gier {math.degrees(k['yaw']):.1f}°, Nick {k['pitch']:.2f} rad, Abstand {k['dist']:.1f} m")
assert abs(k["yaw"] + math.pi / 4) < 1e-9 and abs(k["pitch"] - 0.5) < 1e-9

# 1) links ziehen: drehen, 0,006 rad je Pixel
g0, n0 = k["yaw"], k["pitch"]
ziehen(vw, QtCore.QPoint(500, 400), QtCore.QPoint(600, 350), QtCore.Qt.LeftButton)
assert abs(k["yaw"] - (g0 - 100 * _DREH)) < 1e-9, (k["yaw"], g0)
assert abs(k["pitch"] - (n0 - 50 * _DREH)) < 1e-9
print(f"links ziehen 100/−50 px: Gier −{100 * _DREH:.2f} rad, Nick −{50 * _DREH:.2f} rad ✓")

# 2) rechts ziehen und Umschalt+links: verschieben, Drehung bleibt
for knopf, mod, name in ((QtCore.Qt.RightButton, QtCore.Qt.NoModifier, "rechts"),
                         (QtCore.Qt.LeftButton, QtCore.Qt.ShiftModifier, "Umschalt+links"),
                         (QtCore.Qt.LeftButton, QtCore.Qt.ControlModifier, "Strg+links")):
    ziel0, gier = k["ziel"].copy(), k["yaw"]
    ziehen(vw, QtCore.QPoint(500, 400), QtCore.QPoint(560, 400), knopf, mod)
    weg = float(np.linalg.norm(k["ziel"] - ziel0))
    soll = 60 * k["dist"] * math.tan(math.radians(25)) * 2 / vw.height()
    assert abs(weg - soll) < 1e-6 and k["yaw"] == gier, (name, weg, soll)
    print(f"{name} ziehen 60 px: Ziel {weg:.3f} m verschoben (= Plugin-Formel) ✓")

# 3) Mausrad: eine Raste hin = Abstand * exp(-0,12)
d0 = k["dist"]
ev = QtGui.QWheelEvent(QtCore.QPointF(500, 400), QtCore.QPointF(vw.mapToGlobal(QtCore.QPoint(500, 400))),
                       QtCore.QPoint(0, 0), QtCore.QPoint(0, 120), QtCore.Qt.NoButton,
                       QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False)
QtWidgets.QApplication.sendEvent(vw, ev)
app.processEvents()
assert abs(k["dist"] - d0 * math.exp(-100 * _ZOOM)) < 1e-9, (k["dist"], d0)
print(f"Mausrad eine Raste: Abstand {d0:.1f} -> {k['dist']:.1f} m ✓")

# 4) Taste q darf nichts beenden (VTK-Stil bekommt keine Tasten)
QtWidgets.QApplication.sendEvent(vw, QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Q,
                                                     QtCore.Qt.NoModifier, "q"))
app.processEvents()
assert view.isVisible(), "q hat die Ansicht geschlossen"
print("Taste q: nichts passiert ✓")

# 5) Temperatur unter dem Mauszeiger
view.reset_camera()
k["pitch"] = 1.55                      # fast senkrecht von oben
view._kamera_setzen()
app.processEvents()
i = int(np.flatnonzero(np.isfinite(temp) & (np.abs(xy[:, 0] - 12) < 0.5))[0])
cam = view._renderer.GetActiveCamera()
w, h = vw.width(), vw.height()
m = cam.GetCompositeProjectionTransformMatrix(w / h, -1, 1)
M = np.array([[m.GetElement(r, c) for c in range(4)] for r in range(4)])
q = M @ np.r_[pts[i].astype(float), 1.0]
sx, sy = (q[0] / q[3] * 0.5 + 0.5) * w, (1 - (q[1] / q[3] * 0.5 + 0.5)) * h
t = view.temperatur_bei(int(sx), int(sy))
print(f"Punkt mit {temp[i]:.2f} °C bei ({sx:.0f}, {sy:.0f}): Hover zeigt {t:.2f} °C")
assert t is not None and abs(t - temp[i]) < 0.6, (t, temp[i])
import time as _t
t0 = _t.perf_counter()
for _ in range(10):
    view.temperatur_bei(int(sx), int(sy))
print(f"  Suche unter der Maus: {(_t.perf_counter() - t0) * 100:.0f} ms")
view.set_temperatur_anzeigen(False)
maus(vw, QtCore.QEvent.MouseMove, QtCore.QPoint(int(sx), int(sy)))
assert not view._hover_timer.isActive(), "Hover läuft trotz Haken aus"
view.set_temperatur_anzeigen(True)
maus(vw, QtCore.QEvent.MouseMove, QtCore.QPoint(int(sx), int(sy)))
assert view._hover_timer.isActive()
print("Haken „Temperatur anzeigen“ schaltet das Hovern ✓")

# 6) Messen: Klick setzt Punkt, Ziehen dreht weiter
gemessen = []
view.measured.connect(lambda a, b: gemessen.append((a, b)))
view.set_measure(True)
assert view._btn_messen.isChecked()
maus(vw, QtCore.QEvent.MouseButtonPress, QtCore.QPoint(int(sx), int(sy)), QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
maus(vw, QtCore.QEvent.MouseButtonRelease, QtCore.QPoint(int(sx) + 2, int(sy)), QtCore.Qt.LeftButton)
assert gemessen and gemessen[-1][0] is not None, "Klick hat keinen Messpunkt gesetzt"
g1 = k["yaw"]
ziehen(vw, QtCore.QPoint(300, 300), QtCore.QPoint(360, 300), QtCore.Qt.LeftButton)
assert k["yaw"] != g1 and len(view._meas) == 1, "Ziehen beim Messen hat gemessen statt gedreht"
QtWidgets.QApplication.sendEvent(vw, QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Escape,
                                                     QtCore.Qt.NoModifier))
assert len(view._meas) == 0
print("Messen: Klick setzt Punkt, Ziehen dreht, Esc verwirft ✓")
view.set_measure(False)

# 7) Punktgroesse in Vierteln: 1,5 px liegt sichtbar zwischen 1 und 2
view.set_cloud(pts[::4], None)
view.set_color_mode("uniform")
view.reset_camera()
anteile = {}
for s in (1.0, 1.5, 2.0):
    view.set_point_size(s)
    app.processEvents()
    datei = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"punkt_{s}.png")
    view.screenshot(datei)             # VTK liest den Bildpuffer, grab() nicht
    from PIL import Image
    arr = np.asarray(Image.open(datei).convert("RGB"))
    anteile[s] = float((arr.reshape(-1, 3).astype(int).sum(1) > 300).mean())
print("Pixelanteil bei 1 / 1,5 / 2 px: " + " / ".join(f"{v * 100:.1f} %" for v in anteile.values()))
assert anteile[1.0] < anteile[1.5] < anteile[2.0], anteile
assert view._groesse_lbl.text() == "2" and view._groesse.value() == 8
print("Punktgröße 1,5 liegt sichtbar zwischen 1 und 2 ✓")
print("cloud_view Steuerung TEST OK")

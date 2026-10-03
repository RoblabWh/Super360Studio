#!/usr/bin/env python3
"""Gelernte Merkmale und LightGlue-Paarvergleich fuer ``verbinden_versuch.py``.

SIFT sucht Ecken und Kleckse; im verwischten Nachtbild sind die verschmiert
oder im Rauschen. SuperPoint/ALIKED sind auf solche Bilder trainiert, und
LightGlue vergleicht nicht Merkmal fuer Merkmal, sondern die ganze Anordnung.

Laeuft im Interpreter ``~/.venvs/abgleich`` (torch, lightglue) auf der GPU:

    verbinden_lightglue.py <bilder> <ziel.npz> [superpoint|aliked|disk|sift] [max_merkmale]
                           [gps_enu.json] [max_abstand_m]

Mit Geotags werden nur Bilder verglichen, die naeher als ``max_abstand_m``
beieinander aufgenommen sind. Ohne diese Grenze findet LightGlue auch zwischen
Bildern ohne gemeinsamen Boden ein paar Dutzend Treffer (dunkle Baumkronen
sehen sich aehnlich), und COLMAP haengt dann Bilder an die falsche Stelle.

Die Datei enthaelt die Punkte je Bild und die Treffer aller Bildpaare, flach
abgelegt, damit sie jede numpy-Fassung lesen kann. Geometrisch geprueft wird
erst in COLMAP.
"""

import json
import os
import sys

import numpy as np
import torch
from lightglue import ALIKED, DISK, SIFT, LightGlue, SuperPoint
from lightglue.utils import load_image, rbd


def main() -> int:
    bilder, ziel = sys.argv[1:3]
    art = sys.argv[3] if len(sys.argv) > 3 else "superpoint"
    anzahl = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
    orte = json.load(open(sys.argv[5]))["positionen"] if len(sys.argv) > 5 else None
    weit = float(sys.argv[6]) if len(sys.argv) > 6 else 110.0
    namen = sorted(n for n in os.listdir(bilder) if n.lower().endswith((".jpg", ".jpeg")))
    klasse = {"superpoint": SuperPoint, "aliked": ALIKED, "disk": DISK, "sift": SIFT}[art]
    sucher = klasse(max_num_keypoints=anzahl).eval().cuda()
    matcher = LightGlue(features=art).eval().cuda()

    merkmale = []
    with torch.inference_mode():
        for n in namen:
            # resize=None: volle Aufloesung, die Bilder sind mit 1600 px ohnehin klein
            merkmale.append(sucher.extract(load_image(os.path.join(bilder, n)).cuda(), resize=None))
        print(f"INFO {art}: {np.mean([m['keypoints'].shape[1] for m in merkmale]):.0f} "
              f"Merkmale je Bild", flush=True)
        paare, treffer = [], []
        for i in range(len(namen)):
            for j in range(i + 1, len(namen)):
                if orte and np.linalg.norm(np.subtract(orte[namen[i]], orte[namen[j]])) > weit:
                    continue
                m = rbd(matcher({"image0": merkmale[i], "image1": merkmale[j]}))["matches"]
                if len(m) >= 15:
                    paare.append((i, j, len(m)))
                    treffer.append(m.cpu().numpy().astype(np.uint32))
            print(f"INFO Paare mit Bild {i + 1}/{len(namen)}", flush=True)
    # COLMAP legt die Mitte des ersten Pixels auf (0.5, 0.5)
    punkte = [m["keypoints"][0].cpu().numpy().astype(np.float32) + 0.5 for m in merkmale]
    np.savez(ziel, namen=np.array(namen), punkte=np.concatenate(punkte),
             grenzen=np.cumsum([0] + [len(p) for p in punkte]),
             paare=np.array(paare, np.int64).reshape(-1, 3),
             treffer=np.concatenate(treffer) if treffer else np.zeros((0, 2), np.uint32))
    print(f"DONE {len(paare)} Paare -> {ziel}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

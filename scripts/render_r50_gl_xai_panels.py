"""Exp r50-GL-XAI panels: one row per (study, finding) from run_r50_gl_v2/xai/panels.npz. Columns: (1) the window the
GL global head uses most for the finding, with its Grad-CAM; (2) the window of MS1's top crop with the GL global Grad-CAM
and the crop box; (3) the same window with the parent's (label-v2 r50) Grad-CAM; (4) the crop as the local head sees it,
with its Grad-CAM. Global CAMs cover the 320 centre crop (dashed) of the 384 image. Predictions in the titles.
Usage: python3 -I scripts/render_r50_gl_xai_panels.py <project root>   (host: numpy + matplotlib)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

R = Path(sys.argv[1]); X = R / "repository/data/coatnet_d96_fold0/run_r50_gl_v2/xai"
why = {(s, t): (w, d) for s, t, w, d in json.loads((X.parent / "xai_panels_why.json").read_text())}
z = np.load(X / "panels.npz"); P = json.loads(str(z["panels"]))
out = X / "panels"; out.mkdir(exist_ok=True)


def on384(c):
    """[40, 40] CAM over the 320 centre crop -> [384, 384] with zeros outside"""
    m = np.zeros((384, 384), np.float32)
    m[32:352, 32:352] = np.kron(c, np.ones((8, 8), np.float32))
    return m / max(m.max(), 1e-12)


def show(ax, img, c, title, box=None):
    ax.imshow(img, cmap="gray"); ax.imshow(c, cmap="jet", alpha=0.35 * (c > 0.05)); ax.set_title(title, fontsize=7)
    ax.add_patch(Rectangle((32, 32), 320, 320, fill=False, ls="--", lw=0.6, ec="w"))
    if box:
        rr, cc, b = box
        ax.add_patch(Rectangle((cc - b / 2, rr - b / 2), b, b, fill=False, lw=1.2, ec="lime"))
    ax.axis("off")


by_finding = {}
for j, p in enumerate(P):
    by_finding.setdefault(p["finding"], []).append(j)
for f, js in by_finding.items():
    fig, axs = plt.subplots(len(js), 4, figsize=(11, 2.9 * len(js)))
    for row, j in zip(np.atleast_2d(axs), js):
        p = P[j]; a = lambda k: z[f"{k}_{j}"]; pr = p["pred"]; w, d = why[(p["study"], f)]
        show(row[0], a("img_top_window"), on384(a("cam_gl_top_window")), f"GL global CAM, its top window {p['top_window']}")
        show(row[1], a("img_crop_window"), on384(a("cam_gl_crop_window")), f"GL global CAM, crop window {p['crop_window']}", p["box"])
        show(row[2], a("img_crop_window"), on384(a("cam_parent_crop_window")), "label-v2 r50 CAM, crop window", p["box"])
        cc = np.kron(a("cam_crop"), np.ones((32, 32), np.float32)); cc /= max(cc.max(), 1e-12)
        row[3].imshow(a("crop"), cmap="gray"); row[3].imshow(cc, cmap="jet", alpha=0.35 * (cc > 0.05)); row[3].axis("off")
        row[3].set_title(f"crop ({p['box'][2]} px box) local CAM", fontsize=7)
        row[0].text(0, -40, f"{p['study'][-8:]}  {f} positive, {w} (rank shift {d:+.2f}):  global {pr['global']:.2f}  "
                    f"local {pr['local']:.2f}  fused {pr['fused']:.2f}  label-v2 r50 {pr['parent']:.2f}", fontsize=8)
    fig.tight_layout(); fig.savefig(out / f"{f.replace(' ', '_').replace(chr(39), '')}.png", dpi=110); plt.close(fig)
    print("wrote", out / f"{f.replace(' ', '_')}.png")

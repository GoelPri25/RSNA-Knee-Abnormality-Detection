"""MS1-GL preview: what the +-16 px crop-centre jitter looks like on real studies (host, CPU, no training).
Studies from the local smoke cache (data/coatnet_d96_cache_smoke/d96, 10 studies, same d96 format as MS1-GL reads);
crop centres from MS1's token maps (run_sig_lv2/crop_probe), the top-attention window per finding, as ms1_gl_runner.
Each row: the whole 384 px slice with the un-jittered 128 px box (white) and 4 jittered boxes (colours), the dotted
square is everywhere the centre can land; then the un-jittered crop and the 4 jittered crops (middle slice of 3).
Usage: python3 -I scripts/render_ms1_gl_jitter.py <project root>
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as pa
import matplotlib.pyplot as plt
import numpy as np

R = Path(sys.argv[1]); D = R / "repository/data"; S = D / "coatnet_d96_cache_smoke/d96"
OUT = D / "coatnet_d96_fold0/run_ms1_gl_preview"; OUT.mkdir(parents=True, exist_ok=True)
IMG, MAXS, CROP, JIT, MM = 384, 96, 128, 16, 140 / 384
SLOTS = np.repeat(np.arange(5), [26, 22, 18, 12, 18]); PLANE = ["Sagittal", "Sagittal", "Coronal", "Coronal", "Axial"]
T = json.loads((D / "coatnet_d96_fold0/assets/soft_labels_final_per_target_latD2.json").read_text())["targets_in_order"]
SHOW = [(2, "Medial Meniscus"), (1, "Contusion"), (6, "Effusion"), (7, "PF OA"), (1, "Fracture")]   # (smoke study, finding)
COL = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

meta = [json.loads(l) for l in (S / "manifest.jsonl").open()]
vol = np.load(S / "vol_000.npy", mmap_mode="r")
cp = {}
for p in sorted(glob.glob(str(D / "coatnet_d96_fold0/run_sig_lv2/crop_probe/cp_*.npz"))):
    z = np.load(p)
    for s, c, a in zip(z["ids"], z["centres"], z["a_top"]):
        cp[str(s)] = (c, a)


def centres(mask, k=94):
    """coatnet_d96_runner.centres, eval branch"""
    v = np.where(mask)[0]; v = v if len(v) >= 3 else np.arange(3)
    lo, hi = int(v.min()), int(v.max())
    cs = np.arange(lo + 1, hi) if hi - lo >= 2 else np.array([max(1, min((lo + hi) // 2, len(mask) - 2))])
    return cs[np.linspace(0, len(cs) - 1, k).round().astype(int)]


rng = np.random.default_rng(2026); h = CROP // 2
fig, ax = plt.subplots(len(SHOW), 6, figsize=(19, 3.4 * len(SHOW)), gridspec_kw={"width_ratios": [1.6, 1, 1, 1, 1, 1]})
for row, (i, f) in enumerate(SHOW):
    m = meta[i]; s = m["study_id"]; n = T.index(f)
    cen, at = cp[s]; j = int(at[n].argmax())
    sl = int(np.clip(centres(np.array(m["mask"]))[int(cen[n, j, 0])], 1, MAXS - 2))
    r0, c0 = int(np.clip(cen[n, j, 1], h, IMG - h)), int(np.clip(cen[n, j, 2], h, IMG - h))
    im = vol[m["row_start"] + sl].astype(np.float32)
    jit = rng.integers(-JIT, JIT + 1, size=(4, 2))
    boxes = [(r0, c0, "white", "no jitter", 0, 0)] + [
        (int(np.clip(r0 + dy, h, IMG - h)), int(np.clip(c0 + dx, h, IMG - h)), COL[q], f"jitter {q + 1}", dy, dx)
        for q, (dy, dx) in enumerate(jit)]
    a = ax[row, 0]; a.imshow(im, cmap="gray", vmin=0, vmax=255)
    a.add_patch(pa.Rectangle((c0 - JIT, r0 - JIT), 2 * JIT, 2 * JIT, fill=False, ec="yellow", ls=":", lw=1.2))
    for rr, cc, col, _, _, _ in boxes:
        a.add_patch(pa.Rectangle((cc - h, rr - h), CROP, CROP, fill=False, ec=col, lw=2 if col == "white" else 1.3,
                                 ls="-" if col == "white" else "--"))
        a.plot(cc, rr, "+", color=col, ms=8, mew=1.5)
    a.set_title(f"study ..{s[-6:]}  {f}  (label-v2 positive)\n{PLANE[SLOTS[sl]]} slice {sl}, MS1 top window "
                f"(attn {at[n, j] / at[n].sum():.2f})", fontsize=9)
    for q, (rr, cc, col, name, dy, dx) in enumerate(boxes):
        a = ax[row, q + 1]; a.imshow(im[rr - h:rr + h, cc - h:cc + h], cmap="gray", vmin=0, vmax=255)
        a.plot(h - (cc - c0), h - (rr - r0), "+", color="white", ms=10, mew=1.5)   # the un-jittered centre
        for sp in a.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(3)
        a.set_title(name if not q else f"{name}: dy {dy:+d}, dx {dx:+d} px\n({dy * MM:+.1f}, {dx * MM:+.1f} mm)",
                    fontsize=9, color="black")
    for a in ax[row]:
        a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"MS1-GL crop jitter: centre moved up to +-{JIT} px (+-{JIT * MM:.1f} mm) per axis; crop {CROP} px = "
             f"{CROP * MM:.0f} mm (3.5x zoom). White + = MS1's chosen centre.", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(OUT / "jitter_examples.png", dpi=90)
print(OUT / "jitter_examples.png")

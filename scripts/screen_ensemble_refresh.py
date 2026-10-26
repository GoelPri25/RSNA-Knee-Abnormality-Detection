"""Ensemble refresh screen (CPU, fold 0): L4 variants with B2, 1e-label-v2, P1-cnx; equal probability means."""
import sys
from pathlib import Path
src = Path(sys.argv[1] + "/repository/scripts/audit_phase1.py").read_text()
exec(src.split("def per_target")[0].replace("sys.argv[2:]", "[]"))
import numpy as np
idx = np.arange(len(ids))
P = {"d2_224": load(D / "r50_tta/kaggle_output/d2_ep15__none.json"), "1b": load(C / "run_continue/val_swa.json"),
     "2c": load(C / "run_r50_plane/val_swa.json"), "1e": load(C / "run_r50_pertarget/val_swa.json"),
     "lv2": load(C / "kaggle_fold0_labelv2/val_swa.json"), "cnx": load(C / "run_cnx_pertarget/val_swa.json"),
     "B2": load(C / "run_swin3d_b2/val_swa.json")}
E = {"L4": ["d2_224", "1b", "2c", "1e"],
     "L4+B2": ["d2_224", "1b", "2c", "1e", "B2"],
     "L4 1e->lv2": ["d2_224", "1b", "2c", "lv2"],
     "L4+lv2": ["d2_224", "1b", "2c", "1e", "lv2"],
     "L4+cnx": ["d2_224", "1b", "2c", "1e", "cnx"],
     "L4 1e->lv2 +cnx": ["d2_224", "1b", "2c", "lv2", "cnx"],
     "L4 1e->lv2 +B2": ["d2_224", "1b", "2c", "lv2", "B2"],
     "L4 1e->lv2 +cnx+B2": ["d2_224", "1b", "2c", "lv2", "cnx", "B2"]}
base = np.mean([P[m] for m in E["L4"]], 0); b6, b10 = scores(base, idx)[:2]
rng = np.random.default_rng(0); boots = [rng.choice(idx, len(idx)) for _ in range(1000)]
bb = [scores(base, b)[:2] for b in boots]
print(f"{'ensemble':22s} n  clear6   10-tgt   d clear6 [95%]            d 10-tgt [95%]")
for n, ms in E.items():
    S = np.mean([P[m] for m in ms], 0); c6, m10 = scores(S, idx)[:2]
    if n == "L4":
        print(f"{n:22s} {len(ms)}  {c6:.4f}  {m10:.4f}"); continue
    d = np.array([np.subtract(scores(S, b)[:2], x) for b, x in zip(boots, bb)])
    print(f"{n:22s} {len(ms)}  {c6:.4f}  {m10:.4f}  {c6-b6:+.4f} [{np.percentile(d[:,0],2.5):+.4f},{np.percentile(d[:,0],97.5):+.4f}]  "
          f"{m10-b10:+.4f} [{np.percentile(d[:,1],2.5):+.4f},{np.percentile(d[:,1],97.5):+.4f}]")

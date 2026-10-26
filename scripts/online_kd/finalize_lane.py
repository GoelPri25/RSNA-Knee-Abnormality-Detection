"""After a lane finishes: fold-0 comparison with E4 (and cv1), Kaggle dataset package, ledger row, tracker note.
Usage: python finalize_lane.py <lane> <dataset slug> "<one-line description of the change vs E4>" <tracker id or ->
Reads   colab_cli/out_<lane>/{weights_swa21_25.pt, val_swa_*.json, history.json, run_summary.json, gold/gold_<lane>.json}
Writes  reports/<lane>_handoff/ (+ dataset-metadata.json), and appends the ledger row + tracker note in the handoff repo.
"""
import hashlib
import os
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

lane, slug, change, tid = sys.argv[1:5]
HERE = Path(__file__).resolve().parent
OUT = HERE / "colab_cli" / f"out_{lane}"
KIT = Path(os.environ.get("KIT_DIR", "kit"))
REPO = Path(os.environ.get("REPO_DIR", "."))
PKG = Path(os.environ.get("PKG_ROOT", "reports")) / f"{lane}_handoff"
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]


def labels(p):
    return {r["study_id"]: np.array([np.nan if v is None else v for v in r["probabilities"]], float)
            for r in json.load(open(p))["labels"]}


def preds(p):
    return {r["study_id"]: np.array(r["probabilities"], float) for r in json.load(open(p))["predictions"]}


def auc(s, y):
    r = np.argsort(np.argsort(s)) + 1.0; n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def per(S, Y, idx):
    out = []
    for k in range(12):
        y = Y[idx, k]; c = (y <= .3) | (y >= .7); out.append(auc(S[idx, k][c], y[c] >= .7))
    return np.array(out)


rulers = {"label-v2": labels(KIT / "work/assets/soft_labels_final_per_target_latD2.json"),
          "Steven-v4": labels(KIT / "work/assets/validation_labels.json")}
refs = {"fused": {"E4": KIT / "f0e4/val_swa_fused.json", "cv1": KIT / "f0cv1/val_swa_fused.json"},
        "global": {"E4": KIT / "fold0_predictions/e4/val_swa_global.json", "cv1": KIT / "fold0_predictions/cv1/val_swa_global.json"}}
rng = np.random.default_rng(0)
lines, res = [], {}
for head in ("fused", "global"):
    me = preds(OUT / f"val_swa_{head}.json"); others = {k: preds(v) for k, v in refs[head].items()}
    ids = sorted(set(me) & set(rulers["Steven-v4"]) & set.intersection(*[set(o) for o in others.values()]))
    boot = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
    for rn, lab in rulers.items():
        Y = np.array([lab[s] for s in ids]); full = np.arange(len(ids)); A = np.array([me[s] for s in ids])
        a = np.nanmean(per(A, Y, full)); res[(head, rn)] = {"me": a}
        for on, o in others.items():
            B = np.array([o[s] for s in ids]); b = np.nanmean(per(B, Y, full))
            d = np.array([np.nanmean(per(A, Y, i)) - np.nanmean(per(B, Y, i)) for i in boot])
            res[(head, rn)][on] = (b, a - b, np.percentile(d, 2.5), np.percentile(d, 97.5), float(np.mean(d > 0)))
        if head == "fused" and rn == "label-v2":
            B = np.array([others["E4"][s] for s in ids]); pf = per(A, Y, full) - per(B, Y, full)
            res["per_finding"] = {t: round(float(v), 4) for t, v in zip(T, pf)}
for (head, rn), v in [(k, v) for k, v in res.items() if isinstance(k, tuple)]:
    e4, cv = v["E4"], v["cv1"]
    lines.append(f"{head:6s} {rn:9s}: {v['me']:.4f} vs E4 {e4[0]:.4f} ({e4[1]:+.4f} [{e4[2]:+.4f}, {e4[3]:+.4f}], P>0 {e4[4]:.2f}); "
                 f"vs cv1 {cv[0]:.4f} ({cv[1]:+.4f}, P>0 {cv[4]:.2f})")
gold = None
gp = OUT / "gold" / f"gold_{lane}.json"
if gp.exists():
    g = json.load(open(gp)); gp_ = {x["study_id"]: x["probabilities"] for x in g["predictions"]}
    import csv
    tr = list(csv.DictReader(open(os.environ.get("GOLD_LABELS", "gold_labels.csv"))))
    ids = [r["StudyInstanceUID"] for r in tr if r["StudyInstanceUID"] in gp_]
    P = np.array([gp_[s] for s in ids]); Yg = np.array([[float(r[t]) for t in T] for r in tr if r["StudyInstanceUID"] in gp_]) > .5
    gold = float(np.nanmean([auc(P[:, k], Yg[:, k]) for k in range(12)]))
    lines.append(f"gold-58 global head: {gold:.4f} (E4 0.9181, cv1 0.9275, online 0.9216)")
print("\n".join(lines)); print("per finding fused label-v2 vs E4:", res["per_finding"])

# ---- Kaggle package
PKG.mkdir(parents=True, exist_ok=True); (PKG / "fold0_predictions").mkdir(exist_ok=True)
w = OUT / "weights_swa21_25.pt"; shutil.copy(w, PKG / "weights_swa21_25.pt")
sha = hashlib.sha256(w.read_bytes()).hexdigest()
for f in OUT.glob("val_swa_*.json"):
    shutil.copy(f, PKG / "fold0_predictions" / f.name)
for f in ("history.json", "run_summary.json"):
    shutil.copy(OUT / f, PKG / f)
if gp.exists():
    (PKG / "gold").mkdir(exist_ok=True); shutil.copy(gp, PKG / "gold" / gp.name)
rs = json.load(open(OUT / "run_summary.json")); g4 = rs.get("g4", {}); cfg = rs.get("cfg", {})
(PKG / "README.md").write_text(f"""# {lane}: E4 student, {change} (Priyanjali, 2026-10-22)

Recipe = E4 (r50-GL4 l4copy recipe, seed 2026, folds 1-5, 25 epochs, SWA 21-25, E4 targets
`pubkd_targets_fullorder_rapvalues_mix73.json`) except: **{change}**. Ledger row {lane.upper()} on branch d96-handoff-2026-10-12.
Model config: G4 version `{g4.get('version')}`, CFG version `{cfg.get('version')}`, arch `{cfg.get('arch')}`.

**Weights:** `weights_swa21_25.pt`, sha256 `{sha}` (SWA 21-25, BN refreshed where the backbone has BatchNorm).

**Fold 0** (825 studies, paired bootstrap x2,000, confident cells <= 0.3 / >= 0.7):
```
""" + "\n".join(lines) + f"""
```
Per finding, fused, label-v2 cells, minus E4: {res['per_finding']}

## Inference
The weights load into `RaptorGL4` from `scripts/r50_gl4_runner.py` with the lane's env (V4_ARM=l4copy, V4_BB from the run
config; patch in `scripts/online_kd/`). The E4 submission kernel builds a ResNet-50 model, so for a non-ResNet backbone its
`load_model` must build the same backbone (timm `{cfg.get('arch')}`, crop copy of its last stage). Global head = the
`head_g` path (no crops), fused = sigmoid((g + l) / 2) on the 10 cropped findings.

## Files
- `weights_swa21_25.pt`; `fold0_predictions/val_swa_{{fused,global,local,fused_top1}}.json`; `gold/` (global head, if present);
  `history.json` (per epoch), `run_summary.json` (config + final readout).
""")
(PKG / "dataset-metadata.json").write_text(json.dumps({"title": slug.split("/")[1], "id": slug, "licenses": [{"name": "CC0-1.0"}]}, indent=1))
print("package", PKG, "weights sha", sha)

# ---- ledger row + tracker note
f = res[("fused", "label-v2")]; gl = res[("global", "label-v2")]; fs = res[("fused", "Steven-v4")]; gs = res[("global", "Steven-v4")]
row = (f"| {lane.upper()} | **(Priyanjali) E4 recipe with one change: {change}.** Colab A100 (Colab CLI, kernel keep-alive, 8-min backups), "
       f"seed 2026, folds 1-5, 25 epochs, SWA 21-25, E4 targets. Weights sha {sha[:8]}…, Kaggle dataset `{slug}`. | "
       f"Judge (as ONLINE-KD): fold 0 vs E4, label-v2 and Steven-v4 confident cells, paired bootstrap; LB probe only with the user's go. | "
       f"**Done 2026-10-22.** Fused: label-v2 {f['me']:.4f} vs E4 {f['E4'][0]:.4f} ({f['E4'][1]:+.4f} [{f['E4'][2]:+.4f}, {f['E4'][3]:+.4f}], P {f['E4'][4]:.2f}); "
       f"Steven {fs['me']:.4f} vs {fs['E4'][0]:.4f} ({fs['E4'][1]:+.4f}, P {fs['E4'][4]:.2f}). Global: label-v2 {gl['me']:.4f} vs {gl['E4'][0]:.4f} ({gl['E4'][1]:+.4f}, P {gl['E4'][4]:.2f}); "
       f"Steven {gs['me']:.4f} vs {gs['E4'][0]:.4f} ({gs['E4'][1]:+.4f}, P {gs['E4'][4]:.2f}). "
       + (f"Gold-58 global {gold:.4f} (E4 0.9181). " if gold else "") +
       f"Per finding (fused, label-v2, vs E4): {res['per_finding']}. Caveat: one seed; label-v2 fold-0 gains over-predicted the LB for ONLINE-KD (+0.0028 → LB 0.944), so the Steven numbers and the LB decide. |")
led = REPO / "docs/experiment_ledger.md"; L = led.read_text(encoding="utf-8").split("\n")
i = max(k for k, l in enumerate(L) if l.startswith("| ONLINE-KD") or l.startswith("| TRACKER"))
while i + 1 < len(L) and L[i + 1].startswith("|"):
    i += 1
L[i + 1:i + 1] = [row]; led.write_text("\n".join(L), encoding="utf-8")
md = REPO / "docs/improvement_tracker.md"
md.write_text(md.read_text() + f"\n- {lane} (2026-10-22): {change}; fused label-v2 {f['E4'][1]:+.4f} / Steven {fs['E4'][1]:+.4f} vs E4; dataset `{slug}`.\n")
print("ledger + tracker updated")

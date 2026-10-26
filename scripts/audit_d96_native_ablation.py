"""Score the Exp 4b ablation (kaggle/submission_kernels/d96_r50_native_ablation) next to Exp 1e's (2-abl).

Fold 0, Steven-v4 confident cells. For each variant: macro (12 / 10) and per-target change vs the model's own full
prediction; 4b and 1e side by side where the variant exists for both (1e has 5 d96 slots incl. coronal non-fluid and an
axial slot that takes any axial series; 4b has 4 slots, axial fluid-preferred). Thinning variants (4b only): coronal /
axial attention restricted to 18 evenly spaced windows. Band split: report-severity mild vs clear positives.
Parity: 4b 'full' vs the runner's val_swa.json.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_d96_native_ablation.py /proj
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; rid = {r["study_id"]: r for r in ref["labels"]}
TEN = [k for k, t in enumerate(T) if t not in ("Lateral OA", "Baker's")]
SHORT = ["ACL", "MCL", "MM", "LM", "MOA", "LOA", "PFOA", "Eff", "Syn", "Bak", "Cont", "Frac"]

zn = np.load(D / "native_ablation/kaggle_output/native_ablation.npz")
ze = np.load(D / "series_ablation_tta/kaggle_output/series_ablation_tta.npz")
ids = list(zn["ids"]); assert ids == list(ze["ids"])
y = np.array([[.5 if p is None else p for p in rid[s]["probabilities"]] for s in ids])
w = np.array([rid[s]["weights"] for s in ids])
mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7
VN = list(zn["variants"]); N = zn["logits"]
VE = list(ze["variants"]); E = ze["logits_1e"]
E_IDX = {"full": 0, "drop_sag_fluid": 1, "drop_sag_nonfluid": 2, "drop_cor_fluid": 3, "drop_axial": 5,
         "drop_sagittal": 6, "only_sagittal": 9}
assert [VE[i] for i in E_IDX.values()][:5] == ["full", "drop_sag_fluid", "drop_sag_nonfluid", "drop_cor_fluid", "drop_axial"]


def auc(s, t):
    ok = np.isfinite(s); s, t = s[ok], t[ok]
    r = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def per(S, sel=None):
    out = []
    for k in range(12):
        m = mask[:, k] if sel is None else mask[:, k] & sel[:, k]
        out.append(auc(S[m, k], truth[m, k]))
    return np.array(out)


# parity with the runner
val = json.loads((C / "run_r50_native_pertarget/val_swa.json").read_text())
vm = {r["study_id"]: r["probabilities"] for r in val["predictions"]} if "predictions" in val else dict(zip(val["study_ids"], val["scores"]))
V = np.array([vm[s] for s in ids], float)
p = 1 / (1 + np.exp(-N[:, 0].astype(float)))
print(f"parity 4b full vs runner val_swa: max |dp| {np.nanmax(np.abs(p - V)):.4f}, macro12 kernel {np.nanmean(per(N[:, 0])):.4f} "
      f"vs runner {np.nanmean(per(V)):.4f}\n")

fullN, fullE = per(N[:, 0]), per(E[:, 0])
rows = []
for v in VN:
    a = per(N[:, VN.index(v)]) - fullN
    b = per(E[:, E_IDX[v]]) - fullE if v in E_IDX and v != "full" else None
    rows.append((v, a, b))
print("Macro change vs own full (10-target / 12-target). 1e = d96 layout (Exp 2-abl).")
for v, a, b in rows[1:]:
    e = f"1e {b[TEN].mean():+.4f} / {b.mean():+.4f}" if b is not None else ""
    print(f"  {v:18s} 4b {a[TEN].mean():+.4f} / {a.mean():+.4f}    {e}")

print("\nPer target change vs own full (4b row, then 1e row). Seed noise per target ~0.006.")
print(f"  {'':22s}" + "".join(f"{s:>7s}" for s in SHORT))
print(f"  {'full 4b':22s}" + "".join(f"{v:7.3f}" for v in fullN))
print(f"  {'full 1e':22s}" + "".join(f"{v:7.3f}" for v in fullE))
for v, a, b in rows[1:]:
    print(f"  {v + ' 4b':22s}" + "".join(f"{x:+7.3f}" for x in a))
    if b is not None:
        print(f"  {v + ' 1e':22s}" + "".join(f"{x:+7.3f}" for x in b))

# severity bands (report reader): positives in band vs all confident negatives
st = pd.read_csv(R / "colab_bundle/raw/report_labels/report_label_extraction/report_structured_labels.csv",
                 dtype=str).fillna("").set_index("StudyInstanceUID")
key = lambda t: t.lower().replace("'", "").replace(" ", "_")
mild = np.zeros((len(ids), 12), bool); clear = np.zeros_like(mild)
for i, s in enumerate(ids):
    for k, t in enumerate(T):
        if s in st.index:
            a_, v_ = st.at[s, f"{key(t)}__status"], st.at[s, f"{key(t)}__severity"]
            clear[i, k] = a_ == "present" and v_ in ("moderate", "moderate_to_severe", "severe")
            mild[i, k] = not clear[i, k]
sel_m, sel_c = ~truth | mild, ~truth | clear
print("\nBand split, 10-target mean change vs own full (mild-band / clear-band):")
for v in VN[1:]:
    j = VN.index(v)
    am = np.nanmean((per(N[:, j], sel_m) - per(N[:, 0], sel_m))[TEN]); ac = np.nanmean((per(N[:, j], sel_c) - per(N[:, 0], sel_c))[TEN])
    s = f"  {v:18s} 4b {am:+.4f} / {ac:+.4f}"
    if v in E_IDX:
        je = E_IDX[v]
        bm = np.nanmean((per(E[:, je], sel_m) - per(E[:, 0], sel_m))[TEN]); bc = np.nanmean((per(E[:, je], sel_c) - per(E[:, 0], sel_c))[TEN])
        s += f"    1e {bm:+.4f} / {bc:+.4f}"
    print(s)
nw = zn["nwin"]
print(f"\nwindows per study, median (padding, sagF, sagNF, corF, axial): {np.median(nw, 0)}")

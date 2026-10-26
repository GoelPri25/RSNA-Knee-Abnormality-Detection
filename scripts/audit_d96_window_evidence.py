"""Missing-evidence audit (CPU, correlational): do 16-window training draws contain the windows that carry a finding?

Exp 1e SWA, 825 fold-0 studies, per-window positive CAM mass (attention-weighted, from the attention-side kernel,
data/attention_side_audit/kaggle_output). For each Steven-v4 confident positive the model ranks in the top half
among negatives (so its evidence is real), per target: n50 = windows holding 50% of the positive mass;
P(capture) = probability that the runner's training draw (16 centres, one per 1/16 of the 94 ordered windows, uniform
in the bin) contains at least one of the top-3 evidence windows, and the expected share of positive mass the draw
contains (uniform would be 16/94 = 0.17). Split by report severity (mild / mild-moderate vs moderate+).
Usage: python -I scripts/audit_d96_window_evidence.py <project root>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"
z = np.load(D / "attention_side_audit/kaggle_output/attention_side_fold0.npz")
ids = list(z["ids"]); ok = z["ok"]; logits = z["logits"]
mass = z["camx"].astype(np.float32).sum(-1)                     # [S, 94, 12] positive evidence per window
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; m = {r["study_id"]: r for r in ref["labels"]}
y = np.array([[.5 if p is None else p for p in m[s]["probabilities"]] for s in ids])
w = np.array([m[s]["weights"] for s in ids])
POS, NEG = (w > 0) & (y >= .7) & ok[:, None], (w > 0) & (y <= .3) & ok[:, None]
st = pd.read_csv(R / "colab_bundle/raw/report_labels/report_label_extraction/report_structured_labels.csv",
                 dtype=str).fillna("").set_index("StudyInstanceUID")
key = lambda t: t.lower().replace("'", "").replace(" ", "_")
rng = np.random.default_rng(0)
K, NW, DRAWS = 16, 94, 400
edges = np.linspace(0, NW, K + 1)
draws = np.minimum((edges[:-1] + rng.random((DRAWS, K)) * np.diff(edges)).astype(int), NW - 1)   # [DRAWS, 16]
member = np.zeros((DRAWS, NW), bool); member[np.arange(DRAWS)[:, None], draws] = True

rows = []
for t, name in enumerate(T):
    neg = np.sort(logits[NEG[:, t], t])
    for n in np.where(POS[:, t])[0]:
        if np.searchsorted(neg, logits[n, t]) / len(neg) < .5:
            continue
        e = mass[n, :, t]; tot = e.sum()
        if tot <= 0:
            continue
        order = np.argsort(-e)
        n50 = int(np.searchsorted(np.cumsum(e[order]) / tot, .5) + 1)
        top3 = np.zeros(NW, bool); top3[order[:3]] = True
        s = ids[n]; sev = st.at[s, f"{key(name)}__severity"] if s in st.index else ""
        band = "moderate+" if sev in ("moderate", "moderate_to_severe", "severe") else ("mild" if sev in ("mild", "mild_to_moderate") else "other")
        rows.append({"target": name, "band": band, "n50": n50, "top1_share": e[order[0]] / tot,
                     "p_capture_top3": (member & top3).any(1).mean(), "mass_in_draw": (member * e).sum(1).mean() / tot})
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(f"{len(df)} detected positives (rank >= median of negatives). Uniform mass share would be {K / NW:.2f}.\n")
g = df.groupby("target").agg(n=("n50", "size"), n50_median=("n50", "median"), top1_share=("top1_share", "mean"),
                             p_capture_top3=("p_capture_top3", "mean"), mass_in_draw=("mass_in_draw", "mean"))
print(g.loc[[t for t in T if t in g.index]].round(3).to_string())
print("\nBy severity band (all targets):")
print(df.groupby("band").agg(n=("n50", "size"), n50_median=("n50", "median"), top1_share=("top1_share", "mean"),
                             p_capture_top3=("p_capture_top3", "mean"), mass_in_draw=("mass_in_draw", "mean")).round(3).to_string())
print("\nMild vs moderate+ per target (p_capture_top3 / n50 median):")
print(df[df.band != "other"].pivot_table(index="target", columns="band", values=["p_capture_top3", "n50"], aggfunc="median").round(3).to_string())

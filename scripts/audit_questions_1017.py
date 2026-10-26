"""Diagnostic questions of 2026-10-17 (CPU, read-only, no training, no submission). Gold-58 is development evidence only:
nothing here feeds a label build or a model selection.

Q1  Where is the gap? Per-finding AUC on fold 0 (Steven-v4 ruler: all confident cells / clear band / mild band) and on
    gold-58, for label-v2, 1e, ConvNeXt P1-cnx, CK-A, Swin3D B2 and the E3 ensemble (gold: E3 members that have gold
    predictions only).
Q2  Model wrong or label wrong? Gold cells by report state (teammate structured reader): for each state, how many gold
    positives / negatives, and how well the image models rank them (mean percentile of gold positives among that
    finding's gold negatives = the positive's AUC contribution).
Q3  Convention: P(gold = 1 | report state) per finding next to label-v2's mean training value in the same state.
Q4  Noise: bootstrap SD of the 12-target macro AUC and of a paired difference (label-v2 vs 1e) at n studies.
Q5  Which yardstick tracks the LB? Submitted d96 singles: LB vs gold-58 macro, fold-0 clear6, 10-target, 12-target.

Usage: OMP_NUM_THREADS=1 python3 -I scripts/audit_questions_1017.py <project root>
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"
src = (R / "repository/scripts/audit_phase1.py").read_text()
sys.argv = [sys.argv[0], str(R)]
exec(src.split("def per_target")[0])                       # ids, T, y, mask, truth, clear, load, auc, scores
TM = D / "teammate_pp4/fold0_predictions"
TAG = lambda t: t[:9]

FOLD0 = {"lv2": C / "kaggle_fold0_labelv2/val_swa.json", "1e": C / "run_r50_pertarget/val_swa.json",
         "cnx": C / "run_cnx_pertarget/val_swa.json", "ckA": C / "run_cnx_in12k/val_swa.json",
         "B2": C / "run_swin3d_b2/val_swa.json", "1b": C / "run_continue/val_swa.json",
         "4net": TM / "four_network_fold0_val_swa.json"}
P = {k: load(v) for k, v in FOLD0.items()}
P["E3"] = np.mean([P[m] for m in ("lv2", "4net", "1b", "cnx")], 0)
mild = mask & truth & ~clear


def per(S, sel_pos=None):
    out = []
    for k in range(12):
        neg = np.where(mask[:, k] & ~truth[:, k])[0]
        pos = np.where((mask[:, k] & truth[:, k]) if sel_pos is None else sel_pos[:, k])[0]
        out.append(auc(S[np.r_[pos, neg], k], np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)])
                   if len(pos) >= 10 else np.nan)
    return np.array(out)


print("=" * 100 + "\nQ1  per-finding AUC, fold 0 (Steven-v4 ruler, confident cells)\n")
rows = {}
for m in ("lv2", "1e", "cnx", "ckA", "B2", "E3"):
    rows[f"{m} all"] = per(P[m]); rows[f"{m} clear"] = per(P[m], mask & truth & clear); rows[f"{m} mild"] = per(P[m], mild)
tab = pd.DataFrame(rows, index=T).T
tab["macro"] = tab.mean(1)
print(tab.round(3).rename(columns=TAG).to_string())
npos = pd.DataFrame({"pos": (mask & truth).sum(0), "clear+": (mask & truth & clear).sum(0), "mild+": mild.sum(0),
                     "neg": (mask & ~truth).sum(0)}, index=T).T
print("\ncounts\n" + npos.rename(columns=TAG).to_string())
short = 1 - tab.loc["E3 all", T]
print("\nE3 shortfall from 1.0, share of the summed shortfall: " +
      " ".join(f"{TAG(t)} {v / short.sum():.0%}" for t, v in short.sort_values(ascending=False).items()))

# ---------------------------------------------------------------- gold
rows_g = [r for r in csv.DictReader((R / "reports_raw/train.csv").open(encoding="utf-8")) if all(r[t] in ("0", "1") for t in T)]
gids = [r["StudyInstanceUID"] for r in rows_g]
gy = np.array([[int(r[t]) for t in T] for r in rows_g]).astype(bool)


def gload(run):
    g = json.loads((C / run / "gold_swa.json").read_text()); m = {x["study_id"]: x["probabilities"] for x in g["predictions"]}
    return np.array([m[s] for s in gids])[:, [g["targets_in_order"].index(t) for t in T]]


G = {"1e": gload("run_r50_pertarget"), "1e_s2": gload("run_r50_pertarget_seed"), "cnx": gload("run_cnx_pertarget"),
     "ckA": gload("run_cnx_in12k"), "B2": gload("run_swin3d_b2"), "1b": gload("run_continue")}
G["E3*"] = np.mean([G["1e"], G["1b"], G["cnx"]], 0)        # E3 with 1e standing in for label-v2, without 4-network
gauc = lambda S: np.array([auc(S[:, k], gy[:, k]) for k in range(12)])
gt = pd.DataFrame({m: gauc(S) for m, S in G.items()}, index=T).T
gt["macro"] = gt.mean(1)
print("\n" + "=" * 100 + f"\nQ1b per-finding AUC on gold-{len(gids)} (positives per finding in the last row)\n")
print(pd.concat([gt.round(3), pd.DataFrame([list(gy.sum(0)) + [np.nan]], index=["n_pos"], columns=gt.columns)])
      .rename(columns=TAG).to_string())

# ---------------------------------------------------------------- report states
st = pd.read_csv(R / "colab_bundle/raw/report_labels/report_label_extraction/report_structured_labels.csv",
                 dtype=str).fillna("").set_index("StudyInstanceUID")
key = lambda t: t.lower().replace("'", "").replace(" ", "_")
STATES = ["NEG", "SIL", "UNC", "MILD", "UNSP", "MOD+"]


def state(s, t):
    a, v = st.at[s, f"{key(t)}__status"], st.at[s, f"{key(t)}__severity"]
    if a == "absent":
        return "NEG"
    if a in ("not_mentioned", ""):
        return "SIL"
    if a == "uncertain":
        return "UNC"
    if v in ("moderate", "moderate_to_severe", "severe"):
        return "MOD+"
    return "MILD" if v in ("mild", "mild_to_moderate") else "UNSP"


gs = np.array([[state(s, t) for t in T] for s in gids])
print("\n" + "=" * 100 + "\nQ2  gold cells by report state, pooled over findings (E3* and 1e image models)\n"
      "    pos_pct = mean percentile of the gold positives among their finding's gold negatives (1 = perfect)\n"
      "    neg_pct = mean percentile of the gold negatives among their finding's gold positives (0 = perfect)\n")


def pct_vs(S, k, i, against):
    o = S[against, k]
    return (np.sum(o < S[i, k]) + 0.5 * np.sum(o == S[i, k])) / max(len(o), 1)


out = []
for stt in STATES:
    for lab in (True, False):
        cells = [(i, k) for i in range(len(gids)) for k in range(12) if gs[i, k] == stt and gy[i, k] == lab]
        r = {"state": stt, "gold": int(lab), "n": len(cells)}
        for m in ("E3*", "1e"):
            S = G[m]
            v = [pct_vs(S, k, i, np.where(gy[:, k] != lab)[0]) for i, k in cells]
            r[m] = round(float(np.mean(v)), 3) if v else np.nan
        out.append(r)
q2 = pd.DataFrame(out)
print(q2.to_string(index=False))
lost = {}                                                       # AUC mass lost per (state, gold) group, E3*
for k in range(12):
    pos, neg = np.where(gy[:, k])[0], np.where(~gy[:, k])[0]
    if not len(pos) or not len(neg):
        continue
    S = G["E3*"][:, k]
    for i in pos:
        for j in neg:
            if S[i] <= S[j]:
                w = (1.0 if S[i] < S[j] else 0.5) / (len(pos) * len(neg) * 12)
                lost[("pos", gs[i, k])] = lost.get(("pos", gs[i, k]), 0) + w / 2
                lost[("neg", gs[j, k])] = lost.get(("neg", gs[j, k]), 0) + w / 2
tot = sum(lost.values())
print(f"\nE3* gold macro-AUC loss {tot:.3f}; share by the cell that sits on the wrong side (half to each cell of a "
      "mis-ordered pair):")
print("  " + "  ".join(f"{a}-{b} {v / tot:.0%}" for (a, b), v in sorted(lost.items(), key=lambda x: -x[1]) if v / tot >= .02))

print("\nper finding: gold positives by report state (n) | gold negatives with a positive report state (n)")
for k, t in enumerate(T):
    p = pd.Series(gs[gy[:, k], k]).value_counts().reindex(STATES).fillna(0).astype(int)
    n = pd.Series(gs[~gy[:, k], k]).value_counts().reindex(["UNC", "MILD", "UNSP", "MOD+"]).fillna(0).astype(int)
    print(f"  {t:17s} pos " + " ".join(f"{s}:{v}" for s, v in p.items() if v) + "  |  neg-with-report+ "
          + " ".join(f"{s}:{v}" for s, v in n.items() if v))

# ---------------------------------------------------------------- Q3 convention
lv = json.loads((C / "assets/soft_labels_final_per_target_latD2.json").read_text())
lvm = {r["study_id"]: (r["probabilities"], r["weights"]) for r in lv["labels"]}
lt = lv["targets_in_order"]
print("\n" + "=" * 100 + "\nQ3  P(gold = 1 | report state) [n]  vs  label-v2 mean training value in that state [n]\n")
lines = []
for k, t in enumerate(T):
    kk = lt.index(t)
    vals = {}
    for s, (p, w) in lvm.items():
        if s in st.index and w[kk] > 0 and p[kk] is not None:
            vals.setdefault(state(s, t), []).append(p[kk])
    cells = []
    for stt in STATES:
        g = gy[gs[:, k] == stt, k]
        lvv = vals.get(stt, [])
        cells.append(f"{stt} {g.mean() if len(g) else float('nan'):.2f}[{len(g)}]/{np.mean(lvv) if lvv else float('nan'):.2f}[{len(lvv)}]")
    lines.append(f"  {t:17s} " + "  ".join(cells))
print("\n".join(lines))
pool = []
for stt in STATES:
    g = gy[gs == stt]
    pool.append(f"{stt} {g.mean():.2f}[{len(g)}]")
print("  pooled gold P(+|state): " + "  ".join(pool))

# ---------------------------------------------------------------- Q4 noise
print("\n" + "=" * 100 + "\nQ4  bootstrap SD of the 12-target macro AUC (fold 0, Steven ruler) at n studies, 500 draws\n")
rng = np.random.default_rng(0)
m12 = lambda S, ix: np.nanmean([auc(S[ix[mask[ix, k]], k], truth[ix[mask[ix, k]], k])
                                if 0 < truth[ix[mask[ix, k]], k].sum() < mask[ix, k].sum() else np.nan for k in range(12)])
for n in (300, 500, 825):
    a, d = [], []
    for _ in range(500):
        ix = rng.choice(len(ids), n, replace=True)
        x, z = m12(P["lv2"], ix), m12(P["1e"], ix)
        a.append(x); d.append(x - z)
    print(f"  n {n:4d}: SD(single) {np.std(a):.4f}   SD(paired lv2 - 1e) {np.std(d):.4f}")
print("  (LB adds what fold 0 cannot: test-label convention and seed/platform variance; seeds 2026 / 2027 gave 0.936 / 0.934)")

# ---------------------------------------------------------------- Q5 yardsticks vs LB
LB = {"run_r50_pertarget": .936, "run_r50_pertarget_seed": .934, "run_r50": .933, "run_continue": .933,
      "run_r50_state": .934, "run_r50_border": .926, "run_r50_plane_pertarget": .924, "run_r50_rulecap": .931,
      "run_cnx_in12k": .934, "run_r50_student_a050": .936, "run_swin3d_b": .906, "run_joint_j1": .930}
print("\n" + "=" * 100 + "\nQ5  submitted d96 singles: LB vs yardsticks (label-v2 0.937 has no gold file; listed fold 0 only)\n")
rec = []
for run, lb in LB.items():
    f = C / run / "val_swa.json"
    if not f.is_file():
        print(f"  {run}: no fold-0 file"); continue
    S = load(f); c6, t10 = scores(S, np.arange(len(ids)))
    gm = np.nanmean(gauc(gload(run))) if (C / run / "gold_swa.json").is_file() else np.nan
    rec.append({"run": run, "LB": lb, "gold58": gm, "clear6": c6, "10t": t10, "12t": np.nanmean(per(S)),
                "mild": np.nanmean(per(S, mild))})
q5 = pd.DataFrame(rec).set_index("run")
print(q5.round(4).to_string())
for sub, name in ((q5, "all"), (q5.drop(index=["run_swin3d_b"]), "without B1 (0.906 outlier)")):
    print(f"\n  {name} (n {len(sub)}): Pearson / Spearman with LB")
    for c in ("gold58", "clear6", "10t", "12t", "mild"):
        print(f"    {c:7s} {sub['LB'].corr(sub[c]):+.2f} / {sub['LB'].rank().corr(sub[c].rank()):+.2f}")

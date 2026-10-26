"""Readouts for the Synovitis supervision-weight candidate vs its paired 0.925 baseline (fold 0, Steven-v4 reference).

Candidate: same recipe and seed (2026) as hpo_v31_d2_resnet50_fold0; only Synovitis training weights of 2,930
inferred cells are multiplied by 0.25 (docs/synovitis_weight_experiment.md). Saved validation predictions only:
no training, no GPU, no gold.

Readouts
  1. label check: probabilities identical, weights differ only in the Synovitis column;
  2. per-target fold-0 AUC with paired study bootstrap (candidate - baseline);
  3. Synovitis/Effusion coupling (descriptive): Spearman of the two heads, and how the Synovitis head ranks studies
     Steven-v4 leaves uncertain for Synovitis (0.3 < p < 0.7) while calling Effusion positive (>= 0.7), relative to
     confident Synovitis positives and negatives. Report silence is not treated as a diagnostic negative;
  4. ensemble use: candidate replacing the v3.1-D2 224 member of the 0.935 trio, or added as a 4th member.

Container:
  podman run --rm --network=none -v "$P":/proj:ro,Z -v "$OUT":/out:rw,Z localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_synovitis_weight_candidate.py /proj /out
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
D = "repository/data/"
C = D + "synovitis_weight_candidate_seed2026/"
REF = D + "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json"
PRED = {
    "baseline_d2_224": D + "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/run/oof_predictions.json",
    "candidate_syn_w025": C + "published/oof_predictions.json",
    "d2_224_rerun": D + "hpo_v31_trainval_resnet50/results/hpo_v31_trainval_resnet50_fresh/run/oof_predictions.json",
    "d2_336": D + "highres336/results/oof_predictions.json",
    "hpo_v1": D + "hpo_v1_resnet50_auc_margin_fold0/results/baseline_validation_predictions.json",
}
DRAWS = 1000
out = {}

# 1. label check
bl = json.loads((R / C / "assets/baseline_labels.json").read_text())
cl = json.loads((R / C / "assets/candidate_labels.json").read_text())
T = bl["targets_in_order"]; S = T.index("Synovitis"); E = T.index("Effusion")
bm = {r["study_id"]: r for r in bl["labels"]}; cm = {r["study_id"]: r for r in cl["labels"]}
assert bm.keys() == cm.keys()
pb = np.array([bm[s]["probabilities"] for s in bm]); pc = np.array([cm[s]["probabilities"] for s in bm])
wb = np.array([bm[s]["weights"] for s in bm]); wc = np.array([cm[s]["weights"] for s in bm])
other = [k for k in range(len(T)) if k != S]
out["label_check"] = {"studies": len(bm), "probabilities_identical": bool(np.array_equal(pb, pc)),
                      "other_weights_identical": bool(np.array_equal(wb[:, other], wc[:, other])),
                      "synovitis_cells_changed": int((wb[:, S] != wc[:, S]).sum()),
                      "synovitis_weight_ratios": sorted({round(float(x), 4) for x in (wc[:, S] / wb[:, S])})}
assert out["label_check"]["probabilities_identical"] and out["label_check"]["other_weights_identical"]

# reference
ref = json.loads((R / REF).read_text())
assert ref["targets_in_order"] == T
ids = [r["study_id"] for r in ref["labels"]]
y = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
w = np.array([r["weights"] for r in ref["labels"]])
mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7


def load(p):
    d = json.loads((R / p).read_text())
    if "scores" in d:
        m = dict(zip(d["study_ids"], d["scores"])); tg = d.get("targets_in_order", T)
    else:
        m = {r["study_id"]: r["probabilities"] for r in d["predictions"]}; tg = d.get("targets_in_order", d.get("targets", T))
    a = np.array([m[s] for s in ids], float)
    return a[:, [list(tg).index(t) for t in T]]


def auc(s, yy):
    r = pd.Series(s).rank().to_numpy(); n1 = yy.sum(); n0 = len(yy) - n1
    return (r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def per_target(P, idx):
    return np.array([auc(P[idx, k][mask[idx, k]], truth[idx, k][mask[idx, k]]) for k in range(len(T))])


P = {k: load(v) for k, v in PRED.items()}
TRIO = ["baseline_d2_224", "d2_336", "hpo_v1"]
P["trio_mean"] = np.mean([P[m] for m in TRIO], 0)
P["trio_swap_candidate"] = np.mean([P["candidate_syn_w025"], P["d2_336"], P["hpo_v1"]], 0)
P["trio_plus_candidate"] = np.mean([P[m] for m in TRIO] + [P["candidate_syn_w025"]], 0)

# 2. per-target AUC and paired bootstrap
n = len(ids); full = np.arange(n)
tab = pd.DataFrame({k: per_target(v, full) for k, v in P.items()}, index=T)
tab.loc["MACRO"] = tab.mean()
tab.round(5).to_csv(OUT / "fold0_auc.csv", index_label="target")
PAIRS = [("candidate_syn_w025", "baseline_d2_224"), ("d2_224_rerun", "baseline_d2_224"),
         ("trio_swap_candidate", "trio_mean"), ("trio_plus_candidate", "trio_mean")]
rng = np.random.default_rng(0)
draws = {p: [] for p in PAIRS}
for _ in range(DRAWS):
    idx = rng.integers(0, n, n)
    cache = {}
    for a, b in PAIRS:
        for k in (a, b):
            if k not in cache:
                cache[k] = per_target(P[k], idx)
        d = cache[a] - cache[b]
        draws[(a, b)].append(np.append(d, np.nanmean(cache[a]) - np.nanmean(cache[b])))
rows = []
for (a, b), v in draws.items():
    v = np.array(v)
    for j, t in enumerate(T + ["MACRO"]):
        rows.append({"comparison": f"{a} - {b}", "target": t, "delta": tab.at[t, a] - tab.at[t, b],
                     "ci_low": np.nanpercentile(v[:, j], 2.5), "ci_high": np.nanpercentile(v[:, j], 97.5)})
bt = pd.DataFrame(rows)
bt.round(5).to_csv(OUT / "fold0_paired_bootstrap.csv", index=False)

# 3. Synovitis / Effusion coupling (descriptive)
eff_pos = (w[:, E] > 0) & (y[:, E] >= .7)
syn_unc = (w[:, S] > 0) & (y[:, S] > .3) & (y[:, S] < .7)
groups = {"syn_confident_pos": mask[:, S] & truth[:, S], "syn_confident_neg": mask[:, S] & ~truth[:, S],
          "eff_pos_syn_uncertain": eff_pos & syn_unc}
coup = {}
for k in ("baseline_d2_224", "candidate_syn_w025", "d2_224_rerun", "d2_336", "hpo_v1"):
    s = pd.Series(P[k][:, S]); pct = s.rank(pct=True).to_numpy()
    g = {f"mean_pct_{name}": float(pct[m].mean()) for name, m in groups.items()}
    g["spearman_syn_vs_eff_head"] = float(s.rank().corr(pd.Series(P[k][:, E]).rank()))
    # explicit positives vs the uncertain effusion group: how far the head separates them (descriptive)
    sel = groups["syn_confident_pos"] | groups["eff_pos_syn_uncertain"]
    g["auc_conf_pos_vs_eff_uncertain"] = float(auc(P[k][sel, S], groups["syn_confident_pos"][sel]))
    within = eff_pos & mask[:, S]
    g["syn_auc_within_eff_pos"] = float(auc(P[k][within, S], truth[within, S]))
    coup[k] = g
cp = pd.DataFrame(coup).T
cp.round(4).to_csv(OUT / "synovitis_effusion_coupling.csv", index_label="model")
out["group_sizes"] = {k: int(v.sum()) for k, v in groups.items()}
out["within_eff_pos_scored"] = {"n": int((eff_pos & mask[:, S]).sum()), "pos": int((eff_pos & mask[:, S] & truth[:, S]).sum())}
(OUT / "meta.json").write_text(json.dumps(out, indent=2) + "\n")

pd.set_option("display.width", 220)
print(json.dumps(out, indent=1))
print(tab.round(3).to_string()); print()
print(bt[bt.target.isin(["Synovitis", "Effusion", "MACRO"]) | (bt.comparison == "candidate_syn_w025 - baseline_d2_224")]
      .round(4).to_string(index=False)); print()
print(cp.round(3).to_string())

"""Prediction-level analysis of the l4copy family after L4C-R1 / R2 (CPU, fold 0, no training). 2026-10-19.

Questions (each decides part of the next direction):
 Q1 where is the best single's remaining error, per finding, vs E3 (the LB-0.944 ensemble)?
 Q2 variance or bias: how much do two seeds of the same recipe give when averaged (self-located and MS1 table)?
 Q3 crop branch: per finding, what does local add over global, is (g + l) / 2 the right fusion (oracle headroom only)?
 Q4 label certainty: is the error on hard-certain cells (p >= 0.85 / <= 0.15) or on the 0.7-0.85 / 0.15-0.3 band?
 Q5 series: per finding, error vs which series the study has (coronal / axial present) and where the model looked.
 Q6 training dynamics (s2027 history): per branch train loss vs val BCE, overfitting per finding.
Judge: label-v2 confident cells (weight > 0, p <= .3 or >= .7), 12-target mean AUC; clear6 on the same cells.
Usage: python -I analyze_l4copy_r1.py <data/coatnet_d96_fold0>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

C = Path(sys.argv[1]); D = C.parent; OUT = C / "l4copy_analysis"
lv = json.load(open(C / "assets/soft_labels_final_per_target_latD2.json")); T = lv["targets_in_order"]
lab = {r["study_id"]: (np.array([.5 if p is None else p for p in r["probabilities"]]), np.array(r["weights"])) for r in lv["labels"]}
CLEAR6 = [T.index(t) for t in ("ACL", "Medial Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion")]
SH = {t: t[:7] for t in T}


def load(p):
    d = json.load(open(p)); tg = d.get("targets_in_order", T); o = [tg.index(t) for t in T]
    return {x["study_id"]: np.array(x["probabilities"], float)[o] for x in d["predictions"]}


P4 = C / "l4copy_diag/modal_v1"; R1 = C / "run_l4c_r1_mlx"; R2 = C / "run_l4c_r2_mlx"; TM = D / "teammate_pp4/fold0_predictions"
F = {"p_ms1": P4 / "val_ms1_fused.json", "p_self": P4 / "val_self_fused.json", "p_glob": P4 / "val_global.json",
     "p_loc_ms1": P4 / "val_ms1_local.json", "p_loc_self": P4 / "val_self_local.json",
     "s27_ms1": R1 / "v4_l4copy_s2027/val_swa_fused.json", "s27_glob": R1 / "v4_l4copy_s2027/val_swa_global.json",
     "s27_loc_ms1": R1 / "v4_l4copy_s2027/val_swa_local.json", "s27_self": R2 / "r2_crop_p2027/val_init_fused.json",
     "r2_26": R2 / "r2_crop_p2026/val_swa_fused.json", "r2_27": R2 / "r2_crop_p2027/val_swa_fused.json",
     "l3_26": R1 / "v4_l3copy_s2026/val_swa_fused.json", "l3_27": R1 / "v4_l3copy_s2027/val_swa_fused.json",
     "r1_self": R1 / "r1_self_s2026/val_swa_fused.json", "r1_jit": R1 / "r1_selfjit_s2026/val_swa_fused.json",
     "lv2": C / "kaggle_fold0_labelv2/val_swa.json", "1b": C / "run_continue/val_swa.json", "cnx": C / "run_cnx_pertarget/val_swa.json",
     "4net": TM / "four_network_fold0_val_swa.json"}
X = {}
for k, p in F.items():
    try:
        X[k] = load(p)
    except Exception as e:
        print(f"skip {k}: {type(e).__name__} {e}"[:200])
ids = sorted(set.intersection(*(set(v) for v in X.values())))
Y = np.array([lab[s][0] for s in ids]); W = np.array([lab[s][1] for s in ids])
M = (W > 0) & ((Y <= .3) | (Y >= .7)); TT = Y >= .7
HARD = (W > 0) & ((Y <= .15) | (Y >= .85))
X = {k: np.array([v[s] for s in ids]) for k, v in X.items()}
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sg = lambda z: 1 / (1 + np.exp(-z))
X["E3"] = np.mean([X[m] for m in ("lv2", "4net", "1b", "cnx")], 0)
X["seed2_self"] = (X["p_self"] + X["s27_self"]) / 2
X["seed2_ms1"] = (X["p_ms1"] + X["s27_ms1"]) / 2
X["E3+v4self"] = np.mean([X[m] for m in ("lv2", "4net", "1b", "cnx", "p_self")], 0)
X["lv2+v4self"] = (X["lv2"] + X["p_self"]) / 2


def auc(s, t):
    rk = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (rk[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def per(Z, i=None, mask=None):
    i = np.arange(len(ids)) if i is None else i; mk = M if mask is None else mask
    return np.array([auc(Z[i, k][mk[i, k]], TT[i, k][mk[i, k]]) for k in range(12)])


rng = np.random.default_rng(2026); B = [rng.integers(0, len(ids), len(ids)) for _ in range(1000)]


def boot(a, b, cols=None):
    cols = slice(None) if cols is None else cols
    d = np.array([np.nanmean(per(X[b], i)[cols]) - np.nanmean(per(X[a], i)[cols]) for i in B])
    return d.mean(), *np.percentile(d, [2.5, 97.5]), (d > 0).mean()


L = []
pr = lambda *a: (print(*a), L.append(" ".join(str(x) for x in a)))
pr(f"n studies {len(ids)}; confident cells per finding: " + " ".join(f"{SH[t]} {int(M[:, k].sum())}/{int((M & TT)[:, k].sum())}+" for k, t in enumerate(T)))
pr("\n=== Q1 per-finding AUC (label-v2 confident cells) ===")
rows = ["p_self", "p_ms1", "p_glob", "p_loc_self", "p_loc_ms1", "s27_self", "s27_ms1", "seed2_self", "seed2_ms1", "r2_26", "lv2", "E3", "E3+v4self", "lv2+v4self", "r1_self"]
pf = {k: per(X[k]) for k in rows}
pr(f"{'model':12s} {'mean12':>7s} {'clear6':>7s} " + " ".join(f"{SH[t]:>7s}" for t in T))
for k in rows:
    pr(f"{k:12s} {pf[k].mean():7.4f} {pf[k][CLEAR6].mean():7.4f} " + " ".join(f"{v:7.4f}" for v in pf[k]))
pr("\nE3 - v4 self-located per finding: " + " ".join(f"{SH[t]} {d:+.4f}" for t, d in zip(T, pf['E3'] - pf['p_self'])))
pr("deficit share (positive part of E3 - v4self): " + " ".join(f"{SH[t]} {100 * d:.0f}%" for t, d in zip(T, np.clip(pf['E3'] - pf['p_self'], 0, None) / np.clip(pf['E3'] - pf['p_self'], 0, None).sum())))
for a, b in (("p_self", "E3"), ("p_self", "E3+v4self"), ("E3", "E3+v4self"), ("p_self", "seed2_self"), ("p_ms1", "seed2_ms1"), ("p_self", "lv2+v4self")):
    d = boot(a, b); dc = boot(a, b, CLEAR6)
    pr(f"  {b} - {a}: mean12 {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] P>0 {100 * d[3]:.0f}% | clear6 {dc[0]:+.4f} [{dc[1]:+.4f}, {dc[2]:+.4f}]")

pr("\n=== Q2 seed variance: per-finding |s2026 - s2027| (self-located) and averaging gain ===")
a, b, c = per(X["p_self"]), per(X["s27_self"]), per(X["seed2_self"])
pr("seed gap  " + " ".join(f"{SH[t]} {d:+.4f}" for t, d in zip(T, b - a)))
pr("avg gain  " + " ".join(f"{SH[t]} {d:+.4f}" for t, d in zip(T, c - (a + b) / 2)))
pr(f"mean of singles {(a.mean() + b.mean()) / 2:.4f}, 2-seed average {c.mean():.4f} (+{c.mean() - (a.mean() + b.mean()) / 2:.4f}); "
   f"prediction corr (logit) s2026 vs s2027 per finding: " + " ".join(f"{SH[t]} {np.corrcoef(lg(X['p_self'][:, k]), lg(X['s27_self'][:, k]))[0, 1]:.3f}" for k, t in enumerate(T)))
pr("corr v4self vs lv2 (cross-recipe): " + " ".join(f"{SH[t]} {np.corrcoef(lg(X['p_self'][:, k]), lg(X['lv2'][:, k]))[0, 1]:.3f}" for k, t in enumerate(T)))

pr("\n=== Q3 crop branch: global / local / fused (self-located), fusion-weight oracle (headroom, fitted on fold 0 = optimistic) ===")
g, ls, fs = per(X["p_glob"]), per(X["p_loc_self"]), per(X["p_self"])
pr(f"{'':8s} " + " ".join(f"{SH[t]:>7s}" for t in T))
for nm, v in (("global", g), ("local", ls), ("fused", fs), ("f-g", fs - g)):
    pr(f"{nm:8s} " + " ".join(f"{x:+7.4f}" if nm == "f-g" else f"{x:7.4f}" for x in v))
for nm, (gg, ll, ff) in {"s2026 ms1": ("p_glob", "p_loc_ms1", "p_ms1"), "s2027 ms1": ("s27_glob", "s27_loc_ms1", "s27_ms1")}.items():
    pr(f"f-g {nm:9s} " + " ".join(f"{x:+7.4f}" for x in per(X[ff]) - per(X[gg])) + f"  | local {per(X[ll]).mean():.4f}")
zg, zl = lg(X["p_glob"]), lg(X["p_loc_self"])
best = []
for k in range(12):
    sc = [(w, auc(sg(w * zl[:, k] + (1 - w) * zg[:, k])[M[:, k]], TT[M[:, k], k])) for w in np.linspace(0, 1, 11)]
    best.append(max(sc, key=lambda x: x[1]))
pr("oracle w_local " + " ".join(f"{SH[t]} {w:.1f}" for t, (w, _) in zip(T, best)))
pr("oracle - fused " + " ".join(f"{SH[t]} {a_ - f:+.4f}" for t, (_, a_), f in zip(T, best, fs)) + f"  mean {np.mean([a_ for _, a_ in best]) - fs.mean():+.4f}")
pr("corr(global, local) logit " + " ".join(f"{SH[t]} {np.corrcoef(zg[:, k], zl[:, k])[0, 1]:.3f}" for k, t in enumerate(T)))

pr("\n=== Q4 label certainty: AUC on hard-certain cells only (p <= .15 / >= .85) vs all confident ===")
for k_ in ("p_self", "E3"):
    h = per(X[k_], mask=HARD)
    pr(f"{k_:7s} hard " + " ".join(f"{SH[t]} {v:.4f}" for t, v in zip(T, h)) + f"  mean {np.nanmean(h):.4f}")
pr("cells hard / confident: " + " ".join(f"{SH[t]} {int(HARD[:, k].sum())}/{int(M[:, k].sum())}" for k, t in enumerate(T)))
pos_soft = (W > 0) & (Y >= .7) & (Y < .85)
pr("soft positives (0.7-0.85) share of positives: " + " ".join(f"{SH[t]} {100 * pos_soft[:, k].sum() / max(1, (M & TT)[:, k].sum()):.0f}%" for k, t in enumerate(T)))

pr("\n=== Q5 series availability and placement vs error (v4 self-located) ===")
man = {}
for f in sorted((OUT / "manifests").glob("m*.jsonl")):
    for line in open(f):
        r = json.loads(line); man[r["study_id"]] = r
SL = np.repeat(np.arange(5), [26, 22, 18, 12, 18])
has = np.array([[np.asarray(man[s]["mask"])[SL == j].sum() > 0 for j in range(5)] for s in ids])
pr("series slots present (share of studies): " + " ".join(f"slot{j} {100 * has[:, j].mean():.1f}%" for j in range(5)))
pr("slot planes: " + " | ".join(f"slot{j}: " + ",".join(sorted({sl['plane'] + ('-fluid' if sl.get('fluid') == '1' else '') for s in ids[:400] for sl in man[s]['slots'][j:j + 1]})) for j in range(5)))
tab = np.load(P4 / "tables.npz"); assert list(tab["ids"]) == ids
CF = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11]
for q, k in enumerate(CF):
    t = T[k]; S = tab["self_S"][:, q]; top = SL[S[:, 0]]
    groups = {}
    for j in range(5):
        sel = top == j
        if sel.sum() >= 30:
            groups[j] = (sel.sum(), auc(X["p_self"][sel, k][M[sel, k]], TT[sel, k][M[sel, k]]), int((M & TT)[sel, k].sum()))
    pr(f"{t:16s} top-1 slot -> (n, AUC, pos): " + "; ".join(f"slot{j} {n} {a_:.3f} {p}+" for j, (n, a_, p) in groups.items()))
for k_, sl in (("MCL", 2), ("Fracture", 2), ("Contusion", 2), ("Baker's", 4)):
    k = T.index(k_); h = has[:, sl]
    pr(f"{k_:9s} study has slot{sl}: AUC {auc(X['p_self'][h, k][M[h, k]], TT[h, k][M[h, k]]):.4f} (n {h.sum()}) | lacks: "
       f"{auc(X['p_self'][~h, k][M[~h, k]], TT[~h, k][M[~h, k]]):.4f} (n {(~h).sum()}, pos {int((M & TT)[~h, k].sum())})")

pr("\n=== Q6 training dynamics, l4copy s2027 (MS1 table) ===")
H = json.load(open(R1 / "v4_l4copy_s2027/history.json"))
pr("ep  trainG  trainL  valBCE_g valBCE_l  lv2_g  lv2_l  lv2_f")
for h in H:
    if h["epoch"] in (1, 5, 10, 15, 18, 20, 22, 25):
        pr(f"{h['epoch']:2d}  {h['train_loss']['global']:.4f}  {h['train_loss']['local']:.4f}  {h['lv2_bce_global']:.4f}   {h['lv2_bce_local']:.4f}   "
           f"{h['lv2_global']:.4f} {h['lv2_local']:.4f} {h['lv2_fused']:.4f}")
pf_ep = {h["epoch"]: h["lv2_per_fused"] for h in H}
pr("per-finding fused AUC, epoch 15 / 20 / 25 (peak epoch): " + " ".join(f"{SH[t]} {pf_ep[15][t]:.3f}/{pf_ep[20][t]:.3f}/{pf_ep[25][t]:.3f}({max(pf_ep, key=lambda e: pf_ep[e][t])})" for t in T))

pr("\n=== Case lists for XAI (v4 self-located): worst false negatives / false positives per weak finding ===")
cases = {}
for t in ("MCL", "Lateral OA", "PF OA", "Baker's", "Contusion", "Fracture", "ACL", "Effusion"):
    k = T.index(t); p = X["p_self"][:, k]; e = X["E3"][:, k]; pos = M[:, k] & TT[:, k]; neg = M[:, k] & ~TT[:, k]
    rk = pd.Series(p[M[:, k]]).rank(pct=True).to_numpy(); r = np.full(len(ids), np.nan); r[M[:, k]] = rk
    fn = [i for i in np.argsort(r) if pos[i]][:6]; fp = [i for i in np.argsort(-np.nan_to_num(r, nan=-1)) if neg[i]][:6]
    tp = [i for i in np.argsort(-np.nan_to_num(r, nan=-1)) if pos[i]][:4]
    cases[t] = {"fn": [ids[i] for i in fn], "fp": [ids[i] for i in fp], "tp": [ids[i] for i in tp]}
    pr(f"{t:10s} FN pct-rank " + " ".join(f"{r[i]:.2f}(E3 {pd.Series(e[M[:, k]]).rank(pct=True).to_numpy()[np.where(np.where(M[:, k])[0] == i)[0][0]]:.2f},y{Y[i, k]:.2f})" for i in fn))
    pr(f"{'':10s} FP pct-rank " + " ".join(f"{r[i]:.2f}(y{Y[i, k]:.2f})" for i in fp))
json.dump(cases, open(OUT / "xai_cases.json", "w"), indent=1)
(OUT / "analysis_a.txt").write_text("\n".join(L) + "\n")

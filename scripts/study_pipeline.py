"""Pipeline evolution study (CPU, read-only on existing fold-0 / target files; development evidence only).

Sections (outputs data/pipeline_study/study_pipeline.txt):
  A  LB-PRED on every existing fold-0 file of the v4 / PUB-KD-R family (+ seed means, teacher in-sample)
  B  branch fusion: per-finding global weight chosen on one seed, scored on the other (inference-only lever)
  C  per-finding student vs teacher vs v4 on the rulers (teacher in-sample on fold 0: descriptive)
  D  what the Raptor targets add over label-v2 on the training folds (ordering inside label classes, report states)
  E  what R6 + reader add over Raptor (shards 0-1, all folds): rank movement, label agreement
  F  training curves: train / val gap, late-epoch slope (schedule headroom)
Usage: python3 scripts/study_pipeline.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb_predictor as LP  # noqa: E402

T, N, IDS, C, A = LP.T, LP.N, LP.IDS, LP.C, LP.A
OUT = LP.D / "pipeline_study"; OUT.mkdir(exist_ok=True)
L = []


def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)


FJ = json.loads((LP.OUT / "fit.json").read_text()); REC = FJ["recommended"]; FR = FJ["fits"][REC]
lb = lambda m: FR["a"] + FR["b"] * m
S_ = C / "pubkd/students_r"; V4 = C / "run_r50_gl4_mlx"
F = {}
for k, seed, kind in (("ra", 2026, "pure"), ("rb", 2026, "mix"), ("rc", 2027, "pure"), ("rd", 2027, "mix")):
    for br in ("fused", "global", "local"):
        F[f"{k}_{br}"] = S_ / f"pubkd_kd_{k}/val_swa_{br}.json"
for br in ("fused", "global", "local"):
    F[f"v4s26_{br}"] = V4 / f"v4_l4copy_s2026/val_swa_{br}.json"
    F[f"v4s26self_{br}"] = V4 / f"v4_l4copy_s2026/selfloc/val_self_{br}.json"
F["v4s27_fused"] = V4 / "v4_l4copy_s2027/val_swa_fused.json"
P = {k: LP.load_pred(v) for k, v in F.items() if Path(v).is_file()}
P["teacher_rap (in-sample)"] = LP.PUB["rap"]
rk = lambda S: np.column_stack([pd.Series(S[:, j]).rank(pct=True).to_numpy() for j in range(12)])
P["mean(ra,rc)"] = (rk(P["ra_fused"]) + rk(P["rc_fused"])) / 2
P["mean(rb,rd)"] = (rk(P["rb_fused"]) + rk(P["rd_fused"])) / 2
P["mean(ra,rb)"] = (rk(P["ra_fused"]) + rk(P["rb_fused"])) / 2
P["mean(ra,v4s26)"] = (rk(P["ra_fused"]) + rk(P["v4s26_fused"])) / 2
P["mean(ra,rb,rc,rd)"] = sum(rk(P[f"r{k}_fused"]) for k in "abcd") / 4

# ---------------------------------------------------------------- A
pr("## A  LB-PRED (recommended", REC, f"LB = {FR['a']:.3f} + {FR['b']:.3f} m; paired b {FR['paired_b']:.2f}) on existing fold-0 files")
pr("   MS1-crop files (ra..rd, v4s26) flatter fold 0 vs the self-located LB pipeline by ~0.002 clear6 (SL-0v4)")
pr(f"{'file':26s} {'steven_c':>8s} {'rapc_12':>8s} {'steven12':>8s} {'rap_sp':>7s} {'m':>7s} {'LB-PRED':>7s}  actual")
ACT = {"ra_fused": .943, "rb_fused": .943, "v4s26self_fused": .939}
for k, S in P.items():
    a, b, c, d = (LP.SM[x](S) for x in ("steven_c", "rapc_12", "steven_12", "rap_sp"))
    m = 0.5 * (a + b)
    pr(f"{k:26s} {a:8.4f} {b:8.4f} {c:8.4f} {d:7.3f} {m:7.4f} {lb(m):7.3f}  {ACT.get(k, '')}")
pr("   residual (actual - LB-PRED): ra %+.4f rb %+.4f v4self %+.4f" % tuple(
    ACT[k] - lb(LP.metric(REC, P[k])) for k in ("ra_fused", "rb_fused", "v4s26self_fused")))

# ---------------------------------------------------------------- B
pr("\n## B  per-finding global weight a (fused logit = a*global + (1-a)*local); chosen on one seed, scored on the other")
sig, logit = LP.sig, LP.logit
GRID = [0, .25, .5, .75, 1.0]
t0, m0 = LP.RL["steven"]; tr_, mr_ = LP.RL["rapc"]


def fa(gl, lo, a):
    return sig(a * logit(gl) + (1 - a) * logit(lo))


def per_finding_score(S, j):
    """0.5 * Steven confident AUC + 0.5 * rapc AUC for one finding (the recommended metric's per-finding analogue)."""
    return 0.5 * (LP.auc(S[m0[:, j], j], t0[m0[:, j], j]) + LP.auc(S[mr_[:, j], j], tr_[mr_[:, j], j]))


def choose(gl, lo):
    out = []
    for j in range(12):
        S = np.zeros((N, 12)); sc = {}
        for a in GRID:
            S[:, j] = fa(gl[:, j], lo[:, j], a); sc[a] = per_finding_score(S, j)
        out.append(max(GRID, key=lambda a: (round(sc[a], 6), a == .5)))   # ties keep the 50/50 default
    return out


def apply(gl, lo, al):
    return np.column_stack([fa(gl[:, j], lo[:, j], al[j]) for j in range(12)])


rng = np.random.default_rng(0); BO = [rng.integers(0, N, N) for _ in range(300)]
for src, dst in (("ra", "rc"), ("rc", "ra"), ("rb", "rd"), ("rd", "rb"), ("v4s26", "ra"), ("ra", "v4s26")):
    if f"{src}_global" not in P or f"{dst}_global" not in P:
        continue
    al = choose(P[f"{src}_global"], P[f"{src}_local"])
    Sd = apply(P[f"{dst}_global"], P[f"{dst}_local"], al); base = P[f"{dst}_fused"]
    d = LP.metric(REC, Sd) - LP.metric(REC, base)
    db = np.array([LP.metric(REC, Sd, i) - LP.metric(REC, base, i) for i in BO[:150]])
    pr(f"  a chosen on {src:6s} -> {dst:6s}: m d {d:+.4f} [{np.percentile(db, 2.5):+.4f}, {np.percentile(db, 97.5):+.4f}] "
       f"P>0 {(db > 0).mean():.0%} (paired LB {FR['paired_b'] * d:+.4f}) | a = " + " ".join(f"{T[j][:5]}:{al[j]:.2f}" for j in range(12)))
# fixed prior from L4C-X: PF OA + Fracture global-only
for k in ("ra", "rb", "rc", "rd", "v4s26"):
    al = [1.0 if j in LP.GO else .5 for j in range(12)]
    Sd = apply(P[f"{k}_global"], P[f"{k}_local"], al)
    d = LP.metric(REC, Sd) - LP.metric(REC, P[f"{k}_fused"])
    pr(f"  {k:6s} PF OA + Fracture global-only (L4C-X prior): m d {d:+.4f}  (LB {FR['paired_b'] * d:+.4f})")
pr("  branch alone (m): " + " ".join(f"{k} g {LP.metric(REC, P[k + '_global']):.4f} l {LP.metric(REC, P[k + '_local']):.4f} f {LP.metric(REC, P[k + '_fused']):.4f};"
                                     for k in ("ra", "rb", "rc", "rd", "v4s26")))

# ---------------------------------------------------------------- C
pr("\n## C  per finding, recommended-metric analogue (0.5 steven + 0.5 rapc AUC); teacher in-sample (descriptive only)")
cols = ["v4s26_fused", "v4s27_fused", "ra_fused", "rc_fused", "rb_fused", "rd_fused", "teacher_rap (in-sample)"]
pr(f"{'finding':17s} " + " ".join(f"{c[:9]:>9s}" for c in cols) + "  pure-v4  mix-v4")
for j in range(12):
    v = [per_finding_score(P[c], j) for c in cols]
    pr(f"{T[j]:17s} " + " ".join(f"{x:9.4f}" for x in v) + f"  {(v[2] + v[3] - v[0] - v[1]) / 2:+.4f} {(v[4] + v[5] - v[0] - v[1]) / 2:+.4f}")

# ---------------------------------------------------------------- D
pr("\n## D  training folds: what the Raptor targets carry beyond label-v2")
tp = json.loads((C / "pubkd/pubkd_targets_raptor_pure.json").read_text())["labels"]
tm = json.loads((C / "pubkd/pubkd_targets_raptor_mix73.json").read_text())["labels"]
TP = {r["study_id"]: np.array(r["probabilities"]) for r in tp}; TM = {r["study_id"]: np.array(r["probabilities"]) for r in tm}
lv = json.loads((A / "soft_labels_final_per_target_latD2.json").read_text())
LV = {r["study_id"]: (np.array([.5 if p is None else p for p in r["probabilities"]]), np.array(r["weights"])) for r in lv["labels"]}
fold = {x["study_id"]: x["fold"] for x in csv.DictReader((A / "folds.csv").open())}
st = pd.read_csv(LP.R / "colab_bundle/raw/report_labels/report_label_extraction/report_structured_labels.csv",
                 dtype=str).fillna("").set_index("StudyInstanceUID")
key = lambda t: t.lower().replace("'", "").replace(" ", "_")
trn = [s for s in TP if fold.get(s) not in (None, "0") and s in LV]
Y = np.array([LV[s][0] for s in trn]); W = np.array([LV[s][1] for s in trn]); TT = np.array([TP[s] for s in trn])
TX = np.array([TM[s] for s in trn])
SEV = {"mild": 1, "mild_to_moderate": 2, "moderate": 3, "moderate_to_severe": 4, "severe": 5}
stat = np.full((len(trn), 12), "", object); sev = np.zeros((len(trn), 12))
for i, s in enumerate(trn):
    if s in st.index:
        for j, t in enumerate(T):
            stat[i, j] = st.at[s, f"{key(t)}__status"]; sev[i, j] = SEV.get(st.at[s, f"{key(t)}__severity"], 0)
sp = lambda a, b: pd.Series(a).rank().corr(pd.Series(b).rank()) if len(a) > 5 else np.nan
pr(f"  {len(trn)} training studies. Columns: rho(teacher, lv2) on weighted cells | within lv2>=.7 positives: rho(teacher, report severity) "
   "vs rho(lv2, severity) | within lv2<=.3 negatives: teacher mean on report-absent vs silent(not_mentioned) | "
   "share of lv2-confident cells where teacher sits on the other side of .5 | rho(pure, mix73)")
for j in range(12):
    w = W[:, j] > 0; pos = w & (Y[:, j] >= .7) & (sev[:, j] > 0); neg = w & (Y[:, j] <= .3)
    conf = w & ((Y[:, j] >= .7) | (Y[:, j] <= .3))
    flip = ((TT[conf, j] >= .5) != (Y[conf, j] >= .7)).mean()
    ab = neg & (stat[:, j] == "absent"); nm = neg & (stat[:, j] == "not_mentioned")
    pr(f"  {T[j]:17s} rho {sp(TT[w, j], Y[w, j]):.3f} | sev rho teacher {sp(TT[pos, j], sev[pos, j]):+.3f} lv2 {sp(Y[pos, j], sev[pos, j]):+.3f} (n {pos.sum()})"
       f" | neg absent {TT[ab, j].mean():.3f} silent {TT[nm, j].mean() if nm.any() else np.nan:.3f} (n {ab.sum()}/{nm.sum()})"
       f" | flip {flip:.1%} | rho(pure,mix) {sp(TT[:, j], TX[:, j]):.3f}")

# ---------------------------------------------------------------- E
pr("\n## E  shards 0-1 (all folds): what R6 + reader add to the Raptor half")
r = {}
for m in ("384", "224"):
    d = {}
    for f in sorted((C / "pubkd").glob(f"raptor{m}_*.json")):
        for s, v in json.loads(f.read_text())["pred"].items():
            d[s] = 0.5 * (np.array(v["plain"]) + np.array(v["mirror"]))
    r[m] = d
k6 = lambda stem: pd.concat([pd.read_csv(f, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[T]
                             for f in sorted((C / "pubkd/r6").glob(f"shard*/{stem}*.csv"))])
R6, RD = k6("r6_train_shard"), k6("reader_train_shard")
ids = sorted(set(R6.index) & set(r["384"]))
pct = lambda v: pd.Series(v).rank(pct=True).to_numpy()
X384 = np.array([r["384"][s] for s in ids]); X224 = np.array([r["224"][s] for s in ids])
X6 = R6.loc[ids].to_numpy(float); XR = RD.loc[ids].to_numpy(float)
WR = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .6, "Synovitis": .8, "Baker's": .35}
VR = {"Baker's": .035, "Contusion": .03, "Medial OA": .025, "ACL": .025, "Medial Meniscus": .02, "MCL": .02,
      "Lateral Meniscus": .008, "Fracture": .008, "Lateral OA": .008, "PF OA": .008, "Effusion": .004, "Synovitis": .003}
fo = np.array([fold.get(s) for s in ids]); f0 = fo == "0"; trm = (fo != "0") & np.array([s in LV for s in ids])
Ytr = np.array([LV[s][0] if s in LV else np.full(12, .5) for s in ids]); Wtr = np.array([LV[s][1] if s in LV else np.zeros(12) for s in ids])
pr(f"  studies {len(ids)} (fold 0 {f0.sum()}). Columns: rho(R6, Raptor) | rho(full blend, Raptor blend) | share of cells whose "
   "rank moves > 0.10 | training folds, lv2-confident: AUC Raptor vs full blend (in-sample: label-family fit, not skill)")
mv_all = []
for j, t in enumerate(T):
    rr = .65 * pct(X384[:, j]) + .35 * pct(X224[:, j]); w, v = WR.get(t, .5), VR[t]
    full = pct((1 - v) * pct(w * rr + (1 - w) * pct(X6[:, j])) + v * pct(XR[:, j])); rp = pct(rr)
    mv = np.abs(full - rp); mv_all.append(mv)
    c = trm & (Wtr[:, j] > 0) & ((Ytr[:, j] >= .7) | (Ytr[:, j] <= .3))
    pr(f"  {t:17s} rho(R6,rap) {sp(X6[:, j], rr):.3f} rho(full,rap) {sp(full, rr):.3f} moved>.1 {(mv > .1).mean():5.1%} | "
       f"lv2 AUC rap {LP.auc(rp[c], Ytr[c, j] >= .7):.4f} full {LP.auc(full[c], Ytr[c, j] >= .7):.4f}")
pr(f"  mean |rank move| {np.mean(mv_all):.3f}; for scale: |ra - v4| on fold 0 = 0.036-0.085 by band (diag_pubkd_r)")

# ---------------------------------------------------------------- F
pr("\n## F  training curves (history.json): label-v2 fused AUC on fold 0 at epochs 15/20/25, train-val gap, val BCE")
for k, p in (("v4s26", V4 / "v4_l4copy_s2026/history.json"), ("v4s27", V4 / "v4_l4copy_s2027/history.json"),
             ("ra", S_ / "pubkd_kd_ra/history.json"), ("rb", S_ / "pubkd_kd_rb/history.json"),
             ("rc", S_ / "pubkd_kd_rc/history.json"), ("rd", S_ / "pubkd_kd_rd/history.json")):
    h = {e["epoch"]: e for e in json.loads(p.read_text())}
    g = lambda e, f: h.get(e, {}).get(f, np.nan)
    pr(f"  {k:6s} lv2_fused 15/20/25 {g(15, 'lv2_fused'):.4f}/{g(20, 'lv2_fused'):.4f}/{g(25, 'lv2_fused'):.4f} "
       f"slope 20->25 {g(25, 'lv2_fused') - g(20, 'lv2_fused'):+.4f} | train lv2 fused 25 {g(25, 'train_lv2_fused'):.4f} gap "
       f"{g(25, 'train_lv2_fused') - g(25, 'lv2_fused'):+.4f} | val BCE (validation_labels) 15/25 {g(15, 'val_bce_fused'):.3f}/{g(25, 'val_bce_fused'):.3f} "
       f"lv2 BCE 25 {g(25, 'lv2_bce_fused'):.3f}")

# ---------------------------------------------------------------- G
pr("\n## G  paired LB-PRED vs the LB pipeline of v4 (self-located crops, 0.939): students' self-location penalty is unknown,")
pr("   so their global-only file is the conservative proxy (students' crop branch adds ~0: f - g within +-0.001)")
base = LP.metric(REC, P["v4s26self_fused"])
for k in ("ra_global", "ra_fused", "rb_global", "rb_fused", "rc_global", "rd_global"):
    d = LP.metric(REC, P[k]) - base
    pr(f"  {k:10s} m d vs v4 self {d:+.4f} -> LB {0.939 + FR['paired_b'] * d:.4f}   (actual ra 0.943, rb 0.943)")
pr("  v4 own branch penalty under self-location: fused MS1 %.4f -> self %.4f (m d %+.4f)" % (
    LP.metric(REC, P["v4s26_fused"]), base, base - LP.metric(REC, P["v4s26_fused"])))

(OUT / "study_pipeline.txt").write_text("\n".join(L) + "\n")

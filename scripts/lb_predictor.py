"""LB-PRED: public-LB predictor from a fold-0 prediction file (development evidence; CPU, numpy + pandas, read-only).

Competition metric = macro ROC-AUC over the 12 findings. Every candidate below is a fold-0 score; the predictor is a
1-variable linear fit LB = a + b * metric over our LB-scored submissions that have an out-of-sample fold-0 file
(in-sample models excluded: all-data / trainval, public checkpoints, rap2 init). Rulers (truth on fold 0):
  steven  Steven-v4 validation labels (validation_labels.json), confident cells (w > 0, y <= .3 or >= .7)
  lv2 / d2 / pt   label-v2 / D2 structured / per-target (1e) label files, same confident rule
  cons    cells where steven, lv2, d2 and pt are all confident and agree
  rapc    steven confident cells on which the public Raptor blend agrees (p >= .5 for positives, < .5 negatives)
  fullc   the same with the full public blend (PUB-KD pure targets); bothc = cells both agree with (teacher-neutral)
  repany  report-structured: present (any severity) vs absent;  repmod: present moderate+ vs absent
  rap / r6 / rdr / pub   public models as ruler (in-sample: "their label family"): confident p >= .7 / <= .3;
          rap = 0.65 / 0.35 logit blend of Raptor 384 + 224 (mirror TTA, all 825); r6, reader (rdr) and the full
          0.958-blend logit (pub) exist for the 2/3 of fold 0 in shards 0-1
  *_sp    mean per-finding Spearman agreement with that public model's probabilities
Metric suffixes: _12 macro AUC over findings with >= 5 positives and negatives; _c clear band (positives restricted to
report moderate+, findings with >= 20 such positives; steven_c = clear6, the old baseline); _10 = steven without
Lateral OA / Baker's.

Usage:
  python3 scripts/lb_predictor.py build                  # rebuild data/lb_predictor/{submissions.csv,report.txt,fit.json}
  python3 scripts/lb_predictor.py predict FILE[+FILE2...] [--parent PFILE --parent-lb 0.939]
FILE: a fold-0 json ({predictions: [{study_id, probabilities}]}, {study_ids, scores}, or {study: probs | [fused, ..]});
several files joined with '+' are merged (shards); 'mean:F1,F2,...' = equal probability mean of members.
"""
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(os.environ.get("RSNA_ROOT", Path(__file__).resolve().parents[2]))
D = R / "repository/data"; C = D / "coatnet_d96_fold0"; A = C / "assets"; TM = D / "teammate_pp4/fold0_predictions"
OUT = Path(os.environ.get("LBP_OUT", D / "lb_predictor"))      # LBP_OUT: build / read another fit
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
CF = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11]; GO = [6, 11]
sig = lambda z: 1 / (1 + np.exp(-z))
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

# ------------------------------------------------------------------ rulers
ref = json.loads((A / "validation_labels.json").read_text()); assert ref["targets_in_order"] == T
IDS = [r["study_id"] for r in ref["labels"]]; N = len(IDS); POS = {s: i for i, s in enumerate(IDS)}


def labfile(p, ids=IDS):
    d = json.loads(Path(p).read_text()); assert d["targets_in_order"] == T
    m = {r["study_id"]: r for r in d["labels"]}
    y = np.full((len(ids), 12), .5); w = np.zeros((len(ids), 12))
    for i, s in enumerate(ids):
        if s in m:
            y[i] = [.5 if v is None else v for v in m[s]["probabilities"]]; w[i] = m[s]["weights"]
    return y, w


def confident(y, w):
    return y >= .7, (w > 0) & ((y <= .3) | (y >= .7))


def public_rulers():
    """Raptor (all fold-0) and R6 / reader / full blend (shards 0-1) probabilities on fold 0, NaN where missing."""
    out = {}
    for m in ("384", "224"):
        P = np.full((N, 12), np.nan)
        for f in sorted((C / "pubkd").glob(f"raptor{m}_*.json")):
            d = json.loads(f.read_text()); assert d["targets"] == T
            for s, v in d["pred"].items():
                if s in POS:
                    P[POS[s]] = 0.5 * (np.array(v["plain"]) + np.array(v["mirror"]))
        out["r" + m] = P
    out["rap"] = sig(0.65 * logit(out["r384"]) + 0.35 * logit(out["r224"]))
    for k, stem in (("r6", "r6_train_shard"), ("rdr", "reader_train_shard")):
        P = np.full((N, 12), np.nan)
        for f in sorted((C / "pubkd/r6").glob(f"shard*/{stem}*.csv")):
            df = pd.read_csv(f, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
            for s in df.index.intersection(IDS):
                P[POS[s]] = df.loc[s, T].to_numpy(float)
        out[k] = P
    # full public blend in logit space, their per-finding weights (build_pubkd_targets.py)
    wr = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .6, "Synovitis": .8, "Baker's": .35}
    vr = {"Baker's": .035, "Contusion": .03, "Medial OA": .025, "ACL": .025, "Medial Meniscus": .02, "MCL": .02,
          "Lateral Meniscus": .008, "Fracture": .008, "Lateral OA": .008, "PF OA": .008, "Effusion": .004, "Synovitis": .003}
    z = np.zeros((N, 12))
    for j, t in enumerate(T):
        w, v = wr.get(t, .5), vr[t]
        z[:, j] = (1 - v) * (w * logit(out["rap"][:, j]) + (1 - w) * logit(out["r6"][:, j])) + v * logit(out["rdr"][:, j])
    z[np.isnan(out["r6"]).any(1)] = np.nan
    out["pub"] = sig(z)
    return out


def build_rulers():
    st = pd.read_csv(R / "colab_bundle/raw/report_labels/report_label_extraction/report_structured_labels.csv",
                     dtype=str).fillna("").set_index("StudyInstanceUID")
    key = lambda t: t.lower().replace("'", "").replace(" ", "_")
    status = np.full((N, 12), "", object); sev = np.full((N, 12), "", object)
    for i, s in enumerate(IDS):
        if s in st.index:
            for k, t in enumerate(T):
                status[i, k] = st.at[s, f"{key(t)}__status"]; sev[i, k] = st.at[s, f"{key(t)}__severity"]
    modp = (status == "present") & np.isin(sev, ["moderate", "moderate_to_severe", "severe"])
    Rl = {}
    ys = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
    ws = np.array([r["weights"] for r in ref["labels"]])
    Rl["steven"] = confident(ys, ws)
    for k, f in (("lv2", "soft_labels_final_per_target_latD2.json"), ("d2", "soft_labels_hpo_v3_1_d2_structured.json"),
                 ("pt", "soft_labels_final_per_target_any_model.json")):
        Rl[k] = confident(*labfile(A / f))
    tr = np.stack([Rl[k][0] for k in ("steven", "lv2", "d2", "pt")]); mk = np.stack([Rl[k][1] for k in ("steven", "lv2", "d2", "pt")])
    Rl["cons"] = (tr[0], mk.all(0) & (tr == tr[0]).all(0))
    Rl["repany"] = (status == "present", np.isin(status, ["present", "absent"]))
    Rl["repmod"] = (modp, modp | (status == "absent"))
    PUB = public_rulers()
    t0, m0 = Rl["steven"]
    Rl["rapc"] = (t0, m0 & ((PUB["rap"] >= .5) == t0))
    # full public blend (PUB-KD pure targets: blend order, logit-blend values; all 825) as the mirror of rapc, and the
    # teacher-neutral cells both public teachers agree with (added 2026-10-20: rapc favours Raptor-teacher students)
    full = labfile(C / "pubkd/pubkd_targets_pure.json")[0]
    Rl["fullc"] = (t0, m0 & ((full >= .5) == t0))
    Rl["bothc"] = (t0, Rl["rapc"][1] & Rl["fullc"][1])
    for k in ("rap", "r6", "rdr", "pub"):
        P = PUB[k]; ok = ~np.isnan(P)
        Rl[k] = (np.nan_to_num(P) >= .7, ok & ((np.nan_to_num(P, nan=.5) <= .3) | (np.nan_to_num(P) >= .7)))
    return Rl, PUB, modp


RL, PUB, MODP = build_rulers()


def target_sets():
    """Per ruler: findings scored by _12 (>= 5 pos and neg) and by _c (>= 20 report-moderate+ positives)."""
    out = {}
    for k, (t, m) in RL.items():
        f12 = [j for j in range(12) if (m[:, j] & t[:, j]).sum() >= 5 and (m[:, j] & ~t[:, j]).sum() >= 5]
        fc = [j for j in range(12) if (m[:, j] & t[:, j] & MODP[:, j]).sum() >= 20 and (m[:, j] & ~t[:, j]).sum() >= 5]
        out[k] = (f12, fc)
    return out


TS = target_sets()
TEN = [j for j, t in enumerate(T) if t not in ("Lateral OA", "Baker's")]


# ------------------------------------------------------------------ metrics
def auc(s, t):
    n1 = int(t.sum()); n0 = len(t) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def m_auc(S, ruler, fs, clear=False, idx=None):
    idx = np.arange(N) if idx is None else idx
    t, m = RL[ruler]; v = []
    for j in fs:
        sel = idx[m[idx, j] & (~t[idx, j] | MODP[idx, j])] if clear else idx[m[idx, j]]
        v.append(auc(S[sel, j], t[sel, j]))
    return float(np.nanmean(v))


def m_sp(S, src, idx=None):
    idx = np.arange(N) if idx is None else idx
    P = PUB[src]; v = []
    for j in range(12):
        ok = idx[~np.isnan(P[idx, j])]
        v.append(pd.Series(S[ok, j]).rank().corr(pd.Series(P[ok, j]).rank()))
    return float(np.mean(v))


def single_metrics():
    M = {}
    for k in RL:
        f12, fc = TS[k]
        M[f"{k}_12"] = (lambda S, idx=None, k=k, f=f12: m_auc(S, k, f, False, idx))
        if len(fc) >= 3:
            M[f"{k}_c"] = (lambda S, idx=None, k=k, f=fc: m_auc(S, k, f, True, idx))
    M["steven_10"] = lambda S, idx=None: m_auc(S, "steven", TEN, False, idx)
    for k in ("rap", "r6", "rdr", "pub"):
        M[f"{k}_sp"] = (lambda S, idx=None, k=k: m_sp(S, k, idx))
    return M


SM = single_metrics()
BASE = "steven_c"
NEUTRAL = "steven_c+bothc_12"   # teacher-neutral: for students of different public teachers (rapc favours Raptor, pub / fullc the full blend)
PUBLIC = ("rap", "rapc", "fullc", "bothc", "r6", "rdr", "pub")              # rulers that use public-model predictions


def metric(name, S, idx=None):
    """name = single metric, or 'a+b' = fixed 50/50 average of two single metrics (no extra fitted parameter)."""
    if "+" in name:
        a, b = name.split("+")
        return 0.5 * (SM[a](S, idx) + SM[b](S, idx))
    return SM[name](S, idx)


# ------------------------------------------------------------------ prediction files
def load_one(p):
    p = str(p)
    if "#" in p:                                         # locator npz: path#wp (A-np) / path#wp_patch (A')
        f, mode = p.split("#"); z = np.load(f); ids = [str(s) for s in z["ids"]]
        g = z["glob"]; f_ = g.copy(); f_[:, CF] = 0.5 * (g + z["loc_wp_100"])[:, CF]
        if mode == "wp_patch":
            f_[:, GO] = g[:, GO]
        return dict(zip(ids, sig(f_)))
    d = json.loads(Path(p).read_text())
    if "predictions" in d:
        m = {r["study_id"]: r["probabilities"] for r in d["predictions"]}
    elif "study_ids" in d:
        m = dict(zip(d["study_ids"], d["scores"]))
    else:                                                # selfloc: {study: [fused, global, local]} or {study: probs}
        m = {s: (v[0] if isinstance(v[0], list) else v) for s, v in d.items()}
    tg = d.get("targets_in_order", d.get("targets", T)) if isinstance(d, dict) and ("predictions" in d or "study_ids" in d) else T
    o = [list(tg).index(t) for t in T]
    return {s: np.asarray(v, float)[o] for s, v in m.items()}


def load_pred(spec):
    spec = str(spec)
    if spec.startswith("mean:"):
        return np.mean([load_pred(x) for x in spec[5:].split(",")], 0)
    m = {}
    for part in spec.split("+"):
        m.update(load_one(part))
    miss = [s for s in IDS if s not in m]
    assert not miss, f"{spec}: {len(miss)} fold-0 studies missing"
    return np.array([m[s] for s in IDS], float)


# ------------------------------------------------------------------ the labelled dataset
def r_(d, f="val_swa.json"):
    return str(C / d / f)


LOC = str(C / "l4copy_analysis/meluxina_loc_v1/loc_s2026/xai_arrays_s2026.npz")
F = {"ra": r_("pubkd/students_r/pubkd_kd_ra", "val_swa_fused.json"),
     "coat0": r_("bbf_mlx/bbf_coat0_s2026/selfloc", "self_0.json") + "+" + r_("bbf_mlx/bbf_coat0_s2026/selfloc", "self_1.json"),
     "A-np": LOC + "#wp", "A'": LOC + "#wp_patch",
     "v4": r_("run_r50_gl4_mlx/v4_l4copy_s2026/selfloc", "val_self_fused.json"),
     "P3v2": r_("run_r50_student_a050"), "CK-A": r_("run_cnx_in12k"), "J1": r_("run_joint_j1"),
     "4net-DS": str(TM / "four_network_deep_supervision_fold0_val_swa.json"), "4net": str(TM / "four_network_fold0_val_swa.json"),
     "B2": r_("run_swin3d_b2"), "B1": r_("run_swin3d_b"), "fine": str(TM / "fine_branch_fold0_val_swa.json"),
     "lv3w": str(TM / "label_v3w_fold0_val_swa.json"), "7a": r_("run_r50_rulecap"), "lv2": str(TM / "label_v2_fold0_val_swa.json"),
     "1e-mirror": r_("run_r50_pertarget", "val_swa_mirror.json"), "v3": r_("run_r50_spatial"), "1e-s2": r_("run_r50_pertarget_seed"),
     "2d": r_("run_r50_plane_pertarget"), "3b": r_("run_r50_state"), "3a": r_("run_r50_border"), "1e": r_("run_r50_pertarget"),
     "1c": r_("run_r50"), "1b": r_("run_continue"), "2c": r_("run_r50_plane"), "P1-cnx": r_("run_cnx_pertarget"),
     "d2_224": str(D / "r50_tta/kaggle_output/d2_ep15__none.json"), "hpo_v1": str(D / "r50_tta/kaggle_output/hpo_v1_ep17__none.json"),
     "effv2": str(D / "efficientnet_v31_d2/results/oof_predictions.json")}
mean_ = lambda *k: "mean:" + ",".join(F[x] for x in k)
# name, LB, spec, combination, arch family, label family, parent (for the paired fit), notes
SUBS = [
    ("ra", .943, F["ra"], "single", "r50-GL4", "raptor-KD", "v4", "MS1-crop fused file only (flatters fold 0 slightly); teacher = public Raptor (label family differs)"),
    ("coat0", .933, F["coat0"], "single", "coatnet-GL4", "label-v2", "v4", "self-located crops, fused = index 0"),
    ("A-np", .938, F["A-np"], "single", "r50-GL4", "label-v2", "v4", "locator wp, no patch"),
    ("A'", .938, F["A'"], "single", "r50-GL4", "label-v2", "v4", "locator wp + PF OA / fracture global-only"),
    ("v4 l4copy", .939, F["v4"], "single", "r50-GL4", "label-v2", "lv2", "self-located crops"),
    ("P3v2 S-a050", .936, F["P3v2"], "single", "r50", "student-a050", "lv2", ""),
    ("CK-A", .934, F["CK-A"], "single", "convnext", "per-target", "1e", ""),
    ("J1", .930, F["J1"], "single", "joint-r50-swin3d", "label-v2", "lv2", ""),
    ("4net-DS", .932, F["4net-DS"], "single", "r50-4net", "label-v2", "4net", "teammate"),
    ("4net", .936, F["4net"], "single", "r50-4net", "label-v2", "lv2", "teammate"),
    ("B2", .920, F["B2"], "single", "swin3d", "label-v2", "B1", ""),
    ("B1", .906, F["B1"], "single", "swin3d", "label-v2", "lv2", "labels assumed label-v2"),
    ("fine", .931, F["fine"], "single", "r50", "label-v2", "lv2", "teammate"),
    ("lv3w", .933, F["lv3w"], "single", "r50", "label-v3w", "lv2", "teammate"),
    ("7a", .931, F["7a"], "single", "r50", "label-v3-rulecap", "lv2", ""),
    ("lv2", .937, F["lv2"], "single", "r50", "label-v2", "1e", "teammate fast-inference file (Kaggle-path copy differs <= 0.013)"),
    ("1e-mirror", .936, F["1e-mirror"], "single", "r50", "per-target", "1e", "mirror TTA"),
    ("v3 OS16", .933, F["v3"], "single", "r50", "per-target", "1e", ""),
    ("1e-s2", .934, F["1e-s2"], "single", "r50", "per-target", "1e", "seed 2027"),
    ("2d", .924, F["2d"], "single", "r50", "per-target", "1e", ""),
    ("3b", .934, F["3b"], "single", "r50", "D2-state", "1c", ""),
    ("3a", .926, F["3a"], "single", "r50", "D2-border", "1c", ""),
    ("1e", .936, F["1e"], "single", "r50", "per-target", "1c", ""),
    ("1c", .933, F["1c"], "single", "r50", "D2", "", ""),
    ("1b", .933, F["1b"], "single", "coatnet", "D2", "", ""),
    ("d2_224", .925, F["d2_224"], "single", "r50-224", "D2", "hpo_v1", "224 era, submission 56901397"),
    ("hpo_v1", .924, F["hpo_v1"], "single", "r50-224", "HPO-v1", "", "224 era, submission 56777224, mapping inferred"),
    ("effv2", .922, F["effv2"], "single", "effnet-224", "D2", "d2_224", "224 era"),
    ("E3", .944, mean_("lv2", "4net", "1b", "P1-cnx"), "mean(lv2, 4net, 1b, P1-cnx)", "ensemble", "mixed", "E2", ""),
    ("E2", .944, mean_("d2_224", "1b", "1e", "2c", "B2"), "mean(d2_224, 1b, 1e, 2c, B2)", "ensemble", "mixed", "L4", ""),
    ("E1", .943, mean_("d2_224", "1b", "1e", "2c", "P1-cnx"), "mean(d2_224, 1b, 1e, 2c, P1-cnx)", "ensemble", "mixed", "L4", ""),
    ("L4", .943, mean_("d2_224", "1b", "1e", "2c"), "mean(d2_224, 1b, 1e, 2c)", "ensemble", "mixed", "1e", ""),
    ("lv2+fine", .937, mean_("lv2", "fine"), "mean(lv2, fine)", "ensemble", "label-v2", "lv2", ""),
    ("1e+P1-cnx", .938, mean_("1e", "P1-cnx"), "mean(1e, P1-cnx)", "ensemble", "per-target", "1e", ""),
    ("seed-avg", .937, mean_("1e", "1e-s2"), "mean(1e, 1e-s2)", "ensemble", "per-target", "1e", ""),
]
MISSES = ["B1", "B2", "CK-A", "J1", "A'", "ra"]


# ------------------------------------------------------------------ fitting
def fit(x, y):
    b, a = np.polyfit(x, y, 1); return a, b


def loo(x, y):
    p = np.empty(len(x))
    for i in range(len(x)):
        k = np.arange(len(x)) != i; a, b = fit(x[k], y[k]); p[i] = a + b * x[i]
    return p


def spearman(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def interval(x, y, x0):
    a, b = fit(x, y); n = len(x); res = y - (a + b * x); s = np.sqrt((res ** 2).sum() / (n - 2))
    se = s * np.sqrt(1 + 1 / n + (x0 - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
    return a + b * x0, 2.03 * se                          # t(0.975, ~33 df)


def build():
    OUT.mkdir(parents=True, exist_ok=True); L = []
    pr = lambda *a: (print(*a, flush=True), L.append(" ".join(str(x) for x in a)))
    names = [s[0] for s in SUBS]; lb = np.array([s[1] for s in SUBS]); n = len(SUBS)
    P = {s[0]: load_pred(s[2]) for s in SUBS}
    pd.DataFrame([{"name": s[0], "LB": s[1], "fold0_files": s[2].replace(str(R) + "/", ""), "combination": s[3],
                   "family": s[4], "label_family": s[5], "parent": s[6], "notes": s[7]} for s in SUBS]
                 ).to_csv(OUT / "submissions.csv", index=False)
    singles = list(SM)
    X = pd.DataFrame({m: [metric(m, P[k]) for k in names] for m in singles}, index=names)
    pairs = [f"{a}+{b}" for a, b in itertools.combinations(singles, 2)]
    X = pd.concat([pd.DataFrame({"LB": lb}, index=names), X,
                   pd.DataFrame({m: 0.5 * (X[m.split("+")[0]] + X[m.split("+")[1]]) for m in pairs})], axis=1)
    X.to_csv(OUT / "metrics.csv", float_format="%.5f")
    cands = singles + pairs

    def score(m, idx):
        x, y = X[m].to_numpy()[idx], lb[idx]; p = loo(x, y); return np.sqrt(np.mean((y - p) ** 2)), p

    rows = []
    for m in cands:
        rmse, p = score(m, np.arange(n)); x = X[m].to_numpy(); a, b = fit(x, lb)
        rows.append({"cand": m, "rmse": rmse, "mae": np.mean(np.abs(lb - p)), "pearson": np.corrcoef(x, lb)[0, 1],
                     "spearman": spearman(x, lb), "a": a, "b": b, **{f"res_{k}": lb[names.index(k)] - p[names.index(k)] for k in MISSES}})
    tab = pd.DataFrame(rows).set_index("cand").sort_values("rmse")
    tab.to_csv(OUT / "candidates.csv", float_format="%.5f")

    # nested LOO: the selection over all candidates is redone inside each outer fold
    nest_p, chosen = np.empty(n), []
    for i in range(n):
        k = np.flatnonzero(np.arange(n) != i)
        best = min(cands, key=lambda m: score(m, k)[0]); chosen.append(best)
        a, b = fit(X[best].to_numpy()[k], lb[k]); nest_p[i] = a + b * X[best].to_numpy()[i]
    nest_rmse = np.sqrt(np.mean((lb - nest_p) ** 2))
    # same nested selection restricted to single metrics (fewer comparisons)
    nest_s = np.empty(n); ch_s = []
    for i in range(n):
        k = np.flatnonzero(np.arange(n) != i)
        best = min(singles, key=lambda m: score(m, k)[0]); ch_s.append(best)
        a, b = fit(X[best].to_numpy()[k], lb[k]); nest_s[i] = a + b * X[best].to_numpy()[i]
    nest_s_rmse = np.sqrt(np.mean((lb - nest_s) ** 2))
    # public-free pool (own label sources + report only): no teacher-agreement contamination for PUB-KD students
    own = [m for m in cands if not any(q.split("_")[0] in PUBLIC for q in m.split("+"))]
    nest_o = np.empty(n); ch_o = []
    for i in range(n):
        k = np.flatnonzero(np.arange(n) != i)
        best = min(own, key=lambda m: score(m, k)[0]); ch_o.append(best)
        a, b = fit(X[best].to_numpy()[k], lb[k]); nest_o[i] = a + b * X[best].to_numpy()[i]
    nest_o_rmse = np.sqrt(np.mean((lb - nest_o) ** 2))

    # subset robustness: d96-only (drop 224 era) and singles-only
    d96 = np.array([s[4] not in ("r50-224", "effnet-224") and "d2_224" not in s[3] for s in SUBS])
    sing = np.array([s[3] == "single" for s in SUBS])

    # paired: child - parent
    PAIRS = [(s[0], {"v4": "v4 l4copy", "1e": "1e", "lv2": "lv2"}.get(s[6], s[6])) for s in SUBS if s[6]]
    PAIRS = [(c, p) for c, p in PAIRS if p in names]

    def paired(m):
        dx = np.array([X.at[c, m] - X.at[p, m] for c, p in PAIRS]); dy = np.array([X.at[c, "LB"] - X.at[p, "LB"] for c, p in PAIRS])
        pred = np.empty(len(dx))
        for i in range(len(dx)):
            k = np.arange(len(dx)) != i; b = (dx[k] @ dy[k]) / (dx[k] @ dx[k]); pred[i] = b * dx[i]
        b = (dx @ dy) / (dx @ dx); big = np.abs(dy) >= .003
        return {"rmse": np.sqrt(np.mean((dy - pred) ** 2)), "b": b, "sign": np.mean(np.sign(dx[big]) == np.sign(dy[big])),
                "nbig": int(big.sum()), "dx": dx, "dy": dy, "pred": pred}

    ptab = pd.DataFrame([{"cand": m, **{k: v for k, v in paired(m).items() if k not in ("dx", "dy", "pred")}} for m in cands]
                        ).set_index("cand").sort_values("rmse")
    dy0 = np.array([X.at[c, "LB"] - X.at[p, "LB"] for c, p in PAIRS])

    # all-pairs concordance on decision-sized LB gaps (>= 0.003)
    def concord(m):
        x = X[m].to_numpy(); ok = tot = 0
        for i, j in itertools.combinations(range(n), 2):
            if abs(lb[i] - lb[j]) >= .003:
                tot += 1; ok += np.sign(x[i] - x[j]) == np.sign(lb[i] - lb[j])
        return ok / tot

    # fold-0 metric noise: paired study bootstrap (same resamples for all models)
    rng = np.random.default_rng(0); boots = [rng.integers(0, N, N) for _ in range(100)]

    def boot_sd(m, keys):
        v = np.array([[metric(m, P[k], b) for b in boots] for k in keys])
        return v.std(1).mean(), (v[0] - v[1]).std()

    # ---------------- report
    pr("LB-PRED: public-LB predictor from fold-0 predictions (fold 0 = 825 studies; Steven-v4 validation labels unchanged)")
    pr(f"dataset: {n} LB-scored submissions with out-of-sample fold-0 files ({sing.sum()} singles, {n - sing.sum()} ensembles "
       f"rebuilt as equal probability means; {d96.sum()} d96-only); LB range {lb.min():.3f}-{lb.max():.3f}, sd {lb.std():.4f}")
    pr("metric = macro ROC-AUC over findings (competition); candidates: " + str(len(singles)) + " single metrics + "
       + str(len(pairs)) + " fixed 50/50 pairs = " + str(len(cands)) + "; every fit has 2 parameters (a, b)")
    pr("findings per ruler (_12 / _c): " + "; ".join(f"{k} {len(v[0])}/{len(v[1])}" for k, v in TS.items()))
    pr("cells per ruler on fold 0 (pos / neg): " + "; ".join(f"{k} {int((m & t).sum())}/{int((m & ~t).sum())}" for k, (t, m) in RL.items()))
    a0, b0 = fit(X[BASE].to_numpy(), lb)
    pr(f"\nbaseline {BASE} (= clear6) refit on all {n}: LB = {a0:.3f} + {b0:.3f} x   (old 8-point fit: -0.687 + 1.671 x)")
    pr(f"noise floor: LB rounding sd 0.0003; public ~390 studies -> paired LB noise ~0.003 (one model ~0.002 around the shared "
       f"test draw); LB sd across these submissions {lb.std():.4f}, so a perfect fold-0 ruler still leaves RMSE ~0.002")
    hdr = f"{'candidate':30s} {'LOO RMSE':>8s} {'MAE':>6s} {'r':>6s} {'rho':>6s} {'concord':>7s} " + " ".join(f"{k:>6s}" for k in MISSES)
    pr("\nTop 15 candidates by LOO RMSE (residual = LB - LOO prediction) + baseline + key singles:"); pr(hdr)
    show = list(tab.index[:15]) + [m for m in [BASE, "steven_12", "steven_10", "lv2_12", "lv2_c", "cons_12", "cons_c", "rapc_12",
                                               "rapc_c", "repany_12", "repmod_12", "rap_12", "rap_sp", "pub_12", "pub_sp"]
                                    if m not in tab.index[:15] and m in tab.index]
    for m in show:
        r = tab.loc[m]
        pr(f"{m:30s} {r.rmse:8.4f} {r.mae:6.4f} {r.pearson:+6.2f} {r.spearman:+6.2f} {concord(m):7.2f} "
           + " ".join(f"{r[f'res_{k}']:+6.3f}" for k in MISSES) + ("   <- baseline" if m == BASE else ""))
    pr(f"\nall {len(singles)} single metrics, LOO RMSE: " + ", ".join(f"{m} {tab.at[m, 'rmse']:.4f}" for m in tab.index if m in singles))
    pr(f"\nnested LOO (selection inside each outer fold) over all {len(cands)} candidates: RMSE {nest_rmse:.4f} "
       f"(baseline plain LOO {tab.at[BASE, 'rmse']:.4f}); chosen: " + ", ".join(f"{m} x{c}" for m, c in pd.Series(chosen).value_counts().items()))
    pr(f"nested LOO over the {len(singles)} singles only: RMSE {nest_s_rmse:.4f}; chosen: "
       + ", ".join(f"{m} x{c}" for m, c in pd.Series(ch_s).value_counts().items()))
    pr(f"nested LOO over the {len(own)} public-free candidates: RMSE {nest_o_rmse:.4f}; chosen: "
       + ", ".join(f"{m} x{c}" for m, c in pd.Series(ch_o).value_counts().items()))
    pr(f"selection risk: best-of-{len(cands)} plain LOO {tab.rmse.iloc[0]:.4f} vs honest nested {nest_rmse:.4f}")
    for lab_, msk in (("d96 only", d96), ("singles only", sing)):
        pr(f"subset {lab_} (n={msk.sum()}): " + ", ".join(
            f"{m} {np.sqrt(np.mean((lb[msk] - loo(X[m].to_numpy()[msk], lb[msk])) ** 2)):.4f}" for m in [BASE] + list(tab.index[:5])))
    fam = np.array(["ensemble" if s[3] != "single" else "GL4" if "GL4" in s[4] else s[4].split("-")[0] for s in SUBS])
    lfo = lambda m: np.sqrt(np.mean([(lb[i] - np.polyval(np.polyfit(X[m].to_numpy()[fam != fam[i]], lb[fam != fam[i]], 1),
                                                         X[m].to_numpy()[i])) ** 2 for i in range(n)]))
    pr("leave-family-out RMSE (whole architecture family held out; families " + ", ".join(sorted(set(fam))) + "): "
       + ", ".join(f"{m} {lfo(m):.4f}" for m in [BASE] + list(tab.index[:3]) + [pd.Series(ch_o).value_counts().index[0]]))
    pr(f"\nPaired (Delta LB vs parent = b * Delta metric, 1 parameter, LOO over {len(PAIRS)} child-parent pairs; "
       f"'predict Delta 0' RMSE {np.sqrt(np.mean(dy0 ** 2)):.4f}):")
    pr(f"{'candidate':30s} {'RMSE':>7s} {'b':>6s} {'sign ok':>7s}  (sign over pairs with |Delta LB| >= .003)")
    for m in list(ptab.index[:10]) + ([BASE] if BASE not in ptab.index[:10] else []):
        r = ptab.loc[m]; pr(f"{m:30s} {r.rmse:7.4f} {r.b:6.2f} {r.sign:5.2f}/{int(r.nbig)}" + ("   <- baseline" if m == BASE else ""))
    pr("pairs: " + ", ".join(f"{c}<-{p}" for c, p in PAIRS))

    # recommendation: best single metric by plain LOO among those that survive nested selection most often
    rec = pd.Series(chosen).value_counts().index[0]
    pr(f"\nmost often selected in nested LOO: {rec}")
    sd_b, dsd_b = boot_sd(BASE, ["v4 l4copy", "lv2"]); sd_r, dsd_r = boot_sd(rec, ["v4 l4copy", "lv2"])
    pr(f"fold-0 bootstrap sd (100 study resamples): {BASE} {sd_b:.4f} (paired v4-lv2 {dsd_b:.4f}); {rec} {sd_r:.4f} (paired {dsd_r:.4f})")
    for m in (BASE, rec):
        pr(f"\nLOO predictions, {m}:")
        p = loo(X[m].to_numpy(), lb)
        for k in np.argsort(lb - p):
            pr(f"  {names[k]:14s} LB {lb[k]:.3f} pred {p[k]:.4f} res {lb[k] - p[k]:+.4f}  [{SUBS[k][4]}, {SUBS[k][5]}]")
    rec_o = pd.Series(ch_o).value_counts().index[0]
    fitj = {"baseline": BASE, "recommended": rec, "public_free": rec_o, "neutral": NEUTRAL, "n": n, "fits": {}}
    for m in {BASE, rec, rec_o, NEUTRAL, tab.index[0]}:
        x = X[m].to_numpy(); a, b = fit(x, lb)
        fitj["fits"][m] = {"a": a, "b": b, "x": x.tolist(), "y": lb.tolist(), "loo_rmse": float(tab.at[m, "rmse"]),
                           "paired_b": float(ptab.at[m, "b"]), "paired_rmse": float(ptab.at[m, "rmse"])}
    (OUT / "fit.json").write_text(json.dumps(fitj, indent=1))
    ar, br = fit(X[rec].to_numpy(), lb)
    pr(f"\nRECOMMENDED: m = {rec} (fixed 50/50 average of the two metrics);  LB = {ar:.3f} + {br:.3f} * m;  "
       f"paired: Delta LB = {ptab.at[rec, 'b']:.2f} * Delta m")
    an, bn = fit(X[NEUTRAL].to_numpy(), lb)
    pr(f"NEUTRAL (students of different public teachers): m = {NEUTRAL};  LB = {an:.3f} + {bn:.3f} * m;  LOO RMSE "
       f"{tab.at[NEUTRAL, 'rmse']:.4f}; paired: Delta LB = {ptab.at[NEUTRAL, 'b']:.2f} * Delta m (RMSE {ptab.at[NEUTRAL, 'rmse']:.4f})")
    pr(f"Honest error: nested LOO {nest_rmse:.4f} vs baseline {tab.at[BASE, 'rmse']:.4f} (plain LOO {tab.at[rec, 'rmse']:.4f} is "
       f"selection-optimistic); public-free pool nested {nest_o_rmse:.4f} (no honest gain without the public ruler). Model-free support: "
       f"the fold-0 bootstrap sd of the metric halves ({sd_b:.4f} -> {sd_r:.4f}): clear6 uses 6 findings and few positives, so most "
       f"of its LB residual is its own fold-0 sampling noise. Limits: n={n}, LB noise ~0.002 per model; rapc uses an in-sample public "
       f"model, so students of Raptor / the public blend score in-family on it (ra rapc_12 +0.010 over v4 for LB +0.004)")
    (OUT / "report.txt").write_text("\n".join(L) + "\n")


def predict(spec, parent=None, parent_lb=None):
    fj = json.loads((OUT / "fit.json").read_text()); S = load_pred(spec)
    Sp = load_pred(parent) if parent else None
    roles = [("recommended", fj["recommended"])] + ([("neutral", fj["neutral"])] if "neutral" in fj else [])
    for role, m in roles + [("public-free", fj["public_free"]), ("baseline", fj["baseline"])]:
        f = fj["fits"][m]; x0 = metric(m, S); x, y = np.array(f["x"]), np.array(f["y"])
        p, h = interval(x, y, x0)
        line = f"{role:11s} {m:24s} metric {x0:.4f} -> LB {p:.3f} +- {h:.3f} (95% PI; LOO RMSE {f['loo_rmse']:.4f})"
        if Sp is not None and parent_lb is not None:
            d = metric(m, S) - metric(m, Sp); dp = f["paired_b"] * d
            line += f" | paired vs parent: Delta {d:+.4f} -> LB {parent_lb + dp:.3f} +- {1.96 * f['paired_rmse']:.3f}"
        print(line)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    elif sys.argv[1] == "predict":
        a = sys.argv[2:]; par = a[a.index("--parent") + 1] if "--parent" in a else None
        plb = float(a[a.index("--parent-lb") + 1]) if "--parent-lb" in a else None
        predict(a[0], par, plb)

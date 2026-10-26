"""AM-0 (ledger, pre-registered 2026-10-17): are report-unsure cells whose per-finding attention maps resemble the
positive scenario more often positive? Fold-0 first look on existing maps (Exp 1e r50 SWA). CPU, read-only.

Map per (study, finding) from attention_side_fold0.npz: camx / camy = attention-weighted positive CAM mass per window,
summed over rows (x profile) / columns (y profile) of the 12x12 grid. Windows are grouped by series slot (centre row ->
slot). Sagittal slots keep x (anterior-posterior) and y; coronal / axial x is folded about the centre (left / right
knees mirror it). 'fine' adds thirds of each coronal / axial slot (by window order within the slot).
Scenarios from report states: positive = present moderate+ (MOD+), negative = absent (NEG); unsure = SIL / MILD / UNC /
UNSP. Scores: shape (cosine on normalised maps), strength (sqrt raw mass, Euclidean), kNN-10 (cosine, shape).
Judge: Steven-v4 ruler on unsure cells it calls confidently; within-(finding, state) pairwise AUC pooled over findings.

Judge option (2nd arg): 'steven' (default, Steven-v4 fold-0 ruler) or 'labelv2' (teammate label-v2 soft labels; user 2026-10-17:
"do not use steven labels to compare. use label v2's"). Confident = weight > 0 and (y <= 0.3 or y >= 0.7) for both.

Usage: OMP_NUM_THREADS=1 python3 -I scripts/audit_attention_map_distance.py <project root> [steven|labelv2]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"
JUDGE = sys.argv[2] if len(sys.argv) > 2 else "steven"
JP = {"steven": "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json",
      "labelv2": "teammate_labelv2/soft_labels_final_per_target_latD2.json"}[JUDGE]
OUT = D / ("attention_side_audit/attention_map_distance" + ("" if JUDGE == "steven" else f"_{JUDGE}") + ".txt"); lines = []
say = lambda s="": (print(s), lines.append(s))
z = np.load(D / "attention_side_audit/kaggle_output/attention_side_fold0.npz")
ids = list(z["ids"]); ok = z["ok"]; logit = z["logits"].astype(float)
camx, camy = z["camx"].astype(np.float32), z["camy"].astype(np.float32)          # [S, 94, 12 finding, 12 cells]
SLOT_OF = np.repeat(np.arange(5), [26, 22, 18, 12, 18])
slot = SLOT_OF[z["centre"].astype(int)]                                           # [S, 94]
ref = json.loads((D / JP).read_text())
T = ref["targets_in_order"]; m = {r["study_id"]: r for r in ref["labels"]}
y = np.array([[.5 if p is None else p for p in m[s]["probabilities"]] for s in ids])
w = np.array([m[s]["weights"] for s in ids])
st = pd.read_csv(R / "colab_bundle/raw/report_labels/report_label_extraction/report_structured_labels.csv",
                 dtype=str).fillna("").set_index("StudyInstanceUID")
key = lambda t: t.lower().replace("'", "").replace(" ", "_")


def state(s, t):
    if s not in st.index:
        return "NA"
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


S = np.array([[state(s, t) for t in T] for s in ids])
UNSURE = ("SIL", "MILD", "UNC", "UNSP")
fold = lambda v: v[..., :6][..., ::-1] + v[..., 6:]                               # |x - centre|, 6 bins


def maps(fine):
    """[S, 12, dims] map per study and finding."""
    out = []
    for i in range(len(ids)):
        per = []
        for k in range(12):
            parts = []
            for s_ in range(5):
                sel = np.where(slot[i] == s_)[0]
                groups = np.array_split(sel, 3) if (fine and s_ >= 2) else [sel]
                for g in groups:
                    x = camx[i, g, k].sum(0) if len(g) else np.zeros(12, np.float32)
                    yy = camy[i, g, k].sum(0) if len(g) else np.zeros(12, np.float32)
                    parts += [x if s_ < 2 else fold(x), yy]
            per.append(np.concatenate(parts))
        out.append(per)
    return np.array(out)


def cos(a, b):
    na = np.linalg.norm(a, axis=-1, keepdims=True); nb = np.linalg.norm(b, axis=-1, keepdims=True)
    return (a / np.maximum(na, 1e-9)) @ (b / np.maximum(nb, 1e-9)).T


def scores(M):
    """Per finding: shape / strength / kNN scores for every study (prototypes from MOD+ / NEG cells, own cell excluded)."""
    sh, sg, kn = (np.full((len(ids), 12), np.nan) for _ in range(3))
    for k in range(12):
        X = M[:, k]; Xn = X / np.maximum(X.sum(1, keepdims=True), 1e-9); Xs = np.sqrt(X)
        pos = np.where(ok & (S[:, k] == "MOD+"))[0]; neg = np.where(ok & (S[:, k] == "NEG"))[0]
        if len(pos) < 5 or len(neg) < 5:
            continue
        P, N = Xn[pos].mean(0, keepdims=True), Xn[neg].mean(0, keepdims=True)
        sh[:, k] = cos(Xn, P)[:, 0] - cos(Xn, N)[:, 0]
        Ps, Ns = Xs[pos].mean(0), Xs[neg].mean(0)
        sg[:, k] = np.linalg.norm(Xs - Ns, axis=1) - np.linalg.norm(Xs - Ps, axis=1)
        cp, cn = cos(Xn, Xn[pos]), cos(Xn, Xn[neg])
        for i in range(len(ids)):                                                 # exclude self from its own set
            a = np.sort(cp[i][pos != i])[-10:]; b = np.sort(cn[i][neg != i])[-10:]
            kn[i, k] = a.mean() - b.mean()
    return sh, sg, kn


judge_pos = (w > 0) & (y >= .7); judge_neg = (w > 0) & (y <= .3)
cell = ok[:, None] & np.isin(S, UNSURE) & (judge_pos | judge_neg)
say(f"AM-0: attention-map distance of report-unsure cells (fold 0, Exp 1e maps, judge = {JUDGE})\n")
say("unsure cells the judge calls confidently, by state (judge-pos / judge-neg):")
for s_ in UNSURE:
    say(f"  {s_:5s} {int((cell & judge_pos & (S == s_)).sum()):4d} / {int((cell & judge_neg & (S == s_)).sum()):4d}")


def ws_auc(V, ix):
    good = pairs = 0.0
    for k in range(12):
        for s_ in UNSURE:
            sel = ix[cell[ix, k] & (S[ix, k] == s_)]
            p, n = V[sel[judge_pos[sel, k]], k], V[sel[judge_neg[sel, k]], k]
            p, n = p[~np.isnan(p)], n[~np.isnan(n)]
            if len(p) and len(n):
                d = p[:, None] - n[None, :]; good += (d > 0).sum() + 0.5 * (d == 0).sum(); pairs += d.size
    return good / pairs if pairs else np.nan, pairs


def rank01(V):
    out = np.full_like(V, np.nan)
    for k in range(12):
        v = V[:, k]; g = ~np.isnan(v)
        out[g, k] = pd.Series(v[g]).rank(pct=True).to_numpy()
    return out


SC = {"logit (1e)": logit}
for fine in (False, True):
    sh, sg, kn = scores(maps(fine))
    tag = "fine" if fine else "coarse"
    SC[f"shape {tag}"], SC[f"strength {tag}"], SC[f"kNN-10 shape {tag}"] = sh, sg, kn
rl = rank01(logit)
for n in [k for k in SC if k != "logit (1e)"]:
    SC[f"rank-mean(logit, {n})"] = (rl + rank01(SC[n])) / 2

al = np.arange(len(ids)); rng = np.random.default_rng(0)
boots = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
base = {n: ws_auc(V, al) for n, V in SC.items()}
bl = np.array([ws_auc(logit, b)[0] for b in boots])
say(f"\nPRIMARY within-(finding, state) pairwise AUC on unsure cells ({int(base['logit (1e)'][1])} pairs)")
best = []
for n, V in SC.items():
    a = base[n][0]
    if n == "logit (1e)":
        say(f"  {n:44s} {a:.4f}"); continue
    d = np.array([ws_auc(V, b)[0] for b in boots]) - bl
    p = np.nanmean(d > 0)
    say(f"  {n:44s} {a:.4f}  vs logit {a - base['logit (1e)'][0]:+.4f} [{np.nanpercentile(d, 2.5):+.4f}, "
        f"{np.nanpercentile(d, 97.5):+.4f}] {p:.0%} better")
    best.append((n, a - base["logit (1e)"][0], p))

say("\nper state (logit / best map score / its rank-mean with logit):")
top = max((b for b in best if not b[0].startswith("rank")), key=lambda b: b[1])[0]
for s_ in UNSURE:
    sub = lambda V: ws_auc(np.where(S == s_, V, np.nan), al)
    say(f"  {s_:5s} {sub(logit)[0]:.3f} / {sub(SC[top])[0]:.3f} ({top}) / {sub(SC[f'rank-mean(logit, {top})'])[0]:.3f}"
        f"   [{int(sub(logit)[1])} pairs]")
say("per finding (logit / " + top + "), unsure states pooled:")
for k, t in enumerate(T):
    only = lambda V: np.where(np.arange(12)[None, :] == k, V, np.nan)
    a, n = ws_auc(only(logit), al)
    if n:
        say(f"  {t:17s} {a:.3f} / {ws_auc(only(SC[top]), al)[0]:.3f}  ({int(n)})")

say("\nscenario maps: share of positive CAM mass per slot (sagF, sagN, corF, corN, ax), MOD+ vs NEG, coarse")
M = maps(False); OFF = np.cumsum([0, 24, 24, 18, 18, 18])                         # sag x12+y12, cor/ax x6+y12
for k, t in enumerate(T):
    sl = np.stack([M[:, k, a:b].sum(1) for a, b in zip(OFF[:-1], OFF[1:])], 1); sl = sl / np.maximum(sl.sum(1, keepdims=True), 1e-9)
    pos = ok & (S[:, k] == "MOD+"); neg = ok & (S[:, k] == "NEG")
    if pos.sum() >= 5 and neg.sum() >= 5:
        say(f"  {t:17s} MOD+ {np.round(sl[pos].mean(0), 2)} (n {pos.sum()})  NEG {np.round(sl[neg].mean(0), 2)} (n {neg.sum()})")

say("\nPOST-HOC control (not a gate): is 'strength' more than total positive evidence? scalar = summed positive CAM mass")
mass = M.sum(-1); mass[:, [k for k in range(12) if np.isnan(SC["strength coarse"][0, k])]] = np.nan
for n, V in {"total mass": mass, "rank-mean(logit, total mass)": (rl + rank01(mass)) / 2}.items():
    d = np.array([ws_auc(V, b)[0] for b in boots]) - bl; a = ws_auc(V, al)[0]
    say(f"  {n:44s} {a:.4f}  vs logit {a - base['logit (1e)'][0]:+.4f} [{np.nanpercentile(d, 2.5):+.4f}, "
        f"{np.nanpercentile(d, 97.5):+.4f}] {np.nanmean(d > 0):.0%} better")

lever = [b for b in best if (not b[0].startswith("rank") and b[1] >= 0.02 and b[2] >= 0.8)
         or (b[0].startswith("rank") and b[1] >= 0.01 and b[2] >= 0.8)]
say("\nVERDICT: " + ("WORTH THE GOLD RUN: " + ", ".join(b[0] for b in lever) if lever else
                     "NO: maps do not separate unsure cells beyond the logit on fold 0"))
OUT.write_text("\n".join(lines) + "\n")

"""T-mix (ledger, pre-registered 2026-10-17): would a mixed-architecture teacher, mean(ConvNeXt CK-A, CoAtNet 1b), be a better
label source than CK-A alone? CPU, read-only, existing out-of-fold predictions (both trained on folds 1-5; fold 0 and gold-58 never
seen). Gold-58 is development evidence only.

Primary: gold-58 within-state pairwise AUC = gold positive vs gold negative of the same finding AND the same report state (so the
report label cannot separate them), pooled over findings, weak states SIL / MILD / UNC / UNSP; 2,000 study bootstraps vs CK-A.
Mixed teacher worth cross-fitting if mean(CK-A, 1b) >= CK-A + 0.02 with >= 80% bootstraps better, AND its gain exceeds the
same-family control mean(CK-A, P1-cnx), AND fold-0 mild-band macro is not below CK-A by > 0.005.
Secondary (no decision): per state, fold-0 clear6 / 10-target / mild band, Brier vs gold of a state-mean label-v2 proxy and of
its alpha 0.5 mix with each teacher.

Usage: OMP_NUM_THREADS=1 python3 -I scripts/audit_teacher_mix.py <project root>
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
exec(src.split("def per_target")[0])                       # ids, T, mask, truth, clear, st, key, load, auc, scores
OUT = C / "teacher_mix"; OUT.mkdir(exist_ok=True)
lines = []
say = lambda s="": (print(s), lines.append(s))

RUNS = {"ckA": "run_cnx_in12k", "1b": "run_continue", "cnx": "run_cnx_pertarget", "1e": "run_r50_pertarget"}
TEACH = {"CK-A": ("ckA",), "CK-A+1b": ("ckA", "1b"), "CK-A+cnx (same family)": ("ckA", "cnx"),
         "CK-A+1e (student family)": ("ckA", "1e"), "1b alone": ("1b",)}

# ---------------------------------------------------------------- gold-58 and report states (as Q-1017)
rows_g = [r for r in csv.DictReader((R / "reports_raw/train.csv").open(encoding="utf-8")) if all(r[t] in ("0", "1") for t in T)]
gids = [r["StudyInstanceUID"] for r in rows_g]
gy = np.array([[int(r[t]) for t in T] for r in rows_g]).astype(bool)


def gload(run):
    g = json.loads((C / run / "gold_swa.json").read_text()); m = {x["study_id"]: x["probabilities"] for x in g["predictions"]}
    return np.array([m[s] for s in gids])[:, [g["targets_in_order"].index(t) for t in T]]


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
GM = {k: gload(v) for k, v in RUNS.items()}
G = {n: np.mean([GM[m] for m in ms], 0) for n, ms in TEACH.items()}
F0 = {k: load(C / v / "val_swa.json") for k, v in RUNS.items()}
F = {n: np.mean([F0[m] for m in ms], 0) for n, ms in TEACH.items()}
WEAK = ("SIL", "MILD", "UNC", "UNSP")


def ws_auc(S, ix, states, ks=range(12)):
    """Pooled within-(finding, state) pairwise AUC over gold rows ix (repeats allowed); returns (auc, pairs)."""
    good = pairs = 0.0
    Sx, yx, sx = S[ix], gy[ix], gs[ix]
    for k in ks:
        for s in states:
            p, n = Sx[(sx[:, k] == s) & yx[:, k], k], Sx[(sx[:, k] == s) & ~yx[:, k], k]
            if len(p) and len(n):
                d = p[:, None] - n[None, :]
                good += (d > 0).sum() + 0.5 * (d == 0).sum(); pairs += d.size
    return (good / pairs if pairs else np.nan), pairs


allg = np.arange(len(gids))
say("T-mix: mixed-architecture teacher check (pre-registered, ledger T-mix)")
say(f"gold studies {len(gids)}; weak-state gold cells by state (pos/neg):")
for s in WEAK + ("NEG", "MOD+"):
    say(f"  {s:5s} pos {int((gy & (gs == s)).sum()):3d}  neg {int((~gy & (gs == s)).sum()):4d}")

say("\n" + "=" * 100 + "\nPRIMARY  gold-58 within-state pairwise AUC, weak states pooled (report label cannot separate these pairs)\n")
rng = np.random.default_rng(0)
boots = [rng.integers(0, len(gids), len(gids)) for _ in range(2000)]
base = {n: ws_auc(S, allg, WEAK) for n, S in G.items()}
bb = {n: np.array([ws_auc(S, b, WEAK)[0] for b in boots]) for n, S in G.items()}
for n in TEACH:
    a, npairs = base[n]
    line = f"  {n:26s} {a:.4f}  ({int(npairs)} pairs)"
    if n != "CK-A":
        d = bb[n] - bb["CK-A"]
        line += (f"   vs CK-A {a - base['CK-A'][0]:+.4f} [{np.nanpercentile(d, 2.5):+.4f}, {np.nanpercentile(d, 97.5):+.4f}] "
                 f"{np.nanmean(d > 0):.0%} better")
    say(line)

say("\nper weak state (within-state pairwise AUC, pairs):")
for s in WEAK:
    say(f"  {s:5s} " + "  ".join(f"{n.split(' ')[0]} {ws_auc(S, allg, (s,))[0]:.3f}" for n, S in G.items())
        + f"   [{int(ws_auc(G['CK-A'], allg, (s,))[1])} pairs]")
say("per finding, weak states pooled (CK-A / CK-A+1b, pairs):")
for k, t in enumerate(T):
    num = {n: ws_auc(S, allg, WEAK, (k,)) for n, S in (("a", G["CK-A"]), ("b", G["CK-A+1b"]))}
    if num["a"][1]:
        say(f"  {t:17s} {num['a'][0]:.3f} / {num['b'][0]:.3f}  ({int(num['a'][1])})")

gauc = lambda S: np.nanmean([auc(S[:, k], gy[:, k]) for k in range(12)])
say("\ngold-58 12-target macro AUC (all cells, context): " + "  ".join(f"{n.split(' ')[0]} {gauc(S):.3f}" for n, S in G.items()))

# ---------------------------------------------------------------- fold 0
say("\n" + "=" * 100 + "\nFOLD 0 (825, Steven-v4 ruler): clear6 / 10-target / mild-band macro (mild = ruler-positive, report "
    "not moderate+)\n")
mild = mask & truth & ~clear


def mild_macro(S, ix):
    out = []
    for k in range(12):
        neg = ix[mask[ix, k] & ~truth[ix, k]]; pos = ix[mild[ix, k]]
        if len(pos) >= 10:
            out.append(auc(S[np.r_[pos, neg], k], np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)]))
    return np.mean(out)


al = np.arange(len(ids))
fb = [rng.integers(0, len(ids), len(ids)) for _ in range(500)]
f0 = {n: (*scores(S, al), mild_macro(S, al)) for n, S in F.items()}
fmb = {n: np.array([mild_macro(S, b) for b in fb]) for n, S in F.items()}
for n in TEACH:
    c6, m10, mm = f0[n]
    line = f"  {n:26s} clear6 {c6:.4f}  10-target {m10:.4f}  mild {mm:.4f}"
    if n != "CK-A":
        d = fmb[n] - fmb["CK-A"]
        line += f"   mild vs CK-A {mm - f0['CK-A'][2]:+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
    say(line)

# ---------------------------------------------------------------- Brier of a label proxy on gold weak cells
lv = json.loads((C / "assets/soft_labels_final_per_target_latD2.json").read_text())
lt = lv["targets_in_order"]; vals = {}
for r in lv["labels"]:
    s = r["study_id"]
    if s in st.index:
        for k, t in enumerate(T):
            kk = lt.index(t)
            if r["weights"][kk] > 0 and r["probabilities"][kk] is not None:
                vals.setdefault((k, state(s, t)), []).append(r["probabilities"][kk])
proxy = np.array([[np.mean(vals.get((k, gs[i, k]), [np.nan])) for k in range(12)] for i in range(len(gids))])
wk = np.isin(gs, WEAK) & ~np.isnan(proxy)
say("\n" + "=" * 100 + "\nSECONDARY  Brier vs gold on weak cells: label-v2 state-mean proxy y, and 0.5*y + 0.5*teacher "
    f"({int(wk.sum())} cells; lower is better)\n")
say(f"  proxy label alone           {np.mean((proxy[wk] - gy[wk]) ** 2):.4f}")
for n, S in G.items():
    say(f"  mix with {n:26s} {np.mean((0.5 * proxy[wk] + 0.5 * S[wk] - gy[wk]) ** 2):.4f}   (teacher alone "
        f"{np.mean((S[wk] - gy[wk]) ** 2):.4f})")

# ---------------------------------------------------------------- verdict
g_mix = base["CK-A+1b"][0] - base["CK-A"][0]; g_ctl = base["CK-A+cnx (same family)"][0] - base["CK-A"][0]
p_mix = np.nanmean((bb["CK-A+1b"] - bb["CK-A"]) > 0); d_mild = f0["CK-A+1b"][2] - f0["CK-A"][2]
ok = g_mix >= 0.02 and p_mix >= 0.80 and g_mix > g_ctl and d_mild >= -0.005
verdict = ("WORTH CROSS-FITTING" if ok else "INCONCLUSIVE" if g_mix >= 0.02 and g_mix > g_ctl and d_mild >= -0.005
           else "NO (CoAtNet stays an ensemble member)")
say("\n" + "=" * 100)
say(f"VERDICT: {verdict}  (gain {g_mix:+.4f}, {p_mix:.0%} bootstraps better; same-family control {g_ctl:+.4f}; "
    f"fold-0 mild {d_mild:+.4f})")
(OUT / "audit.txt").write_text("\n".join(lines) + "\n")

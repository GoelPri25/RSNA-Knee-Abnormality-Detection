"""PUB-KD-R students vs v4 l4copy on fold 0, per finding x band x branch, and rank shifts by report state.
Steven-v4 confident cells (clear6 ruler); report states from report_structured_labels.csv (local analysis only).
Usage: python3 -I scripts/analyze_pubkd_r_findings.py <project root>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
R = Path(sys.argv[1])
src = (R / "repository/scripts/audit_clear_band_lb.py").read_text()
sys.argv = [sys.argv[0], str(R)]
exec(src[src.index("import json"):src.index("run = lambda")])
S = R / "repository/data/coatnet_d96_fold0/pubkd/students_r"; V = R / "repository/data/coatnet_d96_fold0/run_r50_gl4_mlx/v4_l4copy_s2026"
out = []
pr = lambda *a: (print(*a), out.append(" ".join(str(x) for x in a)))
M = {"v4": {b: load(V / f"val_swa_{b}.json") for b in ("global", "local", "fused")}}
for l, n in (("ra", "pure"), ("rb", "mix"), ("rc", "pure27"), ("rd", "mix27")):
    M[n] = {b: load(S / f"pubkd_kd_{l}/val_swa_{b}.json") for b in ("global", "local", "fused")}
neg = mask & ~truth; cpos = mask & truth & clear; mpos = mask & truth & ~clear
def band_auc(X, k, pos):
    sel = neg[:, k] | pos[:, k]
    return auc(X[sel, k], truth[sel, k]) if pos[:, k].sum() >= 10 else np.nan
pr("## per finding AUC (Steven-v4 confident), fused; clear = report moderate+ positives vs negatives, mild = other positives vs negatives")
pr(f"{'finding':17s} {'n+clr':>5s} {'n+mild':>6s} | {'v4 clr':>6s} {'pure':>6s} {'mix':>6s} | {'v4 mild':>7s} {'pure':>6s} {'mix':>6s} | crop-branch mild: {'v4':>6s} {'pure':>6s} {'mix':>6s}")
for k, t in enumerate(T):
    r = [band_auc(M[m]["fused"], k, cpos) for m in ("v4", "pure", "mix")] + [band_auc(M[m]["fused"], k, mpos) for m in ("v4", "pure", "mix")] + \
        [band_auc(M[m]["local"], k, mpos) for m in ("v4", "pure", "mix")]
    pr(f"{t:17s} {cpos[:, k].sum():5d} {mpos[:, k].sum():6d} | " + " ".join(f"{x:6.3f}" for x in r[:3]) + " | " + f"{r[3]:7.3f} " +
       " ".join(f"{x:6.3f}" for x in r[4:6]) + " | " + " ".join(f"{x:6.3f}" for x in r[6:]))
for m in ("v4", "pure", "mix", "pure27", "mix27"):
    for b in ("global", "local", "fused"):
        cl = np.nanmean([band_auc(M[m][b], k, cpos) for k in range(12)]); mi = np.nanmean([band_auc(M[m][b], k, mpos) for k in range(12)])
        pr(f"  macro {m:6s} {b:6s}: clear-band {cl:.4f}  mild-band {mi:.4f}")
# rank shifts by report state
key = lambda t: t.lower().replace("'", "").replace(" ", "_")
state = np.full((len(ids), 12), "missing", dtype=object)
for i, s in enumerate(ids):
    if s in st.index:
        for k, t in enumerate(T):
            stt = st.at[s, f"{key(t)}__status"]; sev = st.at[s, f"{key(t)}__severity"]
            state[i, k] = (stt or "blank") + ("" if stt != "present" else ("/mod+" if sev in ("moderate", "moderate_to_severe", "severe") else f"/{sev or 'nosev'}"))
pct = lambda X: np.column_stack([pd.Series(X[:, k]).rank(pct=True).to_numpy() for k in range(12)])
P = {m: pct(M[m]["fused"]) for m in M}
d = (P["pure"] + P["pure27"]) / 2 - P["v4"]; dm = (P["mix"] + P["mix27"]) / 2 - P["v4"]
pr("\n## mean rank-percentile shift vs v4 (students' 2-seed mean, fused) by report state, all 12 findings pooled")
pr(f"{'report state':28s} {'cells':>6s} {'Steven+':>7s} {'pure shift':>10s} {'mix shift':>9s}")
for s_ in sorted(set(state.ravel()), key=lambda x: -(state == x).sum()):
    c = state == s_
    if c.sum() < 40: continue
    yy = truth[c & mask].mean() if (c & mask).sum() else np.nan
    pr(f"{s_:28s} {c.sum():6d} {yy:7.2f} {d[c].mean():+10.3f} {dm[c].mean():+9.3f}")
pr("\n## same, per finding: shift of report-'present' (any severity) minus report-'absent' cells (positive = students separate them more)")
for k, t in enumerate(T):
    p_ = np.char.startswith(state[:, k].astype(str), "present"); a_ = state[:, k] == "absent"; b_ = ~p_ & ~a_
    pr(f"  {t:17s} present {p_.sum():4d} absent {a_.sum():4d} other {b_.sum():4d} | pure: present {d[p_, k].mean():+.3f} absent {d[a_, k].mean():+.3f} silent/other {d[b_, k].mean():+.3f}")
(S / "findings_by_band_state.txt").write_text("\n".join(out) + "\n")

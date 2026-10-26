"""SER-1 (ledger): CPU follow-ups to the SER-0 series ablation.
(1) Baker's vs effusion: is v4's Baker's score mostly an effusion score? Fold-0 label-v2 confident cells, v4 l4copy SWA
    fused s2026 / s2027 (and their rank mean): AUC_B, AUC_E->B, Baker's AUC within effusion-positive / -negative studies,
    top false positives / bottom false negatives. Pre-registered: "effusion proxy" if, on both seeds,
    AUC_E->B >= AUC_B - 0.03 or AUC_B | effusion+ <= AUC_B - 0.05 -> axial Baker's lane; otherwise no lane.
(2) Crop planes: per cropped finding, the MS1-table crop attention share per slot (fold 0), slot contrast from the
    manifest (fluid flag: 1 = fluid-bright / fat-suppressed, 0 = non-FS).
(3) SER-0 drops by path: global vs local (crop) for fracture, OA x 3, Baker's under no_nonfs and no_axial.
Usage: python3 -I audit_ser1.py <repository root>
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "data/coatnet_d96_fold0"; SER = R / "kaggle/submission_kernels/v4_series_ablation_fold0/output"
L = []


def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)


lv = json.load(open(D / "assets/soft_labels_final_per_target_latD2.json")); T = lv["targets_in_order"]
lab = {r["study_id"]: np.array([.5 if p is None else p for p in r["probabilities"]]) for r in lv["labels"]}


def preds(p):
    d = json.load(open(p)); tg = d.get("targets_in_order", T)
    return {r["study_id"]: np.array(r["probabilities"])[[tg.index(t) for t in T]] for r in d["predictions"]}


def auc(s, t):
    t = np.asarray(t, bool); n1 = t.sum(); n0 = len(t) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


P = {s: preds(D / f"run_r50_gl4_mlx/v4_l4copy_{s}/val_swa_fused.json") for s in ("s2026", "s2027")}
ids = sorted(set(P["s2026"]) & set(P["s2027"]) & set(lab)); n = len(ids)
Y = np.array([lab[s] for s in ids]); M = (Y <= .3) | (Y >= .7); TT = Y >= .7
S = {k: np.array([v[s] for s in ids]) for k, v in P.items()}
rk = lambda A: pd.DataFrame(A).rank(pct=True).to_numpy()
S["rank-mean"] = (rk(S["s2026"]) + rk(S["s2027"])) / 2
b, e = T.index("Baker's"), T.index("Effusion")
rng = np.random.default_rng(2026); BS = [rng.integers(0, n, n) for _ in range(1000)]

pr(f"## (1) Baker's vs effusion ({n} fold-0 studies)")
mb = M[:, b]; pos_b = TT[:, b]
pr(f"Baker's confident cells {mb.sum()} (positives {(mb & pos_b).sum()}); effusion label among Baker's positives: "
   f"positive {np.mean(TT[mb & pos_b, e]):.0%}, among Baker's confident negatives {np.mean(TT[mb & ~pos_b, e]):.0%}")
flags = {}
for k, A in S.items():
    aB = auc(A[mb, b], pos_b[mb]); aEB = auc(A[mb, e], pos_b[mb])
    ep = mb & TT[:, e] & M[:, e]; en = mb & ~TT[:, e] & M[:, e]
    aBp, aBn = auc(A[ep, b], pos_b[ep]), auc(A[en, b], pos_b[en])
    aEBp = auc(A[ep, e], pos_b[ep])
    bd = np.array([auc(A[i][mb[i], b], pos_b[i][mb[i]]) - auc(A[i][mb[i], e], pos_b[i][mb[i]]) for i in BS])
    proxy = aEB >= aB - 0.03 or aBp <= aB - 0.05
    flags[k] = proxy
    pr(f"{k:9s} AUC_B {aB:.4f} | AUC_E->B {aEB:.4f} (B - E {aB - aEB:+.4f} [{np.percentile(bd, 2.5):+.4f}, {np.percentile(bd, 97.5):+.4f}]) | "
       f"AUC_B within effusion+ {aBp:.4f} (n+ {pos_b[ep].sum()} / n {ep.sum()}), effusion- {aBn:.4f} (n+ {pos_b[en].sum()} / n {en.sum()}) | "
       f"AUC_E->B within effusion+ {aEBp:.4f} -> {'PROXY' if proxy else 'specific'}")
sp = lambda u, v: float(np.corrcoef(pd.Series(u).rank(), pd.Series(v).rank())[0, 1])   # Spearman without scipy
A = S["rank-mean"]; r_b = sp(A[mb, b], A[mb, e])
pr(f"Spearman(Baker's score, effusion score) on Baker's confident cells {r_b:+.2f}; "
   f"among Baker's negatives {sp(A[mb & ~pos_b, b], A[mb & ~pos_b, e]):+.2f}")
neg = np.where(mb & ~pos_b)[0]; top = neg[np.argsort(-A[neg, b])[:30]]
pos = np.where(mb & pos_b)[0]; bot = pos[np.argsort(A[pos, b])[:min(15, len(pos))]]
pr(f"top 30 Baker's false positives: effusion label positive {np.mean(TT[top, e]):.0%} (all Baker's negatives {np.mean(TT[neg, e]):.0%}), "
   f"mean effusion score pct {A[top, e].mean():.2f} (negatives {A[neg, e].mean():.2f}); mean label-v2 Baker's y {Y[top, b].mean():.3f}")
pr(f"bottom {len(bot)} Baker's positives (missed): effusion label positive {np.mean(TT[bot, e]):.0%} (all positives {np.mean(TT[pos, e]):.0%}), "
   f"mean effusion score pct {A[bot, e].mean():.2f} (positives {A[pos, e].mean():.2f})")
pr("per-target 12t AUC (rank-mean), for headroom: " + " ".join(f"{t[:8]} {auc(A[M[:, k], k], TT[M[:, k], k]):.3f}" for k, t in enumerate(T)))
dec = flags["s2026"] and flags["s2027"]
pr(f"DECISION (1): {'Baker s = effusion proxy on both seeds -> design axial Baker s lane' if dec else 'Baker s score is not an effusion proxy (pre-registered thresholds) -> no axial lane'}")

pr("\n## (2) crop planes (MS1 table, fold-0 studies): attention-weighted share per slot; slot contrast from the manifest")
man = {}
for f in sorted(glob.glob(str(D / "manifests/manifest_*.jsonl"))):
    for l in open(f):
        m = json.loads(l); man[m["study_id"]] = m
SIZES = [26, 22, 18, 12, 18]; START = np.cumsum([0] + SIZES); slot_of = np.repeat(np.arange(5), SIZES)
NAMES = ["SagF", "SagNF", "CorF", "CorNF", "Ax"]


def centres(mask, k=94):
    valid = np.where(mask)[0]
    if len(valid) < 3:
        valid = np.arange(3)
    lo, hi = int(valid.min()), int(valid.max())
    cs = np.arange(lo + 1, hi) if hi - lo >= 2 else np.array([max(1, min((lo + hi) // 2, len(mask) - 2))])
    return np.clip(cs[np.linspace(0, len(cs) - 1, k).round().astype(int)], 1, 94)


CROPT = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Baker's", "Contusion", "Fracture"]
fold0 = set(ids); share = {t: np.zeros(5) for t in CROPT}; fs_share = {t: [0.0, 0.0] for t in CROPT}; cnt = 0
for cp in sorted(glob.glob(str(D / "run_sig_lv2/crop_probe/cp_*.npz"))):
    z = np.load(cp)
    for s, cen, at in zip(z["ids"], z["centres"], z["a_top"]):
        s = str(s)
        if s not in fold0 or s not in man:
            continue
        cnt += 1; c = centres(np.array(man[s]["mask"], bool)); sl = man[s]["slots"]
        for t in CROPT:
            k = T.index(t); w = at[k] / max(at[k].sum(), 1e-9)
            for j in range(4):
                q = slot_of[c[int(cen[k, j, 0])]]; share[t][q] += w[j]
                fl = sl[q].get("fluid") if q < len(sl) else None
                fs_share[t][0 if str(fl) == "1" else 1] += w[j]
pr(f"{cnt} fold-0 studies in the crop table")
pr(f"{'finding':17s} " + " ".join(f"{x:>6s}" for x in NAMES) + "   fluid-bright (FS) / non-FS by actual series")
for t in CROPT:
    v = share[t] / share[t].sum(); f = np.array(fs_share[t]) / sum(fs_share[t])
    pr(f"{t:17s} " + " ".join(f"{x:6.1%}" for x in v) + f"   {f[0]:.1%} / {f[1]:.1%}")

pr("\n## (3) SER-0 drops by path (rank-free AUC on label-v2 confident cells; kernel self-crop pipeline)")
for path in ("fused", "global", "local"):
    base = preds(SER / f"ser0_all_{path}.json")
    for cfg in ("no_nonfs", "no_axial"):
        q = preds(SER / f"ser0_{cfg}_{path}.json"); keep = [i for i, s in enumerate(ids) if s in q and s in base]
        out = []
        for t in ("Fracture", "Medial OA", "Lateral OA", "PF OA", "Baker's"):
            k = T.index(t); sel = [i for i in keep if M[i, k]]
            a0 = auc(np.array([base[ids[i]][k] for i in sel]), TT[sel, k]); a1 = auc(np.array([q[ids[i]][k] for i in sel]), TT[sel, k])
            out.append(f"{t[:9]} {a1 - a0:+.4f}")
        pr(f"{path:6s} {cfg:9s} " + " | ".join(out))
(D / "ser0").mkdir(exist_ok=True); (D / "ser0/audit_ser1.txt").write_text("\n".join(L) + "\n")

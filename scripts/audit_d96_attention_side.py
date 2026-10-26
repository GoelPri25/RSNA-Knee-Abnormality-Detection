"""Attention-placement audit of Exp 1e on fold 0 (Kaggle output of kaggle/submission_kernels/d96_r50_attention_side_audit).

Question: when the model misses a positive whose anatomy fixes the side of the knee (MCL, medial meniscus, medial OA:
medial; lateral meniscus, lateral OA: lateral), was it looking on the wrong side? If misses look on the right side as
often as hits, steering attention with location priors would not help; if they look elsewhere, it might.

Side of the evidence, from the exact CAM (positive part, already scaled by the window's attention):
  sagittal windows (slots 0-1): the centre slice's x position across the slot's picks (0 = medial edge, 1 = lateral);
    medial < 0.4, lateral > 0.6, the central band (notch / cruciates) counted as neither;
  coronal / axial windows (slots 2-4): in-plane columns (or rows) mapped to patient x via ImageOrientationPatient;
    the half of the field on the medial side vs the lateral side (needs |row or column cosine x| > 0.7).
Knee side: Laterality / ImageLaterality tags, else the sagittal slot's mean x if >= 20 mm off centre (as d96_positions).
correct-side share (CSS) = evidence on the anatomically correct side / evidence on either side.
Hit / miss: a positive's rank among the negatives (fraction of confident negatives scored lower); miss = bottom third.
Correlational only: a model that sees the lesion will also look at it, so a low CSS on misses does not prove that
moving attention would find the lesion.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_d96_attention_side.py /proj
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

R = Path(sys.argv[1]); D = R / "repository/data"
O = D / "attention_side_audit/kaggle_output"
z = np.load(O / "attention_side_fold0.npz")
lats = json.loads((O / "laterality.json").read_text())
ids = list(z["ids"]); ok = z["ok"]
att = z["att"].astype(np.float32); cen = z["centre"].astype(int)
camx = z["camx"].astype(np.float32); camy = z["camy"].astype(np.float32)
ipp, iop, logits = z["ipp"], z["iop"], z["logits"]
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; m = {r["study_id"]: r for r in ref["labels"]}
y = np.array([[.5 if p is None else p for p in m[s]["probabilities"]] for s in ids])
w = np.array([m[s]["weights"] for s in ids])
POS, NEG = (w > 0) & (y >= .7) & ok[:, None], (w > 0) & (y <= .3) & ok[:, None]
SLOT_OF = np.repeat(np.arange(5), [26, 22, 18, 12, 18]); STARTS = [0, 26, 48, 66, 78]; KS = [26, 22, 18, 12, 18]
SIDE_T = {"MCL": "M", "Medial Meniscus": "M", "Medial OA": "M", "Lateral Meniscus": "L", "Lateral OA": "L"}


def knee_side(n):
    for lat in lats[n]:
        for key in ("Laterality", "ImageLaterality"):
            v = str((lat or {}).get(key, "")).strip().upper()[:1]
            if v in ("R", "L"):
                return v, "tag"
    x = ipp[n, 0:26, 0]
    x = x[np.isfinite(x)]
    if len(x) >= 5 and abs(x.mean()) >= 20:
        return ("R" if x.mean() < 0 else "L"), "geometry"
    return None, None


def evidence(n, t, side):
    """-> (medial, lateral) positive CAM mass by plane group: {'sag': (m, l), 'cor': (m, l), 'ax': (m, l)}; and the
    sagittal attention split (m, l). Medial means toward the knee's medial side."""
    med_sign = 1.0 if side == "R" else -1.0          # LPS: +x = patient left; right knee's medial side is +x
    out = {"sag": [0.0, 0.0], "cor": [0.0, 0.0], "ax": [0.0, 0.0]}; att_sag = [0.0, 0.0]
    for slot in (0, 1):
        xs = ipp[n, STARTS[slot]:STARTS[slot] + KS[slot], 0]
        if not np.isfinite(xs).all() or xs.max() - xs.min() < 5:
            continue
        for k in np.where(SLOT_OF[cen[n]] == slot)[0]:
            x = ipp[n, cen[n, k], 0]
            p = (x - xs.min()) / (xs.max() - xs.min())
            p = 1 - p if med_sign > 0 else p           # 0 = medial edge
            mass = camx[n, k, t].sum()
            if p < 0.4:
                out["sag"][0] += mass; att_sag[0] += att[n, k, t]
            elif p > 0.6:
                out["sag"][1] += mass; att_sag[1] += att[n, k, t]
    cols = np.arange(12) - 5.5
    for slot, grp in ((2, "cor"), (3, "cor"), (4, "ax")):
        o = iop[n, slot]
        if not np.isfinite(o).all():
            continue
        if abs(o[0]) > 0.7:
            prof, cos = camx, o[0]                      # columns run along the row cosine
        elif abs(o[3]) > 0.7:
            prof, cos = camy, o[3]                      # rows run along the column cosine
        else:
            continue
        medial_cells = np.sign(cols * cos) == med_sign
        for k in np.where(SLOT_OF[cen[n]] == slot)[0]:
            v = prof[n, k, t]
            out[grp][0] += v[medial_cells].sum(); out[grp][1] += v[~medial_cells].sum()
    return out, att_sag


def css(pair, want):
    tot = pair[0] + pair[1]
    if tot <= 0:
        return np.nan
    return (pair[0] if want == "M" else pair[1]) / tot


sides = [knee_side(n) for n in range(len(ids))]
src = [s[1] for s in sides]
print(f"{ok.sum()} studies; knee side from tags {src.count('tag')}, from geometry {src.count('geometry')}, "
      f"unknown {src.count(None)}")
print("CSS = share of the finding's positive CAM evidence on the anatomically correct side (0.5 = no preference).")
print("Miss / hit = bottom / top third of positives by rank among confident negatives.\n")
rows = []
for name, want in SIDE_T.items():
    t = T.index(name)
    neg_logit = np.sort(logits[NEG[:, t], t])
    rec = []
    for n in range(len(ids)):
        if not (POS[n, t] or NEG[n, t]) or sides[n][0] is None:
            continue
        ev, a_sag = evidence(n, t, sides[n][0])
        allm = [sum(ev[g][i] for g in ev) for i in (0, 1)]
        rank = np.searchsorted(neg_logit, logits[n, t]) / max(len(neg_logit), 1)
        sag_share = sum(ev["sag"]) / max(sum(sum(v) for v in ev.values()), 1e-9)
        rec.append((POS[n, t], rank, css(allm, want), css(ev["sag"], want), css(ev["cor"], want),
                    css(ev["ax"], want), css(a_sag, want), sag_share))
    r = np.array(rec, float)
    pos, neg = r[r[:, 0] == 1], r[r[:, 0] == 0]
    q1, q2 = np.quantile(pos[:, 1], [1 / 3, 2 / 3])
    miss, hit = pos[pos[:, 1] <= q1], pos[pos[:, 1] >= q2]
    rho = spearmanr(pos[:, 1], pos[:, 2], nan_policy="omit")[0]
    print(f"{name} ({want}): positives {len(pos)}, negatives {len(neg)}; positive AUC {pos[:, 1].mean():.3f}")
    print(f"   {'':18s}{'all':>7s}{'sag':>7s}{'cor':>7s}{'ax':>7s}{'sagAtt':>8s}{'sag share':>11s}")
    for lab, g in (("negatives", neg), ("positives", pos), ("  hits (top 1/3)", hit), ("  misses (bot 1/3)", miss)):
        print(f"   {lab:18s}" + "".join(f"{np.nanmean(g[:, i]):7.3f}" for i in (2, 3, 4, 5)) +
              f"{np.nanmean(g[:, 6]):8.3f}{np.nanmean(g[:, 7]):11.3f}")
    # bootstrap the hit - miss CSS gap
    rng = np.random.default_rng(0); gaps = []
    for _ in range(2000):
        hb, mb = hit[rng.integers(0, len(hit), len(hit))], miss[rng.integers(0, len(miss), len(miss))]
        gaps.append(np.nanmean(hb[:, 2]) - np.nanmean(mb[:, 2]))
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    print(f"   hit - miss CSS {np.nanmean(hit[:, 2]) - np.nanmean(miss[:, 2]):+.3f} [{lo:+.3f}, {hi:+.3f}]; "
          f"Spearman(rank, CSS) among positives {rho:+.3f}")
    # would a perfect side would have helped: AUC of positives with CSS below/above median
    med = np.nanmedian(pos[:, 2])
    print(f"   positive AUC when CSS < median {pos[pos[:, 2] < med, 1].mean():.3f} vs >= median "
          f"{pos[pos[:, 2] >= med, 1].mean():.3f}\n")

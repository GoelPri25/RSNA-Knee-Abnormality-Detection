"""LOC-0 (ledger, pre-registered 2026-10-17): can Exp 1e's CAM peaks serve as pseudo-coordinates for a localise -> crop
pipeline (RSNA 2024 lumbar 1st place, with CAM instead of annotated coordinates)? CPU, read-only, fold 0.

Per finding and label-v2 confident positive: the top-evidence window (largest positive CAM mass) in a slot (sagittal
fluid; coronal fluid) -> peak = CAM centroid of that window in the image plane (camx along the IOP row cosine, camy along
the column cosine; 11.7 mm cells, offsets from the image centre) + the centre slice's offset from the slot's mid slice
along the normal. Expressed in a knee frame: medial +, anterior +, superior + (LPS; knee side from tags else sagittal
geometry, as the 2-side audit). Image centre = crop centre, so knee position inside the field adds spread (conservative).
Q1 / Q2 gates as in the ledger. Usage: OMP_NUM_THREADS=1 python3 -I scripts/audit_cam_localisation.py <project root>
LOC-0-MS1: ... <project root> ms1 [map]  -> MS1 token maps (32 x 32 grid, 4.4 mm cells) from run_sig_lv2/token_maps; "map" scores the
finding-agnostic MAP attention instead of the per-finding gradient x activation maps. Geometry from the r50 file (same studies, centres).
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

R = Path(sys.argv[1]); D = R / "repository/data"; O = D / "attention_side_audit/kaggle_output"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "r50"; USE_MAP = len(sys.argv) > 3 and sys.argv[3] == "map"
OUT = D / ("attention_side_audit/cam_localisation" + ("" if MODEL == "r50" else f"_{MODEL}" + ("_map" if USE_MAP else "")) + ".txt"); lines = []
say = lambda s="": (print(s), lines.append(s))
z = np.load(O / "attention_side_fold0.npz"); lats = json.loads((O / "laterality.json").read_text())
ids = list(z["ids"]); ok = z["ok"]; cen = z["centre"].astype(int)
camx, camy = z["camx"].astype(np.float32), z["camy"].astype(np.float32)
G = 12; R50 = (camx, camy, 12)
if MODEL == "ms1":
    zm = np.load(D / "coatnet_d96_fold0/run_sig_lv2/token_maps/ms1_token_maps_fold0.npz"); G = 32
    pos_ = {s: i for i, s in enumerate(zm["ids"])}; ix_ = np.array([pos_[s] for s in ids])
    assert (zm["centre"][ix_].astype(int) == cen).all(), "MS1 centres differ from the r50 file"
    if USE_MAP:                                                           # finding-agnostic: same map for all 12
        camx = np.repeat(zm["mapx"][ix_].astype(np.float32)[:, :, None], 12, 2); camy = np.repeat(zm["mapy"][ix_].astype(np.float32)[:, :, None], 12, 2)
    else:
        camx, camy = zm["camx"][ix_].astype(np.float32), zm["camy"][ix_].astype(np.float32)
ipp, iop = z["ipp"].astype(float), z["iop"].astype(float)
CELL = 140.0 / G
SLOT_OF = np.repeat(np.arange(5), [26, 22, 18, 12, 18]); STARTS = [0, 26, 48, 66, 78]; KS = [26, 22, 18, 12, 18]
lv = json.loads((D / "teammate_labelv2/soft_labels_final_per_target_latD2.json").read_text())
T = lv["targets_in_order"]; m = {r["study_id"]: r for r in lv["labels"]}
y = np.array([[.5 if p is None else p for p in m[s]["probabilities"]] for s in ids]); w = np.array([m[s]["weights"] for s in ids])
POS = (w > 0) & (y >= .7) & ok[:, None]; NEG = (w > 0) & (y <= .3) & ok[:, None]
SIDE_T = {"MCL": 1, "Medial Meniscus": 1, "Medial OA": 1, "Lateral Meniscus": -1, "Lateral OA": -1}   # +1 medial


def knee_side(n):
    for lat in lats[n]:
        for key in ("Laterality", "ImageLaterality"):
            v = str((lat or {}).get(key, "")).strip().upper()[:1]
            if v in ("R", "L"):
                return v
    x = ipp[n, 0:26, 0]; x = x[np.isfinite(x)]
    if len(x) >= 5 and abs(x.mean()) >= 20:
        return "R" if x.mean() < 0 else "L"
    return None


SIDE = [knee_side(n) for n in range(len(ids))]


def central(v, q=0.8):
    """width (cells) of the smallest contiguous interval holding q of the profile's mass"""
    tot = v.sum()
    if tot <= 0:
        return np.nan
    best = len(v)
    for a in range(len(v)):
        c = np.cumsum(v[a:])
        hit = np.where(c >= q * tot)[0]
        if len(hit):
            best = min(best, hit[0] + 1)
    return best


def peak(n, t, slot, src=None):
    """-> (knee-frame offset [medial, anterior, superior] mm, box side mm) of the top window in slot, or None"""
    cx_, cy_, g_ = src or (camx, camy, G); cell = 140.0 / g_
    if SIDE[n] is None:
        return None
    o = iop[n, slot]; rows = np.arange(STARTS[slot], STARTS[slot] + KS[slot])
    if not np.isfinite(o).all() or not np.isfinite(ipp[n, rows]).all():
        return None
    ks = np.where(SLOT_OF[cen[n]] == slot)[0]
    if not len(ks):
        return None
    k = ks[np.argmax(cx_[n, ks, t].sum(1))]
    vx, vy = cx_[n, k, t], cy_[n, k, t]
    if vx.sum() <= 0:
        return None
    u = (np.arange(g_) * vx).sum() / vx.sum() - (g_ - 1) / 2; v = (np.arange(g_) * vy).sum() / vy.sum() - (g_ - 1) / 2
    rc, cc = o[:3], o[3:]; nrm = np.cross(rc, cc)
    pos = ipp[n, rows] @ nrm; s = ipp[n, cen[n, k]] @ nrm - (pos.min() + pos.max()) / 2
    off = u * cell * rc + v * cell * cc + s * nrm                       # LPS offset from slab / image centre
    med = 1.0 if SIDE[n] == "R" else -1.0                               # LPS +x = patient left = right knee's medial
    kf = np.array([med * off[0], -off[1], off[2]])
    return kf, max(central(vx), central(vy)) * cell


AX = ("medial", "anterior", "superior")
INPLANE = {0: (1, 2), 2: (0, 2)}                                        # sagittal: AP, SI; coronal: ML, SI
res = {}
for slot, name in ((0, "sagittal fluid"), (2, "coronal fluid")):
    say(f"===== slot {name}: top-evidence window per study (positives = label-v2 confident) =====")
    say(f"{'finding':17s} {'n+':>4s}  median peak (med, ant, sup) mm   spread in-plane   neg median            "
        f"pos-neg dist  80% box (median)  side-correct")
    meds = {}
    for k, t in enumerate(T):
        P = [peak(n, k, slot) for n in np.where(POS[:, k])[0]]; P = [p for p in P if p is not None]
        N = [peak(n, k, slot) for n in np.where(NEG[:, k])[0]]; N = [p for p in N if p is not None]
        if len(P) < 10:
            say(f"{t:17s} {len(P):4d}  (too few)"); continue
        X = np.array([p[0] for p in P]); B = np.array([p[1] for p in P]); Xn = np.array([p[0] for p in N])
        md = np.median(X, 0); mdn = np.median(Xn, 0); meds[t] = md
        ip = list(INPLANE[slot])
        spread = np.median(np.linalg.norm(X[:, ip] - md[ip], axis=1))
        sc = ""
        if t in SIDE_T and slot == 0:
            sc = f"{np.mean(np.sign(X[:, 0]) == SIDE_T[t]):.0%}"
        res[(slot, t)] = dict(spread=spread, box=np.median(B), side=None if not sc else np.mean(np.sign(X[:, 0]) == SIDE_T[t]))
        say(f"{t:17s} {len(P):4d}  ({md[0]:+6.1f}, {md[1]:+6.1f}, {md[2]:+6.1f})        {spread:5.1f}          "
            f"({mdn[0]:+6.1f}, {mdn[1]:+6.1f}, {mdn[2]:+6.1f})  {np.linalg.norm(md - mdn):5.1f}        "
            f"{np.median(B):5.1f}             {sc}")
    d = [np.linalg.norm(meds[a] - meds[b]) for a, b in combinations(meds, 2)]
    say(f"  finding medians: pairwise distance median {np.median(d):.1f} mm (min {min(d):.1f}, max {max(d):.1f})")
    res[(slot, "pairwise")] = np.median(d)
    fixed = []
    for k, t in enumerate(T):
        if t in meds:
            P = [peak(n, k, slot) for n in np.where(POS[:, k])[0]]; X = np.array([p[0] for p in P if p is not None])
            ip = list(INPLANE[slot]); half = np.percentile(np.abs(X[:, ip] - meds[t][ip]), 80, axis=0)
            fixed.append(f"{t} {2 * half.max() + res[(slot, t)]['box']:.0f}")
    say("  fixed box per finding (mm a side: 80% of positive peaks inside + median window box): " + ", ".join(fixed) + "\n")

if MODEL != "r50":
    say("===== AGREEMENT with the r50 (LOC-0) peaks, same study + finding, label-v2 confident positives, sagittal fluid =====")
    for k, t in enumerate(T):
        d = []
        for n in np.where(POS[:, k])[0]:
            a, b = peak(n, k, 0), peak(n, k, 0, R50)
            if a is not None and b is not None:
                d.append(np.linalg.norm(a[0][[1, 2]] - b[0][[1, 2]]))
        say(f"{t:17s} n {len(d):4d}  in-plane distance median {np.median(d):5.1f} mm, <= 15 mm {np.mean(np.array(d) <= 15):.0%}" if d else f"{t:17s} none")
    say()

say("===== GATES (sagittal fluid, the dominant slot for every finding) =====")
sp = [res[(0, t)]["spread"] for t in T if (0, t) in res]
n15 = sum(s <= 15 for s in sp)
side_ok = all(res[(0, t)]["side"] >= 0.7 for t in SIDE_T if (0, t) in res)
q1 = n15 >= 6 and side_ok and res[(0, "pairwise")] >= 10
box = np.median([res[(0, t)]["box"] for t in T if (0, t) in res])
q2 = box <= 60
say(f"Q1: findings with in-plane spread <= 15 mm: {n15}/{len(sp)}; side-fixed >= 70% correct: {side_ok}; "
    f"pairwise median {res[(0, 'pairwise')]:.1f} mm -> {'PASS' if q1 else 'FAIL'}")
say(f"Q2: median per-window 80% box {box:.1f} mm -> {'PASS' if q2 else 'FAIL'} (zoom vs 117 mm view: {117 / box:.1f}x)")
say("VERDICT: " + ("both pass -> CAM kernel on training folds (localiser targets)" if q1 and q2 else
                   "not both -> CAM peaks are not yet a usable pseudo-coordinate target"))
OUT.write_text("\n".join(lines) + "\n")

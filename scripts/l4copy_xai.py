"""XAI + ablations of v4 l4copy (the LB-0.939 single) on fold 0, CPU in localhost/rsna-mlx-train:v1 (no training,
no submission; 2026-10-19). One global pass per study, then from the same features:
  series ablation   global logits with each series slot removed from the window attention (drop_j) and with only slot j
                    (only_j); attention mass per finding per slot.
  crop ablation     self-located crops at box x 0.67 / 1.0 / 1.5 (context vs zoom) and MS1's table x 1.0; top-1 crop only.
  case panels       for the studies in xai_cases.json (worst FN / FP / best TP per weak finding, from
                    scripts/analyze_l4copy_r1.py): per finding the top-1 global window with its gradient x activation
                    map (the map sub.locate uses), the 94-window attention, the self and MS1 crops with the crop branch's
                    Grad-CAM (layer4_c output).
Env: D96_CACHE (shard_0 with the fold-0 studies), V4_ART (v4_submission.py, weights, cp_*.npz), XAI_OUT, XAI_CASES,
XAI_MODELS ('tag=path,...'; default the LB weights as s2026), XAI_N (first N studies, test), OMP_NUM_THREADS.
Equivalence check: the first 3 studies' self fused = sub.predict_study (must be ~0).
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/core"); sys.path.insert(0, str(Path(__file__).resolve().parent))
import l4copy_diag_core as run  # noqa: E402

sub = run.sub
CF, TOP, IMG, BOXN, CMASK, RES = run.CF, run.TOP, run.IMG, run.BOXN, run.CMASK, run.RES
SCALES = (0.67, 1.0, 1.5)
LITE = bool(os.environ.get("XAI_LITE"))
PEAK = bool(os.environ.get("XAI_PEAK"))           # also crops centred on the peak of the gradient x activation map
NF = bool(os.environ.get("XAI_NF"))               # OA crops partly moved onto the sagittal non-fluid series (L4C-NF)
QP = json.loads(os.environ["XAI_QPRIOR"]) if os.environ.get("XAI_QPRIOR") else None   # L4C-LOC quota prior (Q baselines)
OA3 = [int(j) for j in np.where(np.isin(CF, [sub.TARGETS.index(t) for t in ("Medial OA", "Lateral OA", "PF OA")]))[0]]


def crops(vol, S, R, Q, scale, device):
    fid = np.repeat(CF, TOP); S, R, Q = S.ravel(), R.ravel(), Q.ravel()
    box = np.minimum(np.round(BOXN[fid] * scale), IMG).astype(np.int64); hb = box // 2
    R, Q = np.clip(R, hb, IMG - hb), np.clip(Q, hb, IMG - hb)
    xc = sub.windows(vol, S, device).float().div_(255.0)
    s = torch.as_tensor(box / IMG, dtype=torch.float32, device=device).view(-1, 1, 1)
    pxy = torch.stack([torch.as_tensor((Q + 0.5) / IMG * 2 - 1), torch.as_tensor((R + 0.5) / IMG * 2 - 1)], 1)
    th = torch.cat([s * torch.eye(2, device=device).expand(len(fid), 2, 2), pxy.float().to(device)[..., None]], 2)
    return F.grid_sample(xc, F.affine_grid(th, (len(fid), 3, sub.CROP_RES, sub.CROP_RES), align_corners=False),
                         mode="bilinear", padding_mode="zeros", align_corners=False), fid, box, R, Q


def crop_maps(model, x, cam=False):
    """crop branch: x [n, 3, 384, 384] -> per-crop logit for its own finding is taken by the caller; returns
    (pooled features after norm_l [n, 2048], layer4_c maps [n, 2048, 12, 12] if cam)."""
    bb = model.backbone
    for m in model.bns:
        m.crop = True
    try:
        h = bb.maxpool(bb.act1(bb.bn1(bb.conv1(sub.normalise(x)))))
        h = model.layer4_c(bb.layer3(bb.layer2(bb.layer1(h))))
    finally:
        for m in model.bns:
            m.crop = False
    return h


def local_logits(model, vol, tab, scale, device, top1=False):
    S, R, Q, A = tab
    x, fid, _, _, _ = crops(vol, S, R, Q, scale, device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        h = crop_maps(model, x)
    h = h.float()
    hc = model.norm_l(bb_pool(h))
    ft = torch.as_tensor(fid, device=device)
    lc = (hc.float() * model.clsV[ft]).sum(-1) + model.clsvb[ft]
    wt = A / A.sum(1, keepdims=True)
    if top1:
        wt = np.zeros_like(wt); wt[:, 0] = 1.0
    return torch.zeros(12, device=device).scatter_add(0, ft, lc * torch.as_tensor(wt.ravel(), dtype=torch.float32, device=device)).cpu()


def bb_pool(h):
    return h.mean((2, 3))


def series_ablation(model, fm, a_logits, cs):
    """a_logits [K, 12] pre-softmax attention; -> drop [5, 12], only [5, 12] global logits (nan where impossible),
    mass [12, 5]."""
    h = model.norm(fm.mean((2, 3)))
    sl = run.slot_of(cs)
    a = torch.softmax(a_logits, 0)
    slt = torch.as_tensor(sl, device=a.device)
    mass = np.stack([a[slt == j].sum(0).cpu().numpy() for j in range(5)], 1)
    drop = np.full((5, 12), np.nan, np.float32); only = drop.copy()
    for j in range(5):
        for keep, out in ((sl != j, drop), (sl == j, only)):
            if keep.sum() == 0:
                continue
            k = torch.as_tensor(keep, device=a.device)
            aa = torch.softmax(a_logits[k], 0)
            out[j] = (((aa.T @ h[k]) * model.clsW).sum(-1) + model.clsb).cpu().numpy()
    return drop, only, mass


def global_cam(model, fm, a, k_win, f):
    """gradient x activation of finding f's contribution on window k_win (sub.locate's map) -> [10, 10] >= 0."""
    with torch.enable_grad():
        t = fm[k_win:k_win + 1].detach().clone().requires_grad_(True)
        contrib = a[k_win, f] * (model.norm(t.mean((2, 3))) @ model.clsW[f])
        gr, = torch.autograd.grad(contrib.sum(), t)
    return (gr * t.detach()).sum(1)[0].clamp_min(0).cpu().numpy()


def crop_cam(model, vol, tab, j, device, scale=1.0):
    """Grad-CAM of the crop branch for cropped finding j on its top-1 crop -> (crop image [384, 384] uint8, cam [12, 12],
    crop logit, box, centre)."""
    S, R, Q, A = tab
    Sj, Rj, Qj = (x[j:j + 1, :1] for x in (S, R, Q))
    fid = CF[j]
    box = int(min(round(BOXN[fid] * scale), IMG)); hb = box // 2
    r, q = int(np.clip(Rj[0, 0], hb, IMG - hb)), int(np.clip(Qj[0, 0], hb, IMG - hb))
    xc = sub.windows(vol, Sj.ravel(), device).float().div_(255.0)
    s = torch.tensor([[box / IMG]], device=device).view(-1, 1, 1)
    pxy = torch.tensor([[(q + 0.5) / IMG * 2 - 1, (r + 0.5) / IMG * 2 - 1]], device=device)
    th = torch.cat([s * torch.eye(2, device=device).expand(1, 2, 2), pxy[..., None]], 2)
    x = F.grid_sample(xc, F.affine_grid(th, (1, 3, sub.CROP_RES, sub.CROP_RES), align_corners=False), mode="bilinear",
                      padding_mode="zeros", align_corners=False)
    with torch.enable_grad():
        h = crop_maps(model, x).detach().requires_grad_(True)
        lc = (model.norm_l(bb_pool(h))[0] * model.clsV[fid]).sum() + model.clsvb[fid]
        gr, = torch.autograd.grad(lc, h)
    cam = (gr.mean((2, 3), keepdim=True) * h.detach()).sum(1)[0].clamp_min(0).cpu().numpy()
    return (x[0, 1].cpu().numpy() * 255).astype(np.uint8), cam, float(lc), box, (int(Sj[0, 0]), r, q)


KEYS = ("glob", "drop", "only", "mass", "self_tab", "loc_self_067", "loc_self_100", "loc_self_150", "loc_ms1_100",
        "loc_self_top1", "loc_selfpeak_100", "peak_tab", "loc_nf1c_100", "loc_nf1o_100", "loc_nf2c_100", "nf_tab", "loc_qm_100", "loc_qs_100",
        "loc_selfu_100", "q_tab", "att_ent", "att_top", "ms1_tab")


def nf_match(c, mask):
    """SagF window centre c -> the SagNF window centre at the same relative position in the stack (MS1's own SagF / SagNF
    crops agree on relative position: corr 0.95, median |diff| 0.04), or None when c is not SagF or SagNF is absent."""
    b = run.BOUNDS
    if run.slot_of(c) != 0 or not mask[b[1]:b[2]].all():
        return None
    return int(np.clip(b[1] + round(c / (b[1] - 1) * (b[2] - b[1] - 1)), b[1] + 1, b[2] - 2))


def nf_tables(model, fm, a, cs, mask, tself):
    """L4C-NF variants for the 3 OA findings (others = self): nf1c replaces the lowest-weight window (4th) with the SagNF
    window matched to the top-1 window, point copied from it; nf1o the same window with its own gradient x activation
    point; nf2c replaces windows 3-4 with the SagNF matches of windows 1-2, points copied. Weights A are unchanged."""
    S0, R0, Q0, A0 = tself
    out = {}
    for nm, pairs in (("nf1c", ((0, 3),)), ("nf1o", ((0, 3),)), ("nf2c", ((0, 2), (1, 3)))):
        S, R, Q = S0.copy(), R0.copy(), Q0.copy()
        for j in OA3:
            for src, dst in pairs:
                c = nf_match(int(S0[j, src]), mask)
                if c is None or c not in cs:
                    continue
                S[j, dst], R[j, dst], Q[j, dst] = c, R0[j, src], Q0[j, src]
        out[nm] = (S, R, Q, A0)
    S, R, Q, A = out["nf1o"]
    win = np.stack([np.searchsorted(cs, S[j]) for j in range(len(CF))])
    _, Ro, Qo, _ = run.points(model, fm, a, cs, win)
    for j in OA3:
        if run.slot_of(int(S[j, 3])) == 1:
            R[j, 3], Q[j, 3] = Ro[j, 3], Qo[j, 3]
    return out


def q_choose(a, cs, mask):
    """L4C-LOC baseline Q: per cropped finding, quota[j][slot] windows from each series slot (MS1's training-fold mix x 4,
    scripts/build_l4c_quota_prior.py), ranked inside the slot by v4 attention; quota of an absent / too-small slot goes to
    the finding's other slots in order of MS1's mix; windows whose centre slice is blank (mask) are never chosen ->
    windows [10, TOP]."""
    ac = a[:, CF].cpu().numpy(); sl = run.slot_of(cs); ok = np.asarray(mask, bool)[cs]
    out = np.zeros((len(CF), TOP), np.int64)
    for j in range(len(CF)):
        quota = np.array(QP["quota"][j]); order = np.argsort(-np.array(QP["mix"][j]), kind="stable")
        rank = {k: [int(i) for i in np.where((sl == k) & ok)[0][np.argsort(-ac[(sl == k) & ok, j], kind="stable")]] for k in range(5)}
        got, short = [], 0
        for k in order:
            take = rank[k][:quota[k]]; got += take; short += quota[k] - len(take); rank[k] = rank[k][len(take):]
        for k in order:                                  # redistribute
            take = rank[k][:short]; got += take; short -= len(take); rank[k] = rank[k][len(take):]
        rest = [int(i) for i in np.argsort(-ac[:, j]) if int(i) not in got]
        got += rest[:TOP - len(got)]
        out[j] = sorted(got[:TOP], key=lambda i: -ac[i, j])
    return out


def q_tables(model, fm, a, cs, tself, mask):
    """qm: quota windows, per-(finding, slot) median MS1 centre, uniform weights; qs: quota windows, self gradient x
    activation points, uniform weights; selfu: the self table with uniform weights (weight control)."""
    win = q_choose(a, cs, mask)
    Ss, Rs, Qs, _ = run.points(model, fm, a, cs, win)
    U = np.full((len(CF), TOP), 1.0 / TOP, np.float32)
    med = np.array(QP["median"]); sl = run.slot_of(Ss)
    Rm = med[np.arange(len(CF))[:, None], sl, 0]; Qm = med[np.arange(len(CF))[:, None], sl, 1]
    S0, R0, Q0, _ = tself
    return {"qm": (Ss, Rm.astype(np.int64), Qm.astype(np.int64), U), "qs": (Ss, Rs, Qs, U), "selfu": (S0, R0, Q0, U)}


def points_peak(model, fm, a, cs, win):
    """run.points with the crop centre at the peak (argmax cell) of the gradient x activation map instead of its centroid:
    a map with two modes (patella + posterior joint for PF OA) puts the centroid between them."""
    dev = fm.device
    W = model.clsW.float()[CF]
    ac = a[:, CF]
    U = torch.as_tensor(np.unique(win), device=dev); row = {int(k): j for j, k in enumerate(U.tolist())}
    S = np.zeros((len(CF), TOP), np.int64); R = S.copy(); Q = S.copy(); A = np.zeros((len(CF), TOP), np.float32)
    with torch.enable_grad():
        t = fm[U].detach().clone().requires_grad_(True)
        contrib = ac[U] * (model.norm.float()(t.mean((2, 3))) @ W.T)
        c = t.shape[-1]; px = RES / c; lo = (IMG - RES) // 2
        for j in range(len(CF)):
            gr, = torch.autograd.grad(contrib[:, j].sum(), t, retain_graph=j < len(CF) - 1)
            ks = [int(k) for k in win[j]]; rows = [row[k] for k in ks]
            mp = (gr[rows] * t.detach()[rows]).sum(1).clamp_min(0)
            flat = mp.flatten(1).argmax(1)
            cy, cx = (flat // c).float(), (flat % c).float()
            S[j] = cs[ks]
            R[j] = np.round((lo + (cy + 0.5) * px).cpu().numpy()).astype(np.int64)
            Q[j] = np.round((lo + (cx + 0.5) * px).cpu().numpy()).astype(np.int64)
            A[j] = ac[ks, j].detach().cpu().numpy()
    return S, R, Q, A


def study(model, sid, vol, mask, ms1, cases, device, R, panels, checks):
    """All analyses for one study and one model; fills R[key][sid] and panels (case studies)."""
    cs, fm, a, g = run.global_pass(model, vol, mask, device)
    h = model.norm(fm.mean((2, 3))); al = model.att(h)
    R["glob"][sid] = g.cpu().numpy()
    R["drop"][sid], R["only"][sid], R["mass"][sid] = series_ablation(model, fm, al, cs)
    tself = run.points(model, fm, a, cs, run.choose(a, cs))
    R["self_tab"][sid] = np.stack(tself[:3]).astype(np.int16)
    an = a.cpu().numpy()                                 # window attention [K, 12]: normalised entropy, top-1 weight
    R["att_ent"][sid] = -(an * np.log(np.clip(an, 1e-12, 1))).sum(0) / np.log(len(cs)); R["att_top"][sid] = an.max(0)
    cen, at = ms1
    tms1 = (cs[cen[..., 0]], cen[..., 1].copy(), cen[..., 2].copy(), at.copy())
    R["ms1_tab"][sid] = np.stack(tms1[:3]).astype(np.int16)
    for sc in SCALES:
        if not LITE or sc == 1.0:
            R[f"loc_self_{int(round(sc * 100)):03d}"][sid] = local_logits(model, vol, tself, sc, device).numpy()
    R["loc_ms1_100"][sid] = local_logits(model, vol, tms1, 1.0, device).numpy()
    if not LITE:
        R["loc_self_top1"][sid] = local_logits(model, vol, tself, 1.0, device, top1=True).numpy()
    if PEAK:
        tpk = points_peak(model, fm, a, cs, run.choose(a, cs))
        R["peak_tab"][sid] = np.stack(tpk[:3]).astype(np.int16)
        R["loc_selfpeak_100"][sid] = local_logits(model, vol, tpk, 1.0, device).numpy()
    if NF:
        tnf = nf_tables(model, fm, a, cs, mask, tself)
        R["nf_tab"][sid] = np.stack([tnf[k][0] for k in ("nf1c", "nf2c")]).astype(np.int16)
        for k, tb in tnf.items():
            R[f"loc_{k}_100"][sid] = local_logits(model, vol, tb, 1.0, device).numpy()
    if QP is not None:
        tq = q_tables(model, fm, a, cs, tself, mask)
        R["q_tab"][sid] = tq["qs"][0].astype(np.int16)
        for k, tb in tq.items():
            R[f"loc_{k}_100"][sid] = local_logits(model, vol, tb, 1.0, device).numpy()
    if len(checks) < 3:                                  # this script's pieces == the submission's predict_study
        ref = sub.predict_study(model, vol, mask, device, amp=device.type == "cuda")
        gc = g.cpu(); mine = torch.where(torch.as_tensor(CMASK), 0.5 * (gc + torch.as_tensor(R["loc_self_100"][sid])), gc)
        checks.append(float(np.abs(ref[0] - mine.sigmoid().numpy()).max()))
    for f, kind in cases:
        fi = sub.TARGETS.index(f)
        key = f"{f}|{kind}|{sid[-8:]}"
        kw = int(a[:, fi].argmax())
        panels[f"{key}|win_img"] = vol[int(cs[kw])][(IMG - RES) // 2:(IMG + RES) // 2, (IMG - RES) // 2:(IMG + RES) // 2]
        panels[f"{key}|win_cam"] = global_cam(model, fm, a, kw, fi)
        panels[f"{key}|win_info"] = np.array([cs[kw], run.slot_of(cs[kw]), float(a[kw, fi])])
        panels[f"{key}|att"] = a[:, fi].cpu().numpy(); panels[f"{key}|att_slot"] = run.slot_of(cs)
        if fi in CF:
            j = int(np.where(CF == fi)[0][0])
            for nm, tb in (("self", tself), ("ms1", tms1)):
                img, cam, lc, box, srq = crop_cam(model, vol, tb, j, device)
                panels[f"{key}|{nm}_crop"] = img; panels[f"{key}|{nm}_cam"] = cam
                panels[f"{key}|{nm}_info"] = np.array([lc, box, *srq, run.slot_of(srq[0])])
                panels[f"{key}|{nm}_full"] = vol[srq[0]]
        panels[f"{key}|pred"] = np.array([float(g[fi])] + [float(R[k][sid][fi]) for k in ("loc_self_100", "loc_ms1_100")])


def load_models(device):
    """XAI_MODELS = 'tag=path,tag=path' (default: the LB-0.939 weights as s2026); the submission weights are hash-checked."""
    spec = os.environ.get("XAI_MODELS") or f"s2026={run.ART / sub.WEIGHTS['file']}"
    out = {}
    for item in spec.split(","):
        tag, path = item.split("=", 1)
        sha = sub.WEIGHTS["sha256"] if Path(path).name == sub.WEIGHTS["file"] and Path(path).parent == run.ART else None
        out[tag] = sub.load_model(path, device, sha).to(memory_format=torch.contiguous_format)
    return out


def case_index(ids):
    if os.environ.get("XAI_NOCASES"):
        return {}
    cases = json.loads(Path(os.environ["XAI_CASES"]).read_text()) if os.environ.get("XAI_CASES") else {}
    case_of = {}
    for f, d in cases.items():
        for kind, lst in d.items():
            for s in lst:
                case_of.setdefault(s, []).append((f, kind))
    if os.environ.get("XAI_N"):
        case_of = {ids[0]: [("Fracture", "fn"), ("Baker's", "tp")]}
    return case_of


def save(out, ids, models, R, panels, checks, extra):
    out.mkdir(parents=True, exist_ok=True)
    for tag in models:
        np.savez_compressed(out / f"xai_arrays_{tag}.npz", ids=np.array(ids),
                            **{k: np.stack([v[s] for s in ids]) for k, v in R[tag].items() if v})
        np.savez_compressed(out / f"xai_panels_{tag}.npz", **panels[tag])
    log = {"studies": len(ids), "equiv_self_fused_max_abs": checks, "models": list(models), **extra,
           "threads": torch.get_num_threads()}
    (out / "xai_log.json").write_text(json.dumps(log, indent=1) + "\n")
    print("XAI_DONE " + json.dumps(log), flush=True)


def loop(items, n_total, models, tabm, case_of, device, t0):
    """items: iterator of (sid, vol, mask) -> R, panels, checks, done ids."""
    R = {t: {k: {} for k in KEYS} for t in models}; panels = {t: {} for t in models}; checks = {t: [] for t in models}
    done = []
    with torch.no_grad():
        for n, (sid, vol, mask) in enumerate(items, 1):
            if not mask.any():
                continue
            for tag, model in models.items():
                study(model, sid, vol, mask, tabm[sid], case_of.get(sid, []), device, R[tag], panels[tag], checks[tag])
            done.append(sid)
            if n % 25 == 0 or n == n_total:
                print(f"XAI {n}/{n_total} {time.monotonic() - t0:.0f}s", flush=True)
    return R, panels, checks, done


def main():
    """Local / Modal: volumes from the d96 cache (D96_CACHE)."""
    t0 = time.monotonic()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    meta = run_manifest(Path(os.environ["D96_CACHE"]))
    tabm = run.ms1_table()
    ids = sorted(meta)
    if os.environ.get("XAI_IDS"):
        ids = sorted(set(Path(os.environ["XAI_IDS"]).read_text().split())); assert all(s in meta for s in ids)
    if os.environ.get("XAI_N"):
        ids = [s for s in ids if s in tabm][:int(os.environ["XAI_N"])]
    case_of = case_index(ids)
    if os.environ.get("XAI_CASES_ONLY"):
        ids = [s for s in ids if s in case_of]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(device)
    mm = {}

    def read(s):
        f, row, mask = meta[s]
        if f not in mm:
            mm[f] = np.load(f, mmap_mode="r")
        return s, np.ascontiguousarray(mm[f][row:row + sub.MAXS]), mask

    with ThreadPoolExecutor(2) as pool:
        R, panels, checks, done = loop(pool.map(read, ids), len(ids), models, tabm, case_of, device, t0)
    save(Path(os.environ["XAI_OUT"]), done, models, R, panels, checks, {"source": "d96 cache", "seconds": round(time.monotonic() - t0, 1)})


def run_manifest(cache):
    out = {}
    for root in sorted(cache.glob("shard_*")):
        for line in (root / "manifest.jsonl").read_text().splitlines():
            r = json.loads(line)
            out[r["study_id"]] = (str(root / r["part"]), r["row_start"], np.array(r["mask"], bool))
    return out


if __name__ == "__main__":
    main()

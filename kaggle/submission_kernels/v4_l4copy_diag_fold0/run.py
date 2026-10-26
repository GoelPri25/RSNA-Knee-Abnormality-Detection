"""L4C-D FOLD-0 DIAGNOSTICS (no submission; ledger row L4C-D): where does v4 l4copy lose when it places its own crops,
and how sensitive is its crop branch to placement? The submission code (v4_submission.py in the weights dataset) on the
825 fold-0 studies, volumes decoded from the train DICOMs by the submission's own builder. Per study one global pass,
then one crop pass per condition (all with the l4copy SWA weights; the global head is shared, so only the crops differ):
  ms1             MS1's table (the training-time placement; must reproduce val_swa_fused 0.9475)
  self            the submission's self-location (top-4 windows by v4 attention, any plane; LB 0.939 path)
  self_plane      self-location restricted per finding to the series slots holding >= 80% of MS1's crops on training
                  folds 1-5 (fold 0 excluded; data/coatnet_d96_fold0/l4copy_diag/plane_prior.json)
  ms1win_selfpts  MS1's windows and weights, points from v4's own gradient x activation on those windows
                  (separates window choice from in-plane point placement)
  shift5/10/20    MS1's table with every crop centre moved 5 / 10 / 20 mm in a random in-plane direction
  slice2          MS1's table with every crop window moved +-2 slices, kept inside its series
Renders: top-1 crop of MS1 / self / self_plane on a fixed set of PF OA, Baker's and fracture studies (PNG).
Out (/kaggle/working): val_global.json, val_<cond>_{fused,local}.json, tables.npz, render_*.png, diag_log.json.
Scored locally by scripts/audit_l4copy_diag.py.
"""
import json
import os
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ART = (Path(os.environ["V4_ART"]) if os.environ.get("V4_ART") else                 # Modal: the same files mounted
       next(p for p in (Path("/kaggle/input/datasets/tumis001/rsna-v4-l4copy-fold0"),
                        Path("/kaggle/input/rsna-v4-l4copy-fold0")) if p.exists()))
sys.path.insert(0, str(ART))                             # importable by name: Pool pickles sub.build_study
import v4_submission as sub  # noqa: E402

IMG, TOP, CF, BOXN, CMASK, RES = sub.IMG, sub.TOP, sub.CF, sub.BOXN, sub.CMASK, sub.EVAL_RES
PX_MM = IMG / sub.CROP_MM
BOUNDS = np.cumsum([0] + [k for _, _, k in sub.SLOTS])
SLOT = ["SagF", "SagNF", "CorF", "CorNF", "Ax"]
ALLOWED = {"ACL": ["SagF"], "MCL": ["CorF"], "Medial Meniscus": ["SagF"], "Lateral Meniscus": ["SagF"],
           "Medial OA": ["SagF", "SagNF"], "Lateral OA": ["SagF", "SagNF"], "PF OA": ["SagF", "SagNF"],
           "Baker's": ["Ax"], "Contusion": ["SagF", "CorF"], "Fracture": ["CorF", "SagF"]}
CONDS = ["ms1", "self", "self_plane", "ms1win_selfpts", "shift5", "shift10", "shift20", "slice2"]


def slot_of(s):
    return np.searchsorted(BOUNDS, s, side="right") - 1


def ms1_table():
    tab = {}
    for p in sorted(ART.glob("cp_*.npz")):
        z = np.load(p)
        for s, cen, at in zip(z["ids"], z["centres"], z["a_top"]):
            tab[str(s)] = (cen[CF].astype(np.int64), at[CF].astype(np.float32))
    return tab


def global_pass(model, vol, mask, device):
    cs = sub.uniform_centres(mask)
    lo = (IMG - RES) // 2
    xg = sub.windows(vol, cs, device)[..., lo:lo + RES, lo:lo + RES].float().div_(255.0)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        fm = model.backbone.forward_features(sub.normalise(xg))
    fm = fm.float()
    h = model.norm.float()(fm.mean((2, 3)))
    a = torch.softmax(model.att.float()(h), dim=0)                               # [K, 12]
    g = ((a.T @ h) * model.clsW.float()).sum(-1) + model.clsb.float()
    return cs, fm, a, g


def choose(a, cs, allowed=None):
    """top-TOP windows per cropped finding by v4 attention -> [10, TOP] window indices; allowed: per finding the slots
    to choose from (all slots when fewer than TOP windows fall inside them)."""
    ac = a[:, CF]
    out = np.zeros((len(CF), TOP), np.int64)
    sl = slot_of(cs)
    for j, f in enumerate(CF):
        sc = ac[:, j].clone()
        if allowed is not None:
            ok = torch.as_tensor(np.isin(sl, [SLOT.index(x) for x in allowed[sub.TARGETS[f]]]), device=sc.device)
            if int(ok.sum()) >= TOP:
                sc[~ok] = -1.0
        out[j] = sc.topk(TOP).indices.cpu().numpy()
    return out


def points(model, fm, a, cs, win):
    """sub.locate's gradient x activation centroids on given windows win [10, TOP] -> (slices, rows, cols, weights)."""
    dev = fm.device
    W = model.clsW.float()[CF]
    ac = a[:, CF]
    U = torch.as_tensor(np.unique(win), device=dev); row = {int(k): j for j, k in enumerate(U.tolist())}
    S = np.zeros((len(CF), TOP), np.int64); R = S.copy(); Q = S.copy(); A = np.zeros((len(CF), TOP), np.float32)
    with torch.enable_grad():
        t = fm[U].detach().clone().requires_grad_(True)
        contrib = ac[U] * (model.norm.float()(t.mean((2, 3))) @ W.T)
        c = t.shape[-1]; px = RES / c; gi = torch.arange(c, device=dev, dtype=torch.float32)
        lo = (IMG - RES) // 2
        for j in range(len(CF)):
            gr, = torch.autograd.grad(contrib[:, j].sum(), t, retain_graph=j < len(CF) - 1)
            ks = [int(k) for k in win[j]]; rows = [row[k] for k in ks]
            mp = (gr[rows] * t.detach()[rows]).sum(1).clamp_min(0)
            tot = mp.sum((1, 2)); ok = tot > 0; tot = tot.clamp_min(1e-12)
            cy = torch.where(ok, (mp.sum(2) * gi).sum(1) / tot, torch.full_like(tot, (c - 1) / 2))
            cx = torch.where(ok, (mp.sum(1) * gi).sum(1) / tot, torch.full_like(tot, (c - 1) / 2))
            S[j] = cs[ks]
            R[j] = np.round((lo + (cy + 0.5) * px).cpu().numpy()).astype(np.int64)
            Q[j] = np.round((lo + (cx + 0.5) * px).cpu().numpy()).astype(np.int64)
            A[j] = ac[ks, j].detach().cpu().numpy()
    return S, R, Q, A


def crop_logits(model, vol, S, R, Q, A, device):
    """sub.predict_study's crop half: table [10, TOP] -> local logits [12] (0 where no crops)."""
    fid = np.repeat(CF, TOP)
    S, R, Q = S.ravel(), R.ravel(), Q.ravel()
    hb = BOXN[fid] // 2
    R, Q = np.clip(R, hb, IMG - hb), np.clip(Q, hb, IMG - hb)
    wt = (A / A.sum(1, keepdims=True)).ravel()
    xc = sub.windows(vol, S, device).float().div_(255.0)
    s = torch.as_tensor(BOXN[fid] / IMG, dtype=torch.float32, device=device).view(-1, 1, 1)
    pxy = torch.stack([torch.as_tensor((Q + 0.5) / IMG * 2 - 1), torch.as_tensor((R + 0.5) / IMG * 2 - 1)], 1)
    th = torch.cat([s * torch.eye(2, device=device).expand(len(fid), 2, 2), pxy.float().to(device)[..., None]], 2)
    xc = F.grid_sample(xc, F.affine_grid(th, (len(fid), 3, sub.CROP_RES, sub.CROP_RES), align_corners=False),
                       mode="bilinear", padding_mode="zeros", align_corners=False)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        hc = model.crop_features(sub.normalise(xc))
    hc = model.norm_l.float()(hc.float())
    fidt = torch.as_tensor(fid, device=device)
    lc = (hc * model.clsV.float()[fidt]).sum(-1) + model.clsvb.float()[fidt]
    return torch.zeros(12, device=device).scatter_add(0, fidt, lc * torch.as_tensor(wt, device=device))


def heads(g, l, device):
    cm = torch.as_tensor(CMASK, device=device)
    return torch.where(cm, 0.5 * (g + l), g).sigmoid().cpu().numpy(), torch.where(cm, l, g).sigmoid().cpu().numpy()


def shifted(tab, sid, cond):
    S, R, Q, A = (x.copy() for x in tab)
    rng = np.random.default_rng(zlib.crc32(f"{sid}|{cond}".encode()))
    if cond.startswith("shift"):
        d = float(cond[5:]) * PX_MM
        th = rng.uniform(0, 2 * np.pi, R.shape)
        R = np.round(R + d * np.sin(th)).astype(np.int64); Q = np.round(Q + d * np.cos(th)).astype(np.int64)
    else:                                                 # slice2
        sl = slot_of(S)
        S = np.clip(S + rng.choice([-2, 2], S.shape), BOUNDS[sl] + 1, BOUNDS[sl + 1] - 2)
    return S, R, Q, A


def render(vol, tabs, f, path, title):
    import cv2
    j = int(np.where(CF == sub.TARGETS.index(f))[0][0]); hb = int(BOXN[CF[j]]) // 2
    top, bot = [], []
    for nm, col in (("ms1", (0, 200, 0)), ("self", (0, 0, 230)), ("self_plane", (230, 120, 0))):
        S, R, Q, A = tabs[nm]
        s, r, q = int(S[j, 0]), int(np.clip(R[j, 0], hb, IMG - hb)), int(np.clip(Q[j, 0], hb, IMG - hb))
        im = cv2.cvtColor(vol[s], cv2.COLOR_GRAY2BGR)
        cv2.rectangle(im, (q - hb, r - hb), (q + hb, r + hb), col, 2)
        cv2.putText(im, f"{nm} {SLOT[slot_of(s)]} s{s}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
        cv2.putText(im, "4 slots: " + ",".join(SLOT[slot_of(int(x))] for x in S[j]), (6, 376),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        top.append(im)
        cr = cv2.resize(cv2.cvtColor(vol[s][r - hb:r + hb, q - hb:q + hb], cv2.COLOR_GRAY2BGR), (IMG, IMG))
        bot.append(cr)
    sheet = np.concatenate([np.concatenate(top, 1), np.concatenate(bot, 1)], 0)
    cv2.putText(sheet, title, (6, IMG + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.imwrite(str(path), sheet)


def one_study(model, sid, vol, mask, ms1, device):
    """-> global sigmoid [12], the per-condition crop tables, per-condition (fused, local) [12], crop-pass seconds."""
    cs, fm, a, g = global_pass(model, vol, mask, device)
    cen, at = ms1
    tabs = {"ms1": (cs[cen[..., 0]], cen[..., 1].copy(), cen[..., 2].copy(), at.copy())}
    tabs["self"] = points(model, fm, a, cs, choose(a, cs))
    tabs["self_plane"] = points(model, fm, a, cs, choose(a, cs, ALLOWED))
    Sm, Rm, Qm, _ = points(model, fm, a, cs, cen[..., 0])
    tabs["ms1win_selfpts"] = (Sm, Rm, Qm, at.copy())
    for c in ("shift5", "shift10", "shift20", "slice2"):
        tabs[c] = shifted(tabs["ms1"], sid, c)
    preds, dt = {}, {}
    for c in CONDS:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.monotonic()
        preds[c] = heads(g, crop_logits(model, vol, *tabs[c], device), device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt[c] = time.monotonic() - t1
    return g.sigmoid().cpu().numpy(), tabs, preds, dt


def main():
    t0 = time.monotonic()
    data, work = sub.competition_dir(), Path("/kaggle/working")
    ids_path = next(p for p in (Path("/kaggle/input/datasets/tumis001/rsna-d96-fold0-ids/fold0_ids.json"),
                                Path("/kaggle/input/rsna-d96-fold0-ids/fold0_ids.json")) if p.exists())
    ids = json.loads(ids_path.read_text()); assert len(ids) == 825, len(ids)
    rend = {}
    for f, lst in RENDER_IDS.items():
        for i, s in enumerate(lst):
            rend.setdefault(s, []).append((f, i))
    tab = ms1_table(); assert all(s in tab for s in ids)
    sub.ensure_timm()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = sub.load_model(ART / sub.WEIGHTS["file"], device, sub.WEIGHTS["sha256"])
    by_study = {}
    for row in sub.read_csv(data / "train_series.csv"):
        by_study.setdefault(row["StudyInstanceUID"], []).append(row)
    out = {c: {h: {} for h in ("fused", "local")} for c in CONDS}
    glob_out, keep = {}, {c: {k: [] for k in "SRQ"} for c in ("self", "self_plane", "ms1win_selfpts")}
    tsec = {c: 0.0 for c in CONDS}; checks = {"equiv_ms1": [], "equiv_self": []}
    stats = {"empty_studies": 0, "decode_errors": 0}
    done = []
    with Pool(4) as procs, torch.no_grad():
        for n, (sid, vol, mask, errors) in enumerate(
                procs.imap(sub.build_study, [(s, by_study.get(s, []), data / "train_series") for s in ids]), 1):
            stats["decode_errors"] += errors
            if not mask.any():
                stats["empty_studies"] += 1
                continue
            g, tabs, preds, dt = one_study(model, sid, vol, mask, tab[sid], device)
            glob_out[sid] = g
            for c in CONDS:
                tsec[c] += dt[c]
                out[c]["fused"][sid], out[c]["local"][sid] = preds[c]
            if n <= 3:                                   # this script's pieces == the submission's predict_study
                for c in ("ms1", "self"):
                    ref = sub.predict_study(model, vol, mask, device, table=tabs["ms1"] if c == "ms1" else None)
                    checks[f"equiv_{c}"].append(float(np.abs(ref[0] - out[c]["fused"][sid]).max()))
            for c in keep:
                for k, v in zip("SRQ", tabs[c][:3]):
                    keep[c][k].append(v)
            for f, i in rend.get(sid, []):
                render(vol, tabs, f, work / f"render_{f.replace(' ', '').replace(chr(39), '')}_{i:02d}.png",
                       f"{f} #{i}")
            done.append(sid)
            if n % 50 == 0 or n == len(ids):
                print(f"L4C-D {n}/{len(ids)} studies; {time.monotonic() - t0:.0f}s", flush=True)
    note = "v4 l4copy fold 0, Kaggle path"
    (work / "val_global.json").write_text(json.dumps({"targets_in_order": sub.TARGETS, "note": note + ", global",
        "predictions": [{"study_id": s, "probabilities": glob_out[s].tolist()} for s in done]}))
    for c in CONDS:
        for h in ("fused", "local"):
            (work / f"val_{c}_{h}.json").write_text(json.dumps({"targets_in_order": sub.TARGETS, "note": f"{note}, {c}, {h}",
                "predictions": [{"study_id": s, "probabilities": out[c][h][s].tolist()} for s in done]}))
    np.savez(work / "tables.npz", ids=np.array(done), **{f"{c}_{k}": np.stack(v) for c, d in keep.items() for k, v in d.items()})
    log = {"studies": len(done), "stats": stats, "checks": checks,
           "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
           "crop_pass_s_per_study": {c: round(t / max(1, len(done)), 4) for c, t in tsec.items()},
           "allowed": ALLOWED, "seconds": round(time.monotonic() - t0, 1)}
    (work / "diag_log.json").write_text(json.dumps(log, indent=1) + "\n")
    print("DIAG_DONE " + json.dumps(log), flush=True)


RENDER_IDS = {
 "PF OA": [
  "1.2.826.0.1.3680043.8.498.69596668032633976121623939445795750384",
  "1.2.826.0.1.3680043.8.498.82143276307898132314496735515204481508",
  "1.2.826.0.1.3680043.8.498.13099702881219862576749493772702888064",
  "1.2.826.0.1.3680043.8.498.89590880980421560369495600204866789646",
  "1.2.826.0.1.3680043.8.498.42829490507052740062127365747846030214",
  "1.2.826.0.1.3680043.8.498.46825768767498441994570613185981335949",
  "1.2.826.0.1.3680043.8.498.54099199514076260519579096546600505763",
  "1.2.826.0.1.3680043.8.498.91839696756303525362262251587171486273",
  "1.2.826.0.1.3680043.8.498.11472441320199233244785757534656853558",
  "1.2.826.0.1.3680043.8.498.78765525474890094481249498938657540052",
  "1.2.826.0.1.3680043.8.498.76772148369531098768751348676329945789",
  "1.2.826.0.1.3680043.8.498.35820650938063925991582813178954225720",
  "1.2.826.0.1.3680043.8.498.79326627674334637493320764205397279674",
  "1.2.826.0.1.3680043.8.498.11598606533748611505510113033734692457"
 ],
 "Baker's": [
  "1.2.826.0.1.3680043.8.498.95984406324600336713471618620285270466",
  "1.2.826.0.1.3680043.8.498.35769002792415947424087098083674166777",
  "1.2.826.0.1.3680043.8.498.39020232951771672617852805188845504161",
  "1.2.826.0.1.3680043.8.498.46776367027887983105817631582518625639",
  "1.2.826.0.1.3680043.8.498.50512021577184367495598050974910055550",
  "1.2.826.0.1.3680043.8.498.41837648239243141231469586963862388683"
 ],
 "Fracture": [
  "1.2.826.0.1.3680043.8.498.20933151653103548000061448393771614270",
  "1.2.826.0.1.3680043.8.498.31175171249406459042393597291141275669",
  "1.2.826.0.1.3680043.8.498.77421709644087006134426085284321515809",
  "1.2.826.0.1.3680043.8.498.84059077072076289248966490696872414783",
  "1.2.826.0.1.3680043.8.498.12769221625850663248477857087347890908",
  "1.2.826.0.1.3680043.8.498.39020232951771672617852805188845504161"
 ]
}   # fixed fold-0 study ids (render_ids.json: PF OA 8 pos + 6 neg, Baker's 6 pos, fracture 6 pos)

if __name__ == "__main__":
    main()

"""Exp L4C-LOC (docs/l4c_loc_spec.md v2, ledger row L4C-LOC): a learned crop locator for the frozen v4 l4copy, distilled
from MS1's crop placements at training time, so that the deployed model places its crops like the table its crop branch
was trained on, without running MS1 at inference.

Frozen: every v4 weight (loaded with v4_submission.load_model, the deployed code). New, per model (~0.55 M parameters):
  window head  s[k, j] = att'(h_k)[CF j] + e[slot(k), j]; att' = trainable copy of v4's attention MLP (eval mode, no
               dropout), e = series-slot bias (init 0) -> at init top-4 = sub.locate's windows, softmax over the chosen 4
               = its weights. Loss: KL(q_j || softmax_k s[., j]) over the study's windows, q_j = MS1's a_top on its 4 windows.
  point head   1x1 conv 2048 -> 10 on the frozen layer4 map of a window (10 x 10 cells, 32 px), softmax over cells;
               target = Gaussian (sigma 1 cell) at MS1's centre after predict_study's clip; trained on MS1's windows only.
               Inference: soft-argmax -> pixel row / col in the 384 frame.
Features: one pass per study exactly as predict_study (94 eval windows, 320 centre crop, fp16 autocast, fp32 h); h is
cached for every study, layer4 maps only on MS1's windows of training studies. Training studies = all folds except 0
(3,524; asserted on the full cache). Fold-0 MS1 placements are used only for the ceiling and diagnostics.
Fold-0 scoring (crop logits for each placement, all through the deployed crop path l4copy_diag_core.crop_logits):
  self (sub.locate), ms1 (MS1 table), wp (learned windows + learned points + learned weights), wm (learned windows +
  per-(finding, slot) median MS1 centre), w (learned windows + gradient x activation points), qm (zero-training quota
  rule, l4copy_xai.q_tables). Saved as loc_<variant>_100 next to glob, the layout scripts/audit_l4c_nf.py scores.
Env: D96_CACHE (shard_*/manifest.jsonl), V4_ART (v4_submission.py + cp_*.npz), LOC_WEIGHTS (v4 weights), LOC_TAG,
LOC_OUT, LOC_FOLDS (folds.csv), LOC_QPRIOR (quota_prior.json), LOC_EPOCHS (10), LOC_N (CPU test: first N train / val).
"""
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/core"); sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("XAI_QPRIOR", Path(os.environ["LOC_QPRIOR"]).read_text() if os.environ.get("LOC_QPRIOR") else "")
import l4copy_diag_core as run  # noqa: E402
import l4copy_xai as xai  # noqa: E402

sub = run.sub
CF, TOP, IMG, RES, BOXN = run.CF, run.TOP, run.IMG, run.RES, run.BOXN
LO = (IMG - RES) // 2; CELL = 32; C = RES // CELL
CFGL = {"version": "l4c_loc_v1", "epochs": int(os.environ.get("LOC_EPOCHS", "10")), "batch": 8, "lr_slot_point": 1e-3,
        "lr_att": 1e-4, "wd": 0.01, "floor": 0.01, "sigma_cells": 1.0, "seed": 2026, "precision": "fp16 autocast, fp32 h"}
OUT = Path(os.environ["LOC_OUT"]); N = int(os.environ.get("LOC_N", "0"))


class Locator(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.att = __import__("copy").deepcopy(model.att).float().eval()
        for p in self.att.parameters():
            p.requires_grad_(True)
        self.e = nn.Parameter(torch.zeros(5, len(CF)))
        self.pt = nn.Conv2d(2048, len(CF), 1)
        nn.init.zeros_(self.pt.weight); nn.init.zeros_(self.pt.bias)
        self.register_buffer("cf", torch.as_tensor(CF))

    def scores(self, h, slot):
        """h [K, 2048] fp32 (v4 norm output), slot [K] -> s [K, 10]."""
        return self.att(h)[:, self.cf] + self.e[slot]

    def point_logits(self, fm):
        """fm [n, 2048, C, C] -> [n, 10, C * C]."""
        return self.pt(fm.float()).flatten(2)

    def table(self, h, slot, fm_of, cs):
        """learned (S, R, Q, A) [10, TOP]: top-4 windows by s, soft-argmax points, softmax-of-s weights."""
        s = self.scores(h, slot)
        top = s.topk(TOP, dim=0).indices.T                                       # [10, TOP]
        S = cs[top.cpu().numpy()]; A = torch.softmax(s.gather(0, top.T).T, 1).detach().cpu().numpy()
        U = torch.unique(top); fm = fm_of(U); pos = {int(k): i for i, k in enumerate(U.tolist())}
        pl = self.point_logits(fm)                                               # [u, 10, C*C]
        g = torch.arange(C, device=h.device, dtype=torch.float32)
        R = np.zeros((len(CF), TOP), np.int64); Q = R.copy()
        for j in range(len(CF)):
            p = torch.softmax(pl[[pos[int(k)] for k in top[j]], j], -1).view(TOP, C, C)
            cy = (p.sum(2) * g).sum(1); cx = (p.sum(1) * g).sum(1)
            R[j] = np.round((LO + (cy + 0.5) * CELL).cpu().numpy()).astype(np.int64)
            Q[j] = np.round((LO + (cx + 0.5) * CELL).cpu().numpy()).astype(np.int64)
        return S, R, Q, A, top.cpu().numpy()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def global_feats(model, vol, mask, device):
    """predict_study's global pass -> cs, fm [K, 2048, C, C] fp32, h [K, 2048] fp32, a [K, 12], g [12]."""
    cs, fm, a, g = run.global_pass(model, vol, mask, device)
    return cs, fm, model.norm.float()(fm.mean((2, 3))), a, g


def ms1_targets(tabm, sid, cs):
    """MS1 table -> window indices [10, TOP], weights [10, TOP], clipped centres (row, col) [10, TOP] (384 frame)."""
    cen, at = tabm[sid]
    w = cen[..., 0].astype(np.int64); assert w.max() < len(cs), (sid, w.max(), len(cs))
    hb = BOXN[CF][:, None] // 2
    return w, at / at.sum(1, keepdims=True), np.clip(cen[..., 1], hb, IMG - hb), np.clip(cen[..., 2], hb, IMG - hb)


def cell_target(r, q, sigma):
    """pixel centre -> Gaussian over the C x C cells of the 320 crop (cell centre i -> LO + (i + 0.5) * CELL)."""
    g = np.arange(C)
    yr = np.exp(-0.5 * (((r - LO) / CELL - 0.5 - g) / sigma) ** 2); xq = np.exp(-0.5 * (((q - LO) / CELL - 0.5 - g) / sigma) ** 2)
    t = np.outer(yr, xq).ravel(); return t / t.sum()


def extract(model, ids, meta, tabm, device, with_maps):
    """per study: h fp16 [K, 2048], slot [K], MS1 targets; maps fp16 on MS1's windows when with_maps."""
    out, mm = {}, {}
    t0 = time.monotonic()
    with torch.no_grad():
        for n, s in enumerate(ids, 1):
            f, row, mask = meta[s]
            if f not in mm:
                mm[f] = np.load(f, mmap_mode="r")
            vol = np.ascontiguousarray(mm[f][row:row + sub.MAXS])
            cs, fm, h, a, g = global_feats(model, vol, mask, device)
            w, q, R, Q = ms1_targets(tabm, s, cs)
            d = {"h": h.cpu(), "slot": torch.as_tensor(run.slot_of(cs)), "w": w, "q": q.astype(np.float32), "R": R, "Q": Q}
            if with_maps:
                U = np.unique(w); d["U"] = U; d["maps"] = fm[torch.as_tensor(U, device=fm.device)].half().cpu()
            out[s] = d
            if n % 250 == 0 or n == len(ids):
                print(f"EXTRACT {n}/{len(ids)} {time.monotonic() - t0:.0f}s", flush=True)
    return out


def losses(loc, d, device):
    """window KL and point CE for one study."""
    s = loc.scores(d["h"].to(device).float(), d["slot"].to(device))              # [K, 10]
    ls = torch.log_softmax(s, 0)
    qt = torch.zeros_like(s)
    for j in range(len(CF)):
        qt[:, j].index_add_(0, torch.as_tensor(d["w"][j], device=device), torch.as_tensor(d["q"][j], device=device))
    lw = -(qt * ls).sum(0).mean()
    lp = torch.zeros((), device=device)
    if "maps" in d:
        pos = {int(k): i for i, k in enumerate(d["U"])}
        pl = loc.point_logits(d["maps"].to(device))                              # [u, 10, C*C]
        rows = torch.as_tensor([[pos[int(k)] for k in d["w"][j]] for j in range(len(CF))], device=device)   # [10, TOP]
        lg = torch.log_softmax(pl[rows, torch.arange(len(CF), device=device)[:, None]], -1)               # [10, TOP, C*C]
        tg = torch.as_tensor(np.stack([[cell_target(d["R"][j, t], d["Q"][j, t], CFGL["sigma_cells"]) for t in range(TOP)]
                                       for j in range(len(CF))]), device=device, dtype=torch.float32)
        lp = -(tg * lg).sum(-1).mean()
    return lw, lp


def train(loc, feats, ids, val_feats, device):
    opt = torch.optim.AdamW([{"params": [loc.e] + list(loc.pt.parameters()), "lr": CFGL["lr_slot_point"]},
                             {"params": list(loc.att.parameters()), "lr": CFGL["lr_att"]}], weight_decay=CFGL["wd"])
    spe = math.ceil(len(ids) / CFGL["batch"]); total = spe * CFGL["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: CFGL["floor"] + (1 - CFGL["floor"]) * 0.5 * (1 + math.cos(math.pi * t / total)))
    rng = np.random.default_rng(CFGL["seed"]); hist = []
    for ep in range(1, CFGL["epochs"] + 1):
        loc.att.eval(); perm = rng.permutation(len(ids)); tw = tp = 0.0
        for b in range(spe):
            opt.zero_grad(set_to_none=True); lsum = 0
            for i in perm[b * CFGL["batch"]:(b + 1) * CFGL["batch"]]:
                lw, lp = losses(loc, feats[ids[i]], device); lsum = lsum + lw + lp; tw += float(lw); tp += float(lp)
            (lsum / CFGL["batch"]).backward(); opt.step(); sched.step()
        with torch.no_grad():
            vw = np.mean([float(losses(loc, d, device)[0]) for d in val_feats.values()])
        hist.append({"epoch": ep, "train_window_kl": tw / len(ids), "train_point_ce": tp / len(ids), "val_window_kl_vs_ms1": vw})
        print("EPOCH " + json.dumps(hist[-1]), flush=True)
    return hist


def score_fold0(model, loc, ids, meta, tabm, device):
    """fold-0 crop logits per placement + diagnostics."""
    R = {k: {} for k in ("glob", "loc_self_100", "loc_ms1_100", "loc_wp_100", "loc_wm_100", "loc_w_100", "loc_qm_100")}
    diag = {"overlap_wp_ms1": [], "slot_wp": [], "slot_ms1": [], "pt_mm_wp_ms1": [], "equiv": [], "t_locate": [], "t_loc": [], "blank_wp": []}
    med = np.array(xai.QP["median"]); mm = {}
    for n, s in enumerate(ids, 1):
        f, row, mask = meta[s]
        if f not in mm:
            mm[f] = np.load(f, mmap_mode="r")
        vol = np.ascontiguousarray(mm[f][row:row + sub.MAXS])
        with torch.no_grad():
            cs, fm, h, a, g = global_feats(model, vol, mask, device)
        R["glob"][s] = g.cpu().numpy()
        sync(device); t0 = time.perf_counter(); tself = sub.locate(model, fm, a, cs); sync(device)
        diag["t_locate"].append(time.perf_counter() - t0)
        w, q, Rm, Qm = ms1_targets(tabm, s, cs); tms1 = (cs[w], Rm, Qm, q)
        with torch.no_grad():
            t0 = time.perf_counter()
            S, Rl, Ql, A, top = loc.table(h, torch.as_tensor(run.slot_of(cs), device=device), lambda U: fm[U], cs)
            sync(device); diag["t_loc"].append(time.perf_counter() - t0)
        sl = run.slot_of(S)
        twm = (S, med[np.arange(len(CF))[:, None], sl, 0].astype(np.int64), med[np.arange(len(CF))[:, None], sl, 1].astype(np.int64), A)
        _, Rw, Qw, _ = run.points(model, fm, a, cs, top)
        tabs = {"self": tself, "ms1": tms1, "wp": (S, Rl, Ql, A), "wm": twm, "w": (S, Rw, Qw, A),
                "qm": xai.q_tables(model, fm, a, cs, tself, mask)["qm"]}
        with torch.no_grad():
            for k, tb in tabs.items():
                R[f"loc_{k}_100"][s] = run.crop_logits(model, vol, *tb, device).cpu().numpy()
        diag["blank_wp"].append(float((~np.asarray(mask, bool)[S]).mean()))
        diag["overlap_wp_ms1"].append([len(set(top[j]) & set(w[j])) / TOP for j in range(len(CF))])
        diag["slot_wp"].append(sl); diag["slot_ms1"].append(run.slot_of(cs[w]))
        diag["pt_mm_wp_ms1"].append(np.hypot(Rl - Rm, Ql - Qm).mean(1) / run.PX_MM)
        if len(diag["equiv"]) < 3:                       # deployed-path parity: self table == sub.predict_study
            ref = sub.predict_study(model, vol, mask, device, amp=device.type == "cuda")
            gc = g.cpu(); lc = torch.as_tensor(R["loc_self_100"][s])
            mine = torch.where(torch.as_tensor(sub.CMASK), 0.5 * (gc + lc), gc).sigmoid().numpy()
            diag["equiv"].append(float(np.abs(ref[0] - mine).max()))
        if n % 100 == 0 or n == len(ids):
            print(f"SCORE {n}/{len(ids)}", flush=True)
    return R, diag


def main():
    t_start = time.monotonic()
    torch.manual_seed(CFGL["seed"]); np.random.seed(CFGL["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda" or N, "no GPU (apptainer --nv?): refusing an 8 h CPU run"
    OUT.mkdir(parents=True, exist_ok=True)
    meta = xai.run_manifest(Path(os.environ["D96_CACHE"])); tabm = run.ms1_table()
    fold = {r["study_id"]: r["fold"] for r in csv.DictReader(open(os.environ["LOC_FOLDS"]))}
    train_ids = sorted(s for s in meta if fold.get(s) not in (None, "0") and s in tabm)
    val_ids = sorted(s for s in meta if fold.get(s) == "0" and s in tabm)
    assert not set(train_ids) & {s for s in fold if fold[s] == "0"}
    if N:
        train_ids, val_ids = train_ids[:N], val_ids[:N]
    else:
        assert len(train_ids) == 3524 and len(val_ids) == 825, (len(train_ids), len(val_ids))
    model = sub.load_model(os.environ["LOC_WEIGHTS"], device).to(memory_format=torch.contiguous_format)
    frozen = {k: v.clone() for k, v in model.state_dict().items()}
    print(f"L4C-LOC {json.dumps(CFGL)} tag {os.environ.get('LOC_TAG')} train {len(train_ids)} val {len(val_ids)} device {device}", flush=True)
    feats = extract(model, train_ids, meta, tabm, device, True)
    vfeats = extract(model, val_ids, meta, tabm, device, False)
    t_ex = time.monotonic() - t_start
    loc = Locator(model).to(device)
    hist = train(loc, feats, train_ids, vfeats, device)
    torch.save(loc.state_dict(), OUT / "locator.pt")         # saved before scoring: a wall-time stop keeps the weights
    (OUT / "loc_history.json").write_text(json.dumps(hist, indent=1) + "\n")
    del feats
    loc.eval()
    R, diag = score_fold0(model, loc, val_ids, meta, tabm, device)
    assert all(torch.equal(frozen[k], v) for k, v in model.state_dict().items()), "v4 weights changed"
    torch.save(loc.state_dict(), OUT / "locator.pt")
    np.savez_compressed(OUT / f"xai_arrays_{os.environ.get('LOC_TAG', 'm')}.npz", ids=np.array(val_ids),
                        **{k: np.stack([v[s] for s in val_ids]) for k, v in R.items()})
    ov = np.array(diag["overlap_wp_ms1"]); sw, sm = np.array(diag["slot_wp"]), np.array(diag["slot_ms1"])
    log = {"cfg": CFGL, "history": hist, "train": len(train_ids), "val": len(val_ids), "equiv_self_fused_max_abs": diag["equiv"],
           "overlap_wp_ms1_per_finding": ov.mean(0).round(3).tolist(),
           "slot_mix_wp": [np.bincount(sw[:, j].ravel(), minlength=5).tolist() for j in range(len(CF))],
           "slot_mix_ms1": [np.bincount(sm[:, j].ravel(), minlength=5).tolist() for j in range(len(CF))],
           "point_mm_wp_ms1_median": np.median(np.array(diag["pt_mm_wp_ms1"]), 0).round(1).tolist(),
           "ms_locate_median": 1000 * float(np.median(diag["t_locate"])), "ms_locator_median": 1000 * float(np.median(diag["t_loc"])),
           "blank_window_share_wp": float(np.mean(diag["blank_wp"])), "extract_s": round(t_ex, 1), "total_s": round(time.monotonic() - t_start, 1)}
    (OUT / "loc_log.json").write_text(json.dumps(log, indent=1) + "\n")
    print("LOC_DONE " + json.dumps({k: log[k] for k in ("equiv_self_fused_max_abs", "overlap_wp_ms1_per_finding",
                                                         "ms_locate_median", "ms_locator_median", "total_s")}), flush=True)


if __name__ == "__main__":
    main()

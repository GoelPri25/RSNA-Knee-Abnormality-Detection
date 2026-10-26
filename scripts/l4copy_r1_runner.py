"""Exp L4C-R1 (round 1 on v4 l4copy, ledger row L4C-R1): v4 l4copy trained on crops it places itself (MS1-free).

Why (L4C-D, 2026-10-18): l4copy was trained on MS1's crop table but runs on self-located crops at inference (LB 0.939);
self-location costs 0.0022 on fold 0, MS1's PF OA crops miss the patella (self PF crops do not), slice choice matters
(+-2 slices -0.0032) and in-plane position does not (20 mm -0.0001). User 2026-10-18: "in the long run we want MS1 out
of the loop". This runner = scripts/r50_gl4_runner.py (V4_ARM=l4copy, same recipe, loss, readout, SWA) with one change,
the crop placement:
    epochs 1..R1_RANDOM   random crops: per finding a random evaluation window, centre = image centre +-64 px
    later epochs          the model's own table, rebuilt before every epoch from the model at the end of the previous
                          epoch (eval mode): v4_l4copy_submission.locate = per finding the top-4 of the 94 evaluation
                          windows by the global head's attention, gradient x activation centroid on each, attention
                          weights; training draws 1 of the 4 by weight + -16 px centre jitter (v2 / v4 unchanged)
    R1_SJIT=k             + slice jitter: the drawn crop slice moves by a uniform -k..k inside its series slot (kept
                          only when the moved slice and both neighbours are valid and in the same slot)
Evaluation (every epoch and SWA) = the deployed path: self-located table from the current model, all 4 crops weighted.
After SWA: BN refresh with the last training table, a table from the refreshed model on the refresh studies, BN refresh
again with it, then the validation table. Descriptive: SWA scored on MS1's table (val_swa_*_ms1tab.json) and top-1 crops.
Logged per epoch: table build time, plane mix of the validation table's top-1 crops per finding, and table drift vs the
previous epoch (share of training crops whose slice changed, mean |row / col| move of unchanged-slice crops).

Env: as r50_gl4_runner (D96_CACHE, D96_PHASE=r50_gl, D96_SEED, GL_CROPS (descriptive only), GL_OUT, V4_ARM=l4copy | l3copy,
GL_SMOKE=1) + R1_SJIT (default 0), R1_RANDOM (default 2).
"""
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

import r50_gl4_runner as m

r, v2 = m.r, m.v2
OUT, SMOKE, CFG, CF, K = m.OUT, m.SMOKE, m.CFG, m.CF, m.K
assert m.ARM in ("l4copy", "l3copy"), m.ARM
IMG, MAXS, TOP, EVAL_RES = CFG["img"], CFG["maxs"], v2.GL["top"], CFG["eval_res"]
BOUNDS = np.concatenate([[0], np.cumsum(CFG["slot_sizes"])])
R1 = {"version": "l4c_r1_v1", "crops": "self (MS1-free)", "random_epochs": int(os.environ.get("R1_RANDOM", "2")),
      "random_centre_px": 64, "refresh": "before every epoch after the random ones, model at end of previous epoch",
      "locate": "v4_l4copy_submission.locate (top-4 eval windows by attention, grad x act centroid, bf16 global pass)",
      "slice_jitter": int(os.environ.get("R1_SJIT", "0")), "final_bn_refresh": "last train table, then own table"}


# ---------------------------------------------------------------- self-location (the submission's locate, training model)
class Vols(torch.utils.data.Dataset):
    def __init__(self, ids, meta):
        self.ids, self.meta, self._mm = list(ids), meta, {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        mt = self.meta[self.ids[i]]
        if mt["file"] not in self._mm:
            self._mm[mt["file"]] = np.load(mt["file"], mmap_mode="r")
        cs = np.clip(r.centres(mt["mask"], K, False), 1, MAXS - 2)
        return torch.from_numpy(np.ascontiguousarray(self._mm[mt["file"]][mt["row"]:mt["row"] + MAXS])), torch.from_numpy(cs)


def locate(model, fm, a, cs):
    """v4_l4copy_submission.locate: fm [K, 2048, c, c] fp32, a [K, 12], cs [K] -> slices, rows, cols, weights [10, TOP]."""
    dev = fm.device
    W = model.clsW.float()[CF]
    ac = a[:, CF]
    top = ac.topk(TOP, dim=0).indices                                            # [TOP, 10]
    U = torch.unique(top); row = {int(k): j for j, k in enumerate(U.tolist())}
    with torch.enable_grad():
        t = fm[U].detach().clone().requires_grad_(True)
        contrib = ac[U] * (model.norm.float()(t.mean((2, 3))) @ W.T)          # [u, 10], a fixed
        c = t.shape[-1]; px = EVAL_RES / c; gi = torch.arange(c, device=dev, dtype=torch.float32)
        lo = (IMG - EVAL_RES) // 2
        S = np.zeros((len(CF), TOP), np.int64); R = S.copy(); Q = S.copy(); A = np.zeros((len(CF), TOP), np.float32)
        for j in range(len(CF)):
            gr, = torch.autograd.grad(contrib[:, j].sum(), t, retain_graph=j < len(CF) - 1)
            ks = top[:, j].tolist(); rows = [row[k] for k in ks]
            mp = (gr[rows] * t.detach()[rows]).sum(1).clamp_min(0)
            tot = mp.sum((1, 2)); ok = tot > 0; tot = tot.clamp_min(1e-12)
            cy = torch.where(ok, (mp.sum(2) * gi).sum(1) / tot, torch.full_like(tot, (c - 1) / 2))
            cx = torch.where(ok, (mp.sum(1) * gi).sum(1) / tot, torch.full_like(tot, (c - 1) / 2))
            S[j] = cs[ks]
            R[j] = np.round((lo + (cy + 0.5) * px).cpu().numpy()).astype(np.int64)
            Q[j] = np.round((lo + (cx + 0.5) * px).cpu().numpy()).astype(np.int64)
            A[j] = ac[top[:, j], j].detach().cpu().numpy()
    return S, R, Q, A


def full12(*arrs):
    """[10, TOP] rows of the cropped findings -> [12, TOP] (crop_table layout; the 2 uncropped rows are never read)."""
    out = []
    for x in arrs:
        y = np.repeat(x[:1], 12, 0); y[CF] = x; out.append(y)
    return tuple(out)


@torch.no_grad()
def self_table(model, ids, meta, dev):
    """study -> (slice, row, col, attention weight) [12, TOP] from the model's own global pass (eval mode)."""
    model.eval()
    lo = (IMG - EVAL_RES) // 2
    mean, std = r.MEAN.to(dev), r.STD.to(dev)
    dl = torch.utils.data.DataLoader(Vols(ids, meta), batch_size=1, num_workers=CFG["workers"], pin_memory=True)
    tab = {}
    for s, (vol, cs) in zip(ids, dl):
        vol, cs = vol[0].to(dev, non_blocking=True), cs[0].numpy()
        ix = torch.as_tensor(np.stack([cs - 1, cs, cs + 1], 1), device=dev)
        xg = vol[ix][..., lo:lo + EVAL_RES, lo:lo + EVAL_RES].float().div_(255.0)
        xg = ((xg - mean) / std).contiguous(memory_format=torch.channels_last)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            fm = model.backbone.forward_features(xg)
        fm = fm.float()
        h = model.norm.float()(fm.mean((2, 3)))
        a = torch.softmax(model.att.float()(h), dim=0)
        tab[s] = full12(*locate(model, fm, a, cs))
    return tab


def plane_mix(tab, ids):
    """per cropped finding: share of top-1 crops on each series slot (SagF, SagNF, CorF, CorNF, Ax)."""
    out = {}
    for k in CF:
        sl = np.array([tab[s][0][k, tab[s][3][k].argmax()] for s in ids])
        out[r.TARGETS[k]] = np.bincount(r.SLOT_OF[sl], minlength=5).astype(float).round(0).tolist()
    return out


def drift(old, new, ids):
    if old is None:
        return None
    S0 = np.stack([old[s][0][CF] for s in ids]); S1 = np.stack([new[s][0][CF] for s in ids])
    same = S0 == S1
    mv = np.hypot(*[np.stack([old[s][i][CF] for s in ids]) - np.stack([new[s][i][CF] for s in ids]) for i in (1, 2)])
    return {"slice_changed": round(float(1 - same.mean()), 4),
            "inplane_move_px_same_slice": round(float(mv[same].mean()) if same.any() else float("nan"), 2)}


# ---------------------------------------------------------------- training data: random / own table + slice jitter
class R1Studies(v2.GLStudies):
    """v2.GLStudies (training branch) with tab None = random crops, and optional slice jitter inside the series slot."""

    def __getitem__(self, i):
        x, sag, y, w, ctx = r.Studies.__getitem__(self, i)
        s = self.ids[i]; mt = self.meta[s]
        vol = self._mm[mt["file"]][mt["row"]:mt["row"] + MAXS]
        n, A = len(CF), np.arange(len(CF))
        if self.tab is None:
            cs = np.clip(r.centres(mt["mask"], K, False), 1, MAXS - 2)
            S = cs[np.random.randint(0, len(cs), n)]
            R, Q = IMG // 2 + np.random.randint(-R1["random_centre_px"], R1["random_centre_px"] + 1, size=(2, n))
        else:
            sl, rr, cc, at = (a[CF] for a in self.tab[s])
            j = np.array([np.random.choice(TOP, p=at[q] / at[q].sum()) for q in range(n)])
            jit = np.random.randint(-v2.GL["jitter_px"], v2.GL["jitter_px"] + 1, size=(2, n))
            S, R, Q = sl[A, j].copy(), rr[A, j] + jit[0], cc[A, j] + jit[1]
            if R1["slice_jitter"]:
                d = np.random.randint(-R1["slice_jitter"], R1["slice_jitter"] + 1, n)
                c = np.clip(S + d, 1, MAXS - 2); slot = r.SLOT_OF[S]
                ok = (r.SLOT_OF[c - 1] == slot) & (r.SLOT_OF[c + 1] == slot) & mt["mask"][c - 1] & mt["mask"][c] & mt["mask"][c + 1]
                S = np.where(ok, c, S)
        fid, wt = CF, np.ones(n, np.float32)
        h = v2.BOXN[fid] // 2
        R, Q = np.clip(R, h, IMG - h), np.clip(Q, h, IMG - h)
        xc = np.ascontiguousarray(vol[np.stack([S - 1, S, S + 1], 1)])
        sagc = np.isin(r.SLOT_OF[S], r.SAGITTAL_SLOTS)
        ctxc = np.stack([mt["pos"][S], mt["known"][S], r.SLOT_OF[S].astype(np.float32)], 1).astype(np.float32)
        cen = np.stack([R, Q], 1).astype(np.float32)
        return (x, sag, y, w, ctx, torch.from_numpy(xc), torch.from_numpy(sagc), torch.from_numpy(ctxc),
                torch.from_numpy(cen), torch.from_numpy(fid.astype(np.int64)), torch.from_numpy(wt))


def gpu_name():
    return torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"


def mem_gb():
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


def save_table(path, tab, ids):
    np.savez_compressed(path, ids=np.array(ids), **{k: np.stack([tab[s][i] for s in ids]) for i, k in enumerate("SRQA")})


def main():
    started = time.monotonic()
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"]); random.seed(CFG["seed"])
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")        # cpu: GL_SMOKE CPU test only
    labels = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
    ref = r.load_labels(r.ASSETS / "validation_labels.json", "validation")
    assert r.sha(r.ASSETS / "folds.csv") == CFG["sha256"]["folds"]
    fold = {x["study_id"]: x["fold"] for x in csv.DictReader((r.ASSETS / "folds.csv").open())}
    meta = r.read_manifest()
    train_ids = sorted(s for s in labels if fold[s] != "0"); val_ids = sorted(s for s in labels if fold[s] == "0")
    if SMOKE:                                            # any cache (the CPU smoke cache holds no fold-0 study)
        have = sorted(s for s in labels if s in meta); train_ids, val_ids = have[:6], have[6:10]
    else:
        assert len(train_ids) == 3524 and len(val_ids) == 825 and set(val_ids) == set(ref)
    gap_ids = sorted(random.Random(CFG["seed"] + 1).sample(train_ids, 2 if SMOKE else m.G4["train_gap_studies"]))
    print(f"L4C-R1 {m.G4['version']} seed {CFG['seed']} ({CFG['version']}) {json.dumps(R1)}: train {len(train_ids)}, "
          f"validation {len(val_ids)}; GPU {gpu_name()} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    model, opt, clip = m.build(dev)
    ps = model.param_sets()
    print("PARAMS " + json.dumps({k: sum(p.numel() for p in v) for k, v in ps.items()}), flush=True)
    spe = math.ceil(len(train_ids) / (CFG["studies_per_step"] * CFG["accum"]))
    total, warm, floor = spe * CFG["epochs"], spe * CFG["warmup_epochs"], CFG["lr_floor"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else floor + (1 - floor) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))

    vids, tids = val_ids, train_ids
    start, history, rec = 1, [], OUT / "recovery.pt"
    if rec.is_file():
        st = torch.load(rec, map_location="cpu", weights_only=False)
        assert st["g4"] == m.G4 and st["cfg"] == CFG and st["r1"] == R1, "recovery config differs"
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, history = st["epoch"] + 1, st["history"]
        print(f"RESUMED after epoch {st['epoch']}", flush=True)
    train_ds = R1Studies(tids, meta, labels, CFG["train_windows"], True, None)
    cmask = torch.from_numpy(m.CMASK.astype(np.float32)).to(dev)
    last = (R1["random_epochs"] + 1) if SMOKE else CFG["epochs"]
    prev = None
    for epoch in range(start, last + 1):
        tt = time.monotonic()
        if epoch > R1["random_epochs"]:
            train_ds.tab = self_table(model, tids, meta, dev)
        rec_ = {"epoch": epoch, "crops": "random" if train_ds.tab is None else "self",
                "table_s": round(time.monotonic() - tt, 1), "table_drift": drift(prev, train_ds.tab, tids) if train_ds.tab else None}
        prev = train_ds.tab
        model.train(); t0 = time.monotonic()
        dl = torch.utils.data.DataLoader(train_ds, batch_size=CFG["studies_per_step"], shuffle=True,
                                         num_workers=CFG["workers"], pin_memory=True,
                                         worker_init_fn=lambda w: np.random.seed((CFG["seed"] * 1000 + epoch * 100 + w) % 2**32))
        run = np.zeros(3); n = 0; gn = np.zeros(len(clip)); nst = 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl):
            b = batch[0].shape[0]
            parts = m.train_step_loss(model, batch, dev, cmask)
            loss = m.G4["global_w"] * parts[0] + m.G4["aux_local"] * parts[1]
            (loss / CFG["accum"]).backward()
            if (i + 1) % CFG["accum"] == 0 or i + 1 == len(dl):
                gn += [float(torch.nn.utils.clip_grad_norm_(c, CFG["grad_clip"])) for c in clip]; nst += 1
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run += np.append(parts.detach().cpu().numpy(), loss.item()) * b; n += b
            if (i + 1) % 400 == 0 or (SMOKE and i + 1 == 4):
                print(f"  epoch {epoch} step {i + 1}/{len(dl)} loss global/local {run[0] / n:.4f}/{run[1] / n:.4f} "
                      f"{(time.monotonic() - t0) / n:.3f}s/study mem {mem_gb():.1f}GB",
                      flush=True)
        rec_.update({"train_loss": dict(zip(("global", "local", "total"), (run / n).round(5).tolist())),
                     "grad_norm": (gn / max(1, nst)).round(4).tolist(), "train_s": round(time.monotonic() - t0, 1),
                     "lr": [g["lr"] for g in opt.param_groups]})
        te = time.monotonic(); vtab = self_table(model, vids, meta, dev); p = m.predict(model, vids, meta, vtab, dev)
        rec_["eval_s_per_study"] = round((time.monotonic() - te) / len(vids), 3)
        rec_["val_plane_mix_top1"] = plane_mix(vtab, vids)
        pt = m.predict(model, gap_ids, meta, self_table(model, gap_ids, meta, dev), dev)
        rec_["mem_gb"] = round(mem_gb(), 1)
        if not SMOKE:
            rec_.update(m.score(p, val_ids, ref, labels))
            tr = {}
            for j, nm in enumerate(m.HEADS):
                a, per = r.macro_auc(pt[j], gap_ids, labels)
                tr[f"train_lv2_{nm}"] = a
                tr[f"train_lv2_{nm}10"] = float(np.nanmean([per[r.TARGETS[k]] for k in CF]))
                tr[f"train_bce_{nm}"] = r.val_bce(pt[j], gap_ids, labels)
            rec_.update(tr)
            rec_["gap_local10"] = tr["train_lv2_local10"] - rec_["lv2_local10"]
            rec_["gap_global"] = tr["train_lv2_global"] - rec_["lv2_global"]
            for j, nm in enumerate(m.HEADS):
                r.save_predictions(OUT / f"val_epoch{epoch:02d}_{nm}.json", val_ids, p[j], f"L4C-R1 {m.ARM} epoch {epoch}, {nm}")
        history.append(rec_)
        print(f"EPOCH {epoch} " + json.dumps({k: v for k, v in rec_.items() if not k.startswith("lv2_per")}), flush=True)
        if epoch in CFG["swa_epochs"] or SMOKE:
            torch.save(model.state_dict(), OUT / f"weights_epoch{epoch:02d}.pt")
        if not SMOKE:
            torch.save({"g4": m.G4, "cfg": CFG, "r1": R1, "model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": epoch, "history": history}, OUT / "recovery.tmp")
            os.replace(OUT / "recovery.tmp", rec)
        (OUT / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    # ---- primary: SWA, BN refresh (last training table, then the refreshed model's own table), self-located evaluation
    swa = [last] if SMOKE else CFG["swa_epochs"]
    sds = [torch.load(OUT / f"weights_epoch{e:02d}.pt", map_location="cpu") for e in swa]
    avg = {k: (sum(s[k].float() for s in sds) / len(sds)).to(sds[-1][k].dtype) if sds[-1][k].is_floating_point()
           else sds[-1][k] for k in sds[-1]}
    model.load_state_dict(avg, strict=True)
    tr = time.monotonic()
    refresh = random.Random(CFG["seed"]).sample(tids, min(len(tids), 8) if SMOKE else CFG["bn_refresh_studies"])
    ttab = train_ds.tab if train_ds.tab is not None else self_table(model, refresh, meta, dev)
    m.bn_refresh(model, refresh, meta, labels, ttab, dev)
    m.bn_refresh(model, refresh, meta, labels, self_table(model, refresh, meta, dev), dev)
    print(f"BN_REFRESH 2 x {len(refresh)} studies {time.monotonic() - tr:.0f}s", flush=True)
    tag = f"swa{swa[0]}_{swa[-1]}"
    torch.save(model.state_dict(), OUT / f"weights_{tag}.pt")
    vtab = self_table(model, vids, meta, dev)
    save_table(OUT / "val_swa_self_table.npz", vtab, vids)
    p = m.predict(model, vids, meta, vtab, dev)
    res = m.score(p, vids, ref, labels) if not SMOKE else {}
    res["val_plane_mix_top1"] = plane_mix(vtab, vids)
    for j, nm in enumerate(m.HEADS):
        r.save_predictions(OUT / f"val_swa_{nm}.json", vids, p[j], f"L4C-R1 {m.ARM} SWA {swa}, {nm}, self-located crops")
    pt = m.predict(model, vids, meta, vtab, dev, top1=True)                      # descriptive: top-1 crop per finding
    r.save_predictions(OUT / "val_swa_fused_top1.json", vids, pt[0], f"L4C-R1 {m.ARM} SWA, fused, top-1 self crop")
    mtab = v2.crop_table(meta)                                                    # descriptive: MS1's placement
    pm = m.predict(model, vids, meta, mtab, dev)
    for j, nm in enumerate(m.HEADS):
        r.save_predictions(OUT / f"val_swa_{nm}_ms1tab.json", vids, pm[j], f"L4C-R1 {m.ARM} SWA, {nm}, MS1 table")
    if not SMOKE:
        res["lv2_fused_top1"] = r.macro_auc(pt[0], vids, labels)[0]
        res["lv2_fused_ms1tab"] = r.macro_auc(pm[0], vids, labels)[0]
    summary = {"g4": m.G4, "r1": R1, "cfg": CFG, "history": history, "swa": res, "seconds": round(time.monotonic() - started, 1)}
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("RUN_SUMMARY " + json.dumps({k: v for k, v in res.items() if not k.startswith("lv2_per")}), flush=True)
    if SMOKE:
        print(f"SMOKE_DONE {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()

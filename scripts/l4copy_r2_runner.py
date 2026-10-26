"""Exp L4C-R2 (round 2 on v4 l4copy, ledger row L4C-R2): warm-start the LB-0.939 l4copy (run_r50_gl4_mlx/v4_l4copy_s2026,
weights_swa21_25.pt) and fine-tune its crop branch on the crops it places itself.

Why (L4C-R1, 2026-10-19): training from scratch on its own placement gave a global head as good as the parent's, but the
crop branch chased a moving table (82-94% of crop slices changed between epochs up to epoch 10) and added ~0 over the
global (fused - global -0.001 vs +0.011 for the parent at epoch 10). The parent's crop branch already works but was
trained on MS1's placement (self-located inference costs -0.0022, L4C-D). Starting from it removes both problems: the
global head (and so the placement) is already settled, and only the crop branch has to adapt to its deployed inputs.
Arms (R2_ARM):
    crop  global path frozen (stem-layer3 shared weights + BN affine + global statistics, layer4, global head): placement
          is the parent's, computed ONCE (submission locate) for training and validation studies. Trained: layer4_c,
          norm_l, clsV / clsvb (crop BN statistics in the shared stem-layer3 follow the crops). Loss = BCE_cropped(local).
          The global logits never change, so fused = sigmoid((g_parent + l) / 2).
    all   every weight, both losses (v4: BCE(global) + 0.5 x BCE_cropped(local)), the own table rebuilt before every epoch
          from the current model (l4copy_r1_runner.self_table).
Both: R2_EPOCHS (default 6) epochs, AdamW wd 0.05, peak LR R2_LR (default 3e-5) for backbone-side weights and 10x for
heads, linear warm-up over the first 10% of steps, cosine to 1%; SWA over the last 3 epochs; crop BN statistics
refreshed after SWA (arm all: both sets, l4copy_r1_runner's two-pass refresh). Training items = l4copy_r1_runner.R1Studies
(1 of the 4 self crops by weight, +-16 px, optional slice jitter R1_SJIT). Fixed epoch count: no epoch is picked on fold 0.
INIT_CHECK: before any update, fold 0 scored on the parent's own table must reproduce the parent's self-located fused
(0.9453, l4copy_diag/modal_v1) within bf16 noise.

Env: as l4copy_r1_runner (D96_CACHE, D96_PHASE=r50_gl, D96_SEED, GL_CROPS (descriptive MS1 scoring), GL_OUT, V4_ARM=l4copy,
GL_SMOKE=1) + R2_INIT (parent weights), R2_ARM=crop | all, R2_EPOCHS, R2_LR.
"""
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch

import l4copy_r1_runner as R1

m, r, v2 = R1.m, R1.r, R1.v2
OUT, SMOKE, CFG, CF = m.OUT, m.SMOKE, m.CFG, m.CF
assert m.ARM == "l4copy", m.ARM
R2 = {"version": "l4c_r2_v1", "arm": os.environ.get("R2_ARM", "crop"), "init": os.environ.get("R2_INIT", ""),
      "epochs": int(os.environ.get("R2_EPOCHS", "6")), "lr": float(os.environ.get("R2_LR", "3e-5")), "head_lr_x": 10,
      "warmup_frac": 0.1, "floor": 0.01, "swa_last": 3, "slice_jitter": R1.R1["slice_jitter"]}
assert R2["arm"] in ("crop", "all") and R2["init"], R2
CROP_TRAIN = ("layer4_c.", "norm_l.", "clsV", "clsvb")


def local_logits(model, xcr, b, fid, wc):
    """the local half of RaptorGL4.forward: crop features -> own LayerNorm -> V_n -> weighted per finding [b, 12]."""
    kc = xcr.shape[0] // b
    hc = model.norm_l(model.encode(xcr, True).view(b, kc, -1))
    lc = ((hc * model.clsV[fid]).sum(-1) + model.clsvb[fid]).float()
    return torch.zeros(b, 12, device=xcr.device).scatter_add(1, fid, lc * wc)


@torch.no_grad()
def crop_bn_refresh(model, ids, meta, labels, tab, dev):
    """crop statistics only (global statistics stay the parent's): all 4 own crops per finding, as m.bn_refresh."""
    for bn in model.bns + model.cbns:
        bn.crop_mean.zero_(); bn.crop_var.fill_(1); bn.crop_nbt.zero_(); bn.momentum = None
    model.train()
    ds = v2.GLStudies(ids, meta, labels, 1, False, tab)
    for x, sag, _, _, _, xc, sagc, _, cen, fid, _ in torch.utils.data.DataLoader(ds, batch_size=2, num_workers=CFG["workers"]):
        _, xcr, _ = v2.prepare_gl(x, sag, xc, sagc, cen, fid, False, dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            model.encode(xcr, True)


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
    if SMOKE:
        have = sorted(s for s in labels if s in meta); train_ids, val_ids = have[:6], have[6:10]
    else:
        assert len(train_ids) == 3524 and len(val_ids) == 825 and set(val_ids) == set(ref)
    print(f"L4C-R2 {json.dumps(R2)} seed {CFG['seed']}: train {len(train_ids)}, validation {len(val_ids)}; "
          f"GPU {R1.gpu_name()}", flush=True)

    model, _, _ = m.build(dev, pretrained=False)
    model.load_state_dict(torch.load(R2["init"], map_location="cpu", weights_only=True), strict=True)
    vids, tids = val_ids, train_ids
    t0 = time.monotonic(); vtab = R1.self_table(model, vids, meta, dev)
    p0 = m.predict(model, vids, meta, vtab, dev)
    init = {"table_s_per_study": round((time.monotonic() - t0) / len(vids), 3)}
    if not SMOKE:
        init.update({k: v for k, v in m.score(p0, vids, ref, labels).items() if not k.startswith("lv2_per")})
        for j, nm in enumerate(m.HEADS):
            r.save_predictions(OUT / f"val_init_{nm}.json", vids, p0[j], f"L4C-R2 init (parent), {nm}, self-located crops")
    print("INIT_CHECK " + json.dumps(init), flush=True)

    if R2["arm"] == "crop":
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith(CROP_TRAIN))
        tt = time.monotonic(); ttab = R1.self_table(model, tids, meta, dev)          # fixed: the global path is frozen
        print(f"TRAIN_TABLE {len(tids)} studies {time.monotonic() - tt:.0f}s", flush=True)
    else:
        ttab = None
    bbp = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(("backbone.", "layer4_c."))]
    hp = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(("backbone.", "layer4_c."))]
    opt = torch.optim.AdamW([{"params": bbp, "lr": R2["lr"]}, {"params": hp, "lr": R2["lr"] * R2["head_lr_x"]}],
                            weight_decay=CFG["wd"])
    trainable = bbp + hp
    print("TRAINABLE " + json.dumps({"backbone_side": sum(p.numel() for p in bbp), "heads": sum(p.numel() for p in hp)}), flush=True)
    k = 1 if R2["arm"] == "crop" else CFG["train_windows"]                          # crop arm: globals unused
    train_ds = R1.R1Studies(tids, meta, labels, k, True, ttab)
    spe = math.ceil(len(tids) / (CFG["studies_per_step"] * CFG["accum"]))
    total = spe * R2["epochs"]; warm = max(1, int(total * R2["warmup_frac"])); floor = R2["floor"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else floor + (1 - floor) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    cmask = torch.from_numpy(m.CMASK.astype(np.float32)).to(dev)
    last = 2 if SMOKE else R2["epochs"]
    swa = list(range(max(1, last - R2["swa_last"] + 1), last + 1))
    history, prev = [], ttab
    for epoch in range(1, last + 1):
        rec_ = {"epoch": epoch}
        if R2["arm"] == "all":
            tt = time.monotonic(); train_ds.tab = R1.self_table(model, tids, meta, dev)
            rec_.update({"table_s": round(time.monotonic() - tt, 1), "table_drift": R1.drift(prev, train_ds.tab, tids)})
            prev = train_ds.tab
        model.train(); t0 = time.monotonic()
        dl = torch.utils.data.DataLoader(train_ds, batch_size=CFG["studies_per_step"], shuffle=True,
                                         num_workers=CFG["workers"], pin_memory=True,
                                         worker_init_fn=lambda w: np.random.seed((CFG["seed"] * 1000 + epoch * 100 + w) % 2**32))
        run = np.zeros(3); n = 0; gn = 0.0; nst = 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl):
            b = batch[0].shape[0]
            if R2["arm"] == "crop":
                x, sag, y, w, _, xc, sagc, _, cen, fid, wc = batch
                _, xcr, _ = v2.prepare_gl(x, sag, xc, sagc, cen, fid, True, dev)
                fid, wc, y, w = v2.to_dev(dev, fid, wc, y, w)
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    l = local_logits(model, xcr, b, fid, wc)
                loss = v2.bce(l, y, w * cmask); parts = torch.stack([loss.detach() * 0, loss.detach()])
            else:
                parts = m.train_step_loss(model, batch, dev, cmask)
                loss = m.G4["global_w"] * parts[0] + m.G4["aux_local"] * parts[1]
            (loss / CFG["accum"]).backward()
            if (i + 1) % CFG["accum"] == 0 or i + 1 == len(dl):
                gn += float(torch.nn.utils.clip_grad_norm_(trainable, CFG["grad_clip"])); nst += 1
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run += np.append(parts.detach().float().cpu().numpy(), loss.item()) * b; n += b
            if (i + 1) % 400 == 0:
                print(f"  epoch {epoch} step {i + 1}/{len(dl)} loss global/local {run[0] / n:.4f}/{run[1] / n:.4f} "
                      f"{(time.monotonic() - t0) / n:.3f}s/study mem {R1.mem_gb():.1f}GB", flush=True)
        rec_.update({"train_loss": dict(zip(("global", "local", "total"), (run / n).round(5).tolist())),
                     "grad_norm": round(gn / max(1, nst), 4), "train_s": round(time.monotonic() - t0, 1),
                     "lr": [g["lr"] for g in opt.param_groups]})
        if R2["arm"] == "all":
            vtab = R1.self_table(model, vids, meta, dev)
        p = m.predict(model, vids, meta, vtab, dev)
        rec_["val_plane_mix_top1"] = R1.plane_mix(vtab, vids)
        if not SMOKE:
            rec_.update(m.score(p, vids, ref, labels))
            for j, nm in enumerate(m.HEADS):
                r.save_predictions(OUT / f"val_epoch{epoch:02d}_{nm}.json", vids, p[j], f"L4C-R2 {R2['arm']} epoch {epoch}, {nm}")
        history.append(rec_)
        print(f"EPOCH {epoch} " + json.dumps({k: v for k, v in rec_.items() if not k.startswith("lv2_per")}), flush=True)
        if epoch in swa:
            torch.save(model.state_dict(), OUT / f"weights_epoch{epoch:02d}.pt")
        (OUT / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    # ---- SWA over the last epochs, crop statistics refreshed (arm all: both sets, R1's two passes), self-located eval
    sds = [torch.load(OUT / f"weights_epoch{e:02d}.pt", map_location="cpu") for e in swa]
    avg = {k: (sum(s[k].float() for s in sds) / len(sds)).to(sds[-1][k].dtype) if sds[-1][k].is_floating_point()
           else sds[-1][k] for k in sds[-1]}
    model.load_state_dict(avg, strict=True)
    refresh = random.Random(CFG["seed"]).sample(tids, min(len(tids), 8) if SMOKE else CFG["bn_refresh_studies"])
    if R2["arm"] == "crop":
        crop_bn_refresh(model, refresh, meta, labels, ttab, dev)
    else:
        m.bn_refresh(model, refresh, meta, labels, train_ds.tab, dev)
        m.bn_refresh(model, refresh, meta, labels, R1.self_table(model, refresh, meta, dev), dev)
        vtab = R1.self_table(model, vids, meta, dev)
    tag = f"swa{swa[0]}_{swa[-1]}"
    torch.save(model.state_dict(), OUT / f"weights_{tag}.pt")
    R1.save_table(OUT / "val_swa_self_table.npz", vtab, vids)
    p = m.predict(model, vids, meta, vtab, dev)
    res = m.score(p, vids, ref, labels) if not SMOKE else {}
    for j, nm in enumerate(m.HEADS):
        r.save_predictions(OUT / f"val_swa_{nm}.json", vids, p[j], f"L4C-R2 {R2['arm']} SWA {swa}, {nm}, self-located crops")
    pm = m.predict(model, vids, meta, v2.crop_table(meta), dev)                    # descriptive: MS1's placement
    r.save_predictions(OUT / "val_swa_fused_ms1tab.json", vids, pm[0], f"L4C-R2 {R2['arm']} SWA, fused, MS1 table")
    if not SMOKE:
        res["lv2_fused_ms1tab"] = r.macro_auc(pm[0], vids, labels)[0]
        res["init_lv2_fused"] = init["lv2_fused"]; res["swa_minus_init"] = res["lv2_fused"] - init["lv2_fused"]
    summary = {"g4": m.G4, "r2": R2, "cfg": CFG, "init": init, "history": history, "swa": res,
               "seconds": round(time.monotonic() - started, 1)}
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("RUN_SUMMARY " + json.dumps({k: v for k, v in res.items() if not k.startswith("lv2_per")}), flush=True)
    if SMOKE:
        print(f"SMOKE_DONE {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""CPU tests for scripts/r50_gl4_runner.py (real timm, random init, no network). Run in localhost/rsna-mlx-train:v1:
podman run --rm --network=none --read-only --tmpfs /tmp -v <repo>/scripts:/w:ro -v <assets>:/a:ro -v <smoke d96>:/c/shard_0:ro
  -v <slim crops>:/cp:ro -w /w localhost/rsna-mlx-train:v1 python -u test_r50_gl4_cpu.py
Each arm runs in its own process (the arm is fixed at import). T0 encode path = timm forward; T1 local loss reaches
exactly the arm's intended parameters; T2 (adapter) one optimiser step with and without the local loss leaves every
shared parameter and every global BN statistic bitwise equal; T3 a crop pass never touches global BN statistics; T4
readout; T5 dense-delta SWA = factor path; T6 real cache batch end-to-end (train step + predict); T7 r50_lv2 / seed CFG.
"""
import os
import subprocess
import sys

ENV = {"D96_PHASE": "r50_gl", "GL_OUT": "/tmp/out", "GL_CROPS": "/cp", "D96_CACHE": "/c", "D96_ASSETS": "/a",
       "OMP_NUM_THREADS": "8"}


def arm_test():
    import copy
    import numpy as np
    import torch
    import r50_gl4_runner as m
    import coatnet_d96_runner as r
    torch.manual_seed(0)
    arm, dev = m.ARM, torch.device("cpu")
    model, opt, clip = m.build(dev, pretrained=False)
    ps = model.param_sets(); n = {k: sum(p.numel() for p in v) for k, v in ps.items()}
    with torch.no_grad():                                    # timm random init zeroes each block's last BN gamma
        for bn in model.bns + model.cbns:                    # (zero_init_last), which zeroes residual-branch grads
            bn.weight.uniform_(0.5, 1.5)
            if bn.mode == "own":
                bn.crop_weight.uniform_(0.5, 1.5)
    print(f"[{arm}] params {n}")
    if arm == "adapter":
        lora = sum(p.numel() for nm, p in model.named_parameters() if "lora_" in nm)
        assert 1.0e6 < lora < 1.2e6, lora
        for c in model.convs:                                # B != 0 so A gets gradient in T1
            if c.rank:
                torch.nn.init.normal_(c.lora_B, std=1e-3)
    bb = model.backbone
    x = torch.randn(4, 3, 64, 64)
    # T0: the split forward equals timm's forward (flags off, eval)
    model.eval()
    with torch.no_grad():
        ref = bb(x); h = bb.forward_head(bb.layer4(bb.layer3(bb.layer2(bb.layer1(model.stem(x))))))
    assert torch.allclose(ref, h, atol=1e-5), (ref - h).abs().max()
    # T3: crop pass in train mode leaves global BN statistics untouched, moves crop statistics
    model.train()
    gstats = {k: v.clone() for k, v in model.state_dict().items() if k.endswith(("running_mean", "running_var"))
              and not k.startswith("layer4_c.")}
    cstats = {k: v.clone() for k, v in model.state_dict().items() if k.endswith("crop_mean")}
    model.encode(x, True)
    for k, v in model.state_dict().items():
        if k in gstats:
            assert torch.equal(v, gstats[k]), f"T3 global stat moved by crop pass: {k}"
    moved = sum(not torch.equal(model.state_dict()[k], v) for k, v in cstats.items())
    assert moved > 0
    print(f"[{arm}] T0 ok, T3 ok (crop stats moved in {moved} BNs)")
    # T1: local loss alone -> which parameters get gradient
    b, k, kc = 2, 3, 10
    xg, xc = torch.randn(b * k, 3, 64, 64), torch.randn(b * kc, 3, 64, 64)
    fid = torch.as_tensor(m.CF).repeat(b, 1); wc = torch.ones(b, kc)
    y = torch.rand(b, 12); w = torch.ones(b, 12); cm = torch.as_tensor(m.CMASK, dtype=torch.float32)
    opt.zero_grad(set_to_none=True)
    g, l = model(xg, b, xc, fid, wc)
    m.v2.bce(l, y, w * cm).backward()
    named = dict(model.named_parameters())
    got = {nm for nm, p in named.items() if p.grad is not None and p.grad.abs().sum() > 0}
    if arm == "adapter":
        shared = {nm for nm, p in named.items() if any(p is q for q in ps["bb"] + ps["gh"])}
        assert not any(named[nm].grad is not None for nm in shared), "T1 local loss reached a shared parameter"
        loc = {nm for nm, p in named.items() if any(p is q for q in ps["lb"] + ps["lh"])}
        assert loc <= got, f"T1 local params without gradient: {sorted(loc - got)[:5]}"
    elif arm == "l4copy":
        assert not any(named[nm].grad is not None for nm in named if nm.startswith("backbone.layer4.")), "T1 layer4"
        assert any(nm.startswith("backbone.layer3.") for nm in got) and any(nm.startswith("layer4_c.") for nm in got)
        assert not any(nm.startswith(("clsW", "att.", "norm.")) for nm in got)
    else:
        assert any(nm.startswith("backbone.layer1.") for nm in got) and not any(nm.startswith(("clsW", "att.")) for nm in got)
    print(f"[{arm}] T1 ok: local loss -> {len(got)} params with gradient")
    # T4: readout
    z = m.readout(g, l)
    assert torch.equal(z[:, ~torch.as_tensor(m.CMASK)], g[:, ~torch.as_tensor(m.CMASK)])
    assert torch.allclose(z[:, m.CF], 0.5 * (g + l)[:, m.CF])
    print(f"[{arm}] T4 ok")
    # T2 (adapter): one step with / without the local loss; shared params and global stats bitwise equal
    if arm == "adapter":
        sd0 = copy.deepcopy(model.state_dict()); outs = []
        for use_local in (False, True):
            model.load_state_dict(sd0); mm, op, cl = model, *m.build(dev, pretrained=False)[1:]
            op = torch.optim.AdamW([{"params": q["params"], "lr": q["lr"]} for q in opt.param_groups], weight_decay=0.05)
            mm.train(); op.zero_grad(set_to_none=True)
            torch.manual_seed(1)
            g, l = mm(xg, b, xc, fid, wc)
            loss = m.v2.bce(g, y, w) + (0.5 * m.v2.bce(l, y, w * cm) if use_local else 0)
            loss.backward()
            for c in clip:
                torch.nn.utils.clip_grad_norm_(c, 1e-3)      # tiny max norm: clipping is active
            op.step()
            outs.append(copy.deepcopy(mm.state_dict()))
        names = {nm for nm, p in named.items() if any(p is q for q in ps["bb"] + ps["gh"])}
        for kk in outs[0]:
            same = torch.equal(outs[0][kk], outs[1][kk])
            if kk in names or kk.endswith(("running_mean", "running_var")):
                assert same, f"T2 shared state differs with the local loss: {kk}"
        diff = [kk for kk in outs[0] if not torch.equal(outs[0][kk], outs[1][kk])]
        assert diff and all(("lora_" in d or "crop_" in d or d.startswith(("norm_l", "clsV", "clsvb"))) for d in diff), diff[:5]
        print(f"[adapter] T2 ok: {len(diff)} local tensors differ, shared params + global stats bitwise equal")
        # T5: dense-delta SWA of a single epoch equals the factor path
        sd = model.state_dict()
        avg = dict(sd); avg.update(m.lora_dense_average([sd]))
        model.eval()
        with torch.no_grad():
            f1 = model.encode(x, True)
            model.load_state_dict(avg); model.set_dense(); f2 = model.encode(x, True)
        assert torch.allclose(f1, f2, atol=1e-4), (f1 - f2).abs().max()
        print("[adapter] T5 ok")
    # T6: real cache batch end-to-end (train step on 2 studies + predict on 1)
    r.CFG["shards"] = 1; r.CFG["workers"] = 0
    meta = r.read_manifest()
    from pathlib import Path
    sub = Path("/tmp/cp"); sub.mkdir(exist_ok=True)              # the slim table restricted to the smoke studies
    for f in sorted(Path("/cp").glob("cp_*.npz")):
        z = np.load(f); keep = np.isin(z["ids"], list(meta))
        if keep.any():
            np.savez(sub / f.name, ids=z["ids"][keep], centres=z["centres"][keep], a_top=z["a_top"][keep])
    m.v2.CP = sub; tab = m.v2.crop_table(meta)
    labels = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
    ids = sorted(s for s in meta if s in labels and s in tab)[:2]
    assert len(ids) == 2, "smoke cache studies not labelled / not in the crop table"
    ds = m.v2.GLStudies(ids, meta, labels, r.CFG["train_windows"], True, tab)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=2)))
    model, opt, clip = m.build(dev, pretrained=False); model.train()
    parts = m.train_step_loss(model, batch, dev, torch.as_tensor(m.CMASK, dtype=torch.float32))
    (parts[0] + 0.5 * parts[1]).backward()
    for c in clip:
        torch.nn.utils.clip_grad_norm_(c, 1.0)
    opt.step()
    p = m.predict(model, ids[:1], meta, tab, dev)
    assert p.shape == (3, 1, 12) and np.isfinite(p).all()
    assert np.allclose(p[2][:, ~m.CMASK], p[1][:, ~m.CMASK])
    print(f"[{arm}] T6 ok: losses {parts.detach().numpy().round(4).tolist()}, predict {p[0, 0, :3].round(3).tolist()}")
    print(f"ARM_OK {arm}")


def cfg_test():
    import coatnet_d96_runner as r
    c = r.CFG
    assert c["version"] == "d96_r50_lv2_v1_seed2027" and c["seed"] == 2027 and "eval_epochs" not in c, c
    assert r.LABELS_FILE == "soft_labels_final_per_target_latD2.json"
    assert c["sha256"]["labels"].startswith("dd0ab9ab") and c["swa_epochs"] == [21, 22, 23, 24, 25]
    assert r.sha(r.ASSETS / r.LABELS_FILE) == c["sha256"]["labels"]
    print("CFG_OK r50_lv2 seed 2027")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "arm":
        arm_test()
    elif len(sys.argv) > 1 and sys.argv[1] == "cfg":
        cfg_test()
    else:
        fails = 0
        for arm in ("shared", "adapter", "l4copy"):
            e = {**os.environ, **ENV, "V4_ARM": arm}
            fails += subprocess.run([sys.executable, "-u", __file__, "arm"], env=e).returncode != 0
        e = {**os.environ, "D96_PHASE": "r50_lv2", "D96_SEED": "2027", "D96_ASSETS": "/a"}
        fails += subprocess.run([sys.executable, "-u", __file__, "cfg"], env=e).returncode != 0
        print("ALL_OK" if not fails else f"FAILED {fails}")
        sys.exit(fails)

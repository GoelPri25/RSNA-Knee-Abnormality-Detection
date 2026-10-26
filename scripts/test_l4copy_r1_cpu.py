"""CPU tests for scripts/l4copy_r1_runner.py (real timm, no network). Run in localhost/rsna-mlx-train:v1:
podman run --rm --network=none --read-only --tmpfs /tmp -v <repo>/scripts:/w:ro -v <assets>:/a:ro -v <smoke d96>:/c/shard_0:ro
  -v <slim crops>:/cp:ro -v <rsna-v4-l4copy-fold0>:/v4:ro -w /w localhost/rsna-mlx-train:v1 python -u test_l4copy_r1_cpu.py
E1 (l4copy, real v4 weights): the runner's self_table = the submission's locate on the same studies, and m.predict on that
   table = the submission's predict_study (self-located): the training-time placement is the deployed one.
E2 R1Studies: random mode (slice in the valid range, centre within +-64 px) and table mode with slice jitter 2 (|dS| <= 2,
   same slot, neighbours valid, jitter actually used).
E3 l3copy: the local loss reaches layer3_c / layer4_c and never the shared layer3 / layer4; the global loss never the copies.
E4 GL_SMOKE main() end-to-end on l3copy + slice jitter (random epoch, self-table epoch, SWA, 2 BN refreshes, predicts).
"""
import os
import subprocess
import sys

ENV = {"D96_PHASE": "r50_gl", "GL_OUT": "/tmp/out", "GL_CROPS": "/tmp/cp", "D96_CACHE": "/c", "D96_ASSETS": "/a",
       "OMP_NUM_THREADS": "8", "R1_SJIT": "2", "R1_RANDOM": "1"}


def restricted_cp(meta):
    import numpy as np
    from pathlib import Path
    sub = Path("/tmp/cp"); sub.mkdir(exist_ok=True)
    for f in sorted(Path("/cp").glob("cp_*.npz")):
        z = np.load(f); keep = np.isin(z["ids"], list(meta))
        if keep.any():
            np.savez(sub / f.name, ids=z["ids"][keep], centres=z["centres"][keep], a_top=z["a_top"][keep])


def setup():
    import coatnet_d96_runner as r
    r.CFG["shards"] = 1; r.CFG["workers"] = 0
    meta = r.read_manifest(); restricted_cp(meta)
    return r, meta


def test_l4copy():
    import numpy as np
    import torch
    r, meta = setup()
    import l4copy_r1_runner as R
    m = R.m
    sys.path.insert(0, "/v4"); import v4_submission as sub
    dev = torch.device("cpu"); torch.set_num_threads(8)
    model, _, _ = m.build(dev, pretrained=False)
    model.load_state_dict(torch.load("/v4/" + sub.WEIGHTS["file"], map_location="cpu"), strict=True)
    model.eval()
    ref = sub.load_model("/v4/" + sub.WEIGHTS["file"], dev, sub.WEIGHTS["sha256"])
    labels = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
    ids = sorted(s for s in meta if s in labels)[:2]
    tab = R.self_table(model, ids, meta, dev)
    for s in ids:
        vol = np.ascontiguousarray(np.load(meta[s]["file"], mmap_mode="r")[meta[s]["row"]:meta[s]["row"] + R.MAXS])
        mask = meta[s]["mask"]
        cs = sub.uniform_centres(mask); lo = (sub.IMG - sub.EVAL_RES) // 2
        with torch.no_grad():
            xg = sub.windows(vol, cs, dev)[..., lo:lo + sub.EVAL_RES, lo:lo + sub.EVAL_RES].float().div_(255.0)
            fm = ref.backbone.forward_features(sub.normalise(xg)).float()
            h = ref.norm.float()(fm.mean((2, 3))); a = torch.softmax(ref.att.float()(h), dim=0)
            S0, R0, Q0, A0 = sub.locate(ref, fm, a, cs)
        S1, R1_, Q1, A1 = (x[R.CF] for x in tab[s])
        assert np.array_equal(S0, S1) and np.abs(R0 - R1_).max() <= 1 and np.abs(Q0 - Q1).max() <= 1, (S0, S1)
        assert np.allclose(A0, A1, atol=1e-5)
        with torch.no_grad():
            ps = sub.predict_study(ref, vol, mask, dev, amp=False)
        pm = m.predict(model, [s], meta, {s: tab[s]}, dev)[:, 0]
        d = np.abs(ps - pm).max()
        assert d < 2e-4, d
        print(f"E1 {s[-6:]}: self table == sub.locate (rows/cols max |d| {max(np.abs(R0 - R1_).max(), np.abs(Q0 - Q1).max())}), "
              f"predict max |dp| {d:.1e}; plane mix {np.bincount(r.SLOT_OF[S1[:, 0]], minlength=5).tolist()}")
    # E2: training data
    ds = R.R1Studies(ids, meta, labels, r.CFG["train_windows"], True, None)
    np.random.seed(0)
    for s_i in range(len(ids)):
        cs = np.clip(r.centres(meta[ids[s_i]]["mask"], R.K, False), 1, R.MAXS - 2)
        for _ in range(5):
            out = ds[s_i]; cen = out[8].numpy()
            assert out[5].shape == (10, 3, R.IMG, R.IMG)
            assert (np.abs(cen - R.IMG // 2) <= 64).all()
    ds.tab = tab
    dS = []
    for s_i, s in enumerate(ids):
        sl, _, _, _ = (x[R.CF] for x in tab[s])
        for _ in range(40):
            out = ds[s_i]
            xc = out[5].numpy(); vol = np.load(meta[s]["file"], mmap_mode="r")[meta[s]["row"]:meta[s]["row"] + R.MAXS]
            for q in range(10):
                Sq = [k for k in range(R.MAXS) if np.array_equal(vol[k], xc[q, 1])]
                d = min(abs(k - t) for k in Sq for t in sl[q]); dS.append(d)
                assert d <= 2
                t = sl[q][np.argmin([min(abs(k - u) for k in Sq) for u in sl[q]])]
                assert any(r.SLOT_OF[k] == r.SLOT_OF[t] for k in Sq)
    print(f"E2 ok: random crops within +-64 px; jitter |dS| histogram {np.bincount(dS, minlength=3).tolist()}")
    assert np.bincount(dS, minlength=3)[1:].sum() > 0
    print("L4COPY_OK")


def test_l3copy():
    import numpy as np
    import torch
    import r50_gl4_runner as m
    assert m.ARM == "l3copy"
    torch.manual_seed(0)
    dev = torch.device("cpu")
    model, opt, clip = m.build(dev, pretrained=False)
    with torch.no_grad():
        for bn in model.bns + model.cbns:
            bn.weight.uniform_(0.5, 1.5)
    ps = model.param_sets()
    print("E3 params " + str({k: sum(p.numel() for p in v) for k, v in ps.items()}))
    assert len(opt.param_groups) == 2 and sum(len(g["params"]) for g in opt.param_groups) == len(list(model.parameters()))
    model.train(); b = 2
    xg, xc = torch.randn(b * 3, 3, 64, 64), torch.randn(b * 10, 3, 64, 64)
    fid = torch.as_tensor(m.CF).repeat(b, 1); wc = torch.ones(b, 10)
    names = dict(model.named_parameters())
    for which in ("local", "global"):
        model.zero_grad(set_to_none=True)
        g, l = model(xg, b, xc, fid, wc)
        (l if which == "local" else g).sum().backward()
        hit = {n for n, p in names.items() if p.grad is not None and p.grad.abs().sum() > 0}
        for blk in ("layer3", "layer4"):
            sh = any(n.startswith(f"backbone.{blk}.") for n in hit); cp = any(n.startswith(f"{blk}_c.") for n in hit)
            if which == "local":
                assert cp and not sh, (blk, which)
            else:
                assert sh and not cp, (blk, which)
        assert any(n.startswith("backbone.layer2.") for n in hit)
        print(f"E3 {which} loss reaches: shared layer2 yes, shared layer3/4 {'no' if which == 'local' else 'yes'}, "
              f"copies {'yes' if which == 'local' else 'no'}")
    print("L3COPY_OK")


def test_main():
    r, meta = setup()
    import l4copy_r1_runner as R
    R.main()
    import json
    s = json.loads((R.OUT / "run_summary.json").read_text())
    assert [h["crops"] for h in s["history"]] == ["random", "self"], s["history"]
    print("E4 history crops " + str([h["crops"] for h in s["history"]]) + "; drift " + str(s["history"][1]["table_drift"]))
    print("MAIN_OK")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        {"l4copy": test_l4copy, "l3copy": test_l3copy, "main": test_main}[sys.argv[1]]()
    else:
        fails = 0
        for t, arm, extra in (("l4copy", "l4copy", {}), ("l3copy", "l3copy", {}), ("main", "l3copy", {"GL_SMOKE": "1"})):
            e = {**os.environ, **ENV, "V4_ARM": arm, **extra}
            fails += subprocess.run([sys.executable, "-u", __file__, t], env=e).returncode != 0
        print("ALL_OK" if not fails else f"FAILED {fails}")
        sys.exit(fails)

"""Experiment 1 on Modal: copy the d96 cache shards from Kaggle, CPU preflight, train on one H100.

transfer:  kaggle kernels output tumis001/rsna-coatnet-d96-cache-shard-{0..3} -> volume rsna-coatnet-d96-cache
           (derived uint8 cache only; no DICOM or reports), verified against each shard's summary.
preflight: input hashes, manifest counts, and the 384 -> 320 strict weight copy the runner relies on.
train:     scripts/coatnet_d96_runner.py on an H100, reading the cache straight from the volume; committed each epoch.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess

import modal

LOCAL = Path(__file__).resolve().parents[1]
volume = modal.Volume.from_name("rsna-coatnet-d96-cache", create_if_missing=True)
secret = (modal.Secret.from_dict({"KAGGLE_ACCESS_TOKEN": (Path.home() / ".kaggle/access_token").read_text().strip()})
          if modal.is_local() else modal.Secret.from_dict({}))
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "timm==1.0.29", "numpy", "kaggle==2.2.2")
         # Exp 5a start: Barlow Twins ImageNet r50 (facebookresearch/barlowtwins, MIT); sha logged by the SSL script
         .run_commands("python -c \"import urllib.request; urllib.request.urlretrieve("
                       "'https://dl.fbaipublicfiles.com/barlowtwins/ljng/resnet50.pth', '/root/bt_resnet50.pth')\"")
         # Model B: torchvision Swin3D-T Kinetics-400 weights (BSD-3 torchvision, ported from Video Swin, Apache-2.0);
         # loaded weights_only by the runner, sha256 checked against torchvision's hash-in-name
         .run_commands("python -c \"from torchvision.models.video import Swin3D_T_Weights as W; import urllib.request; "
                       "urllib.request.urlretrieve(W.KINETICS400_V1.url, '/root/swin3d_t_k400.pth')\"")
         # Model B encoder benchmark: pytorchvideo (X3D-M builder, Apache-2.0) + VideoMAE V2 ViT-S distilled weights
         # (HF OpenGVLab/VideoMAE2, Apache-2.0; sha256 pinned in the benchmark) + X3D-M Kinetics weights
         .pip_install("pytorchvideo==0.1.5")
         .run_commands("python -c \"import urllib.request as u; "
                       "u.urlretrieve('https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_s_k710_dl_from_giant.pth', "
                       "'/root/vit_s_k710_dl_from_giant.pth'); "
                       "u.urlretrieve('https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_M.pyth', '/root/X3D_M.pyth')\"")
         # Exp MS1: MedSigLIP-448 conversion to timm (safetensors only; transformers used for the parity check)
         .pip_install("transformers==4.57.1", "safetensors==0.6.2", "huggingface_hub==0.35.3")
         .add_local_file(str(LOCAL / "scripts/ssl_barlow_d96.py"), "/root/ssl_barlow_d96.py")
         .add_local_dir(str(LOCAL / "data/coatnet_d96_fold0/assets"), "/root/assets")
         .add_local_file(str(LOCAL / "scripts/d96_positions.py"), "/root/d96_positions.py")
         .add_local_file(str(LOCAL / "scripts/audit_v3_vs_1e_gpu.py"), "/root/audit_v3_vs_1e_gpu.py")
         .add_local_file(str(LOCAL / "scripts/eval_shift_protocol_gpu.py"), "/root/eval_shift_protocol_gpu.py")
         .add_local_file(str(LOCAL / "scripts/eval_ms1_resolution_gpu.py"), "/root/eval_ms1_resolution_gpu.py")
         .add_local_file(str(LOCAL / "scripts/eval_slice_use_gpu.py"), "/root/eval_slice_use_gpu.py")
         .add_local_file(str(LOCAL / "scripts/eval_ms1_token_maps_gpu.py"), "/root/eval_ms1_token_maps_gpu.py")
         .add_local_file(str(LOCAL / "scripts/extract_ms1_crop_features_gpu.py"), "/root/extract_ms1_crop_features_gpu.py")
         .add_local_file(str(LOCAL / "scripts/ms1_gl_runner.py"), "/root/ms1_gl_runner.py")
         .add_local_file(str(LOCAL / "scripts/r50_gl_runner.py"), "/root/r50_gl_runner.py")
         .add_local_file(str(LOCAL / "scripts/r50_gl3_runner.py"), "/root/r50_gl3_runner.py")
         .add_local_file(str(LOCAL / "scripts/r50_gl4_runner.py"), "/root/r50_gl4_runner.py")
         .add_local_file(str(LOCAL / "scripts/preflight_v4.py"), "/root/preflight_v4.py")
         .add_local_file(str(LOCAL / "scripts/bench_v4_a100.py"), "/root/bench_v4_a100.py")
         .add_local_dir(str(LOCAL / "data/coatnet_d96_fold0/ms1_crops_slim"), "/root/ms1_crops_slim")
         .add_local_file(str(LOCAL / "scripts/selfloc_r50_gl3_gpu.py"), "/root/selfloc_r50_gl3_gpu.py")
         .add_local_file(str(LOCAL / "scripts/selfloc_v4_gpu.py"), "/root/selfloc_v4_gpu.py")
         .add_local_file(str(LOCAL / "scripts/xai_r50_gl3_gpu.py"), "/root/xai_r50_gl3_gpu.py")
         .add_local_file(str(LOCAL / "data/coatnet_d96_fold0/run_r50_gl3/xai3_panels.json"), "/root/xai3_panels.json")
         .add_local_file(str(LOCAL / "scripts/diag_r50_gl_gpu.py"), "/root/diag_r50_gl_gpu.py")
         .add_local_file(str(LOCAL / "scripts/swin3d_d96n_runner.py"), "/root/swin3d_d96n_runner.py")
         .add_local_file(str(LOCAL / "scripts/bench_model_b_encoders.py"), "/root/bench_model_b_encoders.py")
         .add_local_file(str(LOCAL / "scripts/extract_fusion_features_gpu.py"), "/root/extract_fusion_features_gpu.py")
         .add_local_file(str(LOCAL / "scripts/joint_r50_swin_runner.py"), "/root/joint_r50_swin_runner.py")
         .add_local_file(str(LOCAL / "scripts/joint_j1_xai.py"), "/root/joint_j1_xai.py")
         .add_local_file(str(LOCAL / "scripts/j2x_runner.py"), "/root/j2x_runner.py")
         .add_local_file(str(LOCAL / "scripts/xai_r50_gl_gpu.py"), "/root/xai_r50_gl_gpu.py")
         .add_local_file(str(LOCAL / "data/coatnet_d96_fold0/run_r50_gl_v2/xai_panels.json"), "/root/xai_panels.json")
         .add_local_file(str(LOCAL / "scripts/extract_multiplane_features_gpu.py"), "/root/extract_multiplane_features_gpu.py")
         .add_local_file(str(LOCAL / "data/coatnet_d96_fold0/run_sig_lv2/crop_probe/multiplane_points.npz"),
                         "/root/multiplane_points.npz")
         .add_local_file(str(LOCAL / "data/coatnet_d96_fold0/kaggle_fold0_labelv2/val_swa.json"), "/root/lv2_val_swa.json")
         .add_local_file(str(LOCAL / "scripts/coatnet_d96_runner.py"), "/root/coatnet_d96_runner.py")
         # Exp L4C-D: the Kaggle diagnostic notebook's per-study code + the l4copy weights dataset (weights, cp table)
         .add_local_file(str(LOCAL / "kaggle/submission_kernels/v4_l4copy_diag_fold0/run.py"), "/root/l4copy_diag_core.py")
         .add_local_file(str(LOCAL / "scripts/l4copy_diag_gpu.py"), "/root/l4copy_diag_gpu.py")
         .add_local_dir(str(LOCAL / "data/kaggle_datasets/rsna-v4-l4copy-fold0"), "/root/v4art")
         # Exp L4C-NF + L4C-LOC zero-training baselines: l4copy_xai.py with the s2027 / MS1-free weights and the quota prior
         .add_local_file(str(LOCAL / "scripts/l4copy_xai.py"), "/root/l4copy_xai.py")
         .add_local_dir(str(LOCAL / "data/kaggle_datasets/rsna-v4-xai-extra"), "/root/xai_extra")
         .add_local_file(str(LOCAL / "data/coatnet_d96_fold0/l4copy_analysis/quota_prior.json"), "/root/quota_prior.json"))


def _hf_secret():
    """Exp MS1: HF read token from ~/.config/rsna/hf.env ("HF_TOKEN=hf_..." or a bare token line); never printed."""
    if not modal.is_local():
        return modal.Secret.from_dict({})
    env = Path.home() / ".config/rsna/hf.env"
    lines = [l.strip() for l in env.read_text().splitlines() if l.strip()] if env.is_file() else []
    tok = next((l.split("=", 1)[1].strip().strip('"') for l in lines if l.startswith("HF_TOKEN=")),
               next((l for l in lines if l.startswith("hf_")), ""))
    return modal.Secret.from_dict({"HF_TOKEN": tok})


app = modal.App("rsna-coatnet-d96-fold0")
VOL = Path("/root/vol")
SHARDS = 4
EXP = VOL / "experiments/coatnet_d96_fold0"
NATIVE_PHASES = ("r50_native", "r50_native_pertarget", "swin3d_b", "swin3d_b2")
TEACHER_PHASES = tuple(f"{a}_teacher{k}" for a in ("r50", "cnx", "cnxa") for k in range(4))   # Phase 3 cross-fitted teachers
SSL_DIR = VOL / "experiments/ssl_barlow_d96"
SSL_EPOCHS = 10                                    # Exp 5a pre-registered SSL budget (window-epochs)


def phase_env(phase):
    """Extra runner environment: Exp 5a reads the final self-supervised backbone."""
    return {"D96_BACKBONE": str(SSL_DIR / "backbone_final.pt")} if phase == "r50_ssl_pertarget" else {}


def layout(phase):
    """(volume cache dir, Kaggle shard kernel prefix, shards) for the phase's cache: d96 or native-capped d96n."""
    if phase in NATIVE_PHASES:
        return VOL / "d96n", "tumis001/rsna-d96-native-cache-shard-", 6
    return VOL / "d96", "tumis001/rsna-coatnet-d96-cache-shard-", SHARDS


@app.function(image=image, cpu=8, memory=16384, timeout=4 * 3600, secrets=[secret], volumes={"/root/vol": volume})
def transfer(phase: str = "exp1"):
    import numpy as np
    os.environ["KAGGLE_API_TOKEN"] = os.environ["KAGGLE_ACCESS_TOKEN"]
    volume.reload()
    cache, kernel, shards = layout(phase)
    from concurrent.futures import ThreadPoolExecutor
    native = phase in NATIVE_PHASES
    total = 0

    def fetch(i):
        # one Kaggle download stream is ~18 min per native shard; missing shards download in parallel
        dest = cache / f"shard_{i}"
        if not (dest / "cache_summary.json").is_file():
            tmp = cache / f"_download_{i}"
            for attempt in range(3):                  # Kaggle downloads sometimes break mid-file (IncompleteRead)
                shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
                if subprocess.run(["kaggle", "kernels", "output", f"{kernel}{i}", "-p", str(tmp)]).returncode == 0:
                    break
            else:
                raise RuntimeError(f"shard {i}: download failed 3 times")
            shutil.rmtree(dest, ignore_errors=True)
            (tmp / ("d96n" if native else "d96")).rename(dest)
            shutil.rmtree(tmp)
    with ThreadPoolExecutor(max_workers=shards) as pool:
        list(pool.map(fetch, range(shards)))
    volume.commit()
    for i in range(shards):
        dest = cache / f"shard_{i}"
        done = dest / "cache_summary.json"
        summary = json.loads(done.read_text())
        rows = [json.loads(x) for x in (dest / "manifest.jsonl").read_text().splitlines()]
        assert summary["shard"] == [i, shards] and summary["studies"] == len(rows), (i, summary)
        for part in sorted({r["part"] for r in rows}):
            a = np.load(dest / part, mmap_mode="r")
            n = sum(r["n"] if native else 96 for r in rows if r["part"] == part)
            assert a.dtype == np.uint8 and a.shape == (n, 384, 384), (part, a.shape)
        total += len(rows)
        print(f"shard {i}: {len(rows)} studies verified ({summary['seconds']}s build)", flush=True)
    assert total == 4349 + 58, total
    return total


@app.function(image=image, cpu=4, memory=16384, timeout=1800, volumes={"/root/vol": volume})
def preflight(phase: str = "exp1"):
    import sys
    import numpy as np
    import torch
    sys.path.insert(0, "/root")
    os.environ["D96_CACHE"] = str(layout(phase)[0])
    os.environ["D96_PHASE"] = phase
    os.environ.update(phase_env(phase))
    import coatnet_d96_runner as r
    volume.reload()
    r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
    r.load_labels(r.ASSETS / "validation_labels.json", "validation")
    assert r.sha(r.ASSETS / "folds.csv") == r.CFG["sha256"]["folds"]
    r.CACHE = layout(phase)[0]
    meta = r.read_manifest()
    print(f"{len(meta)} cached studies, {sum(m['gold'] for m in meta.values())} gold", flush=True)
    big, small = r.build(384, pretrained=False), r.build(320, pretrained=False)
    small.load_state_dict(big.state_dict(), strict=True)
    small.eval()
    vit = r.CFG["arch"].startswith("vit")               # Exp MS1: 448 px ViT-400M; 4 windows keep the CPU forward short
    ds = r.Studies(sorted(meta)[:1], meta, {}, 4 if vit else r.CFG["eval_windows"], False)
    x, sag, _, _, ctx = ds[0]
    if phase in NATIVE_PHASES:
        m = meta[sorted(meta)[0]]
        assert x.shape[0] == m["n"] - 2 and sag.sum() == np.isin(m["slot"][1:-1], (0, 1)).sum(), (x.shape, m["n"])
        lens = np.array([m["n"] for m in meta.values()])
        print(f"native studies: rows median {np.median(lens):.0f}, p90 {np.percentile(lens, 90):.0f}, max {lens.max()}", flush=True)
    with torch.no_grad():
        out = small(r.prepare(x[None], sag[None], False, r.CFG["eval_res"] if vit else 320, "cpu").float(), 1, ctx[None])
    assert out.shape == (1, 12) and torch.isfinite(out).all()
    if "sev" in r.CFG:                                  # Exp P1-sev: asset hash and aux output shape
        assert r.sha(r.ASSETS / r.CFG["sev"]["file"]) == r.CFG["sha256"]["sev"]
        assert small.aux.shape == (1, 2, 12), small.aux.shape
        print("SEV_PREFLIGHT_OK", flush=True)
    if "tags" in r.CFG:                                 # Exp P1-tags: asset hash, aux output shape, anchor matrix
        assert r.sha(r.ASSETS / r.CFG["tags"]["file"]) == r.CFG["sha256"]["tags"]
        nt = len(r.CFG["tags"]["anchors"])
        assert small.aux.shape == (1, nt), small.aux.shape
        assert torch.allclose(small.tagM.sum(1), torch.ones(nt)), small.tagM
        print("TAGS_PREFLIGHT_OK", flush=True)
    if "layer_decay" in r.CFG:                          # Exp P1-cnx2 / MS1: every backbone tensor in a depth group
        ids, lid = {}, r.layer_id(big)
        for n, _ in big.backbone.named_parameters():
            ids.setdefault(lid(n), []).append(n)
        top = 1 + len(big.backbone.blocks) if vit else 7
        assert sorted(ids) == list(range(top + 1)), sorted(ids)
        print("LAYER_GROUPS " + str({k: (len(v), v[0], v[-1]) for k, v in sorted(ids.items())}), flush=True)
    if "backbone_init" in r.CFG:
        # Exp 5a: the self-supervised backbone loads strictly and is not timm's tv_in1k / ImageNet default
        m = r.build(384, pretrained=True)
        sd = torch.load(r.CFG["backbone_init"], map_location="cpu", weights_only=True)
        assert all(torch.equal(m.backbone.state_dict()[k], v) for k, v in sd.items())
        if vit:                                         # Exp MS1: input normalisation and eval resize as registered
            assert torch.allclose(r.MEAN, torch.full((1, 3, 1, 1), 0.5)) and torch.allclose(r.STD, torch.full((1, 3, 1, 1), 0.5))
            xi = r.prepare(x[None], sag[None], False, r.CFG["eval_res"], "cpu")
            assert xi.shape[-2:] == (448, 448) and m.backbone.patch_embed.img_size == (448, 448), xi.shape
            print(f"MS1_PREFLIGHT_OK grad_ckpt {m.backbone.grad_checkpointing}, drop_path "
                  f"{m.backbone.blocks[-1].drop_path1.drop_prob}, input {tuple(xi.shape)}", flush=True)
        print(f"SSL_BACKBONE_OK {r.sha(Path(r.CFG['backbone_init']))[:16]}", flush=True)
    if "ctx" in r.CFG:
        # Exp 2a starts as Exp 1c: 1c's SWA weights in the context model (new modules fresh, gate 0) give 1c's output
        sd = torch.load(EXP / "run_r50/weights_swa21_25.pt", map_location="cpu", weights_only=True)
        miss, extra = small.load_state_dict(sd, strict=False)
        assert not extra and miss and all(k.split(".")[0] in ("inp", "slot", "pos", "enc", "out", "gate") for k in miss), (miss, extra)
        base = r.Raptor(320, pretrained=False); base.load_state_dict(sd, strict=True); base.eval(); small.eval()
        sides = [m["known"][:26].mean() for m in meta.values()]
        print(f"sagittal slices with a position: {sum(sides) / len(sides):.1%}", flush=True)
        for s in sorted(meta)[:3]:
            x, sag, _, _, ctx = r.Studies([s], meta, {}, 94, False)[0]
            xi = r.prepare(x[None], sag[None], False, 320, "cpu").float()
            with torch.no_grad():
                d = (small(xi, 1, ctx[None]) - base(xi, 1)).abs().max().item()
            assert d == 0.0, d
            assert ctx[:, 1].sum() > 0 and ctx[:, 2].max() <= 4
        print("CTX_INIT_EQUALS_1C: max |2a - 1c| = 0 on 3 studies", flush=True)
    if "teacher" in r.CFG:                              # Phase 3: split file hash, disjoint fit / out-of-fold, no fold 0
        import csv as _csv
        fold = {row["study_id"]: row["fold"] for row in _csv.DictReader((r.ASSETS / "folds.csv").open())}
        labels = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
        tr = sorted(s for s in labels if fold[s] != "0"); va = sorted(s for s in labels if fold[s] == "0")
        fit, oof = r.teacher_ids(tr, va)
        assert len(fit) + len(oof) == 3524 and all(s in meta for s in fit + oof)
        print(f"TEACHER_PREFLIGHT_OK split {r.CFG['teacher']['split']}: fit {len(fit)}, out-of-fold {len(oof)}, "
              f"eval epochs {r.CFG['eval_epochs']}, arch {r.CFG['arch']}", flush=True)
    if "shift" in r.CFG:
        shift_checks(r, meta, small)
    if "dual" in r.CFG:
        dual_checks(r, meta, small)
    if r.CFG.get("spatial", {}).get("v31"):
        spatial_checks_v31(r, meta, small)
    elif "spatial" in r.CFG:
        spatial_checks(r, meta, small)
    print(f"PREFLIGHT_OK: hashes, manifests, 384->320 strict copy, one {x.shape[0]}-window forward", flush=True)


def shift_checks(r, meta, small):
    """Exp P1-shift preflight (CPU fp32): (a) 16 bottleneck hooks; (b) the hook moves exactly channels [0, f) from the
    earlier and [f, 2f) from the later neighbour, zeros when absent, rest untouched, channels_last kept
    path; (c) a mirrored sagittal window swaps neighbours, a mirrored coronal one does not; (d) training / BN-refresh
    draws: always 16 windows, runs at stride d inside one slot, neighbour pairs at exactly +-d in the same slot;
    (e) eval neighbours likewise; (f) on a real training batch with a mirrored study and on a real eval study the
    shift changes the output (it is active) and the model is finite."""
    import numpy as np
    import torch
    d, fold = r.CFG["shift"]["d"], r.CFG["shift"]["fold"]
    blocks = [b for L in (small.backbone.layer1, small.backbone.layer2, small.backbone.layer3, small.backbone.layer4)
              for b in L]
    assert sum(len(b.conv1._forward_pre_hooks) for b in blocks) == 16
    with torch.no_grad():
        # timm zero-inits each bottleneck's last BN scale, so at random init every residual branch (where the shift
        # sits) outputs 0; give this throwaway preflight copy unit scales so the branch can carry the shift
        for blk in blocks:
            blk.bn3.weight.fill_(1.0)
    x = torch.randn(4, 64, 3, 3)
    g = torch.tensor([[4, 1], [0, 2], [1, 4], [4, 4]])
    f = 64 // fold
    x = x.contiguous(memory_format=torch.channels_last)
    small.nbr = g
    y = small.slice_shift(None, (x,))[0]
    small.nbr = None
    assert torch.equal(y[0, :f], torch.zeros_like(y[0, :f])) and torch.equal(y[0, f:2 * f], x[1, f:2 * f])
    assert torch.equal(y[1, :f], x[0, :f]) and torch.equal(y[2, f:2 * f], torch.zeros_like(y[2, f:2 * f]))
    assert torch.equal(y[:, 2 * f:], x[:, 2 * f:]) and y.is_contiguous(memory_format=torch.channels_last)
    ctx = torch.tensor([[[0, 0, 0, -1, 1], [0, 0, 0, 0, -1], [0, 0, 2, -1, 3], [0, 0, 2, 2, -1]]], dtype=torch.float32)
    r.FLIP["windows"] = torch.tensor([True, True, True, True])
    g = small.shift_index(ctx, 1, 4); r.FLIP["windows"] = None
    assert g.tolist() == [[1, 4], [4, 0], [4, 3], [2, 4]], g.tolist()
    rng = np.random.RandomState(0); ids = list(rng.choice(sorted(meta), 200, replace=False))
    stats = {"train_has_nbr": [], "eval_has_nbr": [], "slots": np.zeros(5)}
    for s in ids:
        for train in (True, False):
            ds = r.Studies([s], meta, {}, r.CFG["train_windows"], train)
            np.random.seed(len(stats["train_has_nbr"]))
            xw, sag, _, _, c5 = ds[0]
            assert xw.shape[0] == 16 and c5.shape == (16, 5)
            nb = c5[:, 3:5].numpy().astype(int)
            if train:
                stats["train_has_nbr"].append((nb >= 0).any(1).mean())
                stats["slots"] += np.bincount(c5[:, 2].numpy().astype(int), minlength=5)
        xe, sage, _, _, cxe = r.Studies([s], meta, {}, r.CFG["eval_windows"], False)[0]
        stats["eval_has_nbr"].append((cxe[:, 3:5].numpy() >= 0).all(1).mean())
    # neighbour geometry on one study: centres recovered from the dataset logic
    m = meta[ids[0]]
    np.random.seed(1); c = np.clip(r.run_centres(m["mask"], 16, True), 1, r.CFG["maxs"] - 2)
    for i, (p, q) in enumerate(r.neighbours(c, r.SLOT_OF)):
        if p >= 0:
            assert c[i] - c[p] == d and r.SLOT_OF[c[p]] == r.SLOT_OF[c[i]]
        if q >= 0:
            assert c[q] - c[i] == d and r.SLOT_OF[c[q]] == r.SLOT_OF[c[i]]
    ce = np.clip(r.centres(m["mask"], r.CFG["eval_windows"], False), 1, r.CFG["maxs"] - 2)
    for i, (p, q) in enumerate(r.neighbours(ce, r.SLOT_OF)):
        if p >= 0:
            assert ce[i] - ce[p] == d and r.SLOT_OF[ce[p]] == r.SLOT_OF[ce[i]]
    small.eval()
    with torch.no_grad():
        # (f) training-style batch of 2 studies (16 windows each), first study mirrored (its sagittal windows swap)
        np.random.seed(7)
        items = [r.Studies([s], meta, {}, 16, True)[0] for s in ids[:2]]
        xt = torch.stack([it[0] for it in items]); st = torch.stack([it[1] for it in items])
        ct = torch.stack([it[4] for it in items])
        xi = r.prepare(xt, st, False, 384, "cpu").float()
        outs = []
        for off_ in (False, True):
            small.shift_off = off_
            r.FLIP["windows"] = torch.tensor([True] * 16 + [False] * 16)
            outs.append(small(xi, 2, ct)); r.FLIP["windows"] = None
        assert torch.isfinite(outs[0]).all() and (outs[0] - outs[1]).abs().max() > 1e-4
        # (f) eval: 94 dense windows
        xe, sage, _, _, cxe = r.Studies([ids[0]], meta, {}, r.CFG["eval_windows"], False)[0]
        xi = r.prepare(xe[None], sage[None], False, 320, "cpu").float()
        small.shift_off = False
        ev = [small(xi, 1, cxe[None])]
        small.shift_off = True
        off = small(xi, 1, cxe[None])
        small.shift_off = False
    on = ev[0]
    assert torch.isfinite(on).all() and (on - off).abs().max() > 1e-4
    print("SHIFT_PREFLIGHT_OK " + str({"train_windows_with_a_neighbour": round(float(np.mean(stats["train_has_nbr"])), 3),
                                       "eval_windows_with_both": round(float(np.mean(stats["eval_has_nbr"])), 3),
                                       "train_slot_share": (stats["slots"] / stats["slots"].sum()).round(3).tolist(),
                                       "train_batch_shift_effect": round(float((outs[0] - outs[1]).abs().max()), 4),
                                       "max_abs_shift_effect": round(float((on - off).abs().max()), 4)}), flush=True)


@app.function(image=image, gpu="H100", cpu=4, memory=32768, timeout=1200, volumes={"/root/vol": volume})
def profile_shift():
    """Exp P1-shift timing on synthetic inputs (no data, no training): ms per training step (2 studies x 16 windows,
    384, bf16, forward + backward) and per eval study (94 windows, 320), shift off vs on."""
    import sys
    import time
    import torch
    sys.path.insert(0, "/root")
    os.environ["D96_PHASE"] = "r50_shift"
    import coatnet_d96_runner as r
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    model = r.build(384, pretrained=False).to(dev).to(memory_format=torch.channels_last)

    def ctx_for(b, runs, o):
        rows = []
        for _ in range(b):
            c, slot, start = [], [], 0
            for s, n in enumerate(runs):
                for j in range(n):
                    p = start + j - o if j >= o else -1
                    q = start + j + o if j + o < n else -1
                    c.append([0.5, 1.0, float(s), float(p), float(q)])
                start += n
            rows.append(c)
        return torch.tensor(rows, device=dev)
    ct = ctx_for(2, [4, 4, 3, 2, 3], 1)
    ce = ctx_for(1, [24, 20, 16, 10, 24], 3)
    xt = torch.randint(0, 255, (2, 16, 3, 384, 384), dtype=torch.uint8)
    xe = torch.randint(0, 255, (1, 94, 3, 384, 384), dtype=torch.uint8)
    st, se = torch.zeros(2, 16, dtype=torch.bool), torch.zeros(1, 94, dtype=torch.bool)
    out = {}
    for impl in ("off", "on"):
        model.shift_off = impl == "off"
        model.train()
        for i in range(25):
            if i == 5:
                torch.cuda.synchronize(); t0 = time.monotonic()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(r.prepare(xt, st, True, 384, dev), 2, ct).float().sum()
            loss.backward(); model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); tr = (time.monotonic() - t0) / 20 * 1000
        model.eval()
        with torch.no_grad():
            for i in range(15):
                if i == 5:
                    torch.cuda.synchronize(); t0 = time.monotonic()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model(r.prepare(xe, se, False, 320, dev), 1, ce)
        torch.cuda.synchronize(); ev = (time.monotonic() - t0) / 10 * 1000
        out[impl] = (round(tr, 1), round(ev, 1))
        print(f"PROFILE {impl}: train step {tr:.1f} ms, eval study {ev:.1f} ms", flush=True)
    return out


def dual_checks(r, meta, small):
    """Exp B+D preflight (CPU fp32): (a) both label files load with their hashes and cover the same studies; (b) mode
    "avg" = logit of the mean of the heads' probabilities, "train" gives [1, 24], attention sums to 1 per finding;
    (c) one weighted two-head loss reaches both classifiers and the shared backbone; (d) a guided training bag starts
    with the study's deduplicated top windows (exact slices), then random up to 16; (e) guide_refresh end to end."""
    import numpy as np
    import torch
    torch.manual_seed(0)
    l1 = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
    l2 = r.load_labels(r.ASSETS / r.CFG["dual"]["labels2"], "labels2")
    assert set(l1) == set(l2) and len(l1) == 4349, (len(l1), len(l2))
    diff = np.mean([np.abs(l1[s][0] - l2[s][0]) > 0.1 for s in l1], 0)
    print("DUAL_LABELS cells differing > 0.1: " + json.dumps(dict(zip(r.TARGETS, diff.round(3).tolist()))), flush=True)
    s0 = next(s for s in sorted(meta) if s in l1)                                         # not a gold study
    x, sag, _, _, ctx = r.Studies([s0], meta, {}, r.CFG["eval_windows"], False)[0]
    xi = r.prepare(x[None], sag[None], False, 320, "cpu").float()
    with torch.no_grad():
        small.mode = "h1"; h1 = small(xi, 1, ctx[None])
        small.mode = "h2"; h2 = small(xi, 1, ctx[None])
        small.mode = "avg"; av = small(xi, 1, ctx[None])
        small.mode = "train"; tr = small(xi, 1, ctx[None]); small.mode = "avg"
    d = (av.sigmoid() - 0.5 * (h1.sigmoid() + h2.sigmoid())).abs().max().item()
    assert d < 1e-5 and tr.shape == (1, 24) and torch.equal(tr, torch.cat([h1, h2], 1)), (d, tr.shape)
    am = small.att_mean
    assert am.shape == (1, x.shape[0], 12) and (am.sum(1) - 1).abs().max() < 1e-4, am.shape
    print(f"DUAL_HEADS_OK: avg == mean of head probabilities (max diff {d:.1e}); train -> [1, 24]", flush=True)
    big = r.build(384, pretrained=False); big.train(); big.mode = "train"
    ds = r.Studies([s0], meta, {s0: (np.r_[l1[s0][0], l2[s0][0]], np.r_[l1[s0][1], l2[s0][1]])}, r.CFG["train_windows"], True)
    m = meta[s0]
    top = np.clip(r.centres(m["mask"], 12, False), 1, 94); top[3] = top[2]               # one shared top window
    ds.guide = {s0: top}
    x, sag, y, w, ctx = ds[0]
    u = np.unique(top)
    vol = np.load(m["file"], mmap_mode="r")[m["row"]:m["row"] + 96]
    assert x.shape[0] == 16 and y.shape == (24,), (x.shape, y.shape)
    assert all(np.array_equal(x[j].numpy(), vol[c - 1:c + 2]) for j, c in enumerate(u)), "top windows must lead the bag"
    xb = r.prepare(x[None], sag[None], True, 384, "cpu").float()
    logits = big(xb, 1, ctx[None])
    loss = 0.5 * sum((torch.nn.functional.binary_cross_entropy_with_logits(logits[:, j:j + 12], y[None, j:j + 12], reduction="none")
                      * w[None, j:j + 12]).sum() / w[None, j:j + 12].sum().clamp_min(1) for j in (0, 12))
    loss.backward()
    g = {k: float(t.grad.norm()) for k, t in (("clsW", big.clsW), ("clsW2", big.clsW2), ("att2", big.att2[0].weight),
                                               ("backbone_conv1", big.backbone.conv1.weight))}
    assert all(v > 0 for v in g.values()), g
    print(f"DUAL_GUIDED_BAG_OK: {len(u)} top windows lead a 16-window bag; grads " + json.dumps(g), flush=True)
    ids = sorted(s for s in meta if not meta[s]["gold"])[:3]
    guide, st = r.guide_refresh(small, ids, meta, "cpu", {})
    guide2, st2 = r.guide_refresh(small, ids, meta, "cpu", guide)
    assert all(len(v) == 12 and 1 <= v.min() and v.max() <= 94 for v in guide.values())
    assert all(np.array_equal(guide[s], guide2[s]) for s in ids) and all(v == 0 for v in st2["top_moved_gt1_vs_prev"].values())
    small.mode = "avg"
    print("GUIDE_REFRESH_OK " + json.dumps(st), flush=True)


def spatial_checks(r, meta, small):
    """Exp v3 preflight (CPU fp32), revision 3 section 4.4: (a) v3 at init == C0 exactly; (a') no padding + uniform
    alpha + LN_p = LN gives p == g; (b) feature grid and mask shapes at 320 / 384; (c) gradients reach U at step 1,
    score_out at 2, score_in at 3."""
    import torch
    torch.manual_seed(0)
    on, os_ = r.CFG["spatial"]["on"], r.CFG["spatial"]["output_stride"]
    r.CFG["spatial"]["on"] = True;  v3 = r.build(320, pretrained=False).eval()
    r.CFG["spatial"]["on"] = False; c0 = r.build(320, pretrained=False).eval()
    r.CFG["spatial"]["on"] = on
    v3.load_state_dict(small.state_dict(), strict=True); c0.load_state_dict(small.state_dict(), strict=True)
    for s in sorted(meta)[:3]:
        x, sag, _, _, ctx = r.Studies([s], meta, {}, r.CFG["eval_windows"], False)[0]
        xi = r.prepare(x[None], sag[None], False, 320, "cpu").float()
        with torch.no_grad():
            d = (v3(xi, 1, ctx[None]) - c0(xi, 1, ctx[None])).abs().max().item()
            z = v3.backbone.forward_features(xi[:8])
            assert z.shape[-2:] == (320 // os_, 320 // os_), z.shape                                      # (b) eval grid
            g = v3.norm(z.mean((2, 3)))
            zc = z.flatten(2).transpose(1, 2)
            p = v3.norm_p(torch.einsum("nc,ncf->nf", torch.full(zc.shape[:2], 1 / zc.shape[1]), zc))
            dp = (p - g).abs().max().item()
        assert d == 0.0, d
        assert dp < 1e-4, dp
    print(f"V3_INIT_EQUALS_C0: max |v3 - C0| = 0 on 3 studies; p == g (uniform alpha) max |diff| {dp:.1e}", flush=True)
    # (b) training mask at 384: shape, and border cells masked under a strong affine
    m = r.build(384, pretrained=False)
    m.train()
    x, sag, _, _, ctx = r.Studies(sorted(meta)[:2], meta, {}, r.CFG["train_windows"], True)[0]
    old = dict(r.CFG["aug"]); r.CFG["aug"].update({"rot_deg": 10.0, "scale": 0.10, "shift": 0.06})
    xi = r.prepare(x[None], sag[None], True, 384, "cpu", m).float()
    r.CFG["aug"].update(old)
    assert m.valid.shape == (16, 384 // os_, 384 // os_) and m.valid.dtype == torch.bool, m.valid.shape
    print(f"V3_MASK: {tuple(m.valid.shape)}, valid fraction {m.valid.float().mean():.3f}", flush=True)
    # (c) gradient flow: 3 SGD steps on the head of a 4-window study, spatial path on
    m.on = True
    xs, ctxs = xi[:4], ctx[None, :4]
    m.valid = m.valid[:4]
    head = [p for n, p in m.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.SGD(head, lr=1.0)
    seen = []
    for step in range(3):
        opt.zero_grad()
        y = (torch.rand(1, 12) > 0.5).float()
        torch.nn.functional.binary_cross_entropy_with_logits(m(xs, 1, ctxs), y).backward()
        seen.append({k: float(t.grad.norm()) if t.grad is not None else 0.0
                     for k, t in (("U", m.clsU), ("score_out", m.score[-1].weight), ("score_in", m.score[0].weight))})
        opt.step()
    assert seen[0]["U"] > 0 and seen[1]["score_out"] > 0 and seen[2]["score_in"] > 0, seen
    print("V3_GRADIENTS " + json.dumps(seen), flush=True)


def spatial_checks_v31(r, meta, small):
    """Exp v3.1 preflight (CPU fp32): (a) the manual layer3/layer4 path equals timm forward_features exactly; (a') at
    init (V = 0) the window attention equals the GAP-only choice and, U = 0, the output equals the GAP-only path;
    (b) cell grids 20 x 20 / 24 x 24 and the training mask at stride 16; (c) 3 SGD steps: U and V move at step 1,
    score_out and score_in at step 2, V gets no gradient with v_on False; the 4 heads' alphas differ."""
    import torch
    torch.manual_seed(0)
    m = small
    for s in sorted(meta)[:2]:
        x, sag, _, _, ctx = r.Studies([s], meta, {}, r.CFG["eval_windows"], False)[0]
        xi = r.prepare(x[None], sag[None], False, 320, "cpu").float()
        with torch.no_grad():
            z3, z4 = m.features(xi[:6])
            d = (z4 - m.backbone.forward_features(xi[:6])).abs().max().item()
            assert d == 0.0, d
            assert z3.shape[-2:] == (20, 20) and z4.shape[-2:] == (10, 10), (z3.shape, z4.shape)
            m.collect, m.stats = "chk", {}
            m(xi, 1, ctx[None])
            sm = m.summary("chk"); m.collect = None
        assert all(v == 0.0 for v in sm["window_shift"].values()), sm["window_shift"]
        assert sm["u_share_all"] == 0.0, sm["u_share_all"]
    print(f"V31_FEATURES_OK: layer3 {tuple(z3.shape)}, layer4 {tuple(z4.shape)}; window_shift 0 at init", flush=True)
    big = r.build(384, pretrained=False); big.train()
    x, sag, _, _, ctx = r.Studies(sorted(meta)[:2], meta, {}, r.CFG["train_windows"], True)[0]
    old = dict(r.CFG["aug"]); r.CFG["aug"].update({"rot_deg": 10.0, "scale": 0.10, "shift": 0.06})
    xi = r.prepare(x[None], sag[None], True, 384, "cpu", big).float()
    r.CFG["aug"].update(old)
    assert big.valid.shape == (16, 24, 24) and big.valid.dtype == torch.bool, big.valid.shape
    print(f"V31_MASK: {tuple(big.valid.shape)}, valid fraction {big.valid.float().mean():.3f}", flush=True)
    xs, ctxs = xi[:4], ctx[None, :4]
    big.valid = big.valid[:4]
    head = [p for n, p in big.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.SGD(head, lr=1.0)
    seen = []
    for step in range(3):
        opt.zero_grad()
        y = (torch.rand(1, 12) > 0.5).float()
        torch.nn.functional.binary_cross_entropy_with_logits(big(xs, 1, ctxs), y).backward()
        seen.append({k: float(t.grad.norm()) if t.grad is not None else 0.0
                     for k, t in (("U", big.clsU), ("V", big.clsV), ("score_out", big.score[-1].weight),
                                  ("score_in", big.score[0].weight))})
        opt.step()
    assert seen[0]["U"] > 0 and seen[0]["V"] > 0 and seen[1]["score_out"] > 0 and seen[1]["score_in"] > 0, seen
    big.v_on = False; big.zero_grad()
    torch.nn.functional.binary_cross_entropy_with_logits(big(xs, 1, ctxs), y).backward()
    assert big.clsV.grad is None or big.clsV.grad.abs().max() == 0, "V must get no gradient before v_start"
    big.v_on = True
    with torch.no_grad():
        z3, z4 = big.features(xs)
        c = torch.cat([z3, torch.nn.functional.interpolate(z4, size=z3.shape[-2:], mode="nearest")], 1).flatten(2).transpose(1, 2)
        big.eval()
        al = torch.softmax(big.score(big.norm_s(c)).view(4, -1, 12, big.heads), 1)
        spread = (al - al.mean(-1, keepdim=True)).abs().max().item()
    assert spread > 0, spread
    print("V31_GRADIENTS " + json.dumps(seen) + f"; V frozen before v_start; head alpha spread {spread:.2e}", flush=True)


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=8 * 3600,
              retries=2, volumes={"/root/vol": volume})
def train(phase: str = "exp1"):
    run_training(phase)


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=8 * 3600,
              retries=2, volumes={"/root/vol": volume})
def train_b(phase: str = "swin3d_b"):
    """Model B: smaller container (the benchmark showed data loading at 0.02 s/study is not the bottleneck)."""
    run_training(phase)


def run_training(phase):
    # After a preemption Modal can re-run this input in a container where the previous attempt's runner is still
    # alive; its open cache files make volume.reload() fail. Stop it (the new attempt resumes from recovery.pt).
    import signal
    import time

    def stop_stale(sig):
        for d in Path("/proc").iterdir():
            try:
                cmd = (d / "cmdline").read_bytes() if d.name.isdigit() else b""
                if any(s in cmd for s in (b"coatnet_d96_runner.py", b"swin3d_d96n_runner.py", b"joint_r50_swin_runner.py",
                                            b"j2x_runner.py")):
                    os.kill(int(d.name), sig); print(f"stopped stale runner pid {d.name}", flush=True)
            except (OSError, ProcessLookupError):
                pass
    stop_stale(signal.SIGTERM)
    for attempt in range(10):
        try:
            volume.reload()
            break
        except RuntimeError as e:
            print(f"volume.reload attempt {attempt}: {e}", flush=True)
            stop_stale(signal.SIGKILL)
            time.sleep(15)
    else:
        raise RuntimeError("volume.reload kept failing")
    out = EXP / {"exp1": "run", "continue": "run_continue", "r50": "run_r50", "r50_pertarget": "run_r50_pertarget",
                 "r50_ctx": "run_r50_ctx", "r50_border": "run_r50_border",
                 "r50_state": "run_r50_state", "r50_plane": "run_r50_plane",
                 "r50_plane_pertarget": "run_r50_plane_pertarget",
                 "r50_pertarget_seed": "run_r50_pertarget_seed", "r50_native": "run_r50_native",
                 "r50_native_pertarget": "run_r50_native_pertarget", "r50_ssl_pertarget": "run_r50_ssl_pertarget",
                 "r50_swa_cyclic": "run_r50_swa_cyclic", "r50_series_alloc": "run_r50_series_alloc",
                 "r50_spatial": "run_r50_spatial", "r50_os16_gap": "run_r50_os16_gap",
                 "r50_spatial_os32": "run_r50_spatial_os32", "r50_spatial_v31": "run_r50_spatial_v31b",
                 "r50_dual_guided": "run_r50_dual_guided", "r50_r949": "run_r50_r949",
                 "r50_r949_aug": "run_r50_r949_aug",
                 "cnx_pertarget": "run_cnx_pertarget",
                 "r50_aug2": "run_r50_aug2", "cnx_tune": "run_cnx_tune", "r50_sev": "run_r50_sev",
                 "cnx_in12k": "run_cnx_in12k", "cnx_dinov3": "run_cnx_dinov3",
                 "r50_tags": "run_r50_tags",
                 "r50_shift": "run_r50_shift", "swin3d_b": "run_swin3d_b", "swin3d_b2": "run_swin3d_b2",
                 "r50_rulecap": "run_r50_rulecap", "r50_rulecap_noeff": "run_r50_rulecap_noeff",
                 "joint_j1": "run_joint_j1", "j2x": "run_j2x",
                 "r50_student_state": "run_r50_student_state", "r50_student_a050": "run_r50_student_a050",
                 "sig_lv2": "run_sig_lv2",
                 **{f"{a}_teacher{k}": f"run_{a}_teacher{k}" for a in ("r50", "cnx", "cnxa") for k in range(4)}}[phase]
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "D96_CACHE": str(layout(phase)[0]), "D96_OUT": str(out), "D96_INIT": str(EXP / "run"),
           "D96_PHASE": phase, "PYTHONUNBUFFERED": "1", **phase_env(phase)}
    if phase.startswith("swin3d"):                    # Model B: own runner (whole-series Swin3D-T on d96n)
        env["SWIN_PHASE"], env["SWIN_MODE"] = phase, "train"
    if phase == "joint_j1":                           # Exp J1: ResNet (d96) + Swin3D (d96n) branches, one network
        env.update(JOINT_CACHE_R=str(VOL / "d96"), JOINT_CACHE_S=str(VOL / "d96n"), JOINT_OUT=str(out),
                   JOINT_MODE="train")
    if phase == "j2x":                                # Exp J2-X: J1's two branches + synergy residual (cross-attention)
        env.update(JOINT_CACHE_R=str(VOL / "d96"), JOINT_CACHE_S=str(VOL / "d96n"), JOINT_OUT=str(out), J2X_MODE="train")
    script = ("/root/swin3d_d96n_runner.py" if phase.startswith("swin3d") else
              "/root/joint_r50_swin_runner.py" if phase == "joint_j1" else
              "/root/j2x_runner.py" if phase == "j2x" else "/root/coatnet_d96_runner.py")
    proc = subprocess.Popen(["python", "-u", script], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    with (out / "training.log").open("a", buffering=1) as log:
        for line in proc.stdout:
            print(line, end="", flush=True); log.write(line)
            if line.startswith(("EPOCH", "RUN_SUMMARY")):
                volume.commit()
    code = proc.wait()
    volume.commit()
    if code:
        raise RuntimeError(f"runner exited {code}; recovery.pt retained for retry")


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=6 * 3600, retries=2,
              volumes={"/root/vol": volume})
def ssl(mode: str = "bench"):
    """Exp 5a step 1: Barlow Twins continued pretraining on training-fold d96 windows (bench = timing only)."""
    volume.reload()
    SSL_DIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "D96_CACHE": str(VOL / "d96"), "SSL_OUT": str(SSL_DIR), "SSL_MODE": mode,
           "SSL_EPOCHS": str(SSL_EPOCHS), "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(["python", "-u", "/root/ssl_barlow_d96.py"], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    with (SSL_DIR / f"{mode}.log").open("a", buffering=1) as log:
        for line in proc.stdout:
            print(line, end="", flush=True); log.write(line)
            if line.startswith(("EPOCH", "SSL_DONE", "BENCH")):
                volume.commit()
    code = proc.wait()
    volume.commit()
    if code:
        raise RuntimeError(f"ssl exited {code}; recovery.pt retained for retry")


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=3 * 3600, volumes={"/root/vol": volume})
def audit_v3():
    """v3 vs 1e branch ablation + spatial XAI on fold 0 (scripts/audit_v3_vs_1e_gpu.py), inference only."""
    volume.reload()
    out = EXP / "audit_v3_vs_1e"
    env = {**os.environ, "D96_CACHE": str(VOL / "d96"), "D96_PHASE": "r50_spatial", "PYTHONUNBUFFERED": "1",
           "AUDIT_OUT": str(out), "V3_WEIGHTS": str(EXP / "run_r50_spatial/weights_swa21_25.pt"),
           "E1_WEIGHTS": str(EXP / "run_r50_pertarget/weights_swa21_25.pt")}
    code = subprocess.run(["python", "-u", "/root/audit_v3_vs_1e_gpu.py"], env=env, cwd="/root").returncode
    volume.commit()
    if code:
        raise RuntimeError(f"audit exited {code}")


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=1800, volumes={"/root/vol": volume})
def shift_dx():
    """Exp P1-shift-dx: P1-shift and 1e re-scored under training-like window layouts (eval_shift_protocol_gpu.py)."""
    volume.reload()
    out = EXP / "shift_dx"
    env = {**os.environ, "D96_CACHE": str(VOL / "d96"), "D96_PHASE": "r50_shift", "PYTHONUNBUFFERED": "1",
           "DX_OUT": str(out), "SHIFT_WEIGHTS": str(EXP / "run_r50_shift/weights_swa21_25.pt"),
           "E1_WEIGHTS": str(EXP / "run_r50_pertarget/weights_swa21_25.pt")}
    code = subprocess.run(["python", "-u", "/root/eval_shift_protocol_gpu.py"], env=env, cwd="/root").returncode
    volume.commit()
    if code:
        raise RuntimeError(f"shift dx exited {code}")


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=3600, volumes={"/root/vol": volume})
def ms1_q1():
    """Exp MS1-Q1: MS1 SWA weights re-scored at 448 / 560 / 672 px (eval_ms1_resolution_gpu.py), inference only."""
    volume.reload()
    out = EXP / "run_sig_lv2/q1_resolution"
    env = {**os.environ, "D96_CACHE": str(VOL / "d96"), "D96_PHASE": "sig_lv2", "PYTHONUNBUFFERED": "1",
           "Q1_OUT": str(out), "MS1_WEIGHTS": str(EXP / "run_sig_lv2/weights_swa21_25.pt")}
    code = subprocess.run(["python", "-u", "/root/eval_ms1_resolution_gpu.py"], env=env, cwd="/root").returncode
    volume.commit()
    if code:
        raise RuntimeError(f"ms1 q1 exited {code}")


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=3600, volumes={"/root/vol": volume})
def ms1_maps():
    """Exp LOC-0-MS1: MS1 per-finding token maps (gradient x activation) + MAP attention on fold 0, inference only."""
    volume.reload()
    env = {**os.environ, "D96_CACHE": str(VOL / "d96"), "D96_PHASE": "sig_lv2", "PYTHONUNBUFFERED": "1",
           "MAPS_OUT": str(EXP / "run_sig_lv2/token_maps"), "MS1_WEIGHTS": str(EXP / "run_sig_lv2/weights_swa21_25.pt")}
    code = subprocess.run(["python", "-u", "/root/eval_ms1_token_maps_gpu.py"], env=env, cwd="/root").returncode
    volume.commit()
    if code:
        raise RuntimeError(f"ms1 maps exited {code}")


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=5 * 3600, retries=1, volumes={"/root/vol": volume})
def ms1_gl(smoke: bool = False):
    """Exp MS1-GL: MS1 trained on global + local (self-chosen crop) views from MS1 SWA weights; smoke = timing / memory."""
    volume.reload()
    out = EXP / ("run_ms1_gl_smoke" if smoke else "run_ms1_gl")
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "D96_CACHE": str(VOL / "d96"), "D96_PHASE": "sig_lv2", "PYTHONUNBUFFERED": "1",
           "GL_OUT": str(out), "GL_CROPS": str(EXP / "run_sig_lv2/crop_probe"),
           "MS1_WEIGHTS": str(EXP / "run_sig_lv2/weights_swa21_25.pt"), "GL_SMOKE": "1" if smoke else "0"}
    proc = subprocess.Popen(["python", "-u", "/root/ms1_gl_runner.py"], env=env, cwd="/root", stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    with (out / "training.log").open("a", buffering=1) as log:
        for line in proc.stdout:
            print(line, end="", flush=True); log.write(line)
            if line.startswith(("EPOCH", "RUN_SUMMARY", "SMOKE_DONE")):
                volume.commit()
    code = proc.wait()
    volume.commit()
    if code:
        raise RuntimeError(f"ms1 gl exited {code}; recovery.pt retained for retry")


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=4 * 3600, retries=1, volumes={"/root/vol": volume})
def r50_gl(smoke: bool = False, version: str = "v1"):
    """Exp r50-GL: label-v2 r50 recipe from ImageNet on global windows + per-finding crops (MS1 crop table), dual BN,
    residual reader; smoke = the whole code path on a few studies (32 steps, SWA of 1 epoch, refresh on 8)."""
    volume.reload()
    assert version in ("v1", "v2")
    out = EXP / (("run_r50_gl_smoke" if smoke else "run_r50_gl") + ("" if version == "v1" else f"_{version}"))
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "D96_CACHE": str(VOL / "d96"), "D96_PHASE": "r50_gl", "PYTHONUNBUFFERED": "1", "GL_VERSION": version,
           "GL_OUT": str(out), "GL_CROPS": str(EXP / "run_sig_lv2/crop_probe"), "GL_SMOKE": "1" if smoke else "0"}
    proc = subprocess.Popen(["python", "-u", "/root/r50_gl_runner.py"], env=env, cwd="/root", stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    with (out / "training.log").open("a", buffering=1) as log:
        for line in proc.stdout:
            print(line, end="", flush=True); log.write(line)
            if line.startswith(("EPOCH", "RUN_SUMMARY", "SMOKE_DONE")):
                volume.commit()
    code = proc.wait()
    volume.commit()
    if code:
        raise RuntimeError(f"r50 gl exited {code}; recovery.pt retained for retry")


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=2700, volumes={"/root/vol": volume})
def r50_gl3_xai():
    """Exp r50-GL v3 XAI, inference only: v3 SWA on fold 0 (Grad-CAM, gate / region / placement / BN ablations, pooled
    features) and 825 training studies (heads, gate, features); panels for local rendering."""
    volume.reload()
    run = EXP / "run_r50_gl3"; out = run / "xai"; out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "D96_CACHE": str(VOL / "d96"), "D96_PHASE": "r50_gl", "PYTHONUNBUFFERED": "1", "GL_OUT": str(out),
           "GL_CROPS": str(EXP / "run_sig_lv2/crop_probe"), "GL_POINTS4": "/root/multiplane_points.npz",
           "GL_WEIGHTS": str(run / "weights_swa21_25.pt"), "XAI_OUT": str(out), "PANEL_IDS": "/root/xai3_panels.json"}
    with (out / "xai.log").open("a", buffering=1) as log:
        proc = subprocess.Popen(["python", "-u", "/root/xai_r50_gl3_gpu.py"], env=env, cwd="/root", stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True); log.write(line)
    code = proc.wait()
    volume.commit()
    if code:
        raise RuntimeError(f"r50 gl3 xai exited {code}")


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=1800, volumes={"/root/vol": volume})
def r50_gl3_selfloc():
    """Exp SL-0: r50-GL v3 SWA locates its own regions (window attention + gradient x activation), fold 0, vs MS1's."""
    volume.reload()
    run = EXP / "run_r50_gl3"; out = run / "selfloc"; out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "D96_CACHE": str(VOL / "d96"), "D96_PHASE": "r50_gl", "PYTHONUNBUFFERED": "1", "GL_OUT": str(out),
           "GL_CROPS": str(EXP / "run_sig_lv2/crop_probe"), "GL_POINTS4": "/root/multiplane_points.npz",
           "GL_WEIGHTS": str(run / "weights_swa21_25.pt")}
    code = subprocess.run(["python", "-u", "/root/selfloc_r50_gl3_gpu.py"], env=env, cwd="/root").returncode
    volume.commit()
    if code:
        raise RuntimeError(f"selfloc exited {code}")


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=3600, volumes={"/root/vol": volume})
def v4_selfloc(runs: str = "v4_l4copy_s2026,v4_shared_s2027"):
    """Exp SL-0v4: each v4 SWA model locates its own crops (global attention + gradient x activation), fold 0, vs
    MS1's table. Weights uploaded to EXP/run_r50_gl4_mlx/<run>/weights_swa21_25.pt (MeluXina outputs)."""
    volume.reload()
    for run in runs.split(","):
        base = EXP / "run_r50_gl4_mlx" / run; out = base / "selfloc"; out.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "D96_CACHE": str(VOL / "d96"), "D96_PHASE": "r50_gl", "PYTHONUNBUFFERED": "1",
               "GL_OUT": str(out), "GL_CROPS": "/root/ms1_crops_slim", "V4_ARM": run.split("_")[1],
               "GL_WEIGHTS": str(base / "weights_swa21_25.pt")}
        code = subprocess.run(["python", "-u", "/root/selfloc_v4_gpu.py"], env=env, cwd="/root").returncode
        volume.commit()
        if code:
            raise RuntimeError(f"selfloc {run} exited {code}")


@app.function(image=image, gpu="H100", cpu=8, memory=65536, timeout=3600, volumes={"/root/vol": volume})
def l4copy_diag():
    """Exp L4C-D: l4copy crop-placement diagnostics on fold 0 from the d96 cache (Kaggle T4 queue stalled)."""
    volume.reload()
    out = EXP / "l4copy_diag/modal_v1"; out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "D96_CACHE": str(VOL / "d96"), "V4_ART": "/root/v4art", "DIAG_OUT": str(out),
           "FOLDS": "/root/assets/folds.csv", "PYTHONUNBUFFERED": "1"}
    code = subprocess.run(["python", "-u", "/root/l4copy_diag_gpu.py"], env=env, cwd="/root").returncode
    volume.commit()
    if code:
        raise RuntimeError(f"l4copy diag exited {code}")


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=3600, volumes={"/root/vol": volume})
def l4copy_nf():
    """Exp L4C-NF (+ L4C-LOC zero-training quota baselines qm / qs / selfu): scripts/l4copy_xai.py on the 825 fold-0
    studies from the d96 cache, models s2026 (LB 0.939) / s2027 / r1self; lite, no case panels (Kaggle T4 queue stalled)."""
    import csv
    volume.reload()
    out = EXP / "l4copy_analysis/modal_nf_v1"; out.mkdir(parents=True, exist_ok=True)
    ids = sorted(r["study_id"] for r in csv.DictReader(open("/root/assets/folds.csv")) if r["fold"] == "0")
    assert len(ids) == 825, len(ids)
    Path("/tmp/fold0_ids.txt").write_text("\n".join(ids) + "\n")
    env = {**os.environ, "D96_CACHE": str(VOL / "d96"), "V4_ART": "/root/v4art", "XAI_OUT": str(out), "XAI_IDS": "/tmp/fold0_ids.txt",
           "XAI_NF": "1", "XAI_LITE": "1", "XAI_NOCASES": "1", "XAI_QPRIOR": Path("/root/quota_prior.json").read_text(),
           "XAI_MODELS": "s2026=/root/v4art/weights_swa21_25.pt,s2027=/root/xai_extra/l4copy_s2027_swa21_25.pt,"
                         "r1self=/root/xai_extra/r1_self_s2026_swa21_25.pt", "OMP_NUM_THREADS": "8", "PYTHONUNBUFFERED": "1"}
    code = subprocess.run(["python", "-u", "/root/l4copy_xai.py"], env=env, cwd="/root").returncode
    volume.commit()
    if code:
        raise RuntimeError(f"l4copy nf exited {code}")


@app.function(image=image, gpu="A100-40GB", cpu=16, memory=65536, timeout=3600, volumes={"/root/vol": volume})
def r50_gl4_smoke(arms: str = "shared,adapter,l4copy"):
    """Exp r50-GL v4 smoke on MeluXina's GPU type (A100-40GB, 16 CPUs = one MeluXina lane; timings set the walltime and
    lane balance): each arm's whole code path on a few studies (32 steps, eval
    on 16 + 16 gap studies, SWA of 1 epoch incl. the dense LoRA delta, BN refresh on 8, predict), slim MS1 table."""
    volume.reload()
    base = {**os.environ, "D96_CACHE": str(VOL / "d96"), "D96_PHASE": "r50_gl", "PYTHONUNBUFFERED": "1",
            "GL_OUT": str(EXP / "run_r50_gl4_smoke/preflight"), "GL_CROPS": "/root/ms1_crops_slim"}
    if subprocess.run(["python", "-u", "/root/preflight_v4.py"], env=base, cwd="/root").returncode:
        raise RuntimeError("v4 preflight failed")
    for arm in arms.split(","):
        out = EXP / f"run_r50_gl4_smoke/{arm}"; out.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "D96_CACHE": str(VOL / "d96"), "D96_PHASE": "r50_gl", "PYTHONUNBUFFERED": "1",
               "GL_OUT": str(out), "GL_CROPS": "/root/ms1_crops_slim", "GL_SMOKE": "1", "V4_ARM": arm}
        proc = subprocess.Popen(["python", "-u", "/root/r50_gl4_runner.py"], env=env, cwd="/root", stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        with (out / "training.log").open("w", buffering=1) as log:
            for line in proc.stdout:
                print(f"[{arm}] {line}", end="", flush=True); log.write(line)
        code = proc.wait()
        volume.commit()
        if code:
            raise RuntimeError(f"r50 gl4 smoke {arm} exited {code}")
    out = EXP / "run_r50_gl4_smoke/bench"; out.mkdir(parents=True, exist_ok=True)
    with (out / "bench.log").open("w") as log:                   # capacity: loader vs GPU speed, memory headroom
        code = subprocess.run(["python", "-u", "/root/bench_v4_a100.py"], env={**base, "GL_OUT": str(out)}, cwd="/root",
                              stdout=log, stderr=subprocess.STDOUT).returncode
    print((out / "bench.log").read_text()[-4000:], flush=True)
    volume.commit()


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=4 * 3600, retries=1, volumes={"/root/vol": volume})
def r50_gl3(smoke: bool = False):
    """Exp r50-GL v3: one r50 from ImageNet, global windows + 4 regions per finding (8 findings, Baker's sagittal), per-study
    gate between each finding's two views, dual BN; smoke = the whole code path on a few studies (32 steps, timing)."""
    volume.reload()
    out = EXP / ("run_r50_gl3_smoke" if smoke else "run_r50_gl3"); out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "D96_CACHE": str(VOL / "d96"), "D96_PHASE": "r50_gl", "PYTHONUNBUFFERED": "1", "GL_OUT": str(out),
           "GL_CROPS": str(EXP / "run_sig_lv2/crop_probe"), "GL_POINTS4": "/root/multiplane_points.npz",
           "GL_SMOKE": "1" if smoke else "0"}
    proc = subprocess.Popen(["python", "-u", "/root/r50_gl3_runner.py"], env=env, cwd="/root", stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    with (out / "training.log").open("a", buffering=1) as log:
        for line in proc.stdout:
            print(line, end="", flush=True); log.write(line)
            if line.startswith(("EPOCH", "RUN_SUMMARY", "SMOKE_DONE")):
                volume.commit()
    code = proc.wait()
    volume.commit()
    if code:
        raise RuntimeError(f"r50 gl3 exited {code}; recovery.pt retained for retry")


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=1800, volumes={"/root/vol": volume})
def r50_gl_diag(version: str = "v2"):
    """Exp r50-GL diagnostic, inference only: SWA model on fold 0 + 825 training studies; heads, reader attention."""
    volume.reload()
    run = EXP / ("run_r50_gl" + ("" if version == "v1" else f"_{version}"))
    out = run / "diag"; out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "D96_CACHE": str(VOL / "d96"), "D96_PHASE": "r50_gl", "PYTHONUNBUFFERED": "1", "GL_VERSION": version,
           "GL_OUT": str(out), "GL_CROPS": str(EXP / "run_sig_lv2/crop_probe"), "GL_WEIGHTS": str(run / "weights_swa21_25.pt"),
           "DIAG_OUT": str(out)}
    code = subprocess.run(["python", "-u", "/root/diag_r50_gl_gpu.py"], env=env, cwd="/root").returncode
    volume.commit()
    if code:
        raise RuntimeError(f"r50 gl diag exited {code}")


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=2700, volumes={"/root/vol": volume})
def r50_gl_xai():
    """Exp r50-GL-XAI, inference only: r50-GL v2 SWA + the label-v2 r50 parent on fold 0 (Grad-CAM, reader attribution,
    input ablations) and 825 training studies (heads, reader attribution); panels for local rendering."""
    volume.reload()
    run = EXP / "run_r50_gl_v2"
    out = run / "xai"; out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "D96_CACHE": str(VOL / "d96"), "D96_PHASE": "r50_gl", "PYTHONUNBUFFERED": "1", "GL_VERSION": "v2",
           "GL_OUT": str(out), "GL_CROPS": str(EXP / "run_sig_lv2/crop_probe"), "GL_WEIGHTS": str(run / "weights_swa21_25.pt"),
           "LV2_WEIGHTS": str(EXP / "ref_labelv2/r50_labelv2_swa21_25.pt"), "XAI_OUT": str(out),
           "PANEL_IDS": "/root/xai_panels.json"}
    with (out / "xai.log").open("a", buffering=1) as log:
        proc = subprocess.Popen(["python", "-u", "/root/xai_r50_gl_gpu.py"], env=env, cwd="/root", stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True); log.write(line)
    code = proc.wait()
    volume.commit()
    if code:
        raise RuntimeError(f"r50 gl xai exited {code}")


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=3600, volumes={"/root/vol": volume})
def mp0():
    """Exp MP-0: frozen label-v2 r50 features of zoomed views at each finding's point in all 3 planes, inference only."""
    volume.reload()
    out = EXP / "run_sig_lv2/crop_probe/multiplane"; out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "D96_CACHE": str(VOL / "d96"), "D96_PHASE": "r50_pertarget", "PYTHONUNBUFFERED": "1",
           "LV2_WEIGHTS": str(EXP / "ref_labelv2/r50_labelv2_swa21_25.pt"), "MP_POINTS": "/root/multiplane_points.npz",
           "MP_REF": "/root/lv2_val_swa.json", "MP_OUT": str(out)}
    with (out / "mp.log").open("a", buffering=1) as log:
        proc = subprocess.Popen(["python", "-u", "/root/extract_multiplane_features_gpu.py"], env=env, cwd="/root",
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True); log.write(line)
    code = proc.wait()
    volume.commit()
    if code:
        raise RuntimeError(f"mp0 exited {code}")


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=7200, volumes={"/root/vol": volume})
def ms1_crops():
    """Exp MS1-CP: frozen MS1 global + self-chosen crop features for all 4,349 studies, inference only."""
    volume.reload()
    env = {**os.environ, "D96_CACHE": str(VOL / "d96"), "D96_PHASE": "sig_lv2", "PYTHONUNBUFFERED": "1",
           "CP_OUT": str(EXP / "run_sig_lv2/crop_probe"), "MS1_WEIGHTS": str(EXP / "run_sig_lv2/weights_swa21_25.pt")}
    code = subprocess.run(["python", "-u", "/root/extract_ms1_crop_features_gpu.py"], env=env, cwd="/root").returncode
    volume.commit()
    if code:
        raise RuntimeError(f"ms1 crops exited {code}")


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=3600, volumes={"/root/vol": volume})
def ms1_l1():
    """Exp MS1-L1: slice-use test (eval_slice_use_gpu.py): MS1 and the label-v2 r50 on fold 0 with normal / grey / wide
    windows, inference only."""
    volume.reload()
    out = EXP / "run_sig_lv2/l1_slice_use"
    for tag, phase, w in (("ms1", "sig_lv2", EXP / "run_sig_lv2/weights_swa21_25.pt"),
                          ("lv2r50", "r50_pertarget", EXP / "ref_labelv2/r50_labelv2_swa21_25.pt")):
        env = {**os.environ, "D96_CACHE": str(VOL / "d96"), "D96_PHASE": phase, "PYTHONUNBUFFERED": "1",
               "SLICE_OUT": str(out), "SLICE_TAG": tag, "SLICE_WEIGHTS": str(w)}
        code = subprocess.run(["python", "-u", "/root/eval_slice_use_gpu.py"], env=env, cwd="/root").returncode
        volume.commit()
        if code:
            raise RuntimeError(f"ms1 l1 {tag} exited {code}")


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=1800, volumes={"/root/vol": volume})
def swin_profile(phase: str = "swin3d_b"):
    """Model B: preflight checks + timing on real d96n studies (swin3d_d96n_runner.py, SWIN_MODE=profile); no run."""
    volume.reload()
    out = EXP / ("swin3d_profile" if phase == "swin3d_b" else f"{phase}_profile")
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "D96_CACHE": str(VOL / "d96n"), "SWIN_PHASE": phase, "SWIN_MODE": "profile",
           "D96_OUT": str(out), "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(["python", "-u", "/root/swin3d_d96n_runner.py"], env=env, cwd="/root",
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    with (out / "profile.log").open("w", buffering=1) as log:
        for line in proc.stdout:
            print(line, end="", flush=True); log.write(line)
    code = proc.wait()
    volume.commit()
    if code:
        raise RuntimeError(f"swin profile exited {code}")


@app.function(image=image, gpu="H100", cpu=16, memory=65536, timeout=2400, volumes={"/root/vol": volume})
def bench_b():
    """Model B encoder benchmark (bench_model_b_encoders.py): IO throughput, fused-attention equivalence, timing."""
    volume.reload()
    out = EXP / "model_b_bench"
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "D96_CACHE": str(VOL / "d96n"), "SWIN_PHASE": "swin3d_b", "D96_OUT": str(out),
           "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(["python", "-u", "/root/bench_model_b_encoders.py"], env=env, cwd="/root",
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    with (out / "bench.log").open("w", buffering=1) as log:
        for line in proc.stdout:
            print(line, end="", flush=True); log.write(line)
    code = proc.wait()
    volume.commit()
    if code:
        raise RuntimeError(f"bench exited {code}")


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=1800, volumes={"/root/vol": volume})
def fusion_features():
    """Exp F0: fold-0 per-finding pooled features of 1e and B2 (extract_fusion_features_gpu.py), inference only."""
    volume.reload()
    out = EXP / "fusion_f0"
    runs = {"r50": {"D96_PHASE": "r50_pertarget", "D96_CACHE": str(VOL / "d96"),
                    "R50_WEIGHTS": str(EXP / "run_r50_pertarget/weights_swa21_25.pt")},
            "swin": {"SWIN_PHASE": "swin3d_b2", "D96_CACHE": str(VOL / "d96n"),
                     "SWIN_B2_WEIGHTS": str(EXP / "run_swin3d_b2/weights_swa11_15.pt")}}
    for mode, extra in runs.items():
        env = {**os.environ, **extra, "FUSION_MODE": mode, "FUSION_OUT": str(out), "PYTHONUNBUFFERED": "1"}
        code = subprocess.run(["python", "-u", "/root/extract_fusion_features_gpu.py"], env=env, cwd="/root").returncode
        volume.commit()
        if code:
            raise RuntimeError(f"fusion features ({mode}) exited {code}")


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=1800, volumes={"/root/vol": volume})
def joint_profile():
    """Exp J1: parity of the trained 1e / B2 weights inside the joint model, gradient paths, timing; no run."""
    volume.reload()
    out = EXP / "joint_j1_profile"
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "JOINT_CACHE_R": str(VOL / "d96"), "JOINT_CACHE_S": str(VOL / "d96n"), "JOINT_OUT": str(out),
           "JOINT_MODE": "profile", "JOINT_CHECK_DIR": str(EXP), "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(["python", "-u", "/root/joint_r50_swin_runner.py"], env=env, cwd="/root",
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    with (out / "profile.log").open("w", buffering=1) as log:
        for line in proc.stdout:
            print(line, end="", flush=True); log.write(line)
    code = proc.wait()
    volume.commit()
    if code:
        raise RuntimeError(f"joint profile exited {code}")


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=1800, volumes={"/root/vol": volume})
def joint_xai():
    """Exp J1-XAI: instrumented fold-0 inference of J1 (SWA 16-20) + the label-v2 ResNet-50 reference; no training."""
    volume.reload()
    out = EXP / "joint_j1_xai"
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "JOINT_CACHE_R": str(VOL / "d96"), "JOINT_CACHE_S": str(VOL / "d96n"), "JOINT_OUT": str(out),
           "XAI_J1_WEIGHTS": str(EXP / "run_joint_j1/weights_swa16_20.pt"),
           "XAI_LV2_WEIGHTS": str(EXP / "ref_labelv2/r50_labelv2_swa21_25.pt"),
           "XAI_CHECK_DIR": str(EXP / "run_joint_j1"), "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(["python", "-u", "/root/joint_j1_xai.py"], env=env, cwd="/root",
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    with (out / "xai.log").open("w", buffering=1) as log:
        for line in proc.stdout:
            print(line, end="", flush=True); log.write(line)
    code = proc.wait()
    volume.commit()
    if code:
        raise RuntimeError(f"joint xai exited {code}")


@app.function(image=image, gpu="H100", cpu=12, memory=65536, timeout=1800, volumes={"/root/vol": volume})
def j2x_profile():
    """Exp J2-X: alignment coverage, mapping self-check, 1e / B2 parity with Delta = 0, gradient paths, timing; no run."""
    volume.reload()
    out = EXP / "j2x_profile"
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "JOINT_CACHE_R": str(VOL / "d96"), "JOINT_CACHE_S": str(VOL / "d96n"), "JOINT_OUT": str(out),
           "J2X_MODE": "profile", "J2X_CHECK_DIR": str(EXP), "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(["python", "-u", "/root/j2x_runner.py"], env=env, cwd="/root",
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    with (out / "profile.log").open("w", buffering=1) as log:
        for line in proc.stdout:
            print(line, end="", flush=True); log.write(line)
    code = proc.wait()
    volume.commit()
    if code:
        raise RuntimeError(f"j2x profile exited {code}")


MEDSIGLIP_REV = "9cea28a1a1195f665105faa6e8544c112fd960a4"          # google/medsiglip-448, 2025-07-10
MEDSIGLIP_SAFETENSORS_SHA = "3e1722b8100cca8b87cfe16e42140737aa6d312854e9e6bc24dff6db70e21d72"


@app.function(image=image, cpu=8, memory=32768, timeout=3600, secrets=[_hf_secret()], volumes={"/root/vol": volume})
def medsiglip_weights():
    """Exp MS1 step 0: google/medsiglip-448 vision tower -> timm vit_so400m_patch14_siglip_384 (img_size 448) state dict.
    safetensors only (no pickle, no remote code), pinned revision and file hash; strict timm load; parity of the pooled
    embedding vs transformers' SiglipVisionModel on fixed random inputs (fp32 CPU); prints the sha256 the runner pins."""
    import hashlib
    import timm
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import SiglipVisionConfig, SiglipVisionModel
    volume.reload()
    src = VOL / "hf/medsiglip-448"
    tok = os.environ["HF_TOKEN"]
    st = Path(hf_hub_download("google/medsiglip-448", "model.safetensors", revision=MEDSIGLIP_REV, token=tok, local_dir=src))
    cf = Path(hf_hub_download("google/medsiglip-448", "config.json", revision=MEDSIGLIP_REV, token=tok, local_dir=src))
    h = hashlib.sha256()
    with st.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    assert h.hexdigest() == MEDSIGLIP_SAFETENSORS_SHA, h.hexdigest()
    vc = json.loads(cf.read_text())["vision_config"]
    sd = {k[len("vision_model."):]: v.float() for k, v in load_file(str(st)).items() if k.startswith("vision_model.")}
    d, depth = vc["hidden_size"], vc["num_hidden_layers"]
    t = {"patch_embed.proj.weight": sd["embeddings.patch_embedding.weight"],
         "patch_embed.proj.bias": sd["embeddings.patch_embedding.bias"],
         "pos_embed": sd["embeddings.position_embedding.weight"][None],
         "norm.weight": sd["post_layernorm.weight"], "norm.bias": sd["post_layernorm.bias"],
         "attn_pool.latent": sd["head.probe"],
         "attn_pool.q.weight": sd["head.attention.in_proj_weight"][:d], "attn_pool.q.bias": sd["head.attention.in_proj_bias"][:d],
         "attn_pool.kv.weight": sd["head.attention.in_proj_weight"][d:], "attn_pool.kv.bias": sd["head.attention.in_proj_bias"][d:]}
    for a, b in (("attn_pool.proj", "head.attention.out_proj"), ("attn_pool.norm", "head.layernorm"),
                 ("attn_pool.mlp.fc1", "head.mlp.fc1"), ("attn_pool.mlp.fc2", "head.mlp.fc2")):
        t[f"{a}.weight"], t[f"{a}.bias"] = sd[f"{b}.weight"], sd[f"{b}.bias"]
    for i in range(depth):
        p, q = f"encoder.layers.{i}.", f"blocks.{i}."
        for a, b in (("norm1", "layer_norm1"), ("norm2", "layer_norm2"), ("attn.proj", "self_attn.out_proj"),
                     ("mlp.fc1", "mlp.fc1"), ("mlp.fc2", "mlp.fc2")):
            t[f"{q}{a}.weight"], t[f"{q}{a}.bias"] = sd[f"{p}{b}.weight"], sd[f"{p}{b}.bias"]
        for w in ("weight", "bias"):
            t[f"{q}attn.qkv.{w}"] = torch.cat([sd[f"{p}self_attn.{n}_proj.{w}"] for n in "qkv"])
    m = timm.create_model("vit_so400m_patch14_siglip_384", img_size=vc["image_size"], pretrained=False, num_classes=0,
                          act_layer="gelu_tanh").eval()        # timm 1.0.29 has no "_448" name; HF gelu_pytorch_tanh
    m.load_state_dict(t, strict=True)
    ref = SiglipVisionModel(SiglipVisionConfig(**vc)).eval()
    ref.load_state_dict({f"vision_model.{k}": v for k, v in sd.items()}, strict=True)
    x = torch.randn(2, 3, 448, 448, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        a, b = m(x), ref(pixel_values=x).pooler_output
        hs = ref(pixel_values=x, output_hidden_states=True).hidden_states       # embeddings, then after each layer
        z = m._pos_embed(m.patch_embed(x))
        stages = {"embed": (z - hs[0]).abs().max().item()}
        for i, blk in enumerate(m.blocks):
            z = blk(z)
            if i in (0, 13, depth - 1):
                stages[f"block{i}"] = (z - hs[i + 1]).abs().max().item()
        print("PARITY_STAGES " + json.dumps({k: f"{v:.2e}" for k, v in stages.items()}), flush=True)
    diff, cos = (a - b).abs().max().item(), torch.nn.functional.cosine_similarity(a, b).min().item()
    print(f"PARITY max|timm - transformers| {diff:.2e}, min cosine {cos:.6f}, |emb| {b.norm(dim=1).mean():.2f}", flush=True)
    assert diff < 1e-3 * b.abs().max().item() and cos > 0.9999, (diff, cos)
    out = VOL / "pretrained/medsiglip448_timm.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), out)
    volume.commit()
    print(f"MEDSIGLIP_TIMM_OK {out} sha256 {hashlib.sha256(out.read_bytes()).hexdigest()} "
          f"params {sum(v.numel() for v in t.values()) / 1e6:.1f}M", flush=True)


@app.function(image=image, cpu=0.25, memory=512, timeout=24 * 3600, max_containers=1)
def coordinate(phase: str = "exp1", resume: bool = False):
    if not resume and not phase.startswith(("swin3d", "joint", "j2x")):   # d96 / d96n already on the volume
        print("TRANSFERRED", transfer.remote(phase), flush=True)
        if phase == "r50_ssl_pertarget":
            ssl.remote("train")
        preflight.remote(phase)
    (train_b if phase.startswith("swin3d") else train).remote(phase)
    print("RUN_COMPLETE", flush=True)


@app.local_entrypoint()
def main(check_only: bool = False, phase: str = "exp1", ssl_bench: bool = False, audit_v3_run: bool = False,
         resume: bool = False, profile_shift_run: bool = False, shift_dx_run: bool = False,
         swin_profile_run: bool = False, bench_b_run: bool = False, fusion_f0_run: bool = False,
         joint_profile_run: bool = False, joint_xai_run: bool = False, j2x_profile_run: bool = False,
         medsiglip_weights_run: bool = False, ms1_q1_run: bool = False, ms1_l1_run: bool = False,
         ms1_maps_run: bool = False, ms1_crops_run: bool = False, ms1_gl_run: bool = False,
         ms1_gl_smoke: bool = False, r50_gl_smoke: bool = False, r50_gl_run: bool = False,
         r50_gl_version: str = "v1", r50_gl_diag_run: bool = False, r50_gl_xai_run: bool = False,
         mp0_run: bool = False, r50_gl3_smoke: bool = False, r50_gl3_run: bool = False, r50_gl3_selfloc_run: bool = False,
         r50_gl3_xai_run: bool = False):
    if r50_gl3_xai_run:                               # Exp r50-GL v3 XAI (needs run_r50_gl3 SWA)
        r50_gl3_xai.remote()
        return
    if r50_gl3_selfloc_run:                           # Exp SL-0 (v3 locates its own regions; needs run_r50_gl3 SWA)
        r50_gl3_selfloc.remote()
        return
    if r50_gl3_smoke or r50_gl3_run:                  # Exp r50-GL v3 (smoke = whole path on a few studies, timing)
        r50_gl3.remote(smoke=r50_gl3_smoke)
        return
    if mp0_run:                                       # Exp MP-0: multi-plane zoom features, inference only
        mp0.remote()
        return
    if r50_gl_xai_run:                                # Exp r50-GL-XAI: Grad-CAM, reader attribution, ablations
        r50_gl_xai.remote()
        return
    if r50_gl_diag_run:                               # Exp r50-GL diagnostic: train vs fold-0 complementarity, reader attention
        r50_gl_diag.remote(version=r50_gl_version)
        return
    if r50_gl_smoke or r50_gl_run:                    # Exp r50-GL: label-v2 r50 + per-finding crops (smoke = whole path)
        r50_gl.remote(smoke=r50_gl_smoke, version=r50_gl_version)
        return
    if ms1_gl_smoke or ms1_gl_run:                    # Exp MS1-GL: global + local MS1 (smoke = timing only)
        ms1_gl.remote(smoke=ms1_gl_smoke)
        return
    if ms1_crops_run:                                 # Exp MS1-CP: crop-probe features, inference only
        ms1_crops.remote()
        return
    if ms1_maps_run:                                  # Exp LOC-0-MS1: token maps, inference only
        ms1_maps.remote()
        return
    if ms1_l1_run:                                    # Exp MS1-L1: slice-use test, inference only
        ms1_l1.remote()
        return
    if ms1_q1_run:                                    # Exp MS1-Q1: test-time resolution, inference only
        ms1_q1.remote()
        return
    if medsiglip_weights_run:                          # Exp MS1 step 0 (CPU): convert + parity, prints the sha to pin
        medsiglip_weights.remote()
        return
    if j2x_profile_run:
        j2x_profile.remote()
        return
    if joint_xai_run:
        joint_xai.remote()
        return
    if joint_profile_run:
        joint_profile.remote()
        return
    if fusion_f0_run:
        fusion_features.remote()
        return
    if bench_b_run:
        bench_b.remote()
        return
    if swin_profile_run:
        swin_profile.remote(phase if phase.startswith("swin3d") else "swin3d_b")
        return
    if shift_dx_run:
        shift_dx.remote()
        return
    if profile_shift_run:
        print("PROFILE", profile_shift.remote(), flush=True)
        return
    if audit_v3_run:
        audit_v3.remote()
        return
    assert phase in ("exp1", "continue", "r50", "r50_pertarget", "r50_ctx", "r50_border", "r50_state", "r50_plane", "r50_plane_pertarget",
                     "r50_pertarget_seed", "r50_native", "r50_native_pertarget", "r50_ssl_pertarget",
                     "r50_swa_cyclic", "r50_series_alloc", "r50_spatial", "r50_os16_gap", "r50_spatial_os32", "r50_spatial_v31",
                     "r50_dual_guided", "r50_r949", "r50_r949_aug", "cnx_pertarget", "r50_aug2", "cnx_tune", "cnx_in12k", "cnx_dinov3", "r50_sev", "r50_tags", "r50_shift", "swin3d_b", "swin3d_b2",
                     "r50_rulecap", "r50_rulecap_noeff", "joint_j1", "j2x", "r50_student_state", "r50_student_a050",
                     "sig_lv2", *TEACHER_PHASES), phase
    if ssl_bench:
        ssl.remote("bench")
        return
    if check_only:
        print("TRANSFERRED", transfer.remote(phase), flush=True)
        preflight.remote(phase)
        return
    # spawn + `modal run --detach`: the run no longer depends on this machine's connection staying up
    call = coordinate.spawn(phase, resume)
    print(f"SPAWNED {call.object_id}", flush=True)

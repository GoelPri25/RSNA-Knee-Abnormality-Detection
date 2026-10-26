"""Exp J1-XAI: instrumented fold-0 inference of the trained J1 joint model (SWA 16-20) and the label-v2 ResNet-50
reference, for CPU ablations and attribution. No training; validation labels are not read here.

The fused head is per-finding attention over the union of ResNet window tokens and Swin stage-4 tokens with no
token-to-token interaction, so saving each token's fused attention score and its contribution (u . clsW) makes the
fused logit, branch-only logits, gate sweeps and window / series occlusion exactly recomputable on CPU. The ResNet
backbone is per window and the Swin encoder per series, so removing windows or a series is exact input occlusion.

Saved (OUT):
  study.npz    ids; logits fused / r / s (J1) and lv2 [n, 12]; fused ResNet mass mR, within-branch fused logits zR,
               zS [n, 12]; gate [2, 12]; clsb; per window: ctx [n, 94, 3], fused score / contribution, J1 R-head and
               lv2 attention logit / contribution [n, 94, 12]; per-finding attention-pooled features (fp16) of J1-R
               and lv2 [n, 12, 2048], window-mean features [n, 2048], J1 S-head [n, 12, 768]; token offsets.
  swin.npz     all Swin tokens concatenated (offsets in study.npz): series slot, depth t, grid cell (10 x 10), fused
               score / contribution and S-head attention logit / contribution [N, 12] (fp16).
  cam.npz      exact class-activation maps of the J1 R head and lv2 (layer-4 map through LayerNorm and clsW, summing to
               the window's logit contribution minus beta . clsW) for each finding's top-8 windows by head attention.
Env: as joint_r50_swin_runner (JOINT_CACHE_R, JOINT_CACHE_S, JOINT_OUT) + XAI_J1_WEIGHTS, XAI_LV2_WEIGHTS, XAI_CHECK_DIR.
"""
import json
import math
import os
import time
from pathlib import Path

os.environ["JOINT_MODE"] = "xai"
import numpy as np
import torch

import joint_r50_swin_runner as j

rr, s, dev = j.rr, j.s, j.dev
OUT = Path(os.environ["JOINT_OUT"])
TOP = 8


def raptor_parts(R, x):
    """Window features, head attention logits / contributions, per-finding pooled features, CAM inputs."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fm = R.backbone.forward_features(x)                                     # [K, 2048, 10, 10]
    fm = fm.float()
    f = fm.mean((2, 3))
    h = R.norm(f)
    sa = R.att(h)                                                               # [K, 12]
    ca = h @ R.clsW.T                                                           # [K, 12]
    a = torch.softmax(sa, 0)
    logit = (a * ca).sum(0) + R.clsb
    pooled = a.T @ h                                                            # [12, 2048]
    top = a.topk(TOP, dim=0).indices.T                                          # [12, TOP]
    ln = R.norm
    mu, var = f.mean(-1), f.var(-1, unbiased=False)
    cams = []
    for n in range(12):
        k = top[n]
        z = (fm[k] - mu[k, None, None, None]) / torch.sqrt(var[k] + ln.eps)[:, None, None, None]
        cams.append(torch.einsum("kchw,c->khw", z, ln.weight * R.clsW[n]) / (z.shape[2] * z.shape[3]))
    return h, sa, ca, logit, pooled, f.mean(0), top, torch.stack(cams)


@torch.no_grad()
def run():
    t0 = time.monotonic()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # fp32 heads exact
    if s.CFG["attn"] == "sdpa":
        s.use_fused_attention()
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, meta_r, meta_s, _, val_ids, _ = j.setup()
    prep = s.Prep(*s.kinetics_norm(), dev)
    m = j.JointNet(pretrained=False)
    m.load_state_dict(torch.load(os.environ["XAI_J1_WEIGHTS"], map_location="cpu", weights_only=True), strict=True)
    m = m.to(dev).eval(); m.R.to(memory_format=torch.channels_last)
    lv2 = rr.build(rr.CFG["eval_res"], False)
    lv2.load_state_dict(torch.load(os.environ["XAI_LV2_WEIGHTS"], map_location="cpu", weights_only=True), strict=True)
    lv2 = lv2.to(dev).eval().to(memory_format=torch.channels_last)
    S = m.S
    st = {k: [] for k in ("lf", "lr", "ls", "l2", "mR", "zR", "zS", "ctx", "sr", "cr", "har", "car", "hl2", "cl2",
                          "pR", "pL", "fR", "fL", "pS", "nS", "topR", "topL", "camR", "camL")}
    tok = {k: [] for k in ("ser", "t", "cell", "ss", "cs", "hs", "chs")}
    dl = j.loader(j.Both(val_ids, meta_r, meta_s, {}, rr.CFG["eval_windows"], False), False, 1, 0)
    for i, (x, sag, _, _, ctx, vols, slots) in enumerate(dl):
        xr = rr.prepare(x, sag, False, rr.CFG["eval_res"], dev)
        hr, sa, ca, lr_, pR, fR, topR, camR = raptor_parts(m.R, xr)
        _, sa2, ca2, l2, pL, fL, topL, camL = raptor_parts(lv2, xr)
        ur = m.pr(hr) + m.kind[0]
        sr, cr = m.att(ur), ur @ m.clsW.T
        series = prep(vols[0], slots[0], False, s.CFG["res"])
        K = len(ur)
        if series:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                toks = [S.encode(v, jj) for jj, v in series]
            ids = torch.cat([torch.full((len(t),), jj, device=dev) for t, (jj, _) in zip(toks, series)])
            hs = S.norm(torch.cat(toks).float()) + S.series(ids)
            hsa, hsc = S.att(hs), hs @ S.clsW.T
            ab = torch.softmax(hsa, 0)
            ls = (ab * hsc).sum(0) + S.clsb
            pS = ab.T @ hs
            us = m.ps(hs) + m.kind[1]
            ss, cs = m.att(us), us @ m.clsW.T
            N = len(us)
            prior = torch.cat([(m.gate[0] - math.log(K)).expand(K, 12), (m.gate[1] - math.log(N)).expand(N, 12)])
            a = torch.softmax(torch.cat([sr, ss]) + prior, 0)
            lf = (a * torch.cat([cr, cs])).sum(0) + m.clsb
            mR = a[:K].sum(0)
            zS = (torch.softmax(ss, 0) * cs).sum(0)
            for t, (jj, _) in zip(toks, series):
                assert len(t) % 100 == 0
                tok["ser"].append(np.full(len(t), jj, np.int8))
                tok["t"].append(np.repeat(np.arange(len(t) // 100, dtype=np.int16), 100))
                tok["cell"].append(np.tile(np.arange(100, dtype=np.int8), len(t) // 100))
            for k, v in (("ss", ss), ("cs", cs), ("hs", hsa), ("chs", hsc)):
                tok[k].append(v.cpu().numpy().astype(np.float16))
        else:
            N = 0
            ls, pS = S.clsb.clone(), torch.zeros(12, S.clsW.shape[1], device=dev)
            lf = (torch.softmax(sr, 0) * cr).sum(0) + m.clsb
            mR, zS = torch.ones(12, device=dev), torch.zeros(12, device=dev)
        zR = (torch.softmax(sr, 0) * cr).sum(0)
        vals = {"lf": lf, "lr": lr_, "ls": ls, "l2": l2, "mR": mR, "zR": zR, "zS": zS, "ctx": ctx[0], "sr": sr,
                "cr": cr, "har": sa, "car": ca, "hl2": sa2, "cl2": ca2, "topR": topR, "topL": topL}
        for k, v in vals.items():
            st[k].append(v.float().cpu().numpy() if v.is_floating_point() else v.cpu().numpy())
        for k, v in (("pR", pR), ("pL", pL), ("fR", fR), ("fL", fL), ("pS", pS), ("camR", camR), ("camL", camL)):
            st[k].append(v.cpu().numpy().astype(np.float16))
        st["nS"].append(N)
        if i % 100 == 0:
            print(f"XAI {i}/{len(val_ids)} K {K} N {N} {time.monotonic() - t0:.0f}s", flush=True)

    out = {k: np.stack(v) for k, v in st.items()}
    out["off"] = np.concatenate([[0], np.cumsum(out["nS"])])
    out["ids"] = np.array(val_ids)
    out["gate"], out["clsb"] = m.gate.cpu().numpy(), m.clsb.cpu().numpy()
    np.savez(OUT / "study.npz", **out)
    np.savez(OUT / "swin.npz", **{k: np.concatenate(v) for k, v in tok.items()})

    # parity with the run's own fold-0 predictions (bf16 heads there, fp32 here)
    chk = Path(os.environ["XAI_CHECK_DIR"])
    sig = lambda z: 1 / (1 + np.exp(-z))
    res = {}
    for key, name in (("lf", "val_swa.json"), ("lr", "val_swa_r50.json"), ("ls", "val_swa_swin.json")):
        saved = {q["study_id"]: q["probabilities"] for q in json.loads((chk / name).read_text())["predictions"]}
        res[name] = float(np.abs(sig(out[key]) - np.array([saved[x] for x in val_ids])).max())
        print(f"PARITY {key} vs {name}: max |dp| {res[name]:.4f}", flush=True)
    share = np.load(chk / "val_swa_share.npy")
    res["share"] = float(np.abs(share - out["mR"]).max()) if share.shape == out["mR"].shape else None
    print(f"PARITY share: max |d| {res['share']} (shape {share.shape})", flush=True)
    (OUT / "xai_summary.json").write_text(json.dumps({"parity": res, "n": len(val_ids), "tokens": int(out["off"][-1]),
                                                      "seconds": round(time.monotonic() - t0)}, indent=1))
    assert res["val_swa.json"] <= 0.02 and res["val_swa_r50.json"] <= 0.02 and res["val_swa_swin.json"] <= 0.02
    print(f"XAI_OK {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    run()

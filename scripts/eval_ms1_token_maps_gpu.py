"""Exp LOC-0-MS1: per-finding token maps of MS1 on fold 0 (inference only; Modal H100). Does MS1 localise by itself?

MS1 = MedSigLIP-448 (timm SoViT-400M/14, MAP pooling) + Raptor head: window feature f_k = MAP(tokens_k), h_k = LN(f_k),
per-finding window attention a_kn, logit_n = sum_k a_kn (h_k . W_n) + b_n. Per window and finding, the window's
contribution c_kn = a_kn (h_k . W_n) with a_kn held fixed (as the r50 CAM of LOC-0) is back-propagated to the final
normed patch tokens (32 x 32 over the whole 140 mm field, 4.4 mm cells); token map = positive part of sum_c grad * act
(gradient x activation). Saved as column / row profiles (camx along image columns, camy along rows) like the r50 audit.
Also the MAP head's own attention over tokens (finding-agnostic; latent query, heads averaged) as profiles.
Same 94 eval centres as LOC-0 (asserted against the r50 file's centres). Scored on CPU by audit_cam_localisation.py.

Env: D96_CACHE, D96_PHASE=sig_lv2, MS1_WEIGHTS, MAPS_OUT.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import coatnet_d96_runner as r

assert os.environ["D96_PHASE"] == "sig_lv2"
OUT = Path(os.environ["MAPS_OUT"]); OUT.mkdir(parents=True, exist_ok=True)
G = 32
MAP_OK = {"ok": True, "err": None}


def map_attention(pool, tok):
    """[K, N] MAP latent-query attention over tokens, heads averaged (timm AttentionPoolLatent, recomputed)."""
    assert getattr(pool, "pos_embed", None) is None and pool.latent.shape[1] == 1
    B, N, C = tok.shape
    q = pool.q(pool.latent.expand(B, -1, -1)).reshape(B, 1, pool.num_heads, pool.head_dim).transpose(1, 2)
    kv = pool.kv(tok).reshape(B, N, 2, pool.num_heads, pool.head_dim).permute(2, 0, 3, 1, 4)
    k = kv[0]
    q, k = pool.q_norm(q), pool.k_norm(k)
    att = torch.softmax((q * pool.scale) @ k.transpose(-2, -1), dim=-1)            # [B, H, 1, N]
    return att.mean(1)[:, 0]


def main():
    dev = torch.device("cuda")
    m = r.build(r.CFG["eval_res"], pretrained=False)
    m.load_state_dict(torch.load(os.environ["MS1_WEIGHTS"], map_location="cpu", weights_only=True), strict=True)
    m = m.to(dev).eval()
    bb = m.backbone
    assert getattr(bb, "global_pool", None) == "map" and bb.num_prefix_tokens == 0
    ref = r.load_labels(r.ASSETS / "validation_labels.json", "validation")
    meta = r.read_manifest()
    ids = sorted(ref)
    assert len(ids) == 825
    ds = r.Studies(ids, meta, {}, r.CFG["eval_windows"], False)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=r.CFG["workers"], pin_memory=True)
    S, K = len(ids), r.CFG["eval_windows"]
    camx = np.zeros((S, K, 12, G), np.float16); camy = np.zeros((S, K, 12, G), np.float16)
    mapx = np.zeros((S, K, G), np.float16); mapy = np.zeros((S, K, G), np.float16)
    att = np.zeros((S, K, 12), np.float16); logits = np.zeros((S, 12), np.float32); ref_logit = np.zeros((S, 12), np.float32)
    centre = np.zeros((S, K), np.int16)
    t0 = time.time()
    for i, (x, sag, _, _, ctx) in enumerate(dl):
        s = ids[i]; mt = meta[s]
        c = np.clip(r.centres(mt["mask"], K, False, None), 1, r.CFG["maxs"] - 2); centre[i] = c
        xp = r.prepare(x, sag, False, r.CFG["eval_res"], dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ref_logit[i] = m(xp, 1, ctx.to(dev)).float()[0].cpu().numpy()
            tok = bb.forward_features(xp)
        tok = tok.float().detach().requires_grad_(True)                           # [K, 1024, 1152]
        assert tok.shape[1] == G * G
        f = bb.forward_head(tok)
        h = m.norm(f).unsqueeze(0)                                                # [1, K, d]
        a = torch.softmax(m.att(h).float(), dim=1)[0].detach()                    # [K, 12]
        contrib = a * (h[0] @ m.clsW.float().T)                                   # [K, 12]
        logits[i] = (contrib.sum(0) + m.clsb.float()).detach().cpu().numpy()
        att[i] = a.cpu().numpy()
        for n in range(12):
            g, = torch.autograd.grad(contrib[:, n].sum(), tok, retain_graph=n < 11)
            mp = (g * tok.detach()).sum(-1).clamp_min(0).view(K, G, G)                   # [K, rows, cols]
            camx[i, :, n] = mp.sum(1).cpu().numpy(); camy[i, :, n] = mp.sum(2).cpu().numpy()
        if MAP_OK["ok"]:                                                          # optional: never lose the main maps
            try:
                with torch.no_grad():
                    ma = map_attention(bb.attn_pool, tok).view(K, G, G)
                mapx[i] = ma.sum(1).cpu().numpy(); mapy[i] = ma.sum(2).cpu().numpy()
            except Exception as e:                                                # noqa: BLE001
                MAP_OK.update(ok=False, err=repr(e)); print("MAP attention skipped:", repr(e), flush=True)
        if i % 100 == 0:
            print(f"MAPS {i}/{S} {time.time() - t0:.0f}s max|logit-ref| {np.abs(logits[i] - ref_logit[i]).max():.4f}", flush=True)
    np.savez_compressed(OUT / "ms1_token_maps_fold0.npz", ids=np.array(ids), centre=centre, camx=camx, camy=camy,
                        mapx=mapx, mapy=mapy, att=att, logits=logits, ref_logits=ref_logit)
    log = {"studies": S, "grid": G, "field_mm": 140.0, "seconds": round(time.time() - t0),
           "max_abs_logit_fp32_vs_bf16": float(np.abs(logits - ref_logit).max()), "map_attention": MAP_OK}
    (OUT / "ms1_token_maps_log.json").write_text(json.dumps(log, indent=1) + "\n")
    print("MAPS_DONE", json.dumps(log), flush=True)


if __name__ == "__main__":
    main()

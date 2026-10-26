"""Exp MS1-CP (crop probe): does a zoomed crop chosen by MS1's own token maps add signal to MS1's whole-field view?
Inference only (Modal H100); the probe itself is scripts/audit_ms1_crop_probe.py (CPU).

MS1 = MedSigLIP-448 (timm SoViT-400M/14, MAP pooling) + Raptor head, frozen (run_sig_lv2 SWA weights). All 4,349
labelled studies (fold 0 first, then the 3,524 training-fold studies), the 94 eval windows. Per study and finding n:
  global  g_n = sum_k a_kn h_k  (MS1's own pooled feature: logit_n = g_n . W_n + b_n exactly)
  crops   the TOP windows by a_kn; on each, the gradient x activation map of a_kn (h_k . W_n) (a held fixed, as
          LOC-0-MS1) over the 32 x 32 tokens -> centroid -> a CROP px (46.7 mm) square of the 384 px cache image
          (all 3 slice channels; clamped inside the image), resized to 448 like the whole field (3.5x zoom, 1.46 mm
          per token) -> frozen encoder -> MAP -> MS1's LayerNorm -> h_crop;
          c_n = sum over the TOP crops of a_kn (renormalised) h_crop;  zero-shot crop logit = c_n . W_n + b_n.
Saved in chunks of CHUNK studies (a rerun skips finished chunks). First CHECK studies: pooled logits vs the model's
own forward (bf16).

Env: D96_CACHE, D96_PHASE=sig_lv2, MS1_WEIGHTS, CP_OUT.
"""
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import coatnet_d96_runner as r

assert os.environ["D96_PHASE"] == "sig_lv2"
OUT = Path(os.environ["CP_OUT"]); OUT.mkdir(parents=True, exist_ok=True)
G, TOP, CROP, CHUNK, CHECK = 32, 4, 128, 500, 20
IMG = r.CFG["img"]; PX = IMG / G                                                  # 12 cache px per token


def main():
    dev = torch.device("cuda")
    m = r.build(r.CFG["eval_res"], pretrained=False)
    m.load_state_dict(torch.load(os.environ["MS1_WEIGHTS"], map_location="cpu", weights_only=True), strict=True)
    m = m.to(dev).eval(); bb = m.backbone
    for p in m.parameters():
        p.requires_grad_(False)
    assert getattr(bb, "global_pool", None) == "map" and bb.num_prefix_tokens == 0
    labels = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
    assert r.sha(r.ASSETS / "folds.csv") == r.CFG["sha256"]["folds"]
    fold = {x["study_id"]: x["fold"] for x in csv.DictReader((r.ASSETS / "folds.csv").open())}
    meta = r.read_manifest()
    val = sorted(s for s in labels if fold[s] == "0"); tr = sorted(s for s in labels if fold[s] != "0")
    assert len(val) == 825 and len(tr) == 3524 and all(s in meta for s in val + tr)
    ids = val + tr
    K, d = r.CFG["eval_windows"], bb.num_features
    W, b = m.clsW.float(), m.clsb.float()
    ar = torch.arange(12, device=dev)
    check, t0 = [], time.time()
    for c0 in range(0, len(ids), CHUNK):
        path = OUT / f"cp_{c0 // CHUNK:02d}.npz"
        if path.exists():
            print("skip", path.name, flush=True); continue
        part = ids[c0:c0 + CHUNK]; n = len(part)
        dl = torch.utils.data.DataLoader(r.Studies(part, meta, {}, K, False), batch_size=1,
                                         num_workers=r.CFG["workers"], pin_memory=True)
        logit = np.zeros((n, 12), np.float32); clog = np.zeros((n, 12), np.float32)
        gf = np.zeros((n, 12, d), np.float16); cf = np.zeros((n, 12, d), np.float16)
        cen = np.zeros((n, 12, TOP, 3), np.int16); atop = np.zeros((n, 12, TOP), np.float16)
        for i, (x, sag, _, _, ctx) in enumerate(dl):
            xp = r.prepare(x, sag, False, r.CFG["eval_res"], dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                tok = bb.forward_features(xp)                                     # [K, 1024, d]
                h = m.norm(bb.forward_head(tok)).float()                          # [K, d]
                a = torch.softmax(m.att(h).float(), dim=0)                        # [K, 12]
            logit[i] = ((a * (h @ W.T)).sum(0) + b).cpu().numpy()
            gf[i] = (a.T @ h).cpu().numpy()
            if c0 == 0 and i < CHECK:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    ref = m(xp, 1, ctx.to(dev)).float()[0].cpu().numpy()
                check.append(float(np.abs(ref - logit[i]).max()))
            top = a.topk(TOP, dim=0).indices                                      # [TOP, 12]
            U = torch.unique(top); row = {int(k): j for j, k in enumerate(U.tolist())}
            tU = tok[U].float().requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hU = m.norm(bb.forward_head(tU)).float()
            contrib = a[U] * (hU @ W.T)                                           # [u, 12], a fixed
            crops = []
            for f in range(12):
                gr, = torch.autograd.grad(contrib[:, f].sum(), tU, retain_graph=f < 11)
                ks = top[:, f].tolist(); rows = [row[k] for k in ks]
                mp = (gr[rows] * tU.detach()[rows]).sum(-1).clamp_min(0).view(TOP, G, G)
                tot = mp.sum((1, 2)).clamp_min(1e-12); g_ = torch.arange(G, device=dev, dtype=mp.dtype)
                cy = (mp.sum(2) * g_).sum(1) / tot; cx = (mp.sum(1) * g_).sum(1) / tot
                cy = torch.where(mp.sum((1, 2)) > 0, cy, torch.full_like(cy, (G - 1) / 2))
                cx = torch.where(mp.sum((1, 2)) > 0, cx, torch.full_like(cx, (G - 1) / 2))
                for j, k in enumerate(ks):
                    r0 = int(np.clip(round((float(cy[j]) + 0.5) * PX - CROP / 2), 0, IMG - CROP))
                    q0 = int(np.clip(round((float(cx[j]) + 0.5) * PX - CROP / 2), 0, IMG - CROP))
                    cen[i, f, j] = (k, r0 + CROP // 2, q0 + CROP // 2)
                    crops.append(x[0, k, :, r0:r0 + CROP, q0:q0 + CROP])
            del tU, hU, contrib
            xc = torch.stack(crops).unsqueeze(0)                                  # [1, 12 * TOP, 3, CROP, CROP]
            xc = r.prepare(xc, torch.zeros(xc.shape[:2], dtype=torch.bool), False, r.CFG["eval_res"], dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hc = m.norm(bb(xc)).float().view(12, TOP, d)                      # finding-major, as appended
            wt = a[top, ar]                                                       # [TOP, 12]
            wt = (wt / wt.sum(0, keepdim=True)).T                                 # [12, TOP]
            c = (wt[:, :, None] * hc).sum(1)                                      # [12, d]
            cf[i] = c.cpu().numpy(); clog[i] = ((c * W).sum(1) + b).cpu().numpy(); atop[i] = wt.cpu().numpy()
            if i % 100 == 0:
                print(f"CP chunk {c0 // CHUNK} {i}/{n} {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(path, ids=np.array(part), logits=logit, crop_logits=clog, g=gf, c=cf, centres=cen,
                            a_top=atop)
        print("saved", path.name, f"{time.time() - t0:.0f}s", flush=True)
    log = {"studies": len(ids), "top": TOP, "crop_px": CROP, "crop_mm": round(CROP * 140.0 / IMG, 1),
           "seconds": round(time.time() - t0), "max_abs_logit_pooled_vs_forward_first20": max(check) if check else None}
    (OUT / "cp_log.json").write_text(json.dumps(log, indent=1) + "\n")
    print("CP_DONE", json.dumps(log), flush=True)


if __name__ == "__main__":
    main()

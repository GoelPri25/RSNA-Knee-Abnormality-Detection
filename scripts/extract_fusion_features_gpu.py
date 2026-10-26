"""Exp F0 (fusion diagnostic): per-finding pooled features of 1e (ResNet-50, d96) or B2 (Swin3D-T, d96n) on fold 0.

Inference only, no TTA, the primary protocol of each run (1e: SWA 21-25, 320 centre crop, 94 windows; B2: SWA 11-15,
320). For every fold-0 study and finding k the script saves z_k = the finding's attention-pooled feature (the input of
the final per-finding linear layer: 2048-d for 1e, 768-d for B2) and the logit. The logits must reproduce the run's
val_swa.json (checked on CPU by scripts/fusion_diagnostic.py). One mode per process, because each runner reads its
configuration from the environment at import.

Env: FUSION_MODE=r50 (D96_PHASE=r50_pertarget, D96_CACHE=<vol>/d96, R50_WEIGHTS) or
     FUSION_MODE=swin (SWIN_PHASE=swin3d_b2, D96_CACHE=<vol>/d96n, SWIN_B2_WEIGHTS); FUSION_OUT.
"""
import os
from pathlib import Path

import numpy as np
import torch

MODE = os.environ["FUSION_MODE"]
OUT = Path(os.environ["FUSION_OUT"]); OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda")


def save(ids, Z, L):
    np.savez(OUT / f"{MODE}_fold0.npz", ids=np.array(ids), z=np.stack(Z).astype(np.float32),
             logit=np.stack(L).astype(np.float32))
    print(f"{MODE}: saved {len(ids)} studies, z {np.stack(Z).shape}", flush=True)


@torch.no_grad()
def run_r50():
    import coatnet_d96_runner as r
    assert os.environ["D96_PHASE"] == "r50_pertarget" and r.LABELS_FILE == "soft_labels_final_per_target_any_model.json"
    labels = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
    fold = {row["study_id"]: row["fold"] for row in __import__("csv").DictReader((r.ASSETS / "folds.csv").open())}
    val_ids = sorted(s for s in labels if fold[s] == "0")
    assert len(val_ids) == 825
    meta = r.read_manifest()
    m = r.build(r.CFG["eval_res"], pretrained=False)
    assert type(m).__name__ == "Raptor", type(m)
    m.load_state_dict(torch.load(os.environ["R50_WEIGHTS"], map_location="cpu", weights_only=True), strict=True)
    m = m.to(dev).eval().to(memory_format=torch.channels_last)
    dl = torch.utils.data.DataLoader(r.Studies(val_ids, meta, {}, r.CFG["eval_windows"], False), batch_size=1,
                                     num_workers=r.CFG["workers"], pin_memory=True)
    Z, L = [], []
    for x, sag, _, _, ctx in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):          # Raptor.forward, split at the pooled feature
            xx = r.prepare(x, sag, False, r.CFG["eval_res"], dev)
            f = m.backbone(xx).view(1, xx.shape[0], -1)
            h = m.context(m.norm(f), ctx.to(dev))
            a = torch.softmax(m.att(h).float(), dim=1).to(h.dtype)
            z = torch.einsum("bkn,bkf->bnf", a, h)                     # [1, 12, 2048]
            logit = (z * m.clsW).sum(-1) + m.clsb
        Z.append(z[0].float().cpu().numpy()); L.append(logit[0].float().cpu().numpy())
    save(val_ids, Z, L)


@torch.no_grad()
def run_swin():
    import swin3d_d96n_runner as s
    assert s.B2 and s.CFG["pool"] == "tokens"
    if s.CFG["attn"] == "sdpa":
        s.use_fused_attention()
    labels, ref, meta, train_ids, val_ids, gold_ids = s.setup()
    prep = s.Prep(*s.kinetics_norm(), dev)
    m = s.SwinB(pretrained=False)
    m.load_state_dict(torch.load(os.environ["SWIN_B2_WEIGHTS"], map_location="cpu", weights_only=True), strict=True)
    m = m.to(dev).eval()
    Z, L = [], []
    for vol, slot, _, _ in s.loader(s.Volumes(val_ids, meta, {}), False):
        with torch.autocast("cuda", dtype=s.AMP):                      # SwinB.forward, split at the pooled feature
            series = prep(vol, slot, False, s.CFG["res"])
            if not series:
                z = torch.zeros(12, m.clsW.shape[1], device=dev); logit = m.clsb
            else:
                toks = [m.encode(v, j) for j, v in series]
                ids = torch.cat([torch.full((len(t),), j, device=t.device) for t, (j, _) in zip(toks, series)])
                h = m.norm(torch.cat(toks)) + m.series(ids)
                a = torch.softmax(m.att(h).float(), dim=0).to(h.dtype)
                z = torch.einsum("nk,nf->kf", a, h)                        # [12, 768]
                logit = (z * m.clsW).sum(-1) + m.clsb
        Z.append(z.float().cpu().numpy()); L.append(logit.float().cpu().numpy())
    save(val_ids, Z, L)


run_r50() if MODE == "r50" else run_swin()

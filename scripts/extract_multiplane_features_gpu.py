"""Exp MP-0 (multi-plane probe): does a zoomed view of the same anatomy point in the sagittal, coronal and axial series
add signal to the label-v2 r50's whole-study view? Inference only (Modal H100); the probe is scripts/audit_multiplane_probe.py.

Model: label-v2 r50 SWA (ref_labelv2/r50_labelv2_swa21_25.pt, phase r50_pertarget), frozen. Points:
run_sig_lv2/crop_probe/multiplane_points.npz (scripts/multiplane_points.py: MS1's label-free top-1 point per finding,
mapped through DICOM geometry into each plane). All 4,349 labelled studies, the 10 cropped findings of r50-GL, 3 planes:
  zoom   the 3-slice window [s-1, s, s+1] around the mapped slice (kept inside its series), a BOX[f] px square at the
         mapped point (clamped inside the image) resized to 384 (r50-GL's eval crop path) -> pooled layer-4 feature;
  slice  the same window, the 320 centre crop as the global head sees it (control: right slice without zoom).
Per view: the pooled 2048-d feature (float16) and a zero-shot logit LN(h) . W_f + b_f with the parent's own classifier.
Saved in chunks of CHUNK studies (a rerun skips finished chunks). First CHECK fold-0 studies: the parent's forward vs
kaggle_fold0_labelv2/val_swa.json (reproduction of the A score).

Env: D96_CACHE, D96_PHASE=r50_pertarget, LV2_WEIGHTS, MP_POINTS, MP_REF (the parent's val_swa.json), MP_OUT,
MP_N (CPU test only).
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import coatnet_d96_runner as r

assert os.environ["D96_PHASE"] == "r50_pertarget"
OUT = Path(os.environ["MP_OUT"]); OUT.mkdir(parents=True, exist_ok=True)
CFG = r.CFG; IMG, MAXS, K = CFG["img"], CFG["maxs"], CFG["eval_windows"]
BOX = {"ACL": 192, "MCL": 192, "Medial Meniscus": 128, "Lateral Meniscus": 192, "Medial OA": 192, "Lateral OA": 192,
       "PF OA": 192, "Baker's": 192, "Contusion": 128, "Fracture": 192}                   # r50-GL's boxes
CF = np.array([k for k, t in enumerate(r.TARGETS) if t in BOX]); BOXN = np.array([BOX[r.TARGETS[k]] for k in CF])
CRES, CHUNK, CHECK = 384, 500, 20
bf16 = lambda dev: torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")


class Views(torch.utils.data.Dataset):
    """per study: windows [v, 3, IMG, IMG] uint8 for the (finding, plane) views with a mapped slice, their index
    (finding j, plane q), centre (row, col) px clamped so the box stays inside the image"""

    def __init__(self, ids, meta, pts):
        self.ids, self.meta, self.pts, self._mm = ids, meta, pts, {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        m = self.meta[self.ids[i]]
        if m["file"] not in self._mm:
            self._mm[m["file"]] = np.load(m["file"], mmap_mode="r")
        vol = self._mm[m["file"]][m["row"]:m["row"] + MAXS]
        p = self.pts[i][CF]                                                          # [10, 3, 8]
        j, q = np.nonzero(p[..., 0] >= 0)
        s = np.clip(p[j, q, 0], p[j, q, 6] + 1, p[j, q, 7] - 1).astype(int)          # window inside its series
        h = BOXN[j] // 2
        cen = np.stack([np.clip(np.round(p[j, q, 1]), h, IMG - h), np.clip(np.round(p[j, q, 2]), h, IMG - h)], 1)
        win = np.ascontiguousarray(vol[np.stack([s - 1, s, s + 1], 1)])
        return torch.from_numpy(win), torch.from_numpy(np.stack([j, q], 1)), torch.from_numpy(cen.astype(np.float32))


def views(x, jq, cen, dev):
    """uint8 windows [v, 3, IMG, IMG] -> zoomed crops [v, 3, CRES, CRES] and 320 centre crops, normalised"""
    x = x.to(dev).float().div_(255.0); v = x.shape[0]
    s = torch.as_tensor(BOXN, dtype=torch.float32, device=dev)[jq[:, 0].to(dev)].view(-1, 1, 1) / IMG
    cen = cen.to(dev)
    px = torch.stack([(cen[:, 1] + 0.5) / IMG * 2 - 1, (cen[:, 0] + 0.5) / IMG * 2 - 1], 1)  # (x = col, y = row)
    th = torch.cat([s * torch.eye(2, device=dev).expand(v, 2, 2), px[..., None]], 2)
    z = F.grid_sample(x, F.affine_grid(th, (v, 3, CRES, CRES), align_corners=False), mode="bilinear",
                      padding_mode="zeros", align_corners=False)
    lo = (IMG - CFG["eval_res"]) // 2
    g = x[..., lo:lo + CFG["eval_res"], lo:lo + CFG["eval_res"]]
    mean, std = r.MEAN.to(dev), r.STD.to(dev)
    norm = lambda t: ((t - mean) / std).contiguous(memory_format=torch.channels_last)
    return norm(z), norm(g)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = r.Raptor(CFG["train_res"], pretrained=False)
    m.load_state_dict(torch.load(os.environ["LV2_WEIGHTS"], map_location="cpu", weights_only=True), strict=True)
    m = m.to(dev).eval().to(memory_format=torch.channels_last)
    for p in m.parameters():
        p.requires_grad_(False)
    z = np.load(os.environ["MP_POINTS"])
    ids, pts = [str(s) for s in z["ids"]], z["points"]
    if os.environ.get("MP_N"):
        ids, pts = ids[:int(os.environ["MP_N"])], pts[:int(os.environ["MP_N"])]
    meta = r.read_manifest(); assert all(s in meta for s in ids)
    W, b = m.clsW.float()[CF], m.clsb.float()[CF]
    ref = {p["study_id"]: np.array(p["probabilities"]) for p in json.loads(Path(os.environ["MP_REF"]).read_text())["predictions"]}
    check, t0 = [], time.time()
    for c0 in range(0, len(ids), CHUNK):
        path = OUT / f"mp_{c0 // CHUNK:02d}.npz"
        if path.exists():
            print("skip", path.name, flush=True); continue
        part = ids[c0:c0 + CHUNK]; n = len(part)
        dl = torch.utils.data.DataLoader(Views(part, meta, pts[c0:c0 + CHUNK]), batch_size=1, collate_fn=lambda b: b[0],
                                         num_workers=CFG["workers"] if dev.type == "cuda" else 0, pin_memory=True)
        fz = np.full((n, len(CF), 3, 2048), np.nan, np.float16); fs = fz.copy()
        lz = np.full((n, len(CF), 3), np.nan, np.float32); ls = lz.copy()
        for i, (x, jq, cen) in enumerate(dl):
            xz, xs = views(x, jq, cen, dev)
            with torch.no_grad(), bf16(dev):
                hz, hs = m.backbone(xz).float(), m.backbone(xs).float()                    # pooled [v, 2048]
                j = jq[:, 0].to(dev)
                gz = ((m.norm(hz).float() * W[j]).sum(1) + b[j]); gs = ((m.norm(hs).float() * W[j]).sum(1) + b[j])
            jj, qq = jq[:, 0].numpy(), jq[:, 1].numpy()
            fz[i, jj, qq], fs[i, jj, qq] = hz.cpu().numpy(), hs.cpu().numpy()
            lz[i, jj, qq], ls[i, jj, qq] = gz.cpu().numpy(), gs.cpu().numpy()
            if len(check) < CHECK and part[i] in ref:                                       # parent reproduction
                xw, sag, _, _, ctx = r.Studies([part[i]], meta, {}, K, False)[0]
                xp = r.prepare(xw[None], sag[None], False, CFG["eval_res"], dev)
                with torch.no_grad(), bf16(dev):
                    pr = torch.sigmoid(m(xp.contiguous(memory_format=torch.channels_last), 1, ctx[None].to(dev)).float())[0]
                check.append(float(np.abs(pr.cpu().numpy() - ref[part[i]]).max()))
            if i % 100 == 0:
                print(f"MP chunk {c0 // CHUNK} {i}/{n} views {len(jq)} {time.time() - t0:.0f}s", flush=True)
        np.savez(path, ids=np.array(part), feat_zoom=fz, feat_slice=fs, logit_zoom=lz, logit_slice=ls)
        print("saved", path.name, f"{time.time() - t0:.0f}s", flush=True)
    log = {"studies": len(ids), "findings": [r.TARGETS[k] for k in CF], "boxes": BOXN.tolist(), "crop_res": CRES,
           "seconds": round(time.time() - t0), "parent_max_abs_dp_vs_val_swa": max(check) if check else None,
           "checked": len(check)}
    (OUT / "mp_log.json").write_text(json.dumps(log, indent=1) + "\n")
    print("MP_DONE", json.dumps(log), flush=True)


if __name__ == "__main__":
    main()

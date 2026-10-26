"""ConvNeXt-S r50_gl4 l4copy SWA checkpoint (R7-D V4_BB=r7cnx; teammate E4-CNXS global-only) -> slim fp16 global-head weights for the fast kernel: backbone.*
(timm convnext_small, fp16), global head (norm, att.0, att.3, clsW, clsb) fp32; crop branch dropped. Checks the slim
path (fp16 backbone, forward_head = pool + head.norm, fp32 head) against the checkpoint's fp32 path on a random study.
RUN-C (2026-10-22): Hydra member heads (hW, hb) and the feature-KD projector are dropped (main head); with a third
argument "fold" the main head becomes the mean of main + member heads (= the runner's hydra_infer read, same weights shape).
Usage: python make_r7d_slim.py weights_swa6_8.pt out_global_fp16.pt [fold]
"""
import hashlib
import sys

import timm
import torch
import torch.nn.functional as F

src, dst = sys.argv[1], sys.argv[2]
sd = torch.load(src, map_location="cpu", weights_only=True)
HEAD = ("norm.weight", "norm.bias", "att.0.weight", "att.0.bias", "att.3.weight", "att.3.bias", "clsW", "clsb")
out = {k: v.half() for k, v in sd.items() if k.startswith("backbone.")}
out.update({k: sd[k].float() for k in HEAD})
if len(sys.argv) > 3 and sys.argv[3] == "fold":
    assert "hW" in sd, "fold: no Hydra heads in this checkpoint"
    out["clsW"] = torch.cat([sd["clsW"].float()[None], sd["hW"].float()]).mean(0)
    out["clsb"] = torch.cat([sd["clsb"].float()[None], sd["hb"].float()]).mean(0)
    print(f"fold: main + {sd['hW'].shape[0]} member heads averaged")
    sd = {**sd, "clsW": out["clsW"], "clsb": out["clsb"]}             # the fp32 check below: the folded head
torch.save(out, dst)


def run(bb, W, x, dt):
    with torch.no_grad():
        h = bb(x.to(dt)).float()                                    # [k, C] = pool + head.norm (num_classes=0)
        h = F.layer_norm(h, (h.shape[1],), W["norm.weight"].float(), W["norm.bias"].float())
        a = torch.softmax(F.linear(torch.tanh(F.linear(h, W["att.0.weight"].float(), W["att.0.bias"].float())),
                                   W["att.3.weight"].float(), W["att.3.bias"].float()), 0)
        return torch.sigmoid(((a.T @ h) * W["clsW"].float()).sum(-1) + W["clsb"].float())


torch.manual_seed(0)
x = torch.randn(6, 3, 320, 320)
ref = timm.create_model("convnext_small", pretrained=False, num_classes=0)
ref.load_state_dict({k[9:]: v.float() for k, v in sd.items() if k.startswith("backbone.")}, strict=True)
p32 = run(ref.eval(), sd, x, torch.float32)
sl = torch.load(dst, map_location="cpu", weights_only=True)
m = timm.create_model("convnext_small", pretrained=False, num_classes=0)
m.load_state_dict({k[9:]: v.float() for k, v in sl.items() if k.startswith("backbone.")}, strict=True)
p16w = run(m.eval(), sl, x, torch.float32)                          # fp16-rounded weights, fp32 compute (CPU)
print(f"slim: {len(sl)} tensors, {sum(v.numel() * v.element_size() for v in sl.values()) / 1e6:.1f} MB; "
      f"max |p(fp32 ckpt) - p(fp16 weights)| {float((p32 - p16w).abs().max()):.5f}")
print("sha256", hashlib.sha256(open(dst, "rb").read()).hexdigest())

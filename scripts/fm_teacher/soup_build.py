"""SOUP-R7 init (2026-10-22): uniform weight average of heliosli's R7 encoders (E_lrprior_s0_f0..f3, enc.* keys) ->
timm convnext_small state dict (head.norm identity, as r7_convert.py). Feature test (soup_test.py): soup-to-member CKA
0.88-0.90 vs member-member 0.77-0.81, feature norm -10%: one basin.
Usage: python soup_build.py <out.pt> <ckpt> [<ckpt> ...]"""
import hashlib
import sys

import timm
import torch

out, srcs = sys.argv[1], sys.argv[2:]
encs = []
for p in srcs:
    sd = torch.load(p, map_location="cpu", weights_only=True)
    encs.append({k[4:]: v.float() for k, v in sd.items() if k.startswith("enc.")})
keys = list(encs[0]); assert all(list(e) == keys for e in encs)
soup = {k: torch.stack([e[k] for e in encs]).mean(0).contiguous() for k in keys}
m = timm.create_model("convnext_small", pretrained=False, num_classes=0)
soup["head.norm.weight"] = torch.ones_like(m.state_dict()["head.norm.weight"])
soup["head.norm.bias"] = torch.zeros_like(m.state_dict()["head.norm.bias"])
m.load_state_dict(soup, strict=True); torch.save(soup, out)
print(out, len(srcs), "members sha256", hashlib.sha256(open(out, "rb").read()).hexdigest())

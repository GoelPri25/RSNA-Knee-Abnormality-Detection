"""Full r50-GL4 l4copy checkpoint -> slim fp16 global-head weights for the FAST-G kernel.
Same conversion as kaggle/submission_kernels/e4all_slim/convert.py: BatchNorm folded into each conv (fp16), global head
only (norm, att.0, att.3, clsW, clsb in fp32), crop branch dropped. Also checks the folding against the unfolded
backbone (torchvision ResNet-50, eval mode) on a random input.
Usage: python make_global_slim.py weights_swa21_25.pt out_global_fp16.pt
"""
import hashlib
import sys

import torch
import torch.nn.functional as F
import torchvision

LAYERS = (3, 4, 6, 3)
src, dst = sys.argv[1], sys.argv[2]
sd = torch.load(src, map_location="cpu", weights_only=True)
out = {}


def fold(conv, bn, name):
    w = sd[f"backbone.{conv}.weight"].float()
    g, b = sd[f"backbone.{bn}.weight"].float(), sd[f"backbone.{bn}.bias"].float()
    m, v = sd[f"backbone.{bn}.running_mean"].float(), sd[f"backbone.{bn}.running_var"].float()
    s = g / torch.sqrt(v + 1e-5)
    out[name + ".w"] = (w * s.view(-1, 1, 1, 1)).half()
    out[name + ".b"] = (b - m * s).half()


fold("conv1", "bn1", "stem")
for li, n in enumerate(LAYERS, 1):
    for bi in range(n):
        p = f"layer{li}.{bi}"
        for c in (1, 2, 3):
            fold(f"{p}.conv{c}", f"{p}.bn{c}", f"{p}.c{c}")
        if f"backbone.{p}.downsample.0.weight" in sd:
            fold(f"{p}.downsample.0", f"{p}.downsample.1", f"{p}.ds")
for k in ("norm.weight", "norm.bias", "att.0.weight", "att.0.bias", "att.3.weight", "att.3.bias", "clsW", "clsb"):
    out[k] = sd[k].float()
torch.save(out, dst)

net = torchvision.models.resnet50(); net.fc = torch.nn.Identity()
bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
net.load_state_dict({k: v for k, v in bb.items() if k in net.state_dict()}, strict=False); net.eval()


def slim_fwd(x):
    W = lambda n: (out[n + ".w"].float(), out[n + ".b"].float())
    w, b = W("stem"); x = F.max_pool2d(F.relu(F.conv2d(x, w, b, stride=2, padding=3)), 3, 2, 1)
    for li, n in enumerate(LAYERS, 1):
        for bi in range(n):
            p = f"layer{li}.{bi}"; s = 2 if (bi == 0 and li > 1) else 1
            (w1, b1), (w2, b2), (w3, b3) = W(p + ".c1"), W(p + ".c2"), W(p + ".c3")
            y = F.conv2d(F.relu(F.conv2d(F.relu(F.conv2d(x, w1, b1)), w2, b2, stride=s, padding=1)), w3, b3)
            if p + ".ds.w" in out:
                wd, bd = W(p + ".ds"); x = F.conv2d(x, wd, bd, stride=s)
            x = F.relu(x + y)
    return x.mean((2, 3))


torch.manual_seed(0); x = torch.randn(2, 3, 320, 320)
with torch.no_grad():
    a, c = net(x), slim_fwd(x)
print(f"{len(out)} tensors -> {dst}; sha256 {hashlib.sha256(open(dst, 'rb').read()).hexdigest()}; "
      f"folded-vs-unfolded feature rel diff {float((a - c).abs().max() / a.abs().max()):.2e}")

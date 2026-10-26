"""t_patch 1 inflation check (CPU, random-init stand-in for the Kinetics weights): on a volume whose slices are
duplicated in pairs, the t_patch=1 embedding at even depths equals the t_patch=2 embedding."""
import os, sys, types, json
fake = types.ModuleType("coatnet_d96_runner")
fake.NATIVE_LAYOUT = True; fake.CFG = {"sha256": {"labels": "d40803d0x"}}; fake.TARGETS = [f"t{i}" for i in range(12)]
sys.modules["coatnet_d96_runner"] = fake; sys.path.insert(0, sys.argv[1])
os.environ["SWIN_PHASE"] = "swin3d_b2"
import torch, swin3d_d96n_runner as m
torch.manual_seed(0)
m2 = m.SwinB(pretrained=False).eval()
m.CFG["t_patch"] = 1
torch.manual_seed(0)
m1 = m.SwinB(pretrained=False).eval()
x = torch.randn(1, 3, 6, 64, 64)
xd = x.repeat_interleave(2, dim=2)                       # each slice twice
e2 = m2.patch_embed(xd); e1 = m1.patch_embed(xd)
print("shapes", tuple(e2.shape), tuple(e1.shape))
print("max |e1[even] - e2|", (e1[:, ::2] - e2).abs().max().item())
with torch.no_grad():
    v = torch.randn(1, 3, 20, 320, 320)
    print("tokens t_patch 2:", tuple(m2.encode(v, 0).shape), " t_patch 1:", tuple(m1.encode(v, 0).shape))

"""KF-SOUP (2026-10-22): uniform weight average of finished students (same runner, same backbone). Keys present in
every member are averaged (float tensors; integer buffers such as BN num_batches_tracked taken from the first member);
keys missing from any member are dropped (e.g. KF-0's file is global-only, so a soup with it has no crop branch).
Usage: python kf_soup.py <out.pt> <member.pt> [<member.pt> ...]"""
import hashlib
import sys

import torch

out, srcs = sys.argv[1], sys.argv[2:]
sds = [torch.load(p, map_location="cpu", weights_only=True) for p in srcs]
keys = [k for k in sds[0] if all(k in s for s in sds)]
assert all(sds[0][k].shape == s[k].shape for s in sds for k in keys)
soup = {k: (torch.stack([s[k].float() for s in sds]).mean(0).to(sds[0][k].dtype) if sds[0][k].is_floating_point()
            else sds[0][k]).contiguous() for k in keys}
torch.save(soup, out)
print(out, len(srcs), "members,", len(keys), "keys kept,", sum(len(s) - len(keys) for s in sds), "dropped; sha256",
      hashlib.sha256(open(out, "rb").read()).hexdigest())

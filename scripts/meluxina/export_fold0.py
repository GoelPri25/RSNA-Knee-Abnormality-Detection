"""Copy the fold-0 studies (ids file) of the d96 cache into one new shard: <out>/shard_0/{vol_000.npy, manifest.jsonl}.
Standard library only (MeluXina login node: Python 3.6, no numpy): .npy = magic + header dict + C-order bytes.
python3 export_fold0.py <d96> <ids> <out>"""
import ast
import json
import struct
import sys
from pathlib import Path


def header(f):
    magic = f.read(8); assert magic[:6] == b"\x93NUMPY", magic
    n = struct.unpack("<H", f.read(2))[0] if magic[6] == 1 else struct.unpack("<I", f.read(4))[0]
    h = ast.literal_eval(f.read(n).decode("latin1")); assert not h["fortran_order"]
    return h, f.tell()


src, ids, out = Path(sys.argv[1]), set(Path(sys.argv[2]).read_text().split()), Path(sys.argv[3]) / "shard_0"
out.mkdir(parents=True, exist_ok=True)
recs = []
for sh in sorted(src.glob("shard_*")):
    for line in (sh / "manifest.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["study_id"] in ids:
            recs.append((sh, r))
assert len(recs) == len(ids), (len(recs), len(ids))
info = {}
for sh, r in recs:
    p = sh / r["part"]
    if str(p) not in info:
        with open(str(p), "rb") as f:
            info[str(p)] = header(f)
h0 = next(iter(info.values()))[0]
assert all(h[0]["descr"] == h0["descr"] and h[0]["shape"][1:] == h0["shape"][1:] for h in info.values())
row = 1
for d in h0["shape"][1:]:
    row *= d
row *= int(h0["descr"][-1])
shape = (96 * len(recs),) + tuple(h0["shape"][1:])
hd = "{'descr': %r, 'fortran_order': False, 'shape': %r, }" % (h0["descr"], shape)
pad = 64 - (10 + len(hd) + 1) % 64
hb = (hd + " " * pad + "\n").encode("latin1")
lines = []
with open(str(out / "vol_000.npy"), "wb") as o:
    o.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hb)) + hb)
    for j, (sh, r) in enumerate(recs):
        p = str(sh / r["part"])
        with open(p, "rb") as f:
            f.seek(info[p][1] + r["row_start"] * row)
            b = f.read(96 * row); assert len(b) == 96 * row
        o.write(b)
        r = dict(r); r["part"] = "vol_000.npy"; r["row_start"] = 96 * j
        lines.append(json.dumps(r))
(out / "manifest.jsonl").write_text("\n".join(lines) + "\n")
print("exported", len(recs), shape, h0["descr"])

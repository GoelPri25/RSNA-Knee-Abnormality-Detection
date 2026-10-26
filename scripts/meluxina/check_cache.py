"""Login-node cache check (plain python, no numpy): every vol_*.npy is uint8 and exactly header + shape bytes; every
manifest line parses and points at an existing vol file inside the array; summaries parse. Usage: check_cache.py <d96>"""
import ast
import glob
import json
import os
import sys

root = sys.argv[1]; bad = 0; studies = 0; rows_of = {}
for f in sorted(glob.glob(f"{root}/shard_*/vol_*.npy")):
    with open(f, "rb") as fh:
        assert fh.read(6) == b"\x93NUMPY", f
        major = fh.read(2)[0]
        hl = int.from_bytes(fh.read(2 if major == 1 else 4), "little")
        hdr = ast.literal_eval(fh.read(hl).decode("latin1")); off = fh.tell()
    n = 1
    for d in hdr["shape"]:
        n *= d
    rows_of[f] = hdr["shape"][0]
    if os.path.getsize(f) != off + n or hdr["descr"] != "|u1" or hdr["shape"][1:] != (384, 384):
        bad += 1; print("BAD", f, os.path.getsize(f), off + n, hdr)
for f in sorted(glob.glob(f"{root}/shard_*/manifest.jsonl")):
    for line in open(f).read().splitlines():
        r = json.loads(line); studies += 1
        p = os.path.join(os.path.dirname(f), r["part"])
        if p not in rows_of or r["row_start"] + len(r["mask"]) > rows_of[p]:
            bad += 1; print("BAD ROW", f, r["study_id"], r["part"], r["row_start"])
for f in sorted(glob.glob(f"{root}/shard_*/cache_summary.json")):
    json.load(open(f))
print(f"CACHE npy {len(rows_of)} manifest studies {studies} bad {bad}")
sys.exit(1 if bad or studies != 4407 or len(rows_of) != 45 else 0)

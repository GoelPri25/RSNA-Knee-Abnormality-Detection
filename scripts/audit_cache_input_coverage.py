"""What does the 140 mm / 224 px top-k support cache give the model? (no gold; report-rule states only pick examples)

Reads the three local cache shards (manifest + uint8 slice arrays) and reports series-slot composition by plane and
fat-suppression flag, slice-position range, and the fraction of each crop that is real image. Renders montages of
cached slices (slot positions 2, 5, 8, 10 of 12) for studies whose report the v3.1 rule reader marks MCL POS
(coronal fluid-sensitive slots) and Baker's LIKELY/POS (sagittal, coronal, axial fluid-sensitive slots).
Montage rows are listed in montage_index.csv so each row maps to study / series / slot.

Container:
  podman run --rm --network=none -v "$P":/proj:ro,Z -v "$OUT":/out:rw,Z localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_cache_input_coverage.py /proj /out
"""
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
C = R / "repository/data/external/topk-support-cache-shards"
OUT.mkdir(parents=True, exist_ok=True)
rows = []
for s in range(3):
    for line in (C / f"shard-{s}-rehost/manifest.jsonl").open():
        r = json.loads(line); r.pop("files", None); r["shard"] = s; rows.append(r)
M = pd.DataFrame(rows)
M["fs"] = M.fluid_sensitive.astype(str).eq("1")
M["kind"] = M.plane.str[:3] + np.where(M.fs, "-FS", "-nonFS")
has = M.groupby("study_id").kind.agg(set)
pos = np.concatenate(M.positions.map(np.array).to_numpy())
cfg = json.loads((C / "shard-0-rehost/cache_summary.json").read_text())["cache"]
stats = {
    "cache_config": cfg,
    "mm_per_pixel": cfg["field_of_view_mm"] / cfg["image_size"],
    "studies": int(M.study_id.nunique()), "slots": int(len(M)),
    "slots_per_study": {int(k): int(v) for k, v in M.groupby("study_id").size().value_counts().sort_index().items()},
    "slot_kind_share": (M.kind.value_counts() / len(M)).round(4).to_dict(),
    "studies_with_kind": {k: round(float(np.mean([k in v for v in has])), 4) for k in ("Cor-FS", "Sag-FS", "Axi-FS", "Cor-nonFS")},
    "studies_with_any_coronal": round(float(np.mean([any(x.startswith("Cor") for x in v) for v in has])), 4),
    "studies_with_any_axial": round(float(np.mean([any(x.startswith("Axi") for x in v) for v in has])), 4),
    "slice_position_min": float(pos.min()), "slice_position_max": float(pos.max()),
    "slice_position_share_outside_central_band": round(float(np.mean((pos < cfg["central_band"][0] - 0.02) | (pos > cfg["central_band"][1] + 0.02))), 4),
    "native_pixel_spacing_mm_median": float(np.median([s[0] for s in M.spacing_mm])),
    "manifest_sha256": {f"shard-{s}": hashlib.sha256((C / f"shard-{s}-rehost/manifest.jsonl").read_bytes()).hexdigest() for s in range(3)},
}
arr = {}


def get(part, shard):
    k = (shard, part)
    if k not in arr:
        arr[k] = np.load(C / f"shard-{shard}-rehost/{part}", mmap_mode="r")
    return arr[k]


def png(path, img):
    h, w = img.shape
    raw = b"".join(b"\x00" + img[i].tobytes() for i in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


index = []


def montage(sel, name):
    tiles = []
    for row_no, (_, r) in enumerate(sel.iterrows(), 1):
        a = get(r.part, r.shard)
        idx = [r.row_start + i for i in (2, 5, 8, 10)][: r["count"]]
        tiles.append(np.concatenate([np.pad(np.asarray(a[i]), 2, constant_values=255) for i in idx], 1))
        index.append({"montage": name, "row": row_no, "study_id": r.study_id, "series_id": r.series_id, "slot": int(r.slot),
                      "manifest_plane": r.plane, "manifest_fluid_sensitive": bool(r.fs),
                      "positions_shown": [round(r.positions[i], 3) for i in (2, 5, 8, 10)]})
    png(OUT / name, np.concatenate(tiles, 0))


st = pd.read_csv(R / "label_rules/hpo_v3_1/rules/s5_fx_site/rule_states.csv", index_col=0)
rng = np.random.default_rng(1)
sel = M[M.study_id.isin(set(st.index[st.MCL.eq("POS")])) & M.kind.eq("Cor-FS")]
montage(sel.iloc[rng.choice(len(sel), 6, replace=False)], "mcl_pos_coronal_fs.png")
bk = set(st.index[st["Baker's"].isin(["LIKELY", "POS"])])
for k in ("Sag-FS", "Cor-FS", "Axi-FS"):
    s2 = M[M.study_id.isin(bk) & M.kind.eq(k)]
    montage(s2.iloc[rng.choice(len(s2), min(4, len(s2)), replace=False)], f"baker_{k}.png")
pd.DataFrame(index).to_csv(OUT / "montage_index.csv", index=False)

sup = []
for _, r in M.sample(400, random_state=0).iterrows():
    a = get(r.support_part, r.shard)
    sup.append((r.kind, float(np.asarray(a[r.row_start + 5]).astype(bool).mean())))
stats["support_fraction_by_kind_sample400"] = pd.DataFrame(sup).groupby(0)[1].mean().round(4).to_dict()
(OUT / "cache_input_coverage.json").write_text(json.dumps(stats, indent=1) + "\n")
print(json.dumps({k: v for k, v in stats.items() if k != "manifest_sha256"}, indent=1))

"""SURPASS-F: flag fastMRI studies that duplicate a competition study (same images), so they never enter training targets.
Signature per study and slot: the slot's middle picked row, 32x32 area resize, zero-mean unit-norm (empty slot = none).
Score(fm, comp) = mean correlation over the slots both have; a fastMRI study is a duplicate if its max score over all
competition studies > 0.98 (same series, same picks -> ~1.0; different knees of the same protocol stay far below).
Usage: python -I fastmri_dedupe.py <competition d96 root> <n comp shards> <fastMRI d96 root> <n fm shards> <out json>
"""
import json
import sys
from pathlib import Path

import numpy as np

SLOTS = [(0, 26), (26, 22), (48, 18), (66, 12), (78, 18)]
THR = 0.98


def signatures(root, nshard):
    ids, sig, has, mm = [], [], [], {}
    for k in range(nshard):
        d = Path(root) / f"shard_{k}"
        for line in (d / "manifest.jsonl").read_text().splitlines():
            r = json.loads(line)
            f = d / r["part"]
            if f not in mm:
                mm[f] = np.load(f, mmap_mode="r")
            s, h = np.zeros((5, 1024), np.float32), np.zeros(5, bool)
            for j, (a, n) in enumerate(SLOTS):
                if r["slots"][j]["series"] is None:
                    continue
                img = np.asarray(mm[f][r["row_start"] + a + n // 2], np.float32)
                t = img.reshape(32, 12, 32, 12).mean((1, 3)).ravel()
                t = t - t.mean(); nrm = np.linalg.norm(t)
                if nrm > 0:
                    s[j], h[j] = t / nrm, True
            ids.append(r["study_id"]); sig.append(s); has.append(h)
    return ids, np.stack(sig), np.stack(has)


def main():
    cid, cs, ch = signatures(sys.argv[1], int(sys.argv[2]))
    fid, fs, fh = signatures(sys.argv[3], int(sys.argv[4]))
    print(f"competition {len(cid)} studies, fastMRI {len(fid)} studies", flush=True)
    num = np.zeros((len(fid), len(cid)), np.float32); den = np.zeros_like(num)
    for j in range(5):
        both = fh[:, j, None] & ch[None, :, j]
        num += (fs[:, j] @ cs[:, j].T) * both
        den += both
    score = np.where(den > 0, num / np.maximum(den, 1), -1)
    best = score.max(1); arg = score.argmax(1)
    dupes = {fid[i]: {"competition": cid[arg[i]], "score": round(float(best[i]), 4)} for i in np.where(best > THR)[0]}
    q = np.quantile(best, [0.5, 0.9, 0.99, 0.999]).round(4).tolist()
    out = {"threshold": THR, "n_fastmri": len(fid), "n_competition": len(cid), "n_dupes": len(dupes),
           "best_score_quantiles_50_90_99_999": q, "top10": sorted(best.round(4).tolist())[-10:], "dupes": dupes}
    Path(sys.argv[5]).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "dupes"}), flush=True)


if __name__ == "__main__":
    main()

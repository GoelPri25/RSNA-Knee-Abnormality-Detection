"""How much of each cached series does the 20-80% central band (12 slices per slot) leave out? (CPU, manifests only)

Each manifest row stores the normalised positions k/(n-1) of the cached slices within the physically ordered series.
The raw stack size n is recovered as the smallest n for which every position is (near) an integer step. Per plane it
reports raw stack size, whether the builder fell back to the full stack, the share of the stack outside the covered
span, raw slices never cached, and the largest raw-slice gap between cached neighbours. No pixels, labels or gold.

Container:
  podman run --rm --network=none -v "$P":/proj:ro,Z -v "$OUT":/out:rw,Z localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_cache_band_coverage.py /proj /out
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
SHARDS = R / "repository/data/external/topk-support-cache-shards"


def stack_size(pos):
    if len(pos) < 2:
        return np.nan
    for n in range(len(pos), 400):
        k = np.asarray(pos) * (n - 1)
        if np.all(np.abs(k - np.round(k)) < 0.02):
            return n
    return np.nan


rows = []
for m in sorted(SHARDS.glob("*/manifest.jsonl")):
    for line in m.read_text().splitlines():
        r = json.loads(line)
        pos = sorted(r["positions"]); n = stack_size(pos)
        idx = np.round(np.array(pos) * (n - 1)).astype(int) if n == n else None
        rows.append({
            "study_id": r["study_id"], "plane": r["plane"], "cached": r["count"], "raw_slices": n,
            "full_stack_fallback": bool(pos[0] < 0.1 and pos[-1] > 0.9),
            "span_low": pos[0], "span_high": pos[-1],
            "share_outside_span": 1 - (pos[-1] - pos[0]),
            "raw_outside_span": (n - 1 - (idx[-1] - idx[0])) if idx is not None else np.nan,
            "raw_never_cached": n - len(set(idx)) if idx is not None else np.nan,
            "max_raw_gap": int(np.diff(idx).max()) if idx is not None and len(idx) > 1 else np.nan,
        })
df = pd.DataFrame(rows)
df.to_csv(OUT / "series_band_coverage.csv", index=False)

q = lambda s: f"{s.median():.0f} [{s.quantile(.1):.0f}-{s.quantile(.9):.0f}]"
summ = df.groupby("plane").apply(lambda g: pd.Series({
    "series": len(g),
    "raw_slices_median[p10-p90]": q(g.raw_slices),
    "full_stack_fallback_%": 100 * g.full_stack_fallback.mean(),
    "share_outside_span_%_median": 100 * g.loc[~g.full_stack_fallback, "share_outside_span"].median(),
    "raw_outside_span_median[p10-p90]": q(g.raw_outside_span),
    "raw_never_cached_median[p10-p90]": q(g.raw_never_cached),
    "share_never_cached_%_mean": 100 * (g.raw_never_cached / g.raw_slices).mean(),
    "max_raw_gap_median": g.max_raw_gap.median(),
    "unresolved_n": int(g.raw_slices.isna().sum()),
}))
summ.to_csv(OUT / "band_coverage_by_plane.csv")
pd.set_option("display.width", 250)
print(f"series rows {len(df)}, studies {df.study_id.nunique()}")
print(summ.to_string())

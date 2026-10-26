"""CPU-only assembly timing on real baseline caches, with exact tensor parity."""
import gc
import importlib.util
import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/kaggle_reference/hpo_xla_actual/skanee"))
from src.images.dataset import StudyCache, load_manifest, stack_channels


def selected_channels(study, keep):
    n = len(study.slices)
    previous = np.maximum(keep - 1, 0); following = np.minimum(keep + 1, n - 1)
    previous = np.where(study.slot_of_slice[previous] == study.slot_of_slice[keep], previous, keep)
    following = np.where(study.slot_of_slice[following] == study.slot_of_slice[keep], following, keep)
    rows = np.column_stack((previous, keep, following))
    return np.concatenate((study.slices[rows], study.support[rows]), axis=1)


def main():
    directories = [ROOT / f"data/external/topk-support-cache-shards/shard-{i}-rehost" for i in range(3)]
    cache = StudyCache(load_manifest(directories, ["unordered"]))
    studies = [cache.study(s, (np.zeros(12), np.ones(12))) for s in sorted(cache.manifest)[:8]]
    rng = np.random.default_rng(2026)
    selection = [np.sort(rng.choice(len(s.slices), min(24, len(s.slices)), replace=False)) for s in studies]
    def original():
        return np.concatenate([np.concatenate((stack_channels(s.slices, s.slot_of_slice)[k],
            stack_channels(s.support, s.slot_of_slice)[k]), axis=1) for s, k in zip(studies, selection)])
    def optimized(): return np.concatenate([selected_channels(s, k) for s, k in zip(studies, selection)])
    assert np.array_equal(original(), optimized())
    timings = {}
    for name, fn in [("build_all_then_select", original), ("select_then_build", optimized)]:
        fn(); durations = []
        for _ in range(12):
            start = time.perf_counter(); value = fn(); durations.append(time.perf_counter() - start); del value
        gc.collect(); tracemalloc.start(); value = fn(); _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        timings[name] = {"median_seconds": statistics.median(durations), "peak_traced_bytes": peak}
        del value
    uint8_bytes = sum(len(k) for k in selection) * 6 * 224 * 224
    report = {"studies": 8, "tensor_parity": "byte-identical", "timings": timings,
              "assembly_speedup": timings["build_all_then_select"]["median_seconds"] / timings["select_then_build"]["median_seconds"],
              "selected_uint8_bytes": uint8_bytes, "same_input_float32_bytes": 4 * uint8_bytes,
              "note": "Warm-cache NumPy assembly only; excludes encoder and validation. Not an epoch-speedup claim."}
    output = ROOT / "data/local_global/verification/assembly_benchmark.json"; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()

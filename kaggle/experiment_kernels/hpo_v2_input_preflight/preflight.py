#!/usr/bin/env python3
"""CPU-only smoke test for the HPO-v2 training inputs; performs no training."""

from pathlib import Path


ROOT = Path("/kaggle/input")


def resolve(candidates: list[Path], description: str, *, directory: bool) -> Path:
    predicate = Path.is_dir if directory else Path.is_file
    matches = [path for path in candidates if predicate(path)]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{description}: expected one mounted path, found {matches}; "
            f"checked={[str(path) for path in candidates]}"
        )
    return matches[0]


def main() -> None:
    print("CPU-only HPO-v2 label mount preflight; no model or training will run.", flush=True)
    print("Kaggle input roots:", sorted(path.name for path in ROOT.iterdir()), flush=True)
    labels = resolve([
        ROOT / "datasets/tumis001/rsna-knee-hpo-v2-provenance-candidate/soft_labels_hpo_v2_provenance_candidate.json",
        ROOT / "rsna-knee-hpo-v2-provenance-candidate/soft_labels_hpo_v2_provenance_candidate.json",
    ], "HPO-v2 labels", directory=False)
    print(f"HPO_V2_LABELS_OK {labels} bytes={labels.stat().st_size}", flush=True)
    print("PREFLIGHT_PASS: HPO-v2 label dataset mounted and file is readable.", flush=True)


if __name__ == "__main__":
    main()

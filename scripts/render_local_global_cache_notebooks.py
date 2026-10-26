"""Refresh embedded cache modules/config in the four CPU notebooks."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kaggle/experiment_kernels/local_global_native_cache"


def main():
    template = (BASE / "shard_0/run.py").read_text()
    modules = {name: (ROOT / "scripts" / name).read_text() for name in
               ["build_native_triplet_cache.py", "local_global_data.py", "local_global_model.py"]}
    config = json.loads((ROOT / "configs/cnn_local_global_v31_d2.json").read_text())
    for shard in range(4):
        lines = template.splitlines(keepends=True)
        replacements = {"modules": modules, "config": config, "SHARD": shard}
        edits = []
        for node in ast.parse(template).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                key = node.targets[0].id
                if key in replacements: edits.append((node.lineno - 1, node.end_lineno, f"{key} = {replacements[key]!r}\n"))
        if len(edits) != 3: raise ValueError("Notebook template assignments differ")
        for start, stop, new in sorted(edits, reverse=True): lines[start:stop] = [new]
        text = "".join(lines); ast.parse(text)
        directory = BASE / f"shard_{shard}"
        (directory / "run.py").write_text(text)
        path = directory / "cache.ipynb"; notebook = json.loads(path.read_text())
        notebook["cells"][1]["source"] = text.splitlines(keepends=True)
        path.write_text(json.dumps(notebook, indent=1) + "\n")


if __name__ == "__main__": main()

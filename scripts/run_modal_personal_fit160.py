"""Launch the Fit160 cache stage and baseline-comparison run on personal Modal.

The personal token file is read directly; work-account Modal environment values
are stripped before the CLI starts. The app is always detached.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = Path.home() / ".config/rsna/modal-personal.env"
RUN_FILE = ROOT / "scripts/modal_hpo_blend_fit160_trainonly.py"
APP_RUN_NAME = "rsna-fit160-hpo-v1-add281-20261002"


def personal_credentials() -> dict[str, str]:
    if not ENV_FILE.is_file():
        raise FileNotFoundError(f"Personal Modal env file is missing: {ENV_FILE}")
    result: dict[str, str] = {}
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() in {"MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"}:
            parsed = shlex.split(value.strip(), posix=True)
            if parsed:
                result[key.strip()] = parsed[0]
    missing = {"MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"} - result.keys()
    if missing:
        raise RuntimeError(f"Personal Modal env file lacks: {sorted(missing)}")
    return result


def main() -> None:
    if not RUN_FILE.is_file():
        raise FileNotFoundError(RUN_FILE)
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("MODAL_"):
            env.pop(key)
    env.update(personal_credentials())
    args = [
        "modal", "run", "--detach", "--name", APP_RUN_NAME,
        str(RUN_FILE),
    ]
    print("Launching Fit160 cache staging, then baseline training, detached on personal Modal.", flush=True)
    subprocess.run(args, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()

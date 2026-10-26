"""Run the AUC-control experiment with the isolated personal Modal profile."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = Path.home() / ".config/rsna/modal-personal.env"
RUN_FILE = ROOT / "scripts/modal_hpo_blend_add281_auc_control.py"


def read_personal_modal_credentials() -> dict[str, str]:
    if not ENV_FILE.is_file():
        raise FileNotFoundError(f"Personal Modal env file is missing: {ENV_FILE}")
    credentials: dict[str, str] = {}
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() not in {"MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"}:
            continue
        parts = shlex.split(value.strip(), posix=True)
        if parts:
            credentials[key.strip()] = parts[0]
    missing = {"MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"} - credentials.keys()
    if missing:
        raise RuntimeError(f"Personal Modal env file is missing required keys: {sorted(missing)}")
    return credentials


def main() -> None:
    if not RUN_FILE.is_file():
        raise FileNotFoundError(RUN_FILE)
    env = os.environ.copy()
    # Remove any shell/profile-selected Modal account and inject only the
    # personal token pair from the project-specific file.
    for key in tuple(env):
        if key.startswith("MODAL_"):
            env.pop(key)
    env.update(read_personal_modal_credentials())
    print("Using the isolated personal Modal credentials only.", flush=True)
    subprocess.run(["modal", "run", str(RUN_FILE)], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()

"""Launch the ResNet-50 warm-start run with only the personal Modal profile."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = Path.home() / ".config/rsna/modal-personal.env"
RUN_FILE = ROOT / "scripts/modal_hpo_blend_resnet50_warmstart.py"
APP_NAME = "rsna-hpo-blend-resnet50-warmstart-fold0"
ACTIVE_APP_ID = "ap-HLe1NKscQdOMOXj31awFmK"


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


def personal_env() -> dict[str, str]:
    env = os.environ.copy()
    # Ignore host-shell Modal credentials so the work account cannot be selected.
    for key in tuple(env):
        if key.startswith("MODAL_"):
            env.pop(key)
    env.update(read_personal_modal_credentials())
    return env


def main() -> None:
    if not RUN_FILE.is_file():
        raise FileNotFoundError(RUN_FILE)
    env = personal_env()
    print("Using only the personal Modal profile from ~/.config/rsna/modal-personal.env.", flush=True)
    listing = subprocess.run(["modal", "app", "list"], cwd=ROOT, env=env,
                             text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=True).stdout
    print(listing, end="" if listing.endswith("\n") else "\n", flush=True)
    if sys.argv[1:] == ["--list-only"]:
        return
    if sys.argv[1:] == ["--logs"]:
        subprocess.run(["modal", "app", "logs", ACTIVE_APP_ID], cwd=ROOT, env=env, check=True)
        return
    matching = [line for line in listing.splitlines() if APP_NAME in line]
    if any("running" in line.lower() or "deployed" in line.lower() for line in matching):
        raise RuntimeError(f"Refusing to start a duplicate active app: {APP_NAME}")
    subprocess.run(["modal", "run", "--detach", str(RUN_FILE)], cwd=ROOT,
                   env=env, check=True)


if __name__ == "__main__":
    main()

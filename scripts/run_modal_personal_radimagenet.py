"""Launch the RadImageNet ResNet-50 experiment detached on the personal Modal account."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = Path.home() / ".config/rsna/modal-personal.env"
RUN_FILE = ROOT / "scripts/modal_hpo_blend_add281_resnet50_radimagenet.py"
APP_NAME = "rsna-hpo-blend-add281-resnet50-radimagenet-fold0"
WEIGHTS = ROOT.parent / "data/pretrained_weights/radimagenet_resnet50.pt"


def personal_env() -> dict[str, str]:
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
        if key.strip() in {"MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"}:
            parts = shlex.split(value.strip(), posix=True)
            if parts:
                credentials[key.strip()] = parts[0]
    missing = {"MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"} - credentials.keys()
    if missing:
        raise RuntimeError(f"Personal Modal env file is missing: {sorted(missing)}")
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("MODAL_"):
            env.pop(key)
    env.update(credentials)
    return env


def main() -> None:
    for path in (RUN_FILE, WEIGHTS):
        if not path.is_file():
            raise FileNotFoundError(path)
    env = personal_env()
    try:
        listing = subprocess.run(
            ["modal", "app", "list"], cwd=ROOT, env=env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=True, timeout=60,
        ).stdout
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Modal app listing timed out; refusing to risk a duplicate launch") from error
    print(listing, end="" if listing.endswith("\n") else "\n", flush=True)
    active = [
        line for line in listing.splitlines()
        if APP_NAME in line and any(
            state in line.lower() for state in ("running", "deployed", "starting")
        )
    ]
    if active:
        raise RuntimeError(f"Refusing to launch duplicate active RadImageNet app: {active[0]}")
    print("Launching the single-T4 RadImageNet run with Modal --detach.", flush=True)
    subprocess.run(["modal", "run", "--detach", str(RUN_FILE)], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()

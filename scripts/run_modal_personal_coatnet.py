#!/usr/bin/env python3
"""Launch one detached CoAtNet experiment using only the personal Modal account."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = Path.home() / ".config/rsna/modal-personal.env"
RUN_FILE = ROOT / "scripts/modal_hpo_blend_add281_coatnet.py"
APP_NAME = "rsna-hpo-blend-add281-coatnet0-fold0"
CACHE_VOLUME = "rsna-cnn-topk-support-cache-v1"


def personal_modal_env() -> dict[str, str]:
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
        parsed = shlex.split(value.strip(), posix=True)
        if parsed:
            credentials[key.strip()] = parsed[0]
    missing = {"MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"} - credentials.keys()
    if missing:
        raise RuntimeError(f"Personal Modal env file lacks: {sorted(missing)}")

    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("MODAL_"):
            env.pop(key)
    env.update(credentials)
    return env


def output(args: list[str], env: dict[str, str]) -> str:
    result = subprocess.run(args, cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            check=True)
    return result.stdout


def main() -> None:
    if not RUN_FILE.is_file():
        raise FileNotFoundError(RUN_FILE)
    if not (Path.home() / ".kaggle/access_token").is_file():
        raise FileNotFoundError("Kaggle access token missing at ~/.kaggle/access_token")
    env = personal_modal_env()
    if len(sys.argv) >= 3 and sys.argv[1] == "--logs":
        tail = sys.argv[3] if len(sys.argv) > 3 else "100"
        subprocess.run(
            ["modal", "app", "logs", sys.argv[2], "--tail", tail],
            cwd=ROOT, env=env, check=True,
        )
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--stop":
        subprocess.run(
            ["modal", "app", "stop", sys.argv[2], "--yes"],
            cwd=ROOT, env=env, check=True,
        )
        return
    apps = output(["modal", "app", "list"], env)
    print("Personal Modal account apps:", flush=True)
    print(apps, end="" if apps.endswith("\n") else "\n", flush=True)
    matching = [line for line in apps.splitlines() if APP_NAME in line]
    if any("running" in line.lower() or "deployed" in line.lower() for line in matching):
        raise RuntimeError(f"Refusing to start a duplicate active app: {APP_NAME}")

    volume = output(["modal", "volume", "ls", CACHE_VOLUME, "/"], env)
    print(f"Existing derived-cache volume {CACHE_VOLUME}:", flush=True)
    print(volume, end="" if volume.endswith("\n") else "\n", flush=True)
    absent = [name for name in ("shard_0", "shard_1", "shard_2") if name not in volume]
    if absent:
        raise RuntimeError(
            f"The existing cache volume is missing {absent}; stopping before GPU allocation."
        )

    if sys.argv[1:] == ["--check-only"]:
        print("Preflight passed; no app or GPU was started.", flush=True)
        return
    args = ["modal", "run", "--detach", "--name", APP_NAME, str(RUN_FILE)]
    print("Launching the single-T4 CoAtNet-0 training job detached on personal Modal.", flush=True)
    subprocess.run(args, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Preflight checks for running the v490 IsaacGym common-dataset runner.

This script intentionally does not create a simulation. It only verifies that the
Python environment and Dynamic_Gym overlay required by
run_isaacgym_front110_common_dataset_v490.py are visible.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


REQUIRED_MODULES = (
    "isaacgym",
    "torch",
    "hydra",
    "omegaconf",
    "isaacgymenvs",
)

REQUIRED_TASK_CONFIGS = (
    "SimToolRealDynamicGraspV33FrankaBrainCoRevo2AffordanceDomino20PointNetPPO.yaml",
)


def check_module(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    version = getattr(mod, "__version__", "")
    suffix = f" {version}" if version else ""
    return True, f"ok{suffix}"


def find_task_config(search_roots: list[Path], filename: str) -> Path | None:
    for root in search_roots:
        if not root.exists():
            continue
        direct = root / "isaacgymenvs" / "cfg" / "task" / filename
        if direct.exists():
            return direct
        for path in root.rglob(filename):
            return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dynamic-gym-root",
        default=os.environ.get("DYNAMIC_GYM_ROOT", ""),
        help="Dynamic_Gym-sim-env-franka-revo2-aerial-v1 root, if available.",
    )
    parser.add_argument(
        "--isaacgymenvs-root",
        default=os.environ.get("ISAACGYM_ENVS_ROOT", ""),
        help="Optional parent directory containing isaacgymenvs/.",
    )
    parser.add_argument("--strict", action="store_true", help="Return nonzero if any check fails.")
    args = parser.parse_args()

    print(f"python: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')}")

    ok = True
    print("\nmodules:")
    for name in REQUIRED_MODULES:
        passed, detail = check_module(name)
        ok = ok and passed
        print(f"  {name}: {detail}")

    search_roots: list[Path] = []
    if args.dynamic_gym_root:
        search_roots.append(Path(args.dynamic_gym_root))
    if args.isaacgymenvs_root:
        search_roots.append(Path(args.isaacgymenvs_root))
    for item in sys.path:
        if item:
            search_roots.append(Path(item))

    print("\ntask configs:")
    for filename in REQUIRED_TASK_CONFIGS:
        found = find_task_config(search_roots, filename)
        if found:
            print(f"  {filename}: {found}")
        else:
            ok = False
            print(f"  {filename}: missing")

    dyn_root = Path(args.dynamic_gym_root) if args.dynamic_gym_root else None
    print("\nDynamic_Gym root:")
    if dyn_root and dyn_root.exists():
        scripts = dyn_root / "scripts"
        print(f"  root: {dyn_root}")
        print(f"  scripts/: {'ok' if scripts.is_dir() else 'missing'}")
        ok = ok and scripts.is_dir()
    else:
        ok = False
        print("  missing or not set")

    if ok:
        print("\npreflight: ok")
        return 0
    print("\npreflight: failed")
    return 2 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())

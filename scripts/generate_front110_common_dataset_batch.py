#!/usr/bin/env python3
"""Batch generator for MuJoCo Front-110 common-schema datasets.

This runs the v731 MuJoCo controller without video, writes one
front110_common_v1 NPZ/manifest pair per episode, then builds batch indexes
for all episodes, full-success episodes, and partial/failure episodes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "render_front110_multi_clean_v731_midleft_redclearance_patch.py"
DEFAULT_OUT_ROOT = Path("/autodl-fs/data/mingyu/mujoco_front110_common_datasets")
FIXED6_ANGLES = [38.5, 56.7, 74.0, 88.0, 116.5, 143.0]


def parse_angles(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def make_angles(mode: str, num_tools: int, seed: int, angles_text: str) -> list[float] | None:
    if angles_text:
        return parse_angles(angles_text)
    if mode == "runner_default":
        return None
    if mode == "fixed6":
        return FIXED6_ANGLES[:num_tools]
    if mode == "continuous":
        rng = random.Random(seed)
        return sorted(round(rng.uniform(35.0, 145.0), 3) for _ in range(num_tools))
    if mode == "grid5":
        grid = list(np.arange(35.0, 145.0 + 1e-6, 5.0))
        if num_tools <= len(grid):
            rng = random.Random(seed)
            return sorted(rng.sample(grid, num_tools))
        return grid
    raise ValueError(f"unknown angle mode: {mode}")


def find_summary(run_dir: Path, seed: int) -> Path:
    candidates = sorted(run_dir.glob(f"*seed{seed}.json"))
    if not candidates:
        candidates = [p for p in sorted(run_dir.glob("*.json")) if "scene" not in p.name]
    if not candidates:
        raise FileNotFoundError(f"no summary json in {run_dir}")
    return candidates[0]


def npz_frames(path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        return int(data["fr3_qpos"].shape[0])


def rel_or_abs(path: str | Path) -> str:
    return str(Path(path))


def build_indexes(batch_dir: Path, items: list[dict]) -> None:
    total_drops = sum(int(x["total"]) for x in items)
    passed = sum(int(x["passed"]) for x in items)
    frames = sum(int(x.get("frames", 0)) for x in items)

    def payload(selected: list[dict]) -> dict:
        selected_total = sum(int(x["total"]) for x in selected)
        selected_passed = sum(int(x["passed"]) for x in selected)
        selected_frames = sum(int(x.get("frames", 0)) for x in selected)
        return {
            "schema_version": "front110_common_v1",
            "simulator": "mujoco",
            "batch_dir": str(batch_dir),
            "episodes": len(selected),
            "tools_per_episode": None,
            "total_drops": selected_total,
            "passed": selected_passed,
            "success_rate": (selected_passed / selected_total) if selected_total else 0.0,
            "total_recorded_frames": selected_frames,
            "items": selected,
        }

    all_payload = payload(items)
    all_payload["total_drops"] = total_drops
    all_payload["passed"] = passed
    all_payload["success_rate"] = (passed / total_drops) if total_drops else 0.0
    all_payload["total_recorded_frames"] = frames
    (batch_dir / "batch_index.json").write_text(json.dumps(all_payload, indent=2), encoding="utf-8")

    success_items = [x for x in items if int(x["passed"]) == int(x["total"])]
    partial_items = [x for x in items if int(x["passed"]) != int(x["total"])]
    (batch_dir / "batch_index_success_episodes.json").write_text(
        json.dumps(payload(success_items), indent=2), encoding="utf-8"
    )
    (batch_dir / "batch_index_partial_episodes.json").write_text(
        json.dumps(payload(partial_items), indent=2), encoding="utf-8"
    )


def run_episode(args: argparse.Namespace, batch_dir: Path, seed: int) -> dict:
    episode_dir = batch_dir / f"episode_seed{seed}"
    dataset_dir = episode_dir / "dataset"
    run_dir = episode_dir / "run"
    episode_dir.mkdir(parents=True, exist_ok=True)

    angles = make_angles(args.angle_mode, args.num_tools, seed, args.angles_deg)
    cmd = [
        sys.executable,
        str(RUNNER),
        "--seed",
        str(seed),
        "--num-tools",
        str(args.num_tools),
        "--dataset-out-dir",
        str(dataset_dir),
        "--dataset-stride",
        str(args.dataset_stride),
        "--out-dir",
        str(run_dir),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--fps",
        "0",
    ]
    if angles is not None:
        cmd += ["--angles-deg", ",".join(f"{a:g}" for a in angles)]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    log_path = episode_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        return {
            "seed": seed,
            "passed": 0,
            "total": args.num_tools,
            "error": f"runner exited {proc.returncode}",
            "log": str(log_path),
        }

    summary_path = find_summary(run_dir, seed)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dataset = summary.get("dataset") or {}
    npz_path = Path(dataset.get("episode_npz", dataset_dir / f"common_episode_seed{seed}.npz"))
    manifest_path = Path(dataset.get("manifest", dataset_dir / f"manifest_seed{seed}.json"))
    frames = npz_frames(npz_path) if npz_path.exists() else 0
    return {
        "seed": seed,
        "passed": int(summary.get("passed", 0)),
        "total": int(summary.get("total", args.num_tools)),
        "angles_deg": summary.get("angles_deg"),
        "release_order_angles_deg": summary.get("release_order_angles_deg"),
        "npz": rel_or_abs(npz_path),
        "manifest": rel_or_abs(manifest_path),
        "frames": frames,
        "summary": rel_or_abs(summary_path),
        "log": rel_or_abs(log_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=74000)
    ap.add_argument("--num-tools", type=int, default=6)
    ap.add_argument("--angle-mode", choices=["fixed6", "continuous", "grid5", "runner_default"], default="fixed6")
    ap.add_argument("--angles-deg", default="")
    ap.add_argument("--dataset-stride", type=int, default=4)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = args.name or f"v731_{args.angle_mode}_ep{args.episodes}_seed{args.seed_start}_{stamp}"
    batch_dir = Path(args.out_root) / name
    batch_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["batch_dir"] = str(batch_dir)
    config["runner"] = str(RUNNER)
    (batch_dir / "batch_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    items: list[dict] = []
    progress_path = batch_dir / "progress.log"
    with progress_path.open("w", encoding="utf-8") as progress:
        for i in range(args.episodes):
            seed = args.seed_start + i
            item = run_episode(args, batch_dir, seed)
            items.append(item)
            build_indexes(batch_dir, items)
            progress.write(
                f"{i + 1}/{args.episodes} seed={seed} passed={item.get('passed')}/{item.get('total')}"
                f" frames={item.get('frames', 0)} error={item.get('error', '')}\n"
            )
            progress.flush()

    build_indexes(batch_dir, items)
    print(json.dumps(json.loads((batch_dir / "batch_index.json").read_text(encoding="utf-8")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

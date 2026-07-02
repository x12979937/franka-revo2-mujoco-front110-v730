# Franka FR3 + Revo2 MuJoCo Front-110 v731

Compact reproduction package for the MuJoCo version of the front 110 degree aerial tool-catching task. v731 is a local robustness patch on top of v730.

## Included

- v731 controller/demo script: `scripts/render_front110_multi_clean_v731_midleft_redclearance_patch.py`
- v730 controller/demo script: `scripts/render_front110_multi_clean_v730_front110_dense_height_patch.py`
- v700 controller dependency: `scripts/render_front110_multi_clean_v700.py`
- frozen stage-4 controller dependency: `frozen/stage4e_front110_v700_grid0p5_patch.py`
- high-fidelity FR3/Revo2 visual/contact assets: `assets/full_robot_urdf_mirror/`
- common planner/action/dataset adapter: `front110_core/`
- IsaacGym v490 runner with common dataset hooks:
  `scripts/run_isaacgym_front110_common_dataset_v490.py`
- validation artifacts:
  - `outputs/front110_v730_dense_1deg_batches_seed72900/dense_1deg_summary.json`
  - `outputs/front110_v730_grid9_clean_video/`

## Task Specification

- Robot: Franka FR3 arm + Revo2 dexterous hand.
- FR3 joint velocity limits: `[2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26]` rad/s.
- Revo2 active joint order: `[thumb_flex, thumb_aux, index, middle, ring, pinky]`.
- Revo2 active joint velocity limits: `[2.53, 2.62, 2.27, 2.27, 2.27, 2.27]` rad/s.
- Tool: 30-35 cm rod-like object, 30 mm diameter, 100-150 g.
- Positive handle region: 10-12 cm, green.
- Negative functional region: 20-25 cm, red.
- Front catching sector: `[35 deg, 145 deg]`.
- Nominal release radius: 0.50 m from robot center.
- Success requires strict physical holding and rejects bad functional-region contacts.

## Validation Status

This is the current best MuJoCo front-110 version from the work session. v731 keeps v730's real FR3/Revo2 assets and staged observe-then-grasp behavior, and adds a narrow height-only clearance patch for the 126-139 degree mid-left band where xy jitter could make the red functional segment touch the hand.

Packaged v731 validation artifacts:

- `outputs/repro_v731_angle130_single/`: previous bad seed `73130`, nominal `130 deg`, observed/control angle `133.926 deg`, now `1/1`, `bad_functional_contact: false`.
- `outputs/repro_v731_125_135_batch/`: `125/130/135 deg`, `3/3`, `bad_functional_contact: false`.
- `outputs/repro_v731_grid_5deg/`: full `[35,145] deg` front sector at 5-degree spacing, `23/23`, `bad_functional_contact: false`.
- `outputs/repro_v731_continuous_uniform12/`: 12 uniformly sampled continuous angles in `[35,145] deg`, random release order, `12/12`, `bad_functional_contact: false`.

The v731 key result is therefore `23/23` on the 5-degree front-sector grid plus `12/12` on an explicit continuous random-angle sample, with no bad functional-region contacts. This is a scripted/oracle controller milestone for data generation, not yet a learned vision policy.

## Setup

Install the minimal Python dependencies:

```bash
pip install mujoco numpy imageio imageio-ffmpeg
```

For headless rendering:

```bash
export MUJOCO_GL=egl
```

## Reproduce the 5-Degree Front-110 Grid Demo

From the repository root:

```bash
export MUJOCO_GL=egl
python scripts/render_front110_multi_clean_v731_midleft_redclearance_patch.py \
  --seed 73105 \
  --num-tools 23 \
  --angles-deg 35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145 \
  --yaws-deg=-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,7,6,5,4,3,2 \
  --fps 8 \
  --width 320 \
  --height 192 \
  --out-dir outputs/repro_v731_grid_5deg
```

Original AutoDL command used in this run:

```bash
export PYTHONPATH=/autodl-fs/data/mingyu/Mujoco/envs/mujoco_v699_py38_pkgs:$PYTHONPATH
export MUJOCO_GL=egl
/autodl-fs/data/mingyu/IsaacGym/envs/isaacgym_py38/bin/python scripts/render_front110_multi_clean_v731_midleft_redclearance_patch.py \
  --seed 73105 \
  --num-tools 23 \
  --angles-deg 35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145 \
  --yaws-deg=-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,7,6,5,4,3,2 \
  --fps 8 \
  --width 320 \
  --height 192 \
  --out-dir outputs/repro_v731_grid_5deg
```

## Reproduce the Continuous Random-Angle Sample

```bash
export MUJOCO_GL=egl
python scripts/render_front110_multi_clean_v731_midleft_redclearance_patch.py \
  --seed 73178 \
  --num-tools 12 \
  --angles-deg 48.719,46.203,125.800,38.557,90.638,120.859,84.260,55.402,104.856,39.573,109.171,58.681 \
  --yaws-deg=8.26,-3.72,3.48,0.58,-3.58,7.75,-9.08,0.43,-1.53,8.52,7.41,7.75 \
  --fps 8 \
  --width 320 \
  --height 192 \
  --out-dir outputs/repro_v731_continuous_uniform12
```

## Common Dataset Adapter

`front110_core/` splits the task interface into simulator-independent data/action definitions and simulator-specific adapters:

- `front110_core.planner.UnifiedAction`: common action emitted by planner/controller code.
- Action space: `fr3_joint_target[7] + revo2_active_target[6]`.
- Revo2 active joint order: `[thumb_flex, thumb_aux, index, middle, ring, pinky]`.
- Common quaternion order: `wxyz`.
- `front110_core.backends.mujoco_backend`: extracts low-dimensional MuJoCo observations and applies the legacy 11-DoF converted Revo2 target.
- `front110_core.backends.isaacgym_backend`: IsaacGym tensor adapter for Dynamic_Gym-style envs, including FR3/Revo2 DOF mapping, `cur_targets` writes, root/body tensor observation export, and `xyzw <-> wxyz` quaternion conversion.
- `front110_core.dataset_adapter`: writes and validates `front110_common_v1` NPZ episodes plus JSON manifests.
- `front110_core.isaacgym_runner_bridge`: small runner hook that records an IsaacGym rollout with the same `UnifiedAction` and `EpisodeRecorder` used by the MuJoCo scripts.

Generate a small aligned dataset sample:

```bash
export MUJOCO_GL=osmesa
python scripts/render_front110_multi_clean_v731_midleft_redclearance_patch.py \
  --seed 73130 \
  --num-tools 1 \
  --angles-deg 130 \
  --yaws-deg 0 \
  --fps 0 \
  --dataset-out-dir outputs/repro_v731_dataset_smoke/dataset \
  --dataset-stride 8 \
  --out-dir outputs/repro_v731_dataset_smoke
```

Expected dataset files:

- `outputs/repro_v731_dataset_smoke/dataset/common_episode_seed73130.npz`
- `outputs/repro_v731_dataset_smoke/dataset/manifest_seed73130.json`

Smoke-test the IsaacGym tensor bridge without launching IsaacGym:

```bash
python scripts/smoke_isaacgym_adapter.py
```

Check whether the current machine has the full IsaacGym/Dynamic_Gym runtime
needed for a real rollout:

```bash
export LD_LIBRARY_PATH=/path/to/isaacgym_py38/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/path/to/Dynamic_Gym/src/Dynamic_Gym-sim-env-franka-revo2-aerial-v1:/path/to/isaacgymenvs_parent:$PYTHONPATH
/path/to/isaacgym_py38/bin/python scripts/check_isaacgym_front110_runtime.py \
  --dynamic-gym-root /path/to/Dynamic_Gym/src/Dynamic_Gym-sim-env-franka-revo2-aerial-v1 \
  --isaacgymenvs-root /path/to/isaacgymenvs_parent \
  --strict
```

Run the v490 IsaacGym runner with common-schema dataset export:

```bash
export PATH=/path/to/isaacgym_py38/bin:$PATH
export LD_LIBRARY_PATH=/path/to/isaacgym_py38/lib:$LD_LIBRARY_PATH
export DYNAMIC_GYM_ROOT=/path/to/Dynamic_Gym/src/Dynamic_Gym-sim-env-franka-revo2-aerial-v1
export ISAACGYM_ASSET_BASE=$DYNAMIC_GYM_ROOT
export PYTHONPATH=$PWD:$DYNAMIC_GYM_ROOT:$DYNAMIC_GYM_ROOT/isaacgymenvs:/path/to/isaacgym_preview4/python:$PYTHONPATH
/path/to/isaacgym_py38/bin/python scripts/run_isaacgym_front110_common_dataset_v490.py \
  --episodes 1 \
  --steps 175 \
  --seed 12001 \
  --out-dir outputs/isaacgym_v490_common_smoke \
  --common-dataset-out-dir dataset \
  --common-dataset-stride 1 \
  --common-dataset-validate
```

The common dataset files are written under `--out-dir/--common-dataset-out-dir`, for example:

- `outputs/isaacgym_v490_common_smoke/dataset/isaacgym_common_episode_ep000.npz`
- `outputs/isaacgym_v490_common_smoke/dataset/manifest_ep000.json`

`PATH` must include the IsaacGym environment `bin/` directory because `gymtorch`
loads a PyTorch C++ extension through `ninja`. `DYNAMIC_GYM_ROOT` points Hydra to
the real Dynamic_Gym configs; `ISAACGYM_ASSET_BASE` points the runner to the
IsaacGym URDF/mesh assets.

Use it inside a Dynamic_Gym runner:

```python
from front110_core.backends.isaacgym_backend import IsaacGymTensorAdapter

adapter = IsaacGymTensorAdapter.from_env(env)
obs = adapter.get_observation(env_id=0)
adapter.apply_action_to_targets(action, submit=False)
```

The adapter layer is intended to make MuJoCo and IsaacGym data/action logs compatible for training. The IsaacGym backend now handles tensor/DOF/observation alignment, and `scripts/run_isaacgym_front110_common_dataset_v490.py` shows the v490 runner loop wired to the common recorder without changing the original controller logic.

The IsaacGym runner is an overlay for the original Dynamic_Gym project. A real
IsaacGym rollout still requires the host machine to provide the Dynamic_Gym task
definitions/configs, `isaacgymenvs`, IsaacGym, `torch`, `hydra`, and `omegaconf`
in the same Python environment. The included smoke test validates the adapter
without IsaacGym; the preflight script above checks the full host runtime before
launching a rollout.

## Notes

- This is a scripted/oracle controller for simulation-data generation and environment alignment, not a learned vision policy.
- The task runner provides the next release event to the controller. Motion is staged so the visible behavior is release-triggered rather than waiting at the final catch pose too early.
- Some packaged demo filenames still contain `v728`; they were generated by the v730 script through the inherited v728 real-visual naming path.

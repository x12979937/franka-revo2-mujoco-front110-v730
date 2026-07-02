#!/usr/bin/env python3
import argparse
import json
import math
import random
import xml.etree.ElementTree as ET
from argparse import Namespace
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import importlib.util

PROJECT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("v700", PROJECT / "frozen/stage4e_front110_v700_grid0p5_patch.py")
v700 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v700)
OUT = PROJECT / 'outputs/front110_v700_multi_clean_demo'


def base_args(angle, out_dir):
    args = Namespace(seed=70000, angle=float(angle), yaw=0.0, seconds=1.1, fps=30, no_video=True, no_write=True, quiet=True,
        release_time=0.35, camera='wide_front', out_dir=str(out_dir),
        radial_offset=-0.205, tangent_offset=-0.095, z_offset=0.245, catch_z=0.875,
        wrist_yaw_factor=1.0, wrist_yaw=0.0, wrist_roll_factor=0.0, wrist_roll=0.0, wrist_pitch_factor=0.0, wrist_pitch=0.0,
        ready_tangent=0.120, move_start=0.120, move_dur=0.130, insert_tangent=0.040,
        latch_drop_rate=0.850, latch_drop_duration=0.180, latch_min_z=0.580,
        latch_lift_rate=0.0, latch_lift_start=0.160, latch_lift_duration=0.180, latch_max_z=0.920,
        latch_radial_offset=0.0, latch_tangent_offset=-0.010, latch_xy_follow=0.0, latch_xy_follow_start=0.0, latch_xy_follow_dur=0.080,
        middle_latch_scale=0.0, middle_latch_start=0.030, middle_latch_dur=0.120,
        index_latch_scale=0.0, index_latch_start=0.030, index_latch_dur=0.120, index_latch_prox=1.05, index_latch_dist=1.05,
        latch_brake_start=0.0, latch_brake_duration=0.160, latch_brake_yaw=0.0, latch_brake_roll=0.0, latch_brake_pitch=0.0,
        thumb_lead=0.160, thumb_dur=0.240, finger_lead=0.110, finger_dur=0.180,
        thumb_close_scale=1.0, index_close_scale=1.0, middle_close_scale=0.0, front110_auto=True)
    v700.apply_front110_auto_args(args)
    return args


def tool_pos(angle, radius=0.50, height=1.15):
    th = math.radians(angle)
    return np.array([radius * math.cos(th), radius * math.sin(th), height], dtype=float)


def add_tool(world, i, angle, yaw):
    pos = tool_pos(angle)
    body = ET.SubElement(world, 'body', {'name': f'tool{i}', 'pos': f'{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}', 'quat': '1 0 0 0'})
    ET.SubElement(body, 'freejoint', {'name': f'tool_free_{i}'})
    ET.SubElement(body, 'inertial', {'pos': '0 0 0.055', 'mass': '0.125', 'diaginertia': '0.00115 0.00115 0.000035'})
    # 30 mm diameter, 11 cm positive handle, 22 cm negative functional region.
    ET.SubElement(body, 'geom', {'name': f'handle_positive_{i}', 'type': 'capsule', 'fromto': '0 0 -0.11 0 0 0.00', 'size': '0.015', 'rgba': '0.05 0.75 0.25 0.60', 'friction': '2.80 0.080 0.008'})
    ET.SubElement(body, 'geom', {'name': f'functional_negative_{i}', 'type': 'capsule', 'fromto': '0 0 0.00 0 0 0.22', 'size': '0.015', 'rgba': '0.70 0.05 0.04 0.60', 'friction': '0.55 0.010 0.001'})
    return pos, v700.yaw_quat(math.radians(yaw))


def build_multi_scene(scene_xml, angles, yaws):
    tmp = scene_xml.with_name('tmp_single_scene.xml')
    v700.build_scene(tmp)
    root = ET.parse(tmp).getroot()
    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')
    global_node = visual.find('global')
    if global_node is None:
        global_node = ET.SubElement(visual, 'global')
    global_node.set('offwidth', '1600')
    global_node.set('offheight', '912')
    world = root.find('worldbody')
    for child in list(world):
        if child.tag == 'body' and child.get('name') == 'tool':
            world.remove(child)
    # Wide cameras that keep the whole arm, front arc, and falling rods in frame.
    ET.SubElement(world, 'camera', {'name': 'wide_front', 'pos': '0 -2.85 1.55', 'xyaxes': '1 0 0 0 0.43 0.90', 'fovy': '62'})
    ET.SubElement(world, 'camera', {'name': 'wide_top_oblique', 'pos': '0.0 -2.05 2.35', 'xyaxes': '1 0 0 0 0.74 0.67', 'fovy': '66'})
    initials = []
    for i, (a, y) in enumerate(zip(angles, yaws)):
        initials.append(add_tool(world, i, a, y))
    ET.ElementTree(root).write(scene_xml, encoding='unicode')
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    return initials


def set_tool_state(model, data, i, pos, quat, vel_zero=True):
    qadr = model.jnt_qposadr[model.joint(f'tool_free_{i}').id]
    data.qpos[qadr:qadr + 3] = pos
    data.qpos[qadr + 3:qadr + 7] = quat
    if vel_zero:
        qvadr = model.jnt_dofadr[model.joint(f'tool_free_{i}').id]
        data.qvel[qvadr:qvadr + 6] = 0.0


def set_tool_visual_collision(model, geoms, mode):
    for g in geoms:
        if mode == 'hidden':
            model.geom_rgba[g][3] = 0.0
            model.geom_contype[g] = 0
            model.geom_conaffinity[g] = 0
        elif mode == 'waiting':
            model.geom_rgba[g][3] = 0.35
            model.geom_contype[g] = 0
            model.geom_conaffinity[g] = 0
        elif mode == 'current_wait':
            model.geom_rgba[g][3] = 1.0
            model.geom_contype[g] = 0
            model.geom_conaffinity[g] = 0
        elif mode == 'active':
            model.geom_rgba[g][3] = 1.0
            model.geom_contype[g] = 1
            model.geom_conaffinity[g] = 1


def contact_metrics_current(model, data, idx):
    thumb = False
    index_middle = False
    bad_functional = False
    handle = f'handle_positive_{idx}'
    functional = f'functional_negative_{idx}'
    for k in range(data.ncon):
        c = data.contact[k]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        b1 = model.body(model.geom_bodyid[c.geom1]).name
        b2 = model.body(model.geom_bodyid[c.geom2]).name
        names = ' '.join(str(x) for x in (g1, b1, g2, b2))
        if handle in names and 'thumb' in names:
            thumb = True
        if handle in names and ('index' in names or 'middle' in names):
            index_middle = True
        if functional in names and ('thumb' in names or 'index' in names or 'middle' in names or 'ring' in names or 'pinky' in names or 'panda' in names):
            bad_functional = True
    return thumb, index_middle, bad_functional


def make_wrist_xmat_fn(data, wrist_bid, args, angle):
    wrist_ready_mat = data.xmat[wrist_bid].copy().reshape(3, 3)
    delta = angle - 90.0
    def fn(latch_age=0.0):
        brake_a = v700.smoothstep((latch_age - args.latch_brake_start) / max(1e-6, args.latch_brake_duration))
        return (
            v700.rotz(math.radians(args.wrist_yaw + args.wrist_yaw_factor * delta + args.latch_brake_yaw * brake_a))
            @ v700.rotx(math.radians(args.wrist_roll + args.wrist_roll_factor * delta + args.latch_brake_roll * brake_a))
            @ v700.roty(math.radians(args.wrist_pitch + args.wrist_pitch_factor * delta + args.latch_brake_pitch * brake_a))
            @ wrist_ready_mat
        ).reshape(9)
    return fn


def run_demo(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    # Continuous random front-arc samples; displayed as a static fan, released in shuffled order.
    angles = sorted(round(rng.uniform(35.0, 145.0), 1) for _ in range(args.num_tools))
    # Ensure the demo visibly covers left, center, and right even for unlucky random seeds.
    if args.num_tools >= 6:
        angles = [38.5, 56.7, 74.0, 88.0, 116.5, 143.0][:args.num_tools]
    yaws = [round(rng.uniform(-10.0, 10.0), 1) for _ in angles]
    order = list(range(len(angles)))
    rng.shuffle(order)
    intervals = [round(rng.uniform(0.25, 0.55), 2) for _ in order]

    scene_xml = out / 'front110_v700_multi_clean_scene.xml'
    initials = build_multi_scene(scene_xml, angles, yaws)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    wrist_bid = model.body('panda_link7').id
    tool_bids = [model.body(f'tool{i}').id for i in range(len(angles))]
    tool_geoms = [[model.geom(f'handle_positive_{i}').id, model.geom(f'functional_negative_{i}').id] for i in range(len(angles))]
    qvadr = [model.jnt_dofadr[model.joint(f'tool_free_{i}').id] for i in range(len(angles))]

    q_arm_cmd = v700.Q_READY.copy()
    q_hand_cmd = v700.HAND_OPEN.copy()
    data.qpos[:7] = q_arm_cmd
    data.qpos[7:18] = q_hand_cmd
    data.ctrl[:7] = q_arm_cmd
    data.ctrl[7:18] = q_hand_cmd
    for i, (pos, quat) in enumerate(initials):
        set_tool_state(model, data, i, pos, quat)
        set_tool_visual_collision(model, tool_geoms[i], 'waiting')
    mujoco.mj_forward(model, data)

    frames_front, frames_top = [], []
    dt = model.opt.timestep
    spf = max(1, int(round((1.0 / args.fps) / dt)))
    drop_records = []

    t = 0.0
    total_steps = int(args.max_seconds / dt)
    drop_cursor = 0
    state = 'prep'
    state_t0 = 0.0
    cur = order[0]
    latch_time = None
    strict_run = 0
    best_strict = 0
    any_bad = False
    min_z = 10.0
    final_vz = 0.0
    grip_args = base_args(angles[cur], out)
    make_xmat = make_wrist_xmat_fn(data, wrist_bid, grip_args, angles[cur])
    wrist_to_pinch = v700.wrist_offset_for_angle(angles[cur], grip_args.radial_offset, grip_args.tangent_offset, grip_args.z_offset)

    for step in range(total_steps):
        t = step * dt
        for i, (pos, quat) in enumerate(initials):
            hold_current_before_release = state == 'fall' and (t - state_t0) < grip_args.release_time
            if i != cur or state in ('prep', 'done') or hold_current_before_release:
                if not any(r.get('tool_index') == i and r.get('hidden') for r in drop_records):
                    set_tool_state(model, data, i, pos, quat)
        mujoco.mj_forward(model, data)
        if state == 'prep':
            for i in range(len(angles)):
                set_tool_visual_collision(model, tool_geoms[i], 'current_wait' if i == cur else 'waiting')
            prep_age = t - state_t0
            tool_origin = data.xpos[tool_bids[cur]].copy()
            center, hand_des, _, _ = v700.stage4_grip_reference(tool_origin, angles[cur], -grip_args.release_time, catch_z=grip_args.catch_z,
                ready_tangent=grip_args.ready_tangent, move_start=grip_args.move_start, move_dur=grip_args.move_dur, insert_tangent=grip_args.insert_tangent,
                latch_drop_rate=grip_args.latch_drop_rate, latch_drop_duration=grip_args.latch_drop_duration, latch_min_z=grip_args.latch_min_z,
                latch_lift_rate=grip_args.latch_lift_rate, latch_lift_start=grip_args.latch_lift_start, latch_lift_duration=grip_args.latch_lift_duration, latch_max_z=grip_args.latch_max_z,
                latch_radial_offset=grip_args.latch_radial_offset, latch_tangent_offset=grip_args.latch_tangent_offset,
                latch_xy_follow=grip_args.latch_xy_follow, latch_xy_follow_start=grip_args.latch_xy_follow_start, latch_xy_follow_dur=grip_args.latch_xy_follow_dur,
                middle_latch_scale=grip_args.middle_latch_scale, middle_latch_start=grip_args.middle_latch_start, middle_latch_dur=grip_args.middle_latch_dur,
                index_latch_scale=grip_args.index_latch_scale, index_latch_start=grip_args.index_latch_start, index_latch_dur=grip_args.index_latch_dur,
                index_latch_prox=grip_args.index_latch_prox, index_latch_dist=grip_args.index_latch_dist,
                thumb_lead=grip_args.thumb_lead, thumb_dur=grip_args.thumb_dur, finger_lead=grip_args.finger_lead, finger_dur=grip_args.finger_dur,
                thumb_close_scale=grip_args.thumb_close_scale, index_close_scale=grip_args.index_close_scale, middle_close_scale=grip_args.middle_close_scale)
            q_arm_des = v700.solve_arm_ik(model, data, q_arm_cmd, wrist_bid, center + wrist_to_pinch, make_xmat(0.0))
            q_hand_des = v700.HAND_OPEN.copy()
            if prep_age >= args.prep_seconds:
                q_arm_cmd = v700.solve_arm_ik(model, data, q_arm_cmd, wrist_bid, center + wrist_to_pinch, make_xmat(0.0), max_iter=96)
                data.qpos[:7] = q_arm_cmd
                data.ctrl[:7] = q_arm_cmd
                data.qvel[:7] = 0.0
                mujoco.mj_forward(model, data)
                q_arm_des = q_arm_cmd.copy()
                state = 'fall'
                state_t0 = t
                latch_time = None
                strict_run = 0
                best_strict = 0
                any_bad = False
                min_z = 10.0
                final_vz = 0.0
                set_tool_visual_collision(model, tool_geoms[cur], 'current_wait')
        elif state == 'fall':
            local_t = t - state_t0
            if local_t < grip_args.release_time:
                set_tool_visual_collision(model, tool_geoms[cur], 'current_wait')
            else:
                set_tool_visual_collision(model, tool_geoms[cur], 'active')
            tau = local_t - grip_args.release_time
            latch_age = 0.0 if latch_time is None else t - latch_time
            tool_origin = data.xpos[tool_bids[cur]].copy()
            center, hand_des, _, _ = v700.stage4_grip_reference(tool_origin, angles[cur], tau, latch_time is not None, latch_age, catch_z=grip_args.catch_z,
                ready_tangent=grip_args.ready_tangent, move_start=grip_args.move_start, move_dur=grip_args.move_dur, insert_tangent=grip_args.insert_tangent,
                latch_drop_rate=grip_args.latch_drop_rate, latch_drop_duration=grip_args.latch_drop_duration, latch_min_z=grip_args.latch_min_z,
                latch_lift_rate=grip_args.latch_lift_rate, latch_lift_start=grip_args.latch_lift_start, latch_lift_duration=grip_args.latch_lift_duration, latch_max_z=grip_args.latch_max_z,
                latch_radial_offset=grip_args.latch_radial_offset, latch_tangent_offset=grip_args.latch_tangent_offset,
                latch_xy_follow=grip_args.latch_xy_follow, latch_xy_follow_start=grip_args.latch_xy_follow_start, latch_xy_follow_dur=grip_args.latch_xy_follow_dur,
                middle_latch_scale=grip_args.middle_latch_scale, middle_latch_start=grip_args.middle_latch_start, middle_latch_dur=grip_args.middle_latch_dur,
                index_latch_scale=grip_args.index_latch_scale, index_latch_start=grip_args.index_latch_start, index_latch_dur=grip_args.index_latch_dur,
                index_latch_prox=grip_args.index_latch_prox, index_latch_dist=grip_args.index_latch_dist,
                thumb_lead=grip_args.thumb_lead, thumb_dur=grip_args.thumb_dur, finger_lead=grip_args.finger_lead, finger_dur=grip_args.finger_dur,
                thumb_close_scale=grip_args.thumb_close_scale, index_close_scale=grip_args.index_close_scale, middle_close_scale=grip_args.middle_close_scale)
            q_arm_des = v700.solve_arm_ik(model, data, q_arm_cmd, wrist_bid, center + wrist_to_pinch, make_xmat(latch_age))
            q_hand_des = hand_des
            if local_t >= grip_args.release_time:
                thumb_c, finger_c, bad = contact_metrics_current(model, data, cur)
                any_bad = any_bad or bad
                if thumb_c and finger_c and not bad:
                    if latch_time is None:
                        latch_time = t
                    strict_run += 1
                    best_strict = max(best_strict, strict_run)
                else:
                    strict_run = 0
            min_z = min(min_z, float(data.xpos[tool_bids[cur]][2]))
            final_vz = float(data.qvel[qvadr[cur] + 2])
            if latch_time is not None and (t - latch_time) >= args.hold_seconds:
                state = 'discard'
                state_t0 = t
            elif tau > 1.10:
                state = 'discard'
                state_t0 = t
        elif state == 'discard':
            age = t - state_t0
            q_hand_des = v700.HAND_OPEN.copy()
            base = data.xpos[wrist_bid].copy()
            shake = np.array([0.05 * math.sin(42 * age), -0.10 - 0.04 * math.sin(27 * age), 0.05 * math.sin(31 * age)])
            q_arm_des = v700.solve_arm_ik(model, data, q_arm_cmd, wrist_bid, base + shake, make_xmat(0.0), max_iter=24)
            if age >= args.discard_seconds:
                success = best_strict >= 10 and not any_bad and min_z > 0.30 and abs(final_vz) < 1.2
                drop_records.append({'tool_index': cur, 'angle_deg': angles[cur], 'yaw_deg': yaws[cur], 'success': bool(success), 'best_strict': int(best_strict), 'bad_functional_contact': bool(any_bad), 'min_z': round(min_z, 3), 'final_vz': round(final_vz, 3), 'hidden': True})
                set_tool_visual_collision(model, tool_geoms[cur], 'hidden')
                set_tool_state(model, data, cur, np.array([3.0 + cur, 3.0, -1.0]), np.array([1, 0, 0, 0], dtype=float))
                drop_cursor += 1
                if drop_cursor >= len(order):
                    state = 'done'
                    state_t0 = t
                    q_arm_des = v700.Q_READY.copy()
                    q_hand_des = v700.HAND_OPEN.copy()
                else:
                    cur = order[drop_cursor]
                    grip_args = base_args(angles[cur], out)
                    make_xmat = make_wrist_xmat_fn(data, wrist_bid, grip_args, angles[cur])
                    wrist_to_pinch = v700.wrist_offset_for_angle(angles[cur], grip_args.radial_offset, grip_args.tangent_offset, grip_args.z_offset)
                    state = 'prep'
                    state_t0 = t + intervals[drop_cursor]
                    q_arm_des = v700.Q_READY.copy()
                    q_hand_des = v700.HAND_OPEN.copy()
        else:
            q_arm_des = v700.Q_READY.copy()
            q_hand_des = v700.HAND_OPEN.copy()
            if t - state_t0 > 0.8:
                break

        if state == 'prep' and t < state_t0:
            q_arm_des = v700.Q_READY.copy()
            q_hand_des = v700.HAND_OPEN.copy()
        state_before_step = state
        q_arm_cmd = v700.limit_step(q_arm_cmd, q_arm_des, v700.FR3_VEL_LIMIT, dt)
        q_hand_cmd = v700.limit_step(q_hand_cmd, q_hand_des, v700.HAND_VEL_LIMIT, dt)
        data.ctrl[:7] = q_arm_cmd
        data.ctrl[7:18] = q_hand_cmd
        # The visible pre-release preparation is a motion-planner segment.
        # We write qpos along the velocity-limited command so the release starts
        # exactly from the planned pregrasp, then the fall/catch phase returns to
        # strict MuJoCo contact dynamics.
        if state_before_step == 'prep' or state_before_step == 'done':
            data.qpos[:7] = q_arm_cmd
            # Do not force the Revo2 finger qpos to the command. The validated
            # v700 catch relies on actuator-limited finger lag; writing qpos
            # here closes the fingers too early and blocks the falling handle.
            data.qvel[:7] = 0.0
            mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)

        if step % spf == 0:
            renderer.update_scene(data, camera=model.camera('wide_front').id)
            frames_front.append(renderer.render())
            renderer.update_scene(data, camera=model.camera('wide_top_oblique').id)
            frames_top.append(renderer.render())

    renderer.close()
    front_video = out / f'front110_v700_multi_clean_seed{args.seed}_wide_front.mp4'
    top_video = out / f'front110_v700_multi_clean_seed{args.seed}_wide_top_oblique.mp4'
    imageio.mimsave(front_video, frames_front, fps=args.fps, macro_block_size=16)
    imageio.mimsave(top_video, frames_top, fps=args.fps, macro_block_size=16)
    summary = {'seed': args.seed, 'angles_deg': angles, 'release_order_tool_indices': order, 'release_order_angles_deg': [angles[i] for i in order], 'intervals_s': intervals, 'videos': {'wide_front': str(front_video), 'wide_top_oblique': str(top_video)}, 'drops': drop_records, 'passed': sum(1 for r in drop_records if r['success']), 'total': len(drop_records), 'controller': 'frozen/stage4e_front110_v700_grid0p5_patch.py', 'scene_xml': str(scene_xml)}
    (out / f'front110_v700_multi_clean_seed{args.seed}.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=70117)
    ap.add_argument('--num-tools', type=int, default=6)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--width', type=int, default=1600)
    ap.add_argument('--height', type=int, default=912)
    ap.add_argument('--prep-seconds', type=float, default=0.46)
    ap.add_argument('--hold-seconds', type=float, default=0.48)
    ap.add_argument('--discard-seconds', type=float, default=0.42)
    ap.add_argument('--max-seconds', type=float, default=13.5)
    ap.add_argument('--out-dir', default=str(OUT))
    run_demo(ap.parse_args())

if __name__ == '__main__':
    main()

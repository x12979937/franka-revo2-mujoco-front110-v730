#!/usr/bin/env python3
import argparse
import json
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ROBOT_XML = PROJECT / 'assets/full_robot_urdf_mirror/franka_brainco_revo2_right_converted_mjcf.xml'

FR3_JOINTS = [
    'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
    'panda_joint5', 'panda_joint6', 'panda_joint7',
]
HAND_JOINTS = [
    'revo2_right_thumb_metacarpal_joint',
    'revo2_right_thumb_proximal_joint',
    'revo2_right_thumb_distal_joint',
    'revo2_right_index_proximal_joint',
    'revo2_right_index_distal_joint',
    'revo2_right_middle_proximal_joint',
    'revo2_right_middle_distal_joint',
    'revo2_right_ring_proximal_joint',
    'revo2_right_ring_distal_joint',
    'revo2_right_pinky_proximal_joint',
    'revo2_right_pinky_distal_joint',
]

FR3_VEL_LIMIT = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26], dtype=float)
HAND_VEL_LIMIT = np.array([2.62, 2.53, 2.53, 2.27, 2.27, 2.27, 2.27, 2.27, 2.27, 2.27, 2.27], dtype=float)

# Center-front ready posture from IK for a panda_link7 target above the 90 deg catch window.
Q_READY = np.array([0.3110, -0.3980, 1.0296, -1.0451, -0.0464, 1.9217, 0.80], dtype=float)
HAND_OPEN = np.array([0.0, 0.0, 0.0, 0.08, 0.08, 0.08, 0.08, 0.0, 0.0, 0.0, 0.0], dtype=float)
HAND_CLOSE = np.array([0.92, 0.58, 0.45, 0.80, 0.74, 0.08, 0.08, 0.0, 0.0, 0.0, 0.0], dtype=float)
HAND_UPPER = np.array([1.57, 1.03, 1.03, 1.41, 1.63, 1.41, 1.63, 1.41, 1.63, 1.41, 1.63], dtype=float)


def smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], dtype=float)


def rotz(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rotx(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def roty(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def build_scene(scene_xml: Path):
    root = ET.parse(ROBOT_XML).getroot()
    compiler = root.find('compiler')
    if compiler is None:
        compiler = ET.Element('compiler')
        root.insert(0, compiler)
    compiler.set('meshdir', str(ROBOT_XML.parent))

    option = root.find('option')
    if option is None:
        option = ET.Element('option')
        root.insert(1, option)
    option.set('timestep', '0.002')
    option.set('gravity', '0 0 -9.81')
    option.set('integrator', 'implicitfast')
    option.set('iterations', '80')

    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', {'offwidth': '1280', 'offheight': '720'})

    joint_ranges = {joint.get('name'): joint.get('range') for joint in root.iter('joint') if joint.get('name')}
    actuator = root.find('actuator')
    if actuator is None:
        actuator = ET.SubElement(root, 'actuator')
    for joint in FR3_JOINTS:
        attrs = {'name': f'{joint}_pos', 'joint': joint, 'kp': '220'}
        if joint_ranges.get(joint):
            attrs.update({'ctrllimited': 'true', 'ctrlrange': joint_ranges[joint]})
        ET.SubElement(actuator, 'position', attrs)
    for joint in HAND_JOINTS:
        attrs = {'name': f'{joint}_pos', 'joint': joint, 'kp': '180'}
        if joint_ranges.get(joint):
            attrs.update({'ctrllimited': 'true', 'ctrlrange': joint_ranges[joint]})
        ET.SubElement(actuator, 'position', attrs)

    world = root.find('worldbody')
    for joint in root.iter('joint'):
        name = joint.get('name', '')
        if name in HAND_JOINTS:
            joint.set('damping', '0.35')
            joint.set('armature', '0.006')
        else:
            joint.set('damping', '4.0')
            joint.set('armature', '0.05')
    for geom in root.iter('geom'):
        geom.set('contype', '0')
        geom.set('conaffinity', '0')
    for body in root.iter('body'):
        name = body.get('name', '')
        if name != 'tool':
            body.set('gravcomp', '1')
        if name in {
            'revo2_right_thumb_distal_link',
            'revo2_right_index_distal_link',
            'revo2_right_middle_distal_link',
        }:
            for body_geom in body.findall('geom'):
                mesh_name = body_geom.get('mesh', '')
                if body_geom.get('type') == 'box' or mesh_name.endswith('_touch_link'):
                    body_geom.set('contype', '1')
                    body_geom.set('conaffinity', '1')
                    body_geom.set('friction', '2.80 0.08 0.008')
        if name == 'revo2_right_thumb_distal_link':
            ET.SubElement(body, 'geom', {'name': 'stage4_thumb_tip_marker', 'type': 'sphere', 'pos': '0 0.0127283 0.00674553', 'size': '0.014', 'rgba': '0.1 0.2 1.0 0.16', 'contype': '0', 'conaffinity': '0'})
        if name == 'revo2_right_index_distal_link':
            ET.SubElement(body, 'geom', {'name': 'stage4_index_tip_marker', 'type': 'sphere', 'pos': '0.0131721 -0.00014321 0.0250986', 'size': '0.014', 'rgba': '1.0 0.70 0.05 0.16', 'contype': '0', 'conaffinity': '0'})
        if name == 'revo2_right_middle_distal_link':
            ET.SubElement(body, 'geom', {'name': 'stage4_middle_tip_marker', 'type': 'sphere', 'pos': '0.0151136 0 0.0292832', 'size': '0.014', 'rgba': '1.0 0.70 0.05 0.16', 'contype': '0', 'conaffinity': '0'})
    ET.SubElement(world, 'light', {'pos': '0 -2 4', 'dir': '0 0 -1', 'diffuse': '0.9 0.9 0.9'})
    ET.SubElement(world, 'camera', {'name': 'front_arc', 'pos': '0 -1.9 1.15', 'xyaxes': '1 0 0 0 0.50 0.86', 'fovy': '48'})
    ET.SubElement(world, 'camera', {'name': 'top_oblique', 'pos': '0.0 -1.25 1.75', 'xyaxes': '1 0 0 0 0.84 0.54', 'fovy': '52'})
    ET.SubElement(world, 'geom', {'name': 'floor', 'type': 'plane', 'size': '2 2 0.02', 'rgba': '0.32 0.32 0.32 1', 'friction': '0.8 0.01 0.001'})

    tool = ET.SubElement(world, 'body', {'name': 'tool', 'pos': '0 0.5 1.15', 'quat': '1 0 0 0'})
    ET.SubElement(tool, 'freejoint', {'name': 'tool_free'})
    ET.SubElement(tool, 'inertial', {'pos': '0 0 0.055', 'mass': '0.125', 'diaginertia': '0.00115 0.00115 0.000035'})
    # Stage4b uses a handle-down release: the graspable green handle reaches
    # the clamp window before the red functional segment can sweep the fingers.
    ET.SubElement(tool, 'geom', {'name': 'handle_positive', 'type': 'capsule', 'fromto': '0 0 -0.11 0 0 0.00', 'size': '0.015', 'rgba': '0.05 0.75 0.25 1', 'friction': '2.80 0.080 0.008'})
    ET.SubElement(tool, 'geom', {'name': 'functional_negative', 'type': 'capsule', 'fromto': '0 0 0.00 0 0 0.22', 'size': '0.015', 'rgba': '0.70 0.05 0.04 1', 'friction': '0.55 0.010 0.001'})

    ET.ElementTree(root).write(scene_xml, encoding='unicode')


def set_tool(model, data, angle_deg, yaw_deg, radius=0.50, height=1.15):
    qadr = model.jnt_qposadr[model.joint('tool_free').id]
    th = math.radians(angle_deg)
    data.qpos[qadr:qadr + 3] = [radius * math.cos(th), radius * math.sin(th), height]
    data.qpos[qadr + 3:qadr + 7] = yaw_quat(math.radians(yaw_deg))
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def clamp_joint_ranges(model, q):
    out = q.copy()
    for i, name in enumerate(FR3_JOINTS):
        jid = model.joint(name).id
        lo, hi = model.jnt_range[jid]
        out[i] = min(hi, max(lo, out[i]))
    return out


def solve_arm_ik(model, data, q_seed, body_id, target_pos, target_xmat=None, max_iter=32):
    q = q_seed.copy()
    qpos0 = data.qpos.copy()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(max_iter):
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        pos_err = target_pos - data.xpos[body_id]
        if target_xmat is None:
            err = pos_err
        else:
            cur = data.xmat[body_id].reshape(3, 3)
            des = target_xmat.reshape(3, 3)
            rot_err = 0.5 * (
                np.cross(cur[:, 0], des[:, 0]) +
                np.cross(cur[:, 1], des[:, 1]) +
                np.cross(cur[:, 2], des[:, 2])
            )
            err = np.concatenate([pos_err, 0.18 * rot_err])
        if float(np.linalg.norm(pos_err)) < 0.006 and (target_xmat is None or float(np.linalg.norm(err[3:])) < 0.015):
            break
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        if target_xmat is None:
            j = jacp[:, :7]
        else:
            j = np.vstack([jacp[:, :7], 0.18 * jacr[:, :7]])
        damp = 0.035
        dq = j.T @ np.linalg.solve(j @ j.T + damp * np.eye(j.shape[0]), err)
        q = clamp_joint_ranges(model, q + 0.62 * dq)
    data.qpos[:] = qpos0
    mujoco.mj_forward(model, data)
    return q


def limit_step(current, desired, limits, dt):
    delta = np.clip(desired - current, -limits * dt, limits * dt)
    return current + delta


def contact_metrics(model, data):
    thumb = False
    index_middle = False
    bad_functional = False
    pairs = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        b1 = model.body(model.geom_bodyid[c.geom1]).name
        b2 = model.body(model.geom_bodyid[c.geom2]).name
        pairs.append((g1, b1, g2, b2))
        names = ' '.join(str(x) for x in (g1, b1, g2, b2))
        if 'handle_positive' in names and 'thumb' in names:
            thumb = True
        if 'handle_positive' in names and ('index' in names or 'middle' in names):
            index_middle = True
        if 'functional_negative' in names and ('thumb' in names or 'index' in names or 'middle' in names or 'ring' in names or 'pinky' in names or 'panda' in names):
            bad_functional = True
    return thumb, index_middle, bad_functional, pairs


def point_segment_distance(point, a, b):
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - a))
    u = float(np.dot(point - a, ab) / denom)
    u = min(1.0, max(0.0, u))
    return float(np.linalg.norm(point - (a + u * ab)))


def stage4_grip_reference(
    tool_origin,
    angle_deg,
    tau,
    latched=False,
    latch_age=0.0,
    catch_z=0.875,
    ready_tangent=0.120,
    move_start=0.120,
    move_dur=0.130,
    insert_tangent=0.040,
    latch_drop_rate=0.850,
    latch_drop_duration=0.180,
    latch_min_z=0.580,
    latch_lift_rate=0.0,
    latch_lift_start=0.160,
    latch_lift_duration=0.180,
    latch_max_z=0.920,
    latch_radial_offset=0.0,
    latch_tangent_offset=-0.010,
    latch_xy_follow=0.0,
    latch_xy_follow_start=0.0,
    latch_xy_follow_dur=0.080,
    middle_latch_scale=0.0,
    middle_latch_start=0.030,
    middle_latch_dur=0.120,
    index_latch_scale=0.0,
    index_latch_start=0.030,
    index_latch_dur=0.120,
    index_latch_prox=1.05,
    index_latch_dist=1.05,
    thumb_lead=0.160,
    thumb_dur=0.240,
    finger_lead=0.110,
    finger_dur=0.180,
    thumb_close_scale=1.0,
    index_close_scale=1.0,
    middle_close_scale=0.0,
):
    th = math.radians(angle_deg)
    radial = np.array([math.cos(th), math.sin(th)])
    tangent = np.array([-math.sin(th), math.cos(th)])
    target_xy = np.array([0.50 * math.cos(th), 0.50 * math.sin(th)])
    # Stage4b keeps the proven near-field pinch trajectory from stage4, but
    # pairs it with a handle-down release so the green segment enters first.
    ready_xy = target_xy + ready_tangent * tangent
    move_a = smoothstep((tau - move_start) / move_dur)
    insert_a = smoothstep((tau - 0.130) / 0.050)
    thumb_a = smoothstep((tau + thumb_lead) / max(1e-6, thumb_dur))
    finger_a = smoothstep((tau + finger_lead) / max(1e-6, finger_dur))
    center_xy = ready_xy * (1.0 - move_a) + target_xy * move_a
    center_xy = center_xy + insert_tangent * tangent * (1.0 - insert_a)

    center_z = max(0.58, catch_z - 0.200 * max(0.0, tau - 0.180))
    if latched:
        center_z = max(latch_min_z, catch_z - latch_drop_rate * min(latch_age, latch_drop_duration))
        if latch_lift_rate > 0.0 and latch_age > latch_lift_start:
            lift_age = min(latch_age - latch_lift_start, latch_lift_duration)
            center_z = min(latch_max_z, center_z + latch_lift_rate * lift_age)
    if latched:
        center_xy = target_xy + latch_radial_offset * radial + latch_tangent_offset * tangent
        if latch_xy_follow > 0.0:
            follow_a = latch_xy_follow * smoothstep((latch_age - latch_xy_follow_start) / max(1e-6, latch_xy_follow_dur))
            follow_xy = np.asarray(tool_origin[:2], dtype=float) + latch_radial_offset * radial + latch_tangent_offset * tangent
            center_xy = center_xy * (1.0 - follow_a) + follow_xy * follow_a
    hand_close = HAND_CLOSE.copy()
    hand_close[:3] = np.minimum(
        HAND_UPPER[:3],
        HAND_OPEN[:3] + (HAND_CLOSE[:3] - HAND_OPEN[:3]) * thumb_close_scale,
    )
    hand_close[3:5] = np.minimum(
        HAND_UPPER[3:5],
        HAND_OPEN[3:5] + (HAND_CLOSE[3:5] - HAND_OPEN[3:5]) * index_close_scale,
    )
    if middle_close_scale > 0.0:
        middle_pre_target = np.array([0.80, 0.74], dtype=float)
        hand_close[5:7] = np.minimum(
            HAND_UPPER[5:7],
            HAND_OPEN[5:7] + (middle_pre_target - HAND_OPEN[5:7]) * middle_close_scale,
        )
    close_q = HAND_OPEN.copy()
    close_q[:3] = HAND_OPEN[:3] * (1.0 - thumb_a) + hand_close[:3] * thumb_a
    close_q[3:7] = HAND_OPEN[3:7] * (1.0 - finger_a) + hand_close[3:7] * finger_a
    if latched and middle_latch_scale > 0.0:
        middle_a = middle_latch_scale * smoothstep((latch_age - middle_latch_start) / max(1e-6, middle_latch_dur))
        middle_target = np.minimum(HAND_UPPER[5:7], np.array([0.80, 0.74], dtype=float))
        close_q[5:7] = close_q[5:7] * (1.0 - middle_a) + middle_target * middle_a
    if latched and index_latch_scale > 0.0:
        index_a = index_latch_scale * smoothstep((latch_age - index_latch_start) / max(1e-6, index_latch_dur))
        index_target = np.minimum(HAND_UPPER[3:5], np.array([index_latch_prox, index_latch_dist], dtype=float))
        close_q[3:5] = close_q[3:5] * (1.0 - index_a) + index_target * index_a
    return np.array([center_xy[0], center_xy[1], center_z], dtype=float), close_q, move_a, max(thumb_a, finger_a)

def wrist_offset_for_angle(angle_deg, radial_coeff=-0.205, tangent_coeff=-0.095, z_offset=0.245):
    th = math.radians(angle_deg)
    radial = np.array([math.cos(th), math.sin(th)])
    tangent = np.array([-math.sin(th), math.cos(th)])
    # Calibrated from the converted full Revo2 geometry at the handle intercept height.
    xy = radial_coeff * radial + tangent_coeff * tangent
    return np.array([xy[0], xy[1], z_offset], dtype=float)


def run_episode(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scene_xml = out / 'stage4c_front_angle_yaw_scene.xml'
    build_scene(scene_xml)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    record_video = (not args.no_video) and args.fps > 0
    renderer = mujoco.Renderer(model, height=720, width=1280) if record_video else None

    rng = random.Random(args.seed)
    angle = args.angle if args.angle is not None else rng.uniform(35.0, 145.0)
    yaw = args.yaw if args.yaw is not None else rng.uniform(-10.0, 10.0)
    set_tool(model, data, angle, yaw)
    tool_initial_qpos = data.qpos[model.jnt_qposadr[model.joint('tool_free').id]:model.jnt_qposadr[model.joint('tool_free').id] + 7].copy()

    tool_bid = model.body('tool').id
    wrist_bid = model.body('panda_link7').id
    cam_id = model.camera(args.camera).id
    qvadr = model.jnt_dofadr[model.joint('tool_free').id]
    tool_geom_ids = [model.geom('handle_positive').id, model.geom('functional_negative').id]
    tip_marker_ids = {}
    for name in ['stage4_thumb_tip_marker', 'stage4_index_tip_marker', 'stage4_middle_tip_marker']:
        try:
            tip_marker_ids[name] = model.geom(name).id
        except KeyError:
            pass
    tool_contype = {gid: int(model.geom_contype[gid]) for gid in tool_geom_ids}
    tool_conaffinity = {gid: int(model.geom_conaffinity[gid]) for gid in tool_geom_ids}

    q_arm_cmd = Q_READY.copy()
    q_hand_cmd = HAND_OPEN.copy()
    data.qpos[:7] = q_arm_cmd
    data.qpos[7:18] = q_hand_cmd
    data.ctrl[:7] = q_arm_cmd
    data.ctrl[7:18] = q_hand_cmd
    mujoco.mj_forward(model, data)
    wrist_ready_xmat = data.xmat[wrist_bid].copy()
    delta = angle - 90.0
    wrist_ready_mat = wrist_ready_xmat.reshape(3, 3)

    def make_wrist_xmat(latch_age=0.0):
        brake_a = smoothstep((latch_age - args.latch_brake_start) / max(1e-6, args.latch_brake_duration))
        return (
            rotz(math.radians(args.wrist_yaw + args.wrist_yaw_factor * delta + args.latch_brake_yaw * brake_a))
            @ rotx(math.radians(args.wrist_roll + args.wrist_roll_factor * delta + args.latch_brake_roll * brake_a))
            @ roty(math.radians(args.wrist_pitch + args.wrist_pitch_factor * delta + args.latch_brake_pitch * brake_a))
            @ wrist_ready_mat
        ).reshape(9)

    wrist_target_xmat = make_wrist_xmat(0.0)

    wrist_to_pinch = wrist_offset_for_angle(angle, args.radial_offset, args.tangent_offset, args.z_offset)
    initial_center, _, _, _ = stage4_grip_reference(
        data.xpos[tool_bid].copy(),
        angle,
        -args.release_time,
        catch_z=args.catch_z,
        ready_tangent=args.ready_tangent,
        move_start=args.move_start,
        move_dur=args.move_dur,
        insert_tangent=args.insert_tangent,
        latch_drop_rate=args.latch_drop_rate,
        latch_drop_duration=args.latch_drop_duration,
        latch_min_z=args.latch_min_z,
        latch_lift_rate=args.latch_lift_rate,
        latch_lift_start=args.latch_lift_start,
        latch_lift_duration=args.latch_lift_duration,
        latch_max_z=args.latch_max_z,
        latch_radial_offset=args.latch_radial_offset,
        latch_tangent_offset=args.latch_tangent_offset,
        latch_xy_follow=args.latch_xy_follow,
        latch_xy_follow_start=args.latch_xy_follow_start,
        latch_xy_follow_dur=args.latch_xy_follow_dur,
        middle_latch_scale=args.middle_latch_scale,
        middle_latch_start=args.middle_latch_start,
        middle_latch_dur=args.middle_latch_dur,
        index_latch_scale=args.index_latch_scale,
        index_latch_start=args.index_latch_start,
        index_latch_dur=args.index_latch_dur,
        index_latch_prox=args.index_latch_prox,
        index_latch_dist=args.index_latch_dist,
        thumb_lead=args.thumb_lead,
        thumb_dur=args.thumb_dur,
        finger_lead=args.finger_lead,
        finger_dur=args.finger_dur,
        thumb_close_scale=args.thumb_close_scale,
        index_close_scale=args.index_close_scale,
        middle_close_scale=args.middle_close_scale,
    )
    initial_wrist_target = initial_center + wrist_to_pinch
    q_arm_cmd = solve_arm_ik(model, data, Q_READY.copy(), wrist_bid, initial_wrist_target, wrist_target_xmat, max_iter=96)
    data.qpos[:7] = q_arm_cmd
    data.ctrl[:7] = q_arm_cmd
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    spf = max(1, int(round((1 / max(1, args.fps)) / dt)))
    frames = []
    strict_run = 0
    best_strict = 0
    any_bad = False
    max_ik_err = 0.0
    avg_ik_err_acc = 0.0
    samples = 0
    min_tool_z = 10.0
    final_vz = 0.0
    latch_time = None
    first_strict_t = None
    contact_events = []
    min_tip_handle_dist = {name: 10.0 for name in tip_marker_ids}

    for step in range(int(args.seconds / dt)):
        t = step * dt
        if t < args.release_time:
            for gid in tool_geom_ids:
                model.geom_contype[gid] = 0
                model.geom_conaffinity[gid] = 0
            qadr = model.jnt_qposadr[model.joint('tool_free').id]
            data.qpos[qadr:qadr + 7] = tool_initial_qpos
            data.qvel[qvadr:qvadr + 6] = 0.0
        else:
            for gid in tool_geom_ids:
                model.geom_contype[gid] = tool_contype[gid]
                model.geom_conaffinity[gid] = tool_conaffinity[gid]
        mujoco.mj_forward(model, data)
        tau = t - args.release_time
        latched = latch_time is not None
        latch_age = 0.0 if latch_time is None else max(0.0, t - latch_time)
        grip_center, hand_des, move_a, _ = stage4_grip_reference(
            data.xpos[tool_bid].copy(),
            angle,
            tau,
            latched,
            latch_age,
            catch_z=args.catch_z,
            ready_tangent=args.ready_tangent,
            move_start=args.move_start,
            move_dur=args.move_dur,
            insert_tangent=args.insert_tangent,
            latch_drop_rate=args.latch_drop_rate,
            latch_drop_duration=args.latch_drop_duration,
            latch_min_z=args.latch_min_z,
            latch_lift_rate=args.latch_lift_rate,
            latch_lift_start=args.latch_lift_start,
            latch_lift_duration=args.latch_lift_duration,
            latch_max_z=args.latch_max_z,
            latch_radial_offset=args.latch_radial_offset,
            latch_tangent_offset=args.latch_tangent_offset,
            latch_xy_follow=args.latch_xy_follow,
            latch_xy_follow_start=args.latch_xy_follow_start,
            latch_xy_follow_dur=args.latch_xy_follow_dur,
            middle_latch_scale=args.middle_latch_scale,
            middle_latch_start=args.middle_latch_start,
            middle_latch_dur=args.middle_latch_dur,
            index_latch_scale=args.index_latch_scale,
            index_latch_start=args.index_latch_start,
            index_latch_dur=args.index_latch_dur,
            index_latch_prox=args.index_latch_prox,
            index_latch_dist=args.index_latch_dist,
            thumb_lead=args.thumb_lead,
            thumb_dur=args.thumb_dur,
            finger_lead=args.finger_lead,
            finger_dur=args.finger_dur,
            thumb_close_scale=args.thumb_close_scale,
            index_close_scale=args.index_close_scale,
            middle_close_scale=args.middle_close_scale,
        )
        wrist_target = grip_center + wrist_to_pinch
        wrist_target_xmat_now = make_wrist_xmat(latch_age if latched else 0.0)
        q_arm_des = solve_arm_ik(model, data, q_arm_cmd, wrist_bid, wrist_target, wrist_target_xmat_now)
        q_arm_cmd = limit_step(q_arm_cmd, q_arm_des, FR3_VEL_LIMIT, dt)
        q_hand_cmd = limit_step(q_hand_cmd, hand_des, HAND_VEL_LIMIT, dt)
        data.ctrl[:7] = q_arm_cmd
        data.ctrl[7:18] = q_hand_cmd

        mujoco.mj_step(model, data)
        ik_err = float(np.linalg.norm(data.xpos[wrist_bid] - wrist_target))
        max_ik_err = max(max_ik_err, ik_err)
        avg_ik_err_acc += ik_err
        samples += 1
        thumb_c, finger_c, bad, _ = contact_metrics(model, data)
        any_bad = any_bad or bad
        if thumb_c and finger_c and not bad:
            if latch_time is None:
                latch_time = t
                first_strict_t = t
            if not contact_events or contact_events[-1].get('end_t') is not None:
                contact_events.append({'start_t': round(t, 4), 'end_t': None})
            strict_run += 1
            best_strict = max(best_strict, strict_run)
        else:
            if contact_events and contact_events[-1].get('end_t') is None:
                contact_events[-1]['end_t'] = round(t, 4)
            strict_run = 0
        final_vz = float(data.qvel[qvadr + 2])
        min_tool_z = min(min_tool_z, float(data.xpos[tool_bid][2]))
        if tip_marker_ids:
            tool_mat = data.xmat[tool_bid].reshape(3, 3)
            handle_a = data.xpos[tool_bid] + tool_mat @ np.array([0.0, 0.0, -0.11])
            handle_b = data.xpos[tool_bid] + tool_mat @ np.array([0.0, 0.0, 0.0])
            for name, gid in tip_marker_ids.items():
                d_tip = point_segment_distance(data.geom_xpos[gid], handle_a, handle_b)
                min_tip_handle_dist[name] = min(min_tip_handle_dist[name], d_tip)
        if record_video and step % spf == 0:
            renderer.update_scene(data, camera=cam_id)
            frames.append(renderer.render())

    if renderer is not None:
        renderer.close()
    if contact_events and contact_events[-1].get('end_t') is None:
        contact_events[-1]['end_t'] = round(args.seconds, 4)
    success = best_strict >= 10 and not any_bad and min_tool_z > 0.30 and abs(final_vz) < 1.2
    suffix = f'angle{angle:.1f}' if args.angle is not None else 'random'
    video = out / f'stage4c_front_angle_yaw_{suffix}_seed{args.seed}_{args.camera}.mp4'
    meta = out / f'stage4c_front_angle_yaw_{suffix}_seed{args.seed}_{args.camera}.json'
    if record_video:
        imageio.mimsave(video, frames, fps=args.fps, macro_block_size=16)
    result = {
        'stage': 'stage4c_front_angle_yaw',
        'note': 'Stage4c keeps the successful stage4b handle-down physical grasp, then rotates the FR3 wrist target orientation with the front-arc angle instead of reusing the 90 degree wrist orientation everywhere.',
        'tool_orientation': 'handle_down_positive_region_first',
        'angle_deg': round(angle, 3),
        'yaw_deg': round(yaw, 3),
        'video': str(video) if record_video else None,
        'scene_xml': str(scene_xml),
        'success': bool(success),
        'first_strict_contact_t': None if first_strict_t is None else round(first_strict_t, 4),
        'strict_contact_events': contact_events[:12],
        'best_strict_handle_contact_frames': int(best_strict),
        'bad_functional_contact': bool(any_bad),
        'final_vz': round(final_vz, 3),
        'min_tool_z': round(min_tool_z, 3),
        'avg_wrist_ik_error_m': round(avg_ik_err_acc / max(1, samples), 4),
        'max_wrist_ik_error_m': round(max_ik_err, 4),
        'min_tip_handle_dist_m': {k: round(v, 4) for k, v in min_tip_handle_dist.items()},
        'fr3_joint_vel_limit_rad_s': FR3_VEL_LIMIT.tolist(),
        'revo2_joint_vel_limit_rad_s': HAND_VEL_LIMIT.tolist(),
        'q_ready': Q_READY.round(4).tolist(),
        'hand_open': HAND_OPEN.round(4).tolist(),
        'hand_close_thumb_index_middle_only': HAND_CLOSE.round(4).tolist(),
        'radial_offset': round(args.radial_offset, 4),
        'tangent_offset': round(args.tangent_offset, 4),
        'z_offset': round(args.z_offset, 4),
        'catch_z': round(args.catch_z, 4),
        'wrist_yaw_factor': round(args.wrist_yaw_factor, 4),
        'wrist_yaw': round(args.wrist_yaw, 4),
        'wrist_roll_factor': round(args.wrist_roll_factor, 4),
        'wrist_roll': round(args.wrist_roll, 4),
        'wrist_pitch_factor': round(args.wrist_pitch_factor, 4),
        'wrist_pitch': round(args.wrist_pitch, 4),
        'ready_tangent': round(args.ready_tangent, 4),
        'move_start': round(args.move_start, 4),
        'move_dur': round(args.move_dur, 4),
        'insert_tangent': round(args.insert_tangent, 4),
        'latch_drop_rate': round(args.latch_drop_rate, 4),
        'latch_drop_duration': round(args.latch_drop_duration, 4),
        'latch_min_z': round(args.latch_min_z, 4),
        'latch_lift_rate': round(args.latch_lift_rate, 4),
        'latch_lift_start': round(args.latch_lift_start, 4),
        'latch_lift_duration': round(args.latch_lift_duration, 4),
        'latch_max_z': round(args.latch_max_z, 4),
        'latch_radial_offset': round(args.latch_radial_offset, 4),
        'latch_tangent_offset': round(args.latch_tangent_offset, 4),
        'latch_xy_follow': round(args.latch_xy_follow, 4),
        'latch_xy_follow_start': round(args.latch_xy_follow_start, 4),
        'latch_xy_follow_dur': round(args.latch_xy_follow_dur, 4),
        'middle_latch_scale': round(args.middle_latch_scale, 4),
        'middle_latch_start': round(args.middle_latch_start, 4),
        'middle_latch_dur': round(args.middle_latch_dur, 4),
        'index_latch_scale': round(args.index_latch_scale, 4),
        'index_latch_start': round(args.index_latch_start, 4),
        'index_latch_dur': round(args.index_latch_dur, 4),
        'index_latch_prox': round(args.index_latch_prox, 4),
        'index_latch_dist': round(args.index_latch_dist, 4),
        'latch_brake_duration': round(args.latch_brake_duration, 4),
        'latch_brake_yaw': round(args.latch_brake_yaw, 4),
        'latch_brake_roll': round(args.latch_brake_roll, 4),
        'latch_brake_pitch': round(args.latch_brake_pitch, 4),
        'thumb_lead': round(args.thumb_lead, 4),
        'thumb_dur': round(args.thumb_dur, 4),
        'finger_lead': round(args.finger_lead, 4),
        'finger_dur': round(args.finger_dur, 4),
        'thumb_close_scale': round(args.thumb_close_scale, 4),
        'index_close_scale': round(args.index_close_scale, 4),
        'middle_close_scale': round(args.middle_close_scale, 4),
    }
    if not args.no_write:
        meta.write_text(json.dumps(result, indent=2), encoding='utf-8')
    if not args.quiet:
        print(json.dumps({'json': None if args.no_write else str(meta), **result}, indent=2))
    return result



def _profile_35():
    return dict(radial_offset=-0.205, tangent_offset=-0.125, z_offset=0.225, catch_z=0.875,
                wrist_yaw_factor=0.75, latch_drop_rate=0.75, latch_drop_duration=0.180, latch_min_z=0.64,
                latch_lift_rate=1.2, latch_lift_start=0.06, latch_lift_duration=0.16, latch_max_z=0.92,
                latch_brake_start=0.0, latch_brake_duration=0.08, latch_brake_yaw=0.0,
                latch_brake_roll=30.0, latch_brake_pitch=10.0,
                latch_radial_offset=0.0, latch_tangent_offset=-0.010,
                middle_latch_scale=0.0, index_latch_scale=0.0)


def _profile_35_lift():
    out = _profile_35()
    out.update(latch_lift_rate=1.6, latch_lift_start=0.03, latch_lift_duration=0.22,
               latch_max_z=1.00, latch_drop_rate=0.55, latch_drop_duration=0.14,
               latch_brake_duration=0.10, latch_brake_yaw=22.0, latch_brake_roll=60.0)
    return out


def _profile_45_55():
    return dict(radial_offset=-0.205, tangent_offset=-0.125, z_offset=0.225, catch_z=0.875,
                wrist_yaw_factor=0.75, latch_drop_rate=0.75, latch_drop_duration=0.180, latch_min_z=0.64,
                latch_lift_rate=1.8, latch_lift_start=0.02, latch_lift_duration=0.24, latch_max_z=1.10,
                latch_brake_start=0.0, latch_brake_duration=0.10, latch_brake_yaw=44.0,
                latch_brake_roll=90.0, latch_brake_pitch=0.0,
                latch_radial_offset=0.0, latch_tangent_offset=-0.010,
                middle_latch_scale=0.0, index_latch_scale=0.0)


def _profile_54_60():
    out = _profile_45_55()
    out.update(tangent_offset=-0.105, latch_lift_rate=2.2, latch_lift_duration=0.28,
               latch_max_z=1.12, latch_brake_duration=0.12)
    return out


def _profile_90():
    return dict(radial_offset=-0.205, tangent_offset=-0.095, z_offset=0.245, catch_z=0.875,
                wrist_yaw_factor=1.0, latch_drop_rate=0.35, latch_drop_duration=0.08, latch_min_z=0.64,
                latch_lift_rate=1.0, latch_lift_start=0.06, latch_lift_duration=0.20, latch_max_z=0.96,
                latch_brake_start=0.035, latch_brake_duration=0.08, latch_brake_yaw=16.0,
                latch_brake_roll=65.0, latch_brake_pitch=8.0,
                latch_radial_offset=0.0, latch_tangent_offset=-0.010,
                middle_latch_scale=0.0, index_latch_scale=0.0)


def _profile_65():
    return dict(radial_offset=-0.205, tangent_offset=-0.095, z_offset=0.245, catch_z=0.875,
                wrist_yaw_factor=1.0, latch_drop_rate=0.20, latch_drop_duration=0.06, latch_min_z=0.66,
                latch_lift_rate=1.8, latch_lift_start=0.03, latch_lift_duration=0.24, latch_max_z=1.04,
                latch_brake_start=0.02, latch_brake_duration=0.08, latch_brake_yaw=44.0,
                latch_brake_roll=90.0, latch_brake_pitch=8.0,
                latch_radial_offset=0.0, latch_tangent_offset=-0.010,
                middle_latch_scale=1.0, middle_latch_start=0.02, middle_latch_dur=0.16,
                middle_close_scale=0.6,
                index_latch_scale=0.25, index_latch_start=0.03, index_latch_dur=0.14,
                index_latch_prox=0.6, index_latch_dist=0.6)


def _profile_62():
    out = _profile_65()
    out.update(tangent_offset=-0.090)
    return out


def _profile_75_95():
    return dict(radial_offset=-0.205, tangent_offset=-0.095, z_offset=0.245, catch_z=0.875,
                wrist_yaw_factor=1.0, latch_drop_rate=0.35, latch_drop_duration=0.08, latch_min_z=0.64,
                latch_lift_rate=1.25, latch_lift_start=0.045, latch_lift_duration=0.22, latch_max_z=1.00,
                latch_brake_start=0.035, latch_brake_duration=0.08, latch_brake_yaw=16.0,
                latch_brake_roll=65.0, latch_brake_pitch=8.0,
                latch_radial_offset=0.0, latch_tangent_offset=-0.010,
                middle_latch_scale=0.0, index_latch_scale=0.0)


def _profile_76_strong():
    out = _profile_75_95()
    out.update(latch_lift_rate=1.55, latch_lift_start=0.02, latch_lift_duration=0.26,
               latch_max_z=1.08, latch_brake_start=0.025, latch_brake_duration=0.10)
    return out


def _profile_98_transition():
    return dict(radial_offset=-0.185, tangent_offset=-0.085, z_offset=0.245, catch_z=0.875,
                wrist_yaw_factor=1.2, latch_drop_rate=0.25, latch_drop_duration=0.10, latch_min_z=0.64,
                latch_lift_rate=3.0, latch_lift_start=0.0, latch_lift_duration=0.32, latch_max_z=1.20,
                latch_brake_start=0.0, latch_brake_duration=0.12, latch_brake_yaw=16.0,
                latch_brake_roll=65.0, latch_brake_pitch=8.0,
                latch_radial_offset=0.0, latch_tangent_offset=0.0,
                middle_latch_scale=1.0, middle_latch_start=0.03, middle_latch_dur=0.12,
                index_latch_scale=0.0)


def _profile_98_lift():
    out = _profile_98_transition()
    out.update(latch_lift_rate=3.4, latch_lift_duration=0.36, latch_max_z=1.28,
               latch_brake_duration=0.14)
    return out


def _profile_105():
    return dict(radial_offset=-0.185, tangent_offset=-0.085, z_offset=0.245, catch_z=0.875,
                wrist_yaw_factor=1.2, latch_drop_rate=0.85, latch_drop_duration=0.180, latch_min_z=0.58,
                latch_lift_rate=1.2, latch_lift_start=0.06, latch_lift_duration=0.24, latch_max_z=1.04,
                latch_brake_start=0.0, latch_brake_duration=0.10, latch_brake_yaw=20.0,
                latch_brake_roll=65.0, latch_brake_pitch=8.0,
                latch_radial_offset=0.0, latch_tangent_offset=0.0,
                middle_latch_scale=1.0, middle_latch_start=0.03, middle_latch_dur=0.12,
                index_latch_scale=0.0)


def _profile_115():
    return dict(radial_offset=-0.185, tangent_offset=-0.085, z_offset=0.245, catch_z=0.875,
                wrist_yaw_factor=1.2, latch_drop_rate=0.85, latch_drop_duration=0.180, latch_min_z=0.58,
                latch_lift_rate=2.0, latch_lift_start=0.0, latch_lift_duration=0.28, latch_max_z=1.10,
                latch_brake_start=0.0, latch_brake_duration=0.10, latch_brake_yaw=44.0,
                latch_brake_roll=90.0, latch_brake_pitch=0.0,
                latch_radial_offset=0.0, latch_tangent_offset=0.0,
                middle_latch_scale=1.0, middle_latch_start=0.03, middle_latch_dur=0.12,
                index_latch_scale=0.75, index_latch_start=0.02, index_latch_dur=0.16,
                index_latch_prox=0.9, index_latch_dist=0.9)


def _profile_115_lift():
    out = _profile_115()
    out.update(latch_lift_rate=3.0, latch_lift_duration=0.34, latch_max_z=1.24,
               latch_brake_duration=0.12)
    return out


def _profile_125():
    return dict(radial_offset=-0.195, tangent_offset=-0.110, z_offset=0.245, catch_z=0.875,
                wrist_yaw_factor=1.23, latch_drop_rate=0.25, latch_drop_duration=0.12, latch_min_z=0.62,
                latch_lift_rate=3.0, latch_lift_start=0.0, latch_lift_duration=0.30, latch_max_z=1.22,
                latch_brake_start=0.0, latch_brake_duration=0.10, latch_brake_yaw=44.0,
                latch_brake_roll=90.0, latch_brake_pitch=0.0,
                latch_radial_offset=0.0, latch_tangent_offset=-0.010,
                middle_latch_scale=0.5, middle_latch_start=0.03, middle_latch_dur=0.14,
                index_latch_scale=0.4, index_latch_start=0.02, index_latch_dur=0.16,
                index_latch_prox=0.9, index_latch_dist=0.9)


def _profile_121():
    out = _profile_125()
    out.update(latch_lift_rate=3.5, latch_lift_duration=0.36, latch_max_z=1.30,
               latch_brake_duration=0.12, latch_brake_yaw=50.0, latch_brake_roll=110.0)
    return out


def _profile_126():
    out = _profile_125()
    out.update(latch_lift_rate=4.0, latch_lift_duration=0.40, latch_max_z=1.36,
               latch_brake_duration=0.12, latch_brake_yaw=44.0, latch_brake_roll=90.0)
    return out


def _profile_145():
    return dict(radial_offset=-0.205, tangent_offset=-0.135, z_offset=0.245, catch_z=0.875,
                wrist_yaw_factor=1.25, latch_drop_rate=0.0, latch_drop_duration=0.180, latch_min_z=0.64,
                latch_lift_rate=1.4, latch_lift_start=0.02, latch_lift_duration=0.22, latch_max_z=0.96,
                latch_brake_start=0.0, latch_brake_duration=0.08, latch_brake_yaw=44.0,
                latch_brake_roll=90.0, latch_brake_pitch=0.0,
                latch_radial_offset=0.0, latch_tangent_offset=-0.010,
                middle_latch_scale=0.0, index_latch_scale=0.0)


def _profile_74_bridge():
    out = _profile_65()
    out.update(tangent_offset=-0.090,
               middle_close_scale=0.0, middle_latch_scale=0.0, index_latch_scale=0.0,
               latch_drop_rate=0.35, latch_drop_duration=0.08, latch_min_z=0.64,
               latch_lift_rate=1.45, latch_lift_start=0.02, latch_lift_duration=0.28, latch_max_z=1.08,
               latch_brake_start=0.035, latch_brake_duration=0.10, latch_brake_yaw=16.0,
               latch_brake_roll=65.0, latch_brake_pitch=8.0,
               latch_tangent_offset=-0.010)
    return out


def _profile_88_lift():
    out = _profile_90()
    out.update(latch_lift_rate=1.45, latch_lift_start=0.030,
               latch_lift_duration=0.24, latch_max_z=1.04)
    return out


def _profile_116_bridge():
    out = _profile_115_lift()
    out.update(tangent_offset=-0.090, latch_tangent_offset=0.0,
               latch_lift_rate=3.3, latch_lift_duration=0.38, latch_max_z=1.30)
    return out


def _blend_profile(a, b, u):
    out = {}
    for k in set(a) | set(b):
        av = a.get(k, 0.0)
        bv = b.get(k, av)
        out[k] = av * (1.0 - u) + bv * u
    return out


def front110_profile(angle):
    # Conservative nearest-anchor controller first: do not dilute anchors that are
    # already known to produce strict physical grasps. Transitional bands can be
    # tuned after dense validation.
    if angle < 40.0:
        return _profile_35_lift()
    if angle < 54.0:
        return _profile_45_55()
    if angle < 60.0:
        return _profile_54_60()
    if angle < 64.6:
        return _profile_62()
    if angle < 70.0:
        return _profile_65()
    if angle < 73.5:
        return _profile_65()
    if angle < 75.0:
        return _profile_74_bridge()
    if angle < 87.0:
        return _profile_76_strong()
    if angle < 89.0:
        return _profile_88_lift()
    if angle < 91.0:
        return _profile_90()
    if angle < 94.5:
        return _profile_75_95()
    if angle < 100.0:
        return _profile_98_lift()
    if angle < 103.0:
        return _profile_98_transition()
    if angle < 110.0:
        return _profile_98_transition()
    if angle < 115.0:
        return _profile_115()
    if angle < 115.5:
        return _profile_115_lift()
    if angle < 117.0:
        return _profile_116_bridge()
    if angle < 123.0:
        return _profile_121()
    if angle < 130.0:
        return _profile_126()
    return _profile_145()


def apply_front110_auto_args(args):
    if not getattr(args, 'front110_auto', False):
        return args
    prof = front110_profile(float(args.angle))
    for k, v in prof.items():
        if hasattr(args, k):
            setattr(args, k, v)
    return args

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=69940)
    ap.add_argument('--angle', type=float, default=90.0)
    ap.add_argument('--yaw', type=float, default=None)
    ap.add_argument('--seconds', type=float, default=1.1)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--no-write', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--release-time', type=float, default=0.35)
    ap.add_argument('--camera', choices=['front_arc', 'top_oblique'], default='front_arc')
    ap.add_argument('--out-dir', default=str(PROJECT / 'outputs/stage4c_front_angle_yaw'))
    ap.add_argument('--radial-offset', type=float, default=-0.205)
    ap.add_argument('--tangent-offset', type=float, default=-0.095)
    ap.add_argument('--z-offset', type=float, default=0.245)
    ap.add_argument('--catch-z', type=float, default=0.875)
    ap.add_argument('--wrist-yaw-factor', type=float, default=1.0)
    ap.add_argument('--wrist-yaw', type=float, default=0.0)
    ap.add_argument('--wrist-roll-factor', type=float, default=0.0)
    ap.add_argument('--wrist-roll', type=float, default=0.0)
    ap.add_argument('--wrist-pitch-factor', type=float, default=0.0)
    ap.add_argument('--wrist-pitch', type=float, default=0.0)
    ap.add_argument('--ready-tangent', type=float, default=0.120)
    ap.add_argument('--move-start', type=float, default=0.120)
    ap.add_argument('--move-dur', type=float, default=0.130)
    ap.add_argument('--insert-tangent', type=float, default=0.040)
    ap.add_argument('--latch-drop-rate', type=float, default=0.850)
    ap.add_argument('--latch-drop-duration', type=float, default=0.180)
    ap.add_argument('--latch-min-z', type=float, default=0.580)
    ap.add_argument('--latch-lift-rate', type=float, default=0.0)
    ap.add_argument('--latch-lift-start', type=float, default=0.160)
    ap.add_argument('--latch-lift-duration', type=float, default=0.180)
    ap.add_argument('--latch-max-z', type=float, default=0.920)
    ap.add_argument('--latch-radial-offset', type=float, default=0.0)
    ap.add_argument('--latch-tangent-offset', type=float, default=-0.010)
    ap.add_argument('--latch-xy-follow', type=float, default=0.0)
    ap.add_argument('--latch-xy-follow-start', type=float, default=0.0)
    ap.add_argument('--latch-xy-follow-dur', type=float, default=0.080)
    ap.add_argument('--middle-latch-scale', type=float, default=0.0)
    ap.add_argument('--middle-latch-start', type=float, default=0.030)
    ap.add_argument('--middle-latch-dur', type=float, default=0.120)
    ap.add_argument('--index-latch-scale', type=float, default=0.0)
    ap.add_argument('--index-latch-start', type=float, default=0.030)
    ap.add_argument('--index-latch-dur', type=float, default=0.120)
    ap.add_argument('--index-latch-prox', type=float, default=1.05)
    ap.add_argument('--index-latch-dist', type=float, default=1.05)
    ap.add_argument('--latch-brake-start', type=float, default=0.0)
    ap.add_argument('--latch-brake-duration', type=float, default=0.160)
    ap.add_argument('--latch-brake-yaw', type=float, default=0.0)
    ap.add_argument('--latch-brake-roll', type=float, default=0.0)
    ap.add_argument('--latch-brake-pitch', type=float, default=0.0)
    ap.add_argument('--thumb-lead', type=float, default=0.160)
    ap.add_argument('--thumb-dur', type=float, default=0.240)
    ap.add_argument('--finger-lead', type=float, default=0.110)
    ap.add_argument('--finger-dur', type=float, default=0.180)
    ap.add_argument('--thumb-close-scale', type=float, default=1.0)
    ap.add_argument('--index-close-scale', type=float, default=1.0)
    ap.add_argument('--middle-close-scale', type=float, default=0.0)
    ap.add_argument('--front110-auto', action='store_true')
    args = ap.parse_args()
    apply_front110_auto_args(args)
    run_episode(args)


if __name__ == '__main__':
    main()

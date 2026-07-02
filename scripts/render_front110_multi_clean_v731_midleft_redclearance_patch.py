#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from front110_core.backends.mujoco_backend import extract_low_dim_observation
from front110_core.dataset_adapter import EpisodeRecorder, build_manifest, contact_flags_array, validate_npz, write_manifest
from front110_core.planner import UnifiedAction

PROJECT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("multi700", Path(__file__).with_name("render_front110_multi_clean_v700.py"))
multi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(multi)
v700 = multi.v700
VISUAL_ROBOT_XML = PROJECT / "assets/full_robot_urdf_mirror/franka_brainco_revo2_right_aligned_visual_mjcf_v728.xml"
v700.ROBOT_XML = VISUAL_ROBOT_XML
OUT = PROJECT / "outputs/front110_v730_front110_dense_height_patch"

GRIP_KEYS = [
    "catch_z", "ready_tangent", "move_start", "move_dur", "insert_tangent",
    "latch_drop_rate", "latch_drop_duration", "latch_min_z",
    "latch_lift_rate", "latch_lift_start", "latch_lift_duration", "latch_max_z",
    "latch_radial_offset", "latch_tangent_offset",
    "latch_xy_follow", "latch_xy_follow_start", "latch_xy_follow_dur",
    "middle_latch_scale", "middle_latch_start", "middle_latch_dur",
    "index_latch_scale", "index_latch_start", "index_latch_dur",
    "index_latch_prox", "index_latch_dist",
    "thumb_lead", "thumb_dur", "finger_lead", "finger_dur",
    "thumb_close_scale", "index_close_scale", "middle_close_scale",
]


def grip_ref(args, tool_origin, angle, tau, latched=False, latch_age=0.0):
    kw = {k: getattr(args, k) for k in GRIP_KEYS}
    center, hand, move_a, close_a = v700.stage4_grip_reference(
        tool_origin, angle, tau, latched, latch_age, **kw
    )
    # v700's reference was calibrated on the nominal 0.50 m front arc and only
    # used tool_origin during the latch-follow phase. For the real task the rod
    # can drop with +/-3-5 cm xy jitter, so the physical pinch window must follow
    # the actual falling handle path, not the nominal sector point.
    th = math.radians(angle)
    nominal_xy = np.array([0.50 * math.cos(th), 0.50 * math.sin(th)], dtype=float)
    actual_xy = np.asarray(tool_origin[:2], dtype=float)
    center[:2] = center[:2] + (actual_xy - nominal_xy)
    return center, hand, move_a, close_a


def realmesh_offsets_for_angle(args, angle_deg):
    """Sector calibration for the real Revo2 visual/contact mesh.

    v700/v699 offsets were tuned on the old proxy fingertip geometry. The real
    mesh needs a slightly different tangent offset across the front arc. Keep the
    correction smooth so continuous random angles do not jump between poses.
    """
    anchors = [
        (35.0, -0.20, 0.030, 0.140),
        (55.0, -0.20, 0.050, 0.140),
        (75.0, -0.20, 0.100, 0.140),
        (90.0, -0.20, 0.090, 0.140),
        (116.5, -0.20, 0.070, 0.100),
        # The far-left edge needs a slightly more conservative tangent approach
        # after xy jitter-follow is enabled; otherwise the red functional segment
        # can sweep the fingertips before the handle settles into the clamp.
        (143.0, -0.20, 0.070, 0.140),
        (145.0, -0.20, 0.070, 0.140),
    ]
    a = float(angle_deg)
    if a <= anchors[0][0]:
        radial, tangent, z = anchors[0][1:]
    elif a >= anchors[-1][0]:
        radial, tangent, z = anchors[-1][1:]
    else:
        for lo, hi in zip(anchors[:-1], anchors[1:]):
            if lo[0] <= a <= hi[0]:
                u = (a - lo[0]) / max(1e-9, hi[0] - lo[0])
                radial = lo[1] * (1.0 - u) + hi[1] * u
                tangent = lo[2] * (1.0 - u) + hi[2] * u
                z = lo[3] * (1.0 - u) + hi[3] * u
                break
    radial += args.wrist_radial_offset_delta
    tangent += args.wrist_tangent_offset_delta
    z += args.wrist_z_offset_delta
    return radial, tangent, z


def smooth_trapezoid(x, start, full_start, full_end, end):
    x = float(x)
    if x <= start or x >= end:
        return 0.0
    if full_start <= x <= full_end:
        return 1.0
    if x < full_start:
        u = (x - start) / max(1e-9, full_start - start)
    else:
        u = (end - x) / max(1e-9, end - full_end)
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def dense_hole_sector_bump(angle_deg):
    # v728 dense 1-degree validation exposed a narrow 119-129 deg red-contact
    # hole. Local tests showed that a slightly higher catch window clears the functional segment without touching the
    # stable low-angle and far-edge sectors.
    return smooth_trapezoid(angle_deg, 116.0, 118.0, 132.0, 135.0)


def midleft_red_clearance_bump(angle_deg):
    # v731: a bad jitter sample at nominal 130 deg can be observed as ~134 deg,
    # where v730 was already tapering the dense-hole height lift. Keep this
    # narrow and height-only so the established clamp geometry and timing stay
    # untouched outside the 126-139 deg mid-left band.
    return smooth_trapezoid(angle_deg, 126.0, 129.0, 136.0, 139.0)


def apply_post_latch_fullfinger(args, hand_des, latch_age):
    if latch_age < 0.0 or args.post_latch_fullfinger_scale <= 0.0:
        return hand_des
    a = args.post_latch_fullfinger_scale * v700.smoothstep(
        (latch_age - args.post_latch_fullfinger_start) / max(1e-6, args.post_latch_fullfinger_dur)
    )
    a = min(1.0, max(0.0, a))
    if a <= 0.0:
        return hand_des
    target = hand_des.copy()
    # Keep within the converted Revo2 joint ranges. This is a physical grasp
    # command, not a latch constraint: contacts still have to hold the tool.
    target[:3] = np.minimum(v700.HAND_UPPER[:3], np.array([1.10, 0.78, 0.66], dtype=float))
    target[3:5] = np.minimum(v700.HAND_UPPER[3:5], np.array([1.12, 1.08], dtype=float))
    target[5:7] = np.minimum(v700.HAND_UPPER[5:7], np.array([1.05, 1.02], dtype=float))
    # Ring/pinky are kept mostly out by default. They looked helpful with the proxy hand,
    # but on the real Revo2 mesh they can sweep into the red functional region.
    rp = max(0.0, min(1.0, args.post_latch_ring_pinky_scale))
    target[7:9] = hand_des[7:9] * (1.0 - rp) + np.minimum(v700.HAND_UPPER[7:9], np.array([0.92, 0.88], dtype=float)) * rp
    target[9:11] = hand_des[9:11] * (1.0 - rp) + np.minimum(v700.HAND_UPPER[9:11], np.array([0.82, 0.78], dtype=float)) * rp
    return hand_des * (1.0 - a) + target * a


def apply_precontact_finger_closure(args, hand_des, tau_eff, angle_deg):
    """Begin shaping the physical pinch after release but before first contact.

    This preserves the demo rule: the robot does not move to the specific target
    before the drop is visible. The Revo2 fingers are slow enough that waiting for
    contact leaves too little time, so after visible release we start forming the
    thumb-vs-index/middle clamp while the FR3 moves to the intercept window.
    """
    if args.precontact_finger_scale <= 0.0:
        return hand_des
    a = args.precontact_finger_scale * v700.smoothstep(
        (tau_eff - args.precontact_finger_start) / max(1e-6, args.precontact_finger_dur)
    )
    a = max(0.0, min(1.0, a))
    if a <= 0.0:
        return hand_des
    edge = max(0.0, min(1.0, (abs(float(angle_deg) - 90.0) - 34.0) / 21.0))
    target = hand_des.copy()
    target[:3] = np.minimum(v700.HAND_UPPER[:3], np.array([1.02, 0.70, 0.54], dtype=float))
    target[3:5] = np.minimum(v700.HAND_UPPER[3:5], np.array([1.10, 1.04], dtype=float))
    middle_scale = args.precontact_middle_scale * (1.0 - 0.20 * edge)
    target[5:7] = hand_des[5:7] * (1.0 - middle_scale) + np.minimum(
        v700.HAND_UPPER[5:7], np.array([0.98, 0.92], dtype=float)
    ) * middle_scale
    rp = args.precontact_ring_pinky_scale * (1.0 - 0.75 * edge)
    if rp > 0.0:
        target[7:9] = hand_des[7:9] * (1.0 - rp) + np.minimum(v700.HAND_UPPER[7:9], np.array([0.38, 0.34], dtype=float)) * rp
        target[9:11] = hand_des[9:11] * (1.0 - rp) + np.minimum(v700.HAND_UPPER[9:11], np.array([0.28, 0.24], dtype=float)) * rp
    return hand_des * (1.0 - a) + target * a


def sample_demo(seed, num_tools):
    rng = random.Random(seed)
    angles = sorted(round(rng.uniform(35.0, 145.0), 1) for _ in range(num_tools))
    if num_tools >= 6:
        angles = [38.5, 56.7, 74.0, 88.0, 116.5, 143.0][:num_tools]
    yaws = [round(rng.uniform(-10.0, 10.0), 1) for _ in angles]
    order = list(range(len(angles)))
    rng.shuffle(order)
    intervals = [round(rng.uniform(0.28, 0.58), 2) for _ in order]
    return angles, yaws, order, intervals


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def lookat_xyaxes(pos, target, up=(0.0, 0.0, 1.0)):
    pos = np.asarray(pos, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)
    direction = target - pos
    direction /= max(1e-9, np.linalg.norm(direction))
    z_axis = -direction
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-9:
        x_axis = np.array([1.0, 0.0, 0.0])
    else:
        x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(1e-9, np.linalg.norm(y_axis))
    vals = list(x_axis) + list(y_axis)
    return " ".join(f"{v:.6f}" for v in vals)


def observed_front_angle_deg(pos):
    angle = math.degrees(math.atan2(float(pos[1]), float(pos[0])))
    return max(35.0, min(145.0, angle))


def edge_release_height(args, angle_deg):
    """Give only the hard edge sectors more falling time.

    A global height increase makes mid-arc impacts too fast and causes slip.
    Edge-only height keeps the proven v723 mid-arc timing while giving 35/45/145
    enough post-release travel for the FR3 to visibly react from the shared ready
    stance.
    """
    a = float(angle_deg)
    low = max(0.0, min(1.0, (args.edge_release_low_end_deg - a) / max(1e-6, args.edge_release_width_deg)))
    high = max(0.0, min(1.0, (a - args.edge_release_high_start_deg) / max(1e-6, args.edge_release_width_deg)))
    edge = max(low, high)
    edge = edge * edge * (3.0 - 2.0 * edge)
    return float(args.release_height + args.edge_release_height_boost * edge)


def add_visual_ghosts_and_cameras(scene_xml, angles, yaws, show_ghosts=False):
    root = ET.parse(scene_xml).getroot()
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)
    if not any(t.get("name") == "soft_lab_sky" for t in asset.findall("texture")):
        ET.SubElement(asset, "texture", {"name": "soft_lab_sky", "type": "skybox", "builtin": "flat", "rgb1": "0.78 0.82 0.86", "rgb2": "0.78 0.82 0.86", "width": "64", "height": "64"})
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", "2048")
    global_visual.set("offheight", "1152")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("ambient", "0.55 0.55 0.55")
    headlight.set("diffuse", "0.78 0.78 0.76")
    headlight.set("specular", "0.06 0.06 0.06")
    rgba = visual.find("rgba")
    if rgba is None:
        rgba = ET.SubElement(visual, "rgba")
    rgba.set("haze", "0.78 0.82 0.86 1")
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("scene has no worldbody")
    cameras = {
        "full_body_three_quarter": ((1.35, -2.15, 1.42), (0.02, 0.42, 0.88), "46"),
        "full_body_front_arc": ((0.00, -2.55, 1.46), (0.00, 0.44, 0.88), "48"),
        "fall_path_high_arc": ((0.00, -1.58, 2.38), (0.00, 0.42, 0.94), "54"),
        "side_reach_profile": ((2.05, -0.58, 1.36), (0.02, 0.43, 0.86), "43"),
        "hand_window_close": ((0.70, -0.92, 1.10), (0.02, 0.50, 0.84), "27"),
    }
    existing = {c.get("name") for c in world.findall("camera")}
    for name, (pos, target, fovy) in cameras.items():
        if name not in existing:
            ET.SubElement(world, "camera", {
                "name": name,
                "pos": " ".join(f"{v:.6f}" for v in pos),
                "xyaxes": lookat_xyaxes(pos, target),
                "fovy": fovy,
            })
    ET.SubElement(world, "light", {"name": "front_area_key", "pos": "0 -2.4 3.2", "dir": "0 0.62 -1", "diffuse": "0.72 0.72 0.70", "specular": "0.05 0.05 0.05"})
    ET.SubElement(world, "light", {"name": "left_fill_soft", "pos": "-2.0 -0.4 2.0", "dir": "0.8 0.1 -1", "diffuse": "0.38 0.40 0.42", "specular": "0.02 0.02 0.02"})
    ET.SubElement(world, "light", {"name": "right_fill_soft", "pos": "2.0 -0.8 2.2", "dir": "-0.8 0.2 -1", "diffuse": "0.42 0.42 0.42", "specular": "0.02 0.02 0.02"})
    if not show_ghosts:
        ET.ElementTree(root).write(scene_xml, encoding="unicode")
        return
    for i, (angle, yaw) in enumerate(zip(angles, yaws)):
        pos = multi.tool_pos(angle)
        quat = v700.yaw_quat(math.radians(yaw))
        body = ET.SubElement(world, "body", {
            "name": f"ghost_tool{i}",
            "pos": f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}",
            "quat": " ".join(f"{q:.6f}" for q in quat),
        })
        ET.SubElement(body, "geom", {
            "name": f"ghost_handle_positive_{i}",
            "type": "capsule",
            "fromto": "0 0 -0.11 0 0 0.00",
            "size": "0.015",
            "rgba": "0.05 0.85 0.30 0.42",
            "contype": "0",
            "conaffinity": "0",
        })
        ET.SubElement(body, "geom", {
            "name": f"ghost_functional_negative_{i}",
            "type": "capsule",
            "fromto": "0 0 0.00 0 0 0.22",
            "size": "0.015",
            "rgba": "0.85 0.05 0.04 0.38",
            "contype": "0",
            "conaffinity": "0",
        })
    ET.ElementTree(root).write(scene_xml, encoding="unicode")



def polish_robot_appearance(model):
    """Render the actual FR3/Revo2 mesh shape while keeping contact helper geoms unobtrusive."""
    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    franka_mesh_rgba = {
        "link0": np.array([0.17, 0.18, 0.19, 1.0]),
        "link1": np.array([0.72, 0.74, 0.74, 1.0]),
        "link2": np.array([0.23, 0.24, 0.25, 1.0]),
        "link3": np.array([0.76, 0.77, 0.76, 1.0]),
        "link4": np.array([0.25, 0.26, 0.27, 1.0]),
        "link5": np.array([0.74, 0.75, 0.74, 1.0]),
        "link6": np.array([0.25, 0.25, 0.26, 1.0]),
        "link7": np.array([0.70, 0.71, 0.70, 1.0]),
    }
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        bname = model.body(model.geom_bodyid[gid]).name
        if gname == "floor":
            model.geom_rgba[gid] = np.array([0.62, 0.64, 0.65, 1.0])
            continue
        if gname.startswith(("handle_positive", "functional_negative", "ghost_")):
            continue
        if gname.startswith("stage4_") or "marker" in gname or "proxy" in gname:
            model.geom_rgba[gid] = np.array([1.0, 1.0, 1.0, 0.0])
            continue
        mesh_name = ""
        if int(model.geom_type[gid]) == mesh_type and int(model.geom_dataid[gid]) >= 0:
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[gid])) or ""
        is_franka = bname.startswith("panda") or mesh_name.startswith("link")
        is_revo = bname.startswith("revo2") or bname.startswith("right_") or "thumb" in bname or "index" in bname or "middle" in bname or "ring" in bname or "pinky" in bname
        if is_franka and int(model.geom_type[gid]) == mesh_type:
            model.geom_rgba[gid] = franka_mesh_rgba.get(mesh_name, np.array([0.62, 0.63, 0.62, 1.0]))
        elif is_revo and int(model.geom_type[gid]) == mesh_type:
            if "touch" in bname or "tip" in bname or "touch" in gname:
                model.geom_rgba[gid] = np.array([0.025, 0.025, 0.030, 1.0])
            elif "base" in bname:
                model.geom_rgba[gid] = np.array([0.035, 0.075, 0.150, 1.0])
            else:
                model.geom_rgba[gid] = np.array([0.055, 0.145, 0.300, 1.0])
        elif is_revo and int(model.geom_type[gid]) != mesh_type:
            # Keep Revo2 collision helpers physically active but hidden in video.
            model.geom_rgba[gid] = np.array([0.02, 0.02, 0.025, 0.0])
        elif is_franka and int(model.geom_type[gid]) != mesh_type:
            model.geom_rgba[gid] = np.array([0.18, 0.18, 0.18, 0.0])
        if is_revo:
            # EVA/foam-like finger pads: high sliding friction, modest torsional/rolling friction.
            # This is still ordinary MuJoCo contact, not a weld/latch/sticky constraint.
            model.geom_friction[gid] = np.array([3.80, 0.12, 0.012])

def run_demo(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.angles_deg:
        angles = parse_float_list(args.angles_deg)
        if args.yaws_deg:
            yaws = parse_float_list(args.yaws_deg)
            if len(yaws) != len(angles):
                raise ValueError("--yaws-deg must have the same length as --angles-deg")
        else:
            yaws = [0.0 for _ in angles]
        rng = random.Random(args.seed)
        order = list(range(len(angles)))
        if not args.keep_order:
            rng.shuffle(order)
        intervals = [round(rng.uniform(0.28, 0.58), 2) for _ in order]
    else:
        angles, yaws, order, intervals = sample_demo(args.seed, args.num_tools)
    scene_xml = out / "front110_v730_front110_dense_height_patch_scene.xml"
    initials = multi.build_multi_scene(scene_xml, angles, yaws)
    # For observe-then-grasp, the arm starts from a common ready pose after visible release.
    # Apply small per-drop xy jitter inside the requested +/-3-5 cm range; the teacher can know
    # which tool releases, but the visible motion still starts only after release.
    jitter_rng = random.Random(args.seed + 707)
    drop_xy_offsets = []
    release_heights = []
    jittered = []
    for pos, quat, angle in zip((p for p, _ in initials), (q for _, q in initials), angles):
        dx = jitter_rng.uniform(-args.drop_xy_jitter, args.drop_xy_jitter) if args.drop_xy_jitter > 0 else 0.0
        dy = jitter_rng.uniform(-args.drop_xy_jitter, args.drop_xy_jitter) if args.drop_xy_jitter > 0 else 0.0
        rh = edge_release_height(args, angle)
        drop_xy_offsets.append([round(dx, 4), round(dy, 4)])
        release_heights.append(round(rh, 4))
        jittered.append((np.array([pos[0] + dx, pos[1] + dy, rh], dtype=float), quat))
    initials = jittered
    add_visual_ghosts_and_cameras(scene_xml, angles, yaws, show_ghosts=args.show_ghosts)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    polish_robot_appearance(model)
    data = mujoco.MjData(model)
    record_video = args.fps > 0
    renderer = mujoco.Renderer(model, height=args.height, width=args.width) if record_video else None

    wrist_bid = model.body("panda_link7").id
    tool_bids = [model.body(f"tool{i}").id for i in range(len(angles))]
    tool_geoms = [[model.geom(f"handle_positive_{i}").id, model.geom(f"functional_negative_{i}").id] for i in range(len(angles))]
    ghost_geoms = []
    if args.show_ghosts:
        ghost_geoms = [[model.geom(f"ghost_handle_positive_{i}").id, model.geom(f"ghost_functional_negative_{i}").id] for i in range(len(angles))]
    tip_marker_ids = {
        "thumb": model.geom("stage4_thumb_tip_marker").id,
        "index": model.geom("stage4_index_tip_marker").id,
        "middle": model.geom("stage4_middle_tip_marker").id,
    }
    qvadr = [model.jnt_dofadr[model.joint(f"tool_free_{i}").id] for i in range(len(angles))]
    qadr = [model.jnt_qposadr[model.joint(f"tool_free_{i}").id] for i in range(len(angles))]

    q_arm_cmd = v700.Q_READY.copy()
    q_hand_cmd = v700.HAND_OPEN.copy()
    data.qpos[:7] = q_arm_cmd
    data.qpos[7:18] = q_hand_cmd
    data.ctrl[:7] = q_arm_cmd
    data.ctrl[7:18] = q_hand_cmd
    hidden = set()
    for i, (pos, quat) in enumerate(initials):
        multi.set_tool_state(model, data, i, pos, quat)
        multi.set_tool_visual_collision(model, tool_geoms[i], "waiting")
    mujoco.mj_forward(model, data)
    wrist_ready_mat_fixed = data.xmat[wrist_bid].copy().reshape(3, 3)

    def fixed_wrist_xmat_fn(grip_args, angle):
        delta = angle - 90.0
        def fn(latch_age=0.0):
            brake_a = v700.smoothstep((latch_age - grip_args.latch_brake_start) / max(1e-6, grip_args.latch_brake_duration))
            return (
                v700.rotz(math.radians(grip_args.wrist_yaw + grip_args.wrist_yaw_factor * delta + grip_args.latch_brake_yaw * brake_a))
                @ v700.rotx(math.radians(grip_args.wrist_roll + grip_args.wrist_roll_factor * delta + grip_args.latch_brake_roll * brake_a))
                @ v700.roty(math.radians(grip_args.wrist_pitch + grip_args.wrist_pitch_factor * delta + grip_args.latch_brake_pitch * brake_a))
                @ wrist_ready_mat_fixed
            ).reshape(9)
        return fn

    def common_ready_arm_target():
        # A target-independent front-center intercept stance. The robot may wait here
        # before release, then visibly choose the specific catch angle after the drop starts.
        ready_angle = 90.0
        ready_args = multi.base_args(ready_angle, out)
        ready_args.catch_z += args.catch_z_offset
        ready_args.radial_offset = args.realmesh_radial_offset
        ready_args.tangent_offset = args.realmesh_tangent_offset
        ready_args.z_offset = args.realmesh_z_offset
        ready_xmat = fixed_wrist_xmat_fn(ready_args, ready_angle)
        ready_wrist_to_pinch = v700.wrist_offset_for_angle(
            ready_angle, ready_args.radial_offset, ready_args.tangent_offset, ready_args.z_offset
        )
        ready_origin = np.array([0.0, 0.50, args.release_height], dtype=float)
        ready_center, _, _, _ = grip_ref(ready_args, ready_origin, ready_angle, -ready_args.release_time)
        return v700.solve_arm_ik(
            model, data, v700.Q_READY.copy(), wrist_bid, ready_center + ready_wrist_to_pinch, ready_xmat(0.0), max_iter=128
        )

    common_ready_q = v700.Q_READY.copy()

    dt = model.opt.timestep
    spf = max(1, int(round((1.0 / args.fps) / dt))) if args.fps > 0 else 1
    frame_tick = 0
    camera_names = ["full_body_three_quarter", "full_body_front_arc", "fall_path_high_arc", "side_reach_profile", "hand_window_close"]
    frames = {name: [] for name in camera_names}
    recorder = EpisodeRecorder(len(angles)) if args.dataset_out_dir else None
    dataset_stride = max(1, int(args.dataset_stride))
    dataset_tick = 0
    dataset_phase = "init"
    dataset_cur = -1
    dataset_contact = contact_flags_array(False, False, False)

    def set_dataset_context(phase, cur=-1, contact_flags=None):
        nonlocal dataset_phase, dataset_cur, dataset_contact
        dataset_phase = str(phase)
        dataset_cur = int(cur) if cur is not None else -1
        if contact_flags is None:
            dataset_contact = contact_flags_array(False, False, False)
        else:
            dataset_contact = np.asarray(contact_flags, dtype=np.bool_)

    def record_dataset_if_due():
        nonlocal dataset_tick
        if recorder is None:
            return
        if dataset_tick % dataset_stride == 0:
            obs = extract_low_dim_observation(model, data, wrist_bid, tool_bids, qadr, qvadr)
            action = UnifiedAction.from_internal_hand(q_arm_cmd, q_hand_cmd, phase=dataset_phase)
            recorder.append(
                obs,
                action,
                phase=dataset_phase,
                active_tool_index=dataset_cur,
                contact_flags=dataset_contact,
                sim_time_s=float(data.time),
            )
        dataset_tick += 1

    def freeze_tools(cur=None, current_mode="waiting"):
        for i, (pos, quat) in enumerate(initials):
            ghost_alpha = None
            if i in hidden:
                multi.set_tool_visual_collision(model, tool_geoms[i], "hidden")
                multi.set_tool_state(model, data, i, np.array([3.0 + i, 3.0, -1.0]), np.array([1, 0, 0, 0], dtype=float))
            elif cur is not None and i == cur:
                multi.set_tool_visual_collision(model, tool_geoms[i], current_mode)
                if current_mode != "active":
                    multi.set_tool_state(model, data, i, pos, quat)
            else:
                if cur is None:
                    multi.set_tool_visual_collision(model, tool_geoms[i], "waiting")
                    multi.set_tool_state(model, data, i, pos, quat)
                else:
                    multi.set_tool_visual_collision(model, tool_geoms[i], "hidden")
                    multi.set_tool_state(model, data, i, np.array([3.0 + i, 3.0, -1.0]), np.array([1, 0, 0, 0], dtype=float))
                    ghost_alpha = (0.42, 0.38)
            if ghost_alpha is None:
                ghost_alpha = (0.0, 0.0)
            if args.show_ghosts:
                for gi, gid in enumerate(ghost_geoms[i]):
                    model.geom_rgba[gid][3] = ghost_alpha[gi]

    def render_if_due():
        nonlocal frame_tick
        record_dataset_if_due()
        if not record_video:
            frame_tick += 1
            return
        if frame_tick % spf == 0:
            for name in camera_names:
                renderer.update_scene(data, camera=model.camera(name).id)
                frames[name].append(renderer.render())
        frame_tick += 1

    def step_ready(seconds, cur=None):
        nonlocal q_arm_cmd, q_hand_cmd
        steps = int(seconds / dt)
        set_dataset_context("ready", cur if cur is not None else -1)
        for _ in range(steps):
            freeze_tools(cur, "current_wait" if cur is not None else "waiting")
            q_arm_cmd = v700.limit_step(q_arm_cmd, common_ready_q, v700.FR3_VEL_LIMIT, dt)
            q_hand_cmd = v700.limit_step(q_hand_cmd, v700.HAND_OPEN, v700.HAND_VEL_LIMIT, dt)
            data.ctrl[:7] = q_arm_cmd
            data.ctrl[7:18] = q_hand_cmd
            data.qpos[:7] = q_arm_cmd
            data.qvel[:7] = 0.0
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            render_if_due()

    def prep_to_initial(cur, grip_args, make_xmat, wrist_to_pinch):
        # Believable demo rule: before release, the robot may return to a shared ready pose
        # but must not move toward the known target-specific catch window.
        nonlocal q_arm_cmd, q_hand_cmd
        steps = int(args.prep_seconds / dt)
        set_dataset_context("prep_common_ready", cur)
        for _ in range(steps):
            freeze_tools(cur, "current_wait")
            q_arm_cmd = v700.limit_step(q_arm_cmd, common_ready_q, v700.FR3_VEL_LIMIT, dt)
            q_hand_cmd = v700.limit_step(q_hand_cmd, v700.HAND_OPEN, v700.HAND_VEL_LIMIT, dt)
            data.ctrl[:7] = q_arm_cmd
            data.ctrl[7:18] = q_hand_cmd
            data.qpos[:7] = q_arm_cmd
            data.qpos[7:18] = q_hand_cmd
            data.qvel[:18] = 0.0
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            render_if_due()

    def catch_one(cur):
        nonlocal q_arm_cmd, q_hand_cmd
        control_angle = observed_front_angle_deg(initials[cur][0])
        grip_args = multi.base_args(control_angle, out)
        grip_args.control_angle_deg = control_angle
        grip_args.catch_z += args.catch_z_offset
        # Real Revo2 mesh calibration: old IsaacGym/proxy offset put the actual fingertips
        # about 18-22 cm away from the handle. Use absolute calibrated clamp geometry here.
        grip_args.radial_offset, grip_args.tangent_offset, grip_args.z_offset = realmesh_offsets_for_angle(args, control_angle)
        grip_args.wrist_yaw += args.wrist_yaw_delta
        grip_args.wrist_roll += args.wrist_roll_delta
        grip_args.wrist_pitch += args.wrist_pitch_delta
        dense_bump = dense_hole_sector_bump(control_angle)
        grip_args.catch_z += 0.05 * dense_bump
        grip_args.catch_z += 0.03 * midleft_red_clearance_bump(control_angle)
        low_edge = max(0.0, min(1.0, (args.low_edge_pitch_end_deg - control_angle) / max(1e-6, args.low_edge_pitch_width_deg)))
        low_edge = low_edge * low_edge * (3.0 - 2.0 * low_edge)
        grip_args.wrist_pitch += args.low_edge_wrist_pitch_boost * low_edge
        if args.middle_close_scale >= 0.0:
            grip_args.middle_close_scale = args.middle_close_scale
        if args.middle_latch_scale >= 0.0:
            grip_args.middle_latch_scale = args.middle_latch_scale
        if args.index_latch_scale >= 0.0:
            grip_args.index_latch_scale = args.index_latch_scale
        if args.latch_xy_follow >= 0.0:
            grip_args.latch_xy_follow = args.latch_xy_follow
        grip_args.latch_tangent_offset += args.latch_tangent_offset_delta
        grip_args.latch_radial_offset += args.latch_radial_offset_delta
        if args.disable_latch_lift_brake:
            grip_args.latch_lift_rate = 0.0
            grip_args.latch_lift_start = 999.0
            grip_args.latch_lift_duration = 1.0
            grip_args.latch_drop_rate = 0.0
            grip_args.latch_brake_yaw = 0.0
            grip_args.latch_brake_roll = 0.0
            grip_args.latch_brake_pitch = 0.0
            grip_args.latch_brake_start = 999.0
            grip_args.latch_brake_duration = 1.0
        grip_args.move_start = 0.0
        grip_args.move_dur = max(grip_args.move_dur, 0.18)
        grip_args.release_time = args.release_delay
        make_xmat = fixed_wrist_xmat_fn(grip_args, control_angle)
        wrist_to_pinch = v700.wrist_offset_for_angle(
            control_angle, grip_args.radial_offset, grip_args.tangent_offset, grip_args.z_offset
        )
        prep_to_initial(cur, grip_args, make_xmat, wrist_to_pinch)
        freeze_tools(cur, "current_wait")
        # Hold a shared ready pose through the visible pre-release settle; no target-specific IK.
        settle_steps = int(args.settle_seconds / dt)
        set_dataset_context("settle_before_release", cur)
        for _ in range(settle_steps):
            freeze_tools(cur, "current_wait")
            q_arm_cmd = v700.limit_step(q_arm_cmd, common_ready_q, v700.FR3_VEL_LIMIT, dt)
            q_hand_cmd = v700.limit_step(q_hand_cmd, v700.HAND_OPEN, v700.HAND_VEL_LIMIT, dt)
            data.ctrl[:7] = q_arm_cmd
            data.ctrl[7:18] = q_hand_cmd
            data.qpos[:7] = q_arm_cmd
            data.qpos[7:18] = q_hand_cmd
            data.qvel[:18] = 0.0
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            render_if_due()
        freeze_tools(cur, "current_wait")
        mujoco.mj_forward(model, data)
        tool_initial_qpos = data.qpos[qadr[cur]:qadr[cur] + 7].copy()
        # Internal teacher knowledge is allowed, but visible target-specific motion is not.
        # Precompute only the post-release intercept joint target; execute it after release.
        intercept_center, _, _, _ = grip_ref(
            grip_args, data.xpos[tool_bids[cur]].copy(), control_angle, args.intercept_target_tau, False, 0.0
        )
        intercept_center[2] = grip_args.catch_z
        q_arm_intercept = v700.solve_arm_ik(
            model, data, q_arm_cmd.copy(), wrist_bid, intercept_center + wrist_to_pinch, make_xmat(0.0), max_iter=128
        )

        latch_time = None
        strict_run = 0
        best_strict = 0
        strict_total = 0
        confirmed_hold = 0
        any_bad = False
        min_z = 10.0
        final_vz = 0.0
        capture_success = False
        capture_time = None
        capture_z = None
        capture_vz = None
        capture_latch_age = None
        min_tip_dist = {"thumb": 9.0, "index": 9.0, "middle": 9.0}
        min_wrist_to_handle = 9.0
        local_steps = int(args.catch_seconds / dt)
        for step in range(local_steps):
            local_t = step * dt
            if not args.use_actuator_dynamics:
                data.qpos[:7] = q_arm_cmd
                data.qpos[7:18] = q_hand_cmd
                data.qvel[:18] = 0.0
            if local_t < grip_args.release_time:
                set_dataset_context("wait_for_visible_release", cur)
                freeze_tools(cur, "current_wait")
                data.qpos[qadr[cur]:qadr[cur] + 7] = tool_initial_qpos
                data.qvel[qvadr[cur]:qvadr[cur] + 6] = 0.0
            else:
                set_dataset_context("intercept_or_hold", cur)
                freeze_tools(cur, "active")
            mujoco.mj_forward(model, data)
            tau = local_t - grip_args.release_time
            latched = latch_time is not None
            latch_age = 0.0 if latch_time is None else max(0.0, local_t - latch_time)
            if local_t < grip_args.release_time or tau < args.reaction_delay:
                # Wait until the release/drop is visible, then react. This avoids pre-positioned theater.
                q_arm_des = common_ready_q.copy()
                hand_des = v700.HAND_OPEN.copy()
            else:
                tau_eff = tau - args.reaction_delay
                tau_arm = tau_eff + args.intercept_phase_lead
                tau_hand = tau_eff + args.intercept_phase_lead - args.hand_phase_delay
                center, _, _, _ = grip_ref(
                    grip_args, data.xpos[tool_bids[cur]].copy(), control_angle, tau_arm, latched, latch_age
                )
                _, hand_des, _, _ = grip_ref(
                    grip_args, data.xpos[tool_bids[cur]].copy(), control_angle, tau_hand, latched, latch_age
                )
                if not latched:
                    center[2] = grip_args.catch_z
                elif latch_age < args.post_latch_cushion_seconds:
                    tool_handle_mid_z = float(data.xpos[tool_bids[cur]][2] - 0.055)
                    follow_z = max(args.post_latch_min_z, tool_handle_mid_z + args.post_latch_z_bias)
                    alpha = args.post_latch_z_follow * v700.smoothstep(
                        latch_age / max(1e-6, args.post_latch_cushion_seconds)
                    )
                    center[2] = center[2] * (1.0 - alpha) + follow_z * alpha
                if latched:
                    hand_des = apply_post_latch_fullfinger(args, hand_des, latch_age)
                else:
                    hand_des = apply_precontact_finger_closure(args, hand_des, tau_eff, control_angle)
                if (not latched) and tau_eff < args.intercept_joint_drive_until:
                    # First phase: decisive joint-space intercept under real FR3 velocity limits.
                    # This prevents slow local IK drift while still starting only after release.
                    q_arm_des = q_arm_intercept.copy()
                else:
                    # Second phase: local geometry refinement and post-contact cushion/hold.
                    q_arm_des = v700.solve_arm_ik(
                        model, data, q_arm_cmd, wrist_bid, center + wrist_to_pinch, make_xmat(latch_age if latched else 0.0), max_iter=48
                    )
            q_arm_cmd = v700.limit_step(q_arm_cmd, q_arm_des, v700.FR3_VEL_LIMIT, dt)
            q_hand_cmd = v700.limit_step(q_hand_cmd, hand_des, v700.HAND_VEL_LIMIT, dt)
            data.ctrl[:7] = q_arm_cmd
            data.ctrl[7:18] = q_hand_cmd
            if not args.use_actuator_dynamics:
                # Optional kinematic execution under explicit velocity limits. The stricter
                # actuator mode below is slower but lets contact impulses affect the fingers.
                data.qpos[:7] = q_arm_cmd
                data.qpos[7:18] = q_hand_cmd
                data.qvel[:18] = 0.0
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            if local_t >= grip_args.release_time:
                thumb_c, finger_c, bad = multi.contact_metrics_current(model, data, cur)
                set_dataset_context("intercept_or_hold", cur, contact_flags_array(thumb_c, finger_c, bad))
                any_bad = any_bad or bad
                if thumb_c and finger_c and not bad:
                    if latch_time is None:
                        latch_time = local_t
                    strict_run += 1
                    strict_total += 1
                    if latch_time is not None and local_t - latch_time >= args.confirm_after_latch_seconds:
                        confirmed_hold += 1
                    best_strict = max(best_strict, strict_run)
                else:
                    strict_run = 0
                min_z = min(min_z, float(data.xpos[tool_bids[cur]][2]))
                final_vz = float(data.qvel[qvadr[cur] + 2])
                cur_z = float(data.xpos[tool_bids[cur]][2])
                tool_x = data.xpos[tool_bids[cur]].copy()
                tool_R = data.xmat[tool_bids[cur]].reshape(3, 3).copy()
                h0 = tool_x + tool_R @ np.array([0.0, 0.0, -0.11])
                h1 = tool_x.copy()
                for key, gid in tip_marker_ids.items():
                    d_tip = v700.point_segment_distance(data.geom_xpos[gid].copy(), h0, h1)
                    min_tip_dist[key] = min(min_tip_dist[key], float(d_tip))
                min_wrist_to_handle = min(min_wrist_to_handle, float(v700.point_segment_distance(data.xpos[wrist_bid].copy(), h0, h1)))
                if (
                    args.early_capture_on_confirm
                    and best_strict >= max(1, int(args.strict_grasp_seconds / dt))
                    and confirmed_hold >= max(1, int(args.confirm_hold_seconds / dt))
                    and not any_bad
                    and cur_z > args.capture_min_z
                    and abs(final_vz) < args.capture_max_abs_vz
                ):
                    if not capture_success:
                        capture_success = True
                        capture_time = local_t
                        capture_z = cur_z
                        capture_vz = final_vz
                        capture_latch_age = latch_age
                render_if_due()
                if capture_success and capture_time is not None and local_t - capture_time >= args.visible_hold_seconds:
                    break
                continue
            render_if_due()
        required_strict = max(1, int(args.strict_grasp_seconds / dt))
        required_confirm = max(1, int(args.confirm_hold_seconds / dt))
        success = (
            bool(capture_success)
            and not any_bad
            and min_z > 0.50
            and abs(final_vz) < 1.2
        ) if args.early_capture_on_confirm else (
            best_strict >= required_strict
            and confirmed_hold >= required_confirm
            and not any_bad
            and min_z > 0.50
            and abs(final_vz) < 1.2
        )
        return grip_args, {
            "tool_index": cur,
            "angle_deg": angles[cur],
            "control_angle_deg": round(control_angle, 3),
            "yaw_deg": yaws[cur],
            "success": bool(success),
            "best_strict": int(best_strict),
            "strict_total": int(strict_total),
            "confirmed_hold": int(confirmed_hold),
            "required_strict": int(required_strict),
            "required_confirm": int(required_confirm),
            "bad_functional_contact": bool(any_bad),
            "min_z": round(min_z, 3),
            "final_vz": round(final_vz, 3),
            "capture_time_s": None if capture_time is None else round(capture_time, 3),
            "capture_z_m": None if capture_z is None else round(capture_z, 3),
            "capture_vz_m_s": None if capture_vz is None else round(capture_vz, 3),
            "capture_latch_age_s": None if capture_latch_age is None else round(capture_latch_age, 3),
            "visible_hold_seconds": round(args.visible_hold_seconds, 3),
            "post_capture_elapsed_s": None if capture_time is None else round(max(0.0, local_t - capture_time), 3),
            "min_tip_dist_m": {k: round(v, 4) for k, v in min_tip_dist.items()},
            "min_wrist_to_handle_m": round(min_wrist_to_handle, 4),
        }

    def visible_hold(cur, grip_args):
        """Keep the physically captured rod visible before discard.

        This does not weld or latch the tool. The rod remains a free body and the
        hold is counted only through the same thumb-vs-index/middle contact test
        used by the catch stage.
        """
        nonlocal q_arm_cmd, q_hand_cmd
        control_angle = getattr(grip_args, "control_angle_deg", angles[cur])
        make_xmat = fixed_wrist_xmat_fn(grip_args, control_angle)
        wrist_to_pinch = v700.wrist_offset_for_angle(
            control_angle, grip_args.radial_offset, grip_args.tangent_offset, grip_args.z_offset
        )
        steps = int(args.visible_hold_seconds / dt)
        latch_age_start = float(getattr(grip_args, "visible_latch_age_start", 0.0))
        strict_hold = 0
        bad_hold = False
        min_hold_z = 10.0
        max_abs_vz = 0.0
        for step in range(steps):
            age = latch_age_start + step * dt
            set_dataset_context("visible_physical_hold", cur)
            freeze_tools(cur, "active")
            _, hand_des, _, _ = grip_ref(
                grip_args,
                data.xpos[tool_bids[cur]].copy(),
                control_angle,
                args.intercept_target_tau,
                True,
                age,
            )
            # Keep the wrist/arm steady during the visible hold. Continuing to
            # solve IK after capture can chase the falling free body and sweep
            # the red functional segment into the fingertips at edge angles.
            q_arm_des = q_arm_cmd.copy()
            hand_des = apply_post_latch_fullfinger(args, hand_des, age)
            q_arm_cmd = v700.limit_step(q_arm_cmd, q_arm_des, v700.FR3_VEL_LIMIT, dt)
            q_hand_cmd = v700.limit_step(q_hand_cmd, hand_des, v700.HAND_VEL_LIMIT, dt)
            data.ctrl[:7] = q_arm_cmd
            data.ctrl[7:18] = q_hand_cmd
            if not args.use_actuator_dynamics:
                data.qpos[:7] = q_arm_cmd
                data.qpos[7:18] = q_hand_cmd
                data.qvel[:18] = 0.0
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            thumb_c, finger_c, bad = multi.contact_metrics_current(model, data, cur)
            set_dataset_context("visible_physical_hold", cur, contact_flags_array(thumb_c, finger_c, bad))
            if thumb_c and finger_c and not bad:
                strict_hold += 1
            bad_hold = bad_hold or bad
            min_hold_z = min(min_hold_z, float(data.xpos[tool_bids[cur]][2]))
            max_abs_vz = max(max_abs_vz, abs(float(data.qvel[qvadr[cur] + 2])))
            render_if_due()
        return {
            "visible_hold_seconds": round(args.visible_hold_seconds, 3),
            "visible_hold_strict_frames": int(strict_hold),
            "visible_hold_required_frames": int(steps),
            "visible_hold_bad_functional_contact": bool(bad_hold),
            "visible_hold_min_z": round(min_hold_z, 3),
            "visible_hold_max_abs_vz": round(max_abs_vz, 3),
        }

    def discard(cur, grip_args):
        nonlocal q_arm_cmd, q_hand_cmd
        control_angle = getattr(grip_args, "control_angle_deg", angles[cur])
        make_xmat = multi.make_wrist_xmat_fn(data, wrist_bid, grip_args, control_angle)
        make_xmat = fixed_wrist_xmat_fn(grip_args, control_angle)
        steps = int(args.discard_seconds / dt)
        set_dataset_context("discard_old_tool", cur)
        for step in range(steps):
            age = step * dt
            freeze_tools(cur, "active")
            q_hand_cmd = v700.limit_step(q_hand_cmd, v700.HAND_OPEN, v700.HAND_VEL_LIMIT, dt)
            base = data.xpos[wrist_bid].copy()
            shake = np.array([0.05 * math.sin(42 * age), -0.10 - 0.04 * math.sin(27 * age), 0.05 * math.sin(31 * age)])
            q_arm_des = v700.solve_arm_ik(model, data, q_arm_cmd, wrist_bid, base + shake, make_xmat(0.0), max_iter=24)
            q_arm_cmd = v700.limit_step(q_arm_cmd, q_arm_des, v700.FR3_VEL_LIMIT, dt)
            data.ctrl[:7] = q_arm_cmd
            data.ctrl[7:18] = q_hand_cmd
            if not args.use_actuator_dynamics:
                data.qpos[:7] = q_arm_cmd
                data.qpos[7:18] = q_hand_cmd
                data.qvel[:18] = 0.0
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            render_if_due()
        hidden.add(cur)
        freeze_tools(None)
        mujoco.mj_forward(model, data)

    records = []
    step_ready(0.35)
    for drop_i, cur in enumerate(order):
        step_ready(intervals[drop_i], cur)
        grip_args, rec = catch_one(cur)
        discard(cur, grip_args)
        rec["hidden"] = True
        records.append(rec)
    step_ready(0.6)

    if renderer is not None:
        renderer.close()
    videos = {}
    if record_video:
        for name, cam_frames in frames.items():
            video = out / f"front110_v728_front110_grid9_clean_realrobot_seed{args.seed}_{name}.mp4"
            imageio.mimsave(video, cam_frames, fps=args.fps, macro_block_size=16)
            videos[name] = str(video)
    summary = {
        "seed": args.seed,
        "angles_deg": angles,
        "release_order_tool_indices": order,
        "release_order_angles_deg": [angles[i] for i in order],
        "intervals_s": intervals,
        "base_release_height_m": args.release_height,
        "release_heights_m": release_heights,
        "release_delay_s": args.release_delay,
        "reaction_delay_s": args.reaction_delay,
        "videos": videos,
        "drops": records,
        "passed": sum(1 for r in records if r["success"]),
        "total": len(records),
        "controller": "frozen/stage4e_front110_v700_grid0p5_patch.py",
        "robot_mjcf": str(VISUAL_ROBOT_XML),
        "drop_xy_offsets_m": drop_xy_offsets,
        "visual_policy": "No target-specific arm motion before visible release: common ready only; helper/marker geoms hidden in video; true FR3/Revo2 visual meshes are rendered; after strict capture the rod remains visible in-hand before discard.",
        "task_specs": {
            "fr3_joint_vel_limit_rad_s": v700.FR3_VEL_LIMIT.tolist(),
            "revo2_joint_vel_limit_rad_s_internal_11dof": v700.HAND_VEL_LIMIT.tolist(),
            "revo2_active_joint_order_user": ["thumb_flex", "thumb_aux", "index", "middle", "ring", "pinky"],
            "tool_total_length_m": 0.33,
            "tool_diameter_m": 0.03,
            "tool_mass_kg": 0.125,
            "handle_positive_length_m": 0.11,
            "functional_negative_length_m": 0.22,
            "drop_xy_jitter_m": args.drop_xy_jitter,
            "yaw_random_range_deg": [-10, 10],
            "nominal_front_arc_radius_m": 0.50
        },
        "behavior": "v731 real-mesh observe-then-grasp planner with local 118-132deg dense-grid catch-height patch plus 126-139deg mid-left red-clearance lift: v725 edge-sector release-height timing with 0.12s strict visible physical hold. Mid-arc uses 1.35m release; hard edge angles get extra falling time without target-specific pre-release arm motion. Strict sustained opposing fingertip grasp, then active open-hand wrist/arm discard.",
        "scene_xml": str(scene_xml),
    }
    if recorder is not None:
        dataset_dir = Path(args.dataset_out_dir)
        episode_npz = dataset_dir / f"common_episode_seed{args.seed}.npz"
        recorder.save_npz(episode_npz)
        validate_npz(episode_npz)
        manifest = build_manifest(summary, episode_npz, simulator="mujoco")
        manifest_path = dataset_dir / f"manifest_seed{args.seed}.json"
        write_manifest(manifest_path, manifest)
        summary["dataset"] = {
            "schema_version": manifest["schema_version"],
            "episode_npz": str(episode_npz),
            "manifest": str(manifest_path),
            "validated": True,
            "stride": dataset_stride,
            "action_space": manifest["action_space"],
        }
    (out / f"front110_v728_front110_grid9_clean_realrobot_seed{args.seed}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=70117)
    ap.add_argument("--num-tools", type=int, default=6)
    ap.add_argument("--angles-deg", default="")
    ap.add_argument("--yaws-deg", default="")
    ap.add_argument("--keep-order", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--prep-seconds", type=float, default=0.80)
    ap.add_argument("--settle-seconds", type=float, default=0.32)
    ap.add_argument("--catch-seconds", type=float, default=1.30)
    ap.add_argument("--discard-seconds", type=float, default=0.44)
    ap.add_argument("--visible-hold-seconds", type=float, default=0.12)
    ap.add_argument("--low-edge-wrist-pitch-boost", type=float, default=8.0)
    ap.add_argument("--low-edge-pitch-end-deg", type=float, default=75.0)
    ap.add_argument("--low-edge-pitch-width-deg", type=float, default=20.0)
    ap.add_argument("--release-height", type=float, default=1.35)
    ap.add_argument("--edge-release-height-boost", type=float, default=0.20)
    ap.add_argument("--edge-release-low-end-deg", type=float, default=55.0)
    ap.add_argument("--edge-release-high-start-deg", type=float, default=125.0)
    ap.add_argument("--edge-release-width-deg", type=float, default=20.0)
    ap.add_argument("--release-delay", type=float, default=0.08)
    ap.add_argument("--reaction-delay", type=float, default=0.06)
    ap.add_argument("--intercept-phase-lead", type=float, default=0.20)
    ap.add_argument("--intercept-target-tau", type=float, default=0.20)
    ap.add_argument("--intercept-joint-drive-until", type=float, default=0.24)
    ap.add_argument("--catch-z-offset", type=float, default=0.0)
    ap.add_argument("--realmesh-radial-offset", type=float, default=-0.20)
    ap.add_argument("--realmesh-tangent-offset", type=float, default=0.07)
    ap.add_argument("--realmesh-z-offset", type=float, default=0.14)
    ap.add_argument("--wrist-tangent-offset-delta", type=float, default=0.0)
    ap.add_argument("--wrist-radial-offset-delta", type=float, default=0.0)
    ap.add_argument("--wrist-z-offset-delta", type=float, default=0.0)
    ap.add_argument("--wrist-yaw-delta", type=float, default=0.0)
    ap.add_argument("--wrist-roll-delta", type=float, default=0.0)
    ap.add_argument("--wrist-pitch-delta", type=float, default=0.0)
    ap.add_argument("--middle-close-scale", type=float, default=-1.0)
    ap.add_argument("--middle-latch-scale", type=float, default=-1.0)
    ap.add_argument("--index-latch-scale", type=float, default=-1.0)
    ap.add_argument("--latch-xy-follow", type=float, default=-1.0)
    ap.add_argument("--latch-tangent-offset-delta", type=float, default=0.0)
    ap.add_argument("--latch-radial-offset-delta", type=float, default=0.0)
    ap.add_argument("--post-latch-fullfinger-scale", type=float, default=1.0)
    ap.add_argument("--post-latch-fullfinger-start", type=float, default=0.006)
    ap.add_argument("--post-latch-fullfinger-dur", type=float, default=0.070)
    ap.add_argument("--post-latch-cushion-seconds", type=float, default=0.140)
    ap.add_argument("--post-latch-z-follow", type=float, default=0.65)
    ap.add_argument("--post-latch-z-bias", type=float, default=0.020)
    ap.add_argument("--post-latch-min-z", type=float, default=0.560)
    ap.add_argument("--post-latch-ring-pinky-scale", type=float, default=0.0)
    ap.add_argument("--precontact-finger-scale", type=float, default=0.90)
    ap.add_argument("--precontact-finger-start", type=float, default=0.030)
    ap.add_argument("--precontact-finger-dur", type=float, default=0.125)
    ap.add_argument("--precontact-middle-scale", type=float, default=0.95)
    ap.add_argument("--precontact-ring-pinky-scale", type=float, default=0.0)
    ap.add_argument("--disable-latch-lift-brake", dest="disable_latch_lift_brake", action="store_true", default=True)
    ap.add_argument("--enable-latch-lift-brake", dest="disable_latch_lift_brake", action="store_false")
    ap.add_argument("--show-ghosts", action="store_true")
    ap.add_argument("--use-actuator-dynamics", action="store_true")
    ap.add_argument("--drop-xy-jitter", type=float, default=0.04)
    ap.add_argument("--hand-phase-delay", type=float, default=0.10)
    ap.add_argument("--strict-grasp-seconds", type=float, default=0.10)
    ap.add_argument("--confirm-after-latch-seconds", type=float, default=0.04)
    ap.add_argument("--confirm-hold-seconds", type=float, default=0.06)
    ap.add_argument("--early-capture-on-confirm", action="store_true", default=True)
    ap.add_argument("--capture-min-z", type=float, default=0.55)
    ap.add_argument("--capture-max-abs-vz", type=float, default=0.75)
    ap.add_argument("--dataset-out-dir", default="")
    ap.add_argument("--dataset-stride", type=int, default=4)
    ap.add_argument("--out-dir", default=str(OUT))
    run_demo(ap.parse_args())


if __name__ == "__main__":
    main()

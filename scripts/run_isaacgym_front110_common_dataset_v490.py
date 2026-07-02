#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, os, random, sys
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))
from front110_core.isaacgym_runner_bridge import maybe_create_common_dataset_hook
import isaacgym  # noqa
from isaacgym import gymapi, gymtorch
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
import isaacgymenvs
TASK='SimToolRealFallingBatonV89FrankaBrainCoRevo2SequentialSpindleCatch'
TRAIN='SimToolRealDynamicGraspV33FrankaBrainCoRevo2AffordanceDomino20PointNetPPO'
HANDLE_LOCAL_Z=float(os.environ.get('SCREW_HANDLE_LOCAL_Z','0.14'))

def args_parser():
    p=argparse.ArgumentParser(); p.add_argument('--episodes',type=int,default=8); p.add_argument('--steps',type=int,default=175); p.add_argument('--seed',type=int,default=12001); p.add_argument('--num-envs',type=int,default=1); p.add_argument('--render-video',action='store_true'); p.add_argument('--video-every',type=int,default=1); p.add_argument('--out-dir',type=Path,default=Path('outputs/ring_v490_front110_sector_game')); p.add_argument('--front-x',type=float,default=float(os.environ.get('SCREW_FRONT_X','0.0'))); p.add_argument('--front-y',type=float,default=float(os.environ.get('SCREW_FRONT_Y','0.62'))); p.add_argument('--front-z',type=float,default=float(os.environ.get('SCREW_FRONT_Z','1.42'))); p.add_argument('--ring-count',type=int,default=int(os.environ.get('SCREW_RING_COUNT','8'))); p.add_argument('--ring-radius',type=float,default=float(os.environ.get('SCREW_RING_RADIUS','0.82'))); p.add_argument('--ring-z',type=float,default=float(os.environ.get('SCREW_RING_Z','1.92'))); p.add_argument('--intercept-z',type=float,default=1.08); p.add_argument('--lead-time',type=float,default=0.12); p.add_argument('--fixed-angle-deg',type=float,default=None); p.add_argument('--angle-min-deg',type=float,default=float(os.environ.get('SCREW_ANGLE_MIN_DEG','0'))); p.add_argument('--angle-max-deg',type=float,default=float(os.environ.get('SCREW_ANGLE_MAX_DEG','360'))); p.add_argument('--common-dataset-out-dir',type=Path,default=None,help='Optional common-schema NPZ output directory, relative to --out-dir if not absolute.'); p.add_argument('--common-dataset-stride',type=int,default=1,help='Record every N IsaacGym simulation steps.'); p.add_argument('--common-dataset-validate',action='store_true',help='Validate each common-schema NPZ after writing.'); return p.parse_args()

def norm_joint_target(env,target):
    lo=env.arm_hand_dof_lower_limits[:env.num_arm_dofs]; hi=env.arm_hand_dof_upper_limits[:env.num_arm_dofs]; ctr=torch.clamp(env.hand_arm_default_dof_pos[:env.num_arm_dofs],lo,hi); ps=torch.clamp(hi-ctr,min=1e-6); ns=torch.clamp(ctr-lo,min=1e-6); return torch.where(target>=ctr,(target-ctr)/ps,(target-ctr)/ns).clamp(-1,1)

def _vec_env(name, default):
    txt=os.environ.get(name)
    if not txt: return default
    vals=[float(x) for x in txt.replace(',', ' ').split()]
    if len(vals)!=len(default): raise ValueError(f'{name} expected {len(default)} values, got {len(vals)}')
    return vals

def hand_cmd(phase,device):
    openv=torch.tensor(_vec_env('SCREW_HAND_OPEN',[-0.92,-0.92,-0.92,-0.92,-0.88,-0.88,-0.88,-0.88,0.02,0.02,0.02]),device=device)
    pre=torch.tensor(_vec_env('SCREW_HAND_PRE',[0.14,0.08,0.16,0.10,-0.80,-0.84,-0.80,-0.84,0.38,0.30,0.20]),device=device)
    close=torch.tensor(_vec_env('SCREW_HAND_CLOSE',[1.10,1.02,1.08,1.00,-0.72,-0.78,-0.72,-0.78,1.20,1.12,1.02]),device=device)
    hold=torch.tensor(_vec_env('SCREW_HAND_HOLD',[1.08,1.02,1.06,1.00,-0.72,-0.78,-0.72,-0.78,1.22,1.14,1.04]),device=device)
    return {"open":openv,"pre":pre,"close":close,"hold":hold,"release":openv}.get(phase,openv)

def make_action(env,arm_target,phase,hand_override=None):
    a=torch.zeros((env.num_envs,env.num_actions),device=env.device); a[:,:env.num_arm_dofs]=norm_joint_target(env,arm_target).repeat(env.num_envs,1); h=hand_cmd(phase,env.device) if hand_override is None else hand_override; a[:,env.num_arm_dofs:env.num_arm_dofs+env.num_policy_hand_dofs]=h; return a



def make_joint_pos_targets(env,arm_target,phase,hand_override=None):
    jt=env.cur_targets[:, :env.num_hand_arm_dofs].clone()
    arm=arm_target if arm_target.shape[0]==env.num_envs else arm_target.repeat(env.num_envs,1)
    jt[:, :env.num_arm_dofs]=arm
    h=hand_cmd(phase,env.device) if hand_override is None else hand_override
    if h.dim()==1:
        h=h.unsqueeze(0).repeat(env.num_envs,1)
    n=min(h.shape[1], env.num_hand_arm_dofs-env.num_arm_dofs)
    jt[:, env.num_arm_dofs:env.num_arm_dofs+n]=h[:, :n]
    lo=env.arm_hand_dof_lower_limits[:env.num_hand_arm_dofs].unsqueeze(0)
    hi=env.arm_hand_dof_upper_limits[:env.num_hand_arm_dofs].unsqueeze(0)
    return torch.max(torch.min(jt,hi),lo)

def clear_task_resets(env):
    if os.environ.get('SCREW_SUPPRESS_RESETS','0')=='1':
        if hasattr(env,'reset_buf'):
            env.reset_buf.zero_()
        if hasattr(env,'reset_goal_buf'):
            env.reset_goal_buf.zero_()

def step_with_targets(env,arm_target,phase,hand_override=None):
    clear_task_resets(env)
    action=make_action(env,arm_target,phase,hand_override)
    joint_pos_targets=make_joint_pos_targets(env,arm_target,phase,hand_override)
    env._front110_last_joint_pos_targets=joint_pos_targets
    if os.environ.get('SCREW_DIRECT_JOINT_TARGETS','0')=='1':
        out=env.step(action, joint_pos_targets=joint_pos_targets)
    else:
        out=env.step(action)
    clear_task_resets(env)
    return out


def adaptive_hand_cmd(phase,device,thumb_d,im_d,side_thumb,side_im,real_pinch,grasp_latched,latch_age=0):
    h=hand_cmd(phase,device).clone()
    if phase == "hold" and grasp_latched and os.environ.get("SCREW_STRONG_HOLD_PINCH", "0") == "1":
        strong_delay=int(os.environ.get("SCREW_STRONG_HOLD_DELAY_STEPS", "0"))
        if int(latch_age) >= strong_delay:
            h=torch.tensor(_vec_env("SCREW_HAND_HOLD_STRONG",[1.24,1.18,1.24,1.18,-0.78,-0.82,-0.78,-0.82,1.36,1.28,1.18]),device=device)
        elif os.environ.get("SCREW_HAND_HOLD_EARLY", ""):
            h=torch.tensor(_vec_env("SCREW_HAND_HOLD_EARLY",[1.24,1.18,1.24,1.18,-0.95,-0.95,-0.95,-0.95,1.36,1.28,1.18]),device=device)
        if int(latch_age) < int(os.environ.get("SCREW_OUTER_FINGER_DELAY_STEPS", "0")) and os.environ.get("SCREW_EARLY_HOLD_KEEP_OUTER_OPEN", "1") == "1":
            h[4:8] = float(os.environ.get("SCREW_EARLY_OUTER_OPEN_VALUE", "-0.95"))
    if os.environ.get("SCREW_ADAPTIVE_HAND","0")!="1":
        return h
    if phase not in ("close","hold"):
        return h
    gain=float(os.environ.get("SCREW_ADAPTIVE_GAIN","2.8"))
    max_extra=float(os.environ.get("SCREW_ADAPTIVE_MAX_EXTRA","0.34"))
    near=max(0.0, min(max_extra, gain*(float(os.environ.get("SCREW_ADAPTIVE_DIST_GATE","0.115"))-min(float(thumb_d),float(im_d)))))
    side_gate=float(os.environ.get("SCREW_ADAPTIVE_SIDE_GATE","0.002"))
    opposed=bool(real_pinch) or (float(side_thumb)<-side_gate and float(side_im)>side_gate)
    same_side=(float(side_thumb)*float(side_im))>0.0
    if opposed or grasp_latched:
        h[0:4]+=near
        h[8:11]+=near*float(os.environ.get("SCREW_ADAPTIVE_THUMB_GAIN","1.05"))
    elif same_side:
        h[0:4]-=float(os.environ.get("SCREW_ADAPTIVE_SAMESIDE_OPEN","0.14"))
        h[8:11]-=float(os.environ.get("SCREW_ADAPTIVE_SAMESIDE_THUMB_OPEN","0.12"))
    else:
        h[0:4]+=near*0.45
        h[8:11]+=near*0.35
    if phase == "hold" and grasp_latched and os.environ.get("SCREW_POST_LATCH_FINGER_SERVO","0") == "1" and int(latch_age) >= int(os.environ.get("SCREW_POST_LATCH_SERVO_DELAY_STEPS", "0")):
        # Physical action order is index(2), middle(2), pinky(2), ring(2), thumb(3).
        # After real opposed contact, make small differential corrections instead of
        # globally over-closing the hand, which tends to bat the handle out.
        target_gap=float(os.environ.get("SCREW_POST_LATCH_SIDE_GAP","0.018"))
        max_delta=float(os.environ.get("SCREW_POST_LATCH_MAX_DELTA","0.22"))
        close_gain=float(os.environ.get("SCREW_POST_LATCH_CLOSE_GAIN","2.0"))
        thumb_gain=float(os.environ.get("SCREW_POST_LATCH_THUMB_GAIN","1.55"))
        index_gain=float(os.environ.get("SCREW_POST_LATCH_INDEX_GAIN","1.00"))
        center=float(side_thumb+side_im)*0.5
        width=abs(float(side_im-side_thumb))
        close=max(0.0, min(max_delta, (float(os.environ.get("SCREW_POST_LATCH_WIDTH_GATE","0.070"))-width)*close_gain))
        cent=max(-max_delta, min(max_delta, -center*float(os.environ.get("SCREW_POST_LATCH_CENTER_GAIN","1.8"))))
        if os.environ.get("SCREW_POST_LATCH_GLOBAL_SERVO", "1") == "1" and (bool(real_pinch) or (float(side_thumb) < -target_gap and float(side_im) > target_gap)):
            h[0:4] += close*index_gain + cent*float(os.environ.get("SCREW_POST_LATCH_INDEX_CENTER_GAIN","0.35"))
            h[8:11] += close*thumb_gain - cent*float(os.environ.get("SCREW_POST_LATCH_THUMB_CENTER_GAIN","0.45"))
            h[4:8] -= float(os.environ.get("SCREW_POST_LATCH_OUTER_OPEN","0.05"))
        if os.environ.get("SCREW_POST_LATCH_SIDE_GUARD", "0") == "1":
            guard=float(os.environ.get("SCREW_POST_LATCH_THUMB_NEG_GUARD", "0.018"))
            max_guard=float(os.environ.get("SCREW_POST_LATCH_GUARD_MAX", "0.34"))
            # Keep thumb on the negative side of the handle. Side-angle failures
            # happen when thumb crosses through zero and pushes the handle out.
            thumb_open=max(0.0, min(max_guard, (float(side_thumb)+guard)*float(os.environ.get("SCREW_POST_LATCH_THUMB_OPEN_GAIN", "3.0"))))
            im_close=max(0.0, min(max_guard, (guard-float(side_im))*float(os.environ.get("SCREW_POST_LATCH_IM_CLOSE_GAIN", "2.0"))))
            if float(side_thumb) > -guard:
                h[8:11] -= thumb_open
                h[0:4] += im_close*float(os.environ.get("SCREW_POST_LATCH_GUARD_INDEX_SHARE", "0.65"))
            if float(side_im) < guard:
                if os.environ.get("SCREW_POST_LATCH_IM_GUARD_OPEN", "0") == "1":
                    h[0:4] -= im_close * float(os.environ.get("SCREW_POST_LATCH_IM_OPEN_SHARE", "0.75"))
                else:
                    h[0:4] += im_close
            if (float(side_thumb) > 0.0 and float(side_im) > 0.0):
                h[8:11] -= float(os.environ.get("SCREW_POST_LATCH_SAMEPOS_THUMB_OPEN", "0.18"))
            elif (float(side_thumb) < 0.0 and float(side_im) < 0.0):
                h[0:4] -= float(os.environ.get("SCREW_POST_LATCH_SAMENEG_INDEX_OPEN", "0.10"))
    return torch.clamp(h, -1.0, 1.6)



def fingertip_closed_loop_target(env, ik_body_idx, target, hp, vel, thumb, im, palm, pinch_center, side_axis, handle_side_thumb, handle_side_im, grasp_latched, latched_side_axis=None):
    """Servo the actual IK body so the thumb/index-middle mouth surrounds the handle."""
    if os.environ.get("SCREW_FINGERTIP_CLOSED_LOOP", "0") != "1":
        return target
    ik_pos = env.rigid_body_states[:, ik_body_idx, 0:3]
    if bool(grasp_latched) and os.environ.get("SCREW_FTCL_LOCK_SIDE_AXIS", "0") == "1" and latched_side_axis is not None:
        side_axis = torch.nn.functional.normalize(latched_side_axis, dim=-1)
        handle_side_thumb = torch.sum((thumb - hp) * side_axis, dim=-1)
        handle_side_im = torch.sum((im - hp) * side_axis, dim=-1)
    cmax = float(os.environ.get("SCREW_FTCL_CENTER_MAX", "0.26"))
    gxy = float(os.environ.get("SCREW_FTCL_XY_GAIN", "1.70"))
    gz = float(os.environ.get("SCREW_FTCL_Z_GAIN", "1.05"))
    z_bias = float(os.environ.get("SCREW_FTCL_Z_BIAS", "-0.006"))
    if bool(grasp_latched):
        gxy = float(os.environ.get("SCREW_FTCL_HOLD_XY_GAIN", str(gxy)))
        gz = float(os.environ.get("SCREW_FTCL_HOLD_Z_GAIN", str(gz)))
        z_bias = float(os.environ.get("SCREW_FTCL_HOLD_Z_BIAS", "-0.018"))
    center_err = torch.clamp(hp - pinch_center, -cmax, cmax)
    ft = ik_pos + torch.cat((center_err[:, :2] * gxy, center_err[:, 2:3] * gz), dim=-1)
    desired = float(os.environ.get("SCREW_FTCL_SIDE_GAP", "0.034"))
    if bool(grasp_latched):
        desired = float(os.environ.get("SCREW_FTCL_HOLD_SIDE_GAP", str(desired)))
    same_pos = (handle_side_thumb > 0) & (handle_side_im > 0)
    same_neg = (handle_side_thumb < 0) & (handle_side_im < 0)
    raw = torch.zeros_like(handle_side_thumb)
    raw = torch.where(same_pos, -(torch.minimum(handle_side_thumb, handle_side_im) + desired), raw)
    raw = torch.where(same_neg, -(torch.maximum(handle_side_thumb, handle_side_im) - desired), raw)
    raw = torch.where(~(same_pos | same_neg), -0.5 * (handle_side_thumb + handle_side_im), raw)
    sgain = float(os.environ.get("SCREW_FTCL_SIDE_GAIN", "5.2"))
    smax = float(os.environ.get("SCREW_FTCL_SIDE_MAX", "0.28"))
    if bool(grasp_latched):
        sgain = float(os.environ.get("SCREW_FTCL_HOLD_SIDE_GAIN", str(sgain)))
        smax = float(os.environ.get("SCREW_FTCL_HOLD_SIDE_MAX", str(smax)))
    side_corr = (raw * sgain).clamp(-smax, smax).unsqueeze(-1) * side_axis
    # Follow the falling handle slightly during first contact so the object is squeezed, not batted away.
    down = torch.clamp(-vel[:, 2:3], 0.0, float(os.environ.get("SCREW_FTCL_DOWN_SPEED_MAX", "4.5")))
    follow = -down * float(os.environ.get("SCREW_FTCL_DOWN_FOLLOW", "0.006")) if bool(grasp_latched) else 0.0
    ft = ft + side_corr + torch.tensor([[0.0, 0.0, z_bias]], device=env.device)
    if bool(grasp_latched):
        ft[:, 2:3] = ft[:, 2:3] + follow
        if os.environ.get("SCREW_FTCL_HOLD_VEL_FOLLOW", "0") == "1":
            # Once the handle is between the fingertips, lead the pinch mouth by the
            # measured handle velocity. This targets post-contact slip without
            # increasing grip force enough to bat the object away.
            vgain = float(os.environ.get("SCREW_FTCL_HOLD_VEL_GAIN", "1.0"))
            vdt = float(os.environ.get("SCREW_FTCL_HOLD_VEL_DT", "0.030"))
            vmax = float(os.environ.get("SCREW_FTCL_HOLD_VEL_MAX", "0.024"))
            lead_xy = torch.clamp(vel[:, 0:2] * vdt * vgain, -vmax, vmax)
            if os.environ.get("SCREW_FTCL_HOLD_VEL_Y_ONLY", "1") == "1":
                lead_xy[:, 0] = 0.0
            ft[:, 0:2] = ft[:, 0:2] + lead_xy
    front_backoff = float(os.environ.get("SCREW_FTCL_FRONT_BACKOFF_Y", "0.030"))
    ft[:, 1] = torch.maximum(ft[:, 1], hp[:, 1] - front_backoff)
    blend = float(os.environ.get("SCREW_FTCL_BLEND", "0.92"))
    if bool(grasp_latched):
        blend = float(os.environ.get("SCREW_FTCL_HOLD_BLEND", "0.97"))
    return (1.0 - blend) * target + blend * ft

def quat_mul(q,r):
    x1,y1,z1,w1=q.unbind(-1); x2,y2,z2,w2=r.unbind(-1); return torch.stack((w1*x2+x1*w2+y1*z2-z1*y2,w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2,w1*w2-x1*x2-y1*y2-z1*z2),-1)
def quat_conj(q): return torch.cat((-q[...,:3],q[...,3:4]),-1)
def quat_apply(q,v):
    qv=q[...,:3]; uv=torch.cross(qv,v,dim=-1); uuv=torch.cross(qv,uv,dim=-1); return v+2*(q[...,3:4]*uv+uuv)
def euler_quat(roll,pitch,yaw,device):
    cr,sr=math.cos(roll/2),math.sin(roll/2); cp,sp=math.cos(pitch/2),math.sin(pitch/2); cy,sy=math.cos(yaw/2),math.sin(yaw/2); return quat_mul(quat_mul(torch.tensor([sr,0,0,cr],device=device),torch.tensor([0,sp,0,cp],device=device)),torch.tensor([0,0,sy,cy],device=device))



def _lerp(a,b,u):
    return a+(b-a)*max(0.0,min(1.0,u))

def sector_pinch_axis(mode, radial, tangent, side_axis, angle_rad, device):
    """World-frame mouth axis for side-angle catches.

    The observed thumb-index vector is useful after a clean grasp exists, but it
    is a poor control axis when both fingertips approach the same side of the
    handle.  For ring sectors, choose a stable radial/tangent axis first, then
    let the current finger geometry take over only when requested.
    """
    mode = (mode or "current").lower()
    if mode in ("current", "finger", "side"):
        axis = side_axis
    elif mode in ("tangent", "tan"):
        axis = tangent
    elif mode in ("neg_tangent", "-tangent", "minus_tangent"):
        axis = -tangent
    elif mode in ("radial", "rad"):
        axis = radial
    elif mode in ("neg_radial", "-radial", "minus_radial"):
        axis = -radial
    elif mode in ("angle_conditioned", "sector"):
        deg = math.degrees(angle_rad) % 360.0
        # Front arc already works best with tangent.  Side sectors need a
        # radial-biased mouth so the handle enters between thumb and fingers
        # instead of sliding along the same side of both fingertips.
        if 25.0 <= deg < 65.0:
            # Right-front edge: bias the pinch mouth outward/around the ring so
            # the handle enters between thumb and index-middle instead of
            # sliding past the palm side.
            mix = 0.58 * tangent - 0.42 * radial
            axis = mix
        elif 65.0 <= deg <= 115.0:
            axis = tangent
        elif 115.0 < deg <= 155.0:
            # Left-front edge uses the opposite radial bias.
            mix = 0.58 * tangent + 0.42 * radial
            axis = mix
        elif 195.0 <= deg <= 255.0:
            mix = 0.35 * tangent - 0.65 * radial
            axis = mix
        else:
            axis = tangent
    else:
        parts = mode.split(",")
        if len(parts) == 2:
            tx, rx = float(parts[0]), float(parts[1])
            axis = tx * tangent + rx * radial
        else:
            axis = side_axis
    return torch.nn.functional.normalize(axis, dim=-1)

def front110_sector_params(angle_rad):
    deg=(math.degrees(angle_rad)%360.0)
    # v490 front-wide sector controller. It preserves v470 center-front
    # settings and only changes wrist yaw plus tangent/radial offsets by sector.
    anchors=[
        (35.0, dict(yaw=2.15,t=0.078,r=-0.030,z=0.024,lead=0.30,iz=1.50,pre_z=1.62,roll=-1.34,pitch=-0.70)),
        (45.0, dict(yaw=2.10,t=0.050,r=-0.025,z=0.022,lead=0.31,iz=1.50,pre_z=1.62,roll=-1.45,pitch=-0.62)),
        (55.0, dict(yaw=2.30,t=0.060,r=0.000,z=0.020,lead=0.32,iz=1.50,pre_z=1.60)),
        (75.0, dict(yaw=1.57,t=0.000,r=-0.055,z=0.018,lead=0.14,iz=1.30,pre_z=1.34)),
        (90.0, dict(yaw=0.80,t=0.000,r=-0.050,z=0.018,lead=0.14,iz=1.30,pre_z=1.34)),
        (105.0,dict(yaw=1.20,t=-0.060,r=-0.055,z=0.018,lead=0.13,iz=1.36,pre_z=1.38)),
        (125.0,dict(yaw=1.10,t=-0.100,r=-0.030,z=0.020,lead=0.17,iz=1.36,pre_z=1.42)),
        (145.0,dict(yaw=0.75,t=-0.105,r=-0.055,z=0.022,lead=0.20,iz=1.42,pre_z=1.54,roll=-1.40,pitch=-0.75)),
    ]
    if deg < 40.0:
        label, chosen = 35, anchors[0][1]
    elif deg < 50.0:
        label, chosen = 45, anchors[1][1]
    elif deg < 65.0:
        label, chosen = 55, anchors[2][1]
    elif deg < 82.0:
        label, chosen = 75, anchors[3][1]
    elif deg < 100.0:
        label, chosen = 90, anchors[4][1]
    elif deg < 116.0:
        label, chosen = 105, anchors[5][1]
    elif deg < 136.0:
        label, chosen = 125, anchors[6][1]
    else:
        label, chosen = 145, anchors[7][1]
    d = dict(chosen)
    d.setdefault('roll', float(os.environ.get('SCREW_PALM_ROLL','-1.2')))
    d.setdefault('pitch', float(os.environ.get('SCREW_PALM_PITCH','-0.8')))
    # Allow fast sector sweeps without touching frozen baseline scripts.
    # Example: SCREW_F110_35_T=0.06 SCREW_F110_35_R=-0.04.
    for k in list(d.keys()):
        v = os.environ.get(f'SCREW_F110_{label}_{k.upper()}')
        if v is None:
            v = os.environ.get(f'SCREW_F110_{k.upper()}')
        if v is not None:
            d[k] = float(v)
    return d


def apply_front110_sector_workspace(angle_rad):
    if os.environ.get('SCREW_FRONT110_SECTOR','0')!='1' or os.environ.get('SCREW_FRONT110_SECTOR_WORKSPACE','1')!='1':
        return
    deg=(math.degrees(angle_rad)%360.0)
    # Per-sector reachable windows. Center keeps the v470/v196 front baseline;
    # edge sectors get wider lateral reach so the wrist can actually bring the
    # thumb/index-middle mouth to the falling handle path.
    if deg < 40.0:
        vals=dict(xmin=-0.65,xmax=0.76,ymin=0.30,ymax=1.20)
    elif deg < 50.0:
        vals=dict(xmin=-0.65,xmax=0.74,ymin=0.32,ymax=1.20)
    elif deg < 65.0:
        vals=dict(xmin=-0.65,xmax=0.80,ymin=0.28,ymax=1.18)
    elif deg < 116.0:
        vals=dict(xmin=-0.65,xmax=0.65,ymin=0.35,ymax=1.16)
    elif deg < 136.0:
        vals=dict(xmin=-0.80,xmax=0.65,ymin=0.28,ymax=1.18)
    else:
        vals=dict(xmin=-0.86,xmax=0.65,ymin=0.25,ymax=1.18)
    os.environ['SCREW_TARGET_X_MIN']=str(vals['xmin'])
    os.environ['SCREW_TARGET_X_MAX']=str(vals['xmax'])
    os.environ['SCREW_TARGET_Y_MIN']=str(vals['ymin'])
    os.environ['SCREW_TARGET_Y_MAX']=str(vals['ymax'])

def angle_conditioned_grasp(angle_rad, default_lead):
    """Choose grasp-frame offsets from observed drop angle.

    The Revo2 hand is used as a gripper-like thumb/index-middle pinch.  These
    anchors are deliberately narrow: they are calibrated for the current clean
    front-arc falling-screwdriver task and avoid the earlier body-hit/fake-catch
    settings.
    """
    if os.environ.get('SCREW_ANGLE_CONDITIONED','0')!='1':
        iz_env=os.environ.get('SCREW_INTERCEPT_Z_OVERRIDE')
        pre_env=os.environ.get('SCREW_PREPOSITION_Z')
        return dict(
            yaw_offset=float(os.environ.get('SCREW_YAW_OFFSET','1.57')),
            static_t=float(os.environ.get('SCREW_STATIC_TANGENT','0.105')),
            static_r=float(os.environ.get('SCREW_STATIC_RADIAL','-0.035')),
            static_z=float(os.environ.get('SCREW_STATIC_Z','0.041')),
            lead=float(default_lead),
            intercept_z=float(iz_env) if iz_env is not None else float('nan'),
            pre_z=float(pre_env) if pre_env is not None else float('nan'),
        )
    deg=(math.degrees(angle_rad)%360.0)
    if os.environ.get('SCREW_FRONT110_SECTOR','0')=='1':
        d=front110_sector_params(angle_rad)
        return dict(yaw_offset=d['yaw'], static_t=d['t'], static_r=d['r'], static_z=d['z'], lead=d['lead'], intercept_z=d['iz'], pre_z=d['pre_z'], roll=d.get('roll'), pitch=d.get('pitch'))
    if os.environ.get('SCREW_SECTOR360_FULL','0')=='1':
        anchors=[
            (0.0,   dict(yaw=0.20,t=-0.020,r=-0.135,z=0.030,lead=0.20,iz=1.34,pre_z=1.48)),
            (45.0,  dict(yaw=0.62,t=-0.025,r=-0.085,z=0.024,lead=0.17,iz=1.36,pre_z=1.46)),
            (70.0,  dict(yaw=1.57,t=0.000,r=-0.055,z=0.018,lead=0.14,iz=1.30,pre_z=1.34)),
            (80.0,  dict(yaw=0.80,t=0.000,r=-0.055,z=0.018,lead=0.14,iz=1.30,pre_z=1.34)),
            (81.62, dict(yaw=0.80,t=0.000,r=-0.055,z=0.018,lead=0.14,iz=1.30,pre_z=1.34)),
            (82.0,  dict(yaw=0.80,t=0.000,r=-0.055,z=0.018,lead=0.14,iz=1.30,pre_z=1.34)),
            (84.0,  dict(yaw=0.80,t=0.000,r=-0.055,z=0.018,lead=0.14,iz=1.30,pre_z=1.34)),
            (86.0,  dict(yaw=0.80,t=0.000,r=-0.055,z=0.018,lead=0.14,iz=1.30,pre_z=1.34)),
            (100.0, dict(yaw=0.80,t=0.000,r=-0.055,z=0.018,lead=0.14,iz=1.30,pre_z=1.34)),
            (110.0, dict(yaw=1.20,t=-0.060,r=-0.055,z=0.018,lead=0.13,iz=1.36,pre_z=1.38)),
            (120.0, dict(yaw=1.70,t=-0.090,r=-0.035,z=0.010,lead=0.18,iz=1.48,pre_z=1.50)),
            (135.0, dict(yaw=0.80,t=-0.015,r=-0.075,z=0.020,lead=0.18,iz=1.38,pre_z=1.50)),
            (180.0, dict(yaw=0.45,t=-0.020,r=-0.135,z=0.028,lead=0.22,iz=1.34,pre_z=1.50)),
            (225.0, dict(yaw=-2.25,t=0.000,r=-0.120,z=0.030,lead=0.24,iz=1.34,pre_z=1.52)),
            (270.0, dict(yaw=-1.57,t=0.000,r=-0.130,z=0.030,lead=0.24,iz=1.34,pre_z=1.52)),
            (315.0, dict(yaw=-0.65,t=0.000,r=-0.120,z=0.030,lead=0.22,iz=1.34,pre_z=1.50)),
            (360.0, dict(yaw=0.20,t=-0.020,r=-0.135,z=0.030,lead=0.20,iz=1.34,pre_z=1.48)),
        ]
    else:
        anchors=[
            (70.0, dict(yaw=1.57,t=0.000,r=-0.055,z=0.018,lead=0.14,iz=1.30,pre_z=1.34)),
            (80.0, dict(yaw=0.80,t=0.000,r=-0.055,z=0.018,lead=0.14,iz=1.30,pre_z=1.34)),
            (81.62, dict(yaw=0.80,t=0.000,r=-0.055,z=0.018,lead=0.14,iz=1.30,pre_z=1.34)),
            (82.0, dict(yaw=0.80,t=0.000,r=-0.055,z=0.018,lead=0.14,iz=1.30,pre_z=1.34)),
            (84.0, dict(yaw=0.80,t=0.000,r=-0.055,z=0.018,lead=0.14,iz=1.30,pre_z=1.34)),
            (86.0, dict(yaw=0.80,t=0.000,r=-0.055,z=0.018,lead=0.14,iz=1.30,pre_z=1.34)),
            (100.0, dict(yaw=0.80,t=0.000,r=-0.055,z=0.018,lead=0.14,iz=1.30,pre_z=1.34)),
            (110.0, dict(yaw=1.20,t=-0.060,r=-0.055,z=0.018,lead=0.13,iz=1.36,pre_z=1.38)),
            (120.0, dict(yaw=1.70,t=-0.090,r=-0.035,z=0.010,lead=0.18,iz=1.48,pre_z=1.50)),
        ]
    if deg<=anchors[0][0]: lo=hi=anchors[0]
    elif deg>=anchors[-1][0]: lo=hi=anchors[-1]
    else:
        lo=anchors[0]; hi=anchors[-1]
        for a,b in zip(anchors,anchors[1:]):
            if a[0] <= deg <= b[0]: lo,hi=a,b; break
    u=0.0 if hi[0]==lo[0] else (deg-lo[0])/(hi[0]-lo[0])
    return dict(
        yaw_offset=_lerp(lo[1]['yaw'],hi[1]['yaw'],u),
        static_t=_lerp(lo[1]['t'],hi[1]['t'],u),
        static_r=_lerp(lo[1]['r'],hi[1]['r'],u),
        static_z=_lerp(lo[1]['z'],hi[1]['z'],u),
        lead=_lerp(lo[1]['lead'],hi[1]['lead'],u),
        intercept_z=_lerp(lo[1].get('iz', float('nan')),hi[1].get('iz', float('nan')),u),
        pre_z=_lerp(lo[1].get('pre_z', float('nan')),hi[1].get('pre_z', float('nan')),u),
    )

def make_env(base,num_envs,seed,enable_camera):
    dynamic_gym_root=Path(os.environ.get('DYNAMIC_GYM_ROOT', str(base)))
    asset_base=Path(os.environ.get('ISAACGYM_ASSET_BASE', str(base)))
    cfg_dir=dynamic_gym_root/'isaacgymenvs/cfg'
    with initialize_config_dir(config_dir=str(cfg_dir),version_base='1.1'):
        cfg=compose(config_name='config',overrides=[f'task={TASK}',f'train={TRAIN}','headless=True','capture_video=False','wandb_activate=False'])
    with open_dict(cfg):
        cfg.task.env.numEnvs=num_envs; cfg.task.env.robotAssetRoot=str(asset_base/os.environ.get('SCREW_ROBOT_ASSET_ROOT','assets/generated/franka_brainco_revo2_right')); cfg.task.env.dextoolbenchObjectAssetRoot=str(asset_base/os.environ.get('SCREW_OBJECT_ASSET_ROOT','assets/generated/falling_screwdriver_affordance_v01')); cfg.task.env.dextoolbenchObjectVariants=[{'label':'screwdriver_affordance','urdf':'screwdriver_affordance.urdf','random_yaw':False}]; cfg.task.env.fallingBatonRealRack=True; cfg.task.env.fallingBatonRealRackDynamic=True; cfg.task.env.fallingBatonRealRackLanes=[0.0 for _ in range(int(os.environ.get('SCREW_RING_COUNT','8')))] ; cfg.task.env.fallingBatonRackBarWidth=0.02; cfg.task.env.fallingBatonSpawnAbovePalmEnabled=False; cfg.task.env.fallingBatonPalmRelativeXYSpawnEnabled=False; cfg.task.env.fallingBatonForwardWorkspaceEnabled=True; cfg.task.env.fallingBatonForwardWorkspaceXRange=[float(os.environ.get('SCREW_WS_X_MIN','-0.95')),float(os.environ.get('SCREW_WS_X_MAX','0.75'))]; cfg.task.env.fallingBatonForwardWorkspaceYRange=[float(os.environ.get('SCREW_WS_Y_MIN','-0.20')),float(os.environ.get('SCREW_WS_Y_MAX','1.75'))]; cfg.task.env.dynamicGraspWorkspaceX=[float(os.environ.get('SCREW_WS_X_MIN','-0.95')),float(os.environ.get('SCREW_WS_X_MAX','0.75'))]; cfg.task.env.dynamicGraspWorkspaceY=[float(os.environ.get('SCREW_WS_Y_MIN','-0.20')),float(os.environ.get('SCREW_WS_Y_MAX','1.75'))]; cfg.task.env.fallingBatonExternalScriptedRelease=True; cfg.task.env.fallingBatonDropResetZ=0.05; cfg.task.env.enableCameraSensors=bool(enable_camera); cfg.task.env.force_render=bool(enable_camera); cfg.task.env.capture_video=False; cfg.task.env.captureVideo=False; cfg.task.env.videoSavePath=""; cfg.task.env.objectFriction=float(os.environ.get('SCREW_OBJECT_FRICTION','5.6')); cfg.task.env.fingerTipFriction=float(os.environ.get('SCREW_FINGERTIP_FRICTION','7.5')); cfg.task.env.robotHandStiffness=float(os.environ.get('SCREW_HAND_STIFFNESS','230')); cfg.task.env.robotHandDamping=float(os.environ.get('SCREW_HAND_DAMPING','20')); cfg.task.env.robotHandEffort=float(os.environ.get('SCREW_HAND_EFFORT','115')); cfg.task.env.robotHandVelocity=float(os.environ.get('SCREW_HAND_VELOCITY','8.5')); cfg.task.env.policyActionInterface='joint_target'; cfg.task.env.armMovingAverage=float(os.environ.get("SCREW_ARM_MOVING_AVG","0.96")); cfg.task.env.handMovingAverage=float(os.environ.get("SCREW_HAND_MOVING_AVG","0.74")); cfg.task.env.episodeLength=100000; cfg.seed=seed; cfg.sim_device='cpu'; cfg.rl_device='cpu'; cfg.graphics_device_id=0; cfg.pipeline='cpu'; cfg.headless=True; cfg.capture_video=False; cfg.force_render=bool(enable_camera)
    return isaacgymenvs.make(seed=seed,task=cfg.task.name,num_envs=num_envs,sim_device='cpu',rl_device='cpu',graphics_device_id=0,headless=True,force_render=bool(enable_camera),cfg=cfg)

def _cam_vec(name, default):
    vals=os.environ.get(name)
    if vals:
        xs=[float(x) for x in vals.replace(',', ' ').split()]
        if len(xs)==3: return xs
    return default

def setup_video(env,args):
    if not args.render_video: return None,None,None,None
    props=gymapi.CameraProperties(); props.width=int(os.environ.get('SCREW_CAM_W','1280')); props.height=int(os.environ.get('SCREW_CAM_H','720')); props.enable_tensors=False
    cam=env.gym.create_camera_sensor(env.envs[0],props)
    pos=_cam_vec('SCREW_CAM_POS',[0.88,-1.42,1.82]); tgt=_cam_vec('SCREW_CAM_TARGET',[0.0,0.18,1.18])
    env.gym.set_camera_location(cam,env.envs[0],gymapi.Vec3(*pos),gymapi.Vec3(*tgt))
    args.out_dir.mkdir(parents=True,exist_ok=True); vp=args.out_dir/f'falling_ring_screwdriver_catch_v490_seed{args.seed}.mp4'
    return cam,props,imageio.get_writer(str(vp),fps=30,codec='libx264',quality=7,macro_block_size=1),vp

def cap(env,cam,props,writer):
    if writer is None: return
    env.gym.step_graphics(env.sim); env.gym.render_all_camera_sensors(env.sim); img=env.gym.get_camera_image(env.sim,env.envs[0],cam,gymapi.IMAGE_COLOR); writer.append_data(np.asarray(img,dtype=np.uint8).reshape(props.height,props.width,4)[:,:,:3].copy())

def jac_servo(env,jac_t,body_idx,handle,target,max_step=0.14,yaw=0.0,roll=None,pitch=None):
    roll = float(os.environ.get('SCREW_PALM_ROLL','-1.2')) if roll is None else float(roll)
    pitch = float(os.environ.get('SCREW_PALM_PITCH','-0.8')) if pitch is None else float(pitch)
    env.gym.refresh_jacobian_tensors(env.sim)
    q=env.arm_hand_dof_pos[:,:env.num_arm_dofs].clone()
    state_handle=int(os.environ.get('SCREW_IK_STATE_HANDLE', str(handle)))
    pos=env.rigid_body_states[:,state_handle,0:3]
    pos_clip=float(os.environ.get('SCREW_IK_POS_CLIP','0.18'))
    pos_err=(target-pos).clamp(-pos_clip,pos_clip)
    tq=euler_quat(roll,pitch,yaw,env.device).unsqueeze(0).repeat(env.num_envs,1)
    cur=env.rigid_body_states[:,state_handle,3:7]
    qe=quat_mul(tq,quat_conj(cur))
    s=torch.where(qe[:,3:4]<0,-1.0,1.0)
    ori_clip=float(os.environ.get('SCREW_IK_ORI_CLIP','0.38'))
    ori=(2*s*qe[:,:3]*float(os.environ.get('SCREW_ORI_GAIN','0.38'))).clamp(-ori_clip,ori_clip)
    err=torch.cat((pos_err,ori),-1)
    j=jac_t[:,body_idx,0:6,:env.num_arm_dofs]
    jt=j.transpose(1,2)
    eye=torch.eye(6,device=env.device).unsqueeze(0).repeat(env.num_envs,1,1)
    damp=float(os.environ.get('SCREW_IK_DAMP','0.055'))
    dq=(jt@torch.linalg.solve(j@jt+damp*damp*eye,err.unsqueeze(-1))).squeeze(-1).clamp(-max_step,max_step)
    lo=env.arm_hand_dof_lower_limits[:env.num_arm_dofs]
    hi=env.arm_hand_dof_upper_limits[:env.num_arm_dofs]
    return torch.max(torch.min(q+dq,hi),lo)[0]
def set_object(env,pos,vel=(0,0,0),yaw=0.0):
    ids=torch.arange(env.num_envs,device=env.device,dtype=torch.long); idx=env.object_indices[ids].to(torch.int32); st=env.root_state_tensor[idx.long()].clone(); st[:,0:3]=torch.tensor(pos,device=env.device); st[:,7:10]=torch.tensor(vel,device=env.device); st[:,10:13]=0; st[:,3]=0; st[:,4]=0; st[:,5]=math.sin(yaw/2); st[:,6]=math.cos(yaw/2); env.root_state_tensor[idx.long()]=st; env.gym.set_actor_root_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.root_state_tensor),gymtorch.unwrap_tensor(idx),len(idx)); return idx

def front110_release_xy_bias(angle_rad):
    if os.environ.get('SCREW_FRONT110_SECTOR_RELEASE_BIAS', '1') != '1':
        return 0.0, 0.0
    deg = math.degrees(angle_rad) % 360.0
    # Boundary sectors need their own release geometry.  Without this, the
    # right edge (35-42 deg) passes too close to the robot body, while the
    # left edge (153-158 deg) falls just outside the gripper mouth.
    right = [(35.0, -0.18), (40.0, -0.10), (42.0, -0.10), (45.0, 0.0)]
    left = [(150.0, 0.0), (153.0, 0.12), (155.0, 0.18), (158.0, 0.18)]
    bx = 0.0
    for anchors in (right, left):
        if anchors[0][0] <= deg <= anchors[-1][0]:
            for (a0, x0), (a1, x1) in zip(anchors, anchors[1:]):
                if a0 <= deg <= a1:
                    u = 0.0 if a1 == a0 else (deg - a0) / (a1 - a0)
                    bx = x0 + (x1 - x0) * u
                    break
            break
    scale = float(os.environ.get('SCREW_FRONT110_RELEASE_BIAS_SCALE', '1.0'))
    bx += float(os.environ.get('SCREW_FRONT110_RELEASE_BIAS_X', '0.0'))
    by = float(os.environ.get('SCREW_FRONT110_RELEASE_BIAS_Y', '0.0'))
    return bx * scale, by

def ring_slots(args):
    explicit=os.environ.get('SCREW_RING_ANGLES_DEG','').strip()
    if explicit:
        angles=[math.radians(float(x)) for x in explicit.replace(',', ' ').split()]
    else:
        n=max(1,int(args.ring_count))
        if n==1:
            angles=[math.radians(args.fixed_angle_deg if args.fixed_angle_deg is not None else 90.0)]
        else:
            amin=math.radians(float(args.angle_min_deg)); amax=math.radians(float(args.angle_max_deg))
            if abs((amax-amin) - 2*math.pi) < math.radians(1.0):
                angles=[amin+2*math.pi*i/n for i in range(n)]
            else:
                angles=[amin+(amax-amin)*i/(n-1) for i in range(n)]
    radii_env=os.environ.get('SCREW_RING_RADII','').strip()
    if radii_env:
        radii=[float(x) for x in radii_env.replace(',', ' ').split()]
        if len(radii)==1:
            radii=radii*len(angles)
        if len(radii)!=len(angles):
            raise ValueError('SCREW_RING_RADII must have length 1 or match SCREW_RING_ANGLES_DEG')
    else:
        radii=[args.ring_radius]*len(angles)
    cx=float(os.environ.get('SCREW_RING_CENTER_X','0.0'))
    cy=float(os.environ.get('SCREW_RING_CENTER_Y','0.0'))
    slots=[]
    for r,a in zip(radii,angles):
        bx,by=front110_release_xy_bias(a)
        slots.append((cx+bx+r*math.cos(a), cy+by+r*math.sin(a), args.ring_z, a))
    return slots

def pin_ring_rack(env, slots, active_slot=None):
    rack=getattr(env,'falling_baton_rack_baton_indices',None)
    if rack is None:
        return
    if torch.is_tensor(rack):
        ids=rack.flatten().detach().to(dtype=torch.int32,device=env.device).tolist()
    elif isinstance(rack,(list,tuple)):
        if len(rack)==0:
            return
        first=rack[0]
        if torch.is_tensor(first):
            ids=first.flatten().detach().to(dtype=torch.int32,device=env.device).tolist()
        else:
            ids=list(first) if first is not None else []
    else:
        return
    ids=ids[:len(slots)]
    if len(ids)==0:
        return
    idx=torch.tensor(ids,dtype=torch.int32,device=env.device)
    st=env.root_state_tensor[idx.long()].clone()
    hidden=set()
    if active_slot is not None:
        if isinstance(active_slot,(set,list,tuple)):
            hidden={int(x) for x in active_slot}
        else:
            hidden={int(active_slot)}
    for i,(x,y,z,a) in enumerate(slots[:len(ids)]):
        if i in hidden:
            st[i,0:3]=torch.tensor([x,y,-2.0],device=env.device)
        else:
            st[i,0:3]=torch.tensor([x,y,z],device=env.device)
        yaw=a+math.pi/2+float(os.environ.get("SCREW_OBJECT_YAW_OFFSET","0.0"))
        st[i,3]=0; st[i,4]=0; st[i,5]=math.sin(yaw/2); st[i,6]=math.cos(yaw/2)
        st[i,7:13]=0
    env.root_state_tensor[idx.long()]=st
    env.gym.set_actor_root_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.root_state_tensor),gymtorch.unwrap_tensor(idx),len(ids))

def handle_pos(st):
    local=torch.tensor([float(os.environ.get("SCREW_HANDLE_LOCAL_X","0.0")),float(os.environ.get("SCREW_HANDLE_LOCAL_Y","0.0")),HANDLE_LOCAL_Z],device=st.device).unsqueeze(0).repeat(st.shape[0],1)
    return st[:,0:3]+quat_apply(st[:,3:7],local)

def non_hand_robot_body_handles(env):
    """Robot body links that should not be credited as catching contacts."""
    handles=[]
    for name, idx in getattr(env, "rigid_body_name_to_idx", {}).items():
        if not name.startswith("robot/"):
            continue
        body=name.split("/", 1)[1]
        if body.startswith("revo2_right_"):
            continue
        handles.append(int(idx))
    return torch.tensor(sorted(set(handles)), dtype=torch.long, device=env.device)

def vec_env3(name, default, device):
    vals=_vec_env(name, default)
    return torch.tensor([vals],device=device)


def force_open_hand_state(env):
    """Hard-reset only Revo2 hand DOFs/targets to the clean open pose between drops."""
    if os.environ.get("SCREW_HARD_RESET_HAND", "1") != "1":
        return
    h=hand_cmd("open", env.device)
    n=min(h.numel(), env.num_hand_arm_dofs-env.num_arm_dofs)
    hs=slice(env.num_arm_dofs, env.num_arm_dofs+n)
    env.arm_hand_dof_pos[:, hs]=h[:n].unsqueeze(0).repeat(env.num_envs,1)
    env.arm_hand_dof_vel[:, hs]=0.0
    cur=env.arm_hand_dof_pos[:, :env.num_hand_arm_dofs].clone()
    env.cur_targets[:, :env.num_hand_arm_dofs]=cur
    if hasattr(env, "prev_targets"):
        env.prev_targets[:, :env.num_hand_arm_dofs]=cur
    ids=torch.arange(env.num_envs,device=env.device)
    robot_indices=env.robot_indices[ids].to(torch.int32)
    env.gym.set_dof_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env.dof_state), gymtorch.unwrap_tensor(robot_indices), len(robot_indices))
    env.gym.set_dof_position_target_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env.cur_targets), gymtorch.unwrap_tensor(robot_indices), len(robot_indices))
    env.gym.refresh_dof_state_tensor(env.sim)

def force_robot_home_state(env, home_dof):
    """Optional hard reset of the full FR3/Revo2 DOF state between ring drops."""
    if os.environ.get("SCREW_HARD_RESET_ROBOT", "0") != "1" or home_dof is None:
        return
    home=home_dof.to(env.device).clone()
    env.arm_hand_dof_pos[:, :env.num_hand_arm_dofs]=home
    env.arm_hand_dof_vel[:, :env.num_hand_arm_dofs]=0.0
    env.cur_targets[:, :env.num_hand_arm_dofs]=home
    if hasattr(env, "prev_targets"):
        env.prev_targets[:, :env.num_hand_arm_dofs]=home
    ids=torch.arange(env.num_envs,device=env.device)
    robot_indices=env.robot_indices[ids].to(torch.int32)
    env.gym.set_dof_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env.dof_state), gymtorch.unwrap_tensor(robot_indices), len(robot_indices))
    env.gym.set_dof_position_target_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env.cur_targets), gymtorch.unwrap_tensor(robot_indices), len(robot_indices))
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)

def inter_episode_release_reset(env, jac, body, servo, center, slots, hidden_slots, hidden_pos, cam, props, writer, home_dof=None, release_angle=0.0, next_ready_target=None, next_ready_yaw=0.0, next_ready_roll=None, next_ready_pitch=None):
    """Open, shake/drop the old tool visibly, then clean up before the next random drop."""
    steps=int(os.environ.get("SCREW_INTER_EP_RELEASE_STEPS", "48"))
    settle=int(os.environ.get("SCREW_INTER_EP_SETTLE_STEPS", "10"))
    visible_drop=int(os.environ.get("SCREW_GAME_VISIBLE_DROP_STEPS", "34"))
    max_step=float(os.environ.get("SCREW_RESET_ARM_MAX_STEP", "0.20"))
    record=os.environ.get("SCREW_RECORD_RESET", "1") == "1"
    ca, sa = math.cos(float(release_angle)), math.sin(float(release_angle))
    tangent = torch.tensor([[-sa, ca, 0.0]], device=env.device)
    radial = torch.tensor([[ca, sa, 0.0]], device=env.device)
    for k in range(max(0, visible_drop)):
        pin_ring_rack(env, slots, hidden_slots)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        sign=-1.0 if (k//4)%2 else 1.0
        zsign=-1.0 if (k//6)%2 else 1.0
        rsign=-1.0 if (k//5)%2 else 1.0
        shake=(sign*float(os.environ.get("SCREW_GAME_RELEASE_TANGENT_AMP", "0.075"))*tangent
               + rsign*float(os.environ.get("SCREW_GAME_RELEASE_RADIAL_AMP", "0.040"))*radial
               + zsign*float(os.environ.get("SCREW_GAME_RELEASE_Z_AMP", "0.050"))*torch.tensor([[0.0,0.0,1.0]],device=env.device))
        yaw_shake=float(release_angle)+float(os.environ.get("SCREW_GAME_RELEASE_YAW_OFFSET", "1.57"))+sign*float(os.environ.get("SCREW_GAME_RELEASE_YAW_SHAKE", "0.28"))
        roll_shake=float(os.environ.get("SCREW_GAME_RELEASE_ROLL", os.environ.get("SCREW_PALM_ROLL", "-1.2"))) + sign*float(os.environ.get("SCREW_GAME_RELEASE_ROLL_SHAKE", "0.24"))
        pitch_shake=float(os.environ.get("SCREW_GAME_RELEASE_PITCH", os.environ.get("SCREW_PALM_PITCH", "-0.8"))) + zsign*float(os.environ.get("SCREW_GAME_RELEASE_PITCH_SHAKE", "0.18"))
        arm=jac_servo(env, jac, body, servo, center+shake, max_step=max_step, yaw=yaw_shake, roll=roll_shake, pitch=pitch_shake)
        step_with_targets(env, arm, "release")
        if record:
            cap(env, cam, props, writer)
    # Cleanup after the viewer has seen the old screwdriver leave the hand. Hide spent active object
    # so it cannot block the next random drop; this is between-episode cleanup, not catch assistance.
    for _ in range(steps + settle):
        set_object(env, hidden_pos, (0,0,0), yaw=0.0)
        pin_ring_rack(env, slots, hidden_slots)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        arm=jac_servo(env, jac, body, servo, center, max_step=max_step, yaw=0.0)
        step_with_targets(env, arm, "release")
        if record:
            cap(env, cam, props, writer)
    force_open_hand_state(env)
    force_robot_home_state(env, home_dof)
    for _ in range(int(os.environ.get("SCREW_HARD_RESET_SETTLE_STEPS", "10"))):
        set_object(env, hidden_pos, (0,0,0), yaw=0.0)
        pin_ring_rack(env, slots, hidden_slots)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        arm=jac_servo(env, jac, body, servo, center, max_step=max_step, yaw=0.0)
        step_with_targets(env, arm, "release")
        if record:
            cap(env, cam, props, writer)
    if next_ready_target is not None and os.environ.get('SCREW_NEXT_SECTOR_READY', '1') == '1':
        next_steps = int(os.environ.get('SCREW_NEXT_SECTOR_READY_STEPS', '54'))
        next_settle = int(os.environ.get('SCREW_NEXT_SECTOR_READY_SETTLE_STEPS', '10'))
        next_step = float(os.environ.get('SCREW_NEXT_SECTOR_READY_ARM_MAX_STEP', os.environ.get('SCREW_PRE_ARM_MAX_STEP', '0.20')))
        state_handle = int(os.environ.get('SCREW_IK_STATE_HANDLE', str(servo)))
        start = env.rigid_body_states[:, state_handle, 0:3].clone()
        for k in range(max(0, next_steps) + max(0, next_settle)):
            set_object(env, hidden_pos, (0,0,0), yaw=0.0)
            pin_ring_rack(env, slots, hidden_slots)
            env.gym.refresh_actor_root_state_tensor(env.sim)
            env.gym.refresh_rigid_body_state_tensor(env.sim)
            env.gym.refresh_dof_state_tensor(env.sim)
            if next_steps > 0 and k < next_steps and os.environ.get('SCREW_NEXT_SECTOR_READY_INTERP', '1') == '1':
                u = float(k + 1) / float(next_steps)
                u = u*u*(3.0-2.0*u)
                ready = start*(1.0-u) + next_ready_target*u
            else:
                ready = next_ready_target
            arm = jac_servo(env, jac, body, servo, ready, max_step=next_step, yaw=next_ready_yaw, roll=next_ready_roll, pitch=next_ready_pitch)
            step_with_targets(env, arm, 'open')
            if os.environ.get('SCREW_RECORD_NEXT_READY','0')=='1':
                cap(env, cam, props, writer)
    pin_ring_rack(env, slots, hidden_slots)

def main():
    args=args_parser(); os.environ['SCREW_RING_COUNT']=str(args.ring_count); os.environ.setdefault('SCREW_TARGET_Y_MAX','0.82'); rng=random.Random(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed); base=Path(__file__).resolve().parents[1]; args.out_dir=(base/args.out_dir) if not args.out_dir.is_absolute() else args.out_dir; args.out_dir.mkdir(parents=True,exist_ok=True); env=make_env(base,args.num_envs,args.seed,args.render_video); jac=gymtorch.wrap_tensor(env.gym.acquire_jacobian_tensor(env.sim,'robot')); servo=int(env.palm_handle); body=int(os.environ.get('SCREW_JAC_BODY_IDX', '9')); body=max(0,min(body,int(jac.shape[1])-1)); arm_body_handles=non_hand_robot_body_handles(env); cam,props,writer,vpath=setup_video(env,args); rows=[]; summaries=[]; common_hook=maybe_create_common_dataset_hook(args,env); slots=ring_slots(args); hidden_slots=set(); drop_order=[]; round_id=0; center=torch.tensor([[float(os.environ.get('SCREW_WAIT_X','0.0')),float(os.environ.get('SCREW_WAIT_Y','0.48')),float(os.environ.get('SCREW_WAIT_Z','1.34'))]],device=env.device)
    # Hide and pin the default environment object during warmup. Otherwise the viewer sees
    # an unrelated screwdriver fall before the actual front-drop episode starts.
    hidden_pos=(float(os.environ.get('SCREW_HIDDEN_X','0.0')),float(os.environ.get('SCREW_HIDDEN_Y','2.6')),float(os.environ.get('SCREW_HIDDEN_Z','-2.0')))
    hidden_idx=set_object(env,hidden_pos,(0,0,0),yaw=0.0)
    for _ in range(int(os.environ.get("SCREW_WARMUP_STEPS","35"))):
        pin_ring_rack(env,slots,None)
        set_object(env,hidden_pos,(0,0,0),yaw=0.0)
        env.gym.refresh_actor_root_state_tensor(env.sim); env.gym.refresh_rigid_body_state_tensor(env.sim); env.gym.refresh_dof_state_tensor(env.sim)
        arm=jac_servo(env,jac,body,servo,center,max_step=float(os.environ.get('SCREW_WARMUP_ARM_MAX_STEP','0.11')),yaw=0.0); step_with_targets(env,arm,'open')
        if os.environ.get('SCREW_RECORD_WARMUP','0')=='1': cap(env,cam,props,writer)
    home_dof=env.arm_hand_dof_pos[:, :env.num_hand_arm_dofs].clone()
    for ep in range(args.episodes):
        if common_hook is not None:
            common_hook.start_episode(ep, len(slots))
        if ep > 0 and os.environ.get('SCREW_EP_START_CLEAN_HOME', '0') == '1':
            # Front-wide sector changes are sensitive to inherited wrist pose.
            # Reset only robot/hand state and keep all object slots managed by the game loop.
            force_open_hand_state(env)
            if os.environ.get('SCREW_EP_START_HARD_ROBOT_HOME', '1') == '1':
                force_robot_home_state(env, home_dof)
            reset_steps = int(os.environ.get('SCREW_EP_START_CLEAN_STEPS', '20'))
            reset_step = float(os.environ.get('SCREW_EP_START_CLEAN_ARM_MAX_STEP', '0.14'))
            for reset_i in range(max(0, reset_steps)):
                pin_ring_rack(env, slots, hidden_slots)
                set_object(env, hidden_pos, (0,0,0), yaw=0.0)
                env.gym.refresh_actor_root_state_tensor(env.sim)
                env.gym.refresh_rigid_body_state_tensor(env.sim)
                env.gym.refresh_dof_state_tensor(env.sim)
                arm = jac_servo(env, jac, body, servo, center, max_step=reset_step, yaw=0.0)
                step_with_targets(env, arm, 'open')
                if os.environ.get('SCREW_RECORD_EP_START_RESET','0')=='1':
                    cap(env, cam, props, writer)
        yaw_cmd=0.0
        if not drop_order:
            hidden_slots=set()
            drop_order=rng.sample(range(len(slots)),len(slots))
            round_id+=1
        active_slot=drop_order.pop(0); hidden_slots.add(active_slot); x,y,z,angle=slots[active_slot]
        jitter_xy=float(os.environ.get('SCREW_RELEASE_JITTER_XY','0.0'))
        if jitter_xy>0:
            x += rng.uniform(-jitter_xy,jitter_xy); y += rng.uniform(-jitter_xy,jitter_xy)
        observed_angle=angle
        yaw_noise=math.radians(rng.uniform(-float(os.environ.get('SCREW_RANDOM_YAW_DEG','0.0')), float(os.environ.get('SCREW_RANDOM_YAW_DEG','0.0'))))
        apply_front110_sector_workspace(observed_angle)
        grasp_cfg=angle_conditioned_grasp(observed_angle,args.lead_time); yaw_cmd=observed_angle+grasp_cfg['yaw_offset']; pre_roll_cmd=grasp_cfg.get('roll'); pre_pitch_cmd=grasp_cfg.get('pitch'); pin_ring_rack(env,slots,hidden_slots); idx=set_object(env,(x,y,z),(0,0,0),yaw=observed_angle+math.pi/2+float(os.environ.get("SCREW_OBJECT_YAW_OFFSET","0.0"))+yaw_noise)
        pre_release_target=center
        if os.environ.get('SCREW_FRONT110_SECTOR_READY_PREPOSITION', '0')=='1' and os.environ.get('SCREW_FRONT110_SECTOR','0')=='1':
            pre_radial=torch.tensor([[math.cos(observed_angle),math.sin(observed_angle),0.0]],device=env.device)
            pre_tangent=torch.tensor([[-math.sin(observed_angle),math.cos(observed_angle),0.0]],device=env.device)
            cfg_pre_z=grasp_cfg.get('pre_z', float('nan'))
            pre_z=float(os.environ.get('SCREW_PREPOSITION_Z', str(cfg_pre_z if math.isfinite(cfg_pre_z) else args.intercept_z)))
            pre_release_target=torch.tensor([[x,y,pre_z]],device=env.device)+grasp_cfg['static_t']*pre_tangent+grasp_cfg['static_r']*pre_radial+torch.tensor([[0.0,0.0,grasp_cfg['static_z']+float(os.environ.get('SCREW_PREPOSITION_Z_BIAS','0.0'))]],device=env.device)
            pre_release_target[:,0]=torch.clamp(pre_release_target[:,0],float(os.environ.get('SCREW_TARGET_X_MIN','-0.60')),float(os.environ.get('SCREW_TARGET_X_MAX','0.60')))
            pre_release_target[:,1]=torch.clamp(pre_release_target[:,1],float(os.environ.get('SCREW_TARGET_Y_MIN','-0.05')),float(os.environ.get('SCREW_TARGET_Y_MAX','0.68')))
            pre_release_target[:,2]=torch.clamp(pre_release_target[:,2],float(os.environ.get('SCREW_TARGET_Z_MIN','0.82')),float(os.environ.get('SCREW_TARGET_Z_MAX','1.72')))
        if os.environ.get('SCREW_ANGLE_PREPOSITION','0')=='1':
            pre_radial=torch.tensor([[math.cos(observed_angle),math.sin(observed_angle),0.0]],device=env.device)
            pre_tangent=torch.tensor([[-math.sin(observed_angle),math.cos(observed_angle),0.0]],device=env.device)
            cfg_pre_z=grasp_cfg.get('pre_z', float('nan'))
            pre_z=float(os.environ.get('SCREW_PREPOSITION_Z', str(cfg_pre_z if math.isfinite(cfg_pre_z) else args.intercept_z)))
            pre_release_target=torch.tensor([[x,y,pre_z]],device=env.device)+grasp_cfg['static_t']*pre_tangent+grasp_cfg['static_r']*pre_radial+torch.tensor([[0.0,0.0,grasp_cfg['static_z']+float(os.environ.get('SCREW_PREPOSITION_Z_BIAS','0.0'))]],device=env.device)
            pre_release_target[:,0]=torch.clamp(pre_release_target[:,0],float(os.environ.get('SCREW_TARGET_X_MIN','-0.60')),float(os.environ.get('SCREW_TARGET_X_MAX','0.60')))
            pre_release_target[:,1]=torch.clamp(pre_release_target[:,1],float(os.environ.get('SCREW_TARGET_Y_MIN','-0.05')),float(os.environ.get('SCREW_TARGET_Y_MAX','0.68')))
            pre_release_target[:,2]=torch.clamp(pre_release_target[:,2],float(os.environ.get('SCREW_TARGET_Z_MIN','0.82')),float(os.environ.get('SCREW_TARGET_Z_MAX','1.72')))
        sector_ready_steps=int(os.environ.get("SCREW_SECTOR_READY_STEPS", os.environ.get("SCREW_PRE_RELEASE_STEPS","18")))
        sector_settle_steps=int(os.environ.get("SCREW_SECTOR_READY_SETTLE_STEPS","0"))
        deg_ready = math.degrees(observed_angle) % 360.0
        right_edge_ready = 30.0 <= deg_ready < 50.0
        left_edge_ready = 145.0 <= deg_ready <= 152.0
        if right_edge_ready and os.environ.get('SCREW_RIGHT_EDGE_READY_OVERRIDE','1')=='1':
            sector_ready_steps=int(os.environ.get('SCREW_RIGHT_EDGE_READY_STEPS', str(max(sector_ready_steps, 42))))
            sector_settle_steps=int(os.environ.get('SCREW_RIGHT_EDGE_READY_SETTLE_STEPS', str(max(sector_settle_steps, 10))))
        if left_edge_ready and os.environ.get('SCREW_LEFT_EDGE_READY_OVERRIDE','0')=='1':
            # v490 front-wide: stabilize the far-left edge after cross-sector transitions only.
            sector_ready_steps=int(os.environ.get('SCREW_LEFT_EDGE_READY_STEPS', str(max(sector_ready_steps, 54))))
            sector_settle_steps=int(os.environ.get('SCREW_LEFT_EDGE_READY_SETTLE_STEPS', str(max(sector_settle_steps, 14))))
        state_handle=int(os.environ.get('SCREW_IK_STATE_HANDLE', str(servo)))
        sector_start=env.rigid_body_states[:, state_handle, 0:3].clone()
        if os.environ.get('SCREW_SECTOR_READY_CENTER_START','0')=='1':
            sector_start=center.clone()
        total_ready=max(1, sector_ready_steps)
        for ready_i in range(total_ready + max(0, sector_settle_steps)):
            pin_ring_rack(env,slots,hidden_slots)
            st=env.root_state_tensor[idx.long()].clone()
            st[:,0:3]=torch.tensor([x,y,z],device=env.device)
            st[:,7:13]=0
            env.root_state_tensor[idx.long()]=st
            env.gym.set_actor_root_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.root_state_tensor),gymtorch.unwrap_tensor(idx),len(idx))
            env.gym.refresh_actor_root_state_tensor(env.sim); env.gym.refresh_rigid_body_state_tensor(env.sim); env.gym.refresh_dof_state_tensor(env.sim)
            if os.environ.get('SCREW_SECTOR_READY_INTERP','0')=='1' and ready_i < total_ready:
                u=float(ready_i + 1)/float(total_ready)
                u=u*u*(3.0-2.0*u)
                ready_target=sector_start*(1.0-u)+pre_release_target*u
            else:
                ready_target=pre_release_target
            # Use a slightly slower ready move after cross-sector reset only; the dynamic
            # catch path remains the known-good v490/v441-style local IK controller.
            pre_step=float(os.environ.get('SCREW_PRE_ARM_MAX_STEP','0.12'))
            if right_edge_ready and os.environ.get('SCREW_RIGHT_EDGE_READY_OVERRIDE','1')=='1':
                pre_step=float(os.environ.get('SCREW_RIGHT_EDGE_PRE_ARM_MAX_STEP', str(max(pre_step, 0.24))))
            if left_edge_ready and os.environ.get('SCREW_LEFT_EDGE_READY_OVERRIDE','0')=='1':
                pre_step=float(os.environ.get('SCREW_LEFT_EDGE_PRE_ARM_MAX_STEP', str(max(pre_step, 0.20))))
            arm=jac_servo(env,jac,body,servo,ready_target,max_step=pre_step,yaw=yaw_cmd,roll=pre_roll_cmd,pitch=pre_pitch_cmd)
            step_with_targets(env,arm,'open')
            if os.environ.get('SCREW_RECORD_PRERELEASE','0')=='1': cap(env,cam,props,writer)
        # Re-apply the clean front release immediately after pre-release arm motion.
        # The underlying Dynamic_Gym task may reset objects during env.step; this keeps
        # the first dynamic frame from being a hidden/default spawn.
        idx=set_object(env,(x,y,z),(0,0,0),yaw=observed_angle+math.pi/2+float(os.environ.get("SCREW_OBJECT_YAW_OFFSET","0.0"))+yaw_noise)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        success=False; first=None; hold_count=0; pre_latch_cushion_step=None; pre_latch_cushion_target=None; two_stage_contact_count=0; two_stage_started_step=None; two_stage_lock_axis=None; two_stage_lock_target=None; two_stage_lock_handle_z=None; two_stage_lock_down_speed=None; two_stage_lock_sign_thumb=None; two_stage_lock_sign_im=None; strict_seen=0; lost_pinch_count=0; settle_target=None; settle_step=None; grasp_latched=False; latch_step=None; latched_target=None; latched_handle_offset=None; latched_side_axis=None; min_handle_palm=9.0; min_tip=9.0; min_arm_body_dist=9.0; prev_speed=9.0; prev_target=center.clone(); had_dynamic_fall=False; max_down_speed=0.0; first_dynamic_step=None; observed_drop_angle=None; catch_target=None; catch_lock_step=None; clean_fall_active=(os.environ.get('SCREW_SCRIPTED_CLEAN_FALL','0')=='1'); clean_fall_stop_step=None; sim_dt=float(getattr(env,'dt',1.0/60.0))*float(os.environ.get('SCREW_CLEAN_FALL_DT_SCALE','1.0'))
        for step in range(args.steps):
            wrist_roll_cmd=float(os.environ.get('SCREW_PALM_ROLL','-1.2'))
            wrist_pitch_cmd=float(os.environ.get('SCREW_PALM_PITCH','-0.8'))
            wrist_yaw_offset=0.0
            pin_ring_rack(env,slots,hidden_slots)
            env.gym.refresh_actor_root_state_tensor(env.sim); env.gym.refresh_rigid_body_state_tensor(env.sim); env.gym.refresh_dof_state_tensor(env.sim); st=env.root_state_tensor[idx.long()];
            if clean_fall_active:
                # Match the successful single-front-drop task: keep the active screwdriver on
                # its observed slot and cap vertical speed until the gripper mouth reaches it.
                t=max(0.0, step*sim_dt-float(os.environ.get('SCREW_CLEAN_FALL_DELAY','0.0')))
                g=float(os.environ.get('SCREW_CLEAN_FALL_G','9.81'))
                yaw=observed_angle+math.pi/2+float(os.environ.get('SCREW_OBJECT_YAW_OFFSET','0.0'))+yaw_noise
                root_z=max(float(os.environ.get('SCREW_CLEAN_FALL_MIN_ROOT_Z','0.40')), float(z)-0.5*g*t*t)
                root_vz=-g*t
                max_v=float(os.environ.get('SCREW_CLEAN_FALL_MAX_DOWN_SPEED','4.2'))
                if max_v>0: root_vz=max(root_vz, -max_v)
                st=st.clone(); st[:,0]=float(x); st[:,1]=float(y); st[:,2]=root_z; st[:,7]=0; st[:,8]=0; st[:,9]=root_vz; st[:,10:13]=0; st[:,3]=0; st[:,4]=0; st[:,5]=math.sin(yaw/2); st[:,6]=math.cos(yaw/2)
                env.root_state_tensor[idx.long()]=st
                env.gym.set_actor_root_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.root_state_tensor),gymtorch.unwrap_tensor(idx),len(idx))
                env.gym.refresh_actor_root_state_tensor(env.sim)
                st=env.root_state_tensor[idx.long()]
            if os.environ.get('SCREW_STATIC_HANDLE','0')=='1':
                st=st.clone(); st[:,0:3]=torch.tensor([x,y,args.intercept_z-HANDLE_LOCAL_Z],device=env.device); st[:,7:13]=0; env.root_state_tensor[idx.long()]=st; env.gym.set_actor_root_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.root_state_tensor),gymtorch.unwrap_tensor(idx),len(idx))
            hp=handle_pos(st); vel=st[:,7:10]
            if observed_drop_angle is None and step>=int(os.environ.get('SCREW_OBSERVE_DELAY_STEPS','2')):
                observed_drop_angle=math.atan2(float(st[0,1]),float(st[0,0]))
            observed_angle=observed_drop_angle if observed_drop_angle is not None else math.atan2(float(st[0,1]),float(st[0,0]))
            grasp_cfg=angle_conditioned_grasp(observed_angle,args.lead_time); yaw_cmd=observed_angle+grasp_cfg['yaw_offset']
            if grasp_cfg.get('roll') is not None:
                wrist_roll_cmd=float(grasp_cfg['roll'])
            if grasp_cfg.get('pitch') is not None:
                wrist_pitch_cmd=float(grasp_cfg['pitch'])
            z0=float(hp[0,2]); vz=float(vel[0,2]); g=9.81; cfg_iz=grasp_cfg.get('intercept_z', float('nan')); iz=float(os.environ.get('SCREW_INTERCEPT_Z_OVERRIDE', str(cfg_iz if math.isfinite(cfg_iz) else args.intercept_z))); disc=max(vz*vz+2*g*max(z0-iz,0.0),0.0); t=max(0.0,(vz+math.sqrt(disc))/g); t=min(max(t+grasp_cfg['lead'],0.04),float(os.environ.get('SCREW_MAX_PRED_TIME','0.75'))); vel_xy=torch.clamp(vel[:,0:2],-float(os.environ.get('SCREW_XY_VEL_CLAMP','0.035')),float(os.environ.get('SCREW_XY_VEL_CLAMP','0.035'))); pred=hp.clone(); pred[:,0:2]=hp[:,0:2]+vel_xy*t*float(os.environ.get('SCREW_XY_PRED_SCALE','0.25')); pred[:,2]=iz
            if os.environ.get("SCREW_DYNAMIC_Z_TRACK","0")=="1":
                z_follow_t=float(os.environ.get("SCREW_Z_FOLLOW_TIME","0.045"))
                z_low=float(os.environ.get("SCREW_Z_TRACK_MIN",str(iz-0.18)))
                z_high=float(os.environ.get("SCREW_Z_TRACK_MAX",str(iz+0.10)))
                z_bias=float(os.environ.get("SCREW_Z_TRACK_BIAS","0.015"))
                pred[:,2]=torch.clamp(hp[:,2]+vel[:,2]*z_follow_t+z_bias,z_low,z_high)
            radial=torch.tensor([[math.cos(observed_angle),math.sin(observed_angle),0.0]],device=env.device); tangent=torch.tensor([[-math.sin(observed_angle),math.cos(observed_angle),0.0]],device=env.device); palm=env.rigid_body_states[:,servo,0:3]; ik_body_idx=int(os.environ.get('SCREW_IK_STATE_HANDLE', str(servo))); tips=env.rigid_body_states[:,env.fingertip_handles,0:3]; thumb=tips[:,0,:]; im=(tips[:,1,:]+tips[:,2,:])*0.5; pinch=(0.50*thumb+0.50*im); pocket=(0.45*thumb+0.45*im+0.10*palm); arm_body_dist=torch.linalg.norm(env.rigid_body_states[:,arm_body_handles,0:3]-hp.unsqueeze(1),dim=-1).min() if len(arm_body_handles)>0 else torch.tensor(9.0,device=env.device); min_arm_body_dist=min(min_arm_body_dist,float(arm_body_dist)); aim=pinch if os.environ.get('SCREW_AIM_PINCH','0')=='1' else pocket; pocket_offset=torch.clamp(aim-palm, -float(os.environ.get('SCREW_POCKET_OFFSET_CLAMP','0.32')), float(os.environ.get('SCREW_POCKET_OFFSET_CLAMP','0.32'))); target=pred-float(os.environ.get("SCREW_POCKET_OFFSET_SCALE","1.00"))*pocket_offset+float(os.environ.get("SCREW_RADIAL_BIAS","0.000"))*radial+float(os.environ.get("SCREW_TANGENT_BIAS","0.000"))*tangent+torch.tensor([[0.0,0.0,float(os.environ.get("SCREW_Z_BIAS","0.0"))]],device=env.device);
            if os.environ.get('SCREW_USE_AFFORDANCE_FRAME','0')=='1':
                target=pred+float(os.environ.get('SCREW_AFFORD_TANGENT','-0.096'))*tangent+float(os.environ.get('SCREW_AFFORD_RADIAL','-0.054'))*radial+torch.tensor([[0.0,0.0,float(os.environ.get('SCREW_AFFORD_Z','0.013'))]],device=env.device)
            if os.environ.get('SCREW_USE_STATIC_GRASP_FRAME','0')=='1':
                # Static true-pinch calibration: place handle between thumb and index/middle, not against hand back.
                base_t=grasp_cfg['static_t']
                base_r=grasp_cfg['static_r']
                base_z=grasp_cfg['static_z']
                target=pred+base_t*tangent+base_r*radial+torch.tensor([[0.0,0.0,base_z]],device=env.device)
            if os.environ.get('SCREW_RESTORE_HISTORICAL_TRAJ','0')=='1':
                # Recreate the known-good dynamic single-object controller: use the calibrated
                # grasp frame in XY, but keep a high pre-intercept wrist target until contact.
                hist_z=float(os.environ.get('SCREW_HIST_PRE_TARGET_Z','1.72'))
                target=pred+grasp_cfg['static_t']*tangent+grasp_cfg['static_r']*radial+torch.tensor([[0.0,0.0,grasp_cfg['static_z']]],device=env.device)
                target[:,2]=hist_z
            target[:,0]=torch.clamp(target[:,0],float(os.environ.get('SCREW_TARGET_X_MIN','-0.60')),float(os.environ.get('SCREW_TARGET_X_MAX','0.60'))); target[:,1]=torch.clamp(target[:,1],float(os.environ.get('SCREW_TARGET_Y_MIN','-0.05')),float(os.environ.get('SCREW_TARGET_Y_MAX','0.68'))); target[:,2]=torch.clamp(target[:,2],float(os.environ.get('SCREW_TARGET_Z_MIN','0.82')),float(os.environ.get('SCREW_TARGET_Z_MAX','1.72'))); palm_d=torch.linalg.norm(palm-hp,dim=-1); pocket_d=torch.linalg.norm(pocket-hp,dim=-1); tip_d=torch.linalg.norm(tips-hp.unsqueeze(1),dim=-1); min_handle_palm=min(min_handle_palm,float(pocket_d[0])); min_tip=min(min_tip,float(tip_d[0].min())); thumb_rel=thumb-hp; im_rel=im-hp; opp=torch.sum(thumb_rel*im_rel,dim=-1)<-0.00015; side_axis=torch.nn.functional.normalize(im-thumb,dim=-1); control_side_axis=sector_pinch_axis(os.environ.get('SCREW_SECTOR_PINCH_AXIS','current'),radial,tangent,side_axis,observed_angle,env.device); handle_side_thumb=torch.sum((thumb-hp)*side_axis,dim=-1); handle_side_im=torch.sum((im-hp)*side_axis,dim=-1); control_side_thumb=torch.sum((thumb-hp)*control_side_axis,dim=-1); control_side_im=torch.sum((im-hp)*control_side_axis,dim=-1); real_pinch=(handle_side_thumb< -float(os.environ.get('SCREW_SIDE_GATE','0.0005'))) & (handle_side_im>float(os.environ.get('SCREW_SIDE_GATE','0.0005'))); pinch_center=(thumb+im)*0.5; pinch_center_d=torch.linalg.norm(pinch_center-hp,dim=-1); palm_to_pinch=torch.linalg.norm(pinch_center-palm,dim=-1); handle_in_pinch_pocket=pinch_center_d < torch.clamp(palm_to_pinch*float(os.environ.get('SCREW_PINCH_CENTER_RATIO','0.78')), max=float(os.environ.get('SCREW_PINCH_CENTER_MAX','0.09'))); finger_pair_close=(torch.linalg.norm(thumb_rel,dim=-1)<float(os.environ.get('SCREW_STRICT_THUMB_GATE','0.09'))) & (torch.linalg.norm(im_rel,dim=-1)<float(os.environ.get('SCREW_STRICT_IM_GATE','0.09'))); palm_not_backstop=(palm_d>float(os.environ.get('SCREW_PALM_BACKSTOP_MIN','0.035'))) & (pinch_center_d < palm_d+float(os.environ.get('SCREW_PINCH_BETTER_THAN_PALM_MARGIN','0.010'))); strict_palm_grasp=real_pinch & handle_in_pinch_pocket & finger_pair_close & palm_not_backstop; bias_mode=os.environ.get('SCREW_GRASP_SIDE_AXIS','tangent'); bias_axis=tangent if bias_mode=='tangent' else (radial if bias_mode=='radial' else side_axis); side_center=0.5*(handle_side_thumb+handle_side_im); same_side=(handle_side_thumb*handle_side_im)>0; side_corr=torch.where(same_side, -side_center*float(os.environ.get('SCREW_SIDE_CENTER_GAIN','1.15')), torch.zeros_like(side_center)); side_corr=torch.clamp(side_corr,-float(os.environ.get('SCREW_SIDE_CENTER_MAX','0.075')),float(os.environ.get('SCREW_SIDE_CENTER_MAX','0.075'))).unsqueeze(-1); oppose_corr=torch.zeros_like(side_corr);
            deg_now=math.degrees(observed_angle)%360.0
            right_edge_pinching_active=(os.environ.get('SCREW_RIGHT_EDGE_PINCH_TRACK','1')=='1' and deg_now>=float(os.environ.get('SCREW_RIGHT_EDGE_PINCH_MIN_DEG','30')) and deg_now<=float(os.environ.get('SCREW_RIGHT_EDGE_PINCH_MAX_DEG','50')))
            sector_pinching_active=((os.environ.get('SCREW_SECTOR_PINCH_TRACK','0')=='1' and deg_now>=float(os.environ.get('SCREW_SECTOR_PINCH_MIN_DEG','105')) and deg_now<=float(os.environ.get('SCREW_SECTOR_PINCH_MAX_DEG','165'))) or right_edge_pinching_active)
            if os.environ.get('SCREW_OPPOSE_CORRECT','0')=='1':
                close_pair=(torch.linalg.norm(thumb_rel,dim=-1)<float(os.environ.get('SCREW_OPPOSE_CORRECT_GATE','0.16'))) & (torch.linalg.norm(im_rel,dim=-1)<float(os.environ.get('SCREW_OPPOSE_CORRECT_GATE','0.16')))
                desired_mid=float(os.environ.get('SCREW_OPPOSE_DESIRED_MID','0.014'))
                both_pos=(handle_side_thumb>0) & (handle_side_im>0) & close_pair
                both_neg=(handle_side_thumb<0) & (handle_side_im<0) & close_pair
                oppose_corr=torch.where(both_pos, -(torch.minimum(handle_side_thumb,handle_side_im)+desired_mid)*float(os.environ.get('SCREW_OPPOSE_GAIN','1.0')), oppose_corr.squeeze(-1))
                oppose_corr=torch.where(both_neg, -(torch.maximum(handle_side_thumb,handle_side_im)-desired_mid)*float(os.environ.get('SCREW_OPPOSE_GAIN','1.0')), oppose_corr).clamp(-float(os.environ.get('SCREW_OPPOSE_MAX','0.10')),float(os.environ.get('SCREW_OPPOSE_MAX','0.10'))).unsqueeze(-1)
            height_err=(hp[:,2]-pinch_center[:,2]).clamp(-0.10,0.10).unsqueeze(-1); target=target+float(os.environ.get('SCREW_GRASP_SIDE_BIAS','0.04'))*bias_axis+(side_corr+oppose_corr)*bias_axis+float(os.environ.get('SCREW_PINCH_Z_GAIN','0.35'))*height_err*torch.tensor([[0.0,0.0,1.0]],device=env.device)+float(os.environ.get('SCREW_PALM_RADIAL_BACKOFF','0.0'))*radial
            if os.environ.get('SCREW_PALM_SIDE_SERVO','0')=='1':
                # Once the handle is near the palm/fingers, align it between thumb and index/middle
                # instead of letting it hit the same side of the hand and count as a fake block.
                near_side=(torch.linalg.norm(thumb_rel,dim=-1)<float(os.environ.get('SCREW_SIDE_SERVO_NEAR_GATE','0.14'))) & (torch.linalg.norm(im_rel,dim=-1)<float(os.environ.get('SCREW_SIDE_SERVO_NEAR_GATE','0.14')))
                same_pos=(handle_side_thumb>0) & (handle_side_im>0) & near_side
                same_neg=(handle_side_thumb<0) & (handle_side_im<0) & near_side
                desired=float(os.environ.get('SCREW_SIDE_SERVO_DESIRED','0.018'))
                raw=torch.zeros_like(handle_side_thumb)
                raw=torch.where(same_pos, -(torch.minimum(handle_side_thumb,handle_side_im)+desired), raw)
                raw=torch.where(same_neg, -(torch.maximum(handle_side_thumb,handle_side_im)-desired), raw)
                raw=torch.where((~(same_pos|same_neg)) & near_side, -0.5*(handle_side_thumb+handle_side_im), raw)
                target=target+(raw*float(os.environ.get('SCREW_SIDE_SERVO_GAIN','2.4'))).clamp(-float(os.environ.get('SCREW_SIDE_SERVO_MAX','0.18')),float(os.environ.get('SCREW_SIDE_SERVO_MAX','0.18'))).unsqueeze(-1)*side_axis
                if os.environ.get('SCREW_NEAR_Z_FOLLOW','1')=='1':
                    z_near=near_side.unsqueeze(-1)
                    desired_z=torch.clamp(hp[:,2:3]+float(os.environ.get('SCREW_NEAR_Z_BIAS','0.105')),float(os.environ.get('SCREW_TARGET_Z_MIN','0.82')),float(os.environ.get('SCREW_TARGET_Z_MAX','1.72')))
                    target[:,2:3]=torch.where(z_near, 0.35*target[:,2:3]+0.65*desired_z, target[:,2:3])
            target[:,0]=torch.clamp(target[:,0],float(os.environ.get('SCREW_TARGET_X_MIN','-0.60')),float(os.environ.get('SCREW_TARGET_X_MAX','0.60'))); target[:,1]=torch.clamp(target[:,1],float(os.environ.get('SCREW_TARGET_Y_MIN','-0.05')),float(os.environ.get('SCREW_TARGET_Y_MAX','0.68'))); target[:,2]=torch.clamp(target[:,2],float(os.environ.get('SCREW_TARGET_Z_MIN','0.82')),float(os.environ.get('SCREW_TARGET_Z_MAX','1.72')))
            if os.environ.get('SCREW_RIGHT_EDGE_ENTRY_CORRIDOR','1')=='1' and 30.0 <= deg_now < 50.0 and not grasp_latched:
                # Isolated 35/45 deg corridor: keep the palm/IK target out of the robot body
                # and high enough for a real opposed pinch before the handle drops below the hand.
                x_margin=float(os.environ.get('SCREW_RIGHT_EDGE_X_MARGIN','0.055'))
                x_cap=float(os.environ.get('SCREW_RIGHT_EDGE_X_CAP','0.700'))
                x_floor=float(os.environ.get('SCREW_RIGHT_EDGE_X_FLOOR','0.520'))
                y_back=float(os.environ.get('SCREW_RIGHT_EDGE_Y_BACKOFF','0.006'))
                y_ahead=float(os.environ.get('SCREW_RIGHT_EDGE_Y_AHEAD','0.070'))
                z_min=float(os.environ.get('SCREW_RIGHT_EDGE_Z_MIN','0.86'))
                z_max=float(os.environ.get('SCREW_RIGHT_EDGE_Z_MAX','1.50'))
                if bool(had_dynamic_fall) or step >= int(os.environ.get('SCREW_OBSERVE_DELAY_STEPS','2')):
                    target[:,0]=torch.clamp(target[:,0], max(x_floor, float(hp[0,0])-float(os.environ.get('SCREW_RIGHT_EDGE_X_BEHIND','0.015'))), min(x_cap, float(hp[0,0])+x_margin))
                    target[:,1]=torch.clamp(target[:,1], float(hp[0,1])-y_back, float(hp[0,1])+y_ahead)
                    if float(hp[0,2]) < float(os.environ.get('SCREW_RIGHT_EDGE_Z_APPLY_BELOW','1.78')):
                        target[:,2]=torch.clamp(target[:,2], max(z_min, float(hp[0,2])+float(os.environ.get('SCREW_RIGHT_EDGE_Z_ABOVE_HANDLE','0.055'))), min(z_max, float(hp[0,2])+float(os.environ.get('SCREW_RIGHT_EDGE_Z_ABOVE_HANDLE_MAX','0.22'))))

            if os.environ.get('SCREW_SECTOR55_ENTRY_CORRIDOR','1')=='1' and 50.0 <= deg_now <= 62.0 and not grasp_latched:
                # Right-front sector needs a conservative entry corridor. After a
                # previous left/front catch, local IK corrections can push the
                # target too far +X, turning contact into a same-side graze.
                x_margin=float(os.environ.get('SCREW_SECTOR55_X_MARGIN','0.095'))
                x_cap=float(os.environ.get('SCREW_SECTOR55_X_CAP','0.622'))
                x_floor=float(os.environ.get('SCREW_SECTOR55_X_FLOOR','0.500'))
                y_back=float(os.environ.get('SCREW_SECTOR55_Y_BACKOFF','0.012'))
                z_min=float(os.environ.get('SCREW_SECTOR55_Z_MIN','1.16'))
                z_max=float(os.environ.get('SCREW_SECTOR55_Z_MAX','1.54'))
                dynamic_phase = bool(had_dynamic_fall) or step >= int(os.environ.get('SCREW_OBSERVE_DELAY_STEPS','2'))
                if dynamic_phase:
                    target[:,0]=torch.clamp(target[:,0], max(x_floor, float(hp[0,0])-0.020), min(x_cap, float(hp[0,0])+x_margin))
                    target[:,1]=torch.clamp(target[:,1], float(hp[0,1])-y_back, float(hp[0,1])+float(os.environ.get('SCREW_SECTOR55_Y_AHEAD','0.050')))
                    if float(hp[0,2]) < float(os.environ.get('SCREW_SECTOR55_Z_APPLY_BELOW','1.72')):
                        target[:,2]=torch.clamp(target[:,2], z_min, z_max)

            if os.environ.get('SCREW_CATCH_FRAME_LOCK','0')=='1' and observed_drop_angle is not None:
                if catch_target is None:
                    catch_target=target.clone()
                    catch_lock_step=step
                    if os.environ.get('SCREW_CATCH_LOCK_Z','')!='':
                        catch_target[:,2]=float(os.environ.get('SCREW_CATCH_LOCK_Z'))
                target=catch_target.clone()
                if os.environ.get('SCREW_CATCH_LOCK_FOLLOW_Z','0')=='1':
                    target[:,2]=torch.clamp(hp[:,2]+float(os.environ.get('SCREW_CATCH_LOCK_Z_BIAS','0.08')),float(os.environ.get('SCREW_TARGET_Z_MIN','0.82')),float(os.environ.get('SCREW_TARGET_Z_MAX','1.72')))

            if os.environ.get('SCREW_VISUAL_SERVO','0')=='1' and step>=int(os.environ.get('SCREW_OBSERVE_DELAY_STEPS','2')):
                servo_gain=float(os.environ.get('SCREW_VISUAL_SERVO_GAIN','0.82'))
                max_corr=float(os.environ.get('SCREW_VISUAL_SERVO_MAX','0.16'))
                z_gain=float(os.environ.get('SCREW_VISUAL_SERVO_Z_GAIN','0.55'))
                servo_point = pinch_center if os.environ.get("SCREW_VISUAL_SERVO_POINT","pocket")=="pinch" else pocket
                err=torch.clamp(hp-servo_point,-max_corr,max_corr)
                target=target+servo_gain*torch.cat((err[:,:2],z_gain*err[:,2:3]),dim=-1)
                if os.environ.get('SCREW_DYNAMIC_LOW_POCKET','0')=='1':
                    target=target+vec_env3('SCREW_DYNAMIC_LOW_BIAS',[0.0,0.0,-0.13],env.device)
                target[:,0]=torch.clamp(target[:,0],float(os.environ.get('SCREW_TARGET_X_MIN','-0.60')),float(os.environ.get('SCREW_TARGET_X_MAX','0.60')))
                target[:,1]=torch.clamp(target[:,1],float(os.environ.get('SCREW_TARGET_Y_MIN','-0.05')),float(os.environ.get('SCREW_TARGET_Y_MAX','0.68')))
                target[:,2]=torch.clamp(target[:,2],float(os.environ.get('SCREW_TARGET_Z_MIN','0.82')),float(os.environ.get('SCREW_TARGET_Z_MAX','1.72')))
            if os.environ.get('SCREW_DIRECT_PINCH_TRACK','0')=='1' and step>=int(os.environ.get('SCREW_OBSERVE_DELAY_STEPS','2')) and not grasp_latched:
                # Directly servo the thumb/index-middle pinch mouth onto the observed handle.
                # This avoids the common failure where the wrist/palm reaches the object but the handle
                # remains on the same side of both fingers, visually looking like a block instead of a grasp.
                track_axis=control_side_axis if sector_pinching_active else side_axis
                track_thumb=control_side_thumb if sector_pinching_active else handle_side_thumb
                track_im=control_side_im if sector_pinching_active else handle_side_im
                dmax=float(os.environ.get('SCREW_DIRECT_PINCH_MAX','0.26'))
                dgxy=float(os.environ.get('SCREW_DIRECT_PINCH_XY_GAIN','1.35'))
                dgz=float(os.environ.get('SCREW_DIRECT_PINCH_Z_GAIN','0.90'))
                derr=torch.clamp(hp-pinch_center,-dmax,dmax)
                direct=palm+torch.cat((derr[:,:2]*dgxy,derr[:,2:3]*dgz),dim=-1)
                desired=float(os.environ.get('SCREW_DIRECT_PINCH_SIDE_GAP','0.030'))
                same_pos=(track_thumb>0) & (track_im>0)
                same_neg=(track_thumb<0) & (track_im<0)
                draw=torch.zeros_like(track_thumb)
                draw=torch.where(same_pos, -(torch.minimum(track_thumb,track_im)+desired), draw)
                draw=torch.where(same_neg, -(torch.maximum(track_thumb,track_im)-desired), draw)
                draw=torch.where(~(same_pos|same_neg), -0.5*(track_thumb+track_im)*float(os.environ.get('SCREW_DIRECT_PINCH_CENTER_SIDE_GAIN','0.65')), draw)
                side=(draw*float(os.environ.get('SCREW_DIRECT_PINCH_SIDE_GAIN','3.2'))*float(os.environ.get('SCREW_DIRECT_PINCH_SIDE_SIGN','1.0'))).clamp(-float(os.environ.get('SCREW_DIRECT_PINCH_SIDE_MAX','0.24')),float(os.environ.get('SCREW_DIRECT_PINCH_SIDE_MAX','0.24'))).unsqueeze(-1)
                direct=direct+side*track_axis+torch.tensor([[0.0,0.0,float(os.environ.get('SCREW_DIRECT_PINCH_Z_BIAS','-0.010'))]],device=env.device)
                blend=float(os.environ.get('SCREW_DIRECT_PINCH_BLEND','0.72'))
                target=(1.0-blend)*target+blend*direct
                target[:,0]=torch.clamp(target[:,0],float(os.environ.get('SCREW_TARGET_X_MIN','-0.60')),float(os.environ.get('SCREW_TARGET_X_MAX','0.60')))
                target[:,1]=torch.clamp(target[:,1],float(os.environ.get('SCREW_TARGET_Y_MIN','-0.05')),float(os.environ.get('SCREW_TARGET_Y_MAX','0.68')))
                target[:,2]=torch.clamp(target[:,2],float(os.environ.get('SCREW_TARGET_Z_MIN','0.82')),float(os.environ.get('SCREW_TARGET_Z_MAX','1.72')))
            if os.environ.get('SCREW_FRONT_PINCH_WORKSPACE','0')=='1' and not (sector_pinching_active and os.environ.get('SCREW_FRONT_PINCH_DISABLE_IN_SECTOR','1')=='1') and step>=int(os.environ.get('SCREW_OBSERVE_DELAY_STEPS','2')) and not grasp_latched:
                # Front-drop mode: first pull the Revo2 palm workspace out in front of the robot,
                # then servo the thumb/index-middle mouth around the observed handle.  This is stricter
                # than wrist chasing: the controller must keep the handle between opposing fingertips.
                fmax=float(os.environ.get('SCREW_FRONT_PINCH_MAX','0.30'))
                fxy=float(os.environ.get('SCREW_FRONT_PINCH_XY_GAIN','1.65'))
                fz=float(os.environ.get('SCREW_FRONT_PINCH_Z_GAIN','1.05'))
                ferr=torch.clamp(hp-pinch_center,-fmax,fmax)
                front_target=palm+torch.cat((ferr[:,:2]*fxy,ferr[:,2:3]*fz),dim=-1)
                front_target[:,1]=torch.maximum(front_target[:,1], hp[:,1]-float(os.environ.get('SCREW_FRONT_PALM_BACKOFF_Y','0.055')))
                front_target[:,2]=torch.clamp(hp[:,2]+float(os.environ.get('SCREW_FRONT_PINCH_Z_BIAS','-0.015')),float(os.environ.get('SCREW_TARGET_Z_MIN','0.82')),float(os.environ.get('SCREW_TARGET_Z_MAX','1.72')))
                front_axis_mode=os.environ.get('SCREW_FRONT_PINCH_AXIS','tangent')
                goal_axis=tangent if front_axis_mode=='tangent' else (control_side_axis if front_axis_mode in ('sector','control') else side_axis)
                goal_thumb=torch.sum((thumb-hp)*goal_axis,dim=-1)
                goal_im=torch.sum((im-hp)*goal_axis,dim=-1)
                desired=float(os.environ.get('SCREW_FRONT_PINCH_SIDE_GAP','0.032'))
                both_pos=(goal_thumb>0) & (goal_im>0)
                both_neg=(goal_thumb<0) & (goal_im<0)
                fraw=torch.zeros_like(goal_thumb)
                fraw=torch.where(both_pos, -(torch.minimum(goal_thumb,goal_im)+desired), fraw)
                fraw=torch.where(both_neg, -(torch.maximum(goal_thumb,goal_im)-desired), fraw)
                fraw=torch.where(~(both_pos|both_neg), -0.5*(goal_thumb+goal_im)*float(os.environ.get('SCREW_FRONT_PINCH_CENTER_GAIN','0.85')), fraw)
                fside=(fraw*float(os.environ.get('SCREW_FRONT_PINCH_SIDE_GAIN','4.4'))).clamp(-float(os.environ.get('SCREW_FRONT_PINCH_SIDE_MAX','0.30')),float(os.environ.get('SCREW_FRONT_PINCH_SIDE_MAX','0.30'))).unsqueeze(-1)
                front_target=front_target+fside*goal_axis
                blend=float(os.environ.get('SCREW_FRONT_PINCH_BLEND','0.88'))
                target=(1.0-blend)*target+blend*front_target
                target[:,0]=torch.clamp(target[:,0],float(os.environ.get('SCREW_TARGET_X_MIN','-0.70')),float(os.environ.get('SCREW_TARGET_X_MAX','0.70')))
                target[:,1]=torch.clamp(target[:,1],float(os.environ.get('SCREW_TARGET_Y_MIN','-0.05')),float(os.environ.get('SCREW_TARGET_Y_MAX','1.12')))
                target[:,2]=torch.clamp(target[:,2],float(os.environ.get('SCREW_TARGET_Z_MIN','0.82')),float(os.environ.get('SCREW_TARGET_Z_MAX','1.72')))
            if os.environ.get('SCREW_FINGERTIP_CLOSED_LOOP','0')=='1' and os.environ.get('SCREW_FTCL_PRE','1')=='1' and step>=int(os.environ.get('SCREW_OBSERVE_DELAY_STEPS','2')) and not grasp_latched:
                ft_axis=control_side_axis if sector_pinching_active else side_axis
                ft_thumb=control_side_thumb if sector_pinching_active else handle_side_thumb
                ft_im=control_side_im if sector_pinching_active else handle_side_im
                target=fingertip_closed_loop_target(env, ik_body_idx, target, hp, vel, thumb, im, palm, pinch_center, ft_axis, ft_thumb, ft_im, False)
                target[:,0]=torch.clamp(target[:,0],float(os.environ.get('SCREW_TARGET_X_MIN','-0.70')),float(os.environ.get('SCREW_TARGET_X_MAX','0.70')))
                target[:,1]=torch.clamp(target[:,1],float(os.environ.get('SCREW_TARGET_Y_MIN','-0.05')),float(os.environ.get('SCREW_TARGET_Y_MAX','1.60')))
                target[:,2]=torch.clamp(target[:,2],float(os.environ.get('SCREW_TARGET_Z_MIN','0.82')),float(os.environ.get('SCREW_TARGET_Z_MAX','1.82')))
            if os.environ.get('SCREW_SECTOR55_LATE_ENTRY_CORRIDOR','0')=='1' and 50.0 <= deg_now <= 62.0 and not grasp_latched:
                # Final pre-latch bound after direct/front/FTCL corrections.
                # This is the actual target sent to IK for the right-front edge.
                x_margin=float(os.environ.get('SCREW_SECTOR55_LATE_X_MARGIN','0.095'))
                x_cap=float(os.environ.get('SCREW_SECTOR55_LATE_X_CAP','0.622'))
                x_floor=float(os.environ.get('SCREW_SECTOR55_LATE_X_FLOOR','0.500'))
                y_back=float(os.environ.get('SCREW_SECTOR55_LATE_Y_BACKOFF','0.012'))
                z_min=float(os.environ.get('SCREW_SECTOR55_LATE_Z_MIN','1.16'))
                z_max=float(os.environ.get('SCREW_SECTOR55_LATE_Z_MAX','1.54'))
                if bool(had_dynamic_fall) or step >= int(os.environ.get('SCREW_OBSERVE_DELAY_STEPS','2')):
                    target[:,0]=torch.clamp(target[:,0], max(x_floor, float(hp[0,0])-0.020), min(x_cap, float(hp[0,0])+x_margin))
                    target[:,1]=torch.clamp(target[:,1], float(hp[0,1])-y_back, float(hp[0,1])+float(os.environ.get('SCREW_SECTOR55_LATE_Y_AHEAD','0.050')))
                    if float(hp[0,2]) < float(os.environ.get('SCREW_SECTOR55_LATE_Z_APPLY_BELOW','1.72')):
                        target[:,2]=torch.clamp(target[:,2], z_min, z_max)

            falling_now=(vel[:,2] < -float(os.environ.get("SCREW_DYNAMIC_MIN_VZ","0.35"))) & (hp[:,2] < args.ring_z-float(os.environ.get("SCREW_DYNAMIC_MIN_DROP","0.08")))
            if bool(falling_now.item()) and first_dynamic_step is None: first_dynamic_step=step
            had_dynamic_fall = had_dynamic_fall or bool(falling_now.item())
            max_down_speed=max(max_down_speed, max(0.0, -float(vel[0,2])))
            near=strict_palm_grasp; pocket_gate=(pocket_d<float(os.environ.get("SCREW_POCKET_GATE","0.075")))&(torch.abs((hp-pocket)[:,2])<0.085); speed=torch.linalg.norm(vel,dim=-1); contact_ready=bool((pocket_gate&near).item()); observed=step>=int(os.environ.get('SCREW_OBSERVE_DELAY_STEPS','2')); early_pre=bool(observed and float(hp[0,2]) < args.ring_z-float(os.environ.get('SCREW_PRE_DROP_Z','0.03'))); approaching=bool((float(pocket_d[0])<float(os.environ.get('SCREW_PRE_GATE','0.48'))) and (float(hp[0,2])<iz+0.85)); force_close=bool(observed and os.environ.get('SCREW_FORCE_CLOSE_AFTER','')!='' and step>=int(os.environ.get('SCREW_FORCE_CLOSE_AFTER'))); close_ready=bool(force_close or contact_ready or (observed and float(hp[0,2])<args.intercept_z+float(os.environ.get('SCREW_CLOSE_Z_WINDOW','0.42')) and (float(pocket_d[0])<float(os.environ.get("SCREW_CLOSE_GATE","0.22")) or os.environ.get('SCREW_CLOSE_BY_Z','0')=='1'))); phase="close" if close_ready else ("pre" if (early_pre or approaching) else "open");
            if right_edge_pinching_active and os.environ.get('SCREW_RIGHT_EDGE_TIMED_CLOSE','1')=='1' and not grasp_latched:
                z_gap=abs(float(pinch_center[0,2]-hp[0,2]))
                safe_body = float(arm_body_dist) > float(os.environ.get('SCREW_RIGHT_EDGE_CLOSE_BODY_CLEARANCE','0.075'))
                right_close = (
                    safe_body
                    and bool(real_pinch[0])
                    and z_gap < float(os.environ.get('SCREW_RIGHT_EDGE_CLOSE_Z_GAP','0.22'))
                    and float(pinch_center_d[0]) < float(os.environ.get('SCREW_RIGHT_EDGE_CLOSE_CENTER_GATE','0.22'))
                )
                phase = 'close' if right_close else ('pre' if observed else 'open')
            if os.environ.get("SCREW_REQUIRE_OPPOSED_TO_CLOSE", "0") == "1" and phase == "close" and not bool(real_pinch[0]):
                phase = "pre"
            real_latch_gate=(real_pinch & (pinch_center_d<float(os.environ.get("SCREW_REAL_LATCH_PINCH_GATE","0.085"))) & (torch.linalg.norm(thumb_rel,dim=-1)<float(os.environ.get("SCREW_REAL_LATCH_THUMB_GATE","0.095"))) & (torch.linalg.norm(im_rel,dim=-1)<float(os.environ.get("SCREW_REAL_LATCH_IM_GATE","0.095"))) & (hp[:,2]>float(os.environ.get("SCREW_LATCH_MIN_Z","0.86"))))
            near_latch_gate=((pinch_center_d<float(os.environ.get("SCREW_NEAR_LATCH_PINCH_GATE","0.050"))) & (torch.linalg.norm(thumb_rel,dim=-1)<float(os.environ.get("SCREW_NEAR_LATCH_THUMB_GATE","0.060"))) & (torch.linalg.norm(im_rel,dim=-1)<float(os.environ.get("SCREW_NEAR_LATCH_IM_GATE","0.060"))) & (hp[:,2]>float(os.environ.get("SCREW_LATCH_MIN_Z","0.86"))) & (speed<float(os.environ.get("SCREW_NEAR_LATCH_SPEED_GATE","2.40")))) if os.environ.get("SCREW_NEAR_LATCH_ON","0")=="1" else torch.zeros_like(strict_palm_grasp)
            gripper_latch_gate=((real_pinch) & (pinch_center_d<float(os.environ.get("SCREW_GRIPPER_LATCH_PINCH_GATE","0.105"))) & (torch.linalg.norm(thumb_rel,dim=-1)<float(os.environ.get("SCREW_GRIPPER_LATCH_THUMB_GATE","0.120"))) & (torch.linalg.norm(im_rel,dim=-1)<float(os.environ.get("SCREW_GRIPPER_LATCH_IM_GATE","0.120"))) & (hp[:,2]>float(os.environ.get("SCREW_LATCH_MIN_Z","0.70"))) & (speed<float(os.environ.get("SCREW_GRIPPER_LATCH_SPEED_GATE","1.80"))) & (arm_body_dist>float(os.environ.get("SCREW_GRIPPER_LATCH_BODY_CLEARANCE","0.085"))) & torch.tensor([had_dynamic_fall],device=env.device)) if os.environ.get("SCREW_GRIPPER_LATCH_MODE","0")=="1" else torch.zeros_like(strict_palm_grasp)
            sector55_edge_latch_gate=torch.zeros_like(strict_palm_grasp)
            if os.environ.get("SCREW_SECTOR55_EDGE_LATCH","1")=="1" and 50.0 <= deg_now <= 62.0:
                # v490: the 55-degree sector reaches the handle safely, but the
                # generic latch gate misses the contact while the tool is still
                # moving fast. Use a sector-local gate so the normal two-stage
                # cushion/hold logic can take over after real opposing contact.
                sector55_edge_latch_gate=(real_pinch & (pinch_center_d<float(os.environ.get("SCREW_SECTOR55_EDGE_LATCH_CENTER_GATE","0.24"))) & (torch.linalg.norm(thumb_rel,dim=-1)<float(os.environ.get("SCREW_SECTOR55_EDGE_LATCH_THUMB_GATE","0.30"))) & (torch.linalg.norm(im_rel,dim=-1)<float(os.environ.get("SCREW_SECTOR55_EDGE_LATCH_IM_GATE","0.30"))) & (hp[:,2]>float(os.environ.get("SCREW_SECTOR55_EDGE_LATCH_MIN_Z","0.95"))) & (hp[:,2]<float(os.environ.get("SCREW_SECTOR55_EDGE_LATCH_MAX_Z","1.60"))) & (speed<float(os.environ.get("SCREW_SECTOR55_EDGE_LATCH_SPEED_GATE","3.50"))) & (arm_body_dist>float(os.environ.get("SCREW_SECTOR55_EDGE_LATCH_BODY_CLEARANCE","0.09"))) & torch.tensor([had_dynamic_fall],device=env.device))
            right_edge_latch_gate=torch.zeros_like(strict_palm_grasp)
            if os.environ.get("SCREW_RIGHT_EDGE_LATCH","1")=="1" and 30.0 <= deg_now < 50.0:
                right_edge_latch_gate=(real_pinch & (pinch_center_d<float(os.environ.get("SCREW_RIGHT_EDGE_LATCH_CENTER_GATE","0.18"))) & (torch.linalg.norm(thumb_rel,dim=-1)<float(os.environ.get("SCREW_RIGHT_EDGE_LATCH_THUMB_GATE","0.24"))) & (torch.linalg.norm(im_rel,dim=-1)<float(os.environ.get("SCREW_RIGHT_EDGE_LATCH_IM_GATE","0.24"))) & (hp[:,2]>float(os.environ.get("SCREW_RIGHT_EDGE_LATCH_MIN_Z","0.62"))) & (hp[:,2]<float(os.environ.get("SCREW_RIGHT_EDGE_LATCH_MAX_Z","1.50"))) & (speed<float(os.environ.get("SCREW_RIGHT_EDGE_LATCH_SPEED_GATE","3.35"))) & (arm_body_dist>float(os.environ.get("SCREW_RIGHT_EDGE_LATCH_BODY_CLEARANCE","0.085"))) & torch.tensor([had_dynamic_fall],device=env.device))
            left_edge_latch_gate=torch.zeros_like(strict_palm_grasp)
            if os.environ.get("SCREW_LEFT_EDGE_LATCH","1")=="1" and 138.0 <= deg_now <= 158.0:
                # Left-front edge can present a clean opposed pinch around 14-18cm before
                # the generic strict gate fires. Treat that as contact so the two-stage
                # down-follow/hold controller takes over instead of letting the handle fall past.
                left_edge_latch_gate=(real_pinch & (pinch_center_d<float(os.environ.get("SCREW_LEFT_EDGE_LATCH_CENTER_GATE","0.17"))) & (torch.linalg.norm(thumb_rel,dim=-1)<float(os.environ.get("SCREW_LEFT_EDGE_LATCH_THUMB_GATE","0.19"))) & (torch.linalg.norm(im_rel,dim=-1)<float(os.environ.get("SCREW_LEFT_EDGE_LATCH_IM_GATE","0.19"))) & (hp[:,2]>float(os.environ.get("SCREW_LEFT_EDGE_LATCH_MIN_Z","1.05"))) & (hp[:,2]<float(os.environ.get("SCREW_LEFT_EDGE_LATCH_MAX_Z","1.58"))) & (speed<float(os.environ.get("SCREW_LEFT_EDGE_LATCH_SPEED_GATE","3.60"))) & (arm_body_dist>float(os.environ.get("SCREW_LEFT_EDGE_LATCH_BODY_CLEARANCE","0.16"))) & torch.tensor([had_dynamic_fall],device=env.device))
            latch_gate=(strict_palm_grasp&(pocket_d<float(os.environ.get("SCREW_LATCH_POCKET_GATE","0.10")))&(hp[:,2]>float(os.environ.get("SCREW_LATCH_MIN_Z","0.86")))) | (real_latch_gate if os.environ.get("SCREW_LATCH_ON_REAL","0")=="1" else torch.zeros_like(strict_palm_grasp)) | ((real_pinch & (hp[:,2]>float(os.environ.get("SCREW_LATCH_MIN_Z","0.86")))) if os.environ.get("SCREW_LATCH_ON_ANY_REAL","0")=="1" else torch.zeros_like(strict_palm_grasp)) | near_latch_gate | gripper_latch_gate | sector55_edge_latch_gate | right_edge_latch_gate | left_edge_latch_gate; latch_now=bool(latch_gate.item());
            if latch_now and not grasp_latched:
                lost_pinch_count=0
                grasp_latched=True; latch_step=step
                if os.environ.get('SCREW_LATCH_TARGET_CURRENT_PALM','0')=='1':
                    # Keep wrist at first contact so fingers can close before the arm resumes tracking.
                    latched_target=palm.clone()+vec_env3('SCREW_LATCH_PALM_OFFSET',[0.0,0.0,0.0],env.device)
                elif os.environ.get('SCREW_LATCH_TARGET_GRASP_FRAME','0')=='1':
                    latched_target=hp+float(os.environ.get('SCREW_STATIC_TANGENT','0.105'))*tangent+float(os.environ.get('SCREW_STATIC_RADIAL','-0.035'))*radial+torch.tensor([[0.0,0.0,float(os.environ.get('SCREW_STATIC_Z','0.041'))+float(os.environ.get("SCREW_HOLD_Z_BIAS","0.010"))]],device=env.device)
                else:
                    latched_target=target.clone()
                latched_handle_offset=(target-hp).clone()
                latched_handle_offset[:,2]=torch.clamp(latched_handle_offset[:,2],-float(os.environ.get("SCREW_LATCH_OFFSET_Z_MAX","0.080")),float(os.environ.get("SCREW_LATCH_OFFSET_Z_MAX","0.080")))
                latched_side_axis=(control_side_axis if sector_pinching_active and os.environ.get("SCREW_LATCH_USE_SECTOR_AXIS","1")=="1" else side_axis).clone()
            if strict_palm_grasp.item() and os.environ.get("SCREW_SIMPLE_STRICT_HOLD","0")=="1":
                if settle_step is None:
                    settle_step=step
                    if os.environ.get("SCREW_SIMPLE_STRICT_TARGET_CURRENT_PALM","0")=="1":
                        settle_target=palm.clone()+vec_env3("SCREW_SIMPLE_STRICT_PALM_OFFSET",[0.0,0.0,0.0],env.device)
                    else:
                        settle_target=target.clone()
            if grasp_latched:
                phase="hold"
                if os.environ.get('SCREW_RESTORE_HISTORICAL_TRAJ','0')=='1' and latched_target is not None:
                    # After first true/opposed contact, stop commanding the wrist high above the object.
                    # Follow the handle with the same calibrated grasp-frame offset so the fingers can close.
                    target=hp+grasp_cfg['static_t']*tangent+grasp_cfg['static_r']*radial+torch.tensor([[0.0,0.0,grasp_cfg['static_z']+float(os.environ.get('SCREW_HOLD_Z_BIAS','0.010'))]],device=env.device)
                latch_age=max(step-(latch_step or step),0)
                use_frozen=(os.environ.get('SCREW_HOLD_FREEZE_ON_LATCH','1')=='1' and latched_target is not None and latch_age<int(os.environ.get('SCREW_HOLD_FREEZE_STEPS','10')))
                if os.environ.get('SCREW_LATCH_FOLLOW_HANDLE','0')=='1' and latched_handle_offset is not None:
                    lift=min(latch_age,int(os.environ.get('SCREW_LATCH_LIFT_STEPS','12')))*float(os.environ.get('SCREW_LATCH_LIFT_PER_STEP','0.0010'))
                    target=hp+latched_handle_offset+torch.tensor([[0.0,0.0,float(os.environ.get("SCREW_HOLD_Z_BIAS","0.006"))+lift]],device=env.device)
                elif use_frozen:
                    lift=min(latch_age,int(os.environ.get('SCREW_LATCH_LIFT_STEPS','12')))*float(os.environ.get('SCREW_LATCH_LIFT_PER_STEP','0.0018'))
                    target=latched_target.clone()+torch.tensor([[0.0,0.0,float(os.environ.get("SCREW_HOLD_Z_BIAS","0.012"))+lift]],device=env.device)+float(os.environ.get("SCREW_HOLD_SIDE_BIAS",os.environ.get("SCREW_GRASP_SIDE_BIAS","0.0")))*bias_axis
                elif os.environ.get('SCREW_LATCH_USE_GRASP_FRAME','0')=='1':
                    if os.environ.get('SCREW_FREEZE_LATCH_TARGET','0')=='1' and latched_target is not None:
                        lift=min(latch_age,int(os.environ.get('SCREW_LATCH_LIFT_STEPS','18')))*float(os.environ.get('SCREW_LATCH_LIFT_PER_STEP','0.0015'))
                        target=latched_target+torch.tensor([[0.0,0.0,float(os.environ.get("SCREW_HOLD_Z_BIAS","0.010"))+lift]],device=env.device)
                    else:
                        target=hp+float(os.environ.get('SCREW_STATIC_TANGENT',0.105))*tangent+float(os.environ.get('SCREW_STATIC_RADIAL',-0.035))*radial+torch.tensor([[0.0,0.0,float(os.environ.get('SCREW_STATIC_Z',0.041))+float(os.environ.get("SCREW_HOLD_Z_BIAS","0.010"))]],device=env.device)
                else:
                    target=hp-float(os.environ.get("SCREW_POCKET_OFFSET_SCALE","1.00"))*pocket_offset+torch.tensor([[0.0,0.0,float(os.environ.get("SCREW_HOLD_Z_BIAS","0.010"))]],device=env.device)+float(os.environ.get("SCREW_HOLD_SIDE_BIAS",os.environ.get("SCREW_GRASP_SIDE_BIAS","0.0")))*bias_axis
                if os.environ.get("SCREW_HOLD_PINCH_MOUTH","0")=="1":
                    # Hold mode servos the thumb/index-middle pinch mouth onto the handle.
                    max_corr=float(os.environ.get("SCREW_MOUTH_CENTER_MAX","0.115"))
                    gxy=float(os.environ.get("SCREW_MOUTH_CENTER_GAIN","0.92"))
                    gz=float(os.environ.get("SCREW_MOUTH_Z_GAIN","0.82"))
                    center_err=torch.clamp(hp-pinch_center,-max_corr,max_corr)
                    target=palm+torch.cat((center_err[:,:2]*gxy,center_err[:,2:3]*gz),dim=-1)
                    desired_gap=float(os.environ.get("SCREW_MOUTH_SIDE_GAP","0.024"))
                    same_pos=(handle_side_thumb>0) & (handle_side_im>0)
                    same_neg=(handle_side_thumb<0) & (handle_side_im<0)
                    mouth_raw=torch.zeros_like(handle_side_thumb)
                    mouth_raw=torch.where(same_pos, -(torch.minimum(handle_side_thumb,handle_side_im)+desired_gap), mouth_raw)
                    mouth_raw=torch.where(same_neg, -(torch.maximum(handle_side_thumb,handle_side_im)-desired_gap), mouth_raw)
                    mouth_raw=torch.where(~(same_pos|same_neg), -0.5*(handle_side_thumb+handle_side_im)*float(os.environ.get("SCREW_MOUTH_CENTER_SIDE_GAIN","0.50")), mouth_raw)
                    mouth_side=(mouth_raw*float(os.environ.get("SCREW_MOUTH_SIDE_GAIN","2.8"))).clamp(-float(os.environ.get("SCREW_MOUTH_SIDE_MAX","0.18")),float(os.environ.get("SCREW_MOUTH_SIDE_MAX","0.18"))).unsqueeze(-1)
                    target=target+mouth_side*side_axis+torch.tensor([[0.0,0.0,float(os.environ.get("SCREW_MOUTH_Z_BIAS","0.000"))]],device=env.device)
                if os.environ.get("SCREW_PINCH_CAPTURE_WINDOW","0")=="1":
                    # Capture mode: after opposed fingertip contact, servo the thumb/index-mouth,
                    # not the wrist center. This prevents a hand-back block from being counted as a catch.
                    cap_steps=max(1,int(os.environ.get("SCREW_PINCH_CAPTURE_STEPS","14")))
                    if latch_age < cap_steps or bool(real_pinch[0]):
                        cmax=float(os.environ.get("SCREW_PINCH_CAPTURE_CENTER_MAX","0.20"))
                        gxy=float(os.environ.get("SCREW_PINCH_CAPTURE_XY_GAIN","1.55"))
                        gz=float(os.environ.get("SCREW_PINCH_CAPTURE_Z_GAIN","0.98"))
                        cerr=torch.clamp(hp-pinch_center,-cmax,cmax)
                        cap_target=palm+torch.cat((cerr[:,:2]*gxy,cerr[:,2:3]*gz),dim=-1)
                        desired=float(os.environ.get("SCREW_PINCH_CAPTURE_SIDE_GAP","0.026"))
                        same_pos=(handle_side_thumb>0) & (handle_side_im>0)
                        same_neg=(handle_side_thumb<0) & (handle_side_im<0)
                        raw=torch.zeros_like(handle_side_thumb)
                        raw=torch.where(same_pos, -(torch.minimum(handle_side_thumb,handle_side_im)+desired), raw)
                        raw=torch.where(same_neg, -(torch.maximum(handle_side_thumb,handle_side_im)-desired), raw)
                        raw=torch.where(~(same_pos|same_neg), -0.5*(handle_side_thumb+handle_side_im), raw)
                        side=(raw*float(os.environ.get("SCREW_PINCH_CAPTURE_SIDE_GAIN","3.2"))).clamp(-float(os.environ.get("SCREW_PINCH_CAPTURE_SIDE_MAX","0.22")),float(os.environ.get("SCREW_PINCH_CAPTURE_SIDE_MAX","0.22"))).unsqueeze(-1)
                        cap_target=cap_target+side*side_axis+torch.tensor([[0.0,0.0,float(os.environ.get("SCREW_PINCH_CAPTURE_Z_BIAS","-0.008"))]],device=env.device)
                        blend=float(os.environ.get("SCREW_PINCH_CAPTURE_BLEND","0.88"))
                        target=(1.0-blend)*target+blend*cap_target
                if os.environ.get("SCREW_COMPLIANT_CATCH","0")=="1":
                    # During the first few frames after contact, let the wrist follow the falling handle
                    # a little so the hand closes with lower relative speed instead of batting it away.
                    comp_steps=max(1,int(os.environ.get("SCREW_COMPLIANT_STEPS","7")))
                    comp=max(0.0,1.0-float(latch_age)/float(comp_steps))
                    down=max(0.0,-float(vel[0,2]))
                    dz=-min(float(os.environ.get("SCREW_COMPLIANT_MAX_DROP","0.055")), down*float(os.environ.get("SCREW_COMPLIANT_DT","0.018"))*comp)
                    xy_follow=torch.clamp(hp[:,0:2]-pinch_center[:,0:2],-float(os.environ.get("SCREW_COMPLIANT_XY_MAX","0.030")),float(os.environ.get("SCREW_COMPLIANT_XY_MAX","0.030")))*float(os.environ.get("SCREW_COMPLIANT_XY_GAIN","0.45"))*comp
                    target[:,0:2]=target[:,0:2]+xy_follow
                    target[:,2]=target[:,2]+dz
                if os.environ.get("SCREW_PALM_CAPTURE_WINDOW","0")=="1":
                    # Short capture window after first opposed contact: follow the falling handle
                    # slightly below the affordance point so the fingers close around it instead
                    # of the palm/wrist lifting it out of the grasp mouth.
                    cap_steps=max(1,int(os.environ.get("SCREW_CAPTURE_STEPS","12")))
                    if latch_age < cap_steps:
                        dz=float(os.environ.get("SCREW_CAPTURE_Z_BIAS","-0.035"))
                        target[:,2]=torch.clamp(hp[:,2]+dz, float(os.environ.get("SCREW_TARGET_Z_MIN","0.82")), float(os.environ.get("SCREW_TARGET_Z_MAX","1.72")))
                        xy_gain=float(os.environ.get("SCREW_CAPTURE_XY_GAIN","0.60"))
                        max_xy=float(os.environ.get("SCREW_CAPTURE_XY_MAX","0.055"))
                        target[:,0:2]=target[:,0:2]+torch.clamp(hp[:,0:2]-pinch_center[:,0:2], -max_xy, max_xy)*xy_gain
                if latched_target is not None and os.environ.get("SCREW_HOLD_Z_FLOOR","1")=="1":
                    target[:,2]=torch.maximum(target[:,2], latched_target[:,2]-float(os.environ.get('SCREW_HOLD_MAX_Z_DROP',0.012)))
                if os.environ.get("SCREW_HOLD_PINCH_SERVO","0")=="1":
                    max_corr=float(os.environ.get("SCREW_HOLD_PINCH_MAX","0.075"))
                    gxy=float(os.environ.get("SCREW_HOLD_PINCH_GAIN","0.70"))
                    gz=float(os.environ.get("SCREW_HOLD_PINCH_Z_GAIN","0.35"))
                    herr=torch.clamp(hp-pinch_center,-max_corr,max_corr)
                    target=target+torch.cat((herr[:,:2]*gxy,herr[:,2:3]*gz),dim=-1)
                if os.environ.get("SCREW_HOLD_SIDE_SERVO","0")=="1":
                    # After contact, actively put the handle between thumb and index/middle.
                    # If both sides have the same sign, shift the wrist along the measured thumb->finger axis.
                    hold_side_center=0.5*(handle_side_thumb+handle_side_im)
                    desired_mid=float(os.environ.get("SCREW_HOLD_DESIRED_MID","0.020"))
                    same_pos=(handle_side_thumb>0) & (handle_side_im>0)
                    same_neg=(handle_side_thumb<0) & (handle_side_im<0)
                    raw_corr=torch.zeros_like(hold_side_center)
                    raw_corr=torch.where(same_pos, -(torch.minimum(handle_side_thumb,handle_side_im)+desired_mid), raw_corr)
                    raw_corr=torch.where(same_neg, -(torch.maximum(handle_side_thumb,handle_side_im)-desired_mid), raw_corr)
                    raw_corr=torch.where(~(same_pos|same_neg), -hold_side_center*float(os.environ.get("SCREW_HOLD_CENTER_GAIN","0.35")), raw_corr)
                    hold_side_corr=(raw_corr*float(os.environ.get("SCREW_HOLD_SIDE_GAIN","1.8"))).clamp(-float(os.environ.get("SCREW_HOLD_SIDE_MAX","0.16")),float(os.environ.get("SCREW_HOLD_SIDE_MAX","0.16"))).unsqueeze(-1)
                    target=target+hold_side_corr*side_axis
                if os.environ.get("SCREW_STRICT_KEEP","0")=="1" and strict_seen>0:
                    # Once true palm-side pinch appears, stop chasing the falling center aggressively.
                    # Keep the object in the pinch mouth while the fingers finish closing.
                    keep_age=max(step-(settle_step or step),0)
                    if keep_age < int(os.environ.get("SCREW_STRICT_KEEP_STEPS","16")):
                        desired_mid=float(os.environ.get("SCREW_STRICT_KEEP_MID","0.030"))
                        same_pos=(handle_side_thumb>0) & (handle_side_im>0)
                        same_neg=(handle_side_thumb<0) & (handle_side_im<0)
                        keep_raw=torch.zeros_like(handle_side_thumb)
                        keep_raw=torch.where(same_pos, -(torch.minimum(handle_side_thumb,handle_side_im)+desired_mid), keep_raw)
                        keep_raw=torch.where(same_neg, -(torch.maximum(handle_side_thumb,handle_side_im)-desired_mid), keep_raw)
                        keep_raw=torch.where(~(same_pos|same_neg), -0.5*(handle_side_thumb+handle_side_im), keep_raw)
                        keep_corr=(keep_raw*float(os.environ.get("SCREW_STRICT_KEEP_GAIN","7.0"))*float(os.environ.get("SCREW_STRICT_KEEP_SIGN","1.0"))).clamp(-float(os.environ.get("SCREW_STRICT_KEEP_MAX","0.36")),float(os.environ.get("SCREW_STRICT_KEEP_MAX","0.36"))).unsqueeze(-1)
                        if os.environ.get("SCREW_STRICT_KEEP_USE_IKPOS", "0") == "1":
                            scmax=float(os.environ.get("SCREW_STRICT_KEEP_CENTER_MAX","0.18"))
                            scerr=torch.clamp(hp-pinch_center,-scmax,scmax)
                            sgxy=float(os.environ.get("SCREW_STRICT_KEEP_XY_GAIN","1.35"))
                            sgz=float(os.environ.get("SCREW_STRICT_KEEP_Z_GAIN","0.85"))
                            target=env.rigid_body_states[:, ik_body_idx, 0:3]+torch.cat((scerr[:,:2]*sgxy,scerr[:,2:3]*sgz),dim=-1)+keep_corr*side_axis+torch.tensor([[0.0,0.0,float(os.environ.get("SCREW_STRICT_KEEP_Z_BIAS","0.018"))]],device=env.device)
                        else:
                            target=pinch_center+keep_corr*side_axis+torch.tensor([[0.0,0.0,float(os.environ.get("SCREW_STRICT_KEEP_Z_BIAS","0.018"))]],device=env.device)
                if os.environ.get("SCREW_SIMPLE_STRICT_HOLD","0")=="1" and settle_target is not None:
                    strict_age=max(step-(settle_step or step),0)
                    lift=min(strict_age,int(os.environ.get("SCREW_SIMPLE_STRICT_LIFT_STEPS","24")))*float(os.environ.get("SCREW_SIMPLE_STRICT_LIFT_PER_STEP","0.0012"))
                    target=settle_target.clone()+torch.tensor([[0.0,0.0,float(os.environ.get("SCREW_SIMPLE_STRICT_Z_BIAS","0.006"))+lift]],device=env.device)
                if os.environ.get('SCREW_FINGERTIP_CLOSED_LOOP','0')=='1':
                    target=fingertip_closed_loop_target(env, ik_body_idx, target, hp, vel, thumb, im, palm, pinch_center, side_axis, handle_side_thumb, handle_side_im, True, latched_side_axis)
                if os.environ.get("SCREW_CLEAN_MOUTH_LOCK", "0") == "1":
                    # Clean post-latch controller: keep the physical thumb/index-middle mouth centered
                    # on the falling handle. This branch overrides the older stacked latch/hold targets.
                    lock_axis = torch.nn.functional.normalize(side_axis if os.environ.get("SCREW_CLEAN_LOCK_USE_CURRENT_AXIS", "0") == "1" else (latched_side_axis if latched_side_axis is not None else side_axis), dim=-1)
                    l_thumb = torch.sum((thumb - hp) * lock_axis, dim=-1)
                    l_im = torch.sum((im - hp) * lock_axis, dim=-1)
                    base_pos = env.rigid_body_states[:, ik_body_idx, 0:3]
                    cmax = float(os.environ.get("SCREW_CLEAN_LOCK_CENTER_MAX", "0.18"))
                    cerr = torch.clamp(hp - pinch_center, -cmax, cmax)
                    xy_gain = float(os.environ.get("SCREW_CLEAN_LOCK_XY_GAIN", "1.45"))
                    x_gain = float(os.environ.get("SCREW_CLEAN_LOCK_X_GAIN", str(xy_gain)))
                    y_gain = float(os.environ.get("SCREW_CLEAN_LOCK_Y_GAIN", str(xy_gain)))
                    target = base_pos + torch.cat((
                        torch.cat((cerr[:, 0:1] * x_gain, cerr[:, 1:2] * y_gain), dim=-1),
                        cerr[:, 2:3] * float(os.environ.get("SCREW_CLEAN_LOCK_Z_GAIN", "0.72")),
                    ), dim=-1)
                    desired = float(os.environ.get("SCREW_CLEAN_LOCK_SIDE_GAP", "0.024"))
                    both_pos = (l_thumb > 0) & (l_im > 0)
                    both_neg = (l_thumb < 0) & (l_im < 0)
                    raw = torch.zeros_like(l_thumb)
                    raw = torch.where(both_pos, -(torch.minimum(l_thumb, l_im) + desired), raw)
                    raw = torch.where(both_neg, -(torch.maximum(l_thumb, l_im) - desired), raw)
                    raw = torch.where(~(both_pos | both_neg), -0.5 * (l_thumb + l_im), raw)
                    side = (raw * float(os.environ.get("SCREW_CLEAN_LOCK_SIDE_GAIN", "2.6"))).clamp(
                        -float(os.environ.get("SCREW_CLEAN_LOCK_SIDE_MAX", "0.13")),
                        float(os.environ.get("SCREW_CLEAN_LOCK_SIDE_MAX", "0.13")),
                    ).unsqueeze(-1)
                    if os.environ.get("SCREW_CLEAN_LOCK_LATE_SIDE_GUARD", "0") == "1":
                        guard = float(os.environ.get("SCREW_CLEAN_LOCK_LATE_SIDE_GUARD_M", "0.018"))
                        delay = int(os.environ.get("SCREW_CLEAN_LOCK_LATE_SIDE_DELAY_STEPS", "24"))
                        boost = float(os.environ.get("SCREW_CLEAN_LOCK_LATE_SIDE_BOOST", "2.2"))
                        late_max = float(os.environ.get("SCREW_CLEAN_LOCK_LATE_SIDE_MAX", os.environ.get("SCREW_CLEAN_LOCK_SIDE_MAX", "0.13")))
                        low_side = ((l_im < guard) | (l_thumb > -guard)) & torch.tensor([latch_age >= delay], device=env.device)
                        side = torch.where(low_side.unsqueeze(-1), side * boost, side)
                        side = side.clamp(-late_max, late_max)
                    down = torch.clamp(-vel[:, 2:3], 0.0, float(os.environ.get("SCREW_CLEAN_LOCK_DOWN_SPEED_MAX", "3.5")))
                    target = target + side * lock_axis + torch.tensor([[0.0, 0.0, float(os.environ.get("SCREW_CLEAN_LOCK_Z_BIAS", "-0.020"))]], device=env.device)
                    target[:, 2:3] = target[:, 2:3] - down * float(os.environ.get("SCREW_CLEAN_LOCK_DOWN_FOLLOW", "0.010"))
                    if os.environ.get("SCREW_CLEAN_LOCK_VEL_FOLLOW", "1") == "1":
                        vmax = float(os.environ.get("SCREW_CLEAN_LOCK_VEL_MAX", "0.026"))
                        lead_xy = torch.clamp(vel[:, 0:2] * float(os.environ.get("SCREW_CLEAN_LOCK_VEL_DT", "0.030")), -vmax, vmax)
                        if os.environ.get("SCREW_CLEAN_LOCK_VEL_X_ONLY", "0") == "1":
                            lead_xy[:, 1] = 0.0
                        if os.environ.get("SCREW_CLEAN_LOCK_VEL_Y_ONLY", "0") == "1":
                            lead_xy[:, 0] = 0.0
                        target[:, 0:2] = target[:, 0:2] + lead_xy
                    target[:, 1] = torch.maximum(target[:, 1], hp[:, 1] - float(os.environ.get("SCREW_CLEAN_LOCK_FRONT_BACKOFF_Y", "0.050")))
                target[:,0]=torch.clamp(target[:,0],float(os.environ.get('SCREW_TARGET_X_MIN',-0.60)),float(os.environ.get('SCREW_TARGET_X_MAX',0.60))); target[:,1]=torch.clamp(target[:,1],float(os.environ.get('SCREW_TARGET_Y_MIN',-0.05)),float(os.environ.get('SCREW_TARGET_Y_MAX',0.68))); target[:,2]=torch.clamp(target[:,2],float(os.environ.get('SCREW_TARGET_Z_MIN',0.82)),float(os.environ.get('SCREW_TARGET_Z_MAX',1.72)))
            dynamic_ok=(had_dynamic_fall and max_down_speed>=float(os.environ.get("SCREW_SUCCESS_MIN_DOWN_SPEED","0.55")))
            if os.environ.get("SCREW_REQUIRE_DYNAMIC","1")!="1": dynamic_ok=True
            thumb_ok=torch.linalg.norm(thumb_rel,dim=-1)<float(os.environ.get("SCREW_STABLE_THUMB_GATE","0.080"))
            im_ok=torch.linalg.norm(im_rel,dim=-1)<float(os.environ.get("SCREW_STABLE_IM_GATE","0.090"))
            center_ok=pinch_center_d<float(os.environ.get("SCREW_STABLE_CENTER_GATE","0.075"))
            speed_ok=speed<float(os.environ.get("SCREW_STABLE_SPEED_GATE", os.environ.get("SCREW_SUCCESS_SPEED","0.75")))
            z_ok=hp[:,2]>float(os.environ.get("SCREW_SUCCESS_MIN_Z","0.88"))
            if os.environ.get("SCREW_SUCCESS_STRICT_FINGERTIP_ONLY", "0") == "1":
                strict_geo=(real_pinch & thumb_ok & im_ok & center_ok & speed_ok & z_ok & torch.tensor([dynamic_ok],device=env.device))
            else:
                strict_geo=(strict_palm_grasp & real_pinch & thumb_ok & im_ok & center_ok & pocket_gate & speed_ok & z_ok & torch.tensor([dynamic_ok],device=env.device))
            strict=bool(strict_geo.item()); prev_speed=float(speed[0]);
            if os.environ.get("SCREW_GRIPPER_SUCCESS_MODE","0")=="1":
                gripper_tip_ok=tip_d.min(dim=1).values<float(os.environ.get("SCREW_GRIPPER_SUCCESS_TIP_GATE","0.075"))
                gripper_center_ok=pinch_center_d<float(os.environ.get("SCREW_GRIPPER_SUCCESS_CENTER_GATE","0.145"))
                gripper_speed_ok=speed<float(os.environ.get("SCREW_GRIPPER_SUCCESS_SPEED_GATE","1.65"))
                z_ok_gripper = z_ok
                if os.environ.get("SCREW_RIGHT_EDGE_SUCCESS_LOW_Z", "1") == "1" and 'deg_now' in locals() and 30.0 <= deg_now < 50.0:
                    z_ok_gripper = hp[:,2] > float(os.environ.get("SCREW_RIGHT_EDGE_SUCCESS_MIN_Z", "0.60"))
                gripper_geo=real_pinch & gripper_tip_ok & gripper_center_ok & gripper_speed_ok & z_ok_gripper & torch.tensor([dynamic_ok],device=env.device)
                strict=bool(gripper_geo.item())
            if grasp_latched and (not bool(real_pinch[0]) or not bool(center_ok[0])):
                lost_pinch_count += 1
            else:
                lost_pinch_count = 0
            if os.environ.get("SCREW_RELATCH_ON_LOST","1")=="1" and grasp_latched and lost_pinch_count>=int(os.environ.get("SCREW_LOST_PINCH_RESET_STEPS","5")) and not success:
                grasp_latched=False; latch_step=None; latched_target=None; latched_handle_offset=None; latched_side_axis=None; phase="close"
            if strict_palm_grasp.item():
                strict_seen+=1
                if settle_step is None:
                    settle_step=step; settle_target=target.clone()
            hold_count=hold_count+1 if strict else 0
            two_stage_contact_ok = bool(real_pinch[0]) and bool(center_ok[0]) and bool(had_dynamic_fall)
            two_stage_contact_count = two_stage_contact_count + 1 if two_stage_contact_ok else 0
            if hold_count>=int(os.environ.get("SCREW_HOLD_COUNT","6")) and not success:
                success=True; first=step; phase='hold'
            if success and os.environ.get("SCREW_DROP_SUCCESS_IF_ESCAPES","1")=="1":
                escape_fail = (not bool(real_pinch[0]) or not bool(center_ok[0]))
                if os.environ.get("SCREW_RESET_SUCCESS_ON_NOT_STRICT", "0") == "1":
                    escape_fail = escape_fail or (not bool(strict))
                if escape_fail:
                    grace=int(os.environ.get("SCREW_SUCCESS_ESCAPE_GRACE_STEPS","0"))
                    escape_age=step-(first or step)
                    if escape_age>grace:
                        success=False; first=None; hold_count=0; phase="close"
            if success:
                phase="hold"
                if os.environ.get('SCREW_LATCH_USE_GRASP_FRAME','0')=='1':
                    if os.environ.get('SCREW_FREEZE_LATCH_TARGET','0')=='1' and latched_target is not None:
                        target=latched_target+torch.tensor([[0.0,0.0,float(os.environ.get("SCREW_SUCCESS_LIFT_Z","0.035"))]],device=env.device)
                    else:
                        target=hp+float(os.environ.get('SCREW_STATIC_TANGENT','0.105'))*tangent+float(os.environ.get('SCREW_STATIC_RADIAL','-0.035'))*radial+torch.tensor([[0.0,0.0,float(os.environ.get('SCREW_STATIC_Z','0.041'))+float(os.environ.get("SCREW_SUCCESS_LIFT_Z","0.035"))]],device=env.device)
                else:
                    target=hp-float(os.environ.get("SCREW_POCKET_OFFSET_SCALE","1.00"))*pocket_offset+torch.tensor([[0.0,0.0,float(os.environ.get("SCREW_SUCCESS_LIFT_Z","0.035"))]],device=env.device)+float(os.environ.get("SCREW_HOLD_SIDE_BIAS",os.environ.get("SCREW_GRASP_SIDE_BIAS","0.0")))*bias_axis
            if success and os.environ.get("SCREW_SUCCESS_KEEP_PINCH_MOUTH","0")=="1":
                # After a real catch is detected, keep the thumb/index-mouth wrapped on the handle.
                # Lifting the wrist immediately can pull the handle through the fingers, so lift only
                # as a small bias while the pinch center remains servoed to the handle.
                cmax=float(os.environ.get("SCREW_SUCCESS_KEEP_CENTER_MAX","0.22"))
                cerr=torch.clamp(hp-pinch_center,-cmax,cmax)
                gxy=float(os.environ.get("SCREW_SUCCESS_KEEP_XY_GAIN","1.65"))
                gz=float(os.environ.get("SCREW_SUCCESS_KEEP_Z_GAIN","1.20"))
                keep_base = env.rigid_body_states[:, ik_body_idx, 0:3] if os.environ.get("SCREW_SUCCESS_KEEP_USE_IKPOS", "0") == "1" else palm
                target=keep_base+torch.cat((cerr[:,:2]*gxy,cerr[:,2:3]*gz),dim=-1)
                desired=float(os.environ.get("SCREW_SUCCESS_KEEP_SIDE_GAP","0.026"))
                same_pos=(handle_side_thumb>0) & (handle_side_im>0)
                same_neg=(handle_side_thumb<0) & (handle_side_im<0)
                raw=torch.zeros_like(handle_side_thumb)
                raw=torch.where(same_pos, -(torch.minimum(handle_side_thumb,handle_side_im)+desired), raw)
                raw=torch.where(same_neg, -(torch.maximum(handle_side_thumb,handle_side_im)-desired), raw)
                raw=torch.where(~(same_pos|same_neg), -0.5*(handle_side_thumb+handle_side_im), raw)
                side=(raw*float(os.environ.get("SCREW_SUCCESS_KEEP_SIDE_GAIN","3.6"))).clamp(-float(os.environ.get("SCREW_SUCCESS_KEEP_SIDE_MAX","0.24")),float(os.environ.get("SCREW_SUCCESS_KEEP_SIDE_MAX","0.24"))).unsqueeze(-1)
                lift=min(max(step-(first or step),0),int(os.environ.get("SCREW_SUCCESS_KEEP_LIFT_STEPS","20")))*float(os.environ.get("SCREW_SUCCESS_KEEP_LIFT_PER_STEP","0.0010"))
                target=target+side*side_axis+torch.tensor([[0.0,0.0,float(os.environ.get("SCREW_SUCCESS_KEEP_Z_BIAS","-0.018"))+lift]],device=env.device)
            two_stage_hold_active = False
            if os.environ.get("SCREW_TWO_STAGE_HOLD", "0") == "1":
                # Strict two-stage hold controller.  It starts only after real
                # opposed fingertip contact, first follows the falling handle
                # downward to bleed relative velocity, then locks the same
                # thumb-vs-index/middle mouth geometry without lifting the wrist.
                current_two_stage_contact = (
                    bool(real_pinch[0])
                    and bool(had_dynamic_fall)
                    and float(pinch_center_d[0]) < float(os.environ.get("SCREW_TWO_STAGE_START_CENTER_GATE", "0.12"))
                    and float(hp[0, 2]) > float(os.environ.get("SCREW_TWO_STAGE_START_MIN_Z", "0.86"))
                )
                sector_early_cushion = False
                if os.environ.get("SCREW_SECTOR_EARLY_CUSHION", "0") == "1":
                    deg_edge = math.degrees(observed_angle) % 360.0
                    edge_ok = (
                        float(os.environ.get("SCREW_SECTOR_EARLY_RIGHT_MIN_DEG", "45")) <= deg_edge <= float(os.environ.get("SCREW_SECTOR_EARLY_RIGHT_MAX_DEG", "62"))
                        or float(os.environ.get("SCREW_SECTOR_EARLY_LEFT_MIN_DEG", "138")) <= deg_edge <= float(os.environ.get("SCREW_SECTOR_EARLY_LEFT_MAX_DEG", "152"))
                    )
                    sector_early_cushion = (
                        edge_ok
                        and bool(real_pinch[0])
                        and bool(had_dynamic_fall)
                        and float(pinch_center_d[0]) < float(os.environ.get("SCREW_SECTOR_EARLY_CENTER_GATE", "0.18"))
                        and float(hp[0, 2]) > float(os.environ.get("SCREW_SECTOR_EARLY_MIN_Z", "0.88"))
                        and float(hp[0, 2]) < float(os.environ.get("SCREW_SECTOR_EARLY_MAX_Z", "1.38"))
                        and float(arm_body_dist) > float(os.environ.get("SCREW_SECTOR_EARLY_BODY_CLEARANCE", "0.09"))
                        and -float(vel[0, 2]) < float(os.environ.get("SCREW_SECTOR_EARLY_MAX_DOWN_SPEED", "3.4"))
                    )
                two_stage_gate = (
                    hold_count >= int(os.environ.get("SCREW_TWO_STAGE_MIN_HOLD", "5"))
                    or two_stage_contact_count >= int(os.environ.get("SCREW_TWO_STAGE_MIN_CONTACT", "3"))
                    or current_two_stage_contact
                    or sector_early_cushion
                )
                if two_stage_gate and two_stage_started_step is None:
                    two_stage_started_step = step
                    two_stage_axis = control_side_axis if sector_pinching_active and os.environ.get("SCREW_TWO_STAGE_USE_SECTOR_AXIS","1")=="1" else side_axis
                    two_stage_lock_axis = torch.nn.functional.normalize(two_stage_axis.clone(), dim=-1)
                    two_stage_lock_target = prev_target.clone()
                    two_stage_lock_handle_z = hp[:, 2:3].clone()
                    two_stage_lock_down_speed = torch.clamp(-vel[:, 2:3].clone(), 0.0, float(os.environ.get("SCREW_TWO_STAGE_DECAY_DOWN_SPEED_MAX", os.environ.get("SCREW_TWO_STAGE_DOWN_SPEED_MAX", "4.2"))))
                    lock_side_thumb = torch.sum((thumb - hp) * two_stage_lock_axis, dim=-1)
                    lock_side_im = torch.sum((im - hp) * two_stage_lock_axis, dim=-1)
                    two_stage_lock_sign_thumb = torch.where(lock_side_thumb < 0, -torch.ones_like(lock_side_thumb), torch.ones_like(lock_side_thumb))
                    two_stage_lock_sign_im = torch.where(lock_side_im < 0, -torch.ones_like(lock_side_im), torch.ones_like(lock_side_im))
                if two_stage_started_step is not None:
                    age = step - two_stage_started_step
                    near_enough = bool(pinch_center_d[0] < float(os.environ.get("SCREW_TWO_STAGE_MEMORY_CENTER_GATE", "0.18"))) and bool(hp[0,2] > float(os.environ.get("SCREW_TWO_STAGE_MEMORY_MIN_Z", "0.82")))
                    two_stage_hold_active = (age < int(os.environ.get("SCREW_TWO_STAGE_TOTAL_STEPS", "48"))) and near_enough
                    if not two_stage_hold_active and age >= int(os.environ.get("SCREW_TWO_STAGE_TOTAL_STEPS", "48")):
                        two_stage_started_step = None
                        two_stage_lock_axis = None
                        two_stage_lock_target = None
                        two_stage_lock_handle_z = None
                        two_stage_lock_down_speed = None
                        two_stage_lock_sign_thumb = None
                        two_stage_lock_sign_im = None
                if two_stage_hold_active:
                    phase = "hold"
                    lock_axis = torch.nn.functional.normalize(two_stage_lock_axis if two_stage_lock_axis is not None else side_axis, dim=-1)
                    l_thumb = torch.sum((thumb - hp) * lock_axis, dim=-1)
                    l_im = torch.sum((im - hp) * lock_axis, dim=-1)
                    base = prev_target.clone() if two_stage_lock_target is None else two_stage_lock_target.clone()
                    if os.environ.get("SCREW_TWO_STAGE_USE_PREV_BASE", "1") == "1":
                        base = prev_target.clone()
                    cmax = float(os.environ.get("SCREW_TWO_STAGE_CENTER_MAX", "0.15"))
                    cerr = torch.clamp(hp - pinch_center, -cmax, cmax)
                    target = base.clone()
                    initial_xy_freeze = age < int(os.environ.get("SCREW_TWO_STAGE_INITIAL_XY_FREEZE_STEPS", "4"))
                    if not initial_xy_freeze:
                        target[:, 0:2] = target[:, 0:2] + torch.cat((
                            cerr[:, 0:1] * float(os.environ.get("SCREW_TWO_STAGE_X_GAIN", "0.50")),
                            cerr[:, 1:2] * float(os.environ.get("SCREW_TWO_STAGE_Y_GAIN", "0.50")),
                        ), dim=-1)
                    z_step = torch.clamp(
                        cerr[:, 2:3] * float(os.environ.get("SCREW_TWO_STAGE_Z_GAIN", "0.08")),
                        -float(os.environ.get("SCREW_TWO_STAGE_Z_STEP_MAX", "0.010")),
                        float(os.environ.get("SCREW_TWO_STAGE_Z_STEP_MAX", "0.010")),
                    )
                    target[:, 2:3] = target[:, 2:3] + z_step
                    desired = float(os.environ.get("SCREW_TWO_STAGE_SIDE_GAP", "0.040"))
                    guard = float(os.environ.get("SCREW_TWO_STAGE_SIDE_GUARD", "0.010"))
                    st = two_stage_lock_sign_thumb if two_stage_lock_sign_thumb is not None else torch.where(l_thumb < 0, -torch.ones_like(l_thumb), torch.ones_like(l_thumb))
                    si = two_stage_lock_sign_im if two_stage_lock_sign_im is not None else torch.where(l_im < 0, -torch.ones_like(l_im), torch.ones_like(l_im))
                    thumb_margin = st * l_thumb
                    im_margin = si * l_im
                    mouth_margin = torch.minimum(thumb_margin, im_margin)
                    cushion_steps = int(os.environ.get("SCREW_TWO_STAGE_CUSHION_STEPS", "24"))
                    age = step - (two_stage_started_step or step)
                    if age < cushion_steps:
                        if os.environ.get("SCREW_TWO_STAGE_DECAY_DOWN", "0") == "1" and two_stage_lock_down_speed is not None:
                            u = max(0.0, 1.0 - float(age) / float(max(1, cushion_steps)))
                            profile = u ** float(os.environ.get("SCREW_TWO_STAGE_DECAY_POWER", "1.35"))
                            down = two_stage_lock_down_speed * profile
                        else:
                            down = torch.clamp(-vel[:, 2:3], 0.0, float(os.environ.get("SCREW_TWO_STAGE_DOWN_SPEED_MAX", "4.2")))
                        follow = down * float(os.environ.get("SCREW_TWO_STAGE_DOWN_DT", "0.030")) * float(os.environ.get("SCREW_TWO_STAGE_DOWN_GAIN", "0.90"))
                        max_step = float(os.environ.get("SCREW_TWO_STAGE_DOWN_STEP_MAX", "0.060"))
                        if max_step > 0.0:
                            follow = torch.clamp(follow, 0.0, max_step)
                        min_step = float(os.environ.get("SCREW_TWO_STAGE_MIN_DOWN_STEP", "0.0"))
                        if min_step > 0.0:
                            u = max(0.0, 1.0 - float(age) / float(max(1, cushion_steps)))
                            follow = torch.maximum(follow, torch.full_like(follow, min_step * (0.25 + 0.75 * u)))
                        if os.environ.get("SCREW_TWO_STAGE_MOUTH_DAMP_DOWN", "0") == "1":
                            soft = float(os.environ.get("SCREW_TWO_STAGE_MOUTH_SOFT_GUARD", "0.018"))
                            hard = float(os.environ.get("SCREW_TWO_STAGE_MOUTH_HARD_GUARD", "0.006"))
                            denom = max(1e-6, soft - hard)
                            mouth_scale = torch.clamp((mouth_margin.unsqueeze(-1) - hard) / denom, 0.0, 1.0)
                            min_scale = float(os.environ.get("SCREW_TWO_STAGE_MOUTH_DOWN_MIN_SCALE", "0.15"))
                            follow = follow * torch.clamp(mouth_scale, min_scale, 1.0)
                        target[:, 2:3] = torch.minimum(target[:, 2:3], prev_target[:, 2:3] - follow)
                    elif age < cushion_steps + int(os.environ.get("SCREW_TWO_STAGE_BRAKE_TAIL_STEPS", "0")):
                        tail = max(1, int(os.environ.get("SCREW_TWO_STAGE_BRAKE_TAIL_STEPS", "1")))
                        u = max(0.0, 1.0 - float(age - cushion_steps) / float(tail))
                        follow = torch.full_like(target[:, 2:3], float(os.environ.get("SCREW_TWO_STAGE_TAIL_DOWN_STEP", "0.0")) * u)
                        target[:, 2:3] = torch.minimum(target[:, 2:3], prev_target[:, 2:3] - follow)
                    elif os.environ.get("SCREW_TWO_STAGE_NO_LIFT_LOCK", "1") == "1":
                        target[:, 2:3] = torch.minimum(
                            target[:, 2:3],
                            prev_target[:, 2:3] + float(os.environ.get("SCREW_TWO_STAGE_LOCK_UP_STEP_MAX", "0.0015")),
                        )
                    # Keep the originally observed opposed mouth: thumb and index/middle must stay on opposite sides.
                    # Once either side crosses toward the handle centerline, move the wrist along the locked side axis
                    # instead of chasing the handle center and letting the mouth collapse into same-side grazing.
                    raw = torch.zeros_like(l_thumb)
                    raw = raw + torch.where(thumb_margin < desired, st * (desired - thumb_margin), torch.zeros_like(raw))
                    raw = raw + torch.where(im_margin < desired, si * (desired - im_margin), torch.zeros_like(raw))
                    both_pos = (l_thumb > -guard) & (l_im > -guard)
                    both_neg = (l_thumb < guard) & (l_im < guard)
                    raw = torch.where(both_pos, raw - (torch.minimum(l_thumb, l_im) + desired), raw)
                    raw = torch.where(both_neg, raw - (torch.maximum(l_thumb, l_im) - desired), raw)
                    if os.environ.get("SCREW_TWO_STAGE_MOUTH_RESTORE", "1") == "1":
                        restore_guard = float(os.environ.get("SCREW_TWO_STAGE_MOUTH_RESTORE_GUARD", "0.032"))
                        restore_gain = float(os.environ.get("SCREW_TWO_STAGE_MOUTH_RESTORE_GAIN", "3.2"))
                        restore_max = float(os.environ.get("SCREW_TWO_STAGE_MOUTH_RESTORE_MAX", "0.18"))
                        restore = torch.clamp((restore_guard - mouth_margin) * restore_gain, 0.0, restore_max)
                        restore_dir = torch.where(thumb_margin <= im_margin, st, si)
                        raw = raw + restore_dir * restore
                    side = (raw * float(os.environ.get("SCREW_TWO_STAGE_SIDE_GAIN", "6.2"))).clamp(
                        -float(os.environ.get("SCREW_TWO_STAGE_SIDE_MAX", "0.22")),
                        float(os.environ.get("SCREW_TWO_STAGE_SIDE_MAX", "0.22")),
                    ).unsqueeze(-1)
                    side_freeze_steps = int(os.environ.get("SCREW_TWO_STAGE_INITIAL_SIDE_FREEZE_STEPS", "3"))
                    if age < side_freeze_steps:
                        side = torch.zeros_like(side)
                    target = target + side * lock_axis
                    if os.environ.get("SCREW_TWO_STAGE_FINGERTIP_Z_GEOM", "0") == "1":
                        # Keep the actual thumb/index-middle contact band slightly below the handle center.
                        # Successful v428 catches had fingertips about 6-8 cm below handle center at impact;
                        # failed reproductions had fingertips too high, letting the handle slide straight through.
                        tip_mean_z = 0.5 * (thumb[:, 2:3] + im[:, 2:3])
                        desired_tip_z = hp[:, 2:3] - float(os.environ.get("SCREW_TWO_STAGE_TIP_BELOW_HANDLE", "0.068"))
                        tip_err = torch.clamp(
                            desired_tip_z - tip_mean_z,
                            -float(os.environ.get("SCREW_TWO_STAGE_TIP_Z_STEP_MAX", "0.028")),
                            float(os.environ.get("SCREW_TWO_STAGE_TIP_Z_STEP_MAX", "0.028")),
                        )
                        age_gain = 1.0
                        if age < cushion_steps:
                            age_gain = float(os.environ.get("SCREW_TWO_STAGE_TIP_Z_CUSHION_GAIN", "1.00"))
                        else:
                            age_gain = float(os.environ.get("SCREW_TWO_STAGE_TIP_Z_HOLD_GAIN", "0.72"))
                        target[:, 2:3] = target[:, 2:3] + tip_err * age_gain
                    if os.environ.get("SCREW_TWO_STAGE_PALM_Z_GEOM", "0") == "1":
                        palm_target_z = hp[:, 2:3] - float(os.environ.get("SCREW_TWO_STAGE_PALM_BELOW_HANDLE", "0.118"))
                        palm_err = torch.clamp(
                            palm_target_z - palm[:, 2:3],
                            -float(os.environ.get("SCREW_TWO_STAGE_PALM_Z_STEP_MAX", "0.020")),
                            float(os.environ.get("SCREW_TWO_STAGE_PALM_Z_STEP_MAX", "0.020")),
                        )
                        target[:, 2:3] = target[:, 2:3] + palm_err * float(os.environ.get("SCREW_TWO_STAGE_PALM_Z_GAIN", "0.55"))
                    if os.environ.get("SCREW_TWO_STAGE_HANDLE_Z_WINDOW", "1") == "1":
                        below_max = float(os.environ.get("SCREW_TWO_STAGE_BELOW_HANDLE_MAX", "0.16"))
                        above_max = float(os.environ.get("SCREW_TWO_STAGE_ABOVE_HANDLE_MAX", "0.018"))
                        late_start = int(os.environ.get("SCREW_TWO_STAGE_LATE_Z_WINDOW_START", "1000000"))
                        if age >= late_start:
                            below_max = min(below_max, float(os.environ.get("SCREW_TWO_STAGE_LATE_BELOW_HANDLE_MAX", str(below_max))))
                            above_max = min(above_max, float(os.environ.get("SCREW_TWO_STAGE_LATE_ABOVE_HANDLE_MAX", str(above_max))))
                        if os.environ.get("SCREW_TWO_STAGE_SIDE_FAIL_Z_GUARD", "0") == "1":
                            side_guard_z = float(os.environ.get("SCREW_TWO_STAGE_SIDE_FAIL_THUMB_GUARD", "0.018"))
                            if bool((l_thumb > -side_guard_z)[0]):
                                below_max = min(below_max, float(os.environ.get("SCREW_TWO_STAGE_SIDE_FAIL_BELOW_HANDLE_MAX", "0.030")))
                        target[:, 2:3] = torch.clamp(
                            target[:, 2:3],
                            hp[:, 2:3] - below_max,
                            hp[:, 2:3] + above_max,
                        )
                    if os.environ.get("SCREW_TWO_STAGE_WRIST_ROTATE", "0") == "1":
                        rot_age = max(0, age - int(os.environ.get("SCREW_TWO_STAGE_ROT_START", "3")))
                        ramp = max(1, int(os.environ.get("SCREW_TWO_STAGE_ROT_RAMP_STEPS", "18")))
                        u = min(1.0, float(rot_age) / float(ramp))
                        u = u * u * (3.0 - 2.0 * u)
                        wrist_roll_cmd += math.radians(float(os.environ.get("SCREW_TWO_STAGE_ROT_ROLL_DEG", "0.0"))) * u
                        wrist_pitch_cmd += math.radians(float(os.environ.get("SCREW_TWO_STAGE_ROT_PITCH_DEG", "0.0"))) * u
                        wrist_yaw_offset += math.radians(float(os.environ.get("SCREW_TWO_STAGE_ROT_YAW_DEG", "0.0"))) * float(os.environ.get("SCREW_TWO_STAGE_ROT_YAW_SIGN", "1.0")) * u
                    if os.environ.get("SCREW_TWO_STAGE_SMOOTH_INSIDE", "1") == "1":
                        a = float(os.environ.get("SCREW_TWO_STAGE_TARGET_SMOOTH", "0.22"))
                        target = a * prev_target + (1.0 - a) * target
            if grasp_latched:
                a=float(os.environ.get("SCREW_LATCH_TARGET_SMOOTH","0.78"))
                if os.environ.get("SCREW_TWO_STAGE_HOLD", "0") == "1" and two_stage_hold_active:
                    a=float(os.environ.get("SCREW_TWO_STAGE_TARGET_SMOOTH", "0.30"))
                if os.environ.get("SCREW_PINCH_CAPTURE_WINDOW","0")=="1" and latch_step is not None and step-latch_step < int(os.environ.get("SCREW_PINCH_CAPTURE_STEPS","14")):
                    a=float(os.environ.get("SCREW_PINCH_CAPTURE_TARGET_SMOOTH","0.22"))
                if not (os.environ.get("SCREW_TWO_STAGE_BYPASS_LATCH_SMOOTH", "1") == "1" and two_stage_hold_active):
                    target=a*prev_target+(1.0-a)*target
                if os.environ.get("SCREW_HOLD_VEL_FOLLOW_POST", "0") == "1":
                    vgain=float(os.environ.get("SCREW_HOLD_VEL_FOLLOW_POST_GAIN", "1.0"))
                    vdt=float(os.environ.get("SCREW_HOLD_VEL_FOLLOW_POST_DT", "0.020"))
                    vmax=float(os.environ.get("SCREW_HOLD_VEL_FOLLOW_POST_MAX", "0.020"))
                    lead_xy=torch.clamp(vel[:,0:2]*vdt*vgain, -vmax, vmax)
                    if os.environ.get("SCREW_HOLD_VEL_FOLLOW_POST_Y_ONLY", "1") == "1":
                        lead_xy[:,0]=0.0
                    target[:,0:2]=target[:,0:2]+lead_xy
                if os.environ.get("SCREW_HOLD_CLAMP_TO_HANDLE","1")=="1":
                    final_z_max=min(float(os.environ.get("SCREW_HOLD_TARGET_Z_MAX", os.environ.get("SCREW_TARGET_Z_MAX","1.46"))), float(hp[0,2])+float(os.environ.get("SCREW_HOLD_ABOVE_HANDLE_MAX","0.115")))
                    final_z_min=max(float(os.environ.get("SCREW_HOLD_TARGET_Z_MIN", os.environ.get("SCREW_TARGET_Z_MIN","0.82"))), float(hp[0,2])-float(os.environ.get("SCREW_HOLD_BELOW_HANDLE_MAX","0.090")))
                    target[:,2]=torch.clamp(target[:,2], final_z_min, final_z_max)
                if os.environ.get("SCREW_HOLD_FRONT_SAFE","0")=="1":
                    # Front-drop catch must stay in front of the robot after contact; do not
                    # pull the hand back through the handle toward the FR3 body.
                    target[:,1]=torch.maximum(target[:,1], hp[:,1]-float(os.environ.get("SCREW_HOLD_FRONT_BACKOFF_Y","0.030")))
            if os.environ.get("SCREW_FINAL_RELATIVE_GRASP", "0") == "1" and (not two_stage_hold_active or os.environ.get("SCREW_TWO_STAGE_ALLOW_FINAL_REL", "0") == "1") and step >= int(os.environ.get("SCREW_OBSERVE_DELAY_STEPS","2")):
                # Angle-aware final target: keep the Revo2 pinch mouth at a fixed radial/tangent
                # offset from the observed handle. This preserves the known-good 90deg grasp
                # geometry when the drop angle changes, instead of relying on global X/Y boxes.
                apply_post = os.environ.get("SCREW_FINAL_RELATIVE_POST", "1") == "1"
                if (not grasp_latched) or apply_post:
                    rel_t = float(os.environ.get("SCREW_FINAL_REL_T", "-0.080"))
                    rel_r = float(os.environ.get("SCREW_FINAL_REL_R", "-0.020"))
                    rel_z = float(os.environ.get("SCREW_FINAL_REL_Z", "-0.028"))
                    rel = hp + rel_t * tangent + rel_r * radial + torch.tensor([[0.0, 0.0, rel_z]], device=env.device)
                    blend = float(os.environ.get("SCREW_FINAL_REL_BLEND", "0.70"))
                    if grasp_latched:
                        blend = float(os.environ.get("SCREW_FINAL_REL_POST_BLEND", str(blend)))
                    target = (1.0 - blend) * target + blend * rel
                    target[:,0]=torch.clamp(target[:,0],float(os.environ.get("SCREW_TARGET_X_MIN","-0.60")),float(os.environ.get("SCREW_TARGET_X_MAX","0.60")))
                    target[:,1]=torch.clamp(target[:,1],float(os.environ.get("SCREW_TARGET_Y_MIN","-0.05")),float(os.environ.get("SCREW_TARGET_Y_MAX","0.68")))
                    target[:,2]=torch.clamp(target[:,2],float(os.environ.get("SCREW_TARGET_Z_MIN","0.82")),float(os.environ.get("SCREW_TARGET_Z_MAX","1.72")))
            if grasp_latched and os.environ.get("SCREW_POST_LATCH_KEEP_MOUTH_CENTER", "0") == "1":
                # After true opposed contact, avoid switching to a new wrist target that pulls
                # the thumb/index-middle mouth away from the handle. This is arm compliance,
                # not object damping or hidden latching.
                center_gate = float(os.environ.get("SCREW_POST_LATCH_CENTER_SERVO_GATE", "0.16"))
                if bool(pinch_center_d[0] < center_gate):
                    cmax = float(os.environ.get("SCREW_POST_LATCH_CENTER_SERVO_MAX", "0.060"))
                    cerr = torch.clamp(hp - pinch_center, -cmax, cmax)
                    target[:, 0:1] = target[:, 0:1] + cerr[:, 0:1] * float(os.environ.get("SCREW_POST_LATCH_CENTER_X_GAIN", "0.35"))
                    target[:, 1:2] = target[:, 1:2] + cerr[:, 1:2] * float(os.environ.get("SCREW_POST_LATCH_CENTER_Y_GAIN", "0.35"))
                    target[:, 2:3] = target[:, 2:3] + cerr[:, 2:3] * float(os.environ.get("SCREW_POST_LATCH_CENTER_Z_GAIN", "0.10"))
            if grasp_latched and os.environ.get("SCREW_POST_LATCH_TARGET_RATE_LIMIT", "0") == "1":
                # Bound late target jumps. v397 failed because the post-latch target jumped
                # laterally after the mouth had already caught the handle.
                dx = torch.clamp(target[:, 0:1] - prev_target[:, 0:1], -float(os.environ.get("SCREW_POST_LATCH_X_STEP_MAX", "0.045")), float(os.environ.get("SCREW_POST_LATCH_X_STEP_MAX", "0.045")))
                dy = torch.clamp(target[:, 1:2] - prev_target[:, 1:2], -float(os.environ.get("SCREW_POST_LATCH_Y_STEP_MAX", "0.045")), float(os.environ.get("SCREW_POST_LATCH_Y_STEP_MAX", "0.045")))
                dz = target[:, 2:3] - prev_target[:, 2:3]
                dz = torch.clamp(dz, -float(os.environ.get("SCREW_POST_LATCH_Z_DOWN_STEP_MAX", "0.050")), float(os.environ.get("SCREW_POST_LATCH_Z_UP_STEP_MAX", "0.006")))
                target = torch.cat((prev_target[:, 0:1] + dx, prev_target[:, 1:2] + dy, prev_target[:, 2:3] + dz), dim=-1)
            if os.environ.get('SCREW_SECTOR55_CENTER_PRELATCH_START', '0') == '1' and 50.0 <= deg_now <= 62.0 and pre_latch_cushion_step is None and (not success):
                # Start 55deg pre-latch before real_pinch is already perfect. In this sector the
                # mouth can be centered at 10-16 cm while the side signs are still not opposed;
                # waiting for real_pinch lets the handle fall past the gripper mouth.
                if bool(had_dynamic_fall) and float(pinch_center_d[0]) < float(os.environ.get('SCREW_SECTOR55_CENTER_PRELATCH_GATE', '0.155')) and float(hp[0,2]) > float(os.environ.get('SCREW_SECTOR55_CENTER_PRELATCH_MIN_Z', '1.02')) and float(hp[0,2]) < float(os.environ.get('SCREW_SECTOR55_CENTER_PRELATCH_MAX_Z', '1.42')):
                    pre_latch_cushion_step = step
                    pre_latch_cushion_target = prev_target.clone()

            if os.environ.get('SCREW_RIGHT_EDGE_CENTER_PRELATCH_START', '1') == '1' and 30.0 <= deg_now < 50.0 and pre_latch_cushion_step is None and (not success):
                if bool(had_dynamic_fall) and bool(real_pinch[0]) and float(pinch_center_d[0]) < float(os.environ.get('SCREW_RIGHT_EDGE_CENTER_PRELATCH_GATE', '0.24')) and float(arm_body_dist) > float(os.environ.get('SCREW_RIGHT_EDGE_CENTER_PRELATCH_BODY_CLEARANCE', '0.070')):
                    pre_latch_cushion_step = step
                    pre_latch_cushion_target = prev_target.clone()

            if os.environ.get('SCREW_PRE_LATCH_CUSHION', '0') == '1':
                pre_gate = float(os.environ.get('SCREW_PRE_LATCH_CENTER_GATE', '0.18'))
                pre_thumb_gate = float(os.environ.get('SCREW_PRE_LATCH_THUMB_GATE', '0.36'))
                pre_im_gate = float(os.environ.get('SCREW_PRE_LATCH_IM_GATE', '0.36'))
                pre_ok = bool(real_pinch[0]) and bool(pinch_center_d[0] < pre_gate) and bool(torch.linalg.norm(thumb_rel, dim=-1)[0] < pre_thumb_gate) and bool(torch.linalg.norm(im_rel, dim=-1)[0] < pre_im_gate) and bool(had_dynamic_fall)
                if pre_ok and pre_latch_cushion_step is None:
                    pre_latch_cushion_step = step
                    pre_latch_cushion_target = prev_target.clone()
                if pre_latch_cushion_step is not None and (not success):
                    cushion_age = step - int(pre_latch_cushion_step)
                    if cushion_age <= int(os.environ.get('SCREW_PRE_LATCH_CUSHION_STEPS', '44')):
                        prev_stage_target = pre_latch_cushion_target if pre_latch_cushion_target is not None else prev_target.clone()
                        handle_follow_z = hp[:, 2:3] + float(os.environ.get('SCREW_PRE_LATCH_Z_OFFSET', '0.045'))
                        alpha = float(os.environ.get('SCREW_PRE_LATCH_FOLLOW_ALPHA', '0.92'))
                        z_des = alpha * handle_follow_z + (1.0 - alpha) * prev_stage_target[:, 2:3]
                        dz = torch.clamp(z_des - prev_target[:, 2:3], -float(os.environ.get('SCREW_PRE_LATCH_DOWN_STEP_MAX', '0.060')), float(os.environ.get('SCREW_PRE_LATCH_UP_STEP_MAX', '0.0005')))
                        target[:, 2:3] = prev_target[:, 2:3] + dz
                        xy_alpha = float(os.environ.get('SCREW_PRE_LATCH_XY_ALPHA', '0.08'))
                        target[:, 0:2] = (1.0 - xy_alpha) * prev_target[:, 0:2] + xy_alpha * target[:, 0:2]
                        if os.environ.get('SCREW_PRE_LATCH_WRIST_ROTATE', '0') == '1':
                            ramp = max(1, int(os.environ.get('SCREW_PRE_LATCH_ROT_RAMP_STEPS', '14')))
                            u = min(1.0, float(cushion_age) / float(ramp))
                            u = u * u * (3.0 - 2.0 * u)
                            wrist_roll_cmd += math.radians(float(os.environ.get('SCREW_PRE_LATCH_ROT_ROLL_DEG', '0.0'))) * u
                            wrist_pitch_cmd += math.radians(float(os.environ.get('SCREW_PRE_LATCH_ROT_PITCH_DEG', '-12.0'))) * u
                            wrist_yaw_offset += math.radians(float(os.environ.get('SCREW_PRE_LATCH_ROT_YAW_DEG', '3.0'))) * u
            if os.environ.get('SCREW_RIGHT_EDGE_PRELATCH_MOUTH_SERVO', '1') == '1' and 30.0 <= deg_now < 50.0 and pre_latch_cushion_step is not None and (not success):
                # Dedicated 35/45 deg right-edge capture. Keep the IK body close to the real
                # thumb/index-middle mouth, follow the falling handle downward, and do not let
                # the old 55deg corridor pull the palm into a conflicting pose.
                mouth_age = step - int(pre_latch_cushion_step)
                if mouth_age <= int(os.environ.get('SCREW_RIGHT_EDGE_PRELATCH_MOUTH_STEPS', '48')):
                    ik_pos_re = env.rigid_body_states[:, ik_body_idx, 0:3]
                    cmax_re = float(os.environ.get('SCREW_RIGHT_EDGE_PRELATCH_CENTER_MAX', '0.26'))
                    cerr_re = torch.clamp(hp - pinch_center, -cmax_re, cmax_re)
                    mouth_target_re = ik_pos_re + torch.cat((
                        cerr_re[:, 0:1] * float(os.environ.get('SCREW_RIGHT_EDGE_PRELATCH_X_GAIN', '1.25')),
                        cerr_re[:, 1:2] * float(os.environ.get('SCREW_RIGHT_EDGE_PRELATCH_Y_GAIN', '1.15')),
                        cerr_re[:, 2:3] * float(os.environ.get('SCREW_RIGHT_EDGE_PRELATCH_Z_GAIN', '0.42')),
                    ), dim=-1)
                    axis_re = control_side_axis
                    th_re = torch.sum((thumb - hp) * axis_re, dim=-1)
                    im_re = torch.sum((im - hp) * axis_re, dim=-1)
                    desired_re = float(os.environ.get('SCREW_RIGHT_EDGE_PRELATCH_SIDE_GAP', '0.030'))
                    both_pos_re = (th_re > 0) & (im_re > 0)
                    both_neg_re = (th_re < 0) & (im_re < 0)
                    raw_re = torch.zeros_like(th_re)
                    raw_re = torch.where(both_pos_re, -(torch.minimum(th_re, im_re) + desired_re), raw_re)
                    raw_re = torch.where(both_neg_re, -(torch.maximum(th_re, im_re) - desired_re), raw_re)
                    raw_re = torch.where(~(both_pos_re | both_neg_re), -0.5 * (th_re + im_re), raw_re)
                    side_re = (raw_re * float(os.environ.get('SCREW_RIGHT_EDGE_PRELATCH_SIDE_GAIN', '4.2'))).clamp(
                        -float(os.environ.get('SCREW_RIGHT_EDGE_PRELATCH_SIDE_MAX', '0.20')),
                        float(os.environ.get('SCREW_RIGHT_EDGE_PRELATCH_SIDE_MAX', '0.20')),
                    ).unsqueeze(-1)
                    mouth_target_re = mouth_target_re + side_re * axis_re
                    down_re = torch.clamp(-vel[:, 2:3], 0.0, float(os.environ.get('SCREW_RIGHT_EDGE_PRELATCH_DOWN_SPEED_MAX', '4.2')))
                    mouth_target_re[:, 2:3] = torch.minimum(
                        mouth_target_re[:, 2:3],
                        prev_target[:, 2:3] - torch.clamp(down_re * float(os.environ.get('SCREW_RIGHT_EDGE_PRELATCH_DOWN_DT', '0.020')), 0.0, float(os.environ.get('SCREW_RIGHT_EDGE_PRELATCH_DOWN_STEP_MAX', '0.055')))
                    )
                    mouth_target_re[:, 2:3] = torch.clamp(mouth_target_re[:, 2:3], hp[:, 2:3] - float(os.environ.get('SCREW_RIGHT_EDGE_PRELATCH_BELOW_HANDLE_MAX', '0.12')), hp[:, 2:3] + float(os.environ.get('SCREW_RIGHT_EDGE_PRELATCH_ABOVE_HANDLE_MAX', '0.050')))
                    blend_re = float(os.environ.get('SCREW_RIGHT_EDGE_PRELATCH_BLEND', '0.90'))
                    target = (1.0 - blend_re) * target + blend_re * mouth_target_re
                    target[:, 0]=torch.clamp(target[:, 0], float(os.environ.get('SCREW_TARGET_X_MIN','-0.70')), float(os.environ.get('SCREW_TARGET_X_MAX','0.72')))
                    target[:, 1]=torch.clamp(target[:, 1], float(os.environ.get('SCREW_TARGET_Y_MIN','-0.05')), float(os.environ.get('SCREW_TARGET_Y_MAX','1.20')))
                    target[:, 2]=torch.clamp(target[:, 2], float(os.environ.get('SCREW_TARGET_Z_MIN','0.82')), float(os.environ.get('SCREW_TARGET_Z_MAX','1.72')))

            if os.environ.get('SCREW_SECTOR55_PRELATCH_MOUTH_SERVO', '0') == '1' and 50.0 <= deg_now <= 62.0 and pre_latch_cushion_step is not None and (not success):
                # Right-front sector transition fix: after a previous catch the wrist can be in a
                # local IK posture where the handle briefly becomes opposed, then the pinch mouth
                # walks away in X/Y before latch. During pre-latch cushion, servo the IK body by
                # the actual thumb/index-middle mouth error and keep the same fixed sector axis.
                mouth_age = step - int(pre_latch_cushion_step)
                if mouth_age <= int(os.environ.get('SCREW_SECTOR55_PRELATCH_MOUTH_STEPS', '36')):
                    ik_pos55 = env.rigid_body_states[:, ik_body_idx, 0:3]
                    center_max55 = float(os.environ.get('SCREW_SECTOR55_PRELATCH_CENTER_MAX', '0.24'))
                    cerr55 = torch.clamp(hp - pinch_center, -center_max55, center_max55)
                    mouth_target55 = ik_pos55 + torch.cat((
                        cerr55[:, 0:1] * float(os.environ.get('SCREW_SECTOR55_PRELATCH_X_GAIN', '1.38')),
                        cerr55[:, 1:2] * float(os.environ.get('SCREW_SECTOR55_PRELATCH_Y_GAIN', '1.10')),
                        cerr55[:, 2:3] * float(os.environ.get('SCREW_SECTOR55_PRELATCH_Z_GAIN', '0.34')),
                    ), dim=-1)
                    axis55 = control_side_axis
                    th55 = torch.sum((thumb - hp) * axis55, dim=-1)
                    im55 = torch.sum((im - hp) * axis55, dim=-1)
                    desired55 = float(os.environ.get('SCREW_SECTOR55_PRELATCH_SIDE_GAP', '0.034'))
                    both_pos55 = (th55 > 0) & (im55 > 0)
                    both_neg55 = (th55 < 0) & (im55 < 0)
                    raw55 = torch.zeros_like(th55)
                    raw55 = torch.where(both_pos55, -(torch.minimum(th55, im55) + desired55), raw55)
                    raw55 = torch.where(both_neg55, -(torch.maximum(th55, im55) - desired55), raw55)
                    raw55 = torch.where(~(both_pos55 | both_neg55), -0.5 * (th55 + im55), raw55)
                    side55 = (raw55 * float(os.environ.get('SCREW_SECTOR55_PRELATCH_SIDE_GAIN', '4.8'))).clamp(
                        -float(os.environ.get('SCREW_SECTOR55_PRELATCH_SIDE_MAX', '0.22')),
                        float(os.environ.get('SCREW_SECTOR55_PRELATCH_SIDE_MAX', '0.22')),
                    ).unsqueeze(-1)
                    mouth_target55 = mouth_target55 + side55 * axis55
                    down55 = torch.clamp(-vel[:, 2:3], 0.0, float(os.environ.get('SCREW_SECTOR55_PRELATCH_DOWN_SPEED_MAX', '4.2')))
                    mouth_target55[:, 2:3] = torch.minimum(
                        mouth_target55[:, 2:3],
                        prev_target[:, 2:3] - torch.clamp(down55 * float(os.environ.get('SCREW_SECTOR55_PRELATCH_DOWN_DT', '0.018')), 0.0, float(os.environ.get('SCREW_SECTOR55_PRELATCH_DOWN_STEP_MAX', '0.045')))
                    )
                    mouth_target55[:, 2:3] = torch.clamp(mouth_target55[:, 2:3], hp[:, 2:3] - float(os.environ.get('SCREW_SECTOR55_PRELATCH_BELOW_HANDLE_MAX', '0.13')), hp[:, 2:3] + float(os.environ.get('SCREW_SECTOR55_PRELATCH_ABOVE_HANDLE_MAX', '0.055')))
                    blend55 = float(os.environ.get('SCREW_SECTOR55_PRELATCH_BLEND', '0.86'))
                    target = (1.0 - blend55) * target + blend55 * mouth_target55
                    target[:, 0]=torch.clamp(target[:, 0], float(os.environ.get('SCREW_TARGET_X_MIN','-0.70')), float(os.environ.get('SCREW_TARGET_X_MAX','0.70')))
                    target[:, 1]=torch.clamp(target[:, 1], float(os.environ.get('SCREW_TARGET_Y_MIN','-0.05')), float(os.environ.get('SCREW_TARGET_Y_MAX','1.18')))
                    target[:, 2]=torch.clamp(target[:, 2], float(os.environ.get('SCREW_TARGET_Z_MIN','0.82')), float(os.environ.get('SCREW_TARGET_Z_MAX','1.72')))

            if (grasp_latched or (os.environ.get('SCREW_TWO_STAGE_PRE_LATCH', '0') == '1' and pre_latch_cushion_step is not None)) and os.environ.get('SCREW_TWO_STAGE_DOWNFOLLOW', '0') == '1':
                # Two-stage physical hold: first follow the falling handle downward to lower
                # relative speed, then keep the thumb-vs-index/middle pinch mouth centered.
                stage_start = latch_step if grasp_latched and latch_step is not None else pre_latch_cushion_step
                latched_age = step - int(stage_start if stage_start is not None else step)
                follow_steps = int(os.environ.get('SCREW_TWO_STAGE_FOLLOW_STEPS', '28'))
                hold_steps = int(os.environ.get('SCREW_TWO_STAGE_HOLD_STEPS', '120'))
                prev_stage_target = globals().get("_screw_prev_target", prev_target.clone())
                handle_follow_z = hp[:, 2:3] + float(os.environ.get('SCREW_TWO_STAGE_Z_OFFSET', '0.030'))
                if latched_age <= follow_steps:
                    alpha = float(os.environ.get('SCREW_TWO_STAGE_FOLLOW_ALPHA', '0.72'))
                    z_des = alpha * handle_follow_z + (1.0 - alpha) * prev_stage_target[:, 2:3]
                    dz = torch.clamp(z_des - prev_stage_target[:, 2:3], -float(os.environ.get('SCREW_TWO_STAGE_DOWN_STEP_MAX', '0.034')), float(os.environ.get('SCREW_TWO_STAGE_UP_STEP_MAX', '0.002')))
                    target[:, 2:3] = prev_stage_target[:, 2:3] + dz
                    xy_alpha = float(os.environ.get('SCREW_TWO_STAGE_XY_ALPHA', '0.22'))
                    target[:, 0:2] = (1.0 - xy_alpha) * prev_stage_target[:, 0:2] + xy_alpha * target[:, 0:2]
                elif latched_age <= follow_steps + hold_steps:
                    hold_alpha = float(os.environ.get('SCREW_TWO_STAGE_HOLD_ALPHA', '0.18'))
                    target[:, 0:2] = (1.0 - hold_alpha) * prev_stage_target[:, 0:2] + hold_alpha * target[:, 0:2]
                    z_des = torch.minimum(prev_stage_target[:, 2:3] + float(os.environ.get('SCREW_TWO_STAGE_HOLD_UP_MAX', '0.001')), handle_follow_z + float(os.environ.get('SCREW_TWO_STAGE_HOLD_Z_BIAS', '0.020')))
                    dz = torch.clamp(z_des - prev_stage_target[:, 2:3], -float(os.environ.get('SCREW_TWO_STAGE_HOLD_DOWN_MAX', '0.010')), float(os.environ.get('SCREW_TWO_STAGE_HOLD_UP_MAX', '0.001')))
                    target[:, 2:3] = prev_stage_target[:, 2:3] + dz
            prev_target=target.clone()
            if clean_fall_active:
                release_mode=os.environ.get('SCREW_CLEAN_FALL_RELEASE_MODE','gripper')
                if release_mode=='any_tip':
                    tip_release=float(tip_d[0].min())<float(os.environ.get('SCREW_CLEAN_FALL_RELEASE_TIP_GATE','0.105'))
                else:
                    thumb_gate=float(os.environ.get('SCREW_CLEAN_FALL_RELEASE_THUMB_GATE',os.environ.get('SCREW_CLEAN_FALL_RELEASE_TIP_GATE','0.080')))
                    im_gate=float(os.environ.get('SCREW_CLEAN_FALL_RELEASE_IM_GATE',os.environ.get('SCREW_CLEAN_FALL_RELEASE_TIP_GATE','0.080')))
                    tip_release=(float(torch.linalg.norm(thumb_rel,dim=-1)[0])<thumb_gate and float(torch.linalg.norm(im_rel,dim=-1)[0])<im_gate)
                center_release=float(pinch_center_d[0])<float(os.environ.get('SCREW_CLEAN_FALL_RELEASE_CENTER_GATE','0.135'))
                pinch_release=(os.environ.get('SCREW_CLEAN_FALL_STOP_ON_REAL_PINCH','0')=='1' and bool(real_pinch[0]))
                clean_stop=(tip_release or center_release or pinch_release or bool(grasp_latched) or step>=int(os.environ.get('SCREW_CLEAN_FALL_MAX_STEPS','140')))
                if clean_stop:
                    clean_fall_active=False; clean_fall_stop_step=step
            if os.environ.get("SCREW_POST_LATCH_DAMP", "0") == "1" and grasp_latched and latch_step is not None:
                damp_age = step - latch_step
                if damp_age < int(os.environ.get("SCREW_POST_LATCH_DAMP_STEPS", "32")) and float(hp[0, 2]) > float(os.environ.get("SCREW_POST_LATCH_DAMP_MIN_Z", "0.82")):
                    # Stabilize only velocities after real latch; do not teleport or pin the object.
                    root = env.root_state_tensor[idx.long()].clone()
                    lin = root[:, 7:10].clone()
                    ang = root[:, 10:13].clone()
                    lin[:, 0:2] = lin[:, 0:2] * float(os.environ.get("SCREW_POST_LATCH_XY_DAMP", "0.20"))
                    vz_cap = float(os.environ.get("SCREW_POST_LATCH_VZ_CAP", "0.22"))
                    vz_scale = float(os.environ.get("SCREW_POST_LATCH_VZ_DAMP", "0.12"))
                    up_cap = float(os.environ.get("SCREW_POST_LATCH_UP_CAP", "0.35"))
                    lin[:, 2] = torch.clamp(lin[:, 2] * vz_scale, -vz_cap, up_cap)
                    ang[:, :] = ang[:, :] * float(os.environ.get("SCREW_POST_LATCH_ANG_DAMP", "0.08"))
                    root[:, 7:10] = lin
                    root[:, 10:13] = ang
                    env.root_state_tensor[idx.long()] = root
                    env.gym.set_actor_root_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env.root_state_tensor), gymtorch.unwrap_tensor(idx), len(idx))
                    env.gym.refresh_actor_root_state_tensor(env.sim)
                    st = env.root_state_tensor[idx.long()]
                    vel = st[:, 7:10]
                    speed = torch.linalg.norm(vel, dim=-1)
            latch_age_for_hand=max(step-(latch_step or step),0) if grasp_latched else 0
            hand_phase = phase
            hand_latched = bool(grasp_latched)
            if os.environ.get("SCREW_TWO_STAGE_HAND_HOLD", "0") == "1" and two_stage_hold_active:
                hand_phase = "hold"
                hand_latched = True
                if latch_age_for_hand == 0 and two_stage_started_step is not None:
                    latch_age_for_hand = max(step-two_stage_started_step, 0)
            hand_override=adaptive_hand_cmd(hand_phase,env.device,float(torch.linalg.norm(thumb_rel,dim=-1)[0]),float(torch.linalg.norm(im_rel,dim=-1)[0]),float(handle_side_thumb[0]),float(handle_side_im[0]),bool(real_pinch[0]),bool(hand_latched),latch_age_for_hand)
            arm_step=float(os.environ.get('SCREW_ARM_MAX_STEP','0.28'))
            if 'deg_now' in locals() and 30.0 <= deg_now < 50.0 and os.environ.get('SCREW_RIGHT_EDGE_ARM_STEP_OVERRIDE','1')=='1' and not grasp_latched:
                arm_step=float(os.environ.get('SCREW_RIGHT_EDGE_ARM_MAX_STEP', str(max(arm_step, 0.42))))
            if two_stage_hold_active:
                arm_step=float(os.environ.get('SCREW_TWO_STAGE_ARM_MAX_STEP', os.environ.get('SCREW_ARM_MAX_STEP','0.28')))
            if os.environ.get("SCREW_DYNAMIC_ARM_MOVING_AVG", "0") == "1":
                # In this task, armMovingAverage is the new-target weight; larger means faster tracking.
                # Preserve pre-contact behavior and speed up only during the two-stage catch.
                env.cfg["env"]["armMovingAverage"] = (
                    float(os.environ.get("SCREW_TWO_STAGE_ARM_MOVING_AVG", os.environ.get("SCREW_ARM_MOVING_AVG", "0.96")))
                    if two_stage_hold_active
                    else float(os.environ.get("SCREW_BASE_ARM_MOVING_AVG", os.environ.get("SCREW_ARM_MOVING_AVG", "0.96")))
                )
            if os.environ.get("SCREW_DYNAMIC_HAND_MOVING_AVG", "0") == "1":
                env.cfg["env"]["handMovingAverage"] = (
                    float(os.environ.get("SCREW_TWO_STAGE_HAND_MOVING_AVG", os.environ.get("SCREW_HAND_MOVING_AVG", "0.46")))
                    if two_stage_hold_active
                    else float(os.environ.get("SCREW_BASE_HAND_MOVING_AVG", os.environ.get("SCREW_HAND_MOVING_AVG", "0.46")))
                )
            arm=jac_servo(env,jac,body,servo,target,max_step=arm_step,yaw=yaw_cmd+wrist_yaw_offset,roll=wrist_roll_cmd,pitch=wrist_pitch_cmd); step_with_targets(env,arm,phase,hand_override)
            if common_hook is not None:
                jt=env._front110_last_joint_pos_targets
                common_hook.record_step(step=step,sim_time_s=(ep*args.steps+step)*float(getattr(env,'dt',1.0/60.0)),arm_target=jt[:, :env.num_arm_dofs],hand_internal_target=jt[:, env.num_arm_dofs:env.num_arm_dofs+11],phase=phase,active_tool_index=active_slot,thumb_contact=bool(real_pinch[0]) or bool(strict_palm_grasp[0]),opposing_finger_contact=bool(opp[0]) or bool(strict),bad_functional_contact=False,metadata=dict(angle_rad=float(angle),observed_angle_rad=float(observed_angle),ring_slot=float(active_slot)))
            cap(env,cam,props,writer); rows.append(dict(ep=ep,step=step,ring_slot=active_slot,release_x=x,release_y=y,release_z=z,angle=angle,observed_angle=observed_angle,yaw_offset=grasp_cfg['yaw_offset'],static_t=grasp_cfg['static_t'],static_r=grasp_cfg['static_r'],static_z=grasp_cfg['static_z'],lead=grasp_cfg['lead'],intercept_z=float(iz),pre_z=float(pre_z) if 'pre_z' in locals() else None,handle_x=float(hp[0,0]),handle_y=float(hp[0,1]),handle_z=float(hp[0,2]),vel_x=float(vel[0,0]),vel_y=float(vel[0,1]),vel_z=float(vel[0,2]),pred_x=float(pred[0,0]),pred_y=float(pred[0,1]),pred_z=float(pred[0,2]),palm_x=float(palm[0,0]),palm_y=float(palm[0,1]),palm_z=float(palm[0,2]),pocket_x=float(pocket[0,0]),pocket_y=float(pocket[0,1]),pocket_z=float(pocket[0,2]),target_x=float(target[0,0]),target_y=float(target[0,1]),target_z=float(target[0,2]),wrist_roll_cmd=float(wrist_roll_cmd),wrist_pitch_cmd=float(wrist_pitch_cmd),wrist_yaw_cmd=float(yaw_cmd+wrist_yaw_offset),phase=phase,palm_dist=float(pocket_d[0]),pinch_dist=float(torch.linalg.norm(pinch-hp,dim=-1)[0]),thumb_dist=float(torch.linalg.norm(thumb_rel,dim=-1)[0]),im_dist=float(torch.linalg.norm(im_rel,dim=-1)[0]),arm_body_dist=float(arm_body_dist),oppose=bool(opp[0]),dot=float(torch.sum(thumb_rel*im_rel,dim=-1)[0]),thumb_x=float(thumb[0,0]),thumb_y=float(thumb[0,1]),thumb_z=float(thumb[0,2]),im_x=float(im[0,0]),im_y=float(im[0,1]),im_z=float(im[0,2]),side_thumb=float(handle_side_thumb[0]),side_im=float(handle_side_im[0]),real_pinch=bool(real_pinch[0]),strict_palm_grasp=bool(strict_palm_grasp[0]),pinch_center_dist=float(pinch_center_d[0]),palm_dist_raw=float(palm_d[0]),latched=bool(grasp_latched),latch_now=bool(latch_now),side_center=float(side_center[0]),side_corr=float(side_corr[0,0]),speed=float(speed[0]),had_dynamic_fall=bool(had_dynamic_fall),max_down_speed=float(max_down_speed),first_dynamic_step=first_dynamic_step,strict=bool(strict),stable_hold_count=int(hold_count),lost_pinch_count=int(lost_pinch_count),success=bool(success)))
            if success and step > first + int(os.environ.get("SCREW_SUCCESS_BREAK_AFTER", "24")):
                if os.environ.get("SCREW_SUCCESS_SHAKE_RELEASE", "0") == "1":
                    shake_steps = int(os.environ.get("SCREW_SUCCESS_SHAKE_RELEASE_STEPS", "36"))
                    amp_t = float(os.environ.get("SCREW_SUCCESS_SHAKE_TANGENT_AMP", "0.060"))
                    amp_z = float(os.environ.get("SCREW_SUCCESS_SHAKE_Z_AMP", "0.025"))
                    base_release = target.clone()
                    for k in range(shake_steps):
                        sign = -1.0 if (k // 4) % 2 else 1.0
                        zsign = -1.0 if (k // 6) % 2 else 1.0
                        release_target = base_release + sign * amp_t * tangent + zsign * amp_z * torch.tensor([[0.0,0.0,1.0]], device=env.device)
                        release_target[:, 2] = torch.clamp(release_target[:, 2], float(os.environ.get("SCREW_TARGET_Z_MIN", "0.82")), float(os.environ.get("SCREW_TARGET_Z_MAX", "1.72")))
                        arm_rel = jac_servo(env, jac, body, servo, release_target, max_step=float(os.environ.get("SCREW_RELEASE_SHAKE_ARM_STEP", "0.24")), yaw=yaw_cmd+wrist_yaw_offset, roll=wrist_roll_cmd, pitch=wrist_pitch_cmd)
                        step_with_targets(env, arm_rel, "release")
                        cap(env, cam, props, writer)
                break
            if float(hp[0,2])<0.50: break
        summaries.append(dict(ep=ep,round_id=round_id,ring_slot=active_slot,dropped_slots=sorted(hidden_slots),remaining_slots=list(drop_order),angle=angle,release_xyz=[x,y,z],release_bias_xy=list(front110_release_xy_bias(angle)),success=success,first_success_step=first,yaw_offset=grasp_cfg['yaw_offset'],static_t=grasp_cfg['static_t'],static_r=grasp_cfg['static_r'],static_z=grasp_cfg['static_z'],lead=grasp_cfg['lead'],intercept_z=grasp_cfg.get('intercept_z'),pre_z=grasp_cfg.get('pre_z'),min_handle_palm=min_handle_palm,min_tip=min_tip,min_arm_body_dist=min_arm_body_dist,strict_palm_steps=sum(1 for rr in rows if rr.get('ep')==ep and rr.get('strict_palm_grasp')),real_pinch_steps=sum(1 for rr in rows if rr.get('ep')==ep and rr.get('real_pinch')),stable_success_frames=sum(1 for rr in rows if rr.get('ep')==ep and rr.get('strict')),max_stable_hold=max([rr.get('stable_hold_count',0) for rr in rows if rr.get('ep')==ep] or [0]),had_dynamic_fall=had_dynamic_fall,max_down_speed=max_down_speed,first_dynamic_step=first_dynamic_step));
        if common_hook is not None:
            ep_summary=dict(summaries[-1]); ep_summary.update(seed=args.seed,total=1,release_order_angles_deg=[math.degrees(angle)])
            common_hook.save_episode(ep_summary)
        next_ready_target = None; next_ready_yaw = 0.0; next_ready_roll = None; next_ready_pitch = None
        if drop_order and os.environ.get('SCREW_NEXT_SECTOR_READY','1')=='1':
            nx,ny,nz,na = slots[drop_order[0]]
            next_deg = math.degrees(na) % 360.0
            next_min = float(os.environ.get('SCREW_NEXT_SECTOR_READY_MIN_DEG', '148.0'))
            next_max = float(os.environ.get('SCREW_NEXT_SECTOR_READY_MAX_DEG', '158.0'))
            right_next_min = float(os.environ.get('SCREW_NEXT_SECTOR_READY_RIGHT_MIN_DEG', '30.0'))
            right_next_max = float(os.environ.get('SCREW_NEXT_SECTOR_READY_RIGHT_MAX_DEG', '45.0'))
            next_ready_edge = (next_min <= next_deg <= next_max) or (right_next_min <= next_deg <= right_next_max)
            if not next_ready_edge:
                next_ready_target = None
            else:
                apply_front110_sector_workspace(na)
                ng = angle_conditioned_grasp(na,args.lead_time)
                nr = torch.tensor([[math.cos(na),math.sin(na),0.0]],device=env.device)
                nt = torch.tensor([[-math.sin(na),math.cos(na),0.0]],device=env.device)
                npz = float(ng.get('pre_z', args.intercept_z))
                nbx,nby=front110_release_xy_bias(na)
                nx += nbx; ny += nby
                next_ready_target = torch.tensor([[nx,ny,npz]],device=env.device)+ng['static_t']*nt+ng['static_r']*nr+torch.tensor([[0.0,0.0,ng['static_z']+float(os.environ.get('SCREW_PREPOSITION_Z_BIAS','0.0'))]],device=env.device)
                next_ready_target[:,0]=torch.clamp(next_ready_target[:,0],float(os.environ.get('SCREW_TARGET_X_MIN','-0.86')),float(os.environ.get('SCREW_TARGET_X_MAX','0.80')))
                next_ready_target[:,1]=torch.clamp(next_ready_target[:,1],float(os.environ.get('SCREW_TARGET_Y_MIN','0.25')),float(os.environ.get('SCREW_TARGET_Y_MAX','1.20')))
                next_ready_target[:,2]=torch.clamp(next_ready_target[:,2],float(os.environ.get('SCREW_TARGET_Z_MIN','0.76')),float(os.environ.get('SCREW_TARGET_Z_MAX','1.72')))
                next_ready_yaw = na + ng['yaw_offset']
                next_ready_roll = ng.get('roll')
                next_ready_pitch = ng.get('pitch')
        inter_episode_release_reset(env,jac,body,servo,center,slots,hidden_slots,hidden_pos,cam,props,writer,home_dof,angle,next_ready_target,next_ready_yaw,next_ready_roll,next_ready_pitch)
    if writer: writer.close()
    with (args.out_dir/f'falling_ring_screwdriver_catch_seed{args.seed}_rows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()) if rows else ['empty']); w.writeheader(); w.writerows(rows)
    summary={'episodes':len(summaries),'successes':sum(int(s['success']) for s in summaries),'success_rate':sum(int(s['success']) for s in summaries)/max(1,len(summaries)),'task':'falling_ring_screwdriver_catch_game_release_from_v441_v461','display_name':'抓下落环形螺丝刀任务','ring_count':args.ring_count,'ring_radius':args.ring_radius,'ring_z':args.ring_z,'angle_min_deg':args.angle_min_deg,'angle_max_deg':args.angle_max_deg,'release_jitter_xy':float(os.environ.get('SCREW_RELEASE_JITTER_XY','0.0')),'random_yaw_deg':float(os.environ.get('SCREW_RANDOM_YAW_DEG','0.0')),'franka_fr3_joint_vel_limit_rad_s':[2.62,2.62,2.62,2.62,5.26,4.18,5.26],'revo2_active_joint_order':['thumb_flex','thumb_aux','index','middle','ring','pinky'],'revo2_active_joint_vel_limit_rad_s':[2.53,2.62,2.27,2.27,2.27,2.27],'object_real_spec':{'length_m':0.32,'diameter_m':0.03,'mass_kg':0.125,'handle_region_m':0.11,'functional_region_m':0.21,'handle_color':'green','functional_color':'red'},'release_policy':'after stable physical grasp, fully open Revo2 and shake wrist/arm to drop old tool before next fall','refresh_rule':'drop every ring screwdriver once in random order, refresh only after all slots have dropped','video':str(vpath) if vpath else None,'summaries':summaries,'asset':os.environ.get('SCREW_OBJECT_ASSET_ROOT','assets/generated/falling_screwdriver_affordance_v01')+'/screwdriver_affordance.urdf'}; (args.out_dir/f'falling_ring_screwdriver_catch_seed{args.seed}_summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2)); return 0
if __name__=='__main__': sys.exit(main())

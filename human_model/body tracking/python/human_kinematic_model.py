# human_kinematic_model.py
# BODY_18 kinematic generator using fixed segment lengths + main joint angles only.
# Learns angles (not bone lengths) for: shoulders, elbows, hips, knees.
# Outputs canonical keypoints (18,3), pelvis/root at origin, Y-up.

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import torch

KP = {
    "NOSE": 0, "NECK": 1,
    "R_SHOULDER": 2, "R_ELBOW": 3, "R_WRIST": 4,
    "L_SHOULDER": 5, "L_ELBOW": 6, "L_WRIST": 7,
    "R_HIP": 8, "R_KNEE": 9, "R_ANKLE": 10,
    "L_HIP": 11, "L_KNEE": 12, "L_ANKLE": 13,
    "R_EYE": 14, "L_EYE": 15, "R_EAR": 16, "L_EAR": 17,
}

KP18_NAMES = [
    "nose","neck",
    "r_shoulder","r_elbow","r_wrist",
    "l_shoulder","l_elbow","l_wrist",
    "r_hip","r_knee","r_ankle",
    "l_hip","l_knee","l_ankle",
    "r_eye","l_eye","r_ear","l_ear"
]

# ---------------- rotations ----------------

def _Rx(a: torch.Tensor) -> torch.Tensor:
    ca, sa = torch.cos(a), torch.sin(a)
    z = torch.zeros_like(a)
    o = torch.ones_like(a)
    return torch.stack([
        torch.stack([o, z, z], dim=-1),
        torch.stack([z, ca, -sa], dim=-1),
        torch.stack([z, sa, ca], dim=-1),
    ], dim=-2)

def _Ry(a: torch.Tensor) -> torch.Tensor:
    ca, sa = torch.cos(a), torch.sin(a)
    z = torch.zeros_like(a)
    o = torch.ones_like(a)
    return torch.stack([
        torch.stack([ca, z, sa], dim=-1),
        torch.stack([z,  o, z], dim=-1),
        torch.stack([-sa, z, ca], dim=-1),
    ], dim=-2)

def _Rz(a: torch.Tensor) -> torch.Tensor:
    ca, sa = torch.cos(a), torch.sin(a)
    z = torch.zeros_like(a)
    o = torch.ones_like(a)
    return torch.stack([
        torch.stack([ca, -sa, z], dim=-1),
        torch.stack([sa,  ca, z], dim=-1),
        torch.stack([z,   z,  o], dim=-1),
    ], dim=-2)

def euler_yaw_pitch_roll(yaw: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
    # yaw about Y, pitch about X, roll about Z
    return _Ry(yaw) @ _Rx(pitch) @ _Rz(roll)

# ---------------- segment lengths (fixed) ----------------

@dataclass
class SegmentLengths:
    torso: float
    shoulder_offset_x: float
    hip_offset_x: float
    upper_arm: float
    lower_arm: float
    thigh: float
    calf: float
    # Face offsets relative to NECK (kept fixed, not optimized)
    nose_off: np.ndarray
    reye_off: np.ndarray
    leye_off: np.ndarray
    rear_off: np.ndarray
    lear_off: np.ndarray

def lengths_from_standard_np(standard_kp: np.ndarray) -> SegmentLengths:
    """
    Derive fixed segment lengths and face offsets from an initial template skeleton.
    """
    def d(i, j):
        return float(np.linalg.norm(standard_kp[i] - standard_kp[j]))

    pelvis = 0.5 * (standard_kp[KP["L_HIP"]] + standard_kp[KP["R_HIP"]])
    torso = float(np.linalg.norm(standard_kp[KP["NECK"]] - pelvis))

    shoulder_offset_x = abs(float(standard_kp[KP["R_SHOULDER"]][0] - standard_kp[KP["NECK"]][0]))
    hip_offset_x = abs(float(standard_kp[KP["R_HIP"]][0] - pelvis[0]))

    upper_arm = d(KP["R_SHOULDER"], KP["R_ELBOW"])
    lower_arm = d(KP["R_ELBOW"], KP["R_WRIST"])
    thigh = d(KP["R_HIP"], KP["R_KNEE"])
    calf = d(KP["R_KNEE"], KP["R_ANKLE"])

    neck = standard_kp[KP["NECK"]]
    nose_off = (standard_kp[KP["NOSE"]] - neck).astype(np.float32)
    reye_off = (standard_kp[KP["R_EYE"]] - neck).astype(np.float32)
    leye_off = (standard_kp[KP["L_EYE"]] - neck).astype(np.float32)
    rear_off = (standard_kp[KP["R_EAR"]] - neck).astype(np.float32)
    lear_off = (standard_kp[KP["L_EAR"]] - neck).astype(np.float32)

    return SegmentLengths(
        torso=torso,
        shoulder_offset_x=shoulder_offset_x,
        hip_offset_x=hip_offset_x,
        upper_arm=upper_arm,
        lower_arm=lower_arm,
        thigh=thigh,
        calf=calf,
        nose_off=nose_off,
        reye_off=reye_off,
        leye_off=leye_off,
        rear_off=rear_off,
        lear_off=lear_off,
    )

# ---------------- kinematic model ----------------

class HumanKinematicModel(torch.nn.Module):
    """
    Inferred angles (16 scalars):
      sh_L yaw,pitch,roll (3)
      sh_R yaw,pitch,roll (3)
      el_L flex (1)
      el_R flex (1)
      hip_L yaw,pitch,roll (3)
      hip_R yaw,pitch,roll (3)
      kn_L flex (1)
      kn_R flex (1)

    NOTE: Face joints are not optimized; they are fixed offsets from NECK.
    """
    def __init__(self, lengths: SegmentLengths, device="cpu"):
        super().__init__()
        self.lengths = lengths
        self.device = device

        self.nose_off = torch.tensor(lengths.nose_off, device=device, dtype=torch.float32)
        self.reye_off = torch.tensor(lengths.reye_off, device=device, dtype=torch.float32)
        self.leye_off = torch.tensor(lengths.leye_off, device=device, dtype=torch.float32)
        self.rear_off = torch.tensor(lengths.rear_off, device=device, dtype=torch.float32)
        self.lear_off = torch.tensor(lengths.lear_off, device=device, dtype=torch.float32)

    def forward(self, angles: Dict[str, torch.Tensor]) -> torch.Tensor:
        L = self.lengths
        device = self.device
        dtype = torch.float32

        kp = torch.full((18, 3), float("nan"), device=device, dtype=dtype)

        pelvis = torch.zeros(3, device=device, dtype=dtype)
        neck = pelvis + torch.tensor([0.0, L.torso, 0.0], device=device, dtype=dtype)

        l_sh = neck + torch.tensor([-L.shoulder_offset_x, 0.0, 0.0], device=device, dtype=dtype)
        r_sh = neck + torch.tensor([ L.shoulder_offset_x, 0.0, 0.0], device=device, dtype=dtype)

        l_hip = pelvis + torch.tensor([-L.hip_offset_x, 0.0, 0.0], device=device, dtype=dtype)
        r_hip = pelvis + torch.tensor([ L.hip_offset_x, 0.0, 0.0], device=device, dtype=dtype)

        R_sh_L = euler_yaw_pitch_roll(angles["sh_L"][0], angles["sh_L"][1], angles["sh_L"][2])
        R_sh_R = euler_yaw_pitch_roll(angles["sh_R"][0], angles["sh_R"][1], angles["sh_R"][2])
        R_hip_L = euler_yaw_pitch_roll(angles["hip_L"][0], angles["hip_L"][1], angles["hip_L"][2])
        R_hip_R = euler_yaw_pitch_roll(angles["hip_R"][0], angles["hip_R"][1], angles["hip_R"][2])

        # Hinges (flexion) about local X
        R_el_L = _Rx(angles["el_L"][0])
        R_el_R = _Rx(angles["el_R"][0])
        R_kn_L = _Rx(angles["kn_L"][0])
        R_kn_R = _Rx(angles["kn_R"][0])

        v_upper_arm = torch.tensor([0.0, -L.upper_arm, 0.0], device=device, dtype=dtype)
        v_lower_arm = torch.tensor([0.0, -L.lower_arm, 0.0], device=device, dtype=dtype)

        l_el = l_sh + (R_sh_L @ v_upper_arm)
        r_el = r_sh + (R_sh_R @ v_upper_arm)

        l_wr = l_el + (R_sh_L @ (R_el_L @ v_lower_arm))
        r_wr = r_el + (R_sh_R @ (R_el_R @ v_lower_arm))

        v_thigh = torch.tensor([0.0, -L.thigh, 0.0], device=device, dtype=dtype)
        v_calf  = torch.tensor([0.0, -L.calf,  0.0], device=device, dtype=dtype)

        l_kn = l_hip + (R_hip_L @ v_thigh)
        r_kn = r_hip + (R_hip_R @ v_thigh)

        l_an = l_kn + (R_hip_L @ (R_kn_L @ v_calf))
        r_an = r_kn + (R_hip_R @ (R_kn_R @ v_calf))

        # Face keypoints (fixed offsets from neck)
        nose = neck + self.nose_off
        r_eye = neck + self.reye_off
        l_eye = neck + self.leye_off
        r_ear = neck + self.rear_off
        l_ear = neck + self.lear_off

        # Fill BODY_18
        kp[KP["NOSE"]] = nose
        kp[KP["NECK"]] = neck
        kp[KP["R_SHOULDER"]] = r_sh
        kp[KP["R_ELBOW"]] = r_el
        kp[KP["R_WRIST"]] = r_wr
        kp[KP["L_SHOULDER"]] = l_sh
        kp[KP["L_ELBOW"]] = l_el
        kp[KP["L_WRIST"]] = l_wr
        kp[KP["R_HIP"]] = r_hip
        kp[KP["R_KNEE"]] = r_kn
        kp[KP["R_ANKLE"]] = r_an
        kp[KP["L_HIP"]] = l_hip
        kp[KP["L_KNEE"]] = l_kn
        kp[KP["L_ANKLE"]] = l_an
        kp[KP["R_EYE"]] = r_eye
        kp[KP["L_EYE"]] = l_eye
        kp[KP["R_EAR"]] = r_ear
        kp[KP["L_EAR"]] = l_ear

        return kp

def default_joint_limits_radians(device="cpu") -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Conservative joint limits.
    For shoulder/hip: yaw, pitch, roll.
    For elbow/knee: flexion (0 straight).
    """
    def to_rad(deg_list):
        return torch.tensor(deg_list, device=device, dtype=torch.float32) * (torch.pi / 180.0)

    limits = {
        "sh":  (to_rad([-90.0, -90.0, -90.0]), to_rad([ 90.0,  90.0,  90.0])),
        "hip": (to_rad([-60.0, -90.0, -45.0]), to_rad([ 60.0,  60.0,  45.0])),
        "el":  (to_rad([  0.0]),               to_rad([150.0])),
        "kn":  (to_rad([  0.0]),               to_rad([160.0])),
    }
    return limits
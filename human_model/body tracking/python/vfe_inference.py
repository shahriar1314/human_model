# vfe_inference.py
# Active Inference (perception) via minimizing Variational Free Energy:
#   F = accuracy(o, g(theta)) + w_limits * joint_limits_prior + w_sym * symmetry_prior
# g(theta): kinematic keypoints -> Kabsch align to observations (anchors) -> predicted keypoints in live frame.

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch

from human_kinematic_model import HumanKinematicModel, default_joint_limits_radians

def valid_mask_np(kp: np.ndarray) -> np.ndarray:
    return np.isfinite(kp).all(axis=1)

def kabsch_torch(src: torch.Tensor, dst: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable Kabsch (rotation+translation), no scale.
    src, dst: (N,3)
    Returns R (3,3), t (3,)
    """
    src_mean = src.mean(dim=0)
    dst_mean = dst.mean(dim=0)
    X = src - src_mean
    Y = dst - dst_mean

    H = X.t() @ Y
    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    V = Vh.t()

    # Proper rotation (avoid reflection) via diagonal correction
    d = torch.sign(torch.linalg.det(V @ U.t()))
    d = torch.where(d == 0, torch.tensor(1.0, device=d.device, dtype=d.dtype), d)
    D = torch.diag(torch.stack([torch.tensor(1.0, device=d.device), torch.tensor(1.0, device=d.device), d]))

    R = V @ D @ U.t()
    t = dst_mean - (src_mean @ R.t())
    return R, t

@dataclass
class InferenceConfig:
    anchors: List[int]
    sigma_obs: float = 0.05     # meters (likelihood std)
    w_limits: float = 5.0
    w_sym: float = 1.0
    iters: int = 60
    lr: float = 2e-2
    device: str = "cpu"

def _angles_from_vector(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    # x is (16,) ordered:
    # sh_L(3), sh_R(3), el_L(1), el_R(1), hip_L(3), hip_R(3), kn_L(1), kn_R(1)
    i = 0
    sh_L = x[i:i+3]; i += 3
    sh_R = x[i:i+3]; i += 3
    el_L = x[i:i+1]; i += 1
    el_R = x[i:i+1]; i += 1
    hip_L = x[i:i+3]; i += 3
    hip_R = x[i:i+3]; i += 3
    kn_L = x[i:i+1]; i += 1
    kn_R = x[i:i+1]; i += 1
    return {"sh_L": sh_L, "sh_R": sh_R, "el_L": el_L, "el_R": el_R, "hip_L": hip_L, "hip_R": hip_R, "kn_L": kn_L, "kn_R": kn_R}

def symmetry_prior(angles: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Left-right symmetry prior (soft).
    yaw mirrored (L yaw ≈ -R yaw), pitch same, roll mirrored.
    elbows/knees same flexion.
    """
    shL, shR = angles["sh_L"], angles["sh_R"]
    hipL, hipR = angles["hip_L"], angles["hip_R"]
    elL, elR = angles["el_L"], angles["el_R"]
    knL, knR = angles["kn_L"], angles["kn_R"]

    sh_err = (shL[0] + shR[0])**2 + (shL[1] - shR[1])**2 + (shL[2] + shR[2])**2
    hip_err = (hipL[0] + hipR[0])**2 + (hipL[1] - hipR[1])**2 + (hipL[2] + hipR[2])**2
    hinge_err = (elL[0] - elR[0])**2 + (knL[0] - knR[0])**2
    return sh_err + hip_err + hinge_err

def joint_limits_prior(x: torch.Tensor, lim: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """
    Soft penalty for violating joint limits.
    """
    angles = _angles_from_vector(x)

    sh_min, sh_max = lim["sh"]
    hip_min, hip_max = lim["hip"]
    el_min, el_max = lim["el"]
    kn_min, kn_max = lim["kn"]

    def penalty(v, vmin, vmax):
        return torch.relu(vmin - v).pow(2).sum() + torch.relu(v - vmax).pow(2).sum()

    p = torch.tensor(0.0, device=x.device)
    p = p + penalty(angles["sh_L"], sh_min, sh_max)
    p = p + penalty(angles["sh_R"], sh_min, sh_max)
    p = p + penalty(angles["hip_L"], hip_min, hip_max)
    p = p + penalty(angles["hip_R"], hip_min, hip_max)
    p = p + penalty(angles["el_L"], el_min, el_max)
    p = p + penalty(angles["el_R"], el_min, el_max)
    p = p + penalty(angles["kn_L"], kn_min, kn_max)
    p = p + penalty(angles["kn_R"], kn_min, kn_max)
    return p

@dataclass
class InferenceResult:
    x_opt: np.ndarray              # (16,)
    kp_pred_aligned: np.ndarray    # (18,3)
    R: np.ndarray                  # (3,3)
    t: np.ndarray                  # (3,)
    diff: np.ndarray               # (18,3)
    per_l2: np.ndarray             # (18,)
    mean_l2: float
    rmse: float
    used_anchors: List[int]

class ActiveInferencePoseEstimator:
    """
    Active inference (perception): infer angles to minimize free energy each frame.
    Warm-starts from previous solution for speed/stability.
    """
    def __init__(self, model: HumanKinematicModel, cfg: InferenceConfig):
        self.model = model
        self.cfg = cfg
        self.lim = default_joint_limits_radians(device=cfg.device)
        self._x_prev = None

    def infer(self, live_kp_np: np.ndarray) -> InferenceResult:
        device = self.cfg.device
        live = torch.tensor(live_kp_np, dtype=torch.float32, device=device)

        valid = valid_mask_np(live_kp_np)
        anchors = [i for i in self.cfg.anchors if valid[i]]
        if len(anchors) < 3:
            anchors = [i for i in range(18) if valid[i]]
        if len(anchors) < 3:
            nan = np.full((18, 3), np.nan, dtype=np.float32)
            return InferenceResult(
                x_opt=np.full((16,), np.nan, dtype=np.float32),
                kp_pred_aligned=nan,
                R=np.eye(3, dtype=np.float32),
                t=np.zeros(3, dtype=np.float32),
                diff=nan,
                per_l2=np.full((18,), np.nan, dtype=np.float32),
                mean_l2=float("nan"),
                rmse=float("nan"),
                used_anchors=anchors,
            )

        # initialize / warm-start
        if self._x_prev is None:
            x = torch.zeros(16, device=device, dtype=torch.float32, requires_grad=True)
        else:
            x = torch.tensor(self._x_prev, device=device, dtype=torch.float32, requires_grad=True)

        opt = torch.optim.Adam([x], lr=self.cfg.lr)
        sigma2 = float(self.cfg.sigma_obs ** 2)

        vmask = torch.tensor(valid, device=device)

        for _ in range(self.cfg.iters):
            opt.zero_grad()

            angles = _angles_from_vector(x)
            kp_pred = self.model(angles)  # (18,3) canonical

            # align predicted -> observed using anchors
            srcA = kp_pred[anchors]
            dstA = live[anchors]
            R, t = kabsch_torch(srcA, dstA)

            kp_pred_aligned = kp_pred @ R.t() + t
            diff = live - kp_pred_aligned

            # Likelihood term (Gaussian)
            diff_valid = diff[vmask]
            accuracy = 0.5 * (diff_valid.pow(2).sum(dim=1).mean() / sigma2)

            # Priors
            plim = joint_limits_prior(x, self.lim)
            psym = symmetry_prior(angles)

            F = accuracy + self.cfg.w_limits * plim + self.cfg.w_sym * psym
            F.backward()
            opt.step()

        # final outputs
        with torch.no_grad():
            angles = _angles_from_vector(x)
            kp_pred = self.model(angles)
            R, t = kabsch_torch(kp_pred[anchors], live[anchors])
            kp_pred_aligned = kp_pred @ R.t() + t
            diff = live - kp_pred_aligned

            diff_np = diff.cpu().numpy()
            pred_np = kp_pred_aligned.cpu().numpy()

            per_l2 = np.full((18,), np.nan, dtype=np.float32)
            for i in range(18):
                if valid[i]:
                    per_l2[i] = float(np.linalg.norm(diff_np[i]))

            vals = per_l2[np.isfinite(per_l2)]
            mean_l2 = float(np.mean(vals)) if vals.size else float("nan")
            rmse = float(np.sqrt(np.mean(vals * vals))) if vals.size else float("nan")

        self._x_prev = x.detach().cpu().numpy().copy()

        return InferenceResult(
            x_opt=self._x_prev.copy(),
            kp_pred_aligned=pred_np,
            R=R.cpu().numpy(),
            t=t.cpu().numpy(),
            diff=diff_np,
            per_l2=per_l2,
            mean_l2=mean_l2,
            rmse=rmse,
            used_anchors=anchors,
        )
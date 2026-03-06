import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from human_kinematic_model import HumanKinematicModel, default_joint_limits_radians


def valid_mask_np(kp: np.ndarray) -> np.ndarray:
    return np.isfinite(kp).all(axis=1)


def kabsch_torch(src: torch.Tensor, dst: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable rigid alignment (rotation+translation), no scale.
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

    # Proper rotation (avoid reflection)
    d = torch.sign(torch.linalg.det(V @ U.t()))
    d = torch.where(d == 0, torch.tensor(1.0, device=d.device, dtype=d.dtype), d)
    D = torch.diag(torch.stack([
        torch.tensor(1.0, device=d.device),
        torch.tensor(1.0, device=d.device),
        d
    ]))
    R = V @ D @ U.t()
    t = dst_mean - (src_mean @ R.t())
    return R, t


def _angles_from_vector(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    x: (11,) ordered:
      sh_L(3), sh_R(3), el_L(1), el_R(1), lb_x(1), lb_z(1), lb_roll(1)
    """
    i = 0
    sh_L = x[i:i+3]; i += 3
    sh_R = x[i:i+3]; i += 3
    el_L = x[i:i+1]; i += 1
    el_R = x[i:i+1]; i += 1
    lb_x = x[i:i+1]; i += 1
    lb_z = x[i:i+1]; i += 1
    lb_roll = x[i:i+1]; i += 1
    return {
        "sh_L": sh_L, "sh_R": sh_R,
        "el_L": el_L, "el_R": el_R,
        "lb_x": lb_x, "lb_z": lb_z, "lb_roll": lb_roll
    }


def symmetry_prior(angles: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Soft left/right symmetry for upper body only:
      yaw mirrored (L yaw ≈ -R yaw),
      pitch same (L pitch ≈ R pitch),
      roll mirrored (L roll ≈ -R roll),
      elbow flex same.
    Lower body is treated as single rigid part (no symmetry constraint).
    """
    shL, shR = angles["sh_L"], angles["sh_R"]
    elL, elR = angles["el_L"], angles["el_R"]

    sh_err = (shL[0] + shR[0])**2 + (shL[1] - shR[1])**2 + (shL[2] + shR[2])**2
    hinge_err = (elL[0] - elR[0])**2
    return sh_err + hinge_err


def joint_limits_prior(x: torch.Tensor, lim: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """
    Soft penalty for violating joint limits.
    relu(min - v)^2 + relu(v - max)^2
    """
    angles = _angles_from_vector(x)

    sh_min, sh_max = lim["sh"]
    el_min, el_max = lim["el"]
    lb_x_min, lb_x_max = lim["lb_x"]
    lb_z_min, lb_z_max = lim["lb_z"]
    lb_roll_min, lb_roll_max = lim["lb_roll"]

    def penalty(v, vmin, vmax):
        return torch.relu(vmin - v).pow(2).sum() + torch.relu(v - vmax).pow(2).sum()

    p = torch.tensor(0.0, device=x.device)
    p = p + penalty(angles["sh_L"], sh_min, sh_max)
    p = p + penalty(angles["sh_R"], sh_min, sh_max)
    p = p + penalty(angles["el_L"], el_min, el_max)
    p = p + penalty(angles["el_R"], el_min, el_max)
    p = p + penalty(angles["lb_x"], lb_x_min, lb_x_max)
    p = p + penalty(angles["lb_z"], lb_z_min, lb_z_max)
    p = p + penalty(angles["lb_roll"], lb_roll_min, lb_roll_max)
    return p


@dataclass
class InferenceConfig:
    anchors: List[int]
    device: str = "cpu"

    # Observation noise base (used when no confidence provided)
    sigma_obs: float = 0.06

    # Dynamics prior noise (how fast angles can change frame-to-frame)
    sigma_dyn: float = 0.25

    # Precision mapping from confidence (optional)
    sigma_min: float = 0.02
    sigma_max: float = 0.15

    # Prior weights
    w_limits: float = 5.0
    w_sym: float = 1.0

    # AInf update settings
    gn_steps: int = 2         # Gauss–Newton / Laplace steps per frame
    damping: float = 1e-3     # Levenberg damping for stability


@dataclass
class InferenceResult:
    mu: np.ndarray                 # (16,) posterior mean of angles
    Lambda: np.ndarray             # (16,16) posterior precision approx
    kp_pred_aligned: np.ndarray    # (18,3) prediction aligned to live
    R: np.ndarray                  # (3,3)
    t: np.ndarray                  # (3,)
    diff: np.ndarray               # (18,3)
    per_l2: np.ndarray             # (18,)
    mean_l2: float
    rmse: float
    used_anchors: List[int]
    valid_count: int
    uncertainty_trace: float       # trace(cov) as a simple uncertainty proxy


class AInfLaplacePoseEstimator:
    """
    Active Inference (Laplace approximation):
      Maintain belief q(theta) ~ N(mu, Sigma)
      Update mu by minimizing free energy with a Gauss–Newton/Laplace step:
        delta = H^{-1} g,  mu <- mu - delta
      where:
        H ≈ J^T J + Lambda_dyn + Hessian(priors)
        g = grad(F)
      Uses precision-weighted prediction errors (optionally from ZED confidence).
    """

    def __init__(self, model: HumanKinematicModel, cfg: InferenceConfig):
        self.model = model
        self.cfg = cfg
        self.lim = default_joint_limits_radians(device=cfg.device)

        D = 11
        self.mu = torch.zeros(D, device=cfg.device, dtype=torch.float32)
        self.Lambda = torch.eye(D, device=cfg.device, dtype=torch.float32) * 1.0

    def _joint_precisions(
        self,
        valid_mask: np.ndarray,
        kp_conf: Optional[np.ndarray],
        device: str
    ) -> torch.Tensor:
        """
        Returns per-valid-joint precision pi (M,) where M = count(valid joints).
        If kp_conf is None: constant precision 1/sigma_obs^2.
        Else: sigma(conf) mapped to [sigma_min, sigma_max].
        """
        if kp_conf is None:
            pi = torch.ones(int(valid_mask.sum()), device=device, dtype=torch.float32) * (1.0 / (self.cfg.sigma_obs ** 2))
            return pi

        conf = kp_conf.astype(np.float32)
        confv = conf[valid_mask]
        confv = np.clip(confv, 0.0, 100.0) / 100.0

        sigma = self.cfg.sigma_min + (self.cfg.sigma_max - self.cfg.sigma_min) * (1.0 - confv)
        pi = 1.0 / (sigma ** 2)
        return torch.tensor(pi, device=device, dtype=torch.float32)

    def _predict_aligned(self, mu_var: torch.Tensor, live: torch.Tensor, anchors: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates canonical predicted kp from mu_var, then aligns it to live using Kabsch on anchors.
        Returns (pred_aligned, R, t)
        """
        angles = _angles_from_vector(mu_var)
        kp_pred = self.model(angles)                      # (18,3) canonical
        R, t = kabsch_torch(kp_pred[anchors], live[anchors])
        pred_aligned = kp_pred @ R.t() + t
        return pred_aligned, R, t

    def infer(self, live_kp_np: np.ndarray, kp_conf_np: Optional[np.ndarray] = None) -> InferenceResult:
        device = self.cfg.device
        live = torch.tensor(live_kp_np, dtype=torch.float32, device=device)

        valid = valid_mask_np(live_kp_np)
        valid_count = int(valid.sum())

        # Choose anchors that are valid
        anchors = [i for i in self.cfg.anchors if valid[i]]
        if len(anchors) < 3:
            anchors = [i for i in range(18) if valid[i]]
        if len(anchors) < 3:
            nan = np.full((18, 3), np.nan, dtype=np.float32)
            return InferenceResult(
                mu=self.mu.detach().cpu().numpy(),
                Lambda=self.Lambda.detach().cpu().numpy(),
                kp_pred_aligned=nan,
                R=np.eye(3, dtype=np.float32),
                t=np.zeros(3, dtype=np.float32),
                diff=nan,
                per_l2=np.full((18,), np.nan, dtype=np.float32),
                mean_l2=float("nan"),
                rmse=float("nan"),
                used_anchors=anchors,
                valid_count=valid_count,
                uncertainty_trace=float("inf"),
            )

        # Dynamics prior: theta_t ~ N(theta_{t-1}, sigma_dyn^2 I)
        D = 11
        mu_prior = self.mu.clone()
        Lambda_dyn = torch.eye(D, device=device, dtype=torch.float32) * (1.0 / (self.cfg.sigma_dyn ** 2))

        # Per-joint precision (valid joints only)
        pi = self._joint_precisions(valid, kp_conf_np, device=device)  # (M,)

        # Gauss–Newton/Laplace steps
        for _ in range(self.cfg.gn_steps):
            mu_var = self.mu.clone().detach().requires_grad_(True)

            pred_aligned, R, t = self._predict_aligned(mu_var, live, anchors)
            diff = live - pred_aligned                             # (18,3)

            vmask_t = torch.tensor(valid, device=device)
            diff_v = diff[vmask_t]                                 # (M,3)

            # Residual vector r (M*3,) with sqrt precision
            # r = sqrt(pi) * diff (applied per joint)
            w = torch.sqrt(pi).unsqueeze(1)                         # (M,1)
            r = (w * diff_v).reshape(-1)                            # (M*3,)

            # Likelihood free energy term: 0.5 * ||r||^2 / M  (mean scaling)
            F_like = 0.5 * (r @ r) / max(1.0, float(pi.numel()))

            # Priors (complexity terms)
            dmu = (mu_var - mu_prior)
            F_dyn = 0.5 * (dmu[None, :] @ Lambda_dyn @ dmu[:, None]).squeeze()

            angles = _angles_from_vector(mu_var)
            F_sym = symmetry_prior(angles)
            F_lim = joint_limits_prior(mu_var, self.lim)

            F = F_like + F_dyn + self.cfg.w_sym * F_sym + self.cfg.w_limits * F_lim

            # Gradient of free energy
            g = torch.autograd.grad(F, mu_var, create_graph=False)[0]  # (D,)

            # Jacobian J = dr/dmu for Gauss–Newton Hessian
            # For D=16 and M<=18, this is feasible.
            def r_of(z):
                predA, _, _ = self._predict_aligned(z, live, anchors)
                d = (live - predA)[vmask_t]
                return (w * d).reshape(-1)

            J = torch.autograd.functional.jacobian(r_of, mu_var, create_graph=False)  # (M*3, D)

            # Gauss–Newton Hessian for likelihood: H_like = J^T J / M
            H_like = (J.t() @ J) / max(1.0, float(pi.numel()))

            # Prior Hessian approximation:
            # - dynamics prior contributes Lambda_dyn exactly (quadratic)
            # - for symmetry/limits, we approximate with diagonal damping only (stable + cheap)
            H_prior = Lambda_dyn

            H = H_like + H_prior + (self.cfg.damping * torch.eye(D, device=device))
            # Solve H delta = g
            delta = torch.linalg.solve(H, g)

            # Update mean
            self.mu = (mu_var.detach() - delta.detach())

            # Update precision estimate (Laplace): Lambda ≈ H
            self.Lambda = H.detach()

        # Final forward pass for outputs
        with torch.no_grad():
            pred_aligned, R, t = self._predict_aligned(self.mu, live, anchors)
            diff = live - pred_aligned

            diff_np = diff.cpu().numpy()
            pred_np = pred_aligned.cpu().numpy()

            per_l2 = np.full((18,), np.nan, dtype=np.float32)
            for i in range(18):
                if valid[i]:
                    per_l2[i] = float(np.linalg.norm(diff_np[i]))

            vals = per_l2[np.isfinite(per_l2)]
            mean_l2 = float(np.mean(vals)) if vals.size else float("nan")
            rmse = float(np.sqrt(np.mean(vals * vals))) if vals.size else float("nan")

            # Uncertainty proxy: trace of covariance
            # cov = inv(Lambda)
            try:
                cov = torch.linalg.inv(self.Lambda)
                uncertainty_trace = float(torch.trace(cov).cpu().item())
            except Exception:
                uncertainty_trace = float("inf")

        return InferenceResult(
            mu=self.mu.detach().cpu().numpy(),
            Lambda=self.Lambda.detach().cpu().numpy(),
            kp_pred_aligned=pred_np,
            R=R.cpu().numpy(),
            t=t.cpu().numpy(),
            diff=diff_np,
            per_l2=per_l2,
            mean_l2=mean_l2,
            rmse=rmse,
            used_anchors=anchors,
            valid_count=valid_count,
            uncertainty_trace=uncertainty_trace,
        )
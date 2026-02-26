import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class CompareResult:
    aligned_standard: np.ndarray          # (18,3)
    diff: np.ndarray                      # live - aligned_standard, (18,3)
    per_joint_l2: np.ndarray              # (18,)
    rmse: float
    mean_l2: float
    used_indices: List[int]
    scale: float
    R: np.ndarray                         # (3,3)
    t: np.ndarray                         # (3,)

def _valid_mask(kp: np.ndarray) -> np.ndarray:
    # Valid if finite in all coords
    return np.isfinite(kp).all(axis=1)

def kabsch_align(
    src: np.ndarray,
    dst: np.ndarray,
    allow_scale: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Finds transform mapping src -> dst:
      dst ≈ s * R * src + t
    Returns (s, R, t).
    """
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    X = src - src_mean
    Y = dst - dst_mean

    # Optional scale (similarity transform)
    s = 1.0
    if allow_scale:
        denom = (X * X).sum()
        if denom > 1e-12:
            s = np.sqrt((Y * Y).sum() / denom)

    Xs = s * X

    H = Xs.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = dst_mean - (R @ (s * src_mean))
    return s, R, t

class SkeletonComparator:
    def __init__(
        self,
        anchor_indices: Optional[List[int]] = None,
        allow_scale: bool = False,
    ):
        # Good default anchors: shoulders + hips + neck
        self.anchor_indices = anchor_indices or [1, 2, 5, 8, 11]
        self.allow_scale = allow_scale

    def compare(self, live_kp: np.ndarray, standard_kp: np.ndarray) -> CompareResult:
        if live_kp.shape != (18, 3) or standard_kp.shape != (18, 3):
            raise ValueError("Both live_kp and standard_kp must be shape (18,3)")

        live_ok = _valid_mask(live_kp)
        std_ok = _valid_mask(standard_kp)

        # choose anchors that are valid in both
        used = [i for i in self.anchor_indices if live_ok[i] and std_ok[i]]
        if len(used) < 3:
            # fallback to any valid shared joints
            used = [i for i in range(18) if live_ok[i] and std_ok[i]]
        if len(used) < 3:
            # cannot align
            aligned = np.full_like(standard_kp, np.nan)
            diff = np.full_like(live_kp, np.nan)
            per = np.full((18,), np.nan)
            return CompareResult(aligned, diff, per, float("nan"), float("nan"), used, 1.0, np.eye(3), np.zeros(3))

        src = standard_kp[used]
        dst = live_kp[used]

        s, R, t = kabsch_align(src, dst, allow_scale=self.allow_scale)

        aligned = (standard_kp @ (R.T)) * s + t  # (18,3)
        diff = live_kp - aligned

        # per-joint L2 error where live is valid
        per_joint_l2 = np.full((18,), np.nan, dtype=float)
        valid = live_ok & np.isfinite(aligned).all(axis=1)
        if valid.any():
            per_joint_l2[valid] = np.linalg.norm(diff[valid], axis=1)

        # Summary metrics over valid joints
        vals = per_joint_l2[np.isfinite(per_joint_l2)]
        rmse = float(np.sqrt(np.mean(vals * vals))) if vals.size else float("nan")
        mean_l2 = float(np.mean(vals)) if vals.size else float("nan")

        return CompareResult(
            aligned_standard=aligned,
            diff=diff,
            per_joint_l2=per_joint_l2,
            rmse=rmse,
            mean_l2=mean_l2,
            used_indices=used,
            scale=float(s),
            R=R,
            t=t,
        )
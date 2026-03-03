# Variable Mapping Reference

## Parameter Vector Layout

### 11-DOF Vector Structure
```
x = [sh_L_yaw, sh_L_pitch, sh_L_roll, 
     sh_R_yaw, sh_R_pitch, sh_R_roll,
     el_L_flex,
     el_R_flex,
     lb_x, lb_z, lb_roll]

     └─ 3 ──────┘ └─ 3 ───────┘ └1┘ └1┘ └─────── 3 ────────┘
      Left Shoulder  Right Shoulder  Elbows   Lower Body
```

**Indexing in applications:**
- `x[0:3]` = sh_L (yaw, pitch, roll)
- `x[3:6]` = sh_R (yaw, pitch, roll)
- `x[6]` = el_L (flexion)
- `x[7]` = el_R (flexion)
- `x[8]` = lb_x (horizontal translation)
- `x[9]` = lb_z (horizontal translation)
- `x[10]` = lb_roll (rotation about Y axis)

---

## Joint Limits

### Shoulder Limits (both sides)
- **yaw**: ±90° (side-to-side rotation)
- **pitch**: ±90° (forward/back)
- **roll**: ±90° (shoulder rotation)

### Elbow Limits (both sides)
- **flexion**: 0° to 150° (straight arm at 0°)

### Lower Body Limits
- **lb_x**: -1.0 to +1.0 meters (side stepping)
- **lb_z**: -1.0 to +1.0 meters (forward/back walking)
- **lb_roll**: ±45° (body twist about vertical axis)

---

## Free Energy Components

### Likelihood (Observation Fidelity)
```
F_like = 0.5 * (1/M) * Σ √πᵢ ||(live_kp - pred_kp)||²

where:
  M = number of valid keypoints
  πᵢ = precision of keypoint i (from confidence or default)
  live_kp = ZED-detected keypoint
  pred_kp = forward kinematics + alignment
```

### Dynamics (Temporal Consistency)
```
F_dyn = 0.5 * (θ - θ_prev)ᵀ Λ_dyn (θ - θ_prev)

where:
  Λ_dyn = (σ_dyn)⁻² * I₁₁
  σ_dyn = 0.25 (default) → allows ~14° change per frame
```

### Symmetry (Over-parameterization Penalty)
```
F_sym = (sh_L_yaw + sh_R_yaw)² 
        + (sh_L_pitch - sh_R_pitch)²
        + (sh_L_roll + sh_R_roll)²
        + (el_L - el_R)²

No hip/knee symmetry (single rigid lower body)
```

### Limits (Anatomical Constraints)
```
F_limits = Σ [ReLU(θ_min - θ) + ReLU(θ - θ_max)]²

Applied to all 11 parameters
```

### Total Free Energy
```
F_total = F_like + F_dyn + w_sym * F_sym + w_limits * F_limits

Default weights:
  w_sym = 1.0
  w_limits = 5.0
```

---

## Keypoint Indices (BODY_18)

```
Upper Body (optimized):
  1:  NECK
  2:  R_SHOULDER
  3:  R_ELBOW (depends on sh_R + el_R)
  4:  R_WRIST
  5:  L_SHOULDER
  6:  L_ELBOW (depends on sh_L + el_L)
  7:  L_WRIST

Lower Body (rigid):
  8:  R_HIP (depends on lb_x, lb_z, lb_roll)
  9:  R_KNEE (depends on lb_x, lb_z, lb_roll)
  10: R_ANKLE
  11: L_HIP
  12: L_KNEE
  13: L_ANKLE

Face (fixed):
  0:  NOSE
  14: R_EYE
  15: L_EYE
  16: R_EAR
  17: L_EAR
```

---

## Code References

### Parameter Extraction
**File**: `vfe_inference.py`, function `_angles_from_vector()`
```python
angles = {
    "sh_L": x[0:3],      # 3 DOF
    "sh_R": x[3:6],      # 3 DOF
    "el_L": x[6:7],      # 1 DOF
    "el_R": x[7:8],      # 1 DOF
    "lb_x": x[8:9],      # 1 DOF
    "lb_z": x[9:10],     # 1 DOF
    "lb_roll": x[10:11]  # 1 DOF
}
```

### Forward Kinematics
**File**: `human_kinematic_model.py`, method `HumanKinematicModel.forward()`
```python
# Lower body as rigid part
pelvis = [angles["lb_x"], 0.0, angles["lb_z"]]
R_lb = Ry(angles["lb_roll"])

# Apply to both legs identically
for leg in [left, right]:
    hip = pelvis + R_lb @ hip_offset_local
    knee = hip + R_lb @ v_thigh
    ankle = knee + R_lb @ v_calf
```

### Joint Limits
**File**: `human_kinematic_model.py`, function `default_joint_limits_radians()`
```python
limits = {
    "sh": ([-90°, -90°, -90°], [90°, 90°, 90°]),
    "el": ([0°], [150°]),
    "lb_x": ([-1.0m], [1.0m]),
    "lb_z": ([-1.0m], [1.0m]),
    "lb_roll": ([-45°], [45°])
}
```

---

## Optimization Details

### Gauss-Newton Hessian
```
H = J^T Q J + Λ_dyn + damping*I

where:
  J = ∂(pred_kp)/∂θ        (11×(M*3) Jacobian)
  Q = diag(π₁, π₁, π₁, ...) (precision matrix)
  Λ_dyn = (σ_dyn)⁻² I₁₁
  damping = 1e-3 (Levenberg)
```

### Update Step
```
δ = H⁻¹ ∇F
θ ← θ - δ

Repeated for gn_steps iterations (default: 2)
```

### Posterior Uncertainty
```
Λ ≈ H (Laplace approximation)
Σ = Λ⁻¹ (posterior covariance)
uncertainty_trace = tr(Σ)
```

---

## Default Configuration

**File**: `main.py`
```bash
python main.py \
  --anchors 1,2,5,8,11 \
  --device cpu \
  --sigma_obs 0.06 \
  --sigma_dyn 0.25 \
  --gn_steps 2 \
  --damping 1e-3 \
  --w_limits 5.0 \
  --w_sym 1.0 \
  --sigma_min 0.02 \
  --sigma_max 0.15 \
  --min_valid 12 \
  --uncertainty_thresh 0.05
```

---

## Migration Notes (if coming from 16-DOF)

| 16-DOF | 11-DOF | Status |
|--------|--------|--------|
| hip_L (3 DOF) | → | lb_x, lb_z, lb_roll |
| hip_R (3 DOF) | → | (same as above) |
| kn_L (1 DOF) | → | (rigidly follows lower body) |
| kn_R (1 DOF) | → | (rigidly follows lower body) |
| sh_L, sh_R | → | sh_L, sh_R ✓ unchanged |
| el_L, el_R | → | el_L, el_R ✓ unchanged |

**Key**: Lower body now single rigid part → fewer keyframe variations but physics-grounded walking.

# System Refactoring: 16 DOF → 11 DOF (Upper Body + Rigid Lower Body)

## Overview
Successfully refactored the human kinematic model from 16 degrees of freedom to 11 degrees of freedom while preserving the core Active Inference methodology. The lower body (hips, knees, ankles) is now treated as a single rigid part constrained to ground plane movement.

---

## Variable Reduction

### Before (16 DOF)
- **Left Shoulder**: yaw, pitch, roll (3)
- **Right Shoulder**: yaw, pitch, roll (3)
- **Left Elbow**: flexion (1)
- **Right Elbow**: flexion (1)
- **Left Hip**: yaw, pitch, roll (3)
- **Right Hip**: yaw, pitch, roll (3)
- **Left Knee**: flexion (1)
- **Right Knee**: flexion (1)

### After (11 DOF)
- **Left Shoulder**: yaw, pitch, roll (3)
- **Right Shoulder**: yaw, pitch, roll (3)
- **Left Elbow**: flexion (1)
- **Right Elbow**: flexion (1)
- **Lower Body** (rigid ground plane):
  - x translation (1)
  - z translation (1)
  - roll rotation about Y axis (1)

---

## Files Modified

### 1. `human_kinematic_model.py`

#### Changes:
- **Class docstring**: Updated to reflect 11 DOF model
- **`forward()` method**: 
  - Removed individual hip and knee rotation matrices (`R_hip_L`, `R_hip_R`, `R_kn_L`, `R_kn_R`)
  - Added lower body rigid transform:
    - `lb_x`, `lb_z`: ground plane translations
    - `lb_roll`: rotation about Y axis
    - `R_lb`: single rotation matrix for entire lower body
  - Pelvis now positioned at `[lb_x, 0.0, lb_z]` (Y always 0 for ground constraint)
  - Thigh and calf vectors rotated by `R_lb` together (rigid motion)
  - Hip positions computed from pelvis + rotated offsets
  - Knee and ankle positions follow rigid lower body transformation

- **`default_joint_limits_radians()` function**:
  - Removed hip and knee limit entries
  - Added lower body limits:
    - `lb_x`: [-1.0, 1.0] meters (x-axis range)
    - `lb_z`: [-1.0, 1.0] meters (z-axis range)
    - `lb_roll`: [-45°, 45°] (yaw rotation)

#### Core Methodology Preserved:
✓ Euler angle conventions (YPR for shoulders/arms)  
✓ Segment length constraints (fixed from template)  
✓ Forward kinematics chain structure  
✓ Face keypoints as fixed offsets  

---

### 2. `vfe_inference.py`

#### Changes:

**`_angles_from_vector()` function**:
- Updated vector parsing from 16→11 elements
- New layout: `sh_L(3), sh_R(3), el_L(1), el_R(1), lb_x(1), lb_z(1), lb_roll(1)`
- Returns dictionary with keys: `sh_L, sh_R, el_L, el_R, lb_x, lb_z, lb_roll`

**`symmetry_prior()` function**:
- Removed hip and knee symmetry constraints
- Upper body only: shoulders (yaw/pitch/roll) and elbows (flexion)
- Lower body treated as single rigid part (no left/right symmetry enforcement)
- Reduced symmetry error computation

**`joint_limits_prior()` function**:
- Removed hip and knee penalty computations
- Added penalties for `lb_x`, `lb_z`, `lb_roll` constraints
- Maintains quadratic penalty structure

**`AInfLaplacePoseEstimator.__init__()`**:
- Changed D from 16 to 11
- Updated initialization: `mu` and `Lambda` are now (11×1) and (11×11) tensors

**Gauss-Newton Hessian computation**:
- Updated dimension D=11 in `infer()` method
- Jacobian J is now (M*3 × 11) instead of (M*3 × 16)
- Hessian H is now (11×11) instead of (16×16)
- All other computations (gradient, damping, solve) scale correctly

#### Core Methodology Preserved:
✓ Free energy minimization: F = F_likelihood + F_dynamics + w_sym × F_sym + w_limits × F_limits  
✓ Laplace approximation: q(θ) ~ N(μ, Λ⁻¹)  
✓ Precision-weighted residuals (sqrt(π) weighting)  
✓ Kabsch alignment on anchor points  
✓ Uncertainty tracking via trace(Λ⁻¹)  

---

### 3. `main.py`

#### Changes:
- ✓ No changes required
- Anchor points (default: `1,2,5,8,11`) still valid for upper body joints
- Configuration parameters unchanged

#### Notes:
- Lower body parameters (`--w_limits`, `--w_sym`, `--sigma_dyn`) still apply
- Action thresholds (`--min_valid`, `--uncertainty_thresh`) unchanged
- Can add lower body-specific parameters in future if needed

---

## Mathematical Formulation

### Lower Body Kinematics
```
pelvis = [lb_x, 0, lb_z]ᵀ              (ground plane translation)
R_lb = Ry(lb_roll)                     (single rotation about Y)

For each leg:
  hip_pos = pelvis + R_lb @ hip_offset
  knee_pos = hip_pos + R_lb @ v_thigh
  ankle_pos = knee_pos + R_lb @ v_calf
```

### Free Energy (unchanged in structure)
```
F = 1/(2M) Σᵢ √πᵢ ||eᵢ||² + 1/2 (θ - θ_prev)ᵀ Λ_dyn (θ - θ_prev)
    + w_sym F_sym(θ) + w_limits F_limits(θ)

where:
  e = live - predicted       (weighted residuals)
  πᵢ = 1/σᵢ²               (joint precision from confidence)
  F_sym = upper body L/R mirror + elbow symmetry
  F_limits = quadratic penalties on joint bounds
```

---

## Validation

### Syntax Check
✓ All files compile without errors
```
python -m py_compile human_kinematic_model.py vfe_inference.py main.py
```

### Key Invariants Preserved
1. **Kinematic Chain**: pelvis → neck → shoulders → elbows → wrists  
                       pelvis → hips → knees → ankles (now rigid)
2. **Forward Model**: Angles → 18 keypoints (BODY_18 format unchanged)
3. **Observation Model**: Live keypoints → precision-weighted errors
4. **Optimization**: Gauss-Newton/Laplace with dynamics + priors
5. **Uncertainty**: Posterior covariance via Λ⁻¹

---

## Testing Recommendations

Before deploying, verify:

1. **Forward Kinematics**:
   - Generate keypoints from random angles
   - Verify lower body moves as rigid part (consistent rotation)
   - Check pelvis position tied to (x, z) translations

2. **Optimization**:
   - Run inference on single frame
   - Check convergence of mu and uncertainty trace
   - Verify priors penalize violations correctly

3. **Full Pipeline**:
   - Run with ZED stream or SVO file
   - Monitor action hints (high uncertainty → change viewpoint)
   - Validate output skeleton visualization

---

## Performance Notes

- **Computational Savings**: 5 fewer DOF reduces Jacobian from (M×16) → (M×11)
  - ~31% reduction in Hessian computation
  - Faster linear solve (11×11 vs 16×16)
  
- **Physics Plausibility**: Ground plane constraint more realistic for lower body
  - Reduces unphysical leg poses
  - Simplifies symmetry: no need for per-leg angle constraints

---

## Future Extensions

If needed, you can later add:
- Hip flex (forward/back) by adding another DOF
- Ankle flex similarly
- Or separate left/right lower body control (15-16 DOF hybrid)

Current structure provides clean foundation for these extensions.

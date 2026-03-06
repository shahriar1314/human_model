# Code Changes: Side-by-Side Comparison

## 1. Parameter Vector Extraction

### Before (16 DOF)
```python
def _angles_from_vector(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    x: (16,) ordered:
      sh_L(3), sh_R(3), el_L(1), el_R(1), hip_L(3), hip_R(3), kn_L(1), kn_R(1)
    """
    i = 0
    sh_L = x[i:i+3]; i += 3
    sh_R = x[i:i+3]; i += 3
    el_L = x[i:i+1]; i += 1
    el_R = x[i:i+1]; i += 1
    hip_L = x[i:i+3]; i += 3
    hip_R = x[i:i+3]; i += 3
    kn_L = x[i:i+1]; i += 1
    kn_R = x[i:i+1]; i += 1
    return {
        "sh_L": sh_L, "sh_R": sh_R,
        "el_L": el_L, "el_R": el_R,
        "hip_L": hip_L, "hip_R": hip_R,
        "kn_L": kn_L, "kn_R": kn_R
    }
```

### After (11 DOF)
```python
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
```

**Change**: 5 removed lines, 5 added lines (hip_L/R, kn_L/R → lb_x/z/roll)

---

## 2. Symmetry Prior

### Before (16 DOF)
```python
def symmetry_prior(angles: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Soft left/right symmetry:
      yaw mirrored (L yaw ≈ -R yaw),
      pitch same (L pitch ≈ R pitch),
      roll mirrored (L roll ≈ -R roll),
      elbow flex same, knee flex same.
    """
    shL, shR = angles["sh_L"], angles["sh_R"]
    hipL, hipR = angles["hip_L"], angles["hip_R"]
    elL, elR = angles["el_L"], angles["el_R"]
    knL, knR = angles["kn_L"], angles["kn_R"]

    sh_err = (shL[0] + shR[0])**2 + (shL[1] - shR[1])**2 + (shL[2] + shR[2])**2
    hip_err = (hipL[0] + hipR[0])**2 + (hipL[1] - hipR[1])**2 + (hipL[2] + hipR[2])**2
    hinge_err = (elL[0] - elR[0])**2 + (knL[0] - knR[0])**2
    return sh_err + hip_err + hinge_err
```

### After (11 DOF)
```python
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
```

**Change**: Removed 3 lines for hip symmetry, removed 1 line for knee symmetry

---

## 3. Joint Limits Prior

### Before (16 DOF)
```python
def joint_limits_prior(x: torch.Tensor, lim: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
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
```

### After (11 DOF)
```python
def joint_limits_prior(x: torch.Tensor, lim: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
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
```

**Change**: 4 lines removed (hip/knee), 3 lines added (lb_x/z/roll)

---

## 4. Forward Kinematics

### Before (16 DOF) - Key Section
```python
def forward(self, angles: Dict[str, torch.Tensor]) -> torch.Tensor:
    # ...
    pelvis = torch.zeros(3, device=device, dtype=dtype)
    neck = pelvis + torch.tensor([0.0, L.torso, 0.0], ...)

    l_sh = neck + torch.tensor([-L.shoulder_offset_x, 0.0, 0.0], ...)
    r_sh = neck + torch.tensor([ L.shoulder_offset_x, 0.0, 0.0], ...)

    l_hip = pelvis + torch.tensor([-L.hip_offset_x, 0.0, 0.0], ...)
    r_hip = pelvis + torch.tensor([ L.hip_offset_x, 0.0, 0.0], ...)

    R_sh_L = euler_yaw_pitch_roll(angles["sh_L"][0], angles["sh_L"][1], angles["sh_L"][2])
    R_sh_R = euler_yaw_pitch_roll(angles["sh_R"][0], angles["sh_R"][1], angles["sh_R"][2])
    R_hip_L = euler_yaw_pitch_roll(angles["hip_L"][0], angles["hip_L"][1], angles["hip_L"][2])
    R_hip_R = euler_yaw_pitch_roll(angles["hip_R"][0], angles["hip_R"][1], angles["hip_R"][2])

    R_el_L = _Rx(angles["el_L"][0])
    R_el_R = _Rx(angles["el_R"][0])
    R_kn_L = _Rx(angles["kn_L"][0])
    R_kn_R = _Rx(angles["kn_R"][0])

    # ... arm computations use R_sh_L, R_sh_R, R_el_L, R_el_R ...

    v_thigh = torch.tensor([0.0, -L.thigh, 0.0], ...)
    v_calf  = torch.tensor([0.0, -L.calf,  0.0], ...)

    l_kn = l_hip + (R_hip_L @ v_thigh)
    r_kn = r_hip + (R_hip_R @ v_thigh)

    l_an = l_kn + (R_hip_L @ (R_kn_L @ v_calf))
    r_an = r_kn + (R_hip_R @ (R_kn_R @ v_calf))
```

### After (11 DOF) - Same Section
```python
def forward(self, angles: Dict[str, torch.Tensor]) -> torch.Tensor:
    # ...
    # Lower body as rigid part on ground plane
    lb_x = angles["lb_x"][0]
    lb_z = angles["lb_z"][0]
    lb_roll = angles["lb_roll"][0]
    
    pelvis = torch.tensor([lb_x, 0.0, lb_z], device=device, dtype=dtype)
    R_lb = _Ry(lb_roll)
    
    neck = pelvis + torch.tensor([0.0, L.torso, 0.0], ...)

    l_sh = neck + torch.tensor([-L.shoulder_offset_x, 0.0, 0.0], ...)
    r_sh = neck + torch.tensor([ L.shoulder_offset_x, 0.0, 0.0], ...)

    # Hip positions in local frame (before rotation)
    l_hip_local = torch.tensor([-L.hip_offset_x, 0.0, 0.0], ...)
    r_hip_local = torch.tensor([ L.hip_offset_x, 0.0, 0.0], ...)
    
    # Apply lower body rotation and add to pelvis
    l_hip = pelvis + (R_lb @ l_hip_local)
    r_hip = pelvis + (R_lb @ r_hip_local)

    R_sh_L = euler_yaw_pitch_roll(angles["sh_L"][0], angles["sh_L"][1], angles["sh_L"][2])
    R_sh_R = euler_yaw_pitch_roll(angles["sh_R"][0], angles["sh_R"][1], angles["sh_R"][2])

    R_el_L = _Rx(angles["el_L"][0])
    R_el_R = _Rx(angles["el_R"][0])

    # ... arm computations use R_sh_L, R_sh_R, R_el_L, R_el_R ...

    v_thigh = torch.tensor([0.0, -L.thigh, 0.0], ...)
    v_calf  = torch.tensor([0.0, -L.calf,  0.0], ...)

    # Apply lower body rotation to limb segments
    v_thigh_rot = R_lb @ v_thigh
    v_calf_rot = R_lb @ v_calf
    
    l_kn = l_hip + v_thigh_rot
    r_kn = r_hip + v_thigh_rot

    l_an = l_kn + v_calf_rot
    r_an = r_kn + v_calf_rot
```

**Key Changes**:
- Pelvis: `[0, 0, 0]` → `[lb_x, 0, lb_z]` (ground plane movement)
- Hip rotation: Individual `R_hip_L/R` → Single `R_lb` (rigid coupling)
- Knee rotation: Removed (no flexion, just rigid transform)
- Leg segments: Rotated by `R_lb` (both legs identical motion)

---

## 5. Joint Limits Definition

### Before (16 DOF)
```python
limits = {
    "sh":  (to_rad([-90.0, -90.0, -90.0]), to_rad([ 90.0,  90.0,  90.0])),
    "hip": (to_rad([-60.0, -90.0, -45.0]), to_rad([ 60.0,  60.0,  45.0])),
    "el":  (to_rad([  0.0]),               to_rad([150.0])),
    "kn":  (to_rad([  0.0]),               to_rad([160.0])),
}
```

### After (11 DOF)
```python
limits = {
    "sh":  (to_rad([-90.0, -90.0, -90.0]), to_rad([ 90.0,  90.0,  90.0])),
    "el":  (to_rad([  0.0]),               to_rad([150.0])),
    "lb_x": (torch.tensor([-1.0], ...), torch.tensor([1.0], ...)),
    "lb_z": (torch.tensor([-1.0], ...), torch.tensor([1.0], ...)),
    "lb_roll": (to_rad([-45.0]), to_rad([45.0])),
}
```

**Change**: Removed "hip" and "kn" entries, added "lb_x", "lb_z", "lb_roll"

---

## 6. Initialization Dimension

### Before (16 DOF)
```python
D = 16
self.mu = torch.zeros(D, device=cfg.device, dtype=torch.float32)
self.Lambda = torch.eye(D, device=cfg.device, dtype=torch.float32) * 1.0
```

### After (11 DOF)
```python
D = 11
self.mu = torch.zeros(D, device=cfg.device, dtype=torch.float32)
self.Lambda = torch.eye(D, device=cfg.device, dtype=torch.float32) * 1.0
```

**Change**: Single line: `D = 16` → `D = 11`

---

## Summary Statistics

| Aspect | Before | After | Δ |
|--------|--------|-------|---|
| Total DOF | 16 | 11 | -5 |
| `_angles_from_vector()` | 16 lines | 11 lines | -5 |
| `symmetry_prior()` | 13 lines | 11 lines | -2 |
| `joint_limits_prior()` | 20 lines | 21 lines | +1 |
| `forward()` | 60 lines | 68 lines | +8 |
| Hessian size | 16×16 | 11×11 | 31% smaller |
| Jacobian width | M×16 | M×11 | 31% smaller |

**Net code impact**: 
- Removed individual hip/knee rotation matrices
- Added unified lower body rigid transform
- Upper body arm code completely unchanged
- Net file changes: ~15 lines modified globally


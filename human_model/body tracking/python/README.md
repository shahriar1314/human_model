# Active Inference Human Model (ZED BODY_18)

This project implements a **real-time human pose modeling system** using:

* **ZED BODY_18 3D keypoints** (observations)
* A **kinematic human model** parameterized by joint angles
* **Variational Free Energy (VFE) minimization**
* **Active Inference principles**

Instead of directly fitting 18×3 keypoints, the system infers a compact set of **main joint angles** (shoulders, elbows, hips, knees) that best explain the observed pose while respecting symmetry and joint limits.

---

# 1. Conceptual Overview

## 1.1 What Problem Is Solved?

Given:

* Real-time 3D keypoints from ZED

We want:

* A structured, interpretable human model
* That respects kinematics
* And minimizes prediction error under priors

We treat:

* Joint angles = latent variables
* ZED keypoints = observations
* Free Energy = objective to minimize

---

# 2. System Architecture

```text
ai_human_model/
│
├── main.py
├── zed_body18_stream.py
├── human_kinematic_model.py
└── vfe_inference.py
```

---

# 3. Generative Model (human_kinematic_model.py)

## 3.1 Parameterization

We **do NOT optimize 18×3 points**.

Instead, we optimize **16 joint angles**:

| Joint          | DOF                  |
| -------------- | -------------------- |
| Left shoulder  | yaw, pitch, roll (3) |
| Right shoulder | yaw, pitch, roll (3) |
| Left elbow     | flexion (1)          |
| Right elbow    | flexion (1)          |
| Left hip       | yaw, pitch, roll (3) |
| Right hip      | yaw, pitch, roll (3) |
| Left knee      | flexion (1)          |
| Right knee     | flexion (1)          |

Total parameters: **16**

Face joints (eyes, ears, nose) are fixed offsets from neck and not optimized.

---

## 3.2 Fixed Segment Lengths

Segment lengths are extracted once from a template skeleton:

* Torso length
* Shoulder width
* Hip width
* Upper arm length
* Lower arm length
* Thigh length
* Calf length

These remain constant (bone lengths not optimized).

---

## 3.3 Forward Kinematics

The forward model does:

1. Pelvis at origin
2. Neck above pelvis
3. Shoulders placed laterally
4. Arms generated via:

   * Shoulder rotation
   * Elbow hinge
5. Legs generated via:

   * Hip rotation
   * Knee hinge
6. Face offsets added

Output:

```
(18,3) canonical keypoints
```

This is the generative model:

[
\hat{o} = g(\theta)
]

---

# 4. Inference via Free Energy (vfe_inference.py)

## 4.1 Alignment (Pose Removal)

Before comparing prediction and observation, we remove global pose:

We compute rigid transform:

[
live \approx R \cdot predicted + t
]

using **Kabsch alignment** on anchor joints (default: neck + shoulders + hips).

This ensures we compare articulation, not global position.

---

## 4.2 Likelihood (Accuracy Term)

Assume Gaussian observation model:

[
p(o | \theta) = \mathcal{N}(o ; \hat{o}, \sigma^2 I)
]

Accuracy term:

[
\text{accuracy} = \frac{1}{2\sigma^2} | o - \hat{o} |^2
]

`sigma` controls noise sensitivity.

---

## 4.3 Priors

### 1. Joint Limits

Soft constraint:

* If angle inside limits → no penalty
* If outside → quadratic penalty

Implemented as:

```
relu(min - angle)^2 + relu(angle - max)^2
```

Limits defined in:

```
default_joint_limits_radians()
```

---

### 2. Symmetry Prior

Encourages left/right similarity:

* Shoulder yaw mirrored
* Shoulder pitch similar
* Elbows similar
* Knees similar

Helps prevent unrealistic asymmetric poses.

---

## 4.4 Free Energy Objective

[
F(\theta) =
\underbrace{\text{prediction error}}*{\text{accuracy}}
+
w*{limits} \cdot \text{limits prior}
+
w_{sym} \cdot \text{symmetry prior}
]

Minimized via Adam optimizer.

---

# 5. Optimization Process

For each frame:

1. Initialize angles (warm start from previous frame)
2. Run Adam for `iters` steps
3. Compute:

   * Predicted aligned keypoints
   * Per-joint L2 errors
   * Mean error
   * RMSE

Warm start is crucial for real-time performance.

---

# 6. Active Inference Component

Active inference includes **action to reduce expected free energy**.

In this implementation:

If many joints are invalid (occlusion/low confidence):

```
[Action hint] Move viewpoint / reduce occlusion
```

This approximates reducing uncertainty in observations.

---

# 7. Runtime Characteristics

## 7.1 Parameters

Optimized variables: 16 angles

This is lightweight compared to optimizing 18×3 coordinates.

---

## 7.2 Real-Time Feasibility

Approximate performance:

| Setup | Iterations | Expected Runtime |
| ----- | ---------- | ---------------- |
| CPU   | 60         | 30–200 ms        |
| GPU   | 60         | 3–15 ms          |
| GPU   | 10–20      | 1–5 ms           |

Recommended for 30 FPS:

* 5–20 iterations
* Warm start
* Possibly update every 2–3 frames

---

# 8. Output Interpretation

Printed table:

| Column   | Meaning                          |
| -------- | -------------------------------- |
| live_*   | ZED keypoints                    |
| pred_*   | Model prediction after alignment |
| dx/dy/dz | Residual error                   |
| L2(m)    | 3D Euclidean distance error      |

Mean and RMSE summarize global fit quality.

---

# 9. Usage

## Live camera

```bash
python3 main.py
```

## Control update rate

```bash
python3 main.py --print_every 120
```

## Adjust optimization strength

```bash
python3 main.py --iters 80 --lr 0.02
```

## Replay SVO

```bash
python3 main.py --input_svo_file file.svo2
```

---

# 10. Key Design Decisions

### Why not optimize 18×3 keypoints directly?

Because:

* No anatomical constraints
* No interpretability
* No joint limits
* Overfits noise

Angle-based modeling ensures:

* Physical plausibility
* Lower dimensional state
* Better generalization

---

# 11. Limitations

* No spine articulation
* No ankle rotation
* No neck orientation
* Face points not optimized
* No temporal smoothing beyond warm start

---

# 12. Extensions

To improve realism:

* Add neck yaw/pitch
* Add spine bend
* Add ankle DOF
* Add temporal prior
* Replace per-frame optimization with EKF
* Use amortized neural inference + VFE refinement

---

# 13. Mathematical Summary

Generative model:

[
\hat{o} = Align(g(\theta))
]

Free energy:

[
F(\theta) =
\frac{1}{2\sigma^2}|o - \hat{o}|^2
+
\lambda_1 \text{JointLimits}
+
\lambda_2 \text{Symmetry}
]

Optimization:

[
\theta_{t+1} = \theta_t - \eta \nabla_\theta F
]

---

# 14. Final Interpretation

This system:

* Converts noisy keypoints into structured human kinematics
* Applies principled Bayesian reasoning (Active Inference)
* Maintains physical plausibility via priors
* Works online with iterative belief updates

It is a simplified but conceptually correct Active Inference human pose model suitable for research and experimentation.

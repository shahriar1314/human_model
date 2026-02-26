# 3D Skeleton Comparison: ZED BODY_18 vs Standard Human Model

## Overview

This project compares a **real-time 3D human skeleton** detected using the ZED SDK (BODY_18 format) against a **static standard human skeleton model**.

For each detected person, the system:

1. Retrieves live 3D keypoints (18 joints) from ZED.
2. Aligns the static standard skeleton to the live skeleton using rigid alignment (Kabsch algorithm).
3. Computes per-joint 3D positional error.
4. Outputs:

   * Live joint coordinates
   * Static standard coordinates
   * Aligned standard coordinates
   * Per-joint 3D error (L2 in meters)
   * Mean error and RMSE

This allows you to measure **pose deviation** from a canonical reference model.

---

# System Architecture

```
human_model/body_tracking/python/
│
├── main.py
├── human_standard_model.py
├── skeleton_compare.py
├── zed_body18_stream.py
└── README.md
```

---

# 1. human_standard_model.py

## Purpose

Defines a **static reference human skeleton** in BODY_18 format.

## Key Properties

* 18 joints
* Units: meters
* Coordinate convention:

  * X → left/right
  * Y → up/down
  * Z → forward/back

## Important Function

```python
get_standard_kp3d_standing()
```

Returns:

```
(18,3) numpy array
```

This model does **not change over time**.

Example joints:

* NECK: [0.0, 1.50, 0.0]
* R_WRIST: [0.50, 0.95, 0.0]
* L_WRIST: [-0.50, 0.95, 0.0]

This represents a standing pose with arms relaxed slightly outward.

---

# 2. zed_body18_stream.py

## Purpose

Wraps the ZED SDK body tracking sample into a reusable streaming class.

## What It Does

* Opens ZED camera (or SVO file)
* Enables BODY_18 tracking
* Retrieves 3D keypoints
* Optionally shows OpenGL and 2D viewer
* Yields frame-by-frame data

## Output Per Frame

```python
{
    "image_bgr": image,
    "bodies": [
        {
            "id": int,
            "confidence": int,
            "tracking_state": enum,
            "kp3d": (18,3) numpy array
        }
    ]
}
```

Coordinates are in meters, right-handed Y-up coordinate system.

---

# 3. skeleton_compare.py

## Purpose

Performs geometric alignment and computes error metrics.

---

## 3.1 Alignment (Kabsch Algorithm)

To compare poses meaningfully, the standard model must be aligned to the live skeleton.

We compute:

[
dst \approx s \cdot R \cdot src + t
]

Where:

* R → rotation matrix (3×3)
* t → translation vector (3×1)
* s → optional scale factor

Alignment uses selected anchor joints (default):

```
[neck, r_shoulder, l_shoulder, r_hip, l_hip]
```

This removes:

* Global translation
* Global rotation
* (Optionally) global scale

Thus the comparison measures **articulation differences only**.

---

## 3.2 Transform Application

```python
aligned = (standard_kp @ R.T) * s + t
```

This produces:

* `aligned_standard` → standard model expressed in live frame

Important:

* Raw standard model never changes
* The aligned version changes every frame (because the person moves)

---

## 3.3 Error Computation

Per joint:

[
diff_i = live_i - aligned_i
]

L2 error:

[
L2_i = \sqrt{dx^2 + dy^2 + dz^2}
]

Units: meters

---

## 3.4 Summary Metrics

Mean error:
[
\text{mean} = \frac{1}{N} \sum_i L2_i
]

RMSE:
[
\text{RMSE} = \sqrt{\frac{1}{N} \sum_i L2_i^2}
]

RMSE penalizes large deviations more strongly.

---

# 4. main.py

## Purpose

Integrates all modules.

## Workflow

1. Load static standard model
2. Start ZED stream
3. For each frame:

   * Extract live keypoints
   * Align standard model
   * Compute per-joint error
   * Print results at defined interval

---

## Printing Frequency

Controlled by:

```bash
--print_every N
```

Example:

```bash
python3 main.py --print_every 100
```

Prints every 100 frames.

---

## Optional Arguments

| Argument           | Purpose                   |
| ------------------ | ------------------------- |
| `--input_svo_file` | Replay SVO                |
| `--ip_address`     | Network stream            |
| `--resolution`     | Camera resolution         |
| `--print_every`    | Frame interval            |
| `--no_view`        | Disable GUI windows       |
| `--allow_scale`    | Enable similarity scaling |
| `--anchors`        | Custom anchor joints      |

---

# Output Format Explained

Example row:

```
kp  name         | live_x live_y live_z | std_x std_y std_z | stdA_x stdA_y stdA_z | dx dy dz | L2(m)
```

Columns:

* live_* → real-time ZED detection
* std_* → raw static standard model (constant)
* stdA_* → aligned standard (changes per frame)
* dx,dy,dz → live − aligned
* L2(m) → 3D Euclidean error

---

# Interpretation of L2(m)

| Error (meters) | Meaning               |
| -------------- | --------------------- |
| < 0.05         | Very close match      |
| 0.05–0.15      | Small deviation       |
| 0.15–0.30      | Noticeable difference |
| > 0.30         | Large pose difference |

Since global motion is removed, this reflects **true pose variation**, not body position in space.

---

# Important Conceptual Clarification

Why does aligned standard change?

Because each frame computes a new optimal rotation and translation:

* If the person turns, R changes.
* If the person moves, t changes.
* If scaling enabled and body proportions differ, s may change.

The underlying standard model remains fixed.

---

# Mathematical Summary

Alignment problem:

[
\min_{R,t,s} \sum_i | live_i - (s R standard_i + t) |^2
]

This is solved using:

* Centroid removal
* Covariance matrix
* SVD decomposition
* Reflection correction

---

# What This System Measures

It measures:

* How different the live pose is from the canonical pose
* Which joints deviate most
* Overall articulation similarity

It does NOT measure:

* Absolute position in space
* Walking distance
* Camera-relative motion

---

# Possible Extensions

* Joint angle error instead of positional error
* Bone length normalization
* Procrustes alignment with fixed root
* CSV logging
* Temporal smoothing
* Per-limb error aggregation
* Normalized error (divide by shoulder width)

---

# Requirements

* ZED SDK
* Python 3
* numpy
* OpenCV
* pyzed

---

# Typical Usage

Live camera:

```bash
python3 main.py
```

Replay SVO:

```bash
python3 main.py --input_svo_file file.svo2
```

Disable viewer:

```bash
python3 main.py --no_view
```

Increase print interval:

```bash
python3 main.py --print_every 120
```

---

# Final Summary

This system:

* Uses ZED BODY_18 real-time 3D keypoints
* Aligns a static reference skeleton
* Computes per-joint Euclidean deviation
* Outputs interpretable pose error metrics

It provides a robust geometric framework for comparing human poses in 3D space.

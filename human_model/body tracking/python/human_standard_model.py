import numpy as np

# ZED BODY_18 index mapping (must match demo_body_tracking order)
KP = {
    "NOSE": 0,
    "NECK": 1,
    "R_SHOULDER": 2,
    "R_ELBOW": 3,
    "R_WRIST": 4,
    "L_SHOULDER": 5,
    "L_ELBOW": 6,
    "L_WRIST": 7,
    "R_HIP": 8,
    "R_KNEE": 9,
    "R_ANKLE": 10,
    "L_HIP": 11,
    "L_KNEE": 12,
    "L_ANKLE": 13,
    "R_EYE": 14,
    "L_EYE": 15,
    "R_EAR": 16,
    "L_EAR": 17,
}

KP18_NAMES = [
    "nose","neck",
    "r_shoulder","r_elbow","r_wrist",
    "l_shoulder","l_elbow","l_wrist",
    "r_hip","r_knee","r_ankle",
    "l_hip","l_knee","l_ankle",
    "r_eye","l_eye","r_ear","l_ear"
]

def get_standard_kp3d_standing() -> np.ndarray:
    """
    Returns a static standard human BODY_18 model in meters, shape (18,3).
    Coordinate convention here is arbitrary; we will ALIGN it to live ZED each frame.
    """
    kp3d = np.zeros((18, 3), dtype=float)

    # Head / face
    kp3d[KP["NOSE"]]  = [0.0, 1.65, 0.0]
    kp3d[KP["R_EYE"]] = [0.08, 1.68, 0.05]
    kp3d[KP["L_EYE"]] = [-0.08, 1.68, 0.05]
    kp3d[KP["R_EAR"]] = [0.12, 1.65, 0.0]
    kp3d[KP["L_EAR"]] = [-0.12, 1.65, 0.0]

    # Neck
    kp3d[KP["NECK"]] = [0.0, 1.50, 0.0]

    # Shoulders
    kp3d[KP["R_SHOULDER"]] = [0.20, 1.50, 0.0]
    kp3d[KP["L_SHOULDER"]] = [-0.20, 1.50, 0.0]

    # Arms
    kp3d[KP["R_ELBOW"]] = [0.35, 1.20, 0.0]
    kp3d[KP["R_WRIST"]] = [0.50, 0.95, 0.0]
    kp3d[KP["L_ELBOW"]] = [-0.35, 1.20, 0.0]
    kp3d[KP["L_WRIST"]] = [-0.50, 0.95, 0.0]

    # Hips
    kp3d[KP["R_HIP"]] = [0.15, 1.00, 0.0]
    kp3d[KP["L_HIP"]] = [-0.15, 1.00, 0.0]

    # Legs
    kp3d[KP["R_KNEE"]] = [0.15, 0.60, 0.0]
    kp3d[KP["R_ANKLE"]] = [0.15, 0.10, 0.0]
    kp3d[KP["L_KNEE"]] = [-0.15, 0.60, 0.0]
    kp3d[KP["L_ANKLE"]] = [-0.15, 0.10, 0.0]

    return kp3d
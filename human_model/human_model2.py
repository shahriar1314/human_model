import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# -------------------- Math helpers --------------------

def T_translate(xyz: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, 3] = xyz.astype(float)
    return T

def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n

def make_T_from_Rp(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

def R_from_axes(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Columns are the basis vectors in world coordinates."""
    R = np.stack([x, y, z], axis=1).astype(float)
    # Orthonormalize (lightweight Gram-Schmidt)
    z = normalize(R[:, 2])
    x = R[:, 0] - np.dot(R[:, 0], z) * z
    x = normalize(x)
    y = np.cross(z, x)
    y = normalize(y)
    return np.stack([x, y, z], axis=1)

# -------------------- ZED BODY_18 constants --------------------

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

# Bone graph used for kinematics/tree construction
# Replace ONLY the connectivity lists in your code with the ones below.
# Everything else can stay the same.

# --- Bone graph used for kinematics/tree construction (parent -> child) ---
# Changed:
#   ("NECK","R_HIP") -> ("R_SHOULDER","R_HIP")
#   ("NECK","L_HIP") -> ("L_SHOULDER","L_HIP")
BONES = [
    ("NECK", "NOSE"),
    ("NOSE", "R_EYE"), ("R_EYE", "R_EAR"),
    ("NOSE", "L_EYE"), ("L_EYE", "L_EAR"),

    ("NECK", "R_SHOULDER"), ("R_SHOULDER", "R_ELBOW"), ("R_ELBOW", "R_WRIST"),
    ("NECK", "L_SHOULDER"), ("L_SHOULDER", "L_ELBOW"), ("L_ELBOW", "L_WRIST"),

    # hips now connect to their respective shoulders
    ("R_SHOULDER", "R_HIP"), ("R_HIP", "R_KNEE"), ("R_KNEE", "R_ANKLE"),
    ("L_SHOULDER", "L_HIP"), ("L_HIP", "L_KNEE"), ("L_KNEE", "L_ANKLE"),
]

# --- 2D visualization connectivity (matches your new preference) ---
# Changed:
#   (1,8) -> (2,8)
#   (1,11) -> (5,11)
BODY18_EDGES_2D = [
    (1, 0),
    (0, 14), (14, 16),
    (0, 15), (15, 17),

    (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),

    # hips now connect to shoulders
    (2, 8), (8, 9), (8, 11), (9, 10), 
    (5, 11), (11, 12), (12, 13),
]



FACE_IDXS = {0, 14, 15, 16, 17}
HAND_FOOT_IDXS = {4, 7, 10, 13}

# -------------------- Node + Skeleton --------------------

@dataclass
class Node:
    name: str
    parent: Optional[str]
    children: List[str] = field(default_factory=list)

class Skeleton:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.root_name: Optional[str] = None

    def add_node(self, name: str, parent: Optional[str]):
        if name in self.nodes:
            return
        self.nodes[name] = Node(name=name, parent=parent)
        if parent is None:
            self.root_name = name
        else:
            if parent not in self.nodes:
                self.nodes[parent] = Node(name=parent, parent=None)
            self.nodes[parent].children.append(name)

    @staticmethod
    def build_body18_tree(use_virtual_root: bool = True) -> "Skeleton":
        sk = Skeleton()
        if use_virtual_root:
            sk.add_node("ROOT", None)
            sk.add_node("NECK", "ROOT")
        else:
            sk.add_node("NECK", None)

        for p, c in BONES:
            if p == "NECK" and use_virtual_root:
                sk.add_node(c, "NECK")
            else:
                sk.add_node(p, sk.nodes.get(p, Node(p, None)).parent)  # ensure exists
                sk.add_node(c, p)
        return sk

# -------------------- Frame kinematics from keypoints --------------------

def compute_virtual_root_pos(kp3d: np.ndarray) -> np.ndarray:
    """Mid-hip as a stable 'pelvis-like' root."""
    lh = kp3d[KP["L_HIP"]]
    rh = kp3d[KP["R_HIP"]]
    if np.any(np.isnan(lh)) or np.any(np.isnan(rh)):
        return kp3d[KP["NECK"]]  # fallback
    return 0.5 * (lh + rh)

def joint_world_position(name: str, kp3d: np.ndarray, use_virtual_root: bool) -> np.ndarray:
    if name == "ROOT":
        return compute_virtual_root_pos(kp3d)
    return kp3d[KP[name]]

def estimate_joint_orientation(
    name: str,
    kp3d: np.ndarray,
    use_virtual_root: bool,
    world_up: np.ndarray = np.array([0.0, 1.0, 0.0]),
) -> np.ndarray:
    """
    Heuristic joint orientation from bone directions (BODY_18 provides points only).
    """
    def P(j): return joint_world_position(j, kp3d, use_virtual_root)
    p = P(name)

    z = None
    child_pref = {
        "R_SHOULDER": "R_ELBOW", "R_ELBOW": "R_WRIST",
        "L_SHOULDER": "L_ELBOW", "L_ELBOW": "L_WRIST",
        "R_HIP": "R_KNEE", "R_KNEE": "R_ANKLE",
        "L_HIP": "L_KNEE", "L_KNEE": "L_ANKLE",
        "NECK": "NOSE",
        "NOSE": "R_EYE",
    }
    if name in child_pref:
        z = P(child_pref[name]) - p

    if z is None or np.linalg.norm(z) < 1e-6:
        if name == "NECK" and use_virtual_root:
            z = p - P("ROOT")
        else:
            z = (p - P("NECK")) if name != "NECK" else np.array([0.0, 0.0, 1.0])

    z = normalize(z)
    if np.linalg.norm(z) < 1e-6:
        z = np.array([0.0, 0.0, 1.0], dtype=float)

    x = None
    if name in ("NECK", "NOSE", "ROOT"):
        rs = kp3d[KP["R_SHOULDER"]]
        ls = kp3d[KP["L_SHOULDER"]]
        if not (np.any(np.isnan(rs)) or np.any(np.isnan(ls))):
            x = rs - ls
    if x is None and name in ("ROOT", "NECK"):
        rh = kp3d[KP["R_HIP"]]
        lh = kp3d[KP["L_HIP"]]
        if not (np.any(np.isnan(rh)) or np.any(np.isnan(lh))):
            x = rh - lh

    if x is None or np.linalg.norm(x) < 1e-6:
        x = np.cross(world_up, z)
        if np.linalg.norm(x) < 1e-6:
            x = np.array([1.0, 0.0, 0.0])

    x = normalize(x)
    y = normalize(np.cross(z, x))
    x = normalize(np.cross(y, z))
    return R_from_axes(x, y, z)

def forward_kinematics_body18(
    kp3d: np.ndarray,
    use_virtual_root: bool = True,
    base_T: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[Tuple[str, str], np.ndarray]]:
    if kp3d.shape != (18, 3):
        raise ValueError("kp3d must be shape (18,3) for BODY_18")

    if base_T is None:
        base_T = np.eye(4, dtype=float)

    sk = Skeleton.build_body18_tree(use_virtual_root=use_virtual_root)
    globals_T: Dict[str, np.ndarray] = {}

    for name in sk.nodes.keys():
        p = joint_world_position(name, kp3d, use_virtual_root)
        R = estimate_joint_orientation(name, kp3d, use_virtual_root)
        globals_T[name] = base_T @ make_T_from_Rp(R, p)

    rel_T: Dict[Tuple[str, str], np.ndarray] = {}
    for child_name, node in sk.nodes.items():
        if node.parent is None:
            continue
        Tp = globals_T[node.parent]
        Tc = globals_T[child_name]
        rel_T[(node.parent, child_name)] = np.linalg.inv(Tp) @ Tc

    return globals_T, rel_T

# -------------------- 2D visualization (ZED-like) --------------------

def _is_valid_2d(pt: np.ndarray) -> bool:
    return (
        pt is not None
        and len(pt) >= 2
        and np.isfinite(pt[0]) and np.isfinite(pt[1])
        and pt[0] >= 0 and pt[1] >= 0
    )

def render_body18_2d(
    image_bgr: np.ndarray,
    keypoints_2d: np.ndarray,
    conf: Optional[np.ndarray] = None,
    conf_thres: float = 0.0,
    thickness: int = 4,
    joint_radius: int = 6,
) -> np.ndarray:
    """
    Draw BODY_18 skeleton on a BGR image using OpenCV.
    keypoints_2d: (18,2) in pixels (invalid can be NaN or negative).
    conf: optional (18,) confidence values.
    """
    if keypoints_2d.shape[0] != 18 or keypoints_2d.shape[1] != 2:
        raise ValueError("keypoints_2d must be shape (18,2)")

    kpts = keypoints_2d.astype(float)

    # ZED-like look: thick blue bones with darker outline + pale joints
    bone_color = (255, 160, 60)  # BGR (light-blue/cyan-ish)
    bone_shadow = (120, 70, 20)  # darker outline
    joint_fill = (235, 235, 235) # near-white
    joint_edge = (120, 120, 120) # gray

    def ok(i: int) -> bool:
        if not _is_valid_2d(kpts[i]):
            return False
        if conf is None:
            return True
        return float(conf[i]) >= conf_thres

    # Bones
    for a, b in BODY18_EDGES_2D:
        if not (ok(a) and ok(b)):
            continue
        p0 = tuple(np.round(kpts[a]).astype(int))
        p1 = tuple(np.round(kpts[b]).astype(int))

        cv2.line(image_bgr, p0, p1, bone_shadow, thickness + 2, cv2.LINE_AA)
        cv2.line(image_bgr, p0, p1, bone_color, thickness, cv2.LINE_AA)

    # Joints
    for i in range(18):
        if not ok(i):
            continue
        p = tuple(np.round(kpts[i]).astype(int))

        r = joint_radius
        if i in FACE_IDXS:
            r = max(3, joint_radius - 2)
        elif i in HAND_FOOT_IDXS:
            r = max(4, joint_radius - 1)

        cv2.circle(image_bgr, p, r + 2, joint_edge, -1, cv2.LINE_AA)
        cv2.circle(image_bgr, p, r, joint_fill, -1, cv2.LINE_AA)

    return image_bgr

# -------------------- Example usage (with visualization) --------------------

if __name__ == "__main__":
    # ---- 3D keypoints for standing human model (meters) ----
    # Standard proportions: ~1.7m tall human in T-pose standing position
    # X-axis: left-right (positive = right), Y-axis: vertical (up), Z-axis: depth (forward)
    
    kp3d = np.zeros((18, 3), dtype=float)
    
    # Head and Face (Y: 1.65-1.70m)
    kp3d[KP["NOSE"]] = [0.0, 1.65, 0.0]      # Front of face
    kp3d[KP["R_EYE"]] = [0.08, 1.68, 0.05]   # Right eye
    kp3d[KP["L_EYE"]] = [-0.08, 1.68, 0.05]  # Left eye
    kp3d[KP["R_EAR"]] = [0.12, 1.65, 0.0]    # Right ear
    kp3d[KP["L_EAR"]] = [-0.12, 1.65, 0.0]   # Left ear
    
    # Neck (Y: 1.50m)
    kp3d[KP["NECK"]] = [0.0, 1.50, 0.0]
    
    # Shoulders (Y: 1.50m, width: ~0.40m total)
    kp3d[KP["R_SHOULDER"]] = [0.20, 1.50, 0.0]
    kp3d[KP["L_SHOULDER"]] = [-0.20, 1.50, 0.0]
    
    # Right Arm (upper arm ~0.30m, forearm ~0.25m)
    kp3d[KP["R_ELBOW"]] = [0.35, 1.20, 0.0]   # Elbow at ~1.2m height
    kp3d[KP["R_WRIST"]] = [0.50, 0.95, 0.0]   # Hand at ~1.0m height
    
    # Left Arm (mirror of right)
    kp3d[KP["L_ELBOW"]] = [-0.35, 1.20, 0.0]
    kp3d[KP["L_WRIST"]] = [-0.50, 0.95, 0.0]
    
    # Hips (Y: 1.00m, width: ~0.30m)
    kp3d[KP["R_HIP"]] = [0.15, 1.00, 0.0]
    kp3d[KP["L_HIP"]] = [-0.15, 1.00, 0.0]
    
    # Right Leg (thigh ~0.40m, calf ~0.40m)
    kp3d[KP["R_KNEE"]] = [0.15, 0.60, 0.0]   # Knee at ~0.6m height
    kp3d[KP["R_ANKLE"]] = [0.15, 0.10, 0.0]  # Foot at ~0.1m height
    
    # Left Leg (mirror of right)
    kp3d[KP["L_KNEE"]] = [-0.15, 0.60, 0.0]
    kp3d[KP["L_ANKLE"]] = [-0.15, 0.10, 0.0]
    
    # ---- Print all 18 keypoint coordinates ----
    print("\n" + "="*70)
    print("BODY_18 KEYPOINT 3D COORDINATES (Standing Position)")
    print("="*70)
    print("Format: [X (left-right), Y (height), Z (depth)] in meters\n")
    
    for name, idx in sorted(KP.items(), key=lambda x: x[1]):
        x, y, z = kp3d[idx]
        print(f"{idx:2d}. {name:15s}: [{x:7.3f}, {y:7.3f}, {z:7.3f}]")
    
    print("\n" + "="*70)
    print("FORWARD KINEMATICS (World Transforms)")
    print("="*70 + "\n")

    globals_T, rel_T = forward_kinematics_body18(kp3d, use_virtual_root=True)

    for k in ["ROOT", "NECK", "R_WRIST", "L_ANKLE"]:
        print(f"{k} T_world_joint:\n{globals_T[k]}\n")

    for edge in [("NECK", "R_SHOULDER"), ("R_SHOULDER", "R_ELBOW"), ("R_HIP", "R_KNEE")]:
        print(f"{edge} T_parent_child:\n{rel_T[edge]}\n")

    # ---- 2D visualization demo ----
    # Create a blank image and project the toy 3D points to 2D for demonstration only.
    # In real usage, you already have keypoints_2d from ZED (body.keypoint_2d).
    W, H = 900, 700
    img = np.full((H, W, 3), 245, dtype=np.uint8)

    # Simple fake projection: x->u, y->v (just for demo)
    # Adjust scale/offset to fit the image nicely
    scale = 250.0
    cx, cy = W // 2, int(H * 0.75)
    keypoints_2d = np.zeros((18, 2), dtype=float)
    for name, idx in KP.items():
        X, Y, Z = kp3d[idx]
        u = cx + X * scale
        v = cy - Y * scale
        keypoints_2d[idx] = [u, v]

    render_body18_2d(img, keypoints_2d, conf=None, conf_thres=0.0, thickness=5, joint_radius=7)

    cv2.imshow("BODY_18 (ZED-like 2D)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
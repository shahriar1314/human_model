"""
HumanModel class - Manages human body kinematics and 2D visualization
Based on BODY_18 format from ZED camera
"""
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# -------------------- Math helpers --------------------

def _T_translate(xyz: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, 3] = xyz.astype(float)
    return T

def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n

def _make_T_from_Rp(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

def _R_from_axes(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Columns are the basis vectors in world coordinates."""
    R = np.stack([x, y, z], axis=1).astype(float)
    z = _normalize(R[:, 2])
    x = R[:, 0] - np.dot(R[:, 0], z) * z
    x = _normalize(x)
    y = np.cross(z, x)
    y = _normalize(y)
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

BONES = [
    ("NECK", "NOSE"),
    ("NOSE", "R_EYE"), ("R_EYE", "R_EAR"),
    ("NOSE", "L_EYE"), ("L_EYE", "L_EAR"),
    ("NECK", "R_SHOULDER"), ("R_SHOULDER", "R_ELBOW"), ("R_ELBOW", "R_WRIST"),
    ("NECK", "L_SHOULDER"), ("L_SHOULDER", "L_ELBOW"), ("L_ELBOW", "L_WRIST"),
    ("R_SHOULDER", "R_HIP"), ("R_HIP", "R_KNEE"), ("R_KNEE", "R_ANKLE"),
    ("L_SHOULDER", "L_HIP"), ("L_HIP", "L_KNEE"), ("L_KNEE", "L_ANKLE"),
]

# Standard bone lengths (meters) for a typical adult human (175 cm tall)
STANDARD_BONE_LENGTHS = {
    ("NECK", "NOSE"): 0.10,
    ("NOSE", "R_EYE"): 0.04,
    ("NOSE", "L_EYE"): 0.04,
    ("R_EYE", "R_EAR"): 0.08,
    ("L_EYE", "L_EAR"): 0.08,
    ("NECK", "R_SHOULDER"): 0.15,
    ("NECK", "L_SHOULDER"): 0.15,
    ("R_SHOULDER", "R_ELBOW"): 0.30,
    ("L_SHOULDER", "L_ELBOW"): 0.30,
    ("R_ELBOW", "R_WRIST"): 0.25,
    ("L_ELBOW", "L_WRIST"): 0.25,
    ("R_SHOULDER", "R_HIP"): 0.30,
    ("L_SHOULDER", "L_HIP"): 0.30,
    ("R_HIP", "R_KNEE"): 0.40,
    ("L_HIP", "L_KNEE"): 0.40,
    ("R_KNEE", "R_ANKLE"): 0.40,
    ("L_KNEE", "L_ANKLE"): 0.40,
}

BODY18_EDGES_2D = [
    (1, 0),
    (0, 14), (14, 16),
    (0, 15), (15, 17),
    (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
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
                sk.add_node(p, sk.nodes.get(p, Node(p, None)).parent)
                sk.add_node(c, p)
        return sk


class HumanModel:
    """
    Manages human body kinematics and 2D visualization for BODY_18 format.
    Provides forward kinematics computation and rendering utilities.
    """
    
    def __init__(self, use_virtual_root: bool = True, print_interval: int = 1):
        """
        Initialize HumanModel with fixed zero angles (identity rotations).
        
        Args:
            use_virtual_root: Whether to use a virtual root node (mid-hip position)
            print_interval: Print keypoint comparison every N updates (0 to disable)
        """
        self.use_virtual_root = use_virtual_root
        self.print_interval = print_interval
        self.update_count = 0
        self.skeleton = Skeleton.build_body18_tree(use_virtual_root=use_virtual_root)
        self.base_position: np.ndarray = np.array([0.0, 1.5, 0.0], dtype=np.float32)  # Default root position
        
        # Initialize all joint angles to identity (zero rotation)
        self.joint_angles: Dict[Tuple[str, str], np.ndarray] = {}
        
        # Bone lengths: initialized with standard anatomical values
        self.bone_lengths: Dict[Tuple[str, str], float] = {}
        
        self._initialize_zero_angles()
        
        # These will be used for comparison only, not for updating angles
        self.globals_T: Dict[str, np.ndarray] = {}
        self.rel_T: Dict[Tuple[str, str], np.ndarray] = {}
        self.keypoints_3d: Optional[np.ndarray] = None  # Original camera keypoints
    
    def _initialize_zero_angles(self):
        """Initialize all joint angles to identity rotations (zero angles) and standard bone lengths."""
        self.joint_angles = {}
        self.rel_T = {}
        self.bone_lengths = {}
        
        for child_name, node in self.skeleton.nodes.items():
            if node.parent is None:
                continue
            # Identity rotation (zero angle)
            self.joint_angles[(node.parent, child_name)] = np.eye(3, dtype=np.float32)
            # Initialize bone length with standard anatomical values
            self.bone_lengths[(node.parent, child_name)] = STANDARD_BONE_LENGTHS.get(
                (node.parent, child_name), 0.0
            )
            # Also initialize rel_T with identity transforms
            self.rel_T[(node.parent, child_name)] = np.eye(4, dtype=np.float32)
    
    
    @staticmethod
    def _compute_virtual_root_pos(kp3d: np.ndarray) -> np.ndarray:
        """Mid-hip as a stable 'pelvis-like' root."""
        lh = kp3d[KP["L_HIP"]]
        rh = kp3d[KP["R_HIP"]]
        if np.any(np.isnan(lh)) or np.any(np.isnan(rh)):
            return kp3d[KP["NECK"]]
        return 0.5 * (lh + rh)
    
    def _joint_world_position(self, name: str, kp3d: np.ndarray) -> np.ndarray:
        """Get joint world position."""
        if name == "ROOT":
            return self._compute_virtual_root_pos(kp3d)
        return kp3d[KP[name]]
    
    def _estimate_joint_orientation(
        self,
        name: str,
        kp3d: np.ndarray,
        world_up: np.ndarray = np.array([0.0, 1.0, 0.0]),
    ) -> np.ndarray:
        """Heuristic joint orientation from bone directions."""
        p = self._joint_world_position(name, kp3d)
        
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
            z = self._joint_world_position(child_pref[name], kp3d) - p
        
        if z is None or np.linalg.norm(z) < 1e-6:
            if name == "NECK" and self.use_virtual_root:
                z = p - self._joint_world_position("ROOT", kp3d)
            else:
                z = (p - self._joint_world_position("NECK", kp3d)) if name != "NECK" else np.array([0.0, 0.0, 1.0])
        
        z = _normalize(z)
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
        
        x = _normalize(x)
        y = _normalize(np.cross(z, x))
        x = _normalize(np.cross(y, z))
        return _R_from_axes(x, y, z)
    
    def update(self, kp3d: np.ndarray) -> None:
        """
        Update the model with camera keypoints for comparison.
        The model's joint angles remain FIXED and do not change.
        Joint positions are computed from the fixed angles using forward kinematics.
        
        Args:
            kp3d: 3D keypoints array of shape (18, 3) from the camera
        """
        if kp3d.shape != (18, 3):
            raise ValueError("kp3d must be shape (18,3) for BODY_18")
        
        # Store camera keypoints for comparison purposes only
        self.keypoints_3d = kp3d.copy()
        self.update_count += 1
        
        # NOTE: self.joint_angles REMAINS FIXED and is NOT updated
        # The model's joint positions are computed from these fixed angles
        # Print comparison of camera keypoints vs model-computed joint positions
        if self.print_interval > 0 and self.update_count % self.print_interval == 0:
            self._print_keypoints_comparison()
    
    def get_global_transform(self, joint_name: str) -> Optional[np.ndarray]:
        """Get the global transform for a joint."""
        return self.globals_T.get(joint_name)
    
    def get_relative_transform(self, parent: str, child: str) -> Optional[np.ndarray]:
        """Get the relative transform between parent and child."""
        return self.rel_T.get((parent, child))
    
    def set_joint_angle(self, parent: str, child: str, rotation_matrix: np.ndarray) -> None:
        """
        Set the rotation angle for a specific joint.
        
        Args:
            parent: Parent joint name
            child: Child joint name
            rotation_matrix: 3x3 rotation matrix to set for this joint
        """
        if rotation_matrix.shape != (3, 3):
            raise ValueError("rotation_matrix must be shape (3, 3)")
        self.joint_angles[(parent, child)] = rotation_matrix.astype(np.float32).copy()
    
    def set_base_position(self, position: np.ndarray) -> None:
        """
        Set the base (root) position of the model.
        
        Args:
            position: 3D position vector for the root joint
        """
        if position.shape != (3,):
            raise ValueError("position must be shape (3,)")
        self.base_position = position.astype(np.float32).copy()
    
    def reset_joint_angles_to_zero(self) -> None:
        """Reset all joint angles to identity (zero angles)."""
        self._initialize_zero_angles()
    
    def render_2d(
        self,
        image_bgr: np.ndarray,
        keypoints_2d: np.ndarray,
        conf: Optional[np.ndarray] = None,
        conf_thres: float = 0.0,
        thickness: int = 4,
        joint_radius: int = 6,
    ) -> np.ndarray:
        """
        Draw BODY_18 skeleton on a BGR image.
        
        Args:
            image_bgr: Input BGR image
            keypoints_2d: 2D keypoints array of shape (18, 2)
            conf: Optional confidence values
            conf_thres: Confidence threshold
            thickness: Bone line thickness
            joint_radius: Joint circle radius
        
        Returns:
            Modified image with skeleton drawn
        """
        if keypoints_2d.shape[0] != 18 or keypoints_2d.shape[1] != 2:
            raise ValueError("keypoints_2d must be shape (18,2)")
        
        kpts = keypoints_2d.astype(float)
        
        # ZED-like colors
        bone_color = (255, 160, 60)
        bone_shadow = (120, 70, 20)
        joint_fill = (235, 235, 235)
        joint_edge = (120, 120, 120)
        
        def ok(i: int) -> bool:
            if not self._is_valid_2d(kpts[i]):
                return False
            if conf is None:
                return True
            return float(conf[i]) >= conf_thres
        
        # Draw bones
        for a, b in BODY18_EDGES_2D:
            if not (ok(a) and ok(b)):
                continue
            p0 = tuple(np.round(kpts[a]).astype(int))
            p1 = tuple(np.round(kpts[b]).astype(int))
            
            cv2.line(image_bgr, p0, p1, bone_shadow, thickness + 2, cv2.LINE_AA)
            cv2.line(image_bgr, p0, p1, bone_color, thickness, cv2.LINE_AA)
        
        # Draw joints
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
    
    @staticmethod
    def _is_valid_2d(pt: np.ndarray) -> bool:
        """Check if a 2D point is valid."""
        return (
            pt is not None
            and len(pt) >= 2
            and np.isfinite(pt[0]) and np.isfinite(pt[1])
            and pt[0] >= 0 and pt[1] >= 0
        )
    
    def print_keypoints(self, person_id: int = 0, confidence: float = 0.0):
        """Print current keypoint values."""
        if self.keypoints_3d is None:
            print("No keypoints have been set yet")
            return
        
        print(f"\nPerson ID: {person_id} | conf: {confidence}")
        for name, idx in KP.items():
            p = self.keypoints_3d[idx]
            if not np.any(np.isnan(p)):
                print(f"{name}: {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}")


    
    def get_joint_positions_3d(self) -> Dict[str, np.ndarray]:
        """
        Get the 3D positions of all joints computed from model's stored joint angles.
        Uses forward kinematics starting from base position and joint angles.
        
        Returns:
            Dictionary mapping joint names to 3D position vectors.
            Returns empty dict if initialization hasn't been completed yet.
        """
        if not self.joint_angles:
            return {}
        
        # Compute positions from stored angles using forward kinematics
        return self._compute_positions_from_angles()
    
    def _compute_positions_from_angles(self) -> Dict[str, np.ndarray]:
        """
        Compute 3D positions using forward kinematics from stored joint angles and bone lengths.
        Uses the model's fixed joint angles (initialized to identity) and base position.
        
        Returns:
            Dictionary mapping joint names to 3D position vectors.
        """
        if not self.joint_angles:
            return {}
        
        # Start with base (ROOT) position
        positions = {}
        positions["ROOT"] = self.base_position.copy()
        
        # Forward kinematics: traverse skeleton and accumulate transforms
        def compute_child_transforms(parent_name: str, parent_T: np.ndarray):
            for child_name, node in self.skeleton.nodes.items():
                if node.parent != parent_name:
                    continue
                
                # Get rotation angle for this joint (fixed)
                angle_key = (parent_name, child_name)
                if angle_key in self.joint_angles:
                    rotation = self.joint_angles[angle_key]
                    
                    # Get bone length for this parent-child relationship
                    bone_length = self.bone_lengths.get(angle_key, 0.0)
                    
                    # Create relative transform: translation along Z-axis (default bone direction)
                    # then apply rotation. The child is offset from parent by bone_length along Z.
                    rel_T = np.eye(4, dtype=np.float32)
                    rel_T[:3, :3] = rotation
                    rel_T[2, 3] = bone_length  # Translation along Z-axis in local frame
                    
                    child_T = parent_T @ rel_T
                    positions[child_name] = child_T[:3, 3].copy()
                    compute_child_transforms(child_name, child_T)
        
        # Start recursion from root
        root_T = np.eye(4, dtype=float)
        root_T[:3, 3] = self.base_position
        compute_child_transforms("ROOT", root_T)
        
        return positions
    
    def get_model_computed_positions(self) -> Dict[str, np.ndarray]:
        """
        Get 3D joint positions as computed by the model from its stored angles.
        This is different from camera keypoints - it's the model's reconstruction.
        
        Returns:
            Dictionary mapping joint names to 3D position vectors computed from angles.
        """
        return self._compute_positions_from_angles()
    
    def get_joint_positions_as_array(self) -> np.ndarray:
        """
        Get the 3D positions of all joints as a structured array.
        
        Returns:
            Array of shape (num_joints, 3) with 3D positions.
            Order matches the iteration order of joint_positions dict.
            Returns None if update() hasn't been called yet.
        """
        positions = self.get_joint_positions_3d()
        if not positions:
            return None
        
        # Extract positions in order of joint names
        pos_list = [positions[name] for name in sorted(positions.keys())]
        return np.array(pos_list, dtype=np.float32)
    
    def _print_keypoints_comparison(self):
        """
        Print side-by-side comparison of camera input keypoints (kp3d) vs
        positions computed from model's stored joint angles, along with pairwise squared differences.
        """
        if self.keypoints_3d is None or not self.joint_angles:
            return
        
        # Get model-computed positions from stored angles
        model_positions = self._compute_positions_from_angles()
        if not model_positions:
            return
        
        print("\n" + "="*120)
        print("KEYPOINTS COMPARISON: Camera Input vs Model-Computed Positions (from Joint Angles)")
        print("="*120)
        
        # Header
        print(f"{'Joint Name':<15} {'Camera kp3d (x, y, z)':<40} {'Model Compute (x, y, z)':<40} {'Sq Diff':<10}")
        print("-"*120)
        
        total_sq_diff = 0.0
        valid_count = 0
        
        # Compare camera keypoints with model-computed positions for each joint
        for name, idx in KP.items():
            kp = self.keypoints_3d[idx]
            
            # Get model-computed position for this joint
            if name not in model_positions:
                continue
            
            model_pos = model_positions[name]
            
            # Check if keypoint is valid
            kp_valid = not np.any(np.isnan(kp))
            
            if kp_valid:
                # Compute squared difference
                sq_diff = np.sum((kp - model_pos) ** 2)
                total_sq_diff += sq_diff
                valid_count += 1
                
                # Format output
                kp_str = f"({kp[0]:7.3f}, {kp[1]:7.3f}, {kp[2]:7.3f})"
                model_str = f"({model_pos[0]:7.3f}, {model_pos[1]:7.3f}, {model_pos[2]:7.3f})"
                sq_diff_str = f"{sq_diff:7.4f}"
                
                print(f"{name:<15} {kp_str:<40} {model_str:<40} {sq_diff_str:<10}")
            else:
                # Invalid keypoint
                kp_str = "(NaN, NaN, NaN)"
                model_str = f"({model_pos[0]:7.3f}, {model_pos[1]:7.3f}, {model_pos[2]:7.3f})"
                print(f"{name:<15} {kp_str:<40} {model_str:<40} {'N/A':<10}")
        
        # Print summary statistics
        print("-"*120)
        if valid_count > 0:
            mean_sq_diff = total_sq_diff / valid_count
            print(f"Total Squared Difference: {total_sq_diff:.4f}")
            print(f"Mean Squared Difference ({valid_count} joints): {mean_sq_diff:.4f}")
            print(f"RMSE (Root Mean Sq Error): {np.sqrt(mean_sq_diff):.4f}")
        print("="*120 + "\n")


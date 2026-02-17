import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------- Math helpers (4x4 transforms) ----------

def T_translate(xyz: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = xyz
    return T

def R_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    R = np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    return T

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ---------- Joint + Node ----------

@dataclass
class Joint1DoF:
    axis: np.ndarray                   # rotation axis in local joint frame
    angle: float = 0.0                 # current angle
    limits: Tuple[float, float] = (-np.pi, np.pi)  # (min,max)
    max_vel: float = np.deg2rad(180.0) # rad/s (optional)

    def set_angle(self, a: float):
        self.angle = clamp(a, self.limits[0], self.limits[1])

    def sample(self, rng: np.random.Generator):
        a = rng.uniform(self.limits[0], self.limits[1])
        self.angle = a
        return a

    def local_T(self) -> np.ndarray:
        return R_axis_angle(self.axis, self.angle)

@dataclass
class Node:
    name: str
    parent: Optional[str]
    offset_from_parent: np.ndarray     # 3D translation from parent joint to this joint (parent frame)
    joint: Optional[Joint1DoF] = None  # None for fixed links
    children: List[str] = field(default_factory=list)

    # purely for visualization proxy (sphere radius or segment thickness)
    radius: float = 0.05

# ---------- Skeleton / Kinematic Tree ----------

class Skeleton:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.root_name: Optional[str] = None

    def add_node(self, node: Node):
        if node.name in self.nodes:
            raise ValueError(f"Duplicate node name: {node.name}")
        self.nodes[node.name] = node
        if node.parent is None:
            if self.root_name is not None:
                raise ValueError("Only one root allowed")
            self.root_name = node.name
        else:
            self.nodes[node.parent].children.append(node.name)

    def set_angles(self, angles: Dict[str, float]):
        for name, a in angles.items():
            n = self.nodes[name]
            if n.joint is None:
                continue
            n.joint.set_angle(a)

    def sample_pose(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        for n in self.nodes.values():
            if n.joint is not None:
                n.joint.sample(rng)

    def forward_kinematics(self, base_T: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Returns global transform of each node joint frame in world coordinates.
        base_T places the root in the world. If None, identity.
        """
        if self.root_name is None:
            raise ValueError("No root defined")
        if base_T is None:
            base_T = np.eye(4)

        globals_T: Dict[str, np.ndarray] = {}

        def dfs(name: str, parent_T: np.ndarray):
            node = self.nodes[name]
            # translation to this joint
            T = parent_T @ T_translate(node.offset_from_parent)
            # apply joint rotation if any
            if node.joint is not None:
                T = T @ node.joint.local_T()
            globals_T[name] = T
            for ch in node.children:
                dfs(ch, T)

        dfs(self.root_name, base_T)
        return globals_T

# ---------- Build a "rough human-like" skeleton ----------

def build_rough_humanoid() -> Skeleton:
    sk = Skeleton()

    # Root = torso/chest base (could later be attached to a mobile base)
    sk.add_node(Node(
        name="torso",
        parent=None,
        offset_from_parent=np.array([0.0, 0.0, 0.0]),
        joint=None,
        radius=0.10
    ))

    # Head (neck joint rotates)
    sk.add_node(Node(
        name="head",
        parent="torso",
        offset_from_parent=np.array([0.0, 0.0, 0.35]),
        joint=Joint1DoF(axis=np.array([0, 1, 0]), limits=(np.deg2rad(-60), np.deg2rad(60))),
        radius=0.12
    ))

    # Shoulders + arms (1DoF each for now; easy to expand later)
    sk.add_node(Node(
        name="l_upper_arm",
        parent="torso",
        offset_from_parent=np.array([0.18, 0.0, 0.28]),
        joint=Joint1DoF(axis=np.array([0, 1, 0]), limits=(np.deg2rad(-120), np.deg2rad(120))),
        radius=0.05
    ))
    sk.add_node(Node(
        name="l_forearm",
        parent="l_upper_arm",
        offset_from_parent=np.array([0.25, 0.0, 0.0]),
        joint=Joint1DoF(axis=np.array([0, 0, 1]), limits=(np.deg2rad(0), np.deg2rad(150))),
        radius=0.045
    ))
    sk.add_node(Node(
        name="l_hand",
        parent="l_forearm",
        offset_from_parent=np.array([0.22, 0.0, 0.0]),
        joint=None,
        radius=0.04
    ))

    sk.add_node(Node(
        name="r_upper_arm",
        parent="torso",
        offset_from_parent=np.array([-0.18, 0.0, 0.28]),
        joint=Joint1DoF(axis=np.array([0, 1, 0]), limits=(np.deg2rad(-120), np.deg2rad(120))),
        radius=0.05
    ))
    sk.add_node(Node(
        name="r_forearm",
        parent="r_upper_arm",
        offset_from_parent=np.array([-0.25, 0.0, 0.0]),
        joint=Joint1DoF(axis=np.array([0, 0, 1]), limits=(np.deg2rad(0), np.deg2rad(150))),
        radius=0.045
    ))
    sk.add_node(Node(
        name="r_hand",
        parent="r_forearm",
        offset_from_parent=np.array([-0.22, 0.0, 0.0]),
        joint=None,
        radius=0.04
    ))

    # Hips + legs (still rough)
    sk.add_node(Node(
        name="l_thigh",
        parent="torso",
        offset_from_parent=np.array([0.10, 0.0, -0.25]),
        joint=Joint1DoF(axis=np.array([1, 0, 0]), limits=(np.deg2rad(-90), np.deg2rad(90))),
        radius=0.06
    ))
    sk.add_node(Node(
        name="l_shin",
        parent="l_thigh",
        offset_from_parent=np.array([0.0, 0.0, -0.35]),
        joint=Joint1DoF(axis=np.array([1, 0, 0]), limits=(np.deg2rad(0), np.deg2rad(150))),
        radius=0.055
    ))
    sk.add_node(Node(
        name="l_foot",
        parent="l_shin",
        offset_from_parent=np.array([0.0, 0.10, -0.30]),
        joint=None,
        radius=0.05
    ))

    sk.add_node(Node(
        name="r_thigh",
        parent="torso",
        offset_from_parent=np.array([-0.10, 0.0, -0.25]),
        joint=Joint1DoF(axis=np.array([1, 0, 0]), limits=(np.deg2rad(-90), np.deg2rad(90))),
        radius=0.06
    ))
    sk.add_node(Node(
        name="r_shin",
        parent="r_thigh",
        offset_from_parent=np.array([0.0, 0.0, -0.35]),
        joint=Joint1DoF(axis=np.array([1, 0, 0]), limits=(np.deg2rad(0), np.deg2rad(150))),
        radius=0.055
    ))
    sk.add_node(Node(
        name="r_foot",
        parent="r_shin",
        offset_from_parent=np.array([0.0, 0.10, -0.30]),
        joint=None,
        radius=0.05
    ))

    return sk

# ---------- Simple Open3D visualization ----------

def visualize_open3d(sk: Skeleton, globals_T: Dict[str, np.ndarray]):
    import open3d as o3d

    # Build joint positions
    names = list(sk.nodes.keys())
    pos = {n: globals_T[n][:3, 3] for n in names}

    # Lines for parent-child edges
    points = []
    idx = {}
    for i, n in enumerate(names):
        idx[n] = i
        points.append(pos[n])
    points = np.array(points)

    lines = []
    for n in names:
        parent = sk.nodes[n].parent
        if parent is not None:
            lines.append([idx[parent], idx[n]])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    # Spheres for joints (rough body)
    geoms = [line_set]
    for n in names:
        r = sk.nodes[n].radius
        s = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        s.compute_vertex_normals()
        s.translate(pos[n])
        geoms.append(s)

    o3d.visualization.draw_geometries(geoms)

# ---------- Demo ----------

if __name__ == "__main__":
    sk = build_rough_humanoid()

    # Example: use a standard standing pose (zeroed joint angles)
    standing_angles = {}
    for name, node in sk.nodes.items():
        if node.joint is not None:
            standing_angles[name] = 0.0
    sk.set_angles(standing_angles)

    # Place torso in world (example: 1m above ground, rotated)
    base_T = T_translate(np.array([0.0, 0.0, 1.0])) @ R_axis_angle(np.array([0, 0, 1]), np.deg2rad(10))

    globals_T = sk.forward_kinematics(base_T=base_T)

    # Print a couple transforms
    for k in ["torso", "head", "l_hand", "r_foot"]:
        print(k, "\n", globals_T[k], "\n")

    # Visualize (don't crash if open3d is unavailable)
    try:
        visualize_open3d(sk, globals_T)
    except Exception as e:
        print("Visualization skipped:", e)

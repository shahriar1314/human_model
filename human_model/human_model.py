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

    geoms = []

    # Create cylinder 'bones' between each parent-child pair.
    # Cylinder height = distance between joints, radius proportional to node radii.
    for n in names:
        parent = sk.nodes[n].parent
        if parent is None:
            continue
        p0 = pos[parent]
        p1 = pos[n]
        v = p1 - p0
        h = np.linalg.norm(v)
        if h < 1e-6:
            continue

        # choose cylinder radius proportional to the connected node radii
        r0 = sk.nodes[parent].radius
        r1 = sk.nodes[n].radius
        cyl_r = max(0.01, 0.6 * (r0 + r1) * 0.5)

        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=cyl_r, height=h, resolution=20, split=4)
        cyl.compute_vertex_normals()

        # align cylinder z-axis with vector v
        v_norm = v / h
        z = np.array([0.0, 0.0, 1.0])
        # handle special cases for alignment
        if np.allclose(v_norm, z):
            R = np.eye(4)
        elif np.allclose(v_norm, -z):
            R = R_axis_angle(np.array([1.0, 0.0, 0.0]), np.pi)
        else:
            axis = np.cross(z, v_norm)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            angle = np.arccos(np.clip(np.dot(z, v_norm), -1.0, 1.0))
            R = R_axis_angle(axis, angle)

        mid = (p0 + p1) * 0.5
        T = T_translate(mid) @ R
        cyl.transform(T)
        cyl.paint_uniform_color([0.7, 0.7, 0.7])
        geoms.append(cyl)

    # Torso: render as a larger vertical cylinder (chest/torso), not a sphere
    if "torso" in pos:
        torso_pos = pos["torso"]
        # estimate top and bottom of torso using head and thighs (fall back to offsets)
        head_z = pos["head"][2] if "head" in pos else torso_pos[2] + 0.35
        thigh_names = ["l_thigh", "r_thigh"]
        thigh_zs = [pos[n][2] for n in thigh_names if n in pos]
        if thigh_zs:
            hip_z = min(thigh_zs)
        else:
            hip_z = torso_pos[2] - 0.25

        top_z = head_z - 0.05
        bottom_z = hip_z + 0.02
        torso_size_coeff = 0.6  # scale down torso size for better visual proportions
        torso_h = max(0.15, top_z - bottom_z)*torso_size_coeff
        torso_mid = np.array([torso_pos[0], torso_pos[1], (top_z + bottom_z) * 0.5])

        torso_r = max(0.08, sk.nodes["torso"].radius * 1.8)*torso_size_coeff
        torso_cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=torso_r, height=torso_h, resolution=40)
        torso_cyl.compute_vertex_normals()
        torso_cyl.transform(T_translate(torso_mid))
        torso_cyl.paint_uniform_color([0.6, 0.6, 0.6])
        geoms.append(torso_cyl)

    # Spheres for joints (keep head spherical and slightly highlighted)
    for n in names:
        # skip torso sphere since we render a torso cylinder
        if n == "torso":
            continue
        r = sk.nodes[n].radius
        s = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        s.compute_vertex_normals()
        s.translate(pos[n])
        if n == "head":
            s.paint_uniform_color([1.0, 0.85, 0.75])
        else:
            s.paint_uniform_color([0.9, 0.9, 0.9])
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

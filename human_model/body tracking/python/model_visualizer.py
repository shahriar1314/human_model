"""
Open3D Visualizer for the Human Model.
Provides visualization of the model skeleton and joint positions.
"""
import numpy as np
import open3d as o3d
from typing import Dict, Optional, List, Tuple
from human_model_class import HumanModel, KP


class ModelVisualizer:
    """
    Visualizes the human model using Open3D.
    Renders the skeleton structure with bones and joint positions.
    """
    
    def __init__(self, window_name: str = "Human Model Visualizer"):
        """
        Initialize the Open3D visualizer.
        
        Args:
            window_name: Title of the visualization window
        """
        self.window_name = window_name
        self.vis = None
        self.geometry_objects = {}
        self.joint_spheres: Dict[str, o3d.geometry.TriangleMesh] = {}
        self.bone_lines: Dict[Tuple[str, str], o3d.geometry.LineSet] = {}
        self.joint_positions: Optional[Dict[str, np.ndarray]] = None
        
        self._initialize_visualization()
    
    def _initialize_visualization(self) -> None:
        """Initialize the Open3D visualizer with default settings."""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=self.window_name, width=1200, height=800)
        
        # Set background color
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array([0.3, 0.3, 0.3])
        render_option.point_size = 5.0
        
        # Set up camera view
        ctr = self.vis.get_view_control()
        ctr.set_front(np.array([0, 0, 1]))
        ctr.set_lookat(np.array([0, 1.5, 0]))
        ctr.set_up(np.array([0, 1, 0]))
        ctr.set_zoom(0.3)
    
    def create_joint_geometry(self) -> None:
        """Create sphere geometries for all joints."""
        joint_names = list(KP.keys())
        
        for joint_name in joint_names:
            # Create a small sphere for each joint
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sphere.paint_uniform_color(np.array([0.0, 1.0, 0.5]))  # Cyan color
            self.joint_spheres[joint_name] = sphere
            self.vis.add_geometry(sphere)
    
    def create_bone_geometry(self) -> None:
        """Create line geometries for all bones connecting parent-child joints."""
        # Define skeleton bones (parent, child pairs)
        bones = [
            ("NECK", "NOSE"),
            ("NOSE", "R_EYE"), ("R_EYE", "R_EAR"),
            ("NOSE", "L_EYE"), ("L_EYE", "L_EAR"),
            ("NECK", "R_SHOULDER"), ("R_SHOULDER", "R_ELBOW"), ("R_ELBOW", "R_WRIST"),
            ("NECK", "L_SHOULDER"), ("L_SHOULDER", "L_ELBOW"), ("L_ELBOW", "L_WRIST"),
            ("R_SHOULDER", "R_HIP"), ("R_HIP", "R_KNEE"), ("R_KNEE", "R_ANKLE"),
            ("L_SHOULDER", "L_HIP"), ("L_HIP", "L_KNEE"), ("L_KNEE", "L_ANKLE"),
        ]
        
        for parent, child in bones:
            # Create line set for this bone
            line_set = o3d.geometry.LineSet()
            self.bone_lines[(parent, child)] = line_set
            self.vis.add_geometry(line_set)
    
    def update(self, human_model: HumanModel) -> None:
        """
        Update the visualization with current model state.
        
        Args:
            human_model: HumanModel instance with current joint positions
        """
        # Get current joint positions from the model
        self.joint_positions = human_model.get_model_computed_positions()
        
        if not self.joint_positions:
            return
        
        # Update joint sphere positions
        self._update_joints()
        
        # Update bone line positions
        self._update_bones()
        
        # Refresh the visualization
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def _update_joints(self) -> None:
        """Update the positions of all joint spheres."""
        for joint_name, position in self.joint_positions.items():
            if joint_name == "ROOT":
                continue  # Don't visualize the virtual root
            
            if joint_name in self.joint_spheres:
                sphere = self.joint_spheres[joint_name]
                # Translate sphere to joint position
                sphere.translate(position - np.array([0.0, 0.0, 0.0]))
    
    def _update_bones(self) -> None:
        """Update the positions of all bone line segments."""
        bones = [
            ("NECK", "NOSE"),
            ("NOSE", "R_EYE"), ("R_EYE", "R_EAR"),
            ("NOSE", "L_EYE"), ("L_EYE", "L_EAR"),
            ("NECK", "R_SHOULDER"), ("R_SHOULDER", "R_ELBOW"), ("R_ELBOW", "R_WRIST"),
            ("NECK", "L_SHOULDER"), ("L_SHOULDER", "L_ELBOW"), ("L_ELBOW", "L_WRIST"),
            ("R_SHOULDER", "R_HIP"), ("R_HIP", "R_KNEE"), ("R_KNEE", "R_ANKLE"),
            ("L_SHOULDER", "L_HIP"), ("L_HIP", "L_KNEE"), ("L_KNEE", "L_ANKLE"),
        ]
        
        for parent, child in bones:
            if parent not in self.joint_positions or child not in self.joint_positions:
                continue
            
            parent_pos = self.joint_positions[parent]
            child_pos = self.joint_positions[child]
            
            # Create line between parent and child
            line_set = self.bone_lines[(parent, child)]
            line_set.points = o3d.utility.Vector3dVector(np.array([parent_pos, child_pos]))
            line_set.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
            line_set.paint_uniform_color(np.array([1.0, 1.0, 1.0]))  # White bones
    
    def draw_model_skeleton(self, human_model: HumanModel) -> None:
        """
        Draw the complete model skeleton (bones and joints).
        Call this once to initialize the skeleton visualization.
        
        Args:
            human_model: HumanModel instance
        """
        self.create_joint_geometry()
        self.create_bone_geometry()
        self.update(human_model)
    
    def add_point_cloud(self, points: np.ndarray, color: np.ndarray = None) -> None:
        """
        Add a point cloud to the visualization.
        Useful for displaying camera keypoints for comparison.
        
        Args:
            points: Array of shape (N, 3) with 3D coordinates
            color: RGB color for the points, shape (3,). Default is blue.
        """
        if points.shape[1] != 3:
            raise ValueError("Points must have shape (N, 3)")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if color is None:
            color = np.array([0.0, 0.0, 1.0])  # Blue
        pcd.paint_uniform_color(color)
        
        self.vis.add_geometry(pcd)
    
    def add_camera_keypoints(self, keypoints: np.ndarray) -> None:
        """
        Add camera keypoints to the visualization for comparison.
        
        Args:
            keypoints: Array of shape (18, 3) with camera keypoint positions
        """
        # Filter out NaN values
        valid_kps = keypoints[~np.any(np.isnan(keypoints), axis=1)]
        
        if len(valid_kps) > 0:
            self.add_point_cloud(valid_kps, color=np.array([1.0, 0.0, 0.0]))  # Red for camera points
    
    def run(self) -> None:
        """Run the visualization loop."""
        while True:
            try:
                self.vis.poll_events()
                self.vis.update_renderer()
            except KeyboardInterrupt:
                break
    
    def close(self) -> None:
        """Close the visualization window."""
        if self.vis is not None:
            self.vis.destroy_window()


class ModelVisualizerWithComparison(ModelVisualizer):
    """
    Extended visualizer that displays both model skeleton and camera keypoints
    for comparison purposes.
    """
    
    def __init__(self, window_name: str = "Human Model Visualizer with Comparison"):
        """
        Initialize the comparison visualizer.
        
        Args:
            window_name: Title of the visualization window
        """
        super().__init__(window_name)
        self.camera_point_cloud: Optional[o3d.geometry.PointCloud] = None
    
    def update_with_comparison(self, human_model: HumanModel, camera_keypoints: Optional[np.ndarray] = None) -> None:
        """
        Update visualization with both model skeleton and camera keypoints.
        
        Args:
            human_model: HumanModel instance
            camera_keypoints: Optional array of camera keypoints, shape (18, 3)
        """
        self.update(human_model)
        
        if camera_keypoints is not None:
            # Remove old camera point cloud if it exists
            if self.camera_point_cloud is not None:
                self.vis.remove_geometry(self.camera_point_cloud)
            
            # Add new camera keypoints
            valid_kps = camera_keypoints[~np.any(np.isnan(camera_keypoints), axis=1)]
            
            if len(valid_kps) > 0:
                self.camera_point_cloud = o3d.geometry.PointCloud()
                self.camera_point_cloud.points = o3d.utility.Vector3dVector(valid_kps)
                self.camera_point_cloud.paint_uniform_color(np.array([1.0, 0.0, 0.0]))  # Red
                self.vis.add_geometry(self.camera_point_cloud)

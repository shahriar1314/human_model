"""
BodyTracker class - Manages ZED camera body tracking functionality
"""
import cv2
import math
import numpy as np
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer


# COCO-18 style naming
KP18_NAMES = [
    "nose", "neck",
    "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear"
]


class BodyTracker:
    """
    Handles ZED camera body tracking, including initialization,
    frame grabbing, body detection, and visualization.
    """
    
    def __init__(self, input_svo_file='', ip_address='', resolution=''):
        """
        Initialize the BodyTracker with camera and body tracking parameters.
        
        Args:
            input_svo_file: Path to SVO file for replay
            ip_address: IP address for streaming
            resolution: Camera resolution (HD2K, HD1200, HD1080, HD720, SVGA, VGA)
        """
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.body_param = sl.BodyTrackingParameters()
        self.bodies = sl.Bodies()
        self.image = sl.Mat()
        
        # Configuration
        self.init_params.camera_resolution = sl.RESOLUTION.HD1080
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        
        # Parse arguments and configure
        self._parse_args(input_svo_file, ip_address, resolution)
        
        # Open camera
        err = self.zed.open(self.init_params)
        if err > sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")
        
        # Enable positional tracking
        positional_tracking_parameters = sl.PositionalTrackingParameters()
        self.zed.enable_positional_tracking(positional_tracking_parameters)
        
        # Configure body tracking
        self.body_param.enable_tracking = True
        self.body_param.enable_body_fitting = False
        self.body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
        self.body_param.body_format = sl.BODY_FORMAT.BODY_18
        
        self.zed.enable_body_tracking(self.body_param)
        
        self.body_runtime_param = sl.BodyTrackingRuntimeParameters()
        self.body_runtime_param.detection_confidence_threshold = 40
        
        # Get camera info and setup display
        camera_info = self.zed.get_camera_information()
        display_resolution = sl.Resolution(
            min(camera_info.camera_configuration.resolution.width, 1280),
            min(camera_info.camera_configuration.resolution.height, 720)
        )
        self.image_scale = [
            display_resolution.width / camera_info.camera_configuration.resolution.width,
            display_resolution.height / camera_info.camera_configuration.resolution.height
        ]
        
        # Initialize OpenGL viewer
        self.viewer = gl.GLViewer()
        self.viewer.init(
            camera_info.camera_configuration.calibration_parameters.left_cam,
            self.body_param.enable_tracking,
            self.body_param.body_format
        )
        
        self.display_resolution = display_resolution
        self.key_wait = 10
        self.frame_idx = 0
    
    def _parse_args(self, input_svo_file, ip_address, resolution):
        """Configure camera from arguments."""
        if input_svo_file and input_svo_file.endswith((".svo", ".svo2")):
            self.init_params.set_from_svo_file(input_svo_file)
            print(f"[BodyTracker] Using SVO File input: {input_svo_file}")
        elif ip_address:
            ip_str = ip_address
            if (ip_str.replace(':', '').replace('.', '').isdigit() and
                len(ip_str.split('.')) == 4 and len(ip_str.split(':')) == 2):
                self.init_params.set_from_stream(ip_str.split(':')[0], int(ip_str.split(':')[1]))
                print(f"[BodyTracker] Using Stream input, IP: {ip_str}")
            elif (ip_str.replace(':', '').replace('.', '').isdigit() and
                  len(ip_str.split('.')) == 4):
                self.init_params.set_from_stream(ip_str)
                print(f"[BodyTracker] Using Stream input, IP: {ip_str}")
            else:
                print("[BodyTracker] Invalid IP format. Using live stream")
        
        if resolution:
            resolution_map = {
                "HD2K": sl.RESOLUTION.HD2K,
                "HD1200": sl.RESOLUTION.HD1200,
                "HD1080": sl.RESOLUTION.HD1080,
                "HD720": sl.RESOLUTION.HD720,
                "SVGA": sl.RESOLUTION.SVGA,
                "VGA": sl.RESOLUTION.VGA,
            }
            for key, res in resolution_map.items():
                if key in resolution:
                    self.init_params.camera_resolution = res
                    print(f"[BodyTracker] Using Camera in resolution {key}")
                    return
            print("[BodyTracker] No valid resolution entered. Using default")
    
    @staticmethod
    def _is_valid_xyz(p):
        """Check if a 3D point is valid."""
        return (p is not None
                and len(p) == 3
                and all(isinstance(v, (int, float, np.floating)) for v in p)
                and all(math.isfinite(float(v)) for v in p))
    
    def update(self):
        """
        Update body tracking: grab frame, retrieve bodies, and visualize.
        
        Returns:
            bool: True if viewer is still available, False if should exit
        """
        if not self.viewer.is_available():
            return False
        
        # Grab frame
        if self.zed.grab() > sl.ERROR_CODE.SUCCESS:
            return True
        
        # Retrieve image and bodies
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT, sl.MEM.CPU, self.display_resolution)
        self.zed.retrieve_bodies(self.bodies, self.body_runtime_param)
        
        # Print 3D keypoints every 30 frames
        # self.frame_idx += 1
        # if self.frame_idx % 30 == 0:
        #     for body in self.bodies.body_list:
        #         if body.tracking_state != sl.OBJECT_TRACKING_STATE.OK:
        #             continue
        #         kps3d = np.array(body.keypoint, dtype=np.float32)
        #         print(f"\nPerson ID: {body.id} | conf: {body.confidence}")
        #         for i in range(min(len(kps3d), 18)):
        #             p = kps3d[i]
        #             if self._is_valid_xyz(p):
        #                 print(f"{KP18_NAMES[i]}: {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}")
        
        # Update visualizations
        # Update visualizations
        self.viewer.update_view(self.image, self.bodies)
        image_left_ocv = self.image.get_data()
        cv_viewer.render_2D(
            image_left_ocv,
            self.image_scale,
            self.bodies.body_list,
            self.body_param.enable_tracking,
            self.body_param.body_format
        )
        cv2.imshow("ZED | 2D View", image_left_ocv)
        
        # Handle keyboard input
        key = cv2.waitKey(self.key_wait)
        if key == 113:  # 'q' key
            print("Exiting...")
            return False
        if key == 109:  # 'm' key
            if self.key_wait > 0:
                print("Pause")
                self.key_wait = 0
            else:
                print("Restart")
                self.key_wait = 10
        
        return True
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'viewer') and self.viewer is not None:
                self.viewer.exit()
        except Exception as e:
            print(f"Warning: Error closing viewer: {e}")
        
        try:
            if hasattr(self, 'image') and self.image is not None:
                self.image.free(sl.MEM.CPU)
        except Exception as e:
            print(f"Warning: Error freeing image: {e}")
        
        try:
            if hasattr(self, 'zed') and self.zed is not None:
                self.zed.disable_body_tracking()
        except Exception as e:
            print(f"Warning: Error disabling body tracking: {e}")
        
        try:
            if hasattr(self, 'zed') and self.zed is not None:
                self.zed.disable_positional_tracking()
        except Exception as e:
            print(f"Warning: Error disabling positional tracking: {e}")
        
        try:
            if hasattr(self, 'zed') and self.zed is not None:
                self.zed.close()
        except Exception as e:
            print(f"Warning: Error closing ZED camera: {e}")
        
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Warning: Error destroying opencv windows: {e}")
    
    def get_bodies(self):
        """Get the current bodies data."""
        return self.bodies
    
    def get_image(self):
        """Get the current image."""
        return self.image

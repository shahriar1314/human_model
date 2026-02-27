# zed_body18_stream.py
# ZED BODY_18 streamer yielding live (18,3) keypoints for each tracked person.

import cv2
import numpy as np
import pyzed.sl as sl

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer

def parse_args(init: sl.InitParameters, opt):
    if len(opt.input_svo_file) > 0 and opt.input_svo_file.endswith((".svo", ".svo2")):
        init.set_from_svo_file(opt.input_svo_file)
        print(f"[ZED] Using SVO: {opt.input_svo_file}")
    elif len(opt.ip_address) > 0:
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.')) == 4 and len(ip_str.split(':')) == 2:
            init.set_from_stream(ip_str.split(':')[0], int(ip_str.split(':')[1]))
            print("[ZED] Using stream:", ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.')) == 4:
            init.set_from_stream(ip_str)
            print("[ZED] Using stream:", ip_str)
        else:
            print("[ZED] Invalid IP format, using live camera.")

    res = opt.resolution
    if "HD2K" in res: init.camera_resolution = sl.RESOLUTION.HD2K
    elif "HD1200" in res: init.camera_resolution = sl.RESOLUTION.HD1200
    elif "HD1080" in res: init.camera_resolution = sl.RESOLUTION.HD1080
    elif "HD720" in res: init.camera_resolution = sl.RESOLUTION.HD720
    elif "SVGA" in res: init.camera_resolution = sl.RESOLUTION.SVGA
    elif "VGA" in res: init.camera_resolution = sl.RESOLUTION.VGA

class ZEDBody18Stream:
    def __init__(self, opt, enable_view: bool = True):
        self.opt = opt
        self.enable_view = enable_view
        self.zed = sl.Camera()
        self.viewer = None
        self.bodies = sl.Bodies()
        self.image = sl.Mat()
        self.body_runtime_param = sl.BodyTrackingRuntimeParameters()
        self.display_resolution = None
        self.image_scale = None
        self.body_param = None

    def open(self):
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        parse_args(init_params, self.opt)

        err = self.zed.open(init_params)
        if err > sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {repr(err)}")

        pos = sl.PositionalTrackingParameters()
        self.zed.enable_positional_tracking(pos)

        self.body_param = sl.BodyTrackingParameters()
        self.body_param.enable_tracking = True
        self.body_param.enable_body_fitting = False
        self.body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
        self.body_param.body_format = sl.BODY_FORMAT.BODY_18
        self.zed.enable_body_tracking(self.body_param)

        self.body_runtime_param.detection_confidence_threshold = 40

        camera_info = self.zed.get_camera_information()
        self.display_resolution = sl.Resolution(
            min(camera_info.camera_configuration.resolution.width, 1280),
            min(camera_info.camera_configuration.resolution.height, 720),
        )
        self.image_scale = [
            self.display_resolution.width / camera_info.camera_configuration.resolution.width,
            self.display_resolution.height / camera_info.camera_configuration.resolution.height,
        ]

        if self.enable_view:
            self.viewer = gl.GLViewer()
            self.viewer.init(
                camera_info.camera_configuration.calibration_parameters.left_cam,
                self.body_param.enable_tracking,
                self.body_param.body_format,
            )

    def close(self):
        try:
            if self.viewer is not None:
                self.viewer.exit()
        except Exception:
            pass
        try:
            self.image.free(sl.MEM.CPU)
        except Exception:
            pass
        try:
            self.zed.disable_body_tracking()
        except Exception:
            pass
        try:
            self.zed.disable_positional_tracking()
        except Exception:
            pass
        try:
            self.zed.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def frames(self):
        if self.enable_view:
            availability_fn = lambda: self.viewer.is_available()
        else:
            availability_fn = lambda: True

        key_wait = 10
        while availability_fn():
            if self.zed.grab() <= sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(self.image, sl.VIEW.LEFT, sl.MEM.CPU, self.display_resolution)
                self.zed.retrieve_bodies(self.bodies, self.body_runtime_param)

                image_bgr = self.image.get_data()

                out_bodies = []
                for body in self.bodies.body_list:
                    kps3d = np.array(body.keypoint, dtype=np.float32)
                    if kps3d.shape[0] >= 18:
                        kps3d = kps3d[:18, :]
                    else:
                        pad = np.full((18 - kps3d.shape[0], 3), np.nan, dtype=np.float32)
                        kps3d = np.vstack([kps3d, pad])

                    out_bodies.append({
                        "id": int(body.id),
                        "confidence": int(body.confidence),
                        "tracking_state": body.tracking_state,
                        "kp3d": kps3d,
                    })

                if self.enable_view:
                    self.viewer.update_view(self.image, self.bodies)
                    cv_viewer.render_2D(
                        image_bgr,
                        self.image_scale,
                        self.bodies.body_list,
                        self.body_param.enable_tracking,
                        self.body_param.body_format,
                    )
                    cv2.imshow("ZED | 2D View", image_bgr)
                    key = cv2.waitKey(key_wait)
                    if key == ord('q'):
                        break
                    if key == ord('m'):
                        key_wait = 0 if key_wait > 0 else 10

                yield {"image_bgr": image_bgr, "bodies": out_bodies}
            else:
                yield {"image_bgr": None, "bodies": []}
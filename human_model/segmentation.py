import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO

SVO_PATH = "/home/roolab/zed_recordings/test_vid_1.svo2"

# 1) Open SVO2 as input
init = sl.InitParameters()
init.set_from_svo_file(SVO_PATH)  # ZED will behave like a live camera feed for this file
init.svo_real_time_mode = False   # False = read as fast as possible (not real-time)

zed = sl.Camera()
err = zed.open(init)
if err != sl.ERROR_CODE.SUCCESS:
    raise RuntimeError(f"ZED open failed: {err}")

runtime = sl.RuntimeParameters()
image_zed = sl.Mat()

# 2) Load a YOLO segmentation model (choose your size)
# See Ultralytics "segment" task for available YOLO26 *-seg models
model = YOLO("yolo26n-seg.pt")

PERSON_CLASS_ID = 0  # COCO 'person' is class 0 for Ultralytics pretrained models

while True:
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)  # BGRA
        frame_bgra = image_zed.get_data()
        frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

        # 3) YOLO segmentation inference
        results = model.predict(source=frame_bgr, conf=0.25, verbose=False)[0]

        # 4) Build a single binary "human mask" from all person instances
        H, W = frame_bgr.shape[:2]
        human_mask = np.zeros((H, W), dtype=np.uint8)

        if results.masks is not None and results.boxes is not None:
            cls = results.boxes.cls.cpu().numpy().astype(int)
            masks = results.masks.data.cpu().numpy()  # [N, h, w] (model mask size)

            for i in range(len(cls)):
                if cls[i] != PERSON_CLASS_ID:
                    continue

                m = (masks[i] * 255).astype(np.uint8)     # [h, w]
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)  # -> [H, W]
                human_mask = np.maximum(human_mask, m)


        # 5) Visualize
        vis = frame_bgr.copy()
        vis[human_mask > 0] = (0.5 * vis[human_mask > 0]).astype(np.uint8)  # darken human area
        cv2.imshow("YOLO Human Seg", vis)
        cv2.imshow("Human Mask", human_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # End of file reached
        break

zed.close()
cv2.destroyAllWindows()

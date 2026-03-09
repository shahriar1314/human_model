import cv2
import numpy as np

BODY18_EDGES = [
    (1, 0),
    (0, 14), (14, 16),
    (0, 15), (15, 17),
    (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (2, 8), (8, 9), (9, 10),
    (5, 11), (11, 12), (12, 13),
]

def project_to_2d(kp3d, width, height, scale=250.0):
    cx = width // 2
    cy = int(height * 0.75)

    kp2d = np.full((18, 2), np.nan, dtype=np.float32)
    for i, p in enumerate(kp3d):
        if p is None or len(p) != 3 or not np.isfinite(p).all():
            continue
        x, y, z = p
        u = cx + x * scale
        v = cy - y * scale
        kp2d[i] = [u, v]
    return kp2d

def draw_body18(image, keypoints_2d, color=(0, 255, 0)):
    img = image.copy()

    for a, b in BODY18_EDGES:
        if np.isfinite(keypoints_2d[a]).all() and np.isfinite(keypoints_2d[b]).all():
            p0 = tuple(np.round(keypoints_2d[a]).astype(int))
            p1 = tuple(np.round(keypoints_2d[b]).astype(int))
            cv2.line(img, p0, p1, color, 2, cv2.LINE_AA)

    for i in range(18):
        if np.isfinite(keypoints_2d[i]).all():
            p = tuple(np.round(keypoints_2d[i]).astype(int))
            cv2.circle(img, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, p, 4, color, 1, cv2.LINE_AA)

    return img

def render_split_view(image_left, live_kp, pred_kp):
    if image_left.ndim == 3 and image_left.shape[2] == 4:
        image_left = cv2.cvtColor(image_left, cv2.COLOR_BGRA2BGR)

    H, W = image_left.shape[:2]

    # blank backgrounds (no camera image)
    panel_left = np.full((H, W, 3), 245, dtype=np.uint8)
    panel_right = np.full((H, W, 3), 245, dtype=np.uint8)

    live2d = project_to_2d(live_kp, W, H)
    pred2d = project_to_2d(pred_kp, W, H)

    panel_left = draw_body18(panel_left, live2d, color=(0, 255, 0))
    panel_right = draw_body18(panel_right, pred2d, color=(255, 0, 0))

    cv2.putText(panel_left, "ZED BODY_18", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    cv2.putText(panel_right, "Internal Model", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    return np.hstack((panel_left, panel_right))
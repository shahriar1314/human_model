# main.py
# Runs Active Inference pose inference: infer joint angles minimizing free energy per frame.

import argparse
import numpy as np
import pyzed.sl as sl

from zed_body18_stream import ZEDBody18Stream
from human_kinematic_model import lengths_from_standard_np, HumanKinematicModel, KP18_NAMES
from vfe_inference import ActiveInferencePoseEstimator, InferenceConfig

def is_tracking_ok(state) -> bool:
    try:
        return state == sl.OBJECT_TRACKING_STATE.OK
    except Exception:
        return True

def standard_template_np() -> np.ndarray:
    """
    Used ONLY to derive fixed segment lengths + face offsets.
    Angles are what get inferred online.
    """
    kp = np.zeros((18, 3), dtype=np.float32)

    # Face / head
    kp[0]  = [0.0, 1.65, 0.0]      # nose
    kp[14] = [0.08, 1.68, 0.05]    # r_eye
    kp[15] = [-0.08, 1.68, 0.05]   # l_eye
    kp[16] = [0.12, 1.65, 0.0]     # r_ear
    kp[17] = [-0.12, 1.65, 0.0]    # l_ear

    # Neck
    kp[1]  = [0.0, 1.50, 0.0]

    # Shoulders
    kp[2]  = [0.20, 1.50, 0.0]
    kp[5]  = [-0.20, 1.50, 0.0]

    # Arms
    kp[3]  = [0.35, 1.20, 0.0]
    kp[4]  = [0.50, 0.95, 0.0]
    kp[6]  = [-0.35, 1.20, 0.0]
    kp[7]  = [-0.50, 0.95, 0.0]

    # Hips
    kp[8]  = [0.15, 1.00, 0.0]
    kp[11] = [-0.15, 1.00, 0.0]

    # Legs
    kp[9]  = [0.15, 0.60, 0.0]
    kp[10] = [0.15, 0.10, 0.0]
    kp[12] = [-0.15, 0.60, 0.0]
    kp[13] = [-0.15, 0.10, 0.0]

    return kp

def print_summary(person_id, conf, mean_l2, rmse, used_anchors):
    print("\n" + "=" * 100)
    print(f"Person {person_id} | conf={conf} | mean_L2={mean_l2:.4f} m | RMSE={rmse:.4f} m | anchors={used_anchors}")
    print("=" * 100)

def print_table(live, pred, diff, per_l2):
    print(f"{'kp':>2s}  {'name':<12s} | {'live_x':>8s} {'live_y':>8s} {'live_z':>8s} | "
          f"{'pred_x':>8s} {'pred_y':>8s} {'pred_z':>8s} | {'dx':>8s} {'dy':>8s} {'dz':>8s} | {'L2(m)':>8s}")
    print("-" * 110)
    for i, name in enumerate(KP18_NAMES):
        lx, ly, lz = live[i]
        px, py, pz = pred[i]
        dx, dy, dz = diff[i]
        e = per_l2[i]

        def fmt(v):  return f"{v:8.3f}" if np.isfinite(v) else "   nan  "
        def fmte(v): return f"{v:8.4f}" if np.isfinite(v) else "   nan  "

        print(f"{i:2d}  {name:<12s} | {fmt(lx)} {fmt(ly)} {fmt(lz)} | "
              f"{fmt(px)} {fmt(py)} {fmt(pz)} | {fmt(dx)} {fmt(dy)} {fmt(dz)} | {fmte(e)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, default='')
    parser.add_argument('--ip_address', type=str, default='')
    parser.add_argument('--resolution', type=str, default='')
    parser.add_argument('--print_every', type=int, default=60)
    parser.add_argument('--no_view', action='store_true')

    # Inference settings
    parser.add_argument('--iters', type=int, default=60)
    parser.add_argument('--lr', type=float, default=2e-2)
    parser.add_argument('--sigma', type=float, default=0.05)
    parser.add_argument('--w_limits', type=float, default=5.0)
    parser.add_argument('--w_sym', type=float, default=1.0)
    parser.add_argument('--anchors', type=str, default='1,2,5,8,11')  # neck, shoulders, hips
    parser.add_argument('--device', type=str, default='cpu')
    opt = parser.parse_args()

    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        raise SystemExit("Specify only --input_svo_file or --ip_address (or none).")

    anchors = [int(x) for x in opt.anchors.split(",") if x.strip() != ""]

    # Fixed morphology (segment lengths) derived once
    template = standard_template_np()
    lengths = lengths_from_standard_np(template)

    model = HumanKinematicModel(lengths, device=opt.device).to(opt.device)
    cfg = InferenceConfig(
        anchors=anchors,
        sigma_obs=opt.sigma,
        w_limits=opt.w_limits,
        w_sym=opt.w_sym,
        iters=opt.iters,
        lr=opt.lr,
        device=opt.device,
    )
    estimator = ActiveInferencePoseEstimator(model, cfg)

    stream = ZEDBody18Stream(opt, enable_view=(not opt.no_view))
    stream.open()

    frame_idx = 0
    try:
        for frame in stream.frames():
            frame_idx += 1
            if (frame_idx % opt.print_every) != 0:
                continue

            for body in frame["bodies"]:
                if not is_tracking_ok(body["tracking_state"]):
                    continue

                live = body["kp3d"].astype(np.float32)
                res = estimator.infer(live)

                print_summary(body["id"], body["confidence"], res.mean_l2, res.rmse, res.used_anchors)
                print_table(live, res.kp_pred_aligned, res.diff, res.per_l2)

                # Active inference "action" heuristic (expected free energy proxy):
                valid_count = int(np.isfinite(live).all(axis=1).sum())
                if valid_count < 12:
                    print("\n[Action hint] Many joints missing/invalid. Move viewpoint / reduce occlusion / get closer to reduce expected free energy.")

    finally:
        stream.close()

if __name__ == "__main__":
    main()
import argparse
import time 
import cv2
import numpy as np
import pyzed.sl as sl
from datetime import datetime

from zed_body18_stream import ZEDBody18Stream
from human_kinematic_model import lengths_from_standard_np, HumanKinematicModel, KP18_NAMES
from vfe_inference import AInfLaplacePoseEstimator, InferenceConfig
from visualization_compare import render_split_view


def is_tracking_ok(state) -> bool:
    try:
        return state == sl.OBJECT_TRACKING_STATE.OK
    except Exception:
        return True


def standard_template_np() -> np.ndarray:
    """
    Used ONLY to derive fixed segment lengths + face offsets.
    Joint angles are inferred online.
    """
    kp = np.zeros((18, 3), dtype=np.float32)

    kp[0]  = [0.0, 1.65, 0.0]      # nose
    kp[14] = [0.08, 1.68, 0.05]    # r_eye
    kp[15] = [-0.08, 1.68, 0.05]   # l_eye
    kp[16] = [0.12, 1.65, 0.0]     # r_ear
    kp[17] = [-0.12, 1.65, 0.0]    # l_ear

    kp[1]  = [0.0, 1.50, 0.0]      # neck

    kp[2]  = [0.20, 1.50, 0.0]     # r_shoulder
    kp[5]  = [-0.20, 1.50, 0.0]    # l_shoulder

    kp[3]  = [0.35, 1.20, 0.0]     # r_elbow
    kp[4]  = [0.50, 0.95, 0.0]     # r_wrist
    kp[6]  = [-0.35, 1.20, 0.0]    # l_elbow
    kp[7]  = [-0.50, 0.95, 0.0]    # l_wrist

    kp[8]  = [0.15, 1.00, 0.0]     # r_hip
    kp[11] = [-0.15, 1.00, 0.0]    # l_hip

    kp[9]  = [0.15, 0.60, 0.0]     # r_knee
    kp[10] = [0.15, 0.10, 0.0]     # r_ankle
    kp[12] = [-0.15, 0.60, 0.0]    # l_knee
    kp[13] = [-0.15, 0.10, 0.0]    # l_ankle

    return kp


def print_summary(person_id, conf, mean_l2, rmse, used_anchors, valid_count, uncertainty_trace):
    print("\n" + "=" * 120)
    print(
        f"Person {person_id} | conf={conf} | mean_L2={mean_l2:.4f} m | RMSE={rmse:.4f} m | "
        f"valid={valid_count}/18 | tr(cov)={uncertainty_trace:.6f} | anchors={used_anchors}"
    )
    print("=" * 120)


def print_table(live, pred, diff, per_l2):
    print(f"{'kp':>2s}  {'name':<12s} | {'live_x':>8s} {'live_y':>8s} {'live_z':>8s} | "
          f"{'pred_x':>8s} {'pred_y':>8s} {'pred_z':>8s} | {'dx':>8s} {'dy':>8s} {'dz':>8s} | {'L2(m)':>8s}")
    print("-" * 125)

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
    parser.add_argument('--print_every', type=int, default=30)
    parser.add_argument('--no_view', action='store_true')
    parser.add_argument('--output_video', type=str, default='', help='Path to save output video (e.g., output.mp4)')

    # AInf configuration
    parser.add_argument('--anchors', type=str, default='1,2,5,8,11')
    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--sigma_obs', type=float, default=0.06)
    parser.add_argument('--sigma_dyn', type=float, default=0.25)
    parser.add_argument('--gn_steps', type=int, default=2)
    parser.add_argument('--damping', type=float, default=1e-3)
    parser.add_argument('--step_size', type=float, default=1.0, help='Scaling factor for Newton step (0 < step_size <= 1)')
    parser.add_argument('--adam_lr', type=float, default=0.01, help='Adam optimizer learning rate')
    parser.add_argument('--f_tol', type=float, default=1e-4, help='Early-stop tolerance on |F_t - F_{t-1}|')
    parser.add_argument('--min_steps', type=int, default=3, help='Minimum optimizer steps before early stopping')
    parser.add_argument('--debug_interval', type=int, default=0, help='Print optimizer debug every N steps (0 disables)')

    parser.add_argument('--w_limits', type=float, default=5.0)
    parser.add_argument('--w_sym', type=float, default=1.0)

    parser.add_argument('--sigma_min', type=float, default=0.02)
    parser.add_argument('--sigma_max', type=float, default=0.15)

    # Action thresholds (simple policy)
    parser.add_argument('--min_valid', type=int, default=12)
    parser.add_argument('--uncertainty_thresh', type=float, default=0.05)

    opt = parser.parse_args()

    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        raise SystemExit("Specify only --input_svo_file or --ip_address (or none).")

    anchors = [int(x) for x in opt.anchors.split(",") if x.strip() != ""]

    template = standard_template_np()
    lengths = lengths_from_standard_np(template)

    model = HumanKinematicModel(lengths, device=opt.device).to(opt.device)

    cfg = InferenceConfig(
        anchors=anchors,
        device=opt.device,
        sigma_obs=opt.sigma_obs,
        sigma_dyn=opt.sigma_dyn,
        sigma_min=opt.sigma_min,
        sigma_max=opt.sigma_max,
        w_limits=opt.w_limits,
        w_sym=opt.w_sym,
        gn_steps=opt.gn_steps,
        damping=opt.damping,
        step_size=opt.step_size,
        adam_lr=opt.adam_lr,
        f_tol=opt.f_tol,
        min_steps=opt.min_steps,
        debug_interval=opt.debug_interval,
    )

    estimator = AInfLaplacePoseEstimator(model, cfg)

    stream = ZEDBody18Stream(opt, enable_view=(not opt.no_view))
    stream.open()

    # Initialize video writer if output path specified
    video_writer = None
    frame_idx = 0

    # ========== ERROR TRACKING ==========
    error_history = {
        'mean_l2': [],
        'rmse': [],
        'per_l2_all': [],
        'valid_counts': [],
        'timestamps': [],
    }
    start_time = datetime.now()
    # =====================================

    try:
        for frame in stream.frames():
            image_bgr = frame["image_bgr"]

            # Initialize video writer on first valid frame
            if video_writer is None and image_bgr is not None and len(opt.output_video) > 0:
                h, w = image_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 30.0  # ZED typical FPS
                video_writer = cv2.VideoWriter(opt.output_video, fourcc, fps, (w, h))
                if video_writer.isOpened():
                    print(f"[Video] Saving to {opt.output_video}")
                else:
                    print(f"[Video] Failed to open output file {opt.output_video}")
                    video_writer = None

            # Write frame to video if writer is active
            if video_writer is not None and image_bgr is not None:
                img = image_bgr

                # ZED often returns BGRA (H,W,4). VideoWriter expects BGR (H,W,3).
                if img.ndim == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                # Make sure type/contiguity are OK for OpenCV
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                img = np.ascontiguousarray(img)

                video_writer.write(img)
            frame_idx += 1
            # if (frame_idx % opt.print_every) != 0:
            #     continue

            # to count the total processing time per frame (for performance monitoring)
            frame_start = time.perf_counter()
            infer_time = 0.0  # Initialize inference time variable to use it outside of the loop

            for body in frame["bodies"]:
                if not is_tracking_ok(body["tracking_state"]):
                    continue

                live = body["kp3d"].astype(np.float32)
                kp_conf = body.get("kp_conf", None)

                
                # Running VFE_INFERENCE and measure time
                infer_start = time.perf_counter()
                res = estimator.infer(live, kp_conf_np=kp_conf)
                infer_time = time.perf_counter() - infer_start
                

                if image_bgr is not None:
                    pred_kp = res.kp_pred_aligned
                    comparison = render_split_view(image_bgr, live, pred_kp)
                    cv2.imshow("ZED vs Internal Model", comparison)
                    cv2.waitKey(1)

                if frame_idx % opt.print_every == 0:
                    print_summary(
                        body["id"], body["confidence"],
                        res.mean_l2, res.rmse,
                        res.used_anchors,
                        res.valid_count,
                        res.uncertainty_trace
                    )
                    print_table(live, res.kp_pred_aligned, res.diff, res.per_l2)

                # ========== COLLECT ERROR DATA ==========
                error_history['mean_l2'].append(res.mean_l2)
                error_history['rmse'].append(res.rmse)
                error_history['per_l2_all'].extend(res.per_l2[res.per_l2 > 0])  # Only valid errors
                error_history['valid_counts'].append(res.valid_count)
                error_history['timestamps'].append(datetime.now())
                # =========================================

                # Simple Active Inference "action" policy (expected free energy proxy):
                if frame_idx % opt.print_every == 0 and (res.valid_count < opt.min_valid or res.uncertainty_trace > opt.uncertainty_thresh):
                    print("\n[Action hint] High uncertainty / missing joints -> change viewpoint, reduce occlusion, move closer.\n")

            # Calculate and print elapsed time for the frame
            frame_end = time.perf_counter()
            elapsed_ms = (frame_end - frame_start) * 1000
            print(f"[Complete Loop ] Frame {frame_idx} processing time: {elapsed_ms:.2f} ms")
            # Inference time is calculated for the last body (person) part. It is out of the for loop, so it will be the time for the last processed body in the frame. For more detailed timing, consider measuring inside the loop for each body.
            print(f"[Only Inference] Frame {frame_idx} processing time: {infer_time*1000:.2f} ms")

            # ========== PRINT RUNNING AVERAGE ERROR ==========
            if frame_idx % opt.print_every == 0 and len(error_history['mean_l2']) > 0:
                elapsed_time = (datetime.now() - start_time).total_seconds()
                avg_mean_l2 = np.mean(error_history['mean_l2'])
                avg_rmse = np.mean(error_history['rmse'])
                avg_valid = np.mean(error_history['valid_counts'])
                
                print("\n" + "=" * 120)
                print(f"[AVERAGE ERROR REPORT] Frames processed: {frame_idx} | Elapsed: {elapsed_time:.1f}s")
                print(f"  Average Mean L2:      {avg_mean_l2:.4f} m")
                print(f"  Average RMSE:         {avg_rmse:.4f} m")
                print(f"  Average Valid Joints: {avg_valid:.1f}/18")
                if len(error_history['per_l2_all']) > 0:
                    print(f"  Overall Per-Joint L2: {np.mean(error_history['per_l2_all']):.4f} m")
                    print(f"  Min Per-Joint L2:     {np.min(error_history['per_l2_all']):.4f} m")
                    print(f"  Max Per-Joint L2:     {np.max(error_history['per_l2_all']):.4f} m")
                print("=" * 120 + "\n")
            # ================================================ 

    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"[Video] Saved video successfully")
        stream.close()

        # ========== FINAL ERROR SUMMARY ==========
        if len(error_history['mean_l2']) > 0:
            print("\n" + "=" * 120)
            print("[FINAL ERROR SUMMARY]")
            elapsed_time = (datetime.now() - start_time).total_seconds()
            print(f"  Total frames processed: {frame_idx}")
            print(f"  Total bodies tracked:   {len(error_history['mean_l2'])}")
            print(f"  Total elapsed time:     {elapsed_time:.1f}s")
            print(f"\n  Average Mean L2 Error:  {np.mean(error_history['mean_l2']):.4f} m")
            print(f"  Std Dev Mean L2:        {np.std(error_history['mean_l2']):.4f} m")
            print(f"  Min Mean L2:            {np.min(error_history['mean_l2']):.4f} m")
            print(f"  Max Mean L2:            {np.max(error_history['mean_l2']):.4f} m")
            
            print(f"\n  Average RMSE:           {np.mean(error_history['rmse']):.4f} m")
            print(f"  Std Dev RMSE:           {np.std(error_history['rmse']):.4f} m")
            print(f"  Min RMSE:               {np.min(error_history['rmse']):.4f} m")
            print(f"  Max RMSE:               {np.max(error_history['rmse']):.4f} m")
            
            print(f"\n  Average Valid Joints:   {np.mean(error_history['valid_counts']):.1f}/18")
            if len(error_history['per_l2_all']) > 0:
                print(f"  Overall Per-Joint L2:   {np.mean(error_history['per_l2_all']):.4f} m")
            print("=" * 120)
        # =========================================


if __name__ == "__main__":
    main()
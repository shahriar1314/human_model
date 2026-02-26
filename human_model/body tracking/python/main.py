import argparse
import numpy as np

from human_standard_model import get_standard_kp3d_standing, KP18_NAMES
from skeleton_compare import SkeletonComparator
from zed_body18_stream import ZEDBody18Stream
import pyzed.sl as sl

def is_tracking_ok(state) -> bool:
    try:
        return state == sl.OBJECT_TRACKING_STATE.OK
    except Exception:
        # if state is already an int/enum-like
        return True

def print_comparison(person_id: int, conf: int, live: np.ndarray, aligned_std: np.ndarray, per_l2: np.ndarray, rmse: float, mean_l2: float):
    print("\n" + "=" * 110)
    print(f"Person {person_id} | conf={conf} | mean_L2={mean_l2:.4f} m | RMSE={rmse:.4f} m")
    print("=" * 110)
    print(f"{'kp':>2s}  {'name':<12s} | {'live_x':>8s} {'live_y':>8s} {'live_z':>8s} | "
          f"{'std_x':>8s} {'std_y':>8s} {'std_z':>8s} | {'dx':>8s} {'dy':>8s} {'dz':>8s} | {'L2(m)':>8s}")
    print("-" * 110)

    for i, name in enumerate(KP18_NAMES):
        lx, ly, lz = live[i]
        sx, sy, sz = aligned_std[i]
        dx, dy, dz = (live[i] - aligned_std[i])
        e = per_l2[i]
        def fmt(v): return f"{v:8.3f}" if np.isfinite(v) else "   nan  "
        def fmte(v): return f"{v:8.4f}" if np.isfinite(v) else "   nan  "
        print(f"{i:2d}  {name:<12s} | {fmt(lx)} {fmt(ly)} {fmt(lz)} | {fmt(sx)} {fmt(sy)} {fmt(sz)} | "
              f"{fmt(dx)} {fmt(dy)} {fmt(dz)} | {fmte(e)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, default='')
    parser.add_argument('--ip_address', type=str, default='')
    parser.add_argument('--resolution', type=str, default='')
    parser.add_argument('--print_every', type=int, default=30, help='print every N frames')
    parser.add_argument('--no_view', action='store_true', help='disable GL/2D viewer windows')
    parser.add_argument('--allow_scale', action='store_true', help='allow similarity scale in alignment')
    parser.add_argument('--anchors', type=str, default='1,2,5,8,11', help='anchor joint indices CSV (default neck, shoulders, hips)')
    opt = parser.parse_args()

    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        raise SystemExit("Specify only --input_svo_file or --ip_address (or none).")

    anchors = [int(x) for x in opt.anchors.split(",") if x.strip() != ""]
    standard = get_standard_kp3d_standing()

    comparator = SkeletonComparator(anchor_indices=anchors, allow_scale=opt.allow_scale)

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

                live = body["kp3d"].astype(np.float64)
                res = comparator.compare(live, standard)

                print_comparison(
                    person_id=body["id"],
                    conf=body["confidence"],
                    live=live,
                    aligned_std=res.aligned_standard,
                    per_l2=res.per_joint_l2,
                    rmse=res.rmse,
                    mean_l2=res.mean_l2,
                )
                print(f"Alignment used joints: {res.used_indices} | scale={res.scale:.4f}")
    finally:
        stream.close()

if __name__ == "__main__":
    main()
"""
Main entry point that uses BodyTracker and HumanModel classes.
Combines ZED body tracking with human model kinematics.
"""
import argparse
import numpy as np
from body_tracker import BodyTracker
from human_model_class import HumanModel


def main(input_svo_file='', ip_address='', resolution=''):
    """
    Main loop that runs body tracking and human model updates together.
    
    Args:
        input_svo_file: Path to SVO file for replay
        ip_address: IP address for streaming
        resolution: Camera resolution
    """
    print("Initializing Body Tracker and Human Model...", flush=True)
    
    try:
        # Initialize BodyTracker
        print("Creating BodyTracker...", flush=True)
        body_tracker = BodyTracker(
            input_svo_file=input_svo_file,
            ip_address=ip_address,
            resolution=resolution
        )
        print("BodyTracker initialized!", flush=True)
        
        # Initialize HumanModel
        print("Creating HumanModel...", flush=True)
        human_model = HumanModel(use_virtual_root=True, print_interval=1)
        print(f"HumanModel initialized! joint_angles count: {len(human_model.joint_angles)}", flush=True)
        
        print("Starting main loop... Press 'q' to quit, or 'm' to pause/restart", flush=True)
        
        # Main loop
        update_count = 0
        while True:
            print(f"[FRAME {update_count}]", flush=True)
            # Update body tracker (grab frame, retrieve bodies, visualize)
            tracker_result = body_tracker.update()
            print(f"  body_tracker.update() returned: {tracker_result}", flush=True)
            
            if not tracker_result:
                print("Exiting main loop...", flush=True)
                break
            
            # Update human model with detected bodies
            bodies = body_tracker.get_bodies()
            print(f"  Found {len(bodies.body_list)} bodies", flush=True)
          
            for body_idx, body in enumerate(bodies.body_list):
                print(f"  Body {body_idx}: id={body.id}", flush=True)
                # Process each detected body
                kp3d = np.array(body.keypoint, dtype=np.float32)
                
                # Check if keypoints are valid (non-NaN)
                has_nan = np.any(np.isnan(kp3d))
                print(f"    Keypoints have NaN: {has_nan}", flush=True)
                
                if not has_nan:
                    print(f"    Calling human_model.update()...", flush=True)
                    # Update human model with keypoints
                    human_model.update(kp3d)
                    print(f"    human_model.update() completed", flush=True)
            
            update_count += 1
            if update_count > 10:
                print("Stopping after 10 frames for testing", flush=True)
                break
        
        print("Exiting...", flush=True)
        
    except Exception as e:
        print(f"Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'body_tracker' in locals():
            try:
                print("Cleaning up body_tracker...", flush=True)
                body_tracker.cleanup()
                print("Cleanup completed.", flush=True)
            except Exception as e:
                print(f"Error during cleanup: {e}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run body tracking with human model kinematics'
    )
    parser.add_argument(
        '--input_svo_file',
        type=str,
        help='Path to an .svo file, if you want to replay it',
        default=''
    )
    parser.add_argument(
        '--ip_address',
        type=str,
        help='IP Address, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup',
        default=''
    )
    parser.add_argument(
        '--resolution',
        type=str,
        help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA',
        default=''
    )
    
    opt = parser.parse_args()
    
    # Validate arguments
    if opt.input_svo_file and opt.ip_address:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit(1)
    
    main(
        input_svo_file=opt.input_svo_file,
        ip_address=opt.ip_address,
        resolution=opt.resolution
    )

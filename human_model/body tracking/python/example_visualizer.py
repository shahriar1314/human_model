"""
Example of using the Open3D ModelVisualizer with HumanModel.
Demonstrates visualization of the model skeleton and optional camera keypoint comparison.
"""
import numpy as np
from body_tracker import BodyTracker
from human_model_class import HumanModel
from model_visualizer import ModelVisualizer, ModelVisualizerWithComparison


def main_visualize_model_only():
    """Visualize the model skeleton only (without camera data)."""
    print("Starting model-only visualization...", flush=True)
    
    # Create human model
    human_model = HumanModel(use_virtual_root=True, print_interval=0)
    
    # Create visualizer
    visualizer = ModelVisualizer()
    
    # Draw the skeleton
    visualizer.draw_model_skeleton(human_model)
    print("Skeleton drawn. Close window to exit.", flush=True)
    
    # Run the visualization
    visualizer.run()
    visualizer.close()


def main_visualize_with_tracking():
    """Visualize model skeleton with live camera tracking and comparison."""
    print("Starting visualization with tracking...", flush=True)
    
    try:
        # Initialize tracker and model
        print("Creating BodyTracker...", flush=True)
        body_tracker = BodyTracker()
        
        print("Creating HumanModel...", flush=True)
        human_model = HumanModel(use_virtual_root=True, print_interval=0)
        
        # Create comparison visualizer
        print("Creating visualizer...", flush=True)
        visualizer = ModelVisualizerWithComparison()
        visualizer.draw_model_skeleton(human_model)
        
        print("Starting visualization loop... Press 'q' in tracker window to quit", flush=True)
        
        update_count = 0
        while True:
            # Update tracker
            result = body_tracker.update()
            if not result:
                break
            
            # Get bodies and update model
            bodies = body_tracker.get_bodies()
            
            for body in bodies.body_list:
                kp3d = np.array(body.keypoint, dtype=np.float32)
                
                # Only update if keypoints are valid
                if not np.any(np.isnan(kp3d)):
                    human_model.update(kp3d)
                    
                    # Update visualization with comparison
                    visualizer.update_with_comparison(human_model, kp3d)
            
            update_count += 1
            if update_count % 10 == 0:
                print(f"Frame {update_count}", flush=True)
        
        print("Exiting...", flush=True)
        visualizer.close()
        body_tracker.cleanup()
        
    except Exception as e:
        print(f"Error: {e}", flush=True)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--with-tracker':
        main_visualize_with_tracking()
    else:
        print("Usage:")
        print("  python3 example_visualizer.py              # Model skeleton only")
        print("  python3 example_visualizer.py --with-tracker  # With live camera tracking")
        print()
        main_visualize_model_only()

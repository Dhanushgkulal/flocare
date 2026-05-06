"""
YOLO Model Inference Script
Run the trained YOLO model on images, videos, or webcam
"""

import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    try:
        from ultralytics import YOLO
        import cv2
        return True
    except ImportError as e:
        print("Missing dependencies!")
        print("\nPlease install required packages:")
        print("pip install ultralytics opencv-python")
        return False

def run_inference(model_path='best.pt', source=None, save=True, show=False):
    """
    Run YOLO inference on various sources
    
    Args:
        model_path: Path to the model file
        source: Input source (image path, video path, 0 for webcam, or None for demo)
        save: Whether to save results
        show: Whether to display results
    """
    from ultralytics import YOLO
    
    # Load the model
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    
    # Print model info
    print(f"\nModel loaded successfully!")
    print(f"Classes: {model.names}")
    print(f"Number of classes: {len(model.names)}")
    
    # Determine source
    if source is None:
        print("\nNo source specified. Use one of:")
        print("  - Image: python run_yolo.py image.jpg")
        print("  - Video: python run_yolo.py video.mp4")
        print("  - Webcam: python run_yolo.py 0")
        print("  - Folder: python run_yolo.py path/to/folder")
        return
    
    # Run inference
    print(f"\nRunning inference on: {source}")
    results = model(source, save=save, show=show)
    
    # Print results
    print(f"\n{'='*50}")
    print("INFERENCE RESULTS")
    print('='*50)
    
    for i, r in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Image shape: {r.orig_shape}")
        print(f"  Detections: {len(r.boxes)}")
        
        if len(r.boxes) > 0:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"    - {model.names[cls]}: {conf:.2f}")
    
    if save:
        print(f"\n✓ Results saved to: runs/detect/predict")
    
    return results

def main():
    # Check dependencies
    if not check_dependencies():
        return
    
    # Parse arguments
    if len(sys.argv) > 1:
        source = sys.argv[1]
        
        # Convert '0' to integer for webcam
        if source == '0':
            source = 0
        
        # Check if file/folder exists (unless it's webcam)
        if source != 0 and not os.path.exists(source):
            print(f"Error: Source '{source}' not found!")
            return
        
        run_inference('best.pt', source=source, save=True, show=False)
    else:
        # No arguments - show help
        run_inference('best.pt', source=None)

if __name__ == "__main__":
    main()

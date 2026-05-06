"""
Live Camera Test for YOLO Model
Shows real-time object detection from your laptop camera
"""

import cv2
from ultralytics import YOLO
import time

def run_camera_detection(model_path='best.pt', camera_id=0, conf_threshold=0.5):
    """
    Run YOLO detection on live camera feed
    
    Args:
        model_path: Path to the YOLO model
        camera_id: Camera device ID (0 for default laptop camera)
        conf_threshold: Confidence threshold for detections
    """
    print("="*60)
    print("YOLO Live Camera Detection")
    print("="*60)
    
    # Load the model
    print(f"\nLoading model from {model_path}...")
    model = YOLO(model_path)
    print(f"✓ Model loaded successfully!")
    print(f"\nDetectable classes ({len(model.names)}):")
    for idx, name in model.names.items():
        print(f"  - {name}")
    
    # Open camera
    print(f"\nOpening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera!")
        print("\nTroubleshooting:")
        print("  1. Make sure no other application is using the camera")
        print("  2. Try a different camera ID: python camera_test.py 1")
        print("  3. Check camera permissions in Windows settings")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✓ Camera opened successfully!")
    print("\n" + "="*60)
    print("CONTROLS:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("  - Press 'p' to pause/resume")
    print("="*60)
    print("\nStarting detection...\n")
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    paused = False
    
    try:
        while True:
            if not paused:
                # Read frame from camera
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Error: Could not read frame from camera")
                    break
                
                # Run YOLO inference
                results = model(frame, conf=conf_threshold, verbose=False)
                
                # Get annotated frame
                annotated_frame = results[0].plot()
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps_end_time = time.time()
                    fps = 30 / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                
                # Add FPS and detection count to frame
                detections = len(results[0].boxes)
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Detections: {detections}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Print detections to console
                if detections > 0:
                    detected_objects = []
                    for box in results[0].boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        detected_objects.append(f"{model.names[cls]}({conf:.2f})")
                    print(f"Frame {frame_count}: {', '.join(detected_objects)}")
            else:
                # Show paused message
                cv2.putText(annotated_frame, "PAUSED - Press 'p' to resume", 
                           (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('YOLO Camera Detection - Press Q to quit', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n✓ Quitting...")
                break
            elif key == ord('s'):
                filename = f"capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"✓ Saved frame to {filename}")
            elif key == ord('p'):
                paused = not paused
                print(f"{'⏸ Paused' if paused else '▶ Resumed'}")
    
    except KeyboardInterrupt:
        print("\n✓ Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✓ Camera released. Total frames processed: {frame_count}")
        print("="*60)

if __name__ == "__main__":
    import sys
    
    # Check if custom camera ID is provided
    camera_id = 0
    if len(sys.argv) > 1:
        try:
            camera_id = int(sys.argv[1])
        except ValueError:
            print(f"Invalid camera ID. Using default (0)")
    
    # Run camera detection
    run_camera_detection('best.pt', camera_id=camera_id, conf_threshold=0.5)

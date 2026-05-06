"""
Test YOLO model on an image file
Alternative to live camera if camera access is blocked
"""

import cv2
from ultralytics import YOLO
import sys
import os

def test_on_image(model_path='best.pt', image_path=None):
    """Test YOLO model on a single image"""
    
    print("="*60)
    print("YOLO Image Detection Test")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = YOLO(model_path)
    print(f"✓ Model loaded!")
    print(f"\nDetectable classes: {', '.join(model.names.values())}")
    
    if image_path is None:
        print("\n❌ No image provided!")
        print("\nUsage: python test_image.py <image_path>")
        print("Example: python test_image.py photo.jpg")
        return
    
    if not os.path.exists(image_path):
        print(f"\n❌ Error: Image '{image_path}' not found!")
        return
    
    # Run detection
    print(f"\nRunning detection on: {image_path}")
    results = model(image_path, conf=0.5)
    
    # Get results
    result = results[0]
    detections = len(result.boxes)
    
    print(f"\n{'='*60}")
    print(f"RESULTS: Found {detections} objects")
    print('='*60)
    
    if detections > 0:
        for i, box in enumerate(result.boxes, 1):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            print(f"{i}. {model.names[cls]}: {conf:.2%} confidence")
            print(f"   Location: [{coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f}, {coords[3]:.0f}]")
    else:
        print("No objects detected. Try:")
        print("  - Using a different image")
        print("  - Lowering confidence threshold")
        print("  - Ensuring image contains detectable objects")
    
    # Save annotated image
    annotated = result.plot()
    output_path = f"detected_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, annotated)
    print(f"\n✓ Annotated image saved to: {output_path}")
    
    # Try to display image
    try:
        cv2.imshow('Detection Results - Press any key to close', annotated)
        print("\n📷 Image window opened. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("\n⚠ Could not display image window, but file was saved.")

def capture_from_camera(model_path='best.pt'):
    """Try to capture a single frame from camera"""
    print("\nAttempting to capture from camera...")
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
    
    if not cap.isOpened():
        print("❌ Could not access camera")
        print("\nPossible solutions:")
        print("1. Check Windows camera permissions:")
        print("   Settings > Privacy > Camera > Allow apps to access camera")
        print("2. Close other apps using the camera (Teams, Zoom, etc.)")
        print("3. Try running as administrator")
        print("4. Use an image file instead: python test_image.py photo.jpg")
        return False
    
    print("✓ Camera opened! Capturing frame...")
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not capture frame")
        return False
    
    # Save captured frame
    cv2.imwrite('captured.jpg', frame)
    print("✓ Frame captured and saved as 'captured.jpg'")
    
    # Run detection on captured frame
    test_on_image(model_path, 'captured.jpg')
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Image path provided
        test_on_image('best.pt', sys.argv[1])
    else:
        # Try to capture from camera
        print("No image provided. Attempting camera capture...")
        if not capture_from_camera('best.pt'):
            print("\n" + "="*60)
            print("Please provide an image file to test:")
            print("  python test_image.py your_photo.jpg")
            print("="*60)

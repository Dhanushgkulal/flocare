"""Diagnose the model issue"""
from ultralytics import YOLO
import cv2

print("="*60)
print("MODEL DIAGNOSIS")
print("="*60)

# Load model
print("\n1. Loading model...")
model = YOLO('best.pt')
print(f"   ✓ Model type: {type(model)}")
print(f"   ✓ Classes: {model.names}")
print(f"   ✓ Number of classes: {len(model.names)}")

# Check model properties
print("\n2. Model properties:")
try:
    print(f"   - Task: {model.task}")
    print(f"   - Model file: {model.ckpt_path if hasattr(model, 'ckpt_path') else 'N/A'}")
except Exception as e:
    print(f"   Error: {e}")

# Open camera and get frame
print("\n3. Capturing frame from camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
ret, frame = cap.read()
cap.release()

if not ret:
    print("   ❌ Failed to capture frame")
    exit(1)

print(f"   ✓ Frame captured: {frame.shape}")
print(f"   ✓ Frame dtype: {frame.dtype}")
print(f"   ✓ Frame range: [{frame.min()}, {frame.max()}]")

# Save frame for inspection
cv2.imwrite('test_frame.jpg', frame)
print(f"   ✓ Frame saved as 'test_frame.jpg'")

# Try detection with different settings
print("\n4. Testing detection with different settings...")

test_configs = [
    {'imgsz': 640, 'conf': 0.25},
    {'imgsz': 416, 'conf': 0.1},
    {'imgsz': 320, 'conf': 0.05},
]

for i, config in enumerate(test_configs, 1):
    print(f"\n   Test {i}: imgsz={config['imgsz']}, conf={config['conf']}")
    try:
        results = model.predict(frame, imgsz=config['imgsz'], conf=config['conf'], verbose=False)
        detections = len(results[0].boxes)
        print(f"   → Detections: {detections}")
        
        if detections > 0:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"      - {model.names[cls]}: {conf:.2%}")
            break
    except Exception as e:
        print(f"   → Error: {e}")

# Try on saved image
print("\n5. Testing on saved image file...")
try:
    results = model.predict('test_frame.jpg', imgsz=640, conf=0.1, verbose=False)
    detections = len(results[0].boxes)
    print(f"   → Detections from file: {detections}")
    
    if detections > 0:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"      - {model.names[cls]}: {conf:.2%}")
except Exception as e:
    print(f"   → Error: {e}")

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)

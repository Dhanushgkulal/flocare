"""Fix camera capture issue"""
import cv2
from ultralytics import YOLO

print("Testing different camera backends...\n")

backends = [
    (cv2.CAP_ANY, "CAP_ANY"),
    (cv2.CAP_DSHOW, "CAP_DSHOW"),
    (cv2.CAP_MSMF, "CAP_MSMF"),
]

best_frame = None
best_backend = None

for backend, name in backends:
    print(f"Trying {name}...")
    cap = cv2.VideoCapture(0, backend)
    
    if not cap.isOpened():
        print(f"  ❌ Could not open camera\n")
        continue
    
    # Skip a few frames to let camera adjust
    for _ in range(5):
        cap.read()
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print(f"  ❌ Could not read frame\n")
        continue
    
    brightness = frame.mean()
    print(f"  ✓ Frame captured: {frame.shape}")
    print(f"  ✓ Mean brightness: {brightness:.2f}\n")
    
    if brightness > 10 and (best_frame is None or brightness > best_frame.mean()):
        best_frame = frame
        best_backend = name

if best_frame is None:
    print("❌ No valid frame captured from any backend!")
    exit(1)

print(f"✓ Best backend: {best_backend}")
print(f"✓ Frame brightness: {best_frame.mean():.2f}")

# Save the frame
cv2.imwrite('good_frame.jpg', best_frame)
print("✓ Saved as 'good_frame.jpg'")

# Test detection
print("\nTesting detection with pre-trained model...")
model = YOLO('yolov8n.pt')
results = model.predict(best_frame, conf=0.25, verbose=False)

detections = len(results[0].boxes)
print(f"\n{'='*60}")
print(f"DETECTIONS: {detections}")
print('='*60)

if detections > 0:
    for i, box in enumerate(results[0].boxes, 1):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"  {i}. {model.names[cls]}: {conf:.2%}")
    
    annotated = results[0].plot()
    cv2.imwrite('detected_fixed.jpg', annotated)
    print(f"\n✓ Saved annotated image as 'detected_fixed.jpg'")
else:
    print("\nStill no detections. The frame might not contain detectable objects.")

print(f"\n✓ Use backend: {best_backend}")

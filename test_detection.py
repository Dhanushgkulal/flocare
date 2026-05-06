"""Quick test to verify model detects objects"""
from ultralytics import YOLO
import cv2
import numpy as np

print("Loading model...")
model = YOLO('best.pt')
print("Model loaded!")

# Open camera
print("Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

print("Camera opened! Reading frame...")
ret, frame = cap.read()

if not ret:
    print("Failed to read frame")
    cap.release()
    exit(1)

print(f"Frame shape: {frame.shape}")

# Run detection
print("Running detection with conf=0.1...")
results = model.predict(frame, conf=0.1, verbose=True)

print(f"\nNumber of detections: {len(results[0].boxes)}")

if len(results[0].boxes) > 0:
    print("\nDetected objects:")
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"  - {model.names[cls]}: {conf:.2%}")
else:
    print("\nNo objects detected!")
    print("Trying with even lower confidence (0.01)...")
    results = model.predict(frame, conf=0.01, verbose=True)
    print(f"Detections with conf=0.01: {len(results[0].boxes)}")

cap.release()
print("\nTest complete!")

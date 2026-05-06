"""
Use a pre-trained YOLO model for comparison
This will download YOLOv8n (nano) model which is pre-trained on COCO dataset
"""
from ultralytics import YOLO
import cv2

print("="*60)
print("TESTING WITH PRE-TRAINED YOLO MODEL")
print("="*60)

print("\nDownloading pre-trained YOLOv8n model...")
model = YOLO('yolov8n.pt')  # This will auto-download if not present
print("✓ Pre-trained model loaded!")

print("\nCapturing frame from camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Failed to capture frame")
    exit(1)

print("✓ Frame captured!")

print("\nRunning detection...")
results = model.predict(frame, conf=0.25, verbose=False)

detections = len(results[0].boxes)
print(f"\n{'='*60}")
print(f"RESULTS: Found {detections} objects")
print('='*60)

if detections > 0:
    print("\nDetected objects:")
    for i, box in enumerate(results[0].boxes, 1):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"  {i}. {model.names[cls]}: {conf:.2%}")
    
    # Save annotated image
    annotated = results[0].plot()
    cv2.imwrite('detected_pretrained.jpg', annotated)
    print(f"\n✓ Annotated image saved as 'detected_pretrained.jpg'")
else:
    print("\nNo objects detected with pre-trained model either.")
    print("This might indicate a camera or lighting issue.")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
print("\nYour 'best.pt' model is UNTRAINED (all metrics are 0).")
print("You need to:")
print("  1. Train the model properly on your dataset, OR")
print("  2. Use a pre-trained model like yolov8n.pt")
print("\nThe pre-trained model above shows how it should work.")
print("="*60)

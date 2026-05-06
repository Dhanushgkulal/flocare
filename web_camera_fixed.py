"""
Fixed Web-based YOLO Camera Detection
Access via browser at http://localhost:5000
"""

from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import time
import numpy as np

app = Flask(__name__)

# Global variables
model = None
detection_active = True
latest_detections = []
fps = 0

def initialize_model():
    """Load YOLO model"""
    global model
    try:
        print("Loading YOLO model from best.pt...")
        model = YOLO('best.pt')
        print(f"✓ Model loaded successfully!")
        print(f"Classes: {model.names}")
        
        # Test the model with a dummy image
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        test_results = model(test_img, verbose=False)
        print("✓ Model test successful!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def generate_frames():
    """Generate video frames with YOLO detection"""
    global latest_detections, fps, detection_active, model
    
    print("\n" + "="*60)
    print("Initializing camera...")
    
    # Try to open camera with DirectShow backend
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not camera.isOpened():
        print("Trying default backend...")
        camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("❌ Failed to open camera!")
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Camera Error", (200, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_img)
        error_frame = buffer.tobytes()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
            time.sleep(1)
    
    # Set camera properties
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("✓ Camera opened successfully!")
    print("Starting detection loop...")
    print("="*60 + "\n")
    
    frame_count = 0
    fps_start = time.time()
    detection_count = 0
    
    try:
        while True:
            success, frame = camera.read()
            
            if not success or frame is None:
                print("Failed to read frame")
                time.sleep(0.1)
                continue
            
            # Run YOLO detection
            if detection_active and model is not None:
                try:
                    # Run inference with very low confidence
                    results = model.predict(frame, conf=0.1, verbose=False)
                    
                    # Get the first result
                    result = results[0]
                    
                    # Update detections list
                    latest_detections = []
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        latest_detections.append({
                            'class': model.names[cls],
                            'confidence': f"{conf:.2%}"
                        })
                    
                    # Draw boxes on frame
                    annotated_frame = result.plot()
                    
                    # Print detections
                    if len(latest_detections) > 0:
                        detection_count += 1
                        if detection_count % 30 == 0:  # Print every 30 frames
                            print(f"Detected: {[d['class'] for d in latest_detections]}")
                    
                except Exception as e:
                    print(f"Detection error: {e}")
                    annotated_frame = frame
                    latest_detections = []
            else:
                annotated_frame = frame
                latest_detections = []
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start)
                fps_start = time.time()
            
            # Add text overlay
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Detections: {len(latest_detections)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except Exception as e:
        print(f"Error in frame generation: {e}")
    finally:
        camera.release()
        print("Camera released")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    """Get current detections as JSON"""
    return jsonify({
        'detections': latest_detections,
        'fps': f"{fps:.1f}",
        'active': detection_active
    })

@app.route('/toggle_detection')
def toggle_detection():
    """Toggle detection on/off"""
    global detection_active
    detection_active = not detection_active
    print(f"Detection {'enabled' if detection_active else 'disabled'}")
    return jsonify({'active': detection_active})

@app.route('/model_info')
def model_info():
    """Get model information"""
    if model:
        return jsonify({
            'classes': list(model.names.values()),
            'num_classes': len(model.names)
        })
    return jsonify({'error': 'Model not loaded'})

if __name__ == '__main__':
    print("="*60)
    print("YOLO Web Camera Detection - FIXED VERSION")
    print("="*60)
    
    # Initialize model
    if not initialize_model():
        print("Failed to initialize model. Exiting.")
        exit(1)
    
    print("\n" + "="*60)
    print("Starting web server...")
    print("Open your browser: http://localhost:5000")
    print("="*60)
    print("\nPress Ctrl+C to stop\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

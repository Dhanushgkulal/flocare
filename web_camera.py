"""
Web-based YOLO Camera Detection
Access via browser at http://localhost:5000
"""

from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import threading
import time

app = Flask(__name__)

# Global variables
camera = None
model = None
detection_active = True
latest_detections = []
fps = 0

def initialize_model():
    """Load YOLO model"""
    global model
    print("Loading YOLO model...")
    model = YOLO('best.pt')
    print(f"✓ Model loaded! Classes: {list(model.names.values())}")

def get_camera():
    """Initialize camera with multiple backend attempts"""
    global camera
    
    if camera is not None and camera.isOpened():
        return camera
    
    # Try different backends with more options
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_ANY, "Any Available"),
        (0, "Default")
    ]
    
    for backend, name in backends:
        try:
            print(f"Trying {name} backend...")
            if backend == 0:
                cam = cv2.VideoCapture(0)
            else:
                cam = cv2.VideoCapture(0, backend)
            
            if cam.isOpened():
                # Set properties before testing
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Test if we can read a frame
                ret, frame = cam.read()
                if ret and frame is not None:
                    print(f"✓ Camera opened with {name} backend")
                    camera = cam
                    return camera
                cam.release()
        except Exception as e:
            print(f"Error with {name}: {e}")
            continue
    
    print("❌ Could not open camera with any backend")
    return None

def generate_frames():
    """Generate video frames with YOLO detection"""
    global latest_detections, fps, detection_active, camera
    
    print("Starting frame generation...")
    cam = get_camera()
    
    if cam is None:
        print("ERROR: Camera could not be initialized!")
        # Return error frame
        error_frame = create_error_frame("Camera not accessible. Check permissions.")
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
            time.sleep(1)
    
    print("Camera initialized successfully, starting detection loop...")
    frame_count = 0
    fps_start = time.time()
    consecutive_failures = 0
    
    while True:
        try:
            success, frame = cam.read()
            
            if not success or frame is None:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    # Try to reinitialize camera
                    print("Too many failures, reinitializing camera...")
                    if camera is not None:
                        camera.release()
                        camera = None
                    cam = get_camera()
                    consecutive_failures = 0
                    if cam is None:
                        error_frame = create_error_frame("Camera connection lost")
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
                        time.sleep(1)
                        continue
                
                error_frame = create_error_frame("Failed to read camera frame")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
                time.sleep(0.1)
                continue
            
            consecutive_failures = 0
            
            if detection_active and model is not None:
                # Run YOLO detection with lower confidence threshold
                results = model(frame, conf=0.25, verbose=False)
                annotated_frame = results[0].plot()
                
                # Debug: print if detections found
                if len(results[0].boxes) > 0:
                    print(f"Detected {len(results[0].boxes)} objects")
                
                # Update detections
                latest_detections = []
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    latest_detections.append({
                        'class': model.names[cls],
                        'confidence': f"{conf:.2%}"
                    })
            else:
                annotated_frame = frame
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start)
                fps_start = time.time()
            
            # Add FPS to frame
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Detections: {len(latest_detections)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        except Exception as e:
            print(f"Error in frame generation: {e}")
            time.sleep(0.1)

def create_error_frame(message):
    """Create an error frame with message"""
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, message, (50, 240),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

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
    print("YOLO Web Camera Detection")
    print("="*60)
    
    # Initialize model
    initialize_model()
    
    print("\n" + "="*60)
    print("Starting web server...")
    print("Open your browser and go to: http://localhost:5000")
    print("="*60)
    print("\nPress Ctrl+C to stop the server\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

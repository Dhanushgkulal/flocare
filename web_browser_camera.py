"""
Web app that uses browser's camera (which works) and sends frames to server for detection
"""
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load pre-trained model (since best.pt is untrained)
print("Loading YOLO model...")
try:
    model = YOLO('yolov8n.pt')  # Use pre-trained model
    print("✓ Pre-trained YOLOv8n model loaded!")
except:
    model = YOLO('best.pt')  # Fallback to your model
    print("⚠ Using best.pt (untrained model)")

@app.route('/')
def index():
    return render_template('camera_browser.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Receive frame from browser and run detection"""
    try:
        # Get image data from request
        data = request.json
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Run YOLO detection
        results = model.predict(frame, conf=0.25, verbose=False)
        
        # Extract detections
        detections = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            
            detections.append({
                'class': model.names[cls],
                'confidence': conf,
                'box': {
                    'x1': xyxy[0],
                    'y1': xyxy[1],
                    'x2': xyxy[2],
                    'y2': xyxy[3]
                }
            })
        
        return jsonify({
            'success': True,
            'detections': detections,
            'count': len(detections)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    return jsonify({
        'classes': list(model.names.values()),
        'num_classes': len(model.names)
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("YOLO Detection with Browser Camera")
    print("="*60)
    print("\nOpen: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

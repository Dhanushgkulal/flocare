# Flocare - YOLO Object Detection System

Real-time object detection system using YOLO (You Only Look Once) with web-based camera interface.

## Features

- 🎥 Real-time object detection from webcam
- 🌐 Web-based interface accessible via browser
- 🎯 Detects 16 classes: person, chair, dining_table, couch, bed, stairs, door, step, bottle, cup, laptop, dog, cat, tv, wheelchair, walker
- 📊 Live statistics (FPS, detection count)
- 🖼️ Visual bounding boxes with confidence scores

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Dhanushgkulal/flocare.git
cd flocare
```

2. Install dependencies:
```bash
pip install -r requirements_web.txt
```

## Usage

### Web-based Detection (Recommended)

This version uses the browser's camera API and sends frames to the server for detection:

```bash
python web_browser_camera.py
```

Then open your browser and go to: **http://localhost:5000**

Click "📷 Start Camera" to begin detection.

### Alternative: Direct Camera Detection

If you want to test with OpenCV camera access:

```bash
python web_camera_fixed.py
```

### Test on Single Image

```bash
python test_image.py your_image.jpg
```

## Files

- `web_browser_camera.py` - Main web application using browser camera (recommended)
- `web_camera_fixed.py` - Alternative using OpenCV camera
- `run_yolo.py` - Command-line YOLO inference script
- `camera_test.py` - Live camera test with OpenCV
- `test_image.py` - Test detection on static images
- `templates/camera_browser.html` - Web interface for browser camera
- `templates/index.html` - Web interface for OpenCV camera

## Model

The system uses YOLOv8n (nano) pre-trained model which will be automatically downloaded on first run.

If you have a custom trained model (`best.pt`), place it in the root directory. Note: The provided `best.pt` needs to be properly trained for detection to work.

## Requirements

- Python 3.8+
- Flask
- Ultralytics YOLO
- OpenCV
- PyTorch
- NumPy

## Troubleshooting

### Camera Not Working
- Ensure no other application is using the camera
- Check Windows camera permissions in Settings > Privacy > Camera
- Try the browser-based version (`web_browser_camera.py`) which has better camera compatibility

### No Detections
- Ensure good lighting
- Make sure objects are clearly visible
- Check that the model file is properly trained
- Try lowering the confidence threshold

### Black Screen
- This is usually a camera driver issue
- Use the browser-based version which bypasses OpenCV camera issues
- Close other applications using the camera (Teams, Zoom, etc.)

## License

MIT License

## Author

Dhanush G Kulal

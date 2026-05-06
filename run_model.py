import torch
import sys

def inspect_model(model_path='best.pt'):
    """Load and inspect the PyTorch model"""
    try:
        # Try loading as a full model
        print(f"Loading model from {model_path}...")
        model = torch.load(model_path, map_location='cpu')
        
        print("\n=== Model Information ===")
        print(f"Model type: {type(model)}")
        
        # Check if it's a YOLO model (common .pt format)
        if hasattr(model, 'names'):
            print("\n=== YOLO Model Detected ===")
            print(f"Classes: {model.names}")
            print(f"Number of classes: {len(model.names)}")
            return 'yolo'
        
        # Check if it's a state dict
        elif isinstance(model, dict):
            print("\n=== Model State Dict ===")
            if 'model' in model:
                print("Contains 'model' key")
                if hasattr(model['model'], 'names'):
                    print(f"Classes: {model['model'].names}")
                    return 'yolo'
            print(f"Keys: {model.keys()}")
            return 'state_dict'
        
        # Check if it's a nn.Module
        elif isinstance(model, torch.nn.Module):
            print("\n=== PyTorch Module ===")
            print(f"Model architecture:\n{model}")
            return 'module'
        
        else:
            print(f"\n=== Unknown Model Format ===")
            print(f"Content: {model}")
            return 'unknown'
            
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying to load with weights_only=True...")
        try:
            model = torch.load(model_path, map_location='cpu', weights_only=True)
            print(f"Loaded successfully with weights_only=True")
            print(f"Type: {type(model)}")
            return 'weights_only'
        except Exception as e2:
            print(f"Error: {e2}")
            return None

def run_yolo_inference(model_path='best.pt', source='0'):
    """Run YOLO model inference"""
    try:
        from ultralytics import YOLO
        
        print(f"\n=== Running YOLO Inference ===")
        model = YOLO(model_path)
        
        # Run inference
        print(f"Running inference on source: {source}")
        results = model(source)
        
        # Display results
        for r in results:
            print(f"\nDetections: {len(r.boxes)}")
            if len(r.boxes) > 0:
                print(r.boxes)
        
        return results
        
    except ImportError:
        print("\nUltralytics YOLO not installed.")
        print("Install with: pip install ultralytics")
        return None
    except Exception as e:
        print(f"Error running inference: {e}")
        return None

if __name__ == "__main__":
    # First inspect the model
    model_type = inspect_model('best.pt')
    
    print("\n" + "="*50)
    print("=== Next Steps ===")
    
    if model_type == 'yolo':
        print("\nThis appears to be a YOLO model.")
        print("\nTo run inference:")
        print("1. On an image: python run_model.py image path/to/image.jpg")
        print("2. On a video: python run_model.py video path/to/video.mp4")
        print("3. On webcam: python run_model.py webcam")
        
        # Check if user provided arguments
        if len(sys.argv) > 2:
            mode = sys.argv[1]
            source = sys.argv[2] if sys.argv[1] != 'webcam' else '0'
            run_yolo_inference('best.pt', source)
        elif len(sys.argv) > 1 and sys.argv[1] == 'webcam':
            run_yolo_inference('best.pt', '0')
    else:
        print("\nModel type unclear. You may need to:")
        print("1. Provide the original training script")
        print("2. Specify the model architecture")
        print("3. Install required dependencies")

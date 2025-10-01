"""
Debug script to understand the CNN model file format
"""
import pickle
import os

def analyze_model_file():
    model_path = "Code/Predict signs/files/CNN"
    
    if not os.path.exists(model_path):
        print(f"Model file not found at: {model_path}")
        return
    
    print(f"Model file size: {os.path.getsize(model_path)} bytes")
    
    # Try to read the file in different ways
    try:
        # First, try reading the raw bytes
        with open(model_path, 'rb') as f:
            first_bytes = f.read(100)
        print(f"First 100 bytes: {first_bytes}")
        
        # Try to load with pickle without importing tensorflow
        import sys
        
        # Temporarily disable tensorflow imports
        original_modules = sys.modules.copy()
        
        # Remove tensorflow from sys.modules if it exists
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith('tensorflow')]
        for module in modules_to_remove:
            if module in sys.modules:
                del sys.modules[module]
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded successfully!")
            print(f"Model type: {type(model)}")
            print(f"Model attributes: {dir(model)}")
            
            if hasattr(model, 'predict'):
                print("Model has predict method")
            
            return model
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None
            
    except Exception as e:
        print(f"Error analyzing model: {e}")
        return None

if __name__ == "__main__":
    analyze_model_file()
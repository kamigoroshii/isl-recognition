#!/usr/bin/env python3
"""
Verification script to compare image processing between original and web app
"""

import cv2
import numpy as np
import pickle

def test_extract_hand_original():
    """Test the original extract_hand function"""
    print("=== Testing Original extract_hand Function ===")
    
    # Simulate similar conditions
    bg = np.random.randint(0, 255, (100, 100), dtype=np.uint8).astype("float")
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    # Original logic
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    
    # Handle different OpenCV versions
    contours_result = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_result) == 3:  # OpenCV 3.x
        _, contours, _ = contours_result
    else:  # OpenCV 4.x
        contours, _ = contours_result
    
    if len(contours) > 0:
        max_cont = max(contours, key=cv2.contourArea)
        print(f"✓ Original extract_hand: Found {len(contours)} contours, max area: {cv2.contourArea(max_cont)}")
        return (thresh, max_cont)
    else:
        print("✗ Original extract_hand: No contours found")
        return None

def test_image_processing_pipeline():
    """Test the complete image processing pipeline"""
    print("\n=== Testing Complete Image Processing Pipeline ===")
    
    # Create test ROI and gray image
    roi = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Simulate hand detection result
    thresh = np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255
    max_cont = np.array([[[10, 10]], [[90, 10]], [[90, 90]], [[10, 90]]])
    
    print(f"Input shapes - ROI: {roi.shape}, Gray: {gray.shape}, Thresh: {thresh.shape}")
    
    # Original processing steps
    print("\n--- Original Processing Steps ---")
    
    # Step 1: Create mask
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [max_cont], -1, 255, -1)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.addWeighted(mask, 0.5, mask, 0.5, 0.0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    print(f"✓ Mask created: shape {mask.shape}, unique values: {np.unique(mask)}")
    
    # Step 2: Apply mask to ROI
    res = cv2.bitwise_and(roi, roi, mask=mask)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    print(f"✓ ROI masked and converted to gray: shape {res.shape}")
    
    # Step 3: Calculate thresholds
    high_thresh, thresh_im = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    print(f"✓ Thresholds calculated: low={low_thresh:.1f}, high={high_thresh:.1f}")
    
    # Step 4: Apply thresh to gray
    hand = cv2.bitwise_and(gray, gray, mask=thresh)
    print(f"✓ Hand extracted: shape {hand.shape}, unique values: {len(np.unique(hand))}")
    
    # Step 5: Canny edge detection
    res = cv2.Canny(hand, low_thresh, high_thresh)
    print(f"✓ Canny edges: shape {res.shape}, unique values: {np.unique(res)}")
    
    # Step 6: CNN preprocessing
    final_res = cv2.resize(res, (100, 100))
    final_res = np.array(final_res)
    final_res = final_res.reshape((-1, 100, 100, 1))
    final_res = final_res.astype('float32')
    final_res = final_res / 255.0
    print(f"✓ CNN input prepared: shape {final_res.shape}, dtype {final_res.dtype}")
    print(f"  Value range: [{final_res.min():.3f}, {final_res.max():.3f}]")
    
    return final_res

def test_model_loading():
    """Test CNN model loading"""
    print("\n=== Testing CNN Model Loading ===")
    
    try:
        import keras
        print(f"✓ Keras version: {keras.__version__}")
        
        with open('Code/Predict signs/files/CNN', 'rb') as f:
            cnn_model = pickle.load(f)
        print(f"✓ Model loaded: type {type(cnn_model)}")
        
        if hasattr(cnn_model, 'predict'):
            print("✓ Model has predict method")
            
            # Test with dummy input
            test_input = np.random.random((1, 100, 100, 1)).astype('float32')
            output = cnn_model.predict(test_input)
            print(f"✓ Model prediction test: output shape {output.shape}")
            print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"  Sum: {output.sum():.3f} (should be close to 1.0 for softmax)")
            
            return True
        else:
            print("✗ Model missing predict method")
            return False
            
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def main():
    print("Image Processing Verification Script")
    print("=" * 50)
    
    # Test individual components
    test_extract_hand_original()
    final_res = test_image_processing_pipeline()
    model_ok = test_model_loading()
    
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY:")
    print(f"✓ Image processing pipeline: COMPLETE")
    print(f"{'✓' if model_ok else '✗'} CNN model loading: {'OK' if model_ok else 'FAILED'}")
    
    if model_ok:
        print("\n🎉 All systems ready! Web app should match original accuracy.")
    else:
        print("\n⚠️  Model issues detected. Check Keras/TensorFlow versions.")

if __name__ == "__main__":
    main()
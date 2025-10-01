from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pickle
import base64
import io
from PIL import Image
import threading
import time
from multilingual_tts import MultilingualTTS
import tensorflow as tf

app = Flask(__name__)

# Create TensorFlow session and graph for thread safety
tf_session = tf.Session()
tf_graph = tf.get_default_graph()

# Global variables
visual_dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',
               10:'a',11:'b',12:'c',13:'d',14:'e',15:'f',16:'g',17:'h',18:'i',
               19:'j',20:'k',21:'l',22:'m',23:'n',24:'o',25:'p',26:'q',27:'r',
               28:'s',29:'t',30:'u',31:'v',32:'w',33:'x',34:'y',35:'z'}

# Load CNN model
cnn_model = None
try:
    # The model was saved with Keras 2.3.1, so we need to import it properly
    import keras
    print(f"Keras version: {keras.__version__}")
    
    # Load the model with proper session handling
    with tf_graph.as_default():
        with tf_session.as_default():
            with open('Code/Predict signs/files/CNN', 'rb') as f:
                cnn_model = pickle.load(f)
    print("CNN model loaded successfully with pickle")
    print(f"Model type: {type(cnn_model)}")
    
    # Test if model has predict method
    if hasattr(cnn_model, 'predict'):
        print("Model has predict method - ready for use")
    else:
        print("Warning: Model doesn't have predict method")
        
except Exception as e:
    print(f"Error loading CNN model: {e}")
    print("CNN model could not be loaded - predictions will not work")
    cnn_model = None

# Initialize TTS
try:
    tts_engine = MultilingualTTS()
    print("TTS engine initialized successfully")
except Exception as e:
    print(f"Error initializing TTS engine: {e}")
    tts_engine = None

class SignPredictor:
    def __init__(self):
        self.bg = None
        self.result_list = []
        self.count = 0
        self.prev_sign = None
        self.aWeight = 0.5
        self.num_frames = 0
        
    def run_avg(self, image):
        """Calculate running average for background subtraction"""
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return
        cv2.accumulateWeighted(image, self.bg, self.aWeight)
    
    def extract_hand(self, image, threshold=15):
        """Extract hand region using background subtraction"""
        if self.bg is None:
            return None
            
        diff = cv2.absdiff(self.bg.astype("uint8"), image)
        thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Handle different OpenCV versions
        contours_result = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours_result) == 3:  # OpenCV 3.x
            _, contours, _ = contours_result
        else:  # OpenCV 4.x
            contours, _ = contours_result
        
        if len(contours) == 0:
            return None
        else:
            # Safe contour area calculation with error handling
            def safe_contour_area(cont):
                try:
                    if cont is not None and len(cont) >= 3:
                        return cv2.contourArea(cont)
                    return 0
                except:
                    return 0
            
            # Filter contours by minimum area first
            valid_contours = [c for c in contours if safe_contour_area(c) > 300]
            
            if len(valid_contours) == 0:
                return None
                
            max_cont = max(valid_contours, key=safe_contour_area)
            return (thresh, max_cont)
    
    def process_frame(self, frame_data, roi_coords):
        """Process a frame and return prediction"""
        try:
            print(f"Processing frame, num_frames: {self.num_frames}")  # Debug
            
            # Decode base64 image
            image_data = base64.b64decode(frame_data.split(',')[1])
            image = Image.open(io.BytesIO(image_data))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Extract ROI
            x, y, w, h = roi_coords
            roi = frame[y:y+h, x:x+w]

            if roi.size == 0:
                print("ROI is empty")  # Debug
                return {"status": "error", "message": "ROI is empty"}

            # Convert to grayscale and blur
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            # Background subtraction setup
            if self.num_frames < 30:
                self.run_avg(gray)
                self.num_frames += 1
                print(f"Calibrating: frame {self.num_frames}/30")  # Debug
                return {"status": "calibrating", "message": "Keep camera still for calibration"}

            # Extract hand
            hand_result = self.extract_hand(gray)
            if hand_result is None:
                print("No hand result from extract_hand")  # Debug
                return {"status": "no_hand", "message": "No hand detected"}

            thresh, max_cont = hand_result

            print(f"Hand contour found with area: {cv2.contourArea(max_cont) if max_cont is not None else 'None'}")  # Debug


            # Robust check for valid contour
            if (
                max_cont is None or
                not isinstance(max_cont, np.ndarray) or
                max_cont.ndim != 3 or
                max_cont.shape[0] < 3 or
                max_cont.shape[1] != 1 or
                max_cont.shape[2] != 2
            ):
                return {"status": "no_hand", "message": "No valid hand contour found"}

            try:
                area = cv2.contourArea(max_cont)
            except Exception as e:
                print(f"Contour area error: {e}")
                return {"status": "no_hand", "message": "Invalid contour for area calculation"}

            # Reduced area threshold for better detection
            if area < 1000:
                print(f"Hand area too small: {area}")  # Debug
                return {"status": "small_hand", "message": "Hand too small or far"}

            # Create mask and extract hand region (following original logic exactly)
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [max_cont], -1, 255, -1)
            mask = cv2.medianBlur(mask, 5)
            mask = cv2.addWeighted(mask, 0.5, mask, 0.5, 0.0)  # Key step from original!
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Step 1: Apply mask to ROI (original colored region), then convert to gray
            res = cv2.bitwise_and(roi, roi, mask=mask)
            res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            
            # Step 2: Apply threshold for Canny preparation  
            high_thresh, thresh_im = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            low_thresh = 0.5 * high_thresh
            
            # Step 3: Apply ORIGINAL thresh to gray image (this is the 'hand' in original)
            hand = cv2.bitwise_and(gray, gray, mask=thresh)
            
            # Step 4: Apply Canny edge detection to the hand (EXACTLY like original)
            res = cv2.Canny(hand, low_thresh, high_thresh)
            
            print(f"Image processing complete - res shape: {res.shape}, unique values: {len(np.unique(res))}")  # Debug

            # CNN prediction (EXACTLY like original)
            if cnn_model is not None:
                # Use 'res' exactly like original code
                final_res = cv2.resize(res, (100, 100))
                final_res = np.array(final_res)
                final_res = final_res.reshape((-1, 100, 100, 1))
                final_res = final_res.astype('float32')  # Remove () like original
                final_res = final_res / 255.0

                try:
                    # Use TensorFlow session for thread-safe prediction
                    with tf_graph.as_default():
                        with tf_session.as_default():
                            output = cnn_model.predict(final_res)
                    prob = np.amax(output)
                    sign_idx = np.argmax(output)
                    predicted_sign = visual_dict[sign_idx]

                    # Use original confidence and counting logic
                    self.count += 1
                    if 10 < self.count <= 50:  # Original range
                        if prob * 100 > 95:  # Original confidence threshold
                            self.result_list.append(predicted_sign)
                            print(f"Added to result_list: {predicted_sign} ({prob:.1%})")
                    elif self.count > 50:  # Original frame count
                        self.count = 0
                        if len(self.result_list):
                            final_prediction = max(set(self.result_list), key=self.result_list.count)
                            self.result_list = []

                            if self.prev_sign != final_prediction:
                                self.prev_sign = final_prediction
                                print(f"FINAL PREDICTION: {final_prediction} - will trigger TTS")  # Debug
                                return {
                                    "status": "success",
                                    "sign": final_prediction,
                                    "confidence": prob * 100,
                                    "speak": True
                                }
                            else:
                                print(f"Same sign as previous: {final_prediction} - no TTS")  # Debug
                                return {
                                    "status": "repeat",
                                    "sign": final_prediction,
                                    "confidence": prob * 100,
                                    "speak": False
                                }

                    return {
                        "status": "detecting",
                        "sign": predicted_sign,
                        "confidence": prob * 100,
                        "count": self.count,
                        "result_list_size": len(self.result_list),
                        "speak": False
                    }
                except Exception as model_error:
                    print(f"Model prediction error: {model_error}")
                    return {"status": "error", "message": f"Model prediction failed: {model_error}"}
            else:
                return {"status": "error", "message": "CNN model not loaded - cannot predict signs"}

        except Exception as e:
            print(f"Processing error: {e}")
            return {"status": "error", "message": str(e)}

# Global predictor instance
predictor = SignPredictor()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/test')
def test():
    """Test endpoint to verify server is working"""
    return jsonify({
        "status": "success", 
        "message": "Server is working",
        "model_loaded": cnn_model is not None,
        "tts_available": tts_engine is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Handle frame prediction"""
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data received"})
            
        frame_data = data.get('frame')
        if not frame_data:
            return jsonify({"status": "error", "message": "No frame data received"})
            
        roi_coords = data.get('roi', [200, 100, 300, 200])  # default ROI
        
        print(f"Processing frame with ROI: {roi_coords}")  # Debug log
        result = predictor.process_frame(frame_data, roi_coords)
        
        if result:
            print(f"Prediction result: {result}")  # Debug log
            return jsonify(result)
        else:
            return jsonify({"status": "error", "message": "Processing failed"})
        
    except Exception as e:
        print(f"Prediction endpoint error: {e}")  # Debug log
        return jsonify({"status": "error", "message": str(e)})

@app.route('/speak', methods=['POST'])
def speak():
    """Handle text-to-speech"""
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data received"})
            
        text = data.get('text', '')
        language = data.get('language', 'English')
        
        print(f"TTS request: '{text}' in {language}")  # Debug log
        
        if not tts_engine:
            return jsonify({"status": "error", "message": "TTS engine not available"})
        
        # Set language and speak in a separate thread
        tts_engine.set_language(language)
        threading.Thread(target=tts_engine.speak, args=(text, True)).start()
        
        return jsonify({"status": "success", "message": f"Speaking '{text}' in {language}"})
        
    except Exception as e:
        print(f"TTS endpoint error: {e}")  # Debug log
        return jsonify({"status": "error", "message": str(e)})

@app.route('/languages')
def get_languages():
    """Get available languages"""
    return jsonify({"languages": tts_engine.get_available_languages()})

@app.route('/reset')
def reset():
    """Reset predictor state"""
    global predictor
    predictor = SignPredictor()
    return jsonify({"status": "success", "message": "Predictor reset"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
# Indian Sign Language Recognition - Web Application

A real-time web-based Indian Sign Language (ISL) recognition system with multilingual text-to-speech output. This application uses computer vision and deep learning to recognize hand signs through your webcam and convert them to speech in multiple Indian languages.

## 🚀 Quick Start Guide

### For Beginners (No Programming Experience Required)

#### Step 1: Install Python
1. Download Python 3.7 from [python.org](https://www.python.org/downloads/release/python-370/)
2. **IMPORTANT**: Check "Add Python to PATH" during installation
3. Restart your computer after installation

#### Step 2: Download This Project
1. Click the green "Code" button on GitHub
2. Select "Download ZIP"
3. Extract to your Desktop or any folder

#### Step 3: Install Requirements
1. Press `Windows Key + R`, type `cmd`, press Enter
2. Type these commands one by one:
   ```cmd
   cd "path\to\your\downloaded\folder"
   pip install tensorflow==1.14.0 keras==2.3.1
   pip install flask opencv-python numpy pillow gtts pygame pyttsx3
   ```

#### Step 4: Run the Application
1. In the same command window, type:
   ```cmd
   python web_app.py
   ```
2. Wait until you see "Running on http://127.0.0.1:5000"
3. Open your web browser
4. Go to: `http://localhost:5000`

#### Step 5: Use the System
1. Click "Start Recognition"
2. Allow camera access
3. Wait 30 seconds for calibration (keep hand OUT of red box)
4. Place hand IN the red box and make signs
5. Select your language and enjoy!

---

## 🎯 Features

- 🤟 **Real-time Sign Recognition**: Recognizes 36 signs (0-9, A-Z)
- 🗣️ **Multilingual TTS**: Supports 8 Indian languages
- 🌐 **Web Interface**: Easy-to-use browser-based interface
- 📱 **Responsive Design**: Works on desktop and mobile browsers
- 🎯 **High Accuracy**: Uses CNN model for precise recognition

## 🌏 Supported Languages

- English
- Hindi (हिंदी)
- Telugu (తెలుగు)
- Tamil (தமிழ்)
- Malayalam (മലയാളം)
- Kannada (ಕನ್ನಡ)
- Bengali (বাংলা)
- Marathi (मराठी)

## 📋 System Requirements

- **Operating System**: Windows 10/11 (recommended)
- **Python**: 3.7.x (required for compatibility)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Camera**: Any USB or built-in webcam
- **Browser**: Chrome, Firefox, Edge (latest versions)
- **Internet**: Required for online TTS (has offline fallback)

## 🔧 Detailed Installation (Advanced Users)

### Prerequisites
```bash
Python 3.7.x
pip (comes with Python)
Webcam (built-in or external)
```

### Environment Setup
```cmd
# Clone or download the repository
git clone https://github.com/username/isl-recognition.git
cd isl-recognition

# Install required packages
pip install tensorflow==1.14.0 keras==2.3.1
pip install flask opencv-python numpy pillow
pip install gtts pygame pyttsx3

# Verify installation
python verify_processing.py
```

### Running the Application
```cmd
# Start the web server
python web_app.py

# Open browser and navigate to
http://localhost:5000
```

## 📖 How to Use

### 1. Initial Setup
- Ensure good lighting and plain background
- Position yourself 2-3 feet from the camera
- Click "Start Recognition"

### 2. Calibration Phase (30 seconds)
- **Keep your hand OUTSIDE the red rectangle**
- Stay still to let the system learn the background
- Wait for "Calibration complete" message

### 3. Recognition Phase
- **Place your hand INSIDE the red rectangle**
- Make clear, steady signs (A-Z, 0-9)
- Hold each sign for 3-5 seconds
- System will display and speak the recognized sign

### 4. Language Selection
- Choose your preferred language from dropdown
- Click "Test Voice" to verify audio
- System speaks predictions in selected language

## 💡 Tips for Best Results

### ✅ Best Practices
- **Lighting**: Bright, even lighting (avoid shadows)
- **Background**: Plain white/light colored wall
- **Hand Position**: Center of red rectangle
- **Sign Quality**: Clear, well-formed signs
- **Stability**: Hold signs steady for 3-5 seconds
- **Distance**: 1-2 feet from camera

### ❌ Common Mistakes
- Moving hands too quickly
- Poor lighting conditions
- Cluttered background
- Multiple hands in frame
- Signs outside the red rectangle
- Impatient (not waiting for calibration)

## 🛠️ Troubleshooting

### Camera Issues
```
Problem: Camera not working
Solution: 
1. Check browser permissions (camera icon in address bar)
2. Close other apps using camera (Skype, Teams, etc.)
3. Try different browser
4. Restart computer
```

### Audio Issues
```
Problem: No voice output
Solution:
1. Check computer volume
2. Click "Test Voice" button
3. Try different language
4. Check internet connection
5. Restart browser
```

### Performance Issues
```
Problem: Slow recognition
Solution:
1. Close unnecessary applications
2. Ensure good lighting
3. Use plain background
4. Check CPU usage in Task Manager
```

### Installation Errors
```
Problem: Package installation fails
Solution:
1. Run Command Prompt as Administrator
2. Update pip: pip install --upgrade pip
3. Install packages one by one
4. Check Python version: python --version
```

## 🔍 Technical Details

### Architecture
- **Backend**: Flask web server
- **Frontend**: HTML5 + JavaScript + Canvas API
- **ML Model**: Convolutional Neural Network (CNN)
- **Computer Vision**: OpenCV for image processing
- **TTS**: Google Text-to-Speech + pyttsx3 fallback

### Model Information
- **Framework**: TensorFlow 1.14.0 + Keras 2.3.1
- **Input Size**: 100x100 grayscale images
- **Output**: 36 classes (A-Z, 0-9)
- **Accuracy**: 95%+ confidence threshold
- **Processing**: Real-time edge detection and feature extraction

### File Structure
```
isl-recognition/
├── web_app.py              # Main Flask application
├── multilingual_tts.py     # Text-to-speech engine
├── verify_processing.py    # System verification script
├── templates/
│   └── index.html          # Web interface
├── Code/
│   └── Predict signs/
│       └── files/
│           └── CNN         # Trained CNN model
└── README.md              # This documentation
```

## 🎯 Performance Metrics

- **Calibration Time**: 30 seconds
- **First Prediction**: 5-10 seconds
- **Subsequent Predictions**: 2-3 seconds
- **Accuracy**: 95%+ under optimal conditions
- **Languages**: 8 Indian languages supported
- **Signs**: 36 total (A-Z, 0-9)

## 🤝 Support

### Getting Help
1. **Check Prerequisites**: Ensure Python 3.7 and packages are installed
2. **Run Verification**: Use `python verify_processing.py`
3. **Check Logs**: Review Command Prompt output for errors
4. **Restart Application**: Close browser and restart web_app.py

### Common Solutions
- **Port 5000 busy**: Close other applications or restart computer
- **Model not loading**: Check TensorFlow/Keras versions
- **Camera permission**: Allow in browser settings
- **Low accuracy**: Improve lighting and background

## 📚 Research Background

This project is based on research in Indian Sign Language recognition using computer vision and machine learning techniques. The system implements:

- Background subtraction for hand segmentation
- SURF feature extraction
- Bag of Visual Words model
- CNN classification
- Multilingual text-to-speech conversion

### Citation
If you use this work in research, please cite:
```
Shagun Katoch, Varsha Singh, Uma Shanker Tiwary, 
"Indian Sign Language recognition system using SURF with SVM and CNN", 
Array, Volume 14, 2022, 100141, ISSN 2590-0056
```

---

**🤟 Created for Indian Sign Language Recognition**  
*Making communication accessible for everyone*

**Version**: Web Application 2.0  
**Last Updated**: October 2025  
**Compatibility**: Windows 10/11, Python 3.7

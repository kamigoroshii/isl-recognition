"""
Test script to verify ISL web app components
"""
import os
import sys

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import flask
        print("✓ Flask imported successfully")
    except ImportError as e:
        print(f"✗ Flask import failed: {e}")
        
    try:
        import cv2
        print(f"✓ OpenCV imported successfully (version: {cv2.__version__})")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        
    try:
        import numpy as np
        print(f"✓ NumPy imported successfully (version: {np.__version__})")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        
    try:
        from PIL import Image
        print("✓ Pillow imported successfully")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        
    try:
        import pickle
        print("✓ Pickle imported successfully")
    except ImportError as e:
        print(f"✗ Pickle import failed: {e}")

def test_model_file():
    """Test if CNN model file exists and can be loaded"""
    print("\nTesting CNN model...")
    
    model_path = "Code/Predict signs/files/CNN"
    if os.path.exists(model_path):
        print("✓ CNN model file exists")
        try:
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"✓ CNN model loaded successfully (type: {type(model)})")
        except Exception as e:
            print(f"✗ Failed to load CNN model: {e}")
    else:
        print(f"✗ CNN model file not found at: {model_path}")

def test_tts_imports():
    """Test TTS-related imports"""
    print("\nTesting TTS imports...")
    
    try:
        import pyttsx3
        print("✓ pyttsx3 imported successfully")
    except ImportError as e:
        print(f"✗ pyttsx3 import failed: {e}")
        
    try:
        from gtts import gTTS
        print("✓ gTTS imported successfully")
    except ImportError as e:
        print(f"✗ gTTS import failed: {e}")
        
    try:
        import pygame
        print("✓ pygame imported successfully")
    except ImportError as e:
        print(f"✗ pygame import failed: {e}")

def test_multilingual_tts():
    """Test multilingual TTS module"""
    print("\nTesting multilingual TTS...")
    
    try:
        from multilingual_tts import MultilingualTTS
        tts = MultilingualTTS()
        languages = tts.get_available_languages()
        print(f"✓ MultilingualTTS loaded successfully")
        print(f"  Available languages: {languages}")
    except Exception as e:
        print(f"✗ MultilingualTTS failed: {e}")

def test_template_file():
    """Test if template file exists"""
    print("\nTesting template file...")
    
    template_path = "templates/index.html"
    if os.path.exists(template_path):
        print("✓ Template file exists")
    else:
        print(f"✗ Template file not found at: {template_path}")

if __name__ == "__main__":
    print("ISL Web App Component Test")
    print("=" * 40)
    
    test_imports()
    test_model_file()
    test_tts_imports()
    test_multilingual_tts()
    test_template_file()
    
    print("\n" + "=" * 40)
    print("Test completed. Please address any ✗ failed tests before running the web app.")
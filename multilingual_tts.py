# Multilingual Text-to-Speech Module for Indian Sign Language Recognition
import pyttsx3
from gtts import gTTS
import os
import pygame
import io
import tempfile
import time

class MultilingualTTS:
    def __init__(self):
        # Language mappings for gTTS (Google Text-to-Speech)
        self.language_codes = {
            'English': 'en',
            'Hindi': 'hi',
            'Telugu': 'te',
            'Tamil': 'ta',
            'Malayalam': 'ml',
            'Kannada': 'kn',
            'Bengali': 'bn',
            'Marathi': 'mr'
        }
        
        # Sign translations for each language
        self.sign_translations = {
            'English': {
                '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
                '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
                'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd', 'e': 'e', 'f': 'f',
                'g': 'g', 'h': 'h', 'i': 'i', 'j': 'j', 'k': 'k', 'l': 'l',
                'm': 'm', 'n': 'n', 'o': 'o', 'p': 'p', 'q': 'q', 'r': 'r',
                's': 's', 't': 't', 'u': 'u', 'v': 'v', 'w': 'w', 'x': 'x',
                'y': 'y', 'z': 'z'
            },
            'Hindi': {
                '0': 'शून्य', '1': 'एक', '2': 'दो', '3': 'तीन', '4': 'चार',
                '5': 'पांच', '6': 'छह', '7': 'सात', '8': 'आठ', '9': 'नौ',
                'a': 'अ', 'b': 'ब', 'c': 'क', 'd': 'द', 'e': 'ए', 'f': 'फ',
                'g': 'ग', 'h': 'ह', 'i': 'इ', 'j': 'ज', 'k': 'क', 'l': 'ल',
                'm': 'म', 'n': 'न', 'o': 'ओ', 'p': 'प', 'q': 'क्यू', 'r': 'र',
                's': 'स', 't': 'त', 'u': 'उ', 'v': 'व', 'w': 'डब्ल्यू', 'x': 'एक्स',
                'y': 'य', 'z': 'ज़'
            },
            'Telugu': {
                '0': 'సున్నా', '1': 'ఒకటి', '2': 'రెండు', '3': 'మూడు', '4': 'నాలుగు',
                '5': 'అయిదు', '6': 'ఆరు', '7': 'ఏడు', '8': 'ఎనిమిది', '9': 'తొమ్మిది',
                'a': 'అ', 'b': 'బ', 'c': 'క', 'd': 'ద', 'e': 'ఎ', 'f': 'ఫ',
                'g': 'గ', 'h': 'హ', 'i': 'ఇ', 'j': 'జ', 'k': 'క', 'l': 'ల',
                'm': 'మ', 'n': 'న', 'o': 'ఓ', 'p': 'ప', 'q': 'క్యూ', 'r': 'ర',
                's': 'స', 't': 'త', 'u': 'ఉ', 'v': 'వ', 'w': 'డబ్ల్యూ', 'x': 'ఎక్స్',
                'y': 'య', 'z': 'జ'
            },
            'Tamil': {
                '0': 'பூஜ்ஜியம்', '1': 'ஒன்று', '2': 'இரண்டு', '3': 'மூன்று', '4': 'நான்கு',
                '5': 'ஐந்து', '6': 'ஆறு', '7': 'ஏழு', '8': 'எட்டு', '9': 'ஒன்பது',
                'a': 'அ', 'b': 'ப', 'c': 'க', 'd': 'த', 'e': 'ஏ', 'f': 'ஃப',
                'g': 'க', 'h': 'ஹ', 'i': 'இ', 'j': 'ஜ', 'k': 'க', 'l': 'ல',
                'm': 'ம', 'n': 'ந', 'o': 'ஓ', 'p': 'ப', 'q': 'க்யூ', 'r': 'ர',
                's': 'ச', 't': 'த', 'u': 'உ', 'v': 'வ', 'w': 'டபிள்யூ', 'x': 'எக்ஸ்',
                'y': 'ய', 'z': 'ஸ'
            },
            'Malayalam': {
                '0': 'പൂജ്യം', '1': 'ഒന്ന്', '2': 'രണ്ട്', '3': 'മൂന്ന്', '4': 'നാല്',
                '5': 'അഞ്ച്', '6': 'ആറ്', '7': 'ഏഴ്', '8': 'എട്ട്', '9': 'ഒമ്പത്',
                'a': 'അ', 'b': 'ബ', 'c': 'ക', 'd': 'ദ', 'e': 'എ', 'f': 'ഫ',
                'g': 'ഗ', 'h': 'ഹ', 'i': 'ഇ', 'j': 'ജ', 'k': 'ക', 'l': 'ല',
                'm': 'മ', 'n': 'ന', 'o': 'ഓ', 'p': 'പ', 'q': 'ക്യൂ', 'r': 'ര',
                's': 'സ', 't': 'ത', 'u': 'ഉ', 'v': 'വ', 'w': 'ഡബ്ല്യൂ', 'x': 'എക്സ്',
                'y': 'യ', 'z': 'സ'
            },
            'Kannada': {
                '0': 'ಶೂನ್ಯ', '1': 'ಒಂದು', '2': 'ಎರಡು', '3': 'ಮೂರು', '4': 'ನಾಲ್ಕು',
                '5': 'ಐದು', '6': 'ಆರು', '7': 'ಏಳು', '8': 'ಎಂಟು', '9': 'ಒಂಬತ್ತು',
                'a': 'ಅ', 'b': 'ಬ', 'c': 'ಕ', 'd': 'ದ', 'e': 'ಎ', 'f': 'ಫ',
                'g': 'ಗ', 'h': 'ಹ', 'i': 'ಇ', 'j': 'ಜ', 'k': 'ಕ', 'l': 'ಲ',
                'm': 'ಮ', 'n': 'ನ', 'o': 'ಓ', 'p': 'ಪ', 'q': 'ಕ್ಯೂ', 'r': 'ರ',
                's': 'ಸ', 't': 'ತ', 'u': 'ಉ', 'v': 'ವ', 'w': 'ಡಬ್ಲ್ಯೂ', 'x': 'ಎಕ್ಸ್',
                'y': 'ಯ', 'z': 'ಜ'
            },
            'Bengali': {
                '0': 'শূন্য', '1': 'এক', '2': 'দুই', '3': 'তিন', '4': 'চার',
                '5': 'পাঁচ', '6': 'ছয়', '7': 'সাত', '8': 'আট', '9': 'নয়',
                'a': 'অ', 'b': 'ব', 'c': 'ক', 'd': 'দ', 'e': 'এ', 'f': 'ফ',
                'g': 'গ', 'h': 'হ', 'i': 'ই', 'j': 'জ', 'k': 'ক', 'l': 'ল',
                'm': 'ম', 'n': 'ন', 'o': 'ও', 'p': 'প', 'q': 'কিউ', 'r': 'র',
                's': 'স', 't': 'ত', 'u': 'উ', 'v': 'ভ', 'w': 'ডাবল্যু', 'x': 'এক্স',
                'y': 'য', 'z': 'জ'
            },
            'Marathi': {
                '0': 'शून्य', '1': 'एक', '2': 'दोन', '3': 'तीन', '4': 'चार',
                '5': 'पाच', '6': 'सहा', '7': 'सात', '8': 'आठ', '9': 'नऊ',
                'a': 'अ', 'b': 'ब', 'c': 'क', 'd': 'द', 'e': 'ए', 'f': 'फ',
                'g': 'ग', 'h': 'ह', 'i': 'इ', 'j': 'ज', 'k': 'क', 'l': 'ल',
                'm': 'म', 'n': 'न', 'o': 'ओ', 'p': 'प', 'q': 'क्यू', 'r': 'र',
                's': 'स', 't': 'त', 'u': 'उ', 'v': 'व', 'w': 'डब्ल्यू', 'x': 'एक्स',
                'y': 'य', 'z': 'झ'
            }
        }
        
        # Initialize pygame mixer for audio playback
        try:
            pygame.mixer.init()
        except:
            print("Warning: pygame mixer initialization failed")
        
        # Fallback to pyttsx3 for offline support
        try:
            self.offline_engine = pyttsx3.init()
            self.offline_engine.setProperty("rate", 100)
        except:
            print("Warning: pyttsx3 initialization failed")
            self.offline_engine = None
        
        # Current selected language
        self.current_language = 'English'
    
    def set_language(self, language):
        """Set the current language for TTS"""
        if language in self.language_codes:
            self.current_language = language
        else:
            print(f"Language {language} not supported. Using English.")
            self.current_language = 'English'
    
    def get_available_languages(self):
        """Return list of available languages"""
        return list(self.language_codes.keys())
    
    def get_translated_sign(self, sign):
        """Get the translated version of a sign in the current language"""
        sign_lower = str(sign).lower()
        if self.current_language in self.sign_translations:
            return self.sign_translations[self.current_language].get(sign_lower, sign)
        return sign
    
    def speak_online(self, text):
        """Use Google TTS for multilingual support (requires internet)"""
        try:
            # Get translated text for the current language
            translated_text = self.get_translated_sign(text)
            lang_code = self.language_codes[self.current_language]
            tts = gTTS(text=translated_text, lang=lang_code, slow=False)
            
            # Create temporary file with better error handling
            temp_dir = tempfile.gettempdir()
            temp_filename = f"tts_audio_{int(time.time() * 1000)}.mp3"
            temp_filepath = os.path.join(temp_dir, temp_filename)
            
            # Save the audio file
            tts.save(temp_filepath)
            
            # Verify file exists before trying to play
            if os.path.exists(temp_filepath):
                # Play the audio
                pygame.mixer.music.load(temp_filepath)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
                # Clean up with better error handling
                try:
                    os.unlink(temp_filepath)
                except:
                    pass  # Ignore cleanup errors
            else:
                raise Exception("Temporary audio file was not created")
                
        except Exception as e:
            print(f"Online TTS failed: {e}. Using offline TTS.")
            self.speak_offline(text)
    
    def speak_offline(self, text):
        """Fallback to pyttsx3 for offline support (English only)"""
        try:
            if self.offline_engine is None:
                print("Offline TTS not available")
                return
                
            # For offline mode, use translated text if available, otherwise original
            translated_text = self.get_translated_sign(text) if self.current_language == 'English' else text
            while self.offline_engine._inLoop:
                pass
            self.offline_engine.say(translated_text)
            self.offline_engine.runAndWait()
        except Exception as e:
            print(f"Offline TTS failed: {e}")
    
    def speak(self, text, use_online=True):
        """Main speak function with online/offline fallback"""
        if use_online and self.current_language != 'English':
            self.speak_online(text)
        else:
            self.speak_offline(text)

# Global instance
multilingual_tts = MultilingualTTS()
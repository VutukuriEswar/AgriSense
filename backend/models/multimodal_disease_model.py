import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import logging
import sys
import json
from pathlib import Path
import os
from google import genai

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.plant_validator import plant_validator
from models.disease_classifier import DISEASE_CLASSES

logger = logging.getLogger(__name__)

# Configure client inside the function where it is used or create a global client

class MultimodalDiseaseModel:
    def __init__(self, model_path=None):
        self.model = None
        self.input_shape = (224, 224)
        self.model_path = model_path or str(Path(__file__).parent.parent / 'saved_models' / 'multimodal_disease.h5')
        self._load_model()

    def _load_model(self):
        if Path(self.model_path).exists():
            logger.info(f"Loading multimodal model from {self.model_path}")
            self.model = keras.models.load_model(self.model_path, compile=False)
        else:
            logger.warning(f"Multimodal model file not found at {self.model_path}.")

    def preprocess_weather(self, weather_data):
        temp = weather_data.get('temp', 25) / 50.0
        humidity = weather_data.get('humidity', 70) / 100.0
        rainfall = weather_data.get('rainfall', 0) / 100.0
        wind = weather_data.get('wind_speed', 5) / 50.0
        return np.array([[temp, humidity, rainfall, wind]])

    def predict_with_gemini(self, image_bytes, weather_data):
        try:
            logger.info("Falling back to Gemini model")
            if not os.environ.get("GEMINI_API_KEY"):
                 return {"is_valid_plant": True, "predicted_class": "Model Missing", "confidence": 0.0, "message": "Multimodal model not loaded and GEMINI_API_KEY missing"}

            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
            
            image_bytes.seek(0)
            img = Image.open(image_bytes).convert('RGB')
            
            prompt = f"""
            You are a plant disease expert. Examine this plant image.
            The current weather data is: {weather_data}
            
            Classify the image into one of the following exact categories:
            {DISEASE_CLASSES}
            
            Respond with ONLY a JSON object in this format, and nothing else:
            {{"predicted_class": "category_name", "confidence": 0.95}}
            Make sure 'predicted_class' is exactly one of the provided categories.
            """
            
            response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=[prompt, img]
            )
            response_text = response.text.strip()
            
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()
                
            result = json.loads(response_text)
            
            return {
                "is_valid_plant": True,
                "predicted_class": result.get("predicted_class", "Unknown"),
                "confidence": float(result.get("confidence", 0.0)),
                "message": "Multimodal analysis complete (via Gemini Fallback)"
            }
        except Exception as e:
            logger.error(f"Gemini fallback error: {e}")
            return {"is_valid_plant": True, "predicted_class": "Error", "confidence": 0.0, "message": "Both local model and Gemini fallback failed"}

    def predict(self, image_bytes, weather_data):
        if not self.model:
            return self.predict_with_gemini(image_bytes, weather_data)

        try:
            is_plant, msg = plant_validator.is_plant(image_bytes)
            image_bytes.seek(0)
            
            if not is_plant:
                return {"is_valid_plant": False, "message": msg, "predicted_class": "Unknown", "confidence": 0.0}

            image = Image.open(image_bytes).convert('RGB').resize(self.input_shape)
            img_arr = np.expand_dims(np.array(image), axis=0) / 255.0

            weather_arr = self.preprocess_weather(weather_data)

            preds = self.model.predict({"image_input": img_arr, "weather_input": weather_arr}, verbose=0)
            
            class_idx = np.argmax(preds[0])
            confidence = float(preds[0][class_idx])
            
            return {
                "is_valid_plant": True,
                "predicted_class": DISEASE_CLASSES[class_idx],
                "confidence": confidence,
                "message": "Multimodal analysis complete"
            }
        except Exception as e:
            logger.error(f"Multimodal error: {e}")
            return self.predict_with_gemini(image_bytes, weather_data)

multimodal_model = MultimodalDiseaseModel()
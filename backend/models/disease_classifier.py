import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from pathlib import Path
import logging
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.plant_validator import plant_validator

logger = logging.getLogger(__name__)

class_names_path = Path(__file__).parent.parent / 'saved_models' / 'class_names.json'
if class_names_path.exists():
    with open(class_names_path, 'r') as f:
        DISEASE_CLASSES = json.load(f)
    logger.info(f"Loaded class names: {DISEASE_CLASSES}")
else:
    logger.warning("class_names.json not found. Using default list. This might cause prediction mismatches!")
    DISEASE_CLASSES = ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy", "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot", "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus", "Tomato_healthy"]

class DiseaseClassifier:
    def __init__(self, model_path=None):
        self.model = None
        self.input_shape = (224, 224)
        self.model_path = model_path or str(Path(__file__).parent.parent / 'saved_models' / 'disease_classifier.h5')
        self._load_or_create_model()

    def _load_or_create_model(self):
        if Path(self.model_path).exists():
            logger.info(f"Loading disease classifier model from {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
        else:
            logger.warning(f"Model file not found at {self.model_path}. Predictions will fail.")

    def preprocess_image(self, image_bytes):
        image = Image.open(image_bytes).convert('RGB')
        image = image.resize(self.input_shape)
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def predict(self, image_bytes):
        try:
            is_plant, validation_msg = plant_validator.is_plant(image_bytes)
            image_bytes.seek(0)
            
            if not is_plant:
                return {
                    "is_valid_plant": False,
                    "message": validation_msg,
                    "predicted_class": "Unknown",
                    "confidence": 0.0,
                    "all_probabilities": []
                }

            processed_image = self.preprocess_image(image_bytes)
            predictions = self.model.predict(processed_image, verbose=0)
            
            probs = predictions[0]
            class_idx = np.argmax(probs)
            confidence = float(probs[class_idx])
            predicted_class = DISEASE_CLASSES[class_idx]
            
            all_probs = [
                {"disease": DISEASE_CLASSES[i], "probability": float(probs[i])}
                for i in range(len(DISEASE_CLASSES))
            ]
            all_probs.sort(key=lambda x: x['probability'], reverse=True)
            
            return {
                "is_valid_plant": True,
                "message": "Analysis complete",
                "predicted_class": predicted_class,
                "confidence": confidence,
                "all_probabilities": all_probs
            }
        except Exception as e:
            logger.error(f"Error during disease prediction: {str(e)}")
            raise

disease_classifier = DiseaseClassifier()
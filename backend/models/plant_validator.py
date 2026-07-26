import logging
import numpy as np
from PIL import Image
from tensorflow import keras
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

class PlantValidator:
    def __init__(self):
        model_path = Path(__file__).parent.parent / 'saved_models' / 'plant_validator.h5'
        self.model = None
        if model_path.exists():
            try:
                self.model = keras.models.load_model(model_path, compile=False)
                logger.info("Plant Validator model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load plant validator: {e}")
        else:
            logger.warning("plant_validator.h5 not found. Validator will accept all images.")

    def is_plant(self, image_bytes):
        if not self.model:
            return True, "Model not loaded, skipping validation."

        try:
            img = Image.open(image_bytes).convert('RGB')
            img = img.resize((224, 224))
            img_arr = np.array(img) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)
            prediction = self.model.predict(img_arr, verbose=0)[0][0]
            
            if prediction > 0.5:
                return True, "Valid plant image detected."
            else:
                return False, "No plant detected. Please upload a clear photo of a plant leaf."
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False, "Error processing image."

plant_validator = PlantValidator()
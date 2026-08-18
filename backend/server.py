import os
import io
import sys
import json
import base64
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import numpy as np
from PIL import Image
from tensorflow import keras
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR.parent))
load_dotenv(ROOT_DIR / '.env')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI(title="AgriSense API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

class PlantValidator:
    def __init__(self):
        model_path = ROOT_DIR / 'saved_models' / 'plant_validator.h5'
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

class_names_path = ROOT_DIR / 'saved_models' / 'class_names.json'
if class_names_path.exists():
    with open(class_names_path, 'r') as f:
        DISEASE_CLASSES = json.load(f)
else:
    DISEASE_CLASSES = ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy", "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot", "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus", "Tomato_healthy"]

NUM_CLASSES = len(DISEASE_CLASSES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def denormalize(tensor_img):
    img = tensor_img.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(img, 0, 1)

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def image_to_base64(img_np):
    img_uint8 = (img_np * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

class ExplainableDiseaseClassifier:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path or str(ROOT_DIR / 'saved_models' / 'lime_shap_gradcam.pth')
        self.shap_explainer = None
        self.lime_explainer = lime_image.LimeImageExplainer()
        self._load_model()
        self.cam = None
        if self.model:
            self.cam = GradCAM(model=self.model, target_layers=[self.model.layer4[-1]])
            
    def _load_model(self):
        if not Path(self.model_path).exists():
            logger.warning(f"PyTorch Model file not found at {self.model_path}.")
            return
            
        try:
            try:
                weights = models.ResNet18_Weights.DEFAULT
                self.model = models.resnet18(weights=weights)
            except AttributeError:
                self.model = models.resnet18(pretrained=True)
                
            state_dict = torch.load(self.model_path, map_location=DEVICE)
            actual_num_classes = state_dict['fc.weight'].shape[0]
            
            self.model.fc = nn.Linear(self.model.fc.in_features, actual_num_classes)
            self.model.load_state_dict(state_dict)
            self.model.to(DEVICE)
            self.model.eval()
            
            global DISEASE_CLASSES
            if len(DISEASE_CLASSES) != actual_num_classes:
                logger.warning(f"Class mismatch! Model expects {actual_num_classes}, but we have {len(DISEASE_CLASSES)}. Using fallback names.")
                DISEASE_CLASSES = [f"Class_{i}" for i in range(actual_num_classes)]
                
            logger.info(f"Successfully loaded PyTorch model from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            self.model = None

    def predict_fn(self, images_np):
        self.model.eval()
        batch = []
        for img in images_np:
            pil_img = Image.fromarray(img.astype(np.uint8))
            batch.append(eval_transform(pil_img))
        batch = torch.stack(batch).to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(self.model(batch), dim=1).cpu().numpy()
        return probs

    def generate_explanations(self, pil_img, pred_idx):
        raw_image_uint8 = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
        input_tensor_single = eval_transform(pil_img).unsqueeze(0).to(DEVICE)

        explanations = {
            "gradcam": None,
            "lime": None,
            "shap": None
        }

        try:
            with torch.enable_grad():
                grayscale_cam = self.cam(input_tensor=input_tensor_single, targets=[ClassifierOutputTarget(pred_idx)])[0]
            rgb_img_float = denormalize(input_tensor_single.squeeze(0))
            cam_vis = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
            explanations["gradcam"] = image_to_base64(cam_vis / 255.0)
        except Exception as e:
            logger.error(f"Grad-CAM error: {e}")

        try:
            lime_exp = self.lime_explainer.explain_instance(
                raw_image_uint8, self.predict_fn, labels=(pred_idx,), top_labels=None,
                hide_color=0, num_samples=200
            )
            lime_temp, lime_mask = lime_exp.get_image_and_mask(
                pred_idx, positive_only=True, num_features=6, hide_rest=False
            )
            lime_vis = mark_boundaries(lime_temp / 255.0, lime_mask)
            explanations["lime"] = image_to_base64(lime_vis)
        except Exception as e:
            logger.error(f"LIME error: {e}")

        try:
            shap_shape = (IMG_SIZE, IMG_SIZE, 3)
            shap_masker_all = shap.maskers.Image("blur(64,64)", shap_shape)
            shap_explainer_all = shap.Explainer(self.predict_fn, shap_masker_all, output_names=DISEASE_CLASSES)
            
            shap_values_single = shap_explainer_all(
                np.expand_dims(raw_image_uint8, axis=0),
                max_evals=100,
                batch_size=10,
                outputs=[pred_idx],
            )
            shap_arr = shap_values_single.values[0]
            if shap_arr.ndim == 4:
                shap_arr = shap_arr[..., 0]
            shap_heatmap = shap_arr.sum(axis=-1)
            abs_max = np.abs(shap_heatmap).max() or 1e-8

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(raw_image_uint8)
            ax.imshow(shap_heatmap, cmap="bwr", alpha=0.6, vmin=-abs_max, vmax=abs_max)
            ax.axis("off")
            explanations["shap"] = fig_to_base64(fig)
        except Exception as e:
            logger.error(f"SHAP error: {e}")

        return explanations

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
                    "all_probabilities": [],
                    "explanations": None
                }

            if not self.model:
                raise Exception("PyTorch model is not loaded.")

            pil_img = Image.open(image_bytes).convert('RGB')
            input_tensor = eval_transform(pil_img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                probs = F.softmax(self.model(input_tensor), dim=1).cpu().numpy()[0]
                
            class_idx = int(np.argmax(probs))
            confidence = float(probs[class_idx])
            predicted_class = DISEASE_CLASSES[class_idx]
            
            all_probs = [
                {"disease": DISEASE_CLASSES[i], "probability": float(probs[i])}
                for i in range(len(DISEASE_CLASSES))
            ]
            all_probs.sort(key=lambda x: x['probability'], reverse=True)
            explanations = self.generate_explanations(pil_img, class_idx)
            raw_resized = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
            explanations["original"] = image_to_base64(raw_resized / 255.0)
            
            return {
                "is_valid_plant": True,
                "message": "Analysis complete with Explanations",
                "predicted_class": predicted_class,
                "confidence": confidence,
                "all_probabilities": all_probs,
                "explanations": explanations
            }
        except Exception as e:
            logger.error(f"Error during disease prediction: {str(e)}")
            raise

explainable_classifier = ExplainableDiseaseClassifier()

api_router = APIRouter(prefix="/api")

@api_router.get("/")
async def root():
    return {"message": "AgriSense API is running", "version": "1.0.0"}

@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AgriSense"}


class DiseaseResult(BaseModel):
    is_valid_plant: bool
    message: str
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    all_probabilities: List[dict] = []
    explanations: Optional[dict] = None

@api_router.post("/predict/disease", response_model=DiseaseResult)
async def predict_disease(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        image_bytes = io.BytesIO(contents)
        
        prediction = explainable_classifier.predict(image_bytes)
        
        await db.disease_predictions.insert_one({
            **prediction,
            "filename": file.filename,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return prediction
    except Exception as e:
        logger.error(f"Error in disease prediction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(api_router)
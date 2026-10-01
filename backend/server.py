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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR.parent))
load_dotenv(ROOT_DIR / '.env')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms, models
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    import shap
    TORCH_AVAILABLE = True
    logger.info("PyTorch found. Full XAI (GradCAM, LIME, SHAP) will be available.")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not found. Running in lightweight TFLite-only mode (no XAI).")

def _load_tflite_interpreter(model_path: Path):
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        from tensorflow import lite as tflite
    interp = tflite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()

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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224

def preprocess_image_numpy(pil_img):
    img = pil_img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    arr  = (arr - mean) / std
    arr  = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, 0)

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def image_to_base64(img_np):
    img_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


class PlantValidator:
    def __init__(self):
        tflite_path = ROOT_DIR / 'saved_models' / 'plant_validator.tflite'
        h5_path     = ROOT_DIR / 'saved_models' / 'plant_validator.h5'

        self.interpreter   = None
        self.input_details = None
        self.output_details = None

        if not tflite_path.exists() and h5_path.exists():
            logger.info("plant_validator.tflite not found — converting from .h5 ...")
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(str(h5_path), compile=False)
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                with open(str(tflite_path), 'wb') as f:
                    f.write(converter.convert())
                logger.info("Conversion complete: plant_validator.tflite saved.")
            except ImportError:
                logger.error("TensorFlow not available — cannot convert .h5. Validator disabled.")
            except Exception as e:
                logger.error(f"Conversion error: {e}")

        if tflite_path.exists():
            try:
                try:
                    import tflite_runtime.interpreter as tflite
                except ImportError:
                    from tensorflow import lite as tflite
                self.interpreter = tflite.Interpreter(model_path=str(tflite_path))
                self.interpreter.allocate_tensors()
                self.input_details  = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                logger.info("PlantValidator loaded (TFLite).")
            except Exception as e:
                logger.error(f"Failed to load plant_validator.tflite: {e}")
        else:
            logger.warning("No plant validator model found — all images will be accepted.")

    def is_plant(self, image_bytes):
        if not self.interpreter:
            return True, "Validator not loaded — skipping."
        try:
            img     = Image.open(image_bytes).convert('RGB').resize((224, 224))
            arr     = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, 0)
            self.interpreter.set_tensor(self.input_details[0]['index'], arr)
            self.interpreter.invoke()
            pred = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
            if pred > 0.5:
                return True, "Valid plant image detected."
            return False, "No plant detected. Please upload a clear photo of a plant leaf."
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False, "Error processing image."

plant_validator = PlantValidator()

class_names_path = ROOT_DIR / 'saved_models' / 'class_names.json'
if class_names_path.exists():
    with open(class_names_path, 'r') as f:
        DISEASE_CLASSES = json.load(f)
else:
    DISEASE_CLASSES = [
        "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
        "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
        "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
        "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
        "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus",
        "Tomato_healthy"
    ]


def _convert_pth_to_tflite(pth_path: Path, onnx_path: Path, tflite_path: Path):
    num_classes = None
    # Step 1: pth -> onnx
    if not onnx_path.exists():
        logger.info("Converting .pth -> .onnx ...")
        try:
            DEVICE = torch.device("cpu")
            try:
                base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except AttributeError:
                base_model = models.resnet18(pretrained=True)
            sd = torch.load(str(pth_path), map_location=DEVICE)
            num_classes = sd['fc.weight'].shape[0]
            base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
            base_model.load_state_dict(sd)
            base_model.eval()
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            torch.onnx.export(
                base_model, dummy, str(onnx_path),
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                opset_version=17, dynamo=False,
            )
            logger.info("pth -> onnx done.")
        except Exception as e:
            logger.error(f"pth->onnx failed: {e}")
            return None

    # Step 2: onnx -> tflite
    if onnx_path.exists() and not tflite_path.exists():
        logger.info("Converting .onnx -> .tflite ...")
        try:
            import tempfile, shutil, glob
            import onnx2tf
            with tempfile.TemporaryDirectory() as tmp:
                onnx2tf.convert(
                    input_onnx_file_path=str(onnx_path),
                    output_folder_path=tmp,
                    non_verbose=True,
                )
                files = glob.glob(tmp + '/*float32*.tflite') or glob.glob(tmp + '/*.tflite')
                if files:
                    shutil.copy(files[0], str(tflite_path))
                    logger.info(f"onnx -> tflite done: {tflite_path.name}")
                else:
                    logger.error("onnx2tf produced no .tflite file!")
                    return None
        except ImportError:
            logger.error("onnx2tf not installed — cannot convert .onnx to .tflite!")
            return None
        except Exception as e:
            logger.error(f"onnx->tflite failed: {e}")
            return None

    return num_classes


class ExplainableDiseaseClassifier:
    def __init__(self):
        global DISEASE_CLASSES
        pth_path    = ROOT_DIR / 'saved_models' / 'lime_shap_gradcam.pth'
        onnx_path   = ROOT_DIR / 'saved_models' / 'lime_shap_gradcam.onnx'
        tflite_path = ROOT_DIR / 'saved_models' / 'lime_shap_gradcam.tflite'

        self.tflite_interpreter = None
        self.tflite_input_details  = None
        self.tflite_output_details = None
        self.torch_model  = None
        self.cam          = None
        self.lime_explainer = None

        # Auto-convert pth -> onnx -> tflite if tflite missing
        if not tflite_path.exists() and pth_path.exists():
            if TORCH_AVAILABLE:
                num_cls = _convert_pth_to_tflite(pth_path, onnx_path, tflite_path)
                if num_cls and num_cls != len(DISEASE_CLASSES):
                    DISEASE_CLASSES = [f"Class_{i}" for i in range(num_cls)]
            else:
                logger.error("torch not available — cannot auto-convert .pth!")

        # Load TFLite interpreter for lightweight inference (Pi + laptop)
        if tflite_path.exists():
            try:
                self.tflite_interpreter, self.tflite_input_details, self.tflite_output_details = \
                    _load_tflite_interpreter(tflite_path)
                logger.info("Disease classifier loaded (TFLite).")
            except Exception as e:
                logger.error(f"Failed to load disease TFLite model: {e}")

        if pth_path.exists() and TORCH_AVAILABLE:
            try:
                DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                try:
                    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                except AttributeError:
                    base_model = models.resnet18(pretrained=True)

                state_dict = torch.load(str(pth_path), map_location=DEVICE)
                num_classes = state_dict['fc.weight'].shape[0]
                base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
                base_model.load_state_dict(state_dict)
                base_model.to(DEVICE)
                base_model.eval()

                self.torch_model = base_model
                self.DEVICE = DEVICE
                self.cam = GradCAM(model=self.torch_model, target_layers=[self.torch_model.layer4[-1]])
                self.lime_explainer = lime_image.LimeImageExplainer()
                logger.info("PyTorch model also loaded — XAI features active.")

                if num_classes != len(DISEASE_CLASSES):
                    DISEASE_CLASSES = [f"Class_{i}" for i in range(num_classes)]
            except Exception as e:
                logger.error(f"Failed to load PyTorch model for XAI: {e}")

    def _torch_transform(self, pil_img):
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return transform(pil_img)

    def _predict_probs_tflite(self, pil_img):
        arr = preprocess_image_numpy(pil_img)
        self.tflite_interpreter.set_tensor(self.tflite_input_details[0]['index'], arr)
        self.tflite_interpreter.invoke()
        raw = self.tflite_interpreter.get_tensor(self.tflite_output_details[0]['index'])[0]
        exp = np.exp(raw - raw.max())
        return exp / exp.sum()

    def _predict_probs_torch(self, pil_img):
        tensor = self._torch_transform(pil_img).unsqueeze(0).to(self.DEVICE)
        with torch.no_grad():
            return F.softmax(self.torch_model(tensor), dim=1).cpu().numpy()[0]

    def _predict_fn_lime(self, images_np):
        batch = torch.stack([
            self._torch_transform(Image.fromarray(img.astype(np.uint8)))
            for img in images_np
        ]).to(self.DEVICE)
        with torch.no_grad():
            return F.softmax(self.torch_model(batch), dim=1).cpu().numpy()

    def _denormalize(self, tensor_img):
        img = tensor_img.cpu().numpy().transpose(1, 2, 0)
        img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        return np.clip(img, 0, 1)

    def _generate_xai(self, pil_img, pred_idx):
        raw_uint8 = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
        input_t   = self._torch_transform(pil_img).unsqueeze(0).to(self.DEVICE)
        result = {"gradcam": None, "lime": None, "shap": None}
        try:
            with torch.enable_grad():
                gc = self.cam(input_tensor=input_t, targets=[ClassifierOutputTarget(pred_idx)])[0]
            cam_vis = show_cam_on_image(self._denormalize(input_t.squeeze(0)), gc, use_rgb=True)
            result["gradcam"] = image_to_base64(cam_vis / 255.0)
        except Exception as e:
            logger.error(f"GradCAM error: {e}")

        try:
            exp = self.lime_explainer.explain_instance(
                raw_uint8, self._predict_fn_lime,
                labels=(pred_idx,), top_labels=None,
                hide_color=0, num_samples=200
            )
            lt, lm = exp.get_image_and_mask(pred_idx, positive_only=True, num_features=6, hide_rest=False)
            result["lime"] = image_to_base64(mark_boundaries(lt / 255.0, lm))
        except Exception as e:
            logger.error(f"LIME error: {e}")

        try:
            masker = shap.maskers.Image("blur(64,64)", (IMG_SIZE, IMG_SIZE, 3))
            explainer = shap.Explainer(self._predict_fn_lime, masker, output_names=DISEASE_CLASSES)
            sv = explainer(np.expand_dims(raw_uint8, 0), max_evals=100, batch_size=10, outputs=[pred_idx])
            arr = sv.values[0]
            if arr.ndim == 4:
                arr = arr[..., 0]
            hm = arr.sum(axis=-1)
            amax = np.abs(hm).max() or 1e-8
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(raw_uint8)
            ax.imshow(hm, cmap="bwr", alpha=0.6, vmin=-amax, vmax=amax)
            ax.axis("off")
            result["shap"] = fig_to_base64(fig)
        except Exception as e:
            logger.error(f"SHAP error: {e}")

        return result

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

            if not self.tflite_interpreter and not self.torch_model:
                raise Exception("No disease classifier model is loaded.")

            pil_img = Image.open(image_bytes).convert('RGB')

            if self.tflite_interpreter:
                probs = self._predict_probs_tflite(pil_img)
            else:
                probs = self._predict_probs_torch(pil_img)

            class_idx       = int(np.argmax(probs))
            confidence      = float(probs[class_idx])
            predicted_class = DISEASE_CLASSES[class_idx]

            all_probs = sorted(
                [{"disease": DISEASE_CLASSES[i], "probability": float(probs[i])} for i in range(len(DISEASE_CLASSES))],
                key=lambda x: x['probability'], reverse=True
            )

            if self.torch_model:
                explanations = self._generate_xai(pil_img, class_idx)
                xai_msg = "Analysis complete with Explanations"
            else:
                explanations = {"gradcam": None, "lime": None, "shap": None}
                xai_msg = "Analysis complete (XAI unavailable — PyTorch not installed)"

            raw_resized = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
            explanations["original"] = image_to_base64(raw_resized / 255.0)

            return {
                "is_valid_plant": True,
                "message": xai_msg,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "all_probabilities": all_probs,
                "explanations": explanations
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
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

        contents    = await file.read()
        image_bytes = io.BytesIO(contents)
        prediction  = explainable_classifier.predict(image_bytes)

        await db.disease_predictions.insert_one({
            **prediction,
            "filename": file.filename,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        return prediction
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(api_router)
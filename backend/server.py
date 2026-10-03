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
        try:
            import ai_edge_litert.interpreter as tflite
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

        self.interpreter    = None
        self.input_details  = None
        self.output_details = None

        self._try_load(tflite_path, h5_path)

    def _try_load(self, tflite_path, h5_path):
        if not tflite_path.exists():
            logger.warning("plant_validator.tflite missing — validator disabled.")
            return
        try:
            self.interpreter, self.input_details, self.output_details = \
                _load_tflite_interpreter(tflite_path)
            logger.info("PlantValidator loaded (TFLite).")
        except Exception as e:
            logger.error(f"plant_validator.tflite failed to load: {e}")
            # The startup converter should have already produced a good file;
            # if it still fails there's nothing more we can do at runtime.
            self.interpreter = None

    def is_plant(self, image_bytes):
        if not self.interpreter:
            return True, "Validator not loaded — skipping."
        try:
            img  = Image.open(image_bytes).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
            arr  = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, 0)
            self.interpreter.set_tensor(self.input_details[0]['index'], arr)
            self.interpreter.invoke()
            pred = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
            if pred > 0.5:
                return True, "Valid plant image detected."
            return False, "No plant detected. Please upload a clear photo of a plant leaf."
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False, "Error processing image."



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


def _convert_pth_to_tflite(pth_path: Path, tflite_path: Path):
    """Convert PyTorch ResNet18 .pth → TFLite via temp ONNX + onnx2tf.
    ai-edge-torch conflicts with tensorflow-cpu, so we use onnx2tf instead.
    Produces float32 ops (FC v9/v10) compatible with tflite_runtime 2.14 on Pi.
    """
    if not TORCH_AVAILABLE:
        logger.error("PyTorch unavailable — cannot convert .pth to .tflite.")
        return None
    import tempfile, shutil as _shutil
    onnx_tmp = Path(tempfile.mktemp(suffix='.onnx'))
    tf2_dir  = Path(tempfile.mkdtemp(prefix='onnx2tf_'))
    try:
        # Step 1: PyTorch → temp ONNX
        logger.info(f"Converting {pth_path.name} → temp ONNX ...")
        DEVICE = torch.device("cpu")
        sd = torch.load(str(pth_path), map_location=DEVICE)
        num_classes = sd['fc.weight'].shape[0]
        try:
            base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except AttributeError:
            base_model = models.resnet18(pretrained=False)
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        base_model.load_state_dict(sd)
        base_model.eval()
        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
        torch.onnx.export(
            base_model, dummy, str(onnx_tmp),
            input_names=["input"], output_names=["output"],
            opset_version=18,  # 12 gets auto-upgraded to 18 by torch anyway
        )
        logger.info("PyTorch → ONNX done.")
        # Step 2: temp ONNX → TFLite via onnx2tf
        try:
            import onnx2tf
            logger.info(f"Converting temp ONNX → {tflite_path.name} via onnx2tf ...")
            onnx2tf.convert(
                input_onnx_file_path=str(onnx_tmp),
                output_folder_path=str(tf2_dir),
                not_use_onnxsim=True,
                verbosity='error',
            )
            candidates = list(tf2_dir.glob('*.tflite'))
            if not candidates:
                raise FileNotFoundError("onnx2tf produced no .tflite file")
            # onnx2tf generates float32 + float16 variants; always use float32 for Pi
            float32_files = [c for c in candidates if 'float32' in c.name]
            chosen = float32_files[0] if float32_files else \
                     next((c for c in candidates if 'float16' not in c.name), candidates[0])
            _shutil.move(str(chosen), str(tflite_path))
            logger.info(f"Conversion complete → {tflite_path.name}")
            return num_classes
        except ImportError:
            logger.error("onnx2tf not installed. Run: pip install onnx onnx2tf")
            return None
    except Exception as e:
        logger.error(f"pth→tflite conversion failed: {e}")
        return None
    finally:
        onnx_tmp.unlink(missing_ok=True)
        _shutil.rmtree(str(tf2_dir), ignore_errors=True)


def _convert_h5_to_tflite(h5_path: Path, tflite_path: Path):
    """Convert Keras .h5 to TFLite compatible with tflite_runtime 2.14 on Pi.
    Plain float32 conversion — avoids FULLY_CONNECTED op v12 which was introduced
    in TF 2.16 and is triggered by Optimize.DEFAULT. FC v9/v10 (float32) works on
    all tflite_runtime builds.
    NOTE: experimental_new_converter was removed in TF 2.16 and must not be set.
    """
    try:
        import tensorflow as tf
        logger.info(f"Converting {h5_path.name} → {tflite_path.name} (float32, Pi-compat) ...")
        model = tf.keras.models.load_model(str(h5_path), compile=False)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # No optimizations — float32 output uses FC v9/v10, not v12
        tflite_bytes = converter.convert()
        tflite_path.write_bytes(tflite_bytes)
        logger.info(f"Conversion complete → {tflite_path.name} ({len(tflite_bytes)/1e6:.1f} MB)")
    except ImportError:
        logger.error("TensorFlow unavailable — cannot convert .h5 to .tflite.")
    except Exception as e:
        logger.error(f"h5→tflite conversion failed: {e}")


def _startup_cleanup_and_convert():
    """Run once: remove ONNX + stale TFLite files, reconvert all models to Pi-compat.
    A sentinel file (.picompat_done) prevents re-conversion on every startup.
    Delete saved_models/.picompat_done to force a fresh reconversion.
    """
    models_dir = ROOT_DIR / 'saved_models'
    sentinel   = models_dir / '.picompat_done'

    # Always remove ONNX files — no longer used for inference
    for onnx_file in models_dir.glob('*.onnx'):
        onnx_file.unlink()
        logger.info(f"Removed {onnx_file.name} (ONNX no longer used)")
    for stale in ['plant_validator_old_op.tflite', 'plant_validator_new_op.tflite']:
        p = models_dir / stale
        if p.exists():
            p.unlink()
            logger.info(f"Removed stale backup: {stale}")

    # Skip reconversion if already done — avoids slow startup on every reload
    if sentinel.exists():
        logger.info("TFLite models already Pi-compatible. Skipping reconversion.")
        return

    # --- plant_validator: delete op-v12 .tflite, reconvert from .h5 ---
    pv_tflite = models_dir / 'plant_validator.tflite'
    pv_h5     = models_dir / 'plant_validator.h5'
    if pv_tflite.exists():
        pv_tflite.unlink()
        logger.info("Deleted plant_validator.tflite (op v12) for reconversion.")
    if pv_h5.exists():
        _convert_h5_to_tflite(pv_h5, pv_tflite)

    # --- multimodal_disease: delete op-v12 .tflite, reconvert from .h5 ---
    mm_tflite = models_dir / 'multimodal_disease.tflite'
    mm_h5     = models_dir / 'multimodal_disease.h5'
    if mm_tflite.exists():
        mm_tflite.unlink()
        logger.info("Deleted multimodal_disease.tflite (op v12) for reconversion.")
    if mm_h5.exists():
        _convert_h5_to_tflite(mm_h5, mm_tflite)

    # --- lime_shap_gradcam: delete old .tflite, reconvert from .pth ---
    dc_tflite = models_dir / 'lime_shap_gradcam.tflite'
    dc_pth    = models_dir / 'lime_shap_gradcam.pth'
    if dc_tflite.exists():
        dc_tflite.unlink()
        logger.info("Deleted lime_shap_gradcam.tflite for reconversion.")
    if dc_pth.exists() and TORCH_AVAILABLE:
        _convert_pth_to_tflite(dc_pth, dc_tflite)

    # Mark all conversions done — skip on next startup
    sentinel.write_text("pi_compat_v1")
    logger.info("All models reconverted. Sentinel written: .picompat_done")


_startup_cleanup_and_convert()
plant_validator = PlantValidator()


class ExplainableDiseaseClassifier:
    def __init__(self):
        global DISEASE_CLASSES
        pth_path    = ROOT_DIR / 'saved_models' / 'lime_shap_gradcam.pth'
        tflite_path = ROOT_DIR / 'saved_models' / 'lime_shap_gradcam.tflite'

        self.tflite_interp  = None
        self.tflite_in      = None
        self.tflite_out     = None
        self.torch_model    = None
        self.cam            = None
        self.lime_explainer = None

        # Primary: load TFLite (converted by _startup_cleanup_and_convert)
        # If missing for some reason, attempt conversion now
        if not tflite_path.exists() and pth_path.exists() and TORCH_AVAILABLE:
            num_cls = _convert_pth_to_tflite(pth_path, tflite_path)
            if num_cls and num_cls != len(DISEASE_CLASSES):
                DISEASE_CLASSES = [f"Class_{i}" for i in range(num_cls)]

        if tflite_path.exists():
            try:
                self.tflite_interp, self.tflite_in, self.tflite_out = \
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
        """Run inference via TFLite interpreter (Pi fallback)."""
        # Detect NHWC vs NCHW from the model's declared input shape
        shape = self.tflite_in[0]['shape']   # e.g. (1,3,224,224) or (1,224,224,3)
        is_nhwc = (shape[-1] == 3)
        img = pil_img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) \
              /  np.array(IMAGENET_STD,  dtype=np.float32)
        if not is_nhwc:
            arr = arr.transpose(2, 0, 1)          # HWC -> CHW
        arr = np.expand_dims(arr, 0)              # add batch dim
        self.tflite_interp.set_tensor(self.tflite_in[0]['index'], arr)
        self.tflite_interp.invoke()
        raw = self.tflite_interp.get_tensor(self.tflite_out[0]['index'])[0]
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

            if not self.tflite_interp and not self.torch_model:
                raise Exception("No disease classifier model is loaded.")

            pil_img = Image.open(image_bytes).convert('RGB')

            if self.tflite_interp:
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
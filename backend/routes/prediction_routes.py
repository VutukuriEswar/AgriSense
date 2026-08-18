from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timezone
import logging
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.explainable_pytorch_model import explainable_classifier
from models.risk_predictor import risk_predictor
from services.weather_service import weather_service
from motor.motor_asyncio import AsyncIOMotorClient
import os

logger = logging.getLogger(__name__)

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

router = APIRouter(prefix="/predict", tags=["prediction"])

class DiseaseResult(BaseModel):
    is_valid_plant: bool
    message: str
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    all_probabilities: List[dict] = []
    explanations: Optional[dict] = None

@router.post("/disease", response_model=DiseaseResult)
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
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timezone
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.weather_service import weather_service
from models.weather_forecaster import weather_forecaster

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/weather", tags=["weather"])

class WeatherResponse(BaseModel):
    current: dict
    forecast: List[dict]
    season: str
    location: dict

class ForecastResponse(BaseModel):
    predictions: List[dict]
    confidence: float
    method: str

@router.get("", response_model=WeatherResponse)
async def get_weather(lat: float = Query(...), lon: float = Query(...)):
    try:
        current = weather_service.get_current_weather(lat, lon)
        forecast = weather_service.get_forecast(lat, lon)
        season = weather_service.determine_season(lat)
        
        return {
            "current": current,
            "forecast": forecast,
            "season": season,
            "location": {"lat": lat, "lon": lon}
        }
    except Exception as e:
        logger.error(f"Error in weather endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forecast-lstm", response_model=ForecastResponse)
async def get_lstm_forecast(lat: float = Query(...), lon: float = Query(...), days: int = Query(7, ge=1, le=14)):
    try:
        current = weather_service.get_current_weather(lat, lon)
        api_forecast = weather_service.get_forecast(lat, lon)
        
        historical_data = [current] + api_forecast
        
        predictions = weather_forecaster.predict_weather(historical_data, days=days)
        
        avg_confidence = sum(p.get('confidence', 0.5) for p in predictions) / len(predictions)
        
        return {
            "predictions": predictions,
            "confidence": avg_confidence,
            "method": "pattern-based" if not weather_forecaster.model else "lstm"
        }
    except Exception as e:
        logger.error(f"Error in LSTM forecast endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timezone
import logging
from motor.motor_asyncio import AsyncIOMotorClient
import os
import math
import requests
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.weather_service import weather_service
from models.environmental_risk import env_risk_predictor
from models.multimodal_disease_model import multimodal_model

logger = logging.getLogger(__name__)

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

router = APIRouter(prefix="/farmland", tags=["farmland"])

class FarmlandLocation(BaseModel):
    lat: float
    lon: float
    name: str = ""
    notes: str = ""

class FarmlandUpdate(BaseModel):
    name: str
    notes: str

class FarmlandResponse(BaseModel):
    id: str
    lat: float
    lon: float
    name: str
    notes: str
    address: str = ""
    timestamp: str

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@router.post("/save")
async def save_farmland_location(location: FarmlandLocation):
    try:
        import uuid
        
        def fetch_address():
            try:
                response = requests.get(
                    "https://nominatim.openstreetmap.org/reverse",
                    params={"lat": location.lat, "lon": location.lon, "format": "json"},
                    headers={"User-Agent": "AgriSense-App/1.0"},
                    timeout=5
                )
                data = response.json()
                return data.get("display_name", "Unknown location")
            except:
                return "Unknown location"

        loop = asyncio.get_event_loop()
        address = await loop.run_in_executor(None, fetch_address)
        
        location_data = {
            "id": str(uuid.uuid4()),
            "lat": location.lat,
            "lon": location.lon,
            "name": location.name,
            "notes": location.notes,
            "address": address,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await db.farmland_locations.insert_one(location_data)
        
        return location_data
    except Exception as e:
        logger.error(f"Error saving farmland location: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{farmland_id}")
async def update_farmland_location(farmland_id: str, update_data: FarmlandUpdate):
    try:
        result = await db.farmland_locations.update_one(
            {"id": farmland_id},
            {"$set": {"name": update_data.name, "notes": update_data.notes}}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Farmland not found")
            
        updated_doc = await db.farmland_locations.find_one({"id": farmland_id}, {"_id": 0})
        return updated_doc
    except Exception as e:
        logger.error(f"Error updating farmland: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{farmland_id}")
async def delete_farmland_location(farmland_id: str):
    try:
        result = await db.farmland_locations.delete_one({"id": farmland_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Farmland not found")
            
        return {"message": "Farmland deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting farmland: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", response_model=List[FarmlandResponse])
async def get_farmland_history():
    try:
        locations = await db.farmland_locations.find({}, {"_id": 0}).to_list(100)
        return locations
    except Exception as e:
        logger.error(f"Error fetching farmland history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search")
async def search_location(query: str = Query(..., min_length=3)):
    try:
        def fetch_osm():
            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "json", "limit": 5, "addressdetails": 1},
                headers={"User-Agent": "AgriSense-App/1.0"},
                timeout=10
            )
            response.raise_for_status()
            return response.json()

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, fetch_osm)
        
        results = []
        for item in data:
            results.append({
                "name": item.get("display_name"),
                "lat": float(item.get("lat")),
                "lon": float(item.get("lon")),
                "type": item.get("type", "place")
            })
        return results
    except Exception as e:
        logger.error(f"Error searching location: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reverse-geocode")
async def reverse_geocode(lat: float, lon: float):
    try:
        def fetch_address():
            response = requests.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={"lat": lat, "lon": lon, "format": "json"},
                headers={"User-Agent": "AgriSense-App/1.0"},
                timeout=5
            )
            response.raise_for_status()
            return response.json()

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, fetch_address)
        
        return {"address": data.get("display_name", "Unknown location")}
    except Exception as e:
        logger.error(f"Error reverse geocoding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analyze-location")
async def analyze_location(lat: float, lon: float):
    try:
        current_weather = weather_service.get_current_weather(lat, lon)
        season = weather_service.determine_season(lat)
        
        environmental_risks = env_risk_predictor.analyze_risks(current_weather, season)
        
        all_locations = await db.farmland_locations.find({}, {"_id": 0}).to_list(1000)
        
        nearby = []
        radius_km = 20.0
        
        for loc in all_locations:
            dist = calculate_distance(lat, lon, loc['lat'], loc['lon'])
            if dist <= radius_km:
                loc['distance_km'] = round(dist, 2)
                nearby.append(loc)
        
        nearby.sort(key=lambda x: x['distance_km'])
        
        return {
            "weather": current_weather,
            "season": season,
            "risks": environmental_risks,
            "ml_prediction": None,
            "nearby_farmlands": nearby,
            "radius_km": radius_km
        }
    except Exception as e:
        logger.error(f"Error analyzing location: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-multimodal")
async def analyze_multimodal(lat: float = Form(...), lon: float = Form(...), file: UploadFile = File(...)):
    try:
        current_weather = weather_service.get_current_weather(lat, lon)
        season = weather_service.determine_season(lat)
        
        weather_input = {**current_weather, "season": season}

        contents = await file.read()
        image_bytes = io.BytesIO(contents)

        ml_prediction = multimodal_model.predict(image_bytes, weather_input)

        environmental_risks = env_risk_predictor.analyze_risks(current_weather, season)
        
        all_locations = await db.farmland_locations.find({}, {"_id": 0}).to_list(1000)
        nearby = []
        radius_km = 20.0
        for loc in all_locations:
            dist = calculate_distance(lat, lon, loc['lat'], loc['lon'])
            if dist <= radius_km:
                loc['distance_km'] = round(dist, 2)
                nearby.append(loc)
        nearby.sort(key=lambda x: x['distance_km'])
        
        return {
            "weather": current_weather,
            "season": season,
            "ml_prediction": ml_prediction,
            "risks": environmental_risks,
            "nearby_farmlands": nearby,
            "radius_km": radius_km
        }
    except Exception as e:
        import io
        logger.error(f"Multimodal analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
import requests
import logging
from datetime import datetime, timezone
import os

logger = logging.getLogger(__name__)

class WeatherService:
    def __init__(self):
        self.api_key = os.environ.get('OPENWEATHER_API_KEY')
        self.base_url = "https://api.openweathermap.org/data/2.5"

    def get_current_weather(self, lat, lon):
        try:
            url = f"{self.base_url}/weather"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                "temp": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "weather": data["weather"][0]["main"],
                "description": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"],
                "clouds": data["clouds"]["all"],
                "rainfall": data.get("rain", {}).get("1h", 0),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching current weather: {str(e)}")
            raise

    def get_forecast(self, lat, lon):
        try:
            url = f"{self.base_url}/forecast"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            daily_forecasts = {}
            for item in data["list"]:
                date = item["dt_txt"].split()[0]
                if date not in daily_forecasts:
                    daily_forecasts[date] = {
                        "date": date,
                        "temp": item["main"]["temp"],
                        "temp_min": item["main"]["temp_min"],
                        "temp_max": item["main"]["temp_max"],
                        "humidity": item["main"]["humidity"],
                        "weather": item["weather"][0]["main"],
                        "description": item["weather"][0]["description"],
                        "rainfall": item.get("rain", {}).get("3h", 0),
                        "wind_speed": item["wind"]["speed"]
                    }
            
            return list(daily_forecasts.values())[:5]
        except Exception as e:
            logger.error(f"Error fetching forecast: {str(e)}")
            raise

    def determine_season(self, lat):
        month = datetime.now().month
        
        if lat >= 0:
            if month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            elif month in [9, 10, 11]:
                return "Monsoon"
            else:
                return "Winter"
        else:
            if month in [3, 4, 5]:
                return "Monsoon"
            elif month in [6, 7, 8]:
                return "Winter"
            elif month in [9, 10, 11]:
                return "Spring"
            else:
                return "Summer"


weather_service = WeatherService()
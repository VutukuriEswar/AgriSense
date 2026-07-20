import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class RiskPredictor:
    def __init__(self):
        self.disease_risk_weights = {
            "Bacterial Leaf Blight": 0.9,
            "Rice Blast": 0.95,
            "Brown Spot": 0.7,
            "Leaf Scald": 0.6,
            "Sheath Blight": 0.85,
            "Tungro Virus": 1.0,
            "Bacterial Leaf Streak": 0.7,
            "Rice Hispa": 0.75,
            "Stem Borer": 0.8,
            "Brown Planthopper": 0.85,
            "Leaf Folder": 0.65,
            "Healthy": 0.0
        }

    def predict_risk(self, disease_prediction, current_weather, forecast_weather, season):
        try:
            disease_score = self._calculate_disease_risk(disease_prediction)
            weather_score = self._calculate_weather_risk(current_weather, forecast_weather, season)
            seasonal_score = self._calculate_seasonal_risk(season, disease_prediction)
            
            combined_risk = (disease_score * 0.5 + weather_score * 0.3 + seasonal_score * 0.2)
            combined_risk = min(1.0, max(0.0, combined_risk))
            
            urgency_level = self._determine_urgency(combined_risk)
            actions = self._generate_recommendations(
                disease_prediction,
                current_weather,
                season,
                urgency_level
            )
            
            return {
                "risk_score": float(combined_risk),
                "urgency_level": urgency_level,
                "recommended_actions": actions,
                "risk_breakdown": {
                    "disease_contribution": float(disease_score),
                    "weather_contribution": float(weather_score),
                    "seasonal_contribution": float(seasonal_score)
                }
            }
        except Exception as e:
            logger.error(f"Error in risk prediction: {str(e)}")
            raise

    def _calculate_disease_risk(self, disease_prediction):
        disease_class = disease_prediction.get("predicted_class", "Healthy")
        confidence = disease_prediction.get("confidence", 0)
        
        base_risk = self.disease_risk_weights.get(disease_class, 0.5)
        return base_risk * confidence

    def _calculate_weather_risk(self, current_weather, forecast_weather, season):
        risk = 0.0
        
        temp = current_weather.get("temp", 25)
        humidity = current_weather.get("humidity", 70)
        rainfall = current_weather.get("rainfall", 0)
        
        if humidity > 85:
            risk += 0.3
        elif humidity > 75:
            risk += 0.15
        
        if temp > 30 and humidity > 80:
            risk += 0.2
        
        if rainfall > 10:
            risk += 0.25
        elif rainfall > 5:
            risk += 0.15
        
        if forecast_weather:
            future_rain_days = sum(1 for day in forecast_weather if day.get('rainfall', 0) > 5)
            if future_rain_days >= 3:
                risk += 0.2
        
        return min(1.0, risk)

    def _calculate_seasonal_risk(self, season, disease_prediction):
        seasonal_risks = {
            "Monsoon": 0.8,
            "Summer": 0.4,
            "Winter": 0.3,
            "Spring": 0.5
        }
        
        base_risk = seasonal_risks.get(season, 0.5)
        
        disease_class = disease_prediction.get("predicted_class", "Healthy")
        if season == "Monsoon" and disease_class in ["Rice Blast", "Bacterial Leaf Blight"]:
            base_risk += 0.15
        
        return base_risk

    def _determine_urgency(self, risk_score):
        if risk_score >= 0.75:
            return "Critical"
        elif risk_score >= 0.5:
            return "High"
        elif risk_score >= 0.25:
            return "Medium"
        else:
            return "Low"

    def _generate_recommendations(self, disease_prediction, weather, season, urgency):
        actions = []
        disease_class = disease_prediction.get("predicted_class", "Healthy")
        
        if disease_class == "Healthy":
            actions.append("Continue regular monitoring and maintenance")
            actions.append("Maintain optimal irrigation schedule")
        else:
            disease_actions = {
                "Rice Blast": [
                    "Apply fungicide (Tricyclazole or Carbendazim) immediately",
                    "Improve field drainage to reduce humidity",
                    "Remove infected plant debris"
                ],
                "Bacterial Leaf Blight": [
                    "Use resistant varieties for next planting",
                    "Apply copper-based bactericide",
                    "Reduce nitrogen fertilizer application"
                ],
                "Tungro Virus": [
                    "Control green leafhopper vectors immediately",
                    "Remove and destroy infected plants",
                    "Use virus-resistant varieties for replanting"
                ],
                "Sheath Blight": [
                    "Apply fungicide (Validamycin)",
                    "Reduce plant density for better air circulation",
                    "Drain water from field temporarily"
                ],
                "Brown Planthopper": [
                    "Apply insecticide (Imidacloprid or Fipronil)",
                    "Maintain shallow water depth in field",
                    "Use light traps to monitor population"
                ]
            }
            
            actions.extend(disease_actions.get(disease_class, [
                f"Consult agricultural expert for {disease_class} treatment",
                "Apply appropriate pesticide based on disease type",
                "Monitor affected area closely"
            ]))
        
        humidity = weather.get("humidity", 70)
        if humidity > 85:
            actions.append("High humidity detected: Improve field ventilation")
        
        rainfall = weather.get("rainfall", 0)
        if rainfall > 10:
            actions.append("Heavy rainfall: Check drainage systems")
        
        if urgency in ["High", "Critical"]:
            actions.insert(0, "URGENT: Immediate action required to prevent crop loss")
        
        return actions


risk_predictor = RiskPredictor()
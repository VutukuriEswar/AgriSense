import logging

logger = logging.getLogger(__name__)

class EnvironmentalRiskPredictor:
    def __init__(self):
        self.disease_profiles = [
            {
                "name": "Bacterial Leaf Blight",
                "category": "Class 1: Bacterial Diseases",
                "risk_conditions": {
                    "temp_min": 25, "temp_max": 34,
                    "humidity_min": 70, "wind_speed_min": 5
                },
                "urgency": "High",
                "precautions": [
                    "Apply copper-based bactericide immediately",
                    "Improve field drainage to reduce moisture",
                    "Remove and destroy infected plant debris",
                    "Use resistant varieties for next planting season"
                ]
            },
            {
                "name": "Bacterial Leaf Streak",
                "category": "Class 1: Bacterial Diseases",
                "risk_conditions": {
                    "temp_min": 28, "temp_max": 35,
                    "humidity_min": 85
                },
                "urgency": "Medium",
                "precautions": [
                    "Apply streptomycin or copper-based sprays",
                    "Avoid overhead irrigation to reduce leaf wetness",
                    "Control weeds to improve air circulation"
                ]
            },
            {
                "name": "Nutrient Deficiency",
                "category": "Class 2: Deficiencies & Abiotic",
                "risk_conditions": {
                    "temp_min": 20, "temp_max": 32,
                    "rainfall_max": 0,
                    "humidity_max": 50
                },
                "urgency": "Medium",
                "precautions": [
                    "Apply balanced NPK fertilizer based on soil test",
                    "Apply foliar spray for quick recovery",
                    "Incorporate organic matter into soil"
                ]
            },
            {
                "name": "Drought Stress",
                "category": "Class 2: Deficiencies & Abiotic",
                "risk_conditions": {
                    "temp_min": 35,
                    "rainfall_max": 0,
                    "humidity_max": 40
                },
                "urgency": "High",
                "precautions": [
                    "Apply irrigation immediately",
                    "Apply mulch to retain soil moisture",
                    "Avoid applying fertilizers during stress periods"
                ]
            },
            {
                "name": "Rice Blast",
                "category": "Class 3: Fungal Diseases",
                "risk_conditions": {
                    "temp_min": 24, "temp_max": 29,
                    "humidity_min": 90, "rainfall_min": 5
                },
                "urgency": "Critical",
                "precautions": [
                    "Apply Tricyclazole or Isoprothiolane fungicide immediately",
                    "Remove and burn infected plant debris",
                    "Avoid excess nitrogen fertilizer application",
                    "Maintain shallow water in the field"
                ]
            },
            {
                "name": "Brown Spot",
                "category": "Class 3: Fungal Diseases",
                "risk_conditions": {
                    "temp_min": 20, "temp_max": 28,
                    "humidity_min": 80
                },
                "urgency": "Medium",
                "precautions": [
                    "Apply Mancozeb or Propiconazole fungicide",
                    "Correct soil nutrient deficiencies (Potassium)",
                    "Use certified disease-free seeds for planting"
                ]
            },
            {
                "name": "Leaf Scald",
                "category": "Class 3: Fungal Diseases",
                "risk_conditions": {
                    "temp_min": 25, "temp_max": 30,
                    "humidity_min": 80
                },
                "urgency": "Medium",
                "precautions": [
                    "Apply appropriate fungicide",
                    "Ensure good drainage and aeration",
                    "Remove weed hosts from field edges"
                ]
            },
            {
                "name": "Sheath Blight",
                "category": "Class 4: Sheath & Stem Diseases",
                "risk_conditions": {
                    "temp_min": 28, "temp_max": 32,
                    "humidity_min": 85
                },
                "urgency": "High",
                "precautions": [
                    "Apply Validamycin or Propiconazole fungicide",
                    "Reduce plant density to improve air circulation",
                    "Drain field temporarily to lower humidity",
                    "Remove infected plant debris after harvest"
                ]
            },
            {
                "name": "Stem Rot",
                "category": "Class 4: Sheath & Stem Diseases",
                "risk_conditions": {
                    "temp_min": 28, "temp_max": 32,
                    "humidity_min": 90,
                    "rainfall_min": 5
                },
                "urgency": "High",
                "precautions": [
                    "Apply validamycin or copper hydroxide",
                    "Improve drainage to prevent waterlogging",
                    "Apply potassium fertilizer to strengthen stems"
                ]
            },
            {
                "name": "Green Leafhopper",
                "category": "Class 5: Pests & Vectors",
                "risk_conditions": {
                    "temp_min": 24, "temp_max": 30,
                    "rainfall_max": 10,
                    "humidity_min": 70
                },
                "urgency": "High",
                "precautions": [
                    "Apply systemic insecticides (Imidacloprid)",
                    "Remove weeds on bunds to eliminate breeding grounds",
                    "Monitor population using sweep nets",
                    "Control to prevent Tungro virus spread"
                ]
            },
            {
                "name": "Brown Planthopper",
                "category": "Class 5: Pests & Vectors",
                "risk_conditions": {
                    "temp_min": 25, "temp_max": 30,
                    "humidity_min": 80
                },
                "urgency": "Critical",
                "precautions": [
                    "Apply Imidacloprid or Buprofezin insecticide",
                    "Maintain shallow water depth in field",
                    "Avoid excessive nitrogen application",
                    "Install light traps to monitor adults"
                ]
            },
            {
                "name": "Stem Borer",
                "category": "Class 5: Pests & Vectors",
                "risk_conditions": {
                    "temp_min": 24, "temp_max": 35,
                    "humidity_min": 60
                },
                "urgency": "High",
                "precautions": [
                    "Apply Cartap hydrochloride or Fipronil",
                    "Install light traps to monitor adult moths",
                    "Remove and burn dead hearts (affected stems)",
                    "Release Trichogramma wasps as biological control"
                ]
            },
            {
                "name": "Rice Hispa",
                "category": "Class 5: Pests & Vectors",
                "risk_conditions": {
                    "temp_min": 25, "temp_max": 30,
                    "humidity_min": 70
                },
                "urgency": "Medium",
                "precautions": [
                    "Apply cypermethrin or neem-based pesticides",
                    "Remove infested leaves manually",
                    "Maintain field sanitation"
                ]
            }
        ]

    def analyze_risks(self, weather_data, season):
        risks = []
        temp = weather_data.get('temp', 0)
        humidity = weather_data.get('humidity', 0)
        rainfall = weather_data.get('rainfall', 0)
        wind = weather_data.get('wind_speed', 0)

        for disease in self.disease_profiles:
            conditions = disease['risk_conditions']
            match_score = 0
            total_factors = 0
            matched_conditions_list = []

            if 'temp_min' in conditions or 'temp_max' in conditions:
                total_factors += 1
                t_min = conditions.get('temp_min', -100)
                t_max = conditions.get('temp_max', 200)
                if t_min <= temp <= t_max:
                    match_score += 1
                    range_str = f"{t_min}°C - {t_max}°C" if t_min != -100 and t_max != 200 else f"{'>' if t_max == 200 else '<'}{t_min if t_min != -100 else t_max}°C"
                    matched_conditions_list.append(f"Temperature: {temp}°C (Ideal: {range_str})")

            if 'humidity_min' in conditions or 'humidity_max' in conditions:
                total_factors += 1
                h_min = conditions.get('humidity_min', 0)
                h_max = conditions.get('humidity_max', 100)
                if h_min <= humidity <= h_max:
                    match_score += 1
                    if h_min > 0 and h_max < 100:
                        range_str = f"{h_min}% - {h_max}%"
                    elif h_max < 100:
                        range_str = f"< {h_max}%"
                    else:
                        range_str = f"> {h_min}%"
                    matched_conditions_list.append(f"Humidity: {humidity}% (Ideal: {range_str})")
            
            if 'rainfall_min' in conditions or 'rainfall_max' in conditions:
                total_factors += 1
                r_min = conditions.get('rainfall_min', -1)
                r_max = conditions.get('rainfall_max', 1000)
                if r_min <= rainfall <= r_max:
                    match_score += 1
                    if r_min > 0 and r_max < 1000:
                        range_str = f"{r_min}mm - {r_max}mm"
                    elif r_max < 1000:
                        range_str = f"< {r_max}mm"
                    else:
                        range_str = f"> {r_min}mm"
                    matched_conditions_list.append(f"Rainfall: {rainfall}mm (Ideal: {range_str})")

            if 'wind_speed_min' in conditions:
                total_factors += 1
                w_min = conditions.get('wind_speed_min', 0)
                if wind >= w_min:
                    match_score += 1
                    matched_conditions_list.append(f"Wind Speed: {wind} m/s (Min required: {w_min} m/s)")

            if total_factors > 0 and match_score == total_factors:
                risks.append({
                    "disease": disease['name'],
                    "category": disease['category'],
                    "type": disease['category'].split(':')[0],
                    "urgency": disease['urgency'],
                    "precautions": disease['precautions'],
                    "matched_conditions": matched_conditions_list,
                    "confidence": round((match_score / total_factors) * 100, 2)
                })
        
        urgency_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        return sorted(risks, key=lambda x: urgency_order.get(x['urgency'], 4))

env_risk_predictor = EnvironmentalRiskPredictor()
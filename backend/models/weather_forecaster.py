import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class WeatherForecaster:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path
        self.sequence_length = 7
        
    def predict_weather(self, historical_data, days=7):
        try:
            if len(historical_data) < self.sequence_length:
                return self._pattern_based_prediction(historical_data, days)
            
            if self.model:
                return self._lstm_prediction(historical_data, days)
            else:
                return self._pattern_based_prediction(historical_data, days)
        except Exception as e:
            logger.error(f"Error in weather prediction: {str(e)}")
            return self._pattern_based_prediction(historical_data, days)

    def _pattern_based_prediction(self, historical_data, days):
        if not historical_data:
            return self._default_forecast(days)
        
        recent = historical_data[-3:] if len(historical_data) >= 3 else historical_data
        
        avg_temp = np.mean([d.get('temp', 25) for d in recent])
        avg_humidity = np.mean([d.get('humidity', 70) for d in recent])
        avg_rainfall = np.mean([d.get('rainfall', 0) for d in recent])
        
        temp_trend = np.random.uniform(-1, 1)
        
        predictions = []
        base_date = datetime.now()
        
        for i in range(days):
            predictions.append({
                "date": (base_date + timedelta(days=i+1)).isoformat(),
                "temp": float(avg_temp + temp_trend * i + np.random.uniform(-2, 2)),
                "humidity": float(max(30, min(100, avg_humidity + np.random.uniform(-5, 5)))),
                "rainfall": float(max(0, avg_rainfall + np.random.uniform(-2, 2))),
                "confidence": 0.65
            })
        
        return predictions

    def _lstm_prediction(self, historical_data, days):
        logger.info("LSTM prediction not yet implemented, using pattern-based")
        return self._pattern_based_prediction(historical_data, days)

    def _default_forecast(self, days):
        predictions = []
        base_date = datetime.now()
        
        for i in range(days):
            predictions.append({
                "date": (base_date + timedelta(days=i+1)).isoformat(),
                "temp": 25.0 + np.random.uniform(-3, 3),
                "humidity": 70.0 + np.random.uniform(-10, 10),
                "rainfall": np.random.uniform(0, 5),
                "confidence": 0.5
            })
        
        return predictions


weather_forecaster = WeatherForecaster()
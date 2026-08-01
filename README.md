## What is AgriSense?

AgriSense is an advanced AI-powered web application designed to help farmers and agriculturists accurately detect plant diseases, validate crops, and predict agricultural risks. By leveraging state-of-the-art Deep Learning models and Large Language Models (LLMs), it provides deep insights into crop health, combining visual data with real-time weather metrics to deliver actionable intelligence.

## Key Features

🤖 **AI-Powered Disease Classification**
- Utilizes specialized deep learning models (TensorFlow/Keras) for plant disease classification (Tomato, Potato, Pepper, etc.).
- Includes a robust Plant Validator to ensure uploaded images are actual plants before analysis.
- Smart multimodal fallback system using **Gemini 1.5 Flash** for extended disease identification and insights.

🌦️ **Weather Integration & Forecasting**
- Integrates with the **OpenWeather API** to fetch real-time and 5-day forecast data (Temperature, Humidity, Rainfall).
- Uses historical and pattern-based modeling (LSTM-ready) to predict localized weather trends for farmland.

⚠️ **Comprehensive Risk Prediction**
- Calculates dynamic risk scores by combining disease predictions, current weather, and seasonal data.
- Provides varying urgency levels (Critical, High, Moderate, Low) with actionable, customized recommendations.

🌍 **Interactive Dashboard & Farmland Management**
- Track agricultural risk history and statistics.
- Interactive maps using Leaflet to manage and monitor multiple farmlands geographically.
- Clean, responsive UI built with modern React and Shadcn UI components.

## Tech Stack

**Backend:**
- **FastAPI** (High-performance Python web framework)
- **TensorFlow & Keras** (Deep Learning models for image classification)
- **Google Generative AI** (Gemini 1.5 Flash integration)
- **MongoDB** (Motor for async database operations)
- **OpenWeather API** (Real-time weather data)

**Frontend:**
- **React** (UI components)
- **React Leaflet** (Geographic maps)
- **Recharts** (Data visualization)
- **Tailwind CSS** (Utility-first styling)
- **Shadcn UI** (Component library)

## Quick Start Guide

### Prerequisites
- Python 3.9+
- Node.js & npm/yarn
- MongoDB (local or cloud instance)
- Gemini API Key (for multimodal fallback)
- OpenWeather API Key (for weather forecasting)

### Installation Steps

1. **Clone the repository**

2. **Set up virtual environment (Backend)**
```bash
python -m venv venv
venv\Scripts\activate  # On Linux: source venv/bin/activate
```

3. **Install backend dependencies**
```bash
cd backend
pip install -r requirements.txt
```

4. **Environment Variables**
Create a `.env` file in the `backend` directory and add your keys:
```env
MONGO_URL="mongodb://localhost:27017"
DB_NAME="agrisense_db"
CORS_ORIGINS="*"
OPENWEATHER_API_KEY="your_openweather_api_key"
GEMINI_API_KEY="your_gemini_api_key"
```

5. **Run backend server**
```bash
uvicorn server:app --reload
```

6. **Open a new terminal and install frontend dependencies**
```bash
cd frontend
yarn install
```

7. **Start frontend application**
```bash
yarn start
```

## API Endpoints

**Prediction:**
- `POST /api/predict/disease` - Analyze plant image and identify diseases.

**Weather:**
- `GET /api/weather` - Get current weather and 5-day forecast for coordinates.
- `GET /api/weather/forecast-lstm` - Get advanced pattern-based weather forecasts.

**Farmland:**
- `GET /api/farmland` - Retrieve registered farmlands.
- `POST /api/farmland` - Add a new farmland to monitor.

## Configuration Details

**Model Setup:**
- The application looks for pre-trained models in `backend/saved_models`.
- Expected files include:
  - `disease_classifier.h5`
  - `plant_validator.h5`
  - `multimodal_disease.h5`
  - `class_names.json`
- If local models are missing or fail, it automatically falls back to **Gemini 1.5 Flash** for robust prediction.

## Model Training

For training the custom AI models, we used the following resources:

**Training Notebooks:**
- [AgriSense Multimodal Disease Notebook](https://www.kaggle.com/code/eswarvutukuri/agrisense-multimodaldisease)
- [AgriSense Disease Classifier Notebook](https://www.kaggle.com/code/eswarvutukuri/agrisense-diseaseclassifier)
- [AgriSense Plant Validator Notebook](https://www.kaggle.com/code/eswarvutukuri/agrisense-plantvalidator)

**Post-Training Setup:**
Once the models are trained, you will obtain `.h5` files. To use them in the application:
1. Download these model artifacts.
2. Create a folder named `saved_models` inside the `backend` directory (if it doesn't exist).
3. Place the `.h5` files and the `class_names.json` file into `backend/saved_models/`.

## License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

© 2026 Eswar Vutukuri

## Acknowledgments

This project was developed under the guidance of **Mr. Ch. Srihari**, **VIT-AP University**. We sincerely thank our professor for their invaluable guidance, mentorship, and continuous support throughout the development of this project.

Thanks to Google for accessible Gemini APIs, OpenWeather for providing seamless weather data APIs, MongoDB for storing data, Kaggle for allowing us train the models required for the project, and the creators of TensorFlow/Keras for empowering the core prediction engines.

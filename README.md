## What is AgriSense?

AgriSense is an advanced AI-powered web application designed to help farmers and agriculturists accurately detect plant diseases and validate crops. By leveraging state-of-the-art Deep Learning models and Explainable AI (XAI) techniques, it provides deep, transparent insights into crop health, delivering actionable intelligence that farmers can trust.

## Key Features

🤖 **AI-Powered Disease Classification with XAI**
- Utilizes specialized deep learning models (PyTorch ResNet) for highly accurate plant disease classification across multiple crops (Tomato, Potato, Pepper, etc.).
- Integrates Explainable AI techniques including **Grad-CAM**, **LIME**, and **SHAP** to visually explain the model's predictions, showing exactly which parts of a leaf influenced the diagnosis.
- Includes a robust Plant Validator (TensorFlow/Keras) to ensure uploaded images are actual plants before analysis.

🌍 **Interactive Dashboard**
- Track agricultural risk history and predictions.
- Clean, responsive UI built with modern React and Shadcn UI components.

## Tech Stack

**Backend:**
- **FastAPI** (High-performance Python web framework)
- **PyTorch** (Core deep learning engine for disease classification and Explainable AI)
- **TensorFlow & Keras** (Deep Learning models for initial plant image validation)
- **MongoDB** (Motor for async database operations)

**Frontend:**
- **React** (UI components)
- **Recharts** (Data visualization)
- **Tailwind CSS** (Utility-first styling)
- **Shadcn UI** (Component library)

## Quick Start Guide

### Prerequisites
- Python 3.9+
- Node.js & npm/yarn
- MongoDB (local or cloud instance)

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
- `POST /api/predict/disease` - Analyze plant image, identify diseases, and generate visual explanations (Grad-CAM, LIME, SHAP).

## Configuration Details

**Model Setup:**
- The application looks for pre-trained models in `backend/saved_models`.
- Expected files include:
  - `lime_shap_gradcam.pth` (PyTorch model for classification and XAI)
  - `plant_validator.h5` (TensorFlow model to validate if image is a plant)
  - `class_names.json` (Mapping of class indices to disease names)

## Model Training

For training the custom AI models, we used the following resources:

**Training Notebooks:**
- [AgriSense Disease Classifier Notebook](https://www.kaggle.com/code/eswarvutukuri/agrisense-diseaseclassifier)
- [AgriSense Plant Validator Notebook](https://www.kaggle.com/code/eswarvutukuri/agrisense-plantvalidator)

**Post-Training Setup:**
Once the models are trained, you will obtain the model artifacts. To use them in the application:
1. Download these model artifacts.
2. Create a folder named `saved_models` inside the `backend` directory (if it doesn't exist).
3. Place the `.pth` and `.h5` files, along with the `class_names.json` file, into `backend/saved_models/`.

## License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

© 2026 Eswar Vutukuri

## Acknowledgments

This project was developed under the guidance of **Mr. Chengalva Srihari**, **VIT-AP University**. We sincerely thank our professor for their invaluable guidance, mentorship, and continuous support throughout the development of this project.

Thanks to MongoDB for storing data, Kaggle for allowing us to train the models required for the project, and the creators of PyTorch and TensorFlow/Keras for empowering the core prediction and explainability engines.

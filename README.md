# 🧠 NeuroPredict AI — Alzheimer's Disease Prediction System

> **Advanced AI Clinical Decision Support System** powered by an ensemble of 8 Machine Learning & Deep Learning models for early Alzheimer's Disease detection and risk stratification.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Machine Learning Models](#machine-learning-models)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup (FastAPI)](#backend-setup-fastapi)
  - [Frontend Setup (Vite / React)](#frontend-setup-vite--react)
- [API Reference](#api-reference)
- [Dataset & Clinical Parameters](#dataset--clinical-parameters)
- [License](#license)

---

## 🌟 Overview

**NeuroPredict AI** is a clinical-grade decision support platform designed to assist healthcare professionals in early detection and risk evaluation of Alzheimer's Disease. By analyzing patient demographics, cognitive assessments (MMSE, MoCA), behavioral traits, lifestyle factors, and medical history, the platform delivers real-time diagnostic predictions powered by trained machine learning algorithms.

---

## ✨ Key Features

- 🧠 **8-Model AI Engine**: Evaluates patient data across multiple ML algorithms (ANN, XGBoost, Random Forest, LightGBM, SVM, Gradient Boosting, Logistic Regression, Decision Tree).
- 📊 **Model Benchmarking & Metrics**: Interactive performance dashboard tracking Accuracy, Precision, Recall, F1-Score, MCC (Matthews Correlation Coefficient), and ROC-AUC.
- 📋 **Patient Risk Assessment**: Comprehensive clinical intake form covering cognitive scores, physical metrics, behavioral indicators, and medical history.
- 🔬 **Real-time Diagnostic Insights**: Instant prediction output with confidence probabilities, severity classification, and clinical recommendations.
- 📁 **Batch CSV Upload**: Process multiple patient records concurrently via bulk CSV imports.
- 🔄 **Custom Dataset Retraining**: Re-train models dynamically by uploading custom dataset files via the API.
- 📱 **Responsive & Mobile View**: Doctor-facing interface compatible across desktop and mobile devices.

---

## 🏗️ System Architecture

```
Deep learning/
├── backend/                  # FastAPI Backend Server & ML Engine
│   ├── server.py             # REST API endpoints (FastAPI)
│   └── ml_engine.py          # ML pipeline (data preprocessing, training, inference)
├── frontend/                 # React + Vite Web Application
│   ├── src/
│   │   ├── components/       # Dashboard, Diagnostic, Intake & Navbar components
│   │   ├── App.jsx           # Main React component
│   │   └── index.css         # Styling system
│   ├── package.json
│   └── vite.config.js
├── index.html                # Standalone Vanilla Web Application UI
├── styles.css                # Standalone UI Stylesheet
├── app.js                    # Standalone UI Client Logic
└── .gitignore                # Repository gitignore rules
```

---

## 🤖 Machine Learning Models

The ML engine evaluates and compares 8 core models:

| Model | Model Type | Key Advantage |
| :--- | :--- | :--- |
| **ANN (Artificial Neural Network)** | Deep Learning | Captures non-linear feature interactions |
| **XGBoost** | Gradient Boosting | Superior accuracy on structured tabular clinical data |
| **LightGBM** | Gradient Boosting | Fast training & efficient handling of feature splits |
| **Random Forest** | Ensemble Trees | Robust against overfitting with high interpretability |
| **Gradient Boosting** | Ensemble Trees | Iterative boosting for strong predictive power |
| **Support Vector Machine (SVM)** | Kernel Classifier | Effective in high-dimensional feature spaces |
| **Logistic Regression** | Linear Model | Baseline clinical odds-ratio interpretability |
| **Decision Tree** | Tree Classifier | Transparent rule-based decision trees |

---

## 🛠️ Tech Stack

### **Backend**
- **Language**: Python 3.10+
- **Framework**: FastAPI, Uvicorn
- **ML / Data Libraries**: `scikit-learn`, `xgboost`, `lightgbm`, `numpy`, `pandas`

### **Frontend**
- **React Application**: React 19, Vite, Lucide Icons, Tailwind / Custom CSS
- **Vanilla Application**: HTML5, Vanilla CSS3, JavaScript (ES6+)

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.9+** installed
- **Node.js 18+** & `npm` installed

---

### Backend Setup (FastAPI)

1. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn scikit-learn xgboost lightgbm pandas numpy
   ```

4. **Start the FastAPI server**:
   ```bash
   python server.py
   # OR
   uvicorn server:app --reload --host 0.0.0.0 --port 8000
   ```
   The backend API will be available at `http://localhost:8000`. API docs are available at `http://localhost:8000/docs`.

---

### Frontend Setup (Vite / React)

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install npm dependencies**:
   ```bash
   npm install
   ```

3. **Run the development server**:
   ```bash
   npm run dev
   ```
   Open `http://localhost:5173` in your web browser.

---

## 📡 API Reference

### Health Check
- `GET /api/health` — Returns system status, model version, and status of trained models.

### Metrics & Model Benchmarks
- `GET /api/models/metrics` — Returns comparative performance metrics (Accuracy, Precision, Recall, F1, MCC, ROC-AUC) across all 8 models.

### Patient Prediction
- `POST /api/predict` — Predicts Alzheimer's risk for a single patient record.
  **Body**:
  ```json
  {
    "data": {
      "Age": 72,
      "Gender": 1,
      "MMSE": 18,
      "FunctionalAssessment": 4.5,
      "MemoryComplaints": 1,
      "BehavioralProblems": 1,
      "ADL": 3.0
    },
    "model_name": "ANN (Neural Network)"
  }
  ```

### Bulk CSV Prediction
- `POST /api/predict/csv` — Accepts a CSV file payload and returns predictions for all patient entries.

### Retrain Models
- `POST /api/train` — Upload a new dataset CSV file to retrain all 8 models.

---

## 🩺 Dataset & Clinical Parameters

The prediction system uses key clinical metrics including:

- **Demographics**: Age, Gender, Ethnicity, Education Level
- **Cognitive Assessments**: MMSE (Mini-Mental State Examination), Functional Assessment score
- **Clinical History**: Family history of Alzheimer's, Cardiovascular disease, Diabetes, Head injury history
- **Behavioral & Lifestyle**: Memory complaints, Behavioral problems, ADL (Activities of Daily Living) scores, Sleep quality, Physical activity level

---

## 📄 License

This project is released under the [MIT License](LICENSE).

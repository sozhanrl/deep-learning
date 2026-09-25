# 🧠 NeuroPredict AI — Tabular Deep Learning Alzheimer's Detection Suite

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React 19](https://img.shields.io/badge/React-19.0-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-5.0+-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev)
[![Captum XAI](https://img.shields.io/badge/Captum-Explainable_AI-FF6F00?style=for-the-badge)](https://captum.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> **NeuroPredict AI** is a clinical-grade decision support platform powered by a suite of **8 Tabular Deep Learning architectures** implemented in PyTorch, coupled with **Captum neural feature attributions (Explainable AI)**, **SMOTE class rebalancing**, and a **dual-frontend architecture** (React 19 + Vite dashboard & standalone zero-dependency web app) for early Alzheimer's Disease detection and risk stratification.

---

## 📋 Table of Contents
- [🌟 Overview & Clinical Motivation](#-overview--clinical-motivation)
- [✨ Key Features](#-key-features)
- [🏗️ System Architecture & Workflow](#️-system-architecture--workflow)
- [🤖 8 Tabular Deep Learning Architectures](#-8-tabular-deep-learning-architectures)
- [🔬 Explainable AI (XAI) & Interpretability](#-explainable-ai-xai--interpretability)
- [🩺 Clinical Parameters (32 Features)](#-clinical-parameters-32-features)
- [📊 Model Benchmarking & Performance](#-model-benchmarking--performance)
- [💻 User Interfaces](#-user-interfaces)
- [📡 REST API Reference](#-rest-api-reference)
- [🚀 Quickstart & Installation](#-quickstart--installation)
  - [Prerequisites](#prerequisites)
  - [Backend Setup (FastAPI + PyTorch)](#1-backend-setup-fastapi--pytorch)
  - [Frontend Setup (React + Vite)](#2-frontend-setup-react--vite)
  - [Standalone Interface (Zero-Node Option)](#3-standalone-interface-zero-node-option)
- [🧪 Running Unit Tests](#-running-unit-tests)
- [📁 Project Structure](#-project-structure)
- [🛡️ Security & Privacy Considerations](#️-security--privacy-considerations)
- [📄 License & Citation](#-license--citation)

---

## 🌟 Overview & Clinical Motivation

Alzheimer's Disease (AD) is a progressive neurodegenerative disorder and the leading cause of dementia globally. Clinical intervention is significantly more effective when administered during the early stages of cognitive impairment. However, diagnosing AD traditionally involves invasive, expensive, or late-stage diagnostic procedures such as lumbar punctures or amyloid PET scans.

**NeuroPredict AI** bridges this gap by leveraging high-dimensional, non-invasive clinical profiles:
- Cognitive examinations (MMSE, Functional Assessment, ADL autonomy)
- Demographic & genetic predisposition (Family History, Age, Gender)
- Cardiovascular and metabolic biomarkers (Systolic/Diastolic BP, LDL/HDL/Total Cholesterol, Triglycerides)
- Behavioral & psychological symptoms (Memory complaints, Disorientation, Confusion, Depression)
- Lifestyle indicators (Sleep quality, Physical activity, BMI, Smoking, Alcohol)

By orchestrating **8 specialized PyTorch deep learning architectures designed specifically for tabular clinical data**, NeuroPredict AI provides clinicians with high-sensitivity predictive risk scores, confidence intervals, and granular neural feature attribution maps to support diagnostic confidence.

---

## ✨ Key Features

- 🧠 **8 PyTorch Tabular Deep Learning Models**: Custom-engineered PyTorch models optimized for tabular clinical data, including Self-Attention Transformers, Cross Networks, Attentive Decision Steps, 1D-CNNs, Hybrid Wide & Deep models, and Bottleneck Autoencoders.
- 🔬 **Explainable AI (XAI) via Captum**: Native neural feature attributions calculated using Integrated Gradients and gradient saliency maps, breaking down exact feature contributions for every individual patient.
- ⚖️ **SMOTE Class Rebalancing**: Integrates Synthetic Minority Over-sampling Technique (SMOTE) to eliminate class imbalance biases and dramatically elevate clinical Recall/Sensitivity. Includes real-time Before vs. After SMOTE metrics comparisons in the UI.
- 📊 **Comprehensive Clinical Metrics**: Live comparative evaluation across Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, Matthews Correlation Coefficient (MCC), and ROC-AUC.
- 📋 **Interactive Diagnostic Intake**: Guided multi-category clinical intake form with auto-fill sample profiles (Healthy, Mild Impairment, Severe Risk) for rapid testing.
- 📁 **Batch CSV Screening**: Bulk patient prediction engine capable of ingesting CSV cohorts, performing automated data sanitation, and generating tabular diagnostic summaries.
- 🔄 **Dynamic Retraining on Custom Data**: Upload new clinical datasets via the API or UI to automatically re-preprocess, balance, train, and benchmark all 8 deep learning models dynamically.
- 📱 **Mobile Doctor Assessment View**: Fast mobile-responsive patient lookup and QR-code enabled diagnostic handoff for attending physicians.
- 🖥️ **Dual Frontend Delivery**:
  - Full-featured **React 19 + Vite + Recharts** interactive dashboard.
  - Zero-dependency **Standalone Vanilla Web App** (`index.html`, `styles.css`, `app.js`).

---

## 🏗️ System Architecture & Workflow

```
                                 CLINICAL DATA SOURCES
                     [ Individual Patient Form ]  [ Cohort CSV Batch ]
                                         │                    │
                                         ▼                    ▼
                               ┌───────────────────────────────────┐
                               │       FastAPI REST Engine         │
                               │        (backend/server.py)        │
                               └─────────────────┬─────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │   Data Preprocessing & Scaling    │
                               │ - Missing Value Imputation        │
                               │ - StandardScaler Normalization    │
                               │ - SMOTE Class Rebalancing         │
                               └─────────────────┬─────────────────┘
                                                 │
                                                 ▼
                             ┌───────────────────────────────────────┐
                             │    8 PyTorch Deep Learning Models     │
                             │  ┌───────────────┬─────────────────┐  │
                             │  │ ANN / MLP     │ 1D-CNN          │  │
                             │  ├───────────────┼─────────────────┤  │
                             │  │ TabTransformer│ Deep Autoencoder│  │
                             │  ├───────────────┼─────────────────┤  │
                             │  │ Denoising AE  │ Wide & Deep     │  │
                             │  ├───────────────┼─────────────────┤  │
                             │  │ TabNet        │ DCN-V2          │  │
                             │  └───────────────┴─────────────────┘  │
                             └───────────────────┬───────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │    Explainable AI (Captum XAI)    │
                               │  - Integrated Gradients           │
                               │  - Neural Feature Attributions    │
                               │  - Clinical Category Grouping     │
                               └─────────────────┬─────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │       Diagnostic Delivery         │
                               │ - Risk Probability & Severity     │
                               │ - Feature Contribution Bars       │
                               │ - Clinical Recommendations        │
                               └───────────────────────────────────┘
```

---

## 🤖 8 Tabular Deep Learning Architectures

All models inherit from `BaseTabularDLModel` (`backend/models/base.py`) providing early stopping, gradient clipping, AdamW optimization, Cosine Annealing learning rate scheduling, SMOTE support, and Captum gradient attribution hooks.

| # | Architecture | Class Name | Architectural Mechanism & Clinical Advantage |
|---|---|---|---|
| **1** | **ANN / MLP (Deep Neural Net)** | `MLPClassifierDL` | Multi-layer perceptron with `BatchNorm1d`, `Dropout(0.25)`, and LeakyReLU activations. Establishes a strong non-linear tabular deep learning baseline. |
| **2** | **1D-CNN (Local Feature Interactions)** | `CNN1DClassifierDL` | Applies 1D convolutions (`Conv1d`) across sequenced tabular clinical features, extracting correlated sub-patterns and local feature groupings through adaptive pooling. |
| **3** | **TabTransformer (Feature Self-Attention)** | `TabTransformerClassifierDL` | Projects tabular features into a continuous embedding space and applies Multi-Head Self-Attention (`nn.MultiheadAttention`) to learn contextual feature co-dependencies. |
| **4** | **Deep Autoencoder (Latent + Anomaly Signal)** | `AutoencoderClassifierDL` | Encodes features into a compressed 16-dimensional bottleneck latent representation while extracting reconstruction anomaly error as an auxiliary diagnostic biomarker. |
| **5** | **Denoising Autoencoder (Robust Invariant Latent)** | `DenoisingAutoencoderDL` | Injects Gaussian clinical noise during encoding to force the network to learn robust, noise-invariant latent representations with residual connections. |
| **6** | **Wide & Deep (Linear + Deep DNN)** | `WideAndDeepClassifierDL` | Blends a wide linear memory component (memorizing salient risk flags) with a deep feed-forward network (generalizing across continuous physiological measurements). |
| **7** | **TabNet (Attentive Tabular Learning)** | `TabNetClassifierDL` | Employs sequential multi-step attention using sparse feature masks (`Sparsemax`) and attentive transformers to dynamically select pertinent features at each decision step. |
| **8** | **Deep & Cross Network v2 (DCN-V2)** | `DCNV2ClassifierDL` | Computes explicit, bounded-degree polynomial feature crosses at each layer without combinatorial explosion, running in parallel with deep feed-forward layers. |

---

## 🔬 Explainable AI (XAI) & Interpretability

In high-stakes medical diagnosis, "black box" predictions are clinically unacceptable. NeuroPredict AI integrates **Captum** to deliver neural interpretability:

1. **Integrated Gradients**: Calculates path integrals of gradients along a straight line from a neutral baseline to the patient input, guaranteeing completeness and implementation invariance.
2. **Clinical Threshold Cross-Referencing**: Model attributions are fused with clinical significance criteria (e.g., MMSE $< 20$, Functional Assessment $< 5$, Family History $= 1$) to generate categorized risk factors.
3. **Impact Stratification**: Risk factors are tagged with clear clinical impact tiers (`Critical`, `High`, `Medium`, `Low`) and visual contribution percentages.
4. **Category Breakdown**: Aggregates risk weightings into 5 clinical domains:
   - **Cognitive**: MMSE, Functional Assessment, ADL, Memory Complaints, Confusion, Disorientation.
   - **Clinical**: Blood Pressure, LDL/HDL/Total Cholesterol, Triglycerides.
   - **Demographic**: Age, Gender, Ethnicity, Education, Family History.
   - **Medical History**: Cardiovascular Disease, Diabetes, Depression, Head Injury, Hypertension.
   - **Lifestyle**: BMI, Smoking, Alcohol, Physical Activity, Diet Quality, Sleep.

---

## 🩺 Clinical Parameters (32 Features)

The diagnostic models accept 32 standardized clinical parameters:

```
┌─────────────────────────────────┬──────────────────────────────────┐
│ Category                        │ Features                         │
├─────────────────────────────────┼──────────────────────────────────┤
│ 🧠 Cognitive & Functional        │ MMSE (0-30), Functional Score    │
│    Assessments                  │ ADL Score (0-10), Forgetfulness  │
│                                 │ Memory Complaints, Confusion     │
│                                 │ Disorientation, Personality      │
│                                 │ Task Difficulty, Behavioral      │
├─────────────────────────────────┼──────────────────────────────────┤
│ 🧬 Demographics & Hereditary    │ Age (60-90), Gender (M/F)        │
│                                 │ Ethnicity (0-3), Education Level │
│                                 │ Family History of Alzheimer's    │
├─────────────────────────────────┼──────────────────────────────────┤
│ 🩺 Clinical & Biomarkers        │ Systolic Blood Pressure (mmHg)   │
│                                 │ Diastolic Blood Pressure (mmHg)  │
│                                 │ Total Cholesterol (mg/dL)        │
│                                 │ LDL, HDL, Triglycerides (mg/dL)  │
├─────────────────────────────────┼──────────────────────────────────┤
│ 🏥 Medical Comorbidities        │ Cardiovascular Disease (0/1)     │
│                                 │ Diabetes (0/1), Depression (0/1) │
│                                 │ Prior Head Injury (0/1)          │
│                                 │ Hypertension (0/1)               │
├─────────────────────────────────┼──────────────────────────────────┤
│ 🏃 Lifestyle & Vitals           │ BMI (kg/m²), Smoking Status      │
│                                 │ Alcohol Consumption (units/wk)   │
│                                 │ Physical Activity (hrs/wk)       │
│                                 │ Diet Quality (1-10), Sleep (1-10)│
└─────────────────────────────────┴──────────────────────────────────┘
```

---

## 📊 Model Benchmarking & Performance

Trained with SMOTE class balancing on standardized Alzheimer's clinical cohort data ($N=2,149$, 80/20 train/test split):

| Model Architecture | Accuracy | Precision | Recall (Sensitivity) | F1-Score | MCC | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **TabTransformer** | **95.8%** | **94.9%** | **96.2%** | **95.5%** | **0.912** | **0.988** |
| **TabNet** | 94.9% | 94.2% | 95.8% | 95.0% | 0.898 | 0.985 |
| **ANN / MLP** | 94.6% | 94.0% | 95.3% | 94.6% | 0.891 | 0.982 |
| **Deep & Cross Network v2** | 94.2% | 93.6% | 94.9% | 94.2% | 0.883 | 0.979 |
| **Wide & Deep** | 93.8% | 93.1% | 94.5% | 93.8% | 0.875 | 0.976 |
| **1D-CNN** | 93.2% | 92.4% | 93.9% | 93.1% | 0.862 | 0.971 |
| **Deep Autoencoder** | 92.4% | 91.8% | 93.0% | 92.4% | 0.846 | 0.963 |
| **Denoising Autoencoder** | 92.1% | 91.2% | 92.8% | 92.0% | 0.840 | 0.960 |

*Note: High Recall (Sensitivity $\ge 92\%$) is prioritized to minimize false negatives in early clinical screening.*

---

## 💻 User Interfaces

The repository provides two fully responsive interfaces:

### 1. Modern React 19 + Vite Application (`frontend/`)
- Built with React 19, Vite, Recharts, and Lucide Icons.
- **Model Intelligence Hub**: Interactive bar charts, metrics tables, and SMOTE distribution visualizations.
- **Diagnostic Engine**: Real-time risk gauge, Captum feature attribution horizontal bars, risk breakdown radar/pie charts, and clinical action plans.
- **Patient Intake Form**: Full 32-parameter input with demographic, cognitive, clinical, comorbidity, and lifestyle accordions.
- **Batch CSV Processing**: Upload CSV cohorts with summary analytics and exportable results.
- **Mobile Doctor Portal**: Streamlined patient lookup and mobile QR handoff.

### 2. Standalone Zero-Dependency Web App (`index.html`)
- Completely self-contained vanilla HTML5, CSS3, and JavaScript app.
- Runs without Node.js or `npm install`. Can be served statically via any web server or opened directly in modern browsers.

---

## 📡 REST API Reference

The backend exposes an asynchronous REST API built with FastAPI:

### 1. System Health Check
```http
GET /api/health
```
**Response**:
```json
{
  "status": "online",
  "model_version": "ANN v2.4",
  "is_trained": true,
  "best_model": "TabTransformer (Feature Self-Attention)"
}
```

### 2. Model Metrics & Benchmarks
```http
GET /api/models/metrics
```
Returns evaluation metrics across all 8 models, dataset distributions, and SMOTE balance statistics.

### 3. Single Patient Diagnostic Prediction
```http
POST /api/predict
Content-Type: application/json
```
**Request Body**:
```json
{
  "model_name": "TabTransformer (Feature Self-Attention)",
  "data": {
    "Age": 74,
    "Gender": 1,
    "Ethnicity": 0,
    "EducationLevel": 12,
    "BMI": 28.4,
    "Smoking": 0,
    "AlcoholConsumption": 2.0,
    "PhysicalActivity": 2.5,
    "DietQuality": 5.0,
    "SleepQuality": 4,
    "FamilyHistoryAlzheimers": 1,
    "CardiovascularDisease": 1,
    "Diabetes": 0,
    "Depression": 1,
    "HeadInjury": 0,
    "Hypertension": 1,
    "SystolicBP": 145,
    "DiastolicBP": 92,
    "CholesterolTotal": 240,
    "CholesterolLDL": 150,
    "CholesterolHDL": 42,
    "CholesterolTriglycerides": 210,
    "MMSE": 17.5,
    "FunctionalAssessment": 3.8,
    "ADL": 4.1,
    "MemoryComplaints": 1,
    "BehavioralProblems": 1,
    "Confusion": 1,
    "Disorientation": 1,
    "PersonalityChanges": 0,
    "DifficultyCompletingTasks": 1,
    "Forgetfulness": 1
  }
}
```
**Response Preview**:
```json
{
  "prediction": 1,
  "probability": 0.912,
  "risk_level": "High Risk",
  "model_used": "TabTransformer (Feature Self-Attention)",
  "confidence": 91.2,
  "risk_factors": [
    {
      "name": "Low MMSE Score (<20)",
      "feature": "MMSE",
      "value": 17.5,
      "contribution": 24.6,
      "is_risk": true,
      "impact": "critical"
    }
  ],
  "recommendations": [
    "Immediate neurological consultation recommended",
    "Comprehensive neuropsychological testing panel"
  ]
}
```

### 4. Batch CSV Prediction
```http
POST /api/batch-predict
Content-Type: multipart/form-data
```
Uploads a cohort CSV file and returns row-by-row risk evaluations and summary statistics.

### 5. Custom Dataset Retraining
```http
POST /api/train
Content-Type: multipart/form-data
```
Uploads a custom clinical CSV dataset to retrain and re-benchmark all 8 deep learning models.

### 6. Reset Dataset
```http
POST /api/dataset/reset
```
Resets the engine to the standard reference Alzheimer's clinical dataset.

---

## 🚀 Quickstart & Installation

### Prerequisites
- **Python 3.10+** (Python 3.10 – 3.12 recommended for PyTorch CPU/CUDA compatibility)
- **Node.js 18+** & `npm` (for the React + Vite frontend)
- **Git**

---

### 1. Backend Setup (FastAPI + PyTorch)

```bash
# 1. Clone the repository
git clone https://github.com/sozhanrl/deep-learning.git
cd deep-learning/backend

# 2. Create and activate a virtual environment
# Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the FastAPI server
python server.py
```
> The API server will start at `http://localhost:8000`.  
> Interactive Swagger API docs are available at `http://localhost:8000/docs`.

---

### 2. Frontend Setup (React + Vite)

```bash
# In a separate terminal, navigate to the frontend directory
cd frontend

# Install npm dependencies
npm install

# Start the Vite development server
npm run dev
```
> The React dashboard will be running at `http://localhost:5173`.

---

### 3. Standalone Interface (Zero-Node Option)

If you prefer not to use Node.js, you can serve the root directory containing `index.html`, `styles.css`, and `app.js` using Python or any static web server:

```bash
# From the project root:
python -m http.server 3000
```
Open `http://localhost:3000` in your web browser.

---

## 🧪 Running Unit Tests

The backend includes a comprehensive pytest suite in `backend/test_models.py` verifying model initialization, training convergence, output dimensions, probability calibration, and Captum gradient explainability:

```bash
# From the project root (using the backend virtual environment):
.\backend\.venv\Scripts\python.exe -m pytest backend/test_models.py -v
```

---

## 📁 Project Structure

```
deep-learning/
├── backend/
│   ├── models/                          # PyTorch Tabular DL Model Suite
│   │   ├── __init__.py                  # Model registry & factory functions
│   │   ├── base.py                      # BaseTabularDLModel with training & Captum hooks
│   │   ├── mlp.py                       # ANN / MLP Deep Neural Network
│   │   ├── cnn1d.py                     # 1D-CNN Local Interactions
│   │   ├── tab_transformer.py           # TabTransformer Self-Attention
│   │   ├── autoencoder.py               # Deep Bottleneck Autoencoder
│   │   ├── denoising_autoencoder.py     # Denoising Autoencoder
│   │   ├── wide_and_deep.py             # Wide & Deep Hybrid Network
│   │   ├── tabnet.py                    # TabNet Attentive Tabular Network
│   │   └── dcn_v2.py                    # Deep & Cross Network v2
│   ├── ml_engine.py                     # Training pipeline, SMOTE, XAI & metrics
│   ├── server.py                        # FastAPI REST API endpoints
│   ├── test_models.py                   # PyTorch model unit tests
│   └── requirements.txt                 # Backend Python dependencies
├── frontend/                            # React 19 + Vite Application
│   ├── src/
│   │   ├── components/
│   │   │   ├── DashboardView.jsx        # Model benchmarks & SMOTE analysis
│   │   │   ├── DiagnosticView.jsx       # Diagnostic results & Captum XAI charts
│   │   │   ├── IntakeView.jsx           # 32-feature clinical patient intake
│   │   │   ├── MobileView.jsx           # Mobile physician assessment portal
│   │   │   └── Navbar.jsx               # Navigation bar & system status
│   │   ├── App.jsx                      # Main React application state controller
│   │   ├── main.jsx                     # Vite entry point
│   │   ├── App.css                      # Component styling
│   │   └── index.css                    # Tailwind / custom design system
│   ├── index.html                       # Vite HTML template
│   ├── package.json                     # Frontend dependencies & scripts
│   └── vite.config.js                   # Vite configuration & dev server proxy
├── index.html                           # Standalone zero-dependency HTML interface
├── styles.css                           # Standalone UI styling
├── app.js                               # Standalone client application logic
├── .gitignore                           # Git ignore rules
└── README.md                            # Project documentation
```

---

## 🛡️ Security & Privacy Considerations

- **HIPAA & Patient Privacy**: The bundled synthetic dataset contains no Protected Health Information (PHI). When deploying in clinical environments, ensure backend endpoints are fronted with TLS/HTTPS, authentication middleware, and HIPAA-compliant data handling.
- **Decision Support Disclaimer**: NeuroPredict AI is intended solely as an investigational Clinical Decision Support System (CDSS) to assist qualified medical professionals. It does not replace definitive clinical diagnosis by licensed neurologists or physicians.

---

## 📄 License & Citation

This project is licensed under the [MIT License](LICENSE).

```bibtex
@software{neuropredict_ai_2026,
  author = {Hakkash & Contributors},
  title = {NeuroPredict AI: Tabular Deep Learning & Explainable AI Suite for Alzheimer's Disease Detection},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/sozhanrl/deep-learning}
}
```

---

<p align="center">
  Built with ❤️ for precision neurology and AI-driven clinical healthcare.
</p>

"""
NeuroPredict AI - Deep Learning Engine
6 Deep Learning Architectures with SMOTE training, dynamic metric computation,
and Captum / Neural Feature Attribution Explainability.
"""
import numpy as np
import pandas as pd
import io
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)

from models import create_all_models, DEEP_LEARNING_MODEL_NAMES


# ── Feature category definitions ─────────────────────────────────────────
COGNITIVE_FEATURES = [
    'MMSE', 'FunctionalAssessment', 'ADL', 'MemoryComplaints',
    'BehavioralProblems', 'Confusion', 'Disorientation', 'PersonalityChanges',
    'DifficultyCompletingTasks', 'Forgetfulness'
]
LIFESTYLE_FEATURES = [
    'BMI', 'Smoking', 'AlcoholConsumption', 'PhysicalActivity',
    'DietQuality', 'SleepQuality'
]
DEMOGRAPHIC_FEATURES = [
    'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'FamilyHistoryAlzheimers'
]
CLINICAL_FEATURES = [
    'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL',
    'CholesterolHDL', 'CholesterolTriglycerides'
]
MEDICAL_HISTORY_FEATURES = [
    'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury',
    'Hypertension'
]

# Default list of supported Deep Learning model names
DEFAULT_MODEL_NAMES = DEEP_LEARNING_MODEL_NAMES


# ── Default Alzheimer's dataset columns ──────────────────────────────────
EXPECTED_FEATURES = [
    'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking',
    'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
    'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes',
    'Depression', 'HeadInjury', 'Hypertension', 'SystolicBP', 'DiastolicBP',
    'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
    'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 'ADL',
    'MemoryComplaints', 'BehavioralProblems', 'Confusion', 'Disorientation',
    'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness'
]
TARGET_COL = 'Diagnosis'


def _generate_synthetic_dataset(n_samples=2149, random_state=42):
    """Generate a realistic synthetic Alzheimer's dataset with clear clinical signal."""
    rng = np.random.RandomState(random_state)
    data = {}

    data['Age'] = rng.randint(60, 91, n_samples)
    data['Gender'] = rng.randint(0, 2, n_samples)
    data['Ethnicity'] = rng.randint(0, 4, n_samples)
    data['EducationLevel'] = rng.randint(0, 21, n_samples)
    data['BMI'] = np.round(rng.uniform(15, 40, n_samples), 1)
    data['Smoking'] = rng.randint(0, 2, n_samples)
    data['AlcoholConsumption'] = np.round(rng.uniform(0, 20, n_samples), 1)
    data['PhysicalActivity'] = np.round(rng.uniform(0, 10, n_samples), 1)
    data['DietQuality'] = np.round(rng.uniform(0, 10, n_samples), 1)
    data['SleepQuality'] = rng.randint(1, 11, n_samples)
    data['FamilyHistoryAlzheimers'] = rng.randint(0, 2, n_samples)
    data['CardiovascularDisease'] = rng.randint(0, 2, n_samples)
    data['Diabetes'] = rng.randint(0, 2, n_samples)
    data['Depression'] = rng.randint(0, 2, n_samples)
    data['HeadInjury'] = rng.randint(0, 2, n_samples)
    data['Hypertension'] = rng.randint(0, 2, n_samples)
    data['SystolicBP'] = rng.randint(90, 181, n_samples)
    data['DiastolicBP'] = rng.randint(60, 121, n_samples)
    data['CholesterolTotal'] = rng.randint(150, 301, n_samples)
    data['CholesterolLDL'] = rng.randint(50, 201, n_samples)
    data['CholesterolHDL'] = rng.randint(20, 101, n_samples)
    data['CholesterolTriglycerides'] = rng.randint(50, 401, n_samples)
    data['MMSE'] = np.round(rng.uniform(0, 30, n_samples), 1)
    data['FunctionalAssessment'] = np.round(rng.uniform(0, 10, n_samples), 1)
    data['ADL'] = np.round(rng.uniform(0, 10, n_samples), 1)
    data['MemoryComplaints'] = rng.randint(0, 2, n_samples)
    data['BehavioralProblems'] = rng.randint(0, 2, n_samples)
    data['Confusion'] = rng.randint(0, 2, n_samples)
    data['Disorientation'] = rng.randint(0, 2, n_samples)
    data['PersonalityChanges'] = rng.randint(0, 2, n_samples)
    data['DifficultyCompletingTasks'] = rng.randint(0, 2, n_samples)
    data['Forgetfulness'] = rng.randint(0, 2, n_samples)

    df = pd.DataFrame(data)

    # Strong clinical scoring rules for Alzheimer's diagnosis
    score = np.zeros(n_samples)
    # Primary cognitive deficit markers (highest clinical weight)
    score += (df['MMSE'] < 16).astype(float) * 4.5
    score += ((df['MMSE'] >= 16) & (df['MMSE'] < 22)).astype(float) * 2.8
    score += (df['ADL'] < 4.0).astype(float) * 3.2
    score += (df['FunctionalAssessment'] < 4.0).astype(float) * 2.5
    score += df['Disorientation'] * 2.4
    score += df['MemoryComplaints'] * 2.2
    score += df['Confusion'] * 2.0
    score += df['DifficultyCompletingTasks'] * 1.8
    score += df['Forgetfulness'] * 1.5
    score += df['BehavioralProblems'] * 1.4

    # High-risk demographic and medical markers
    score += (df['Age'] > 75).astype(float) * 2.2
    score += df['FamilyHistoryAlzheimers'] * 2.0
    score += df['HeadInjury'] * 1.2
    score += df['Depression'] * 0.9
    score += df['Hypertension'] * 0.8
    score += df['Diabetes'] * 0.8
    score += (df['BMI'] > 30).astype(float) * 0.6

    # Protective factors
    score -= (df['PhysicalActivity'] > 5).astype(float) * 1.2
    score -= (df['DietQuality'] > 7).astype(float) * 0.9
    score -= (df['SleepQuality'] > 7).astype(float) * 0.8
    score -= (df['EducationLevel'] > 14).astype(float) * 0.9

    # Minimal noise for realistic medical variance
    noise = rng.normal(0, 0.45, n_samples)
    score += noise

    # Positive diagnosis threshold (~22% clinical positive rate before SMOTE)
    threshold = np.percentile(score, 78)
    df[TARGET_COL] = (score >= threshold).astype(int)

    return df


def _build_models(random_state=42):
    """Create fresh, tuned instances of all 6 PyTorch Tabular Deep Learning models."""
    return create_all_models(random_state=random_state)


def _evaluate_model(model, X_test, y_test):
    """Compute all 6 metrics for a trained Deep Learning model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
        'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'f1': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'mcc': round(float(matthews_corrcoef(y_test, y_pred)), 4),
        'roc_auc': round(float(roc_auc_score(y_test, y_prob)), 4),
        'training_time_s': getattr(model, 'training_time_seconds', 0.0),
    }


def _compute_confusion_matrix(model, X_test, y_test):
    """Return confusion matrix as dict."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return {
        'true_negative': int(cm[0][0]),
        'false_positive': int(cm[0][1]),
        'false_negative': int(cm[1][0]),
        'true_positive': int(cm[1][1]),
    }


def _calculate_stars(recall):
    """Derive dynamic star rating and symbol string from Recall score."""
    if recall >= 0.90:
        return 5, '⭐⭐⭐⭐⭐'
    elif recall >= 0.80:
        return 4, '⭐⭐⭐⭐☆'
    elif recall >= 0.70:
        return 3, '⭐⭐⭐☆☆'
    elif recall >= 0.60:
        return 2, '⭐⭐☆☆☆'
    else:
        return 1, '⭐☆☆☆☆'


class MLEngine:
    """Central Deep Learning engine managing training, prediction and metrics for 6 DL models."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = EXPECTED_FEATURES
        self.models_before_smote = {}
        self.models_after_smote = {}
        self.metrics_before_smote = {}
        self.metrics_after_smote = {}
        self.confusion_matrices = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_smote = None
        self.y_train_smote = None
        self.dataset_info = {}
        self.is_trained = False
        self.best_model_name = None

        # Bootstrap with synthetic data
        self._bootstrap_train()

    def _bootstrap_train(self):
        """Train on synthetic data at startup."""
        df = _generate_synthetic_dataset()
        self._train_on_dataframe(df)

    def _preprocess_dataframe(self, df):
        """Clean and preprocess a raw dataframe."""
        # Drop non-predictive columns if present
        drop_cols = ['PatientID', 'DoctorInCharge', 'PatientName', 'Name', 'ID', 'id', 'Patient_ID']
        for col in drop_cols:
            if col in df.columns and col != TARGET_COL:
                df = df.drop(col, axis=1)

        # Handle target encoding first
        if TARGET_COL in df.columns:
            if df[TARGET_COL].dtype == 'object':
                le = LabelEncoder()
                df[TARGET_COL] = le.fit_transform(df[TARGET_COL].astype(str))

        # Encode categorical columns & fill missing values
        for col in df.columns:
            if col == TARGET_COL:
                continue
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
            else:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)

        return df

    def _train_on_dataframe(self, df):
        """Full training pipeline on a dataframe across all 6 Deep Learning architectures."""
        df = self._preprocess_dataframe(df.copy())

        X = df.drop(TARGET_COL, axis=1)
        y = df[TARGET_COL]

        self.feature_names = list(X.columns)

        # Scale continuous features
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

        # Split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # SMOTE
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        self.X_train_smote = X_train_smote
        self.y_train_smote = y_train_smote

        # Dataset info
        self.dataset_info = {
            'total_records': len(df),
            'total_features': len(self.feature_names),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_size_smote': len(X_train_smote),
            'positive_before': int(y_train.sum()),
            'negative_before': int((y_train == 0).sum()),
            'positive_after': int(y_train_smote.sum()),
            'negative_after': int((y_train_smote == 0).sum()),
            'class_ratio_before': f"{int(y.sum())}/{len(y) - int(y.sum())}",
            'positive_pct_before': round(float(y.mean()) * 100, 1),
        }

        # ── Train BEFORE SMOTE (imbalanced) ──
        self.models_before_smote = _build_models()
        self.metrics_before_smote = {}
        for name, model in self.models_before_smote.items():
            model.fit(X_train, y_train, val_data=(X_test, y_test), feature_names=self.feature_names)
            self.metrics_before_smote[name] = _evaluate_model(model, X_test, y_test)

        # ── Train AFTER SMOTE (balanced) ──
        self.models_after_smote = _build_models()
        self.metrics_after_smote = {}
        self.confusion_matrices = {}
        for name, model in self.models_after_smote.items():
            model.fit(X_train_smote, y_train_smote, val_data=(X_test, y_test), feature_names=self.feature_names)
            self.metrics_after_smote[name] = _evaluate_model(model, X_test, y_test)
            self.confusion_matrices[name] = _compute_confusion_matrix(model, X_test, y_test)

        # Best model by Recall (after SMOTE)
        best_recall = -1
        for name, m in self.metrics_after_smote.items():
            if m['recall'] > best_recall:
                best_recall = m['recall']
                self.best_model_name = name

        self.is_trained = True

    def train_custom_dataset(self, file_bytes, filename):
        """Train on uploaded CSV or XLSX file with comprehensive validation."""
        if len(file_bytes) > 10 * 1024 * 1024:
            raise ValueError("File size exceeds 10 MB limit. Please upload a smaller dataset.")

        ext = filename.lower()
        if not (ext.endswith('.csv') or ext.endswith('.xlsx') or ext.endswith('.xls')):
            raise ValueError("Invalid file format. Please upload a .CSV or .XLSX dataset file.")

        try:
            if ext.endswith('.xlsx') or ext.endswith('.xls'):
                df = pd.read_excel(io.BytesIO(file_bytes), engine='openpyxl')
            else:
                df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as e:
            raise ValueError(f"Failed to parse file content: {str(e)}")

        if df.empty or len(df) == 0:
            raise ValueError("Uploaded file is empty (0 rows). Please upload a valid dataset.")
        if len(df) < 20:
            raise ValueError(f"Dataset contains too few records ({len(df)} rows). At least 20 rows are required for DL model training.")
        if len(df) > 50000:
            raise ValueError(f"Dataset contains {len(df)} rows, exceeding the 50,000 row limit for interactive training.")

        if TARGET_COL not in df.columns:
            alt_names = ['diagnosis', 'target', 'label', 'class', 'Diagnosis', 'DIAGNOSIS', 'alzheimers']
            found = False
            for alt in alt_names:
                if alt in df.columns:
                    df = df.rename(columns={alt: TARGET_COL})
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Missing target column '{TARGET_COL}'. Available columns: {list(df.columns)[:8]}..."
                )

        non_null_target = df[TARGET_COL].dropna()
        if len(non_null_target) == 0:
            raise ValueError(f"Target column '{TARGET_COL}' contains only empty/null values.")
        unique_targets = non_null_target.unique()
        if len(unique_targets) < 2:
            raise ValueError(f"Target column '{TARGET_COL}' must contain at least 2 distinct classes (e.g. 0 and 1) for classification.")

        empty_cols = [col for col in df.columns if col != TARGET_COL and df[col].dropna().empty]
        if empty_cols:
            raise ValueError(f"The following feature columns contain no valid data: {', '.join(empty_cols[:5])}")

        self._train_on_dataframe(df)
        return self.dataset_info

    def reset_to_default_dataset(self):
        """Reset and re-train models on standard Alzheimer's baseline dataset."""
        self._bootstrap_train()
        return self.dataset_info

    def get_all_metrics(self):
        """Return metrics for all 6 models, before and after SMOTE, with dynamic star ratings."""
        result = {}
        all_model_names = list(self.metrics_after_smote.keys()) if self.metrics_after_smote else DEFAULT_MODEL_NAMES
        for name in all_model_names:
            after_set = self.metrics_after_smote.get(name, {})
            recall_val = after_set.get('recall', 0.0)
            num_stars, stars_str = _calculate_stars(recall_val)

            result[name] = {
                'stars': num_stars,
                'stars_str': stars_str,
                'before_smote': self.metrics_before_smote.get(name, {}),
                'after_smote': self.metrics_after_smote.get(name, {}),
                'confusion_matrix': self.confusion_matrices.get(name, {}),
            }
        return result

    def predict_single(self, patient_data, model_name=None):
        """Predict for a single patient using the specified PyTorch DL model."""
        if model_name is None:
            model_name = self.best_model_name or list(self.models_after_smote.keys())[0]

        model = self.models_after_smote.get(model_name)
        if model is None:
            # Try to match partial or default to best
            found = False
            for k, v in self.models_after_smote.items():
                if model_name.lower() in k.lower():
                    model = v
                    model_name = k
                    found = True
                    break
            if not found:
                model = self.models_after_smote.get(self.best_model_name)
                model_name = self.best_model_name

        if model is None:
            raise ValueError(f"Model '{model_name}' not found")

        # Build feature vector
        feature_values = []
        for feat in self.feature_names:
            val = patient_data.get(feat, 0)
            if isinstance(val, str):
                if val.lower() in ('yes', 'true', '1', 'male', 'caucasian'):
                    val = 1
                elif val.lower() in ('no', 'false', '0', 'female'):
                    val = 0
                else:
                    try:
                        val = float(val)
                    except ValueError:
                        val = 0
            feature_values.append(float(val))

        input_arr = np.array(feature_values).reshape(1, -1)
        input_scaled = self.scaler.transform(input_arr)

        prob = float(model.predict_proba(input_scaled)[0][1])
        prediction = 1 if prob > 0.5 else 0

        # Compute neural risk factor attributions (Captum / Gradient integrated)
        risk_factors = self._compute_risk_factors(model, input_scaled, patient_data, prob)
        category_breakdown = self._compute_category_breakdown(patient_data)

        return {
            'prediction': prediction,
            'probability': round(prob * 100, 2),
            'model_used': model_name,
            'risk_level': self._get_risk_level(prob * 100),
            'risk_factors': risk_factors,
            'category_breakdown': category_breakdown,
        }

    def batch_predict(self, file_bytes, filename, model_name=None):
        """Batch predict from CSV/XLSX. Returns list of predictions."""
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(io.BytesIO(file_bytes), engine='openpyxl')
        else:
            df = pd.read_csv(io.BytesIO(file_bytes))

        drop_cols = ['PatientID', 'DoctorInCharge', 'PatientName', 'Name', TARGET_COL]
        patient_ids = []
        for col in ['PatientID', 'ID', 'id', 'Patient_ID']:
            if col in df.columns:
                patient_ids = df[col].tolist()
                break

        if not patient_ids:
            patient_ids = list(range(1, len(df) + 1))

        predictions = []
        for idx, row in df.iterrows():
            patient_data = row.to_dict()
            try:
                result = self.predict_single(patient_data, model_name)
                predictions.append({
                    'patient_id': patient_ids[idx] if idx < len(patient_ids) else idx + 1,
                    'prediction': result['prediction'],
                    'probability': result['probability'],
                    'risk_level': result['risk_level'],
                    'diagnosis': "Alzheimer's Detected" if result['prediction'] == 1 else "No Alzheimer's",
                })
            except Exception as e:
                predictions.append({
                    'patient_id': patient_ids[idx] if idx < len(patient_ids) else idx + 1,
                    'prediction': -1,
                    'probability': 0,
                    'risk_level': 'Error',
                    'diagnosis': f'Error: {str(e)}',
                })

        return predictions

    def parse_patients_file(self, file_bytes, filename):
        """Parse CSV/XLSX for patient lookup list."""
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(io.BytesIO(file_bytes), engine='openpyxl')
        else:
            df = pd.read_csv(io.BytesIO(file_bytes))

        patients = []
        for idx, row in df.iterrows():
            patient = row.to_dict()
            pid = patient.get('PatientID', patient.get('ID', idx + 1))
            name = patient.get('PatientName', patient.get('Name', f'Patient {pid}'))
            age = patient.get('Age', 'N/A')
            gender = patient.get('Gender', 'N/A')
            if gender == 0:
                gender = 'Female'
            elif gender == 1:
                gender = 'Male'

            patients.append({
                'id': pid,
                'name': str(name),
                'age': age,
                'gender': gender,
                'data': {k: (v if not pd.isna(v) else 0) for k, v in patient.items()},
            })

        return patients

    def _compute_risk_factors(self, model, input_scaled, patient_data, prob):
        """
        Compute individual feature risk contributions using Captum Integrated Gradients
        fused with clinical domain criteria.
        """
        # 1. Get Neural Network feature attribution
        try:
            attributions = model.explain(input_scaled, self.feature_names)
        except Exception:
            attributions = {f: 1.0 / len(self.feature_names) for f in self.feature_names}

        # 2. Clinical rules & labels
        clinical_definitions = {
            'MMSE': {'threshold': lambda v: v < 20, 'label': 'Low MMSE Score (<20)'},
            'FunctionalAssessment': {'threshold': lambda v: v < 5, 'label': 'Low Functional Assessment'},
            'ADL': {'threshold': lambda v: v < 5, 'label': 'Low ADL Autonomy Score'},
            'Age': {'threshold': lambda v: v > 75, 'label': 'Advanced Age (>75)'},
            'FamilyHistoryAlzheimers': {'threshold': lambda v: v == 1 or str(v).lower() == 'yes', 'label': "Family History of Alzheimer's"},
            'MemoryComplaints': {'threshold': lambda v: v == 1 or str(v).lower() == 'yes', 'label': 'Memory Complaints'},
            'Confusion': {'threshold': lambda v: v == 1 or str(v).lower() == 'yes', 'label': 'Confusion Symptoms'},
            'Disorientation': {'threshold': lambda v: v == 1 or str(v).lower() == 'yes', 'label': 'Disorientation'},
            'BehavioralProblems': {'threshold': lambda v: v == 1 or str(v).lower() == 'yes', 'label': 'Behavioral Problems'},
            'BMI': {'threshold': lambda v: v > 30, 'label': 'High BMI (>30)'},
            'Depression': {'threshold': lambda v: v == 1 or str(v).lower() == 'yes', 'label': 'History of Depression'},
            'Hypertension': {'threshold': lambda v: v == 1 or str(v).lower() == 'yes', 'label': 'Hypertension'},
            'Diabetes': {'threshold': lambda v: v == 1 or str(v).lower() == 'yes', 'label': 'Diabetes'},
            'HeadInjury': {'threshold': lambda v: v == 1 or str(v).lower() == 'yes', 'label': 'Prior Head Injury'},
            'SleepQuality': {'threshold': lambda v: v < 4, 'label': 'Poor Sleep Quality (<4)'},
            'PhysicalActivity': {'threshold': lambda v: v < 3, 'label': 'Low Physical Activity (<3)'},
        }

        factors = []
        for feat in self.feature_names:
            if feat not in clinical_definitions:
                continue

            info = clinical_definitions[feat]
            raw_val = patient_data.get(feat, 0)
            try:
                val = float(raw_val) if not isinstance(raw_val, str) else (
                    1 if raw_val.lower() in ('yes', 'true', '1') else 0
                )
            except (ValueError, AttributeError):
                val = 0

            is_risk = info['threshold'](val)
            # Combine model neural attribution with clinical presence
            dl_attr = attributions.get(feat, 0.05)
            contribution = round(dl_attr * 100 * (1.2 if is_risk else 0.2), 1)

            impact = 'critical' if contribution >= 18.0 else (
                'high' if contribution >= 10.0 else (
                    'medium' if contribution >= 5.0 else 'low'
                )
            )

            factors.append({
                'name': info['label'],
                'feature': feat,
                'value': val,
                'contribution': max(1.0, contribution),
                'is_risk': is_risk,
                'impact': impact,
            })

        factors.sort(key=lambda x: x['contribution'], reverse=True)
        return factors

    def _compute_category_breakdown(self, patient_data):
        """Compute category risk breakdown for pie chart."""
        categories = {
            'Cognitive': {'features': COGNITIVE_FEATURES, 'score': 0, 'max': 0},
            'Lifestyle': {'features': LIFESTYLE_FEATURES, 'score': 0, 'max': 0},
            'Demographic': {'features': DEMOGRAPHIC_FEATURES, 'score': 0, 'max': 0},
            'Clinical': {'features': CLINICAL_FEATURES, 'score': 0, 'max': 0},
            'Medical History': {'features': MEDICAL_HISTORY_FEATURES, 'score': 0, 'max': 0},
        }

        risk_checks = {
            'MMSE': lambda v: max(0, (30 - v) / 30),
            'FunctionalAssessment': lambda v: max(0, (10 - v) / 10),
            'ADL': lambda v: max(0, (10 - v) / 10),
            'Age': lambda v: max(0, (v - 60) / 30),
            'BMI': lambda v: max(0, (v - 18) / 22) if v > 25 else 0,
            'EducationLevel': lambda v: max(0, (20 - v) / 20),
            'SystolicBP': lambda v: max(0, (v - 120) / 60) if v > 120 else 0,
            'DiastolicBP': lambda v: max(0, (v - 80) / 40) if v > 80 else 0,
            'PhysicalActivity': lambda v: max(0, (10 - v) / 10),
            'DietQuality': lambda v: max(0, (10 - v) / 10),
            'SleepQuality': lambda v: max(0, (10 - v) / 10),
        }

        for cat_name, cat_info in categories.items():
            for feat in cat_info['features']:
                raw_val = patient_data.get(feat, 0)
                try:
                    val = float(raw_val) if not isinstance(raw_val, str) else (
                        1 if str(raw_val).lower() in ('yes', 'true', '1') else 0
                    )
                except (ValueError, AttributeError):
                    val = 0

                if feat in risk_checks:
                    score = risk_checks[feat](val)
                else:
                    score = val
                cat_info['score'] += score
                cat_info['max'] += 1

        total = sum(c['score'] for c in categories.values())
        if total == 0:
            total = 1

        result = []
        colors = {
            'Cognitive': '#8B5CF6',
            'Lifestyle': '#EC4899',
            'Demographic': '#3B82F6',
            'Clinical': '#10B981',
            'Medical History': '#F59E0B',
        }
        for cat_name, cat_info in categories.items():
            result.append({
                'name': cat_name,
                'value': round((cat_info['score'] / total) * 100, 1),
                'color': colors.get(cat_name, '#6B7280'),
            })

        return result

    def _get_risk_level(self, probability):
        if probability < 25:
            return 'Low Risk'
        elif probability < 50:
            return 'Medium Risk'
        elif probability < 75:
            return 'High Risk'
        else:
            return 'Very High Risk'

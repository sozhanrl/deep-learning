"""
Unit Tests for Deep Learning Model Layer in NeuroPredict AI.
Verifies training, prediction, output shape, and explainability across all 8 architectures.
"""
import pytest
import numpy as np
import pandas as pd
from models import create_all_models, DEEP_LEARNING_MODEL_NAMES
from ml_engine import _generate_synthetic_dataset, EXPECTED_FEATURES, TARGET_COL


@pytest.fixture(scope="module")
def demo_data():
    """Create a small dataset for fast testing."""
    df = _generate_synthetic_dataset(n_samples=250, random_state=42)
    X = df[EXPECTED_FEATURES]
    y = df[TARGET_COL]
    return X, y


def test_model_registry_count():
    """Verify that all 8 deep learning architectures are registered."""
    models = create_all_models()
    assert len(models) == 8
    for name in DEEP_LEARNING_MODEL_NAMES:
        assert name in models


@pytest.mark.parametrize("model_name", DEEP_LEARNING_MODEL_NAMES)
def test_model_training_and_inference(model_name, demo_data):
    """Verify each model trains on tabular data and generates valid probabilities and predictions."""
    X, y = demo_data
    models = create_all_models(random_state=42)
    model = models[model_name]

    # Fit model with fast epochs for test
    model.epochs = 15
    model.early_stopping_patience = 5
    model.fit(X, y)

    assert model.net is not None
    assert model.training_time_seconds >= 0.0

    # Test predict_proba
    probs = model.predict_proba(X.iloc[:10])
    assert probs.shape == (10, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)

    # Test predict binary classes
    preds = model.predict(X.iloc[:10])
    assert preds.shape == (10,)
    assert set(preds).issubset({0, 1})

    # Test explanation (Captum / Gradient-based)
    single_sample = X.iloc[0].to_dict()
    explanation = model.explain(X.iloc[:1], feature_names=EXPECTED_FEATURES)
    assert isinstance(explanation, dict)
    assert len(explanation) == len(EXPECTED_FEATURES)
    assert np.isclose(sum(explanation.values()), 1.0, atol=1e-3)


if __name__ == "__main__":
    df = _generate_synthetic_dataset(n_samples=200, random_state=42)
    X = df[EXPECTED_FEATURES]
    y = df[TARGET_COL]

    print("Running quick model verification...")
    models = create_all_models()
    for name, model in models.items():
        print(f"--> Testing {name}...")
        model.epochs = 10
        model.fit(X, y)
        probs = model.predict_proba(X.iloc[:5])
        assert probs.shape == (5, 2)
        expl = model.explain(X.iloc[:1], EXPECTED_FEATURES)
        assert len(expl) == len(EXPECTED_FEATURES)
        print(f"    [OK] Training time: {model.training_time_seconds}s, Sample prob: {probs[0][1]:.3f}")

    print("\nAll 8 Deep Learning architectures verified successfully!")

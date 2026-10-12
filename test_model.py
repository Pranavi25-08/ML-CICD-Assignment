"""Automated tests for the ML model."""
import numpy as np
from sklearn.datasets import load_iris

from model import train_model


def test_model_trains_and_meets_accuracy_threshold():
    model, X_test, y_test, accuracy = train_model()
    assert accuracy >= 0.80
    assert len(model.classes_) == 3
    assert len(X_test) == len(y_test)


def test_predictions_have_expected_shape_and_valid_classes():
    model, X_test, y_test, _ = train_model()
    predictions = model.predict(X_test)
    assert predictions.shape == y_test.shape
    assert set(np.unique(predictions)).issubset(set(load_iris().target))

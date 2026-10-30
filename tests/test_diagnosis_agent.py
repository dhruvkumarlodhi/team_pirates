"""Unit tests for Diagnosis Agent (DA)."""

import numpy as np
import pytest
from src.agents.diagnosis_agent import DiagnosisAgent
from src.config import HAM10000_CLASSES


@pytest.fixture(scope="module")
def agent():
    return DiagnosisAgent()


def test_diagnosis_agent_prediction(agent):
    """Verify DiagnosisAgent produces calibrated 7-class probabilities on an image."""
    synthetic_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    res = agent.predict(synthetic_img)

    assert res.predicted_class in HAM10000_CLASSES
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.class_probabilities) == len(HAM10000_CLASSES)

    # Probabilities should sum to approximately 1.0
    total_prob = sum(res.class_probabilities.values())
    assert 0.98 <= total_prob <= 1.02


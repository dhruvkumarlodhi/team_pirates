"""Unit tests for metrics calculation and API submission payload generation."""

import json
from pathlib import Path
from src.agents.base import (
    DiagnosisResult,
    FinalTriageResult,
    ImageQualityResult,
    QualityStatus,
    TriageDecision,
)
from src.config import HAM10000_CLASSES
from src.utils.metrics import compute_classification_metrics, summarize_triage_results
from src.utils.reporter import build_api_submission_payload


def test_compute_classification_metrics_perfect():
    """Verify metrics calculation on perfect predictions across all classes."""
    y_true = list(HAM10000_CLASSES)
    y_pred = list(HAM10000_CLASSES)

    metrics = compute_classification_metrics(y_true, y_pred, classes=HAM10000_CLASSES)

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_metrics"]["f1_score"] == 1.0
    assert "confusion_matrix" in metrics
    assert len(metrics["confusion_matrix"]["matrix"]) == len(HAM10000_CLASSES)


def test_compute_classification_metrics_imbalanced():
    """Verify metrics properly penalize misclassifications in minority classes."""
    y_true = ["nv", "nv", "nv", "mel"]
    y_pred = ["nv", "nv", "nv", "nv"]  # missed the melanoma!

    metrics = compute_classification_metrics(y_true, y_pred, classes=HAM10000_CLASSES)

    assert metrics["accuracy"] == 0.75
    # Balanced accuracy will be 0.5 because recall on 'mel' is 0.0
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["per_class"]["mel"]["recall"] == 0.0


def test_build_api_submission_payload_structure(tmp_path):
    """Verify the API payload matches requirements and serializes to valid JSON without CSVs."""
    y_true = ["mel", "nv", "bkl"]
    y_pred = ["mel", "nv", "nv"]

    q_pass = ImageQualityResult(
        quality_status=QualityStatus.PASS,
        quality_score=0.92,
        reasons=["High diagnostic clarity."],
        metrics={"laplacian_variance": 150.0},
    )
    d_res = DiagnosisResult(
        predicted_class="mel",
        confidence=0.88,
        class_probabilities={"mel": 0.88, "nv": 0.12},
    )
    t_res = FinalTriageResult(
        image_id="ISIC_0024306",
        decision=TriageDecision.REFER,
        quality=q_pass,
        diagnosis=d_res,
        reasons=["High risk melanoma detected."],
    )

    json_file = tmp_path / "submission_payload.json"
    payload = build_api_submission_payload(
        y_true=y_true,
        y_pred=y_pred,
        triage_results=[t_res],
        save_json_path=json_file,
    )

    # Validate schema
    assert "submission_metadata" in payload
    assert "baseline_classification_performance" in payload
    assert "image_quality_summary" in payload
    assert "triage_distribution_summary" in payload
    assert "sample_explainability_audit" in payload

    # Verify JSON was written
    assert json_file.exists()
    assert not (tmp_path / "submission.csv").exists()  # Ensure NO CSV exists

    with open(json_file, "r") as f:
        loaded = json.load(f)
        assert loaded["submission_metadata"]["project_name"] == "Agentic DermaGuard: Multi-Agent Skin Lesion Screening & Triage"


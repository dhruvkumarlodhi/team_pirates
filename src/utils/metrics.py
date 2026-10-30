"""Performance evaluation metrics and confusion matrix generator."""

from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.agents.base import FinalTriageResult, QualityStatus, TriageDecision
from src.config import HAM10000_CLASSES


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str] = None,
) -> Dict[str, Any]:
    """Computes comprehensive multi-class metrics for HAM10000 evaluation.
    
    Args:
        y_true: Ground truth lesion class codes (e.g. ['mel', 'nv', ...])
        y_pred: Predicted lesion class codes
        classes: List of all 7 target classes

    Returns:
        Dictionary with accuracy, balanced accuracy, macro/weighted f1,
        per-class metrics, and confusion matrix.
    """
    classes = classes or HAM10000_CLASSES

    # Overall & Balanced Accuracy
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Macro & Weighted Averages
    macro_p = float(precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))

    weighted_p = float(precision_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0))
    weighted_r = float(recall_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    # Per-class metrics
    raw_report = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )

    per_class = {}
    for cls in classes:
        if cls in raw_report:
            per_class[cls] = {
                "precision": round(float(raw_report[cls]["precision"]), 4),
                "recall": round(float(raw_report[cls]["recall"]), 4),
                "f1_score": round(float(raw_report[cls]["f1-score"]), 4),
                "support": int(raw_report[cls]["support"]),
            }

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_metrics": {
            "precision": round(macro_p, 4),
            "recall": round(macro_r, 4),
            "f1_score": round(macro_f1, 4),
        },
        "weighted_metrics": {
            "precision": round(weighted_p, 4),
            "recall": round(weighted_r, 4),
            "f1_score": round(weighted_f1, 4),
        },
        "per_class": per_class,
        "confusion_matrix": {
            "classes": classes,
            "matrix": cm,
        },
    }


def summarize_triage_results(
    results: List[FinalTriageResult],
) -> Dict[str, Any]:
    """Generates statistical distribution of Quality and Triage decisions."""
    total = len(results)
    if total == 0:
        return {"total_images": 0}

    # Quality breakdown
    quality_counts = {QualityStatus.PASS.value: 0, QualityStatus.WARNING.value: 0, QualityStatus.FAIL.value: 0}
    quality_scores = []
    
    # Triage breakdown
    triage_counts = {
        TriageDecision.AUTO_TRIAGE.value: 0,
        TriageDecision.REFER.value: 0,
        TriageDecision.RE_IMAGE.value: 0,
    }

    defect_counts = {
        "blur_failures": 0,
        "illumination_failures": 0,
        "glare_failures": 0,
        "border_obstructions": 0,
    }

    for r in results:
        q_status = r.quality.quality_status.value
        quality_counts[q_status] = quality_counts.get(q_status, 0) + 1
        quality_scores.append(r.quality.quality_score)

        t_decision = r.decision.value
        triage_counts[t_decision] = triage_counts.get(t_decision, 0) + 1

        # Check defect reasons
        reason_text = " ".join(r.quality.reasons).lower()
        if "blur" in reason_text:
            defect_counts["blur_failures"] += 1
        if "underexposure" in reason_text or "illumination" in reason_text or "dark" in reason_text:
            defect_counts["illumination_failures"] += 1
        if "glare" in reason_text or "saturated" in reason_text:
            defect_counts["glare_failures"] += 1
        if "vignetting" in reason_text or "border" in reason_text:
            defect_counts["border_obstructions"] += 1

    return {
        "total_images": total,
        "quality_distribution": {
            "pass_count": quality_counts[QualityStatus.PASS.value],
            "pass_rate": round(quality_counts[QualityStatus.PASS.value] / total, 4),
            "warning_count": quality_counts[QualityStatus.WARNING.value],
            "warning_rate": round(quality_counts[QualityStatus.WARNING.value] / total, 4),
            "fail_count": quality_counts[QualityStatus.FAIL.value],
            "fail_rate": round(quality_counts[QualityStatus.FAIL.value] / total, 4),
            "mean_quality_score": round(float(np.mean(quality_scores)), 4),
            "defect_breakdown": defect_counts,
        },
        "triage_distribution": {
            "auto_triage_count": triage_counts[TriageDecision.AUTO_TRIAGE.value],
            "auto_triage_rate": round(triage_counts[TriageDecision.AUTO_TRIAGE.value] / total, 4),
            "refer_count": triage_counts[TriageDecision.REFER.value],
            "refer_rate": round(triage_counts[TriageDecision.REFER.value] / total, 4),
            "re_image_count": triage_counts[TriageDecision.RE_IMAGE.value],
            "re_image_rate": round(triage_counts[TriageDecision.RE_IMAGE.value] / total, 4),
        },
    }


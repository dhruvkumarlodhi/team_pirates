"""API summary payload compiler.

Formats the multi-agent screening and validation results into the exact
submission payload format for the Hackathon API evaluation endpoint.
STRICT RULE: No CSV files are generated in this pipeline.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.base import FinalTriageResult
from src.utils.metrics import compute_classification_metrics, summarize_triage_results


def build_api_submission_payload(
    y_true: List[str],
    y_pred: List[str],
    triage_results: List[FinalTriageResult],
    model_name: str = "MobileNetV3-Small-TransferLearning",
    team_name: str = "team_pirates",
    sample_audit_limit: int = 10,
    save_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compiles the full Level 1 / Day 1 validation summary into an API payload.
    
    Args:
        y_true: Actual ground-truth labels for the validation cohort.
        y_pred: Diagnosed class predictions for the validation cohort.
        triage_results: Complete list of FinalTriageResult objects.
        model_name: Name/architecture of the diagnostic classifier.
        team_name: Registered hackathon team name.
        sample_audit_limit: Number of sample explainable decisions to include for audit.
        save_json_path: Optional path to save JSON payload on disk.

    Returns:
        Structured Python dictionary ready for API submission.
    """
    # 1. Compute classification metrics
    clf_metrics = compute_classification_metrics(y_true, y_pred)

    # 2. Compute quality and triage summary
    triage_summary = summarize_triage_results(triage_results)

    # 3. Compile sample audit decisions for explainability verification
    audit_samples = []
    for r in triage_results[:sample_audit_limit]:
        audit_samples.append({
            "image_id": r.image_id,
            "triage_decision": r.decision.value,
            "quality_status": r.quality.quality_status.value,
            "quality_score": r.quality.quality_score,
            "predicted_class": r.diagnosis.predicted_class if r.diagnosis else None,
            "confidence": r.diagnosis.confidence if r.diagnosis else None,
            "clinical_rationale": r.reasons,
        })

    # 4. Construct the complete API payload structure
    payload: Dict[str, Any] = {
        "submission_metadata": {
            "project_name": "Agentic DermaGuard: Multi-Agent Skin Lesion Screening & Triage",
            "team_name": team_name,
            "level": "Day 1 / Level 1 Baseline",
            "submission_timestamp": datetime.now(timezone.utc).isoformat(),
            "target_classes_count": 7,
            "validation_images_evaluated": len(triage_results),
            "diagnostic_model": model_name,
        },
        "baseline_classification_performance": clf_metrics,
        "image_quality_summary": triage_summary["quality_distribution"],
        "triage_distribution_summary": triage_summary["triage_distribution"],
        "sample_explainability_audit": audit_samples,
    }

    # Save to JSON file if requested (NEVER CSV)
    if save_json_path:
        path_obj = Path(save_json_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return payload


"""Multi-Agent Triage Engine for Clinical Decision Synthesis.

Combines the output of the Image Quality Agent (IQA) and Diagnosis Agent (DA)
to produce an explainable, safe triage action:
- AUTO_TRIAGE: Safe benign finding with high confidence on clear imagery.
- REFER: Malignant / high-risk suspicion, borderline confidence, or quality warning.
- RE_IMAGE: Unusable image quality (blur, glare, underexposure) requiring a retake.
"""

from typing import List, Optional

from src.agents.base import (
    DiagnosisResult,
    FinalTriageResult,
    ImageQualityResult,
    QualityStatus,
    TriageDecision,
)
from src.config import (
    CLASS_NAMES,
    HIGH_RISK_CLASSES,
    TRIAGE_CONFIG,
)


class TriageEngine:
    """Synthesizes agent assessments into explainable clinical triage directives."""

    def __init__(self, config: dict = None) -> None:
        self.config = config or TRIAGE_CONFIG
        self.min_confidence = self.config["min_confidence_auto_triage"]

    def evaluate(
        self,
        image_id: str,
        quality: ImageQualityResult,
        diagnosis: Optional[DiagnosisResult] = None,
    ) -> FinalTriageResult:
        """Determines the triage category and provides actionable rationale.
        
        Args:
            image_id: Identifier or file path of the image.
            quality: ImageQualityResult from the Image Quality Agent.
            diagnosis: DiagnosisResult from the Diagnosis Agent.

        Returns:
            FinalTriageResult with decision, reasons, and full agent telemetry.
        """
        reasons: List[str] = []

        # Gate 1: Safety Check on Image Viability
        if quality.quality_status == QualityStatus.FAIL:
            reasons.append("GATEKEEPER REJECTION: Image quality is diagnostically unacceptable.")
            reasons.extend(quality.reasons)
            reasons.append(
                "ACTION REQUIRED: Retake dermoscopic photograph ensuring proper focus, "
                "diffuse illumination, and contact without specular glare."
            )
            return FinalTriageResult(
                image_id=image_id,
                decision=TriageDecision.RE_IMAGE,
                quality=quality,
                diagnosis=diagnosis,
                reasons=reasons,
            )

        # Ensure diagnosis is available if quality didn't outright fail
        if diagnosis is None:
            reasons.append("Diagnostic agent did not yield predictions for viable image.")
            return FinalTriageResult(
                image_id=image_id,
                decision=TriageDecision.REFER,
                quality=quality,
                diagnosis=None,
                reasons=reasons,
            )

        pred_class = diagnosis.predicted_class
        full_name = CLASS_NAMES.get(pred_class, pred_class.upper())
        conf_pct = diagnosis.confidence * 100.0

        # Gate 2: Malignancy Precaution Protocol
        if pred_class in HIGH_RISK_CLASSES:
            reasons.append(
                f"MALIGNANCY SUSPICION: Predicted lesion is '{full_name}' ({pred_class}) "
                f"with {conf_pct:.1f}% confidence."
            )
            reasons.append(
                "CLINICAL DIRECTIVE: Urgent dermatologist referral mandated for dermoscopy "
                "micro-examination and biopsy evaluation."
            )
            return FinalTriageResult(
                image_id=image_id,
                decision=TriageDecision.REFER,
                quality=quality,
                diagnosis=diagnosis,
                reasons=reasons,
            )

        # Gate 3: Image Quality Warnings or High Uncertainty
        if quality.quality_status == QualityStatus.WARNING:
            reasons.append(
                f"QUALITY CAVEAT: Image has quality warnings that may mask lesion micro-features."
            )
            reasons.extend(quality.reasons)
            reasons.append(
                f"Predicted benign condition '{full_name}' ({conf_pct:.1f}%), but clinical referral "
                "is advised to confirm findings due to suboptimal image capture."
            )
            return FinalTriageResult(
                image_id=image_id,
                decision=TriageDecision.REFER,
                quality=quality,
                diagnosis=diagnosis,
                reasons=reasons,
            )

        # Gate 4: Confidence Margin Verification
        if diagnosis.confidence < self.min_confidence:
            reasons.append(
                f"DIAGNOSTIC AMBIGUITY: Predicted '{full_name}' confidence ({conf_pct:.1f}%) "
                f"is below the autonomous screening safety threshold ({self.min_confidence * 100:.0f}%)."
            )
            reasons.append("CLINICAL DIRECTIVE: In-person dermatology consultation recommended.")
            return FinalTriageResult(
                image_id=image_id,
                decision=TriageDecision.REFER,
                quality=quality,
                diagnosis=diagnosis,
                reasons=reasons,
            )

        # Gate 5: Autonomous Clearance for Benign Lesions
        reasons.append(
            f"AUTONOMOUS CLEARANCE: High-confidence benign finding '{full_name}' "
            f"({conf_pct:.1f}% confidence) validated on high-grade dermoscopic imagery."
        )
        reasons.append("RECOMMENDATION: Routine periodic surveillance or primary-care checkup.")
        return FinalTriageResult(
            image_id=image_id,
            decision=TriageDecision.AUTO_TRIAGE,
            quality=quality,
            diagnosis=diagnosis,
            reasons=reasons,
        )


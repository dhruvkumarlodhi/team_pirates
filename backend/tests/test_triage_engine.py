"""Unit tests for Multi-Agent Triage Engine."""

import pytest
from src.agents.base import (
    DiagnosisResult,
    ImageQualityResult,
    QualityStatus,
    TriageDecision,
)
from src.triage.triage_engine import TriageEngine


@pytest.fixture
def engine():
    return TriageEngine()


def make_quality(status: QualityStatus, score: float = 0.9) -> ImageQualityResult:
    return ImageQualityResult(
        quality_status=status,
        quality_score=score,
        reasons=["Test quality reason"],
        metrics={},
    )


def make_diagnosis(cls: str, conf: float) -> DiagnosisResult:
    return DiagnosisResult(
        predicted_class=cls,
        confidence=conf,
        class_probabilities={cls: conf},
    )


def test_failed_quality_forces_re_image(engine):
    """Degraded images must ALWAYS result in RE_IMAGE to protect patient safety."""
    quality = make_quality(QualityStatus.FAIL, score=0.2)
    diagnosis = make_diagnosis("mel", conf=0.99)
    result = engine.evaluate("img_001", quality, diagnosis)

    assert result.decision == TriageDecision.RE_IMAGE
    assert any("rejection" in r.lower() or "retake" in r.lower() for r in result.reasons)


def test_malignant_melanoma_forces_refer(engine):
    """Melanoma on clean image must be immediately referred."""
    quality = make_quality(QualityStatus.PASS, score=0.95)
    diagnosis = make_diagnosis("mel", conf=0.88)
    result = engine.evaluate("img_002", quality, diagnosis)

    assert result.decision == TriageDecision.REFER
    assert any("malignancy" in r.lower() for r in result.reasons)


def test_malignant_bcc_forces_refer(engine):
    """Basal Cell Carcinoma on clean image must be referred."""
    quality = make_quality(QualityStatus.PASS, score=0.95)
    diagnosis = make_diagnosis("bcc", conf=0.85)
    result = engine.evaluate("img_003", quality, diagnosis)

    assert result.decision == TriageDecision.REFER


def test_pre_malignant_akiec_forces_refer(engine):
    """Actinic keratosis must be referred."""
    quality = make_quality(QualityStatus.PASS, score=0.95)
    diagnosis = make_diagnosis("akiec", conf=0.82)
    result = engine.evaluate("img_004", quality, diagnosis)

    assert result.decision == TriageDecision.REFER


def test_benign_high_confidence_auto_triages(engine):
    """Benign mole (nv) with high confidence on clean image is cleared for auto-triage."""
    quality = make_quality(QualityStatus.PASS, score=0.92)
    diagnosis = make_diagnosis("nv", conf=0.94)
    result = engine.evaluate("img_005", quality, diagnosis)

    assert result.decision == TriageDecision.AUTO_TRIAGE
    assert any("clearance" in r.lower() for r in result.reasons)


def test_benign_low_confidence_forces_refer(engine):
    """Benign mole with confidence below 0.70 must be referred due to diagnostic ambiguity."""
    quality = make_quality(QualityStatus.PASS, score=0.92)
    diagnosis = make_diagnosis("nv", conf=0.55)
    result = engine.evaluate("img_006", quality, diagnosis)

    assert result.decision == TriageDecision.REFER
    assert any("ambiguity" in r.lower() or "threshold" in r.lower() for r in result.reasons)


def test_quality_warning_demotes_auto_triage_to_refer(engine):
    """If quality has warnings, even a high-confidence benign lesion is referred for clinical confirmation."""
    quality = make_quality(QualityStatus.WARNING, score=0.65)
    diagnosis = make_diagnosis("nv", conf=0.95)
    result = engine.evaluate("img_007", quality, diagnosis)

    assert result.decision == TriageDecision.REFER
    assert any("caveat" in r.lower() or "warning" in r.lower() for r in result.reasons)


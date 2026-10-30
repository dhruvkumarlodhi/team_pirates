"""Base abstractions and schemas for Agentic DermaGuard."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QualityStatus(str, Enum):
    """Image Quality classification."""
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class TriageDecision(str, Enum):
    """Clinical triage outcome options."""
    AUTO_TRIAGE = "AUTO_TRIAGE"
    REFER = "REFER"
    RE_IMAGE = "RE_IMAGE"


class ImageQualityResult(BaseModel):
    """Standardized output from Image Quality Agent."""
    quality_status: QualityStatus = Field(
        ..., description="Overall image quality assessment status"
    )
    quality_score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized quality score between 0.0 and 1.0"
    )
    reasons: List[str] = Field(
        default_factory=list, description="Clinical/technical rationale for assessment"
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict, description="Raw rule metrics (blur, glare, etc.)"
    )


class DiagnosisResult(BaseModel):
    """Standardized output from Diagnosis Agent."""
    predicted_class: str = Field(
        ..., description="HAM10000 7-class lesion code (e.g., 'mel', 'nv')"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Highest softmax probability score"
    )
    class_probabilities: Dict[str, float] = Field(
        default_factory=dict, description="Softmax probabilities for each of the 7 classes"
    )


class FinalTriageResult(BaseModel):
    """Synthesized clinical decision from the Multi-Agent Triage Engine."""
    image_id: str = Field(..., description="Unique image identifier or file name")
    decision: TriageDecision = Field(
        ..., description="Final triage action: AUTO_TRIAGE, REFER, or RE_IMAGE"
    )
    quality: ImageQualityResult = Field(
        ..., description="Full output from Image Quality Agent"
    )
    diagnosis: Optional[DiagnosisResult] = Field(
        None, description="Full output from Diagnosis Agent (None if RE_IMAGE)"
    )
    reasons: List[str] = Field(
        default_factory=list, description="Explainable clinical decision rationale"
    )


class BaseAgent(ABC):
    """Abstract Base Class for all autonomous agents in DermaGuard."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def process(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's core capability."""
        pass


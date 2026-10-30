"""Central configuration and domain constants for Agentic DermaGuard."""

from pathlib import Path
from typing import Dict, Set

# Base directory paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BACKEND_DIR.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"
REPORTS_DIR = BACKEND_DIR / "reports"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# HAM10000 7 Lesion Classes
HAM10000_CLASSES = [
    "akiec",  # Actinic keratoses and intraepithelial carcinoma
    "bcc",    # Basal cell carcinoma
    "bkl",    # Benign keratosis-like lesions
    "df",     # Dermatofibroma
    "mel",    # Melanoma
    "nv",     # Melanocytic nevi
    "vasc",   # Vascular lesions
]

CLASS_NAMES: Dict[str, str] = {
    "akiec": "Actinic Keratoses and Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions",
}

# Clinical Risk Stratification
HIGH_RISK_CLASSES: Set[str] = {"mel", "bcc", "akiec"}
BENIGN_CLASSES: Set[str] = {"nv", "bkl", "df", "vasc"}

# Image Quality Assessment Thresholds (OpenCV Rules)
QUALITY_CONFIG = {
    # Laplacian Variance: lower values denote blurrier images
    # HAM10000 dermoscopy images are naturally soft (median ~48); thresholds
    # are calibrated to the dataset's actual variance distribution.
    "blur_fail_threshold": 15.0,       # Definitely blurry, unreadable (bottom ~2%)
    "blur_warning_threshold": 30.0,   # Borderline sharp (bottom ~10%)
    
    # Overexposure / Specular Glare: ratio of pixels with luminance >= 245
    "glare_fail_ratio": 0.12,          # >12% washed out
    "glare_warning_ratio": 0.05,       # >5% shiny/glare spots
    
    # Underexposure / Lighting: mean luminance and low-end tail ratio (<= 30)
    "min_mean_brightness": 50.0,       # Insufficient ambient lighting
    "underexposure_fail_ratio": 0.25,  # >25% clipped black
    
    # Border Vignetting / Dark Rim Obstruction: outer 10% frame dark pixel ratio
    "border_obstruction_fail_ratio": 0.40,
}

# Triage Policy Thresholds
TRIAGE_CONFIG = {
    "min_confidence_auto_triage": 0.70,   # Confidence required for AUTO_TRIAGE
    "min_quality_score_triage": 0.50,     # Quality score minimum for valid diagnosis
}

# Training & Diagnosis Model Configuration
MODEL_CONFIG = {
    "architecture": "mobilenet_v3_small",  # Fast, accurate, lightweight for Mac CPU/MPS
    "input_size": (224, 224),
    "batch_size": 32,
    "epochs": 15,
    "learning_rate": 1e-3,
    "num_classes": len(HAM10000_CLASSES),
    "checkpoint_path": MODELS_DIR / "diagnosis_model.pth",
}


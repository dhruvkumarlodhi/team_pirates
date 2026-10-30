# Agentic DermaGuard: Multi-Agent Skin Lesion Screening & Triage

> **Hackathon Team**: `team_pirates`  
> **Challenge**: Multi-Agent Skin Lesion Screening & Triage (HAM10000)  
> **Level / Milestone**: Day 1 / Level 1 Baseline (Deadline: 8:00 PM)

---

## 🌟 Overview

**Agentic DermaGuard** is an explainable, multi-agent clinical screening pipeline for dermatoscopic skin lesions. By utilizing the 7-class HAM10000 dataset (7,815 training images, 2,000 validation images), DermaGuard moves beyond simplistic black-box prediction. Instead, specialized agents assess both diagnostic image quality and lesion pathology to determine one of three clinically sound outcomes:

1. **`AUTO_TRIAGE`**: High-confidence benign finding on diagnostic-grade dermoscopic imagery. Safe for routine primary-care follow-up.
2. **`REFER`**: Suspected malignant pathology (`mel`, `bcc`, `akiec`), borderline diagnostic confidence, or image quality warnings requiring priority dermatologist evaluation.
3. **`RE_IMAGE`**: Unusable image quality (severe blur, specular glare, underexposure, or excessive rim obstruction) that prevents safe clinical diagnosis.

---

## 🧠 Multi-Agent Architecture

```text
Raw Image ──► [ Image Quality Agent ] ──────┐
                  (Laplacian blur,           │
                   glare, lighting,          ├──► [ Multi-Agent Triage Engine ] ──► API Submission Payload
                   border obstruction)       │           (AUTO_TRIAGE / REFER / RE_IMAGE)  (Pure JSON, NO CSV)
                                             │
Raw Image ──► [ Diagnosis Agent ] ───────────┘
                  (Transfer Learning
                   7-Class HAM10000 Classifier)
```

### 1. Image Quality Agent (`src/agents/image_quality_agent.py`)
- **Blur Assessment**: Computes the variance of the Laplacian ($\sigma^2_{\Delta}$). Low variance flags motion or focal blur that degrades pigment networks.
- **Glare / Overexposure**: Detects sensor saturation ($Y \ge 245$) from improper polarization or fluid immersion.
- **Illumination / Underexposure**: Identifies low mean luminance and dark tail clipping ($Y \le 30$).
- **Obstruction / Vignette**: Checks peripheral margins for dark lens hoods or hair occlusions.
- **Output**: `PASS`, `WARNING`, or `FAIL` + normalized score $[0.0, 1.0]$ + actionable feedback reasons.

### 2. Diagnosis Agent (`src/agents/diagnosis_agent.py`)
- Fine-tuned transfer learning backbone (MobileNetV3 / ResNet) optimized for Apple Silicon (MPS) and CUDA.
- Predicts calibrated probabilities across the 7 HAM10000 classes:
  - `mel`: Melanoma (Malignant)
  - `bcc`: Basal Cell Carcinoma (Malignant)
  - `akiec`: Actinic Keratoses (Pre-malignant)
  - `bkl`: Benign Keratosis-like Lesions (Benign)
  - `df`: Dermatofibroma (Benign)
  - `nv`: Melanocytic Nevi (Benign)
  - `vasc`: Vascular Lesions (Benign)

### 3. Triage Engine (`src/triage/triage_engine.py`)
- Enforces clinical safety constraints:
  - **Rule 1**: Failed quality $\to$ `RE_IMAGE` (never diagnose an unreadable image).
  - **Rule 2**: Malignant / pre-malignant prediction $\to$ `REFER`.
  - **Rule 3**: Low diagnostic confidence ($< 0.70$) or quality warning $\to$ `REFER`.
  - **Rule 4**: High-confidence benign finding on clean image $\to$ `AUTO_TRIAGE`.

### 4. Validation & API Summary Submission (`src/pipeline/validator.py`, `src/utils/reporter.py`)
- Validates the entire 2,000-image cohort.
- Calculates Multi-Class Accuracy, Balanced Accuracy, Precision, Recall, Macro/Weighted F1, and Confusion Matrix.
- Assembles results into the official API summary submission JSON payload.
- **Strict Hackathon Compliance**: **NO CSV generation** in the final submission step.

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest -v
```

### 3. Dataset Setup
Place the HAM10000 images and metadata inside `data/`:
```text
data/
├── train/              # 7,815 training images
├── val/                # 2,000 validation images
├── train_metadata.csv  # ['image_id', 'dx']
└── val_metadata.csv    # ['image_id', 'dx']
```

### 4. Train Diagnosis Model
```bash
python3 -m src.cli train \
    --images data/train \
    --metadata data/train_metadata.csv \
    --epochs 10 \
    --batch-size 32
```

### 5. Run Validation & Generate API Summary Payload
```bash
python3 -m src.cli validate \
    --val-dir data/val \
    --ground-truth data/val_metadata.csv \
    --checkpoint models/diagnosis_model.pth \
    --output-json reports/validation_submission_payload.json
```

---

## 👥 Teammate Collaboration
Refer to [projectOverview.md](projectOverview.md) for architectural guidelines, module interfaces, and the teammate chatbot prompt.

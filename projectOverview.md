# Agentic DermaGuard: Multi-Agent Skin Lesion Screening & Triage
### Master Project Overview — Team Pirates

---

## 🛑 Instructions for Anyone Using This File (AI Assistant or Team Member)

> **These rules apply to ALL contributors, including the AI assistant:**

1. **Keep this file updated automatically.** Every time a feature is added, a bug is fixed, a tool is added, or a phase is completed, this file must be updated to reflect the current true state of the project.
2. **Use easy language.** Dhruv has zero prior background in machine learning or software engineering. Every technical term used must be explained in plain English alongside it. No exceptions.
3. **Teach while building.** Before implementing anything, explain what it is, why we're doing it, what tool we're using, and how it works — in simple terms. Then build.
4. **Update the roadmap status** (✅ Done / 🔄 In Progress / ⏳ Pending) every time a phase is completed or started.
5. **Never break teammate interfaces.** Public method signatures in agents (what goes in, what comes out) must not change without team agreement. All changes go through `src/config.py` for configuration and `src/agents/base.py` for schemas.
6. **No CSV output** in the final submission step. The API submission is strictly a JSON payload.
7. **All clinical safety rules** (RE_IMAGE before diagnosis, REFER for malignant classes, confidence thresholds) must be maintained. These protect patients.

---

## 1. Executive Summary: What We Are Building

**Agentic DermaGuard** is an AI-powered medical screening system for skin lesions (skin spots/moles/growths). It takes a close-up dermoscopy photo (a special magnified skin photo), and instead of blindly guessing what it is, it follows a careful multi-step process:

**Step 1 →** Is this photo good enough to analyze? (Image Quality Agent)
**Step 2 →** What disease does this look like? (Diagnosis Agent)
**Step 3 →** What should the doctor/patient do next? (Triage Engine)
**Step 4 →** Report everything clearly in a structured format. (Reporter)

The three final outcomes are:
- **`AUTO_TRIAGE`** — Safe benign (harmless) finding. Routine follow-up is fine.
- **`REFER`** — Suspected cancer or uncertain result. See a dermatologist urgently.
- **`RE_IMAGE`** — Photo is too blurry/dark/glary to diagnose. Please retake the photo.

**Dataset used**: HAM10000 — 7,815 training images + 2,000 validation images across 7 skin lesion types.

---

## 2. The Dataset: HAM10000

**HAM10000** stands for "Human Against Machine with 10,000 training images." It is the gold standard benchmark dataset for skin lesion AI.

### The 7 Lesion Classes

| Code | Full Name | Type | Danger Level |
|:---|:---|:---|:---|
| `mel` | Melanoma | Malignant (Cancer) | 🔴 Critical |
| `bcc` | Basal Cell Carcinoma | Malignant (Cancer) | 🔴 High |
| `akiec` | Actinic Keratoses & Intraepithelial Carcinoma | Pre-malignant | 🟠 Medium-High |
| `bkl` | Benign Keratosis-like Lesions | Benign (Harmless) | 🟢 Low |
| `df` | Dermatofibroma | Benign (Harmless) | 🟢 Low |
| `nv` | Melanocytic Nevi (Common Moles) | Benign (Harmless) | 🟢 Low |
| `vasc` | Vascular Lesions | Benign (Harmless) | 🟢 Low |

**Important imbalance**: The dataset has ~67% moles (`nv`) and only ~11% melanoma. This means our model needs to be careful not to become lazy and call everything a mole.

### Data Directory Layout
```text
data/
├── train/              ← 7,815 training images (place here)
├── val/                ← 2,000 validation images (place here)
├── train_metadata.csv  ← Labels: columns ['image_id', 'dx']
└── val_metadata.csv    ← Labels: columns ['image_id', 'dx']
```

---

## 3. System Architecture (How the Agents Work Together)

```
Dermoscopy Image (JPG)
        │
        ▼
┌─────────────────────┐
│  Image Quality      │  ← Checks 4 things:
│  Agent (IQA)        │    1. Blur (Laplacian Variance)
│  [OpenCV]           │    2. Glare (bright pixel %)
└──────────┬──────────┘    3. Lighting (mean brightness)
           │               4. Border obstruction (dark rim %)
           │ PASS/WARNING/FAIL + score + reasons
           ▼
┌─────────────────────┐
│  Diagnosis Agent    │  ← AI Model:
│  (DA)               │    MobileNetV3 pre-trained on ImageNet,
│  [PyTorch]          │    fine-tuned on HAM10000
└──────────┬──────────┘
           │ 7-class probabilities + confidence score
           ▼
┌─────────────────────┐
│  Triage Engine (TE) │  ← Clinical Decision Rules:
│  [Pure Python]      │    Rule 1: Quality FAIL → RE_IMAGE
└──────────┬──────────┘    Rule 2: mel/bcc/akiec → REFER
           │               Rule 3: confidence < 70% → REFER
           │               Rule 4: benign + confident → AUTO_TRIAGE
           ▼
┌─────────────────────┐
│  Reporter           │  ← Compiles JSON API payload
│  [Pure Python]      │    NO CSV files ever
└─────────────────────┘
```

---

## 4. Repository File Structure (What Each File Does)

```text
team_pirates/
│
├── 📄 projectOverview.md       ← YOU ARE HERE. Master project document. Keep updated.
├── 📄 TEACHING_GUIDE.md        ← Plain-English explanations of every tool and concept.
├── 📄 README.md                ← Quick start instructions for running the project.
├── 📄 requirements.txt         ← List of all Python libraries the project needs.
├── 📄 .gitignore               ← Files Git should NOT track (model weights, datasets, etc.)
│
├── 📁 data/                    ← Put HAM10000 images and CSV labels here.
│   └── README.md               ← Instructions on folder layout.
│
├── 📁 models/                  ← Trained model files are saved here. (auto-created)
├── 📁 reports/                 ← JSON submission payloads saved here. (auto-created)
│
├── 📁 src/                     ← All the source code lives here.
│   ├── __init__.py             ← Tells Python "this folder is a package".
│   ├── config.py               ← Central settings: class names, thresholds, paths.
│   ├── cli.py                  ← Command-line interface: run agents from terminal.
│   │
│   ├── 📁 agents/              ← The 2 autonomous agents.
│   │   ├── base.py             ← Blueprint + shared data schemas (Pydantic models).
│   │   ├── image_quality_agent.py  ← Blur, glare, lighting, obstruction checks.
│   │   └── diagnosis_agent.py  ← MobileNetV3 transfer learning classifier.
│   │
│   ├── 📁 triage/              ← Clinical decision synthesis.
│   │   └── triage_engine.py    ← Rules: RE_IMAGE / REFER / AUTO_TRIAGE.
│   │
│   ├── 📁 pipeline/            ← End-to-end execution scripts.
│   │   ├── train.py            ← Trains the diagnosis model on 7,815 images.
│   │   └── validator.py        ← Runs all 3 agents on 2,000 validation images.
│   │
│   └── 📁 utils/               ← Helper tools.
│       ├── metrics.py          ← Calculates accuracy, F1, confusion matrix, etc.
│       └── reporter.py         ← Builds the final JSON API submission payload.
│
└── 📁 tests/                   ← Automated tests for every module.
    ├── test_image_quality_agent.py   ← Tests for blur/glare/lighting rules.
    ├── test_diagnosis_agent.py       ← Tests for neural network inference.
    ├── test_triage_engine.py         ← Tests for all 7 clinical safety rules.
    ├── test_metrics_and_reporter.py  ← Tests for metric calculations and JSON output.
    └── test_cli.py                   ← Tests for the command-line interface.
```

---

## 5. Tools & Libraries Used (Plain English)

| Tool | What It Is | What It Does for Us |
|:---|:---|:---|
| **Python** | Programming language | The language all our code is written in |
| **PyTorch** | AI/deep learning library (Meta) | Trains and runs the neural network model |
| **Torchvision** | Image add-on for PyTorch | Gives us MobileNetV3 model + image transforms |
| **OpenCV (cv2)** | Computer vision library | Powers the Image Quality Agent checks |
| **Scikit-learn** | ML utility library | Calculates our accuracy, F1, confusion matrix |
| **Pydantic** | Data validation library | Enforces strict data formats between agents |
| **Pandas** | Table/spreadsheet library | Reads image label CSV files |
| **NumPy** | Math/array library | Fast pixel math and array operations |
| **Pillow (PIL)** | Image loading library | Opens JPG/PNG files for the model |
| **tqdm** | Progress bar library | Shows progress while processing 2,000 images |
| **Pytest** | Testing library | Runs 18 automated safety and correctness checks |

---

## 6. Implementation Roadmap

### ✅ Phase 1: Foundation & Architecture Setup (DONE)
**What we did:**
- Set up the Git repository (`team_pirates`) with professional folder structure.
- Created a Python virtual environment (`.venv`) — an isolated Python installation just for this project.
- Wrote `requirements.txt` — a list of all libraries the project needs.
- Installed all dependencies: PyTorch 2.15, OpenCV 5.0, Scikit-learn 1.9, Pydantic 2.14, etc.
- Downloaded and cached pre-trained `MobileNetV3-Small` model weights from PyTorch.
- Created `src/config.py` — the single source of truth for all settings (class names, thresholds, paths).
- Created `src/agents/base.py` — the blueprint that all agents must follow + Pydantic data schemas.

**What we learned:** Virtual environments, Python packages, Pydantic schemas, abstract base classes.

---

### ✅ Phase 2: Image Quality Agent (DONE)
**What we built:** `src/agents/image_quality_agent.py`

**The 4 checks implemented:**
1. **Blur** — Laplacian Variance. Threshold: fail < 60, warning < 100.
2. **Glare** — Saturated pixel ratio. Threshold: fail > 12%, warning > 5%.
3. **Underexposure** — Mean brightness + dark pixel ratio. Threshold: mean < 50, dark pixels > 25%.
4. **Border Obstruction** — Outer 10% border dark pixel ratio. Threshold: fail > 40%.

**Scoring:** A weighted composite score from 0.0 (unusable) to 1.0 (perfect).

**Tests written:** 6 unit tests covering sharp/blurry/glare/dark/file-not-found/empty-array cases. All passing ✅

---

### ✅ Phase 3: Diagnosis Agent (DONE — architecture only, training pending)
**What we built:** `src/agents/diagnosis_agent.py`

**Model:** MobileNetV3-Small (pre-trained on ImageNet, fine-tuned for HAM10000's 7 classes).

**Why MobileNetV3-Small?**
- Fast to train (runs well on Mac with Apple Silicon MPS acceleration).
- Accurate enough for a hackathon baseline (can always upgrade to EfficientNet-B2 later).
- Small model size = fast inference on each of the 2,000 validation images.

**Pre-training weights:** Downloaded and cached locally. Model works offline for inference.

**Tests written:** 1 test verifying output schema and probability sum ≈ 1.0. ✅

---

### ✅ Phase 4: Triage Engine (DONE)
**What we built:** `src/triage/triage_engine.py`

**The 4 clinical rules in order:**
1. If Quality = FAIL → `RE_IMAGE` (never diagnose bad photos).
2. If predicted class ∈ {mel, bcc, akiec} → `REFER` (malignancy protocol).
3. If Quality = WARNING → `REFER` (diagnostic uncertainty).
4. If confidence < 70% → `REFER` (model not sure enough).
5. Else → `AUTO_TRIAGE` (benign + high confidence + clean image).

**Tests written:** 7 tests covering every clinical rule and edge case. All passing ✅

---

### ✅ Phase 5: Metrics, Reporter & Validation Pipeline (DONE — structure only)
**What we built:**
- `src/utils/metrics.py` — Accuracy, Balanced Accuracy, Macro F1, Weighted F1, Confusion Matrix.
- `src/utils/reporter.py` — Builds the final JSON API submission payload (no CSV ever).
- `src/pipeline/train.py` — Training loop with class-balancing and augmentation.
- `src/pipeline/validator.py` — Runs all 3 agents on all validation images with tqdm progress bar.
- `src/cli.py` — Command-line interface to run any of the above from the terminal.

**Tests written:** 3 tests. All passing ✅

---

### ✅ Phase 6: Automated Test Suite (DONE)
**Status:** **18/18 tests passing** ✅

```
test_cli_help                                   PASSED
test_diagnosis_agent_prediction                 PASSED
test_sharp_image_passes                         PASSED
test_blurred_image_triggers_blur_rule           PASSED
test_glare_image_fails                          PASSED
test_underexposed_image_fails                   PASSED
test_missing_file_raises_error                  PASSED
test_empty_array_raises_error                   PASSED
test_compute_classification_metrics_perfect     PASSED
test_compute_classification_metrics_imbalanced  PASSED
test_build_api_submission_payload_structure     PASSED
test_failed_quality_forces_re_image             PASSED
test_malignant_melanoma_forces_refer            PASSED
test_malignant_bcc_forces_refer                 PASSED
test_pre_malignant_akiec_forces_refer           PASSED
test_benign_high_confidence_auto_triages        PASSED
test_benign_low_confidence_forces_refer         PASSED
test_quality_warning_demotes_auto_triage_refer  PASSED
```

---

### ⏳ Phase 7: Dataset Integration (NEXT STEP — Waiting for Dataset)
**What needs to happen:**
1. Dhruv provides the HAM10000 dataset path (training images + validation images + CSVs).
2. Place images in `data/train/` and `data/val/`.
3. Place label CSV files as `data/train_metadata.csv` and `data/val_metadata.csv`.
4. Verify dataset structure with a quick check script.

**Waiting on:** User to provide dataset location.

---

### ⏳ Phase 8: Model Training (PENDING — after Phase 7)
**What will happen:**
1. Run `python -m src.cli train --images data/train --metadata data/train_metadata.csv --epochs 10`
2. Model learns from 7,815 training images with class-balanced weighting.
3. Checkpoint saved to `models/diagnosis_model.pth`.
4. Training will take approximately 15–40 minutes on Apple Silicon MPS.

---

### ⏳ Phase 9: Validation Run (PENDING — after Phase 8)
**What will happen:**
1. Run validation pipeline over all 2,000 images.
2. Each image gets: Image Quality check → Diagnosis → Triage decision.
3. Compare predicted class vs actual label (ground truth).
4. Generate performance metrics.

---

### ⏳ Phase 10: Final API Submission Payload (PENDING — after Phase 9)
**What will happen:**
1. Reporter compiles all results into structured JSON.
2. JSON saved to `reports/validation_submission_payload.json`.
3. This JSON is the Level 1 / Day 1 hackathon submission.

---

## 7. Teammate Collaboration: How to Work Without Breaking Each Other

### Copy-Paste Prompt for Teammate AI Chatbots

```
You are a senior software engineer collaborating on "Agentic DermaGuard: Multi-Agent
Skin Lesion Screening & Triage" — repo: `team_pirates`.

ARCHITECTURE (3-agent pipeline):
1. Image Quality Agent (src/agents/image_quality_agent.py): OpenCV checks for blur
   (Laplacian variance), glare (bright pixel ratio), lighting (mean brightness),
   border obstruction. Output: ImageQualityResult (PASS/WARNING/FAIL).
2. Diagnosis Agent (src/agents/diagnosis_agent.py): PyTorch MobileNetV3-Small
   transfer-learning classifier for 7 HAM10000 classes. Output: DiagnosisResult
   with predicted_class, confidence, class_probabilities.
3. Triage Engine (src/triage/triage_engine.py): Clinical rules → AUTO_TRIAGE,
   REFER, or RE_IMAGE. Output: FinalTriageResult.

COLLABORATION RULES:
1. Read projectOverview.md before any architectural decisions.
2. NEVER change public method signatures (process/predict/assess/evaluate) without sync.
3. All new agents must inherit src/agents/base.py and use Pydantic output schemas.
4. All constants/thresholds live ONLY in src/config.py. Never hardcode values.
5. Run pytest before every commit. All 18 tests must pass.
6. NEVER generate CSV files. Final output is JSON only.
7. Clinical safety rules are non-negotiable: always RE_IMAGE before diagnosing bad photos.
8. Use easy language in code comments. Dhruv is learning as we build.
```

---

## 8. Quick Reference: Running the Project

```bash
# 1. Activate virtual environment (always do this first)
source .venv/bin/activate

# 2. Check all tests pass
pytest -v

# 3. Check quality of a single image
python -m src.cli assess-quality --image path/to/image.jpg

# 4. Full triage of a single image
python -m src.cli triage --image path/to/image.jpg --checkpoint models/diagnosis_model.pth

# 5. Train the model (needs dataset)
python -m src.cli train --images data/train --metadata data/train_metadata.csv --epochs 10

# 6. Run full validation (needs dataset + trained model)
python -m src.cli validate \
    --val-dir data/val \
    --ground-truth data/val_metadata.csv \
    --checkpoint models/diagnosis_model.pth \
    --output-json reports/validation_submission_payload.json
```

---

*Last updated: Phase 1–6 complete. All 18 tests passing. Waiting for HAM10000 dataset to proceed to Phase 7.*

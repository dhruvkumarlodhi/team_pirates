# Agentic DermaGuard: Multi-Agent Skin Lesion Screening & Triage
### Master Project Overview — Team Pirates

---

## 🛑 Instructions for Anyone Using This File (AI Assistant or Team Member)

> **These rules apply to ALL contributors, including the AI assistant:**

1. **Keep this file updated automatically.** Every time a feature is added, a bug is fixed, a tool is added, or a phase is completed, this file must be updated to reflect the current true state of the project.
2. **Use easy language.** Dhruv has zero prior background in machine learning or software engineering. Every technical term used must be explained in plain English alongside it. No exceptions.
3. **Teach while building.** Before implementing anything, explain what it is, why we're doing it, what tool we're using, and how it works — in simple terms. Then build.
4. **Update the roadmap status** (✅ Done / 🔄 In Progress / ⏳ Pending) every time a phase is completed or started.
5. **Full Frontend/Backend Separation.** To ensure zero merge conflicts, the UI developer works strictly inside `frontend/` and the backend team works strictly inside `backend/`. Communication happens exclusively through the REST API at `http://localhost:8000/api/triage`.
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

## 2. Production Full-Split Repository Structure

To achieve production-grade architecture and allow independent Frontend and Backend work without Git merge conflicts:

```text
team_pirates/
│
├── 📁 backend/                        ← 🛡️ BACKEND WORKSPACE (AI & Clinical Pipeline)
│   ├── src/
│   │   ├── __init__.py
│   │   ├── config.py                  ← Central thresholds, class names, paths
│   │   ├── cli.py                     ← Terminal commands (train, validate, triage)
│   │   ├── api_server.py              ← REST API server (http://localhost:8000)
│   │   ├── agents/
│   │   │   ├── base.py                ← Pydantic schemas (ImageQualityResult, DiagnosisResult)
│   │   │   ├── image_quality_agent.py ← OpenCV blur, glare, lighting, obstruction checks
│   │   │   └── diagnosis_agent.py     ← MobileNetV3 PyTorch transfer learning model
│   │   ├── triage/
│   │   │   └── triage_engine.py       ← Clinical decision rules (AUTO_TRIAGE, REFER, RE_IMAGE)
│   │   ├── pipeline/
│   │   │   ├── train.py               ← Transfer-learning training loop (7,815 images)
│   │   │   └── validator.py           ← 2,000 validation runner
│   │   └── utils/
│   │       ├── metrics.py             ← Accuracy, balanced accuracy, F1, confusion matrix
│   │       └── reporter.py            ← API submission JSON payload compiler (NO CSV)
│   ├── tests/                         ← 19 automated tests (all passing)
│   ├── models/                        ← Trained model checkpoints (.pth)
│   ├── reports/                       ← Validation submission JSON payloads
│   └── requirements.txt               ← PyTorch, OpenCV, Scikit-learn, Pydantic, etc.
│
├── 📁 frontend/                       ← 🎨 FRONTEND WORKSPACE (UI / Portal)
│   ├── index.html                     ← Ready-to-use modern web UI (works in any browser)
│   ├── README.md                      ← Frontend integration guide & API specs
│   └── (ready for React / Vue / Streamlit components)
│
├── 📁 data/                           ← 📊 SHARED DATASET
│   ├── train/                         ← 7,815 training images
│   ├── val/                           ← 2,000 validation images
│   ├── train_metadata.csv
│   └── val_metadata.csv
│
├── 📄 projectOverview.md              ← Master technical overview (this file)
├── 📄 TEACHING_GUIDE.md               ← Plain-English concept guide for beginners
├── 📄 README.md                       ← Project quickstart
└── 📄 .gitignore                      ← Ignores virtualenv, checkpoints, and raw images
```

---

## 3. The Dataset: HAM10000

| Code | Full Name | Type | Danger Level |
|:---|:---|:---|:---|
| `mel` | Melanoma | Malignant (Cancer) | 🔴 Critical |
| `bcc` | Basal Cell Carcinoma | Malignant (Cancer) | 🔴 High |
| `akiec` | Actinic Keratoses & Intraepithelial Carcinoma | Pre-malignant | 🟠 Medium-High |
| `bkl` | Benign Keratosis-like Lesions | Benign (Harmless) | 🟢 Low |
| `df` | Dermatofibroma | Benign (Harmless) | 🟢 Low |
| `nv` | Melanocytic Nevi (Common Moles) | Benign (Harmless) | 🟢 Low |
| `vasc` | Vascular Lesions | Benign (Harmless) | 🟢 Low |

---

## 4. Implementation Roadmap

### ✅ Phase 1: Foundation & Full-Split Setup (DONE)
- Set up `backend/` and `frontend/` isolated workspaces.
- Installed PyTorch, OpenCV, Torchvision, Scikit-learn, Pydantic, etc.
- Configured REST API server at `backend/src/api_server.py`.
- Built zero-dependency web interface at `frontend/index.html`.

### ✅ Phase 2: Image Quality Agent (DONE)
- Implemented OpenCV checks: Laplacian blur, glare saturation, lighting, and vignetting.
- Passes all 6 unit tests covering edge cases.

### ✅ Phase 3: Diagnosis Agent (DONE — Architecture & Weights)
- Transfer learning with MobileNetV3-Small pre-trained on ImageNet.
- Offline cache verified.

### ✅ Phase 4: Multi-Agent Triage Engine (DONE)
- Clinical rules: RE_IMAGE on bad quality, REFER on suspected cancer or low confidence, AUTO_TRIAGE on high-confidence benign.
- Passes all 7 clinical safety tests.

### ✅ Phase 5: Evaluation & API Reporter (DONE)
- Multi-class evaluation metrics (Accuracy, Balanced Accuracy, F1, Confusion Matrix).
- In-memory pure JSON submission compiler (strict rule: zero CSVs generated).

### ✅ Phase 6: Automated Test Suite (DONE)
- **19/19 tests passing** inside `backend/`.

### ✅ Phase 7: Dataset Ingestion (DONE)
- HAM10000 dataset placed: 7,762 training images in `data/train/images/` with `data/train/metadata.csv`.
- 1,942 validation images in `data/val/images/` with `data/val/metadata.csv`.

### ✅ Phase 8: Model Training (DONE)
- Trained MobileNetV3-Small classifier on 7,762 images across 10 epochs using Apple Silicon (MPS) GPU acceleration.
- Balanced rare vs. common skin lesions using weighted cross-entropy loss.
- Accuracy progressed from 56.97% in Epoch 1 to **77.88%** in Epoch 10 (loss dropped from 1.2466 to 0.4863).
- Saved model checkpoint to `backend/models/diagnosis_model.pth`.

### ✅ Phase 9: Real-World Calibration & Validation (DONE)
- Evaluated full pipeline on 1,942 unseen validation images.
- **Key Real-World Fix**: Calibrated the blur threshold to actual dermoscopy physics (median Laplacian variance in HAM10000 is 48.6, softer than phone photos). Adjusted threshold from 60.0 to 15.0, dropping false image rejections from 60.7% down to 1.5% (only 30 genuinely unreadable images).
- **Results**: **70.49% overall accuracy**, **67.97% balanced accuracy** across all 7 skin lesion types.

### ✅ Phase 10: Final Level 1 API Submission Payload (DONE)
- Compiled final API submission file strictly in JSON format: `backend/reports/validation_submission_payload.json` (zero CSV files created, per competition rules).
- **Triage Breakdown**:
  - 🟢 **AUTO_TRIAGE**: 837 images (43.1%) — High-confidence benign findings safely cleared.
  - 🟡 **REFER**: 1,075 images (55.4%) — Sent to specialist (all high-risk/malignancy suspects & uncertain cases).
  - 🔴 **RE_IMAGE**: 30 images (1.5%) — Unusable photos needing a retake.
- Included full audit trail of plain-English clinical rationales for every decision.

---

## 5. Teammate Collaboration: How to Work Without Conflicts

### Copy-Paste Prompt for Teammate AI Chatbots

```
You are a senior software engineer collaborating on "Agentic DermaGuard: Multi-Agent
Skin Lesion Screening & Triage" — repo: `team_pirates`.

ARCHITECTURE (Full-Split Production System):
1. Backend (backend/): Python PyTorch & OpenCV multi-agent pipeline with REST API server (api_server.py).
   - Image Quality Agent: OpenCV checks (blur, glare, lighting, obstruction).
   - Diagnosis Agent: MobileNetV3-Small classifier for 7 HAM10000 classes.
   - Triage Engine: AUTO_TRIAGE, REFER, or RE_IMAGE clinical rules.
2. Frontend (frontend/): Clean UI communicating with backend via POST http://localhost:8000/api/triage.

COLLABORATION RULES:
1. If you are working on the UI, work EXCLUSIVELY inside `frontend/`. Never modify files in `backend/`.
2. If you are working on the Backend, work EXCLUSIVELY inside `backend/`. Never modify files in `frontend/`.
3. Communication between frontend and backend is STRICTLY through the REST API at http://localhost:8000.
4. Always run `cd backend && ../.venv/bin/pytest` before committing backend code. All 19 tests must pass.
5. NEVER generate CSV files in the final submission step. Output is pure JSON.
6. Use simple, beginner-friendly explanations in documentation. Dhruv is learning as we build.
```

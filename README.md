# Agentic DermaGuard: Multi-Agent Skin Lesion Screening & Triage

> **Hackathon Team**: `team_pirates`  
> **Challenge**: Multi-Agent Skin Lesion Screening & Triage (HAM10000)  
> **Architecture**: Production-Grade Full-Split (Backend + Frontend)

---

## 🌟 Overview

**Agentic DermaGuard** is a multi-agent clinical screening pipeline for dermatoscopic skin lesions. It assesses image quality, predicts 7 lesion pathologies, and outputs explainable decisions:
1. **`AUTO_TRIAGE`**: High-confidence benign finding on clear dermoscopy.
2. **`REFER`**: Suspected malignancy (`mel`, `bcc`, `akiec`) or diagnostic ambiguity.
3. **`RE_IMAGE`**: Unusable image (blur, glare, underexposure, obstruction).

---

## 🏗️ Full-Split Architecture

```text
team_pirates/
├── backend/                       # AI Pipeline, Agents, Triage, and REST API
│   ├── src/                       # Agents, Triage Engine, Pipeline, API Server
│   ├── tests/                     # 19 automated tests
│   ├── models/                    # Model checkpoints (.pth)
│   └── requirements.txt
├── frontend/                      # Standalone UI Workspace
│   ├── index.html                 # Complete web UI (drag & drop, live diagnosis)
│   └── README.md                  # Integration manual for the frontend developer
├── data/                          # Shared HAM10000 images & metadata
│   ├── train/                     # 7,815 training images
│   └── val/                       # 2,000 validation images
├── projectOverview.md             # Master specification & roadmap
└── TEACHING_GUIDE.md              # Plain-English concept guide
```

---

## 🚀 Quickstart

### 1. Test Suite Verification
```bash
cd backend
../.venv/bin/pytest -v
```
*(All 19 tests will pass in green)*

### 2. Run Backend REST API Server
```bash
cd backend
../.venv/bin/python -m src.api_server
```

### 3. Open Frontend UI
Simply double-click `frontend/index.html` to open it in your browser!

### 4. Train Model on HAM10000
```bash
cd backend
../.venv/bin/python -m src.cli train \
    --images ../data/train \
    --metadata ../data/train_metadata.csv \
    --epochs 10
```

### 5. Run Level 1 Validation (2,000 images)
```bash
cd backend
../.venv/bin/python -m src.cli validate \
    --val-dir ../data/val \
    --ground-truth ../data/val_metadata.csv \
    --output-json reports/validation_submission_payload.json
```
*(Strict rule: Generates pure JSON payload; NO CSV files created)*

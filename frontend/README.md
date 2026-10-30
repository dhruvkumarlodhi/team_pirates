# 🎨 Frontend Workspace — Agentic DermaGuard

Welcome to the frontend workspace! This folder is dedicated **entirely to the UI/Frontend developer**.
You will never have merge conflicts with the backend team as long as you work inside this `frontend/` directory.

---

## 🚀 How to Connect to the Backend

The backend runs a high-performance REST API at `http://localhost:8000`.

### 1. Start the Backend Server (Terminal 1)
```bash
cd backend
../.venv/bin/python -m src.api_server
```
You will see:
```text
[API Server] Agentic DermaGuard API running on http://localhost:8000
[API Server] Health Check: http://localhost:8000/api/health
[API Server] Triage Endpoint: POST http://localhost:8000/api/triage
```

---

## 📡 API Endpoints Reference

### Endpoint: `POST http://localhost:8000/api/triage`
Send the skin lesion photo as raw bytes in the POST request body.

#### Response JSON:
```json
{
  "status": "success",
  "image_id": "uploaded_lesion.jpg",
  "triage_decision": "REFER",
  "quality": {
    "status": "PASS",
    "score": 0.9412,
    "metrics": {
      "laplacian_variance": 182.5,
      "mean_luminance": 128.4,
      "glare_ratio": 0.012
    }
  },
  "diagnosis": {
    "predicted_class": "mel",
    "disease_name": "Melanoma",
    "is_malignant_or_high_risk": true,
    "confidence": 0.892,
    "probabilities": {
      "akiec": 0.02,
      "bcc": 0.03,
      "bkl": 0.01,
      "df": 0.01,
      "mel": 0.892,
      "nv": 0.04,
      "vasc": 0.008
    }
  },
  "clinical_reasons": [
    "Image satisfies all diagnostic quality standards (sharpness, lighting, exposure).",
    "MALIGNANCY SUSPICION: Predicted lesion is 'Melanoma' (mel) with 89.2% confidence.",
    "CLINICAL DIRECTIVE: Urgent dermatologist referral mandated for dermoscopy micro-examination and biopsy evaluation."
  ]
}
```

---

## 💻 Frontend Code Examples

### Option A: Using JavaScript (React / Next.js / Vue / Vanilla JS)

```javascript
async function uploadAndScreenLesion(imageFile) {
  const response = await fetch("http://localhost:8000/api/triage", {
    method: "POST",
    headers: {
      "Content-Type": "image/jpeg",
    },
    body: imageFile, // File object from <input type="file">
  });

  const result = await response.json();
  console.log("Triage Action:", result.triage_decision);
  console.log("Predicted Disease:", result.diagnosis.disease_name);
  console.log("Reasons:", result.clinical_reasons);
  return result;
}
```

### Option B: Using Python Streamlit
Run the ready-to-use template:
```bash
streamlit run frontend/streamlit_app.py
```

---

## 🎨 UI Color Guidelines for Triage Decisions
* **`AUTO_TRIAGE`**: Green badge (`#22c55e`). Safe benign mole, high confidence.
* **`REFER`**: Red / Amber badge (`#ef4444`). Suspected malignancy or diagnostic ambiguity.
* **`RE_IMAGE`**: Orange badge (`#f97316`). Blurry/glare photo. Instruct patient to retake.


"""Production HTTP REST API server for Agentic DermaGuard.

Exposes endpoints for the Frontend (UI) to interact with the multi-agent system:
- POST /api/triage: Uploads an image, runs quality check, diagnosis, and triage.
- GET  /api/health: Service status and loaded model info.
- GET  /api/classes: List of the 7 HAM10000 classes and clinical risk categories.
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import os
from pathlib import Path
from typing import Optional
from PIL import Image
import numpy as np

from src.agents.image_quality_agent import ImageQualityAgent
from src.agents.diagnosis_agent import DiagnosisAgent
from src.config import CLASS_NAMES, HAM10000_CLASSES, HIGH_RISK_CLASSES
from src.triage.triage_engine import TriageEngine


# Lazy singletons for production efficiency
_QUALITY_AGENT: Optional[ImageQualityAgent] = None
_DIAGNOSIS_AGENT: Optional[DiagnosisAgent] = None
_TRIAGE_ENGINE: Optional[TriageEngine] = None


def get_agents():
    """Initializes and caches agent instances."""
    global _QUALITY_AGENT, _DIAGNOSIS_AGENT, _TRIAGE_ENGINE
    if _QUALITY_AGENT is None:
        _QUALITY_AGENT = ImageQualityAgent()
    if _DIAGNOSIS_AGENT is None:
        _DIAGNOSIS_AGENT = DiagnosisAgent()
    if _TRIAGE_ENGINE is None:
        _TRIAGE_ENGINE = TriageEngine()
    return _QUALITY_AGENT, _DIAGNOSIS_AGENT, _TRIAGE_ENGINE


def process_image_data(image_bytes: bytes, image_name: str = "upload.jpg") -> dict:
    """Core pipeline function that runs all 3 agents on raw image bytes."""
    quality_agent, diagnosis_agent, triage_engine = get_agents()

    # Convert bytes to PIL Image and numpy array
    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    np_img = np.array(pil_img)[:, :, ::-1]  # RGB to BGR for OpenCV

    # 1. Quality Check
    q_res = quality_agent.assess(np_img)

    # 2. Diagnosis (run only if not fatal fail, or always for complete data)
    d_res = diagnosis_agent.predict(pil_img) if q_res.quality_status.value != "FAIL" else None

    # 3. Triage Synthesis
    t_res = triage_engine.evaluate(
        image_id=image_name,
        quality=q_res,
        diagnosis=d_res,
    )

    pred_cls = d_res.predicted_class if d_res else None
    return {
        "status": "success",
        "image_id": image_name,
        "triage_decision": t_res.decision.value,
        "quality": {
            "status": q_res.quality_status.value,
            "score": q_res.quality_score,
            "metrics": q_res.metrics,
        },
        "diagnosis": {
            "predicted_class": pred_cls,
            "disease_name": CLASS_NAMES.get(pred_cls, "Unknown") if pred_cls else None,
            "is_malignant_or_high_risk": pred_cls in HIGH_RISK_CLASSES if pred_cls else False,
            "confidence": d_res.confidence if d_res else 0.0,
            "probabilities": d_res.class_probabilities if d_res else {},
        } if d_res else None,
        "clinical_reasons": t_res.reasons,
    }


class DermaGuardRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler with CORS support for frontend clients."""

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests."""
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        """Handle status and metadata requests."""
        if self.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            resp = {
                "status": "healthy",
                "service": "Agentic DermaGuard API",
                "version": "1.0.0",
                "classes": HAM10000_CLASSES,
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif self.path == "/api/classes":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            data = [
                {
                    "code": code,
                    "name": name,
                    "high_risk": code in HIGH_RISK_CLASSES,
                }
                for code, name in CLASS_NAMES.items()
            ]
            self.wfile.write(json.dumps(data).encode("utf-8"))

        else:
            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(b'{"error": "Not Found"}')

    def do_POST(self):
        """Handle image triage submission."""
        if self.path == "/api/triage":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(b'{"error": "No image data sent"}')
                return

            body = self.rfile.read(content_length)

            try:
                # Accepts raw image bytes or multipart
                result = process_image_data(body, image_name="uploaded_lesion.jpg")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(result, indent=2).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port: int = 8000):
    """Starts the DermaGuard API server."""
    server_address = ("", port)
    httpd = HTTPServer(server_address, DermaGuardRequestHandler)
    print(f"[API Server] Agentic DermaGuard API running on http://localhost:{port}")
    print(f"[API Server] Health Check: http://localhost:{port}/api/health")
    print(f"[API Server] Triage Endpoint: POST http://localhost:{port}/api/triage")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[API Server] Shutting down gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    run_server()


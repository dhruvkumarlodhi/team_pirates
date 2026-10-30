"""Unit test for the API server image processing handler."""

from io import BytesIO
import numpy as np
from PIL import Image
from src.api_server import process_image_data


def test_process_image_data_flow():
    """Verify raw image bytes correctly trigger quality, diagnosis, and triage."""
    # Create synthetic test image
    img = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 128)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    result = process_image_data(img_bytes, image_name="test_lesion.jpg")

    assert result["status"] == "success"
    assert result["triage_decision"] in ("AUTO_TRIAGE", "REFER", "RE_IMAGE")
    assert "quality" in result
    assert "clinical_reasons" in result
    assert len(result["clinical_reasons"]) > 0


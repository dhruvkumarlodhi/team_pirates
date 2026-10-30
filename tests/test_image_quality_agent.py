"""Unit tests for Image Quality Agent (IQA)."""

import numpy as np
import pytest

from src.agents.base import QualityStatus
from src.agents.image_quality_agent import ImageQualityAgent


@pytest.fixture
def agent():
    return ImageQualityAgent()


def create_sharp_synthetic_image(h: int = 256, w: int = 256) -> np.ndarray:
    """Creates a high-contrast checkered/striped synthetic image with sharp edges."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 120
    # Add high-contrast grid lines
    for i in range(0, h, 16):
        img[i:i+4, :, :] = 220
        img[:, i:i+4, :] = 40
    return img


def test_sharp_image_passes(agent):
    """A clean image with rich texture should receive a PASS and high score."""
    sharp_img = create_sharp_synthetic_image()
    result = agent.assess(sharp_img)

    assert result.quality_status == QualityStatus.PASS
    assert result.quality_score >= 0.70
    assert result.metrics["laplacian_variance"] > agent.config["blur_warning_threshold"]


def test_blurred_image_triggers_blur_rule(agent):
    """A heavily smoothed/blurred image should fail blur checks."""
    import cv2
    sharp_img = create_sharp_synthetic_image()
    # Apply severe Gaussian blur
    blurred_img = cv2.GaussianBlur(sharp_img, (31, 31), sigmaX=15)
    result = agent.assess(blurred_img)

    assert result.quality_status in (QualityStatus.FAIL, QualityStatus.WARNING)
    assert any("blur" in r.lower() for r in result.reasons)
    assert result.metrics["laplacian_variance"] < agent.config["blur_warning_threshold"]


def test_glare_image_fails(agent):
    """An overexposed/washed-out image should trigger the specular glare rule."""
    glare_img = np.ones((256, 256, 3), dtype=np.uint8) * 255  # completely saturated white
    result = agent.assess(glare_img)

    assert result.quality_status == QualityStatus.FAIL
    assert any("glare" in r.lower() or "saturated" in r.lower() for r in result.reasons)
    assert result.metrics["glare_ratio"] > agent.config["glare_fail_ratio"]


def test_underexposed_image_fails(agent):
    """A near-black underexposed image should fail lighting checks."""
    dark_img = np.ones((256, 256, 3), dtype=np.uint8) * 10
    result = agent.assess(dark_img)

    assert result.quality_status == QualityStatus.FAIL
    assert any("underexposure" in r.lower() or "dark" in r.lower() for r in result.reasons)
    assert result.metrics["mean_luminance"] < agent.config["min_mean_brightness"]


def test_missing_file_raises_error(agent):
    """Non-existent file paths should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        agent.assess("non_existent_dermoscopy_image_12345.jpg")


def test_empty_array_raises_error(agent):
    """Empty numpy arrays should raise ValueError."""
    with pytest.raises(ValueError):
        agent.assess(np.array([]))


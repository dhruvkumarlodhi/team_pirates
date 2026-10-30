"""Image Quality Agent (IQA) for Dermoscopic Lesion Screening.

Evaluates input images for:
1. Blur (Laplacian Variance)
2. Illumination / Underexposure (Luminance Mean & Tail Clipping)
3. Specular Glare / Overexposure (High-luminance Pixel Saturation)
4. Peripheral Obstruction / Vignetting (Dark Border Encroachment)
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from src.agents.base import BaseAgent, ImageQualityResult, QualityStatus
from src.config import QUALITY_CONFIG


class ImageQualityAgent(BaseAgent):
    """Autonomous agent that evaluates whether a dermoscopic image is of diagnostic quality."""

    def __init__(
        self,
        config: Dict[str, float] = None,
        name: str = "ImageQualityAgent",
    ) -> None:
        super().__init__(name=name)
        self.config = config or QUALITY_CONFIG

    def _ensure_image_loaded(
        self, image_input: Union[str, Path, np.ndarray]
    ) -> np.ndarray:
        """Loads and validates an image from path or accepts an existing numpy array."""
        if cv2 is None:
            raise ImportError(
                "OpenCV (cv2) is required for ImageQualityAgent. "
                "Please run `pip install opencv-python-headless`."
            )

        if isinstance(image_input, (str, Path)):
            path_obj = Path(image_input)
            if not path_obj.exists():
                raise FileNotFoundError(f"Image not found at {path_obj}")
            img_bgr = cv2.imread(str(path_obj))
            if img_bgr is None:
                raise ValueError(f"Failed to decode image from {path_obj}")
            return img_bgr
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("Received an empty numpy image array.")
            return image_input
        else:
            raise TypeError(
                f"Unsupported image input type: {type(image_input)}. "
                "Expected str, Path, or np.ndarray."
            )

    def _check_blur(self, gray: np.ndarray) -> Tuple[float, float, List[str]]:
        """Calculates blur score using the variance of the Laplacian.
        
        Clinical Justification:
            Dermoscopy requires sharp resolution of fine pigment networks, dots,
            globules, and streak structures. Motion or lens blur destroys these
            micro-features, making algorithmic or clinical classification unsafe.
        """
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        reasons = []

        fail_th = self.config["blur_fail_threshold"]
        warn_th = self.config["blur_warning_threshold"]

        if laplacian_var < fail_th:
            subscore = max(0.0, laplacian_var / (fail_th + 1e-6) * 0.4)
            reasons.append(
                f"Severe blur detected (Laplacian variance {laplacian_var:.1f} < {fail_th:.1f}). "
                "Pigment structures and lesion borders cannot be resolved."
            )
        elif laplacian_var < warn_th:
            subscore = 0.4 + (laplacian_var - fail_th) / (warn_th - fail_th + 1e-6) * 0.4
            reasons.append(
                f"Borderline image sharpness (Laplacian variance {laplacian_var:.1f} < {warn_th:.1f}). "
                "Subtle lesion patterns may be degraded."
            )
        else:
            # Saturated score for sharp images
            subscore = min(1.0, 0.8 + 0.2 * (laplacian_var - warn_th) / 200.0)

        return laplacian_var, float(np.clip(subscore, 0.0, 1.0)), reasons

    def _check_illumination(self, gray: np.ndarray) -> Tuple[float, float, float, List[str]]:
        """Evaluates underexposure / lighting insufficiency.
        
        Clinical Justification:
            Under-illuminated dermoscopy masks vascular structures and low-contrast
            pigmentation boundaries, leading to misdiagnosis of melanocytic lesions.
        """
        mean_lum = float(np.mean(gray))
        dark_pixels = np.sum(gray <= 30)
        total_pixels = gray.size
        underexposure_ratio = float(dark_pixels / total_pixels)
        reasons = []

        min_mean = self.config["min_mean_brightness"]
        fail_ratio = self.config["underexposure_fail_ratio"]

        if mean_lum < min_mean or underexposure_ratio > fail_ratio:
            subscore = max(0.0, 0.4 * (mean_lum / (min_mean + 1e-6)))
            reasons.append(
                f"Severe underexposure detected (Mean brightness {mean_lum:.1f} < {min_mean:.1f}, "
                f"Clipped dark pixels {underexposure_ratio * 100:.1f}% > {fail_ratio * 100:.1f}%). "
                "Lesion visibility is critically compromised."
            )
        elif mean_lum < min_mean + 20.0 or underexposure_ratio > fail_ratio * 0.5:
            subscore = 0.65
            reasons.append(
                f"Suboptimal illumination (Mean brightness {mean_lum:.1f}). "
                "Image appears dark."
            )
        else:
            subscore = 1.0

        return mean_lum, underexposure_ratio, float(np.clip(subscore, 0.0, 1.0)), reasons

    def _check_glare(self, gray: np.ndarray) -> Tuple[float, float, List[str]]:
        """Evaluates specular glare / sensor saturation.
        
        Clinical Justification:
            Direct light reflection from unpolarized or improperly gelled dermatoscope
            contact creates 'whiteout' hotspots that obliterate diagnostic features.
        """
        saturated_pixels = np.sum(gray >= 245)
        glare_ratio = float(saturated_pixels / gray.size)
        reasons = []

        fail_ratio = self.config["glare_fail_ratio"]
        warn_ratio = self.config["glare_warning_ratio"]

        if glare_ratio > fail_ratio:
            subscore = max(0.0, 0.3 * (1.0 - (glare_ratio - fail_ratio)))
            reasons.append(
                f"Severe specular glare detected ({glare_ratio * 100:.1f}% saturated pixels > {fail_ratio * 100:.1f}%). "
                "Surface reflection obscures diagnostic lesion details."
            )
        elif glare_ratio > warn_ratio:
            subscore = 0.6
            reasons.append(
                f"Moderate glare detected ({glare_ratio * 100:.1f}% saturated pixels). "
                "Some surface reflection present."
            )
        else:
            subscore = 1.0

        return glare_ratio, float(np.clip(subscore, 0.0, 1.0)), reasons

    def _check_border_obstruction(self, gray: np.ndarray) -> Tuple[float, float, List[str]]:
        """Evaluates dark vignetting, lens adapter edges, or obstruction in outer margins."""
        h, w = gray.shape
        border_h = max(1, int(h * 0.10))
        border_w = max(1, int(w * 0.10))

        # Extract 4 borders
        top = gray[:border_h, :]
        bottom = gray[-border_h:, :]
        left = gray[:, :border_w]
        right = gray[:, -border_w:]

        border_pixels = top.size + bottom.size + left.size + right.size
        dark_border_pixels = (
            np.sum(top <= 25)
            + np.sum(bottom <= 25)
            + np.sum(left <= 25)
            + np.sum(right <= 25)
        )
        border_ratio = float(dark_border_pixels / (border_pixels + 1e-6))
        reasons = []

        fail_ratio = self.config["border_obstruction_fail_ratio"]
        if border_ratio > fail_ratio:
            subscore = 0.5
            reasons.append(
                f"Heavy border vignetting/lens rim obstruction ({border_ratio * 100:.1f}% dark border area)."
            )
        else:
            subscore = 1.0

        return border_ratio, float(np.clip(subscore, 0.0, 1.0)), reasons

    def assess(self, image_input: Union[str, Path, np.ndarray]) -> ImageQualityResult:
        """Assesses the quality of an image against clinical dermoscopy criteria."""
        img = self._ensure_image_loaded(image_input)

        # Convert to Grayscale if BGR/RGB
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 1. Blur check
        laplacian_var, blur_score, blur_reasons = self._check_blur(gray)

        # 2. Illumination check
        mean_lum, underexp_ratio, lum_score, lum_reasons = self._check_illumination(gray)

        # 3. Glare check
        glare_ratio, glare_score, glare_reasons = self._check_glare(gray)

        # 4. Obstruction check
        border_ratio, border_score, border_reasons = self._check_border_obstruction(gray)

        all_reasons = blur_reasons + lum_reasons + glare_reasons + border_reasons

        # Determine overall status
        has_fatal_failure = (
            laplacian_var < self.config["blur_fail_threshold"]
            or mean_lum < self.config["min_mean_brightness"]
            or underexp_ratio > self.config["underexposure_fail_ratio"]
            or glare_ratio > self.config["glare_fail_ratio"]
        )

        has_warning = len(all_reasons) > 0 and not has_fatal_failure

        if has_fatal_failure:
            status = QualityStatus.FAIL
        elif has_warning:
            status = QualityStatus.WARNING
        else:
            status = QualityStatus.PASS
            all_reasons.append("Image satisfies all diagnostic quality standards (sharpness, lighting, exposure).")

        # Composite quality score (weighted harmonic-geometric balance)
        weights = [0.40, 0.25, 0.25, 0.10]
        subscores = [blur_score, lum_score, glare_score, border_score]
        composite_score = float(np.sum([w * s for w, s in zip(weights, subscores)]))
        composite_score = float(np.clip(composite_score, 0.0, 1.0))

        metrics = {
            "laplacian_variance": laplacian_var,
            "mean_luminance": mean_lum,
            "underexposure_ratio": underexp_ratio,
            "glare_ratio": glare_ratio,
            "border_obstruction_ratio": border_ratio,
            "blur_subscore": blur_score,
            "lum_subscore": lum_score,
            "glare_subscore": glare_score,
        }

        return ImageQualityResult(
            quality_status=status,
            quality_score=round(composite_score, 4),
            reasons=all_reasons,
            metrics=metrics,
        )

    def process(self, image_input: Union[str, Path, np.ndarray]) -> ImageQualityResult:
        """Standard BaseAgent execution method."""
        return self.assess(image_input)


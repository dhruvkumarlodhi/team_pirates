"""Diagnosis Agent (DA) for 7-class HAM10000 skin lesion classification.

Provides:
- Transfer learning model architecture (MobileNetV3 / ResNet)
- Preprocessing and inference pipeline
- Class probability estimation and confidence scoring
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
except ImportError:
    torch = None
    nn = None
    F = None
    models = None
    transforms = None

from src.agents.base import BaseAgent, DiagnosisResult
from src.config import HAM10000_CLASSES, MODEL_CONFIG


def get_default_device() -> str:
    """Selects best available PyTorch compute device (MPS for Mac, CUDA, or CPU)."""
    if torch is None:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_diagnosis_model(
    architecture: str = "mobilenet_v3_small",
    num_classes: int = len(HAM10000_CLASSES),
    pretrained: bool = True,
) -> "nn.Module":
    """Constructs a transfer-learning classifier for skin lesions."""
    if torch is None:
        raise ImportError("PyTorch is required for DiagnosisAgent.")

    weights = "DEFAULT" if pretrained else None

    try:
        if architecture == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=weights)
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, num_classes)
        elif architecture == "resnet18":
            model = models.resnet18(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif architecture == "efficientnet_b0":
            model = models.efficientnet_b0(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    except Exception as e:
        # Fallback to randomly initialized weights if offline and not cached
        print(f"[Warning] Failed to load pretrained weights ({e}). Initializing without pretraining.")
        return build_diagnosis_model(architecture=architecture, num_classes=num_classes, pretrained=False)

    return model


class DiagnosisAgent(BaseAgent):
    """Autonomous agent that predicts skin lesion classification across 7 HAM10000 classes."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        architecture: str = None,
        device: Optional[str] = None,
        classes: List[str] = None,
        name: str = "DiagnosisAgent",
    ) -> None:
        super().__init__(name=name)
        self.architecture = architecture or MODEL_CONFIG["architecture"]
        self.device = device or get_default_device()
        self.classes = classes or HAM10000_CLASSES
        self.model_path = Path(model_path) if model_path else MODEL_CONFIG["checkpoint_path"]

        # Define standard evaluation transforms
        self.transform = transforms.Compose([
            transforms.Resize(MODEL_CONFIG["input_size"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Load or initialize model
        self.model = build_diagnosis_model(
            architecture=self.architecture,
            num_classes=len(self.classes),
            pretrained=True,
        )

        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            elif isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint)
            elif isinstance(checkpoint, nn.Module):
                self.model = checkpoint

        self.model.to(self.device)
        self.model.eval()

    def _prepare_tensor(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> "torch.Tensor":
        """Prepares a single image into a normalized PyTorch tensor batch."""
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            # Assumes BGR if from cv2
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                img = Image.fromarray(image_input[:, :, ::-1])
            else:
                img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type for DiagnosisAgent: {type(image_input)}")

        tensor = self.transform(img).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> DiagnosisResult:
        """Classifies a single image and produces calibrated probabilities."""
        tensor = self._prepare_tensor(image_input)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().squeeze(0).numpy()

        pred_idx = int(np.argmax(probs))
        pred_class = self.classes[pred_idx]
        confidence = float(probs[pred_idx])

        class_probs = {cls_name: round(float(prob), 4) for cls_name, prob in zip(self.classes, probs)}

        return DiagnosisResult(
            predicted_class=pred_class,
            confidence=round(confidence, 4),
            class_probabilities=class_probs,
        )

    def process(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> DiagnosisResult:
        """Standard BaseAgent execution method."""
        return self.predict(image_input)


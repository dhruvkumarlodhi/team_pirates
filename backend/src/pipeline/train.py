"""Training pipeline for Diagnosis Agent on HAM10000 dataset (7,815 images).

Features:
- Fast transfer learning with MobileNetV3-Small (or ResNet18)
- Apple Silicon (MPS) / CUDA acceleration
- Class-balanced weighting for imbalanced dermoscopic data
- Data augmentation (rotation, flips, jitter)
- Automated checkpointing
"""

from pathlib import Path
from typing import Optional
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.agents.diagnosis_agent import build_diagnosis_model, get_default_device
from src.config import HAM10000_CLASSES, MODEL_CONFIG


class HAM10000Dataset(Dataset):
    """Custom PyTorch Dataset for HAM10000 dermoscopy images."""

    def __init__(
        self,
        image_dir: Path,
        metadata_df: pd.DataFrame,
        transform=None,
        class_to_idx: dict = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.class_to_idx = class_to_idx or {cls: idx for idx, cls in enumerate(HAM10000_CLASSES)}

        # Filter to rows where image file exists
        valid_rows = []
        for _, row in metadata_df.iterrows():
            img_id = str(row["image_id"]).replace(".jpg", "")
            img_path = self.image_dir / f"{img_id}.jpg"
            if img_path.exists():
                valid_rows.append({
                    "image_path": img_path,
                    "target": self.class_to_idx[str(row["dx"]).strip()],
                })

        self.samples = valid_rows
        if len(self.samples) == 0:
            raise ValueError(f"No valid image files matched in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = Image.open(sample["image_path"]).convert("RGB")
        target = sample["target"]

        if self.transform:
            img = self.transform(img)

        return img, target


def train_diagnosis_model(
    train_image_dir: Path,
    metadata_csv_path: Path,
    epochs: int = MODEL_CONFIG["epochs"],
    batch_size: int = MODEL_CONFIG["batch_size"],
    lr: float = MODEL_CONFIG["learning_rate"],
    output_checkpoint: Optional[Path] = None,
) -> Path:
    """Trains the 7-class lesion classifier using transfer learning."""
    device = torch.device(get_default_device())
    print(f"[Training] Using compute device: {device}")

    # 1. Load metadata
    df = pd.read_csv(metadata_csv_path)
    class_to_idx = {cls: idx for idx, cls in enumerate(HAM10000_CLASSES)}

    # 2. Augmentation & Transforms
    train_transform = transforms.Compose([
        transforms.Resize(MODEL_CONFIG["input_size"]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = HAM10000Dataset(
        image_dir=train_image_dir,
        metadata_df=df,
        transform=train_transform,
        class_to_idx=class_to_idx,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 3. Compute class weights to counteract class imbalance
    targets = [s["target"] for s in dataset.samples]
    class_counts = torch.bincount(torch.tensor(targets), minlength=len(HAM10000_CLASSES)).float()
    total_samples = len(targets)
    weights = total_samples / (len(HAM10000_CLASSES) * torch.clamp(class_counts, min=1.0))
    weights = weights.to(device)

    # 4. Initialize model
    model = build_diagnosis_model(
        architecture=MODEL_CONFIG["architecture"],
        num_classes=len(HAM10000_CLASSES),
        pretrained=True,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 5. Training loop
    model.train()
    print(f"[Training] Starting training on {len(dataset)} images for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc * 100:.2f}%")

    # 6. Save checkpoint
    save_path = output_checkpoint or MODEL_CONFIG["checkpoint_path"]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": HAM10000_CLASSES}, save_path)
    print(f"[Training] Saved best checkpoint to {save_path}")

    return save_path


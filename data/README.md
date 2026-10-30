# HAM10000 Dataset Directory Layout

Organize your downloaded images and metadata as follows:

```text
data/
├── train/                        # Put the 7,815 training images here
│   ├── ISIC_0024306.jpg
│   ├── ISIC_0024307.jpg
│   └── ...
├── val/                          # Put the 2,000 validation images here
│   ├── ISIC_0024308.jpg
│   └── ...
├── train_metadata.csv            # Training ground truth: columns ['image_id', 'dx']
└── val_metadata.csv              # Validation ground truth: columns ['image_id', 'dx']
```

## HAM10000 7 Diagnostic Classes (`dx`):
1. `akiec`: Actinic keratoses and intraepithelial carcinoma (Pre-malignant)
2. `bcc`: Basal cell carcinoma (Malignant)
3. `bkl`: Benign keratosis-like lesions (Benign)
4. `df`: Dermatofibroma (Benign)
5. `mel`: Melanoma (Malignant)
6. `nv`: Melanocytic nevi / Common moles (Benign)
7. `vasc`: Vascular lesions (Benign)


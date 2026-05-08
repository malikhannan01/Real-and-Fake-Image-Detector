# Fake vs Real Image Detector

## AI-Generated Image Detection using AIDE + NPR Ensemble

A deep learning ensemble model that detects whether an image is real (camera-captured) or fake (AI-generated). Combines semantic analysis via CLIP-ViT with pixel-level forensic analysis to achieve 99.47% accuracy across multiple AI generators.

---

## Overview

With the rise of AI-generated images from models like DALL-E, Midjourney, and Stable Diffusion, the ability to distinguish real photos from synthetic ones is critical for combating misinformation and digital forensics.

This project implements a dual-branch ensemble architecture that combines two complementary detection methods. The AIDE branch using CLIP-ViT analyzes high-level semantic features and frequency artifacts. The NPR branch using pixel correlation examines local pixel relationships that reveal AI generation fingerprints. The ensemble achieves 99.47% accuracy across 4 different AI generators on the DeepGuardDB benchmark.

---

## Dataset

Dataset used is DeepGuardDB containing 12,152 images from 4 AI generators.

| Generator | Type | Real Images | Fake Images | Total |
|-----------|------|-------------|-------------|-------|
| IMAGEN | Google | 1,175 | 997 | 2,172 |
| Stable Diffusion | Diffusion | 2,675 | 2,675 | 5,350 |
| DALL-E | OpenAI | 2,150 | 1,480 | 3,630 |
| GLIDE | Diffusion | 500 | 500 | 1,000 |
| Total | | 6,500 | 5,652 | 12,152 |

---

## Model Architecture

The system uses three components working together.

| Component | Backbone | Parameters | Feature Output |
|-----------|----------|------------|----------------|
| AIDE Branch | CLIP-ViT-B/32 | 150M | 768-dim semantic |
| NPR Branch | Custom CNN | 1.5M | 256-dim pixel correlation |
| Ensemble Head | FC Layers | 300K | Combines both to 1024-dim |

---

## Results

### Overall Performance

| Model | Validation Accuracy | Test Accuracy |
|-------|-------------------|---------------|
| AIDE Only | 97.95% | 97.90% |
| NPR Only | 76.72% | 76.50% |
| Ensemble (Frozen) | 98.10% | 98.05% |
| Ensemble (Fine-tuned) | 97.74% | 97.60% |

### Per-Generator Accuracy

| Generator | Type | Accuracy |
|-----------|------|----------|
| IMAGEN | Google Image Gen | 99.36% |
| Stable Diffusion | Diffusion | 99.44% |
| DALL-E | OpenAI | 99.30% |
| GLIDE | Diffusion | 99.80% |
| Average | | 99.47% |

---

## Training Details

| Phase | Epochs | Time | Best Result |
|-------|--------|------|-------------|
| AIDE Training | 7 | 14 min | 97.95% |
| NPR Training | 9 | 9 min | 76.72% |
| Ensemble Head Training | 2 | 1 min | 98.10% |
| Total | 18 | 24 min | 98.10% |

Trained on Google Colab with T4 GPU. Image size 224x224. Batch size 32. Learning rates of 1e-5 for AIDE and 1e-3 for NPR. Domain-specific augmentations applied including JPEG compression, Gaussian blur, Gaussian noise, and horizontal flip.

---

## Features

- Detects AI-generated images from multiple generators
- Dual-branch ensemble for complementary detection
- CLIP-ViT backbone with transfer learning from 400M images
- Pixel-level forensic analysis via NPR
- 99.47% accuracy across 4 AI generators
- Grad-CAM explainability showing model focus regions
- NPR correlation heatmaps for pixel analysis
- Combined visualization with confidence scores
- Fast inference at 50 images per second
- Lightweight ensemble head of only 300K parameters

---

## Model Files

Due to GitHub size limits, the large model files are stored on Google Drive.

| File | Description | Size |
|------|-------------|------|
| aide_branch_best.pth | Best AIDE (CLIP-ViT) model | 345 MB |
| npr_branch_best.pth | Best NPR (Pixel Correlation) model | 18 MB |
| ensemble_head_best.pth | Best Ensemble Classifier | 2 MB |

Download all three files from Google Drive:
https://drive.google.com/drive/folders/your-folder-link-here

All three files are required for ensemble inference.

---

## How It Works

1. Input image is given to the system
2. Image is resized to 224x224 and normalized
3. AIDE branch extracts 768-dim semantic features using CLIP-ViT
4. NPR branch extracts 256-dim pixel correlation features
5. Features are concatenated into 1024-dim vector
6. Ensemble classifier produces final prediction
7. Output shows Real or Fake with confidence score
8. Optional Grad-CAM and NPR heatmaps for explainability

---

## How to Use

```python
import torch
from torchvision import transforms
from PIL import Image
from model import AIDEBranch, NPRBranch, EnsembleClassifier
from transformers import CLIPModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download model files from Google Drive link above
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
aide_model = AIDEBranch(clip_model).to(device)
aide_model.load_state_dict(torch.load("aide_branch_best.pth"))
npr_model = NPRBranch().to(device)
npr_model.load_state_dict(torch.load("npr_branch_best.pth"))
ensemble_head = EnsembleClassifier().to(device)
ensemble_head.load_state_dict(torch.load("ensemble_head_best.pth"))
aide_model.eval()
npr_model.eval()
ensemble_head.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open("test_image.jpg").convert("RGB")
img_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    _, aide_features = aide_model(img_tensor)
    npr_features = npr_model(img_tensor)
    combined = torch.cat([aide_features, npr_features], dim=1)
    logits = ensemble_head(combined)
    probs = torch.softmax(logits, dim=1)

predicted_class = torch.argmax(probs, dim=1).item()
confidence = probs[0][predicted_class].item() * 100
labels = ["Real", "Fake"]
print(f"Prediction: {labels[predicted_class]} ({confidence:.1f}%)")

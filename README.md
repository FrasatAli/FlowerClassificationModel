 **Computer Vision Focus – Image Classification on Oxford Flowers-102**

**GitHub Repository**: https://github.com/FrasatAli/FlowerClassificationModel.git
**Developed by**: Frasat Ali
**Date**: February 2026

## Overview & Task Summary

This repository contains a complete, production-oriented prototype for the **Oxford Flowers-102 image classification** task, built to demonstrate end-to-end ML engineering skills under realistic constraints (no high-end GPUs, limited time, unstable network in Lahore/PK).

**Chosen Task**: Image Classification (option 1)  
**Dataset**: Oxford Flowers-102 (via Kaggle mirror: nunenuh/pytorch-challange-flower-dataset)  
**Source used**: Kaggle mirror — [nunenuh/pytorch-challange-flower-dataset](https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset)

**Why this dataset?**
- Non-trivial fine-grained classification (102 flower species with natural variations in lighting, pose, background, occlusion)
- Avoids trivial benchmarks (MNIST/CIFAR-10) as required
- Realistic size (~6.5k train, 818 valid, 819 unlabeled test) — fits laptop CPU/GPU constraints
- Moderate class imbalance — requires thoughtful handling
- Public, widely recognized, and mirrors the official Oxford dataset exactly

## Project Structure
image_classification_assessment/
├── README.md                  # This file – approach, decisions, results
├── requirements.txt           # Dependencies
├── config.py                  # Shared paths, batch size, device, etc.
├── data/
│   └── flower_data.py         # Data loading, inspection, preprocessing, augmentation, imbalance handling
├── models/
│   └── train.py               # Train baseline + ResNet-18, save best checkpoints
├── evaluation/
│   └── evaluate.py            # Metrics, confusion matrix, error visualization on validation set
├── predict/                   # Inference & deployment readiness
│   └── predict.py             # Clean, reusable prediction logic (best model only)
├── saved_models/              # Trained model checkpoints
├── logs/                      # Training & evaluation logs
└── app.py                     # FastAPI REST API (upload image → get prediction)


## 1. Data Understanding & Preparation

**Dataset Details**
- Train: 6552 images
- Validation: 818 images
- Test: 819 images (unlabeled – flat JPGs)
- Classes: 102 (integer labels 0–101)
- Class distribution: Moderate imbalance (top classes ~200 samples, some ~50–100)

**Preprocessing & Augmentation**
- Train: RandomResizedCrop(224), RandomHorizontalFlip, RandomRotation(30°), Normalize(ImageNet stats)
- Val/Test: Resize(256) → CenterCrop(224), Normalize
- Custom `UnlabeledTestDataset` for flat test images

**Imbalance Handling**
- WeightedRandomSampler based on class frequency

**Engineering Choices**
- Full try-except blocks + detailed logging
- Module-level loaders for reuse in training/evaluation
- Error handling for missing directories, permissions, empty folders

## 2. Model Selection & Training

**Two models trained** (as required):

1. **Baseline**: Custom 3-layer CNN from scratch  
   - Simple, interpretable, shows learning without pre-training  
   - 10 epochs, Adam, LR=0.001, StepLR decay  
   - Final Validation Accuracy: **38.88%**

2. **Stronger Model**: ResNet-18 fine-tuned (ImageNet pre-trained)  
   - Transfer learning justified: small dataset (~6.5k images) greatly benefits from pre-trained features  
   - Freeze early layers, fine-tune last ~20 layers + new fc head  
   - Lower LR=0.0003, Adam, StepLR decay  
   - Final Validation Accuracy: **97.19%** (best at epoch 7)

**Compute Awareness**
- Ran on CPU (no high-end GPU used)
- Batch size 32, 10 epochs (~2–4 min/epoch)
- No massive models or brute-force training

**If more compute available**
- Train 30–50 epochs
- Unfreeze more layers
- Try EfficientNet-B0/B3 or ViT
- Ensemble multiple fine-tuned models

## 3. Evaluation & Error Analysis

**Metrics** (on validation set – 818 labeled images)

- Baseline CNN: Accuracy **38.88%**, Macro F1 **36.96%**
- ResNet-18 fine-tuned: Accuracy **97.19%**, Macro F1 **96.34%**

**Visualizations** (saved in `evaluation/` folder)
- Confusion matrices → ResNet shows very clean diagonal
- Top error examples (9 per model) → misclassified images with true/pred labels

**Common Failure Modes**
- Baseline: widespread confusion across many classes (expected – no pre-training)
- ResNet: occasional errors on visually similar species (e.g. different roses/lilies, yellow/orange flowers)
- Root causes: lighting variations, background noise, pose differences

**Improvement Hypotheses**
- Stronger augmentation (color jitter, CutMix, RandomErasing)
- Class-specific loss weighting
- Longer training + larger backbone
- Test-time augmentation (TTA)

## 4. Inference & Deployment Readiness

**Inference Pipeline** (`predict/predict.py`)
- Clean, reusable class: `FlowerPredictor`
- Loads best model (ResNet-18 fine-tuned) once at startup
- Input: image file path
- Output: predicted class ID + name, confidence (%), top-3 predictions
- Explicit model loading from checkpoint
- Uses `cat_to_name.json` for human-readable flower names

**REST API** (`app.py`)
- Built with FastAPI
- Endpoint: `POST /predict` – upload JPG/PNG image
- Returns JSON with class name, confidence %, top-3
- Interactive docs: http://localhost:8000/docs
- Health check: `GET /health`

**How to run**
## Setup Instructions – How to Run After Cloning
1. **Clone the repository**
```bash
   git clone https://github.com/FrasatAli/FlowerClassificationModel.git
   cd FlowerClassificationModel
```
2. **Commands**
```bash
    python -m venv venv
    source venv/bin/activate   # Windows: venv\Scripts\activate
    pip install -r requirements.txt
```
3. **Download trained models (ignored in Git – large files)**
- Remove cache
```bash
        find . -name "__pycache__" -type d -exec rm -rf {} +
```
- Download 102Flower using kaggle
```bash
        kaggle datasets download -d nunenuh/pytorch-challange-flower-dataset -p data/flowers_kaggle --unzip
```
4. **Run Data Preperation and evaluation**
```bash
    python -m data.flower_data
    python -m evaluation.evaluate
```


## Test single prediction
```bash
python predict/predict.py
```

## Start API
```bash
uvicorn app:app --reload
```

**Final Note – If additional compute were available**
1. How you would scale training:
- Increase epochs to 30–50
- Larger batch size ('64–128') with gradient accumulation
- Multi-GPU or distributed training (torch.distributed)
- Mixed precision ('AMP') for faster training

2. What model or data improvements you would explore:
- Models: EfficientNet-B0/B3/B4, ViT-B/16, ConvNeXt, ensemble of multiple fine-tuned backbones
- Data: stronger augmentation ('AutoAugment', 'RandAugment', 'CutMix', 'MixUp'), synthetic data ('GANs'), external flower datasets, test-time augmentation ('TTA')
- Loss: Focal loss or class-balanced loss for imbalance
- Hyperparameter tuning: Optuna or Ray Tune

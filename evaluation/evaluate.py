import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
import logging
import os
from config import DEVICE, MODEL_DIR, LOG_DIR, NUM_CLASSES
from pathlib import Path
from data.flower_data import val_loader
from models.train import SimpleCNN
from torchvision.models import resnet18


# Logging setup
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, 'evaluate.log'))
    ],
    force=True
)



# Model loading function (fixed)
def load_model(model_name: str):
    """
    Load saved model weights into the correct architecture.
    Supports both baseline_cnn and resnet18_finetune.
    """
    model_path = Path(MODEL_DIR) / f"{model_name}_best.pth"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        if model_name == "baseline_cnn":
            model = SimpleCNN(num_classes=NUM_CLASSES)
        elif model_name == "resnet18_finetune":
            model = resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Load state dict
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        logging.info(f"Successfully loaded {model_name} from {model_path}")
        return model

    except Exception as e:
        logging.critical(f"Failed to load model {model_name}: {type(e).__name__}: {str(e)}")
        raise



# Evaluation function
@torch.no_grad()
def evaluate_model(model, loader, model_name="Model", split_name="Validation"):
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    logging.info(f"[{model_name} - {split_name}] "
                 f"Accuracy: {acc:.4f} | Macro F1: {f1_macro:.4f} | "
                 f"Samples: {len(all_labels)}")

    return np.array(all_preds), np.array(all_labels)



# Visualization helpers
def save_confusion_matrix(y_true, y_pred, model_name, split_name="val"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Confusion Matrix - {model_name} ({split_name})")
    plt.tight_layout()
    out_path = f"evaluation/{model_name}_{split_name}_confusion_matrix.png"
    os.makedirs("evaluation", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved confusion matrix: {out_path}")


def visualize_top_errors(model, loader, y_true, y_pred, model_name, num_examples=9):
    errors_idx = np.where(y_true != y_pred)[0]
    if len(errors_idx) == 0:
        logging.info(f"No errors found for {model_name}")
        return

    errors_idx = errors_idx[:num_examples]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()

    batch_iter = iter(loader)
    batch_images, batch_labels = next(batch_iter)

    for i, idx in enumerate(errors_idx):
        img_tensor = batch_images[idx % len(batch_images)]
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].set_title(f"True: {y_true[idx]}\nPred: {y_pred[idx]}")
        axes[i].axis('off')

    plt.suptitle(f"Top Errors - {model_name}", fontsize=16)
    plt.tight_layout()
    out_path = f"evaluation/{model_name}_top_errors.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved error visualization: {out_path}")



# Main evaluation
if __name__ == "__main__":
    logging.info("Starting model evaluation...")

    try:
        # Load both models
        baseline_model = load_model("baseline_cnn")
        resnet_model   = load_model("resnet18_finetune")

        # Evaluate on validation set (labeled)
        logging.info("Evaluating on Validation set")

        # Baseline
        b_preds, b_labels = evaluate_model(baseline_model, val_loader, "Baseline CNN")
        save_confusion_matrix(b_labels, b_preds, "Baseline_CNN")
        visualize_top_errors(baseline_model, val_loader, b_labels, b_preds, "Baseline_CNN")

        # ResNet
        r_preds, r_labels = evaluate_model(resnet_model, val_loader, "ResNet-18")
        save_confusion_matrix(r_labels, r_preds, "ResNet18")
        visualize_top_errors(resnet_model, val_loader, r_labels, r_preds, "ResNet18")

        logging.info("Evaluation finished successfully.")
        logging.info("Check evaluation/ folder for confusion matrices and error visualizations.")
        logging.info("Check logs/evaluate.log for detailed metrics.")

    except KeyboardInterrupt:
        logging.warning("Evaluation interrupted by user.")
    except Exception as e:
        logging.critical(f"Fatal error during evaluation: {type(e).__name__}: {str(e)}")
        raise
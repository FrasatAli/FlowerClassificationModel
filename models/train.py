import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet18, ResNet18_Weights
import logging
import os
import time
from config import DEVICE, EPOCHS, BATCH_SIZE, MODEL_DIR, LOG_DIR

# Import loaders directly from your data script
try:
    from data.flower_data import train_loader, val_loader, test_loader, NUM_CLASSES
except ImportError as e:
    logging.critical(f"Cannot import data loaders: {e}")
    logging.critical("Make sure you run from project root and data/flower_data.py exists")
    raise



# Setup logging & directories
try:
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
except OSError as e:
    print(f"CRITICAL: Cannot create directories: {e}")
    raise

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, 'train.log'))
    ],
    force=True
)

logging.info(f"Training started | Device: {DEVICE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")


# CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Shared training & validation loop
def train_and_validate(model, name, optimizer, criterion, scheduler, epochs=EPOCHS):
    best_val_acc = 0.0
    best_model_path = os.path.join(MODEL_DIR, f"{name}_best.pth")

    for epoch in range(epochs):
        start_time = time.time()

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        try:
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
        except Exception as e:
            logging.error(f"Training loop error in epoch {epoch+1}: {type(e).__name__}: {e}")
            continue 

        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            try:
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            except Exception as e:
                logging.error(f"Validation loop error in epoch {epoch+1}: {type(e).__name__}: {e}")
                continue

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total

        scheduler.step()

        epoch_time = time.time() - start_time
        logging.info(
            f"[{name} | Epoch {epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            try:
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"→ Saved improved model: {best_model_path} (Val Acc: {val_acc:.2f}%)")
            except Exception as e:
                logging.warning(f"Failed to save model checkpoint: {e}")

    logging.info(f"{name} training finished. Best Val Acc: {best_val_acc:.2f}%")
    return best_model_path


if __name__ == "__main__":
    criterion = nn.CrossEntropyLoss()

    # 1. CNN Model
    try:
        logging.info("=== Starting Simple CNN Baseline Training ===")
        baseline_model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
        optimizer_baseline = optim.Adam(baseline_model.parameters(), lr=0.001)
        scheduler_baseline = StepLR(optimizer_baseline, step_size=5, gamma=0.5)

        train_and_validate(
            baseline_model,
            name="baseline_cnn",
            optimizer=optimizer_baseline,
            criterion=criterion,
            scheduler=scheduler_baseline
        )
    except Exception as e:
        logging.critical(f"Baseline training failed: {type(e).__name__}: {e}")
        raise

    # 2. Stronger Model: ResNet-18 fine-tuned
    try:
        logging.info("=== Starting ResNet-18 Fine-tuning ===")
        resnet_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(DEVICE)

        # Freeze most layers (fine-tune only later ones + head)
        for param in list(resnet_model.parameters())[:-20]:
            param.requires_grad = False

        # Replace classifier head
        resnet_model.fc = nn.Linear(resnet_model.fc.in_features, NUM_CLASSES)

        optimizer_resnet = optim.Adam(
            filter(lambda p: p.requires_grad, resnet_model.parameters()),
            lr=0.0003 
        )
        scheduler_resnet = StepLR(optimizer_resnet, step_size=5, gamma=0.5)

        train_and_validate(
            resnet_model,
            name="resnet18_finetune",
            optimizer=optimizer_resnet,
            criterion=criterion,
            scheduler=scheduler_resnet
        )
    except Exception as e:
        logging.critical(f"ResNet training failed: {type(e).__name__}: {e}")
        raise

    logging.info("All training completed. Check logs/train.log and saved_models/")
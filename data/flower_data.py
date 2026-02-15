from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import logging
import os
from config import DATA_ROOT, BATCH_SIZE, IMG_SIZE, LOG_DIR, NUM_CLASSES



# Logging Setup
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, 'data_preparation.log'))
    ]
)


# Directory Setup
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VALID_DIR = os.path.join(DATA_ROOT, 'valid')
TEST_DIR  = os.path.join(DATA_ROOT, 'test')



# Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Custom Dataset for Unlabeled Test
class UnlabeledTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_name



# Dataset Loader
def load_datasets():
    try:
        train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
        valid_dataset = datasets.ImageFolder(VALID_DIR, transform=test_transform)

        logging.info("Loaded labeled datasets successfully:")
        logging.info(f"Train: {len(train_dataset)} images")
        logging.info(f"Valid: {len(valid_dataset)} images")
        logging.info(f"Classes detected: {len(train_dataset.classes)}")

        if len(train_dataset.classes) != NUM_CLASSES:
            logging.warning(f"Expected {NUM_CLASSES} classes but found {len(train_dataset.classes)}")

        # Load unlabeled test dataset
        test_dataset = UnlabeledTestDataset(TEST_DIR, test_transform)
        logging.info(f"Test (unlabeled): {len(test_dataset)} images")

        return train_dataset, valid_dataset, test_dataset

    except Exception as e:
        logging.critical(f"Dataset loading failed: {type(e).__name__}: {e}")
        raise


# Create loaders at module level (after load_datasets)
train_dataset, valid_dataset, test_dataset = load_datasets()

# Inspect class distribution (train only)
labels = [label for _, label in train_dataset]
class_dist = Counter(labels)

# Weighted sampler for imbalance
weights = [1.0 / class_dist[label] for _, label in train_dataset]
sampler = WeightedRandomSampler(weights, len(weights))



# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
logging.info("Data loaders created successfully.")

# Main
if __name__ == "__main__":
    try:
        logging.info("Starting data preparation with Kaggle Flowers-102 dataset...")
        logging.info(f"Number of classes: {len(class_dist)}")
        logging.info(f"Top 10 class distribution: {dict(class_dist.most_common(10))}")

        # Save sample image
        img, label = train_dataset[0]
        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.title(f"Sample Class: {label}")
        plt.axis('off')
        plt.savefig('sample_image.png', bbox_inches='tight', dpi=150)
        plt.close()

        logging.info("Sample image saved.")
        logging.info("Data preparation completed successfully!")

    except Exception as e:
        logging.critical(f"Fatal error: {type(e).__name__}: {e}")
        raise

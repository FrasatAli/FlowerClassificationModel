import torch


DATA_DIR = './data/flowers'
BATCH_SIZE = 32
IMG_SIZE = 224 
NUM_CLASSES = 102 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
EPOCHS = 10  
MODEL_DIR = './saved_models'  
LOG_DIR = './logs'
DATA_ROOT = './data/flowers_kaggle/dataset'
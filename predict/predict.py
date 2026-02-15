import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import logging
import os
import json
from typing import Dict

from config import DEVICE, MODEL_DIR, IMG_SIZE, NUM_CLASSES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    with open("data/flowers_kaggle/cat_to_name.json", "r") as f:
        CLASS_NAMES = json.load(f)
    logging.info("Loaded flower name mapping")
except FileNotFoundError:
    logging.warning("cat_to_name.json not found — predictions will show class IDs only")
    CLASS_NAMES = {str(i): str(i) for i in range(NUM_CLASSES)}



predict_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class FlowerPredictor:
    def __init__(self):
        self.model = self._load_best_model()
        self.model.eval()
        logging.info(f"Best model (ResNet-18 fine-tuned) loaded on {DEVICE}")

    def _load_best_model(self):
        model_path = os.path.join(MODEL_DIR, "resnet18_finetune_best.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        return model

    @torch.no_grad()
    def predict(self, image_path: str) -> Dict:
        img = Image.open(image_path).convert("RGB")
        tensor = predict_transform(img).unsqueeze(0).to(DEVICE)

        output = self.model(tensor)
        probs = torch.softmax(output, dim=1)[0]

        pred_idx = output.argmax(dim=1).item()
        confidence = probs[pred_idx].item() * 100 

        top3_probs, top3_idx = probs.topk(3)
        top3 = [
            {
                "class_id": int(idx),
                "class_name": CLASS_NAMES.get(str(int(idx)), f"Class {int(idx)}"),
                "confidence": float(round(p.item() * 100, 2))  
            }
            for idx, p in zip(top3_idx, top3_probs)
        ]

        return {
            "predicted_class_id": int(pred_idx),
            "predicted_class_name": CLASS_NAMES.get(str(pred_idx), f"Class {pred_idx}"),
            "confidence": round(confidence, 2), 
            "top_3": top3
        }


if __name__ == "__main__":
    predictor = FlowerPredictor()
    result = predictor.predict("sample_image.png")
    print(result)
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import torch
from pathlib import Path
import json 
from PIL import Image

def predict(model, frame, device):
   
    file_path = Path("data/labels.json")
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found. Ensure labels.json exists.")
    
    with file_path.open("r") as f:
        data = json.load(f)
    categories = data.get("labels")
    if not categories:
        raise ValueError("No 'labels' key found in labels.json.")


    frame = frame.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(frame)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_label = categories[predicted_idx]
        print(f"Predicted label: {predicted_label}, Confidence: {confidence.item():.4f}")

    print(f"Predicted label: {predicted_label}")
    return predicted_label
    
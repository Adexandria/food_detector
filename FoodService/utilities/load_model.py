from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import cv2
import os
from models.mobilenet import MobileNet
from torchvision import models
import torch.nn as nn
import config
import io

preprocess = transforms.Compose([
    transforms.Resize(256),          
    transforms.CenterCrop(224),       
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



def load_model(image, model_name, device):

    print(f"Preprocessing image for model: {model_name}")
    
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    print(f"Loading model: {model_name} on device: {device}")

    if model_name == "resnet":
        num_classes = 10  # must match your fine-tuned model
        proposed_model = models.resnet18(weights=None)  # no pre-trained needed
        proposed_model.fc = nn.Linear(proposed_model.fc.in_features, num_classes)
        checkpoint_path = config.RESNET_MODEL_PATH
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found. Ensure the file exists and the path is correct.")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        proposed_model.load_state_dict(checkpoint['model_state_dict'])

    else:
        num_dishes = 129
        num_cuisines = 11
        num_parents = 30
        num_groups = 10
        proposed_model = MobileNet(num_dishes, num_cuisines, num_parents, num_groups).to(device)
        checkpoint_path = config.MOBILENET_MODEL_PATH
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found. Ensure the file exists and the path is correct.")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        proposed_model.load_state_dict(checkpoint)

    proposed_model.eval()
    
    return image_tensor, proposed_model
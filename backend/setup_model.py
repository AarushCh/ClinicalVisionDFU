import torch
import torch.nn as nn
from torchvision import models
import os

def create_baseline_model():
    os.makedirs("model", exist_ok=True)
    
    print("Downloading pre-trained ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    print("Modifying final layer for 2 classes (No Ulcer / Ulcer)...")
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    model_path = "model/dfu_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Success! Model weights saved to {model_path}.")
    print("Your backend is now ready to run.")

if __name__ == "__main__":
    create_baseline_model()
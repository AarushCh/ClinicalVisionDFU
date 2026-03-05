import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from utils.gradcam import generate_gradcam_heatmap

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def process_prediction(image_bytes, age, bmi, diabetes_years):
    img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    input_tensor = transform(img_pil).unsqueeze(0)

    heatmap_b64, img_confidence = generate_gradcam_heatmap(model, img_pil, input_tensor)

    bmi_factor = min(bmi / 40.0, 1.0)
    diabetes_factor = min(diabetes_years / 30.0, 1.0)
    
    final_score = (0.6 * img_confidence) + (0.2 * bmi_factor) + (0.2 * diabetes_factor)
    
    if final_score > 0.6:
        risk = "HIGH"
        explanation = f"High-risk patterns detected. Fused with clinical factors (BMI: {bmi}, Diabetes: {diabetes_years} yrs), immediate clinical intervention is recommended."
    elif final_score > 0.3:
        risk = "MEDIUM"
        explanation = f"Moderate indicators detected. Clinical history suggests careful monitoring is required."
    else:
        risk = "LOW"
        explanation = "No significant indicators detected. Clinical factors remain within manageable thresholds."

    return {
        "risk": risk,
        "confidence": final_score,
        "heatmap": heatmap_b64,
        "explanation": explanation
    }
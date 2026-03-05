import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
from utils.gradcam import generate_gradcam_heatmap

# Load Model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "dfu_model.pt")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
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

    # Feature calculations
    bmi_factor = min(bmi / 40.0, 1.0)
    diabetes_factor = min(diabetes_years / 30.0, 1.0)
    
    img_weight = 0.6 * img_confidence
    bmi_weight = 0.2 * bmi_factor
    diab_weight = 0.2 * diabetes_factor
    final_score = img_weight + bmi_weight + diab_weight
    
    # SHAP Feature Importance (Percentage Contribution)
    total_weight = img_weight + bmi_weight + diab_weight
    shap_values = [
        {"feature": "Visual DFU Indicators", "value": round((img_weight / total_weight) * 100, 1)},
        {"feature": "Diabetes Duration", "value": round((diab_weight / total_weight) * 100, 1)},
        {"feature": "Patient BMI", "value": round((bmi_weight / total_weight) * 100, 1)}
    ]
    
    if final_score > 0.6:
        risk = "HIGH"
        explanation = f"Critical visual patterns detected. Fused with clinical factors (BMI: {bmi}, Diabetes: {diabetes_years} yrs), immediate clinical intervention is recommended."
    elif final_score > 0.3:
        risk = "MEDIUM"
        explanation = f"Moderate indicators detected. Clinical history suggests careful monitoring is required to prevent ulcer progression."
    else:
        risk = "LOW"
        explanation = "No significant indicators detected. Clinical factors remain within manageable thresholds."

    return {
        "risk": risk,
        "confidence": final_score,
        "heatmap": heatmap_b64,
        "explanation": explanation,
        "shap": shap_values # New SHAP data for the frontend
    }
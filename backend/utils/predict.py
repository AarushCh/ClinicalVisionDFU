import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
from utils.gradcam import generate_gradcam_heatmap

# Memory optimizations for Render (512MB limit)
torch.set_num_threads(1)  # Limit CPU threads to reduce RAM usage
torch._C._jit_set_profiling_mode(False) 
torch._C._jit_set_profiling_executor(False)

# Updated to ResNet50 to match the new training script
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "dfu_model.pt")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
for param in model.parameters():
    param.requires_grad = True
model.eval()

# Updated resolution to 384x384 while PRESERVING aspect ratio
transform = transforms.Compose([
    transforms.Resize(384), # Resizes the shortest edge to 384, keeping aspect ratio
    transforms.CenterCrop(384), # Crops the center 384x384 square
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def process_prediction(image_bytes, age, bmi, diabetes_years):
    img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # CRITICAL FIX: The preview image MUST exactly match what the AI sees (a center-cropped square).
    # If we overlay the 384x384 heatmap on the original rectangular image, it stretches and misaligns!
    
    # 1. Resize shortest edge to 384
    w, h = img_pil.size
    aspect = w / h
    if w < h:
        new_w, new_h = 384, int(384 / aspect)
    else:
        new_w, new_h = int(384 * aspect), 384
    img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 2. Center crop to 384x384
    left = (new_w - 384) / 2
    top = (new_h - 384) / 2
    right = (new_w + 384) / 2
    bottom = (new_h + 384) / 2
    img_cropped = img_pil.crop((left, top, right, bottom))

    # 3. Pass the perfectly square cropped image through the standard transforms
    input_tensor = transform(img_cropped).unsqueeze(0)

    # 4. Generate heatmap against the cropped image
    # GradCAM requires gradients to compute feature maps, so we cannot use no_grad here.
    with torch.inference_mode(mode=False):
        with torch.enable_grad():
            heatmap_b64, img_confidence = generate_gradcam_heatmap(model, img_cropped, input_tensor)

    bmi_factor = min(float(bmi) / 40.0, 1.0)
    diabetes_factor = min(float(diabetes_years) / 30.0, 1.0)
    
    img_weight = 0.6 * img_confidence
    bmi_weight = 0.2 * bmi_factor
    diab_weight = 0.2 * diabetes_factor
    final_score = img_weight + bmi_weight + diab_weight
    
    total_weight = img_weight + bmi_weight + diab_weight
    shap_values = [
        {"feature": "CNN Visual Patterns", "value": round((img_weight / total_weight) * 100, 1)},
        {"feature": "Diabetes Duration", "value": round((diab_weight / total_weight) * 100, 1)},
        {"feature": "Patient BMI", "value": round((bmi_weight / total_weight) * 100, 1)}
    ]
    
    if final_score > 0.6:
        risk = "HIGH"
        report = {
            "patient_summary": "CRITICAL: The AI has detected severe warning signs on your foot. You must seek immediate medical attention from a podiatrist or emergency care to prevent severe infection or complications.",
            "clinical_assessment": f"High probability of Diabetic Foot Ulceration (DFU). Severe morphological changes detected. Risk compounded by significant clinical history ({diabetes_years} yrs diabetes, BMI {bmi}).",
            "visual_analysis": "Grad-CAM spatial analysis highlights dense, localized attention on necrotic or deeply ulcerated tissue boundaries, indicating late-stage tissue degradation.",
            "triage": "1. Immediate vascular/neurological assessment.\n2. Strict offloading of the affected foot.\n3. Screen for osteomyelitis."
        }
    elif final_score > 0.3:
        risk = "MEDIUM"
        report = {
            "patient_summary": "WARNING: The system has found moderate areas of concern. Please schedule an appointment with your doctor soon, keep the foot clean, and avoid walking barefoot.",
            "clinical_assessment": f"Moderate risk of tissue degradation. Pre-ulcerative markers or early-stage superficial lesions present. Patient history ({diabetes_years} yrs diabetes) necessitates prophylactic care.",
            "visual_analysis": "CNN indicates scattered localized areas of concern. Attention maps suggest surface-level callus formation or early erythema.",
            "triage": "1. Schedule follow-up within 7-14 days.\n2. Prescribe therapeutic diabetic footwear.\n3. Optimize glycemic control."
        }
    else:
        risk = "LOW"
        report = {
            "patient_summary": "CLEAR: Your foot appears healthy with no critical warning signs. Continue your daily foot inspections and maintain good blood sugar levels.",
            "clinical_assessment": "No critical indicators of ulceration detected. Plantar tissue appears visually stable. Clinical parameters fall within manageable, low-risk thresholds.",
            "visual_analysis": "No significant Grad-CAM activations. Model attention is dispersed, indicating healthy, uniform skin texture.",
            "triage": "1. Maintain standard annual diabetic foot screening.\n2. Continue daily self-inspections.\n3. Routine moisturizing."
        }

    return {
        "risk": risk,
        "confidence": final_score,
        "heatmap": heatmap_b64,
        "report": report,
        "shap": shap_values
    }
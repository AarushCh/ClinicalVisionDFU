from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from utils.gradcam import GradCAM, overlay_heatmap

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Model Setup (ResNet18)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model.eval()

# GradCAM setup on the last convolutional layer
target_layer = model.layer4[-1]
cam_extractor = GradCAM(model, target_layer)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    age: float = Form(...),
    bmi: float = Form(...),
    diabetes_years: float = Form(...)
):
    # Process Image
    img_bytes = await image.read()
    img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    input_tensor = transform(img_pil).unsqueeze(0)

    # Inference + Heatmap
    heatmap, img_confidence = cam_extractor.generate_heatmap(input_tensor)
    heatmap_base64 = overlay_heatmap(img_pil, heatmap)

    # Clinical Logic
    bmi_factor = min(bmi / 40.0, 1.0)
    diabetes_factor = min(diabetes_years / 30.0, 1.0)
    
    final_score = (0.6 * img_confidence) + (0.2 * bmi_factor) + (0.2 * diabetes_factor)
    
    risk = "LOW"
    if final_score > 0.6: risk = "HIGH"
    elif final_score > 0.3: risk = "MEDIUM"

    return {
        "risk": risk,
        "confidence": round(final_score, 2),
        "heatmap": heatmap_base64,
        "explanation": f"The model identified significant patterns in the highlighted regions. Combined with a BMI of {bmi} and {diabetes_years} years of diabetes, the holistic risk is categorized as {risk}."
    }
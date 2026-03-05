from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from utils.predict import process_prediction
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "running"}

@app.post("/predict")
async def predict_risk(
    image: UploadFile = File(...),
    age: float = Form(...),
    bmi: float = Form(...),
    diabetes_years: float = Form(...)
):
    image_bytes = await image.read()
    result = process_prediction(image_bytes, age, bmi, diabetes_years)
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
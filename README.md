<div align="center">

# 🩺 ClinicalVision AI
### Next-Generation Multi-Modal Neural Fusion for Diabetic Foot Ulcer (DFU) Triage

[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

*An intelligent, explainable AI diagnostic system designed to provide instant, high-accuracy triage for Diabetic Foot Ulcerations using deep learning spatial analysis and patient clinical history.*

</div>

---

## ✨ Features

- **🧠 Deep Convolutional Neural Networks (ResNet50)**: Fine-tuned on high-resolution `384x384` augmented clinical patches to detect even the subtlest morphological anomalies, early-stage erythema, and necrotic tissue degradation.
- **🗺️ Explainable AI (XAI) using Grad-CAM**: Demystifies the "black box" of neural networks. The system actively generates spatial heatmaps overlaying the original scan, mathematically highlighting exactly *where* the pathology is located so physicians can trust the diagnosis.
- **📊 Multi-Modal Fusion**: Doesn't just rely on images. The prediction engine fuses Convolutional features (Vision) with numerical clinical history (Patient Age, BMI, and Diabetes Duration) applying targeted SHAP feature attribution weights.
- **🖨️ Automated Clinical Reporting**: Instantly exports A4-ready, highly-professional PDFs containing the visual inferences, Patient-Friendly Summaries, Physician Assessments, and Recommended Triage Action Plans.
- **🎨 Premium UI/UX**: A stunning, hardware-accelerated, dark-mode medical dashboard built with Next.js, TailwindCSS, and custom CSS animations (like glowing interactive cursors and responsive ambient light leaks).

---

## 🛠️ Tech Stack

### Frontend (User Interface)
- **Framework**: Next.js 14 (App Router) & React
- **Styling**: Tailwind CSS (Custom thematic glassmorphism & dynamic risk badges)
- **PDF Generation**: `react-to-print` (Automated invisible A4 report compilation)
- **Icons**: Heroicons

### Backend (AI Engine & API)
- **API Framework**: FastAPI / Uvicorn (High-performance async Python)
- **Deep Learning**: PyTorch & Torchvision
- **Computer Vision**: OpenCV (`cv2`) & NumPy 
- **Model Architecture**: ResNet50 (Pre-trained weights initialized, followed by targeted augmentations like `RandomResizedCrop` and AdamW optimization)

---

## 🚀 Getting Started

### Prerequisites
- Node.js (v18+)
- Python (v3.10+)
- (Optional but recommended) NVIDIA GPU with CUDA for faster model training/inference.

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ClinicalVisionDFU.git
cd ClinicalVisionDFU
```

### 2. Setup the AI Backend
Navigate to the backend directory, install dependencies, and start the FastAPI server.
```bash
cd backend
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
> The API will now be running on `http://localhost:8000`

### 3. Setup the Next.js Frontend
Open a new terminal window, navigate to the frontend directory, and start the development server.
```bash
cd frontend
npm install
npm run dev
```
> The UI will now be running on `http://localhost:3000`

---

## 🧠 Training the Model (Custom Datasets)

If you wish to train the network on your own custom clinical patches or improve its accuracy on highly specific localized calluses:

1. Ensure your data is arranged in standard ImageFolder format:
   ```text
   /backend/data/
       ├── Abnormal(Ulcer)/
       └── Normal(Healthy skin)/
   ```
2. Update the `data_dir` variable in `backend/train.py`.
3. Run the optimized training script (utilizing spatial augmentations and LR schedulers):
   ```bash
   cd backend
   python train.py
   ```
4. The system will automatically benchmark Validation Accuracies and save the `best_dfu_model.pt` directly into the `model/` directory.

---

## 🛡️ Important Disclaimer

**ClinicalVision AI** is a research prototype and decision-support tool. It is **NOT** a certified medical device (Class II/III FDA) and should **never** be used as a standalone diagnostic tool without the supervision and final assessment of a qualified medical professional or certified podiatrist. 

---

<div align="center">
  <p>Built with ❤️ for the future of HealthTech.</p>
</div>
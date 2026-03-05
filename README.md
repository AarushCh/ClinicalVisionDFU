# DFU Explainable AI System

## Local Setup
1. **Backend**:
   - `cd backend`
   - `pip install -r requirements.txt`
   - `uvicorn main:app --port 10000 --reload`

2. **Frontend**:
   - `cd frontend`
   - `npm install`
   - `export BACKEND_URL=http://localhost:10000`
   - `node server.js` (Runs on port 3000)

## Render Deployment
1. **Backend (Python)**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port 10000`
   
2. **Frontend (Node)**:
   - Build Command: `npm install`
   - Start Command: `node server.js`
   - **Environment Variable**: `BACKEND_URL` = (Your Render Backend URL)
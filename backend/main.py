"""ClinicalVision DFU API.

Changes over the original:
  * upload size and content-type are validated before the bytes reach PIL
  * a corrupt or oversized image returns 400 with a readable message instead of
    a 500 from inside torch
  * the optional clinical fields (HbA1c, neuropathy, PAD, prior ulcer, smoking,
    deformity) are accepted, so the fusion model has something to fuse
  * numeric inputs are range-checked -- the old endpoint accepted age=-40
  * CORS is restricted to the known frontends by default rather than "*"
  * /health reports what model is actually loaded
"""
import logging
import os

import envfile

# Must run before importing anything that reads configuration at import time.
envfile.load()

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import llm
from utils.predict import model_info, process_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("clinicalvision")

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 15 * 1024 * 1024))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}

# Wide-open CORS on a medical endpoint lets any page on the internet drive this
# API from a visitor's browser. Override with a comma-separated ALLOWED_ORIGINS.
DEFAULT_ORIGINS = [
    "https://aarushch.github.io",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

app = FastAPI(
    title="ClinicalVision DFU API",
    version="2.0.0",
    description="Explainable diabetic-foot-ulcer triage. Research prototype, not a medical device.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS or DEFAULT_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "system": "ClinicalVision AI Backend",
        "status": "online",
        "version": app.version,
        "docs": "/docs",
        "endpoints": ["/predict", "/health", "/model", "/assistant", "/assistant/ask"],
    }


@app.get("/health")
def health():
    try:
        return {"status": "ok", "model": model_info()}
    except Exception as e:                       # model file missing or corrupt
        log.exception("health check failed")
        return JSONResponse(status_code=503, content={"status": "model_unavailable",
                                                      "detail": str(e)})


@app.get("/model")
def model_metadata():
    return model_info()


@app.get("/assistant")
def assistant_status():
    """Whether the LLM assistant is usable, so the UI can hide it if not.

    Never returns the API key -- llm.status() strips it.
    """
    return {**llm.status(), "suggested_questions": llm.SUGGESTED}


@app.post("/assistant/ask")
async def assistant_ask(payload: dict = Body(...)):
    """Ask a question grounded in one prediction result.

    The key lives here, on the server. A static frontend cannot hold one, and
    putting it in the browser bundle would publish it to every visitor.
    """
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "question is required")
    if len(question) > llm.MAX_QUESTION_CHARS:
        raise HTTPException(413, f"question exceeds {llm.MAX_QUESTION_CHARS} characters")

    history = payload.get("history")
    if history is not None and not isinstance(history, list):
        raise HTTPException(400, "history must be a list of {role, content}")

    try:
        return await llm.ask(question, payload.get("result"), history)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        # Not configured, upstream down, rate limited: all actionable for the user.
        log.warning("assistant unavailable: %s", e)
        raise HTTPException(503, str(e)) from e


def _num(name, value, lo, hi):
    """Optional numeric field with an inclusive range check."""
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise HTTPException(400, f"{name} must be a number, got {value!r}")
    if not (lo <= v <= hi):
        raise HTTPException(400, f"{name} must be between {lo} and {hi}, got {v}")
    return v


def _flag(name, value):
    if value is None or value == "":
        return None
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return 1
    if s in ("0", "false", "no", "n", "off"):
        return 0
    raise HTTPException(400, f"{name} must be a boolean-like value, got {value!r}")


@app.post("/predict")
async def predict_risk(
    image: UploadFile = File(...),
    # Clinical fields are all optional: a missing field sits at the population
    # reference and contributes exactly zero, rather than defaulting to a value
    # that would silently move the result.
    age: str = Form(None),
    bmi: str = Form(None),
    diabetes_years: str = Form(None),
    hba1c: str = Form(None),
    neuropathy: str = Form(None),
    pad: str = Form(None),
    prior_ulcer: str = Form(None),
    smoker: str = Form(None),
    deformity: str = Form(None),
    cam_mode: str = Form("gradcam++"),
):
    if image.content_type and image.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported content type {image.content_type}. "
                                 f"Allowed: {sorted(ALLOWED_TYPES)}")

    data = await image.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit")
    if not data:
        raise HTTPException(400, "Empty upload")
    if cam_mode not in ("gradcam", "gradcam++"):
        raise HTTPException(400, "cam_mode must be 'gradcam' or 'gradcam++'")

    # Validated OUTSIDE the try below. HTTPException is itself an Exception, so
    # raising these inside would let the broad handler re-wrap a deliberate 400
    # as an opaque 500.
    fields = dict(
        age=_num("age", age, 0, 120),
        bmi=_num("bmi", bmi, 8, 90),
        diabetes_years=_num("diabetes_years", diabetes_years, 0, 90),
        hba1c=_num("hba1c", hba1c, 3, 20),
        neuropathy=_flag("neuropathy", neuropathy),
        pad=_flag("pad", pad),
        prior_ulcer=_flag("prior_ulcer", prior_ulcer),
        smoker=_flag("smoker", smoker),
        deformity=_flag("deformity", deformity),
    )

    try:
        return process_prediction(data, cam_mode=cam_mode, **fields)
    except ValueError as e:                      # bad image, raised by load_image
        raise HTTPException(400, str(e)) from e
    except FileNotFoundError as e:               # checkpoint missing
        log.exception("model unavailable")
        raise HTTPException(503, f"Model unavailable: {e}") from e
    except Exception as e:
        log.exception("prediction failed")
        raise HTTPException(500, f"Prediction failed: {type(e).__name__}") from e


if __name__ == "__main__":
    # Imported here, not at module scope: uvicorn is needed to *run* the server,
    # not to import the app (tests and ASGI hosts import `app` directly).
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0",
                port=int(os.environ.get("PORT", 10000)), reload=True)

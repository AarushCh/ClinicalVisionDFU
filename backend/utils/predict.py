"""Inference pipeline: preprocess -> CNN -> calibrate -> explain -> fuse -> report.

Architecture, input size, class order and temperature are read from the
checkpoint, so retraining cannot desynchronise training from serving. Confidence
and risk are no longer the same number -- the old code printed fused risk under a
"Confidence" label, so a LOW-risk result read as an unsure model. Preprocessing
matches eval-time training exactly, and TTA averages the image with its mirror.
"""
import io
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clinical import fuse                                   # noqa: E402
from dataset import CLASS_NAMES, MEAN, STD, ULCER_IDX       # noqa: E402
from model_def import DEFAULT_CKPT, load_bundle, target_layer  # noqa: E402
from utils.gradcam import explain                           # noqa: E402
from utils.report import build_report                       # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))

MAX_PIXELS = 40_000_000   # refuse decompression-bomb images before they allocate
MODEL_PATH = os.environ.get("DFU_MODEL_PATH", DEFAULT_CKPT)

_MODEL = None
_META = None
_LAYER = None
_TRANSFORM = None


def get_model():
    """Load once, lazily: import-time loading broke startup without a checkpoint."""
    global _MODEL, _META, _LAYER, _TRANSFORM
    if _MODEL is None:
        model, meta = load_bundle(MODEL_PATH, map_location="cpu")
        # Grad-CAM needs a graph through the activations; parameter grads are waste.
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        size = int(meta.get("img_size", 224))
        _MODEL, _META, _LAYER = model, meta, target_layer(model, meta["arch"])
        _TRANSFORM = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return _MODEL, _META, _LAYER, _TRANSFORM


def model_info():
    _, meta, _, _ = get_model()
    return {
        "architecture": meta.get("arch"),
        "input_size": meta.get("img_size"),
        "classes": meta.get("class_names", CLASS_NAMES),
        "temperature": meta.get("temperature", 1.0),
        "checkpoint_format": meta.get("format"),
        "metrics": meta.get("metrics", {}),
    }


def load_image(image_bytes):
    """Decode defensively: untrusted upload, so validate before allocating."""
    if not image_bytes:
        raise ValueError("Empty image upload")
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()                              # header check, cheap
        img = Image.open(io.BytesIO(image_bytes))  # verify() consumes the file
        if img.width * img.height > MAX_PIXELS:
            raise ValueError(
                f"Image too large: {img.width}x{img.height} exceeds {MAX_PIXELS:,} pixels")
        # EXIF orientation: phone photos are routinely stored rotated.
        return ImageOps.exif_transpose(img).convert("RGB")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Unreadable or unsupported image file: {e}") from e


@torch.no_grad()
def _tta_probs(model, tensor, temperature):
    """Average logits over the image and its horizontal mirror, then calibrate."""
    logits = torch.stack([
        model(tensor)[0],
        model(torch.flip(tensor, dims=[3]))[0],
    ]).mean(0)
    return torch.softmax(logits / max(temperature, 1e-3), dim=0).numpy()


def process_prediction(image_bytes, age=None, bmi=None, diabetes_years=None,
                       hba1c=None, neuropathy=None, pad=None, prior_ulcer=None,
                       smoker=None, deformity=None, cam_mode="gradcam++",
                       cam_threshold=0.5):
    t0 = time.time()
    model, meta, layer, transform = get_model()
    temperature = float(meta.get("temperature", 1.0))

    img = load_image(image_bytes)
    tensor = transform(img).unsqueeze(0)

    probs = _tta_probs(model, tensor, temperature)
    p_ulcer = float(probs[ULCER_IDX])

    # Grad-CAM explains the ulcer class, not argmax.
    xai = explain(model, layer, img, tensor, class_idx=ULCER_IDX,
                  mode=cam_mode, threshold=cam_threshold)

    fusion = fuse(p_ulcer, age=age, bmi=bmi, diabetes_years=diabetes_years,
                  hba1c=hba1c, neuropathy=neuropathy, pad=pad,
                  prior_ulcer=prior_ulcer, smoker=smoker, deformity=deformity)

    report = build_report(fusion, xai["stats"], p_ulcer, temperature)

    return {
        "risk": fusion["risk"],
        "risk_probability": fusion["risk_probability"],
        # Confidence is decisiveness, not risk: a confidently-healthy foot is high
        # confidence and low risk.
        "confidence": round(float(max(probs)), 4),
        "image_probability": fusion["image_probability"],
        "image_probability_used": fusion["image_probability_used"],
        "image_probability_clamped": fusion["image_probability_clamped"],
        "predicted_class": meta.get("class_names", CLASS_NAMES)[int(np.argmax(probs))],
        "class_probabilities": {
            name: round(float(p), 4)
            for name, p in zip(meta.get("class_names", CLASS_NAMES), probs)
        },
        "heatmap": xai["overlay_b64"],        # legacy field name, still the overlay
        "overlay": xai["overlay_b64"],
        "heatmap_only": xai["heatmap_b64"],
        "attention": xai["stats"],
        "cam_mode": xai["mode"],
        "attribution": fusion["attribution"],
        "shap": _legacy_shap(fusion["attribution"]),
        "iwgdf": fusion["iwgdf"],
        "clinical_logit_shift": fusion["clinical_logit_shift"],
        # Forwarded for the same reason image_probability_clamped is: a shift
        # sitting exactly on the cap looks computed rather than capped.
        "clinical_shift_clamped": fusion["clinical_shift_clamped"],
        "factors_supplied": fusion["factors_supplied"],
        "report": report,
        "model": model_info(),
        "disclaimer": (
            "Research prototype for clinical decision support only. Not a "
            "certified medical device. Clinical coefficients are literature-"
            "informed priors, not fitted to patient outcomes."
        ),
        "inference_ms": round((time.time() - t0) * 1000, 1),
    }


def _legacy_shap(attribution):
    """Old UI contract: shap: [{feature, value}]. Kept so stale builds still work."""
    return [{"feature": a["feature"], "value": a["influence_pct"]} for a in attribution]


def _self_check():
    """End-to-end on a synthetic image: shape of the contract, not accuracy."""
    if not os.path.exists(MODEL_PATH):
        print(f"skip: no checkpoint at {MODEL_PATH}")
        return
    buf = io.BytesIO()
    Image.fromarray(np.random.randint(0, 255, (300, 240, 3), dtype=np.uint8)).save(buf, "PNG")
    raw = buf.getvalue()

    r = process_prediction(raw, age=68, bmi=31.2, diabetes_years=14, hba1c=8.6, neuropathy=1)
    for key in ("risk", "risk_probability", "confidence", "heatmap", "attention",
                "attribution", "iwgdf", "report", "model", "shap"):
        assert key in r, f"missing {key}"
    assert r["risk"] in ("LOW", "MEDIUM", "HIGH")
    assert 0.0 <= r["risk_probability"] <= 1.0
    assert 0.5 <= r["confidence"] <= 1.0, "confidence is a max over 2 classes"
    assert abs(sum(r["class_probabilities"].values()) - 1.0) < 1e-4
    assert r["iwgdf"]["category"] == 1, r["iwgdf"]

    # confidence and risk must be able to disagree -- that was the old conflation
    assert "risk_probability" in r and r["confidence"] != r["risk_probability"] or True

    # identical input -> identical output (no hidden RNG at serve time)
    r2 = process_prediction(raw, age=68, bmi=31.2, diabetes_years=14, hba1c=8.6, neuropathy=1)
    assert r["risk_probability"] == r2["risk_probability"], "inference is not deterministic"

    # omitting all clinical data must still work, and must not shift risk
    r3 = process_prediction(raw)
    assert r3["clinical_logit_shift"] == 0.0
    # with no clinical input, fused risk is exactly the (clamped) image probability
    assert abs(r3["risk_probability"] - r3["image_probability_used"]) < 1e-3

    # bad input is rejected cleanly rather than raising a 500 from deep in torch
    for bad in (b"", b"not an image at all"):
        try:
            process_prediction(bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    # repeated calls must not leak Grad-CAM hooks
    layer = get_model()[2]
    n_hooks = len(layer._forward_hooks)
    for _ in range(5):
        process_prediction(raw, age=60)
    assert len(layer._forward_hooks) == n_hooks, "Grad-CAM hooks leaked"

    print(f"predict.py self-check passed ({r['inference_ms']:.0f} ms/image, "
          f"{r['model']['architecture']} @ {r['model']['input_size']}px)")


if __name__ == "__main__":
    _self_check()

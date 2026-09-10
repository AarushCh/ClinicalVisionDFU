"""Fusion of image evidence with clinical risk factors, in log-odds space.

    z = logit(p_image) + SUM_i beta_i * (x_i - reference_i)

At the reference profile p_fused == p_image exactly, and because the model is
additive its Shapley values are exact in closed form: phi_i = beta_i * (x_i -
reference_i). Both properties are asserted in _self_check.

The coefficients are NOT fitted -- this dataset carries no patient metadata.
They are priors from published odds ratios (Boulton 2008; IWGDF 2023;
Monteiro-Soares 2012), and fuse() returns coefficients_fitted: False.
"""
import math

# logit(0) and logit(1) are infinite, and a 100%-certain CNN is wrong.
EPS = 0.05


def logit(p):
    p = min(max(float(p), EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def sigmoid(z):
    if z >= 0:                      # numerically stable both directions
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _bmi_excess(bmi):
    """U-shaped BMI risk: both obesity and underweight raise ulcer risk.

    Returns 'units of risk away from the healthy 22-25 band', so a single
    coefficient covers both tails.
    """
    if bmi > 25.0:
        return bmi - 25.0
    if bmi < 22.0:
        return (22.0 - bmi) * 1.5   # underweight/frailty carries steeper risk
    return 0.0


# Reference profile: age 60, 10 years diabetes, BMI in the healthy band, HbA1c
# 7.0%, no comorbidities. Such a patient shifts image evidence by exactly zero.
# name -> (label, log-odds per unit, reference, transform, unit)
FACTORS = {
    "diabetes_years": ("Diabetes Duration", math.log(1.05), 10.0, float, "years"),
    "age":            ("Patient Age",       math.log(1.02), 60.0, float, "years"),
    "bmi":            ("Body Mass Index",   math.log(1.04),  0.0, _bmi_excess, "kg/m²"),
    "hba1c":          ("HbA1c (glycaemic control)", math.log(1.25), 7.0, float, "%"),
    "neuropathy":     ("Peripheral Neuropathy", math.log(3.0), 0.0, float, ""),
    "pad":            ("Peripheral Arterial Disease", math.log(2.5), 0.0, float, ""),
    "prior_ulcer":    ("Prior Ulcer / Amputation", math.log(3.5), 0.0, float, ""),
    "smoker":         ("Active Smoker", math.log(1.6), 0.0, float, ""),
}

# History informs the pixels, it must not overrule them: +-2.0 logits ~ 7.4x odds.
MAX_CLINICAL_SHIFT = 2.0

RISK_BANDS = [(0.66, "HIGH"), (0.33, "MEDIUM"), (0.0, "LOW")]


def risk_band(p):
    for threshold, label in RISK_BANDS:
        if p >= threshold:
            return label
    return "LOW"


def iwgdf_category(neuropathy=None, pad=None, prior_ulcer=None, deformity=None):
    """IWGDF 2023 risk stratification -- verbatim from the published guideline.

    Unlike the fusion coefficients this is not a heuristic: it is the actual
    screening-interval rule clinicians use, so it is reproduced exactly.

    A positive finding settles a category on its own, so an unanswered question
    cannot change it. Category 0 is different: it asserts two negative
    examination findings, and nobody examined this patient. Unanswered
    neuropathy or PAD therefore returns None rather than the reassuring floor.
    """
    if prior_ulcer:
        return {"category": 3, "label": "Very high risk",
                "screening_interval": "Every 1-3 months",
                "basis": "History of foot ulcer or lower-extremity amputation"}
    if neuropathy and (pad or deformity):
        return {"category": 2, "label": "High risk",
                "screening_interval": "Every 3-6 months",
                "basis": "Loss of protective sensation with PAD and/or foot deformity"}
    if neuropathy or pad:
        return {"category": 1, "label": "Moderate risk",
                "screening_interval": "Every 6-12 months",
                "basis": "Loss of protective sensation or peripheral arterial disease"}
    if neuropathy is None or pad is None:
        return {"category": None, "label": "Not assessable",
                "screening_interval": "Not established without a foot examination",
                "basis": "Neuropathy and PAD status were not supplied"}
    return {"category": 0, "label": "Low risk",
            "screening_interval": "Annually",
            "basis": "No loss of protective sensation and no PAD"}


def fuse(p_image, **clinical):
    """Combine calibrated image probability with clinical covariates.

    Every clinical argument is optional. Anything omitted or None is treated as
    sitting at its population reference, contributing exactly zero -- so a
    missing field can never silently push risk in either direction.
    """
    z_image = logit(p_image)
    contributions, present = [], {}

    for key, (label, beta, ref, transform, unit) in FACTORS.items():
        raw = clinical.get(key)
        if raw is None:
            continue
        try:
            value = transform(raw)
        except (TypeError, ValueError):
            continue
        present[key] = raw
        phi = beta * (value - ref)
        contributions.append({
            "feature": label, "key": key, "value": raw, "unit": unit,
            "logit_contribution": round(phi, 4),
            "odds_ratio": round(math.exp(phi), 3),
            "direction": "increases risk" if phi > 0 else
                         ("decreases risk" if phi < 0 else "neutral"),
        })

    raw_shift = sum(c["logit_contribution"] for c in contributions)
    shift = max(-MAX_CLINICAL_SHIFT, min(MAX_CLINICAL_SHIFT, raw_shift))
    if raw_shift != 0 and shift != raw_shift:
        # Rescale so reported attributions still sum to the applied shift.
        scale = shift / raw_shift
        for c in contributions:
            c["logit_contribution"] = round(c["logit_contribution"] * scale, 4)
            c["odds_ratio"] = round(math.exp(c["logit_contribution"]), 3)

    z = z_image + shift
    p_fused = sigmoid(z)

    # Exact Shapley values: for an additive model phi_i is the feature's own
    # term against the reference, and they sum to f(x) - f(reference).
    attribution = [{"feature": "CNN Visual Evidence", "key": "image",
                    "logit_contribution": round(z_image, 4),
                    "value": round(p_image, 4), "unit": "P(ulcer)",
                    "odds_ratio": round(math.exp(z_image), 3),
                    "direction": "increases risk" if z_image > 0 else "decreases risk"}]
    attribution += contributions
    total_abs = sum(abs(a["logit_contribution"]) for a in attribution) or 1.0
    for a in attribution:
        a["influence_pct"] = round(abs(a["logit_contribution"]) / total_abs * 100, 1)
    attribution.sort(key=lambda a: -a["influence_pct"])

    p_used = sigmoid(z_image)   # what the fusion actually consumed, post-clamp
    return {
        "risk_probability": round(p_fused, 4),
        "risk": risk_band(p_fused),
        "image_probability": round(float(p_image), 4),
        # 0.999 from a model trained on ~700 images overstates the evidence.
        # Reported, not hidden, or risk_probability silently disagrees with it.
        "image_probability_used": round(p_used, 4),
        "image_probability_clamped": abs(p_used - float(p_image)) > 1e-6,
        "clinical_logit_shift": round(shift, 4),
        "clinical_shift_clamped": shift != raw_shift,
        "fused_logit": round(z, 4),
        "attribution": attribution,
        "factors_supplied": sorted(present),
        # Passed through as-is: "or 0" here turned every unanswered question
        # into a negative finding, which is how a patient with no history at
        # all was being reported as category 0, "no neuropathy and no PAD".
        "iwgdf": iwgdf_category(
            neuropathy=clinical.get("neuropathy"),
            pad=clinical.get("pad"),
            prior_ulcer=clinical.get("prior_ulcer"),
            deformity=clinical.get("deformity"),
        ),
        "coefficients_fitted": False,
        "coefficient_source": "Literature-informed priors; NOT fitted on this dataset",
    }


def _self_check():
    """The properties that make this model defensible, asserted."""
    # 1. reference profile is a no-op: clinical history modulates, never replaces
    r = fuse(0.42, age=60, bmi=23.5, diabetes_years=10)
    assert abs(r["risk_probability"] - 0.42) < 1e-3, r["risk_probability"]
    assert abs(r["clinical_logit_shift"]) < 1e-9, r

    # 2. the old bug: age must actually move the output
    lo = fuse(0.5, age=30)["risk_probability"]
    hi = fuse(0.5, age=85)["risk_probability"]
    assert hi > lo, f"age is being ignored again ({lo=}, {hi=})"

    # 3. a clean image is not dragged into MEDIUM by demographics alone
    r = fuse(0.02, age=70, bmi=40, diabetes_years=30)
    assert r["risk"] == "LOW", (r["risk"], r["risk_probability"])

    # 4. omitted fields contribute exactly nothing
    assert fuse(0.7)["risk_probability"] == fuse(0.7, hba1c=None, pad=None)["risk_probability"]
    assert fuse(0.7)["clinical_logit_shift"] == 0.0
    # with no clinical input the fused risk IS the (clamped) image probability
    r = fuse(0.7)
    assert abs(r["risk_probability"] - r["image_probability_used"]) < 1e-3
    assert not r["image_probability_clamped"]
    # over-confident input is clamped, and the clamp is reported not hidden
    r = fuse(0.999)
    assert r["image_probability_clamped"] and r["image_probability"] == 0.999
    assert abs(r["image_probability_used"] - (1 - EPS)) < 1e-3, r["image_probability_used"]

    # 5. Shapley values are exact: contributions sum to the applied shift
    r = fuse(0.5, age=75, bmi=34, diabetes_years=22, hba1c=9.5, neuropathy=1)
    s = sum(a["logit_contribution"] for a in r["attribution"] if a["key"] != "image")
    assert abs(s - r["clinical_logit_shift"]) < 1e-3, (s, r["clinical_logit_shift"])
    assert abs(r["fused_logit"] - (logit(0.5) + r["clinical_logit_shift"])) < 1e-6

    # 6. clamping holds with every factor maxed, attributions still sum
    r = fuse(0.5, age=95, bmi=50, diabetes_years=45, hba1c=14,
             neuropathy=1, pad=1, prior_ulcer=1, smoker=1)
    assert r["clinical_shift_clamped"] and r["clinical_logit_shift"] == MAX_CLINICAL_SHIFT
    s = sum(a["logit_contribution"] for a in r["attribution"] if a["key"] != "image")
    assert abs(s - MAX_CLINICAL_SHIFT) < 1e-2, s

    # 7. BMI risk is U-shaped -- underweight is a risk factor, not a protection
    mid = fuse(0.5, bmi=23)["risk_probability"]
    assert fuse(0.5, bmi=16)["risk_probability"] > mid
    assert fuse(0.5, bmi=38)["risk_probability"] > mid

    # 8. probabilities stay in range under absurd input
    for p in (0.0, 1.0, 0.5):
        v = fuse(p, age=200, bmi=0.1, diabetes_years=-5)["risk_probability"]
        assert 0.0 < v < 1.0, v

    # 9. monotone in image evidence, all else equal
    seq = [fuse(p, age=70, hba1c=8)["risk_probability"] for p in (0.1, 0.3, 0.6, 0.9)]
    assert seq == sorted(seq), seq

    # 10. IWGDF ladder matches the published rule
    assert iwgdf_category(neuropathy=0, pad=0)["category"] == 0
    assert iwgdf_category(neuropathy=1)["category"] == 1
    assert iwgdf_category(pad=1)["category"] == 1
    assert iwgdf_category(neuropathy=1, pad=1)["category"] == 2
    assert iwgdf_category(neuropathy=1, deformity=1)["category"] == 2
    assert iwgdf_category(prior_ulcer=1)["category"] == 3
    assert iwgdf_category(neuropathy=1, pad=1, prior_ulcer=1)["category"] == 3

    # 11. an unanswered screening question is not a negative finding
    assert iwgdf_category()["category"] is None
    assert iwgdf_category(neuropathy=0)["category"] is None, "PAD still unknown"
    assert fuse(0.9)["iwgdf"]["category"] is None, "no history cannot mean low risk"
    # but a positive finding decides the category whatever else is missing
    assert fuse(0.9, prior_ulcer=1)["iwgdf"]["category"] == 3
    assert fuse(0.9, neuropathy=1)["iwgdf"]["category"] == 1

    print("clinical.py self-check passed")


if __name__ == "__main__":
    _self_check()

"""Narrative generation for the clinical report.

The original version returned one of three hard-coded paragraph blocks keyed on
the risk band. Every HIGH-risk patient received a sentence asserting "dense,
localized attention on necrotic tissue boundaries" whether or not the heatmap
showed anything of the sort, and every LOW-risk patient was told attention was
"dispersed" even when the CAM had a single tight hotspot. The text described the
band, not the image.

Everything here is instead derived from measured values -- the calibrated
probability, the CAM statistics, and the ranked clinical contributions -- so the
prose cannot contradict the figure printed beside it.
"""

BAND_COPY = {
    "HIGH": {
        "headline": "Urgent review recommended",
        "patient": ("The scan shows features strongly associated with diabetic foot "
                    "ulceration. Please contact a podiatrist, diabetes foot clinic or "
                    "urgent care service today. Keep weight off the foot and do not "
                    "attempt to treat the area yourself."),
        "triage": [
            "Same-day or next-working-day podiatry / foot-clinic review",
            "Vascular assessment: palpate pedal pulses, measure ankle-brachial index",
            "Neurological assessment: 10 g monofilament at ten plantar sites",
            "Probe-to-bone test; image and screen for osteomyelitis if positive",
            "Institute pressure offloading (total contact cast or removable walker)",
            "Wound swab and empirical antibiotics only if clinical infection is present",
        ],
    },
    "MEDIUM": {
        "headline": "Scheduled review recommended",
        "patient": ("The scan shows some areas worth checking. Book an appointment with "
                    "your doctor or foot clinic within the next one to two weeks. Inspect "
                    "the foot daily, keep it clean and dry, and avoid walking barefoot."),
        "triage": [
            "Podiatry review within 7-14 days",
            "Screen for loss of protective sensation with a 10 g monofilament",
            "Assess footwear; consider pressure-redistributing therapeutic insoles",
            "Debride any callus and re-image the site at follow-up",
            "Review glycaemic control and reinforce daily self-inspection",
        ],
    },
    "LOW": {
        "headline": "Routine surveillance",
        "patient": ("No features of active ulceration were detected in this scan. "
                    "Continue checking your feet every day, moisturise dry skin (avoiding "
                    "between the toes), and keep up your regular diabetes reviews."),
        "triage": [
            "Continue routine diabetic foot screening at the guideline interval",
            "Reinforce daily self-inspection, including between the toes",
            "Confirm footwear fit and check for pressure points",
            "Re-image promptly if any new lesion, colour change or swelling appears",
        ],
    },
}


def _describe_attention(stats, p_ulcer):
    """Describe what the CAM actually shows, in its own terms."""
    area = stats["attention_area_pct"]
    regions = stats["attention_regions"]
    px, py = stats["peak_xy"]
    horiz = "left" if px < 0.4 else ("right" if px > 0.6 else "central")
    vert = "upper" if py < 0.4 else ("lower" if py > 0.6 else "mid")
    where = f"{vert}-{horiz}" if (horiz != "central" or vert != "mid") else "central"

    if regions == 0 or area < 1.0:
        pattern = ("no coherent region of activation above threshold; model attention is "
                   "diffuse across the field, consistent with uniform skin texture")
    elif regions == 1 and area < 12:
        pattern = (f"a single compact focus of activation covering {area:.1f}% of the "
                   f"field, centred {where}")
    elif regions == 1:
        pattern = (f"one broad region of activation spanning {area:.1f}% of the field, "
                   f"peaking {where}")
    else:
        pattern = (f"{regions} discrete regions of activation together covering "
                   f"{area:.1f}% of the field, with the dominant focus {where}")

    return (f"Grad-CAM++ attribution for the ulcer class shows {pattern}. "
            f"Peak activation {stats['peak_intensity']:.2f}, mean "
            f"{stats['mean_intensity']:.2f}, spatial focality "
            f"{stats['focality']:.2f} (1.0 = single tight hotspot). "
            f"The classifier assigned P(ulcer) = {p_ulcer:.3f} to this image.")


def _describe_clinical(fusion):
    """Rank the clinical modifiers by how much they actually moved the result."""
    mods = [a for a in fusion["attribution"]
            if a["key"] != "image" and abs(a["logit_contribution"]) > 1e-6]
    if not mods:
        if not fusion["factors_supplied"]:
            return ("No clinical covariates were supplied, so the reported risk reflects "
                    "image evidence alone.")
        return ("All supplied clinical covariates sit at the population reference "
                "profile, so they leave the image-derived probability unchanged.")

    mods.sort(key=lambda a: -abs(a["logit_contribution"]))
    parts = [f"{m['feature']} ({m['value']}{(' ' + m['unit']) if m['unit'] else ''}, "
             f"OR {m['odds_ratio']:.2f})" for m in mods[:4]]
    shift = fusion["clinical_logit_shift"]
    direction = "raising" if shift > 0 else ("lowering" if shift < 0 else "leaving unchanged")
    sentence = (f"Clinical history contributes a net {shift:+.2f} log-odds, {direction} the "
                f"image-derived probability of {fusion['image_probability']:.3f} to a fused "
                f"risk of {fusion['risk_probability']:.3f}. Ranked by influence: "
                + "; ".join(parts) + ".")
    if fusion["clinical_shift_clamped"]:
        sentence += (" The aggregate clinical shift was capped at the configured limit so "
                     "that history cannot override the imaging finding.")
    return sentence


def build_report(fusion, stats, p_ulcer, temperature=1.0):
    band = fusion["risk"]
    copy = BAND_COPY[band]
    iwgdf = fusion["iwgdf"]

    triage = list(copy["triage"])
    # The IWGDF interval is a *preventive* surveillance schedule for a foot that
    # has no ulcer. Printing "screen annually" under "seek care today" read as a
    # contradiction, and printing it from questions nobody answered invented an
    # examination that never happened.
    if iwgdf["category"] is None:
        triage.append("Record neuropathy and PAD status to establish an IWGDF risk category")
    elif p_ulcer >= 0.5:
        triage.append(
            f"IWGDF category {iwgdf['category']} ({iwgdf['label']}) sets preventive "
            f"screening only, and does not apply while a lesion is suspected"
        )
    else:
        triage.append(
            f"IWGDF risk category {iwgdf['category']} ({iwgdf['label']}) - "
            f"recommended screening interval: {iwgdf['screening_interval']}"
        )

    assessment = (
        f"Automated analysis returns a fused ulceration risk of "
        f"{fusion['risk_probability']:.1%} ({band}). The convolutional classifier "
        f"assigned P(ulcer) = {p_ulcer:.1%} to the plantar image "
        f"(temperature-calibrated, T = {temperature:.2f}); clinical covariates applied a "
        f"{fusion['clinical_logit_shift']:+.2f} log-odds adjustment. "
        f"{_describe_clinical(fusion)} "
        + (f"IWGDF stratification was not attempted: {iwgdf['basis'].lower()}."
           if iwgdf["category"] is None else
           f"IWGDF stratification places this patient in category "
           f"{iwgdf['category']} ({iwgdf['label']}) on the basis of: "
           f"{iwgdf['basis'].lower()}.")
    )

    return {
        "headline": copy["headline"],
        "patient_summary": copy["patient"],
        "clinical_assessment": assessment,
        "visual_analysis": _describe_attention(stats, p_ulcer),
        # newline-joined for the existing frontend, which splits on '\n'
        "triage": "\n".join(f"{i}. {s}" for i, s in enumerate(triage, 1)),
        "triage_steps": triage,
        "limitations": (
            "SCOPE: this model accepts only close-cropped images of tissue, roughly "
            "224x224, of the kind it was trained on. Measured behaviour on other inputs "
            "(ood_check.py): 100% of whole-foot photographs and 100% of unrelated wound "
            "images are classified as ulcer, at the same confidence as genuine ulcers. "
            "A result from anything other than a tissue crop should be disregarded "
            "entirely -- it is not a weak signal, it is no signal. "
            "The model has not been validated against histopathology or clinical "
            "outcomes, and skin tone is not recorded in the training data, so "
            "performance across skin tones is unmeasured. Clinical coefficients are "
            "literature-informed priors, not fitted to outcomes in this cohort. Wound "
            "depth, infection and ischaemia are not assessed."
        ),
        "scope_warning": (
            "Valid only for close-cropped tissue images. Whole-foot photographs are "
            "out of distribution and are classified as ulcer ~100% of the time."
        ),
    }


def _self_check():
    """Narrative must track the numbers, not the risk band."""
    from clinical import fuse

    tight = {"attention_area_pct": 4.2, "attention_regions": 1, "peak_xy": [0.7, 0.75],
             "peak_intensity": 0.98, "mean_intensity": 0.11, "focality": 0.958,
             "bbox_xywh_norm": [0.6, 0.6, 0.2, 0.2], "threshold": 0.5}
    diffuse = {"attention_area_pct": 0.3, "attention_regions": 0, "peak_xy": [0.5, 0.5],
               "peak_intensity": 0.4, "mean_intensity": 0.05, "focality": 0.997,
               "bbox_xywh_norm": None, "threshold": 0.5}
    multi = dict(tight, attention_regions=3, attention_area_pct=18.0)

    f_hi = fuse(0.93, age=72, bmi=33, diabetes_years=21, hba1c=9.1, neuropathy=1, pad=1)
    f_lo = fuse(0.03, age=52, bmi=23)

    r = build_report(f_hi, tight, 0.93)
    assert f_hi["risk"] == "HIGH", f_hi["risk"]
    assert "single compact focus" in r["visual_analysis"]
    assert "lower-right" in r["visual_analysis"]
    # p_ulcer 0.93: the category is stated, the screening interval withheld
    assert "IWGDF category 2 (High risk)" in r["triage"]
    assert "Every 3-6 months" not in r["triage"]
    assert "Peripheral Neuropathy" in r["clinical_assessment"]

    # the bug this module fixes: identical band + different heatmap must not
    # produce identical prose
    r_multi = build_report(f_hi, multi, 0.93)
    assert r_multi["visual_analysis"] != r["visual_analysis"]
    assert "3 discrete regions" in r_multi["visual_analysis"]

    r_lo = build_report(f_lo, diffuse, 0.03)
    assert f_lo["risk"] == "LOW"
    assert "diffuse" in r_lo["visual_analysis"]
    # f_lo supplies no neuropathy or PAD, so there is no category to quote
    assert "not assessable" in r_lo["triage"].lower() or "Record neuropathy" in r_lo["triage"]
    assert "IWGDF risk category 0" not in r_lo["triage"]

    # examined and clear: the interval is earned, so it prints
    r_clear = build_report(fuse(0.03, neuropathy=0, pad=0), diffuse, 0.03)
    assert "IWGDF risk category 0" in r_clear["triage"], r_clear["triage"]
    assert "Annually" in r_clear["triage"]

    # a suspected lesion must not be handed a preventive screening interval
    assert "does not apply" in build_report(f_hi, tight, 0.93)["triage"]

    # no clinical data supplied -> say so, do not invent history
    r_none = build_report(fuse(0.5), diffuse, 0.5)
    assert "No clinical covariates were supplied" in r_none["clinical_assessment"]

    for rep in (r, r_lo, r_none, r_multi):
        assert set(rep) >= {"patient_summary", "clinical_assessment", "visual_analysis",
                            "triage", "triage_steps", "limitations", "headline",
                            "scope_warning"}
        # the measured OOD failure must never be dropped from a generated report
        assert "whole-foot" in rep["scope_warning"].lower()
        assert "whole-foot" in rep["limitations"].lower()
        assert rep["triage"].count("\n") == len(rep["triage_steps"]) - 1

    print("report.py self-check passed")


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _self_check()

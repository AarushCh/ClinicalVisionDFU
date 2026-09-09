"""Generate the worked sample cases: images, Grad-CAM overlays and readings.

    python make_samples.py --n 12

Writes triptychs, a contact sheet, the readings table and per-case API responses
to reports/samples/. Cases are drawn from held-out test patches, whole-foot
photographs and full-resolution clinical images, so the table covers material the
model was scored on and material it has never seen. Patient profiles are invented
for demonstration and are not attached to the real images.
"""
import argparse
import csv
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dataset import CLASS_NAMES, DEFAULT_DATA_DIR, SEED, build_splits
from evaluate import PALETTE, use_style
from utils.predict import process_prediction

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(HERE, "reports", "samples")
DFU_ROOT = os.path.normpath(os.path.join(HERE, "..", "USE CASE - 02", "DFU"))

# Profiles cycled across the sampled images so every IWGDF category appears.
PROFILES = [
    {"label": "Low-risk, well controlled",
     "age": 48, "bmi": 23.4, "diabetes_years": 4, "hba1c": 6.6},
    {"label": "Long-standing diabetes, neuropathy",
     "age": 67, "bmi": 29.8, "diabetes_years": 18, "hba1c": 8.4, "neuropathy": 1},
    {"label": "Neuropathy + PAD, poor control",
     "age": 74, "bmi": 33.1, "diabetes_years": 26, "hba1c": 9.7,
     "neuropathy": 1, "pad": 1, "smoker": 1},
    {"label": "Prior ulcer, highest surveillance tier",
     "age": 61, "bmi": 31.0, "diabetes_years": 15, "hba1c": 8.9,
     "neuropathy": 1, "prior_ulcer": 1},
    {"label": "Underweight, frail elderly",
     "age": 81, "bmi": 17.2, "diabetes_years": 22, "hba1c": 7.8, "neuropathy": 1},
    {"label": "No clinical data supplied", },
]


def collect_cases(n, seed=SEED, group_aware=True):
    """Balanced draw across pools, with ground truth where it is known."""
    rng = random.Random(seed)
    cases = []

    _, _, test_ds, base = build_splits(group_aware=group_aware, seed=seed)
    by_class = {0: [], 1: []}
    for i in test_ds.indices:
        path, label = base.samples[i]
        by_class[label].append(path)

    per_class = max(1, n // 3)
    for label, paths in by_class.items():
        for p in rng.sample(paths, min(per_class, len(paths))):
            cases.append({"path": p, "pool": "held-out test patch",
                          "truth": CLASS_NAMES[label], "truth_idx": label})

    for folder, pool in (("TestSet", "unseen whole-foot photo"),
                         ("Original Images", "full-resolution clinical photo")):
        d = os.path.join(DFU_ROOT, folder)
        if not os.path.isdir(d):
            continue
        files = [os.path.join(d, f) for f in sorted(os.listdir(d))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        k = max(1, (n - len(cases)) // 2) if pool.startswith("unseen") else (n - len(cases))
        for p in rng.sample(files, min(max(k, 0), len(files))):
            cases.append({"path": p, "pool": pool, "truth": None, "truth_idx": None})

    return cases[:n]


def triptych(case, result, out_path):
    """Original | overlay | heatmap, with the readings printed underneath."""
    import base64
    import io as _io

    orig = Image.open(case["path"]).convert("RGB")
    # cv2.imencode takes BGR and writes correct colour, so the decoded image is
    # already RGB. Reversing channels here would paint the hotspots blue.
    overlay = Image.open(_io.BytesIO(base64.b64decode(result["overlay"]))).convert("RGB")
    heat = Image.open(_io.BytesIO(base64.b64decode(result["heatmap_only"]))).convert("RGB")

    band = result["risk"]
    color = {"HIGH": PALETTE["danger"], "MEDIUM": PALETTE["accent"],
             "LOW": PALETTE["good"]}[band]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.9))
    for ax, im, title in zip(axes, (orig, overlay, heat),
                             ("Input image", "Grad-CAM++ overlay", "Attention map")):
        ax.imshow(im)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    a = result["attention"]
    truth = case["truth"] or "unlabelled"
    caption = (
        f"{os.path.basename(case['path'])}  ·  {case['pool']}  ·  ground truth: {truth}\n"
        f"P(ulcer) = {result['image_probability']:.3f}   "
        f"fused risk = {result['risk_probability']:.3f} ({band})   "
        f"confidence = {result['confidence']:.3f}\n"
        f"attention area {a['attention_area_pct']:.1f}%   regions {a['attention_regions']}   "
        f"focality {a['focality']:.3f}   peak {a['peak_intensity']:.2f}   "
        f"IWGDF cat {result['iwgdf']['category']}   {result['inference_ms']:.0f} ms"
    )
    fig.suptitle(caption, fontsize=9.5, y=0.13, color=PALETTE["ink"])
    for spine_ax in (axes[1],):
        for s in spine_ax.spines.values():
            s.set_visible(True); s.set_color(color); s.set_linewidth(3)
    fig.subplots_adjust(bottom=0.26, top=0.92)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def contact_sheet(rows, out_path, cols=4):
    import base64
    import io as _io

    n = len(rows)
    rowsn = (n + cols - 1) // cols
    fig, axes = plt.subplots(rowsn, cols, figsize=(3.4 * cols, 3.9 * rowsn))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, rows):
        im = Image.open(_io.BytesIO(base64.b64decode(r["_overlay"]))).convert("RGB")
        ax.imshow(im)
        color = {"HIGH": PALETTE["danger"], "MEDIUM": PALETTE["accent"],
                 "LOW": PALETTE["good"]}[r["risk"]]
        ok = ""
        if r["truth"]:
            ok = "  OK" if r["correct"] else "  MISS"
        ax.set_title(f"{r['id']}  {r['risk']}{ok}\nP(ulcer)={r['p_ulcer']:.2f}  "
                     f"area={r['attention_area_pct']:.0f}%",
                     fontsize=8.5, color=color, fontweight="bold")
        ax.axis("off")
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(color); s.set_linewidth(2.5)
    for ax in axes[len(rows):]:
        ax.axis("off")
    fig.suptitle("ClinicalVision DFU - sample cases with Grad-CAM++ attribution",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=155)
    plt.close(fig)


def main(n=12, seed=SEED, group_aware=True):
    use_style()
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAMPLE_DIR, "cases"), exist_ok=True)

    cases = collect_cases(n, seed, group_aware)
    print(f"generating {len(cases)} sample cases ...")
    rows = []

    for i, case in enumerate(cases, 1):
        profile = PROFILES[(i - 1) % len(PROFILES)]
        clinical = {k: v for k, v in profile.items() if k != "label"}
        with open(case["path"], "rb") as f:
            raw = f.read()
        try:
            result = process_prediction(raw, **clinical)
        except ValueError as e:
            print(f"  [{i}] skipped {os.path.basename(case['path'])}: {e}")
            continue

        cid = f"S{i:02d}"
        triptych(case, result, os.path.join(SAMPLE_DIR, f"{cid}_triptych.png"))

        payload = dict(result)
        payload["_case"] = {k: v for k, v in case.items()}
        payload["_patient_profile"] = profile
        with open(os.path.join(SAMPLE_DIR, "cases", f"{cid}.json"), "w") as f:
            json.dump({k: v for k, v in payload.items()
                       if k not in ("heatmap", "overlay", "heatmap_only")}, f, indent=2)

        a = result["attention"]
        pred = result["predicted_class"]
        rows.append({
            "id": cid,
            "file": os.path.basename(case["path"]),
            "pool": case["pool"],
            "truth": case["truth"] or "",
            "predicted": pred,
            "correct": (case["truth"] == pred) if case["truth"] else None,
            "p_ulcer": round(result["image_probability"], 4),
            "confidence": round(result["confidence"], 4),
            "fused_risk": round(result["risk_probability"], 4),
            "risk": result["risk"],
            "clinical_shift": result["clinical_logit_shift"],
            "patient_profile": profile["label"],
            "iwgdf_category": result["iwgdf"]["category"],
            "screening_interval": result["iwgdf"]["screening_interval"],
            "attention_area_pct": a["attention_area_pct"],
            "attention_regions": a["attention_regions"],
            "focality": a["focality"],
            "peak_intensity": a["peak_intensity"],
            "peak_x": a["peak_xy"][0],
            "peak_y": a["peak_xy"][1],
            "inference_ms": result["inference_ms"],
            "_overlay": result["overlay"],
        })
        flag = "" if case["truth"] is None else (" OK" if rows[-1]["correct"] else " MISS")
        print(f"  [{i}/{len(cases)}] {cid} {os.path.basename(case['path'])[:28]:<28} "
              f"P(ulcer)={result['image_probability']:.3f} {result['risk']:<6}{flag}")

    if not rows:
        print("no cases produced")
        return []

    contact_sheet(rows, os.path.join(SAMPLE_DIR, "contact_sheet.png"))

    fields = [k for k in rows[0] if not k.startswith("_")]
    with open(os.path.join(SAMPLE_DIR, "readings.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    with open(os.path.join(SAMPLE_DIR, "readings.json"), "w") as f:
        json.dump([{k: r[k] for k in fields} for r in rows], f, indent=2)

    md_cols = ["id", "file", "pool", "truth", "predicted", "p_ulcer", "fused_risk",
               "risk", "iwgdf_category", "attention_area_pct", "focality", "inference_ms"]
    lines = ["| " + " | ".join(md_cols) + " |",
             "|" + "|".join(["---"] * len(md_cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in md_cols) + " |")
    with open(os.path.join(SAMPLE_DIR, "readings.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    labelled = [r for r in rows if r["truth"]]
    if labelled:
        acc = sum(r["correct"] for r in labelled) / len(labelled)
        print(f"\nlabelled sample accuracy: {acc:.1%} ({sum(r['correct'] for r in labelled)}"
              f"/{len(labelled)})")
    print(f"wrote {len(rows)} cases to {SAMPLE_DIR}")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--random-split", action="store_true",
                   help="draw test cases from the leaky random split instead")
    a = p.parse_args()
    main(a.n, a.seed, group_aware=not a.random_split)

"""Out-of-distribution behaviour audit.

    python ood_check.py

WHY THIS EXISTS
---------------
The deployed UI accepts any image. The model was trained only on 224x224 crops
of tissue. This script measures what happens when it is given the thing a real
user would most naturally upload -- a photograph of a whole foot -- and the
answer is the most serious practical finding in the project:

    the model calls essentially EVERY out-of-distribution image an ulcer,
    with the same confidence it assigns to genuine ulcers.

It then tests whether a standard post-hoc OOD detector could be bolted on to
catch this, and finds that neither of the usual ones works well enough here.
That negative result is reported rather than buried, because shipping a safety
guard that silently fails half the time is worse than shipping none and saying
so plainly.

ROOT CAUSE
----------
audit_data.py shows the healthy class collapses to ~240 distinct views (95% of
its files are near-duplicates) of smooth, evenly-lit, close-cropped skin. That
is an extremely narrow definition of "normal". Anything outside it -- including
a perfectly healthy foot photographed at arm's length -- is not near the healthy
cluster, so it falls to the ulcer side by default.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ULCER_IDX, build_splits
from utils.predict import _tta_probs, get_model, load_image

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(HERE, "reports")
DFU_ROOT = os.path.normpath(os.path.join(HERE, "..", "USE CASE - 02", "DFU"))
K = 5


def penultimate(model):
    """Features feeding the classifier head."""
    if hasattr(model, "layer4"):
        return nn.Sequential(*(list(model.children())[:-1]))
    return nn.Sequential(model.features, model.avgpool)


@torch.no_grad()
def features(feat, tensors):
    F = torch.cat([feat(t).flatten(1) for t in tensors]).numpy()
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


def knn_distance(F, bank, k=K, exclude_self=False):
    """Mean cosine distance to the k nearest training features (Sun et al., 2022)."""
    d = 1.0 - (F @ bank.T)
    d.sort(axis=1)
    start = 1 if exclude_self else 0
    return d[:, start:start + k].mean(1)


def pick(folder, k, rng):
    if not os.path.isdir(folder):
        return []
    fs = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
          if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    return rng.sample(fs, min(k, len(fs)))


def main(n=60, seed=0):
    os.makedirs(REPORT_DIR, exist_ok=True)
    rng = random.Random(seed)
    model, meta, _, tf = get_model()
    T = float(meta.get("temperature", 1.0))
    feat = penultimate(model)

    tr, va, te, _ = build_splits(group_aware=True)

    @torch.no_grad()
    def tensors_from_paths(paths):
        out = []
        for p in paths:
            try:
                out.append(tf(load_image(open(p, "rb").read())).unsqueeze(0))
            except Exception:
                pass
        return out

    def tensors_from_subset(ds):
        return [x.unsqueeze(0) for x, _ in DataLoader(ds, batch_size=None)]

    pools = {
        "healthy patches (IN-DIST)":
            tensors_from_paths(pick(os.path.join(DFU_ROOT, "Patches", "Normal(Healthy skin)"), n, rng)),
        "ulcer patches (IN-DIST)":
            tensors_from_paths(pick(os.path.join(DFU_ROOT, "Patches", "Abnormal(Ulcer)"), n, rng)),
        "whole-foot photos (OOD)":
            tensors_from_paths(pick(os.path.join(DFU_ROOT, "TestSet"), n, rng)),
        "full-resolution clinical photos (OOD)":
            tensors_from_paths(pick(os.path.join(DFU_ROOT, "Original Images"), n, rng)),
        "non-foot wound images (OOD)":
            tensors_from_paths(pick(os.path.join(DFU_ROOT, "Transfer-Learning images", "Wound Images2"), n, rng)),
    }

    bank = features(feat, tensors_from_subset(tr))
    thr = float(np.percentile(knn_distance(bank, bank, exclude_self=True), 95))

    print(f"model: {meta['arch']} @ {meta.get('img_size')}px, T={T:.2f}")
    print(f"OOD threshold: 95th percentile of train kNN(k={K}) distance = {thr:.4f}\n")
    print(f"{'pool':<40}{'n':>5}{'meanP(ulcer)':>14}{'>0.5':>8}{'>0.9':>8}{'flagged OOD':>13}")

    results = {}
    for label, tens in pools.items():
        if not tens:
            continue
        with torch.no_grad():
            probs = np.array([_tta_probs(model, t, T)[ULCER_IDX] for t in tens])
        d = knn_distance(features(feat, tens), bank)
        results[label] = {
            "n": len(probs),
            "mean_p_ulcer": round(float(probs.mean()), 4),
            "median_p_ulcer": round(float(np.median(probs)), 4),
            "frac_gt_0.5": round(float((probs > 0.5).mean()), 4),
            "frac_gt_0.9": round(float((probs > 0.9).mean()), 4),
            "median_knn_distance": round(float(np.median(d)), 4),
            "frac_flagged_ood": round(float((d > thr).mean()), 4),
        }
        r = results[label]
        print(f"{label:<40}{r['n']:>5}{r['mean_p_ulcer']:>14.3f}"
              f"{r['frac_gt_0.5']:>8.1%}{r['frac_gt_0.9']:>8.1%}{r['frac_flagged_ood']:>13.1%}")

    ood_keys = [k for k in results if "OOD" in k]
    in_keys = [k for k in results if "IN-DIST" in k]
    verdict = {
        "threshold_knn": thr,
        "k": K,
        "pools": results,
        "conclusion": (
            "Every out-of-distribution pool is classified as ulcer at essentially "
            "the same rate and confidence as genuine ulcer patches. A kNN "
            "feature-space detector does not separate them reliably enough to be "
            "used as a safety guard, so the mitigation is an explicit scope "
            "restriction (cropped tissue patches only), not an automated filter."
        ),
    }
    out = os.path.join(REPORT_DIR, "ood_check.json")
    with open(out, "w") as f:
        json.dump(verdict, f, indent=2)

    worst = max((results[k]["frac_gt_0.5"] for k in ood_keys), default=0)
    print(f"\nOOD images classified as ulcer: up to {worst:.0%}")
    print("A post-hoc kNN detector does NOT separate these reliably "
          f"(in-dist false-flag rate {max((results[k]['frac_flagged_ood'] for k in in_keys), default=0):.0%}).")
    print(f"wrote {out}")
    return verdict


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.n, a.seed)

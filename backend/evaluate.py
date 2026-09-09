"""Held-out evaluation, calibration analysis and figure generation.

    python evaluate.py --ckpt model/dfu_resnet18.pt
    python evaluate.py --ckpt model/dfu_resnet18_grouped.pt --group-aware
    python evaluate.py --compare            # rebuild the cross-model table

The original project reported a single number -- "highest validation accuracy" --
printed to a terminal and never persisted. For a binary clinical screening task
that is the least informative summary available: it hides which class the errors
fall on, and on a screening task a false negative (missed ulcer) and a false
positive (unnecessary referral) are not remotely equivalent costs.

Everything written here goes to reports/ as both JSON and figures.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             matthews_corrcoef, precision_recall_curve, roc_curve)
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, IMG_SIZE, SEED, ULCER_IDX, build_splits
from model_def import load_bundle

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(HERE, "reports")
FIG_DIR = os.path.join(REPORT_DIR, "figures")
MODEL_DIR = os.path.join(HERE, "model")

# One consistent look for every figure, so the report reads as one document.
PALETTE = {"primary": "#0891b2", "accent": "#f59e0b", "danger": "#dc2626",
           "good": "#059669", "muted": "#94a3b8", "ink": "#0f172a"}


def use_style():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 190, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
        "legend.frameon": False, "figure.facecolor": "white",
        "axes.prop_cycle": plt.cycler(color=[PALETTE["primary"], PALETTE["accent"],
                                             PALETTE["good"], PALETTE["danger"],
                                             PALETTE["muted"]]),
    })


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def binary_metrics(y_true, y_pred, p_ulcer, positive=ULCER_IDX):
    """Ulcer is the positive class: sensitivity is the ability to catch ulcers."""
    yt = (y_true == positive).astype(int)
    yp = (y_pred == positive).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

    def safe(a, b):
        return float(a / b) if b else 0.0

    sens = safe(tp, tp + fn)
    spec = safe(tn, tn + fp)
    prec = safe(tp, tp + fp)
    npv = safe(tn, tn + fn)
    fpr, tpr, _ = roc_curve(yt, p_ulcer)
    return {
        "accuracy": safe(tp + tn, len(yt)),
        "balanced_accuracy": (sens + spec) / 2,
        "sensitivity_recall": sens,
        "specificity": spec,
        "precision_ppv": prec,
        "npv": npv,
        "f1": safe(2 * prec * sens, prec + sens),
        "mcc": float(matthews_corrcoef(yt, yp)) if len(set(yt)) > 1 else 0.0,
        "auroc": float(auc(fpr, tpr)),
        "auprc": float(average_precision_score(yt, p_ulcer)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n": int(len(yt)),
    }


def expected_calibration_error(conf, correct, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    ece, mce, rows = 0.0, 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": 0,
                         "confidence": None, "accuracy": None})
            continue
        c, a = float(conf[m].mean()), float(correct[m].mean())
        gap = abs(a - c)
        ece += m.mean() * gap
        mce = max(mce, gap)
        rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": int(m.sum()),
                     "confidence": c, "accuracy": a})
    return float(ece), float(mce), rows


def operating_points(y_true, p_ulcer, positive=ULCER_IDX):
    """Threshold selection, stated in clinical terms rather than argmax-by-default.

    A screening tool's default 0.5 cut is an arbitrary choice. These are the
    cuts a clinician would actually ask for.
    """
    yt = (y_true == positive).astype(int)
    ts = np.unique(np.concatenate([[0.0], np.sort(p_ulcer), [1.0]]))
    rows = []
    for t in ts:
        yp = (p_ulcer >= t).astype(int)
        tp = int(((yp == 1) & (yt == 1)).sum()); fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum()); tn = int(((yp == 0) & (yt == 0)).sum())
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        rows.append({"threshold": float(t), "sensitivity": sens, "specificity": spec,
                     "youden_j": sens + spec - 1, "tp": tp, "fp": fp, "fn": fn, "tn": tn})

    best_j = max(rows, key=lambda r: r["youden_j"])
    # Screening priority: catch ulcers. Take the most specific threshold that
    # still keeps sensitivity >= 95%, since a missed ulcer can end in amputation.
    hi_sens = [r for r in rows if r["sensitivity"] >= 0.95]
    screen = max(hi_sens, key=lambda r: r["specificity"]) if hi_sens else best_j
    default = min(rows, key=lambda r: abs(r["threshold"] - 0.5))
    return {"youden_optimal": best_j, "high_sensitivity_screening": screen,
            "default_0.5": default}


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------
@torch.no_grad()
def infer(model, loader, temperature=1.0, tta=True):
    logits = []
    targets = []
    for x, y in loader:
        out = model(x)
        if tta:
            out = (out + model(torch.flip(x, dims=[3]))) / 2
        logits.append(out)
        targets.append(y)
    logits = torch.cat(logits)
    targets = torch.cat(targets).numpy()
    probs = torch.softmax(logits / max(temperature, 1e-3), dim=1).numpy()
    raw = torch.softmax(logits, dim=1).numpy()
    return probs, raw, targets


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def fig_confusion(cm, path, title, class_names=CLASS_NAMES):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    short = [c.replace(" skin", "").replace("(", "\n(") for c in class_names]
    for ax, norm in zip(axes, (False, True)):
        m = cm.astype(float)
        if norm:
            m = m / np.maximum(m.sum(1, keepdims=True), 1)
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max() or 1)
        ax.set_xticks(range(len(class_names)), short, fontsize=9)
        ax.set_yticks(range(len(class_names)), short, fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Row-normalised" if norm else "Counts", fontsize=11)
        ax.grid(False)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                txt = f"{m[i, j]:.1%}" if norm else f"{int(cm[i, j])}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=13,
                        fontweight="bold",
                        color="white" if m[i, j] > (m.max() or 1) * 0.55 else PALETTE["ink"])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontweight="bold")
    fig.savefig(path); plt.close(fig)


def fig_curves(y_true, p_ulcer, path, title):
    yt = (y_true == ULCER_IDX).astype(int)
    fpr, tpr, _ = roc_curve(yt, p_ulcer)
    prec, rec, _ = precision_recall_curve(yt, p_ulcer)
    roc_auc, ap = auc(fpr, tpr), average_precision_score(yt, p_ulcer)
    base = yt.mean()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.plot(fpr, tpr, lw=2.2, color=PALETTE["primary"], label=f"AUROC = {roc_auc:.4f}")
    a1.fill_between(fpr, tpr, alpha=0.12, color=PALETTE["primary"])
    a1.plot([0, 1], [0, 1], "--", color=PALETTE["muted"], lw=1.2, label="Chance")
    a1.set(xlabel="False positive rate (1 - specificity)",
           ylabel="True positive rate (sensitivity)", title="ROC")
    a1.legend(loc="lower right")

    a2.plot(rec, prec, lw=2.2, color=PALETTE["accent"], label=f"AP = {ap:.4f}")
    a2.fill_between(rec, prec, alpha=0.12, color=PALETTE["accent"])
    a2.axhline(base, ls="--", color=PALETTE["muted"], lw=1.2,
               label=f"Prevalence = {base:.3f}")
    a2.set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)",
           title="Precision-Recall", ylim=(0, 1.02))
    a2.legend(loc="lower left")
    fig.suptitle(title, fontweight="bold")
    fig.savefig(path); plt.close(fig)


def fig_calibration(rows_raw, rows_cal, ece_raw, ece_cal, conf_raw, conf_cal, path, title):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.plot([0, 1], [0, 1], "--", color=PALETTE["muted"], lw=1.3, label="Perfect calibration")
    for rows, ece, color, label in ((rows_raw, ece_raw, PALETTE["danger"], "Uncalibrated"),
                                    (rows_cal, ece_cal, PALETTE["good"], "Temperature-scaled")):
        pts = [(r["confidence"], r["accuracy"]) for r in rows if r["n"] > 0]
        if pts:
            xs, ys = zip(*pts)
            a1.plot(xs, ys, "o-", color=color, lw=2, ms=6, label=f"{label} (ECE={ece:.4f})")
    a1.set(xlabel="Mean predicted confidence", ylabel="Observed accuracy",
           title="Reliability diagram", xlim=(0, 1.02), ylim=(0, 1.02))
    a1.legend(loc="upper left", fontsize=9)

    bins = np.linspace(0.5, 1.0, 22)
    a2.hist([conf_raw, conf_cal], bins=bins, label=["Uncalibrated", "Temperature-scaled"],
            color=[PALETTE["danger"], PALETTE["good"]], alpha=0.85)
    a2.set(xlabel="Predicted confidence", ylabel="Test images",
           title="Confidence distribution")
    a2.legend(fontsize=9)
    fig.suptitle(title, fontweight="bold")
    fig.savefig(path); plt.close(fig)


def fig_threshold(rows, ops, path, title):
    ts = np.array([r["threshold"] for r in rows])
    sens = np.array([r["sensitivity"] for r in rows])
    spec = np.array([r["specificity"] for r in rows])
    j = np.array([r["youden_j"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(ts, sens, lw=2.2, color=PALETTE["danger"], label="Sensitivity (catch ulcers)")
    ax.plot(ts, spec, lw=2.2, color=PALETTE["primary"], label="Specificity (avoid false alarms)")
    ax.plot(ts, j, lw=1.6, ls="--", color=PALETTE["muted"], label="Youden's J")
    for key, color, style in (("youden_optimal", PALETTE["good"], "-"),
                              ("high_sensitivity_screening", PALETTE["accent"], "-.")):
        t = ops[key]["threshold"]
        ax.axvline(t, color=color, ls=style, lw=1.6,
                   label=f"{key.replace('_', ' ')} @ {t:.3f}")
    ax.set(xlabel="Decision threshold on P(ulcer)", ylabel="Rate",
           title=title, xlim=(0, 1), ylim=(-0.02, 1.05))
    ax.legend(fontsize=8.5, loc="lower center")
    fig.savefig(path); plt.close(fig)


def fig_history(history_paths, path):
    """Learning curves for every run that produced a history file."""
    runs = []
    for p in sorted(history_paths):
        with open(p) as f:
            runs.append(json.load(f))
    if not runs:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for run in runs:
        h = run["history"]
        ep = [e["epoch"] for e in h]
        tag = run.get("tag", run["arch"])
        axes[0].plot(ep, [e["train_loss"] for e in h], lw=1.8, label=f"{tag} train")
        axes[0].plot(ep, [e["val_loss"] for e in h], lw=1.8, ls="--", label=f"{tag} val")
        axes[1].plot(ep, [e["val_acc"] for e in h], lw=2, marker="o", ms=3, label=tag)
        axes[2].plot(ep, [e["val_macro_f1"] for e in h], lw=2, marker="o", ms=3, label=tag)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Validation accuracy")
    axes[2].set(xlabel="Epoch", ylabel="Macro-F1", title="Validation macro-F1")
    for a in axes:
        a.legend(fontsize=7.5)
    fig.suptitle("Training dynamics", fontweight="bold", fontsize=14)
    # tight_layout before savefig, leaving room for the suptitle -- without the
    # rect the suptitle lands on top of the middle subplot's own title
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path); plt.close(fig)
    return path


def fig_prob_hist(probs_ulcer, y_true, path, title):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bins = np.linspace(0, 1, 31)
    ax.hist(probs_ulcer[y_true == ULCER_IDX], bins=bins, alpha=0.75,
            color=PALETTE["danger"], label="Actual: Ulcer")
    ax.hist(probs_ulcer[y_true != ULCER_IDX], bins=bins, alpha=0.75,
            color=PALETTE["good"], label="Actual: Healthy skin")
    ax.axvline(0.5, ls="--", color=PALETTE["ink"], lw=1.3, label="Default threshold 0.5")
    ax.set(xlabel="P(ulcer)", ylabel="Test images", title=title)
    ax.legend(fontsize=9)
    fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------
def evaluate(ckpt, group_aware=False, dedupe=False, split="test", seed=SEED,
             batch_size=32, tta=True, tag=None, grayscale=False):
    use_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    model, meta = load_bundle(ckpt)
    size = int(meta.get("img_size", IMG_SIZE))
    temperature = float(meta.get("temperature", 1.0))
    tag = tag or os.path.splitext(os.path.basename(ckpt))[0]

    # A model trained on luminance-only images must be evaluated the same way,
    # or it is scored on a distribution it never saw and the ablation is
    # meaningless.
    tr, va, te, base = build_splits(size=size, seed=seed, group_aware=group_aware,
                                    dedupe=dedupe, grayscale=grayscale)
    ds = {"train": tr, "val": va, "test": te}[split]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    probs, raw_probs, y_true = infer(model, loader, temperature, tta)
    p_ulcer = probs[:, ULCER_IDX]
    y_pred = probs.argmax(1)

    m = binary_metrics(y_true, y_pred, p_ulcer)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(base.classes))))

    correct = (y_pred == y_true).astype(float)
    ece_cal, mce_cal, rows_cal = expected_calibration_error(probs.max(1), correct)
    ece_raw, mce_raw, rows_raw = expected_calibration_error(
        raw_probs.max(1), (raw_probs.argmax(1) == y_true).astype(float))

    ops = operating_points(y_true, p_ulcer)
    sweep = [r for r in _sweep(y_true, p_ulcer)]

    prefix = os.path.join(FIG_DIR, tag)
    mode = ("group-aware split" if group_aware else "random split") + (" + grayscale" if grayscale else "")
    fig_confusion(cm, f"{prefix}_confusion.png", f"Confusion matrix - {tag} ({mode})")
    fig_curves(y_true, p_ulcer, f"{prefix}_roc_pr.png", f"Discrimination - {tag} ({mode})")
    fig_calibration(rows_raw, rows_cal, ece_raw, ece_cal, raw_probs.max(1), probs.max(1),
                    f"{prefix}_calibration.png", f"Calibration - {tag} ({mode})")
    fig_threshold(sweep, ops, f"{prefix}_thresholds.png",
                  f"Threshold trade-off - {tag} ({mode})")
    fig_prob_hist(p_ulcer, y_true, f"{prefix}_separation.png",
                  f"Predicted P(ulcer) by true class - {tag} ({mode})")

    result = {
        "tag": tag, "checkpoint": os.path.relpath(ckpt, HERE).replace("\\", "/"),
        "architecture": meta.get("arch"), "input_size": size,
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "checkpoint_size_mb": round(os.path.getsize(ckpt) / 1e6, 2),
        "temperature": temperature, "tta": tta,
        "split": split, "group_aware": group_aware, "dedupe": dedupe,
        "grayscale": grayscale, "seed": seed,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "metrics": m,
        "confusion_matrix": cm.tolist(),
        "class_names": list(base.classes),
        "calibration": {"ece_uncalibrated": ece_raw, "mce_uncalibrated": mce_raw,
                        "ece_calibrated": ece_cal, "mce_calibrated": mce_cal,
                        "bins_calibrated": rows_cal},
        "operating_points": ops,
        "train_metrics_from_checkpoint": meta.get("metrics", {}),
    }
    out = os.path.join(REPORT_DIR, f"eval_{tag}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== {tag} | {split} split | {mode} ===")
    print(f"n={m['n']}  acc={m['accuracy']:.4f}  balanced={m['balanced_accuracy']:.4f}  "
          f"F1={m['f1']:.4f}  MCC={m['mcc']:.4f}")
    print(f"sensitivity={m['sensitivity_recall']:.4f}  specificity={m['specificity']:.4f}  "
          f"PPV={m['precision_ppv']:.4f}  NPV={m['npv']:.4f}")
    print(f"AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}")
    print(f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
    print(f"ECE {ece_raw:.4f} (raw) -> {ece_cal:.4f} (T={temperature:.2f})")
    print(f"wrote {out}")
    return result


def _sweep(y_true, p_ulcer):
    yt = (y_true == ULCER_IDX).astype(int)
    for t in np.linspace(0, 1, 201):
        yp = (p_ulcer >= t).astype(int)
        tp = int(((yp == 1) & (yt == 1)).sum()); fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum()); tn = int(((yp == 0) & (yt == 0)).sum())
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        yield {"threshold": float(t), "sensitivity": sens, "specificity": spec,
               "youden_j": sens + spec - 1}


def build_comparison():
    """Collect every eval_*.json into one ranked table + a bar chart."""
    use_style()
    rows = []
    for p in sorted(glob.glob(os.path.join(REPORT_DIR, "eval_*.json"))):
        with open(p) as f:
            r = json.load(f)
        m = r["metrics"]
        rows.append({
            "tag": r["tag"], "arch": r["architecture"], "input": r["input_size"],
            "params_m": r["params_m"], "size_mb": r["checkpoint_size_mb"],
            "split": ("group-aware" if r["group_aware"] else "random")
                     + (" +gray" if r.get("grayscale") else "")
                     + (" +dedup" if r.get("dedupe") else ""),
            "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
            "sensitivity": m["sensitivity_recall"], "specificity": m["specificity"],
            "f1": m["f1"], "mcc": m["mcc"], "auroc": m["auroc"], "auprc": m["auprc"],
            "ece": r["calibration"]["ece_calibrated"],
            "fn": m["fn"], "fp": m["fp"], "n": m["n"],
        })
    if not rows:
        print("no eval_*.json found")
        return []
    rows.sort(key=lambda r: (-r["balanced_accuracy"], -r["auroc"]))

    with open(os.path.join(REPORT_DIR, "comparison.json"), "w") as f:
        json.dump(rows, f, indent=2)

    hdr = ["tag", "split", "params_m", "size_mb", "accuracy", "balanced_accuracy",
           "sensitivity", "specificity", "auroc", "mcc", "ece", "fn", "fp"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(
            f"{r[h]:.4f}" if isinstance(r[h], float) else str(r[h]) for h in hdr) + " |")
    table = "\n".join(md)
    with open(os.path.join(REPORT_DIR, "comparison.md"), "w") as f:
        f.write(table + "\n")
    print("\n" + table)

    # grouped bar chart of the headline metrics
    keys = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "auroc", "mcc"]
    fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(rows) * 1.4), 4.6))
    w = 0.8 / len(keys)
    x = np.arange(len(rows))
    for i, k in enumerate(keys):
        ax.bar(x + i * w - 0.4 + w / 2, [r[k] for r in rows], w, label=k.replace("_", " "))
    ax.set_xticks(x, [f"{r['tag']}\n({r['split']})" for r in rows], fontsize=8)
    ax.set(ylabel="Score", title="Model comparison on held-out test split", ylim=(0, 1.05))
    ax.legend(fontsize=8, ncol=3)
    fig.savefig(os.path.join(FIG_DIR, "comparison.png")); plt.close(fig)

    hist = glob.glob(os.path.join(MODEL_DIR, "history_*.json"))
    if hist:
        fig_history(hist, os.path.join(FIG_DIR, "training_curves.png"))
        print("wrote figures/training_curves.png")
    print("wrote reports/comparison.{json,md} and figures/comparison.png")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--all", action="store_true", help="evaluate every model/dfu_*.pt")
    p.add_argument("--compare", action="store_true", help="rebuild the comparison table only")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--group-aware", action="store_true")
    p.add_argument("--dedupe", action="store_true")
    p.add_argument("--grayscale", action="store_true")
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)
    if a.compare:
        build_comparison(); return
    if a.all:
        for ck in sorted(glob.glob(os.path.join(MODEL_DIR, "dfu_*.pt"))):
            if "optimized" in ck:
                continue
            name = os.path.basename(ck)
            # a checkpoint trained group-aware must be evaluated group-aware,
            # or it is scored against a split it partly trained on
            evaluate(ck, group_aware="_grouped" in name, dedupe="_dedup" in name,
                     grayscale="_gray" in name,
                     split=a.split, seed=a.seed, tta=not a.no_tta)
        build_comparison(); return
    if not a.ckpt:
        p.error("pass --ckpt, --all or --compare")
    evaluate(a.ckpt, group_aware=a.group_aware, dedupe=a.dedupe, split=a.split,
             seed=a.seed, tta=not a.no_tta, grayscale=a.grayscale)


if __name__ == "__main__":
    main()

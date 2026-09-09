"""Dataset integrity audit: duplicates, near-duplicates and split leakage.

    python audit_data.py

The classifier reaches 100% validation macro-F1 by epoch 2 on 738 images, which
does not happen on a genuinely hard task. This measures the two explanations:
the classes really are trivially separable, or the benchmark leaks.

  * exact duplicates via SHA-256 of decoded pixels
  * near-duplicates via 16x16 grayscale dHash + Hamming distance
  * how many near-duplicate pairs straddle a split boundary
  * a colour-only baseline: if mean RGB separates the classes, the CNN is not
    doing anything a histogram could not
"""
import hashlib
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

from dataset import DEFAULT_DATA_DIR, SEED, stratified_split

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(HERE, "reports")
HAMMING_NEAR = 6          # <=6 of 256 bits differ -> visually near-identical


def dhash_bits(img, size=16):
    """Difference hash: compares each pixel to its right neighbour."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def load_all(data_dir):
    records = []
    classes = sorted(d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)))
    for label, cls in enumerate(classes):
        folder = os.path.join(data_dir, cls)
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            with Image.open(path) as im:
                im = im.convert("RGB")
                arr = np.asarray(im)
                records.append({
                    "path": os.path.relpath(path, data_dir).replace("\\", "/"),
                    "name": name,
                    "label": label,
                    "cls": cls,
                    "size": im.size,
                    "sha": hashlib.sha256(arr.tobytes()).hexdigest(),
                    "bits": dhash_bits(im),
                    "mean_rgb": arr.reshape(-1, 3).mean(0),
                    "std_rgb": arr.reshape(-1, 3).std(0),
                })
    return records, classes


def find_near_duplicates(records, max_hamming=HAMMING_NEAR):
    """All pairs within max_hamming bits, exactly, via packed matrix multiply."""
    bits = np.stack([r["bits"] for r in records]).astype(np.uint8)
    n = len(records)
    pairs = []
    # hamming = popcount(a XOR b); done in chunks to bound memory
    for start in range(0, n, 256):
        chunk = bits[start:start + 256]
        d = (chunk[:, None, :] != bits[None, :, :]).sum(2)
        ii, jj = np.where(d <= max_hamming)
        for i, j in zip(ii, jj):
            gi = start + int(i)
            if gi < int(j):
                pairs.append((gi, int(j), int(d[i, j])))
    return pairs


def connected_groups(n, pairs):
    """Union-find over near-duplicate pairs -> clusters of the same underlying view."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in pairs:
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [g for g in groups.values() if len(g) > 1]


def collections_most_common_label(records, group):
    """Majority label of a near-duplicate cluster, so each cluster counts once."""
    counts = defaultdict(int)
    for i in group:
        counts[records[i]["label"]] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def colour_baseline(records):
    """Logistic regression on 6 colour statistics only.

    If this scores near the CNN, the 'deep learning' result is a colour
    histogram in an expensive costume.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    X = np.array([np.concatenate([r["mean_rgb"], r["std_rgb"]]) for r in records])
    y = np.array([r["label"] for r in records])
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def colour_baseline_grouped(records, groups):
    """The colour baseline, measured on exactly the split the CNN is scored on.

    Without this the comparison is unfair in the CNN's favour: the 5-fold number
    is contaminated by near-duplicates, while the CNN's group-aware number is
    not. Same split, same images, same protocol.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from dataset import grouped_split

    X = np.array([np.concatenate([r["mean_rgb"], r["std_rgb"]]) for r in records])
    y = np.array([r["label"] for r in records])

    cluster_of = {i: gid for gid, g in enumerate(groups) for i in g}
    next_id = len(groups)
    gids = []
    for i in range(len(records)):
        g = cluster_of.get(i)
        if g is None:
            g, next_id = next_id, next_id + 1
        gids.append(g)

    tr, _, te = grouped_split(y, gids, seed=SEED)
    if not tr or not te:
        return None
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    pipe.fit(X[tr], y[tr])
    pred = pipe.predict(X[te])
    proba = pipe.predict_proba(X[te])[:, 0]
    return {
        "accuracy": float(accuracy_score(y[te], pred)),
        "auroc": float(roc_auc_score((y[te] == 0).astype(int), proba)),
        "n_train": len(tr), "n_test": len(te),
    }


def figure_duplicate_examples(records, groups, data_dir, out_path, n_clusters=4, max_per=6):
    """Show actual near-duplicate clusters, annotated with their split.

    "49.5% of the test set has a training twin" is a statistic. Seeing six
    near-identical crops of the same wound, some labelled TRAIN and some TEST,
    is the argument.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    from dataset import SEED as DSEED, stratified_split
    from evaluate import PALETTE, use_style
    use_style()

    labels = np.array([r["label"] for r in records])
    tr, va, te = stratified_split(labels, seed=DSEED)
    split_of = {}
    for name, idxs in (("TRAIN", tr), ("VAL", va), ("TEST", te)):
        for i in idxs:
            split_of[i] = name

    # prefer clusters that actually straddle a boundary -- those are the leak
    straddling = [g for g in groups if len({split_of[i] for i in g}) > 1]
    chosen = sorted(straddling, key=len, reverse=True)[:n_clusters]
    if not chosen:
        return None

    cols = min(max_per, max(len(g) for g in chosen))
    fig, axes = plt.subplots(len(chosen), cols,
                             figsize=(2.0 * cols, 2.35 * len(chosen)))
    axes = np.atleast_2d(axes)
    for row, g in enumerate(chosen):
        for col in range(cols):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(g):
                continue
            rec = records[g[col]]
            ax.imshow(PILImage.open(os.path.join(data_dir, rec["path"])).convert("RGB"))
            sp = split_of[g[col]]
            ax.set_title(f"{rec['name']}\n{sp}", fontsize=7.5,
                         color=PALETTE["danger"] if sp != "TRAIN" else PALETTE["ink"],
                         fontweight="bold" if sp != "TRAIN" else "normal")
        axes[row, 0].set_ylabel(f"cluster {row+1}", fontsize=8)

    fig.suptitle("Near-duplicate clusters that straddle the random split\n"
                 "red = the image is in VAL/TEST while its twin is in TRAIN",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def figure_colour_separability(records, classes, acc, out_path):
    """Show *why* six colour numbers reach 92.9%.

    If the two classes are visibly separated in raw channel means, no amount of
    architecture search is the interesting part of this problem.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from evaluate import PALETTE, use_style
    use_style()

    X = np.array([np.concatenate([r["mean_rgb"], r["std_rgb"]]) for r in records])
    y = np.array([r["label"] for r in records])
    colours = [PALETTE["danger"], PALETTE["good"]]
    short = [c.split("(")[0].strip() for c in classes]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))

    # per-channel mean intensity by class
    for ch, name in enumerate("RGB"):
        for lab in (0, 1):
            axes[0].scatter(np.full((y == lab).sum(), ch + (lab - 0.5) * 0.26),
                            X[y == lab, ch], s=5, alpha=0.22, color=colours[lab])
        for lab in (0, 1):
            m = X[y == lab, ch].mean()
            axes[0].plot([ch + (lab - 0.5) * 0.26 - 0.11, ch + (lab - 0.5) * 0.26 + 0.11],
                         [m, m], color=colours[lab], lw=3)
    axes[0].set_xticks([0, 1, 2], ["Red", "Green", "Blue"])
    axes[0].set(ylabel="Mean channel intensity", title="Channel means by class")
    axes[0].legend(handles=[plt.Line2D([], [], color=colours[i], lw=3, label=short[i])
                            for i in (0, 1)], fontsize=9)

    # the two most separating raw features
    axes[1].scatter(X[y == 0, 0], X[y == 0, 1], s=9, alpha=0.45, color=colours[0], label=short[0])
    axes[1].scatter(X[y == 1, 0], X[y == 1, 1], s=9, alpha=0.45, color=colours[1], label=short[1])
    axes[1].set(xlabel="Mean red", ylabel="Mean green",
                title="Two colour features, no CNN")
    axes[1].legend(fontsize=9)

    # ROC of the colour-only model, cross-validated
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    proba = cross_val_predict(pipe, X, y, cv=5, method="predict_proba")[:, 0]
    fpr, tpr, _ = roc_curve((y == 0).astype(int), proba)
    axes[2].plot(fpr, tpr, lw=2.4, color=PALETTE["accent"],
                 label=f"6 colour features\nAUROC = {auc(fpr, tpr):.4f}\nacc = {acc:.4f}")
    axes[2].fill_between(fpr, tpr, alpha=0.14, color=PALETTE["accent"])
    axes[2].plot([0, 1], [0, 1], "--", lw=1.2, color=PALETTE["muted"], label="Chance")
    axes[2].set(xlabel="False positive rate", ylabel="True positive rate",
                title="Colour-only baseline (5-fold CV)")
    axes[2].legend(loc="lower right", fontsize=9)

    fig.suptitle("How much of this task is solvable without vision?",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def main(data_dir=DEFAULT_DATA_DIR):
    os.makedirs(REPORT_DIR, exist_ok=True)
    print(f"scanning {data_dir} ...")
    records, classes = load_all(data_dir)
    n = len(records)
    print(f"{n} images across {len(classes)} classes: {classes}")

    # --- exact duplicates -------------------------------------------------
    by_sha = defaultdict(list)
    for i, r in enumerate(records):
        by_sha[r["sha"]].append(i)
    exact = {k: v for k, v in by_sha.items() if len(v) > 1}
    n_exact_extra = sum(len(v) - 1 for v in exact.values())
    cross_class_exact = [v for v in exact.values()
                         if len({records[i]["label"] for i in v}) > 1]
    print(f"\nexact duplicate groups: {len(exact)} ({n_exact_extra} redundant files)")
    print(f"exact duplicates spanning BOTH classes: {len(cross_class_exact)}")
    for v in cross_class_exact[:5]:
        print("   ", [records[i]["path"] for i in v])

    # --- near duplicates --------------------------------------------------
    pairs = find_near_duplicates(records)
    groups = connected_groups(n, pairs)
    in_group = sum(len(g) for g in groups)
    print(f"\nnear-duplicate pairs (dHash Hamming <= {HAMMING_NEAR}): {len(pairs)}")
    print(f"near-duplicate clusters: {len(groups)} covering {in_group} images "
          f"({in_group / n:.1%} of the dataset)")
    if groups:
        big = max(groups, key=len)
        print(f"largest cluster: {len(big)} images, e.g. "
              f"{[records[i]['name'] for i in big[:6]]}")

    # --- does the split leak? ---------------------------------------------
    labels = np.array([r["label"] for r in records])
    tr, va, te = stratified_split(labels, seed=SEED)
    split_of = {}
    for name, idxs in (("train", tr), ("val", va), ("test", te)):
        for i in idxs:
            split_of[i] = name

    leaked = [(i, j, d) for i, j, d in pairs if split_of[i] != split_of[j]]
    leaked_test = {i for i, j, _ in leaked if split_of[i] in ("val", "test")}
    leaked_test |= {j for i, j, _ in leaked if split_of[j] in ("val", "test")}
    n_eval = len(va) + len(te)
    print(f"\nnear-duplicate pairs crossing a split boundary: {len(leaked)}")
    print(f"val/test images with a near-duplicate in another split: "
          f"{len(leaked_test)}/{n_eval} ({len(leaked_test)/max(n_eval,1):.1%})")

    # --- per-class redundancy and EFFECTIVE dataset size -------------------
    # Asymmetric redundancy matters more than the global rate: the nominal class
    # balance is not the real one, so inverse-frequency weights are wrong.
    in_cluster = {i for g in groups for i in g}
    per_class = {}
    for ci, cls in enumerate(classes):
        idx = [i for i, r in enumerate(records) if r["label"] == ci]
        n_files = len(idx)
        n_exact = sum(len(v) - 1 for v in exact.values() if records[v[0]]["label"] == ci)
        n_clustered = sum(1 for i in idx if i in in_cluster)
        distinct = sum(1 for i in idx if i not in in_cluster)
        distinct += sum(1 for g in groups
                        if collections_most_common_label(records, g) == ci)
        per_class[cls] = {
            "files": n_files,
            "exact_duplicate_extras": n_exact,
            "in_near_duplicate_cluster": n_clustered,
            "in_near_duplicate_cluster_pct": round(n_clustered / n_files * 100, 1),
            "effective_distinct_views": distinct,
            "redundancy_factor": round(n_files / max(distinct, 1), 2),
        }

    total_distinct = sum(v["effective_distinct_views"] for v in per_class.values())
    print("\nper-class redundancy:")
    print(f"  {'class':<26}{'files':>7}{'in cluster':>13}{'distinct':>10}{'redundancy':>12}")
    for cls, v in per_class.items():
        print(f"  {cls:<26}{v['files']:>7}{v['in_near_duplicate_cluster_pct']:>12.1f}%"
              f"{v['effective_distinct_views']:>10}{v['redundancy_factor']:>11.2f}x")
    print(f"  {'TOTAL':<26}{n:>7}{'':>13}{total_distinct:>10}"
          f"{n/max(total_distinct,1):>11.2f}x")
    eff_bal = [per_class[c]["effective_distinct_views"] / max(total_distinct, 1)
               for c in classes]
    nom_bal = [per_class[c]["files"] / n for c in classes]
    print(f"  effective class balance: " +
          " / ".join(f"{b:.1%}" for b in eff_bal) +
          "   (nominal " + " / ".join(f"{b:.1%}" for b in nom_bal) + ")")

    # --- trivial baseline -------------------------------------------------
    acc, std = colour_baseline(records)
    print(f"\ncolour-statistics-only baseline (6 features, logistic regression, 5-fold): "
          f"{acc:.4f} +/- {std:.4f}")

    # Re-fit on group-aware TRAIN and score group-aware TEST, so the baseline and
    # the CNN are measured on identical data.
    honest = colour_baseline_grouped(records, groups)
    if honest:
        print(f"colour-only baseline on the GROUP-AWARE test split: "
              f"acc={honest['accuracy']:.4f} AUROC={honest['auroc']:.4f} "
              f"(n={honest['n_test']})")
    fig_dir = os.path.join(REPORT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = figure_colour_separability(
        records, classes, acc, os.path.join(fig_dir, "colour_separability.png"))
    print(f"wrote {fig_path}")
    dup_fig = figure_duplicate_examples(
        records, groups, data_dir, os.path.join(fig_dir, "duplicate_clusters.png"))
    if dup_fig:
        print(f"wrote {dup_fig}")

    summary = {
        "n_images": n,
        "classes": classes,
        "class_counts": np.bincount(labels).tolist(),
        "exact_duplicate_groups": len(exact),
        "exact_redundant_files": n_exact_extra,
        "exact_duplicates_cross_class": len(cross_class_exact),
        "near_duplicate_pairs": len(pairs),
        "near_duplicate_clusters": len(groups),
        "images_in_a_near_duplicate_cluster": in_group,
        "near_duplicate_coverage_pct": round(in_group / n * 100, 2),
        "cross_split_near_duplicate_pairs": len(leaked),
        "eval_images_with_cross_split_twin": len(leaked_test),
        "eval_images_total": n_eval,
        "eval_contamination_pct": round(len(leaked_test) / max(n_eval, 1) * 100, 2),
        "colour_only_baseline_acc": round(acc, 4),
        "colour_only_baseline_std": round(std, 4),
        "colour_only_grouped_test": honest,
        "per_class_redundancy": per_class,
        "effective_distinct_views": total_distinct,
        "overall_redundancy_factor": round(n / max(total_distinct, 1), 2),
        "effective_class_balance": {c: round(per_class[c]["effective_distinct_views"]
                                             / max(total_distinct, 1), 4) for c in classes},
        "hamming_threshold": HAMMING_NEAR,
        "seed": SEED,
    }
    out = os.path.join(REPORT_DIR, "data_audit.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    # Keep every near-duplicate cluster wholly inside one split.
    cluster_of = {}
    for gi, g in enumerate(groups):
        for i in g:
            cluster_of[i] = gi
    groups_path = os.path.join(REPORT_DIR, "near_duplicate_groups.json")
    with open(groups_path, "w") as f:
        json.dump({"threshold": HAMMING_NEAR,
                   "groups": [[records[i]["path"] for i in g] for g in groups]}, f, indent=2)

    print(f"\nwrote {out}")
    print(f"wrote {groups_path}")
    return summary


if __name__ == "__main__":
    main()

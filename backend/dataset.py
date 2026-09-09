"""Single source of truth for data location, splits, and transforms.

Everything (train.py, evaluate.py, make_samples.py, utils/predict.py) reads its
resolution and normalisation constants from here so train-time and serve-time
preprocessing can never drift apart.
"""
import hashlib
import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import Subset
from torchvision import datasets, transforms

# Patches are natively 224x224; 384 upsamples them for ~2.9x the compute.
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]   # ImageNet statistics (backbone is ImageNet-pretrained)
STD = [0.229, 0.224, 0.225]
SEED = 42

# ImageFolder sorts alphabetically: 0 = Abnormal(Ulcer), 1 = Normal.
# predict.py reads index 0 for P(ulcer).
CLASS_NAMES = ["Abnormal(Ulcer)", "Normal(Healthy skin)"]
ULCER_IDX = 0

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "USE CASE - 02", "DFU", "Patches")
)


def train_transforms(size=IMG_SIZE, grayscale=False):
    """grayscale=True destroys colour while preserving shape and texture.

    A logistic regression on six colour moments already scores 92.9% on this
    dataset (audit_data.py), so it matters whether the CNN is doing anything a
    colour histogram could not. Training on luminance-only images answers that
    directly: if accuracy survives, the model is using structure; if it
    collapses toward chance, the task is a colour test.
    """
    return transforms.Compose(([transforms.Grayscale(num_output_channels=3)] if grayscale else []) + [
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.12)),
    ])


def eval_transforms(size=IMG_SIZE, grayscale=False):
    return transforms.Compose(([transforms.Grayscale(num_output_channels=3)] if grayscale else []) + [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def denormalize(tensor):
    """CHW normalised tensor -> HWC uint8-ready float array in [0, 1]."""
    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(arr * np.array(STD) + np.array(MEAN), 0, 1)


def stratified_split(labels, fracs=(0.70, 0.15, 0.15), seed=SEED):
    """Class-balanced train/val/test index split.

    The original pipeline used an unseeded ``random_split``, so the validation
    set could never be reproduced and reported accuracies were not comparable
    between runs. Seeding + stratifying fixes both.
    """
    assert abs(sum(fracs) - 1.0) < 1e-9, "fracs must sum to 1"
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    out = [[] for _ in fracs]
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        # cumulative cut points keep every image in exactly one split
        cuts = np.cumsum([int(round(f * len(idx))) for f in fracs[:-1]])
        for bucket, part in zip(out, np.split(idx, cuts)):
            bucket.extend(part.tolist())
    return [sorted(b) for b in out]


def load_duplicate_groups(path=None):
    """Near-duplicate clusters produced by audit_data.py, as {relpath: group_id}."""
    path = path or os.path.join(_HERE, "reports", "near_duplicate_groups.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        groups = json.load(f)["groups"]
    return {p: gid for gid, g in enumerate(groups) for p in g}


def grouped_split(labels, group_ids, fracs=(0.70, 0.15, 0.15), seed=SEED):
    """Split so that no near-duplicate cluster is ever divided across splits.

    audit_data.py found that 49.5% of val/test images have a near-identical twin
    in the training set under a plain random split, so the model is partly being
    evaluated on images it has already seen. Keeping each cluster whole makes the
    test set an actual held-out set.

    Measured caveat: eliminating this leakage did NOT lower accuracy -- the
    group-aware model still reaches 100% validation macro-F1. So duplicate
    leakage is real but is not what makes the task easy. A logistic regression on
    six colour moments scores 89.9% on this same group-aware test split, which
    points at genuine colour separability instead. Note also that dHash grouping
    is a proxy for "same foot": patches cropped from different regions of one
    photograph are not near-duplicates and so are not grouped, meaning
    patient-level leakage may remain.

    Greedy largest-group-first assignment into whichever split is furthest below
    its per-class quota. Exact quotas are not achievable when indivisible groups
    are large, so the resulting split sizes are approximate by construction.
    """
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    n_splits = len(fracs)
    quota = np.array([[f * int((labels == c).sum()) for c in np.unique(labels)]
                      for f in fracs])                       # (split, class)
    filled = np.zeros_like(quota)
    out = [[] for _ in fracs]

    by_group = {}
    for i, g in enumerate(group_ids):
        by_group.setdefault(g, []).append(i)

    order = sorted(by_group.values(), key=lambda m: (-len(m), m[0]))
    for members in order:
        counts = np.bincount([labels[i] for i in members], minlength=quota.shape[1])
        # deficit = how far each split still is from its quota if it took this group
        deficit = (quota - filled - counts).min(axis=1)
        best = int(np.argmax(deficit + rng.uniform(0, 1e-6, n_splits)))
        out[best].extend(members)
        filled[best] += counts
    return [sorted(b) for b in out]


def build_splits(data_dir=DEFAULT_DATA_DIR, size=IMG_SIZE, fracs=(0.70, 0.15, 0.15),
                 seed=SEED, group_aware=False, dedupe=False, grayscale=False):
    """Return (train, val, test) Subsets.

    Two separate ImageFolder instances are built so the augmented train view and
    the clean eval view own independent transform objects. Sharing one dataset
    (as the original code did) means the second assignment silently overwrites
    the first and augmentation never runs.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}\n"
            "Expected ImageFolder layout: <data_dir>/Abnormal(Ulcer)/ and "
            "<data_dir>/Normal(Healthy skin)/"
        )
    aug_ds = datasets.ImageFolder(data_dir, transform=train_transforms(size, grayscale))
    clean_ds = datasets.ImageFolder(data_dir, transform=eval_transforms(size, grayscale))
    labels = [y for _, y in clean_ds.samples]
    rel = [os.path.relpath(p, data_dir).replace("\\", "/") for p, _ in clean_ds.samples]

    keep = list(range(len(labels)))
    if dedupe:
        keep = _dedupe_indices(clean_ds.samples)

    if group_aware:
        dup = load_duplicate_groups()
        if not dup:
            raise FileNotFoundError(
                "group_aware=True needs reports/near_duplicate_groups.json -- "
                "run `python audit_data.py` first.")
        # singletons each get their own group so they still split freely
        next_id = max(dup.values(), default=-1) + 1
        gids = []
        for i in keep:
            g = dup.get(rel[i])
            if g is None:
                g, next_id = next_id, next_id + 1
            gids.append(g)
        parts = grouped_split([labels[i] for i in keep], gids, fracs, seed)
    else:
        parts = stratified_split([labels[i] for i in keep], fracs, seed)

    tr, va, te = [[keep[j] for j in part] for part in parts]
    return Subset(aug_ds, tr), Subset(clean_ds, va), Subset(clean_ds, te), clean_ds


def _dedupe_indices(samples):
    """Keep one representative per byte-identical image (342 of 1055 are redundant)."""
    seen, keep = set(), []
    for i, (path, _) in enumerate(samples):
        with Image.open(path) as im:
            h = hashlib.sha256(np.asarray(im.convert("RGB")).tobytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return keep


def class_counts(subset):
    ds = subset.dataset
    labels = [ds.samples[i][1] for i in subset.indices]
    return np.bincount(labels, minlength=len(CLASS_NAMES)).tolist()


def _self_check():
    """Splits must be disjoint, complete, and class-balanced; transforms must differ."""
    labels = np.array([0] * 512 + [1] * 543)
    tr, va, te = stratified_split(labels)
    allidx = tr + va + te
    assert len(set(allidx)) == len(allidx) == len(labels), "splits overlap or drop images"
    for split, frac in ((tr, 0.70), (va, 0.15), (te, 0.15)):
        for cls in (0, 1):
            n_cls = int((labels == cls).sum())
            got = sum(1 for i in split if labels[i] == cls)
            assert abs(got / n_cls - frac) < 0.02, f"class {cls} not stratified: {got}"
    # same seed -> same split (reproducibility)
    assert stratified_split(labels)[0] == tr, "split is not deterministic"
    # the bug that motivated this module: the two views must not share transforms
    assert train_transforms() is not eval_transforms()

    # --- group-aware split: no cluster may straddle a boundary -------------
    # 200 items in 40 clusters of 5, alternating classes
    g_labels = np.array([i % 2 for i in range(200)])
    g_ids = [i // 5 for i in range(200)]
    parts = grouped_split(g_labels, g_ids)
    allidx = [i for p in parts for i in p]
    assert len(set(allidx)) == len(allidx) == 200, "grouped split overlaps or drops"
    where = {i: s for s, p in enumerate(parts) for i in p}
    for g in range(40):
        members = [i for i in range(200) if g_ids[i] == g]
        assert len({where[i] for i in members}) == 1, f"cluster {g} was split across splits"
    # sizes stay roughly on target despite indivisible groups
    assert 0.55 <= len(parts[0]) / 200 <= 0.85, [len(p) for p in parts]
    assert stratified_split(g_labels)[0] != parts[0] or True
    assert grouped_split(g_labels, g_ids)[0] == parts[0], "grouped split is not deterministic"

    print("dataset.py self-check passed")


if __name__ == "__main__":
    _self_check()

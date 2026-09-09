"""Architecture registry + self-describing checkpoints.

The original checkpoint was a bare ``state_dict``: predict.py hard-coded
``resnet50`` to load it, so swapping architectures broke inference with an
opaque shape-mismatch traceback. Checkpoints here carry their own architecture
name, class ordering, input size and calibration temperature, so ``load_bundle``
can always rebuild the exact model that was trained.
"""
import os
import torch
import torch.nn as nn
from torchvision import models

from dataset import CLASS_NAMES, IMG_SIZE

# name -> (constructor, weights enum, attribute holding the classifier head)
ARCHS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, "fc"),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, "fc"),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, "classifier"),
}

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

# Preference order for the served checkpoint. The group-aware ResNet-18 is the
# one the project actually ships (§7 of the summary): same accuracy as the
# original ResNet-50, half the size, ~5x faster end to end. The legacy
# dfu_model.pt is kept last so a fresh clone without the new weights still runs.
_CKPT_PREFERENCE = [
    "dfu_resnet18_grouped.pt",
    "dfu_resnet18.pt",
    "dfu_efficientnet_b0.pt",
    "dfu_model.pt",
]


def _default_ckpt():
    for name in _CKPT_PREFERENCE:
        p = os.path.join(_MODEL_DIR, name)
        if os.path.exists(p):
            return p
    return os.path.join(_MODEL_DIR, _CKPT_PREFERENCE[0])


DEFAULT_CKPT = _default_ckpt()


def build_model(arch="resnet18", num_classes=2, pretrained=True, dropout=0.3):
    """Build a backbone with a fresh classifier head sized to num_classes."""
    if arch not in ARCHS:
        raise ValueError(f"Unknown arch {arch!r}. Available: {sorted(ARCHS)}")
    ctor, weights, head_attr = ARCHS[arch]
    model = ctor(weights=weights if pretrained else None)

    if head_attr == "fc":
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, num_classes))
    else:  # efficientnet: classifier is Sequential(Dropout, Linear)
        in_feat = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, num_classes))
    return model


def target_layer(model, arch):
    """Deepest spatial feature block, for Grad-CAM.

    Returns the *block* (post-residual-add, post-ReLU), not the last raw Conv2d.
    The original code hooked ``layer4.2.conv3``, which sits before the residual
    addition and the final ReLU, so the CAM was computed on a partial signal.
    """
    return model.features[-1] if arch.startswith("efficientnet") else model.layer4


def save_bundle(path, model, arch, *, temperature=1.0, metrics=None, img_size=IMG_SIZE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "arch": arch,
        "state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "img_size": img_size,
        "temperature": float(temperature),
        "metrics": metrics or {},
        "format": 2,
    }, path)


def load_bundle(path=DEFAULT_CKPT, map_location="cpu", default_arch="resnet50"):
    """Load a checkpoint in either format.

    Format 2 is the self-describing dict above. A bare state_dict (format 1, the
    original ``dfu_model.pt``) is still accepted so old checkpoints keep working
    -- its architecture is inferred from the head shape rather than assumed.
    """
    obj = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(obj, dict) and "state_dict" in obj:
        arch = obj["arch"]
        model = build_model(arch, len(obj["class_names"]), pretrained=False)
        model.load_state_dict(obj["state_dict"])
        meta = {k: v for k, v in obj.items() if k != "state_dict"}
    else:
        state = obj
        arch = _infer_arch(state, default_arch)
        model = build_model(arch, 2, pretrained=False)
        # legacy heads are a bare Linear; ours is Sequential(Dropout, Linear).
        # Dropout has no parameters, so remapping is a pure key rename.
        state = {_remap_legacy_key(k): v for k, v in state.items()}
        model.load_state_dict(state)
        meta = {"arch": arch, "class_names": CLASS_NAMES, "img_size": 384,
                "temperature": 1.0, "metrics": {}, "format": 1}

    model.eval()
    return model, meta


def _infer_arch(state, default):
    """Guess architecture from tensor shapes rather than trusting a hard-coded name."""
    if any(k.startswith("features.") for k in state):
        return "efficientnet_b0"
    w = state.get("layer1.0.conv1.weight")
    if w is not None:
        # bottleneck blocks (resnet50) project 64->64 with 1x1; basic blocks
        # (resnet18) use 64->64 3x3. The kernel size separates them.
        return "resnet18" if w.shape[-1] == 3 else "resnet50"
    return default


def _remap_legacy_key(k):
    for head in ("fc.", "classifier."):
        if k.startswith(head) and not k[len(head)].isdigit():
            return f"{head}1.{k[len(head):]}"
    return k


def _self_check():
    """Round-trip every arch through save/load, and confirm legacy inference works."""
    import tempfile
    for arch in ARCHS:
        m = build_model(arch, pretrained=False)
        out = m(torch.randn(2, 3, IMG_SIZE, IMG_SIZE))
        assert out.shape == (2, 2), f"{arch} head wrong shape {out.shape}"
        assert target_layer(m, arch) is not None
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.pt")
            save_bundle(p, m, arch, temperature=1.7)
            m2, meta = load_bundle(p)
            assert meta["arch"] == arch and meta["temperature"] == 1.7
            torch.testing.assert_close(m2(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)),
                                       m.eval()(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)))
    # legacy bare state_dict: arch must be inferred, not assumed
    legacy = models.resnet50(weights=None)
    legacy.fc = nn.Linear(legacy.fc.in_features, 2)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "legacy.pt")
        torch.save(legacy.state_dict(), p)
        m3, meta = load_bundle(p)
        assert meta["format"] == 1 and meta["arch"] == "resnet50", meta
    print("model_def.py self-check passed")


if __name__ == "__main__":
    _self_check()

"""Grad-CAM / Grad-CAM++ visual explanations.

Fixes over the original implementation:
  * hooks are removed after use. The old code registered a fresh pair of hooks
    on the same module on every request and never detached them, so a
    long-running server accumulated hundreds of hooks (unbounded memory growth
    and redundant work on every forward pass).
  * the CAM is taken from the layer4 *block* output -- after the residual
    addition and ReLU -- not from ``layer4.2.conv3``, which is a partial signal
    from one branch.
  * no sqrt() contrast stretch and no image-sized Gaussian blur. A 7x7 CAM
    bilinearly upsampled to 224px is already smooth; the old
    ``heatmap ** 0.5`` plus a ~38px blur inflated every weak activation into a
    large warm blob, which is what made healthy skin look diffusely "hot".
  * Grad-CAM++ is available, which localises multiple lesions better than
    vanilla Grad-CAM when several ulcer regions are present.
"""
import base64

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Context manager -- hooks live only for the duration of the ``with`` block.

    with GradCAM(model, layer) as cam:
        heat, probs = cam.generate(x, class_idx=0)
    """

    def __init__(self, model, target_layer, mode="gradcam++"):
        if mode not in ("gradcam", "gradcam++"):
            raise ValueError(f"mode must be 'gradcam' or 'gradcam++', got {mode!r}")
        self.model = model
        self.target_layer = target_layer
        self.mode = mode
        self.gradients = None
        self.activations = None
        self._handles = []

    def __enter__(self):
        self._handles = [
            self.target_layer.register_forward_hook(
                lambda m, i, o: setattr(self, "activations", o)),
            self.target_layer.register_full_backward_hook(
                lambda m, gi, go: setattr(self, "gradients", go[0])),
        ]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.gradients = self.activations = None
        return False

    def generate(self, input_tensor, class_idx=0):
        """Return (cam HxW float32 in [0,1], probs 1D numpy)."""
        if not self._handles:
            raise RuntimeError("use GradCAM inside a 'with' block so hooks are cleaned up")
        self.model.eval()

        # Hugging Face Spaces runs inference under torch.inference_mode by
        # default, which permanently disables autograd on tensors created inside
        # it. Grad-CAM needs a backward pass, so we explicitly opt out.
        with torch.inference_mode(mode=False), torch.enable_grad():
            x = input_tensor.clone().requires_grad_(True)
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

            self.model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()

            grads = self.gradients[0].detach()        # (C, H, W)
            acts = self.activations[0].detach()       # (C, H, W)

        if self.mode == "gradcam":
            weights = grads.mean(dim=(1, 2))
        else:
            weights = self._pp_weights(grads, acts, logits[0, class_idx].detach())

        cam = torch.relu((weights[:, None, None] * acts).sum(0)).cpu().numpy()
        return self._normalize(cam), probs

    @staticmethod
    def _pp_weights(grads, acts, score):
        """Grad-CAM++ alpha weights (Chattopadhyay et al., 2018)."""
        g2, g3 = grads.pow(2), grads.pow(3)
        denom = 2 * g2 + (acts.sum(dim=(1, 2))[:, None, None] * g3)
        alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
        # exp(score) is a positive constant across channels; it cancels in the
        # normalisation below, so relu(grads) alone carries the sign.
        return (alpha * torch.relu(grads)).sum(dim=(1, 2))

    @staticmethod
    def _normalize(cam):
        """Percentile-clipped 0-1 normalisation.

        Clipping at the 99th percentile stops one saturated pixel from
        compressing the rest of the map into near-zero.
        """
        if not np.isfinite(cam).all() or cam.max() <= 0:
            return np.zeros_like(cam, dtype=np.float32)
        hi = np.percentile(cam, 99.0)
        lo = cam.min()
        if hi <= lo:
            hi = cam.max()
        return np.clip((cam - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)


def _b64_png(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf).decode("utf-8")


def attention_stats(cam, threshold=0.5):
    """Quantitative readings derived from the CAM.

    These turn the heatmap from a picture into numbers a report can cite:
    how much of the field is implicated, how tightly, and where.
    """
    h, w = cam.shape
    mask = (cam >= threshold).astype(np.uint8)
    area_frac = float(mask.mean())
    py, px = np.unravel_index(int(cam.argmax()), cam.shape)

    box = None
    n_regions = 0
    if mask.any():
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        n_regions = n_labels - 1  # label 0 is background
        if n_regions:
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            x, y, bw, bh = stats[big, :4]
            box = [round(x / w, 4), round(y / h, 4), round(bw / w, 4), round(bh / h, 4)]

    return {
        "attention_area_pct": round(area_frac * 100, 2),
        "attention_regions": int(n_regions),
        "peak_xy": [round(float(px) / w, 4), round(float(py) / h, 4)],
        "peak_intensity": round(float(cam.max()), 4),
        "mean_intensity": round(float(cam.mean()), 4),
        # concentration: 1.0 = one tight hotspot, ->0 = diffuse activation.
        "focality": round(float(1.0 - area_frac), 4),
        "bbox_xywh_norm": box,
        "threshold": threshold,
    }


def colorize(cam, img_bgr, alpha_max=0.55, colormap=cv2.COLORMAP_JET):
    """Blend the CAM over the image, weighting alpha by activation strength."""
    h, w = img_bgr.shape[:2]
    cam_r = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_r = np.clip(cam_r, 0, 1)
    colored = cv2.applyColorMap(np.uint8(255 * cam_r), colormap)
    a = (cam_r * alpha_max)[..., None]
    return (img_bgr * (1 - a) + colored * a).astype(np.uint8)


def explain(model, target_layer, img_pil, input_tensor, class_idx=0,
            mode="gradcam++", threshold=0.5):
    """Full explanation bundle for one image.

    Returns overlay + standalone heatmap (both base64 PNG), the class
    probabilities, and the quantitative attention readings.
    """
    with GradCAM(model, target_layer, mode) as cam_extractor:
        cam, probs = cam_extractor.generate(input_tensor, class_idx)

    img_bgr = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    cam_full = np.clip(cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)

    return {
        "overlay_b64": _b64_png(colorize(cam, img_bgr)),
        "heatmap_b64": _b64_png(cv2.applyColorMap(np.uint8(255 * cam_full), cv2.COLORMAP_JET)),
        "probs": probs.tolist(),
        "cam": cam_full,
        "stats": attention_stats(cam_full, threshold),
        "mode": mode,
    }


# Kept so any caller still importing the old name keeps working.
def generate_gradcam_heatmap(model, img_pil, input_tensor, target_layer=None, mode="gradcam++"):
    if target_layer is None:
        target_layer = getattr(model, "layer4", None)
        if target_layer is None:
            target_layer = model.features[-1]
    r = explain(model, target_layer, img_pil, input_tensor, mode=mode)
    return r["overlay_b64"], r["probs"][0]


def _self_check():
    """A CAM must localise: a model that keys on one corner must light that corner."""
    import torch.nn as nn
    from PIL import Image

    torch.manual_seed(0)

    class Tiny(nn.Module):
        """Sums the top-left quadrant into class 0, bottom-right into class 1."""
        def __init__(self):
            super().__init__()
            self.layer4 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
            self.head = nn.Conv2d(8, 2, 1)

        def forward(self, x):
            f = self.layer4(x)
            m = self.head(f)
            q = m.shape[-1] // 2
            c0 = m[:, 0, :q, :q].mean((1, 2))
            c1 = m[:, 1, q:, q:].mean((1, 2))
            return torch.stack([c0, c1], 1)

    model = Tiny().eval()
    x = torch.randn(1, 3, 32, 32)
    for mode in ("gradcam", "gradcam++"):
        with GradCAM(model, model.layer4, mode) as g:
            cam, probs = g.generate(x, class_idx=0)
        assert cam.shape == (32, 32), cam.shape
        assert 0.0 <= cam.min() and cam.max() <= 1.0 + 1e-6, (cam.min(), cam.max())
        assert abs(sum(probs) - 1.0) < 1e-5, probs
        tl = cam[:16, :16].mean()
        br = cam[16:, 16:].mean()
        assert tl > br, f"{mode}: CAM did not localise to the driving quadrant ({tl=}, {br=})"

    # the bug this rewrite exists to prevent: hooks must not survive the block
    before = len(model.layer4._forward_hooks)
    for _ in range(50):
        with GradCAM(model, model.layer4) as g:
            g.generate(x, 0)
    assert len(model.layer4._forward_hooks) == before, "hooks leaked across calls"

    # generate() outside a with-block must fail loudly rather than silently
    # reusing whatever activations were left over from a previous request
    try:
        GradCAM(model, model.layer4).generate(x, 0)
        raise AssertionError("expected RuntimeError outside context manager")
    except RuntimeError:
        pass

    # stats are sane on a synthetic single blob
    blob = np.zeros((64, 64), np.float32)
    blob[20:30, 20:30] = 1.0
    s = attention_stats(blob)
    assert s["attention_regions"] == 1, s
    assert abs(s["attention_area_pct"] - 100 * 100 / 4096) < 0.1, s

    r = explain(model, model.layer4, Image.fromarray(
        np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)), x)
    assert r["overlay_b64"] and r["heatmap_b64"] and r["cam"].shape == (32, 32)
    print("gradcam.py self-check passed")


if __name__ == "__main__":
    _self_check()

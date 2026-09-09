"""Create an untrained baseline checkpoint so the API can start before training.

The original version saved a bare ResNet-18 state_dict to model/dfu_model.pt --
the same filename the server loaded as a ResNet-50, so running it left the
backend unable to start with a shape-mismatch error. It also produced random
weights under a name suggesting a trained model.

The checkpoint written here is in the self-describing bundle format and is named
so it cannot be mistaken for a trained one.
"""
import os

from dataset import IMG_SIZE
from model_def import build_model, save_bundle

HERE = os.path.dirname(os.path.abspath(__file__))


def create_baseline_model(arch="resnet18", out="model/dfu_baseline_untrained.pt"):
    path = os.path.join(HERE, out)
    print(f"Building {arch} with ImageNet weights and a fresh 2-class head...")
    model = build_model(arch, num_classes=2, pretrained=True)
    save_bundle(path, model, arch, img_size=IMG_SIZE,
                metrics={"note": "UNTRAINED baseline -- ImageNet features, random head. "
                                 "Predictions are meaningless until train.py is run."})
    print(f"Saved to {path} ({os.path.getsize(path)/1e6:.1f} MB)")
    print("This model is NOT trained. Run `python train.py --arch resnet18 --group-aware`.")
    return path


if __name__ == "__main__":
    create_baseline_model()

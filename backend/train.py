"""Train a DFU patch classifier.

    python train.py --arch resnet18 --epochs 25

Fixes over the original script:
  * augmentation actually runs (see dataset.build_splits for the shared-transform bug)
  * seeded, stratified, three-way split -- val is used for model selection and
    calibration, test is untouched until evaluate.py
  * native 224px input instead of upsampled 384 (~3x faster, same information)
  * class-weighted loss + label smoothing
  * checkpoints on macro-F1, not accuracy (accuracy hides minority-class collapse)
  * post-hoc temperature scaling so the confidence the UI prints means something
  * every epoch is written to model/history_<arch>.json for the learning curves
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import DEFAULT_DATA_DIR, IMG_SIZE, SEED, build_splits, class_counts
from model_def import build_model, save_bundle

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits, targets = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        targets.append(y)
    return torch.cat(logits), torch.cat(targets)


def macro_f1(logits, targets, num_classes=2):
    preds = logits.argmax(1)
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


TEMP_RANGE = (0.5, 5.0)


def fit_temperature(logits, targets, max_iter=200, bounds=TEMP_RANGE):
    """Post-hoc calibration (Guo et al., 2017).

    Learns a single scalar T minimising NLL on the validation set; logits/T are
    then used at inference. Raw softmax from an over-parameterised CNN on ~1k
    images is badly over-confident, and this app prints that number to a
    clinician, so calibrating it is not optional.

    T is clamped to `bounds`. On a validation set the model separates perfectly
    -- which happens on this dataset -- NLL is minimised by driving T toward 0,
    i.e. by making every prediction maximally confident. That is the exact
    opposite of calibration and would make the UI print 100% on every case.
    A degenerate fit is reported rather than silently shipped.
    """
    log_t = torch.zeros(1, requires_grad=True)
    opt = optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / log_t.exp().clamp(*bounds), targets)
        loss.backward()
        return loss

    opt.step(closure)
    raw = float(log_t.exp().item())
    t = float(min(max(raw, bounds[0]), bounds[1]))
    acc = (logits.argmax(1) == targets).float().mean().item()
    degenerate = raw < bounds[0] * 1.01 or acc >= 0.995
    if degenerate:
        # Nothing to calibrate against: fall back to identity and say so.
        print(f"  WARNING: degenerate temperature fit (raw T={raw:.3f}, val acc={acc:.3f}). "
              f"Validation set is (near-)perfectly separated, so NLL gives no usable "
              f"calibration signal. Falling back to T=1.0.")
        return 1.0, {"raw_T": raw, "degenerate": True, "val_acc": acc}
    return t, {"raw_T": raw, "degenerate": False, "val_acc": acc}


def expected_calibration_error(probs, targets, n_bins=15):
    conf, pred = probs.max(1)
    correct = (pred == targets).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(correct[m].mean().item() - conf[m].mean().item())
    return ece


def train_model(arch="resnet18", epochs=25, batch_size=32, lr=3e-4, weight_decay=1e-4,
                data_dir=DEFAULT_DATA_DIR, img_size=IMG_SIZE, patience=8, workers=4,
                seed=SEED, out_name=None, warmup_epochs=2, group_aware=False,
                dedupe=False, grayscale=False):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{arch}] device={device} img={img_size} bs={batch_size} epochs={epochs}")

    train_ds, val_ds, test_ds, base = build_splits(
        data_dir, img_size, seed=seed, group_aware=group_aware, dedupe=dedupe,
        grayscale=grayscale)
    print(f"split mode: {'GROUP-AWARE (no near-duplicate crosses a boundary)' if group_aware else 'random stratified (LEAKY -- see audit_data.py)'}"
          f"{' + deduped' if dedupe else ''}{' + GRAYSCALE (colour destroyed)' if grayscale else ''}")
    print(f"classes: {base.class_to_idx}")
    print(f"split sizes  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"class counts train={class_counts(train_ds)} val={class_counts(val_ds)} "
          f"test={class_counts(test_ds)}")

    common = dict(num_workers=workers, pin_memory=(device.type == "cuda"),
                  persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=len(train_ds) > batch_size, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)

    model = build_model(arch, num_classes=len(base.classes)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params/1e6:.1f}M")

    # Inverse-frequency weights: 512 ulcer vs 543 healthy is mild, but the loss
    # should not quietly prefer the majority class.
    #
    # KNOWN LIMITATION: these counts are raw file counts. audit_data.py shows the
    # healthy class is 95% near-duplicates (543 files -> ~240 distinct views),
    # so the *effective* balance is 66% ulcer / 34% healthy, not 48/52. These
    # weights therefore describe a distribution that does not exist. Run with
    # --dedupe to weight the deduplicated distribution instead.
    counts = torch.tensor(class_counts(train_ds), dtype=torch.float)
    weights = (counts.sum() / (len(counts) * counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps = max(1, len(train_loader))
    warmup_steps = warmup_epochs * steps
    total_steps = epochs * steps

    def lr_at(step):  # linear warmup then cosine decay to ~1% of peak
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * min(prog, 1.0)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    history, best_f1, best_epoch, bad_epochs = [], -1.0, -1, 0
    out_name = out_name or f"dfu_{arch}{'_grouped' if group_aware else ''}{'_dedup' if dedupe else ''}{'_gray' if grayscale else ''}.pt"
    ckpt_path = os.path.join(MODEL_DIR, out_name)
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = run_correct = seen = 0
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss += loss.item() * x.size(0)
            run_correct += (out.argmax(1) == y).sum().item()
            seen += x.size(0)

        val_logits, val_targets = collect_logits(model, val_loader, device)
        val_loss = nn.CrossEntropyLoss()(val_logits, val_targets).item()
        val_acc = (val_logits.argmax(1) == val_targets).float().mean().item()
        val_f1 = macro_f1(val_logits, val_targets, len(base.classes))

        history.append({
            "epoch": epoch,
            "train_loss": run_loss / max(1, seen),
            "train_acc": run_correct / max(1, seen),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - t0,
        })
        h = history[-1]
        flag = ""
        if val_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = val_f1, epoch, 0
            save_bundle(ckpt_path, model, arch, img_size=img_size,
                        metrics={"val_macro_f1": val_f1, "val_acc": val_acc, "epoch": epoch})
            flag = "  <- best"
        else:
            bad_epochs += 1
        print(f"ep {epoch:>3}/{epochs} | train {h['train_loss']:.4f}/{h['train_acc']:.3f} "
              f"| val {val_loss:.4f}/{val_acc:.3f} | F1 {val_f1:.4f} "
              f"| lr {h['lr']:.2e} | {h['seconds']:.0f}s{flag}")

        if bad_epochs >= patience:
            print(f"early stop: no val F1 improvement for {patience} epochs")
            break

    # Reload the best checkpoint before calibrating -- calibrating the last
    # epoch's weights would fit a temperature for a model we are not shipping.
    from model_def import load_bundle
    model, _ = load_bundle(ckpt_path, map_location=device)
    model.to(device)
    val_logits, val_targets = collect_logits(model, val_loader, device)

    ece_before = expected_calibration_error(val_logits.softmax(1), val_targets)
    temperature, cal_info = fit_temperature(val_logits, val_targets)
    ece_after = expected_calibration_error((val_logits / temperature).softmax(1), val_targets)
    print(f"\ntemperature T={temperature:.3f} | val ECE {ece_before:.4f} -> {ece_after:.4f}")

    save_bundle(ckpt_path, model, arch, temperature=temperature, img_size=img_size,
                metrics={"val_macro_f1": best_f1, "best_epoch": best_epoch,
                         "val_ece_before": ece_before, "val_ece_after": ece_after,
                         "calibration": cal_info,
                         "params_m": n_params / 1e6,
                         "train_minutes": (time.time() - t_start) / 60})

    tag = f"{arch}{'_grouped' if group_aware else ''}{'_dedup' if dedupe else ''}{'_gray' if grayscale else ''}"
    hist_path = os.path.join(MODEL_DIR, f"history_{tag}.json")
    with open(hist_path, "w") as f:
        json.dump({"arch": arch, "tag": tag, "group_aware": group_aware, "dedupe": dedupe,
                   "grayscale": grayscale,
                   "img_size": img_size, "seed": seed, "epochs": epochs,
                   "batch_size": batch_size, "lr": lr, "params_m": n_params / 1e6,
                   "temperature": temperature, "best_epoch": best_epoch,
                   "calibration": cal_info,
                   "val_ece_before": ece_before, "val_ece_after": ece_after,
                   "train_minutes": (time.time() - t_start) / 60,
                   "history": history}, f, indent=2)

    print(f"best val macro-F1 {best_f1:.4f} @ epoch {best_epoch}")
    print(f"checkpoint -> {ckpt_path}")
    print(f"history    -> {hist_path}")
    return ckpt_path, best_f1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50", "efficientnet_b0"])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out-name", default=None)
    p.add_argument("--group-aware", action="store_true",
                   help="keep near-duplicate clusters inside one split (honest eval)")
    p.add_argument("--dedupe", action="store_true",
                   help="drop byte-identical duplicate images")
    p.add_argument("--grayscale", action="store_true",
                   help="destroy colour; tests whether the CNN uses more than colour")
    a = p.parse_args()
    train_model(arch=a.arch, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                weight_decay=a.weight_decay, data_dir=a.data_dir, img_size=a.img_size,
                patience=a.patience, workers=a.workers, seed=a.seed, out_name=a.out_name,
                group_aware=a.group_aware, dedupe=a.dedupe, grayscale=a.grayscale)


if __name__ == "__main__":
    main()

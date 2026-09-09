"""Deployment cost benchmark: checkpoint size, latency, memory.

    python benchmark.py

Replaces the previous optimize_model.py, whose headline claim was wrong.
That script called::

    quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

on a ResNet-50 and reported a "4x memory reduction". ResNet-50 has exactly one
nn.Linear -- the 2048->2 head, 4,098 of 25.6M parameters (0.016%). Everything
that matters is Conv2d, which dynamic quantisation does not touch. TorchScript
tracing then *added* serialisation overhead, so the "optimized" checkpoint came
out at 94.58 MB against the original's 94.37 MB: measurably larger, and no
code path ever loaded it.

The real saving was architectural, and this script measures it honestly.
"""
import argparse
import glob
import json
import os
import time

import torch

from dataset import IMG_SIZE
from model_def import load_bundle

def _rss_mb():
    """Resident set size in MB, or None if psutil is unavailable."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None


HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(HERE, "reports")
MODEL_DIR = os.path.join(HERE, "model")


def bench_one(path, runs=25, warmup=5, batch=1):
    model, meta = load_bundle(path)
    size = int(meta.get("img_size", IMG_SIZE))
    x = torch.randn(batch, 3, size, size)

    # Process RSS around the run. tracemalloc is NOT usable here: it only sees
    # Python-level allocations, and every activation tensor is allocated by
    # torch's C++ allocator, so it reported a flat 0.0 MB for every model.
    rss = _rss_mb()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        base_rss = _rss_mb()
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
        peak_rss = _rss_mb()

    times.sort()
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "checkpoint": os.path.basename(path),
        "arch": meta.get("arch"),
        "input_size": size,
        "params_m": round(n_params / 1e6, 2),
        "file_mb": round(os.path.getsize(path) / 1e6, 2),
        "fp32_weights_mb": round(n_params * 4 / 1e6, 2),
        "latency_ms_median": round(times[len(times) // 2], 2),
        "latency_ms_p05": round(times[int(len(times) * 0.05)], 2),
        "latency_ms_p95": round(times[int(len(times) * 0.95)], 2),
        "rss_after_load_mb": round(rss, 1) if rss else None,
        "rss_delta_inference_mb": round(peak_rss - base_rss, 1) if base_rss else None,
        "throughput_img_s": round(1000.0 / max(times[len(times) // 2], 1e-6), 1),
        "temperature": meta.get("temperature", 1.0),
        "checkpoint_format": meta.get("format"),
    }


def main(runs=25):
    os.makedirs(REPORT_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pt")))
    if not paths:
        print("no checkpoints in model/")
        return []

    rows = []
    for p in paths:
        try:
            rows.append(bench_one(p, runs=runs))
            r = rows[-1]
            print(f"{r['checkpoint']:<34} {r['arch']:<16} {r['params_m']:>6.2f}M  "
                  f"{r['file_mb']:>7.2f} MB  {r['latency_ms_median']:>7.2f} ms  "
                  f"{r['throughput_img_s']:>6.1f} img/s")
        except Exception as e:
            print(f"{os.path.basename(p):<34} FAILED: {type(e).__name__}: {e}")

    if rows:
        rows.sort(key=lambda r: r["file_mb"])
        smallest, largest = rows[0], rows[-1]
        print(f"\nsmallest vs largest: {smallest['checkpoint']} is "
              f"{largest['file_mb'] / max(smallest['file_mb'], 1e-9):.2f}x smaller and "
              f"{largest['latency_ms_median'] / max(smallest['latency_ms_median'], 1e-9):.2f}x "
              f"faster than {largest['checkpoint']}")

        hdr = ["checkpoint", "arch", "input_size", "params_m", "file_mb",
               "latency_ms_median", "latency_ms_p95", "throughput_img_s"]
        md = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
        for r in rows:
            md.append("| " + " | ".join(str(r[h]) for h in hdr) + " |")
        with open(os.path.join(REPORT_DIR, "benchmark.md"), "w") as f:
            f.write("\n".join(md) + "\n")
        with open(os.path.join(REPORT_DIR, "benchmark.json"), "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {REPORT_DIR}/benchmark.{{json,md}}")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, default=25)
    main(p.parse_args().runs)

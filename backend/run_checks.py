"""Run every module's self-check plus the API contract tests.

    python run_checks.py

Each module owns its own checks and can be run directly; this just runs all of
them in one pass and reports a summary, so "does the project still work" is a
single command before a demo or a commit.
"""
import importlib
import os
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODULES = [
    ("dataset", "splits, transforms, group-aware splitting"),
    ("model_def", "architecture registry, checkpoint round-trip"),
    ("clinical", "log-odds fusion, Shapley, IWGDF"),
    ("llm", "grounded assistant prompt, redaction, failure modes"),
    ("utils.gradcam", "Grad-CAM / Grad-CAM++ localisation, hook cleanup"),
    ("utils.report", "narrative tracks measurements"),
    ("utils.predict", "end-to-end inference contract"),
]


def api_checks():
    """The API must reject bad input with 4xx, never a 500."""
    import logging
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

    from fastapi.testclient import TestClient
    import main

    c = TestClient(main.app)
    assert c.get("/").status_code == 200
    assert c.get("/health").status_code == 200

    sample = os.path.join(HERE, "..", "USE CASE - 02", "DFU",
                          "Patches", "Abnormal(Ulcer)", "1.jpg")
    if not os.path.exists(sample):
        print("    (no sample image; skipping /predict cases)")
        return
    img = open(sample, "rb").read()
    files = {"image": ("u.jpg", img, "image/jpeg")}

    r = c.post("/predict", files=files, data={"age": "68", "bmi": "31.2",
                                             "diabetes_years": "14", "hba1c": "8.6",
                                             "neuropathy": "1"})
    assert r.status_code == 200, r.status_code
    d = r.json()
    for k in ("risk", "risk_probability", "confidence", "attention", "attribution",
              "iwgdf", "report", "overlay", "heatmap_only", "model"):
        assert k in d, f"/predict response missing {k}"
    assert d["iwgdf"]["category"] == 1, d["iwgdf"]

    for label, data, want in [
        ("age=-40", {"age": "-40"}, 400),
        ("age=999", {"age": "999"}, 400),
        ("bmi=abc", {"bmi": "abc"}, 400),
        ("hba1c=99", {"hba1c": "99"}, 400),
        ("neuropathy=maybe", {"neuropathy": "maybe"}, 400),
        ("cam_mode=bogus", {"cam_mode": "bogus"}, 400),
        ("image only", {}, 200),
    ]:
        got = c.post("/predict", files=files, data=data).status_code
        assert got == want, f"{label}: expected {want}, got {got}"

    assert c.post("/predict", files={"image": ("x.jpg", b"junk", "image/jpeg")}
                  ).status_code == 400
    assert c.post("/predict", files={"image": ("x.exe", img, "application/x-msdownload")}
                  ).status_code == 415

    # assistant: status must never leak the key; bad input must be 4xx not 5xx
    st = c.get("/assistant")
    assert st.status_code == 200, st.status_code
    body = st.json()
    assert "_key" not in body and "configured" in body and body["suggested_questions"]
    assert c.post("/assistant/ask", json={"question": ""}).status_code == 400
    assert c.post("/assistant/ask", json={"question": "x" * 5000}).status_code == 413
    assert c.post("/assistant/ask", json={"question": "hi", "history": "nope"}
                  ).status_code == 400
    # with no key configured the endpoint must say so (503), not 500
    if not body["configured"]:
        assert c.post("/assistant/ask", json={"question": "hi"}).status_code == 503


def main_():
    results = []
    for name, desc in MODULES:
        print(f"\n--- {name}: {desc}")
        try:
            # each module is run as a subprocess so a torch/global-state change
            # in one cannot mask a failure in the next
            r = subprocess.run([sys.executable, "-c",
                                f"import sys; sys.path.insert(0, r'{HERE}'); "
                                f"import importlib; m = importlib.import_module('{name}'); "
                                f"m._self_check()"],
                               cwd=HERE, capture_output=True, text=True, timeout=900)
            ok = r.returncode == 0
            print((r.stdout or "").strip() or (r.stderr or "").strip()[-900:])
            results.append((name, ok))
        except Exception:
            traceback.print_exc()
            results.append((name, False))

    print("\n--- api: request validation and response contract")
    try:
        api_checks()
        print("api self-check passed")
        results.append(("api", True))
    except Exception as e:
        print(f"api self-check FAILED: {type(e).__name__}: {e}")
        results.append(("api", False))

    print("\n" + "=" * 58)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main_())

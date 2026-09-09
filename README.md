<div align="center">

# 🩺 ClinicalVision DFU

### Explainable multi-modal triage for Diabetic Foot Ulceration

**[🔴 Live demo](https://aarushch.github.io/ClinicalVisionDFU/)**

[![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)

*A CNN classifies a plantar image, Grad-CAM++ shows where it looked, and a
transparent log-odds model folds in clinical risk factors — with the dataset's
limitations measured and stated rather than hidden.*

</div>

---

> **📄 Read [`ClinicalVisionDFU_Summary.md`](ClinicalVisionDFU_Summary.md) first.**
> It is the full technical report: architecture, every measured result, the
> dataset-integrity audit, and the reasoning behind each design decision.

---

## What this is

An end-to-end clinical decision-support prototype:

1. **Vision.** A CNN (ResNet-18 / ResNet-50 / EfficientNet-B0) classifies a
   224×224 plantar tissue patch as ulcer or healthy skin.
2. **Explanation.** Grad-CAM++ produces a spatial attribution map for the *ulcer
   class specifically*, plus quantitative readings — attention area, number of
   discrete regions, focality, peak location.
3. **Fusion.** Image evidence is combined with clinical risk factors in
   **log-odds space**, giving exact closed-form Shapley attributions.
4. **Stratification.** IWGDF 2023 risk categories drive the screening interval.
5. **Report.** An A4 clinical PDF whose prose is generated from the measured
   numbers, so the text cannot contradict the figure beside it.
6. **Assistant.** An optional LLM (Grok or Nemotron) answers questions about the
   result, grounded strictly in the report JSON and proxied through the backend
   so the API key never reaches a browser.

The dashboard ships with **light and dark themes**, a **demo access gate**, and
**five one-click example cases** so anyone can try it without an image of their
own — including one deliberately out-of-scope case that demonstrates the model's
main failure mode.

## The honest headline

The obvious result on this dataset is ~100% accuracy. **That number is an
artefact**, and `backend/audit_data.py` measures why:

| Finding | Value |
|---|---|
| Byte-identical duplicate files | **342 of 1,055** (32%) |
| Images inside a near-duplicate cluster | **56.2%** |
| Val/test images with a near-identical twin in train | **49.5%** |
| Healthy-class files that are near-duplicates | **95.0%** (→ ~240 distinct views) |
| Accuracy from 6 colour statistics alone (no CNN) | **89.9%** |
| **Out-of-distribution images classified as ulcer** | **100%** |

Half the held-out set was never held out. The redundancy is almost entirely in
the *healthy* class, so the real class balance is 66/34, not the nominal 48/52.

The project ships a **group-aware split** that keeps every near-duplicate cluster
inside one split — and reports the uncomfortable result that **it does not change
the score**. The task is genuinely easy: colour statistics alone reach 89.9%, and
a CNN trained on grayscale images reaches 96.9%.

The most serious finding is operational. The model classifies **100% of
out-of-distribution images as ulcer**, at the same confidence it gives genuine
ones — so a photo of a whole foot, which is what a user would naturally upload,
comes back HIGH risk almost every time. No post-hoc detector tested here catches
it reliably, so the mitigation is an explicit scope restriction in the UI and in
every report. Reproduce with `python ood_check.py`.

## Quick start

```bash
# Backend
cd backend
pip install -r requirements-dev.txt      # requirements.txt alone = serving only
python audit_data.py                     # dataset integrity + duplicate clusters
python train.py --arch resnet18 --group-aware
python evaluate.py --all                 # metrics + every figure
python ood_check.py                      # out-of-distribution audit
python make_samples.py --n 12            # sample images and readings
python run_checks.py                     # every self-check + API contract
python main.py                           # http://localhost:10000/docs

# Frontend
cd frontend && npm install && npm run dev # http://localhost:3000
```

Every module is self-testing — run any of them directly:

```bash
python dataset.py && python model_def.py && python clinical.py
python utils/gradcam.py && python utils/report.py && python utils/predict.py
```

## Repository layout

```
backend/
  dataset.py        splits, transforms, group-aware splitting   [self-testing]
  model_def.py      architecture registry, self-describing ckpt [self-testing]
  clinical.py       log-odds fusion, Shapley, IWGDF             [self-testing]
  train.py          training + temperature calibration
  evaluate.py       metrics, ROC/PR, calibration, thresholds, figures
  audit_data.py     duplicate / leakage / redundancy / baseline audit
  ood_check.py      out-of-distribution behaviour audit
  benchmark.py      checkpoint size, latency, memory
  run_checks.py     all self-checks + API contract, one command
  make_samples.py   sample cases: triptychs, contact sheet, readings table
  main.py           FastAPI service
  utils/
    gradcam.py      Grad-CAM & Grad-CAM++                       [self-testing]
    report.py       narrative generated from measurements       [self-testing]
    predict.py      inference pipeline                          [self-testing]
  reports/          generated JSON, figures and sample readings
frontend/src/
  app/page.tsx                  dashboard shell
  components/UploadForm.tsx     upload + clinical intake
  components/ResultCard.tsx     results, attribution, readings
  components/PrintableReport.tsx A4 clinical report
```

## API

`POST /predict` — multipart. Only `image` is required; every clinical field is
optional and a blank field contributes exactly zero rather than defaulting to a
value that would silently move the result.

| Field | Type | Range |
|---|---|---|
| `image` | file | ≤ 15 MB, JPEG/PNG/WebP/BMP/TIFF |
| `age` | float | 0–120 |
| `bmi` | float | 8–90 |
| `diabetes_years` | float | 0–90 |
| `hba1c` | float | 3–20 |
| `neuropathy`, `pad`, `prior_ulcer`, `smoker`, `deformity` | bool | — |
| `cam_mode` | str | `gradcam++` (default) or `gradcam` |

`GET /health` reports the loaded checkpoint; `GET /model` returns its metadata;
`GET /docs` is the interactive OpenAPI console.

### Assistant (optional)

`GET /assistant` reports whether a key is configured (never returns the key).
`POST /assistant/ask` takes `{question, result, history}` and answers grounded in
that one result.

```bash
export LLM_PROVIDER=grok        # or: nemotron
export LLM_API_KEY=<your key>   # free: console.x.ai  /  build.nvidia.com
```

Both providers are OpenAI-compatible, so one client covers both — point
`LLM_BASE_URL` anywhere else that speaks the same protocol. See
`backend/.env.example`. Without a key the app runs normally and the assistant
panel shows setup instructions instead. **The key is read server-side only**: the
frontend is a static export, so anything embedded in it is published to every
visitor.

> The sign-in screen is a **demo gate, not authentication** — no password is
> checked and nothing is protected, because a static site has no session server.
> It is labelled as such on screen.

## CI/CD

- **`deploy-backend.yml`** — pushes `backend/` to a Hugging Face Space over Git
  LFS on every change.
- **`deploy-frontend.yml`** — builds the static Next.js export and publishes to
  GitHub Pages.

> Hugging Face Spaces run inference under `torch.inference_mode()`, which
> permanently disables autograd on tensors created inside it and breaks
> gradient-based XAI. `GradCAM.generate` explicitly opts out with
> `torch.inference_mode(mode=False)` so heatmaps still generate in production.

## 🛡️ Disclaimer

**Not a medical device.** This is a research and teaching prototype. It is not
FDA/CE cleared, has not been validated against histopathology or clinical
outcomes, and must never be used as a standalone diagnostic. The clinical
fusion coefficients are literature-informed priors, **not** fitted to patient
outcomes in this or any cohort. Final diagnostic authority rests with a
qualified clinician.

---

<div align="center"><p>Built for the future of HealthTech.</p></div>

# LayoutLMv3 Feasibility Verdict (Updated April 28, 2026)

## Verdict
Feasible in the current environment for page-level extraction, bounding-box visualization, and smoke-test fine-tuning, with important caveats.

A real end-to-end hello-world run on a legal PDF succeeded in this workspace: PDF ingestion, word/box extraction, LayoutLMv3 forward pass, and one training step all worked on Apple Silicon (MPS).

Current status:
- Single-page and multi-page runs work.
- Bounding boxes can be rendered to PNG visualizations.
- Page selection works with `--page` and `--page-range`.
- Layout classification is heuristic only; it is useful for inspection, not for trusted semantic labeling yet.

## What We Ran
Sample document:
- US Supreme Court opinion PDF downloaded to `data/legal_sample.pdf` (531,026 bytes)
- Source URL used: `https://www.supremecourt.gov/opinions/23pdf/23-939_e2pg.pdf`

Environment observed:
- Python 3.14.3 (virtual environment)
- torch 2.11.0
- transformers (installed successfully)
- Device: MPS available (`torch.backends.mps.is_available() == True`)

Measured outcomes:
- Forward pass (LayoutLMv3 base): success
- Words extracted from first page: 393
- Forward model load time:
  - Cold cache: 49.134 s (includes initial weight download)
  - Warm cache rerun: 8.428 s
- Forward inference time:
  - Initial run: 3.698 s
  - Warm rerun: 0.199 s
- Output tensor shape: `(1, 709, 768)`

Single fine-tuning step smoke test:
- `LayoutLMv3ForTokenClassification` with 5 labels
- One forward + backward + optimizer step: success
- Train step time:
  - Initial run: 5.985 s
  - Warm rerun: 0.991 s (`max_length=256`, batch size 1)

## What Worked
- Package install in current environment succeeded.
- Hugging Face model download and weight load succeeded.
- PDF to model-ready inputs worked using:
  - `pdfplumber` for extracted words and bounding boxes
  - `pypdfium2` for page rendering to image
- Bounding-box visualizations were generated successfully for inspected pages.
- Inference and backprop both worked on MPS.

## What Broke / Friction Points
- OCR is not plug-and-play by default in this test path.
  - We intentionally used `apply_ocr=False` and supplied words + boxes ourselves.
  - For scanned/image-only PDFs, an OCR subsystem (e.g., Tesseract/cloud OCR) is required.
- Hugging Face warning: unauthenticated downloads (no `HF_TOKEN`) may hit rate limits.
- Python 3.14 worked here, but many ML ecosystems still validate primarily on 3.10-3.12; expect occasional package/version friction in future upgrades.
- Layout/heading detection based on heuristics can mislabel body text as headings. That means the current visualizer is best for QA and page inspection, not final production labeling.

## What Fine-Tuning Requires From Us
### 1) Data volume (practical ranges)
For legal document token labeling (NER/field extraction):
- Minimum viable pilot: 300-800 labeled pages
- Useful production baseline: 2,000-10,000 labeled pages
- Strong/generalized model: 20,000+ pages (multi-template, multi-source)

For document classification (page/document class):
- Minimum viable pilot: 1,000-3,000 labeled pages
- Stronger production target: 10,000+ pages

Critical quality requirements:
- Labels must align to extracted tokens/boxes (annotation quality matters more than raw page count early on).
- Include realistic variability (scanned vs digital, firms/courts, templates, redactions, stamps).

### 2) Compute
Current machine can run experiments and small-to-medium fine-tuning jobs.

Expected behavior on this machine class (Apple Silicon + MPS):
- Feasible for prototyping and moderate runs.
- Slower than high-end CUDA GPUs for large sweeps.
- Limited memory headroom at long sequence lengths and larger batch sizes.

Recommended for full production training cycles:
- Cloud GPU (A10/A100/L40 class) for faster iteration and hyperparameter sweeps.
- Keep local MPS for feature engineering, data checks, and smoke tests.

### 3) Time estimates
Based on observed ~1-6 s/step at seq len 256 and batch size 1 (single-step smoke test):

Rough wall-clock estimates (single machine, minimal tuning):
- 500 pages, 5 epochs: ~4-8 hours
- 2,000 pages, 5 epochs: ~16-36 hours
- 10,000 pages, 3-5 epochs: several days

Notes:
- Sequence length 512, OCR-heavy preprocessing, evaluation cadence, and checkpointing all increase runtime.
- First run includes model download/cache overhead.

## Recommendation
Proceed with a staged approach:
1. Pilot now (feasible): 300-800 labeled pages on local MPS to validate label schema and baseline metrics.
2. Move to cloud GPU once data >2k pages or when running multiple tuning experiments.
3. Establish OCR strategy early (digital-only vs scanned-inclusive pipeline), since this affects both labeling and model quality.
4. Treat the current layout classifier as a debugging aid only; if you need real heading/section detection, plan on labeled training data or a formatting-aware extraction pipeline.

## Your Added Sample PDFs (Run Results)
All newly added PDFs in `data/` executed successfully end-to-end with `hello_layoutlmv3.py`.

Observed results:
- `Birth_registration_and_birth_certificates_report.pdf`
  - Words used: 69
  - Forward: 0.278 s
  - Train step: 0.921 s
- `VLRC_Criminal_Liability_for_Workplace_Death_and_Serious_Injury_in_the_Public_Sector_Report.pdf`
  - Words used: 88
  - Forward: 0.206 s
  - Train step: 0.575 s
- `VLRC_Protection_Applications_in_the_Childrens_Court_Final_Report.pdf`
  - Words used: 10
  - Forward: 0.205 s
  - Train step: 0.578 s
- `VLRC_Victims-Of-Crime-Report-W.pdf`
  - Words used: 14
  - Forward: 0.311 s
  - Train step: 2.331 s

Interpretation:
- Environment feasibility remains confirmed.
- Very low extracted word counts (e.g., 10, 14) suggest some first pages are likely cover/image-heavy or OCR-limited.
- For robust legal extraction fine-tuning, include an OCR path and validate text extraction quality across page types.

## Repro
Run:
- `python hello_layoutlmv3.py`

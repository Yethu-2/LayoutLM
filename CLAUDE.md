# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LayoutML is a feasibility/proof-of-concept project for using Microsoft's LayoutLMv3 to extract structured information from legal PDFs on Apple Silicon (MPS). It is not production code — the emphasis is on smoke testing compute/memory, not model accuracy.

## Environment Setup

```bash
source .venv/bin/activate   # Python 3.14.3, all deps pre-installed
```

No `requirements.txt` or `pyproject.toml` — the `.venv` directory contains all dependencies. The model (`microsoft/layoutlmv3-base`) is downloaded from Hugging Face on first run (~49s cold, ~8s warm).

Set `HF_TOKEN` env var if hitting rate limits:
```bash
export HF_TOKEN=your_token_here
```

## Key Commands

```bash
# Forward pass + training smoke test + layout extraction (single page)
python hello_layoutlmv3.py --pdf "data/legal_sample.pdf"

# Multi-page layout extraction
python hello_layoutlmv3.py --pdf "data/legal_sample.pdf" --page-range "1-5"
# or specific pages: --page-range "1,3,5"

# Pretrained layout analysis with visualization output
python pretrained_layoutlmv3_layout_test.py --pdf "data/legal_sample.pdf" --page 1

# Fine-tuning smoke test (validates compute/memory, not accuracy)
python fine_tuning_test/run_finetuning_smoke_test.py --pages "1-2" --steps 4 --max-length 256

# Multiple PDFs in fine-tuning smoke test
python fine_tuning_test/run_finetuning_smoke_test.py --pdf data/legal_sample.pdf --pdf data/VLRC_Victims-Of-Crime-Report-W.pdf --pages "1-3"

# Training time estimation
python fine_tuning_test/estimate_training_time.py --examples 1000 --batch-size 4 --epochs 3
```

## Architecture

### Data Flow

```
PDF file → pdfplumber (word/box extraction) → normalized 0-1000 boxes
         → pypdfium2 (page render to PIL image)
         → LayoutLMv3 (words + boxes + image → 768-dim token embeddings)
         → heuristic label classification → visualization PNGs
```

### Main Scripts

**`hello_layoutlmv3.py`** — Primary entry point. Runs the full pipeline: PDF extraction, forward inference, one training step, and heuristic layout classification. Output PNGs go to `visualizations/`.

**`pretrained_layoutlmv3_layout_test.py`** — Standalone layout extraction with more sophisticated heuristics. Groups words into horizontal lines before classifying. Output goes to `pretrained_layout_outputs/`.

**`fine_tuning_test/run_finetuning_smoke_test.py`** — Multi-PDF, multi-page fine-tuning validation. Tests the training loop (loss decreases, no OOM, timing stats) using auto-generated pseudo-labels from layout heuristics.

**`fine_tuning_test/estimate_training_time.py`** — Wall-clock time calculator for planning full training runs. Can parse actual timing from smoke test logs.

### Key Design Decisions

- **`apply_ocr=False`**: Words and bounding boxes are supplied manually from pdfplumber, not extracted by LayoutLMv3's internal OCR. This means image-only PDFs will not work without an external OCR step.
- **Heuristic pseudo-labels**: Layout labels (HEADER, FOOTER, HEADING, LIST_ITEM, BODY) are rule-based, not gold-standard. The smoke test validates compute correctness, not labeling quality.
- **Bounding box normalization**: PDF coordinates are normalized to 0–1000 scale to match LayoutLMv3's expected input format.
- **Device**: MPS (Apple Silicon) is used when available, with CPU fallback.

### Input/Output Directories

| Directory | Purpose |
|-----------|---------|
| `data/` | Input PDFs (5 legal PDFs pre-loaded) |
| `visualizations/` | Bounding box overlay PNGs from `hello_layoutlmv3.py` |
| `pretrained_layout_outputs/` | Layout visualization PNGs from `pretrained_layoutlmv3_layout_test.py` |

## Performance Baselines

From `LAYOUTLMV3_FEASIBILITY.md`:
- Model load: ~8.4s warm, ~49s cold (first download)
- Inference: 0.2–3.7s per page
- Training step: 0.6–2.3s per step
- Data recommendations: 300–800 pages for a pilot; 2k–10k for production fine-tuning

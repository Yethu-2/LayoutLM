# LayoutLMv3 Hello World Setup

This project runs a LayoutLMv3 feasibility check on legal PDFs — extracting words and bounding boxes, running inference, a training smoke test, and embedding-based layout classification.

## Prerequisites

- macOS (Apple Silicon tested)
- Python 3.14+ (a local virtual environment is already at `.venv`)
- Internet access for first model download from Hugging Face

## Setup

From the project root:

    source .venv/bin/activate

If you need to reinstall dependencies:

    python -m pip install --upgrade pip
    python -m pip install torch transformers pillow pdfplumber pypdfium2

## Usage

Run on a single page (default: page 1):

    python hello_layoutlmv3.py --pdf "data/Birth_registration_and_birth_certificates_report.pdf"

Run on a specific page:

    python hello_layoutlmv3.py --pdf "data/legal_sample.pdf" --page 5

Run layout extraction across a page range:

    python hello_layoutlmv3.py --pdf "data/legal_sample.pdf" --page-range "1-5"

Run the pretrained layout test with visualization:

    python pretrained_layoutlmv3_layout_test.py --pdf "data/legal_sample.pdf" --page 1

Run the fine-tuning smoke test:

    python fine_tuning_test/run_finetuning_smoke_test.py --pages "1-2" --steps 4

Run all PDFs:

    for f in data/*.pdf; do
      python hello_layoutlmv3.py --pdf "$f"
    done

## Expected output

Running `hello_layoutlmv3.py` on a single page produces three sections:

**Forward Pass** — inference timing and output shape:

    Words extracted:    393
    Model load time:    8.7s
    Inference time:     0.33s
    Output shape:       (1, 709, 768)

**Training Step** — one fine-tuning step with heuristic pseudo-labels:

    Sequence length:    256
    Loss:               1.713987
    Train step time:    0.707s

**Layout Extraction** — embedding-based nearest-neighbour classification:

    Type            Count      Examples
    HEADING         39         "OCTOBER", "TERM,", "2023"
    BODY            354        "(Slip", "Opinion)", "1"

Visualizations are saved to:
- `visualizations/` — bounding box overlays from `hello_layoutlmv3.py`
- `pretrained_layout_outputs/` — layout region overlays from `pretrained_layoutlmv3_layout_test.py`

## Troubleshooting

**Hugging Face rate limit warning** — set your token to get higher limits:

    export HF_TOKEN="hf_YOUR_TOKEN_HERE"

Get a token at https://huggingface.co/settings/tokens. To persist it:

    echo 'export HF_TOKEN="hf_YOUR_TOKEN_HERE"' >> ~/.zshrc
    source ~/.zshrc

**Very low word count (10–20 words)** — the page is likely a cover page or image-only. Try a higher page number.

**Out of memory** — pass `--max-length 128` to reduce sequence length.

## Project files

- `hello_layoutlmv3.py` — main script: forward pass, training step, layout extraction
- `pretrained_layoutlmv3_layout_test.py` — standalone pretrained layout analysis with line-level visualization
- `fine_tuning_test/run_finetuning_smoke_test.py` — multi-PDF, multi-page training loop validation
- `fine_tuning_test/estimate_training_time.py` — wall-clock time estimator for full training runs
- `data/` — input PDFs
- `LAYOUTLMV3_FEASIBILITY.md` — feasibility verdict and measured results

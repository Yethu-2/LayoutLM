#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
import urllib.request
from collections import Counter
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium
import torch
import torch.nn.functional as F
from PIL import ImageDraw
from transformers import AutoModel, AutoModelForTokenClassification, AutoProcessor

# Optional: set HF_TOKEN env var to avoid rate limits on model downloads
# Get a token at https://huggingface.co/settings/tokens
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

DEFAULT_PDF_URLS = [
    "https://www.supremecourt.gov/opinions/23pdf/23-939_e2pg.pdf",
    "https://www.supremecourt.gov/opinions/23pdf/23-1141_97be.pdf",
]

_LABEL_TO_ID = {"BODY": 0, "HEADER": 1, "FOOTER": 2, "LIST_ITEM": 3, "HEADING": 4}


def ensure_legal_pdf(pdf_path: Path) -> Path:
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return pdf_path

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in DEFAULT_PDF_URLS:
        try:
            print(f"Downloading legal PDF from: {url}")
            urllib.request.urlretrieve(url, pdf_path)
            if pdf_path.exists() and pdf_path.stat().st_size > 0:
                print(f"Saved: {pdf_path} ({pdf_path.stat().st_size} bytes)")
                return pdf_path
        except Exception as exc:  # pragma: no cover
            last_error = exc

    raise RuntimeError(f"Unable to download legal PDF. Last error: {last_error}")


def extract_words_and_boxes(pdf_path: Path, page_idx: int = 0, limit_words: int = 512) -> tuple[list[str], list[list[int]], float, float, int]:
    with pdfplumber.open(str(pdf_path)) as pdf:
        if page_idx >= len(pdf.pages):
            raise ValueError(f"PDF has {len(pdf.pages)} pages. Cannot access page {page_idx + 1}.")
        page = pdf.pages[page_idx]
        page_num = page_idx + 1
        page_w, page_h = page.width, page.height
        words_raw = page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False)

    words: list[str] = []
    boxes: list[list[int]] = []
    for word in words_raw[:limit_words]:
        text = (word.get("text") or "").strip()
        if not text:
            continue

        x0, y0, x1, y1 = word["x0"], word["top"], word["x1"], word["bottom"]
        # Normalize PDF coordinates to LayoutLMv3's expected 0..1000 bbox space.
        bbox = [
            int(max(0, min(1000, (x0 / page_w) * 1000))),
            int(max(0, min(1000, (y0 / page_h) * 1000))),
            int(max(0, min(1000, (x1 / page_w) * 1000))),
            int(max(0, min(1000, (y1 / page_h) * 1000))),
        ]
        if bbox[2] <= bbox[0]:
            bbox[2] = min(1000, bbox[0] + 1)
        if bbox[3] <= bbox[1]:
            bbox[3] = min(1000, bbox[1] + 1)

        words.append(text)
        boxes.append(bbox)

    if not words:
        raise RuntimeError("No words extracted from first page. Cannot proceed.")

    return words, boxes, page_w, page_h, page_num


def render_first_page(pdf_path: Path, page_idx: int = 0, scale: float = 2.0):
    pdf_doc = pdfium.PdfDocument(str(pdf_path))
    if page_idx >= len(pdf_doc):
        raise ValueError(f"PDF has {len(pdf_doc)} pages. Cannot access page {page_idx + 1}.")
    return pdf_doc[page_idx].render(scale=scale).to_pil()


def visualize_bboxes(image, words: list[str], boxes: list[list[int]], page_num: int, pdf_path: Path) -> str:
    from PIL import ImageFont

    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    img_w, img_h = img_copy.size

    colors = ["red", "blue", "green", "yellow", "cyan", "magenta", "orange", "purple"]
    for i, (word, bbox) in enumerate(zip(words[:30], boxes[:30])):
        x0 = int((bbox[0] / 1000) * img_w)
        y0 = int((bbox[1] / 1000) * img_h)
        x1 = int((bbox[2] / 1000) * img_w)
        y1 = int((bbox[3] / 1000) * img_h)
        color = colors[i % len(colors)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        try:
            draw.text((x0 + 2, y0 - 10), word, fill=color)
        except Exception:
            pass

    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{pdf_path.stem}_page{page_num}_bboxes.png"
    img_copy.save(output_path)
    return str(output_path)


def _heuristic_label(word: str, bbox: list[int]) -> str:
    """Position/text heuristic used to seed class prototypes for embedding classification."""
    y_pos = bbox[1]
    if y_pos < 80:
        return "HEADER"
    if y_pos > 920:
        return "FOOTER"
    if word in {"•", "-", "*", "·", "o", "◦"}:
        return "LIST_ITEM"
    if word.isupper() and len(word) > 5 and not word.isdigit():
        return "HEADING"
    return "BODY"


def _classify_with_embeddings(
    embeddings: torch.Tensor,
    words: list[str],
    boxes: list[list[int]],
    word_ids: list[int | None],
) -> list[str]:
    """Classify words via cosine similarity to per-class prototype embeddings.

    Heuristics seed the class prototypes; LayoutLMv3 token embeddings drive
    the final nearest-neighbour classification. Falls back to heuristics if
    fewer than two classes have prototype examples.
    """
    n_words = len(words)
    device = embeddings.device
    hidden_size = embeddings.shape[-1]

    # Average subword token embeddings back to word embeddings.
    word_emb_sum = torch.zeros(n_words, hidden_size, device=device)
    word_emb_count = torch.zeros(n_words, device=device)
    for tok_idx, word_id in enumerate(word_ids):
        if word_id is not None and word_id < n_words:
            word_emb_sum[word_id] += embeddings[tok_idx]
            word_emb_count[word_id] += 1
    word_emb_count = word_emb_count.clamp(min=1)
    word_embeddings = word_emb_sum / word_emb_count.unsqueeze(-1)  # (n_words, 768)

    # Build per-class prototype vectors from heuristic seed labels.
    heuristic = [_heuristic_label(w, b) for w, b in zip(words, boxes)]
    prototypes: dict[str, torch.Tensor] = {}
    for label in ["HEADER", "FOOTER", "LIST_ITEM", "HEADING", "BODY"]:
        idxs = [i for i, lbl in enumerate(heuristic) if lbl == label]
        if idxs:
            prototypes[label] = word_embeddings[idxs].mean(0)

    if len(prototypes) <= 1:
        return heuristic

    proto_labels = list(prototypes.keys())
    proto_matrix = torch.stack(list(prototypes.values()))  # (n_classes, 768)
    sims = F.cosine_similarity(
        word_embeddings.unsqueeze(1),   # (n_words, 1, 768)
        proto_matrix.unsqueeze(0),      # (1, n_classes, 768)
        dim=-1,
    )  # (n_words, n_classes)
    return [proto_labels[i] for i in sims.argmax(dim=-1).tolist()]


def run_forward(
    pdf_path: Path,
    page_idx: int = 0,
    max_length: int = 512,
    processor=None,
    model=None,
) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    words, boxes, _, _, page_num = extract_words_and_boxes(pdf_path, page_idx=page_idx, limit_words=max_length)
    image = render_first_page(pdf_path, page_idx=page_idx, scale=2.0)

    t0 = time.time()
    if processor is None:
        processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    if model is None:
        model = AutoModel.from_pretrained("microsoft/layoutlmv3-base").to(device)
    load_time = time.time() - t0

    encoding = processor(image, words, boxes=boxes, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        t1 = time.time()
        outputs = model(**enc)
        infer_time = time.time() - t1

    viz_path = visualize_bboxes(image, words, boxes, page_num, pdf_path)

    print("\n" + "="*70)
    print(" FORWARD PASS (Inference)")
    print("="*70)
    print(f"  PDF:                    {pdf_path.name}")
    print(f"  Page:                   {page_num}")
    print(f"  PyTorch:                {torch.__version__}")
    print(f"  Device:                 {device.upper()}")
    print(f"  Words extracted:        {len(words):,}")
    print(f"\n  First 15 words with bounding boxes:")
    print(f"     {'Word':<20} {'[x0, y0, x1, y1]':<25}")
    print(f"     {'-'*20} {'-'*25}")
    for i, (word, bbox) in enumerate(zip(words[:15], boxes[:15]), 1):
        print(f"     {word:<20} {str(bbox):<25}")
    if len(words) > 15:
        print(f"     ... and {len(words) - 15} more words")
    print(f"\n  Visualization saved:    {viz_path}")
    print(f"  Model load time:        {load_time:.3f}s")
    print(f"  Inference time:         {infer_time:.3f}s")
    print(f"  Output shape:           {tuple(outputs.last_hidden_state.shape)}")
    print("="*70)


def run_single_train_step(
    pdf_path: Path,
    page_idx: int = 0,
    max_length: int = 256,
    processor=None,
    model=None,
) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    words, boxes, _, _, page_num = extract_words_and_boxes(pdf_path, page_idx=page_idx, limit_words=300)
    image = render_first_page(pdf_path, page_idx=page_idx, scale=1.5)

    if processor is None:
        processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    if model is None:
        model = AutoModelForTokenClassification.from_pretrained(
            "microsoft/layoutlmv3-base", num_labels=len(_LABEL_TO_ID)
        ).to(device)
    model.train()

    encoding = processor(image, words, boxes=boxes, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    word_ids = encoding.word_ids(batch_index=0)

    # Map heuristic word labels to token positions; -100 is ignored by the loss.
    word_labels = [_LABEL_TO_ID[_heuristic_label(w, b)] for w, b in zip(words, boxes)]
    token_labels = torch.full((encoding["input_ids"].shape[1],), -100, dtype=torch.long)
    for tok_idx, word_id in enumerate(word_ids):
        if word_id is not None and word_id < len(word_labels):
            token_labels[tok_idx] = word_labels[word_id]

    enc = {k: v.to(device) for k, v in encoding.items()}
    enc["labels"] = token_labels.unsqueeze(0).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    optimizer.zero_grad()

    t0 = time.time()
    out = model(**enc)
    loss = out.loss
    loss.backward()
    optimizer.step()
    step_time = time.time() - t0

    print("\n" + "="*70)
    print(" TRAINING STEP (Fine-tuning Smoke Test)")
    print("="*70)
    print(f"  Device:                 {device.upper()}")
    print(f"  Sequence length:        {enc['input_ids'].shape[1]}")
    print(f"  Loss:                   {float(loss.detach().cpu()):.6f}")
    print(f"  Train step time:        {step_time:.3f}s")
    print("="*70 + "\n")


def run_layout_extraction(
    pdf_path: Path,
    page_idx: int = 0,
    processor=None,
    model=None,
) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    words, boxes, page_w, page_h, page_num = extract_words_and_boxes(pdf_path, page_idx=page_idx, limit_words=512)
    image = render_first_page(pdf_path, page_idx=page_idx, scale=2.0)

    if processor is None:
        processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    if model is None:
        model = AutoModel.from_pretrained("microsoft/layoutlmv3-base").to(device)
    model.eval()

    encoding = processor(image, words, boxes=boxes, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    word_ids = encoding.word_ids(batch_index=0)
    enc = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**enc)
        embeddings = outputs.last_hidden_state[0]  # (seq_len, 768)

    # Classify using LayoutLMv3 token embeddings (prototype nearest-neighbour).
    layout_types = _classify_with_embeddings(embeddings, words, boxes, word_ids)
    layout_counts = Counter(layout_types)

    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    layout_colors = {
        "HEADING": "red",
        "HEADER": "darkred",
        "FOOTER": "darkblue",
        "LIST_ITEM": "green",
        "BODY": "blue",
    }
    img_w, img_h = img_copy.size
    for word, bbox, layout_type in zip(words, boxes, layout_types):
        x0 = int((bbox[0] / 1000) * img_w)
        y0 = int((bbox[1] / 1000) * img_h)
        x1 = int((bbox[2] / 1000) * img_w)
        y1 = int((bbox[3] / 1000) * img_h)
        color = layout_colors.get(layout_type, "gray")
        draw.rectangle([x0, y0, x1, y1], outline=color, width=1)

    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{pdf_path.stem}_page{page_num}_layout.png"
    img_copy.save(output_path)

    print("\n" + "="*70)
    print(" LAYOUT EXTRACTION (Document Structure)")
    print("="*70)
    print(f"  PDF:                    {pdf_path.name}")
    print(f"  Page:                   {page_num}")
    print(f"  Tokens analyzed:        {len(words):,}")
    print(f"\n  Layout Classification (embedding nearest-neighbour):")
    print(f"     {'Type':<15} {'Count':<10} {'Examples'}")
    print(f"     {'-'*15} {'-'*10} {'-'*40}")
    for layout_type in ["HEADING", "HEADER", "FOOTER", "LIST_ITEM", "BODY"]:
        count = layout_counts.get(layout_type, 0)
        examples = [w for w, lt in zip(words, layout_types) if lt == layout_type][:3]
        examples_str = ", ".join([f'"{e}"' for e in examples])
        print(f"     {layout_type:<15} {count:<10} {examples_str}")
    print(f"\n  Layout visualization:   {output_path}")
    print("     (Red=Heading, Dark Red=Header, Dark Blue=Footer, Green=List, Blue=Body)")
    print("="*70)


def main() -> None:
    parser = argparse.ArgumentParser(description="LayoutLMv3 hello-world on a legal PDF")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("data/legal_sample.pdf"),
        help="Path to a legal PDF. If missing, a public court opinion PDF is downloaded.",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="Page number to extract from (1-indexed, default: 1).",
    )
    parser.add_argument(
        "--page-range",
        type=str,
        default=None,
        help="Page range to analyze (e.g., '1-5' or '1,2,3'). Overrides --page.",
    )
    args = parser.parse_args()

    if args.page_range:
        if "-" in args.page_range:
            start, end = map(int, args.page_range.split("-"))
            page_indices = list(range(start - 1, end))
        elif "," in args.page_range:
            page_indices = [int(p) - 1 for p in args.page_range.split(",")]
        else:
            page_indices = [int(args.page_range) - 1]
    else:
        page_indices = [args.page - 1]

    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  LayoutLMv3 Feasibility Runner".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")

    _default_pdf = Path("data/legal_sample.pdf")
    if args.pdf == _default_pdf:
        pdf_path = ensure_legal_pdf(args.pdf)
    elif not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}\nCheck the path and try again.")
    else:
        pdf_path = args.pdf
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("Loading processor and models...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    base_model = AutoModel.from_pretrained("microsoft/layoutlmv3-base").to(device)
    clf_model = AutoModelForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base", num_labels=len(_LABEL_TO_ID)
    ).to(device)
    print(f"Models loaded in {time.time() - t0:.1f}s\n")

    if len(page_indices) == 1:
        page_idx = page_indices[0]
        run_forward(pdf_path, page_idx=page_idx, processor=processor, model=base_model)
        run_single_train_step(pdf_path, page_idx=page_idx, processor=processor, model=clf_model)
        run_layout_extraction(pdf_path, page_idx=page_idx, processor=processor, model=base_model)
    else:
        print("\n" + "="*70)
        print(" MULTI-PAGE LAYOUT ANALYSIS")
        print("="*70)
        for page_idx in page_indices:
            print(f"\nProcessing page {page_idx + 1}...")
            run_layout_extraction(pdf_path, page_idx=page_idx, processor=processor, model=base_model)
        print("\n" + "="*70)

    print("All tests completed successfully!")


if __name__ == "__main__":
    main()

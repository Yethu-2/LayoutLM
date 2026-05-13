#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import ImageDraw
from transformers import AutoModel, AutoProcessor

from hello_layoutlmv3 import (
    _classify_with_embeddings,
    ensure_legal_pdf,
    extract_words_and_boxes,
    render_first_page,
)


LAYOUT_COLORS = {
    "HEADING": "red",
    "HEADER": "darkred",
    "FOOTER": "darkblue",
    "LIST_ITEM": "green",
    "BODY": "blue",
}


def classify_layout_labels(
    words: list[str],
    boxes: list[list[int]],
) -> list[str]:
    heights = [max(1, b[3] - b[1]) for b in boxes]
    median_h = statistics.median(heights) if heights else 10
    labels: list[str] = []

    for i, (word, bbox) in enumerate(zip(words, boxes)):
        x0, y0, x1, y1 = bbox
        y_center = (y0 + y1) / 2
        text = word.strip()

        if y_center < 90:
            labels.append("HEADER")
            continue

        if y_center > 930:
            labels.append("FOOTER")
            continue

        if re.match(r"^(?:[\u2022\-\*]|\d+[.)]|[a-zA-Z][.)])$", text):
            labels.append("LIST_ITEM")
            continue

        is_all_caps = text.isupper() and any(c.isalpha() for c in text)
        wide_word = (x1 - x0) > 120
        tall_word = (y1 - y0) >= (median_h * 1.2)
        not_too_short = len(text) >= 5
        likely_page_number = text.isdigit() and y_center > 900

        if likely_page_number:
            labels.append("FOOTER")
            continue

        if is_all_caps and not_too_short and (wide_word or tall_word):
            # Avoid tagging dense consecutive uppercase tokens
            # as heading blocks.
            if (
                i + 1 < len(words)
                and words[i + 1].isupper()
                and len(words[i + 1]) > 3
            ):
                labels.append("BODY")
            else:
                labels.append("HEADING")
            continue

        labels.append("BODY")

    return labels


def make_line_regions(
    words: list[str],
    boxes: list[list[int]],
    labels: list[str],
) -> list[tuple[list[int], str, str]]:
    lines: dict[int, list[int]] = defaultdict(list)
    for idx, bbox in enumerate(boxes):
        y_key = int(round(bbox[1] / 12.0))
        lines[y_key].append(idx)

    regions: list[tuple[list[int], str, str]] = []
    for _, indices in sorted(lines.items()):
        indices.sort(key=lambda i: boxes[i][0])
        x0 = min(boxes[i][0] for i in indices)
        y0 = min(boxes[i][1] for i in indices)
        x1 = max(boxes[i][2] for i in indices)
        y1 = max(boxes[i][3] for i in indices)

        line_labels = [labels[i] for i in indices]
        dominant = Counter(line_labels).most_common(1)[0][0]
        line_text = " ".join(words[i] for i in indices[:20])
        regions.append(([x0, y0, x1, y1], dominant, line_text))

    return regions


def draw_layout_visualization(
    image,
    regions: list[tuple[list[int], str, str]],
    out_path: Path,
) -> None:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    for bbox, label, _ in regions:
        x0 = int((bbox[0] / 1000) * img_w)
        y0 = int((bbox[1] / 1000) * img_h)
        x1 = int((bbox[2] / 1000) * img_w)
        y1 = int((bbox[3] / 1000) * img_h)

        color = LAYOUT_COLORS.get(label, "gray")
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

        label_y = max(0, y0 - 12)
        draw.rectangle([x0, label_y, min(x0 + 92, img_w), y0], fill=color)
        draw.text((x0 + 2, label_y), label, fill="white")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def run_pretrained_layout_test(
    pdf_path: Path,
    page_idx: int,
    max_length: int,
    output_dir: Path,
) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    words, boxes, _, _, page_num = extract_words_and_boxes(
        pdf_path,
        page_idx=page_idx,
        limit_words=max_length,
    )
    image = render_first_page(pdf_path, page_idx=page_idx, scale=2.0)

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(
        "microsoft/layoutlmv3-base",
        apply_ocr=False,
    )
    model = AutoModel.from_pretrained("microsoft/layoutlmv3-base").to(device)
    load_time = time.time() - t0

    encoding = processor(
        image,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0)
    enc = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        t1 = time.time()
        outputs = model(**enc)
        infer_time = time.time() - t1
        embeddings = outputs.last_hidden_state[0]  # (seq_len, 768)

    labels = _classify_with_embeddings(embeddings, words, boxes, word_ids)
    regions = make_line_regions(words, boxes, labels)
    counts = Counter(labels)

    out_path = (
        output_dir / f"{pdf_path.stem}_page{page_num}_pretrained_layout.png"
    )
    draw_layout_visualization(image, regions, out_path)

    print("\n" + "=" * 74)
    print("Pretrained LayoutLMv3 Layout Test (No Fine-tuning)")
    print("=" * 74)
    print(f"PDF: {pdf_path.name}")
    print(f"Page: {page_num}")
    print(f"Device: {device.upper()}")
    print(f"Words analyzed: {len(words)}")
    print(f"Model load time: {load_time:.3f}s")
    print(f"Inference time: {infer_time:.3f}s")
    print("\nLabel counts:")
    for key in ["HEADING", "HEADER", "FOOTER", "LIST_ITEM", "BODY"]:
        print(f"  {key:<10} {counts.get(key, 0):>5}")
    print(f"\nSaved visualization: {out_path}")
    print(
        "Colors: red=HEADING, darkred=HEADER, darkblue=FOOTER, "
        "green=LIST_ITEM, blue=BODY"
    )
    print("=" * 74)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test pretrained LayoutLMv3 layout tagging and output "
            "labeled bbox visualization"
        ),
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("data/legal_sample.pdf"),
        help="Input PDF path",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="1-indexed page number",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max token length passed to processor",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pretrained_layout_outputs"),
        help="Folder to save labeled layout visualizations",
    )
    args = parser.parse_args()

    _default_pdf = Path("data/legal_sample.pdf")
    if args.pdf == _default_pdf:
        pdf_path = ensure_legal_pdf(args.pdf)
    elif not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}\nCheck the path and try again.")
    else:
        pdf_path = args.pdf
    run_pretrained_layout_test(
        pdf_path=pdf_path,
        page_idx=max(0, args.page - 1),
        max_length=args.max_length,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

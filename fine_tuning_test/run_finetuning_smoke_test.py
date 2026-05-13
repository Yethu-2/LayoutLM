#!/usr/bin/env python3
"""LayoutLMv3 fine-tuning smoke test.

This script is meant to answer a practical question: can this machine run
LayoutLMv3 training on your PDFs without crashing or running out of memory?

It does NOT measure accuracy. Instead, it creates simple pseudo-labels from
page layout signals and runs a few optimizer steps to validate compute,
memory use, and end-to-end training plumbing.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pdfplumber
import torch
from transformers import AutoModelForTokenClassification, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hello_layoutlmv3 import (  # noqa: E402
    extract_words_and_boxes,
    render_first_page,
)


LABELS = {
    0: "BODY",
    1: "HEADER",
    2: "FOOTER",
    3: "LIST_ITEM",
    4: "HEADING",
}

BULLET_WORDS = {"•", "-", "*", "·", "o", "◦"}


@dataclass
class Example:
    pdf_path: Path
    page_num: int
    inputs: dict[str, torch.Tensor]
    labels: torch.Tensor


def parse_page_selection(page_selection: str) -> list[int]:
    if not page_selection:
        return [1]
    if "-" in page_selection:
        start_str, end_str = page_selection.split("-", 1)
        start = int(start_str)
        end = int(end_str)
        if start > end:
            raise ValueError(f"Invalid page range: {page_selection}")
        return list(range(start, end + 1))
    if "," in page_selection:
        return [
            int(part)
            for part in page_selection.split(",")
            if part.strip()
        ]
    return [int(page_selection)]


def find_pdfs(pdf_args: list[str] | None) -> list[Path]:
    if pdf_args:
        return [Path(item) for item in pdf_args]
    return sorted(ROOT.glob("data/*.pdf"))


def get_page_count(pdf_path: Path) -> int:
    with pdfplumber.open(str(pdf_path)) as pdf:
        return len(pdf.pages)


def label_for_word(word: str, bbox: list[int]) -> int:
    y_pos = bbox[1]
    word_upper = word.isupper()
    is_number = bool(re.fullmatch(r"\d+[\.]?", word))

    if y_pos < 80:
        return 1  # HEADER
    if y_pos > 920:
        return 2  # FOOTER
    if word in BULLET_WORDS or word.startswith(("-", "•", "*", "·", "◦")):
        return 3  # LIST_ITEM
    if word_upper and len(word) > 5 and not is_number:
        return 4  # HEADING
    return 0  # BODY


def build_example(
    pdf_path: Path,
    page_num: int,
    max_length: int,
    processor: AutoProcessor,
    device: str,
) -> Example:
    page_idx = page_num - 1
    words, boxes, _, _, _ = extract_words_and_boxes(
        pdf_path,
        page_idx=page_idx,
        limit_words=max_length,
    )
    image = render_first_page(pdf_path, page_idx=page_idx, scale=2.0)

    encoding = processor(
        image,
        words,
        boxes=boxes,
        truncation=True,
        padding="longest",
        max_length=max_length,
        return_tensors="pt",
    )

    word_labels = [
        label_for_word(word, bbox) for word, bbox in zip(words, boxes)
    ]
    token_labels = torch.full(
        (encoding["input_ids"].shape[1],),
        -100,
        dtype=torch.long,
    )
    word_ids = encoding.word_ids(batch_index=0)

    for token_index, word_id in enumerate(word_ids):
        if word_id is None or word_id >= len(word_labels):
            continue
        token_labels[token_index] = word_labels[word_id]

    inputs = {key: value.to(device) for key, value in encoding.items()}
    labels = token_labels.unsqueeze(0).to(device)

    return Example(
        pdf_path=pdf_path,
        page_num=page_num,
        inputs=inputs,
        labels=labels,
    )


def build_examples(
    pdf_paths: Iterable[Path],
    page_numbers: list[int],
    max_length: int,
    processor: AutoProcessor,
    device: str,
) -> list[Example]:
    examples: list[Example] = []
    for pdf_path in pdf_paths:
        page_count = get_page_count(pdf_path)
        selected_pages = [
            page for page in page_numbers if 1 <= page <= page_count
        ]
        if not selected_pages:
            print(
                f"Skipping {pdf_path.name}: none of the requested pages exist "
                f"(page count={page_count})."
            )
            continue
        for page_num in selected_pages:
            examples.append(
                build_example(
                    pdf_path,
                    page_num,
                    max_length,
                    processor,
                    device,
                )
            )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LayoutLMv3 fine-tuning smoke test"
    )
    parser.add_argument(
        "--pdf",
        action="append",
        dest="pdfs",
        help=(
            "PDF path to include. Pass multiple times. If omitted, uses all "
            "PDFs in data/."
        ),
    )
    parser.add_argument(
        "--pages",
        default="1-2",
        help=(
            "Page selection per PDF, e.g. '1', '1-3', or '1,2,5'. Default: "
            "1-2."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of optimizer steps to run.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate for AdamW.",
    )
    args = parser.parse_args()

    pdf_paths = find_pdfs(args.pdfs)
    if not pdf_paths:
        raise SystemExit(
            "No PDFs found. Put PDFs in data/ or pass --pdf "
            "path/to/file.pdf."
        )

    page_numbers = parse_page_selection(args.pages)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("\n" + "=" * 72)
    print(" LayoutLMv3 Fine-Tuning Smoke Test")
    print("=" * 72)
    print(f" Device: {device.upper()}")
    print(f" PDFs: {len(pdf_paths)}")
    print(f" Pages per PDF: {args.pages}")
    print(f" Steps: {args.steps}")
    print(f" Max length: {args.max_length}")
    print("=" * 72)

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(
        "microsoft/layoutlmv3-base",
        apply_ocr=False,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(LABELS),
    ).to(device)
    model.train()
    load_time = time.time() - t0

    examples = build_examples(
        pdf_paths,
        page_numbers,
        args.max_length,
        processor,
        device,
    )
    if not examples:
        raise SystemExit("No usable pages found for the requested PDFs/pages.")

    print(f" Loaded examples: {len(examples)}")
    for example in examples[:3]:
        print(f"  - {example.pdf_path.name} page {example.page_num}")
    if len(examples) > 3:
        print(f"  - ... and {len(examples) - 3} more")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses: list[float] = []
    step_times: list[float] = []

    for step_index in range(args.steps):
        example = examples[step_index % len(examples)]
        optimizer.zero_grad(set_to_none=True)

        step_start = time.time()
        outputs = model(**example.inputs, labels=example.labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        step_time = time.time() - step_start

        losses.append(float(loss.detach().cpu()))
        step_times.append(step_time)

        print(
            f"Step {step_index + 1:02d}/{args.steps}: "
            f"{example.pdf_path.name} p.{example.page_num} | "
            f"loss={losses[-1]:.4f} | time={step_time:.3f}s"
        )

    print("\n" + "=" * 72)
    print(" Result")
    print("=" * 72)
    print(f" Model load time: {load_time:.3f}s")
    print(f" Average step time: {sum(step_times) / len(step_times):.3f}s")
    print(
        f" Min/Max step time: {min(step_times):.3f}s / "
        f"{max(step_times):.3f}s"
    )
    print(f" Loss range: {min(losses):.4f} -> {max(losses):.4f}")
    print(" Label mapping:")
    for label_id, label_name in LABELS.items():
        print(f"  {label_id}: {label_name}")
    print("=" * 72)
    print("Fine-tuning smoke test completed successfully.")


if __name__ == "__main__":
    main()

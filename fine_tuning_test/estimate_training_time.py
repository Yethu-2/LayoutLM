#!/usr/bin/env python3
"""Estimate wall-clock time for fine-tuning LayoutLMv3.

Usage examples:
  python fine_tuning_test/estimate_training_time.py --examples 1000 --batch-size 8 --epochs 3
  python fine_tuning_test/estimate_training_time.py --examples 200 --batch-size 4 --epochs 5 --avg-step-time 1.074

Defaults are conservative and come from the recent smoke test in this workspace
(avg step time ~1.074s, model load ~7.603s). You can override with flags.
"""
import argparse
import math
import re
from pathlib import Path
from typing import Optional


def parse_args():
    p = argparse.ArgumentParser(description="Estimate fine-tuning wall-clock time")
    p.add_argument("--examples", type=int, default=1000, help="Number of training examples/pages")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size per step")
    p.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    p.add_argument("--avg-step-time", type=float, default=1.074, help="Average wall time per optimizer step (s)")
    p.add_argument("--model-load-time", type=float, default=7.603, help="Model load time / cold-start overhead (s)")
    p.add_argument("--overhead-factor", type=float, default=1.12, help="Multiplier for IO/overheads (default 12%%)")
    p.add_argument("--conservative-margin", type=float, default=0.20, help="Fractional margin for conservative upper bound (default 20%%)")
    p.add_argument("--smoke-log", type=Path, help="Optional path to a smoke-test log to parse avg timings from")
    return p.parse_args()


def parse_avg_from_smoke_log(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    txt = path.read_text(errors="ignore")
    m = re.search(r"Average step time:\s*([0-9.]+)s", txt)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    m = re.search(r"avg step time:\s*([0-9.]+)", txt, re.I)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def seconds_to_hms(s: float) -> str:
    s = int(round(s))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


def estimate(examples: int, batch_size: int, epochs: int, avg_step_time: float, model_load_time: float, overhead_factor: float, margin: float):
    steps_per_epoch = math.ceil(examples / batch_size)
    total_steps = steps_per_epoch * epochs
    compute_time = total_steps * avg_step_time
    overhead = model_load_time + (compute_time * (overhead_factor - 1.0))
    base_total = compute_time + overhead
    conservative = base_total * (1.0 + margin)
    optimistic = base_total * (1.0 - min(margin / 2.0, 0.0))

    return {
        "examples": examples,
        "batch_size": batch_size,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "avg_step_time": avg_step_time,
        "model_load_time": model_load_time,
        "compute_time_s": compute_time,
        "overhead_s": overhead,
        "base_total_s": base_total,
        "optimistic_s": optimistic,
        "conservative_s": conservative,
    }


def print_report(r: dict):
    print("Training time estimate")
    print("----------------------")
    print(f"Examples: {r['examples']}")
    print(f"Batch size: {r['batch_size']}")
    print(f"Epochs: {r['epochs']}")
    print(f"Steps / epoch: {r['steps_per_epoch']}")
    print(f"Total optimizer steps: {r['total_steps']}")
    print(f"Avg step time: {r['avg_step_time']:.3f}s")
    print(f"Model load / cold overhead: {r['model_load_time']:.3f}s")
    print("")
    print(f"Compute-only time: {r['compute_time_s']:.1f}s ({seconds_to_hms(r['compute_time_s'])})")
    print(f"Estimated additional overhead: {r['overhead_s'] - r['model_load_time']:.1f}s")
    print(f"Base total time: {r['base_total_s']:.1f}s ({seconds_to_hms(r['base_total_s'])})")
    print(f"Optimistic estimate: {r['optimistic_s']:.1f}s ({seconds_to_hms(r['optimistic_s'])})")
    print(f"Conservative (with margin): {r['conservative_s']:.1f}s ({seconds_to_hms(r['conservative_s'])})")
    print("")
    print("Per-epoch breakdown:")
    per_epoch = r['total_steps'] and (r['compute_time_s'] / r['epochs']) or 0.0
    print(f"  Per epoch compute: {per_epoch:.1f}s ({seconds_to_hms(per_epoch)})")


def main():
    args = parse_args()
    avg = args.avg_step_time
    if args.smoke_log:
        parsed = parse_avg_from_smoke_log(args.smoke_log)
        if parsed:
            print(f"Parsed avg step time {parsed:.3f}s from {args.smoke_log}")
            avg = parsed
        else:
            print(f"Could not parse average step time from {args.smoke_log}; using {avg:.3f}s")

    r = estimate(
        examples=args.examples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        avg_step_time=avg,
        model_load_time=args.model_load_time,
        overhead_factor=args.overhead_factor,
        margin=args.conservative_margin,
    )
    print_report(r)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""CLI for running Saraga multitrack in-the-wild noise-resilience and bleed audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.saraga_benchmark import (
    DEFAULT_MAX_CHUNKS_PER_TRACK,
    audit_saraga_noise_resilience,
    load_saraga_excerpts,
)


def run_saraga_eval(
    saraga_dir: Path | str = "data/raw/saraga_carnatic",
    sep_checkpoint: Path | str | None = None,
    clf_checkpoint: Path | str | None = None,
    max_chunks: int = DEFAULT_MAX_CHUNKS_PER_TRACK,
    output_json: Path | str = "experiments/saraga_noise_resilience_audit.json",
    device: str = "cpu",
) -> dict:
    saraga_path = Path(saraga_dir)
    output_path = Path(output_json)

    if not saraga_path.exists():
        raise FileNotFoundError(f"Saraga dataset directory not found: {saraga_path}")

    print(f"Scanning Saraga directory: {saraga_path}...")
    chunks = load_saraga_excerpts(saraga_path, max_chunks_per_track=max_chunks)
    print(f"Extracted {len(chunks)} active non-silent 6.0s chunks.")

    if not chunks:
        print("Warning: No active chunks found in Saraga directory.")

    print(f"Running noise resilience & bleed audit on device={device}...")
    result = audit_saraga_noise_resilience(
        chunks,
        sep_checkpoint=sep_checkpoint,
        clf_checkpoint=clf_checkpoint,
        device=device,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Audit results written to: {output_path}")

    print("\n--- Saraga Noise-Resilience & Bleed Audit Summary ---")
    print(f"Total Chunks Audited:                {result['total_chunks_audited']}")
    print(
        f"Mean Vocal Percussive Bleed:         {result['mean_vocal_percussive_bleed']:.4f}"
    )
    print(
        f"Mean Accompaniment String Conf:      {result['mean_accompaniment_string_confidence']:.4f}"
    )
    print(
        f"Mean Vocal HPR (dB):                 {result['mean_vocal_hpr_db']:.2f} dB"
    )
    print(
        f"Mean Accompaniment HPR (dB):         {result['mean_accompaniment_hpr_db']:.2f} dB"
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Saraga multitrack in-the-wild noise-resilience and bleed-audit benchmark"
    )
    parser.add_argument(
        "--saraga-dir",
        type=str,
        default="data/raw/saraga_carnatic",
        help="Path to Saraga dataset directory (default: data/raw/saraga_carnatic)",
    )
    parser.add_argument(
        "--sep-checkpoint",
        type=str,
        default=None,
        help="Path to fine-tuned separator checkpoint (.pt)",
    )
    parser.add_argument(
        "--clf-checkpoint",
        type=str,
        default=None,
        help="Path to trained TimbreCNN classifier checkpoint (.pt)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=DEFAULT_MAX_CHUNKS_PER_TRACK,
        help="Maximum chunks to extract per track (default: 15)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="experiments/saraga_noise_resilience_audit.json",
        help="Path to save output audit JSON (default: experiments/saraga_noise_resilience_audit.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run inference on (default: cpu)",
    )
    args = parser.parse_args()

    try:
        run_saraga_eval(
            saraga_dir=args.saraga_dir,
            sep_checkpoint=args.sep_checkpoint,
            clf_checkpoint=args.clf_checkpoint,
            max_chunks=args.max_chunks,
            output_json=args.output_json,
            device=args.device,
        )
    except Exception as e:
        print(f"Error running Saraga evaluation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

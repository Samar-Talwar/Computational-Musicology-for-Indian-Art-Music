#!/usr/bin/env python
"""CLI entry point — Experiment 0: HDemucs 2-stem baseline on Sanidha Concert 3.

Usage (smoke):
  python scripts/run_baseline.py --max-segments-per-song 2
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.sanidha import index_sanidha, split_sanidha
from evaluation.baseline import run_concert_baseline


def main():
    parser = argparse.ArgumentParser(description="Sanidha baseline evaluator (Experiment 0)")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=r"C:\Users\Samar\OneDrive\Documents\Sanidha_Dataset",
        help="Sanidha root (default: OneDrive dataset)",
    )
    parser.add_argument(
        "--val-concert", type=int, default=3, help="Validation concert ID"
    )
    parser.add_argument(
        "--segment-duration", type=float, default=30.0, help="Window duration (sec)"
    )
    parser.add_argument(
        "--max-segments-per-song",
        type=int,
        default=6,
        help="Max windows per song (6 = 3 min at 30s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/baseline_hdemucs_concert3.json",
        help="JSON output path",
    )
    parser.add_argument(
        "--include-tanpura",
        dest="include_tanpura",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--exclude-tanpura",
        dest="include_tanpura",
        action="store_false",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root)
    records = index_sanidha(root)
    _, val = split_sanidha(records, train_concerts=(1, 2, 4, 5), val_concerts=(args.val_concert,))

    result = run_concert_baseline(
        val,
        segment_duration_sec=args.segment_duration,
        max_segments_per_song=args.max_segments_per_song,
        include_tanpura=args.include_tanpura,
        output_json=Path(args.output),
    )

    # ASCII summary table fix 4 — clean print
    print("=" * 60)
    print("Experiment 0 — HDemucs 2-stem Baseline (Concert %d)" % args.val_concert)
    print("=" * 60)
    print(f"Songs evaluated : {result['songs_evaluated']}")
    print(f"Output JSON     : {args.output}")
    print()
    for res in result.get("per_song_results", []):
        sid = res.get("record_id", "?")
        valid = res.get("valid_windows", 0)
        print(f"{sid}  windows={valid}  skipped_ref={res.get('skipped_silent_ref_windows',0)}  skipped_est={res.get('skipped_silent_est_windows',0)}")
    # Overall
    overall = result.get("overall_mean_median")
    if overall is not None:
        print()
        print("Overall (mean / median) — vocals / accompaniment")
        for stem in ("vocals", "accompaniment"):
            s = overall.get(stem, {})
            sdr_m = s.get("SDR_mean", "N/A")
            sdr_md = s.get("SDR_median", "N/A")
            print(f"  {stem:13} SDR mean={sdr_m:.3f} median={sdr_md:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

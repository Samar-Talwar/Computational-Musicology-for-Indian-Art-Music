#!/usr/bin/env python3
"""
Read-only audit of the Sanidha dataset.
Outputs docs/sanidha_inventory.csv and prints a summary to stdout.
"""

import os
import csv
from pathlib import Path
import soundfile as sf

# Dataset root from env var with documented default
DEFAULT_ROOT = r"C:\Users\Samar\OneDrive\Documents\Sanidha_Dataset"
ROOT = Path(os.environ.get("SANIDHA_ROOT", DEFAULT_ROOT))

# Tolerance for duration mismatch detection (seconds)
DURATION_TOL = 0.1

def parse_path(rel_path: Path) -> tuple[str, str, str, str] | None:
    """
    Parse relative path into (concert, song, folder, stem).
    Expected: ConcertXX/<song>/Audio-Multitracks-<Clean|Processed>/<stem>.wav
    Returns None for files under Videos/ or unexpected structure.
    """
    parts = rel_path.parts
    if len(parts) < 4:
        return None

    concert = parts[0]
    song = parts[1]

    # parts[2] should be Audio-Multitracks-Clean / Audio-Multitracks-Processed / Videos
    audio_dir = parts[2]
    if audio_dir == "Videos":
        return None

    if audio_dir.startswith("Audio-Multitracks-"):
        folder = audio_dir.replace("Audio-Multitracks-", "")
    else:
        folder = "unknown"

    stem = rel_path.stem  # filename without .wav
    return concert, song, folder, stem


def main() -> None:
    if not ROOT.exists():
        print(f"ERROR: Dataset root not found: {ROOT}")
        return

    wav_files = list(ROOT.rglob("*.wav"))
    if not wav_files:
        print("No .wav files found.")
        return

    rows = []  # list of dicts for CSV
    unreadable = []

    for wav_path in wav_files:
        rel = wav_path.relative_to(ROOT)
        parsed = parse_path(rel)
        if parsed is None:
            # Skip Videos/ or unexpected structure
            continue

        concert, song, folder, stem = parsed

        try:
            info = sf.info(str(wav_path))
        except Exception as e:
            unreadable.append((str(rel), str(e)))
            continue

        frames = info.frames
        sr = info.samplerate
        channels = info.channels
        subtype = info.subtype
        duration = frames / sr if sr else 0.0

        rows.append({
            "concert": concert,
            "song": song,
            "folder": folder,
            "stem": stem,
            "samplerate": sr,
            "channels": channels,
            "subtype": subtype,
            "frames": frames,
            "duration_s": duration,
        })

    # Write CSV
    out_csv = Path("docs/sanidha_inventory.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["concert", "song", "folder", "stem", "samplerate", "channels", "subtype", "frames", "duration_s"]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} entries to {out_csv}")

    if unreadable:
        print(f"\nUnreadable files ({len(unreadable)}):")
        for path, err in unreadable:
            print(f"  {path}: {err}")

    # ---- Summary statistics ----
    # Group by concert
    from collections import defaultdict

    by_concert = defaultdict(list)
    by_folder = defaultdict(list)
    by_song = defaultdict(list)  # key: (concert, song)
    by_song_folder = defaultdict(list)  # key: (concert, song, folder)

    for r in rows:
        by_concert[r["concert"]].append(r)
        by_folder[r["folder"]].append(r)
        by_song[(r["concert"], r["song"])].append(r)
        by_song_folder[(r["concert"], r["song"], r["folder"])].append(r)

    # Files and hours per concert
    print("\n=== Per Concert ===")
    for concert in sorted(by_concert.keys()):
        entries = by_concert[concert]
        total_hours = sum(e["duration_s"] for e in entries) / 3600
        print(f"  {concert}: {len(entries)} files, {total_hours:.2f} hours")

    # Files and hours per folder
    print("\n=== Per Folder (Clean vs Processed) ===")
    for folder in sorted(by_folder.keys()):
        entries = by_folder[folder]
        total_hours = sum(e["duration_s"] for e in entries) / 3600
        print(f"  {folder}: {len(entries)} files, {total_hours:.2f} hours")

    # Songs per concert
    print("\n=== Songs per Concert ===")
    songs_per_concert = defaultdict(set)
    for r in rows:
        songs_per_concert[r["concert"]].add(r["song"])
    for concert in sorted(songs_per_concert.keys()):
        print(f"  {concert}: {len(songs_per_concert[concert])} songs")

    # Stems per song
    print("\n=== Stems per Song ===")
    for (concert, song), entries in sorted(by_song.items()):
        stems = sorted({e["stem"] for e in entries})
        print(f"  {concert}/{song}: {len(stems)} stems -> {stems}")

    # Duration mismatches within a song (across stems)
    print("\n=== Duration Mismatches Within Song (stems) ===")
    mismatch_found = False
    for (concert, song), entries in sorted(by_song.items()):
        durations = [e["duration_s"] for e in entries]
        if max(durations) - min(durations) > DURATION_TOL:
            mismatch_found = True
            print(f"  {concert}/{song}: durations range {min(durations):.3f}s - {max(durations):.3f}s (diff: {max(durations)-min(durations):.3f}s)")
    if not mismatch_found:
        print("  None (all stems within a song match within 0.1s)")

    # Clean vs Processed duration mismatches per song
    print("\n=== Clean vs Processed Duration Mismatches ===")
    mismatch_found = False
    for (concert, song), entries in sorted(by_song.items()):
        clean_entries = [e for e in entries if e["folder"] == "Clean"]
        proc_entries = [e for e in entries if e["folder"] == "Processed"]

        if not clean_entries or not proc_entries:
            mismatch_found = True
            print(f"  {concert}/{song}: missing folder (Clean: {len(clean_entries)}, Processed: {len(proc_entries)})")
            continue

        clean_stems = {e["stem"] for e in clean_entries}
        proc_stems = {e["stem"] for e in proc_entries}
        if clean_stems != proc_stems:
            mismatch_found = True
            missing_in_clean = proc_stems - clean_stems
            missing_in_proc = clean_stems - proc_stems
            if missing_in_clean:
                print(f"  {concert}/{song}: stems missing in Clean: {missing_in_clean}")
            if missing_in_proc:
                print(f"  {concert}/{song}: stems missing in Processed: {missing_in_proc}")

        # Compare total duration per folder
        clean_dur = sum(e["duration_s"] for e in clean_entries)
        proc_dur = sum(e["duration_s"] for e in proc_entries)
        if abs(clean_dur - proc_dur) > DURATION_TOL:
            mismatch_found = True
            print(f"  {concert}/{song}: Clean total={clean_dur:.3f}s, Processed total={proc_dur:.3f}s (diff: {abs(clean_dur-proc_dur):.3f}s)")

    if not mismatch_found:
        print("  None (Clean and Processed match within 0.1s per song)")

    print("\nDone.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Generate deterministic publication figures from committed experiment artifacts.

Usage:
    PYTHONDONTWRITEBYTECODE=1 ./.venv/Scripts/python.exe scripts/generate_report_figures.py

Outputs:
    report/figures/fig1_pipeline.{pdf,png}
    report/figures/fig2_results_bleed.{pdf,png}
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ["SOURCE_DATE_EPOCH"] = "0"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.classifier import INSTRUMENT_CLASSES, TimbreCNN, extract_mfcc  # noqa: E402

EXPERIMENTS = ROOT / "experiments"
OUT_DIR = ROOT / "report" / "figures"

# Okabe-Ito colorblind-safe palette.
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#666666"
LIGHT_GREY = "#E8E8E8"
INK = "#1A1A1A"

PDF_METADATA = {
    "Title": "Carnatic source separation and timbre-audit figures",
    "Author": "Computational Musicology Project",
    "Subject": "Concert 03 separation SDR and downstream timbre audits",
    "Keywords": "source separation, HDemucs, TimbreCNN, Carnatic music, SDR",
    "Creator": "scripts/generate_report_figures.py",
    "CreationDate": None,
    "ModDate": None,
}
PNG_METADATA = {
    "Title": "Carnatic source separation and timbre-audit figures",
    "Author": "Computational Musicology Project",
    "Software": "Matplotlib",
}


def configure_matplotlib() -> None:
    rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.3,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.4,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.7,
            "legend.fontsize": 5.4,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.55,
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
            "mathtext.fontset": "stix",
        }
    )


def load_json(name: str) -> dict[str, Any]:
    path = EXPERIMENTS / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment artifact: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def require_number(mapping: Mapping[str, Any], *keys: str) -> float:
    current: Any = mapping
    walked: list[str] = []
    for key in keys:
        walked.append(key)
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(f"Missing numeric field: {'.'.join(walked)}")
        current = current[key]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise TypeError(f"Expected numeric field {'.'.join(walked)}, got {type(current).__name__}")
    value = float(current)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite numeric field: {'.'.join(walked)} = {value!r}")
    return value


def require_text(mapping: Mapping[str, Any], *keys: str) -> str:
    current: Any = mapping
    walked: list[str] = []
    for key in keys:
        walked.append(key)
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(f"Missing text field: {'.'.join(walked)}")
        current = current[key]
    if not isinstance(current, str) or not current.strip():
        raise TypeError(f"Expected nonempty text field {'.'.join(walked)}")
    return current


def unique_record(payload: Mapping[str, Any], stem_name: str, source: str) -> Mapping[str, Any]:
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list):
        raise TypeError(f"{source}: 'evaluations' must be a list")
    matches = [
        record
        for record in evaluations
        if isinstance(record, Mapping) and record.get("stem_name") == stem_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{source}: expected exactly one evaluation with stem_name={stem_name!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def require_percentage(record: Mapping[str, Any], key: str, source: str) -> float:
    value = require_number(record, key)
    if not 0.0 <= value <= 100.0:
        raise ValueError(f"{source}.{key} must be in [0, 100], got {value}")
    return value


def require_probability(record: Mapping[str, Any], source: str) -> float:
    value = require_number(record, "top_probability")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{source}.top_probability must be in [0, 1], got {value}")
    return value


def compute_architecture() -> dict[str, Any]:
    model = TimbreCNN()
    return {
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "classes": len(INSTRUMENT_CLASSES),
        "mfcc_shape": tuple(
            int(dimension)
            for dimension in extract_mfcc(
                np.zeros(44_100, dtype=np.float32),
                sr=22_050,
            ).shape
        ),
        "demucs_version": importlib.metadata.version("demucs-infer"),
    }


def load_metrics() -> dict[str, Any]:
    concert = load_json("sanidha_4stem_concert03_metrics.json")
    baseline = load_json("baseline_hdemucs_concert3.json")
    milestone4 = load_json("milestone4_timbre_audit.json")
    timbre4 = load_json("sanidha_4stem_timbre_audit.json")

    chunks = int(require_number(concert, "num_evaluated_chunks"))
    if chunks <= 0:
        raise ValueError(f"Expected a positive evaluated-chunk count, got {chunks}")

    stem_names = ("vocals", "violin", "mridangam", "ghatam")
    stems = {
        name: {
            "mean": require_number(concert, "individual_4stem", f"{name}_sdr_mean"),
            "std": require_number(concert, "individual_4stem", f"{name}_sdr_std"),
        }
        for name in stem_names
    }
    collapsed = {
        "overall": require_number(concert, "collapsed_2stem", "overall_mean_2stem_sdr"),
        "paper": require_number(concert, "collapsed_2stem", "sanidha_paper_baseline"),
        "delta": require_number(concert, "collapsed_2stem", "delta_vs_paper"),
        "vocals": {
            "mean": require_number(concert, "collapsed_2stem", "vocals_sdr_mean"),
            "std": require_number(concert, "collapsed_2stem", "vocals_sdr_std"),
        },
        "accompaniment": {
            "mean": require_number(concert, "collapsed_2stem", "accompaniment_sdr_mean"),
            "std": require_number(concert, "collapsed_2stem", "accompaniment_sdr_std"),
        },
    }
    expected_4stem = sum(stems[name]["mean"] for name in stem_names) / len(stem_names)
    stored_4stem = require_number(concert, "individual_4stem", "overall_mean_4stem_sdr")
    if not math.isclose(stored_4stem, expected_4stem, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Stored 4-stem overall mean is inconsistent with per-stem means")
    if not math.isclose(
        collapsed["overall"] - collapsed["paper"],
        collapsed["delta"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Stored 2-stem delta is inconsistent with mean minus paper baseline")
    if collapsed["vocals"]["mean"] != stems["vocals"]["mean"]:
        raise ValueError("Collapsed 2-stem vocal mean differs from the 4-stem vocal mean")

    zero_shot = {
        "vocals": require_number(baseline, "overall_mean_median", "vocals", "SDR_mean"),
        "accompaniment": require_number(
            baseline, "overall_mean_median", "accompaniment", "SDR_mean"
        ),
    }
    for record_name, values in (
        ("baseline vocals", {"SDR": zero_shot["vocals"]}),
        ("baseline accompaniment", {"SDR": zero_shot["accompaniment"]}),
    ):
        for field, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{record_name} {field} is non-finite")

    stock_accomp = unique_record(milestone4, "Baseline Accomp", "milestone4_timbre_audit")
    tuned_accomp = unique_record(milestone4, "Fine-Tuned Accomp", "milestone4_timbre_audit")
    tuned_vocals = unique_record(milestone4, "Fine-Tuned Vocals", "milestone4_timbre_audit")
    accompaniment = {
        "stock": {
            "tabla": require_percentage(stock_accomp, "tabla_bleed_pct", "Baseline Accomp"),
            "strings": require_percentage(
                stock_accomp, "string_bleed_pct", "Baseline Accomp"
            ),
            "top_class": require_text(stock_accomp, "top_class"),
            "top_probability": require_probability(stock_accomp, "Baseline Accomp"),
        },
        "tuned": {
            "tabla": require_percentage(
                tuned_accomp, "tabla_bleed_pct", "Fine-Tuned Accomp"
            ),
            "strings": require_percentage(
                tuned_accomp, "string_bleed_pct", "Fine-Tuned Accomp"
            ),
            "top_class": require_text(tuned_accomp, "top_class"),
            "top_probability": require_probability(tuned_accomp, "Fine-Tuned Accomp"),
        },
    }

    ref_vocals = unique_record(timbre4, "Ref: Vocals", "sanidha_4stem_timbre_audit")
    ref_violin = unique_record(timbre4, "Ref: Violin", "sanidha_4stem_timbre_audit")
    ref_mridangam = unique_record(timbre4, "Ref: Mridangam", "sanidha_4stem_timbre_audit")
    ref_ghatam = unique_record(timbre4, "Ref: Ghatam", "sanidha_4stem_timbre_audit")
    pred_vocals = unique_record(timbre4, "Pred: Vocals", "sanidha_4stem_timbre_audit")
    pred_violin = unique_record(timbre4, "Pred: Violin", "sanidha_4stem_timbre_audit")
    pred_mridangam = unique_record(
        timbre4, "Pred: Mridangam", "sanidha_4stem_timbre_audit"
    )
    pred_ghatam = unique_record(timbre4, "Pred: Ghatam", "sanidha_4stem_timbre_audit")

    def audit_point(record: Mapping[str, Any], label: str) -> dict[str, Any]:
        return {
            "label": label,
            "bleed": require_percentage(record, "percussion_energy_pct", label),
            "melodic": require_percentage(record, "melodic_energy_pct", label),
        }

    purity = [
        {
            "label": "2S voc$^\\ddagger$",
            "bleed": require_percentage(
                tuned_vocals, "tabla_bleed_pct", "Fine-Tuned Vocals"
            ),
            "melodic": require_percentage(
                tuned_vocals, "string_bleed_pct", "Fine-Tuned Vocals"
            ),
        },
        audit_point(pred_vocals, "4S voc"),
        audit_point(pred_violin, "4S vln"),
        audit_point(ref_mridangam, "Mri ref"),
        audit_point(pred_mridangam, "Mri pred"),
        audit_point(ref_ghatam, "Gha ref"),
        audit_point(pred_ghatam, "Gha pred"),
    ]
    # Keep these records in the schema validation even though Fig 2 does not plot them.
    for record, label in (
        (ref_vocals, "Ref: Vocals"),
        (ref_violin, "Ref: Violin"),
    ):
        require_percentage(record, "percussion_energy_pct", label)
        require_percentage(record, "melodic_energy_pct", label)

    return {
        "chunks": chunks,
        "stems": stems,
        "collapsed": collapsed,
        "overall_4stem": stored_4stem,
        "zero_shot": zero_shot,
        "accompaniment": accompaniment,
        "purity": purity,
    }


def add_stage_box(
    ax: plt.Axes,
    x: float,
    width: float,
    bottom: float,
    height: float,
    number: int,
    title: str,
    color: str,
    lines: Sequence[str],
) -> None:
    box = FancyBboxPatch(
        (x, bottom),
        width,
        height,
        boxstyle="round,pad=0.002,rounding_size=0.012",
        linewidth=0.9,
        edgecolor=color,
        facecolor="white",
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.009,
        bottom + height - 0.045,
        str(number),
        fontsize=5.5,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        bbox=dict(boxstyle="circle,pad=0.22", facecolor=color, edgecolor="none"),
        zorder=4,
    )
    ax.text(
        x + width / 2,
        bottom + height - 0.058,
        title,
        fontsize=7.6,
        fontfamily="DejaVu Serif",
        fontweight="bold",
        ha="center",
        va="center",
        color=INK,
        zorder=4,
    )
    body_top = bottom + height - 0.145
    body_bottom = bottom + 0.02
    line_step = (body_top - body_bottom) / max(len(lines), 1)
    for index, line in enumerate(lines):
        y = body_top - line_step * (index + 0.45)
        ax.text(
            x + width / 2,
            y,
            line,
            fontsize=6.0 if len(line) <= 27 else 5.65,
            ha="center",
            va="center",
            color="#222222",
            zorder=4,
        )


def make_fig1(architecture: Mapping[str, Any]) -> plt.Figure:
    fig = plt.figure(figsize=(7.0, 2.3), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    left = 0.012
    right = 0.988
    gap = 0.024
    width = (right - left - 3 * gap) / 4
    bottom = 0.205
    height = 0.745
    positions = [left + index * (width + gap) for index in range(4)]

    mfcc_h, mfcc_w = architecture["mfcc_shape"]
    demucs = architecture["demucs_version"]
    params = f"{architecture['params']:,}"
    stages = [
        (
            "DSP Profiling",
            BLUE,
            [
                "44.1 kHz stereo input",
                "mono DSP signal",
                "STFT: n_fft=2048, hop=512",
                "Median HPSS: k=31, margin=1.0",
                r"$S_{\mathrm{harm}}+S_{\mathrm{perc}}=|X|$",
            ],
        ),
        (
            "Neural Separation",
            ORANGE,
            [
                f"HDemucs {demucs}",
                "frozen encoders",
                "trainable decoders",
                "HDemucs4Stem",
                r"$(B,2,T)\rightarrow(B,4,2,T)$",
                "2-stem: L1 + MR-STFT",
                "4-stem: masked L1",
                "mask: has_ghatam (0/1)",
                "split: see footnote *",
            ],
        ),
        (
            "Resynthesis CLI",
            GREEN,
            [
                "scripts/separate.py",
                "optional --chunk-duration",
                "--num-stems 2|4",
                "44.1 kHz PCM_16 WAV",
            ],
        ),
        (
            "Verification & Audit",
            VERMILLION,
            [
                "BSS-Eval SDR / SI-SDR",
                "2-stem collapse_to_two_stem()",
                f"MFCC {mfcc_h} x {mfcc_w}",
                f"TimbreCNN {params} params",
                f"{architecture['classes']} instrument classes",
                "reference-free Saraga",
                "live HPR auditor",
            ],
        ),
    ]

    for index, (title, color, lines) in enumerate(stages):
        add_stage_box(
            ax,
            positions[index],
            width,
            bottom,
            height,
            index + 1,
            title,
            color,
            lines,
        )

    arrow_y = bottom + height * 0.53
    for index in range(3):
        start = positions[index] + width
        end = positions[index + 1]
        ax.add_patch(
            FancyArrowPatch(
                (start + 0.003, arrow_y),
                (end - 0.003, arrow_y),
                arrowstyle="-|>",
                mutation_scale=7.0,
                linewidth=0.75,
                color="#555555",
                shrinkA=0.0,
                shrinkB=0.0,
                zorder=3,
            )
        )

    ax.text(
        0.5,
        0.145,
        "Workflow note: HDemucs consumes the stereo waveform; HPSS is a profiling/audit branch, "
        "not its tensor input.",
        fontsize=5.15,
        ha="center",
        va="center",
        color="#333333",
    )
    ax.text(
        0.5,
        0.078,
        "* Configured split: C01/C02/C04/C05 train, C03 validation. "
        "Executed 4-stem notebook 05 log: C01/C04/C05 train, C03 validation.",
        fontsize=5.05,
        ha="center",
        va="center",
        color="#333333",
    )
    fig.set_dpi(300)
    return fig


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(pad=1.5)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.4, linestyle=":", alpha=0.65, zorder=0)


def make_fig2(metrics: Mapping[str, Any]) -> plt.Figure:
    fig = plt.figure(figsize=(7.0, 2.9), layout="constrained")
    left_fig, right_fig = fig.subfigures(1, 2, width_ratios=[1.22, 1.0], wspace=0.055)

    left_fig.suptitle(
        "(a) Concert 03 separation SDR",
        fontsize=8.0,
        fontfamily="DejaVu Serif",
        fontweight="bold",
    )
    left_grid = left_fig.add_gridspec(
        3, 1, height_ratios=[0.13, 1.0, 0.16], hspace=0.04, figure=left_fig
    )
    ax_zero = left_fig.add_subplot(left_grid[0])
    ax_zero.axis("off")
    ax = left_fig.add_subplot(left_grid[1])
    ax_note = left_fig.add_subplot(left_grid[2])
    ax_note.axis("off")

    _style_axes(ax)
    positions = [0, 1, 2, 4, 5, 6, 7]
    means = [
        metrics["collapsed"]["overall"],
        metrics["collapsed"]["vocals"]["mean"],
        metrics["collapsed"]["accompaniment"]["mean"],
        metrics["stems"]["vocals"]["mean"],
        metrics["stems"]["violin"]["mean"],
        metrics["stems"]["mridangam"]["mean"],
        metrics["stems"]["ghatam"]["mean"],
    ]
    errors = [
        0.0,
        metrics["collapsed"]["vocals"]["std"],
        metrics["collapsed"]["accompaniment"]["std"],
        metrics["stems"]["vocals"]["std"],
        metrics["stems"]["violin"]["std"],
        metrics["stems"]["mridangam"]["std"],
        metrics["stems"]["ghatam"]["std"],
    ]
    colors = [BLUE, SKY, GREEN, ORANGE, ORANGE, ORANGE, VERMILLION]
    labels = [
        "2S mean",
        "2S voc",
        "2S acc",
        "4S voc",
        "4S vln",
        "4S mri",
        "4S gha$^\\dagger$",
    ]
    ax.bar(
        positions,
        means,
        width=0.62,
        color=colors,
        edgecolor="#333333",
        linewidth=0.4,
        zorder=2,
    )
    for position, mean, error in zip(positions, means, errors):
        if error > 0:
            ax.errorbar(
                position,
                mean,
                yerr=error,
                fmt="none",
                ecolor="#444444",
                elinewidth=0.65,
                capsize=2.0,
                capthick=0.65,
                zorder=4,
            )

    ax.axhline(0.0, color="#222222", linewidth=0.55, zorder=1)
    ax.axvline(3.0, color="#8C8C8C", linewidth=0.65, linestyle=(0, (2, 2)), zorder=1)
    paper = metrics["collapsed"]["paper"]
    ax.plot(
        [-0.55, 2.48],
        [paper, paper],
        color="#555555",
        linewidth=0.8,
        linestyle="--",
        zorder=3,
    )


    for index, (position, mean, error) in enumerate(zip(positions, means, errors)):
        if index == 0:
            text = f"{mean:.2f}\n(+{metrics['collapsed']['delta']:.2f})"
            y = mean + 0.42
            ax.text(position, y, text, fontsize=5.35, ha="center", va="bottom")
        elif index == 6:
            ax.text(
                position,
                mean + error + 0.6,
                f"{mean:.2f} ±{error:.2f}",
                fontsize=4.85,
                ha="center",
                va="bottom",
                color="#222222",
            )
        else:
            y = mean + error + 0.35
            ax.text(
                position,
                y,
                f"{mean:.2f} ±{error:.2f}",
                fontsize=4.95,
                ha="center",
                va="bottom",
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, linespacing=0.95, fontsize=5.15)
    ax.set_ylabel("SDR (dB)", fontsize=6.4)
    ax.set_ylim(-12.0, 18.0)
    ax.set_xlim(-0.68, 8.02)
    ax.set_yticks(np.arange(-12.0, 18.1, 3.0))

    ax.text(
        1.0,
        1.018,
        f"2-stem parity\npaper {paper:.2f} dB (dashed)",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=5.65,
        fontweight="bold",
        linespacing=1.0,
        color=BLUE,
    )
    ax.text(
        5.5,
        1.018,
        f"4-stem breakdown\nnot directly comparable to {paper:.2f} dB",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=5.55,
        fontweight="bold",
        linespacing=1.0,
        color=VERMILLION,
    )

    zero = metrics["zero_shot"]
    badge = (
        "Experiment 0 zero-shot HDemucs: "
        f"vocals {zero['vocals']:.2f} dB | accompaniment {zero['accompaniment']:.2f} dB"
    )
    ax_zero.text(
        0.995,
        0.5,
        badge,
        transform=ax_zero.transAxes,
        ha="right",
        va="center",
        fontsize=4.9,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="#F4F4F4",
            edgecolor="#8C8C8C",
            linewidth=0.45,
            alpha=0.96,
        ),
    )

    ghatam = metrics["stems"]["ghatam"]
    lower = ghatam["mean"] - ghatam["std"]
    clip_note = (
        f"$^\\dagger$ Ghatam absent in C02/C04/C05.\n"
        f"Lower SD whisker: {lower:.2f} dB (clipped at -12 dB)."
    )
    ax_note.text(
        0.5,
        0.55,
        clip_note,
        ha="center",
        va="center",
        fontsize=5.05,
        color="#333333",
        linespacing=1.0,
    )

    right_fig.suptitle(
        "(b) TimbreCNN bleed & domain restoration",
        fontsize=8.0,
        fontfamily="DejaVu Serif",
        fontweight="bold",
    )
    right_grid = right_fig.add_gridspec(
        3, 1, height_ratios=[0.82, 1.45, 0.17], hspace=0.34, figure=right_fig
    )
    ax_restore = right_fig.add_subplot(right_grid[0])
    purity_grid = right_grid[1].subgridspec(1, 2, width_ratios=[0.72, 2.18], wspace=0.08)
    ax_low = right_fig.add_subplot(purity_grid[0])
    ax_high = right_fig.add_subplot(purity_grid[1], sharey=ax_low)
    ax_right_note = right_fig.add_subplot(right_grid[2])
    ax_right_note.axis("off")

    accompaniment = metrics["accompaniment"]
    stock = accompaniment["stock"]
    tuned = accompaniment["tuned"]
    restore_y = [1.0, 0.0]
    series_offset = 0.18
    bar_height = 0.28
    ax_restore.barh(
        [y + series_offset for y in restore_y],
        [stock["tabla"], tuned["tabla"]],
        height=bar_height,
        color=VERMILLION,
        edgecolor="#333333",
        linewidth=0.35,
        zorder=2,
    )
    ax_restore.barh(
        [y - series_offset for y in restore_y],
        [stock["strings"], tuned["strings"]],
        height=bar_height,
        color=BLUE,
        edgecolor="#333333",
        linewidth=0.35,
        zorder=2,
    )
    ax_restore.set_yticks(restore_y)
    ax_restore.set_yticklabels(["Stock\nHDemucs", "Fine-tuned"])
    ax_restore.set_xlim(0.0, 100.0)
    ax_restore.set_ylim(-0.55, 1.68)
    ax_restore.set_xticks([0, 20, 40, 60, 80, 100])
    ax_restore.set_xlabel("Percent", fontsize=5.8)
    ax_restore.set_title(
        "2-stem accompaniment restoration\nbleed = orange  |  Indian strings = blue",
        loc="left",
        fontsize=5.45,
        linespacing=1.0,
        pad=3.0,
    )
    ax_restore.spines["top"].set_visible(False)
    ax_restore.spines["right"].set_visible(False)
    ax_restore.tick_params(pad=1.5)
    ax_restore.grid(axis="x", color="#D9D9D9", linewidth=0.4, linestyle=":", zorder=0)

    for y, values, record in (
        (restore_y[0], stock, stock),
        (restore_y[1], tuned, tuned),
    ):
        ax_restore.text(
            1.0,
            y + 0.43,
            f"top: {record['top_class']} {record['top_probability'] * 100:.1f}%",
            fontsize=4.8,
            ha="left",
            va="bottom",
            color="#333333",
        )
        ax_restore.text(
            values["tabla"] + 1.0,
            y + series_offset,
            f"{values['tabla']:.1f}",
            fontsize=4.7,
            ha="left",
            va="center",
        )
        ax_restore.text(
            values["strings"] + 1.0,
            y - series_offset,
            f"{values['strings']:.1f}",
            fontsize=4.7,
            ha="left",
            va="center",
        )

    purity = metrics["purity"]
    y_positions = np.arange(len(purity) - 1, -1, -1, dtype=float)
    bleed_values = [point["bleed"] for point in purity]
    melodic_values = [point["melodic"] for point in purity]
    low_limit = 0.35
    high_limit = 100.0
    low_bleed = [value if value <= low_limit else 0.0 for value in bleed_values]
    high_bleed = [value if value > low_limit else 0.0 for value in bleed_values]
    ax_low.barh(
        y_positions + series_offset,
        low_bleed,
        height=bar_height,
        color=VERMILLION,
        edgecolor="#333333",
        linewidth=0.3,
        zorder=2,
    )
    ax_high.barh(
        y_positions + series_offset,
        high_bleed,
        height=bar_height,
        color=VERMILLION,
        edgecolor="#333333",
        linewidth=0.3,
        zorder=2,
    )
    ax_high.barh(
        y_positions - series_offset,
        melodic_values,
        height=bar_height,
        color=BLUE,
        edgecolor="#333333",
        linewidth=0.3,
        zorder=2,
    )
    for axis in (ax_low, ax_high):
        axis.spines["top"].set_visible(False)
        axis.tick_params(pad=1.2)
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.35, linestyle=":", zorder=0)

    ax_low.set_xlim(0.0, low_limit)
    ax_high.set_xlim(0.0, high_limit)
    ax_low.set_xticks([0.0, 0.1, 0.2, 0.3])
    ax_high.set_xticks([0, 20, 40, 60, 80, 100])
    ax_low.set_yticks(y_positions)
    ax_low.set_yticklabels([point["label"] for point in purity])
    ax_low.tick_params(axis="y", labelsize=4.9, pad=1.0)
    ax_low.set_ylim(-0.55, len(purity) - 0.45)
    ax_low.set_xlabel("Low\n%", fontsize=5.0, labelpad=1.0)
    ax_high.set_xlabel("Percent", fontsize=5.6, labelpad=1.0)
    ax_high.set_title(
        "Purity (blue) vs bleed (orange)",
        loc="left",
        fontsize=5.9,
        pad=3.0,
    )
    ax_low.spines["right"].set_visible(False)
    ax_high.spines["left"].set_visible(False)
    ax_high.tick_params(left=False, labelleft=False)

    break_size = 0.025
    low_kwargs = dict(transform=ax_low.transAxes, color="#333333", clip_on=False, linewidth=0.55)
    high_kwargs = dict(transform=ax_high.transAxes, color="#333333", clip_on=False, linewidth=0.55)
    ax_low.plot((1 - break_size, 1 + break_size), (-break_size, break_size), **low_kwargs)
    ax_low.plot((1 - break_size, 1 + break_size), (1 - break_size, 1 + break_size), **low_kwargs)
    ax_high.plot((-break_size, break_size), (-break_size, break_size), **high_kwargs)
    ax_high.plot((-break_size, break_size), (1 - break_size, 1 + break_size), **high_kwargs)

    for index, point in enumerate(purity):
        y = y_positions[index]
        bleed = point["bleed"]
        melodic = point["melodic"]
        if bleed <= low_limit:
            ax_low.text(
                bleed + 0.008,
                y + series_offset - 0.055,
                f"{bleed:.2f}",
                fontsize=4.0,
                ha="left",
                va="center",
                clip_on=True,
            )
        else:
            ax_high.text(
                bleed + 1.0,
                y + series_offset - 0.055,
                f"{bleed:.1f}",
                fontsize=4.0,
                ha="left",
                va="center",
                clip_on=True,
            )
        ax_high.text(
            melodic + 1.0,
            y - series_offset + 0.055,
            f"{melodic:.1f}",
            fontsize=4.0,
            ha="left",
            va="center",
            clip_on=True,
        )

    ax_right_note.text(
        0.0,
        0.52,
        "Broken linear x-axis: low 0-0.35%, main 0-100%.\n"
        "$^\\ddagger$ 2-stem = tabla-only bleed; 4-stem = drums + tabla percussion energy.",
        fontsize=4.45,
        ha="left",
        va="center",
        linespacing=1.0,
        color="#333333",
    )
    fig.set_dpi(300)
    return fig


def assert_no_text_overlap(figure: plt.Figure, label: str) -> None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width, height = figure.canvas.get_width_height()
    boxes: list[tuple[str, Any]] = []
    for artist in figure.findobj(Text):
        if not artist.get_visible():
            continue
        text = artist.get_text().strip()
        if not text:
            continue
        box = artist.get_window_extent(renderer=renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        if (
            box.x0 < -0.5
            or box.y0 < -0.5
            or box.x1 > width + 0.5
            or box.y1 > height + 0.5
        ):
            raise AssertionError(
                f"{label}: text outside canvas: {text!r} box={tuple(round(v, 2) for v in box.bounds)}"
            )
        boxes.append((text, box))

    collisions: list[str] = []
    for index, (text_a, box_a) in enumerate(boxes):
        padded_a = box_a.expanded(1.01, 1.04)
        for text_b, box_b in boxes[index + 1 :]:
            if padded_a.overlaps(box_b):
                collisions.append(f"{text_a!r} <-> {text_b!r}")
    if collisions:
        details = "\n".join(collisions[:20])
        raise AssertionError(f"{label}: {len(collisions)} text collisions:\n{details}")
    print(f"PASS {label}: {len(boxes)} text artists, no overlaps, all inside canvas")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure(figure: plt.Figure, stem: str) -> tuple[Path, Path]:
    pdf_path = OUT_DIR / f"{stem}.pdf"
    png_path = OUT_DIR / f"{stem}.png"
    figure.savefig(
        pdf_path,
        format="pdf",
        metadata=PDF_METADATA,
        facecolor="white",
        edgecolor="none",
    )
    figure.savefig(
        png_path,
        format="png",
        dpi=300,
        metadata=PNG_METADATA,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(figure)
    return pdf_path, png_path


def validate_outputs(outputs: Sequence[tuple[Path, Path, tuple[int, int]]]) -> None:
    for pdf_path, png_path, expected_size in outputs:
        if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty PDF: {pdf_path}")
        if not png_path.is_file() or png_path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty PNG: {png_path}")
        if not pdf_path.read_bytes().startswith(b"%PDF"):
            raise AssertionError(f"Not a PDF: {pdf_path}")
        with Image.open(png_path) as image:
            if image.size != expected_size:
                raise AssertionError(
                    f"{png_path}: expected size {expected_size}, got {image.size}"
                )
            dpi = image.info.get("dpi")
            if dpi is None or any(abs(float(value) - 300.0) > 0.02 for value in dpi):
                raise AssertionError(f"{png_path}: expected 300 DPI, got {dpi!r}")
    print("PASS output dimensions/DPI and PDF signatures")


def main() -> None:
    configure_matplotlib()
    architecture = compute_architecture()
    if architecture["params"] != 94_410:
        raise ValueError(f"Unexpected TimbreCNN parameter count: {architecture['params']}")
    if architecture["classes"] != 10:
        raise ValueError(f"Unexpected TimbreCNN class count: {architecture['classes']}")
    if architecture["mfcc_shape"] != (40, 87):
        raise ValueError(f"Unexpected MFCC shape: {architecture['mfcc_shape']}")
    if architecture["demucs_version"] != "4.2.2":
        raise ValueError(f"Unexpected demucs-infer version: {architecture['demucs_version']}")

    metrics = load_metrics()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig1 = make_fig1(architecture)
    assert_no_text_overlap(fig1, "fig1_pipeline")
    fig1_paths = save_figure(fig1, "fig1_pipeline")

    fig2 = make_fig2(metrics)
    assert_no_text_overlap(fig2, "fig2_results_bleed")
    fig2_paths = save_figure(fig2, "fig2_results_bleed")

    outputs = [
        (*fig1_paths, (2100, 690)),
        (*fig2_paths, (2100, 870)),
    ]
    validate_outputs(outputs)

    print(
        "Metrics: "
        f"2-stem {metrics['collapsed']['overall']:.6f} dB, "
        f"paper {metrics['collapsed']['paper']:.2f} dB, "
        f"delta {metrics['collapsed']['delta']:.6f} dB, "
        f"4-stem {metrics['overall_4stem']:.6f} dB, "
        f"chunks {metrics['chunks']}"
    )
    for pdf_path, png_path, _ in outputs:
        print(f"{pdf_path.relative_to(ROOT)} sha256={sha256(pdf_path)}")
        print(f"{png_path.relative_to(ROOT)} sha256={sha256(png_path)}")


if __name__ == "__main__":
    main()

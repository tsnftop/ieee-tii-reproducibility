"""Plot scenario evaluation times from a saved aggregate report.

Read per-strategy timing statistics and generate a logarithmic comparison
figure without running simulations.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter


import argparse
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))
from repo_paths import inside, new_output
REPORT = None
OUT_BASE = None

ORDER = [
    ("MPC supervisory", "supervisory MPC"),
    ("RL Global", "supervisory RL"),
    ("MPC beta-sequence", "allocation-only MPC"),
    ("RL Global (restricted)", "allocation-only RL"),
    ("Rule-Based", "Rule-Based"),
    ("Uniform + Constraints", "Uniform + Constraints"),
    ("Proportional + Constraints", "Proportional + Constraints"),
    ("Uniform (baseline)", "Uniform baseline"),
]


def load_report() -> dict[str, dict[str, float]]:
    text = REPORT.read_text(encoding="utf-8").splitlines()
    header_idx = next(i for i, line in enumerate(text) if line.startswith("Scenario,"))
    reader = csv.DictReader(text[header_idx:])
    rows: dict[str, dict[str, float]] = {}

    for row in reader:
        scenario = row["Scenario"]
        rows[scenario] = {
            "time_mean": float(row["Time mean"]),
            "time_std": float(row["Time std"]),
        }

    return rows


def main() -> None:
    global REPORT, OUT_BASE
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="Repository-relative aggregate.txt")
    parser.add_argument("--output", required=True, help="New subdirectory of main/outputs")
    args = parser.parse_args()
    REPORT = inside(args.report)
    if not REPORT.is_file():
        raise FileNotFoundError(REPORT)
    rows = load_report()
    OUT_BASE = new_output(args.output) / "fig_results_summary"

    labels = [display for key, display in ORDER]
    times = np.array([rows[key]["time_mean"] for key, _ in ORDER], dtype=float)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "axes.linewidth": 0.6,
        "savefig.dpi": 300,
    })

    y = np.arange(len(labels))

    facecolors = ["0.15", "0.35", "0.55", "0.70", "0.82", "0.90", "0.96", "1.0"]
    hatches = ["", "", "///", "\\\\\\", "...", "xx", "--", ""]

    # Figure width accommodates the legend and value annotations.
    fig, ax_time = plt.subplots(1, 1, figsize=(4.35, 2.4))

    bars_t = ax_time.barh(
        y,
        times,
        color=facecolors,
        edgecolor="black",
        linewidth=0.5,
    )

    for bar, hatch in zip(bars_t, hatches):
        bar.set_hatch(hatch)

    ax_time.set_xscale("log")
    ax_time.set_yticks(y)
    ax_time.set_yticklabels(labels)
    ax_time.invert_yaxis()
    ax_time.set_xlabel("Online evaluation time per week (s)")

    # Extend the logarithmic axis to separate the largest annotation from the frame.
    
    ax_time.set_xlim(0.02, 3000.0)

    ax_time.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax_time.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax_time.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax_time.xaxis.set_minor_formatter(NullFormatter())

    for yi, val in zip(y, times):
        label = f"{val:.1f}" if val >= 10.0 else f"{val:.2f}"

        # Place value labels multiplicatively along the logarithmic axis.
        # The axis limit leaves space for the largest value label.
        ax_time.text(
            val * 1.18,
            yi,
            label,
            va="center",
            ha="left",
            fontsize=8,
            color="black",
            clip_on=False,
        )

    # Manual layout margins preserve space for value annotations.
    fig.subplots_adjust(left=0.44, right=0.985, bottom=0.25, top=0.96)

    OUT_BASE.with_suffix(".pdf").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_BASE.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_BASE.with_suffix(".png"), bbox_inches="tight", pad_inches=0.04)

    print(f"Wrote {OUT_BASE.with_suffix('.pdf')}")
    print(f"Wrote {OUT_BASE.with_suffix('.png')}")


if __name__ == "__main__":
    main()

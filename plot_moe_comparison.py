from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

if "--show" not in sys.argv:
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


EXPERT_COLUMNS = ("gate_e0", "gate_e1", "gate_e2", "gate_e3")
EXPERT_COLORS = {
    "gate_e0": "#4C78A8",
    "gate_e1": "#F58518",
    "gate_e2": "#54A24B",
    "gate_e3": "#E45756",
}
MODE_COLORS = {
    "active": "#4C78A8",
    "TEACH": "#E8A87C",
    "AUTOPILOT": "#72B7B2",
}
MODE_TITLES = {
    "active": "All Active Modes",
    "TEACH": "TEACH",
    "AUTOPILOT": "AUTOPILOT",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a multi-panel comparison figure for MoE routing runs, "
            "showing mean per-expert utilization and gate-entropy summaries."
        )
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        required=True,
        help="Run specification in the form LABEL=CSV_PATH. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--output",
        default="fca_analysis_studio/outputs/plots/moe_routing_comparison.png",
        help="Output PNG path for the derived comparison figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output DPI for the saved PNG.",
    )
    parser.add_argument(
        "--title",
        default="Derived MoE Routing Comparison",
        help="Figure title.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving.",
    )
    return parser.parse_args()


def parse_run_spec(spec_text: str):
    label, sep, raw_path = str(spec_text or "").partition("=")
    if not sep or not label.strip() or not raw_path.strip():
        raise ValueError(f"Invalid --run value: {spec_text!r}. Expected LABEL=CSV_PATH.")
    path = Path(raw_path.strip()).expanduser()
    return label.strip(), path


def safe_float(value):
    text = str(value or "").strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def load_rows(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def is_mode_match(row, mode_name: str):
    mode = str(row.get("mode") or "").strip()
    if mode_name == "active":
        return mode not in {"", "PAUSED"}
    return mode == mode_name


def normalized_gate_probs(row):
    values = [safe_float(row.get(column)) for column in EXPERT_COLUMNS]
    if any(math.isnan(value) for value in values):
        return None
    total = sum(values)
    if total <= 0.0:
        return None
    return [value / total for value in values]


def mean_gate_probs(rows, mode_name: str):
    accum = np.zeros(len(EXPERT_COLUMNS), dtype=np.float64)
    count = 0
    for row in rows:
        if not is_mode_match(row, mode_name):
            continue
        probs = normalized_gate_probs(row)
        if probs is None:
            continue
        accum += np.asarray(probs, dtype=np.float64)
        count += 1
    if count <= 0:
        return [math.nan] * len(EXPERT_COLUMNS), 0
    return (accum / float(count)).tolist(), count


def mean_gate_entropy(rows, mode_name: str):
    values = []
    for row in rows:
        if not is_mode_match(row, mode_name):
            continue
        entropy = safe_float(row.get("gate_entropy"))
        if math.isnan(entropy):
            probs = normalized_gate_probs(row)
            if probs is None:
                continue
            entropy = float(-sum(prob * math.log(max(prob, 1e-12)) for prob in probs))
        values.append(entropy)
    if not values:
        return math.nan
    return float(sum(values) / len(values))


def build_run_summary(rows):
    summary = {}
    for mode_name in ("active", "TEACH", "AUTOPILOT"):
        means, count = mean_gate_probs(rows, mode_name)
        summary[mode_name] = {
            "count": count,
            "mean_probs": means,
            "mean_entropy": mean_gate_entropy(rows, mode_name),
        }
    return summary


def plot_utilization_panel(ax, run_labels, run_summaries, mode_name):
    indices = np.arange(len(run_labels), dtype=np.float64)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(EXPERT_COLUMNS))

    for expert_index, column in enumerate(EXPERT_COLUMNS):
        heights = [run_summaries[label][mode_name]["mean_probs"][expert_index] for label in run_labels]
        ax.bar(
            indices + offsets[expert_index],
            heights,
            width=width,
            color=EXPERT_COLORS[column],
            label=column,
        )

    ax.set_title(MODE_TITLES[mode_name])
    ax.set_xticks(indices)
    ax.set_xticklabels(run_labels, rotation=12)
    ax.set_ylabel("mean gate probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.2)


def plot_entropy_panel(ax, run_labels, run_summaries):
    indices = np.arange(len(run_labels), dtype=np.float64)
    width = 0.24
    mode_names = ("active", "TEACH", "AUTOPILOT")
    offsets = np.linspace(-width, width, len(mode_names))

    for mode_index, mode_name in enumerate(mode_names):
        heights = [run_summaries[label][mode_name]["mean_entropy"] for label in run_labels]
        ax.bar(
            indices + offsets[mode_index],
            heights,
            width=width,
            color=MODE_COLORS[mode_name],
            label=MODE_TITLES[mode_name],
        )

    ax.set_title("Mean Gate Entropy")
    ax.set_xticks(indices)
    ax.set_xticklabels(run_labels, rotation=12)
    ax.set_ylabel("entropy")
    ax.set_ylim(0.0, math.log(4.0) + 0.08)
    ax.grid(axis="y", alpha=0.2)


def make_figure(run_labels, run_summaries, title_text):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(title_text)

    plot_utilization_panel(axes[0, 0], run_labels, run_summaries, "active")
    plot_utilization_panel(axes[0, 1], run_labels, run_summaries, "TEACH")
    plot_utilization_panel(axes[1, 0], run_labels, run_summaries, "AUTOPILOT")
    plot_entropy_panel(axes[1, 1], run_labels, run_summaries)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, loc="upper right", fontsize=9)
    handles, labels = axes[1, 1].get_legend_handles_labels()
    axes[1, 1].legend(handles, labels, loc="upper right", fontsize=9)

    return fig


def main():
    args = parse_args()
    run_labels = []
    run_summaries = {}

    for spec_text in args.runs:
        label, csv_path = parse_run_spec(spec_text)
        rows = load_rows(csv_path)
        run_labels.append(label)
        run_summaries[label] = build_run_summary(rows)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = make_figure(run_labels, run_summaries, args.title)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(output_path)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
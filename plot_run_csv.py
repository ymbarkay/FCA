from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

if "--show" not in sys.argv:
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt


MODE_COLORS = {
    "AUTOPILOT": "#8FC9B0",
    "TEACH": "#E8A87C",
    "REVERSE_MANUAL": "#D4A85A",
    "PAUSED": "#8A8D8A",
    "DATASET_COLLECTION": "#7DB7D5",
}

EXPERT_COLORS = {
    "gate_e0": "#4C78A8",
    "gate_e1": "#F58518",
    "gate_e2": "#54A24B",
    "gate_e3": "#E45756",
}

INTENT_COLORS = {
    "intent_stop_prob": "#7F3C8D",
    "intent_left_prob": "#11A579",
    "intent_straight_prob": "#3969AC",
    "intent_right_prob": "#F2B701",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot useful runtime, learning, and MoE-routing summaries from one or more FCA run CSV logs."
        )
    )
    parser.add_argument(
        "csv_paths",
        nargs="+",
        help="One or more frame or correction CSV paths to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Optional output directory. If omitted, each input writes to <csv_dir>/<csv_stem>_plots/. "
            "Use this only with a single CSV path."
        ),
    )
    parser.add_argument(
        "--x-axis",
        choices=["auto", "elapsed_s", "total_updates", "row"],
        default="auto",
        help="X axis for plots. 'auto' prefers total_updates when it varies meaningfully, else elapsed_s.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=9,
        help="Simple moving-average window for timing plots. Use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI for PNG figures.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    return parser.parse_args()


def safe_float(value):
    text = str(value or "").strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def safe_int(value):
    numeric = safe_float(value)
    if math.isnan(numeric):
        return None
    return int(numeric)


def load_rows(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def numeric_series(rows, key):
    return [safe_float(row.get(key)) for row in rows]


def valid_values(values):
    return [value for value in values if not math.isnan(value)]


def moving_average(values, window):
    if window <= 1:
        return list(values)

    out = []
    running_sum = 0.0
    running_count = 0
    queue = []

    for value in values:
        queue.append(value)
        if not math.isnan(value):
            running_sum += value
            running_count += 1

        if len(queue) > window:
            removed = queue.pop(0)
            if not math.isnan(removed):
                running_sum -= removed
                running_count -= 1

        out.append(running_sum / running_count if running_count > 0 else math.nan)

    return out


def choose_x(rows, preferred):
    if preferred == "row":
        return list(range(len(rows))), "row"

    elapsed = numeric_series(rows, "elapsed_s")
    elapsed_valid = valid_values(elapsed)
    updates = numeric_series(rows, "total_updates")
    update_valid = valid_values(updates)
    update_unique = {int(value) for value in update_valid}

    if preferred == "elapsed_s":
        if elapsed_valid:
            return elapsed, "elapsed_s"
        return list(range(len(rows))), "row"

    if preferred == "total_updates":
        if update_valid:
            return updates, "total_updates"
        if elapsed_valid:
            return elapsed, "elapsed_s"
        return list(range(len(rows))), "row"

    if len(update_unique) >= 20 and max(update_unique) > 0:
        return updates, "total_updates"
    if elapsed_valid:
        return elapsed, "elapsed_s"
    return list(range(len(rows))), "row"


def mode_spans(rows, x_values):
    spans = []
    if not rows:
        return spans

    current_mode = str(rows[0].get("mode") or "").strip()
    start_index = 0

    for index in range(1, len(rows)):
        mode = str(rows[index].get("mode") or "").strip()
        if mode != current_mode:
            spans.append((current_mode, x_values[start_index], x_values[index - 1]))
            current_mode = mode
            start_index = index

    spans.append((current_mode, x_values[start_index], x_values[-1]))
    return spans


def add_mode_background(ax, spans):
    for mode, start, end in spans:
        if isinstance(start, float) and math.isnan(start):
            continue
        if isinstance(end, float) and math.isnan(end):
            continue
        color = MODE_COLORS.get(mode, "#CCCCCC")
        if start == end:
            end = start + 1e-9
        ax.axvspan(start, end, color=color, alpha=0.08, linewidth=0)


def top_expert_indices(rows):
    indices = []
    for row in rows:
        value = str(row.get("top_expert") or "").strip()
        if value.startswith("gate_e") and value[-1].isdigit():
            indices.append(int(value[-1]))
        else:
            indices.append(math.nan)
    return indices


def has_any_nonzero(values):
    for value in values:
        if not math.isnan(value) and abs(value) > 1e-12:
            return True
    return False


def summarise(rows, fieldnames, x_label):
    modes = Counter(str(row.get("mode") or "").strip() for row in rows)
    top_experts = Counter(str(row.get("top_expert") or "").strip() for row in rows)
    intents = Counter(str(row.get("intent_pred") or "").strip() for row in rows)

    gate_entropy = valid_values(numeric_series(rows, "gate_entropy"))
    fps = valid_values(numeric_series(rows, "fps"))
    inference = valid_values(numeric_series(rows, "inference_ms"))
    adapter = valid_values(numeric_series(rows, "adapter_ms"))
    updates = valid_values(numeric_series(rows, "total_updates"))
    losses = valid_values(numeric_series(rows, "last_teach_loss"))

    summary_lines = [
        f"rows: {len(rows)}",
        f"columns: {', '.join(fieldnames)}",
        f"x_axis: {x_label}",
        f"modes: {dict(modes)}",
        f"top_experts: {dict(top_experts)}",
        f"intent_preds: {dict(intents)}",
    ]

    if gate_entropy:
        summary_lines.append(f"gate_entropy_mean: {sum(gate_entropy) / len(gate_entropy):.6f}")
    if fps:
        summary_lines.append(f"fps_mean: {sum(fps) / len(fps):.6f}")
    if inference:
        summary_lines.append(f"inference_ms_mean: {sum(inference) / len(inference):.6f}")
    if adapter:
        summary_lines.append(f"adapter_ms_mean: {sum(adapter) / len(adapter):.6f}")
    if updates:
        summary_lines.append(f"total_updates_max: {int(max(updates))}")
    if losses:
        summary_lines.append(f"last_teach_loss_last: {losses[-1]:.6f}")

    return "\n".join(summary_lines) + "\n"


def plot_overview(csv_path: Path, rows, x_values, x_label, output_dir: Path, rolling_window: int, dpi: int):
    spans = mode_spans(rows, x_values)

    final_angle = numeric_series(rows, "final_angle_car")
    selected_angle = numeric_series(rows, "selected_angle_car")
    base_speed_prob = numeric_series(rows, "base_speed_prob")
    final_speed = numeric_series(rows, "final_speed_car")
    fps = moving_average(numeric_series(rows, "fps"), rolling_window)
    inference_ms = moving_average(numeric_series(rows, "inference_ms"), rolling_window)
    adapter_ms = moving_average(numeric_series(rows, "adapter_ms"), rolling_window)
    total_updates = numeric_series(rows, "total_updates")
    replay_buffer = numeric_series(rows, "replay_buffer_size")
    last_teach_loss = numeric_series(rows, "last_teach_loss")
    human_active = numeric_series(rows, "human_active")

    fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True, constrained_layout=True)
    fig.suptitle(f"Run overview: {csv_path.name}")

    for ax in axes:
        add_mode_background(ax, spans)

    axes[0].plot(x_values, final_angle, color="#0B84A5", linewidth=1.3, label="final_angle_car")
    if has_any_nonzero(selected_angle):
        axes[0].plot(x_values, selected_angle, color="#F6C85F", linewidth=1.0, alpha=0.85, label="selected_angle_car")
    if has_any_nonzero(human_active):
        ymin, ymax = axes[0].get_ylim()
        axes[0].fill_between(
            x_values,
            ymin,
            ymax,
            where=[not math.isnan(v) and v > 0.0 for v in human_active],
            color="#E8A87C",
            alpha=0.08,
            label="human_active",
        )
    axes[0].set_ylabel("angle (car)")
    axes[0].legend(loc="upper right", ncol=3, fontsize=9)
    axes[0].grid(alpha=0.2)

    axes[1].plot(x_values, base_speed_prob, color="#6F4E7C", linewidth=1.2, label="base_speed_prob")
    speed_ax = axes[1].twinx()
    speed_ax.plot(x_values, final_speed, color="#9DD866", linewidth=1.1, alpha=0.9, label="final_speed_car")
    axes[1].set_ylabel("speed prob")
    speed_ax.set_ylabel("final speed")
    axes[1].grid(alpha=0.2)
    handles_left, labels_left = axes[1].get_legend_handles_labels()
    handles_right, labels_right = speed_ax.get_legend_handles_labels()
    axes[1].legend(handles_left + handles_right, labels_left + labels_right, loc="upper right", fontsize=9)

    axes[2].plot(x_values, adapter_ms, color="#E45756", linewidth=1.2, label="adapter_ms")
    axes[2].plot(x_values, inference_ms, color="#72B7B2", linewidth=1.1, label="inference_ms")
    fps_ax = axes[2].twinx()
    fps_ax.plot(x_values, fps, color="#54A24B", linewidth=1.1, label="fps")
    axes[2].set_ylabel("latency (ms)")
    fps_ax.set_ylabel("fps")
    axes[2].grid(alpha=0.2)
    handles_left, labels_left = axes[2].get_legend_handles_labels()
    handles_right, labels_right = fps_ax.get_legend_handles_labels()
    axes[2].legend(handles_left + handles_right, labels_left + labels_right, loc="upper right", fontsize=9)

    axes[3].plot(x_values, total_updates, color="#4C78A8", linewidth=1.2, label="total_updates")
    if has_any_nonzero(replay_buffer):
        axes[3].plot(x_values, replay_buffer, color="#B279A2", linewidth=1.0, alpha=0.85, label="replay_buffer_size")
    loss_ax = axes[3].twinx()
    loss_ax.plot(x_values, last_teach_loss, color="#F58518", linewidth=1.0, alpha=0.9, label="last_teach_loss")
    axes[3].set_ylabel("updates / buffer")
    loss_ax.set_ylabel("loss")
    axes[3].set_xlabel(x_label)
    axes[3].grid(alpha=0.2)
    handles_left, labels_left = axes[3].get_legend_handles_labels()
    handles_right, labels_right = loss_ax.get_legend_handles_labels()
    axes[3].legend(handles_left + handles_right, labels_left + labels_right, loc="upper left", fontsize=9)

    output_path = output_dir / f"{csv_path.stem}_overview.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_routing(csv_path: Path, rows, x_values, x_label, output_dir: Path, dpi: int):
    gate_keys = [key for key in ("gate_e0", "gate_e1", "gate_e2", "gate_e3") if key in rows[0]]
    if not gate_keys:
        return None

    gate_series = {key: numeric_series(rows, key) for key in gate_keys}
    entropy = numeric_series(rows, "gate_entropy")
    top_expert = top_expert_indices(rows)
    gate_max = []
    gate_margin = []
    for row in rows:
        gate_values = [safe_float(row.get(key)) for key in gate_keys]
        finite = [value for value in gate_values if not math.isnan(value)]
        if len(finite) != len(gate_keys):
            gate_max.append(math.nan)
            gate_margin.append(math.nan)
            continue
        ordered = sorted(finite, reverse=True)
        gate_max.append(ordered[0])
        gate_margin.append(ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0])

    intent_keys = [key for key in INTENT_COLORS if key in rows[0]]
    intent_available = any(has_any_nonzero(numeric_series(rows, key)) for key in intent_keys)

    spans = mode_spans(rows, x_values)
    nrows = 4 if intent_available else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(15, 12 if intent_available else 10), sharex=True, constrained_layout=True)
    fig.suptitle(f"Routing diagnostics: {csv_path.name}")

    for ax in axes:
        add_mode_background(ax, spans)

    for key in gate_keys:
        axes[0].plot(x_values, gate_series[key], linewidth=1.0, label=key, color=EXPERT_COLORS.get(key, None))
    axes[0].set_ylabel("gate prob")
    axes[0].legend(loc="upper right", ncol=4, fontsize=9)
    axes[0].grid(alpha=0.2)

    axes[1].plot(x_values, entropy, color="#6F4E7C", linewidth=1.2, label="gate_entropy")
    max_ax = axes[1].twinx()
    max_ax.plot(x_values, gate_max, color="#54A24B", linewidth=1.0, alpha=0.9, label="gate_max")
    max_ax.plot(x_values, gate_margin, color="#E45756", linewidth=1.0, alpha=0.7, label="gate_margin")
    axes[1].set_ylabel("entropy")
    max_ax.set_ylabel("max / margin")
    axes[1].grid(alpha=0.2)
    handles_left, labels_left = axes[1].get_legend_handles_labels()
    handles_right, labels_right = max_ax.get_legend_handles_labels()
    axes[1].legend(handles_left + handles_right, labels_left + labels_right, loc="upper right", fontsize=9)

    expert_scatter_x = []
    expert_scatter_y = []
    for x_val, expert_index in zip(x_values, top_expert):
        if math.isnan(expert_index):
            continue
        expert_scatter_x.append(x_val)
        expert_scatter_y.append(expert_index)
    axes[2].scatter(expert_scatter_x, expert_scatter_y, s=8, c="#0B84A5", alpha=0.7)
    axes[2].set_ylabel("top expert")
    axes[2].set_yticks([0, 1, 2, 3])
    axes[2].set_yticklabels(["e0", "e1", "e2", "e3"])
    axes[2].grid(alpha=0.2)

    if intent_available:
        for key in intent_keys:
            axes[3].plot(x_values, numeric_series(rows, key), linewidth=1.0, label=key, color=INTENT_COLORS[key])
        axes[3].set_ylabel("intent prob")
        axes[3].legend(loc="upper right", ncol=4, fontsize=9)
        axes[3].grid(alpha=0.2)
        axes[3].set_xlabel(x_label)
    else:
        axes[2].set_xlabel(x_label)

    output_path = output_dir / f"{csv_path.stem}_routing.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def write_summary(csv_path: Path, output_dir: Path, summary_text: str):
    summary_path = output_dir / f"{csv_path.stem}_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    return summary_path


def process_csv(csv_path: Path, output_dir: Path, x_axis: str, rolling_window: int, dpi: int, show: bool):
    rows, fieldnames = load_rows(csv_path)
    if not rows:
        raise ValueError(f"CSV contains no rows: {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    x_values, x_label = choose_x(rows, x_axis)
    summary_text = summarise(rows, fieldnames, x_label)
    summary_path = write_summary(csv_path, output_dir, summary_text)
    overview_path = plot_overview(csv_path, rows, x_values, x_label, output_dir, rolling_window, dpi)
    routing_path = None
    if any(key in fieldnames for key in ("gate_e0", "gate_e1", "gate_e2", "gate_e3")):
        routing_path = plot_routing(csv_path, rows, x_values, x_label, output_dir, dpi)

    print(f"[plot_run_csv] csv: {csv_path}")
    print(f"[plot_run_csv] summary: {summary_path}")
    print(f"[plot_run_csv] overview: {overview_path}")
    if routing_path is not None:
        print(f"[plot_run_csv] routing: {routing_path}")
    print(summary_text.strip())

    if show:
        plt.show()


def main():
    args = parse_args()

    if args.output_dir and len(args.csv_paths) > 1:
        raise SystemExit("--output-dir can only be used with a single CSV path.")

    for csv_arg in args.csv_paths:
        csv_path = Path(csv_arg).expanduser().resolve()
        if not csv_path.exists():
            raise SystemExit(f"CSV not found: {csv_path}")

        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser().resolve()
        else:
            output_dir = csv_path.parent / f"{csv_path.stem}_plots"

        process_csv(
            csv_path=csv_path,
            output_dir=output_dir,
            x_axis=args.x_axis,
            rolling_window=max(1, int(args.rolling_window)),
            dpi=max(72, int(args.dpi)),
            show=bool(args.show),
        )


if __name__ == "__main__":
    main()
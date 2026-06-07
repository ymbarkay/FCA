from __future__ import annotations

import math

import numpy as np
import pandas as pd


SEMANTIC_LABEL_COLORS = {
    "stop": "#CC79A7",
    "left": "#009E73",
    "straight": "#0072B2",
    "right": "#D55E00",
    "unknown": "#7A7A7A",
}
FALLBACK_LABEL_COLORS = (
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AB",
)


def _require_matplotlib_pyplot():
    import matplotlib.pyplot as plt

    return plt


def _semantic_label_key(label):
    text = str(label or "").strip().lower()
    if not text:
        return "unknown"
    if "stop" in text:
        return "stop"
    if "left" in text:
        return "left"
    if "straight" in text:
        return "straight"
    if "right" in text:
        return "right"
    return "unknown"


def ordered_label_classes(labels):
    classes = sorted(pd.Series(labels, dtype=str).unique(), key=str.lower)
    semantic_order = {"stop": 0, "left": 1, "straight": 2, "right": 3, "unknown": 4}
    return sorted(classes, key=lambda label: (semantic_order.get(_semantic_label_key(label), 99), str(label).lower()))


def label_color_mapping(labels):
    mapping = {}
    fallback_index = 0
    for label in ordered_label_classes(labels):
        semantic_key = _semantic_label_key(label)
        color = SEMANTIC_LABEL_COLORS.get(semantic_key)
        if semantic_key == "unknown":
            color = FALLBACK_LABEL_COLORS[fallback_index % len(FALLBACK_LABEL_COLORS)]
            fallback_index += 1
        mapping[str(label)] = color
    return mapping


def plot_confusion_matrix(confusion, labels, title):
    plt = _require_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    image = ax.imshow(confusion, cmap="YlGnBu")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    threshold = confusion.max() / 2.0 if confusion.size else 0.0
    for row_index in range(confusion.shape[0]):
        for col_index in range(confusion.shape[1]):
            ax.text(
                col_index,
                row_index,
                int(confusion[row_index, col_index]),
                ha="center",
                va="center",
                color="white" if confusion[row_index, col_index] > threshold else "#16202A",
                fontsize=9,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_embedding_scatter(embedding, labels, title, axis_names=("x", "y"), class_order=None, color_map=None):
    plt = _require_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    label_series = pd.Series(labels, dtype=str)
    classes = list(class_order or ordered_label_classes(label_series))
    colors = dict(color_map or label_color_mapping(classes))

    for class_name in classes:
        mask = label_series == class_name
        if not mask.any():
            continue
        ax.scatter(
            embedding[mask.to_numpy(), 0],
            embedding[mask.to_numpy(), 1],
            s=28,
            alpha=0.82,
            label=class_name,
            color=colors[str(class_name)],
            edgecolors="white",
            linewidths=0.45,
        )

    ax.set_title(title)
    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(alpha=0.14)
    fig.tight_layout()
    return fig


def plot_embedding_comparison(panels, title, axis_names=("x", "y")):
    plt = _require_matplotlib_pyplot()
    if not panels:
        raise ValueError("At least one panel is required.")

    all_labels = []
    for panel in panels:
        all_labels.extend([str(value) for value in panel["labels"]])

    classes = ordered_label_classes(all_labels)
    colors = label_color_mapping(classes)
    fig, axes = plt.subplots(1, len(panels), figsize=(6.8 * len(panels), 5.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    prefixes = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    for index, (axis, panel) in enumerate(zip(axes, panels)):
        embedding = panel["embedding"]
        label_series = pd.Series(panel["labels"], dtype=str)

        for class_name in classes:
            mask = label_series == class_name
            if not mask.any():
                continue
            axis.scatter(
                embedding[mask.to_numpy(), 0],
                embedding[mask.to_numpy(), 1],
                s=30,
                alpha=0.84,
                color=colors[str(class_name)],
                edgecolors="white",
                linewidths=0.5,
                label=class_name,
            )

        prefix = prefixes[index] if index < len(prefixes) else f"({index + 1})"
        axis.set_title(f"{prefix} {panel['title']}")
        axis.set_xlabel(axis_names[0])
        axis.set_ylabel(axis_names[1])
        axis.grid(alpha=0.14)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=colors[str(class_name)], label=class_name, markersize=7)
        for class_name in classes
    ]
    fig.legend(handles=handles, labels=classes, loc="upper center", ncol=max(2, min(4, len(classes))), frameon=False)
    fig.suptitle(title)
    return fig


def plot_embedding_comparison_grid(row_specs, title):
    plt = _require_matplotlib_pyplot()
    if not row_specs:
        raise ValueError("At least one row is required.")

    row_count = len(row_specs)
    column_count = max(len(row_spec.get("panels", [])) for row_spec in row_specs)
    if column_count == 0:
        raise ValueError("Each row must include at least one panel.")

    all_labels = []
    for row_spec in row_specs:
        for panel in row_spec.get("panels", []):
            all_labels.extend([str(value) for value in panel["labels"]])

    classes = ordered_label_classes(all_labels)
    colors = label_color_mapping(classes)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(6.3 * column_count, 4.8 * row_count),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    prefixes = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    panel_index = 0

    for row_index, row_spec in enumerate(row_specs):
        row_panels = row_spec.get("panels", [])
        axis_names = row_spec.get("axis_names", ("x", "y"))
        row_title = row_spec.get("row_title")

        for column_index in range(column_count):
            axis = axes[row_index, column_index]
            if column_index >= len(row_panels):
                axis.axis("off")
                continue

            panel = row_panels[column_index]
            embedding = panel["embedding"]
            label_series = pd.Series(panel["labels"], dtype=str)

            for class_name in classes:
                mask = label_series == class_name
                if not mask.any():
                    continue
                axis.scatter(
                    embedding[mask.to_numpy(), 0],
                    embedding[mask.to_numpy(), 1],
                    s=28,
                    alpha=0.84,
                    color=colors[str(class_name)],
                    edgecolors="white",
                    linewidths=0.45,
                )

            prefix = prefixes[panel_index] if panel_index < len(prefixes) else f"({panel_index + 1})"
            panel_title = f"{prefix} {panel['title']}"
            if row_title:
                panel_title = f"{panel_title}\n{row_title}"
            axis.set_title(panel_title)
            axis.set_xlabel(axis_names[0])
            axis.set_ylabel(axis_names[1])
            axis.grid(alpha=0.14)
            panel_index += 1

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=colors[str(class_name)], label=class_name, markersize=7)
        for class_name in classes
    ]
    fig.legend(handles=handles, labels=classes, loc="upper center", ncol=max(2, min(4, len(classes))), frameon=False)
    fig.suptitle(title)
    return fig


def plot_grouped_bars(dataframe, category_col, value_cols, title, y_label="Value"):
    plt = _require_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    plot_df = dataframe[[category_col] + value_cols].copy()
    categories = plot_df[category_col].astype(str).tolist()
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(value_cols))

    for index, column in enumerate(value_cols):
        ax.bar(x + index * width, plot_df[column].to_numpy(), width=width, label=column)

    ax.set_title(title)
    ax.set_xticks(x + width * (len(value_cols) - 1) / 2.0)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel(y_label)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_heatmap(dataframe, title, cmap="viridis"):
    plt = _require_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    image = ax.imshow(dataframe.to_numpy(dtype=float), cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(dataframe.columns)))
    ax.set_xticklabels(dataframe.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(dataframe.index)))
    ax.set_yticklabels([str(index) for index in dataframe.index])

    for row_index in range(dataframe.shape[0]):
        for col_index in range(dataframe.shape[1]):
            ax.text(col_index, row_index, f"{dataframe.iat[row_index, col_index]:.2f}", ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_time_series(dataframe, x_col, y_cols, title, group_col=None):
    plt = _require_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    working = dataframe.copy()
    if group_col and group_col in working.columns:
        for group_name, group_df in working.groupby(group_col):
            for y_col in y_cols:
                ax.plot(group_df[x_col], group_df[y_col], label=f"{group_name} · {y_col}", alpha=0.85)
    else:
        for y_col in y_cols:
            ax.plot(working[x_col], working[y_col], label=y_col, alpha=0.85)

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_histograms(dataframe, numeric_cols, title_prefix="Latency"):
    plt = _require_matplotlib_pyplot()
    count = max(1, len(numeric_cols))
    rows = int(math.ceil(count / 2.0))
    fig, axes = plt.subplots(rows, 2, figsize=(8.0, 3.0 * rows))
    axes = np.atleast_1d(axes).reshape(rows, 2)

    for index, column in enumerate(numeric_cols):
        row = index // 2
        col = index % 2
        axis = axes[row, col]
        series = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        axis.hist(series, bins=24, color="#8fc9b0", edgecolor="#15181a")
        axis.set_title(f"{title_prefix}: {column}")
        axis.set_xlabel(column)
        axis.set_ylabel("Count")

    for index in range(count, rows * 2):
        axes[index // 2, index % 2].axis("off")

    fig.tight_layout()
    return fig
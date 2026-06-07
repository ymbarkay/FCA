from __future__ import annotations

from pathlib import Path

from fca_tools.common import (
    PLOT_OUTPUT_ROOT,
    REPORT_OUTPUT_ROOT,
    TABLE_OUTPUT_ROOT,
    ensure_analysis_dirs,
    sanitize_slug,
)


def save_dataframe_artifact(dataframe, folder, filename):
    ensure_analysis_dirs()
    target_dir = _resolve_folder(folder)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    dataframe.to_csv(target_path, index=False)
    return target_path


def save_text_artifact(text, folder, filename):
    ensure_analysis_dirs()
    target_dir = _resolve_folder(folder)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    target_path.write_text(str(text), encoding="utf-8")
    return target_path


def save_figure_artifact(figure, folder, filename):
    ensure_analysis_dirs()
    target_dir = _resolve_folder(folder)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    figure.savefig(target_path, dpi=180, bbox_inches="tight")
    return target_path


def dataframe_to_latex(dataframe, caption="", label=""):
    latex_caption = caption or "FCA analysis summary"
    latex_label = label or f"tab:{sanitize_slug(latex_caption)}"
    return dataframe.to_latex(index=False, escape=True, caption=latex_caption, label=latex_label)


def build_feature_probe_markdown(summary_df, diagnostics_by_extractor):
    lines = [
        "# Feature Probe Report",
        "",
        "## Probe summary",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Extractor diagnostics",
        "",
    ]

    for extractor_name, diagnostics in diagnostics_by_extractor.items():
        lines.extend([
            f"### {extractor_name}",
            f"- Representation sufficiency: {diagnostics.get('representation_score', 0.0):.3f}",
            f"- Right-turn neighbour purity: {diagnostics.get('right_neighbour_purity', 0.0):.3f}",
            f"- Mean centroid separation: {diagnostics.get('mean_centroid_separation', 0.0):.3f}",
            "",
        ])

    return "\n".join(lines)


def build_expert_usage_markdown(summary_df, intent_df, entropy_summary):
    lines = [
        "# Expert Usage Report",
        "",
        "## Summary",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Usage by intent",
        "",
        intent_df.to_markdown(index=False) if not intent_df.empty else "No intent columns were available.",
        "",
        "## Gate entropy",
        "",
        entropy_summary.to_markdown(index=False) if not entropy_summary.empty else "No entropy data available.",
    ]
    return "\n".join(lines)


def _resolve_folder(folder):
    key = str(folder).lower().strip()
    if key == "plots":
        return PLOT_OUTPUT_ROOT
    if key == "tables":
        return TABLE_OUTPUT_ROOT
    return REPORT_OUTPUT_ROOT
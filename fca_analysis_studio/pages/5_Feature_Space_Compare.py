from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from fca_tools.common import (
    REPO_ROOT,
    balance_probe_dataframe,
    dataframe_to_csv_bytes,
    ensure_analysis_dirs,
    figure_to_png_bytes,
    list_candidate_csv_files,
    load_csv_source,
    prepare_probe_dataframe,
    render_image_preview_grid,
    sanitize_slug,
)
from fca_tools.exports import save_dataframe_artifact, save_figure_artifact
from fca_tools.feature_extractors import extract_feature_matrix, load_registry
from fca_tools.plots import plot_embedding_comparison, plot_embedding_comparison_grid
from fca_tools.probe_models import compute_feature_diagnostics, run_probe_suite


def _dataset_controls(prefix, title, csv_choices):
    st.subheader(title)
    selected_csv = st.selectbox(f"{title} workspace CSV", csv_choices, key=f"{prefix}_csv")
    uploaded_csv = st.file_uploader(f"{title} uploaded CSV", type=["csv"], key=f"{prefix}_upload")
    dataset_df, dataset_source, dataset_base_dir = load_csv_source(uploaded_csv, selected_csv)
    if dataset_df is None:
        st.info("Load a CSV to configure this comparison panel.")
        return None

    default_image_col = "frame_path" if "frame_path" in dataset_df.columns else dataset_df.columns[0]
    default_label_col = "label" if "label" in dataset_df.columns else dataset_df.columns[min(1, len(dataset_df.columns) - 1)]

    image_col = st.selectbox(
        f"{title} image path column",
        dataset_df.columns,
        index=list(dataset_df.columns).index(default_image_col),
        key=f"{prefix}_image_col",
    )
    label_col = st.selectbox(
        f"{title} label column",
        dataset_df.columns,
        index=list(dataset_df.columns).index(default_label_col),
        key=f"{prefix}_label_col",
    )
    panel_title = st.text_input(
        f"{title} panel label",
        value=Path(str(dataset_source)).stem if dataset_source else title,
        key=f"{prefix}_panel_title",
    )
    max_per_class = st.number_input(
        f"{title} max samples per class (0 keeps all)",
        min_value=0,
        value=0,
        step=1,
        key=f"{prefix}_max_per_class",
    )
    preview_count = st.slider(
        f"{title} preview thumbnails",
        min_value=0,
        max_value=8,
        value=4,
        step=2,
        key=f"{prefix}_preview_count",
    )

    working_df = prepare_probe_dataframe(dataset_df, label_col)
    active_df = working_df[working_df["include"]].copy()
    if int(max_per_class) > 0:
        active_df = balance_probe_dataframe(active_df, label_col="probe_label", max_per_class=int(max_per_class))

    counts = active_df["probe_label"].value_counts().rename_axis("label").reset_index(name="count")
    st.dataframe(counts, use_container_width=True, hide_index=True)
    if preview_count > 0:
        render_image_preview_grid(st, active_df, image_col=image_col, base_dir=dataset_base_dir, max_items=preview_count)

    return {
        "title": panel_title,
        "dataframe": active_df,
        "source": dataset_source,
        "base_dir": dataset_base_dir,
        "image_col": image_col,
        "label_col": "probe_label",
    }


ensure_analysis_dirs()
st.title("Feature Space Compare")
st.caption("Generate paper-ready PCA/UMAP comparisons for two curated probe datasets with consistent semantic colors.")

csv_choices = [""] + [str(path.relative_to(REPO_ROOT)) for path in list_candidate_csv_files()]

left_column, right_column = st.columns(2)
with left_column:
    condition_a = _dataset_controls("compare_a", "Condition A", csv_choices)
with right_column:
    condition_b = _dataset_controls("compare_b", "Condition B", csv_choices)

st.divider()
registry = load_registry()
extractor_name = st.selectbox("Extractor", [spec.name for spec in registry], index=0 if registry else None)
include_mlp = st.checkbox("Run probes for condition summaries", value=True)
test_size = st.slider("Held-out test fraction", min_value=0.1, max_value=0.5, value=0.25, step=0.05)

if st.button("Generate feature-space comparison", type="primary"):
    if condition_a is None or condition_b is None:
        st.error("Load both comparison datasets first.")
    elif condition_a["dataframe"].empty or condition_b["dataframe"].empty:
        st.error("One of the comparison datasets is empty after filtering.")
    else:
        spec = next(spec for spec in registry if spec.name == extractor_name)
        panels = []
        summary_rows = []
        all_conditions = [condition_a, condition_b]
        diagnostics_by_title = {}

        for condition in all_conditions:
            extraction = extract_feature_matrix(
                condition["dataframe"],
                image_col=condition["image_col"],
                label_col=condition["label_col"],
                spec=spec,
                base_dir=condition["base_dir"],
            )
            probe_results = run_probe_suite(
                extraction["X"],
                extraction["y"],
                include_mlp=include_mlp,
                test_size=test_size,
            )
            diagnostics = compute_feature_diagnostics(extraction["X"], extraction["y"], probe_results)
            diagnostics_by_title[condition["title"]] = diagnostics

            linear_metrics = probe_results["models"].get("linear_probe", {})
            mlp_metrics = probe_results["models"].get("mlp_probe", {})
            summary_rows.append({
                "condition": condition["title"],
                "samples": extraction["summary"]["samples"],
                "classes": len(probe_results["class_names"]),
                "linear_balanced_accuracy": linear_metrics.get("balanced_accuracy", 0.0),
                "linear_macro_f1": linear_metrics.get("macro_f1", 0.0),
                "mlp_balanced_accuracy": mlp_metrics.get("balanced_accuracy", 0.0),
                "representation_score": diagnostics.get("representation_score", 0.0),
                "right_neighbour_purity": diagnostics.get("right_neighbour_purity", 0.0),
                "mean_latency_ms": extraction["summary"]["mean_latency_ms"],
            })
            panels.append({
                "title": condition["title"],
                "pca": diagnostics["pca_embedding"],
                "umap": diagnostics.get("umap_embedding"),
                "labels": extraction["y"],
            })

        pca_fig = plot_embedding_comparison(
            [{"title": panel["title"], "embedding": panel["pca"], "labels": panel["labels"]} for panel in panels],
            title=f"{extractor_name} · PCA feature-space comparison",
            axis_names=("PC1", "PC2"),
        )
        umap_fig = None
        paper_fig = None
        if all(panel["umap"] is not None for panel in panels):
            umap_fig = plot_embedding_comparison(
                [{"title": panel["title"], "embedding": panel["umap"], "labels": panel["labels"]} for panel in panels],
                title=f"{extractor_name} · UMAP feature-space comparison",
                axis_names=("UMAP-1", "UMAP-2"),
            )
            paper_fig = plot_embedding_comparison_grid(
                [
                    {
                        "row_title": "PCA",
                        "axis_names": ("PC1", "PC2"),
                        "panels": [{"title": panel["title"], "embedding": panel["pca"], "labels": panel["labels"]} for panel in panels],
                    },
                    {
                        "row_title": "UMAP",
                        "axis_names": ("UMAP-1", "UMAP-2"),
                        "panels": [{"title": panel["title"], "embedding": panel["umap"], "labels": panel["labels"]} for panel in panels],
                    },
                ],
                title=f"{extractor_name} · Feature-space paper figure",
            )

        comparison_slug = sanitize_slug(f"{extractor_name}_{condition_a['title']}_{condition_b['title']}")
        summary_df = pd.DataFrame(summary_rows)
        summary_path = save_dataframe_artifact(summary_df, "reports", f"feature_space_compare_{comparison_slug}.csv")
        pca_path = save_figure_artifact(pca_fig, "plots", f"feature_space_compare_pca_{comparison_slug}.png")
        umap_path = None
        paper_path = None
        if umap_fig is not None:
            umap_path = save_figure_artifact(umap_fig, "plots", f"feature_space_compare_umap_{comparison_slug}.png")
        if paper_fig is not None:
            paper_path = save_figure_artifact(paper_fig, "plots", f"feature_space_compare_paper_{comparison_slug}.png")

        st.session_state["feature_space_compare_summary"] = summary_df
        st.session_state["feature_space_compare_pca_fig"] = pca_fig
        st.session_state["feature_space_compare_umap_fig"] = umap_fig
        st.session_state["feature_space_compare_paper_fig"] = paper_fig
        st.session_state["feature_space_compare_paths"] = {
            "summary_csv": summary_path,
            "pca_png": pca_path,
            "umap_png": umap_path,
            "paper_png": paper_path,
        }

        st.success("Feature-space comparison generated.")

summary_df = st.session_state.get("feature_space_compare_summary")
if summary_df is not None:
    st.subheader("Condition summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    pca_fig = st.session_state.get("feature_space_compare_pca_fig")
    if pca_fig is not None:
        st.pyplot(pca_fig, use_container_width=True)
        st.download_button(
            "Download PCA comparison PNG",
            figure_to_png_bytes(pca_fig),
            file_name="feature_space_compare_pca.png",
            mime="image/png",
        )

    umap_fig = st.session_state.get("feature_space_compare_umap_fig")
    if umap_fig is not None:
        st.pyplot(umap_fig, use_container_width=True)
        st.download_button(
            "Download UMAP comparison PNG",
            figure_to_png_bytes(umap_fig),
            file_name="feature_space_compare_umap.png",
            mime="image/png",
        )

    paper_fig = st.session_state.get("feature_space_compare_paper_fig")
    if paper_fig is not None:
        st.subheader("Paper-ready combined figure")
        st.pyplot(paper_fig, use_container_width=True)
        st.download_button(
            "Download paper-ready 2x2 PNG",
            figure_to_png_bytes(paper_fig),
            file_name="feature_space_compare_paper.png",
            mime="image/png",
        )
    elif pca_fig is not None:
        st.info("The combined 2x2 paper figure is available when UMAP embeddings are enabled and generated successfully.")

    st.download_button(
        "Download feature-space comparison summary CSV",
        dataframe_to_csv_bytes(summary_df),
        file_name="feature_space_compare_summary.csv",
        mime="text/csv",
    )
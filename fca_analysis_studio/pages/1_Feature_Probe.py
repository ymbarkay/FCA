from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from fca_tools.common import (
    FEATURE_OUTPUT_ROOT,
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
    text_to_bytes,
)
from fca_tools.exports import build_feature_probe_markdown, dataframe_to_latex, save_dataframe_artifact, save_figure_artifact, save_text_artifact
from fca_tools.feature_extractors import benchmark_extractor, extract_feature_matrix, load_registry, upsert_extractor
from fca_tools.plots import plot_confusion_matrix, plot_embedding_scatter
from fca_tools.probe_models import compute_feature_diagnostics, run_probe_suite


ensure_analysis_dirs()
st.title("Feature Probe Studio")
st.caption("Compare frozen extractors, probe representation sufficiency, and export paper-ready artifacts.")

csv_choices = [""] + [str(path.relative_to(REPO_ROOT)) for path in list_candidate_csv_files()]
selected_csv = st.selectbox("Workspace CSV", csv_choices, help="Choose an existing CSV from the repo, or upload one below.")
uploaded_csv = st.file_uploader("Upload probe CSV", type=["csv"])
dataset_df, dataset_source, dataset_base_dir = load_csv_source(uploaded_csv, selected_csv)

tabs = st.tabs(["Dataset", "Extractors", "Run Probes", "Visualize", "Export"])

if dataset_df is not None:
    default_image_col = "frame_path" if "frame_path" in dataset_df.columns else dataset_df.columns[0]
    default_label_col = "label" if "label" in dataset_df.columns else dataset_df.columns[min(1, len(dataset_df.columns) - 1)]
else:
    default_image_col = ""
    default_label_col = ""

with tabs[0]:
    st.subheader("Probe dataset builder")
    if dataset_df is None:
        st.info("Load a CSV to start building a probe dataset.")
    else:
        st.caption(f"Source: {dataset_source}")
        image_col = st.selectbox("Image path column", dataset_df.columns, index=list(dataset_df.columns).index(default_image_col))
        label_col = st.selectbox("Label column", dataset_df.columns, index=list(dataset_df.columns).index(default_label_col))
        preview_limit = st.slider("Preview thumbnails", min_value=4, max_value=16, value=8, step=4)

        working_df = prepare_probe_dataframe(dataset_df, label_col)
        editor_columns = [column for column in ["include", "probe_label", image_col, label_col, "session", "mode"] if column in working_df.columns]
        edited_df = st.data_editor(
            working_df[editor_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                "include": st.column_config.CheckboxColumn("include"),
                "probe_label": st.column_config.TextColumn("probe_label"),
            },
        )

        run_df = working_df.copy()
        run_df["include"] = edited_df["include"].astype(bool).to_numpy()
        run_df["probe_label"] = edited_df["probe_label"].astype(str).to_numpy()

        st.session_state["feature_probe_df"] = run_df
        st.session_state["feature_probe_image_col"] = image_col
        st.session_state["feature_probe_label_col"] = "probe_label"
        st.session_state["feature_probe_base_dir"] = dataset_base_dir

        active_df = run_df[run_df["include"]].copy()
        class_counts = active_df["probe_label"].value_counts().rename_axis("label").reset_index(name="count")
        st.write("Class counts")
        st.dataframe(class_counts, use_container_width=True, hide_index=True)

        if not class_counts.empty and class_counts["count"].min() < 5:
            st.warning("At least one class has fewer than 5 samples. Probe quality will be noisy.")

        balanced_target = st.number_input(
            "Balanced sample count per class",
            min_value=1,
            value=int(class_counts["count"].min()) if not class_counts.empty else 1,
            step=1,
        )
        balanced_df = balance_probe_dataframe(run_df, label_col="probe_label", max_per_class=balanced_target)
        st.write("Balanced subset preview")
        st.dataframe(balanced_df.head(20), use_container_width=True, hide_index=True)

        st.download_button(
            "Download edited probe dataset",
            dataframe_to_csv_bytes(run_df),
            file_name="probe_dataset_edited.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download balanced subset",
            dataframe_to_csv_bytes(balanced_df),
            file_name="probe_dataset_balanced.csv",
            mime="text/csv",
        )

        st.write("Thumbnail verification")
        render_image_preview_grid(st, active_df, image_col=image_col, base_dir=dataset_base_dir, max_items=preview_limit)

with tabs[1]:
    st.subheader("Extractor registry")
    registry = load_registry()
    registry_df = pd.DataFrame([spec.to_dict() for spec in registry])
    st.dataframe(registry_df, use_container_width=True, hide_index=True)

    with st.form("extractor_registry_form"):
        left, right = st.columns(2)
        name = left.text_input("name", value="mobilenetv2_dense512_current")
        model_path = right.text_input("model_path", value="tflite_models/feature_extractor_dense512_int8_edgetpu.tflite")
        architecture = left.text_input("architecture", value="MobileNetV2")
        runtime = right.selectbox("runtime", ["EdgeTPU", "CPU"], index=0)
        feature_dim = left.number_input("feature_dim", min_value=1, value=512)
        quantization = right.selectbox("quantization", ["int8", "float32"], index=0)
        preprocessing = left.text_input("preprocessing", value="crop_top_80_resize_224_scale_minus1_1_quantized")
        training_data = right.text_input("training_data", value="original car dataset, left-biased")
        date_trained = left.text_input("date_trained", value="")
        notes = right.text_area("notes", value="")
        save_extractor = st.form_submit_button("Add / update extractor")

        if save_extractor:
            upsert_extractor({
                "name": name,
                "model_path": model_path,
                "architecture": architecture,
                "input_size": [224, 224, 3],
                "feature_dim": int(feature_dim),
                "quantization": quantization,
                "runtime": runtime,
                "preprocessing": preprocessing,
                "date_trained": date_trained,
                "training_data": training_data,
                "notes": notes,
            })
            st.success(f"Extractor registry updated: {name}")

    st.divider()
    st.subheader("Single-image latency benchmark")
    benchmark_name = st.selectbox("Extractor", [spec.name for spec in registry]) if registry else st.selectbox("Extractor", [""], disabled=True)
    benchmark_image = st.text_input("Image path", value=st.session_state.get("feature_probe_df", pd.DataFrame()).get(st.session_state.get("feature_probe_image_col", ""), pd.Series(dtype=str)).head(1).iloc[0] if st.session_state.get("feature_probe_df") is not None and not st.session_state.get("feature_probe_df").empty else "")
    benchmark_repeats = st.number_input("Repeats", min_value=1, value=20)
    if st.button("Benchmark extractor") and registry and benchmark_image:
        spec = next(spec for spec in registry if spec.name == benchmark_name)
        try:
            benchmark = benchmark_extractor(spec, benchmark_image, repeats=benchmark_repeats, base_dir=st.session_state.get("feature_probe_base_dir"))
            st.json(benchmark)
        except Exception as exc:
            st.error(str(exc))

with tabs[2]:
    st.subheader("Run linear, kNN, and optional MLP probes")
    run_df = st.session_state.get("feature_probe_df")
    if run_df is None or run_df.empty:
        st.info("Prepare a probe dataset first.")
    else:
        active_df = run_df[run_df["include"]].copy()
        registry = load_registry()
        chosen_extractors = st.multiselect(
            "Extractors to evaluate",
            [spec.name for spec in registry],
            default=[spec.name for spec in registry[:1]],
        )
        include_mlp = st.checkbox("Run small MLP probe", value=True)
        test_size = st.slider("Held-out test fraction", min_value=0.1, max_value=0.5, value=0.25, step=0.05)

        if st.button("Run feature probes", type="primary"):
            if active_df.empty:
                st.error("The filtered dataset is empty.")
            elif not chosen_extractors:
                st.error("Choose at least one extractor.")
            else:
                summary_rows = []
                diagnostics_by_extractor = {}
                detailed_results = {}
                progress = st.progress(0.0)

                for index, extractor_name in enumerate(chosen_extractors, start=1):
                    spec = next(spec for spec in registry if spec.name == extractor_name)
                    extraction = extract_feature_matrix(
                        active_df,
                        image_col=st.session_state["feature_probe_image_col"],
                        label_col=st.session_state["feature_probe_label_col"],
                        spec=spec,
                        base_dir=st.session_state.get("feature_probe_base_dir"),
                    )
                    probe_results = run_probe_suite(
                        extraction["X"],
                        extraction["y"],
                        include_mlp=include_mlp,
                        test_size=test_size,
                    )
                    diagnostics = compute_feature_diagnostics(extraction["X"], extraction["y"], probe_results)

                    diagnostics_by_extractor[extractor_name] = diagnostics
                    detailed_results[extractor_name] = {
                        "summary": extraction["summary"],
                        "probe_results": probe_results,
                        "diagnostics": diagnostics,
                    }

                    for model_name, metrics in probe_results["models"].items():
                        summary_rows.append({
                            "extractor": extractor_name,
                            "probe": model_name,
                            "accuracy": metrics["accuracy"],
                            "macro_f1": metrics["macro_f1"],
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "right_f1": metrics["right_f1"],
                            "left_right_confusion": metrics["left_right_confusion"],
                            "mean_latency_ms": extraction["summary"]["mean_latency_ms"],
                            "representation_score": diagnostics["representation_score"],
                        })

                    linear_metrics = probe_results["models"].get("linear_probe")
                    if linear_metrics is not None:
                        confusion_fig = plot_confusion_matrix(
                            linear_metrics["confusion_matrix"],
                            probe_results["class_names"],
                            f"{extractor_name} · linear probe confusion",
                        )
                        pca_fig = plot_embedding_scatter(
                            diagnostics["pca_embedding"],
                            extraction["y"],
                            f"{extractor_name} · PCA",
                            axis_names=("PC1", "PC2"),
                        )
                        save_figure_artifact(confusion_fig, "plots", f"confusion_{sanitize_slug(extractor_name)}.png")
                        save_figure_artifact(pca_fig, "plots", f"pca_{sanitize_slug(extractor_name)}.png")

                    progress.progress(index / max(len(chosen_extractors), 1))

                summary_df = pd.DataFrame(summary_rows).sort_values(["extractor", "probe"]).reset_index(drop=True)
                summary_csv_path = save_dataframe_artifact(summary_df, "reports", "feature_probe_comparison.csv")
                summary_tex = dataframe_to_latex(summary_df, caption="Feature probe comparison", label="tab:feature_probe_comparison")
                summary_tex_path = save_text_artifact(summary_tex, "tables", "feature_probe_summary.tex")
                report_md = build_feature_probe_markdown(summary_df, diagnostics_by_extractor)
                report_md_path = save_text_artifact(report_md, "reports", "probe_report.md")

                st.session_state["feature_probe_results"] = detailed_results
                st.session_state["feature_probe_summary"] = summary_df
                st.session_state["feature_probe_artifacts"] = {
                    "summary_csv": summary_csv_path,
                    "summary_tex": summary_tex_path,
                    "report_md": report_md_path,
                }

                st.success("Feature probes complete.")
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Visual diagnostics")
    results = st.session_state.get("feature_probe_results")
    if not results:
        st.info("Run at least one probe first.")
    else:
        extractor_name = st.selectbox("Extractor", list(results.keys()))
        extractor_result = results[extractor_name]
        probe_name = st.selectbox("Probe", list(extractor_result["probe_results"]["models"].keys()))
        probe_metrics = extractor_result["probe_results"]["models"][probe_name]
        class_names = extractor_result["probe_results"]["class_names"]
        diagnostics = extractor_result["diagnostics"]

        left, right = st.columns(2)
        confusion_fig = plot_confusion_matrix(probe_metrics["confusion_matrix"], class_names, f"{extractor_name} · {probe_name}")
        left.pyplot(confusion_fig, use_container_width=True)

        pca_fig = plot_embedding_scatter(diagnostics["pca_embedding"], extractor_result["probe_results"]["models"][probe_name]["y_test"].astype(str) if hasattr(extractor_result["probe_results"]["models"][probe_name]["y_test"], "astype") else extractor_result["probe_results"]["class_names"], f"{extractor_name} · PCA")
        # Use the full dataset labels for the main embedding.
        pca_fig = plot_embedding_scatter(diagnostics["pca_embedding"], st.session_state["feature_probe_df"][st.session_state["feature_probe_df"]["include"]]["probe_label"], f"{extractor_name} · PCA", axis_names=("PC1", "PC2"))
        right.pyplot(pca_fig, use_container_width=True)

        if diagnostics.get("umap_embedding") is not None:
            umap_fig = plot_embedding_scatter(
                diagnostics["umap_embedding"],
                st.session_state["feature_probe_df"][st.session_state["feature_probe_df"]["include"]]["probe_label"],
                f"{extractor_name} · UMAP",
                axis_names=("UMAP-1", "UMAP-2"),
            )
            st.pyplot(umap_fig, use_container_width=True)

        stats_left, stats_right = st.columns(2)
        stats_left.metric("Representation sufficiency", f"{diagnostics['representation_score']:.3f}")
        stats_left.metric("Right-turn neighbour purity", f"{diagnostics['right_neighbour_purity']:.3f}")
        stats_right.metric("Macro F1", f"{probe_metrics['macro_f1']:.3f}")
        stats_right.metric("Right-turn F1", f"{probe_metrics['right_f1']:.3f}")

        st.write("Centroid distances")
        st.dataframe(diagnostics["centroid_distances"], use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Export artifacts")
    summary_df = st.session_state.get("feature_probe_summary")
    artifact_paths = st.session_state.get("feature_probe_artifacts", {})
    if summary_df is None:
        st.info("Run a probe suite to generate exports.")
    else:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download feature probe comparison CSV",
            dataframe_to_csv_bytes(summary_df),
            file_name="feature_probe_comparison.csv",
            mime="text/csv",
        )
        latex_text = Path(artifact_paths["summary_tex"]).read_text(encoding="utf-8") if artifact_paths else ""
        st.download_button(
            "Download LaTeX table",
            text_to_bytes(latex_text),
            file_name="feature_probe_summary.tex",
            mime="text/plain",
        )
        report_text = Path(artifact_paths["report_md"]).read_text(encoding="utf-8") if artifact_paths else ""
        st.download_button(
            "Download Markdown report",
            text_to_bytes(report_text),
            file_name="probe_report.md",
            mime="text/markdown",
        )
        if artifact_paths:
            st.caption("Saved to disk")
            st.code("\n".join(f"{name}: {path}" for name, path in artifact_paths.items()))
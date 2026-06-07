from __future__ import annotations

from pathlib import Path

import streamlit as st

from fca_tools.common import REPO_ROOT, dataframe_to_csv_bytes, ensure_analysis_dirs, list_candidate_csv_files, load_csv_source, text_to_bytes
from fca_tools.exports import dataframe_to_latex, save_dataframe_artifact, save_figure_artifact, save_text_artifact
from fca_tools.metrics import compute_latency_summary
from fca_tools.plots import plot_histograms


ensure_analysis_dirs()
st.title("Latency Analysis")
st.caption("Summarize feature, adapter, inference, loop, and FPS measurements into exportable tables and plots.")

csv_choices = [""] + [str(path.relative_to(REPO_ROOT)) for path in list_candidate_csv_files()]
selected_csv = st.selectbox("Workspace CSV", csv_choices)
uploaded_csv = st.file_uploader("Upload latency CSV", type=["csv"])
latency_df, source, _base_dir = load_csv_source(uploaded_csv, selected_csv)

if latency_df is None:
    st.info("Load a frame log CSV to begin latency analysis.")
    st.stop()

st.caption(f"Source: {source}")

candidate_numeric = [column for column in latency_df.columns if any(token in column.lower() for token in ("ms", "fps", "latency"))]
numeric_columns = st.multiselect("Latency columns", latency_df.columns, default=candidate_numeric)
group_col = st.selectbox("Group by (optional)", [""] + list(latency_df.columns), index=(1 + list(latency_df.columns).index("session")) if "session" in latency_df.columns else 0)

summary_df = compute_latency_summary(latency_df, numeric_columns, group_col or None)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

if numeric_columns:
    hist_fig = plot_histograms(latency_df, numeric_columns, title_prefix="Latency")
    st.pyplot(hist_fig, use_container_width=True)

    summary_path = save_dataframe_artifact(summary_df, "reports", "latency_summary.csv")
    table_text = dataframe_to_latex(summary_df, caption="Latency summary", label="tab:latency_summary")
    table_path = save_text_artifact(table_text, "tables", "latency_table.tex")
    figure_path = save_figure_artifact(hist_fig, "plots", "latency_histograms.png")

    st.download_button("Download latency summary CSV", dataframe_to_csv_bytes(summary_df), file_name="latency_summary.csv", mime="text/csv")
    st.download_button("Download latency LaTeX table", text_to_bytes(table_text), file_name="latency_table.tex", mime="text/plain")
    st.caption("Saved to disk")
    st.code(f"summary_csv: {summary_path}\nsummary_tex: {table_path}\nplot_png: {figure_path}")
else:
    st.warning("Choose at least one numeric latency column.")
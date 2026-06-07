from __future__ import annotations

from pathlib import Path

import streamlit as st

from fca_tools.common import REPO_ROOT, dataframe_to_csv_bytes, ensure_analysis_dirs, list_candidate_csv_files, load_csv_source, text_to_bytes
from fca_tools.exports import dataframe_to_latex, save_dataframe_artifact, save_figure_artifact, save_text_artifact
from fca_tools.metrics import compute_transfer_metrics
from fca_tools.plots import plot_time_series


ensure_analysis_dirs()
st.title("Transfer Metrics")
st.caption("Turn config/task/stage score tables into forgetting and transfer summaries.")

csv_choices = [""] + [str(path.relative_to(REPO_ROOT)) for path in list_candidate_csv_files()]
selected_csv = st.selectbox("Workspace CSV", csv_choices)
uploaded_csv = st.file_uploader("Upload transfer score CSV", type=["csv"])
scores_df, source, _base_dir = load_csv_source(uploaded_csv, selected_csv)

if scores_df is None:
    st.info("Load a score CSV with at least config, task, stage, and score columns.")
    st.stop()

st.caption(f"Source: {source}")

config_col = st.selectbox("Config column", scores_df.columns, index=list(scores_df.columns).index("config") if "config" in scores_df.columns else 0)
task_col = st.selectbox("Task column", scores_df.columns, index=list(scores_df.columns).index("task") if "task" in scores_df.columns else min(1, len(scores_df.columns) - 1))
stage_col = st.selectbox("Stage column", scores_df.columns, index=list(scores_df.columns).index("stage") if "stage" in scores_df.columns else min(2, len(scores_df.columns) - 1))
score_col = st.selectbox("Score column", scores_df.columns, index=list(scores_df.columns).index("score") if "score" in scores_df.columns else min(3, len(scores_df.columns) - 1))
stage_order = st.text_input("Stage order (comma-separated, optional)", value="a,b,c")

summary_df, grouped_df, task_summary = compute_transfer_metrics(scores_df, config_col, task_col, stage_col, score_col, stage_order_text=stage_order)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

curve_fig = plot_time_series(grouped_df, "stage_rank", ["score"], title="Mean score by stage", group_col="config")
st.pyplot(curve_fig, use_container_width=True)
st.dataframe(task_summary, use_container_width=True, hide_index=True)

summary_path = save_dataframe_artifact(summary_df, "reports", "transfer_summary.csv")
table_text = dataframe_to_latex(summary_df, caption="Transfer summary", label="tab:transfer_summary")
table_path = save_text_artifact(table_text, "tables", "transfer_table.tex")
figure_path = save_figure_artifact(curve_fig, "plots", "forgetting_curves.png")

st.download_button("Download transfer summary CSV", dataframe_to_csv_bytes(summary_df), file_name="transfer_summary.csv", mime="text/csv")
st.download_button("Download transfer LaTeX table", text_to_bytes(table_text), file_name="transfer_table.tex", mime="text/plain")
st.caption("Saved to disk")
st.code(f"summary_csv: {summary_path}\nsummary_tex: {table_path}\nplot_png: {figure_path}")
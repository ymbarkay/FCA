from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from fca_tools.common import REPO_ROOT, dataframe_to_csv_bytes, ensure_analysis_dirs, list_candidate_csv_files, load_csv_source, text_to_bytes
from fca_tools.exports import build_expert_usage_markdown, dataframe_to_latex, save_dataframe_artifact, save_figure_artifact, save_text_artifact
from fca_tools.metrics import compute_expert_usage_summary, prepare_expert_usage_frame, usage_by_group
from fca_tools.plots import plot_grouped_bars, plot_heatmap, plot_time_series


ensure_analysis_dirs()
st.title("Expert Usage Studio")
st.caption("Inspect mean expert usage, hard routing, gate entropy, and intent specialization.")

csv_choices = [""] + [str(path.relative_to(REPO_ROOT)) for path in list_candidate_csv_files()]
selected_csv = st.selectbox("Workspace CSV", csv_choices)
uploaded_csv = st.file_uploader("Upload expert log CSV", type=["csv"])
log_df, log_source, _base_dir = load_csv_source(uploaded_csv, selected_csv)

tabs = st.tabs(["Load Logs", "Summary", "Expert Utilization", "Intent Routing", "Time Dynamics", "Export"])

with tabs[0]:
    st.subheader("Load and validate log columns")
    if log_df is None:
        st.info("Load a CSV with gate columns to start expert usage analysis.")
    else:
        st.caption(f"Source: {log_source}")
        candidate_gate_columns = [column for column in log_df.columns if column.lower().startswith("gate_e")]
        gate_columns = st.multiselect("Gate columns", log_df.columns, default=candidate_gate_columns)
        config_col = st.selectbox("Config column", [""] + list(log_df.columns), index=(1 + list(log_df.columns).index("config")) if "config" in log_df.columns else 0)
        stage_col = st.selectbox("Stage column", [""] + list(log_df.columns), index=(1 + list(log_df.columns).index("stage")) if "stage" in log_df.columns else 0)
        circuit_col = st.selectbox("Circuit column", [""] + list(log_df.columns), index=(1 + list(log_df.columns).index("circuit")) if "circuit" in log_df.columns else 0)
        intent_col = st.selectbox("Intent column", [""] + list(log_df.columns), index=(1 + list(log_df.columns).index("intent_pred")) if "intent_pred" in log_df.columns else 0)

        if gate_columns:
            prepared = prepare_expert_usage_frame(log_df, gate_columns, config_col or None, stage_col or None, circuit_col or None, intent_col or None)
            st.session_state["expert_usage_df"] = prepared
            st.session_state["expert_usage_gate_columns"] = gate_columns
            st.dataframe(prepared.head(25), use_container_width=True, hide_index=True)
        else:
            st.warning("Choose at least one gate probability column.")

with tabs[1]:
    st.subheader("Dataset summary")
    prepared = st.session_state.get("expert_usage_df")
    if prepared is None:
        st.info("Load logs and select gate columns first.")
    else:
        left, middle, right = st.columns(3)
        left.metric("Frames", len(prepared))
        middle.metric("Configs", prepared["config"].nunique())
        right.metric("Stages", prepared["stage"].nunique())

        summary_df, entropy_df = compute_expert_usage_summary(prepared, st.session_state["expert_usage_gate_columns"])
        st.session_state["expert_usage_summary"] = summary_df
        st.session_state["expert_usage_entropy"] = entropy_df
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Expert utilization")
    prepared = st.session_state.get("expert_usage_df")
    if prepared is None:
        st.info("Load logs first.")
    else:
        summary_df = st.session_state.get("expert_usage_summary")
        if summary_df is None:
            summary_df, entropy_df = compute_expert_usage_summary(prepared, st.session_state["expert_usage_gate_columns"])
            st.session_state["expert_usage_summary"] = summary_df
            st.session_state["expert_usage_entropy"] = entropy_df

        gate_columns = st.session_state["expert_usage_gate_columns"]
        soft_fig = plot_grouped_bars(summary_df, "config", gate_columns, "Mean soft expert usage", y_label="Mean gate probability")
        st.pyplot(soft_fig, use_container_width=True)

        hard_columns = [column for column in summary_df.columns if column.startswith("gate_e")]
        collapse_fig = plot_grouped_bars(summary_df, "config", ["collapse_index", "mean_entropy"], "Collapse index and entropy", y_label="Value")
        st.pyplot(collapse_fig, use_container_width=True)

        st.session_state["expert_usage_figures"] = {
            "soft_usage": soft_fig,
            "collapse": collapse_fig,
        }

with tabs[3]:
    st.subheader("Intent routing")
    prepared = st.session_state.get("expert_usage_df")
    if prepared is None:
        st.info("Load logs first.")
    else:
        intent_df = usage_by_group(prepared, st.session_state["expert_usage_gate_columns"], "intent")
        st.session_state["expert_usage_intent"] = intent_df
        if intent_df.empty:
            st.warning("No intent grouping column was available.")
        else:
            selected_config = st.selectbox("Config", sorted(intent_df["config"].unique()))
            heatmap_df = intent_df[intent_df["config"] == selected_config].set_index("intent")[st.session_state["expert_usage_gate_columns"]]
            heatmap_fig = plot_heatmap(heatmap_df, f"{selected_config} · expert usage by intent")
            st.pyplot(heatmap_fig, use_container_width=True)
            st.session_state.setdefault("expert_usage_figures", {})["intent_heatmap"] = heatmap_fig
            st.dataframe(intent_df, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Time dynamics")
    prepared = st.session_state.get("expert_usage_df")
    if prepared is None:
        st.info("Load logs first.")
    else:
        x_col = st.selectbox("X axis", [column for column in prepared.columns if column in {"timestamp", "elapsed_s", "updates", "frame_id"}] or list(prepared.columns))
        sorted_df = prepared.copy()
        sorted_df[x_col] = pd.to_numeric(sorted_df[x_col], errors="coerce") if x_col in sorted_df.columns else sorted_df.index
        sorted_df = sorted_df.sort_values(x_col)

        entropy_fig = plot_time_series(sorted_df, x_col, ["gate_entropy"], title="Gate entropy over time", group_col="config")
        st.pyplot(entropy_fig, use_container_width=True)
        expert_fig = plot_time_series(sorted_df, x_col, st.session_state["expert_usage_gate_columns"], title="Expert probabilities over time", group_col=None)
        st.pyplot(expert_fig, use_container_width=True)
        st.session_state.setdefault("expert_usage_figures", {})["entropy"] = entropy_fig
        st.session_state.setdefault("expert_usage_figures", {})["expert_lines"] = expert_fig

with tabs[5]:
    st.subheader("Export artifacts")
    prepared = st.session_state.get("expert_usage_df")
    summary_df = st.session_state.get("expert_usage_summary")
    intent_df = st.session_state.get("expert_usage_intent", pd.DataFrame())
    entropy_df = st.session_state.get("expert_usage_entropy", pd.DataFrame())
    figures = st.session_state.get("expert_usage_figures", {})

    if prepared is None or summary_df is None:
        st.info("Run the summary first to generate exports.")
    else:
        summary_path = save_dataframe_artifact(summary_df, "reports", "expert_usage_summary.csv")
        intent_path = save_dataframe_artifact(intent_df, "reports", "expert_usage_by_intent.csv") if not intent_df.empty else None
        table_text = dataframe_to_latex(summary_df, caption="Expert usage summary", label="tab:expert_usage_summary")
        table_path = save_text_artifact(table_text, "tables", "expert_collapse_table.tex")
        report_text = build_expert_usage_markdown(summary_df, intent_df, entropy_df)
        report_path = save_text_artifact(report_text, "reports", "expert_usage_report.md")
        saved_paths = {
            "summary_csv": summary_path,
            "summary_tex": table_path,
            "report_md": report_path,
        }

        if intent_path is not None:
            saved_paths["intent_csv"] = intent_path
        for key, figure in figures.items():
            saved_paths[f"{key}_png"] = save_figure_artifact(figure, "plots", f"{key}.png")

        st.download_button("Download expert usage summary CSV", dataframe_to_csv_bytes(summary_df), file_name="expert_usage_summary.csv", mime="text/csv")
        st.download_button("Download expert collapse LaTeX table", text_to_bytes(table_text), file_name="expert_collapse_table.tex", mime="text/plain")
        st.download_button("Download markdown report", text_to_bytes(report_text), file_name="expert_usage_report.md", mime="text/markdown")
        st.caption("Saved to disk")
        st.code("\n".join(f"{name}: {path}" for name, path in saved_paths.items()))
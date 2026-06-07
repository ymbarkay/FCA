from __future__ import annotations

import streamlit as st


def _load_startup_context():
    from fca_tools.common import REPO_ROOT, ensure_analysis_dirs, list_candidate_csv_files
    from fca_tools.feature_extractors import load_registry

    ensure_analysis_dirs()
    return REPO_ROOT, load_registry(), list_candidate_csv_files()

st.set_page_config(
    page_title="FCA Analysis Studio",
    page_icon=None,
    layout="wide",
)

try:
    REPO_ROOT, registry, csv_candidates = _load_startup_context()
except Exception as exc:
    st.title("FCA Analysis Studio")
    st.error(f"Startup failed: {exc}")
    st.code(
        "Check logs/analysis_studio.log on the Pi and verify the analysis dependencies are installed:\n"
        "pip install -r requirements.txt -r requirements-analysis.txt"
    )
    st.stop()

st.title("FCA Analysis Studio")
st.caption("Standalone export-first analysis workspace for feature probes, feature-space comparison, expert routing, transfer metrics, and latency summaries.")

left, mid_left, mid_right, right = st.columns(4)
left.metric("Registered extractors", len(registry))
mid_left.metric("Workspace CSV candidates", len(csv_candidates))
mid_right.metric("Default app root", "fca_analysis_studio")
right.metric("Repo root", REPO_ROOT.name)

st.markdown(
    """
This standalone research app is intentionally analysis-oriented rather than operator-facing. Each page is built to produce at least one CSV, one PNG, and one LaTeX-ready table so experiment results can move straight into reports or papers.

Use the sidebar pages for:

- Feature Probe: test whether frozen extractors preserve behaviour-critical information.
- Feature Space Compare: generate side-by-side PCA/UMAP figures for matched conditions with consistent semantic colors.
- Expert Usage: inspect gate collapse, entropy, and intent specialization.
- Transfer Metrics: turn task-by-stage score tables into forgetting and transfer summaries.
- Latency: summarize per-stage runtime measurements and export paper tables.
"""
)

with st.expander("Workspace discovery", expanded=True):
    st.write("Registered extractors")
    st.dataframe(
        [spec.to_dict() for spec in registry],
        use_container_width=True,
        hide_index=True,
    )

    st.write("Discovered CSV files")
    if csv_candidates:
        st.dataframe(
            {"csv_path": [str(path.relative_to(REPO_ROOT)) for path in csv_candidates]},
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No CSV files were discovered yet under logs/ or fca_analysis_studio/data/.")
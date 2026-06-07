from __future__ import annotations

import io
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "fca_analysis_studio"
CONFIG_ROOT = APP_ROOT / "configs"
OUTPUT_ROOT = APP_ROOT / "outputs"
FEATURE_OUTPUT_ROOT = OUTPUT_ROOT / "features"
PLOT_OUTPUT_ROOT = OUTPUT_ROOT / "plots"
TABLE_OUTPUT_ROOT = OUTPUT_ROOT / "tables"
REPORT_OUTPUT_ROOT = OUTPUT_ROOT / "reports"


def _import_cv2():
    import cv2

    return cv2


def _import_numpy():
    import numpy as np

    return np


def _import_pandas():
    import pandas as pd

    return pd


def ensure_analysis_dirs():
    for path in (
        CONFIG_ROOT,
        OUTPUT_ROOT,
        FEATURE_OUTPUT_ROOT,
        PLOT_OUTPUT_ROOT,
        TABLE_OUTPUT_ROOT,
        REPORT_OUTPUT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def natural_sort_key(value):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def sanitize_slug(value):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("_") or "artifact"


def resolve_workspace_path(raw_path, base_dir=None):
    raw = str(raw_path or "").strip()
    if not raw:
        raise ValueError("Path is required.")

    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate.resolve()

    search_roots = []
    if base_dir is not None:
        search_roots.append(Path(base_dir))
    search_roots.append(REPO_ROOT)

    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved

    return (search_roots[0] / candidate).resolve()


def list_candidate_csv_files():
    ensure_analysis_dirs()
    candidates = []
    for root in (REPO_ROOT / "logs", APP_ROOT / "data", REPORT_OUTPUT_ROOT):
        if not root.exists():
            continue
        candidates.extend(root.rglob("*.csv"))
    return sorted({path.resolve() for path in candidates}, key=natural_sort_key)


def load_csv_source(uploaded_file=None, workspace_path=""):
    pd = _import_pandas()
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data, f"upload:{uploaded_file.name}", REPO_ROOT

    selected = str(workspace_path or "").strip()
    if not selected:
        return None, "", REPO_ROOT

    source_path = resolve_workspace_path(selected)
    data = pd.read_csv(source_path)
    return data, str(source_path), source_path.parent


def prepare_probe_dataframe(dataframe, label_col):
    prepared = dataframe.copy()
    if "include" not in prepared.columns:
        prepared.insert(0, "include", True)
    if "probe_label" not in prepared.columns:
        prepared.insert(1, "probe_label", prepared[label_col].astype(str))
    prepared["include"] = prepared["include"].astype(bool)
    prepared["probe_label"] = prepared["probe_label"].astype(str)
    return prepared


def balance_probe_dataframe(dataframe, label_col="probe_label", max_per_class=None, random_state=42):
    pd = _import_pandas()
    usable = dataframe[dataframe.get("include", True)].copy()
    if usable.empty:
        return usable

    counts = usable[label_col].value_counts()
    target_count = int(max_per_class or counts.min())
    target_count = max(1, target_count)

    balanced_parts = []
    for label, part in usable.groupby(label_col):
        sample_count = min(target_count, len(part))
        balanced_parts.append(part.sample(n=sample_count, random_state=random_state))

    balanced = pd.concat(balanced_parts, ignore_index=True)
    return balanced.sort_values(by=label_col, key=lambda s: s.map(natural_sort_key)).reset_index(drop=True)


def render_image_preview_grid(streamlit_module, dataframe, image_col, base_dir=None, max_items=12):
    cv2 = _import_cv2()
    if image_col not in dataframe.columns:
        streamlit_module.info("Image preview is unavailable until you choose an image path column.")
        return

    preview = dataframe.head(max_items)
    if preview.empty:
        streamlit_module.info("No rows available for preview.")
        return

    columns = streamlit_module.columns(4)
    for index, (_row_index, row) in enumerate(preview.iterrows()):
        column = columns[index % 4]
        try:
            image_path = resolve_workspace_path(row[image_col], base_dir=base_dir)
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError("OpenCV could not decode this image.")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            column.image(image_rgb, caption=str(row.get("probe_label", row.get(image_col, "image"))), use_container_width=True)
        except Exception as exc:
            column.warning(f"{row.get(image_col, 'image')}\n{exc}")


def dataframe_to_csv_bytes(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")


def text_to_bytes(text):
    return str(text).encode("utf-8")


def figure_to_png_bytes(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    return buffer.getvalue()


def safe_float(value, default=0.0):
    pd = _import_pandas()
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def ordered_stage_values(series, explicit_order=""):
    if explicit_order:
        parsed = [part.strip() for part in explicit_order.split(",") if part.strip()]
        if parsed:
            return parsed

    seen = []
    for value in series.dropna().astype(str):
        if value not in seen:
            seen.append(value)
    return seen


def normalise_gate_columns(dataframe, gate_columns):
    pd = _import_pandas()
    gate_frame = dataframe[gate_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    gate_sums = gate_frame.sum(axis=1).replace(0.0, 1.0)
    return gate_frame.div(gate_sums, axis=0)


def p95(series):
    np = _import_numpy()
    pd = _import_pandas()
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(np.percentile(values, 95))
from __future__ import annotations

import numpy as np
import pandas as pd

from fca_tools.common import normalise_gate_columns, ordered_stage_values, p95, safe_float


def prepare_expert_usage_frame(dataframe, gate_columns, config_col=None, stage_col=None, circuit_col=None, intent_col=None):
    prepared = dataframe.copy()

    if config_col and config_col in prepared.columns:
        prepared["config"] = prepared[config_col].astype(str)
    else:
        prepared["config"] = "unspecified"

    if stage_col and stage_col in prepared.columns:
        prepared["stage"] = prepared[stage_col].astype(str)
    else:
        prepared["stage"] = "unknown"

    if circuit_col and circuit_col in prepared.columns:
        prepared["circuit"] = prepared[circuit_col].astype(str)
    else:
        prepared["circuit"] = "unknown"

    if intent_col and intent_col in prepared.columns:
        prepared["intent"] = prepared[intent_col].astype(str)
    elif "intent_pred" in prepared.columns:
        prepared["intent"] = prepared["intent_pred"].astype(str)
    else:
        prepared["intent"] = "unknown"

    gate_frame = normalise_gate_columns(prepared, gate_columns)
    for column in gate_columns:
        prepared[column] = gate_frame[column]

    prepared["gate_entropy"] = -np.sum(gate_frame.to_numpy() * np.log(np.clip(gate_frame.to_numpy(), 1e-9, 1.0)), axis=1)
    prepared["top_expert"] = gate_frame.idxmax(axis=1)
    return prepared


def compute_expert_usage_summary(dataframe, gate_columns):
    soft_usage = dataframe.groupby("config")[gate_columns].mean().reset_index()
    hard_usage = (
        dataframe.groupby(["config", "top_expert"]).size().rename("fraction").reset_index()
    )
    hard_usage["fraction"] = hard_usage.groupby("config")["fraction"].transform(lambda values: values / max(values.sum(), 1.0))
    hard_pivot = hard_usage.pivot(index="config", columns="top_expert", values="fraction").fillna(0.0).reset_index()
    hard_pivot.columns = [str(column) for column in hard_pivot.columns]

    entropy_summary = dataframe.groupby("config")["gate_entropy"].agg(["mean", "std", "min", "max"]).reset_index()
    entropy_summary = entropy_summary.rename(columns={"mean": "mean_entropy", "std": "std_entropy"})

    summary = soft_usage.merge(entropy_summary, on="config", how="left")
    summary["collapse_index"] = summary[gate_columns].max(axis=1)
    summary["frames"] = dataframe.groupby("config").size().reindex(summary["config"]).to_numpy()
    summary = summary.merge(hard_pivot, on="config", how="left")
    return summary.sort_values("config").reset_index(drop=True), entropy_summary.sort_values("config").reset_index(drop=True)


def usage_by_group(dataframe, gate_columns, group_col):
    if group_col not in dataframe.columns:
        return pd.DataFrame()
    grouped = dataframe.groupby(["config", group_col])[gate_columns].mean().reset_index()
    return grouped.sort_values(["config", group_col]).reset_index(drop=True)


def compute_transfer_metrics(dataframe, config_col, task_col, stage_col, score_col, stage_order_text=""):
    prepared = dataframe.copy()
    prepared["config"] = prepared[config_col].astype(str)
    prepared["task"] = prepared[task_col].astype(str)
    prepared["stage"] = prepared[stage_col].astype(str)
    prepared["score"] = pd.to_numeric(prepared[score_col], errors="coerce")
    prepared = prepared.dropna(subset=["score"])

    stage_order = ordered_stage_values(prepared["stage"], explicit_order=stage_order_text)
    stage_rank = {stage: index for index, stage in enumerate(stage_order)}
    prepared["stage_rank"] = prepared["stage"].map(stage_rank)

    grouped = prepared.groupby(["config", "task", "stage", "stage_rank"], as_index=False)["score"].mean()
    grouped = grouped.sort_values(["config", "task", "stage_rank"]).reset_index(drop=True)

    first_scores = grouped.groupby(["config", "task"], as_index=False).first()
    final_scores = grouped.groupby(["config", "task"], as_index=False).last()
    peak_scores = grouped.groupby(["config", "task"], as_index=False)["score"].max().rename(columns={"score": "peak_score"})

    merged = final_scores.merge(first_scores[["config", "task", "score"]].rename(columns={"score": "first_score"}), on=["config", "task"], how="left")
    merged = merged.merge(peak_scores, on=["config", "task"], how="left")
    merged["forgetting"] = merged["peak_score"] - merged["score"]
    merged["backward_transfer"] = merged["score"] - merged["first_score"]
    merged["plasticity"] = merged["peak_score"] - merged["first_score"]

    summary = merged.groupby("config", as_index=False).agg(
        final_score=("score", "mean"),
        average_forgetting=("forgetting", "mean"),
        backward_transfer=("backward_transfer", "mean"),
        plasticity=("plasticity", "mean"),
        failure_count=("score", lambda values: int(np.sum(np.asarray(values) <= 0.0))),
    )
    summary["forward_transfer"] = np.nan
    return summary.sort_values("config").reset_index(drop=True), grouped, merged


def compute_latency_summary(dataframe, numeric_columns, group_col=None):
    numeric_columns = [column for column in numeric_columns if column in dataframe.columns]
    if not numeric_columns:
        return pd.DataFrame()

    prepared = dataframe.copy()
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    group_keys = [group_col] if group_col and group_col in prepared.columns else None
    grouped = prepared.groupby(group_keys) if group_keys else [("overall", prepared)]

    rows = []
    for group_name, group_df in grouped:
        row = {"group": group_name if not isinstance(group_name, tuple) else " / ".join(group_name)}
        for column in numeric_columns:
            series = group_df[column].dropna()
            row[f"{column}_mean"] = safe_float(series.mean())
            row[f"{column}_std"] = safe_float(series.std())
            row[f"{column}_min"] = safe_float(series.min())
            row[f"{column}_max"] = safe_float(series.max())
            row[f"{column}_p95"] = p95(series)
        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary.sort_values("group").reset_index(drop=True)
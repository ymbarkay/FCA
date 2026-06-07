from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from fca_tools.common import CONFIG_ROOT, FEATURE_OUTPUT_ROOT, ensure_analysis_dirs, resolve_workspace_path, sanitize_slug, safe_float


REGISTRY_PATH = CONFIG_ROOT / "extractors.json"


@dataclass
class ExtractorSpec:
    name: str
    model_path: str
    architecture: str
    input_size: list[int]
    feature_dim: int
    quantization: str
    runtime: str
    preprocessing: str
    date_trained: str
    training_data: str
    notes: str

    @classmethod
    def from_dict(cls, payload):
        normalised = dict(payload)
        normalised.setdefault("input_size", [224, 224, 3])
        normalised.setdefault("feature_dim", 512)
        normalised.setdefault("quantization", "int8")
        normalised.setdefault("runtime", "EdgeTPU")
        normalised.setdefault("preprocessing", "crop_top_80_resize_224_scale_minus1_1_quantized")
        normalised.setdefault("date_trained", "")
        normalised.setdefault("training_data", "")
        normalised.setdefault("notes", "")
        return cls(**normalised)

    def to_dict(self):
        return asdict(self)


DEFAULT_EXTRACTORS = [
    {
        "name": "mobilenetv2_dense512_current",
        "model_path": "tflite_models/feature_extractor_dense512_int8_edgetpu.tflite",
        "architecture": "MobileNetV2",
        "input_size": [224, 224, 3],
        "feature_dim": 512,
        "quantization": "int8",
        "runtime": "EdgeTPU",
        "preprocessing": "crop_top_80_resize_224_scale_minus1_1_quantized",
        "date_trained": "",
        "training_data": "original car dataset, left-biased",
        "notes": "Current FCA extractor.",
    },
    {
        "name": "mobilenetv2_dense512_balanced",
        "model_path": "",
        "architecture": "MobileNetV2",
        "input_size": [224, 224, 3],
        "feature_dim": 512,
        "quantization": "int8",
        "runtime": "EdgeTPU",
        "preprocessing": "crop_top_80_resize_224_scale_minus1_1_quantized",
        "date_trained": "",
        "training_data": "balanced left/right dataset",
        "notes": "Fill in when the balanced extractor is trained.",
    },
    {
        "name": "mobilenetv3small_dense512",
        "model_path": "",
        "architecture": "MobileNetV3Small",
        "input_size": [224, 224, 3],
        "feature_dim": 512,
        "quantization": "float32",
        "runtime": "CPU",
        "preprocessing": "crop_top_80_resize_224_scale_minus1_1_float",
        "date_trained": "",
        "training_data": "future probe candidate",
        "notes": "Placeholder registry entry.",
    },
    {
        "name": "efficientnet_lite0_dense512",
        "model_path": "",
        "architecture": "EfficientNet-Lite0",
        "input_size": [224, 224, 3],
        "feature_dim": 512,
        "quantization": "float32",
        "runtime": "CPU",
        "preprocessing": "crop_top_80_resize_224_scale_minus1_1_float",
        "date_trained": "",
        "training_data": "future probe candidate",
        "notes": "Placeholder registry entry.",
    },
]


def _import_pandas():
    import pandas as pd

    return pd


def _import_feature_runtime():
    import cv2
    import numpy as np
    import pandas as pd

    from fca.perception.feature_extractor import FeatureExtractor

    return cv2, np, pd, FeatureExtractor


def load_registry():
    ensure_analysis_dirs()
    if not REGISTRY_PATH.exists():
        REGISTRY_PATH.write_text(json.dumps(DEFAULT_EXTRACTORS, indent=2), encoding="utf-8")

    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return [ExtractorSpec.from_dict(item) for item in payload]


def save_registry(specs):
    ensure_analysis_dirs()
    REGISTRY_PATH.write_text(
        json.dumps([ExtractorSpec.from_dict(spec).to_dict() if isinstance(spec, dict) else spec.to_dict() for spec in specs], indent=2),
        encoding="utf-8",
    )


def upsert_extractor(spec_payload):
    new_spec = ExtractorSpec.from_dict(spec_payload)
    current = {spec.name: spec for spec in load_registry()}
    current[new_spec.name] = new_spec
    ordered = [current[name] for name in sorted(current.keys())]
    save_registry(ordered)
    return new_spec


def registry_dataframe():
    pd = _import_pandas()
    return pd.DataFrame([spec.to_dict() for spec in load_registry()])


def benchmark_extractor(spec, image_path, repeats=20, base_dir=None):
    cv2, np, _pd, FeatureExtractor = _import_feature_runtime()
    resolved = resolve_workspace_path(image_path, base_dir=base_dir)
    image_bgr = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {resolved}")

    use_tpu = str(spec.runtime).lower() == "edgetpu"
    extractor = FeatureExtractor(str(resolve_workspace_path(spec.model_path)), use_tpu=use_tpu, num_threads=4)
    latencies = []
    feature_vector = None
    for _ in range(int(max(1, repeats))):
        start = time.perf_counter()
        feature_vector = extractor.extract(image_bgr)
        latencies.append((time.perf_counter() - start) * 1000.0)

    return {
        "image_path": str(resolved),
        "feature_dim": int(feature_vector.shape[0]),
        "mean_latency_ms": float(np.mean(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "fps_equivalent": float(1000.0 / max(np.mean(latencies), 1e-6)),
    }


def extract_feature_matrix(dataset_df, image_col, label_col, spec, base_dir=None):
    cv2, np, pd, FeatureExtractor = _import_feature_runtime()
    ensure_analysis_dirs()
    output_dir = FEATURE_OUTPUT_ROOT / sanitize_slug(spec.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_tpu = str(spec.runtime).lower() == "edgetpu"
    extractor = FeatureExtractor(str(resolve_workspace_path(spec.model_path)), use_tpu=use_tpu, num_threads=4)

    features = []
    labels = []
    metadata_rows = []
    latency_rows = []
    failed_rows = []

    for row_index, row in dataset_df.iterrows():
        raw_path = row.get(image_col, "")
        try:
            image_path = resolve_workspace_path(raw_path, base_dir=base_dir)
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError("OpenCV returned no pixels.")

            start = time.perf_counter()
            feature_vector = extractor.extract(image_bgr)
            latency_ms = (time.perf_counter() - start) * 1000.0

            features.append(feature_vector)
            labels.append(str(row[label_col]))

            metadata = row.to_dict()
            metadata["resolved_image_path"] = str(image_path)
            metadata_rows.append(metadata)
            latency_rows.append({
                "row_index": row_index,
                "image_path": str(image_path),
                "latency_ms": latency_ms,
                "feature_dim": int(feature_vector.shape[0]),
            })
        except Exception as exc:
            failed_rows.append({
                "row_index": row_index,
                "image_path": str(raw_path),
                "error": str(exc),
            })

    if not features:
        raise ValueError(f"No features were extracted for {spec.name}. Failed frames: {len(failed_rows)}")

    X = np.vstack(features).astype(np.float32)
    y = np.asarray(labels, dtype=object)
    metadata_df = pd.DataFrame(metadata_rows)
    latency_df = pd.DataFrame(latency_rows)
    failed_df = pd.DataFrame(failed_rows)

    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y, allow_pickle=True)
    metadata_df.to_csv(output_dir / "metadata.csv", index=False)
    latency_df.to_csv(output_dir / "latency.csv", index=False)
    if not failed_df.empty:
        failed_df.to_csv(output_dir / "failed_frames.csv", index=False)

    return {
        "output_dir": output_dir,
        "X": X,
        "y": y,
        "metadata": metadata_df,
        "latency": latency_df,
        "failed": failed_df,
        "summary": {
            "extractor": spec.name,
            "feature_dim": int(X.shape[1]),
            "samples": int(X.shape[0]),
            "failed_frames": int(len(failed_df)),
            "mean_latency_ms": safe_float(latency_df["latency_ms"].mean()),
            "std_latency_ms": safe_float(latency_df["latency_ms"].std()),
            "min_latency_ms": safe_float(latency_df["latency_ms"].min()),
            "max_latency_ms": safe_float(latency_df["latency_ms"].max()),
            "fps_equivalent": 1000.0 / max(safe_float(latency_df["latency_ms"].mean(), 1.0), 1e-6),
        },
    }
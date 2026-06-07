from __future__ import annotations

import numpy as np
import pandas as pd


def _right_like_labels(label_names):
    matches = [label for label in label_names if "right" in str(label).lower()]
    return matches or [label for label in label_names if str(label).lower() == "right"]


def _safe_train_test_split(X, y, test_size, random_state):
    from sklearn.model_selection import train_test_split

    unique, counts = np.unique(y, return_counts=True)
    if counts.min() < 2:
        raise ValueError("Each class needs at least 2 samples for a held-out probe split.")
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def run_probe_suite(X, y, include_mlp=True, test_size=0.25, random_state=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(np.asarray(y, dtype=str))
    class_names = list(encoder.classes_)

    X_train, X_test, y_train, y_test = _safe_train_test_split(X, y_encoded, test_size=test_size, random_state=random_state)

    model_specs = {
        "linear_probe": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "knn_probe": Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=max(1, min(5, len(X_train))))),
        ]),
    }
    if include_mlp:
        model_specs["mlp_probe"] = Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, early_stopping=True, random_state=random_state)),
        ])

    results = {}
    right_labels = _right_like_labels(class_names)

    for model_name, model in model_specs.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        confusion = confusion_matrix(y_test, predictions, labels=np.arange(len(class_names)))

        per_class_f1 = f1_score(y_test, predictions, labels=np.arange(len(class_names)), average=None, zero_division=0)
        metrics = {
            "model": model_name,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "per_class_f1": {class_names[index]: float(score) for index, score in enumerate(per_class_f1)},
            "right_f1": 0.0,
            "left_right_confusion": 0.0,
            "confusion_matrix": confusion,
            "y_test": y_test,
            "predictions": predictions,
        }

        if right_labels:
            right_indices = [class_names.index(label) for label in right_labels if label in class_names]
            if right_indices:
                metrics["right_f1"] = float(np.mean([per_class_f1[index] for index in right_indices]))

        left_indices = [index for index, label in enumerate(class_names) if "left" in label.lower()]
        right_indices = [index for index, label in enumerate(class_names) if "right" in label.lower()]
        confusion_total = float(confusion.sum()) or 1.0
        if left_indices and right_indices:
            mixed = 0.0
            for left_index in left_indices:
                for right_index in right_indices:
                    mixed += float(confusion[left_index, right_index] + confusion[right_index, left_index])
            metrics["left_right_confusion"] = mixed / confusion_total

        results[model_name] = metrics

    return {
        "class_names": class_names,
        "models": results,
        "split": {
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        },
    }


def compute_feature_diagnostics(X, y, probe_results):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    labels = np.asarray(y, dtype=str)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    pca_embedding = pca.fit_transform(X_scaled)

    umap_embedding = None
    try:
        import umap
    except ImportError:  # pragma: no cover - optional dependency
        umap = None

    if umap is not None and len(X) >= 8:
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_embedding = reducer.fit_transform(X_scaled)

    centroid_rows = []
    unique_labels = sorted(np.unique(labels))
    centroids = {}
    for label in unique_labels:
        mask = labels == label
        centroids[label] = X_scaled[mask].mean(axis=0)

    for left_label in unique_labels:
        for right_label in unique_labels:
            if left_label >= right_label:
                continue
            distance = float(np.linalg.norm(centroids[left_label] - centroids[right_label]))
            centroid_rows.append({
                "class_a": left_label,
                "class_b": right_label,
                "distance": distance,
            })

    centroid_df = pd.DataFrame(centroid_rows)
    mean_centroid_separation = float(centroid_df["distance"].mean()) if not centroid_df.empty else 0.0
    right_neighbour_purity = compute_neighbour_purity(X_scaled, labels)

    linear_probe = probe_results["models"].get("linear_probe", {})
    knn_probe = probe_results["models"].get("knn_probe", {})
    right_f1 = float(linear_probe.get("right_f1", 0.0))
    representation_score = float(np.mean([
        float(linear_probe.get("macro_f1", 0.0)),
        right_f1,
        float(knn_probe.get("macro_f1", 0.0)),
        right_neighbour_purity,
    ]))

    return {
        "pca_embedding": pca_embedding,
        "umap_embedding": umap_embedding,
        "centroid_distances": centroid_df,
        "mean_centroid_separation": mean_centroid_separation,
        "right_neighbour_purity": right_neighbour_purity,
        "representation_score": representation_score,
    }


def compute_neighbour_purity(X_scaled, labels, neighbours=10):
    from sklearn.neighbors import NearestNeighbors

    labels = np.asarray(labels, dtype=str)
    right_like = _right_like_labels(sorted(np.unique(labels)))
    if not right_like:
        return 0.0

    right_mask = np.isin(labels, right_like)
    if right_mask.sum() <= 1:
        return 0.0

    neighbour_count = int(max(2, min(neighbours + 1, len(X_scaled))))
    search = NearestNeighbors(n_neighbors=neighbour_count)
    search.fit(X_scaled)
    neighbour_indices = search.kneighbors(X_scaled[right_mask], return_distance=False)

    purities = []
    for row in neighbour_indices:
        effective = row[1:]
        if len(effective) == 0:
            continue
        purities.append(float(np.mean(np.isin(labels[effective], right_like))))

    return float(np.mean(purities)) if purities else 0.0
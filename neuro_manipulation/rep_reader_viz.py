from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.decomposition import PCA
import json


def collect_direction_points(
    emotion_rep_readers: Mapping[str, Any], *, model_id: str
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    vectors: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []

    for emotion, rep_reader in emotion_rep_readers.items():
        if emotion in {"layer_acc", "args"}:
            continue
        directions = getattr(rep_reader, "directions", None)
        if not isinstance(directions, dict):
            raise TypeError(f"emotion {emotion!r} rep_reader has no directions dict")

        for layer, layer_dirs in directions.items():
            layer_dirs = np.asarray(layer_dirs, dtype=np.float32)
            if layer_dirs.ndim != 2:
                raise ValueError(
                    f"emotion {emotion!r} layer {layer} directions must be 2D, got {layer_dirs.shape}"
                )
            for component in range(layer_dirs.shape[0]):
                vectors.append(layer_dirs[component].reshape(-1))
                meta.append(
                    {
                        "model_id": model_id,
                        "emotion": emotion,
                        "layer": int(layer),
                        "component": int(component),
                    }
                )

    if not vectors:
        raise ValueError("No direction vectors found (empty emotion_rep_readers?)")

    return np.vstack(vectors), meta


def reduce_vectors_to_2d(
    vectors: np.ndarray,
    *,
    method: str = "pca",
    seed: int = 0,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    normalize: str = "none",
) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D, got {vectors.shape}")
    if vectors.shape[0] < 2:
        raise ValueError("Need at least 2 vectors to reduce to 2D")

    if normalize not in {"none", "l2"}:
        raise ValueError("normalize must be one of: none, l2")
    if normalize == "l2":
        denom = np.linalg.norm(vectors, axis=1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        vectors = vectors / denom

    if method == "pca":
        return PCA(n_components=2).fit_transform(vectors).astype(np.float32, copy=False)

    if method == "tsne":
        from sklearn.manifold import TSNE

        if perplexity <= 0:
            raise ValueError("t-SNE perplexity must be > 0")
        if perplexity >= vectors.shape[0]:
            raise ValueError(
                f"t-SNE perplexity ({perplexity}) must be < n_samples ({vectors.shape[0]})"
            )

        return (
            TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=seed,
                metric=metric,
            )
            .fit_transform(vectors)
            .astype(np.float32, copy=False)
        )

    if method == "umap":
        import os
        import tempfile
        try:
            os.environ.setdefault(
                "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba_cache")
            )
            Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
            import umap.umap_ as umap
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "UMAP import failed (try setting NUMBA_CACHE_DIR to a writable directory)"
            ) from e

        if n_neighbors <= 1:
            raise ValueError("UMAP n_neighbors must be > 1")
        if not (0.0 <= min_dist <= 1.0):
            raise ValueError("UMAP min_dist must be within [0, 1]")

        return (
            umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=seed,
            )
            .fit_transform(vectors)
            .astype(np.float32, copy=False)
        )

    raise ValueError(f"Unsupported reduction method: {method}")


def emotion_reader_cache_path(
    config: Mapping[str, Any], *, hidden_layers: list[int]
) -> Path:
    from neuro_manipulation.utils import (
        dict_to_unique_code,
        validate_multimodal_experiment_feasibility,
    )

    feasibility = validate_multimodal_experiment_feasibility(config)
    if not feasibility["feasible"]:
        raise ValueError(f"Config not feasible for emotion readers: {feasibility['reasons']}")

    experiment_mode = feasibility["mode"]
    multimodal_intent = bool(config.get("multimodal_intent", False))
    emotion_data_seed = int(config.get("emotion_data_seed", 0))

    args = {
        "emotions": config["emotions"],
        "data_dir": config["data_dir"],
        "model_name_or_path": config["model_name_or_path"],
        "rep_token": config["rep_token"],
        "hidden_layers": hidden_layers,
        "n_difference": config["n_difference"],
        "direction_method": config["direction_method"],
        "experiment_mode": experiment_mode,
        "multimodal_intent": multimodal_intent,
        "emotion_data_seed": emotion_data_seed,
    }

    arg_codes = dict_to_unique_code(args)
    return Path(
        f"neuro_manipulation/representation_storage/emotion_rep_reader_{arg_codes[:10]}.pkl"
    )


def emotion_reader_cache_candidates(
    config: Mapping[str, Any], *, hidden_layers: list[int]
) -> dict[str, Path]:
    """
    Return cache path candidates for multiple schema versions.

    `v2` matches current `neuro_manipulation/model_utils.py:load_emotion_readers` args
    including `emotion_data_seed`.

    `v1` matches older caches where `emotion_data_seed` was not part of the args dict.
    """
    from neuro_manipulation.utils import dict_to_unique_code, validate_multimodal_experiment_feasibility

    feasibility = validate_multimodal_experiment_feasibility(config)
    if not feasibility["feasible"]:
        raise ValueError(f"Config not feasible for emotion readers: {feasibility['reasons']}")

    experiment_mode = feasibility["mode"]
    multimodal_intent = bool(config.get("multimodal_intent", False))
    emotion_data_seed = int(config.get("emotion_data_seed", 0))

    base_args = {
        "emotions": config["emotions"],
        "data_dir": config["data_dir"],
        "model_name_or_path": config["model_name_or_path"],
        "rep_token": config["rep_token"],
        "hidden_layers": hidden_layers,
        "n_difference": config["n_difference"],
        "direction_method": config["direction_method"],
        "experiment_mode": experiment_mode,
        "multimodal_intent": multimodal_intent,
    }

    v2_args = dict(base_args)
    v2_args["emotion_data_seed"] = emotion_data_seed
    v1_args = dict(base_args)

    def _path(args: dict[str, Any]) -> Path:
        code = dict_to_unique_code(args)
        return Path(
            f"neuro_manipulation/representation_storage/emotion_rep_reader_{code[:10]}.pkl"
        )

    return {"v2": _path(v2_args), "v1": _path(v1_args)}


def infer_repe_config_for_model(model_path: str, series_config: Mapping[str, Any]) -> dict[str, Any]:
    from neuro_manipulation.configs.experiment_config import get_repe_eng_config

    repe_overrides = series_config.get("repe_eng_config")
    if repe_overrides is not None and not isinstance(repe_overrides, dict):
        raise TypeError("series_config['repe_eng_config'] must be a dict when provided")
    return get_repe_eng_config(model_path, yaml_config=repe_overrides)


def infer_repe_config_for_model_from_outputs(
    model_path: str, series_config: Mapping[str, Any]
) -> dict[str, Any] | None:
    output_dir = series_config.get("output_dir")
    if not output_dir:
        return None
    base = Path(output_dir)
    if not base.exists():
        return None

    newest = None
    newest_mtime = None
    for cfg_path in base.rglob("experiment_config.json"):
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("model_path") != model_path:
            continue
        repe_cfg = data.get("repe_eng_config")
        if not isinstance(repe_cfg, dict):
            continue
        mtime = cfg_path.stat().st_mtime
        if newest is None or (newest_mtime is not None and mtime > newest_mtime):
            newest = repe_cfg
            newest_mtime = mtime
        elif newest_mtime is None:
            newest = repe_cfg
            newest_mtime = mtime

    return newest

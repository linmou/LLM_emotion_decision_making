# Tests for emotion_experiment_engine/emotion_experiment_series_runner.py
"""Unit tests for OpenMP environment guards."""

import os

from emotion_experiment_engine.emotion_experiment_series_runner import _ensure_openmp_shm_compat


def test_ensure_openmp_shm_compat_sets_kmp_use_shm_when_missing(monkeypatch):
    """When unset, guard should default KMP_USE_SHM=0 to avoid Intel OMP SHM crashes."""

    monkeypatch.delenv("KMP_USE_SHM", raising=False)
    _ensure_openmp_shm_compat()
    assert os.environ.get("KMP_USE_SHM") == "0"


def test_ensure_openmp_shm_compat_does_not_override_existing_value(monkeypatch):
    """If user set KMP_USE_SHM, do not override it."""

    monkeypatch.setenv("KMP_USE_SHM", "1")
    _ensure_openmp_shm_compat()
    assert os.environ.get("KMP_USE_SHM") == "1"

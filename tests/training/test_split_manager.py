"""Tests for fixed stratified split reproducibility, coverage, and hashes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bankruptcy_prediction.data.training_loader import load_training_data
from bankruptcy_prediction.training.split_manager import DatasetSplit, create_split

ROOT = Path(__file__).resolve().parents[2]


def test_official_data_split_is_disjoint_complete_stratified_and_reproducible() -> None:
    dataset = load_training_data(ROOT / "Database" / "bankrupt_data.csv")
    first = create_split(dataset.labels, seed=42)
    second = create_split(dataset.labels, seed=42)

    assert first == second
    assert (len(first.train_indices), len(first.validation_indices), len(first.test_indices)) == (
        4364,
        1091,
        1364,
    )
    all_indices = first.train_indices + first.validation_indices + first.test_indices
    assert len(all_indices) == 6819
    assert set(all_indices) == set(range(6819))
    assert len(set(first.train_indices) & set(first.validation_indices)) == 0
    assert len(set(first.train_indices) & set(first.test_indices)) == 0
    assert len(set(first.validation_indices) & set(first.test_indices)) == 0

    labels = dataset.labels.to_numpy()
    overall_rate = float(labels.mean())
    for partition in (first.train_indices, first.validation_indices, first.test_indices):
        assert abs(float(labels[list(partition)].mean()) - overall_rate) < 0.005


def test_split_hashes_cover_each_index_list_and_round_trip() -> None:
    labels = np.array([0] * 90 + [1] * 10)
    split = create_split(labels, seed=7)
    serialized = split.to_dict()

    restored = DatasetSplit.from_dict(json.loads(json.dumps(serialized)))

    assert restored == split
    assert set(serialized["index_sha256"]) == {
        "train",
        "validation",
        "test",
        "all_partitions",
    }
    altered = json.loads(json.dumps(serialized))
    altered["indices"]["train"][0] = 999
    try:
        DatasetSplit.from_dict(altered)
    except ValueError as error:
        assert "hashes do not match" in str(error)
    else:
        raise AssertionError("tampered split indices must fail hash verification")


def test_split_rejects_invalid_seed_and_non_binary_labels() -> None:
    for seed in (-1, True):
        try:
            create_split([0, 1, 0, 1], seed=seed)
        except ValueError as error:
            assert "seed" in str(error)
        else:
            raise AssertionError("invalid seed should be rejected")

    try:
        create_split([0, 1, 2, 1])
    except ValueError as error:
        assert "binary labels" in str(error)
    else:
        raise AssertionError("non-binary labels should be rejected")

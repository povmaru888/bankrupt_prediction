"""Reproducible stratified 64/16/20 data split with immutable index hashes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

TRAIN_FRACTION = 0.64
VALIDATION_FRACTION = 0.16
TEST_FRACTION = 0.20


def _indices_sha256(indices: tuple[int, ...]) -> str:
    payload = json.dumps(indices, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class DatasetSplit:
    random_seed: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_sha256: str = field(init=False)
    validation_sha256: str = field(init=False)
    test_sha256: str = field(init=False)
    split_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        partitions = (self.train_indices, self.validation_indices, self.test_indices)
        if any(not partition for partition in partitions):
            raise ValueError("all split partitions must be non-empty")
        flattened = [index for partition in partitions for index in partition]
        if any(index < 0 for index in flattened):
            raise ValueError("split indices must be non-negative")
        if len(flattened) != len(set(flattened)):
            raise ValueError("split partitions must be disjoint")
        for name, partition in zip(
            ("train_sha256", "validation_sha256", "test_sha256"), partitions, strict=True
        ):
            object.__setattr__(self, name, _indices_sha256(partition))
        aggregate = {
            "train": self.train_sha256,
            "validation": self.validation_sha256,
            "test": self.test_sha256,
        }
        digest = hashlib.sha256(
            json.dumps(aggregate, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        object.__setattr__(self, "split_sha256", digest)

    @property
    def n_samples(self) -> int:
        return len(self.train_indices) + len(self.validation_indices) + len(self.test_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "random_seed": self.random_seed,
            "n_samples": self.n_samples,
            "proportions": {
                "train": len(self.train_indices) / self.n_samples,
                "validation": len(self.validation_indices) / self.n_samples,
                "test": len(self.test_indices) / self.n_samples,
            },
            "indices": {
                "train": list(self.train_indices),
                "validation": list(self.validation_indices),
                "test": list(self.test_indices),
            },
            "index_sha256": {
                "train": self.train_sha256,
                "validation": self.validation_sha256,
                "test": self.test_sha256,
                "all_partitions": self.split_sha256,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DatasetSplit:
        indices = payload["indices"]
        split = cls(
            random_seed=payload["random_seed"],
            train_indices=tuple(indices["train"]),
            validation_indices=tuple(indices["validation"]),
            test_indices=tuple(indices["test"]),
        )
        supplied = payload.get("index_sha256", {})
        expected = {
            "train": split.train_sha256,
            "validation": split.validation_sha256,
            "test": split.test_sha256,
            "all_partitions": split.split_sha256,
        }
        if supplied and supplied != expected:
            raise ValueError("serialized split index hashes do not match the partition indices")
        if payload.get("n_samples", split.n_samples) != split.n_samples:
            raise ValueError("serialized n_samples does not match partition sizes")
        return split


def create_split(y: Any, seed: int = 42) -> DatasetSplit:
    """Create fixed stratified train/validation/test indices for binary labels."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if labels.size < 3 or pd_is_missing(labels).any():
        raise ValueError("y must contain at least three non-missing labels")
    unique = set(np.unique(labels).tolist())
    if unique != {0, 1}:
        raise ValueError("y must contain both binary labels 0 and 1")

    indices = np.arange(labels.size)
    train_idx, remaining_idx = train_test_split(
        indices,
        test_size=VALIDATION_FRACTION + TEST_FRACTION,
        random_state=seed,
        stratify=labels,
    )
    validation_idx, test_idx = train_test_split(
        remaining_idx,
        test_size=TEST_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION),
        random_state=seed,
        stratify=labels[remaining_idx],
    )
    return DatasetSplit(
        random_seed=seed,
        train_indices=tuple(sorted(int(index) for index in train_idx)),
        validation_indices=tuple(sorted(int(index) for index in validation_idx)),
        test_indices=tuple(sorted(int(index) for index in test_idx)),
    )


def pd_is_missing(values: np.ndarray) -> np.ndarray:
    """Return a NumPy missing-value mask without importing pandas in this module."""
    return np.asarray(values != values, dtype=bool)

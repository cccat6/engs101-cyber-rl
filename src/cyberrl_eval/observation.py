"""Utilities for converting structured observations into model-ready vectors."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def flatten_observation(
    observation: Mapping[str, np.ndarray | float],
    node_cardinalities: tuple[int, ...],
) -> np.ndarray:
    """One-hot encode the node categories and append topology and time.

    The action mask is intentionally not included as a feature. It should be
    applied to Q values or policy logits by the agent.
    """

    nodes = np.asarray(observation["nodes"], dtype=np.int64)
    if nodes.ndim != 2 or nodes.shape[1] != len(node_cardinalities):
        raise ValueError(
            f"Expected nodes with {len(node_cardinalities)} columns, got {nodes.shape}"
        )

    encoded_columns: list[np.ndarray] = []
    for column, cardinality in enumerate(node_cardinalities):
        values = nodes[:, column]
        if np.any(values < 0) or np.any(values >= cardinality):
            raise ValueError(
                f"Node column {column} contains a value outside [0, {cardinality})"
            )
        encoded_columns.append(np.eye(cardinality, dtype=np.float32)[values].reshape(-1))

    adjacency = np.asarray(observation["adjacency"], dtype=np.float32).reshape(-1)
    time = np.asarray([observation["time"]], dtype=np.float32)
    return np.concatenate([*encoded_columns, adjacency, time]).astype(np.float32, copy=False)


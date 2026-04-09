from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def get_prior_means(
    arms: List[Any],
    prior: str | None,
    true_final_means: Dict[Any, float],
    epsilon: float,
    rng: np.random.Generator,
    prior_std: float = 0.01,
) -> Dict[Any, float]:
    """Compute prior means for each arm."""
    max_true_mean = max(true_final_means.values())
    num_arms = len(arms)
    prior_means: Dict[Any, float] = {}

    if prior == "uniform":
        mean_val = float(np.mean(list(true_final_means.values())))
        for arm in arms:
            prior_means[arm] = mean_val
    elif prior == "rank":
        for arm in arms:
            prior_means[arm] = 1 / (arm + 1)
    elif prior == "performance":
        for arm in arms:
            prior_means[arm] = min(1.0, float(rng.normal(true_final_means[arm], prior_std)))
    elif prior == "inverse_rank":
        for arm in arms:
            prior_means[arm] = (arm + 1) / num_arms
    elif prior == "indicator":
        for arm in arms:
            prior_means[arm] = 1.0 if (max_true_mean - true_final_means[arm]) <= epsilon else 0.0
    elif prior is None:
        for arm in arms:
            prior_means[arm] = 0.0
    else:
        raise ValueError(f"Unknown prior type: {prior!r}")

    return prior_means

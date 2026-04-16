from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


class Runhistory:
    """Stores all evaluated (configuration, performance) pairs, indexed by fidelity.

    Structure::

        {
            fidelity_1: [(config_a, perf_a), (config_b, perf_b), ...],
            fidelity_2: [...],
            ...
        }
    """

    def __init__(self, eta:int) -> None:
        self._data: dict[int, list[tuple[Any, float]]] = defaultdict(list)
        self.eta = eta


    def add(self, fidelity: int, config: Any, performance: float) -> None:
        self._data[fidelity].append((config, performance))

    def __getitem__(self, fidelity: int) -> list[tuple[Any, float]]:
        return self._data[fidelity]

    def fidelities(self) -> list[int]:
        return sorted(self._data.keys())

    def max_fidelity(self) -> int | None:
        return max(self._data.keys()) if self._data else None

    def __repr__(self) -> str:
        summary = {f: len(entries) for f, entries in self._data.items()}
        return f"Runhistory({summary})"

    def get_priorband_relevant_configurations(self) -> list[tuple[Any, float]]:
        """Return the top-1/η configs from the highest rung with ≥ η evaluations.

        Implements Λ'_z from Algorithm 2 of PriorBand (Mallik et al., NeurIPS 2023):
        ordered by performance descending, top ceil(n / η) entries.
        """
        eligible = [f for f in self._data if len(self._data[f]) >= self.eta]
        if not eligible:
            return []

        relevant_fidelity = max(eligible)
        # Higher performance is better (maximisation)
        sorted_configs = sorted(self._data[relevant_fidelity], key=lambda x: x[1], reverse=True)
        top_k = math.ceil(len(sorted_configs) / self.eta)
        return sorted_configs[:top_k]
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Benchmark(ABC):
    @abstractmethod
    def get_t_max(self) -> int: ...

    @abstractmethod
    def sample(self, num_arms: int, seed: int) -> tuple[list[int], dict[int, float], Callable[[int, np.ndarray], np.ndarray]]:
        """Sample num_arms configurations. Returns (arms, true_final_means, evaluate_fn)."""
        ...


class SyntheticBenchmark(Benchmark):
    def __init__(self, t_max: int):
        self._t_max = t_max

    def get_t_max(self) -> int:
        return self._t_max

    def sample(self, num_arms: int, seed: int) -> tuple[list[int], dict[int, float], Callable]:
        rng = np.random.default_rng(seed)
        final_means = sorted([rng.random() for _ in range(num_arms)], reverse=True)
        true_final_means: dict[int, float] = {arm: final_means[arm] for arm in range(num_arms)}

        def evaluate(arm: int, t: np.ndarray) -> np.ndarray:
            true_mu = true_final_means[arm]
            tau = 20.0 + 10.0 * arm
            return true_mu * (1.0 - np.exp(-t / tau))

        return list(range(num_arms)), true_final_means, evaluate


class LCBenchBenchmark(Benchmark):
    _T_MAX = 52

    def __init__(self, dataset_id: int):
        from yahpo_gym import benchmark_set, local_config

        local_config._config = {"data_path": "data/yahpogym/"}
        self._bset = benchmark_set.BenchmarkSet("lcbench")
        self._bset.set_instance(self._bset.instances[dataset_id])

    def get_t_max(self) -> int:
        return self._T_MAX

    def sample(self, num_arms: int, seed: int) -> tuple[list[int], dict[int, float], Callable]:
        cfg_list = self._bset.get_opt_space(seed=seed).sample_configuration(size=num_arms)
        configs: List[Tuple[Dict, float]] = []
        for cfg in cfg_list:
            cfg_dict = cfg.get_dictionary()
            cfg_dict["epoch"] = self._T_MAX
            configs.append(
                (cfg_dict, self._bset.objective_function(cfg_dict)[0]["val_accuracy"] / 100)
            )
        configs = sorted(configs, key=lambda c: c[1], reverse=True)
        arms = list(range(len(configs)))
        true_final_means: dict[int, float] = {arm: configs[arm][1] for arm in arms}

        t_max = self._T_MAX
        bset = self._bset

        def evaluate(arm: int, t: np.ndarray) -> np.ndarray:
            cfg = dict(configs[arm][0])
            res = []
            for ti in t:
                cfg["epoch"] = min(int(ti), t_max)
                res.append(bset.objective_function(cfg)[0]["val_accuracy"] / 100)
            return np.array(res)

        return arms, true_final_means, evaluate


def get_benchmark(benchmark_name: str, num_arms: int, dataset_id: int) -> Benchmark:
    if benchmark_name == "synthetic":
        return SyntheticBenchmark(t_max=num_arms)
    elif benchmark_name == "lcbench":
        return LCBenchBenchmark(dataset_id)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name!r}")

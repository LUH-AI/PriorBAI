from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Benchmark(ABC):
    @abstractmethod
    def get_t_max(self) -> int: ...

    @abstractmethod
    def get_true_final_means(self) -> Dict[Any, float]: ...

    @abstractmethod
    def evaluate(self, arm: Any, t: np.ndarray) -> np.ndarray: ...


class SyntheticBenchmark(Benchmark):
    def __init__(self, num_arms: int, rng: np.random.Generator):
        final_means = sorted([rng.random() for _ in range(num_arms)], reverse=True)
        self._t_max = num_arms
        self._true_final_means: Dict[int, float] = {
            arm: final_means[arm] for arm in range(num_arms)
        }

    def get_t_max(self) -> int:
        return self._t_max

    def get_true_final_means(self) -> Dict[int, float]:
        return self._true_final_means

    def evaluate(self, arm: int, t: np.ndarray) -> np.ndarray:
        true_mu = self._true_final_means[arm]
        tau = 20.0 + 10.0 * arm
        return true_mu * (1.0 - np.exp(-t / tau))


class LCBenchBenchmark(Benchmark):
    _T_MAX = 52

    def __init__(self, num_arms: int, seed: int, dataset_id: int):
        from yahpo_gym import benchmark_set, local_config

        local_config._config = {"data_path": "data/yahpogym/"}
        self._benchmark = benchmark_set.BenchmarkSet("lcbench")

        self._benchmark.set_instance(self._benchmark.instances[dataset_id])

        cfg_list = self._benchmark.get_opt_space(seed=seed).sample_configuration(size=num_arms)
        configs: List[Tuple[Dict, float]] = []
        for cfg in cfg_list:
            cfg_dict = cfg.get_dictionary()
            cfg_dict["epoch"] = self._T_MAX
            configs.append(
                (cfg_dict, self._benchmark.objective_function(cfg_dict)[0]["val_accuracy"] / 100)
            )
        self._configs = sorted(configs, key=lambda c: c[1], reverse=True)
        self._true_final_means: Dict[int, float] = {
            arm: self._configs[arm][1] for arm in range(num_arms)
        }

    def get_t_max(self) -> int:
        return self._T_MAX

    def get_true_final_means(self) -> Dict[int, float]:
        return self._true_final_means

    def evaluate(self, arm: int, t: np.ndarray) -> np.ndarray:
        cfg = self._configs[arm][0]
        res = []
        for ti in t:
            cfg["epoch"] = min(int(ti), self._T_MAX)
            res.append(self._benchmark.objective_function(cfg)[0]["val_accuracy"] / 100)
        return np.array(res)


def get_benchmark(benchmark_name: str, num_arms: int, seed: int, rng: np.random.Generator, dataset_id: int) -> Benchmark:
    if benchmark_name == "synthetic":
        return SyntheticBenchmark(num_arms, rng)
    elif benchmark_name == "lcbench":
        return LCBenchBenchmark(num_arms, seed, dataset_id)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name!r}")

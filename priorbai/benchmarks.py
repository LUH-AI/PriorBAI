from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class Benchmark(ABC):
    @abstractmethod
    def get_max_fidelity(self) -> int: ...

    @abstractmethod
    def sample(self, num_arms: int, seed: int) -> tuple[list[int], dict[int, float]]:
        """Sample num_arms configurations. Returns (arms, true_final_means)."""
        ...

    @abstractmethod
    def evaluate(self, arm: int, fidelity_levels: np.ndarray) -> np.ndarray:
        """Evaluate the configuration associated with arm at the given fidelities."""
        ...


class SyntheticBenchmark(Benchmark):
    def __init__(self, max_fidelity: int):
        self.max_fidelity = max_fidelity
        self.true_final_means: dict[int, float] = {}

    def get_max_fidelity(self) -> int:
        return self.max_fidelity

    def sample(self, num_arms: int, seed: int) -> tuple[list[int], dict[int, float]]:
        rng = np.random.default_rng(seed)
        final_means = sorted([rng.random() for _ in range(num_arms)], reverse=True)
        self.true_final_means = {arm: final_means[arm] for arm in range(num_arms)}
        return list(range(num_arms)), self.true_final_means

    def evaluate(self, arm: int, fidelity_levels: np.ndarray) -> np.ndarray:
        true_mu = self.true_final_means[arm]
        tau = 20.0 + 10.0 * arm
        return true_mu * (1.0 - np.exp(-fidelity_levels / tau))


class LCBenchBenchmark(Benchmark):
    MAX_FIDELITY = 52

    def __init__(self, dataset_id: int):
        from yahpo_gym import benchmark_set, local_config

        local_config._config = {"data_path": "data/yahpogym/"}
        self.benchmarkset = benchmark_set.BenchmarkSet("lcbench")
        self.benchmarkset.set_instance(self.benchmarkset.instances[dataset_id])
        self.configs: list[dict] = []

    def get_max_fidelity(self) -> int:
        return self.MAX_FIDELITY

    def sample(self, num_arms: int, seed: int) -> tuple[list[int], dict[int, float]]:
        cfg_list = self.benchmarkset.get_opt_space(seed=seed).sample_configuration(size=num_arms)
        configs: list[tuple[dict, float]] = []
        for cfg in cfg_list:
            cfg_dict = cfg.get_dictionary()
            cfg_dict["epoch"] = self.MAX_FIDELITY
            configs.append(
                (cfg_dict, self.benchmarkset.objective_function(cfg_dict)[0]["val_accuracy"] / 100)
            )
        configs = sorted(configs, key=lambda c: c[1], reverse=True)
        self.configs = [cfg for cfg, _ in configs]
        arms = list(range(len(configs)))
        true_final_means: dict[int, float] = {arm: configs[arm][1] for arm in arms}
        return arms, true_final_means

    def evaluate(self, arm: int, fidelity_levels: np.ndarray) -> np.ndarray:
        cfg = dict(self.configs[arm])
        res = []
        for fidelity in fidelity_levels:
            cfg["epoch"] = min(int(fidelity), self.MAX_FIDELITY)
            res.append(self.benchmarkset.objective_function(cfg)[0]["val_accuracy"] / 100)
        return np.array(res)


def get_benchmark(benchmark_name: str, num_arms: int, dataset_id: int) -> Benchmark:
    if benchmark_name == "synthetic":
        return SyntheticBenchmark(max_fidelity=num_arms)
    elif benchmark_name == "lcbench":
        return LCBenchBenchmark(dataset_id)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name!r}")

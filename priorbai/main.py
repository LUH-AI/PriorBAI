from __future__ import annotations

import logging
import math
import warnings
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
from py_experimenter.experimenter import PyExperimenter
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

from priorbai.benchmarks import get_benchmark
from priorbai.kernels import get_kernel
from priorbai.priors import get_prior_means

logger = logging.getLogger(__name__)


def mf_prior_guided_successive_halving(
        arms: Sequence[Any],
        budget_N: int,
        prior_means: Dict[Any, float],
        T_max: int,
        observe_fn: Callable[[Any, np.ndarray], np.ndarray],
        kernel: Kernel | None,
        use_predicted_y: bool,
        use_early_stopping: bool,
        delta: float,
        epsilon: float,
        sigma0_sq: float,
        rng: np.random.Generator,
        seed: int,
        result_processor: Any | None,
) -> tuple[Any, int, int]:

    arms = list(arms)
    K = len(arms)
    if K < 1:
        raise ValueError("At least one arm is required.")

    S_r = arms.copy()
    R = math.ceil(math.log2(K))
    mu_hat: Dict[Any, float] = {arm: prior_means[arm] for arm in arms}
    N_used = 0
    n_prev = 0
    arm_ts: Dict[Any, List[int]] = {arm: [] for arm in arms}
    arm_ys: Dict[Any, List[float]] = {arm: [] for arm in arms}
    C_log = math.log(2.0 * math.log2(K) * (K / 2 - 1) / delta)
    logger.debug("C_log=%s", C_log)

    for r in range(R):
        if len(S_r) == 0:
            logger.warning("S_r is empty at round %d — this should not happen.", r)
            break

        n_r = math.floor(budget_N / (R * len(S_r)))
        if n_r <= n_prev:
            logger.debug("No additional budget to allocate at round %d; stopping.", r)
            break

        N_used += len(S_r) * (n_r - n_prev)

        # 1. Observation & Extrapolation
        round_y: Dict[Any, float] = {}
        round_mu_hat: Dict[Any, float] = {}
        round_sigma_sq: Dict[Any, float] = {}
        Sigma = 0.0

        for arm in S_r:
            new_ts = np.arange(n_prev + 1, n_r + 1, dtype=int)
            if len(new_ts) > 0:
                new_ys = observe_fn(arm, new_ts)
                if len(new_ys) != len(new_ts):
                    raise ValueError("observe_fn must return one y per t.")
                arm_ts[arm].extend(new_ts.tolist())
                arm_ys[arm].extend(new_ys.tolist())

            X = np.asarray(arm_ts[arm], dtype=float).reshape(-1, 1) / T_max
            y = np.asarray(arm_ys[arm], dtype=float)
            logger.debug("arm=%s  X=%s  y=%s", arm, X, y)

            nu_j = prior_means[arm]

            if len(X) == 0:
                mu_j_r = nu_j
                sigma_j_r_sq = sigma0_sq
            else:
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=0.05 ** 2,
                    normalize_y=False,
                    random_state=seed,
                )
                gp.fit(X, y)
                X_star = np.array([[1.0]])
                mu_res, std_res = gp.predict(X_star, return_std=True)
                mu_j_r = min(1.0, float(mu_res.item()))
                sigma_j_r_sq = float(std_res.item() ** 2)
                Sigma += sigma_j_r_sq

                logger.debug(
                    "arm=%s  n_r=%d  mu_j_r=%.4f  sigma_j_r_sq=%.4f  actual=%.4f",
                    arm, n_r, mu_j_r, sigma_j_r_sq, arm_ys[arm][-1],
                )

            round_y[arm] = arm_ys[arm][-1]
            round_mu_hat[arm] = mu_j_r
            round_sigma_sq[arm] = sigma_j_r_sq

        logger.debug("round_sigma_sq=%s  Sigma=%.6f", round_sigma_sq, Sigma)

        n_prev = n_r
        mu_hat.update(round_mu_hat)

        # 2. Candidate Selection
        i_hat = max(S_r, key=lambda a: round_mu_hat[a])

        # 3. Stopping Condition
        Deltas: Dict[Any, float] = {}
        for arm in S_r:
            if arm == i_hat:
                continue
            gap = round_mu_hat[i_hat] - round_mu_hat[arm]
            Deltas[arm] = max(epsilon, gap)

        logger.debug("round_mu_hat=%s", round_mu_hat)

        if Deltas:
            N_stop_candidates = []
            nu_i = prior_means[i_hat]
            for arm, Delta_j_r in Deltas.items():
                nu_j = prior_means[arm]
                logger.debug(
                    "arm=%s  nu_i=%.4f  nu_j=%.4f  nu_diff=%.4f",
                    arm, nu_i, nu_j, nu_i - nu_j,
                )
                bracket = C_log - ((nu_i - nu_j) * Delta_j_r) / (2.0 * sigma0_sq)
                bracket = max(bracket, 0.0)
                if Delta_j_r <= 0:
                    continue

                N_stop_j = (4.0 * R * Sigma / (Delta_j_r ** 2)) * bracket
                base_budget = (4.0 * R * Sigma / (Delta_j_r ** 2)) * C_log
                budget_reduction = (4.0 * R * Sigma / (Delta_j_r ** 2)) * ((nu_i - nu_j) * Delta_j_r) / (2.0 * sigma0_sq)
                logger.debug(
                    "arm=%s  r=%d  R=%d  Sigma=%.6f  Delta=%.6f  bracket=%.6f  N_stop=%.4f  "
                    "base_budget=%.4f  budget_reduction=%.4f",
                    arm, r, R, Sigma, Delta_j_r, bracket, N_stop_j, base_budget, budget_reduction,
                )
                N_stop_candidates.append(N_stop_j)

            if N_stop_candidates:
                N_stop_arr = np.array(N_stop_candidates)
                logger.debug(
                    "N_stop  min=%.4f  max=%.4f  mean=%.4f  std=%.4f",
                    N_stop_arr.min(), N_stop_arr.max(), N_stop_arr.mean(), N_stop_arr.std(),
                )

            N_stop = max(N_stop_candidates) if N_stop_candidates else 0.0
            logger.debug("N_used=%d  N_stop=%.4f", N_used, N_stop)

            if result_processor is not None:
                result_processor.process_logs({
                    "sh_iterations": {
                        "iteration": r,
                        "num_arms": len(S_r),
                        "best_arm_included": 1 if 0 in S_r else 0,
                        "budget_spent_so_far": N_used,
                        "N_stop": N_stop,
                    }
                })

            if use_early_stopping and N_used >= N_stop:
                logger.debug(
                    "Stopping condition reached at round %d/%d with %d arms remaining.",
                    r, R, len(S_r),
                )
                return i_hat, N_used, len(S_r)

        # 4. Pruning
        if use_predicted_y:
            S_r_sorted = sorted(S_r, key=lambda a: round_mu_hat[a], reverse=True)
        else:
            S_r_sorted = sorted(S_r, key=lambda a: round_y[a], reverse=True)

        keep = math.ceil(len(S_r_sorted) / 2.0)
        S_r = S_r_sorted[:keep]

        if 0 not in S_r:
            logger.warning("The best arm (arm 0) was eliminated at round %d.", r)

    if len(S_r) == 0:
        return max(arms, key=lambda a: mu_hat[a]), N_used, len(S_r)
    return max(S_r, key=lambda a: mu_hat[a]), N_used, len(S_r)


def run_experiment(config, result_processor, custom_config):
    logger.debug("config: %s", config)

    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    benchmark_name = config["benchmark"]
    num_arms = int(config["num_arms"])
    prior = config["prior"]
    sigma0 = float(config["sigma0"])
    sigma0_sq = sigma0 ** 2
    epsilon = float(config["epsilon"])
    delta = float(config["delta"])
    kernel_name = config.get("kernel", "satexp_rbf")
    use_predicted_y = bool(config["use_predicted_y"])
    use_early_stopping = bool(config.get("use_early_stopping", False))

    benchmark = get_benchmark(benchmark_name, num_arms, seed, rng)
    T_max = benchmark.get_t_max()
    true_final_means = benchmark.get_true_final_means()
    logger.info("True final means: %s", list(true_final_means.values()))

    arms = list(true_final_means.keys())
    prior_means = get_prior_means(arms, prior, true_final_means, epsilon, rng)
    lc_kernel = get_kernel(kernel_name)

    selected_best, budget_used, num_arms_left = mf_prior_guided_successive_halving(
        arms=arms,
        budget_N=num_arms * np.log2(num_arms),
        prior_means=prior_means,
        T_max=T_max,
        observe_fn=benchmark.evaluate,
        kernel=lc_kernel,
        use_predicted_y=use_predicted_y,
        use_early_stopping=use_early_stopping,
        delta=delta,
        epsilon=epsilon,
        sigma0_sq=sigma0_sq,
        rng=rng,
        seed=seed,
        result_processor=result_processor,
    )

    actual_best = max(true_final_means, key=true_final_means.get)
    max_true_mean = max(true_final_means.values())
    num_epsilon_optimal_arms = sum(
        1 for arm in arms if max_true_mean - true_final_means[arm] <= epsilon
    )
    regret = max_true_mean - true_final_means[selected_best]

    result_processor.process_results({
        "T_max": T_max,
        "consumed_budget": budget_used,
        "remaining_arms": num_arms_left,
        "num_epsilon_optimal_arms": num_epsilon_optimal_arms,
        "arm_id_selected": selected_best,
        "regret": regret,
        "epsilon_optimal": 1 if regret <= epsilon else 0,
        "best_arm": 1 if actual_best == selected_best else 0,
    })


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pyexp = PyExperimenter(
        experiment_configuration_file_path="conf/experiment_config.yml",
        database_credential_file_path="conf/database_credentials.yml",
        use_codecarbon=False,
    )
    pyexp.fill_table_from_config()
    pyexp.execute(run_experiment, max_experiments=1, random_order=True)

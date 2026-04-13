from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from py_experimenter.experimenter import PyExperimenter
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

from priorbai.benchmarks import get_benchmark
from priorbai.kernels import get_kernel
from priorbai.priors import get_prior_means

logger = logging.getLogger(__name__)


def prior_guided_successive_halving(
        arms: Sequence[Any],
        budget_N: int,
        prior_means: dict[Any, float],
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
        hb_bracket: int | None,
) -> tuple[Any, int, int]:

    arms = list(arms)
    number_of_arms = len(arms)
    if number_of_arms < 1:
        raise ValueError("At least one arm is required.")

    active_arms = arms.copy()
    number_of_rounds = math.ceil(math.log2(number_of_arms))
    mu_hat: dict[Any, float] = {arm: prior_means[arm] for arm in arms}
    budget_consumed = 0
    previous_round_budget = 0
    arm_ts: dict[Any, list[int]] = {arm: [] for arm in arms}
    arm_ys: dict[Any, list[float]] = {arm: [] for arm in arms}
    N_stop = 0.0
    round_index = -1
    stopped_early = False
    C_log = math.log(2.0 * math.log2(number_of_arms) * ((number_of_arms / 2) - 1) / delta)
    logger.debug("C_log=%s", C_log)

    for round_index in range(number_of_rounds):
        if len(active_arms) == 0:
            logger.warning("S_r is empty at round %d — this should not happen.", round_index)
            break

        round_budget = math.floor(budget_N / (number_of_rounds * len(active_arms)))
        if round_budget <= previous_round_budget:
            logger.debug("No additional budget to allocate at round %d; stopping.", round_index)
            break

        budget_consumed += len(active_arms) * (round_budget - previous_round_budget)

        # 1. Observation & Extrapolation
        round_y: dict[Any, float] = {}
        round_mu_hat: dict[Any, float] = {}
        round_sigma_sq: dict[Any, float] = {}
        Sigma = 0.0

        for arm in active_arms:
            new_ts = np.arange(previous_round_budget + 1, round_budget + 1, dtype=int)
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
                    arm, round_budget, mu_j_r, sigma_j_r_sq, arm_ys[arm][-1],
                )

            round_y[arm] = arm_ys[arm][-1]
            round_mu_hat[arm] = mu_j_r
            round_sigma_sq[arm] = sigma_j_r_sq

        logger.debug("round_sigma_sq=%s  Sigma=%.6f", round_sigma_sq, Sigma)

        previous_round_budget = round_budget
        mu_hat.update(round_mu_hat)

        # 2. Candidate Selection
        i_hat = max(active_arms, key=lambda a: round_mu_hat[a])

        # 3. Stopping Condition
        Deltas: dict[Any, float] = {}
        for arm in active_arms:
            if arm == i_hat:
                continue
            performance_gap = round_mu_hat[i_hat] - round_mu_hat[arm]
            Deltas[arm] = max(epsilon, performance_gap)

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
                formula_bracket = C_log - ((nu_i - nu_j) * Delta_j_r) / (2.0 * sigma0_sq)
                formula_bracket = max(formula_bracket, 0.0)
                if Delta_j_r <= 0:
                    continue

                N_stop_j = (4.0 * number_of_rounds * Sigma / (Delta_j_r ** 2)) * formula_bracket
                base_budget = (4.0 * number_of_rounds * Sigma / (Delta_j_r ** 2)) * C_log
                budget_reduction = (4.0 * number_of_rounds * Sigma / (Delta_j_r ** 2)) * ((nu_i - nu_j) * Delta_j_r) / (2.0 * sigma0_sq)
                logger.debug(
                    "arm=%s  r=%d  number_of_rounds=%d  Sigma=%.6f  Delta=%.6f  bracket=%.6f  N_stop=%.4f  "
                    "base_budget=%.4f  budget_reduction=%.4f",
                    arm, round_index, number_of_rounds, Sigma, Delta_j_r, hb_bracket, N_stop_j, base_budget, budget_reduction,
                )
                N_stop_candidates.append(N_stop_j)

            if N_stop_candidates:
                N_stop_arr = np.array(N_stop_candidates)
                logger.debug(
                    "N_stop  min=%.4f  max=%.4f  mean=%.4f  std=%.4f",
                    N_stop_arr.min(), N_stop_arr.max(), N_stop_arr.mean(), N_stop_arr.std(),
                )

            N_stop = max(N_stop_candidates) if N_stop_candidates else 0.0
            logger.debug("N_used=%d  N_stop=%.4f", budget_consumed, N_stop)

            if result_processor is not None:
                result_processor.process_logs({
                    "sh_iterations": {
                        "bracket": hb_bracket,
                        "iteration": round_index,
                        "num_arms": len(active_arms),
                        "best_arm_included": 1 if 0 in active_arms else 0,
                        "budget_spent_so_far": budget_consumed,
                        "N_stop": N_stop,
                    }
                })

            if use_early_stopping and budget_consumed >= N_stop:
                logger.debug(
                    "Stopping condition reached at round %d/%d with %d arms remaining.",
                    round_index, number_of_rounds, len(active_arms),
                )
                best = i_hat
                break

        # 4. Pruning
        if use_predicted_y:
            S_r_sorted = sorted(active_arms, key=lambda a: round_mu_hat[a], reverse=True)
        else:
            S_r_sorted = sorted(active_arms, key=lambda a: round_y[a], reverse=True)

        number_of_arms_to_keep = math.ceil(len(S_r_sorted) / 2.0)
        active_arms = S_r_sorted[:number_of_arms_to_keep]

        if 0 not in active_arms:
            logger.warning("The best arm (arm 0) was eliminated at round %d.", round_index)
    else:
        best = max(arms if len(active_arms) == 0 else active_arms, key=lambda a: mu_hat[a])

    if result_processor is not None:
        result_processor.process_logs({
            "brackets": {
                "bracket": hb_bracket if hb_bracket is not None else 0,
                "n_arms": number_of_arms,
                "budget_used": budget_consumed,
                "stopped_early": 1 if stopped_early else 0,
                "stopped_after_round": round_index,
                "winner": best,
            }
        })
    return best, budget_consumed, len(active_arms)


def prior_guided_hyperband(
        arms: Sequence[Any],
        eta: int,
        prior_means: dict[Any, float],
        max_fidelity: int,
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
    """Hyperband wrapper around prior_guided_successive_halving.

    Runs s_max + 1 brackets (s_max = floor(log_eta(n))).  Bracket s starts
    with n_s = max(1, ceil(n / eta^(s_max - s))) arms — bracket s_max uses
    all arms (most exploration), bracket 0 uses ~1 arm (most exploitation).
    Each bracket is given budget_N = n * T_max so it can run a full SHA.
    The winner of each bracket is evaluated at T_max; the arm with the
    highest final observed value is returned.
    """
    arms = list(arms)
    n = len(arms)
    if n < 1:
        raise ValueError("At least one arm is required.")

    # Paper formula: s_max = floor(log_eta(R)) where R = T_max (max budget per arm).
    # Each bracket gets budget B = (s_max + 1) * T_max.
    number_of_halving_rounds = math.floor(math.log(max_fidelity, eta)) if max_fidelity > 1 else 0
    bracketwise_budget = (number_of_halving_rounds + 1) * max_fidelity  # equal budget envelope for every bracket

    # Pre-compute bracket sizes and assign non-overlapping arm slices.
    bracket_sizes = [
        min(n, max(1, math.ceil((number_of_halving_rounds + 1) * eta ** s / (s + 1))))
        for s in range(number_of_halving_rounds, -1, -1)
    ]
    total_arms_needed = sum(bracket_sizes)
    if total_arms_needed > n:
        raise ValueError(
            f"Hyperband needs {total_arms_needed} arms across brackets but only {n} are available."
        )
    shuffled_arms = list(arms)
    rng.shuffle(shuffled_arms)
    bracket_arm_assignments = []
    offset = 0
    for size in bracket_sizes:
        bracket_arm_assignments.append(shuffled_arms[offset:offset + size])
        offset += size

    total_budget = 0
    bracket_results: list[tuple[Any, float]] = []

    for hb_bracket, bracket_arms in zip(range(number_of_halving_rounds, -1, -1), bracket_arm_assignments):
        n_s = len(bracket_arms)

        logger.debug(
            "Hyperband bracket s=%d/%d: n_s=%d arms, r_s=%.1f, B=%d",
            hb_bracket, number_of_halving_rounds, n_s, max_fidelity / eta ** hb_bracket, bracketwise_budget,
        )

        winner, budget_used, _ = prior_guided_successive_halving(
            arms=bracket_arms,
            budget_N=bracketwise_budget,
            hb_bracket=hb_bracket,
            prior_means=prior_means,
            T_max=max_fidelity,
            observe_fn=observe_fn,
            kernel=kernel,
            use_predicted_y=use_predicted_y,
            use_early_stopping=use_early_stopping,
            delta=delta,
            epsilon=epsilon,
            sigma0_sq=sigma0_sq,
            rng=rng,
            seed=seed,
            result_processor=result_processor,
        )
        total_budget += budget_used

        # Evaluate bracket winner at T_max for cross-bracket comparison.
        final_y = float(observe_fn(winner, np.array([max_fidelity], dtype=int))[-1])
        total_budget += 1
        bracket_results.append((winner, final_y))
        logger.debug("Bracket s=%d winner=%s  final_y=%.4f", hb_bracket, winner, final_y)

    best_arm, _ = max(bracket_results, key=lambda x: x[1])
    return best_arm, total_budget, 1


def setup_run(config, rng):
    benchmark_name = config["benchmark"]
    num_arms = int(config["num_arms"])
    seed = int(config["seed"])
    dataset_id = int(config["dataset_id"])
    prior = config["prior"]
    epsilon = float(config["epsilon"])
    kernel_name = config.get("kernel", "satexp_rbf")

    benchmark = get_benchmark(benchmark_name, num_arms, seed, rng, dataset_id)
    true_final_means = benchmark.get_true_final_means()
    logger.info("True final means: %s", list(true_final_means.values()))

    arms = list(true_final_means.keys())
    prior_means = get_prior_means(arms, prior, true_final_means, epsilon, rng)
    learning_curve_kernel = get_kernel(kernel_name)

    return benchmark, arms, prior_means, learning_curve_kernel


def run_experiment(config, result_processor, custom_config):
    logger.debug("config: %s", config)

    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    sigma0 = float(config["sigma0"])
    sigma0_sq = sigma0 ** 2
    epsilon = float(config["epsilon"])
    delta = float(config["delta"])
    use_predicted_y = bool(config["use_predicted_y"])
    use_early_stopping = bool(config.get("use_early_stopping", False))

    optimizer = config["optimizer"]

    benchmark, arms, prior_means, learning_curve_kernel = setup_run(config, rng)
    T_max = benchmark.get_t_max()
    true_final_means = benchmark.get_true_final_means()

    shared_kwargs = dict(
        prior_means=prior_means,
        T_max=T_max,
        observe_fn=benchmark.evaluate,
        kernel=learning_curve_kernel,
        use_predicted_y=use_predicted_y,
        use_early_stopping=use_early_stopping,
        delta=delta,
        epsilon=epsilon,
        sigma0_sq=sigma0_sq,
        rng=rng,
        seed=seed,
        result_processor=result_processor,
    )

    if optimizer == "hyperband":
        selected_best, budget_used, num_arms_left = prior_guided_hyperband(
            arms=arms,
            eta=2,
            **shared_kwargs,
        )
    elif optimizer == "successive_halving":
        selected_best, budget_used, num_arms_left = prior_guided_successive_halving(
            arms=arms,
            budget_N=len(arms) * np.log2(len(arms)),
            hb_bracket=None,
            **shared_kwargs,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer!r}. Choose 'successive_halving' or 'hyperband'.")

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
    # pyexp.fill_table_from_config()
    pyexp.execute(run_experiment, max_experiments=20, random_order=True)

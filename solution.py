from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


N = 10
T = 100
W0 = 1000.0
PENALTY = 100.0
SEED = 20260518


@dataclass(frozen=True)
class PolicyConfig:
    horizon: int
    terminal_penalty: float
    max_abs_imbalance: int | None = None


@dataclass
class SimulationSummary:
    name: str
    mean_final: float
    std_final: float
    removal_probability: float
    samples: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "mean_final": self.mean_final,
            "std_final": self.std_final,
            "removal_probability": self.removal_probability,
            "samples": self.samples,
        }


def plan_window_no_cap(rho_window: np.ndarray, start_imbalance: int, terminal_penalty: float) -> list[int]:
    horizon = len(rho_window)
    suffix = np.cumsum(rho_window[::-1])[::-1]
    width = 2 * horizon + 1
    offset = horizon
    next_values = np.full(width, -1e18, dtype=float)
    next_values[:] = [-terminal_penalty * abs(start_imbalance + delta) for delta in range(-horizon, horizon + 1)]
    actions = np.zeros((horizon, width), dtype=np.int8)

    for day in range(horizon - 1, -1, -1):
        values = np.full(width, -1e18, dtype=float)
        max_abs_delta = day
        for delta in range(-max_abs_delta, max_abs_delta + 1):
            best_value = -1e18
            best_action = 0
            for action in (-1, 0, 1):
                next_delta = delta + action
                idx = next_delta + offset
                value = action * suffix[day] + next_values[idx]
                if value > best_value:
                    best_value = value
                    best_action = action
            values[delta + offset] = best_value
            actions[day, delta + offset] = best_action
        next_values = values

    plan: list[int] = []
    delta = 0
    for day in range(horizon):
        action = int(actions[day, delta + offset])
        plan.append(action)
        delta += action
    return plan


def plan_window_capped(
    rho_window: np.ndarray,
    start_imbalance: int,
    terminal_penalty: float,
    max_abs_imbalance: int,
) -> list[int]:
    cap = max_abs_imbalance
    horizon = len(rho_window)
    width = 2 * cap + 1
    offset = cap
    next_values = np.array([-terminal_penalty * abs(x) for x in range(-cap, cap + 1)], dtype=float)
    actions = np.zeros((horizon, width), dtype=np.int8)

    for day in range(horizon - 1, -1, -1):
        values = np.full(width, -1e18, dtype=float)
        for imbalance in range(-cap, cap + 1):
            best_value = -1e18
            best_action = 0
            for action in (-1, 0, 1):
                next_imbalance = imbalance + action
                if abs(next_imbalance) > cap:
                    continue
                value = next_imbalance * rho_window[day] + next_values[next_imbalance + offset]
                if value > best_value:
                    best_value = value
                    best_action = action
            values[imbalance + offset] = best_value
            actions[day, imbalance + offset] = best_action
        next_values = values

    plan: list[int] = []
    imbalance = start_imbalance
    for day in range(horizon):
        action = int(actions[day, imbalance + offset])
        plan.append(action)
        imbalance += action
    return plan


def plan_window(rho_window: np.ndarray, start_imbalance: int, config: PolicyConfig) -> list[int]:
    if config.max_abs_imbalance is None:
        return plan_window_no_cap(rho_window, start_imbalance, config.terminal_penalty)
    return plan_window_capped(rho_window, start_imbalance, config.terminal_penalty, config.max_abs_imbalance)


def simulate_policy(
    rho_paths: np.ndarray,
    config: PolicyConfig,
    *,
    block_update: bool,
    w0: float = W0,
    end_penalty: float = PENALTY,
) -> tuple[np.ndarray, np.ndarray]:
    path_count, total_days = rho_paths.shape
    finals = np.empty(path_count, dtype=float)
    removed = np.zeros(path_count, dtype=bool)

    for path_idx in range(path_count):
        w = w0
        imbalance = 0

        if block_update:
            for block_start in range(0, total_days, config.horizon):
                horizon = min(config.horizon, total_days - block_start)
                rho_window = rho_paths[path_idx, block_start : block_start + horizon]
                plan = plan_window(rho_window, imbalance, config)
                for offset, action in enumerate(plan):
                    imbalance += action
                    w += imbalance * rho_window[offset]
                    if w < 0:
                        removed[path_idx] = True
                        finals[path_idx] = w
                        break
                if removed[path_idx]:
                    break
        else:
            for day in range(total_days):
                horizon = min(config.horizon, total_days - day)
                rho_window = rho_paths[path_idx, day : day + horizon]
                action = plan_window(rho_window, imbalance, config)[0]
                imbalance += action
                w += imbalance * rho_paths[path_idx, day]
                if w < 0:
                    removed[path_idx] = True
                    finals[path_idx] = w
                    break

        if not removed[path_idx]:
            finals[path_idx] = w - end_penalty * abs(imbalance)

    return finals, removed


def summarize(name: str, finals: np.ndarray, removed: np.ndarray) -> SimulationSummary:
    return SimulationSummary(
        name=name,
        mean_final=float(finals.mean()),
        std_final=float(finals.std(ddof=1)),
        removal_probability=float(removed.mean()),
        samples=int(len(finals)),
    )


def estimate(
    rng: np.random.Generator,
    config: PolicyConfig,
    *,
    block_update: bool,
    sample_count: int,
    name: str,
) -> SimulationSummary:
    rho_paths = rng.normal(size=(sample_count, T))
    finals, removed = simulate_policy(rho_paths, config, block_update=block_update)
    return summarize(name, finals, removed)


def choose_best_penalty(rng: np.random.Generator, penalty_grid: list[float], sample_count: int) -> tuple[PolicyConfig, list[dict[str, float]]]:
    rho_paths = rng.normal(size=(sample_count, T))
    best_config: PolicyConfig | None = None
    best_mean = -1e18
    table: list[dict[str, float]] = []
    for penalty in penalty_grid:
        config = PolicyConfig(horizon=N, terminal_penalty=penalty, max_abs_imbalance=None)
        finals, removed = simulate_policy(rho_paths, config, block_update=True)
        summary = summarize(f"penalty_{penalty:g}", finals, removed)
        table.append(summary.to_dict())
        if summary.mean_final > best_mean:
            best_mean = summary.mean_final
            best_config = config
    assert best_config is not None
    return best_config, table


def choose_cautious_cap(
    rng: np.random.Generator,
    base_penalty: float,
    caps: list[int],
    sample_count: int,
    risk_limit: float = 0.001,
) -> tuple[PolicyConfig, list[dict[str, float]]]:
    rho_paths = rng.normal(size=(sample_count, T))
    table: list[dict[str, float]] = []
    fallback: PolicyConfig | None = None
    fallback_risk = 1e18

    for cap in caps:
        config = PolicyConfig(horizon=N, terminal_penalty=base_penalty, max_abs_imbalance=cap)
        finals, removed = simulate_policy(rho_paths, config, block_update=True)
        summary = summarize(f"cap_{cap}", finals, removed)
        table.append(summary.to_dict())
        if summary.removal_probability < fallback_risk:
            fallback_risk = summary.removal_probability
            fallback = config
        if summary.removal_probability <= risk_limit:
            return config, table

    assert fallback is not None
    return fallback, table


def main() -> None:
    tune_rng = np.random.default_rng(SEED)
    eval_rng = np.random.default_rng(SEED + 1)

    penalty_grid = [0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0]
    aggressive_config, penalty_table = choose_best_penalty(tune_rng, penalty_grid, sample_count=2000)
    cautious_config, cap_table = choose_cautious_cap(
        tune_rng,
        base_penalty=aggressive_config.terminal_penalty,
        caps=list(range(1, 9)),
        sample_count=3000,
        risk_limit=0.001,
    )

    aggressive_eval = estimate(eval_rng, aggressive_config, block_update=True, sample_count=5000, name="block_strategy")
    cautious_eval = estimate(eval_rng, cautious_config, block_update=True, sample_count=5000, name="cautious_strategy")
    daily_eval = estimate(eval_rng, aggressive_config, block_update=False, sample_count=1000, name="daily_refresh_strategy")

    result = {
        "parameters": {"N": N, "T": T, "W0": W0, "penalty": PENALTY, "seed": SEED},
        "selected_configs": {
            "aggressive": {
                "terminal_penalty": aggressive_config.terminal_penalty,
                "max_abs_imbalance": aggressive_config.max_abs_imbalance,
            },
            "cautious": {
                "terminal_penalty": cautious_config.terminal_penalty,
                "max_abs_imbalance": cautious_config.max_abs_imbalance,
            },
        },
        "tuning_tables": {
            "terminal_penalty_grid": penalty_table,
            "cautious_caps": cap_table,
        },
        "evaluation": {
            "block_strategy": aggressive_eval.to_dict(),
            "cautious_strategy": cautious_eval.to_dict(),
            "daily_refresh_strategy": daily_eval.to_dict(),
        },
    }

    output_dir = Path(__file__).resolve().parent
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["selected_configs"], indent=2))
    for key, value in result["evaluation"].items():
        print(
            f"{key}: mean={value['mean_final']:.3f}, "
            f"std={value['std_final']:.3f}, "
            f"removal_probability={value['removal_probability']:.6f}, "
            f"samples={value['samples']}"
        )


if __name__ == "__main__":
    main()

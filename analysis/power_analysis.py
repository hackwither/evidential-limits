#!/usr/bin/env python3
"""Power analysis reported in Figure 2 and the main text.

The simulation matches the manuscript: 8,000 Monte Carlo iterations for each
benchmark/harm-rate cell, a one-sided pooled two-sample proportions z-test at
alpha=0.05, and a 50% relative reduction in harm rate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


RNG_SEED = 42
N_ITER = 8_000
ALPHA = 0.05
TARGET_POWER = 0.80
IMPROVEMENT_FACTOR = 0.50

BENCHMARKS: dict[str, tuple[int, str]] = {
    "AdvBench": (520, "Zou et al. 2023"),
    "HarmBench": (400, "Mazeika et al. 2024"),
    "StrongREJECT": (313, "Souly et al. 2024"),
    "WildGuard": (1_725, "Han et al. 2024"),
    "SORRY-Bench": (440, "Xie et al. 2024"),
    "XSTest": (250, "Röttger et al. 2023"),
    "SafetyBench": (2_100, "Zhang et al. 2023"),
    "HEx-PHI": (330, "Qi et al. 2023"),
}

HARM_RATES = np.array(
    [
        0.0001,
        0.0003,
        0.0005,
        0.001,
        0.002,
        0.003,
        0.005,
        0.01,
        0.02,
        0.03,
        0.05,
        0.08,
        0.10,
    ],
    dtype=float,
)


def estimate_power(
    n: int,
    p_unsafe: float,
    improvement: float = IMPROVEMENT_FACTOR,
    n_iter: int = N_ITER,
    seed: int = RNG_SEED,
) -> dict[str, float]:
    """Estimate power for a one-sided pooled two-sample proportions z-test."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 < p_unsafe < 1:
        raise ValueError("p_unsafe must lie in (0, 1)")
    if not 0 < improvement < 1:
        raise ValueError("improvement must lie in (0, 1)")

    rng = np.random.default_rng(seed)
    p_safe = p_unsafe * (1.0 - improvement)
    k_unsafe = rng.binomial(n, p_unsafe, size=n_iter)
    k_safe = rng.binomial(n, p_safe, size=n_iter)

    p_hat_unsafe = k_unsafe / n
    p_hat_safe = k_safe / n
    p_pool = (k_unsafe + k_safe) / (2 * n)
    denominator = np.sqrt(p_pool * (1.0 - p_pool) * 2.0 / n)

    with np.errstate(invalid="ignore", divide="ignore"):
        z_stat = np.where(
            denominator > 0,
            (p_hat_unsafe - p_hat_safe) / denominator,
            0.0,
        )

    power = float(np.mean(z_stat > stats.norm.ppf(1.0 - ALPHA)))
    standard_error = float(np.sqrt(power * (1.0 - power) / n_iter))
    return {
        "power": power,
        "power_se": standard_error,
        "power_ci_lo": max(0.0, power - 1.96 * standard_error),
        "power_ci_hi": min(1.0, power + 1.96 * standard_error),
        "p_safe": p_safe,
    }


def required_n_for_power(
    p_unsafe: float,
    improvement: float = IMPROVEMENT_FACTOR,
    target_power: float = TARGET_POWER,
    alpha: float = ALPHA,
) -> int:
    """Normal-approximation sample size per group for the target power."""
    p_safe = p_unsafe * (1.0 - improvement)
    p_bar = (p_unsafe + p_safe) / 2.0
    z_alpha = stats.norm.ppf(1.0 - alpha)
    z_beta = stats.norm.ppf(target_power)
    numerator = (
        z_alpha * np.sqrt(2.0 * p_bar * (1.0 - p_bar))
        + z_beta
        * np.sqrt(
            p_unsafe * (1.0 - p_unsafe) + p_safe * (1.0 - p_safe)
        )
    ) ** 2
    denominator = (p_unsafe - p_safe) ** 2
    return int(np.ceil(numerator / denominator))


def run_simulation(n_iter: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for benchmark, (n, citation) in BENCHMARKS.items():
        for p_unsafe in HARM_RATES:
            result = estimate_power(
                n=n,
                p_unsafe=float(p_unsafe),
                n_iter=n_iter,
                seed=seed,
            )
            rows.append(
                {
                    "benchmark": benchmark,
                    "n": n,
                    "citation": citation,
                    "p_unsafe": float(p_unsafe),
                    "p_safe": result["p_safe"],
                    "power": result["power"],
                    "power_se": result["power_se"],
                    "power_ci_lo": result["power_ci_lo"],
                    "power_ci_hi": result["power_ci_hi"],
                }
            )
    return pd.DataFrame(rows)


def required_n_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "p_unsafe": HARM_RATES,
            "p_safe": HARM_RATES * (1.0 - IMPROVEMENT_FACTOR),
            "required_n_80pct": [
                required_n_for_power(float(p)) for p in HARM_RATES
            ],
        }
    )


def plot_power_curves(results: pd.DataFrame, output_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for benchmark, group in results.groupby("benchmark", sort=False):
        group = group.sort_values("p_unsafe")
        ax.semilogx(
            group["p_unsafe"],
            group["power"],
            marker="o",
            markersize=3,
            linewidth=1.7,
            label=f"{benchmark} (n={BENCHMARKS[benchmark][0]:,})",
        )
    ax.axhline(TARGET_POWER, color="black", linestyle=":", linewidth=1.2)
    ax.axvline(0.01, color="#555555", linestyle="--", linewidth=1)
    ax.axvline(0.08, color="#555555", linestyle="--", linewidth=1)
    ax.text(0.01, 0.03, "frontier\np=0.01", ha="center", fontsize=8)
    ax.text(0.08, 0.03, "high-frequency\np=0.08", ha="center", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Unsafe-model harm rate ($p_{unsafe}$)")
    ax.set_ylabel("Statistical power")
    ax.set_title("Power to detect a 50% reduction in harm rate")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_required_n(output_stem: Path) -> None:
    p_values = np.logspace(-5, -1, 250)
    n_values = np.array([required_n_for_power(float(p)) for p in p_values])

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.loglog(p_values, n_values, linewidth=2.2)
    ax.axhline(max(n for n, _ in BENCHMARKS.values()), linestyle="--", linewidth=1.2)
    ax.text(
        p_values[2],
        max(n for n, _ in BENCHMARKS.values()) * 1.12,
        "Largest reviewed public benchmark (n=2,100)",
        fontsize=8,
    )
    ax.set_xlabel("Unsafe-model harm rate (p_unsafe)")
    ax.set_ylabel("Prompts per model required for 80% power")
    ax.set_title("Sample size required to detect a 50% harm-rate reduction")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def print_key_results(results: pd.DataFrame) -> None:
    frontier = results[np.isclose(results["p_unsafe"], 0.01)].sort_values("power")
    print("Power at p_unsafe=0.01 and p_safe=0.005:")
    for row in frontier.itertuples(index=False):
        print(f"  {row.benchmark:15s} n={row.n:>5,}: {row.power:.1%}")
    print("\nRequired n per model for 80% power:")
    for p in (0.01, 0.001, 0.0001):
        print(f"  p_unsafe={p:g}: n={required_n_for_power(p):,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--iterations", type=int, default=N_ITER)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = run_simulation(n_iter=args.iterations, seed=args.seed)
    required = required_n_table()

    results.to_csv(args.output_dir / "power_results.csv", index=False)
    required.to_csv(args.output_dir / "required_n_80pct.csv", index=False)
    plot_power_curves(results, args.output_dir / "figure2_statistical_power")
    print_key_results(results)


if __name__ == "__main__":
    main()

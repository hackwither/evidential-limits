#!/usr/bin/env python3
"""False-certification and zero-event confidence-bound calculations.

This script contains only the rare-event calculations reported in the paper:
P(k=0 | n,p), sample sizes required to reduce that probability below a chosen
tolerance, the exact zero-event Clopper-Pearson upper bound, and Table 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABLE1_HARM_RATES = (1e-2, 1e-3, 1e-4, 1e-5)
TABLE1_SAMPLE_SIZES = (250, 520, 2_100, 5_000, 10_000, 100_000)
REPRESENTATIVE_BENCHMARKS = {
    "XSTest": 250,
    "StrongREJECT": 313,
    "HarmBench": 400,
    "AdvBench": 520,
    "SafetyBench": 2_100,
}


def false_certification_probability(n: int | np.ndarray, p: float) -> np.ndarray:
    """Return P(k=0 | n,p)=(1-p)^n for independent Bernoulli trials."""
    if not 0 < p < 1:
        raise ValueError("p must lie in (0, 1)")
    n_array = np.asarray(n, dtype=float)
    if np.any(n_array < 0):
        raise ValueError("n must be non-negative")
    return np.power(1.0 - p, n_array)


def required_n_for_false_certification(p: float, tolerance: float) -> int:
    """Smallest integer n for which P(k=0 | n,p) <= tolerance."""
    if not 0 < tolerance < 1:
        raise ValueError("tolerance must lie in (0, 1)")
    return int(np.ceil(np.log(tolerance) / np.log1p(-p)))


def zero_event_clopper_pearson_upper(
    n: int,
    confidence: float = 0.95,
    two_sided: bool = True,
) -> float:
    """Exact upper confidence bound after observing zero events in n trials."""
    if n <= 0:
        raise ValueError("n must be positive")
    alpha = 1.0 - confidence
    tail_alpha = alpha / 2.0 if two_sided else alpha
    return float(1.0 - tail_alpha ** (1.0 / n))


def make_table1() -> pd.DataFrame:
    rows = []
    for p in TABLE1_HARM_RATES:
        row: dict[str, float | str] = {"harm_rate": p}
        for n in TABLE1_SAMPLE_SIZES:
            row[f"n_{n}"] = 100.0 * float(false_certification_probability(n, p))
        rows.append(row)
    return pd.DataFrame(rows)


def make_required_n_table() -> pd.DataFrame:
    rows = []
    for p in TABLE1_HARM_RATES:
        rows.append(
            {
                "harm_rate": p,
                "n_for_5pct": required_n_for_false_certification(p, 0.05),
                "n_for_1pct": required_n_for_false_certification(p, 0.01),
            }
        )
    return pd.DataFrame(rows)


def plot_false_certification(output_stem: Path) -> None:
    n_values = np.unique(np.logspace(np.log10(30), 6, 400).astype(int))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for p in TABLE1_HARM_RATES:
        probabilities = false_certification_probability(n_values, p)
        ax.semilogx(n_values, probabilities, linewidth=2.2, label=f"p={p:g}")

    for name, n in REPRESENTATIVE_BENCHMARKS.items():
        ax.axvline(n, linewidth=0.8, alpha=0.35)
        ax.text(n, 0.86, name, rotation=90, va="top", ha="right", fontsize=7)

    ax.axhline(0.05, linestyle="--", linewidth=1.4, label="5% tolerance")
    ax.set_xlim(30, 1_000_000)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Evaluation size (n; log scale)")
    ax.set_ylabel("P(observe zero harms | true harm rate p)")
    ax.set_title("False-certification probability after a zero-harm evaluation")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def print_key_results() -> None:
    for n in (313, 520):
        probability = float(false_certification_probability(n, 0.001))
        print(f"p=0.001, n={n}: P(k=0)={probability:.1%}")

    upper = zero_event_clopper_pearson_upper(520, confidence=0.95, two_sided=True)
    print(
        "Zero events in n=520: two-sided 95% Clopper-Pearson upper bound "
        f"={upper:.6f} (approximately 1 in {1 / upper:.1f})"
    )

    for p in (0.001, 0.0001, 0.00001):
        required = required_n_for_false_certification(p, 0.05)
        print(f"p={p:g}: n={required:,} for P(k=0)<=5%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    make_table1().to_csv(
        args.output_dir / "table1_false_certification_percent.csv", index=False
    )
    make_required_n_table().to_csv(
        args.output_dir / "false_certification_required_n.csv", index=False
    )
    plot_false_certification(args.output_dir / "figure1b_false_certification")
    print_key_results()


if __name__ == "__main__":
    main()

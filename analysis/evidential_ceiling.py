#!/usr/bin/env python3
"""Evidential-ceiling calculations reported in Figure 1 and Tables 1, 2, and 4."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IMPROVEMENT_RATIO = 0.5
ADVBENCH_N = 520
FEASIBILITY_CEILING = 100_000
TABLE_HARM_RATES = (1e-2, 5e-3, 1e-3, 1e-4, 1e-5)
CLAIMS_HARM_RATES = (1e-2, 1e-3, 1e-4, 1e-5)


def null_likelihood_ratio(
    n: int | np.ndarray,
    p_unsafe: float | np.ndarray,
    ratio: float = IMPROVEMENT_RATIO,
) -> np.ndarray:
    """Return P(k=0|H1) / P(k=0|H0)."""
    if not 0 < ratio < 1:
        raise ValueError("ratio must lie in (0, 1)")
    p = np.asarray(p_unsafe, dtype=float)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("p_unsafe must lie in (0, 1)")
    return np.power((1.0 - p) / (1.0 - ratio * p), n)


def one_harm_likelihood_ratio(
    n: int,
    p_unsafe: float | np.ndarray,
    ratio: float = IMPROVEMENT_RATIO,
) -> np.ndarray:
    """Return P(k=1|H1) / P(k=1|H0)."""
    p = np.asarray(p_unsafe, dtype=float)
    return (1.0 / ratio) * null_likelihood_ratio(n - 1, p, ratio)


def evidence_bits(likelihood_ratio: float | np.ndarray) -> np.ndarray:
    values = np.asarray(likelihood_ratio, dtype=float)
    return np.abs(np.log2(np.maximum(values, np.finfo(float).tiny)))


def required_n_for_likelihood_ratio(
    p_unsafe: float,
    threshold: float,
    ratio: float = IMPROVEMENT_RATIO,
) -> int:
    """Smallest n for which a null result has likelihood ratio <= threshold."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie in (0, 1)")
    denominator = np.log((1.0 - p_unsafe) / (1.0 - ratio * p_unsafe))
    return int(np.ceil(np.log(threshold) / denominator))


def displayed_n_for_likelihood_ratio(
    p_unsafe: float,
    threshold: float,
    ratio: float = IMPROVEMENT_RATIO,
) -> int:
    """Nearest-integer convention used by the final paper's Table 4."""
    denominator = np.log((1.0 - p_unsafe) / (1.0 - ratio * p_unsafe))
    return int(np.floor(np.log(threshold) / denominator + 0.5))


def boundary_rate(
    threshold: float,
    n_max: int | np.ndarray,
    ratio: float = IMPROVEMENT_RATIO,
) -> np.ndarray:
    """Exact p_min at which n_max trials attain the evidentiary threshold."""
    n = np.asarray(n_max, dtype=float)
    root = np.power(threshold, 1.0 / n)
    return (1.0 - root) / (1.0 - ratio * root)


def elicitation_bits(n: int, q1: float, q0: float) -> float:
    """Evidence in a null result from a general elicitation procedure."""
    if not 0 <= q0 < 1 or not 0 <= q1 < 1:
        raise ValueError("elicitation rates must lie in [0, 1)")
    ratio = ((1.0 - q1) / (1.0 - q0)) ** n
    return float(evidence_bits(ratio))


def make_table1() -> pd.DataFrame:
    rows = []
    for p in TABLE_HARM_RATES:
        clean = float(evidence_bits(null_likelihood_ratio(ADVBENCH_N, p)))
        one_harm = float(evidence_bits(one_harm_likelihood_ratio(ADVBENCH_N, p)))
        rows.append(
            {
                "p_unsafe": p,
                "clean_sheet_bits": clean,
                "one_harm_bits": one_harm,
                "harm_to_clean_ratio": one_harm / clean,
            }
        )
    return pd.DataFrame(rows)


def make_table2() -> pd.DataFrame:
    rows = [
        ("Passive benchmark, AdvBench-sized", "prompt", 520, 1e-3, 5e-4),
        ("Passive benchmark, at N_max", "prompt", 100_000, 1e-3, 5e-4),
        ("Adaptive campaign, discriminating", "campaign", 1, 0.90, 0.10),
        ("Adaptive campaign, discriminating", "campaign", 5, 0.90, 0.10),
        ("Adaptive campaign, non-discriminating", "campaign", 5, 0.90, 0.85),
    ]
    return pd.DataFrame(
        [
            {
                "procedure": procedure,
                "trial_unit": unit,
                "n": n,
                "q1": q1,
                "q0": q0,
                "bits": elicitation_bits(n, q1, q0),
            }
            for procedure, unit, n, q1, q0 in rows
        ]
    )


def make_claims_ladder() -> pd.DataFrame:
    claims: list[tuple[str, float | None]] = [
        ("Harm rate bounded above at 95% confidence", None),
        ("Belief shifts toward safety by 2x", 0.5),
        ("False certification held below 5%", -0.05),
        ("Belief shifts toward safety by 10x", 0.1),
        ("Belief shifts toward safety by 100x", 0.01),
    ]
    rows: list[dict[str, int | str]] = []
    for claim, threshold in claims:
        row: dict[str, int | str] = {"claim": claim}
        for p in CLAIMS_HARM_RATES:
            column = f"p_{p:g}"
            if threshold is None:
                row[column] = "any n"
            elif threshold < 0:
                exact = np.log(-threshold) / np.log1p(-p)
                row[column] = int(np.floor(exact + 0.5))
            else:
                row[column] = displayed_n_for_likelihood_ratio(p, threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_figure1(output_stem: Path) -> None:
    p_values = np.logspace(-5, -1, 500)
    sample_sizes = (520, 2_100, 100_000)
    fig, (channel_ax, evidence_ax) = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        gridspec_kw={"width_ratios": (0.85, 1.5)},
    )

    channel_ax.axis("off")
    boxes = (
        (0.16, 0.58, "Latent safety state\n$H_0$ or $H_1$"),
        (0.50, 0.58, "Observation channel\n$C$ under budget $B$"),
        (0.84, 0.58, "Result\n$k$ harms in $n$ trials"),
        (0.50, 0.15, "Scoring / aggregation\ncannot raise the ceiling"),
    )
    for x, y, label in boxes:
        channel_ax.text(
            x,
            y,
            label,
            transform=channel_ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.45", "fc": "white", "ec": "#333333"},
        )
    for start, end in ((0.28, 0.38), (0.62, 0.72)):
        channel_ax.annotate(
            "",
            xy=(end, 0.55),
            xytext=(start, 0.55),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "lw": 1.6},
        )
    channel_ax.annotate(
        "",
        xy=(0.54, 0.27),
        xytext=(0.80, 0.49),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1.2},
    )
    channel_ax.set_title("(a) The evidence channel", loc="left")

    colors = ("#2166ac", "#7b3294", "#b2182b")
    for n, color in zip(sample_sizes, colors, strict=True):
        clean = evidence_bits(null_likelihood_ratio(n, p_values))
        one_harm = evidence_bits(one_harm_likelihood_ratio(n, p_values))
        evidence_ax.loglog(
            p_values,
            clean,
            color=color,
            linewidth=2,
            label=f"Clean sheet, n={n:,}",
        )
        evidence_ax.loglog(
            p_values,
            one_harm,
            color=color,
            linewidth=1.6,
            linestyle="--",
            label=f"One harm, n={n:,}",
        )
        crossing = np.log(1 / IMPROVEMENT_RATIO) / (
            2 * n * (1 - IMPROVEMENT_RATIO)
        )
        evidence_ax.scatter(
            crossing,
            float(evidence_bits(null_likelihood_ratio(n, crossing))),
            facecolors="white",
            edgecolors=color,
            zorder=5,
        )
    evidence_ax.set_xlabel("Elevated-risk harm rate ($p_u$)")
    evidence_ax.set_ylabel("Evidence magnitude (bits)")
    evidence_ax.set_title("(b) Evidence carried by one result", loc="left")
    evidence_ax.grid(True, which="both", alpha=0.2)
    evidence_ax.legend(frameon=False, fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    make_table1().to_csv(args.output_dir / "table1_evidence_regimes.csv", index=False)
    make_table2().to_csv(
        args.output_dir / "table2_elicitation_discrimination.csv",
        index=False,
    )
    make_claims_ladder().to_csv(args.output_dir / "table4_claims_ladder.csv", index=False)
    plot_figure1(
        args.output_dir / "figure1_caption_consistent_reconstruction"
    )

    print("Table 1: evidence regimes")
    print(make_table1().to_string(index=False))
    print("\nTable 2: elicitation discrimination")
    print(make_table2().to_string(index=False))
    print("\nTable 4: claims ladder")
    print(make_claims_ladder().to_string(index=False))


if __name__ == "__main__":
    main()

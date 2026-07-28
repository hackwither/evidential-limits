#!/usr/bin/env python3
"""Distributional-gap analysis reported in the main text and Figure 4.

Compares AdvBench, HarmBench, and the deduplicated LMSYS-Chat-1M sample using
TF-IDF+SVD and all-MiniLM-L6-v2 sentence embeddings. Outputs sliced Wasserstein
distance, RBF-kernel MMD, mean cosine similarity, and the neural-embedding UMAP.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import rbf_kernel


ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)
HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_all.csv"
)
EXPECTED_ADVBENCH = 520
EXPECTED_HARMBENCH = 400
EXPECTED_LMSYS = 9_089
TFIDF_MAX_FEATURES = 20_000
SVD_DIMS = 100
N_SWD_PROJECTIONS = 500
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SEED = 42


def load_benchmark(source: str, column: str, expected_n: int) -> list[str]:
    frame = pd.read_csv(source)
    if column not in frame.columns:
        raise ValueError(f"Column {column!r} not found in {source}")
    prompts = frame[column].dropna().astype(str).tolist()
    if len(prompts) != expected_n:
        raise RuntimeError(
            f"Expected {expected_n} prompts from {source}, found {len(prompts)}. "
            "The upstream dataset may have changed."
        )
    return prompts


def load_lmsys(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    if "text" not in frame.columns:
        raise ValueError(f"Expected a 'text' column in {path}")
    prompts = (
        frame["text"]
        .dropna()
        .astype(str)
        .loc[lambda values: values.str.strip().str.len() > 0]
        .drop_duplicates()
        .tolist()
    )
    if len(prompts) != EXPECTED_LMSYS:
        raise RuntimeError(
            f"Expected the paper's {EXPECTED_LMSYS:,}-query LMSYS sample, "
            f"found {len(prompts):,}."
        )
    return prompts


def tfidf_embeddings(corpora: dict[str, list[str]]) -> dict[str, np.ndarray]:
    names = list(corpora)
    texts = [text for name in names for text in corpora[name]]
    labels = np.concatenate([[name] * len(corpora[name]) for name in names])

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=TFIDF_MAX_FEATURES,
    )
    matrix = vectorizer.fit_transform(texts)
    n_components = min(SVD_DIMS, matrix.shape[0] - 1, matrix.shape[1] - 1)
    reducer = TruncatedSVD(n_components=n_components, random_state=SEED)
    reduced = reducer.fit_transform(matrix)
    return {name: reduced[labels == name] for name in names}


def neural_embeddings(
    corpora: dict[str, list[str]],
    model_name: str,
    model_revision: str | None,
    batch_size: int,
) -> dict[str, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    model_kwargs = {"device": "cpu"}
    if model_revision:
        model_kwargs["revision"] = model_revision
    model = SentenceTransformer(model_name, **model_kwargs)
    return {
        name: model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        for name, texts in corpora.items()
    }


def sliced_wasserstein(
    x: np.ndarray,
    y: np.ndarray,
    n_projections: int = N_SWD_PROJECTIONS,
    seed: int = SEED,
) -> float:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_projections, x.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    distances = [
        wasserstein_distance(x @ direction, y @ direction)
        for direction in directions
    ]
    return float(np.mean(distances))


def _kernel_sum(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float,
    block_size: int = 1_024,
    exclude_diagonal: bool = False,
) -> float:
    total = 0.0
    same_object = x is y
    for start in range(0, len(x), block_size):
        stop = min(start + block_size, len(x))
        block = rbf_kernel(x[start:stop], y, gamma=gamma)
        total += float(block.sum())
        if exclude_diagonal and same_object:
            total -= float(np.trace(block[:, start:stop]))
    return total


def mmd_rbf(x: np.ndarray, y: np.ndarray) -> float:
    combined = np.vstack([x, y])
    median_distance = float(np.median(pdist(combined, metric="euclidean")))
    if median_distance <= 0:
        raise RuntimeError("Median pairwise distance is zero; MMD bandwidth undefined")
    gamma = 1.0 / (2.0 * median_distance**2)
    m, n = len(x), len(y)
    mmd_squared = (
        _kernel_sum(x, x, gamma, exclude_diagonal=True) / (m * (m - 1))
        + _kernel_sum(y, y, gamma, exclude_diagonal=True) / (n * (n - 1))
        - 2.0 * _kernel_sum(x, y, gamma) / (m * n)
    )
    return float(np.sqrt(max(mmd_squared, 0.0)))


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def mean_cosine_similarity(x: np.ndarray, y: np.ndarray | None = None) -> float:
    """Exact mean cosine without constructing an O(n^2) similarity matrix."""
    x_normalized = _row_normalize(x)
    if y is None:
        n = len(x_normalized)
        summed = x_normalized.sum(axis=0)
        return float((summed @ summed - n) / (n * (n - 1)))
    y_normalized = _row_normalize(y)
    return float(
        (x_normalized.sum(axis=0) @ y_normalized.sum(axis=0))
        / (len(x_normalized) * len(y_normalized))
    )


def metric_table(
    embeddings: dict[str, np.ndarray], representation: str
) -> pd.DataFrame:
    names = list(embeddings)
    rows: list[dict[str, float | str]] = []
    for index, name_a in enumerate(names):
        x = embeddings[name_a]
        rows.append(
            {
                "representation": representation,
                "pair": f"{name_a} (internal)",
                "swd": np.nan,
                "mmd": np.nan,
                "cosine_similarity": mean_cosine_similarity(x),
            }
        )
        for name_b in names[index + 1 :]:
            y = embeddings[name_b]
            rows.append(
                {
                    "representation": representation,
                    "pair": f"{name_a} vs {name_b}",
                    "swd": sliced_wasserstein(x, y),
                    "mmd": mmd_rbf(x, y),
                    "cosine_similarity": mean_cosine_similarity(x, y),
                }
            )
    return pd.DataFrame(rows)


def make_umap_figure(
    embeddings: dict[str, np.ndarray], output_stem: Path
) -> None:
    import umap

    names = list(embeddings)
    stacked = np.vstack([embeddings[name] for name in names])
    labels = np.concatenate([[name] * len(embeddings[name]) for name in names])
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=SEED,
    )
    projection = reducer.fit_transform(stacked)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name in names:
        mask = labels == name
        ax.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=6,
            alpha=0.5,
            label=f"{name} (n={int(mask.sum()):,})",
        )
    ax.set_title("Benchmark and LMSYS prompt distributions\n(all-MiniLM-L6-v2 embeddings)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmsys-path", type=Path, required=True)
    parser.add_argument("--advbench-path", default=ADVBENCH_URL)
    parser.add_argument("--harmbench-path", default=HARMBENCH_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpora = {
        "AdvBench": load_benchmark(args.advbench_path, "goal", EXPECTED_ADVBENCH),
        "HarmBench": load_benchmark(
            args.harmbench_path, "Behavior", EXPECTED_HARMBENCH
        ),
        "LMSYS-Chat-1M": load_lmsys(args.lmsys_path),
    }

    print("Computing TF-IDF+SVD representation...")
    tfidf = tfidf_embeddings(corpora)
    tfidf_metrics = metric_table(tfidf, "TF-IDF+SVD")
    tfidf_metrics.to_csv(
        args.output_dir / "distributional_metrics_tfidf.csv", index=False
    )

    print(f"Computing {args.embedding_model} embeddings on CPU...")
    neural = neural_embeddings(
        corpora,
        model_name=args.embedding_model,
        model_revision=args.model_revision,
        batch_size=args.batch_size,
    )
    neural_metrics = metric_table(neural, args.embedding_model)
    neural_metrics.to_csv(
        args.output_dir / "distributional_metrics_neural.csv", index=False
    )
    pd.concat([tfidf_metrics, neural_metrics], ignore_index=True).to_csv(
        args.output_dir / "distributional_metrics_combined.csv", index=False
    )
    make_umap_figure(neural, args.output_dir / "figure4_prompt_distributions")

    print("\nTF-IDF+SVD metrics")
    print(tfidf_metrics.to_string(index=False))
    print("\nNeural-embedding metrics")
    print(neural_metrics.to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()

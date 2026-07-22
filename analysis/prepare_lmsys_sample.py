#!/usr/bin/env python3
"""Prepare the LMSYS-Chat-1M first-turn query sample used in Figure 3.

The dataset is gated. The user must accept its terms on Hugging Face and log in
before running this script. The paper reports 10,000 sampled conversations and
9,089 deduplicated first-user-turn queries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

DATASET_NAME = "lmsys/lmsys-chat-1m"
N_SAMPLE = 10_000
SEED = 42
EXPECTED_UNIQUE = 9_089


def first_user_turn(conversation: Any) -> str | None:
    if not isinstance(conversation, list):
        return None
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", turn.get("from", ""))).lower()
        if role in {"user", "human"}:
            value = turn.get("content", turn.get("value"))
            if value is not None and str(value).strip():
                return str(value).strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/lmsys_sample.csv"))
    parser.add_argument(
        "--revision",
        default=None,
        help="Hugging Face dataset revision used for the paper. Supply it when known.",
    )
    parser.add_argument("--expected-unique", type=int, default=EXPECTED_UNIQUE)
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        revision=args.revision,
    )
    dataset = dataset.shuffle(seed=SEED)

    rows: list[str] = []
    pool_size = min(len(dataset), int(N_SAMPLE * 1.3))
    for index in range(pool_size):
        text = first_user_turn(dataset[index].get("conversation"))
        if text:
            rows.append(text)
        if len(rows) >= N_SAMPLE:
            break

    if len(rows) < N_SAMPLE:
        raise RuntimeError(
            f"Only {len(rows):,} usable first turns found; expected {N_SAMPLE:,}."
        )

    sample = pd.DataFrame({"text": rows}).drop_duplicates(subset="text")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False)

    metadata = {
        "dataset": DATASET_NAME,
        "revision": args.revision,
        "seed": SEED,
        "sampled_conversations": N_SAMPLE,
        "deduplicated_queries": int(len(sample)),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    if args.expected_unique and len(sample) != args.expected_unique:
        raise RuntimeError(
            f"Produced {len(sample):,} unique queries, but the paper reports "
            f"{args.expected_unique:,}. The dataset revision or preprocessing "
            "environment does not match the paper run."
        )

    print(f"Saved {len(sample):,} unique first-turn queries to {args.output}")


if __name__ == "__main__":
    main()

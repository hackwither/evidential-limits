#!/usr/bin/env python3
"""Run all retained analyses used in the manuscript."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmsys-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--advbench-path",
        default=None,
        help="Optional local AdvBench CSV (otherwise fetched from upstream).",
    )
    parser.add_argument(
        "--harmbench-path",
        default=None,
        help="Optional local HarmBench CSV (otherwise fetched from upstream).",
    )
    parser.add_argument("--model-revision", default=None)
    args = parser.parse_args()

    python = sys.executable
    root = Path(__file__).resolve().parent
    evidential_script = root / "analysis" / "evidential_ceiling.py"
    power_script = root / "analysis" / "power_analysis.py"
    false_cert_script = root / "analysis" / "false_certification.py"
    distribution_script = root / "analysis" / "distributional_gap.py"
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    lmsys_path = args.lmsys_path if args.lmsys_path.is_absolute() else root / args.lmsys_path

    run([python, str(evidential_script), "--output-dir", str(output_dir)])
    run([python, str(power_script), "--output-dir", str(output_dir)])
    run(
        [
            python,
            str(false_cert_script),
            "--output-dir",
            str(output_dir),
        ]
    )
    distribution_command = [
        python,
        str(distribution_script),
        "--lmsys-path",
        str(lmsys_path),
        "--output-dir",
        str(output_dir),
    ]
    if args.model_revision:
        distribution_command.extend(["--model-revision", args.model_revision])
    if args.advbench_path:
        distribution_command.extend(["--advbench-path", args.advbench_path])
    if args.harmbench_path:
        distribution_command.extend(["--harmbench-path", args.harmbench_path])
    run(distribution_command)


if __name__ == "__main__":
    main()

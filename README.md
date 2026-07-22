# Analysis for What AI Red-Team Null Results Can and Cannot Tell Us

This respository contains code for the empirical sections of *Not Yet Caught Is Not Safe: What AI Red-Team Null Results Can and Cannot Tell Us*. 
The paper's central claim is statistical: a red-team evaluation that observes zero harmful outputs has not shown a model safe, and for sufficiently rare harms no feasible passive benchmark can. These studies quantify how far current practice sits from the
sample sizes that claim would require.

## What is here

| Script | Paper element | One-line result |
| :--- | :--- | :--- |
| `analysis/power_analysis.py` | Statistical power / Figure 1 (left), Table 2 | Benchmarks at 200–5,000 prompts sit well below 80% power to detect a 50% harm-rate reduction at frontier operating points. |
| `analysis/distribution_gap.py` | Distributional validity (Condition 2) | Benchmark prompts occupy a narrow sub-region of query space relative to a benign deployment proxy, measured by sliced-Wasserstein / MMD distance and by discriminative vocabulary. |
| `analysis/false_certification.py` | False-certification / Figure 1 (center) | The probability of observing zero harms at a genuinely non-zero rare harm rate stays high at benchmark scale; certifying a tail rate needs far more data than estimating a mean. |

`run_all.py` executes all five in sequence and prints a pass/fail summary.

## Install

```bash
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
```

Python 3.11 or newer is recommended. The dependency set is deliberately small. No GPU, model download, or API key is required.`

## License

Code released under the MIT License (see `LICENSE`). The bundled datasets keep their
original licenses.

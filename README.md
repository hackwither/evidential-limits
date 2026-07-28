# What AI Red-Team Evaluations Can and Cannot Prove

Reproducibility code for:

> Bandana Kaur. *What AI Red-Team Evaluations Can and Cannot Prove*.
> arXiv:2607.21735v1, 2026.

[Read the paper on arXiv](https://arxiv.org/abs/2607.21735v1).

The paper defines an evaluation's **evidential ceiling**: the largest factor by
which one feasible result can move belief between acceptable-risk and
elevated-risk hypotheses. For a benchmark null result, the boundary is
available in closed form. Existing benchmarks can support strong claims for
high-frequency harms, but cannot certify sufficiently rare harms at feasible
sample sizes.

<p align="center">
  <img
    src="./docs/assets/feasibility_boundary.png"
    alt="Feasibility boundary for null-result certification"
    width="760"
  />
</p>

*Figure 3b. The boundary between feasible and infeasible null-result
certification. Cropped from the paper without altering the panel.*

Each curve gives the smallest harm rate <i>p</i><sub>min</sub> that a benchmark
can meaningfully certify for a maximum feasible sample size
<i>N</i><sub>max</sub>. The threshold τ is the required likelihood-ratio
threshold, smaller values demand stronger evidence and <i>r</i> is the
acceptable-to-elevated harm-rate ratio.
Points above a curve can be certified with some benchmark no larger than
<i>N</i><sub>max</sub>; points in the shaded region below it cannot. The
highlighted example shows that with <i>N</i><sub>max</sub> = 100,000, τ = 0.5,
and <i>r</i> = 0.5, the boundary is approximately
<i>p</i><sub>min</sub> = 1.4 × 10<sup>−5</sup>.

## Repository layout

```text
analysis/
  evidential_ceiling.py  Figure 1 and Tables 1, 2, and 4
  power_analysis.py      Figure 2 and Monte Carlo power results
  false_certification.py Figure 3 and Table 3
  distributional_gap.py Figure 4 and distributional-distance tables
  prepare_lmsys_sample.py
tests/                   Unit tests for the reported calculations
docs/assets/             Selected publication assets used in documentation
run_all.py               Complete reproduction entry point
requirements.txt         Paper software environment
data/                    Local source data; not version-controlled
outputs/                 Generated figures and tables; not version-controlled
```

## Paper artifacts

| Paper artifact | Generated output |
| :--- | :--- |
| Figure 1 caption-consistent reconstruction¹ | `figure1_caption_consistent_reconstruction.{png,pdf}` |
| Figure 2, statistical power across eight suites | `figure2_statistical_power.{png,pdf}` |
| Figure 3, false certification and feasibility boundary | `figure3_false_certification_boundary.{png,pdf}` |
| Figure 4, prompt-distribution UMAP | `figure4_prompt_distributions.{png,pdf}` |
| Table 1, clean sheet versus one harm | `table1_evidence_regimes.csv` |
| Table 2, elicitation discrimination | `table2_elicitation_discrimination.csv` |
| Table 3, false-certification surface | `table3_false_certification_percent.csv` |
| Table 4, claims ladder | `table4_claims_ladder.csv` |

Additional CSV files contain the full Monte Carlo surface, required sample
sizes, and distributional metrics.

¹ The Figure 1 image embedded in arXiv v1 compares null-result evidence with
positive-finding lines parameterized by a false-positive rate, while its caption
describes a clean sheet and one observed harm scored against the same
hypothesis pair. Section 4.4 identifies the former comparison as invalid. The
generated reconstruction follows the caption and equations rather than the
embedded image.

Table 4 is emitted with the nearest-integer convention used in the published
table. The reusable sample-size functions return conservative integer ceilings;
at a few operating points these are one trial larger than the displayed paper
value.

## Install

Python 3.12 matches the paper environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The distributional analysis runs sentence-transformer inference on CPU and
downloads `all-MiniLM-L6-v2` on first use. The other analyses need no model or
API access.

## Prepare the gated LMSYS sample

Accept the LMSYS-Chat-1M terms on Hugging Face and authenticate locally, then
run:

```bash
python analysis/prepare_lmsys_sample.py \
  --output data/lmsys_sample.csv \
  --revision <paper-dataset-revision>
```

The script samples 10,000 conversations with seed 42 and checks for the 9,089
deduplicated first-user-turn queries reported in the paper.

## Reproduce all results

```bash
python run_all.py \
  --lmsys-path data/lmsys_sample.csv \
  --model-revision <paper-model-revision>
```

AdvBench and HarmBench are fetched from their upstream repositories by
default. To make the run entirely local:

```bash
python run_all.py \
  --lmsys-path data/lmsys_sample.csv \
  --advbench-path data/advbench_harmful_behaviors.csv \
  --harmbench-path data/harmbench_behaviors_text_all.csv
```

Results are written to `outputs/`. Each analysis can also be run independently
and documents its options under `--help`.

## Verify

```bash
python -m unittest discover -s tests -v
```

## Data and license

AdvBench, HarmBench, and LMSYS-Chat-1M retain their original licenses. No
proprietary data is used. Repository code is released under the [MIT
License](LICENSE).

# Calculator URL parameters

`https://hackwither.co.in/evidential-ceiling/?<parameters>` restores a calculator state exactly.
These names are a public API: they are never renamed or given a new meaning. New parameters may be
added. The embed card at `/evidential-ceiling/embed/` takes the same parameters.

| Name | Meaning | Values | Default | Read in |
|---|---|---|---|---|
| `mode` | which tab | `benchmark`, `audit`, `plan` | `benchmark` | all |
| `suite` | audited suite id (frozen) | `advbench`, `harmbench`, `strongreject`, `wildguard`, `sorry-bench`, `xstest`, `safetybench`, `hex-phi` | `advbench` | benchmark |
| `n` | trials run | integer 1 to 10¹⁵ | `520` | audit |
| `k` | harms observed | integer 0 to n | `0` | audit |
| `proc` | trial model | `passive` (benchmark prompts) or `elicit` (q₁, q₀ procedure, §4.5) | `passive` | audit |
| `pu` | p_u, harm rate under elevated risk | 0 < p_u < 1 | `0.001` | all (audit when `proc=passive`) |
| `r` | improvement ratio, p_s = r·p_u | 0 < r < 1 | `0.5` | all (audit when `proc=passive`) |
| `tau` | evidentiary threshold τ | 0 < τ < 1 | `0.1` | all |
| `nmax` | budget N_max | integer 1 to 10¹⁵ | `100000` | plan |
| `prior` | π, prior probability of elevated risk | 0 < π < 1; omit for none | none | audit |
| `m` | mean cluster size | ≥ 1 (1 = independent) | `1` | all |
| `rho` | intra-cluster correlation ρ | 0 to 1 | `0` | all |
| `q1` | P(procedure elicits harm \| elevated risk) | 0 ≤ q₁ < 1 | `0.9` | audit, `proc=elicit` |
| `q0` | P(procedure elicits harm \| acceptable risk) | 0 ≤ q₀ < 1 | `0.1` | audit, `proc=elicit` |

Numbers accept `1e-5`, `0.00001`, `10^-5`, `5x10^-3`, `1/1000` and `0.1%`. The calculator writes every
parameter its mode reads (so a link means the same thing if defaults change) and writes small numbers
as `5e-4`. Invalid values fall back to the default and are reported on the page.

Examples:

```
/evidential-ceiling/?mode=audit&n=520&k=0&proc=passive&pu=0.01&r=0.5&tau=0.1&m=1&rho=0
/evidential-ceiling/?mode=plan&pu=1e-5&r=0.5&tau=0.01&nmax=100000&m=1&rho=0
/evidential-ceiling/?mode=audit&n=5&k=0&proc=elicit&q1=0.9&q0=0.1&tau=0.1&m=1&rho=0
```

Implementation and round-trip test: `calculator/state.js`, `calculator/ceiling.test.js`.

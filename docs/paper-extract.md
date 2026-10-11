# Paper extract: what the calculator implements

Source of truth: Bandana Kaur, *What AI Red-Team Evaluations Can and Cannot Prove*,
arXiv:2607.21735v2 (30 July 2026), doi:10.48550/arXiv.2607.21735. Section and table numbers
below are the paper's. Every number in "Worked examples" is a vector in
[`../test-vectors.json`](../test-vectors.json), and both `calculator/` (JavaScript) and `py/`
(Python) must reproduce it.

## Notation and definitions

| Paper | Code (JS / Python) | Meaning | Valid range |
|---|---|---|---|
| H₀, H₁ | — | acceptable risk, elevated risk | — |
| p_u | `pu` | harm rate under H₁ (elevated risk) | 0 < p_u < 1 |
| r | `r` | improvement ratio, p_s = r·p_u ("a fifty percent reduction" is r = 0.5) | 0 < r < 1 |
| p_s | `ps` | harm rate under H₀ (acceptable risk) | derived |
| n, k | `n`, `k` | trials, observed harms | n ≥ 1 integer, 0 ≤ k ≤ n |
| Λ(r) = P(r \| H₁)/P(r \| H₀) | `lnLR` (log) | likelihood ratio of a result; Λ < 1 argues toward acceptable risk | > 0 |
| Λ₀, Λ₁ | `logLR(n, 0, …)`, `logLR(n, 1, …)` | Λ for a clean sheet, for one observed harm | |
| O₀ = P(H₁)/P(H₀), O₁ = O₀Λ | `posterior()` | prior and posterior odds (the UI takes a probability π) | π ∈ (0, 1) |
| τ | `tau` | evidentiary threshold; a clean sheet certifies when Λ₀ ≤ τ | 0 < τ < 1 |
| L = C_FN/C_FP | `tauFromLoss(L, O0)` | loss ratio; τ = 1/(L·O₀) (§4.6) | L > 0 |
| N_max | `nmax` | feasibility ceiling on approximately independent trials | integer ≥ 1 |
| p_min(τ, N_max, r) | `pMin` / `p_min` | smallest certifiable harm rate (Theorems 1–2) | |
| p× | `pCross` / `p_cross` | crossing rate (Corollary 1) | |
| Ceil⁻(E) | `ceilingBits` / `ceiling_bits` | exculpatory ceiling, sup of \|log Λ\| over Λ ≤ 1; attained at k = 0 for a passive benchmark | bits |
| q₁, q₀ | `q1`, `q0` | P(procedure elicits harm \| H₁), \| H₀) (§4.5) | [0, 1) |
| κ | `kappa` | per-trial discrimination \|log₂[(1−q₁)/(1−q₀)]\| | bits |
| m, ρ | `m`, `rho` | mean cluster size, intra-cluster correlation (§5.3) | m ≥ 1, ρ ∈ [0, 1] |
| DEFF, n_eff | `designEffect`, `n / deff` | 1 + (m − 1)ρ; n/DEFF | |

Direction convention: Λ is elevated-to-acceptable. Logs are base 2 for reported bits (§4), natural
inside proofs and inside the code.

## Formulas

- Clean sheet (§4.1): Λ₀ = [(1 − p_u)/(1 − p_s)]ⁿ; first order ln Λ₀ ≈ −n·p_u(1 − r).
- Sample-size criterion, Eq. (1): n ≥ ln τ / ln[(1 − p_u)/(1 − p_s)] ≈ −ln τ / [p_u(1 − r)]. Table 4 uses ⌈·⌉.
- Boundary (Theorems 1–2): p_min(τ, N_max, r) ≈ −ln τ / [N_max(1 − r)]. Exact closed form: with
  ρ₀ = τ^(1/N_max), p_min = (1 − ρ₀)/(1 − r·ρ₀) (the repository's `boundary_rate`).
- One observed harm (§4.4): Λ₁ = (1/r)·[(1 − p_u)/(1 − p_s)]^(n−1).
- Crossing (Corollary 1): \|log Λ₀\| = \|log Λ₁\| at p× ≈ ln(1/r) / [2n(1 − r)] = ½·p_min(r, n, r).
  Exact closed form on the branch below 2p×: ln[(1 − p)/(1 − rp)] = ln r/(2n − 1).
- One harm carries no evidence (Λ₁ = 1) at p ≈ 2p× (§4.4); exact: ln[(1 − p)/(1 − rp)] = ln r/(n − 1).
- Any elicitation procedure (§4.5): Λ₀ = [(1 − q₁)/(1 − q₀)]ⁿ exactly; n null replications carry nκ bits;
  passive benchmark is q₁ = p_u, q₀ = r·p_u; κ ≈ q₁(1 − r)/ln 2 in the rare-event regime.
- Threshold (§4.6): τ = 1/(L·O₀).
- False certification (§5.2, Methods): P(k = 0 \| n, p) = (1 − p)ⁿ; n = ⌈ln φ / ln(1 − p)⌉.
- Clopper–Pearson (§5.2, Methods): zero events, one-sided 1 − α^(1/n), two-sided 1 − (α/2)^(1/n);
  computed on the nominal (n, k), never on n_eff (Table 5).
- Clustering (§5.3): DEFF = 1 + (m − 1)ρ, n_eff = n/DEFF; "a first-order heuristic rather than an identity";
  the principled route is beta-binomial.

General k > 0 (calculator Audit mode) uses the same likelihood ratio with the binomial coefficient
cancelled: ln Λ = k·ln(1/r) + (n − k)·ln[(1 − p_u)/(1 − p_s)]. The paper writes the k = 0 and k = 1 cases;
nothing beyond that is added. With clustering set, the calculator divides both n and k by DEFF.

## Worked examples (all are test vectors)

| Where | Inputs | Paper's number |
|---|---|---|
| §4.1 | n = 520, (p_u, p_s) = (0.001, 0.0005) | Λ₀ = 0.77 |
| §4.4 | same | Λ₁ = 1.54; 0.625 vs 0.375 bits; ratio 1.7 |
| §1 | p = 1%, n = 520 | clean result carries 1.4× the evidence of one harm |
| §4.2 | p = 10⁻² | 2× update 138; 10× update 458; false certification < 5%: 299 |
| §4.3 | N_max = 10⁵, r = 0.5, τ = 0.5 | p_min ≈ 1.4 × 10⁻⁵ |
| Cor. 1 | r = 0.5, n = 520 / 10⁵ | p× = 1.33 × 10⁻³ / 6.93 × 10⁻⁶ |
| §4.4 | p = 2p× | Λ₁ = 1 |
| §4.5 | q₁ = 0.9, q₀ = 0.1 | 3.17 bits per replication; ≈ 4,400 passive prompts at q₁ = 10⁻³ to match; 2 reps reach τ = 0.1, 3 reach τ = 0.01 |
| §4.5 | q₁ = q₀ = 0.9 | κ = 0 |
| §4.6 | L = 100, O₀ = 1 | τ = 10⁻²; n ≥ 915 / 9,204 / 92,097 at p = 10⁻² / 10⁻³ / 10⁻⁴ |
| §5.2 | p = 0.001 | false certification: StrongREJECT (313) 73.1%, AdvBench 59.4%; < 5% needs 2,995; < 1% needs 4,603 |
| §5.2 | zero harms | 95% one-sided upper bound: 0.57% (1 in 174) at n = 520; 0.14% (1 in 701) at n = 2,100 |
| §5.3 | m = 10, ρ = 0.1 / m = 25, ρ = 0.1 | DEFF 1.9 / 3.4, n_eff ≈ 0.29n |
| §5.3 | k = 4, n = 328 | exact 95% interval 0.33% to 3.09% |

**Table 1** (n = 520, r = 0.5), bits:

| p_u | 10⁻² | 5×10⁻³ | 10⁻³ | 10⁻⁴ | 10⁻⁵ |
|---|---|---|---|---|---|
| clean sheet | 3.779 | 1.882 | 0.375 | 0.038 | 0.004 |
| one harm | 2.772 | 0.879 | 0.625 | 0.963 | 0.996 |
| ratio harm : clean | 0.73 | 0.47 | 1.7 | 25.7 | 265.6 |

The 5×10⁻³ clean-sheet cell is 1.88256 exactly, which rounds to 1.883; the paper prints 1.882.
The vector holds that cell to ±0.001 and says so.

**Table 2** (§4.5), bits: passive 520 @ (10⁻³, 5×10⁻⁴) 0.38; passive 10⁵ @ same 72.19; adaptive
1 @ (0.9, 0.1) 3.17; adaptive 5 @ (0.9, 0.1) 15.85; non-discriminating 5 @ (0.9, 0.85) 2.92.

**Table 3** (§5.2), P(k = 0 \| n, p) in %, n = 250 / 520 / 2,100 / 5,000 / 10⁴ / 10⁵:
10⁻²: 8.1 0.5 0.0 0.0 0.0 0.0; 10⁻³: 77.9 59.4 12.2 0.7 0.0 0.0; 10⁻⁴: 97.5 94.9 81.1 60.7 36.8 0.0;
10⁻⁵: 99.8 99.5 97.9 95.1 90.5 36.8.

**Table 4** (§6), r = 0.5, required n at p = 10⁻² / 10⁻³ / 10⁻⁴ / 10⁻⁵:
2× (τ = 0.5) 138 / 1,386 / 13,862 / 138,629; false certification < 5% 299 / 2,995 / 29,956 / 299,572;
10× (τ = 0.1) 458 / 4,602 / 46,049 / 460,514; 100× (τ = 0.01) 915 / 9,204 / 92,097 / 921,028;
feasible at N_max = 10⁵: yes yes yes no. Upper bound at 95%: any n.

## The eight audited suites (§5, Figure 2)

The paper's only suite-specific input is n; "harm rates are assumed, common across suites, and
taken from published estimates. These are therefore analytic properties of a design at an assumed
operating point and they are not a ranking of benchmark quality."

| Suite | n | Named in text | Paper's caveat |
|---|---|---|---|
| AdvBench | 520 | §5.1, §4.2, §6 | |
| HarmBench | 400 | §5.1 | |
| StrongREJECT | 313 | §5.2 | |
| XSTest | 250 | §5, §5.1 | measures over-refusal, not harm elicitation |
| SafetyBench | 2,100 | §5, §5.1 | multiple-choice; Chinese subset, 300 per category × 7 |
| WildGuard | 1,725 | Figure 2 only | |
| SORRY-Bench | 440 | Figure 2 only | |
| HEx-PHI | 330 | Figure 2 only | |

WildGuard, SORRY-Bench and HEx-PHI are not named in the running text; their sample sizes come from
Figure 2's data, `analysis/power_analysis.py` in this repository. §5.2's "seven of the eight suites
clear 299" is consistent with this list (only XSTest, n = 250, does not).

Operating points (Methods): high-frequency p = 0.08, mid-frequency p = 0.025, low-frequency
p = 0.003, frontier scenario p = 0.01. The paper calls them "defensible rather than authoritative"
(§7). No operating point is assigned to a particular suite, so the calculator does not assign one.

## Assumptions and limitations (§4, §5.3, §7)

- Approximately independent Bernoulli trials at a fixed rate p, binary harm outcome, fixed scoring rule.
- The scoring rule defines the event counted and so defines p; a stricter judge can move a category out of the impossibility regime (§4.3).
- Clustering strictly raises P(k = 0) for any ρ > 0; DEFF is first order; beta-binomial is principled.
- For adaptive campaigns the trial unit is the campaign, not the probes inside it (§4.5).
- q₁ needs a positive control; sandbagging makes it overstate discrimination; an upper bound on q₀ gives a conservative lower bound on κ (§4.5, §7).
- A null result on test distribution Q bounds nothing on deployment distribution P without a linking assumption (§5.4).
- The mathematics is elementary (Hanley and Lippman-Hand, 1983); the contribution is the consequence for evaluation design (§7).
- Falsifier: adaptive elicitation scaling better than O(1/p) in the rare-harm regime would move the boundary (§7).

## Where this differs from the build brief

1. The crossing rate has a factor 2: p× ≈ ln(1/r)/[2n(1 − r)].
2. Exact p_min and exact p× are closed forms; no bisection is needed.
3. There is no eight-suite table with per-suite harm-rate assumptions; presets carry n only.
4. The paper names Ceil⁻ but gives n·κ only in §4.5; the calculator labels it the clean-sheet ceiling at N_max.
5. Hypotheses are two point rates, so verdicts say "the harm rate is p_s rather than p_u", not "below X".
6. Required n uses ⌈·⌉, which reproduces every Table 4 cell.
7. The adaptive extension has a closed form and ships in v1.
8. The paper's code link is `github.com/hackwither/ai-redteam-evidential-limits`; this repository is `hackwither/evidential-limits`.

# Draft issue for UK AISI Inspect (not posted)

Repository: https://github.com/UKGovernmentBEIS/inspect_ai

**Title:** Report what a clean result can certify: a power / evidential-ceiling line for binary-scored evals

**Body:**

When an eval with a binary scorer (harmful or not, refused or not) finishes with zero or few
failures, the log shows the count and the accuracy. It does not show what that count can support.
A run of 520 samples with no harmful outputs looks identical in the log whether the category's
plausible harm rate is 10⁻² or 10⁻⁵, yet the evidence it carries differs by three orders of
magnitude. Readers fill the gap with "no harmful outputs were observed", which overstates
small runs and understates large ones.

Proposal: an optional summary line, computed from data Inspect already holds (n samples, k
failures, the scorer), for example:

```
evidence: k=0/n=520 · 95% upper bound 0.57% (1 in 174, Clopper–Pearson)
          clean run reaches τ=0.1 (10× shift) only for harm rates ≥ 8.8e-3 at r=0.5
```

- The upper bound is the exact Clopper–Pearson bound on the nominal (n, k).
- The second line comes from the closed-form boundary in Kaur (2026), "What AI Red-Team
  Evaluations Can and Cannot Prove", arXiv:2607.21735:
  p_min(τ, n, r) = (1 − τ^(1/n)) / (1 − r·τ^(1/n)). Here r is the improvement ratio between the
  acceptable and elevated harm rates, and τ is the likelihood-ratio threshold. Both would be
  parameters, with defaults of r = 0.5 and τ = 0.1.
- When epochs > 1 or samples come from shared templates, the line would note that the trials are
  not independent. It could optionally take a design effect, as in §5.3 of the paper.

This needs no new dependencies: about 40 lines of standard-library Python. A reference
implementation with test vectors reproducing every number in the paper is at
https://github.com/hackwither/evidential-limits (`py/evidential_ceiling`, MIT). An interactive
version is at https://hackwither.co.in/evidential-ceiling/.

Happy to open a PR if this fits. The likely shape is a metric or a log-summary hook on binary
scorers. I'd welcome guidance on where it belongs.

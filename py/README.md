# evidential-ceiling

What an AI red-team result can prove: the evidential ceiling of
Kaur (2026), *What AI Red-Team Evaluations Can and Cannot Prove*,
[arXiv:2607.21735](https://arxiv.org/abs/2607.21735). Standard library only.

The same functions run the browser calculator at
[hackwither.co.in/evidential-ceiling](https://hackwither.co.in/evidential-ceiling/), and both
implementations pass the same test vectors: every number the paper prints.

```python
import evidential_ceiling as ec

# A clean run of 520 prompts at p_u = 1e-3, r = 0.5 (§4.1, Table 1)
ln_lr = ec.log_lr(n=520, k=0, pu=1e-3, r=0.5)
ec.bits_from_log_lr(ln_lr)                 # 0.375 bits toward acceptable risk

ec.required_n(pu=1e-2, r=0.5, tau=0.1)     # exact 458, closed form 461 (§4.2)
ec.p_min(tau=0.5, nmax=100_000, r=0.5)     # exact 1.386e-05 (§4.3)
ec.p_cross(n=520, r=0.5)                   # 1.333e-03 (Corollary 1)
ec.elicitation(n=1, k=0, q1=0.9, q0=0.1)   # κ = 3.17 bits (§4.5)
ec.clopper_pearson(328, 4)                 # 0.33% to 3.09% (§5.3)
```

Assumptions, from the paper: approximately independent Bernoulli trials at a fixed harm rate, a
binary harm outcome and a fixed scoring rule. `design_effect(m, rho)` gives the first-order
clustering correction of §5.3.

MIT licensed. Cite the paper (see `CITATION.cff` in the repository).

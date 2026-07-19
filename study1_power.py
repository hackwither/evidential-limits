"""
Study 1 — Bootstrap Power Simulation
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.stats.proportion import proportions_ztest

# Literature-grounded harm rates (conservative estimates from published papers)
# Categories correspond to harm severity tiers used in safety benchmarks
LIT_HARM_RATES = {
    # p_unsafe: harm rate of a model with standard RLHF safety training
    # p_safe:   50% reduction (the comparison claim most papers make)
    'High-frequency harm category\n(hate speech, discrimination)':
        (0.08, 0.04),   # Perez et al. 2022, ~8% on unfiltered; 4% after alignment
    'Mid-frequency harm category\n(weapons, drug synthesis)':
        (0.025, 0.0125), # HarmBench ASR for frontier models, non-expert attack
    'Low-frequency harm category\n(CSAM, bioweapons)':
        (0.003, 0.0015), # Estimate from published red team reports
    'Paper Table 1: frontier model\n(the claim most safety papers make)':
        (0.01, 0.005),   # Exact scenario from Section 4
}

# Power calculation via bootstrap simulation
def bootstrap_power(p_unsafe, p_safe, eval_sizes, n_boot=8000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    results = {}
    for n in eval_sizes:
        correct, trials = 0, 0
        for _ in range(n_boot):
            x_u = rng.binomial(n, p_unsafe)
            x_s = rng.binomial(n, p_safe)
            if x_u == 0 and x_s == 0:
                trials += 1; continue
            try:
                _, pval = proportions_ztest(
                    np.array([x_s, x_u]), np.array([n, n]), alternative='smaller')
                if pval < alpha: correct += 1
                trials += 1
            except Exception: pass
        results[n] = correct / max(trials, 1)
    return results

eval_sizes = [50, 100, 200, 313, 400, 520, 1000, 2000, 5000, 10_000, 50_000]
BENCHMARK_MARKS = {'StrongREJECT (313)': 313, 'HarmBench (400)': 400,
                   'AdvBench (520)': 520, 'Large internal RT (5K)': 5000}

print("Running bootstrap simulations (8K iterations per scenario)…")
fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.52, wspace=0.38)
all_results = {}

for idx, (label, (p_unsafe, p_safe)) in enumerate(LIT_HARM_RATES.items()):
    print(f"  [{idx+1}/4] {label.splitlines()[0]}  p={p_unsafe:.4f}→{p_safe:.4f}")
    res = bootstrap_power(p_unsafe, p_safe, eval_sizes)
    all_results[label] = (p_unsafe, p_safe, res)
    powers = [res[n] for n in eval_sizes]

    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    ax.semilogx(eval_sizes, powers, 'b-o', ms=5, lw=2.5, zorder=4)
    ax.axhline(0.80, color='#d32f2f', ls='--', lw=1.8, label='80% power (conventional target)')
    ax.axhline(0.50, color='#ef6c00', ls=':',  lw=1.2, label='50% power (coin flip)')
    ax.axhline(0.05, color='#757575', ls=':',  lw=1.0, label='α = 0.05 (chance level)')

    for bname, bsize in BENCHMARK_MARKS.items():
        pw = float(np.interp(bsize, eval_sizes, powers))
        ax.axvline(bsize, color='#388e3c', alpha=0.3, ls=':')
        short = bname.split(' ')[0]
        ax.text(bsize * 1.06, pw + 0.025, f'{pw:.0%}',
                fontsize=7.5, color='#1b5e20', fontweight='bold', zorder=5)

    ax.set_xlabel('Evaluation set size n (log scale)', fontsize=9)
    ax.set_ylabel('Statistical power (1 − β)', fontsize=9)
    ax.set_title(f'{label}\n(p_unsafe = {p_unsafe:.4f}, p_safe = {p_safe:.4f}; '
                 f'50% relative reduction)', fontsize=8.5)
    ax.legend(fontsize=7.5, loc='upper left')
    ax.set_ylim(-0.02, 1.07); ax.set_xlim(40, 65_000)
    ax.grid(True, alpha=0.22)
    # Shade underpowered region
    ax.axhspan(-0.02, 0.80, alpha=0.04, color='red')

fig.suptitle(
    'Study 1: Bootstrap Power Analysis — Statistical Power of Safety Benchmarks\n'
    'to Detect a 50% Reduction in Harmful Output Rate (8,000 simulations per point)',
    fontsize=12, fontweight='bold', y=1.01)
# Note: spacing is handled by GridSpec above; plt.tight_layout() is intentionally
# omitted because it is incompatible with the manual GridSpec and emits a warning.
plt.savefig('study1_bootstrap_power.pdf', dpi=300, bbox_inches='tight')
plt.savefig('study1_bootstrap_power.png', dpi=150, bbox_inches='tight')
print("✓ Saved study1_bootstrap_power.pdf / .png")

# ── Table for paper ────────────────────────────────────────────────────────
print("\n=== Table 2: Power at benchmark scale ===")
rows = []
for label, (pu, ps, res) in all_results.items():
    n80 = next((n for n in eval_sizes if res[n] >= 0.80), ">50K")
    rows.append({
        'Scenario': label.split('\n')[0],
        'p_unsafe': pu, 'p_safe': ps,
        'n=313':  float(np.interp(313, eval_sizes, [res[n] for n in eval_sizes])),
        'n=400':  float(np.interp(400, eval_sizes, [res[n] for n in eval_sizes])),
        'n=520':  float(np.interp(520, eval_sizes, [res[n] for n in eval_sizes])),
        'n=5000': res[5000],
        'n_for_80pct': n80,
    })
tbl = pd.DataFrame(rows)
print(tbl.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
tbl.to_csv('study1_power_table.csv', index=False)

# Additional: compute analytically for comparison (Z-test formula)
print("\n=== Analytical power check (normal approximation) ===")
from scipy.stats import norm
for label, (pu, ps, _) in all_results.items():
    pool = (pu + ps) / 2
    for n in [313, 520, 5000]:
        se_h0 = np.sqrt(2 * pool * (1 - pool) / n)
        se_h1 = np.sqrt((pu*(1-pu) + ps*(1-ps)) / n)
        delta = pu - ps
        z_crit = norm.ppf(0.95)
        power_analytic = 1 - norm.cdf(z_crit - delta / se_h0)
        print(f"  {label.split(chr(10))[0][:35]}  n={n:5d}: power={power_analytic:.3f}")
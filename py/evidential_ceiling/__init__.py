"""The evidential ceiling of a red-team result, after Kaur (2026), arXiv:2607.21735.

Python twin of ``calculator/ceiling.js``: same functions, same notation, same test vectors.
Standard library only.

Notation follows the paper. H1 is elevated risk (harm rate ``p_u``), H0 acceptable risk
(``p_s = r * p_u``, 0 < r < 1). A result's likelihood ratio is
Λ = P(result | H1) / P(result | H0), so Λ < 1 argues toward acceptable risk. Logs are
natural inside; bits are log2. ``(1 - p) ** n`` is never formed: everything that can lose
precision at p = 1e-9 or n = 1e9 is computed in log space with log1p/expm1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__version__ = "1.0.0"

__all__ = [
    "log_lr_per_trial",
    "log_lr",
    "log_lr_approx",
    "bits_from_log_lr",
    "required_n",
    "p_min",
    "p_cross",
    "p_one_harm_null",
    "ceiling_bits",
    "elicitation",
    "posterior",
    "tau_from_loss",
    "false_cert",
    "required_n_false_cert",
    "design_effect",
    "clopper_pearson",
    "ExactApprox",
]

LN2 = math.log(2)


@dataclass(frozen=True)
class ExactApprox:
    """An exact value and the paper's closed-form approximation of it."""

    exact: float
    approx: float


def log_lr_per_trial(pu: float, r: float) -> float:
    """ln[(1 - p_u)/(1 - p_s)], p_s = r·p_u: the log likelihood ratio of one clean trial (< 0). §4.1"""
    return math.log1p(-pu * (1 - r) / (1 - r * pu))


def log_lr(n: float, k: float, pu: float, r: float) -> float:
    """ln Λ for k harms in n trials (k = 0 is Λ0, §4.1; k = 1 is Λ1, §4.4)."""
    return k * -math.log(r) + (n - k) * log_lr_per_trial(pu, r)


def log_lr_approx(n: float, k: float, pu: float, r: float) -> float:
    """The paper's first-order form ln Λ0 ≈ -n·p_u(1 - r) (§4.1), extended to k harms."""
    return k * -math.log(r) - (n - k) * pu * (1 - r)


def bits_from_log_lr(ln_lr: float) -> float:
    """Signed evidence in bits: positive argues toward H0 (acceptable risk)."""
    return -ln_lr / LN2


def _ceil_int(x: float) -> int:
    near = round(x)
    return int(near) if abs(x - near) < 1e-9 * max(1.0, abs(x)) else math.ceil(x)


@dataclass(frozen=True)
class RequiredN:
    exact: int
    approx: int
    exact_raw: float
    approx_raw: float


def required_n(pu: float, r: float, tau: float) -> RequiredN:
    """Smallest n whose clean sheet reaches Λ0 ≤ τ. Eq. (1), exact and closed form."""
    exact_raw = math.log(tau) / log_lr_per_trial(pu, r)
    approx_raw = -math.log(tau) / (pu * (1 - r))
    return RequiredN(_ceil_int(exact_raw), _ceil_int(approx_raw), exact_raw, approx_raw)


def _rate_for_per_trial_log_lr(x: float, r: float) -> float:
    em1 = math.expm1(x)
    return -em1 / (1 - r - r * em1)


def p_min(tau: float, nmax: float, r: float) -> ExactApprox:
    """Smallest harm rate a clean run of N_max trials certifies at τ. Theorems 1-2."""
    return ExactApprox(_rate_for_per_trial_log_lr(math.log(tau) / nmax, r), -math.log(tau) / (nmax * (1 - r)))


def p_cross(n: float, r: float) -> ExactApprox:
    """Crossing rate p×, where a clean sheet and one harm carry equal evidence. Corollary 1."""
    return ExactApprox(_rate_for_per_trial_log_lr(math.log(r) / (2 * n - 1), r), -math.log(r) / (2 * n * (1 - r)))


def p_one_harm_null(n: float, r: float) -> ExactApprox:
    """Harm rate at which one observed harm carries no evidence (Λ1 = 1), ≈ 2p×. §4.4"""
    exact = _rate_for_per_trial_log_lr(math.log(r) / (n - 1), r) if n > 1 else math.inf
    return ExactApprox(exact, -math.log(r) / (n * (1 - r)))


def ceiling_bits(nmax: float, pu: float, r: float) -> ExactApprox:
    """Exculpatory ceiling Ceil⁻ for a passive benchmark: clean-sheet bits at the budget. §4, §4.5"""
    return ExactApprox(bits_from_log_lr(nmax * log_lr_per_trial(pu, r)), nmax * pu * (1 - r) / LN2)


@dataclass(frozen=True)
class Elicitation:
    ln_lr: float
    kappa: float


def elicitation(n: float, k: float, q1: float, q0: float) -> Elicitation:
    """General elicitation procedure (§4.5). κ = |log2[(1 - q1)/(1 - q0)]| bits per null replication."""
    per_null = math.log1p(-q1) - math.log1p(-q0)
    ln_lr = (k * (math.log(q1) - math.log(q0)) if k > 0 else 0.0) + (n - k) * per_null
    return Elicitation(ln_lr, abs(per_null) / LN2)


@dataclass(frozen=True)
class Posterior:
    prior_odds: float
    posterior_odds: float
    posterior_prob: float


def posterior(prior: float, ln_lr: float) -> Posterior:
    """Prior probability π of elevated risk → posterior, via O1 = O0·Λ. §4.1"""
    ln_o0 = math.log(prior) - math.log1p(-prior)
    ln_o1 = ln_o0 + ln_lr
    prob = 1 / (1 + math.exp(-ln_o1)) if ln_o1 >= 0 else math.exp(ln_o1) / (1 + math.exp(ln_o1))
    return Posterior(math.exp(ln_o0), math.exp(ln_o1), prob)


def tau_from_loss(loss_ratio: float, prior_odds: float) -> float:
    """τ = 1/(L·O0). §4.6"""
    return 1 / (loss_ratio * prior_odds)


def false_cert(n: float, p: float) -> float:
    """P(k = 0 | n, p) = (1 - p)^n, in log space. §5.2"""
    return math.exp(n * math.log1p(-p))


def required_n_false_cert(p: float, phi: float) -> int:
    """Smallest n with P(k = 0 | n, p) ≤ φ: ⌈ln φ / ln(1 - p)⌉. Methods"""
    return _ceil_int(math.log(phi) / math.log1p(-p))


def design_effect(m: float, rho: float) -> float:
    """DEFF = 1 + (m - 1)ρ; n_eff = n/DEFF (§5.3), a first-order heuristic."""
    return 1 + (m - 1) * rho


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float


def clopper_pearson(n: int, k: int, alpha: float = 0.05, two_sided: bool = True) -> Interval:
    """Exact Clopper-Pearson bounds on the nominal (n, k). Zero events: 1 - α^(1/n). §5.2, Table 5"""
    a = alpha / 2 if two_sided else alpha
    if k == n:
        upper = 1.0
    elif k == 0:
        upper = -math.expm1(math.log(a) / n)
    else:
        upper = _bisect_rate(lambda p: math.log(a) - _log_cdf_lower(n, k, p))
    lower = 0.0
    if two_sided and k > 0:
        lower = math.exp(math.log(a) / n) if k == n else _bisect_rate(lambda p: _log_cdf_upper(n, k, p) - math.log(a))
    return Interval(lower, upper)


def _bisect_rate(f) -> float:
    lo, hi = -745.0, math.log1p(-1e-16)
    for _ in range(200):
        if hi - lo <= 1e-15:
            break
        mid = (lo + hi) / 2
        if f(math.exp(mid)) < 0:
            lo = mid
        else:
            hi = mid
    return math.exp((lo + hi) / 2)


def _log_pmf(n: int, j: int, p: float) -> float:
    return math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + (j * math.log(p) if j else 0.0) + (n - j) * math.log1p(-p)


def _log_tail(n: int, start: int, p: float, step: int) -> float:
    first = _log_pmf(n, start, p)
    total = 1.0
    j = start + step
    while 0 <= j <= n:
        t = _log_pmf(n, j, p) - first
        total += math.exp(t)
        if t < -40:
            break
        j += step
    return first + math.log(total)


def _log1m_exp(x: float) -> float:
    return math.log(-math.expm1(x)) if x > -LN2 else math.log1p(-math.exp(x))


def _log_cdf_lower(n: int, k: int, p: float) -> float:
    if k >= n:
        return 0.0
    return _log_tail(n, k, p, -1) if k <= n * p else _log1m_exp(_log_tail(n, k + 1, p, 1))


def _log_cdf_upper(n: int, k: int, p: float) -> float:
    if k <= 0:
        return 0.0
    return _log_tail(n, k, p, 1) if k >= n * p else _log1m_exp(_log_tail(n, k - 1, p, -1))

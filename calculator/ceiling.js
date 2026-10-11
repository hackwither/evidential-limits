// The evidential ceiling of a red-team result, after Kaur (2026),
// "What AI Red-Team Evaluations Can and Cannot Prove", arXiv:2607.21735.
//
// Notation follows the paper. H1 is elevated risk (harm rate p_u), H0 is acceptable
// risk (p_s = r·p_u, 0 < r < 1). A result's likelihood ratio is Λ = P(result | H1) /
// P(result | H0), so Λ < 1 argues toward acceptable risk. Inside this file logs are
// natural; bits are log2. Pure functions, no dependencies, no I/O.
//
// Numerical rule: (1 − p)^n is never formed. Every probability that can underflow or
// lose precision at p = 1e-9 or n = 1e9 is computed in log space with log1p/expm1.

const LN2 = Math.LN2;

/** ln[(1 − p_u)/(1 − p_s)] with p_s = r·p_u: the log likelihood ratio of one clean trial (< 0). §4.1 */
export function logLRPerTrial(pu, r) {
  // Written as log1p(−p_u(1−r)/(1−r·p_u)) rather than a difference of two log1p terms,
  // so it stays exact to rounding as r → 1 or p_u → 0.
  return Math.log1p((-pu * (1 - r)) / (1 - r * pu));
}

/** ln Λ for k harms in n trials. k = 0 is the paper's Λ0 (§4.1), k = 1 its Λ1 (§4.4);
 *  general k is the same binomial ratio (the binomial coefficient cancels). */
export function logLR(n, k, pu, r) {
  return k * -Math.log(r) + (n - k) * logLRPerTrial(pu, r);
}

/** The paper's first-order form: ln Λ0 ≈ −n·p_u(1 − r) (§4.1), extended to k harms. */
export function logLRApprox(n, k, pu, r) {
  return k * -Math.log(r) - (n - k) * pu * (1 - r);
}

/** Signed evidence in bits: positive argues toward H0 (acceptable risk), negative toward H1. */
export function bitsFromLogLR(lnLR) {
  return -lnLR / LN2;
}

/** Smallest n whose clean sheet reaches Λ0 ≤ τ. Eq. (1): exact, and the closed form −ln τ / [p_u(1 − r)]. */
export function requiredN(pu, r, tau) {
  const exactRaw = Math.log(tau) / logLRPerTrial(pu, r);
  const approxRaw = -Math.log(tau) / (pu * (1 - r));
  return { exact: ceilInt(exactRaw), approx: ceilInt(approxRaw), exactRaw, approxRaw };
}

/** Smallest harm rate a clean run of N_max trials certifies at threshold τ. Theorems 1–2.
 *  Exact: solve Λ0(N_max, p) = τ, i.e. (1 − p)/(1 − r·p) = τ^(1/N_max). */
export function pMin(tau, nmax, r) {
  return { exact: rateForPerTrialLogLR(Math.log(tau) / nmax, r), approx: -Math.log(tau) / (nmax * (1 - r)) };
}

/** Crossing rate p×: a clean sheet and one observed harm carry equal evidence. Corollary 1.
 *  Exact: |ln Λ0| = |ln Λ1| on the branch below p = 2p× gives ln[(1−p)/(1−rp)] = ln r / (2n − 1). */
export function pCross(n, r) {
  return { exact: rateForPerTrialLogLR(Math.log(r) / (2 * n - 1), r), approx: -Math.log(r) / (2 * n * (1 - r)) };
}

/** Harm rate at which one observed harm carries no evidence at all (Λ1 = 1), ≈ 2p×. §4.4 */
export function pOneHarmNull(n, r) {
  return { exact: n > 1 ? rateForPerTrialLogLR(Math.log(r) / (n - 1), r) : Infinity, approx: -Math.log(r) / (n * (1 - r)) };
}

/** Exculpatory ceiling Ceil⁻ for a passive benchmark: the clean-sheet bits at the budget, attained at k = 0. §4, §4.5 */
export function ceilingBits(nmax, pu, r) {
  return { exact: bitsFromLogLR(nmax * logLRPerTrial(pu, r)), approx: (nmax * pu * (1 - r)) / LN2 };
}

/** General elicitation procedure (§4.5): q1 = P(harm | H1), q0 = P(harm | H0), n independent replications.
 *  κ = |log2[(1 − q1)/(1 − q0)]| is the per-trial discrimination of a null replication. */
export function elicitation(n, k, q1, q0) {
  const perNull = Math.log1p(-q1) - Math.log1p(-q0);
  const lnLR = (k > 0 ? k * (Math.log(q1) - Math.log(q0)) : 0) + (n - k) * perNull;
  return { lnLR, kappa: Math.abs(perNull) / LN2 };
}

/** Prior probability π that the system is at elevated risk → posterior, via O1 = O0·Λ. §4.1 */
export function posterior(prior, lnLR) {
  const lnO0 = Math.log(prior) - Math.log1p(-prior);
  const lnO1 = lnO0 + lnLR;
  return { priorOdds: Math.exp(lnO0), posteriorOdds: Math.exp(lnO1), posteriorProb: logistic(lnO1) };
}

/** τ = 1/(L·O0), from the loss ratio L = C_FN/C_FP and prior odds O0. §4.6 */
export function tauFromLoss(L, O0) {
  return 1 / (L * O0);
}

/** P(k = 0 | n, p) = (1 − p)^n, the false-certification probability. §5.2 */
export function falseCert(n, p) {
  return Math.exp(n * Math.log1p(-p));
}

/** Smallest n holding false certification at rate p to at most φ: ⌈ln φ / ln(1 − p)⌉. Methods */
export function requiredNFalseCert(p, phi) {
  return ceilInt(Math.log(phi) / Math.log1p(-p));
}

/** Design effect DEFF = 1 + (m − 1)ρ and n_eff = n/DEFF (§5.3). A first-order heuristic, as the paper says. */
export function designEffect(m, rho) {
  return 1 + (m - 1) * rho;
}

/** Exact Clopper–Pearson bounds on the nominal (n, k), 1 − α confidence.
 *  twoSided: [lower, upper] each at α/2. One-sided: upper at α (lower 0). Zero events: 1 − α^(1/n). §5.2, Table 5 */
export function clopperPearson(n, k, alpha = 0.05, twoSided = true) {
  const a = twoSided ? alpha / 2 : alpha;
  let upper;
  if (k === n) upper = 1;
  else if (k === 0) upper = -Math.expm1(Math.log(a) / n);
  else upper = bisectRate((p) => Math.log(a) - logBinomCdfLower(n, k, p)); // P(X ≤ k | p) falls as p rises
  let lower = 0;
  if (twoSided && k > 0) {
    if (k === n) lower = Math.exp(Math.log(a) / n);
    else lower = bisectRate((p) => logBinomCdfUpper(n, k, p) - Math.log(a)); // P(X ≥ k | p) rises with p
  }
  return { lower, upper };
}

// ---- internals ----

// Rate p with ln[(1 − p)/(1 − r·p)] = x (x < 0): p = (1 − e^x)/(1 − r·e^x), written with expm1.
function rateForPerTrialLogLR(x, r) {
  const em1 = Math.expm1(x); // e^x − 1, negative
  return -em1 / (1 - r - r * em1);
}

function ceilInt(x) {
  // Guard against a quotient that should be an integer landing a hair above it.
  const near = Math.round(x);
  return Math.abs(x - near) < 1e-9 * Math.max(1, Math.abs(x)) ? near : Math.ceil(x);
}

function logistic(z) {
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}

// Root of a monotone f on (0, 1), in log space so tiny rates are found as accurately as large ones.
// f(lo) and f(hi) must have opposite signs; f is assumed increasing.
function bisectRate(f) {
  let lo = -745, hi = Math.log1p(-1e-16); // ln p bounds
  for (let i = 0; i < 200 && hi - lo > 1e-15; i++) {
    const mid = (lo + hi) / 2;
    if (f(Math.exp(mid)) < 0) lo = mid;
    else hi = mid;
  }
  return Math.exp((lo + hi) / 2);
}

function logBinomPmf(n, j, p) {
  return lnChoose(n, j) + (j === 0 ? 0 : j * Math.log(p)) + (n - j) * Math.log1p(-p);
}

// Binomial tails in log space. Each sum runs away from the mode, so its terms only fall and it stops
// after a few standard deviations; the tail on the mode's side is taken as a complement instead.

// ln P(X ≤ k)
function logBinomCdfLower(n, k, p) {
  if (k >= n) return 0;
  return k <= n * p ? logTail(n, k, p, -1) : log1mExp(logTail(n, k + 1, p, 1));
}

// ln P(X ≥ k)
function logBinomCdfUpper(n, k, p) {
  if (k <= 0) return 0;
  return k >= n * p ? logTail(n, k, p, 1) : log1mExp(logTail(n, k - 1, p, -1));
}

function logTail(n, start, p, step) {
  const first = logBinomPmf(n, start, p);
  let sum = 1;
  for (let j = start + step; j >= 0 && j <= n; j += step) {
    const t = logBinomPmf(n, j, p) - first;
    sum += Math.exp(t);
    if (t < -40) break;
  }
  return first + Math.log(sum);
}

function log1mExp(x) {
  return x > -LN2 ? Math.log(-Math.expm1(x)) : Math.log1p(-Math.exp(x));
}

function lnChoose(n, k) {
  return lnGamma(n + 1) - lnGamma(k + 1) - lnGamma(n - k + 1);
}

// ln Γ(x) for x > 0: Lanczos (g = 7) below 10, Stirling's series above.
const LANCZOS = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61503916999185, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
export function lnGamma(x) {
  if (x >= 10) {
    const x2 = x * x;
    return (x - 0.5) * Math.log(x) - x + 0.9189385332046728 + 1 / (12 * x) - 1 / (360 * x * x2) + 1 / (1260 * x2 * x2 * x);
  }
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lnGamma(1 - x);
  x -= 1;
  let a = LANCZOS[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += LANCZOS[i] / (x + i);
  return 0.9189385332046728 + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

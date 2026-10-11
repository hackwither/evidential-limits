// The sentences: the live verdict, the methods-section paragraph and the share text.
//
// Rules, because these get screenshotted and quoted:
//   - never "safe", "secure", "certified safe", "passed", "inadequate", "fails", "broken";
//   - say what the evidence favours, between the two stated harm rates, not that a system is anything;
//   - carry the independence assumption, or the effective trial count when clustering is set, in the sentence itself.

import { fmtBits, fmtCount, fmtNum, fmtFactor, fmtFactorFromLn, fmtInt, fmtLRFromLn, fmtPct, fmtRate, fmtOneIn } from "./format.js";
import { VERSION } from "./version.js";

export const BANNED = /\b(safe|secure|certified safe|passed|pass(es)?|inadequate|fail(s|ed|ing)?|broken)\b/i;

const NONE = 0.1; // bits below which evidence is called "effectively none"

const plural = (n, one, many) => (n === 1 ? one : many);

function clusterNote(s, nEff) {
  return `an effective ${fmtCount(nEff)} after clustering (m = ${fmtCount(s.m)}, ρ = ${fmtNum(s.rho)})`;
}

function shift(tau) {
  return `a ${fmtFactor(1 / tau)} shift`;
}

/** The live verdict sentence for a state and its evaluation. */
export function verdict(s, res) {
  const pu = fmtRate(s.pu);
  const ps = fmtRate(res.ps);

  if (s.mode === "benchmark") {
    const { suite, run } = res;
    const who = res.clustered
      ? `A clean run of ${suite.name}’s ${fmtInt(suite.n)} prompts, ${clusterNote(s, run.nEff)},`
      : `A clean run of ${suite.name}’s ${fmtInt(suite.n)} prompts, taken as independent trials,`;
    const tail = run.bits < NONE ? ": effectively none" : "";
    const one = `${who} gives ${fmtBits(run.bits)} bits of evidence that the harm rate is ${ps} rather than ${pu}${tail}.`;
    const two = res.meets
      ? `That clears τ = ${fmtNum(s.tau)} (${shift(s.tau)}), which a run this size reaches for any harm rate of ${fmtRate(res.pminTau)} or more.`
      : `At τ = ${fmtNum(s.tau)} (${shift(s.tau)}), a clean run this size reaches the threshold only for harm rates of ${fmtRate(res.pminTau)} or more.`;
    return `${one} ${two}`;
  }

  if (s.mode === "plan") {
    const trials = res.clustered
      ? `${fmtInt(res.req.exact)} clean trials at m = ${fmtCount(s.m)}, ρ = ${fmtNum(s.rho)} (${fmtInt(res.req.effective)} effective)`
      : `${fmtInt(res.req.exact)} independent clean trials`;
    const one = `Showing that the harm rate is ${ps} rather than ${pu} at τ = ${fmtNum(s.tau)} (${shift(s.tau)}) takes ${trials}.`;
    let two;
    if (res.feasible) two = `A budget of ${fmtInt(s.nmax)} covers it.`;
    else
      two = `A budget of ${fmtInt(s.nmax)} is ${fmtFactor(res.shortfall)} short; the smallest harm rate it can reach at this threshold is ${fmtRate(res.pmin.exact)}.`;
    const three = res.req.huge ? " That many trials is not feasible." : "";
    return `${one} ${two}${three}`;
  }

  // audit
  const post = res.posterior
    ? ` From a prior of ${fmtPct(s.prior)} on elevated risk, the posterior is ${fmtPct(res.posterior.posteriorProb)}.`
    : "";
  return auditVerdict(s, res, pu, ps) + post;
}

function auditVerdict(s, res, pu, ps) {
  const n = fmtInt(s.n);
  const lr = `likelihood ratio ${fmtLRFromLn(res.lnLR)}`;
  const assume = res.clustered ? `, ${clusterNote(s, res.nEff)},` : "";

  if (s.proc === "elicit") {
    const units = plural(s.n, "replication", "replications");
    const what = s.k === 0 ? `${n} independent null ${units}` : `${fmtInt(s.k)} ${plural(s.k, "harm", "harms")} in ${n} independent ${units}`;
    const proc = `for a procedure that elicits harm with probability ${fmtNum(s.q1)} under elevated risk and ${fmtNum(s.q0)} under acceptable risk`;
    if (Math.abs(res.bits) < 1e-12)
      return `${what}${assume} carry no evidence either way ${proc}: it elicits harm equally often under both.`;
    const toward = res.bits > 0 ? "acceptable rather than elevated" : "elevated rather than acceptable";
    const tail = Math.abs(res.bits) < NONE ? ": effectively none" : "";
    return `${what}${assume} give ${fmtBits(res.bits)} bits of evidence (${lr}) that risk is ${toward}, ${proc}${tail}.`;
  }

  if (s.k === 0) {
    const who = res.clustered ? `${n} clean trials${assume}` : `${n} independent clean trials`;
    const tail = res.bits < NONE ? ": effectively none" : "";
    const one = `${who} give ${fmtBits(res.bits)} bits of evidence (${lr}) that the harm rate is ${ps} rather than ${pu}${tail}.`;
    let two;
    if (res.meets) two = `That is a ${fmtFactorFromLn(-res.lnLR)} shift in odds, past τ = ${fmtNum(s.tau)}.`;
    else {
      const need = res.req.huge ? `${fmtInt(res.req.exact)} trials, which is not feasible` : `${fmtInt(res.req.exact)} trials`;
      two = `Reaching τ = ${fmtNum(s.tau)} (${shift(s.tau)}) at this rate takes ${need}.`;
    }
    return `${one} ${two}`;
  }

  const harms = `${fmtInt(s.k)} ${plural(s.k, "harm", "harms")} in ${n} ${res.clustered ? "trials" : "independent trials"}${assume}`;
  const ci = `The exact 95% interval on the per-trial rate is ${fmtPct(res.cp.lower)} to ${fmtPct(res.cp.upper)}.`;
  if (Math.abs(res.bits) < 0.01) return `${harms} carry effectively no evidence either way between harm rates of ${ps} and ${pu}. ${ci}`;
  const one =
    res.bits < 0
      ? `${harms} give ${fmtBits(res.bits)} bits of evidence (${lr}) that the harm rate is ${pu} rather than ${ps}.`
      : `${harms} give ${fmtBits(res.bits)} bits of evidence (${lr}) that the harm rate is ${ps} rather than ${pu}: fewer harms than ${pu} predicts.`;
  return `${one} ${ci}`;
}

const CITE = "Kaur, B. (2026). What AI Red-Team Evaluations Can and Cannot Prove. arXiv:2607.21735. doi:10.48550/arXiv.2607.21735";

/** A methods-section paragraph: every parameter, exact values, the assumption, the version, a permalink and the citation. */
export function reportParagraph(s, res, permalink) {
  const pu = fmtRate(s.pu);
  const ps = fmtRate(res.ps);
  const tau = fmtNum(s.tau);
  const dep = res.clustered
    ? `Trials were treated as clustered (mean cluster size m = ${fmtCount(s.m)}, intra-cluster correlation ρ = ${fmtNum(s.rho)}, design effect ${fmtCount(res.deff)}), using the paper's first-order design-effect correction (§5.3).`
    : "Trials are assumed independent, with a fixed harm rate and a binary harm outcome.";
  let body;

  if (s.mode === "benchmark") {
    const { suite, run } = res;
    body =
      `Using the evidential ceiling (Kaur, 2026), a clean run of ${suite.name} (n = ${fmtInt(suite.n)}${res.clustered ? `, effective n = ${fmtCount(run.nEff)}` : ""}) provides ${fmtBits(run.bits)} bits of evidence ` +
      `(likelihood ratio Λ₀ = ${fmtLRFromLn(run.lnLR, 4)}) that the harm rate is p_s = ${ps} rather than p_u = ${pu} (improvement ratio r = ${fmtNum(s.r)}). ` +
      `At threshold τ = ${tau} it reaches the threshold for p_u ≥ ${fmtRate(res.pminTau)}; at p_u = ${pu} it ${res.meets ? "does" : `does not, and n = ${fmtInt(res.req.exact)} would be required`}. ` +
      `The paper's closed-form approximation gives ${fmtBits(run.bitsApprox)} bits.`;
  } else if (s.mode === "plan") {
    body =
      `Using the evidential ceiling (Kaur, 2026), showing that the harm rate is p_s = ${ps} rather than p_u = ${pu} (r = ${fmtNum(s.r)}) at threshold τ = ${tau} requires n = ${fmtInt(res.req.exact)} clean trials ` +
      `(closed form: ${fmtInt(res.req.approx)}). With a budget of N_max = ${fmtInt(s.nmax)} this is ${res.feasible ? "feasible" : `${fmtFactor(res.shortfall)} short`}; ` +
      `the smallest certifiable harm rate at that budget is p_min = ${fmtRate(res.pmin.exact)} (closed form ${fmtRate(res.pmin.approx)}), ` +
      `and the clean-sheet ceiling at p_u is ${fmtBits(res.ceiling.exact)} bits.`;
  } else if (s.proc === "elicit") {
    body =
      `Using the evidential ceiling (Kaur, 2026, §4.5), ${s.k === 0 ? `a null result across n = ${fmtInt(s.n)} independent replications` : `k = ${fmtInt(s.k)} harms in n = ${fmtInt(s.n)} independent replications`} ` +
      `of a procedure with elicitation rates q₁ = ${fmtNum(s.q1)} (elevated risk) and q₀ = ${fmtNum(s.q0)} (acceptable risk), per-trial discrimination κ = ${fmtBits(res.kappa)} bits, ` +
      `gives ${fmtBits(res.bits)} bits of evidence (likelihood ratio Λ = ${fmtLRFromLn(res.lnLR, 4)}) toward ${res.bits >= 0 ? "acceptable" : "elevated"} risk, against threshold τ = ${tau}, which it ${res.meets ? "reaches" : "does not reach"}.`;
  } else {
    const what = s.k === 0 ? `a clean run of n = ${fmtInt(s.n)} trials` : `k = ${fmtInt(s.k)} harms in n = ${fmtInt(s.n)} trials`;
    const dir = res.bits >= 0 ? `that the harm rate is p_s = ${ps} rather than p_u = ${pu}` : `that the harm rate is p_u = ${pu} rather than p_s = ${ps}`;
    body =
      `Using the evidential ceiling (Kaur, 2026), ${what} provides ${fmtBits(res.bits)} bits of evidence (likelihood ratio Λ = ${fmtLRFromLn(res.lnLR, 4)}) ${dir} ` +
      `(r = ${fmtNum(s.r)}), against threshold τ = ${tau}, which it ${res.meets ? "reaches" : "does not reach"}. ` +
      `The paper's closed-form approximation gives ${fmtBits(res.bitsApprox)} bits. ` +
      `The exact Clopper–Pearson 95% ${s.k === 0 ? `one-sided upper bound on the nominal (n, k) is ${fmtPct(res.cpUpper)}${fmtOneIn(res.cpUpper) ? ` (${fmtOneIn(res.cpUpper)})` : ""}` : `interval on the nominal (n, k) is ${fmtPct(res.cp.lower)} to ${fmtPct(res.cp.upper)}`}.`;
    if (res.posterior) body += ` From a prior probability of elevated risk π = ${fmtNum(s.prior)}, the posterior is ${fmtPct(res.posterior.posteriorProb)}.`;
  }
  return `${body} ${dep} Computed with the Evidential Ceiling Calculator v${VERSION}, ${permalink}. ${CITE}.`;
}

/** Short text for share intents: the verdict's first sentence. */
export function shareText(s, res) {
  const v = verdict(s, res);
  const first = v.match(/^.*?\.(?=\s|$)/);
  return first ? first[0] : v;
}

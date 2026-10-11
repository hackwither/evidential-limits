// Everything the calculator shows, computed from one validated state. Pure; shared by the
// browser, the static site build and the tests.

import * as C from "./ceiling.js";
import { SUITES, suiteById } from "./suites.js";

/** The claims ladder of Table 4 (§6), as evidential thresholds. `fc` is the false-certification row. */
export const CLAIMS = [
  { id: "x2", label: "Belief shifts 2× toward acceptable risk", short: "2×", tau: 0.5 },
  { id: "fc5", label: "False certification held below 5%", short: "FC < 5%", fc: 0.05 },
  { id: "x10", label: "Belief shifts 10×", short: "10×", tau: 0.1 },
  { id: "x100", label: "Belief shifts 100×", short: "100×", tau: 0.01 },
];

/** Relative gap above which exact and closed-form values are flagged (the approximation weakens as p grows). */
export const APPROX_NOTE = 0.01;

const HUGE_N = 1e12;

function relGap(a, b) {
  return Math.abs(a - b) / Math.max(Math.abs(a), Math.abs(b), 1e-300);
}

/** One clean run of n nominal trials at deff: clean-sheet evidence, one-harm comparison, crossing. */
function cleanRun(n, deff, pu, r) {
  const nEff = n / deff;
  const lnLR = C.logLR(nEff, 0, pu, r);
  const lnLRApprox = C.logLRApprox(nEff, 0, pu, r);
  const lnLR1 = C.logLR(nEff, 1, pu, r);
  return {
    nEff,
    lnLR,
    lnLRApprox,
    bits: C.bitsFromLogLR(lnLR),
    bitsApprox: C.bitsFromLogLR(lnLRApprox),
    oneHarmBits: Math.abs(C.bitsFromLogLR(lnLR1)),
    pCross: C.pCross(nEff, r),
    falseCert: C.falseCert(nEff, pu),
  };
}

/** For a clean run of n nominal trials: the smallest p_u each claim reaches, and whether p_u itself does. */
export function ladder(n, deff, pu, r) {
  const nEff = n / deff;
  return CLAIMS.map((c) => {
    if (c.fc) {
      const pmin = -Math.expm1(Math.log(c.fc) / nEff);
      const need = Math.ceil(C.requiredNFalseCert(pu, c.fc) * deff);
      return { ...c, pmin, need, clears: C.falseCert(nEff, pu) <= c.fc };
    }
    const pmin = C.pMin(c.tau, nEff, r).exact;
    const need = Math.ceil(C.requiredN(pu, r, c.tau).exactRaw * deff);
    return { ...c, pmin, need, clears: C.logLR(nEff, 0, pu, r) <= Math.log(c.tau) };
  });
}

function required(pu, r, tau, deff) {
  const q = C.requiredN(pu, r, tau);
  const exact = deff === 1 ? q.exact : Math.ceil(q.exactRaw * deff);
  const approx = deff === 1 ? q.approx : Math.ceil(q.approxRaw * deff);
  return { exact, approx, effective: q.exact, flag: relGap(exact, approx) > APPROX_NOTE, huge: exact > HUGE_N };
}

/** Data for the live figure: required trials against p_u, the budget line and the marks. */
function figure(s, budget, deff) {
  const pmin = C.pMin(s.tau, budget / deff, s.r).exact;
  const cross = C.pCross(budget / deff, s.r).exact;
  return { r: s.r, tau: s.tau, deff, budget, pu: s.pu, pmin, cross };
}

/** Required nominal trials at harm rate p, for drawing the curve. */
export function curveN(p, r, tau, deff) {
  return (Math.log(tau) / C.logLRPerTrial(p, r)) * deff;
}

export function evaluate(s) {
  const deff = C.designEffect(s.m, s.rho);
  const base = { mode: s.mode, deff, clustered: deff !== 1, ps: s.r * s.pu };

  if (s.mode === "benchmark") {
    const suite = suiteById(s.suite) ?? SUITES[0];
    const run = cleanRun(suite.n, deff, s.pu, s.r);
    const req = required(s.pu, s.r, s.tau, deff);
    return {
      ...base,
      suite,
      n: suite.n,
      run,
      req,
      meets: run.lnLR <= Math.log(s.tau),
      shortfall: req.exact / suite.n,
      pminTau: C.pMin(s.tau, run.nEff, s.r).exact,
      upper95: C.clopperPearson(suite.n, 0, 0.05, false).upper,
      ladder: ladder(suite.n, deff, s.pu, s.r),
      grid: SUITES.map((x) => ({ suite: x, ladder: ladder(x.n, deff, s.pu, s.r) })),
      contrast: s.pu / 100 >= 1e-12 ? { pu: s.pu / 100, bits: C.bitsFromLogLR(C.logLR(run.nEff, 0, s.pu / 100, s.r)) } : null,
      approxFlag: relGap(run.bits, run.bitsApprox) > APPROX_NOTE,
      fig: figure(s, suite.n, deff),
    };
  }

  if (s.mode === "plan") {
    const req = required(s.pu, s.r, s.tau, deff);
    const nmaxEff = s.nmax / deff;
    const pm = C.pMin(s.tau, nmaxEff, s.r);
    const ceil = C.ceilingBits(nmaxEff, s.pu, s.r);
    return {
      ...base,
      req,
      feasible: req.exact <= s.nmax,
      shortfall: req.exact / s.nmax,
      pmin: pm,
      pminFlag: relGap(pm.exact, pm.approx) > APPROX_NOTE,
      ceiling: ceil,
      ceilingFlag: relGap(ceil.exact, ceil.approx) > APPROX_NOTE,
      nmaxEff,
      pCross: C.pCross(nmaxEff, s.r),
      fig: figure(s, s.nmax, deff),
    };
  }

  // audit
  const nEff = s.n / deff;
  const kEff = s.k / deff;
  const cp = C.clopperPearson(s.n, s.k, 0.05, true);
  const cpUpper = C.clopperPearson(s.n, s.k, 0.05, false).upper;
  let lnLR, lnLRApprox = null, kappa = null;
  if (s.proc === "elicit") {
    const e = C.elicitation(nEff, kEff, s.q1, s.q0);
    lnLR = e.lnLR;
    kappa = e.kappa;
  } else {
    lnLR = C.logLR(nEff, kEff, s.pu, s.r);
    lnLRApprox = C.logLRApprox(nEff, kEff, s.pu, s.r);
  }
  const bits = C.bitsFromLogLR(lnLR);
  const bitsApprox = lnLRApprox === null ? null : C.bitsFromLogLR(lnLRApprox);
  const out = {
    ...base,
    n: s.n,
    k: s.k,
    nEff,
    kEff,
    lnLR,
    bits,
    bitsApprox,
    approxFlag: bitsApprox !== null && relGap(bits, bitsApprox) > APPROX_NOTE,
    kappa,
    meets: lnLR <= Math.log(s.tau),
    cp,
    cpUpper,
    posterior: s.prior !== null ? C.posterior(s.prior, lnLR) : null,
  };
  if (s.proc === "passive") {
    out.pCross = C.pCross(nEff, s.r);
    out.req = required(s.pu, s.r, s.tau, deff);
    if (s.k === 0) {
      out.falseCert = C.falseCert(nEff, s.pu);
      out.oneHarmBits = Math.abs(C.bitsFromLogLR(C.logLR(nEff, 1, s.pu, s.r)));
    }
    out.fig = figure(s, s.n, deff);
  } else {
    // Replications of this procedure needed for a null result to reach τ.
    out.replications = kappa > 0 ? Math.ceil((-Math.log2(s.tau) / kappa) * deff - 1e-9) : Infinity;
  }
  return out;
}

// node --test calculator/
// Every number the paper prints is a vector in ../test-vectors.json; this file and py/tests share them.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as C from "./ceiling.js";
import { evaluate } from "./model.js";
import { verdict, reportParagraph, shareText, BANNED } from "./verdict.js";
import { DEFAULTS, toQuery, fromQuery, canonical, parseNumber, parseField } from "./state.js";
import { SUITES } from "./suites.js";
import { fmtRate, fmtNum, fmtInt, fmtBits, fmtInput, fmtPct } from "./format.js";

const { vectors, edge } = JSON.parse(readFileSync(new URL("../test-vectors.json", import.meta.url), "utf8"));

const bits = (n, k, pu, r) => Math.abs(C.bitsFromLogLR(C.logLR(n, k, pu, r)));
const kappa = (q1, q0) => C.elicitation(1, 0, q1, q0).kappa;

// Vector `fn` names → implementations. py/tests/test_vectors.py keeps the same table.
const FN = {
  bitsClean: (a) => bits(a.n, 0, a.pu, a.r),
  bitsOneHarm: (a) => bits(a.n, 1, a.pu, a.r),
  ratioHarmToClean: (a) => bits(a.n, 1, a.pu, a.r) / bits(a.n, 0, a.pu, a.r),
  ratioCleanToHarm: (a) => bits(a.n, 0, a.pu, a.r) / bits(a.n, 1, a.pu, a.r),
  lrClean: (a) => Math.exp(C.logLR(a.n, 0, a.pu, a.r)),
  lrOneHarm: (a) => Math.exp(C.logLR(a.n, 1, a.pu, a.r)),
  requiredN: (a) => C.requiredN(a.pu, a.r, a.tau).exact,
  requiredNFalseCert: (a) => C.requiredNFalseCert(a.p, a.phi),
  feasibleAtNmax: (a) => C.requiredN(a.pu, a.r, a.tau).exact <= a.nmax,
  pMin: (a) => C.pMin(a.tau, a.nmax, a.r).exact,
  pMinApprox: (a) => C.pMin(a.tau, a.nmax, a.r).approx,
  lrCleanAtPMin: (a) => Math.exp(C.logLR(a.nmax, 0, C.pMin(a.tau, a.nmax, a.r).exact, a.r)),
  pCross: (a) => C.pCross(a.n, a.r).approx,
  pCrossExact: (a) => C.pCross(a.n, a.r).exact,
  lrOneHarmAtNull: (a) => Math.exp(C.logLR(a.n, 1, C.pOneHarmNull(a.n, a.r).exact, a.r)),
  pOneHarmNullOverPCross: (a) => C.pOneHarmNull(a.n, a.r).exact / C.pCross(a.n, a.r).exact,
  elicitationBits: (a) => Math.abs(C.bitsFromLogLR(C.elicitation(a.n, 0, a.q1, a.q0).lnLR)),
  elicitationKappa: (a) => kappa(a.q1, a.q0),
  replicationsToTau: (a) => Math.ceil(-Math.log2(a.tau) / kappa(a.q1, a.q0)),
  passiveNToMatch: (a) => kappa(a.q1, a.q0) / Math.abs(C.bitsFromLogLR(C.logLRPerTrial(a.pu, a.r))),
  tauFromLoss: (a) => C.tauFromLoss(a.L, a.O0),
  falseCertPct: (a) => 100 * C.falseCert(a.n, a.p),
  cpUpperZeroPct: (a) => 100 * C.clopperPearson(a.n, 0, a.alpha, false).upper,
  cpOneIn: (a) => Math.round(1 / C.clopperPearson(a.n, 0, a.alpha, false).upper),
  cpTwoSidedPct: (a) => {
    const b = C.clopperPearson(a.n, a.k, a.alpha, true);
    return [100 * b.lower, 100 * b.upper];
  },
  designEffect: (a) => C.designEffect(a.m, a.rho),
  neffFraction: (a) => 1 / C.designEffect(a.m, a.rho),
};

function close(got, want, v) {
  if (typeof want === "boolean") return got === want;
  const tol = v.tol ?? 0;
  const rtol = v.rtol ?? 0;
  return Math.abs(got - want) <= tol + rtol * Math.abs(want);
}

for (const v of vectors) {
  test(`${v.id} (${v.source})`, () => {
    const f = FN[v.fn];
    assert.ok(f, `no implementation for ${v.fn}`);
    const got = f(v.args);
    if (Array.isArray(v.expected)) {
      v.expected.forEach((w, i) => assert.ok(close(got[i], w, v), `${v.fn}[${i}] = ${got[i]}, paper ${w} ± ${v.tol}`));
    } else {
      assert.ok(close(got, v.expected, v), `${v.fn}(${JSON.stringify(v.args)}) = ${got}, paper ${v.expected} ± ${v.tol ?? v.rtol}`);
    }
  });
}

// ---- edge cases: finite everywhere, in every mode, with sane verdicts ----

function statesFor(a) {
  const common = { ...DEFAULTS, pu: a.pu, r: a.r, tau: a.tau };
  return [
    { ...common, mode: "audit", n: a.n, k: a.k },
    { ...common, mode: "audit", n: a.n, k: a.k, m: 10, rho: 0.1 },
    { ...common, mode: "plan", nmax: a.nmax },
    { ...common, mode: "plan", nmax: a.nmax, m: 25, rho: 0.1 },
    ...SUITES.map((s) => ({ ...common, mode: "benchmark", suite: s.id })),
  ];
}

function walkNumbers(x, path, out) {
  if (typeof x === "number") out.push([path, x]);
  else if (Array.isArray(x)) x.forEach((y, i) => walkNumbers(y, `${path}[${i}]`, out));
  else if (x && typeof x === "object") for (const [k, y] of Object.entries(x)) walkNumbers(y, `${path}.${k}`, out);
  return out;
}

for (const e of edge) {
  test(`edge: ${e.id}`, () => {
    for (const s of statesFor(e.args)) {
      const res = evaluate(s);
      for (const [p, x] of walkNumbers(res, "res", [])) {
        // Infinity is allowed only where it is the honest answer (replications at κ = 0); NaN never.
        assert.ok(!Number.isNaN(x), `${s.mode}: ${p} is NaN`);
        assert.ok(Number.isFinite(x) || p.endsWith("replications"), `${s.mode}: ${p} = ${x}`);
      }
      const text = verdict(s, res) + reportParagraph(s, res, "https://hackwither.co.in/evidential-ceiling/");
      assert.ok(!/NaN|Infinity|undefined|—/.test(text), `${s.mode}: ${text}`);
    }
  });
}

test("precision holds at p_u = 1e-9, r = 0.999, n = 1e9", () => {
  // ln Λ0 = n·ln[(1 − p_u)/(1 − r·p_u)]; first order −n·p_u(1 − r) = −1e-3, and the exact value agrees to O(p_u).
  const ln = C.logLR(1e9, 0, 1e-9, 0.999);
  assert.ok(Math.abs(ln / -1e-3 - 1) < 1e-8, String(ln));
  // A naive (1 − p)^n would lose most of these digits.
  assert.ok(C.requiredN(1e-9, 0.999, 0.05).exact > 2.99e12);
});

// ---- verdict wording ----

test("verdicts and report paragraphs never use banned words", () => {
  const pus = [0.3, 0.05, 1e-2, 5e-3, 1e-3, 1e-4, 1e-6, 1e-9];
  for (const pu of pus)
    for (const r of [0.1, 0.5, 0.9])
      for (const tau of [0.5, 0.1, 0.01])
        for (const extra of [{}, { m: 10, rho: 0.1 }]) {
          const base = { ...DEFAULTS, pu, r, tau, ...extra };
          const states = [
            ...SUITES.map((s) => ({ ...base, mode: "benchmark", suite: s.id })),
            { ...base, mode: "plan", nmax: 1e5 },
            { ...base, mode: "plan", nmax: 1e3 },
            ...[0, 1, 4].map((k) => ({ ...base, mode: "audit", n: 520, k })),
            { ...base, mode: "audit", n: 5, k: 0, proc: "elicit", q1: 0.9, q0: 0.1 },
            { ...base, mode: "audit", n: 5, k: 1, proc: "elicit", q1: 0.9, q0: 0.85 },
            { ...base, mode: "audit", n: 5, k: 0, proc: "elicit", q1: 0.9, q0: 0.9 },
            { ...base, mode: "audit", n: 520, k: 0, prior: 0.5 },
          ];
          for (const s of states) {
            const res = evaluate(s);
            for (const t of [verdict(s, res), reportParagraph(s, res, "x"), shareText(s, res)])
              assert.ok(!BANNED.test(t), `banned word in: ${t}`);
          }
        }
});

test("default verdict reads as the paper's AdvBench example", () => {
  const res = evaluate({ ...DEFAULTS });
  const v = verdict(DEFAULTS, res);
  assert.match(v, /AdvBench’s 520 prompts/);
  assert.match(v, /0\.375 bits/);
  assert.match(v, /5 × 10⁻⁴ rather than 10⁻³/);
});

test("verdict carries the clustering assumption when set", () => {
  const s = { ...DEFAULTS, mode: "audit", n: 520, k: 0, pu: 0.01, m: 10, rho: 0.1 };
  assert.match(verdict(s, evaluate(s)), /an effective 273\.7 after clustering \(m = 10, ρ = 0\.1\)/);
  assert.match(verdict({ ...s, prior: 0.5 }, evaluate({ ...s, prior: 0.5 })), /posterior is 20\.\d+%/);
});

// ---- URL state ----

test("permalinks round-trip exactly", () => {
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const pick = (a) => a[Math.floor(rnd() * a.length)];
  for (let i = 0; i < 2000; i++) {
    const n = 1 + Math.floor(10 ** (rnd() * 9));
    const s = {
      ...DEFAULTS,
      mode: pick(["benchmark", "audit", "plan"]),
      suite: pick(SUITES).id,
      n,
      k: Math.floor(rnd() * Math.min(n, 50)),
      pu: 10 ** (-rnd() * 9),
      r: Number(rnd().toFixed(3)) || 0.5,
      tau: pick([0.5, 0.1, 0.05, 0.01, 0.001, 1e-6]),
      nmax: 1 + Math.floor(10 ** (rnd() * 12)),
      prior: pick([null, 0.5, 0.01, 0.123456789]),
      m: pick([1, 5, 10.5, 25]),
      rho: pick([0, 0.1, 0.05, 1]),
      proc: pick(["passive", "elicit"]),
      q1: pick([0.9, 0.5, 1e-3]),
      q0: pick([0.1, 0.85, 5e-4]),
    };
    const q = toQuery(s);
    const back = fromQuery(q);
    assert.deepEqual(back.errors, {}, q);
    assert.deepEqual(back.state, canonical(s), q);
    assert.equal(toQuery(back.state), q);
  }
});

test("bad query values fall back to defaults with a message", () => {
  const { state, errors } = fromQuery("mode=audit&n=520&k=900&pu=2&tau=abc&suite=nope");
  assert.equal(state.mode, "audit");
  assert.equal(state.pu, DEFAULTS.pu);
  assert.ok(errors.k && errors.pu && errors.tau && errors.suite);
});

test("number parsing accepts the forms people type", () => {
  for (const t of ["1e-5", "0.00001", "10^-5", "10⁻⁵", "1 × 10^-5", "1x10^-5", "10 ^ -5", "1/100000", "0.001%", "−", ""].slice(0, 9))
    assert.equal(parseNumber(t), 1e-5, t);
  assert.equal(parseNumber("5 × 10⁻³"), 5e-3);
  assert.equal(parseNumber("1,725"), 1725);
  assert.equal(parseNumber("1e5"), 100000);
  assert.ok(Number.isNaN(parseNumber("abc")));
  assert.ok(Number.isNaN(parseNumber("")));
  assert.ok("error" in parseField("pu", "1.5"));
  assert.ok("error" in parseField("n", "2.5"));
  assert.deepEqual(parseField("prior", ""), { value: null });
});

test("formatting", () => {
  assert.equal(fmtRate(1e-3), "10⁻³");
  assert.equal(fmtRate(5e-4), "5 × 10⁻⁴");
  assert.equal(fmtRate(1.3862799e-5), "1.39 × 10⁻⁵");
  assert.equal(fmtRate(0.5), "0.5");
  assert.equal(fmtRate(0.01), "10⁻²");
  assert.equal(fmtNum(0.05), "0.05");
  assert.equal(fmtPct(0.0309283), "3.09%");
  assert.equal(fmtPct(0.0057444), "0.57%");
  assert.equal(fmtInt(1386), "1,386");
  assert.equal(fmtInt(2.9957e12), "3 × 10¹²");
  assert.equal(fmtBits(3.7793), "3.779");
  assert.equal(fmtBits(72.1916), "72.19");
  assert.equal(fmtInput(5e-4), "5e-4");
  assert.equal(fmtInput(1.3862799476737821e-5), "1.3862799476737821e-5");
});

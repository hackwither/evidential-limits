// Calculator state: defaults, input parsing and validation, and the URL query string.
// The query parameter names are a public API (docs/url-parameters.md): never rename one.

import { SUITES } from "./suites.js";
import { fmtInput } from "./format.js";

export const MODES = ["benchmark", "audit", "plan"];
export const PROCS = ["passive", "elicit"];

/** Default view: Benchmark mode, AdvBench, at the paper's worked example (p_u = 10⁻³, r = 0.5, §4.1). */
export const DEFAULTS = Object.freeze({
  mode: "benchmark",
  suite: "advbench",
  n: 520,
  k: 0,
  pu: 0.001,
  r: 0.5,
  tau: 0.1,
  nmax: 100000,
  prior: null,
  m: 1,
  rho: 0,
  proc: "passive",
  q1: 0.9,
  q0: 0.1,
});

const SAY = {
  rate: "between 0 and 1, exclusive, e.g. 1e-3, 0.001 or 10^-3",
  int: "a whole number",
};

/** name → [kind, check(value, state) → error message or ""] */
const RULES = {
  n: ["int", (v) => (v >= 1 && v <= 1e15 ? "" : "n must be a whole number from 1 to 10¹⁵.")],
  k: ["int", (v, s) => (v >= 0 && v <= s.n ? "" : "k must be a whole number from 0 to n.")],
  nmax: ["int", (v) => (v >= 1 && v <= 1e15 ? "" : "N_max must be a whole number from 1 to 10¹⁵.")],
  pu: ["num", (v) => (v > 0 && v < 1 ? "" : `p_u must be ${SAY.rate}.`)],
  r: ["num", (v) => (v > 0 && v < 1 ? "" : "r must be between 0 and 1, exclusive, e.g. 0.5.")],
  tau: ["num", (v) => (v > 0 && v < 1 ? "" : "τ must be between 0 and 1, exclusive, e.g. 0.1 for a 10× shift.")],
  prior: ["num", (v) => (v > 0 && v < 1 ? "" : "π must be between 0 and 1, exclusive, or left empty.")],
  m: ["num", (v) => (v >= 1 && v <= 1e9 ? "" : "m must be at least 1 (1 means no clustering).")],
  rho: ["num", (v) => (v >= 0 && v <= 1 ? "" : "ρ must be between 0 and 1.")],
  q1: ["num", (v) => (v >= 0 && v < 1 ? "" : "q₁ must be at least 0 and below 1.")],
  q0: ["num", (v) => (v >= 0 && v < 1 ? "" : "q₀ must be at least 0 and below 1.")],
};

export const NUMERIC_FIELDS = Object.keys(RULES);

/**
 * Parse a number as people type it: 1e-5, 0.00001, 10^-5, 10⁻⁵, 5×10^-3, 5 x 10^-3, 1/1000, 0.1%, 1,725.
 * Returns NaN when the text is not a number.
 */
export function parseNumber(text) {
  if (typeof text === "number") return text;
  let s = String(text ?? "").trim().toLowerCase();
  if (!s) return NaN;
  s = s
    .replace(/[−–]/g, "-")
    .replace(/[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+/g, (m) => "^" + [...m].map((c) => "-0123456789"["⁻⁰¹²³⁴⁵⁶⁷⁸⁹".indexOf(c)]).join(""))
    .replace(/\s+/g, "")
    .replace(/(\d),(?=\d{3}(\D|$))/g, "$1");
  let pct = false;
  if (s.endsWith("%")) { pct = true; s = s.slice(0, -1); }
  let v;
  let m;
  if ((m = s.match(/^([0-9]*\.?[0-9]+)[x×*·]10\^?([+-]?\d+)$/))) v = Number(`${m[1]}e${m[2]}`);
  else if ((m = s.match(/^10\^([+-]?\d+)$/))) v = Number(`1e${m[1]}`);
  else if ((m = s.match(/^([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)\/([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)$/))) v = Number((Number(m[1]) / Number(m[2])).toPrecision(15));
  else if (/^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)(e[+-]?\d+)?$/.test(s)) v = Number(s);
  else return NaN;
  return pct ? Number((v / 100).toPrecision(15)) : v;
}

/** Validate one field's text in the context of the rest of the state. → { value } or { error } */
export function parseField(name, text, state = DEFAULTS) {
  const rule = RULES[name];
  if (!rule) return { error: `Unknown field ${name}.` };
  if (name === "prior" && String(text ?? "").trim() === "") return { value: null };
  const v = parseNumber(text);
  if (!Number.isFinite(v)) return { error: rule[1](NaN, state) || `${name} must be a number.` };
  if (rule[0] === "int" && !Number.isInteger(v)) return { error: rule[1](NaN, state).replace(/\.$/, "") + `; ${SAY.int}.` };
  const msg = rule[1](v, state);
  return msg ? { error: msg } : { value: v };
}

/** Cross-field checks the single-field rules cannot see. → { field: message } */
export function crossCheck(s) {
  const errors = {};
  if (s.mode === "audit" && !(s.k <= s.n)) errors.k = "k must be a whole number from 0 to n.";
  if (s.mode === "audit" && s.proc === "elicit" && s.k > 0 && !(s.q0 > 0)) errors.q0 = "With harms observed, q₀ must be above 0 (a harm under q₀ = 0 would rule out acceptable risk outright).";
  if (s.mode === "audit" && s.proc === "elicit" && s.k > 0 && !(s.q1 > 0)) errors.q1 = "With harms observed, q₁ must be above 0.";
  return errors;
}

/** The fields a mode reads, in URL order. */
export function fieldsFor(s) {
  if (s.mode === "benchmark") return ["suite", "pu", "r", "tau", "m", "rho"];
  if (s.mode === "plan") return ["pu", "r", "tau", "nmax", "m", "rho"];
  const f = ["n", "k", "proc"];
  if (s.proc === "elicit") f.push("q1", "q0");
  else f.push("pu", "r");
  f.push("tau", "m", "rho");
  if (s.prior !== null) f.push("prior");
  return f;
}

/** The state reduced to what its mode reads, everything else at defaults. Two states with the same canonical form show the same result. */
export function canonical(s) {
  const out = { ...DEFAULTS, mode: s.mode };
  for (const f of fieldsFor(s)) out[f] = s[f];
  return out;
}

/** State → query string (without "?"). Every field the mode reads is written, so a permalink survives a change of defaults. */
export function toQuery(s) {
  const q = [`mode=${s.mode}`];
  for (const f of fieldsFor(s)) {
    const v = s[f];
    q.push(`${f}=${typeof v === "number" ? fmtInput(v) : encodeURIComponent(v)}`);
  }
  return q.join("&");
}

/** Query string → { state, errors }. Unknown or invalid values fall back to defaults and are reported. */
export function fromQuery(search) {
  const params = new URLSearchParams(search);
  const s = { ...DEFAULTS };
  const errors = {};
  const mode = params.get("mode");
  if (mode !== null) {
    if (MODES.includes(mode)) s.mode = mode;
    else errors.mode = `Unknown mode “${mode.slice(0, 20)}”; showing ${DEFAULTS.mode}.`;
  }
  const suite = params.get("suite");
  if (suite !== null) {
    if (SUITES.some((x) => x.id === suite)) s.suite = suite;
    else errors.suite = `Unknown suite “${suite.slice(0, 30)}”; showing ${DEFAULTS.suite}.`;
  }
  const proc = params.get("proc");
  if (proc !== null) {
    if (PROCS.includes(proc)) s.proc = proc;
    else errors.proc = `Unknown procedure “${proc.slice(0, 20)}”.`;
  }
  // n before k, so k is checked against the n in the link.
  for (const f of ["n", ...NUMERIC_FIELDS.filter((x) => x !== "n")]) {
    if (!params.has(f)) continue;
    const r = parseField(f, params.get(f), s);
    if ("error" in r) errors[f] = r.error;
    else s[f] = r.value;
  }
  Object.assign(errors, crossCheck(s));
  for (const f of Object.keys(errors)) if (f in s && f !== "mode") s[f] = DEFAULTS[f];
  if (s.k > s.n) s.k = 0;
  return { state: s, errors };
}

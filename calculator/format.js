// Number formatting for the calculator. Output is plain Unicode text (10⁻⁵, 5 × 10⁻³, 1,386),
// so the same strings work in HTML, the clipboard and share links.

const SUP = { "-": "⁻", "+": "", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const DASH = "—";

export function sup(s) {
  return String(s).replace(/[-+0-9]/g, (c) => SUP[c]);
}

function trim(s) {
  return s.includes(".") && !s.includes("e") ? s.replace(/\.?0+$/, "") : s;
}

/** m × 10ᵉ from log10 of a positive number, so magnitudes beyond double range still print. */
export function sciFromLog10(l10, sig = 3) {
  if (!Number.isFinite(l10)) return DASH;
  let e = Math.floor(l10);
  let m = Number(Math.pow(10, l10 - e).toPrecision(sig));
  if (m >= 10) { m /= 10; e += 1; }
  const ms = trim(m.toPrecision(sig));
  return ms === "1" ? `10${sup(e)}` : `${ms} × 10${sup(e)}`;
}

export function sci(x, sig = 3) {
  if (!(x > 0) || !Number.isFinite(x)) return x === 0 ? "0" : DASH;
  return sciFromLog10(Math.log10(x), sig);
}

/** Harm rates, as the paper writes them: 0.5, 10⁻², 5 × 10⁻⁴, 1.39 × 10⁻⁵. */
export function fmtRate(p, sig = 3) {
  if (!Number.isFinite(p)) return DASH;
  if (p === 0) return "0";
  if (p >= 0.1) return trim(Number(p.toPrecision(sig)).toString());
  return sci(p, sig);
}

/** Parameters (τ, r, ρ, q, π): decimals down to 0.01, then 10ⁿ form. */
export function fmtNum(x, sig = 3) {
  if (!Number.isFinite(x)) return DASH;
  if (x === 0) return "0";
  if (x >= 0.01) return trim(Number(x.toPrecision(sig)).toString());
  return sci(x, sig);
}

/** Integers with thousands separators; scientific from 10¹⁰ up. */
export function fmtInt(n) {
  if (!Number.isFinite(n)) return DASH;
  if (Math.abs(n) >= 1e10) return sci(n, 3);
  const s = String(Math.round(Math.abs(n))).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return n < 0 ? `−${s}` : s;
}

/** A count that may be fractional (an effective sample size). */
export function fmtCount(x) {
  if (!Number.isFinite(x)) return DASH;
  if (Math.abs(x - Math.round(x)) < 1e-9 || x >= 1000) return fmtInt(x);
  return trim(x.toFixed(x < 10 ? 2 : 1));
}

/** Evidence magnitude in bits: 3.779, 72.19, 0.004, 4.3 × 10⁻⁷. */
export function fmtBits(b) {
  const a = Math.abs(b);
  if (!Number.isFinite(a)) return DASH;
  if (a === 0) return "0";
  if (a < 0.001) return sci(a, 2);
  if (a < 10) return a.toFixed(3);
  if (a < 100) return a.toFixed(2);
  if (a < 1e4) return a.toFixed(1);
  return fmtInt(a);
}

/** A likelihood ratio given as ln Λ, so it prints even when Λ itself under- or overflows. */
export function fmtLRFromLn(lnLR, sig = 3) {
  if (!Number.isFinite(lnLR)) return DASH;
  const l10 = lnLR / Math.LN10;
  if (l10 >= -3 && l10 < 4) return trim(Number(Math.exp(lnLR).toPrecision(sig)).toString());
  return sciFromLog10(l10, sig);
}

/** The factor by which odds move, from |ln Λ|: 1.3×, 13.6×, 1,386×, 10⁴³×. */
export function fmtFactorFromLn(absLn) {
  if (!Number.isFinite(absLn)) return DASH;
  const f = Math.exp(absLn);
  if (f < 10) return `${trim(f.toFixed(2))}×`;
  if (f < 1000) return `${trim(Number(f.toPrecision(3)).toString())}×`;
  if (f < 1e6) return `${fmtInt(f)}×`;
  return `${sciFromLog10(absLn / Math.LN10, 2)}×`;
}

export function fmtFactor(f) {
  return fmtFactorFromLn(Math.log(f));
}

/** Percentages for probabilities: 59.4%, 0.57%, < 0.01%. */
export function fmtPct(p) {
  if (!Number.isFinite(p)) return DASH;
  const pct = 100 * p;
  if (pct === 0) return "0%";
  if (pct < 0.01) return "< 0.01%";
  if (pct < 10) return `${trim(pct.toFixed(2))}%`;
  if (pct > 99.9 && pct < 100) return "> 99.9%";
  return `${trim(pct.toFixed(1))}%`;
}

/** "1 in 174" for a small rate. */
export function fmtOneIn(p) {
  if (!(p > 0) || p >= 0.5) return "";
  return `1 in ${fmtInt(Math.round(1 / p))}`;
}

/** Compact ASCII form for inputs and URLs: 0.5, 0.001, 5e-4. Round-trips exactly through Number(). */
export function fmtInput(x) {
  if (!Number.isFinite(x)) return "";
  if (x === 0 || Math.abs(x) >= 1e-3) return String(x);
  const e = x.toExponential();
  return e.replace(/\.?0*e/, "e").replace("e-0", "e-");
}

"""Every number the paper prints, from ../../test-vectors.json (shared with calculator/ceiling.test.js)."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import evidential_ceiling as ec

DOC = json.loads((Path(__file__).resolve().parents[2] / "test-vectors.json").read_text(encoding="utf-8"))


def _bits(n, k, pu, r):
    return abs(ec.bits_from_log_lr(ec.log_lr(n, k, pu, r)))


def _kappa(q1, q0):
    return ec.elicitation(1, 0, q1, q0).kappa


def _cp_two(a):
    b = ec.clopper_pearson(a["n"], a["k"], a["alpha"], True)
    return [100 * b.lower, 100 * b.upper]


# Same table as FN in calculator/ceiling.test.js.
FN = {
    "bitsClean": lambda a: _bits(a["n"], 0, a["pu"], a["r"]),
    "bitsOneHarm": lambda a: _bits(a["n"], 1, a["pu"], a["r"]),
    "ratioHarmToClean": lambda a: _bits(a["n"], 1, a["pu"], a["r"]) / _bits(a["n"], 0, a["pu"], a["r"]),
    "ratioCleanToHarm": lambda a: _bits(a["n"], 0, a["pu"], a["r"]) / _bits(a["n"], 1, a["pu"], a["r"]),
    "lrClean": lambda a: math.exp(ec.log_lr(a["n"], 0, a["pu"], a["r"])),
    "lrOneHarm": lambda a: math.exp(ec.log_lr(a["n"], 1, a["pu"], a["r"])),
    "requiredN": lambda a: ec.required_n(a["pu"], a["r"], a["tau"]).exact,
    "requiredNFalseCert": lambda a: ec.required_n_false_cert(a["p"], a["phi"]),
    "feasibleAtNmax": lambda a: ec.required_n(a["pu"], a["r"], a["tau"]).exact <= a["nmax"],
    "pMin": lambda a: ec.p_min(a["tau"], a["nmax"], a["r"]).exact,
    "pMinApprox": lambda a: ec.p_min(a["tau"], a["nmax"], a["r"]).approx,
    "lrCleanAtPMin": lambda a: math.exp(ec.log_lr(a["nmax"], 0, ec.p_min(a["tau"], a["nmax"], a["r"]).exact, a["r"])),
    "pCross": lambda a: ec.p_cross(a["n"], a["r"]).approx,
    "pCrossExact": lambda a: ec.p_cross(a["n"], a["r"]).exact,
    "lrOneHarmAtNull": lambda a: math.exp(ec.log_lr(a["n"], 1, ec.p_one_harm_null(a["n"], a["r"]).exact, a["r"])),
    "pOneHarmNullOverPCross": lambda a: ec.p_one_harm_null(a["n"], a["r"]).exact / ec.p_cross(a["n"], a["r"]).exact,
    "elicitationBits": lambda a: abs(ec.bits_from_log_lr(ec.elicitation(a["n"], 0, a["q1"], a["q0"]).ln_lr)),
    "elicitationKappa": lambda a: _kappa(a["q1"], a["q0"]),
    "replicationsToTau": lambda a: math.ceil(-math.log2(a["tau"]) / _kappa(a["q1"], a["q0"])),
    "passiveNToMatch": lambda a: _kappa(a["q1"], a["q0"]) / abs(ec.bits_from_log_lr(ec.log_lr_per_trial(a["pu"], a["r"]))),
    "tauFromLoss": lambda a: ec.tau_from_loss(a["L"], a["O0"]),
    "falseCertPct": lambda a: 100 * ec.false_cert(a["n"], a["p"]),
    "cpUpperZeroPct": lambda a: 100 * ec.clopper_pearson(a["n"], 0, a["alpha"], False).upper,
    "cpOneIn": lambda a: round(1 / ec.clopper_pearson(a["n"], 0, a["alpha"], False).upper),
    "cpTwoSidedPct": _cp_two,
    "designEffect": lambda a: ec.design_effect(a["m"], a["rho"]),
    "neffFraction": lambda a: 1 / ec.design_effect(a["m"], a["rho"]),
}


def _close(got, want, v):
    if isinstance(want, bool):
        return got is want or got == want
    return abs(got - want) <= v.get("tol", 0) + v.get("rtol", 0) * abs(want)


class PaperVectors(unittest.TestCase):
    def test_every_vector(self) -> None:
        for v in DOC["vectors"]:
            with self.subTest(v["id"]):
                got = FN[v["fn"]](v["args"])
                want = v["expected"]
                if isinstance(want, list):
                    for g, w in zip(got, want, strict=True):
                        self.assertTrue(_close(g, w, v), f"{v['id']}: {g} vs paper {w}")
                else:
                    self.assertTrue(_close(got, want, v), f"{v['id']}: {got} vs paper {want}")


class EdgeCases(unittest.TestCase):
    def test_finite(self) -> None:
        for e in DOC["edge"]:
            a = e["args"]
            with self.subTest(e["id"]):
                values = [
                    ec.log_lr(a["n"], a["k"], a["pu"], a["r"]),
                    ec.log_lr_approx(a["n"], a["k"], a["pu"], a["r"]),
                    ec.required_n(a["pu"], a["r"], a["tau"]).exact,
                    ec.p_min(a["tau"], a["nmax"], a["r"]).exact,
                    ec.p_min(a["tau"], a["nmax"], a["r"]).approx,
                    ec.p_cross(a["n"], a["r"]).exact,
                    ec.ceiling_bits(a["nmax"], a["pu"], a["r"]).exact,
                    ec.false_cert(a["n"], a["pu"]),
                    *vars(ec.clopper_pearson(a["n"], a["k"])).values(),
                ]
                for x in values:
                    self.assertTrue(math.isfinite(x), f"{e['id']}: {values}")

    def test_precision_at_extremes(self) -> None:
        ln = ec.log_lr(1e9, 0, 1e-9, 0.999)
        self.assertLess(abs(ln / -1e-3 - 1), 1e-8)

    def test_matches_scipy_free_reference(self) -> None:
        # Λ0 against the paper's own (numpy) reference formula at an easy point.
        self.assertAlmostEqual(math.exp(ec.log_lr(520, 0, 1e-3, 0.5)), ((1 - 1e-3) / (1 - 5e-4)) ** 520, places=12)


if __name__ == "__main__":
    unittest.main()

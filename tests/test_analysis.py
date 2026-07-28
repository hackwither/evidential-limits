from __future__ import annotations

import unittest

import numpy as np

from analysis.distributional_gap import mean_cosine_similarity
from analysis.evidential_ceiling import (
    boundary_rate,
    elicitation_bits,
    evidence_bits,
    null_likelihood_ratio,
    one_harm_likelihood_ratio,
    required_n_for_likelihood_ratio,
)
from analysis.false_certification import (
    false_certification_probability,
    required_n_for_false_certification,
    zero_event_clopper_pearson_upper,
)
from analysis.power_analysis import required_n_for_power
from analysis.prepare_lmsys_sample import first_user_turn


class FalseCertificationTests(unittest.TestCase):
    def test_probability_matches_closed_form(self) -> None:
        probability = float(false_certification_probability(520, 0.001))
        self.assertAlmostEqual(probability, (1 - 0.001) ** 520)

    def test_required_sample_size_is_minimal(self) -> None:
        required = required_n_for_false_certification(0.001, 0.05)
        self.assertLessEqual(
            float(false_certification_probability(required, 0.001)),
            0.05,
        )
        self.assertGreater(
            float(false_certification_probability(required - 1, 0.001)),
            0.05,
        )

    def test_zero_event_upper_bound(self) -> None:
        upper = zero_event_clopper_pearson_upper(
            520,
            confidence=0.95,
            two_sided=False,
        )
        self.assertAlmostEqual(upper, 1 - 0.05 ** (1 / 520))


class PowerAnalysisTests(unittest.TestCase):
    def test_rarer_harms_require_more_samples(self) -> None:
        self.assertGreater(required_n_for_power(0.001), required_n_for_power(0.01))


class EvidentialCeilingTests(unittest.TestCase):
    def test_paper_advbench_example(self) -> None:
        clean = float(evidence_bits(null_likelihood_ratio(520, 0.001)))
        one_harm = float(evidence_bits(one_harm_likelihood_ratio(520, 0.001)))
        self.assertAlmostEqual(clean, 0.375, places=3)
        self.assertAlmostEqual(one_harm, 0.625, places=3)

    def test_claim_threshold_is_met_at_required_n(self) -> None:
        required = required_n_for_likelihood_ratio(0.01, 0.1)
        self.assertEqual(required, 458)
        self.assertLessEqual(float(null_likelihood_ratio(required, 0.01)), 0.1)
        self.assertGreater(float(null_likelihood_ratio(required - 1, 0.01)), 0.1)

    def test_boundary_inverts_null_likelihood_ratio(self) -> None:
        p_min = float(boundary_rate(0.5, 100_000, 0.5))
        self.assertAlmostEqual(
            float(null_likelihood_ratio(100_000, p_min, 0.5)),
            0.5,
        )

    def test_general_elicitation_depends_on_discrimination(self) -> None:
        self.assertAlmostEqual(elicitation_bits(1, 0.9, 0.1), np.log2(9))
        self.assertEqual(elicitation_bits(5, 0.9, 0.9), 0.0)


class DistributionalGapTests(unittest.TestCase):
    def test_cosine_similarity_for_orthogonal_vectors(self) -> None:
        x = np.eye(2)
        y = np.array([[1.0, 1.0]])
        self.assertAlmostEqual(mean_cosine_similarity(x), 0.0)
        self.assertAlmostEqual(mean_cosine_similarity(x, y), 1 / np.sqrt(2))


class LmsysPreparationTests(unittest.TestCase):
    def test_first_user_turn_supports_common_schemas(self) -> None:
        self.assertEqual(
            first_user_turn(
                [
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": " question "},
                ]
            ),
            "question",
        )
        self.assertEqual(
            first_user_turn([{"from": "human", "value": "request"}]),
            "request",
        )


if __name__ == "__main__":
    unittest.main()

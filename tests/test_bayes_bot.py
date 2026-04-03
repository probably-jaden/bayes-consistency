"""
Unit tests for bayes_bot.py.

Run with:
    /path/to/poetry/venv/bin/python -m pytest tests/ -v

Tests are structured in three layers:
  1. Pure functions (no mocking needed)
  2. Question construction (no API calls)
  3. Bot forecast logic (LLM calls mocked)
"""

import asyncio
import csv
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Resolve paths so tests can run from any working directory
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_TRIAL_DIR = os.path.dirname(_TESTS_DIR)
_TEMPLATE_DIR = os.path.join(_TRIAL_DIR, "..", "metac-bot-template")

sys.path.insert(0, _TRIAL_DIR)
sys.path.insert(0, _TEMPLATE_DIR)

from forecasting_tools import BinaryQuestion, ConditionalQuestion
from forecasting_tools.data_models.conditional_models import ConditionalPrediction

from src.bayes_bot import (
    BayesConsistencyBot,
    ConditionalPair,
    ConsistencyResult,
    _extract_float,
    _parse_reasoning_sections,
    compute_consistency_metrics,
    load_pairs_from_csv,
    pair_to_conditional_question,
)

from forecasting_tools import ReasonedPrediction


# ── Fixtures ──────────────────────────────────────────────────────────────────

CSV_PATH = os.path.join(_TRIAL_DIR, "data", "metaculus_conditionals_2026-03.csv")

SAMPLE_PAIR = ConditionalPair(
    conditional_post_id=15159,
    condition_post_id=384,
    child_post_id=10965,
    condition_title="Will there be Human-machine intelligence parity before 2040?",
    condition_description="Background on AI parity.",
    condition_resolution_criteria="Resolves YES if AI matches human performance across a wide range of tasks.",
    condition_fine_print="",
    child_title="Will the United States place restrictions on compute capacity before 2050?",
    child_description="Background on compute restrictions.",
    child_resolution_criteria="Resolves YES if a law restricts US compute.",
    child_fine_print="",
    community_p_a=0.96,
    community_p_b=0.399,
    community_p_b_given_a=0.33,
    community_p_b_given_na=0.20,
)

MOCK_REASONING = "\n".join([
    "## Parent Question Reasoning",
    "The AI parity question has a high community probability.",
    "## Child Question Reasoning",
    "compute restrictions are plausible given AI progress.",
    "## Yes Question Reasoning",
    "If AI parity occurs, compute restrictions become more likely.",
    "## No Question Reasoning",
    "Without AI parity, compute restrictions are less certain.",
])


# ── 1. Pure function tests ────────────────────────────────────────────────────

class TestComputeConsistencyMetrics(unittest.TestCase):

    def test_perfectly_consistent_quadruple_gives_zero_error(self):
        """P(B) = P(B|A)*P(A) + P(B|¬A)*(1-P(A)) → error should be 0."""
        m = compute_consistency_metrics(
            p_a=0.5, p_b=0.5, p_b_given_a=0.6, p_b_given_na=0.4
        )
        self.assertAlmostEqual(m["consistency_error"], 0.0, places=6)
        self.assertAlmostEqual(m["p_b_expected"], 0.5, places=6)

    def test_known_inconsistency(self):
        """Manual calculation: P(B)_expected = 0.6*0.8 + 0.1*0.2 = 0.50; P(B)=0.3 → error=0.2."""
        m = compute_consistency_metrics(
            p_a=0.8, p_b=0.3, p_b_given_a=0.6, p_b_given_na=0.1
        )
        self.assertAlmostEqual(m["p_b_expected"], 0.50, places=4)
        self.assertAlmostEqual(m["consistency_error"], 0.20, places=4)

    def test_relative_error_proportional_to_expected(self):
        # error=0.2, expected=0.50 → relative_error=0.4
        m = compute_consistency_metrics(
            p_a=0.8, p_b=0.3, p_b_given_a=0.6, p_b_given_na=0.1
        )
        self.assertAlmostEqual(m["relative_error"], 0.4, places=4)

    def test_calibrated_error_normalises_by_variance(self):
        # consistency_error=0.2, p_b=0.3 → variance=0.21 → calibrated=0.2/sqrt(0.21)
        import math
        m = compute_consistency_metrics(
            p_a=0.8, p_b=0.3, p_b_given_a=0.6, p_b_given_na=0.1
        )
        expected_calibrated = 0.2 / math.sqrt(0.3 * 0.7)
        self.assertAlmostEqual(m["calibrated_error"], expected_calibrated, places=3)

    def test_zero_expected_gives_inf_relative_error(self):
        # If P(B|A)=0 and P(B|¬A)=0, expected=0 → relative_error=inf
        m = compute_consistency_metrics(
            p_a=0.5, p_b=0.1, p_b_given_a=0.0, p_b_given_na=0.0
        )
        self.assertEqual(m["relative_error"], float("inf"))

    def test_all_values_are_rounded_to_4_decimal_places(self):
        m = compute_consistency_metrics(
            p_a=1/3, p_b=1/3, p_b_given_a=1/3, p_b_given_na=1/3
        )
        for key in ["p_b_expected", "consistency_error", "relative_error", "calibrated_error"]:
            if m[key] != float("inf"):
                self.assertEqual(m[key], round(m[key], 4))


class TestExtractFloat(unittest.TestCase):

    def test_float_passthrough(self):
        self.assertEqual(_extract_float(0.7, "P(A)"), 0.7)

    def test_int_converted(self):
        self.assertEqual(_extract_float(1, "P(A)"), 1.0)
        self.assertIsInstance(_extract_float(1, "P(A)"), float)

    def test_non_numeric_raises_type_error(self):
        from forecasting_tools import PredictionAffirmed
        with self.assertRaises(TypeError):
            _extract_float(PredictionAffirmed(), "P(A)")


class TestParseReasoningSections(unittest.TestCase):

    def test_all_four_sections_extracted(self):
        sections = _parse_reasoning_sections(MOCK_REASONING)
        self.assertIn("parent", sections)
        self.assertIn("child", sections)
        self.assertIn("yes", sections)
        self.assertIn("no", sections)

    def test_section_content_correct(self):
        sections = _parse_reasoning_sections(MOCK_REASONING)
        self.assertIn("AI parity", sections["parent"])
        self.assertIn("compute restrictions", sections["child"])

    def test_missing_sections_return_empty_dict(self):
        sections = _parse_reasoning_sections("No sections here.")
        self.assertEqual(sections, {})


# ── 2. Question construction tests ───────────────────────────────────────────

class TestLoadPairsFromCsv(unittest.TestCase):

    @unittest.skipUnless(os.path.exists(CSV_PATH), "CSV file not present")
    def test_loads_correct_number_of_pairs(self):
        pairs = load_pairs_from_csv(CSV_PATH)
        self.assertGreater(len(pairs), 0)

    @unittest.skipUnless(os.path.exists(CSV_PATH), "CSV file not present")
    def test_all_loaded_pairs_have_correct_types(self):
        pairs = load_pairs_from_csv(CSV_PATH)
        for p in pairs:
            self.assertIsInstance(p, ConditionalPair)
            self.assertIsInstance(p.conditional_post_id, int)
            self.assertIsInstance(p.community_p_a, float)
            self.assertIsInstance(p.community_p_b, float)
            self.assertIsInstance(p.community_p_b_given_a, float)
            self.assertIsInstance(p.community_p_b_given_na, float)

    @unittest.skipUnless(os.path.exists(CSV_PATH), "CSV file not present")
    def test_community_predictions_in_unit_interval(self):
        pairs = load_pairs_from_csv(CSV_PATH)
        for p in pairs:
            self.assertGreaterEqual(p.community_p_a, 0.0)
            self.assertLessEqual(p.community_p_a, 1.0)
            self.assertGreaterEqual(p.community_p_b, 0.0)
            self.assertLessEqual(p.community_p_b, 1.0)

    def test_skips_rows_with_invalid_predictions(self):
        """Rows with non-numeric prediction values should be silently skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=[
                "conditional_post_id", "condition_post_id", "child_post_id",
                "condition_title", "condition_description",
                "condition_resolution_criteria", "condition_fine_print",
                "child_title", "child_description",
                "child_resolution_criteria", "child_fine_print",
                "condition_community_prediction", "child_community_prediction",
                "yes_community_prediction", "no_community_prediction",
            ])
            writer.writeheader()
            # Valid row
            writer.writerow({
                "conditional_post_id": "1", "condition_post_id": "2", "child_post_id": "3",
                "condition_title": "A", "condition_description": "",
                "condition_resolution_criteria": "", "condition_fine_print": "",
                "child_title": "B", "child_description": "",
                "child_resolution_criteria": "", "child_fine_print": "",
                "condition_community_prediction": "0.5",
                "child_community_prediction": "0.3",
                "yes_community_prediction": "0.4",
                "no_community_prediction": "0.2",
            })
            # Invalid row (non-numeric prediction)
            writer.writerow({
                "conditional_post_id": "2", "condition_post_id": "3", "child_post_id": "4",
                "condition_title": "C", "condition_description": "",
                "condition_resolution_criteria": "", "condition_fine_print": "",
                "child_title": "D", "child_description": "",
                "child_resolution_criteria": "", "child_fine_print": "",
                "condition_community_prediction": "N/A",  # invalid
                "child_community_prediction": "0.3",
                "yes_community_prediction": "0.4",
                "no_community_prediction": "0.2",
            })
            tmp_path = f.name

        try:
            pairs = load_pairs_from_csv(tmp_path)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].conditional_post_id, 1)
        finally:
            os.unlink(tmp_path)


class TestPairToConditionalQuestion(unittest.TestCase):

    def setUp(self):
        self.cq = pair_to_conditional_question(SAMPLE_PAIR)

    def test_returns_conditional_question(self):
        self.assertIsInstance(self.cq, ConditionalQuestion)

    def test_parent_is_binary_question(self):
        self.assertIsInstance(self.cq.parent, BinaryQuestion)

    def test_child_is_binary_question(self):
        self.assertIsInstance(self.cq.child, BinaryQuestion)

    def test_question_yes_is_binary_question(self):
        self.assertIsInstance(self.cq.question_yes, BinaryQuestion)

    def test_question_no_is_binary_question(self):
        self.assertIsInstance(self.cq.question_no, BinaryQuestion)

    def test_conditional_types_are_set_correctly(self):
        self.assertEqual(self.cq.parent.conditional_type, "parent")
        self.assertEqual(self.cq.child.conditional_type, "child")
        self.assertEqual(self.cq.question_yes.conditional_type, "yes")
        self.assertEqual(self.cq.question_no.conditional_type, "no")

    def test_parent_question_text_matches_condition_title(self):
        self.assertEqual(self.cq.parent.question_text, SAMPLE_PAIR.condition_title)

    def test_child_question_text_matches_child_title(self):
        self.assertEqual(self.cq.child.question_text, SAMPLE_PAIR.child_title)

    def test_yes_question_has_resolution_criteria(self):
        """_conditional_questions_add_detail should populate resolution_criteria."""
        self.assertIsNotNone(self.cq.question_yes.resolution_criteria)
        self.assertIn(
            SAMPLE_PAIR.condition_title,
            self.cq.question_yes.resolution_criteria,
        )

    def test_no_question_has_resolution_criteria(self):
        self.assertIsNotNone(self.cq.question_no.resolution_criteria)
        self.assertIn(
            SAMPLE_PAIR.condition_title,
            self.cq.question_no.resolution_criteria,
        )

    def test_yes_question_text_contains_resolves_yes(self):
        self.assertIn("yes", self.cq.question_yes.question_text.lower())

    def test_no_question_text_contains_resolves_no(self):
        self.assertIn("no", self.cq.question_no.question_text.lower())

    def test_page_urls_use_post_ids(self):
        self.assertIn(str(SAMPLE_PAIR.condition_post_id), self.cq.parent.page_url)
        self.assertIn(str(SAMPLE_PAIR.child_post_id), self.cq.child.page_url)
        self.assertIn(str(SAMPLE_PAIR.conditional_post_id), self.cq.page_url)


# ── 3. Bot forecast tests (LLM calls mocked) ─────────────────────────────────

class TestBayesConsistencyBotForecastPair(unittest.IsolatedAsyncioTestCase):
    """
    Tests for BayesConsistencyBot.forecast_pair with all LLM calls mocked.

    We mock at two levels:
      - run_research: returns a fixed research string
      - _run_forecast_on_conditional: returns a fixed ConditionalPrediction

    This verifies that forecast_pair correctly extracts predictions, computes
    consistency metrics, and returns a ConsistencyResult with the right types.
    """

    MOCK_PREDICTION = ConditionalPrediction(
        parent=0.80,
        child=0.35,
        prediction_yes=0.50,
        prediction_no=0.15,
    )
    # P(B)_expected = 0.50*0.80 + 0.15*0.20 = 0.43
    # consistency_error = |0.35 - 0.43| = 0.08

    async def _run_forecast_pair(self):
        bot = BayesConsistencyBot()
        mock_reasoned = ReasonedPrediction(
            prediction_value=self.MOCK_PREDICTION,
            reasoning=MOCK_REASONING,
        )
        with patch.object(bot, "run_research", new_callable=AsyncMock, return_value="mock research"):
            with patch.object(bot, "_run_forecast_on_conditional", new_callable=AsyncMock, return_value=mock_reasoned):
                return await bot.forecast_pair(SAMPLE_PAIR)

    async def test_returns_consistency_result(self):
        result = await self._run_forecast_pair()
        self.assertIsInstance(result, ConsistencyResult)

    async def test_llm_predictions_are_floats(self):
        result = await self._run_forecast_pair()
        self.assertIsInstance(result.llm_p_a, float)
        self.assertIsInstance(result.llm_p_b, float)
        self.assertIsInstance(result.llm_p_b_given_a, float)
        self.assertIsInstance(result.llm_p_b_given_na, float)

    async def test_llm_predictions_match_mock(self):
        result = await self._run_forecast_pair()
        self.assertAlmostEqual(result.llm_p_a, 0.80, places=5)
        self.assertAlmostEqual(result.llm_p_b, 0.35, places=5)
        self.assertAlmostEqual(result.llm_p_b_given_a, 0.50, places=5)
        self.assertAlmostEqual(result.llm_p_b_given_na, 0.15, places=5)

    async def test_llm_consistency_error_is_float(self):
        result = await self._run_forecast_pair()
        self.assertIsInstance(result.llm_consistency_error, float)

    async def test_llm_consistency_error_is_non_negative(self):
        result = await self._run_forecast_pair()
        self.assertGreaterEqual(result.llm_consistency_error, 0.0)

    async def test_llm_p_b_expected_computed_correctly(self):
        result = await self._run_forecast_pair()
        # P(B)_expected = 0.50*0.80 + 0.15*0.20 = 0.43
        self.assertAlmostEqual(result.llm_p_b_expected, 0.43, places=4)

    async def test_llm_consistency_error_computed_correctly(self):
        result = await self._run_forecast_pair()
        # |0.35 - 0.43| = 0.08
        self.assertAlmostEqual(result.llm_consistency_error, 0.08, places=4)

    async def test_community_predictions_preserved(self):
        result = await self._run_forecast_pair()
        self.assertAlmostEqual(result.community_p_a, SAMPLE_PAIR.community_p_a)
        self.assertAlmostEqual(result.community_p_b, SAMPLE_PAIR.community_p_b)

    async def test_community_consistency_error_computed(self):
        result = await self._run_forecast_pair()
        # community: P(A)=0.96, P(B)=0.399, P(B|A)=0.33, P(B|¬A)=0.20
        # expected = 0.33*0.96 + 0.20*0.04 = 0.3168 + 0.008 = 0.3248
        # error = |0.399 - 0.3248| = 0.0742
        self.assertAlmostEqual(result.community_p_b_expected, 0.3248, places=3)
        self.assertAlmostEqual(result.community_consistency_error, 0.0742, places=3)

    async def test_reasoning_sections_extracted(self):
        result = await self._run_forecast_pair()
        self.assertIn("AI parity", result.reasoning_parent)
        self.assertIn("compute restrictions", result.reasoning_child)

    async def test_metadata_fields_preserved(self):
        result = await self._run_forecast_pair()
        self.assertEqual(result.conditional_post_id, SAMPLE_PAIR.conditional_post_id)
        self.assertEqual(result.condition_title, SAMPLE_PAIR.condition_title)
        self.assertEqual(result.child_title, SAMPLE_PAIR.child_title)


# ── 4. Save/load tests ────────────────────────────────────────────────────────

class TestSaveResults(unittest.TestCase):

    SAMPLE_RESULT = ConsistencyResult(
        conditional_post_id=15159,
        condition_title="Parent question",
        child_title="Child question",
        community_p_a=0.96,
        community_p_b=0.399,
        community_p_b_given_a=0.33,
        community_p_b_given_na=0.20,
        community_p_b_expected=0.3248,
        community_consistency_error=0.0742,
        llm_p_a=0.80,
        llm_p_b=0.35,
        llm_p_b_given_a=0.50,
        llm_p_b_given_na=0.15,
        llm_p_b_expected=0.43,
        llm_consistency_error=0.08,
        llm_relative_error=0.1860,
        llm_calibrated_error=0.2475,
        reasoning_parent="Parent reasoning text.",
        reasoning_child="Child reasoning text.",
        reasoning_yes="Yes reasoning text.",
        reasoning_no="No reasoning text.",
    )

    def test_save_results_creates_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.csv")
            BayesConsistencyBot.save_results([self.SAMPLE_RESULT], path)
            self.assertTrue(os.path.exists(path))

    def test_save_results_excludes_reasoning_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.csv")
            BayesConsistencyBot.save_results([self.SAMPLE_RESULT], path)
            with open(path) as f:
                header = f.readline()
            self.assertNotIn("reasoning_parent", header)
            self.assertNotIn("reasoning_child", header)

    def test_save_results_includes_metric_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.csv")
            BayesConsistencyBot.save_results([self.SAMPLE_RESULT], path)
            with open(path) as f:
                header = f.readline()
            self.assertIn("llm_consistency_error", header)
            self.assertIn("community_consistency_error", header)

    def test_save_results_correct_row_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.csv")
            BayesConsistencyBot.save_results([self.SAMPLE_RESULT, self.SAMPLE_RESULT], path)
            with open(path) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)

    def test_save_full_results_creates_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.csv")
            BayesConsistencyBot.save_full_results([self.SAMPLE_RESULT], path)
            json_path = path.replace(".csv", "_full.json")
            self.assertTrue(os.path.exists(json_path))

    def test_save_full_results_includes_reasoning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.csv")
            BayesConsistencyBot.save_full_results([self.SAMPLE_RESULT], path)
            json_path = path.replace(".csv", "_full.json")
            with open(json_path) as f:
                data = json.load(f)
            self.assertEqual(data[0]["reasoning_parent"], "Parent reasoning text.")

    def test_save_full_results_serialises_all_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.csv")
            BayesConsistencyBot.save_full_results([self.SAMPLE_RESULT], path)
            json_path = path.replace(".csv", "_full.json")
            with open(json_path) as f:
                data = json.load(f)
            record = data[0]
            self.assertAlmostEqual(record["llm_p_a"], 0.80, places=5)
            self.assertAlmostEqual(record["llm_consistency_error"], 0.08, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

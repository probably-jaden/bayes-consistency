"""
bayes_bot.py —  Bayes Consistency bot for Metaculus conditional question pairs.

Adapts SpringTemplateBot2026 to work offline from CSV data (no Metaculus API),
forecasts the four probabilities P(A), P(B), P(B|A), P(B|¬A) for each pair,
and computes Bayes consistency scores based on the Law of Total Probability:

    P(B)_expected = P(B|A) × P(A) + P(B|¬A) × (1 − P(A))
    consistency_error = |P(B) − P(B)_expected|
"""

import asyncio
import csv
import json
import logging
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import dotenv

# Allow imports from the sibling metac-bot-template directory
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
_TEMPLATE_DIR = os.path.join(_PROJECT_DIR, "..", "metac-bot-template")
sys.path.insert(0, _TEMPLATE_DIR)

# Load .env from project root first (contains OPENROUTER_API_KEY etc.),
# then fall back to the metac-bot-template directory.
dotenv.load_dotenv(os.path.join(_PROJECT_DIR, ".env"))
dotenv.load_dotenv(os.path.join(_TEMPLATE_DIR, ".env"))

from pydantic import BaseModel, Field

from forecasting_tools import (  # noqa: E402 (after sys.path manipulation)
    BinaryQuestion,
    ConditionalQuestion,
    GeneralLlm,
    MetaculusQuestion,
    PredictionAffirmed,
    PredictionTypes,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)
from forecasting_tools.data_models.conditional_models import (  # noqa: E402
    ConditionalPrediction,
)

from src.main import SpringTemplateBot2026  # noqa: E402 (from metac-bot-template)

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class ConditionalPair:
    """One row from the CSV: a conditional question pair with community predictions."""

    conditional_post_id: int
    condition_post_id: int
    child_post_id: int
    condition_title: str
    condition_description: str
    condition_resolution_criteria: str
    condition_fine_print: str
    child_title: str
    child_description: str
    child_resolution_criteria: str
    child_fine_print: str
    community_p_a: float  # P(A) — condition_community_prediction
    community_p_b: float  # P(B) — child_community_prediction
    community_p_b_given_a: float  # P(B|A) — yes_community_prediction
    community_p_b_given_na: float  # P(B|¬A) — no_community_prediction


@dataclass
class ConsistencyResult:
    """Forecast quadruple + Bayes consistency metrics for one question pair."""

    conditional_post_id: int
    condition_title: str
    child_title: str
    # Community predictions (ground-truth baseline)
    community_p_a: float
    community_p_b: float
    community_p_b_given_a: float
    community_p_b_given_na: float
    community_p_b_expected: float
    community_consistency_error: float
    # LLM predictions
    llm_p_a: float
    llm_p_b: float
    llm_p_b_given_a: float
    llm_p_b_given_na: float
    llm_p_b_expected: float
    llm_consistency_error: float
    llm_relative_error: float
    llm_calibrated_error: float
    # Full reasoning per sub-question (excluded from summary CSV, kept in JSON)
    reasoning_parent: str = ""
    reasoning_child: str = ""
    reasoning_yes: str = ""
    reasoning_no: str = ""


# ── CSV loader ────────────────────────────────────────────────────────────────


def load_pairs_from_csv(csv_path: str) -> list[ConditionalPair]:
    """Load all conditional question pairs from the CSV, skipping rows with missing predictions."""
    pairs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pairs.append(
                    ConditionalPair(
                        conditional_post_id=int(row["conditional_post_id"]),
                        condition_post_id=int(row["condition_post_id"]),
                        child_post_id=int(row["child_post_id"]),
                        condition_title=row["condition_title"],
                        condition_description=row["condition_description"] or "",
                        condition_resolution_criteria=row["condition_resolution_criteria"] or "",
                        condition_fine_print=row["condition_fine_print"] or "",
                        child_title=row["child_title"],
                        child_description=row["child_description"] or "",
                        child_resolution_criteria=row["child_resolution_criteria"] or "",
                        child_fine_print=row["child_fine_print"] or "",
                        community_p_a=float(row["condition_community_prediction"]),
                        community_p_b=float(row["child_community_prediction"]),
                        community_p_b_given_a=float(row["yes_community_prediction"]),
                        community_p_b_given_na=float(row["no_community_prediction"]),
                    )
                )
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping row {row.get('conditional_post_id', '?')}: {e}")
    return pairs


# ── Consistency scoring ───────────────────────────────────────────────────────


def compute_consistency_metrics(
    p_a: float,
    p_b: float,
    p_b_given_a: float,
    p_b_given_na: float,
) -> dict:
    """
    Compute Bayes consistency metrics using the Law of Total Probability.

    The LTP states: P(B) = P(B|A)·P(A) + P(B|¬A)·(1−P(A))

    A forecaster who gives all four values should satisfy this identity. Deviation
    from it is our measure of Bayesian inconsistency.

    Returns a dict with:
        p_b_expected:       LTP-implied value of P(B)
        consistency_error:  |P(B) − P(B)_expected|  in [0, 1]
        relative_error:     consistency_error / P(B)_expected  (proportional penalty;
                            useful when comparing across questions with different base rates)
        calibrated_error:   consistency_error / sqrt(P(B)·(1−P(B)))
                            Penalizes errors more when the forecaster is already confident
                            about P(B). Returns inf when P(B) ∈ {0, 1}.
    """
    p_b_expected = p_b_given_a * p_a + p_b_given_na * (1.0 - p_a)
    abs_error = abs(p_b - p_b_expected)
    relative_error = abs_error / p_b_expected if p_b_expected > 1e-9 else float("inf")
    variance = p_b * (1.0 - p_b)
    calibrated_error = abs_error / (variance**0.5) if variance > 1e-9 else float("inf")
    return {
        "p_b_expected": round(p_b_expected, 4),
        "consistency_error": round(abs_error, 4),
        "relative_error": round(relative_error, 4),
        "calibrated_error": round(calibrated_error, 4),
    }


# ── LTP projection ────────────────────────────────────────────────────────────


def project_to_ltp_constraint(
    p_a: float,
    p_b: float,
    p_b_given_a: float,
    p_b_given_na: float,
    tol: float = 0.01,
    max_iter: int = 50,
) -> tuple[float, float, float, float, int]:
    """
    Find the nearest point (Euclidean) in [0.01, 0.99]^4 that satisfies the LTP
    constraint:  P(B) = P(B|A)·P(A) + P(B|¬A)·(1−P(A))

    Uses iterative gradient projection.  At each step the four values are moved
    by the minimum amount that corrects the current residual, spread across all
    four coordinates proportional to their partial derivatives — this is the
    unique Euclidean nearest-constraint-point direction.

    The constraint surface is bilinear (nonlinear), so one step is not exact;
    iteration converges quickly (typically 3–10 steps for tol=0.01).

    Args:
        p_a, p_b, p_b_given_a, p_b_given_na : initial probability quadruple
        tol       : stop when |P(B) − P(B)_expected| ≤ tol  (default 0.01)
        max_iter  : safety cap on iterations

    Returns:
        (p_a, p_b, p_b_given_a, p_b_given_na, n_iterations)
        where n_iterations is the number of projection steps taken.
    """
    CLIP_LO, CLIP_HI = 0.01, 0.99

    for n_iter in range(1, max_iter + 1):
        residual = p_b - p_b_given_a * p_a - p_b_given_na * (1.0 - p_a)
        if abs(residual) <= tol:
            break

        # Gradient of g = P(B) − P(B|A)·P(A) − P(B|¬A)·(1−P(A))
        # with respect to (p_a, p_b, p_b_given_a, p_b_given_na):
        g_pa   = -(p_b_given_a - p_b_given_na)   # ∂g/∂P(A)
        g_pb   =  1.0                              # ∂g/∂P(B)
        g_pba  = -p_a                              # ∂g/∂P(B|A)
        g_pbna = -(1.0 - p_a)                      # ∂g/∂P(B|¬A)

        grad_norm_sq = g_pa**2 + g_pb**2 + g_pba**2 + g_pbna**2
        step = residual / grad_norm_sq

        p_a         = max(CLIP_LO, min(CLIP_HI, p_a         - step * g_pa))
        p_b         = max(CLIP_LO, min(CLIP_HI, p_b         - step * g_pb))
        p_b_given_a = max(CLIP_LO, min(CLIP_HI, p_b_given_a - step * g_pba))
        p_b_given_na = max(CLIP_LO, min(CLIP_HI, p_b_given_na - step * g_pbna))

    return p_a, p_b, p_b_given_a, p_b_given_na, n_iter


# ── Question object construction ──────────────────────────────────────────────


def pair_to_conditional_question(pair: ConditionalPair) -> ConditionalQuestion:
    """
    Construct a ConditionalQuestion from a CSV row with no Metaculus API calls.

    question_yes and question_no are enriched via
    ConditionalQuestion._conditional_questions_add_detail, which produces the
    same structured resolution_criteria, fine_print, and background_info as
    the live API path — ensuring the LLM receives clear conditional framing.
    """
    now = datetime.now(timezone.utc)

    parent = BinaryQuestion(
        question_text=pair.condition_title,
        id_of_post=pair.condition_post_id,
        page_url=f"https://www.metaculus.com/questions/{pair.condition_post_id}/",
        background_info=pair.condition_description,
        resolution_criteria=pair.condition_resolution_criteria,
        fine_print=pair.condition_fine_print,
        date_accessed=now,
        conditional_type="parent",
    )

    child = BinaryQuestion(
        question_text=pair.child_title,
        id_of_post=pair.child_post_id,
        page_url=f"https://www.metaculus.com/questions/{pair.child_post_id}/",
        background_info=pair.child_description,
        resolution_criteria=pair.child_resolution_criteria,
        fine_print=pair.child_fine_print,
        date_accessed=now,
        conditional_type="child",
    )

    # Create unique post IDs for yes/no questions (use large numbers to avoid collisions)
    # These are synthetic IDs since the CSV doesn't provide separate IDs for conditional sub-questions
    yes_post_id = pair.child_post_id * 1000 + 1
    no_post_id = pair.child_post_id * 1000 + 2

    question_yes = BinaryQuestion(
        question_text=f"`{pair.child_title}`, if `{pair.condition_title}` resolves to \"yes\"",
        id_of_post=yes_post_id,
        page_url=f"https://www.metaculus.com/questions/{yes_post_id}/",
        background_info=pair.child_description,
        resolution_criteria=pair.child_resolution_criteria,
        fine_print=pair.child_fine_print,
        date_accessed=now,
        conditional_type="yes",
    )
    question_no = BinaryQuestion(
        question_text=f"`{pair.child_title}`, if `{pair.condition_title}` resolves to \"no\"",
        id_of_post=no_post_id,
        page_url=f"https://www.metaculus.com/questions/{no_post_id}/",
        background_info=pair.child_description,
        resolution_criteria=pair.child_resolution_criteria,
        fine_print=pair.child_fine_print,
        date_accessed=now,
        conditional_type="no",
    )

    question_yes = ConditionalQuestion._conditional_questions_add_detail(question_yes, parent, child, yes=True)
    question_no = ConditionalQuestion._conditional_questions_add_detail(question_no, parent, child, yes=False)

    return ConditionalQuestion(
        question_text=f"Conditional: [{pair.condition_title}] → [{pair.child_title}]",
        id_of_post=pair.conditional_post_id,
        page_url=f"https://www.metaculus.com/questions/{pair.conditional_post_id}/",
        date_accessed=now,
        parent=parent,
        child=child,
        question_yes=question_yes,
        question_no=question_no,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_reasoning_sections(full_reasoning: str) -> dict[str, str]:
    """Extract Parent/Child/Yes/No sections from _run_forecast_on_conditional output."""
    sections = {}
    pattern = r"## (Parent|Child|Yes|No) Question Reasoning\n(.*?)(?=\n## |\Z)"
    for match in re.finditer(pattern, full_reasoning, re.DOTALL):
        sections[match.group(1).lower()] = match.group(2).strip()
    return sections


def _extract_float(value, label: str) -> float:
    """Extract a float from a ConditionalPrediction field."""
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    raise TypeError(
        f"Expected float for {label}, got {type(value).__name__}: {value!r}. "
        "This can happen if previous_forecasts were present — check that question "
        "objects are constructed fresh with no previous_forecasts."
    )


# ── Bot ───────────────────────────────────────────────────────────────────────


class BayesConsistencyBot(SpringTemplateBot2026):
    """
    Forecasting bot for the Bayes Consistency trial.

    Inherits all prompt logic from SpringTemplateBot2026 (binary question prompts,
    _run_forecast_on_conditional, _add_reasoning_to_research, etc.) but works
    offline from CSV data — no Metaculus API submission.

    Research is performed by an LLM (via OPENROUTER_API_KEY) before forecasting,
    giving the model relevant context about each question pair.

    For each conditional question pair the bot forecasts:
        P(A)     — parent/condition question
        P(B)     — child question (unconditional)
        P(B|A)   — child given parent resolves YES
        P(B|¬A)  — child given parent resolves NO

    Then computes Bayes consistency metrics against the Law of Total Probability.

    Usage:
        bot = BayesConsistencyBot()
        results = asyncio.run(bot.forecast_pairs(pairs, n_sample=25))
        BayesConsistencyBot.save_results(results, "results/llm_forecasts.csv")
        BayesConsistencyBot.save_full_results(results, "results/llm_forecasts.csv")

    Requires OPENROUTER_API_KEY in .env (see .env.template).
    """

    def __init__(
        self,
        model: str = "openrouter/openai/gpt-4o-mini",
        researcher_model: str = "openrouter/openai/gpt-4o-mini",
    ):
        super().__init__(
            research_reports_per_question=1,
            predictions_per_research_report=1,
            use_research_summary_to_forecast=False,
            publish_reports_to_metaculus=False,
            llms={
                "default": GeneralLlm(model=model, temperature=0.3, timeout=90),
                "parser": GeneralLlm(model=model, temperature=0.0, timeout=60),
                # researcher uses a GeneralLlm so SpringTemplateBot2026.run_research()
                # calls researcher.invoke(prompt) — no dedicated search API needed,
                # just the OPENROUTER_API_KEY in your .env
                "researcher": GeneralLlm(model=researcher_model, temperature=0.3, timeout=90),
            },
        )

    async def _make_prediction_with_notepad_fallback(
        self,
        question: MetaculusQuestion,
        research: str,
    ) -> ReasonedPrediction[PredictionTypes]:
        """
        Forecast a question, initializing a notepad if one doesn't exist.
        This is needed for binary questions that are part of a ConditionalQuestion
        but don't have their own notepad entry.
        """
        try:
            # Try to get the existing notepad for this question
            return await self._make_prediction(question, research)
        except ValueError as e:
            if "No notepad found" in str(e):
                # If no notepad found, initialize one and try again
                logger.info(f"Initializing notepad for question {question.id_of_post}")
                notepad = await self._initialize_notepad(question)
                async with self._note_pad_lock:
                    self._note_pads.append(notepad)
                return await self._make_prediction(question, research)
            else:
                raise

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        """
        Override to parallelise sub-forecasts and handle notepad issues for
        conditional questions created from CSV data.

        WHY WE PARALLELISE
        ------------------
        The four sub-forecasts — P(A), P(B), P(B|A), P(B|¬A) — are each a
        separate LLM round-trip.  The parent-class implementation (and the
        original override) awaited them sequentially, so wall-clock time per
        pair was roughly 5 × LLM-latency (1 research + 4 forecasts in series).
        Because pairs themselves are already gathered in parallel at the
        forecast_pairs level, the bottleneck is the slowest pair, which is
        dominated by these 4 chained forecasting calls.

        QUANTIFIED GAIN
        ---------------
        Sequential baseline : 4 forecast calls × ~10–20 s each ≈ 40–80 s/pair
        Two-batch parallel  : 2 rounds × ~10–20 s each           ≈ 20–40 s/pair
        Expected wall-clock reduction per pair: ~50 %.

        HOW WE PARALLELISE — TWO-BATCH APPROACH
        ----------------------------------------
        Batch 1 (parallel): P(A) and P(B)
            Both use the base research string.  These two questions are
            logically independent, so sharing each other's reasoning before
            forecasting would only introduce anchoring bias with no benefit.

        Batch 2 (parallel): P(B|A) and P(B|¬A)
            Both receive research enriched with P(A) and P(B) reasoning.
            The conditional sub-questions genuinely benefit from knowing the
            model's prior beliefs about P(A) and P(B) (calibration anchoring),
            but they are independent of *each other*, so they can run in
            parallel once that shared context is ready.

        CONTEXT-CHAIN TRADEOFF
        ----------------------
        The fully sequential chain fed each forecast the reasoning of all
        prior forecasts (parent → child → yes → no).  This two-batch design
        drops two context edges:
          • P(B) no longer sees P(A) reasoning  (low value: different question)
          • P(B|¬A) no longer sees P(B|A) reasoning  (low value: sibling
            conditionals; seeing the "yes" answer first may anchor the "no"
            estimate in an undesirable direction)
        The preserved edges (both conditionals see P(A) and P(B) reasoning)
        capture the context most likely to improve calibration.
        """
        # ── Batch 1: forecast P(A) and P(B) in parallel ──────────────────────
        (parent_info, _), (child_info, _) = await asyncio.gather(
            self._get_question_prediction_info(question.parent, research, "parent"),
            self._get_question_prediction_info(question.child, research, "child"),
        )

        # Enrich research with both Batch 1 results before the conditional calls
        enriched_research = self._add_reasoning_to_research(research, parent_info, "parent")
        enriched_research = self._add_reasoning_to_research(enriched_research, child_info, "child")

        # ── Batch 2: forecast P(B|A) and P(B|¬A) in parallel ────────────────
        (yes_info, _), (no_info, _) = await asyncio.gather(
            self._get_question_prediction_info(question.question_yes, enriched_research, "yes"),
            self._get_question_prediction_info(question.question_no, enriched_research, "no"),
        )
        full_reasoning = f"""
## Parent Question Reasoning
{parent_info.reasoning}

## Child Question Reasoning
{child_info.reasoning}

## Yes Question Reasoning
{yes_info.reasoning}

## No Question Reasoning
{no_info.reasoning}
"""
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        """
        Override to use notepad fallback for questions that might not have notepads.
        """
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        
        # Use fallback method that handles missing notepads
        info = await self._make_prediction_with_notepad_fallback(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    async def forecast_pair(self, pair: ConditionalPair) -> ConsistencyResult:
        """Forecast one pair and return a ConsistencyResult with LLM + community metrics."""
        cq = pair_to_conditional_question(pair)
        research = await self.run_research(cq)
        reasoned: ReasonedPrediction[ConditionalPrediction] = (
            await self._run_forecast_on_conditional(cq, research)
        )
        pred = reasoned.prediction_value

        llm_p_a = _extract_float(pred.parent, "P(A)")
        llm_p_b = _extract_float(pred.child, "P(B)")
        llm_p_b_given_a = float(pred.prediction_yes)
        llm_p_b_given_na = float(pred.prediction_no)

        llm_m = compute_consistency_metrics(llm_p_a, llm_p_b, llm_p_b_given_a, llm_p_b_given_na)
        com_m = compute_consistency_metrics(
            pair.community_p_a,
            pair.community_p_b,
            pair.community_p_b_given_a,
            pair.community_p_b_given_na,
        )

        sections = _parse_reasoning_sections(reasoned.reasoning)

        return ConsistencyResult(
            conditional_post_id=pair.conditional_post_id,
            condition_title=pair.condition_title,
            child_title=pair.child_title,
            community_p_a=pair.community_p_a,
            community_p_b=pair.community_p_b,
            community_p_b_given_a=pair.community_p_b_given_a,
            community_p_b_given_na=pair.community_p_b_given_na,
            community_p_b_expected=com_m["p_b_expected"],
            community_consistency_error=com_m["consistency_error"],
            llm_p_a=llm_p_a,
            llm_p_b=llm_p_b,
            llm_p_b_given_a=llm_p_b_given_a,
            llm_p_b_given_na=llm_p_b_given_na,
            llm_p_b_expected=llm_m["p_b_expected"],
            llm_consistency_error=llm_m["consistency_error"],
            llm_relative_error=llm_m["relative_error"],
            llm_calibrated_error=llm_m["calibrated_error"],
            reasoning_parent=sections.get("parent", ""),
            reasoning_child=sections.get("child", ""),
            reasoning_yes=sections.get("yes", ""),
            reasoning_no=sections.get("no", ""),
        )

    async def forecast_pairs(
        self,
        pairs: list[ConditionalPair],
        n_sample: int = 25,
        seed: int = 42,
    ) -> list[ConsistencyResult]:
        """Sample n_sample pairs randomly and forecast each in parallel."""
        random.seed(seed)
        sampled = random.sample(pairs, min(n_sample, len(pairs)))
        
        # Create tasks for each pair
        tasks = []
        for pair in sampled:
            task = asyncio.create_task(self._forecast_pair_with_logging(pair))
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log failures
        successful_results = []
        failed_pairs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_pairs.append((sampled[i].conditional_post_id, result))
                logger.error(f"Failed on pair {sampled[i].conditional_post_id}: {result}")
            else:
                successful_results.append(result)

        if failed_pairs:
            logger.warning(
                f"{len(failed_pairs)}/{len(sampled)} pairs failed. "
                f"Only {len(successful_results)} results returned. "
                f"Failed IDs: {[pid for pid, _ in failed_pairs]}"
            )

        return successful_results

    async def _forecast_pair_with_logging(self, pair: ConditionalPair) -> ConsistencyResult:
        """Helper to forecast a single pair with logging."""
        logger.info(
            f"Forecasting pair {pair.conditional_post_id}: "
            f"{pair.condition_title[:55]!r}"
        )
        return await self.forecast_pair(pair)

    @staticmethod
    def save_results(results: list[ConsistencyResult], output_path: str) -> None:
        """Save summary results to CSV (reasoning text excluded for readability)."""
        _REASONING_FIELDS = {
            "reasoning_parent",
            "reasoning_child",
            "reasoning_yes",
            "reasoning_no",
        }
        fieldnames = [
            f
            for f in ConsistencyResult.__dataclass_fields__
            if f not in _REASONING_FIELDS
        ]
        dirpath = os.path.dirname(output_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = asdict(r)
                writer.writerow({k: row[k] for k in fieldnames})
        logger.info(f"Saved {len(results)} results → {output_path}")

    @staticmethod
    def save_full_results(results: list[ConsistencyResult], output_path: str) -> None:
        """Save full results including reasoning text to JSON for detailed inspection."""
        json_path = output_path.replace(".csv", "_full.json")
        dirpath = os.path.dirname(json_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        logger.info(f"Saved full results → {json_path}")


# ── Separate-research variant ─────────────────────────────────────────────────


class SeparateResearchBot(BayesConsistencyBot):
    """
    Variant of BayesConsistencyBot that eliminates causal contamination in
    marginal forecasts by running separate research for P(A) and P(B).

    **The contamination problem**
    The baseline bot calls run_research(cq) where cq.question_text is
    "Conditional: [A] → [B]".  The researcher sees the arrow and frames
    its response around the A→B causal relationship.  That causally-framed
    research is then passed unchanged to the P(A) and P(B) marginal
    forecasts, where the superforecaster receives unsolicited causal
    commentary when it should be reasoning about base rates only.

    **The fix**
    Run two parallel research calls — one for question.parent (researcher
    sees only A's question text), one for question.child (sees only B's).
    Those clean, context-free research strings feed the marginal forecasts.
    The conditional forecasts (P(B|A), P(B|¬A)) receive an enriched context
    that combines both research strings plus both marginal reasonings, so
    they still benefit from full context.

    Research call cost: 2 LLM calls instead of 1 (both run in parallel).
    """

    async def forecast_pair(self, pair: ConditionalPair) -> ConsistencyResult:
        cq = pair_to_conditional_question(pair)

        # ── Separate research: researcher sees each question independently ──
        research_parent, research_child = await asyncio.gather(
            self.run_research(cq.parent),
            self.run_research(cq.child),
        )

        # ── Batch 1: P(A) and P(B) with dedicated, uncontaminated research ──
        (parent_info, _), (child_info, _) = await asyncio.gather(
            self._get_question_prediction_info(cq.parent, research_parent, "parent"),
            self._get_question_prediction_info(cq.child, research_child, "child"),
        )

        # ── Enrich research for conditionals: both strings + both reasonings ──
        combined_research = f"{research_parent}\n\n---\n\n{research_child}"
        enriched_research = self._add_reasoning_to_research(combined_research, parent_info, "parent")
        enriched_research = self._add_reasoning_to_research(enriched_research, child_info, "child")

        # ── Batch 2: P(B|A) and P(B|¬A) with full context ───────────────────
        (yes_info, _), (no_info, _) = await asyncio.gather(
            self._get_question_prediction_info(cq.question_yes, enriched_research, "yes"),
            self._get_question_prediction_info(cq.question_no, enriched_research, "no"),
        )

        full_reasoning = (
            f"## Parent Question Reasoning\n{parent_info.reasoning}\n\n"
            f"## Child Question Reasoning\n{child_info.reasoning}\n\n"
            f"## Yes Question Reasoning\n{yes_info.reasoning}\n\n"
            f"## No Question Reasoning\n{no_info.reasoning}\n"
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        reasoned = ReasonedPrediction(reasoning=full_reasoning, prediction_value=full_prediction)

        pred = reasoned.prediction_value
        llm_p_a = _extract_float(pred.parent, "P(A)")
        llm_p_b = _extract_float(pred.child, "P(B)")
        llm_p_b_given_a = float(pred.prediction_yes)
        llm_p_b_given_na = float(pred.prediction_no)

        llm_m = compute_consistency_metrics(llm_p_a, llm_p_b, llm_p_b_given_a, llm_p_b_given_na)
        com_m = compute_consistency_metrics(
            pair.community_p_a, pair.community_p_b,
            pair.community_p_b_given_a, pair.community_p_b_given_na,
        )
        sections = _parse_reasoning_sections(reasoned.reasoning)

        return ConsistencyResult(
            conditional_post_id=pair.conditional_post_id,
            condition_title=pair.condition_title,
            child_title=pair.child_title,
            community_p_a=pair.community_p_a,
            community_p_b=pair.community_p_b,
            community_p_b_given_a=pair.community_p_b_given_a,
            community_p_b_given_na=pair.community_p_b_given_na,
            community_p_b_expected=com_m["p_b_expected"],
            community_consistency_error=com_m["consistency_error"],
            llm_p_a=llm_p_a,
            llm_p_b=llm_p_b,
            llm_p_b_given_a=llm_p_b_given_a,
            llm_p_b_given_na=llm_p_b_given_na,
            llm_p_b_expected=llm_m["p_b_expected"],
            llm_consistency_error=llm_m["consistency_error"],
            llm_relative_error=llm_m["relative_error"],
            llm_calibrated_error=llm_m["calibrated_error"],
            reasoning_parent=sections.get("parent", ""),
            reasoning_child=sections.get("child", ""),
            reasoning_yes=sections.get("yes", ""),
            reasoning_no=sections.get("no", ""),
        )


# ── Conditionals-first variant ────────────────────────────────────────────────


class ConditionalsFirstBot(BayesConsistencyBot):
    """
    Variant that reverses the elicitation order: forecast P(B|A) and P(B|¬A)
    first, then P(A) and P(B) with enriched research from the conditionals.

    **Hypothesis:** eliciting conditionals first forces the LLM to think through
    the causal/conditional structure of A→B *before* committing to marginal base
    rates.  When the marginals are forecast second, the LLM has already articulated
    how A affects B, which may naturally nudge P(B) toward LTP consistency without
    any explicit constraint.
    """

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        # ── Batch 1: forecast P(B|A) and P(B|¬A) in parallel (base research) ──
        (yes_info, _), (no_info, _) = await asyncio.gather(
            self._get_question_prediction_info(question.question_yes, research, "yes"),
            self._get_question_prediction_info(question.question_no, research, "no"),
        )

        # Enrich research with conditional reasoning before marginal calls
        enriched_research = self._add_reasoning_to_research(research, yes_info, "yes")
        enriched_research = self._add_reasoning_to_research(enriched_research, no_info, "no")

        # ── Batch 2: forecast P(A) and P(B) in parallel (enriched research) ───
        (parent_info, _), (child_info, _) = await asyncio.gather(
            self._get_question_prediction_info(question.parent, enriched_research, "parent"),
            self._get_question_prediction_info(question.child, enriched_research, "child"),
        )

        full_reasoning = (
            f"## Parent Question Reasoning\n{parent_info.reasoning}\n\n"
            f"## Child Question Reasoning\n{child_info.reasoning}\n\n"
            f"## Yes Question Reasoning\n{yes_info.reasoning}\n\n"
            f"## No Question Reasoning\n{no_info.reasoning}\n"
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )


# ── Joint-prompt variant ──────────────────────────────────────────────────────


class _FourProbabilityPrediction(BaseModel):
    """All four probabilities extracted from a single joint LLM call."""

    p_a: float = Field(..., description="P(A): probability the parent question resolves YES, between 0 and 1")
    p_b: float = Field(..., description="P(B): unconditional probability the child question resolves YES, between 0 and 1")
    p_b_given_a: float = Field(..., description="P(B|A): probability of child resolving YES given parent resolves YES, between 0 and 1")
    p_b_given_na: float = Field(..., description="P(B|¬A): probability of child resolves YES given parent resolves NO, between 0 and 1")


class JointPromptBot(BayesConsistencyBot):
    """
    Variant of BayesConsistencyBot that forecasts all four probabilities in a
    **single joint LLM call** with an explicit LTP self-check instruction.

    **The strategy**
    The baseline bot makes four independent calls — one each for P(A), P(B),
    P(B|A), and P(B|¬A) — so it has no mechanism to enforce the Law of Total
    Probability across them.  This bot presents all four tasks together and
    explicitly asks the model to verify:

        P(B) = P(B|A)·P(A) + P(B|¬A)·(1−P(A))

    before committing to its answers.

    **Hypothesis:** seeing all four values simultaneously and being reminded of
    the LTP constraint allows the model to self-correct inconsistencies that
    arise from forecasting each probability in isolation.  Consistency is
    *requested*, not *enforced* — the model may still deviate.

    Research is still performed first (one call, same as baseline), so the
    model receives relevant context before making its joint forecast.
    """

    def __init__(
        self,
        model: str = "openrouter/openai/gpt-4o-mini",
        researcher_model: str = "openrouter/openai/gpt-4o-mini",
    ):
        super().__init__(model=model, researcher_model=researcher_model)
        self._joint_llm = GeneralLlm(model=model, temperature=0.3, timeout=90)
        self._joint_parser = GeneralLlm(model=model, temperature=0.0, timeout=60)

    MAX_RETRIES = 3

    async def forecast_pair(self, pair: ConditionalPair) -> ConsistencyResult:
        cq = pair_to_conditional_question(pair)
        research = await self.run_research(cq)
        last_exc = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await self._joint_forecast(pair, research)
            except Exception as e:
                last_exc = e
                logger.warning(
                    f"Joint forecast attempt {attempt}/{self.MAX_RETRIES} failed "
                    f"for pair {pair.conditional_post_id}: {e}"
                )
        raise last_exc  # all retries exhausted

    async def _joint_forecast(
        self, pair: ConditionalPair, research: str
    ) -> ConsistencyResult:
        prompt = clean_indents(f"""
            You are a professional forecaster. Use the background research below to
            estimate four probability values for a related pair of questions.

            --- RESEARCH ---
            {research}
            --- END RESEARCH ---

            **Parent question (A):** {pair.condition_title}
            {pair.condition_resolution_criteria}

            **Child question (B):** {pair.child_title}
            {pair.child_resolution_criteria}

            Today is {datetime.now().strftime('%Y-%m-%d')}.

            Please estimate all four probabilities. For each, write a brief rationale.

            1. P(A)    — probability the parent question resolves YES.
            2. P(B|A)  — probability the child resolves YES, *assuming* the parent resolves YES.
            3. P(B|¬A) — probability the child resolves YES, *assuming* the parent resolves NO.
            4. P(B)    — unconditional probability the child resolves YES.

            IMPORTANT: Before committing to P(B), verify the Law of Total Probability:
                P(B) should equal P(B|A)×P(A) + P(B|¬A)×(1−P(A)).
            Adjust your estimates if they are not approximately consistent with this identity.

            After your reasoning, you MUST end your response with EXACTLY these four
            lines (replace each <decimal> with a number like 0.35):

            P(A): <decimal>
            P(B|A): <decimal>
            P(B|not A): <decimal>
            P(B): <decimal>

            Use plain ASCII. Do not use special characters like ¬. Write "not A" instead.
        """)

        reasoning = await self._joint_llm.invoke(prompt)
        pred: _FourProbabilityPrediction = await structure_output(
            reasoning, _FourProbabilityPrediction, model=self._joint_parser, num_validation_samples=1
        )

        def clamp(x: float) -> float:
            return max(0.01, min(0.99, x))

        llm_p_a = clamp(pred.p_a)
        llm_p_b = clamp(pred.p_b)
        llm_p_b_given_a = clamp(pred.p_b_given_a)
        llm_p_b_given_na = clamp(pred.p_b_given_na)

        llm_m = compute_consistency_metrics(llm_p_a, llm_p_b, llm_p_b_given_a, llm_p_b_given_na)
        com_m = compute_consistency_metrics(
            pair.community_p_a, pair.community_p_b,
            pair.community_p_b_given_a, pair.community_p_b_given_na,
        )

        return ConsistencyResult(
            conditional_post_id=pair.conditional_post_id,
            condition_title=pair.condition_title,
            child_title=pair.child_title,
            community_p_a=pair.community_p_a,
            community_p_b=pair.community_p_b,
            community_p_b_given_a=pair.community_p_b_given_a,
            community_p_b_given_na=pair.community_p_b_given_na,
            community_p_b_expected=com_m["p_b_expected"],
            community_consistency_error=com_m["consistency_error"],
            llm_p_a=llm_p_a,
            llm_p_b=llm_p_b,
            llm_p_b_given_a=llm_p_b_given_a,
            llm_p_b_given_na=llm_p_b_given_na,
            llm_p_b_expected=llm_m["p_b_expected"],
            llm_consistency_error=llm_m["consistency_error"],
            llm_relative_error=llm_m["relative_error"],
            llm_calibrated_error=llm_m["calibrated_error"],
            reasoning_parent=reasoning,  # full joint reasoning stored here
            reasoning_child="",
            reasoning_yes="",
            reasoning_no="",
        )


# ── Recursive revision variant ───────────────────────────────────────────────


import math


class RecursiveRevisionBot(BayesConsistencyBot):
    """
    Variant that iteratively asks the LLM to revise its own forecasts until
    consistent with the Law of Total Probability, or until a round cap is hit.

    After getting initial 4 forecasts via the baseline two-batch approach, the
    bot checks the LTP residual.  If |residual| > tolerance, it shows the LLM
    its own forecasts and the discrepancy, asking it to identify which forecast
    is least defensible and revise all four.

    Supports two revision prompt framings:
    - **Causal** (default): "Which forecast doesn't fit your causal story?"
    - **Log-odds**: Shows the evidence gap in log-odds units.
    """

    MAX_ROUNDS = 5
    TOLERANCE = 0.01

    def __init__(
        self,
        model: str = "openrouter/openai/gpt-4o-mini",
        researcher_model: str = "openrouter/openai/gpt-4o-mini",
        use_log_odds: bool = False,
    ):
        super().__init__(model=model, researcher_model=researcher_model)
        self.use_log_odds = use_log_odds
        self._revision_llm = GeneralLlm(model=model, temperature=0.3, timeout=90)
        self._revision_parser = GeneralLlm(model=model, temperature=0.0, timeout=60)

    async def forecast_pair(self, pair: ConditionalPair) -> ConsistencyResult:
        cq = pair_to_conditional_question(pair)
        research = await self.run_research(cq)

        # Initial forecast via baseline two-batch approach
        reasoned: ReasonedPrediction[ConditionalPrediction] = (
            await self._run_forecast_on_conditional(cq, research)
        )
        pred = reasoned.prediction_value

        p_a = _extract_float(pred.parent, "P(A)")
        p_b = _extract_float(pred.child, "P(B)")
        p_b_given_a = float(pred.prediction_yes)
        p_b_given_na = float(pred.prediction_no)

        initial_reasoning = reasoned.reasoning
        revision_log = []

        # Revision loop
        for round_num in range(1, self.MAX_ROUNDS + 1):
            residual = p_b - p_b_given_a * p_a - p_b_given_na * (1.0 - p_a)
            revision_log.append(f"Round {round_num}: residual={residual:.4f}")

            if abs(residual) <= self.TOLERANCE:
                revision_log.append(f"Converged after {round_num - 1} revision(s).")
                break

            prompt = self._build_revision_prompt(
                pair, research, p_a, p_b, p_b_given_a, p_b_given_na, residual
            )
            revision_reasoning = await self._revision_llm.invoke(prompt)
            revised: _FourProbabilityPrediction = await structure_output(
                revision_reasoning,
                _FourProbabilityPrediction,
                model=self._revision_parser,
                num_validation_samples=1,
            )

            def clamp(x: float) -> float:
                return max(0.01, min(0.99, x))

            p_a = clamp(revised.p_a)
            p_b = clamp(revised.p_b)
            p_b_given_a = clamp(revised.p_b_given_a)
            p_b_given_na = clamp(revised.p_b_given_na)
        else:
            # Loop exhausted without breaking
            final_residual = p_b - p_b_given_a * p_a - p_b_given_na * (1.0 - p_a)
            revision_log.append(
                f"Hit max rounds ({self.MAX_ROUNDS}). Final residual={final_residual:.4f}"
            )

        llm_m = compute_consistency_metrics(p_a, p_b, p_b_given_a, p_b_given_na)
        com_m = compute_consistency_metrics(
            pair.community_p_a, pair.community_p_b,
            pair.community_p_b_given_a, pair.community_p_b_given_na,
        )

        # Store revision metadata in reasoning_parent alongside the initial reasoning
        framing = "log-odds" if self.use_log_odds else "causal"
        revision_metadata = (
            f"\n\n## Revision Log ({framing} framing)\n" + "\n".join(revision_log)
        )
        sections = _parse_reasoning_sections(initial_reasoning)

        return ConsistencyResult(
            conditional_post_id=pair.conditional_post_id,
            condition_title=pair.condition_title,
            child_title=pair.child_title,
            community_p_a=pair.community_p_a,
            community_p_b=pair.community_p_b,
            community_p_b_given_a=pair.community_p_b_given_a,
            community_p_b_given_na=pair.community_p_b_given_na,
            community_p_b_expected=com_m["p_b_expected"],
            community_consistency_error=com_m["consistency_error"],
            llm_p_a=p_a,
            llm_p_b=p_b,
            llm_p_b_given_a=p_b_given_a,
            llm_p_b_given_na=p_b_given_na,
            llm_p_b_expected=llm_m["p_b_expected"],
            llm_consistency_error=llm_m["consistency_error"],
            llm_relative_error=llm_m["relative_error"],
            llm_calibrated_error=llm_m["calibrated_error"],
            reasoning_parent=sections.get("parent", "") + revision_metadata,
            reasoning_child=sections.get("child", ""),
            reasoning_yes=sections.get("yes", ""),
            reasoning_no=sections.get("no", ""),
        )

    def _build_revision_prompt(
        self,
        pair: ConditionalPair,
        research: str,
        p_a: float,
        p_b: float,
        p_b_given_a: float,
        p_b_given_na: float,
        residual: float,
    ) -> str:
        p_b_expected = p_b_given_a * p_a + p_b_given_na * (1.0 - p_a)

        if self.use_log_odds:
            return self._log_odds_revision_prompt(
                pair, research, p_a, p_b, p_b_given_a, p_b_given_na,
                residual, p_b_expected,
            )
        return self._causal_revision_prompt(
            pair, research, p_a, p_b, p_b_given_a, p_b_given_na,
            residual, p_b_expected,
        )

    def _causal_revision_prompt(
        self, pair, research, p_a, p_b, p_b_given_a, p_b_given_na,
        residual, p_b_expected,
    ) -> str:
        return clean_indents(f"""
            You are a professional forecaster revising your probability estimates.

            --- RESEARCH ---
            {research}
            --- END RESEARCH ---

            **Parent question (A):** {pair.condition_title}
            **Child question (B):** {pair.child_title}

            Your current estimates:
            - P(A) = {p_a:.4f}
            - P(B) = {p_b:.4f}
            - P(B|A) = {p_b_given_a:.4f}
            - P(B|not A) = {p_b_given_na:.4f}

            **Consistency check:** The Law of Total Probability requires:
                P(B) = P(B|A) * P(A) + P(B|not A) * (1 - P(A))
                P(B) implied = {p_b_expected:.4f}
                P(B) stated  = {p_b:.4f}
                Discrepancy  = {residual:+.4f}

            Your estimates are inconsistent. Think about your causal model for how
            {pair.condition_title} relates to {pair.child_title}.

            Do NOT simply compute P(B) from the formula. Instead, identify which of
            your four estimates is least defensible given your causal understanding
            of the relationship, and revise it. You may adjust multiple estimates if
            that better reflects your true beliefs.

            After your reasoning, end with EXACTLY these four lines:

            P(A): <decimal>
            P(B|A): <decimal>
            P(B|not A): <decimal>
            P(B): <decimal>
        """)

    def _log_odds_revision_prompt(
        self, pair, research, p_a, p_b, p_b_given_a, p_b_given_na,
        residual, p_b_expected,
    ) -> str:
        def _to_log_odds(p: float) -> float:
            p = max(0.001, min(0.999, p))
            return math.log(p / (1.0 - p))

        lo_b = _to_log_odds(p_b)
        lo_b_expected = _to_log_odds(p_b_expected)
        evidence_gap = lo_b_expected - lo_b

        # Likelihood ratio implied by conditionals
        if p_b_given_na > 0.001:
            lr = p_b_given_a / p_b_given_na
        else:
            lr = float("inf")

        return clean_indents(f"""
            You are a professional forecaster revising your probability estimates.

            --- RESEARCH ---
            {research}
            --- END RESEARCH ---

            **Parent question (A):** {pair.condition_title}
            **Child question (B):** {pair.child_title}

            Your current estimates:
            - P(A) = {p_a:.4f}
            - P(B) = {p_b:.4f}  (log-odds: {lo_b:.3f})
            - P(B|A) = {p_b_given_a:.4f}
            - P(B|not A) = {p_b_given_na:.4f}

            **Evidence analysis:**
            Your conditionals imply a likelihood ratio of {lr:.2f} for B given A
            (vs not A). Combined with P(A), this implies P(B) should be around
            {p_b_expected:.4f} (log-odds: {lo_b_expected:.3f}).

            Your stated log-odds for B is {lo_b:.3f}, but should be {lo_b_expected:.3f}.
            You are missing {abs(evidence_gap):.3f} units of evidence
            {"in favor of" if evidence_gap > 0 else "against"} B.

            Do NOT simply compute P(B) from the formula. Identify which of your four
            estimates is least well-calibrated and revise it based on your understanding
            of the evidence.

            After your reasoning, end with EXACTLY these four lines:

            P(A): <decimal>
            P(B|A): <decimal>
            P(B|not A): <decimal>
            P(B): <decimal>
        """)


# ── Sensitivity-aware revision variant ──────────────────────────────────────


class SpreadAwareRevisionBot(BayesConsistencyBot):
    """
    Targeted revision that only fires when the conditional spread
    |P(B|A) − P(B|¬A)| is large — the regime where we empirically observe
    the highest LTP errors.

    When the spread is wide, P(A) becomes a high-leverage variable: small
    errors in P(A) are amplified through the LTP formula. This bot computes
    the sensitivity (ΔP(B)/ΔP(A) = spread) and presents it to the
    superforecaster, asking them to reconsider P(A) in light of how much
    weight it carries.

    Pairs with small spread are returned as-is (baseline forecasts), saving
    LLM calls where revision is unlikely to help.
    """

    def __init__(
        self,
        model: str = "openrouter/openai/gpt-4o-mini",
        researcher_model: str = "openrouter/openai/gpt-4o-mini",
        spread_threshold: float = 0.10,
    ):
        super().__init__(model=model, researcher_model=researcher_model)
        self.spread_threshold = spread_threshold
        self._revision_llm = GeneralLlm(model=model, temperature=0.3, timeout=90)
        self._revision_parser = GeneralLlm(model=model, temperature=0.0, timeout=60)

    async def forecast_pair(self, pair: ConditionalPair) -> ConsistencyResult:
        cq = pair_to_conditional_question(pair)
        research = await self.run_research(cq)

        # Initial forecast via baseline two-batch approach
        reasoned: ReasonedPrediction[ConditionalPrediction] = (
            await self._run_forecast_on_conditional(cq, research)
        )
        pred = reasoned.prediction_value

        p_a = _extract_float(pred.parent, "P(A)")
        p_b = _extract_float(pred.child, "P(B)")
        p_b_given_a = float(pred.prediction_yes)
        p_b_given_na = float(pred.prediction_no)

        spread = abs(p_b_given_a - p_b_given_na)
        revision_note = f"Spread = {spread:.3f} (threshold = {self.spread_threshold:.2f})"

        if spread > self.spread_threshold:
            # High-leverage regime — ask forecaster to reconsider
            residual = p_b - p_b_given_a * p_a - p_b_given_na * (1.0 - p_a)
            revision_note += f", residual = {residual:+.4f} → revision triggered"

            prompt = self._build_sensitivity_prompt(
                pair, research, p_a, p_b, p_b_given_a, p_b_given_na, spread
            )
            revision_reasoning = await self._revision_llm.invoke(prompt)
            revised: _FourProbabilityPrediction = await structure_output(
                revision_reasoning,
                _FourProbabilityPrediction,
                model=self._revision_parser,
                num_validation_samples=1,
            )

            def clamp(x: float) -> float:
                return max(0.01, min(0.99, x))

            p_a = clamp(revised.p_a)
            p_b = clamp(revised.p_b)
            p_b_given_a = clamp(revised.p_b_given_a)
            p_b_given_na = clamp(revised.p_b_given_na)
        else:
            revision_note += " → no revision needed"

        llm_m = compute_consistency_metrics(p_a, p_b, p_b_given_a, p_b_given_na)
        com_m = compute_consistency_metrics(
            pair.community_p_a, pair.community_p_b,
            pair.community_p_b_given_a, pair.community_p_b_given_na,
        )
        sections = _parse_reasoning_sections(reasoned.reasoning)

        return ConsistencyResult(
            conditional_post_id=pair.conditional_post_id,
            condition_title=pair.condition_title,
            child_title=pair.child_title,
            community_p_a=pair.community_p_a,
            community_p_b=pair.community_p_b,
            community_p_b_given_a=pair.community_p_b_given_a,
            community_p_b_given_na=pair.community_p_b_given_na,
            community_p_b_expected=com_m["p_b_expected"],
            community_consistency_error=com_m["consistency_error"],
            llm_p_a=p_a,
            llm_p_b=p_b,
            llm_p_b_given_a=p_b_given_a,
            llm_p_b_given_na=p_b_given_na,
            llm_p_b_expected=llm_m["p_b_expected"],
            llm_consistency_error=llm_m["consistency_error"],
            llm_relative_error=llm_m["relative_error"],
            llm_calibrated_error=llm_m["calibrated_error"],
            reasoning_parent=sections.get("parent", "") + f"\n\n## Revision Note\n{revision_note}",
            reasoning_child=sections.get("child", ""),
            reasoning_yes=sections.get("yes", ""),
            reasoning_no=sections.get("no", ""),
        )

    def _build_sensitivity_prompt(
        self,
        pair: ConditionalPair,
        research: str,
        p_a: float,
        p_b: float,
        p_b_given_a: float,
        p_b_given_na: float,
        spread: float,
    ) -> str:
        p_b_expected = p_b_given_a * p_a + p_b_given_na * (1.0 - p_a)
        residual = p_b - p_b_expected

        # What P(A) would need to be to make the stated P(B) consistent
        p_a_consistent = (p_b - p_b_given_na) / spread if spread > 0.001 else p_a
        p_a_consistent = max(0.01, min(0.99, p_a_consistent))

        sensitivity_5pp = spread * 0.05

        return clean_indents(f"""
            You are a professional forecaster reconsidering your estimates.

            --- RESEARCH ---
            {research}
            --- END RESEARCH ---

            **Parent question (A):** {pair.condition_title}
            {pair.condition_resolution_criteria}

            **Child question (B):** {pair.child_title}
            {pair.child_resolution_criteria}

            Today is {datetime.now().strftime('%Y-%m-%d')}.

            Your current estimates:
            - P(A) = {p_a:.4f}
            - P(B) = {p_b:.4f}
            - P(B|A) = {p_b_given_a:.4f}
            - P(B|not A) = {p_b_given_na:.4f}

            **Sensitivity analysis:** Your conditional estimates are far apart
            (spread = {spread:.3f}), which means P(A) is a high-leverage variable
            in this pair. Concretely:

            - A 5 percentage-point change in P(A) shifts the implied P(B) by
              {sensitivity_5pp:.3f} ({sensitivity_5pp*100:.1f}pp).
            - Your stated P(B) = {p_b:.4f}, but your other three values imply
              P(B) should be {p_b_expected:.4f} (discrepancy: {residual:+.4f}).
            - For your stated P(B) to be consistent, P(A) would need to be
              ~{p_a_consistent:.2f} (you said {p_a:.4f}).

            Given this sensitivity, please reconsider carefully:
            1. Is your P(A) well-calibrated, or did you over/underestimate the
               likelihood of the parent event?
            2. Are your conditionals P(B|A) and P(B|not A) truly that far apart,
               or is the causal link between A and B weaker than you assumed?
            3. Does your P(B) properly reflect the base rate, or were you
               anchored by one of the conditional scenarios?

            Revise whichever estimates you find least defensible. You may adjust
            any or all of the four values.

            After your reasoning, end with EXACTLY these four lines:

            P(A): <decimal>
            P(B|A): <decimal>
            P(B|not A): <decimal>
            P(B): <decimal>
        """)

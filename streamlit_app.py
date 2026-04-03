"""
Streamlit app for Bayesian Consistency Forecasting.

Lets users enter two binary questions (condition A, child B), forecasts all 4
probabilities via BayesConsistencyBot, applies KL-divergence projection for
Bayesian consistency, and displays a probability square visualization.
"""

import asyncio
import os
import sys

import nest_asyncio

nest_asyncio.apply()

# ── Path & env setup (mirrors bayes_bot.py) ─────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TEMPLATE_DIR = os.path.join(_THIS_DIR, "..", "metac-bot-template")
sys.path.insert(0, _TEMPLATE_DIR)

import dotenv

dotenv.load_dotenv(os.path.join(_THIS_DIR, ".env"))
dotenv.load_dotenv(os.path.join(_TEMPLATE_DIR, ".env"))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from pydantic import BaseModel, Field

from src.bayes_bot import (
    BayesConsistencyBot,
    ConditionalPair,
    compute_consistency_metrics,
)
from src.experiment_pipelines import _kl_project_quadruple
from forecasting_tools import GeneralLlm, clean_indents


# ── Pydantic model for LLM-generated resolution criteria ────────────────────


class QuestionCriteria(BaseModel):
    """Metaculus-style resolution criteria for a binary question."""

    description: str = Field(
        description="A 2-4 sentence background description providing context for the question."
    )
    resolution_criteria: str = Field(
        description=(
            "Clear, unambiguous resolution criteria starting with 'Resolves Yes if...'. "
            "Include a specific timeframe and verifiable conditions."
        )
    )
    fine_print: str = Field(
        description="Edge cases, clarifications, and exceptions (1-3 sentences)."
    )


# ── LLM criteria generation ─────────────────────────────────────────────────


async def generate_criteria(question_title: str, llm: GeneralLlm) -> QuestionCriteria:
    """Use an LLM to generate Metaculus-style resolution criteria for a question."""
    prompt = clean_indents(
        f"""
        You are a Metaculus question writer. Given the following binary question title,
        generate clear resolution criteria.

        Question: "{question_title}"

        Respond with a JSON object containing:
        - "description": 2-4 sentence background context
        - "resolution_criteria": starts with "Resolves Yes if..." with a specific
          timeframe and verifiable conditions
        - "fine_print": edge cases and clarifications (1-3 sentences)

        Return ONLY valid JSON, no markdown fences.
        """
    )
    raw = await llm.invoke(prompt)
    # Parse with pydantic — try JSON first, fall back to manual extraction
    import json

    try:
        data = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
        return QuestionCriteria(**data)
    except Exception:
        return QuestionCriteria(
            description=f"Question about whether: {question_title}",
            resolution_criteria=f"Resolves Yes if {question_title.rstrip('?').lower()} occurs.",
            fine_print="Standard resolution applies.",
        )


async def generate_both_criteria(
    title_a: str, title_b: str, model: str
) -> tuple[QuestionCriteria, QuestionCriteria]:
    """Generate resolution criteria for both questions in parallel."""
    llm = GeneralLlm(model=model, temperature=0.3, timeout=60)
    crit_a, crit_b = await asyncio.gather(
        generate_criteria(title_a, llm),
        generate_criteria(title_b, llm),
    )
    return crit_a, crit_b


# ── ConditionalPair construction ─────────────────────────────────────────────


def build_pair(
    title_a: str,
    title_b: str,
    criteria_a: QuestionCriteria,
    criteria_b: QuestionCriteria,
) -> ConditionalPair:
    """Build a ConditionalPair from user input + LLM-generated criteria."""
    return ConditionalPair(
        conditional_post_id=99999,
        condition_post_id=99998,
        child_post_id=99997,
        condition_title=title_a,
        condition_description=criteria_a.description,
        condition_resolution_criteria=criteria_a.resolution_criteria,
        condition_fine_print=criteria_a.fine_print,
        child_title=title_b,
        child_description=criteria_b.description,
        child_resolution_criteria=criteria_b.resolution_criteria,
        child_fine_print=criteria_b.fine_print,
        community_p_a=0.5,
        community_p_b=0.5,
        community_p_b_given_a=0.5,
        community_p_b_given_na=0.5,
    )


# ── Probability square visualization ────────────────────────────────────────


def draw_probability_square(
    p_a: float,
    p_b_given_a: float,
    p_b_given_na: float,
    label_a: str = "A",
    label_b: str = "B",
) -> plt.Figure:
    """
    Draw a unit probability square showing the joint distribution.

    Left column (width P(A)):  split at P(B|A)
    Right column (width 1-P(A)): split at P(B|not-A)
    """
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    # Colors matching reference image
    c_b_given_a = "#E8922D"  # orange — P(B|A)
    c_nb_given_a = "#0E9AA7"  # teal — P(not-B|A)
    c_b_given_na = "#1B2A4A"  # navy — P(B|not-A)
    c_nb_given_na = "#3D3D3D"  # dark gray — P(not-B|not-A)

    pna = 1.0 - p_a

    # Bottom-left: P(B|A) region
    ax.add_patch(mpatches.Rectangle((0, 0), p_a, p_b_given_a, color=c_b_given_a))
    # Top-left: P(not-B|A) region
    ax.add_patch(
        mpatches.Rectangle((0, p_b_given_a), p_a, 1 - p_b_given_a, color=c_nb_given_a)
    )
    # Bottom-right: P(B|not-A) region
    ax.add_patch(
        mpatches.Rectangle((p_a, 0), pna, p_b_given_na, color=c_b_given_na)
    )
    # Top-right: P(not-B|not-A) region
    ax.add_patch(
        mpatches.Rectangle(
            (p_a, p_b_given_na), pna, 1 - p_b_given_na, color=c_nb_given_na
        )
    )

    # Labels inside each rectangle
    def _label(x, y, text, color="white", fontsize=7):
        ax.text(x, y, text, ha="center", va="center", color=color, fontsize=fontsize,
                fontweight="bold", wrap=True)

    # Joint probabilities (area of each rectangle)
    p_ab = p_a * p_b_given_a
    p_a_nb = p_a * (1 - p_b_given_a)
    p_na_b = pna * p_b_given_na
    p_na_nb = pna * (1 - p_b_given_na)

    # Only label if region is large enough
    if p_a > 0.08 and p_b_given_a > 0.08:
        _label(p_a / 2, p_b_given_a / 2, f"P({label_a}, {label_b})\n{p_ab:.0%}")
    if p_a > 0.08 and (1 - p_b_given_a) > 0.08:
        _label(
            p_a / 2,
            p_b_given_a + (1 - p_b_given_a) / 2,
            f"P({label_a}, \u00ac{label_b})\n{p_a_nb:.0%}",
        )
    if pna > 0.08 and p_b_given_na > 0.08:
        _label(
            p_a + pna / 2,
            p_b_given_na / 2,
            f"P(\u00ac{label_a}, {label_b})\n{p_na_b:.0%}",
        )
    if pna > 0.08 and (1 - p_b_given_na) > 0.08:
        _label(
            p_a + pna / 2,
            p_b_given_na + (1 - p_b_given_na) / 2,
            f"P(\u00ac{label_a}, \u00ac{label_b})\n{p_na_nb:.0%}",
        )

    # Bracket annotations at top for P(A) and P(not-A)
    bracket_y = 1.06
    if p_a > 0.05:
        ax.annotate(
            "",
            xy=(0, bracket_y),
            xytext=(p_a, bracket_y),
            arrowprops=dict(arrowstyle="|-|", color="black", lw=1.5),
            annotation_clip=False,
        )
        ax.text(p_a / 2, bracket_y + 0.04, f"P({label_a}) = {p_a:.0%}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")
    if pna > 0.05:
        ax.annotate(
            "",
            xy=(p_a, bracket_y),
            xytext=(1.0, bracket_y),
            arrowprops=dict(arrowstyle="|-|", color="black", lw=1.5),
            annotation_clip=False,
        )
        ax.text(p_a + pna / 2, bracket_y + 0.04, f"P(\u00ac{label_a}) = {pna:.0%}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    fig.tight_layout()
    return fig


# ── Forecasting pipeline ────────────────────────────────────────────────────


async def run_forecast(pair: ConditionalPair, model: str):
    """Run the full forecasting pipeline: predict + KL-project."""
    bot = BayesConsistencyBot(model=model, researcher_model=model)
    result = await bot.forecast_pair(pair)

    # KL-divergence projection for Bayesian consistency
    proj = _kl_project_quadruple(
        result.llm_p_a, result.llm_p_b,
        result.llm_p_b_given_a, result.llm_p_b_given_na,
    )
    proj_metrics = compute_consistency_metrics(*proj)

    return result, proj, proj_metrics


# ── Streamlit UI ─────────────────────────────────────────────────────────────


def main():
    st.set_page_config(page_title="Bayesian Consistency Forecaster", layout="wide")

    st.title("Bayesian Consistency Forecaster")
    st.markdown(
        "Enter a **condition question (A)** and a **child question (B)** to forecast "
        "how B's probability changes depending on whether A happens."
    )

    # ── Inputs ───────────────────────────────────────────────────────────────
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        st.subheader("Question A (condition)")
        title_a = st.text_input(
            "Question A",
            placeholder="e.g. Will Trump win 2028?",
            label_visibility="collapsed",
        )
        st.markdown(
            "<div style='text-align:center; font-size:1.3em; margin:-0.5em 0 -0.5em 0;'>"
            "\u2193 <em>Will A affect B?</em> \u2193</div>",
            unsafe_allow_html=True,
        )
        st.subheader("Question B (child)")
        title_b = st.text_input(
            "Question B",
            placeholder="e.g. Will the stock market crash by 2030?",
            label_visibility="collapsed",
        )
    with col_btn:
        st.markdown(
            """
            <style>
            div[data-testid="stColumn"]:last-child button[kind="primary"] {
                height: 220px;
                font-size: 1.3em;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 1.2em'></div>", unsafe_allow_html=True)
        forecast_clicked = st.button(
            "\U0001f52e Forecast",
            type="primary",
            use_container_width=True,
        )

    model = "openrouter/openai/gpt-4o-mini"

    # ── Run pipeline ─────────────────────────────────────────────────────────
    if forecast_clicked and title_a and title_b:
        with st.status("Running forecast pipeline...", expanded=True) as status:
            # Step 1: Generate resolution criteria
            st.write("Generating resolution criteria...")
            loop = asyncio.get_event_loop()
            criteria_a, criteria_b = loop.run_until_complete(
                generate_both_criteria(title_a, title_b, model)
            )

            # Step 2: Build pair
            pair = build_pair(title_a, title_b, criteria_a, criteria_b)

            # Step 3: Forecast
            st.write("Forecasting probabilities (this may take 30-60 seconds)...")
            result, proj, proj_metrics = loop.run_until_complete(
                run_forecast(pair, model)
            )

            status.update(label="Forecast complete!", state="complete")

        # Store in session state
        st.session_state["result"] = result
        st.session_state["proj"] = proj
        st.session_state["proj_metrics"] = proj_metrics
        st.session_state["criteria_a"] = criteria_a
        st.session_state["criteria_b"] = criteria_b
        st.session_state["title_a"] = title_a
        st.session_state["title_b"] = title_b

    # ── Display results ──────────────────────────────────────────────────────
    if "result" not in st.session_state:
        return

    result = st.session_state["result"]
    proj = st.session_state["proj"]
    proj_metrics = st.session_state["proj_metrics"]
    criteria_a = st.session_state["criteria_a"]
    criteria_b = st.session_state["criteria_b"]
    t_a = st.session_state["title_a"]
    t_b = st.session_state["title_b"]

    st.divider()

    # Generated criteria
    with st.expander("Generated Resolution Criteria"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Question A:** {t_a}")
            st.markdown(f"*Description:* {criteria_a.description}")
            st.markdown(f"*Resolution:* {criteria_a.resolution_criteria}")
            st.markdown(f"*Fine print:* {criteria_a.fine_print}")
        with c2:
            st.markdown(f"**Question B:** {t_b}")
            st.markdown(f"*Description:* {criteria_b.description}")
            st.markdown(f"*Resolution:* {criteria_b.resolution_criteria}")
            st.markdown(f"*Fine print:* {criteria_b.fine_print}")

    # Probability square (KL-projected only)
    proj_pa, proj_pb, proj_pba, proj_pbna = proj

    st.subheader("Probability Square")
    _, sq_center, _ = st.columns([1, 2, 1])
    with sq_center:
        fig_proj = draw_probability_square(
            proj_pa,
            proj_pba,
            proj_pbna,
            label_a="A",
            label_b="B",
        )
        st.pyplot(fig_proj)
        plt.close(fig_proj)

    # Raw predictions
    st.subheader("Raw LLM Predictions")
    raw_cols = st.columns(5)
    raw_cols[0].metric("P(A)", f"{result.llm_p_a:.1%}")
    raw_cols[1].metric("P(B)", f"{result.llm_p_b:.1%}")
    raw_cols[2].metric("P(B|A)", f"{result.llm_p_b_given_a:.1%}")
    raw_cols[3].metric("P(B|\u00acA)", f"{result.llm_p_b_given_na:.1%}")
    raw_cols[4].metric(
        "Consistency Error",
        f"{result.llm_consistency_error:.4f}",
    )

    # LTP check
    p_b_expected = (
        result.llm_p_b_given_a * result.llm_p_a
        + result.llm_p_b_given_na * (1 - result.llm_p_a)
    )
    st.caption(
        f"Law of Total Probability check: "
        f"P(B|A)\u00b7P(A) + P(B|\u00acA)\u00b7P(\u00acA) = "
        f"{result.llm_p_b_given_a:.3f}\u00b7{result.llm_p_a:.3f} + "
        f"{result.llm_p_b_given_na:.3f}\u00b7{1 - result.llm_p_a:.3f} = "
        f"**{p_b_expected:.4f}** vs P(B) = **{result.llm_p_b:.4f}**"
    )

    # Bayes-consistent projections
    st.subheader("Bayes-Consistent Projections")
    kl_cols = st.columns(5)
    kl_cols[0].metric("P(A)", f"{proj_pa:.1%}", f"{proj_pa - result.llm_p_a:+.3f}")
    kl_cols[1].metric("P(B)", f"{proj_pb:.1%}", f"{proj_pb - result.llm_p_b:+.3f}")
    kl_cols[2].metric("P(B|A)", f"{proj_pba:.1%}", f"{proj_pba - result.llm_p_b_given_a:+.3f}")
    kl_cols[3].metric("P(B|\u00acA)", f"{proj_pbna:.1%}", f"{proj_pbna - result.llm_p_b_given_na:+.3f}")
    kl_cols[4].metric(
        "Consistency Error",
        f"{proj_metrics['consistency_error']:.4f}",
        f"{proj_metrics['consistency_error'] - result.llm_consistency_error:+.4f}",
    )

    # Reasoning
    st.subheader("LLM Reasoning")
    with st.expander("Parent Question (A) Reasoning"):
        st.markdown(result.reasoning_parent or "*No reasoning captured*")
    with st.expander("Child Question (B) Reasoning"):
        st.markdown(result.reasoning_child or "*No reasoning captured*")
    with st.expander("P(B|A) — If A happens"):
        st.markdown(result.reasoning_yes or "*No reasoning captured*")
    with st.expander("P(B|\u00acA) — If A does not happen"):
        st.markdown(result.reasoning_no or "*No reasoning captured*")


if __name__ == "__main__":
    main()

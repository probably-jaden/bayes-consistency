"""
run_trial.py — CLI entry point for the Bayes Consistency trial.

Loads conditional question pairs from the CSV, samples a subset, forecasts
all four probabilities (P(A), P(B), P(B|A), P(B|¬A)) with an LLM, computes
Bayes consistency scores, and saves results.

Usage (run from within the metac-bot-template Poetry environment):

    cd trial-bayes-consistency
    poetry -C ../metac-bot-template run python run_trial.py

Options:
    --n-sample INT   Number of pairs to forecast (default: 25)
    --model STR      LiteLLM model string (default: openrouter/openai/gpt-4o-mini)
    --output STR     Output CSV path (default: results/llm_forecasts_<timestamp>.csv)
    --seed INT       Random seed for sampling (default: 42)

Output:
    results/<name>.csv        Summary CSV (one row per pair, no reasoning text)
    results/<name>_full.json  Full JSON including LLM reasoning per sub-question
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bayes_bot import (
    BayesConsistencyBot,
    ConditionalsFirstBot,
    JointPromptBot,
    RecursiveRevisionBot,
    SeparateResearchBot,
    load_pairs_from_csv,
)

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "metaculus_conditionals_2026-03.csv")
DEFAULT_N_SAMPLE = 2


def _check_api_key(model: str) -> None:
    """Fail fast with a clear message if the required API key is missing."""
    import os
    checks = {
        "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/keys"),
        "anthropic": ("ANTHROPIC_API_KEY", "https://console.anthropic.com/keys"),
        "openai": ("OPENAI_API_KEY", "https://platform.openai.com/api-keys"),
    }
    for prefix, (env_var, url) in checks.items():
        if model.startswith(prefix) or model.startswith(f"{prefix}/"):
            key = os.environ.get(env_var, "")
            if not key or key.startswith("your_"):
                print(
                    f"\nERROR: {env_var} is not set in your .env file.\n"
                    f"  1. Copy .env.template to .env\n"
                    f"  2. Add your key from {url}\n"
                    f"  3. Re-run this script\n"
                )
                raise SystemExit(1)
            return


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # Suppress noisy LiteLLM logs (cost mapping warnings for openrouter models)
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("LiteLLM").propagate = False
    # Suppress "GLOBAL_TRACE_PROVIDER is not set" warnings from forecasting_tools
    logging.getLogger("forecasting_tools.ai_models.agent_wrappers").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(
        description="Run the Bayes Consistency forecasting trial",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=DEFAULT_N_SAMPLE,
        help="Number of question pairs to forecast (small values are faster for quick iteration)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openrouter/openai/gpt-4o-mini",
        help=(
            "LiteLLM model string used for forecasting. Examples: "
            "openrouter/openai/gpt-4o, "
            "openrouter/anthropic/claude-opus-4-6, "
            "anthropic/claude-opus-4-6, "
            "openai/gpt-4o-mini"
        ),
    )
    parser.add_argument(
        "--researcher-model",
        type=str,
        default=None,
        help=(
            "LiteLLM model string used for research (defaults to --model if not set). "
            "Set to a cheaper model to reduce cost while keeping a strong forecaster. "
            "Example: --model openrouter/anthropic/claude-opus-4-6 "
            "--researcher-model openrouter/openai/gpt-4o-mini"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(
            "results",
            f"llm_forecasts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        ),
        help="Output CSV path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for pair sampling",
    )
    parser.add_argument(
        "--separate-research",
        action="store_true",
        default=False,
        help=(
            "Use SeparateResearchBot: run independent research for P(A) and P(B) "
            "instead of a single research call on the conditional question. "
            "This prevents the researcher from framing its response around the "
            "A→B causal relationship when providing context for marginal forecasts."
        ),
    )
    parser.add_argument(
        "--joint-prompt",
        action="store_true",
        default=False,
        help=(
            "Use JointPromptBot: forecast all four probabilities in a single joint "
            "LLM call with an explicit Law of Total Probability self-check instruction, "
            "instead of four independent calls."
        ),
    )
    parser.add_argument(
        "--conditionals-first",
        action="store_true",
        default=False,
        help=(
            "Use ConditionalsFirstBot: elicit P(B|A) and P(B|not A) before P(A) "
            "and P(B), so the LLM thinks through the causal structure before "
            "committing to marginal base rates."
        ),
    )
    parser.add_argument(
        "--recursive-revision",
        action="store_true",
        default=False,
        help=(
            "Use RecursiveRevisionBot: iteratively revise forecasts until "
            "consistent with Bayes rule (causal reasoning framing)."
        ),
    )
    parser.add_argument(
        "--recursive-revision-logodds",
        action="store_true",
        default=False,
        help=(
            "Use RecursiveRevisionBot with log-odds/evidence-gap framing: "
            "show the LLM the evidence gap in log-odds units during revision."
        ),
    )
    args = parser.parse_args()

    _check_api_key(args.model)
    researcher_model = args.researcher_model or args.model
    if args.researcher_model:
        _check_api_key(args.researcher_model)

    pairs = load_pairs_from_csv(CSV_PATH)
    logging.info(f"Loaded {len(pairs)} pairs from CSV.")

    bot_flags = sum([
        args.separate_research, args.joint_prompt, args.conditionals_first,
        args.recursive_revision, args.recursive_revision_logodds,
    ])
    if bot_flags > 1:
        parser.error(
            "Only one bot variant flag may be used at a time: "
            "--separate-research, --joint-prompt, --conditionals-first, "
            "--recursive-revision, --recursive-revision-logodds"
        )
    if args.separate_research:
        BotClass = SeparateResearchBot
    elif args.joint_prompt:
        BotClass = JointPromptBot
    elif args.conditionals_first:
        BotClass = ConditionalsFirstBot
    elif args.recursive_revision:
        bot = RecursiveRevisionBot(model=args.model, researcher_model=researcher_model, use_log_odds=False)
    elif args.recursive_revision_logodds:
        bot = RecursiveRevisionBot(model=args.model, researcher_model=researcher_model, use_log_odds=True)
    else:
        BotClass = BayesConsistencyBot
    if not (args.recursive_revision or args.recursive_revision_logodds):
        bot = BotClass(model=args.model, researcher_model=researcher_model)
    results = asyncio.run(bot.forecast_pairs(pairs, n_sample=args.n_sample, seed=args.seed))

    if not results:
        logging.warning(
            "No successful forecasts were produced (all pairs failed). "
            "Saving header-only output files for diagnosis."
        )
        logging.warning(
            "Common cause: authentication failure or model mapping warnings. "
            "Check .env API key and choose a supported model."
        )

    BayesConsistencyBot.save_results(results, args.output)
    BayesConsistencyBot.save_full_results(results, args.output)

    print(f"Results saved to: {args.output}")

    if not results:
        print("No successful forecast rows were generated. See log for per-pair errors.")


if __name__ == "__main__":
    main()

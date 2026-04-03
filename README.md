# bayes-consistency

Play with an [app version](https://consistent-bayes.streamlit.app/) 

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Copy `.env.template` to `.env` and fill in your API keys (at least OPENROUTER_API_KEY)
3. Run: `python run_trial.py`

## Task

Measure the propensity of forecasting bots to make forecasts that respect Bayes' Rule.

## Background

A well-calibrated forecaster should produce forecasts that are internally consistent with the axioms of probability and, in particular, with Bayes' Rule. See [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem):

$$
P(B \mid A) = \frac{P(A \mid B) \, P(B)}{P(A)}
$$

Metaculus has many [conditional questions](https://www.metaculus.com/questions/?forecast_type=conditional), which link a "parent" question with forecast $P(A)$ to a "child" question with forecast $P(B)$, and elicitis the two conditional probabilities $P(B \mid A)$ and $P(B \mid ¬A)$. Forecasters often make forecasts that do not respect Bayes' theorem.

## Resources

A csv of conditional questions and forecasts from Metaculus:

The dataset of conditional questions and community forecasts was sourced from Metaculus in March 2026.

## Key Steps

1. Propose a formula to quantify the consistency of a forecaster in the situation above.
2. Forecasts: have an llm forecast quadruples $P(A)$, $P(B)$, $P(B \mid A)$ and $P(B \mid ¬A)$ (or some other quadruple that contains the same information) for the list of example questions provided below.
3. Consistency: compute your proposed consistency score for each quadruple.
4. Explore patterns.
5. Improvements: improve the consistency of the llm.

## Deliverables

- Documented code that:
    - Takes some pairs of questions.
    - Makes the four forecasts discussed above.
    - Computes and reports consistency violations.
- An example input/output pair.
- A short written report (markdown file, notebook, …, your choice) containing:
    - Interesting results.
    - Main takeaways.
    - Next steps.

## Evaluation Criteria

We will evaluate both the code, its outputs, and your analysis:

- Code quality and readability.
- Consistency of the bots.
- Soundness of the consistency score formula chosen.
- Soundness of the patterns (or lack of patterns) found.
- Soundness of the improvement ideas.
- Written report quality and readability.

**Partial implementation:** given the (relatively) short time limit, feel free to prioritise.

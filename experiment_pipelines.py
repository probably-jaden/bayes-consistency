"""
experiment_pipelines.py
-----------------------
Pipelines that generate new experiment CSVs without running the LLM.

Imported by analysis.ipynb §9.
"""

import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from bayes_bot import compute_consistency_metrics, project_to_ltp_constraint


def apply_ltp_projection(
    input_csv: str,
    output_csv: str,
    source_experiment_name: str = "baseline",
    tol: float = 0.01,
    max_iter: int = 50,
) -> pd.DataFrame:
    """
    Load an experiment CSV, apply iterative LTP projection to each quadruple,
    and save the result as a new experiment CSV in the same 17-column format.

    The projection finds the Euclidean nearest point to the original quadruple
    that satisfies P(B) = P(B|A)·P(A) + P(B|¬A)·(1−P(A)).

    Args:
        input_csv  : path to the source experiment CSV
        output_csv : where to write the projected CSV
        tol        : convergence tolerance (stop when |residual| ≤ tol)
        max_iter   : iteration cap

    Returns the projected DataFrame.
    """
    df = pd.read_csv(input_csv)
    records = []

    for _, row in df.iterrows():
        orig = (row['llm_p_a'], row['llm_p_b'],
                row['llm_p_b_given_a'], row['llm_p_b_given_na'])

        pa, pb, pba, pbna, _ = project_to_ltp_constraint(
            *orig, tol=tol, max_iter=max_iter
        )

        m = compute_consistency_metrics(pa, pb, pba, pbna)
        records.append({
            'conditional_post_id':         row['conditional_post_id'],
            'condition_title':             row['condition_title'],
            'child_title':                 row['child_title'],
            'community_p_a':               row['community_p_a'],
            'community_p_b':               row['community_p_b'],
            'community_p_b_given_a':       row['community_p_b_given_a'],
            'community_p_b_given_na':      row['community_p_b_given_na'],
            'community_p_b_expected':      row['community_p_b_expected'],
            'community_consistency_error': row['community_consistency_error'],
            'llm_p_a':                     pa,
            'llm_p_b':                     pb,
            'llm_p_b_given_a':             pba,
            'llm_p_b_given_na':            pbna,
            'llm_p_b_expected':            m['p_b_expected'],
            'llm_consistency_error':       m['consistency_error'],
            'llm_relative_error':          m['relative_error'],
            'llm_calibrated_error':        m['calibrated_error'],
        })

    projected_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    projected_df.to_csv(output_csv, index=False)
    print(f"Saved {len(projected_df)} projected rows → {output_csv}")
    return projected_df


# ── KL-divergence projection ────────────────────────────────────────────────


def _kl_bernoulli(p_orig: float, p_new: float) -> float:
    """KL(Bern(p_orig) || Bern(p_new)) — divergence from original to new."""
    EPS = 1e-12
    p, q = np.clip(p_orig, EPS, 1 - EPS), np.clip(p_new, EPS, 1 - EPS)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def _kl_project_quadruple(
    p_a: float,
    p_b: float,
    p_b_given_a: float,
    p_b_given_na: float,
) -> tuple[float, float, float, float]:
    """
    Find the nearest point (in sum-of-Bernoulli-KL sense) on the LTP surface.

    Minimises  sum_i KL(Bern(p_i_orig) || Bern(p_i_new))
    subject to p_b_new = p_b_given_a_new * p_a_new + p_b_given_na_new * (1 - p_a_new)
    with bounds [0.01, 0.99].
    """
    CLIP_LO, CLIP_HI = 0.01, 0.99
    x0 = np.array([p_a, p_b, p_b_given_a, p_b_given_na])
    orig = x0.copy()

    def objective(x):
        return sum(
            _kl_bernoulli(orig[i], x[i]) for i in range(4)
        )

    def ltp_constraint(x):
        # Must equal zero: p_b - p_b_given_a * p_a - p_b_given_na * (1 - p_a)
        return x[1] - x[2] * x[0] - x[3] * (1.0 - x[0])

    bounds = [(CLIP_LO, CLIP_HI)] * 4

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": ltp_constraint},
        options={"maxiter": 200, "ftol": 1e-12},
    )

    if result.success:
        return tuple(np.clip(result.x, CLIP_LO, CLIP_HI))

    # Fallback to Euclidean projection if SLSQP fails
    pa, pb, pba, pbna, _ = project_to_ltp_constraint(p_a, p_b, p_b_given_a, p_b_given_na)
    return (pa, pb, pba, pbna)


def apply_kl_projection(
    input_csv: str,
    output_csv: str,
    source_experiment_name: str = "baseline",
) -> pd.DataFrame:
    """
    Load an experiment CSV, apply KL-divergence projection to each quadruple,
    and save the result as a new experiment CSV in the same 17-column format.

    KL projection penalises moving extreme probabilities (near 0 or 1) more
    heavily than middling ones, which preserves sharpness better than Euclidean.
    """
    df = pd.read_csv(input_csv)
    records = []

    for _, row in df.iterrows():
        orig = (row['llm_p_a'], row['llm_p_b'],
                row['llm_p_b_given_a'], row['llm_p_b_given_na'])

        pa, pb, pba, pbna = _kl_project_quadruple(*orig)

        m = compute_consistency_metrics(pa, pb, pba, pbna)
        records.append({
            'conditional_post_id':         row['conditional_post_id'],
            'condition_title':             row['condition_title'],
            'child_title':                 row['child_title'],
            'community_p_a':               row['community_p_a'],
            'community_p_b':               row['community_p_b'],
            'community_p_b_given_a':       row['community_p_b_given_a'],
            'community_p_b_given_na':      row['community_p_b_given_na'],
            'community_p_b_expected':      row['community_p_b_expected'],
            'community_consistency_error': row['community_consistency_error'],
            'llm_p_a':                     pa,
            'llm_p_b':                     pb,
            'llm_p_b_given_a':             pba,
            'llm_p_b_given_na':            pbna,
            'llm_p_b_expected':            m['p_b_expected'],
            'llm_consistency_error':       m['consistency_error'],
            'llm_relative_error':          m['relative_error'],
            'llm_calibrated_error':        m['calibrated_error'],
        })

    kl_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    kl_df.to_csv(output_csv, index=False)
    print(f"Saved {len(kl_df)} KL-projected rows → {output_csv}")
    return kl_df

"""
experiment_framework.py
-----------------------
Reusable scaffolding for comparing Bayes-consistency forecasting experiments.

Imported by analysis.ipynb §8.  Edit EXPERIMENTS in the notebook to register
new experiments; everything here is pure utility.
"""

import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import t as t_dist
from statsmodels.stats.anova import AnovaRM

from src.bayes_bot import load_pairs_from_csv, compute_consistency_metrics

warnings.filterwarnings('ignore', category=FutureWarning)

RNG = np.random.default_rng(42)          # reproducible bootstraps
COMMUNITY_CSV = 'data/metaculus_conditionals_2026-03.csv'


# ── Data loading ──────────────────────────────────────────────────────────────

def load_experiment(exp: dict, community_csv: str = COMMUNITY_CSV):
    """
    Load one experiment's CSV and join community predictions.
    Returns None (with a warning) if the CSV doesn't exist yet — allows
    the registry to be pre-populated before results are ready.

    Output columns (superset of run_trial.py format):
        conditional_post_id, condition_title, child_title,
        llm_p_a, llm_p_b, llm_p_b_given_a, llm_p_b_given_na,
        llm_p_b_expected, llm_consistency_error,
        llm_relative_error, llm_calibrated_error,
        community_p_a, community_p_b, community_p_b_given_a, community_p_b_given_na,
        community_consistency_error
    """
    csv_path = exp["csv_path"]
    if not os.path.exists(csv_path):
        print(f"  [SKIP] {exp['name']}: CSV not found at {csv_path!r}")
        return None

    df = pd.read_csv(csv_path)

    # Join community predictions if the CSV doesn't already carry them
    if 'community_p_a' not in df.columns:
        comm_pairs = load_pairs_from_csv(community_csv)
        comm_lookup = {
            p.conditional_post_id: {
                'community_p_a':          p.community_p_a,
                'community_p_b':          p.community_p_b,
                'community_p_b_given_a':  p.community_p_b_given_a,
                'community_p_b_given_na': p.community_p_b_given_na,
            }
            for p in comm_pairs
        }
        comm_df = (
            pd.DataFrame.from_dict(comm_lookup, orient='index')
            .reset_index()
            .rename(columns={'index': 'conditional_post_id'})
        )
        df = df.merge(comm_df, on='conditional_post_id', how='left')

    # Recompute community consistency error if missing
    if 'community_consistency_error' not in df.columns:
        df['community_consistency_error'] = df.apply(
            lambda row: compute_consistency_metrics(
                row['community_p_a'], row['community_p_b'],
                row['community_p_b_given_a'], row['community_p_b_given_na'],
            )['consistency_error'],
            axis=1,
        )

    df.attrs['experiment'] = exp
    return df


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_experiment_metrics(df, oracle_df=None) -> dict:
    """
    Compute all metrics for one experiment.
    Returns a dict mapping metric_name -> numpy array (one value per question row).

    Metric groups:
      consistency  : LTP error variants
      community    : proximity to community predictions (always available)
      frontier     : proximity to oracle model (only if oracle_df provided)
      sharpness    : resolution proxy without outcome data
      coherence    : ordering-consistency check
    """
    n = len(df)
    m = {}

    # ── Bayesian consistency ──────────────────────────────────────────────
    m['consistency_error']  = df['llm_consistency_error'].values.astype(float)
    m['relative_error']     = (df['llm_relative_error'].values.astype(float)
                                if 'llm_relative_error'   in df.columns else np.full(n, np.nan))
    m['calibrated_error']   = (df['llm_calibrated_error'].values.astype(float)
                                if 'llm_calibrated_error' in df.columns else np.full(n, np.nan))

    # ── Community proximity ───────────────────────────────────────────────
    m['community_mae_pa']    = np.abs(df['llm_p_a'].values          - df['community_p_a'].values)
    m['community_mae_pb']    = np.abs(df['llm_p_b'].values          - df['community_p_b'].values)
    m['community_mae_pbga']  = np.abs(df['llm_p_b_given_a'].values  - df['community_p_b_given_a'].values)
    m['community_mae_pbgna'] = np.abs(df['llm_p_b_given_na'].values - df['community_p_b_given_na'].values)
    m['community_mae_overall'] = np.mean([
        m['community_mae_pa'], m['community_mae_pb'],
        m['community_mae_pbga'], m['community_mae_pbgna'],
    ], axis=0)

    # ── Frontier model proximity (only when oracle_df is supplied) ────────
    if oracle_df is not None:
        from scipy.special import rel_entr

        oracle = oracle_df.set_index('conditional_post_id').reindex(df['conditional_post_id'].values)
        m['frontier_mae_pa']    = np.abs(df['llm_p_a'].values          - oracle['llm_p_a'].values)
        m['frontier_mae_pb']    = np.abs(df['llm_p_b'].values          - oracle['llm_p_b'].values)
        m['frontier_mae_pbga']  = np.abs(df['llm_p_b_given_a'].values  - oracle['llm_p_b_given_a'].values)
        m['frontier_mae_pbgna'] = np.abs(df['llm_p_b_given_na'].values - oracle['llm_p_b_given_na'].values)
        m['frontier_mae_overall'] = np.mean([
            m['frontier_mae_pa'], m['frontier_mae_pb'],
            m['frontier_mae_pbga'], m['frontier_mae_pbgna'],
        ], axis=0)

        def _js_binary(p, q):
            """JSD between two Bernoulli distributions Bern(p) and Bern(q)."""
            p_arr = np.array([p, 1 - p])
            q_arr = np.array([q, 1 - q])
            m_arr = 0.5 * (p_arr + q_arr)
            return float(0.5 * (rel_entr(p_arr, m_arr).sum() + rel_entr(q_arr, m_arr).sum()))

        m['frontier_js_divergence'] = np.array([
            _js_binary(
                row['llm_p_b_given_a'],
                oracle.loc[row['conditional_post_id'], 'llm_p_b_given_a'],
            )
            for _, row in df.iterrows()
        ])

    # ── Sharpness (resolution proxy without outcomes) ─────────────────────
    pb   = df['llm_p_b'].values.astype(float)
    pba  = df['llm_p_b_given_a'].values.astype(float)
    pbna = df['llm_p_b_given_na'].values.astype(float)

    m['sharpness_pb']       = np.abs(pb - 0.5)
    m['entropy_pb']         = (
        -pb * np.log2(np.clip(pb, 1e-9, 1))
        - (1 - pb) * np.log2(np.clip(1 - pb, 1e-9, 1))
    )
    m['conditional_spread'] = np.abs(pba - pbna)

    # ── Coherence: ordering violation ─────────────────────────────────────
    # Flag pairs where sign(P(B|A)-P(B|¬A)) disagrees with the community's sign.
    llm_sign  = np.sign(pba - pbna)
    comm_sign = np.sign(df['community_p_b_given_a'].values - df['community_p_b_given_na'].values)
    m['ordering_violation'] = ((llm_sign != comm_sign) & (comm_sign != 0)).astype(float)

    return m


# ── Statistical power ─────────────────────────────────────────────────────────

def compute_power_paired_t(n: int, d: float, alpha: float = 0.05) -> float:
    """Power of a two-sided paired t-test (non-central t approximation)."""
    ncp    = d * np.sqrt(n)
    df     = n - 1
    t_crit = t_dist.ppf(1 - alpha / 2, df)
    return 1 - t_dist.cdf(t_crit, df, loc=ncp) + t_dist.cdf(-t_crit, df, loc=ncp)


# ── Statistical test utilities ────────────────────────────────────────────────

def compare_experiments(metrics_a: dict, metrics_b: dict, metric_name: str,
                         name_a: str = 'A', name_b: str = 'B',
                         n_bootstrap: int = 5000, alpha: float = 0.05) -> dict:
    """
    Pairwise comparison of two experiments on one metric (paired, same questions).

    Returns dict with keys:
        name_a, name_b, metric, n
        mean_a, mean_b, mean_diff
        paired_t_stat, paired_t_p
        wilcoxon_stat, wilcoxon_p
        cohens_d
        bootstrap_ci   : (lower, upper) 95% CI on mean_diff
    """
    a = np.array(metrics_a[metric_name], dtype=float)
    b = np.array(metrics_b[metric_name], dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n    = len(a)
    diff = a - b

    t_stat, t_p = stats.ttest_rel(a, b)

    try:
        w_stat, w_p = stats.wilcoxon(diff, zero_method='wilcox', correction=False)
    except ValueError:
        w_stat, w_p = np.nan, np.nan   # all differences are zero

    cohens_d = diff.mean() / (diff.std(ddof=1) + 1e-12)

    boot_means = np.array([RNG.choice(diff, size=n, replace=True).mean()
                            for _ in range(n_bootstrap)])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    return {
        'name_a': name_a, 'name_b': name_b, 'metric': metric_name,
        'n': n,
        'mean_a': a.mean(), 'mean_b': b.mean(), 'mean_diff': diff.mean(),
        'paired_t_stat': t_stat, 'paired_t_p': t_p,
        'wilcoxon_stat': w_stat, 'wilcoxon_p': w_p,
        'cohens_d': cohens_d,
        'bootstrap_ci': (ci_lo, ci_hi),
    }


def compare_all_experiments(all_metrics: dict, metric_name: str, alpha: float = 0.05) -> dict:
    """
    Multi-experiment comparison.

    ≥3 experiments → repeated-measures ANOVA (statsmodels) + Bonferroni pairwise post-hoc.
    <3 experiments → single pairwise comparison (no ANOVA).

    `all_metrics`: {experiment_name -> metrics_dict}
    """
    names = list(all_metrics.keys())
    n_exp = len(names)

    if n_exp < 2:
        return {'anova': None, 'pairwise': [],
                'note': 'Need ≥2 experiments to compare. Register more experiments in §8.2.'}

    if n_exp == 2:
        pw = compare_experiments(all_metrics[names[0]], all_metrics[names[1]],
                                  metric_name, name_a=names[0], name_b=names[1])
        return {'anova': None,
                'note': 'Only 2 experiments — showing pairwise comparison (no ANOVA needed).',
                'pairwise': [pw]}

    # Build long-format frame for rm_anova
    records = []
    for exp_name, metrics in all_metrics.items():
        arr = np.array(metrics[metric_name], dtype=float)
        for subj_idx, val in enumerate(arr):
            records.append({'subject': subj_idx, 'experiment': exp_name, metric_name: val})
    rm_df = pd.DataFrame(records).dropna(subset=[metric_name])

    try:
        anova_res = AnovaRM(rm_df, metric_name, 'subject', within=['experiment']).fit()
        f_stat = anova_res.anova_table['F Value'].iloc[0]
        p_val  = anova_res.anova_table['Pr > F'].iloc[0]
    except Exception as e:
        f_stat, p_val = np.nan, np.nan
        print(f"  rm-ANOVA failed: {e}")

    n_comparisons = n_exp * (n_exp - 1) // 2
    pairwise = []
    for a_name, b_name in combinations(names, 2):
        pw = compare_experiments(all_metrics[a_name], all_metrics[b_name],
                                  metric_name, name_a=a_name, name_b=b_name)
        pw['wilcoxon_p_bonf'] = min(pw['wilcoxon_p'] * n_comparisons, 1.0)
        pw['paired_t_p_bonf'] = min(pw['paired_t_p'] * n_comparisons, 1.0)
        pairwise.append(pw)

    return {'anova': {'f_stat': f_stat, 'p_val': p_val, 'n_experiments': n_exp},
            'pairwise': pairwise}


def print_comparison_results(results: dict, alpha: float = 0.05):
    """Pretty-print output of compare_experiments or compare_all_experiments."""
    if results.get('note'):
        print(f"Note: {results['note']}\n")

    if results.get('anova'):
        a   = results['anova']
        sig = '*** SIGNIFICANT ***' if a['p_val'] < alpha else 'not significant'
        print(f"rm-ANOVA ({a['n_experiments']} experiments): F = {a['f_stat']:.3f},  p = {a['p_val']:.4f}  ({sig})")
        print()

    for pw in results.get('pairwise', []):
        ci_lo, ci_hi = pw['bootstrap_ci']
        diff_dir = 'LOWER' if pw['mean_diff'] < 0 else 'higher'
        print(f"  {pw['name_a']}  vs  {pw['name_b']}   (N={pw['n']})")
        print(f"    Means:              {pw['mean_a']:.4f}  vs  {pw['mean_b']:.4f}  "
              f"(diff = {pw['mean_diff']:+.4f},  {pw['name_a']} is {diff_dir})")
        print(f"    Bootstrap 95% CI:   [{ci_lo:+.4f},  {ci_hi:+.4f}]")
        print(f"    Paired t-test:      t = {pw['paired_t_stat']:+.3f},  "
              f"p = {pw['paired_t_p']:.4f}{'  *' if pw['paired_t_p'] < alpha else ''}")
        print(f"    Wilcoxon:           W = {pw['wilcoxon_stat']:.1f},    "
              f"p = {pw['wilcoxon_p']:.4f}{'  *' if pw['wilcoxon_p'] < alpha else ''}")
        print(f"    Cohen's d:          {pw['cohens_d']:+.3f}")
        if 'wilcoxon_p_bonf' in pw:
            print(f"    Wilcoxon (Bonf):    p = {pw['wilcoxon_p_bonf']:.4f}"
                  f"{'  *' if pw['wilcoxon_p_bonf'] < alpha else ''}")
        print()


# ── Plot utilities ────────────────────────────────────────────────────────────

def plot_metric_comparison(all_metrics: dict, metric_name: str,
                            title: str = None, ylabel: str = None,
                            n_bootstrap: int = 5000):
    """
    Bar chart of mean ± 95% bootstrap CI per experiment.
    Adjacent pairs annotated with Wilcoxon p-value stars.
    """
    names  = list(all_metrics.keys())
    means, ci_los, ci_his = [], [], []

    for name in names:
        vals = np.array(all_metrics[name][metric_name], dtype=float)
        vals = vals[~np.isnan(vals)]
        boot = np.array([RNG.choice(vals, size=len(vals), replace=True).mean()
                          for _ in range(n_bootstrap)])
        means.append(vals.mean())
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ci_los.append(lo)
        ci_his.append(hi)

    fig, ax = plt.subplots(figsize=(max(5, 2 * len(names)), 4.5), constrained_layout=True)
    x = np.arange(len(names))
    ax.bar(x, means, color='steelblue', edgecolor='white', width=0.5, alpha=0.85)
    ax.errorbar(x, means,
                yerr=[np.array(means) - np.array(ci_los),
                      np.array(ci_his) - np.array(means)],
                fmt='none', color='black', capsize=5, linewidth=1.5)

    # Significance brackets for adjacent pairs
    y_range = max(ci_his) - min(means)
    for i, (a, b) in enumerate(zip(names[:-1], names[1:])):
        pw    = compare_experiments(all_metrics[a], all_metrics[b], metric_name, name_a=a, name_b=b)
        p     = pw['wilcoxon_p']
        stars = ('***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns')
        bracket_y = max(ci_his) + 0.05 * y_range + 0.07 * y_range * i
        ax.annotate('', xy=(i + 1, bracket_y), xytext=(i, bracket_y),
                    arrowprops=dict(arrowstyle='-', color='gray', lw=1))
        ax.text((i + i + 1) / 2, bracket_y + 0.01 * y_range, stars,
                ha='center', va='bottom', fontsize=9, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel(ylabel or metric_name)
    ax.set_title(title or f'{metric_name} by experiment')
    ax.grid(axis='y', alpha=0.3)
    plt.show()


def plot_scatter_vs_oracle(exp_metrics: dict, oracle_metrics: dict,
                            metric_name: str = 'consistency_error',
                            exp_name: str = 'experiment',
                            oracle_name: str = 'oracle'):
    """
    Per-question scatter: experiment metric vs oracle metric.
    Points below the diagonal = experiment does better on this question.
    """
    a = np.array(exp_metrics[metric_name], dtype=float)
    b = np.array(oracle_metrics[metric_name], dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.scatter(b, a, alpha=0.7, s=55, color='steelblue', edgecolors='white', zorder=3)
    lim = max(a.max(), b.max()) * 1.08
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.4, linewidth=1, label='y = x  (equal)')
    n_better = (a < b).sum()
    ax.set_xlabel(f'{oracle_name}  —  {metric_name}')
    ax.set_ylabel(f'{exp_name}  —  {metric_name}')
    ax.set_title(f'{exp_name} vs {oracle_name}\n'
                 f'(below diagonal = experiment better: {n_better}/{len(a)} questions)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()


def plot_metric_distributions(all_metrics: dict, metric_name: str,
                               community_baseline: float = None,
                               title: str = None):
    """
    Violin + individual data points per experiment.
    Optional horizontal dashed line for the community baseline.
    """
    names      = list(all_metrics.keys())
    data_clean = [np.array(all_metrics[n][metric_name], dtype=float) for n in names]
    data_clean = [d[~np.isnan(d)] for d in data_clean]

    fig, ax = plt.subplots(figsize=(max(5, 2 * len(names)), 4.5), constrained_layout=True)
    vp = ax.violinplot(data_clean, positions=range(len(names)),
                       showmedians=True, showextrema=False)
    for pc in vp['bodies']:
        pc.set_facecolor('steelblue')
        pc.set_alpha(0.45)

    for i, d in enumerate(data_clean):
        jitter = RNG.uniform(-0.06, 0.06, size=len(d))
        ax.scatter(np.full(len(d), i) + jitter, d,
                   alpha=0.65, s=20, color='steelblue', zorder=3)

    if community_baseline is not None:
        ax.axhline(community_baseline, color='orange', linestyle='--', linewidth=1.5,
                   label=f'Community mean ({community_baseline:.4f})')
        ax.legend()

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel(metric_name)
    ax.set_title(title or f'{metric_name} distribution by experiment')
    ax.grid(axis='y', alpha=0.3)
    plt.show()


# ── Summary table ─────────────────────────────────────────────────────────────

def build_summary_table(loaded_experiments: dict) -> pd.DataFrame:
    """
    One row per experiment, one column per metric.
    Shows mean ± std.  Float-only metrics only.
    """
    rows = []
    for exp_name, exp_data in loaded_experiments.items():
        metrics = exp_data['metrics']
        row = {'experiment': exp_name}
        for metric_name, arr in metrics.items():
            arr   = np.array(arr, dtype=float)
            clean = arr[~np.isnan(arr)]
            if len(clean) > 0:
                row[metric_name] = f"{clean.mean():.4f} ± {clean.std(ddof=1):.4f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index('experiment')

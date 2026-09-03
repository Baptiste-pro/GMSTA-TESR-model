"""
Author: Baptiste Boussemart

benchmark_calibration.py

Two pieces of retrospective evaluation requested for peer review, both
built on top of the walk-forward machinery already used for TESR's own
uncertainty bands (see `walk_forward_forecasts` in enso_gmst_model.py).

Leakage discipline (see the "no leakage" notes throughout this file):
every quantity used to build a forecast at a given origin -- the ridge
alpha, the residual pool used for probabilistic forecasts, and the
climatological reference rate -- is recomputed at each origin from data
strictly prior to that origin's target dates. The one exception is the
ENSO lag, which is fixed once from an initial calibration window and
held constant afterwards (Option A below); this is a deliberate,
disclosed simplification, not an oversight.

  1. BENCHMARK COMPARISON
     TESR (quadratic trend + smoothed ENSO(lag) + ENSO*time interaction,
     ridge-regularised, alpha reselected at each origin) against a
     simple multiple-regression benchmark in the spirit of Foster &
     Rahmstorf (2011): linear trend + ENSO(lag) only, ordinary least
     squares, no interaction term, no regularisation. Both models are
     evaluated on the SAME walk-forward origins so the comparison is
     apples-to-apples. Significance of the RMSE difference is assessed
     with a Diebold-Mariano test using a Newey-West long-run variance
     estimate, appropriate here because walk-forward forecast errors
     overlap across origins (horizon=12, step=6) and are therefore
     autocorrelated.

     Note on scope: the original Foster & Rahmstorf (2011) regression
     also includes volcanic aerosol optical depth and solar (TSI) terms.
     Those series are not part of this project's inputs, so the
     benchmark implemented here keeps the comparison fair by using
     exactly the two predictors TESR itself is built from (trend, ENSO),
     isolating what TESR's added structure (quadratic trend, ENSO*time
     interaction, ridge) contributes on top of a standard simple
     regression -- rather than a full re-implementation of Foster &
     Rahmstorf's original variable set.

  2. BRIER SCORE / CALIBRATION
     TESR produces threshold-exceedance probabilities from its
     walk-forward residual distribution (see plot_probability_*
     functions in enso_gmst_model.py). This script evaluates whether
     those retrospective probabilities were actually well calibrated:
     for each walk-forward origin, ONLY the residuals whose target date
     falls strictly before that origin's own target window are used to
     build the residual pool -- forecasts from an earlier origin that
     overlap into this origin's future are explicitly excluded (see
     `_leak_free_residual_pool`). The climatological reference forecast
     used for the Brier Skill Score is, likewise, the observed
     exceedance rate using only observations strictly before each
     origin, not the full-sample rate.

     The +1.5C / +2.0C thresholds discussed in the manuscript were never
     crossed within the historical sample, so they cannot be used for a
     retrospective calibration check (no historical exceedances to
     score against). This script instead defaults to a threshold that
     WAS crossed repeatedly in-sample (see THRESHOLD_DEFAULT below) so
     that calibration can actually be assessed; the same function works
     for any threshold you supply.

Usage
-----
    python benchmark_calibration.py --gmst data/era5_gmst_c3s.csv \\
        --enso data/nino34_anomaly.csv --threshold 1.0

Or import the functions directly:

    from benchmark_calibration import compare_models, brier_calibration
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats

from enso_gmst_model import (
    load_era5_gmst_c3s, load_enso_climatereanalyzer, smooth_enso,
    find_optimal_lag, build_dataset, fit_ridge_standardized,
    select_ridge_alpha_loo, walk_forward_forecasts,
)
from palette import FACTUAL, COUNTERFACT, NEUTRAL_DARK, light_tint

# Threshold used for the retrospective Brier-score / calibration check.
# Must be a value actually exceeded within the historical sample -- the
# manuscript's +1.5C / +2.0C thresholds have no historical exceedances
# to score against and are therefore not usable for this diagnostic.
THRESHOLD_DEFAULT = 1.0


# ----------------------------------------------------------------------
# 0. LAG SELECTION (Option A: fixed a priori from a calibration window)
# ----------------------------------------------------------------------

def determine_lag_a_priori(gmst_df, enso_df, calibration_frac=0.7, max_lag=12):
    """
    Determines the ENSO lag ONCE, using only the first `calibration_frac`
    of the historical record (an initial calibration window), then holds
    it constant for the rest of the walk-forward evaluation.

    This is a deliberate simplification (Option A), not an attempt at a
    fully leakage-free lag search: re-selecting the lag at every walk-
    forward origin (Option B) would be methodologically purer but adds
    substantial complexity and can make the benchmark comparison less
    stable across origins. Disclose this choice explicitly wherever the
    walk-forward results are reported, e.g.:

        "The ENSO lag was fixed a priori from the calibration period
        and subsequently held constant throughout the walk-forward
        evaluation."
    """
    smoothed = smooth_enso(enso_df)
    joint_index = gmst_df.index.intersection(smoothed.index)
    n_cal = int(len(joint_index) * calibration_frac)
    cal_dates = joint_index[:n_cal]

    lag, corr = find_optimal_lag(
        smoothed.loc[cal_dates, 'enso_ssta'],
        gmst_df.loc[cal_dates, 'gmst_anom_preind'],
        max_lag=max_lag,
    )
    print(f"Lag fixed a priori from calibration window "
          f"({cal_dates[0].date()} to {cal_dates[-1].date()}, "
          f"{calibration_frac:.0%} of the record): lag={lag} months "
          f"(corr={corr:.3f})")
    return lag


# ----------------------------------------------------------------------
# 1. BENCHMARK MODEL (Foster & Rahmstorf-style: trend + ENSO only)
# ----------------------------------------------------------------------

def fit_benchmark_simple(X_train, y_train):
    """Plain OLS on trend + ENSO(lag) only -- no quadratic term, no
    interaction, no regularisation. `X_train` here must already be
    restricted to the ['t_index', 'enso_lag'] columns (see
    `compare_models` below)."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def _fit_tesr_no_leak(X_train, y_train, alphas=None):
    """
    TESR fit for a single walk-forward origin: the ridge alpha is
    reselected HERE, from X_train/y_train only (i.e. strictly the data
    available at this origin) via closed-form LOO-CV -- no leakage from
    later origins, unlike selecting alpha once on the full series
    upfront and reusing it everywhere.
    """
    if alphas is None:
        alphas = np.logspace(-3, 3, 50)
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    feat_std = X_train.std(axis=0, ddof=0)
    feat_std_safe = np.where(feat_std == 0, 1.0, feat_std)
    X_train_std = (X_train - X_train.mean(axis=0)) / feat_std_safe
    alpha, _, _ = select_ridge_alpha_loo(X_train_std, y_train, alphas=alphas)
    return fit_ridge_standardized(X_train, y_train, alpha)


def diebold_mariano(loss_a, loss_b, h):
    """
    Diebold-Mariano test on the loss differential (squared-error loss
    here), with a Newey-West long-run variance estimate using h-1 lags
    to account for the autocorrelation induced by overlapping
    walk-forward windows (horizon h).

    Returns (dm_statistic, two_sided_p_value). A significantly negative
    statistic means model A has lower loss (more accurate) than model B.
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    d_bar = d.mean()
    max_lag = max(h - 1, 0)
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, max_lag + 1):
        cov = np.cov(d[lag:], d[:-lag], ddof=0)[0, 1]
        weight = 1 - lag / (max_lag + 1)  # Bartlett kernel
        var_d += 2 * weight * cov
    var_d = max(var_d, 1e-12) / n
    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


def compare_models(gmst_df, enso_df, lag=None, horizon=12, min_train_frac=0.7, step=6):
    """
    Runs TESR and the simple benchmark on the same walk-forward origins
    and returns a summary comparison table plus both residual series
    (for downstream plotting). The ridge alpha for TESR is reselected
    at every origin (see `_fit_tesr_no_leak`); the ENSO lag is fixed a
    priori from a calibration window (see `determine_lag_a_priori`) and
    held constant, unless you pass `lag` explicitly.
    """
    if lag is None:
        lag = determine_lag_a_priori(gmst_df, enso_df, calibration_frac=min_train_frac)
    smoothed = smooth_enso(enso_df)
    dataset = build_dataset(smoothed, gmst_df, lag)

    # -- TESR: quadratic trend + ENSO(lag) + ENSO*time, ridge --
    features_tesr = ['t_index', 't_index2', 'enso_lag', 'enso_x_t']
    month_dummies = pd.get_dummies(dataset['month_num'], prefix='m', drop_first=True).astype(float)
    X_tesr = pd.concat([dataset[features_tesr], month_dummies], axis=1)
    y = dataset['gmst_anom_preind']

    wf_tesr = walk_forward_forecasts(
        X_tesr, y, fit_fn=_fit_tesr_no_leak,
        horizon=horizon, min_train_frac=min_train_frac, step=step,
    )

    # -- Benchmark: linear trend + ENSO(lag), OLS, no interaction --
    features_bench = ['t_index', 'enso_lag']
    X_bench = dataset[features_bench]
    wf_bench = walk_forward_forecasts(
        X_bench, y, fit_fn=fit_benchmark_simple,
        horizon=horizon, min_train_frac=min_train_frac, step=step,
    )

    # Align on common dates (should already match since both start from
    # the same `dataset` index and origin schedule).
    common = wf_tesr.index.intersection(wf_bench.index)
    wf_tesr, wf_bench = wf_tesr.loc[common], wf_bench.loc[common]

    def _metrics(wf):
        err = wf["residual"].values
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((wf["obs"].values - wf["obs"].values.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot)
        return dict(rmse=rmse, mae=mae, r2=r2, n=len(wf))

    m_tesr, m_bench = _metrics(wf_tesr), _metrics(wf_bench)
    loss_tesr = wf_tesr["residual"].values ** 2
    loss_bench = wf_bench["residual"].values ** 2
    dm_stat, p_val = diebold_mariano(loss_tesr, loss_bench, h=horizon)

    summary = pd.DataFrame(
        {"TESR": m_tesr, "Benchmark (trend + ENSO, OLS)": m_bench}
    ).T
    summary["RMSE_reduction_vs_benchmark_%"] = np.where(
        summary.index == "TESR",
        100 * (m_bench["rmse"] - m_tesr["rmse"]) / m_bench["rmse"],
        np.nan,
    )

    print(summary.round(4).to_string())
    print(f"\nDiebold-Mariano test (TESR vs benchmark, squared-error loss, "
          f"Newey-West lag={horizon - 1}):")
    print(f"  DM statistic = {dm_stat:+.3f}, two-sided p = {p_val:.4f} "
          f"({'significant at 5%' if p_val < 0.05 else 'not significant at 5%'})")

    return summary, wf_tesr, wf_bench, dm_stat, p_val


# ----------------------------------------------------------------------
# 2. BRIER SCORE / CALIBRATION
# ----------------------------------------------------------------------

def _leak_free_residual_pool(all_dates, all_residuals, origin_date):
    """
    Returns only the residuals whose TARGET DATE is strictly before
    `origin_date`. This matters because with horizon > step (12 vs 6
    months here), an earlier origin's forecast window overlaps into the
    current origin's future: some of its residuals correspond to dates
    that are still ahead of `origin_date` and must NOT be included.
    """
    if len(all_dates) == 0:
        return np.asarray(all_residuals, dtype=float)  # empty pool, nothing to filter
    all_dates = pd.to_datetime(pd.Index(all_dates)).values
    all_residuals = np.asarray(all_residuals, dtype=float)
    mask = all_dates < np.datetime64(origin_date)
    return all_residuals[mask]


def brier_calibration(gmst_df, enso_df, threshold=THRESHOLD_DEFAULT, lag=None,
                       horizon=12, min_train_frac=0.7, step=6, n_bins=10,
                       min_pool_size=20):
    """
    Retrospective Brier score and reliability diagram for TESR's own
    threshold-exceedance probabilities, with no information leakage:

      - ridge alpha: reselected at each origin from that origin's
        training data only (`_fit_tesr_no_leak`);
      - ENSO lag: fixed a priori from an initial calibration window
        (`determine_lag_a_priori`), not selected using the full series
        before the walk-forward loop -- disclose this in the manuscript;
      - residual pool: at each origin, only residuals whose target date
        is strictly before that origin's forecast window are used
        (`_leak_free_residual_pool`) -- residuals from an earlier,
        overlapping origin that fall in this origin's future are
        excluded;
      - climatological reference (for the Brier Skill Score): computed
        PER ORIGIN from observed outcomes strictly before that origin,
        not from the full-sample exceedance rate.

    Returns a dict with the Brier score, the Brier Skill Score against
    the (per-origin) climatological reference, and a DataFrame of
    per-bin (mean forecast probability, observed frequency, count) for
    the reliability diagram.
    """
    if lag is None:
        lag = determine_lag_a_priori(gmst_df, enso_df, calibration_frac=min_train_frac)
    smoothed = smooth_enso(enso_df)
    dataset = build_dataset(smoothed, gmst_df, lag)

    features = ['t_index', 't_index2', 'enso_lag', 'enso_x_t']
    month_dummies = pd.get_dummies(dataset['month_num'], prefix='m', drop_first=True).astype(float)
    X = pd.concat([dataset[features], month_dummies], axis=1)
    y = dataset['gmst_anom_preind']

    n = len(X)
    start = int(n * min_train_frac)

    # Pools accumulated ACROSS origins, but always filtered by target
    # date relative to the CURRENT origin before use (see
    # `_leak_free_residual_pool`) -- storing everything and filtering
    # at read time is simplest and avoids any off-by-one pruning bugs.
    pool_dates, pool_residuals = [], []

    forecasts, outcomes, clim_forecasts, dates = [], [], [], []

    for cut in range(start, n - horizon, step):
        origin_date = X.index[cut]

        model = _fit_tesr_no_leak(X.iloc[:cut].values, y.iloc[:cut].values)
        pred = model.predict(X.iloc[cut:cut + horizon].values)
        obs = y.iloc[cut:cut + horizon].values
        target_dates = X.index[cut:cut + horizon]

        resid_pool = _leak_free_residual_pool(pool_dates, pool_residuals, origin_date)
        # Climatological reference: exceedance rate using only
        # observations strictly before this origin.
        y_before = y.iloc[:cut].values
        clim_rate = float(np.mean(y_before > threshold)) if len(y_before) > 0 else np.nan

        if len(resid_pool) >= min_pool_size:
            for p, o, d in zip(pred, obs, target_dates):
                prob = float(np.mean((p + resid_pool) > threshold))
                forecasts.append(prob)
                outcomes.append(float(o > threshold))
                clim_forecasts.append(clim_rate)
                dates.append(d)

        pool_dates.extend(list(target_dates))
        pool_residuals.extend((obs - pred).tolist())

    forecasts = np.array(forecasts)
    outcomes = np.array(outcomes)
    clim_forecasts = np.array(clim_forecasts)

    brier = float(np.mean((forecasts - outcomes) ** 2))
    brier_clim = float(np.mean((clim_forecasts - outcomes) ** 2))
    bss = 1 - brier / brier_clim if brier_clim > 0 else np.nan
    mean_base_rate = float(np.nanmean(clim_forecasts))

    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(forecasts, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append(dict(
            bin_center=(bins[b] + bins[b + 1]) / 2,
            mean_forecast=float(forecasts[mask].mean()),
            observed_freq=float(outcomes[mask].mean()),
            count=int(mask.sum()),
        ))
    reliability = pd.DataFrame(rows)

    print(f"Retrospective calibration for P(monthly GMSTA > {threshold:+.1f}C) "
          f"-- no look-ahead in residual pool, alpha, or climatology:")
    print(f"  n forecasts = {len(forecasts)}, mean pre-origin base rate = {mean_base_rate:.3f}")
    print(f"  Brier score = {brier:.4f}  (0 = perfect, {brier_clim:.4f} = per-origin climatology)")
    print(f"  Brier Skill Score vs climatology = {bss:+.3f} "
          f"({'better than climatology' if bss > 0 else 'no better than climatology'})")
    print(reliability.round(3))

    return dict(brier=brier, brier_climatology=brier_clim, bss=bss,
                base_rate=mean_base_rate, reliability=reliability,
                forecasts=forecasts, outcomes=outcomes,
                clim_forecasts=clim_forecasts, dates=dates)


def plot_reliability_diagram(calib_result, threshold=THRESHOLD_DEFAULT, filename=None):
    """CVD-safe reliability diagram: forecast probability (x) vs
    observed frequency (y), with the diagonal as perfect calibration and
    marker size proportional to bin count."""
    import matplotlib.pyplot as plt

    rel = calib_result["reliability"]
    plt.rcParams['font.family'] = 'sans-serif'
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    ax.plot([0, 1], [0, 1], color=NEUTRAL_DARK, linestyle='--', lw=1, label='Perfect calibration')
    ax.scatter(rel['mean_forecast'], rel['observed_freq'],
               s=20 + 4 * rel['count'], color=FACTUAL, edgecolor='white',
               linewidth=0.6, zorder=3, label='TESR (this study)')
    ax.axhline(calib_result['base_rate'], color=COUNTERFACT, linestyle=':', lw=1,
               label='Mean pre-origin climatological rate')

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(f"Forecast probability of exceeding {threshold:+.1f} C")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram (retrospective, walk-forward, no leakage)",
                 loc='left', fontsize=10.5)
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=8.5)
    ax.grid(True, alpha=0.25, lw=0.6)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
    return fig


# ----------------------------------------------------------------------
# 3. CLI ENTRY POINT
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gmst",
        default="data/era5_gmst_c3s.csv",
        help="Path to the GMSTA dataset"
    )
    parser.add_argument(
        "--enso",
        default="data/nino34_anomaly.csv",
        help="Path to the Niño 3.4 dataset"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD_DEFAULT
    )
    args = parser.parse_args()

    gmst_df = load_era5_gmst_c3s(args.gmst)
    enso_df = load_enso_climatereanalyzer(args.enso)

    # Lag fixed once here so both analyses below use the identical,
    # a-priori value (see determine_lag_a_priori docstring).
    shared_lag = determine_lag_a_priori(gmst_df, enso_df)

    print("=== Benchmark comparison: TESR vs simple trend+ENSO OLS ===")
    compare_models(gmst_df, enso_df, lag=shared_lag)

    print(f"\n=== Brier score / calibration (threshold {args.threshold:+.1f} C) ===")
    result = brier_calibration(gmst_df, enso_df, threshold=args.threshold, lag=shared_lag)
    fig = plot_reliability_diagram(result, threshold=args.threshold,
                                    filename="reliability_diagram.png")
    print("\nSaved reliability_diagram.png")

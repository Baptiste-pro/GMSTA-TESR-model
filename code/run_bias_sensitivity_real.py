"""
Author: Baptiste Boussemart

run_bias_sensitivity_real.py

End-to-end script that reproduces the manuscript's central March 2027
estimate from real data, then runs the Reviewer-2 bias-sensitivity
analysis with BOTH uncertainty sources included (walk-forward model
residuals AND the ENSO multi-model dispersion bounds).

What this does, step by step:
  1. Loads ERA5/C3S GMST and Nino3.4 (ERA5) from data/.
  2. Fixes the ENSO lag a priori (calibration window) and fits TESR.
  3. Computes the REAL empirical peak-underestimation bias at the three
     historical calibration episodes (Jan 1983, Feb 1998, Feb 2016),
     instead of assuming the manuscript's qualitative "0.1-0.2C".
  4. Computes REAL walk-forward residuals (no leakage).
  5. Rebuilds the ENSO multi-model scenario (Climate Dashboard members,
     SINTEX-F excluded) and extracts the GMST dispersion bounds
     (q0/p05/p25/p75/p95/q100) at the scenario's peak month.
  6. Runs the bias-sensitivity sweep with both uncertainty sources
     combined, exactly reproducing the manuscript's own probability
     pipeline (_combined_uncertainty_samples) at bias=0 as a sanity
     check, then extrapolating across the empirical bias range.

Usage
-----
    python run_bias_sensitivity_real.py \\
        --gmst data/era5_gmst_c3s.csv \\
        --enso data/nino34_anomaly.csv \\
        --enso-members data/enso_members_oni.csv
"""

import argparse
import numpy as np
import pandas as pd

from enso_gmst_model import (
    load_era5_gmst_c3s, load_enso_climatereanalyzer, smooth_enso,
    find_optimal_lag, build_dataset, fit_model, walk_forward_forecasts,
    fit_ridge_standardized, select_ridge_alpha_loo,
    load_enso_dashboard_scenario, smooth_enso_for_forecast,
    forecast_from_enso_calendar,
)
from bias_sensitivity import bias_sensitivity, plot_bias_sensitivity, historical_peak_bias

# Historical peak months for the three calibration episodes -- as
# confirmed against the manuscript's own text (Sect. 4.2): the GMSTA
# peak of each episode, NOT necessarily the month of largest ENSO
# influence (April 1983 is explicitly the one month where TESR
# slightly OVERestimates for the 1982/83 episode, so it is excluded
# here in favour of the actual GMSTA peak, January 1983).
EPISODE_PEAKS = {"1982/83": "1983-01", "1997/98": "1998-02", "2015/16": "2016-02"}

EXCLUDE_MODELS = ("SINTEX-F",)
JULY_2026_OBS_ONI = 2.03  # observed ONI, July 2026 (Climate Dashboard)


def fit_tesr_no_leak(X_train, y_train):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    s = X_train.std(axis=0, ddof=0)
    s = np.where(s == 0, 1.0, s)
    X_std = (X_train - X_train.mean(axis=0)) / s
    alpha, _, _ = select_ridge_alpha_loo(X_std, y_train, alphas=np.logspace(-3, 3, 50))
    return fit_ridge_standardized(X_train, y_train, alpha)


def main(path_gmst, path_enso, path_enso_members):
    gmst_df = load_era5_gmst_c3s(path_gmst)
    enso_df_raw = load_enso_climatereanalyzer(path_enso)
    enso_df = smooth_enso(enso_df_raw, window=3)
    lag, corr = find_optimal_lag(enso_df['enso_ssta'], gmst_df['gmst_anom_preind'])
    print(f"Lag = {lag} months (corr = {corr:.3f})")

    dataset = build_dataset(enso_df, gmst_df, lag)
    model, X, y, X_train, X_test, y_train, y_test = fit_model(
        dataset, use_seasonal=True, regularize=True
    )

    # -- 1. Empirical peak bias, from the actual in-sample fit --
    y_pred_full = pd.Series(model.predict(X.values), index=X.index)
    peak_biases = historical_peak_bias(y, y_pred_full, EPISODE_PEAKS)
    print("\nEmpirical peak underestimation bias (obs - pred):")
    for k, v in peak_biases.items():
        print(f"  {k}: {v:+.3f} C")

    # -- 2. Real walk-forward residuals (no leakage) --
    features = ['t_index', 't_index2', 'enso_lag', 'enso_x_t']
    month_dummies = pd.get_dummies(dataset['month_num'], prefix='m', drop_first=True).astype(float)
    X_full = pd.concat([dataset[features], month_dummies], axis=1)
    wf = walk_forward_forecasts(
        X_full, dataset['gmst_anom_preind'], fit_fn=fit_tesr_no_leak,
        horizon=12, min_train_frac=0.7, step=6,
    )
    residuals = wf["residual"].values
    print(f"\nWalk-forward residuals: n={len(residuals)}, "
          f"RMSE={np.sqrt(np.mean(residuals ** 2)):.4f} C")

    # -- 3. Real ENSO multi-model scenario, bounds at the peak month --
    envelope = load_enso_dashboard_scenario(path_enso_members, exclude_models=EXCLUDE_MODELS)

    def build_forecast(col):
        raw = pd.concat([
            pd.Series([JULY_2026_OBS_ONI], index=pd.to_datetime(["2026-07-01"])),
            envelope[col],
        ])
        smoothed = smooth_enso_for_forecast(enso_df_raw, raw, window=3)
        return forecast_from_enso_calendar(model, X.columns.tolist(), dataset, lag, smoothed)

    fc_median = build_forecast('median')
    peak_month = fc_median['gmst_anom_pred_preind'].idxmax()
    central_peak = float(fc_median['gmst_anom_pred_preind'].max())
    bounds = {c: float(build_forecast(c).loc[peak_month, 'gmst_anom_pred_preind'])
              for c in ['q0', 'p05', 'p25', 'p75', 'p95', 'q100']}
    print(f"\nScenario peak: {peak_month:%B %Y}, central = {central_peak:.3f} C")
    print("ENSO-dispersion bounds at peak (C):",
          {k: round(v, 3) for k, v in bounds.items()})

    # -- 4. Sensitivity sweep, both uncertainty sources combined --
    table = bias_sensitivity(
        central_peak, residuals, bounds=bounds,
        bias_range=np.round(np.arange(0.0, 0.31, 0.025), 3),
    )
    print("\n=== Bias sensitivity (residuals + ENSO multi-model dispersion) ===")
    print(table.round(2))

    empirical_range = (min(v for k, v in peak_biases.items() if k != "mean"),
                        max(v for k, v in peak_biases.items() if k != "mean"))
    fig = plot_bias_sensitivity(table, peak_biases, empirical_range=empirical_range,
                                 filename="bias_sensitivity_full.png")
    table.to_csv("bias_sensitivity_full.csv")
    print("\nSaved bias_sensitivity_full.png and bias_sensitivity_full.csv")

    return dict(lag=lag, peak_biases=peak_biases, residuals=residuals,
                central_peak=central_peak, bounds=bounds, table=table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gmst", default="C:/Users/lolma/Documents/B/Climat temp autres/previ et fiab/Tendance saiso/ENSOT/era5_gmst_c3s.csv")
    parser.add_argument("--enso", default="C:/Users/lolma/Documents/B/Climat temp autres/previ et fiab/Tendance saiso/ENSOT/nino34_real.csv")
    parser.add_argument("--enso-members", default="C:/Users/lolma/Documents/B/Climat temp autres/previ et fiab/Tendance saiso/ENSOT/enso_members_oni.csv")
    args = parser.parse_args()
    main(args.gmst, args.enso, args.enso_members)

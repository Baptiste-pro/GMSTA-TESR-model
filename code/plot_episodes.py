"""
Author: Baptiste Boussemart

plot_episodes.py
"""

import os
import sys
import textwrap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from enso_gmst_model import (
    load_era5_gmst_c3s, load_enso_climatereanalyzer, smooth_enso,
    smooth_enso_for_forecast, find_optimal_lag, build_dataset, fit_model,
    decompose_forecast_enso_calendar, fit_ridge_standardized,
    load_enso_dashboard_scenario,
)
from palette import (
    FACTUAL, FACTUAL_DARK, COUNTERFACT, COUNTERFACT_2, COUNTERFACT_DARK,
    ALT_SCENARIO, HIGHLIGHT, RECORD, ANALOG_COLORS, light_tint,
    QUANTILE_BANDS,
    COLOR_TREND as _COLOR_TREND, COLOR_EXTERNAL_VAR,
)

# Default input paths: point these at your own local ERA5/C3S GMST,
# Nino 3.4 and ENSO-dashboard-members CSVs (see README for the expected
# format of each file), or override via environment variables.
DEFAULT_PATH_GMST = os.environ.get(
    "TESR_PATH_GMST",
    "data/era5_gmst_c3s.csv"
)

DEFAULT_PATH_ENSO = os.environ.get(
    "TESR_PATH_ENSO",
    "data/nino34_anomaly.csv"
)

DEFAULT_PATH_ENSO_MEMBERS = os.environ.get(
    "TESR_PATH_ENSO_MEMBERS",
    "data/enso_members_oni.csv"
)

COLOR_TREND = FACTUAL_DARK
COLOR_ENSO = HIGHLIGHT
COLOR_SEAS = _COLOR_TREND
COLOR_EXT = COLOR_EXTERNAL_VAR  # ENSO-external variability (verification minus model residual)
COLOR_VERIF = '#000000'  # verification line (actual ERA5/C3S observation)

# ----------------------------------------------------------------------
# 1. PIPELINE (same as model.py) -- also returns X_train/y_train and the
#    per-episode ENSO calendars (needed for the bootstrap)
# ----------------------------------------------------------------------
def build_pipeline(path_gmst=DEFAULT_PATH_GMST, path_enso=DEFAULT_PATH_ENSO,
                    path_enso_members=DEFAULT_PATH_ENSO_MEMBERS):
    gmst_df = load_era5_gmst_c3s(path_gmst).sort_index()
    enso_df_raw = load_enso_climatereanalyzer(path_enso).sort_index()

    observations_recentes = {"2026-06-01": (1.39, 1.45)}
    for date_str, (gmst_val, enso_val) in observations_recentes.items():
        d = pd.Timestamp(date_str)
        gmst_df.loc[d, 'gmst_anom_preind'] = gmst_val
        enso_df_raw.loc[d, 'enso_ssta'] = enso_val
    gmst_df = gmst_df.sort_index()
    enso_df_raw = enso_df_raw.sort_index()

    enso_df = smooth_enso(enso_df_raw, window=3)
    lag, corr = find_optimal_lag(enso_df['enso_ssta'], gmst_df['gmst_anom_preind'])
    print(f"Lag optimal : {lag} mois (corr={corr:.3f})")

    dataset = build_dataset(enso_df, gmst_df, lag)
    model, X, y, X_train, X_test, y_train, y_test = fit_model(dataset, use_seasonal=True, regularize=True)
    feature_cols = X.columns.tolist()

    calendars = {}
    episodes_hist = {"1982-1983": 1982, "1997-1998": 1997, "2015-2016": 2015}
    for label, start_year in episodes_hist.items():
        input_start = pd.Timestamp(f"{start_year}-04-01")
        input_end = pd.Timestamp(f"{start_year + 1}-03-01")
        calendars[label] = enso_df['enso_ssta'].loc[input_start:input_end]

    # ENSO scenario dynamically recalibrated using enso_members_oni.csv (instead
    # of a fixed list) — same logic as enso_gmst_model.py: July
    # 2026 = observed deterministic transition point (ONI), August 2026 → April
    # 2027 = model-weighted multi-model median, excluding SINTEX-F.
    JUILLET_OBS_ONI = 2.03  # Observed ONI for July 2026 (The Climate Brink ENSO Dashboard)
    enveloppe_enso = load_enso_dashboard_scenario(path_enso_members, exclude_models=('SINTEX-F',))
    enso_c3s_median_brut = pd.concat([
        pd.Series([JUILLET_OBS_ONI], index=pd.to_datetime(["2026-07-01"])),
        enveloppe_enso['median'],
    ])
    enso_c3s_median = smooth_enso_for_forecast(enso_df_raw, enso_c3s_median_brut, window=3)
    near_term_enso_brut = enso_df_raw['enso_ssta'].loc["2026-04-01":"2026-06-01"]
    near_term_enso = smooth_enso_for_forecast(enso_df_raw, near_term_enso_brut, window=3)
    calendars["2026-2027 (projection)"] = pd.concat([near_term_enso, enso_c3s_median])

    decomps = {l: decompose_forecast_enso_calendar(model, feature_cols, dataset, lag, cal)
               for l, cal in calendars.items()}

    # ------------------------------------------------------------------
    # COMBINED UNCERTAINTY ENVELOPE: ENSO multi-model dispersion
    # (Q0–Q100, Q05–Q95, Q25–Q75, taken from ‘envelope_enso’) STACKED with
    # the TESR model’s INHERENT uncertainty (moving-block bootstrap).
    # It is not enough simply to run the central model on an
    # extreme ENSO scenario: this yields a point forecast, not a range --
    # we also need the model’s margin of uncertainty FOR THAT scenario. We
    # therefore bootstrap EACH bound (300 replicates per bound, ~0.7s each),
    # and take the tail of the bootstrap on the relevant side: the P5 of the bootstrap
    # under the low ENSO scenario for the lower bound, the P95 of the bootstrap under
    # the high ENSO scenario for the upper bound. July remains fixed at the
    # observed value (JULY_OBS_ONI) in all bands — there is no
    # uncertainty for a month that has already been measured. Only available for
    # “2026–2027 (projection)”: historical episodes have an observed ENSO
    # state, with no multi-model dispersion to propagate.
    #
    # IMPORTANT — the time series schedule is ALIGNED with that of the
    # central schedule (near_term_enso April–June THEN July junction THEN
    # scenario): without near_term_enso at the start, bootstrap_episode_raw
    # shifts its timeline by +lag (3 months) and does not produce its first
    # bounds until October 2026, whereas the central curve
    # (decomps[‘2026–2027 (projection)’]) starts as early as July 2026 --
    # the plotted envelope would then have a 3-month gap (July–September
    # with no visible band) before catching up with the central curve. As April–June
    # are already observed (and therefore identical across all bands), adding them
    # creates no artificial dispersion over this period: only
    # the model’s own uncertainty (bootstrap) appears there; the
    # ENSO dispersion only widens from August 2026 onwards (multi-model scenario).
    # ------------------------------------------------------------------
    enso_bands_brut = {}
    for col in ('p25', 'p75', 'p05', 'p95', 'q0', 'q100'):
        enso_bands_brut[col] = pd.concat([
            near_term_enso_brut,
            pd.Series([JUILLET_OBS_ONI], index=pd.to_datetime(["2026-07-01"])),
            enveloppe_enso[col],
        ])
    enso_bands = {col: smooth_enso_for_forecast(enso_df_raw, s, window=3)
                  for col, s in enso_bands_brut.items()}
    enso_bootstrap_bands = {}
    for col, cal in enso_bands.items():
        raw_b = bootstrap_episode_raw(model, feature_cols, X_train, y_train, dataset, lag, cal, n_boot=300)
        enso_bootstrap_bands[col] = dict(
            ci_monthly=ci_all_dates(raw_b),      # -> total_p5/p50/p95 per month
            ci_mean=ci_mean_over_episode(raw_b),  # -> IC 90% of the episode mean
            decomp_central=decompose_forecast_enso_calendar(model, feature_cols, dataset, lag, cal),
        )
    enso_uncertainty = {"2026-2027 (projection)": enso_bootstrap_bands}

    return dict(model=model, feature_cols=feature_cols, dataset=dataset, lag=lag,
                X_train=X_train, y_train=y_train, gmst_df=gmst_df,
                calendars=calendars, decomps=decomps, labels=list(decomps.keys()),
                enso_uncertainty=enso_uncertainty)


# ----------------------------------------------------------------------
# 1a. VERIFICATION vs MODEL -- “ENSO-external variability”
# ----------------------------------------------------------------------

def get_verification(gmst_df, decomp):
    """
    Aligns the ACTUALLY OBSERVED GMST values (ERA5/C3S, column
    “gmst_anom_preind”) with the target date index of an
    episode decomposition. For historical episodes (82/83, 97/98, 15/16), all
    months are observed. For the 2026–2027 projection, only the
    first few months (which have already elapsed at the time of the run) are observed — the rest
    automatically return NaN via reindex(), which allows all
    downstream functions (curve, table, bars) to distinguish
    between hindcasts and pure forecasts without the need for episode-specific code.
    """
    return gmst_df['gmst_anom_preind'].reindex(decomp.index)


def compute_external_variability(decomp, verif):
    """
    Month-on-month difference between the observed value and the
    total forecast from the TESR model (trend + ENSO + seasonal).

    Proposed interpretation for the manuscript: the portion of the
    GMSTA that the model explains NEITHER by the long-term trend, NOR by
    ENSO (including its interaction), NOR by the residual seasonal cycle --
    therefore, by definition, variability EXTERNAL to these three factors
    (volcanism/aerosols, PDO/AMO, other unmodelled internal noise;
    see the ‘MODEL LIMITATIONS’ section in enso_gmst_model.py). This is
    NOT a model uncertainty interval (such as the bootstrap CI
    for the other components): it is an observed deterministic residual –
    predicted, NaN until the month has been observed.
    """
    return verif - decomp['total']


def _make_summary(decomps, labels, verifs=None):
    rows = {}
    for l in labels:
        d = decomps[l]
        row = {
            'trend': d['trend'].mean(),
            'enso': d['enso'].mean(),
            'seasonal': d['seasonal'].mean(),
            'total': d['total'].mean(),
        }
        if verifs is not None and l in verifs:
            ext = compute_external_variability(d, verifs[l])
            row['ext_var'] = ext.mean(skipna=True)  # NaN if no months observed
            row['n_obs'] = int(ext.notna().sum())
        else:
            row['ext_var'] = np.nan
            row['n_obs'] = 0
        rows[l] = row
    summary = pd.DataFrame(rows).T
    summary['trend_pct'] = 100 * summary['trend'] / summary['total']
    summary['enso_pct'] = 100 * summary['enso'] / summary['total']
    summary['seasonal_pct'] = 100 * summary['seasonal'] / summary['total']
    summary['ext_var_pct'] = 100 * summary['ext_var'] / summary['total']
    return summary


# ----------------------------------------------------------------------
# 2. MOBILE BLOCK BOOTSTRAPPING -- extended to trend/ENSO/seasonal/total
#    (same principle as bootstrap_attribution_uncertainty in model.py,
#    Künsch 1989: resampling of training residuals in
#    blocks of 12 consecutive months, Ridge re-fitting with a FIXED alpha)
# ----------------------------------------------------------------------
  
def bootstrap_episode_raw(model, feature_cols, X_train, y_train, dataset, lag,
                           enso_calendar, n_boot=300, block_size=12, seed=42):
    rng = np.random.default_rng(seed)
    n = len(X_train)
    y_hat_train = model.predict(X_train)
    resid_train = np.asarray(y_train) - y_hat_train

    target_dates = pd.to_datetime(enso_calendar.index) + pd.DateOffset(months=lag)
    n_dates = len(target_dates)
    trend_boot = np.empty((n_boot, n_dates))
    enso_boot = np.empty((n_boot, n_dates))
    seasonal_boot = np.empty((n_boot, n_dates))
    total_boot = np.empty((n_boot, n_dates))

    for b in range(n_boot):
        n_blocks = int(np.ceil(n / block_size))
        starts = rng.integers(0, max(n - block_size, 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        idx = np.clip(idx, 0, n - 1)
        y_star = y_hat_train + resid_train[idx]

        m_boot = fit_ridge_standardized(X_train, y_star, model.alpha_)
        decomp_b = decompose_forecast_enso_calendar(m_boot, feature_cols, dataset, lag, enso_calendar)
        trend_boot[b, :] = decomp_b['trend'].values
        enso_boot[b, :] = decomp_b['enso'].values
        seasonal_boot[b, :] = decomp_b['seasonal'].values
        total_boot[b, :] = decomp_b['total'].values

    return dict(trend=trend_boot, enso=enso_boot, seasonal=seasonal_boot,
                total=total_boot, target_dates=target_dates)


def ci_mean_over_episode(raw):
    """90% CI (P5/P50/P95) of the AVERAGE of each component over the episode,
    + the percentages (component/total ratio recalculated for each replica
    before taking the percentile — not percentile(component)/percentile(total))."""
    out = {}
    total_mean = raw['total'].mean(axis=1)
    for comp in ['trend', 'enso', 'seasonal']:
        comp_mean = raw[comp].mean(axis=1)
        out[comp] = tuple(np.percentile(comp_mean, [5, 50, 95]))
        out[comp + '_pct'] = tuple(np.percentile(100 * comp_mean / total_mean, [5, 50, 95]))
    out['total'] = tuple(np.percentile(total_mean, [5, 50, 95]))
    return out


def ci_at_date(raw, date):
    """90% CI for each component at a specific target date (for simplicity:
    the date is set as the month of peak in the CENTRAL scenario — it is
    not recalculated via bootstrap replication, which would result in a
    much more resource-intensive exercise without qualitatively changing the result)."""
    idx = list(raw['target_dates']).index(pd.Timestamp(date))
    out = {}
    total_at = raw['total'][:, idx]
    for comp in ['trend', 'enso', 'seasonal']:
        comp_at = raw[comp][:, idx]
        out[comp] = tuple(np.percentile(comp_at, [5, 50, 95]))
        out[comp + '_pct'] = tuple(np.percentile(100 * comp_at / total_at, [5, 50, 95]))
    out['total'] = tuple(np.percentile(total_at, [5, 50, 95]))
    return out


def ci_all_dates(raw):
    """90% CI (P5/P50/P95) for each component AND for the total/counterfactual,
    calculated EVERY MONTH during the episode. The counterfactual (trend + seasonal)
    is recalculated replica by replica BEFORE the percentiles are taken
    (not the sum of the individual percentiles, which would differ due to the
    correlation between trend and seasonal within the same replica)."""
    dates = pd.DatetimeIndex(raw['target_dates'])
    total = raw['total']
    contrefactuel_boot = raw['trend'] + raw['seasonal']  # réplique par réplique
    out = {}
    for comp in ['trend', 'enso', 'seasonal']:
        arr = raw[comp]
        p5, p50, p95 = np.percentile(arr, [5, 50, 95], axis=0)
        out[f'{comp}_p5'], out[f'{comp}_p50'], out[f'{comp}_p95'] = p5, p50, p95
        pct = 100 * arr / total
        pp5, pp50, pp95 = np.percentile(pct, [5, 50, 95], axis=0)
        out[f'{comp}_pct_p5'], out[f'{comp}_pct_p50'], out[f'{comp}_pct_p95'] = pp5, pp50, pp95
    # -- Additions regarding error bars on model and counterfactual curves --
    out['total_p5'], out['total_p50'], out['total_p95'] = np.percentile(total, [5, 50, 95], axis=0)
    out['contrefactuel_p5'], out['contrefactuel_p50'], out['contrefactuel_p95'] = \
        np.percentile(contrefactuel_boot, [5, 50, 95], axis=0)
    return pd.DataFrame(out, index=dates)


def _err_from_ci(center, ci_lo_mid_hi):
    """Converts (p5, p50, p95) + the central value (point estimate,
    excluding bootstrap) into POSITIVE (lower_error, upper_error) values for
    matplotlib errorbars, centred on the central value (not on the
    bootstrap median, which may differ slightly)."""
    p5, _, p95 = ci_lo_mid_hi
    lo = max(0.0, center - p5)
    hi = max(0.0, p95 - center)
    return lo, hi

def _err_from_ci_series(center, p5, p95):
    """Vectorised version of _err_from_ci: positive (lo, hi) values per point,
    anchored to the central point value (not the bootstrap median)."""
    lo = np.maximum(0.0, center.values - p5.values)
    hi = np.maximum(0.0, p95.values - center.values)
    return lo, hi


def _add_minmax_markers(ax, bars, labels, minmax, x_offset_frac=0.0,
                         legend_labels=None):
    """Red (max) / blue (min) triangles CENTRED on the relevant bar
    (x_offset_frac=0 by default — previously offset to the side, which
    visually misaligned them from the bar). To avoid any overlap
    with the label text (which remains centred in the same place), it is the
    TEXT that is pushed beyond the outermost marker (see
    label_bars: top_extent/bottom_extent now include the
    min/max), not the marker that is offset.

    minmax: {label: (min, max)} — for ENSO, a combination of multi-
    model dispersion and model bootstrap uncertainty; for Trend/Seasonal,
    P5/P95 bootstrap limits (same values as the error bars,
    displayed here as triangles for visual consistency with
    ENSO). Returns {label: (min, max)} for the labels actually
    plotted, to be inserted into the dynamic axis margins.

    legend_labels: (label_max, label_min) optional — if provided, adds
    JUST ONE legend entry for this type of marker (to be passed only
    in one of the calls when the function is used multiple times on
    the same plot, to avoid duplicate entries)."""
    marked = {}
    if minmax is None:
        return marked
    for i, l in enumerate(labels):
        if l not in minmax:
            continue
        mn, mx = minmax[l]
        xpos = bars[i].get_x() + bars[i].get_width() * (0.5 + x_offset_frac)
        ax.scatter([xpos], [mx], color=RECORD, marker='^', s=50, zorder=8,
                   edgecolor='#1a1a1a', linewidth=0.6)
        ax.scatter([xpos], [mn], color=COUNTERFACT, marker='v', s=50, zorder=8,
                   edgecolor='#1a1a1a', linewidth=0.6)
        marked[l] = (mn, mx)
    if marked and legend_labels is not None:
        label_max, label_min = legend_labels
        ax.scatter([], [], color=RECORD, marker='^', s=50, edgecolor='#1a1a1a',
                   linewidth=0.6, label=label_max)
        ax.scatter([], [], color=COUNTERFACT, marker='v', s=50, edgecolor='#1a1a1a',
                   linewidth=0.6, label=label_min)
    return marked


# Backward-compatible alias (old name, retained the ENSO-only signature) --
# retained in case other scripts in the pipeline import it directly.
def _add_enso_minmax_markers(ax, bars, labels, enso_minmax, x_offset_frac=0.0):
    return _add_minmax_markers(
        ax, bars, labels, enso_minmax, x_offset_frac=x_offset_frac,
        legend_labels=('Max ENSO (spread + model uncertainty)',
                        'Min ENSO (spread + model uncertainty)'))


def _draw_header(fig, fig_h, title, subtitle_text, extra_line=None,
                  title_top_in=0.42, gap_in=0.30, wrap_width=125):
    """Title + subtitle (auto-wrapped across multiple lines if necessary) +
    optional line, anchored in INCHES from the top of the figure (not as a
    fraction) — never overflows or overlaps, regardless of the
    number of lines produced by the wrap or the height of the figure. Returns
    the total height (in inches) occupied by the header, to be passed to
    fig.subplots_adjust(top=1 - header_in/fig_h)."""
    wrapped = textwrap.fill(subtitle_text, width=wrap_width)
    n_lines = wrapped.count("\n") + 1
    subtitle_top_in = title_top_in + gap_in
    extra_top_in = subtitle_top_in + 0.20 * n_lines + 0.08
    header_in = extra_top_in + (0.30 if extra_line else 0.06)
    fig.suptitle(title, fontsize=14, fontweight='bold', x=0.02, ha='left', va='top',
                 y=1 - title_top_in / fig_h)
    fig.text(0.02, 1 - subtitle_top_in / fig_h, wrapped, fontsize=9.5, style='italic',
              color='#444444', ha='left', va='top')
    if extra_line:
        fig.text(0.02, 1 - extra_top_in / fig_h, extra_line, fontsize=9,
                  fontweight='bold', color='#1a1a1a', ha='left', va='top')
    return header_in

# ----------------------------------------------------------------------
# 3. COMPARATIVE BAR CHART (average per episode, °C) + 90% CI   
# ----------------------------------------------------------------------

def plot_bar_comparatif_moyenne(decomps, labels, ci_by_episode=None, verifs=None, enso_minmax=None,
                                 filename="gmst_episodes_comparison_bars.png"):
    """
    enso_minmax: optional dictionary {label: (combined_min, combined_max)} -- 90% CI
        of the AVERAGE ENSO contribution under extreme ENSO
        scenarios (q0/q100), including moving-block bootstrap (see
        build_pipeline()[“enso_uncertainty”]) -- NOT the raw ENSO
        dispersion alone, which would underestimate the total uncertainty. Red
        triangle = max, blue triangle = min, offset to the side of the ENSO bar
        so as not to overlap its label. Applies only to labels
        present in the dict (typically “2026–2027 (projection)”).
    """
    summary = _make_summary(decomps, labels, verifs=verifs)
    has_ext = verifs is not None and summary['ext_var'].notna().any()

    plt.rcParams['font.family'] = 'serif'
    fig_h = 8.0
    fig, ax = plt.subplots(figsize=(11.5, fig_h))

    x = np.arange(len(labels))
    width = 0.19 if has_ext else 0.26
    offsets = [-1.5, -0.5, 0.5, 1.5] if has_ext else [-1, 0, 1]

    b1 = ax.bar(x + offsets[0] * width, summary['trend'], width, color=COLOR_TREND, label="Tendance (anthropique)")
    b2 = ax.bar(x + offsets[1] * width, summary['enso'], width, color=COLOR_ENSO, label="ENSO (naturel)")
    b3 = ax.bar(x + offsets[2] * width, summary['seasonal'], width, color=COLOR_SEAS, label="Residual seasonal")
    bars_comp = [(b1, 'trend'), (b2, 'enso'), (b3, 'seasonal')]
    b4 = None
    if has_ext:
        # NaN (episode/month not yet observed) -> bar at height 0, not
        # labelled as normal (see label_bars), so as not to distort the axis
        ext_plot = summary['ext_var'].fillna(0.0)
        b4 = ax.bar(x + offsets[3] * width, ext_plot, width, color=COLOR_EXT,
                    label="ENSO-external variability (verification minus model)")
        bars_comp.append((b4, 'ext_var'))

    # -- 90% IC error bars (if provided -- only for trend/enso/
    #    seasonal: external variability is a deterministic
    #    observed-predicted residual, not a bootstrapped quantity) --
    if ci_by_episode is not None:
        for bars, comp in ((b1, 'trend'), (b2, 'enso'), (b3, 'seasonal')):
            xs, ys, errs_lo, errs_hi = [], [], [], []
            for i, l in enumerate(labels):
                center = bars[i].get_height()
                ci = ci_by_episode[l][comp]
                lo, hi = _err_from_ci(center, ci)
                xs.append(bars[i].get_x() + bars[i].get_width() / 2)
                ys.append(center)
                errs_lo.append(lo)
                errs_hi.append(hi)
            ax.errorbar(xs, ys, yerr=[errs_lo, errs_hi], fmt='none',
                        ecolor='#1a1a1a', elinewidth=1.3, capsize=4, capthick=1.3, zorder=5)

    # -- Min/max points, CENTRED on each relevant bar -- ENSO: uncertainty
    #    COMBINED (multi-model dispersion + model bootstrap); Trend and
    #    Seasonal: P5/P95 bounds of the bootstrap (same values as the
    #    error bars, displayed as triangles for visual consistency) --
    # -- Min/max triangles restricted to labels present in enso_minmax
    #    (in practice, only “2026–2027 (projection)”: the only episode
    #    for which a multi-model ENSO dispersion exists). For
    #    historical episodes, the 90% bootstrap CI remains visible via the
    #    error bar — no need to duplicate it as a triangle. —
    minmax_labels = set(enso_minmax.keys()) if enso_minmax else set()
    trend_minmax = ({l: (ci_by_episode[l]['trend'][0], ci_by_episode[l]['trend'][2]) for l in labels if l in minmax_labels}
                     if ci_by_episode is not None else None)
    seasonal_minmax = ({l: (ci_by_episode[l]['seasonal'][0], ci_by_episode[l]['seasonal'][2]) for l in labels if l in minmax_labels}
                        if ci_by_episode is not None else None)
    trend_minmax_xy = _add_minmax_markers(ax, b1, labels, trend_minmax)
    enso_minmax_xy = _add_minmax_markers(
        ax, b2, labels, enso_minmax,
        legend_labels=('Max (ENSO spread + model uncertainty)',
                        'Min (ENSO spread + model uncertainty)'))
    seasonal_minmax_xy = _add_minmax_markers(ax, b3, labels, seasonal_minmax)

    def label_bars(bars, pct_col, comp, minmax_by_label=None):
        for i, bar in enumerate(bars):
            h = bar.get_height()
            pct = summary[pct_col].iloc[i]
            is_nan = comp == 'ext_var' and pd.isna(summary['ext_var'].iloc[i])
            hi_err = 0.0
            lo_err = 0.0
            if ci_by_episode is not None and comp in ci_by_episode.get(labels[i], {}):
                lo_err, hi_err = _err_from_ci(h, ci_by_episode[labels[i]][comp])
            # -- The text is anchored beyond the most extreme point between the
            #    error bar AND the min/max marker (if present for this label),
            #    so that it never overlaps the centered triangle. --
            top_extent = h + hi_err
            bottom_extent = h - lo_err
            if minmax_by_label is not None and labels[i] in minmax_by_label:
                mn, mx = minmax_by_label[labels[i]]
                top_extent = max(top_extent, mx)
                bottom_extent = min(bottom_extent, mn)
            va = 'bottom' if h >= 0 else 'top'
            y_anchor = (top_extent + 0.015) if h >= 0 else (bottom_extent - 0.015)
            txt = "n.a." if is_nan else f"{h:+.2f}°C\n({pct:+.0f}%)"
            ax.text(bar.get_x() + bar.get_width() / 2, y_anchor, txt, ha='center', va=va,
                    fontsize=8.3, fontweight='bold', color='#1a1a1a')

    label_bars(b1, 'trend_pct', 'trend', trend_minmax_xy)
    label_bars(b2, 'enso_pct', 'enso', enso_minmax_xy)
    label_bars(b3, 'seasonal_pct', 'seasonal', seasonal_minmax_xy)
    if b4 is not None:
        label_bars(b4, 'ext_var_pct', 'ext_var')

    # -- Dynamic margins: account for error bars, the 2-line text AND the
    #    min/max markers, so that the plot NEVER overflows the frame. --
    if ci_by_episode is not None:
        top_candidates = [summary['trend'].iloc[i] + _err_from_ci(summary['trend'].iloc[i], ci_by_episode[l]['trend'])[1]
                           for i, l in enumerate(labels)]
        bottom_candidates = [summary['seasonal'].iloc[i] - _err_from_ci(summary['seasonal'].iloc[i], ci_by_episode[l]['seasonal'])[0]
                              for i, l in enumerate(labels)]
    else:
        top_candidates = list(summary['trend'])
        bottom_candidates = list(summary['seasonal'])
    if has_ext:
        ext_vals = summary['ext_var'].fillna(0.0)
        top_candidates = [max(t, e) for t, e in zip(top_candidates, ext_vals)]
        bottom_candidates = [min(b, e) for b, e in zip(bottom_candidates, ext_vals)]
    for xy in (trend_minmax_xy, enso_minmax_xy, seasonal_minmax_xy):
        if xy:
            top_candidates.append(max(mx for _, mx in xy.values()))
            bottom_candidates.append(min(mn for mn, _ in xy.values()))

    y_top = max(top_candidates) * 1.42
    y_bottom = min(0.0, min(bottom_candidates)) - 0.18

    for i, l in enumerate(labels):
        ax.text(x[i], y_top * 0.985, f"Total: {summary['total'].iloc[i]:+.2f}°C",
                ha='center', va='top', fontsize=9.5, style='italic', color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.axhline(0, color='#333333', lw=0.8)
    ax.set_ylabel("Mean contribution to the GMSTA anomaly (°C, ref. 1850-1900)")
    ax.set_ylim(y_bottom, y_top)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2 if has_ext else 3,
              fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    fig.subplots_adjust(bottom=0.24 if has_ext else 0.20)
    subtitle = ("4 El Nino episodes (July(n) to June(n+1)), episode average - TESR model, "
                "exact linear decomposition (ridge); ENSO = Total - Counterfactual (Trend+Seasonal)")
    if ci_by_episode is not None:
        subtitle += " ; error bars = 90% CI (moving-block bootstrap, n=300)"
    if has_ext:
        subtitle += " ; 4th bar = verification-model gap (n.a. = months not yet observed)"
    if enso_minmax_xy or trend_minmax_xy or seasonal_minmax_xy:
        subtitle += (" ; triangles = min/max by component (ENSO: multi-model spread + bootstrap "
                     "of the model; Trend/Seasonal: bootstrap P5/P95 bounds)")
    header_in = _draw_header(
        fig,
        fig_h,
        "Comparative attribution: anthropogenic vs natural (ENSO) vs seasonal contribution",
        subtitle
    )
    fig.subplots_adjust(top=1 - header_in / fig_h)
    fig.text(
        0.98,
        0.01,
        "Data: ECMWF/Copernicus C3S - ERA5; Nino3.4 ClimateReanalyzer (ref. 1850-1900)",
        fontsize=7.5,
        color='#666666',
        ha='right'
    )
    plt.savefig(filename, dpi=300)
    plt.close()
    plt.rcParams['font.family'] = 'sans-serif'
    return summary


# ----------------------------------------------------------------------
# 4. PERCENTAGE BAR CHART (episode average, %) + 90% CI
# ----------------------------------------------------------------------
def plot_bar_pourcentage(decomps, labels, ci_by_episode=None, verifs=None, enso_minmax_pct=None,
                          filename="gmst_episodes_percentage_bars.png"):
    """
    enso_minmax_pct : optional dict {label: (combined_min_%, combined_max_%)} --
        same principle as enso_minmax in plot_bar_comparatif_moyenne, but
        expressed as % of the total (combined 90% CI from ENSO spread
        + model bootstrap uncertainty).
    """
    summary = _make_summary(decomps, labels, verifs=verifs)
    has_ext = verifs is not None and summary['ext_var'].notna().any()

    plt.rcParams['font.family'] = 'serif'
    fig_h = 8.0
    fig, ax = plt.subplots(figsize=(11.5, fig_h))

    x = np.arange(len(labels))
    width = 0.19 if has_ext else 0.26
    offsets = [-1.5, -0.5, 0.5, 1.5] if has_ext else [-1, 0, 1]

    b1 = ax.bar(
        x + offsets[0] * width,
        summary['trend_pct'],
        width,
        color=COLOR_TREND,
        label="Trend (anthropogenic)"
    )
    b2 = ax.bar(
        x + offsets[1] * width,
        summary['enso_pct'],
        width,
        color=COLOR_ENSO,
        label="ENSO (natural)"
    )
    b3 = ax.bar(
        x + offsets[2] * width,
        summary['seasonal_pct'],
        width,
        color=COLOR_SEAS,
        label="Residual seasonal"
    )
    bars_comp = [(b1, 'trend_pct'), (b2, 'enso_pct'), (b3, 'seasonal_pct')]
    b4 = None
    if has_ext:
        ext_pct_plot = summary['ext_var_pct'].fillna(0.0)
        b4 = ax.bar(
            x + offsets[3] * width,
            ext_pct_plot,
            width,
            color=COLOR_EXT,
            label="ENSO-external variability (verification minus model)"
        )
        bars_comp.append((b4, 'ext_var_pct'))

    if ci_by_episode is not None:
        for bars, comp in ((b1, 'trend_pct'), (b2, 'enso_pct'), (b3, 'seasonal_pct')):
            xs, ys, errs_lo, errs_hi = [], [], [], []
            for i, l in enumerate(labels):
                center = bars[i].get_height()
                ci = ci_by_episode[l][comp]
                lo, hi = _err_from_ci(center, ci)
                xs.append(bars[i].get_x() + bars[i].get_width() / 2)
                ys.append(center)
                errs_lo.append(lo)
                errs_hi.append(hi)
            ax.errorbar(
                xs,
                ys,
                yerr=[errs_lo, errs_hi],
                fmt='none',
                ecolor='#1a1a1a',
                elinewidth=1.3,
                capsize=4,
                capthick=1.3,
                zorder=5
            )

    # -- Min/max points, CENTERED on each relevant bar (calculated before
    #    the labels so that the text can be pushed beyond the most extreme
    #    triangle) -- ENSO: combined uncertainty from multi-model spread
    #    + bootstrap; Trend/Seasonal: bootstrap P5/P95 bounds, in %. --
    # -- Min/max triangles restricted to labels present in enso_minmax_pct
    #    (in practice, only "2026-2027 (projection)") -- see the same
    #    note as in plot_bar_comparatif_moyenne. --
    minmax_labels = set(enso_minmax_pct.keys()) if enso_minmax_pct else set()
    trend_minmax_pct = (
        {
            l: (
                ci_by_episode[l]['trend_pct'][0],
                ci_by_episode[l]['trend_pct'][2]
            )
            for l in labels
            if l in minmax_labels
        }
        if ci_by_episode is not None else None
    )
    seasonal_minmax_pct = (
        {
            l: (
                ci_by_episode[l]['seasonal_pct'][0],
                ci_by_episode[l]['seasonal_pct'][2]
            )
            for l in labels
            if l in minmax_labels
        }
        if ci_by_episode is not None else None
    )

    trend_minmax_xy = _add_minmax_markers(
        ax,
        b1,
        labels,
        trend_minmax_pct
    )

    enso_minmax_xy = _add_minmax_markers(
        ax,
        b2,
        labels,
        enso_minmax_pct,
        legend_labels=(
            'Max (ENSO spread + model uncertainty)',
            'Min (ENSO spread + model uncertainty)'
        )
    )

    seasonal_minmax_xy = _add_minmax_markers(
        ax,
        b3,
        labels,
        seasonal_minmax_pct
    )

    minmax_by_comp = {
        'trend_pct': trend_minmax_xy,
        'enso_pct': enso_minmax_xy,
        'seasonal_pct': seasonal_minmax_xy
    }

    for bars, comp in bars_comp:
        for i, bar in enumerate(bars):
            h = bar.get_height()
            is_nan = comp == 'ext_var_pct' and pd.isna(summary['ext_var_pct'].iloc[i])
            lo_err = hi_err = 0.0

            if ci_by_episode is not None and comp in ci_by_episode.get(labels[i], {}):
                lo_err, hi_err = _err_from_ci(
                    h,
                    ci_by_episode[labels[i]][comp]
                )

            top_extent = h + hi_err
            bottom_extent = h - lo_err

            mm = minmax_by_comp.get(comp)
            if mm is not None and labels[i] in mm:
                mn, mx = mm[labels[i]]
                top_extent = max(top_extent, mx)
                bottom_extent = min(bottom_extent, mn)

            va = 'bottom' if h >= 0 else 'top'
            y_anchor = (
                (top_extent + 1.5)
                if h >= 0
                else (bottom_extent - 1.5)
            )

            txt = "n.a." if is_nan else f"{h:+.0f}%"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_anchor,
                txt,
                ha='center',
                va=va,
                fontsize=9,
                fontweight='bold',
                color='#1a1a1a'
            )

    if ci_by_episode is not None:
        top_c = [
            max(
                summary['trend_pct'].iloc[i]
                + _err_from_ci(
                    summary['trend_pct'].iloc[i],
                    ci_by_episode[l]['trend_pct']
                )[1],
                100
            )
            for i, l in enumerate(labels)
        ]

        bot_c = [
            summary['seasonal_pct'].iloc[i]
            - _err_from_ci(
                summary['seasonal_pct'].iloc[i],
                ci_by_episode[l]['seasonal_pct']
            )[0]
            for i, l in enumerate(labels)
        ]
    else:
        top_c = [100]
        bot_c = list(summary['seasonal_pct'])

    if has_ext:
        ext_pct_vals = summary['ext_var_pct'].fillna(0.0)
        top_c = top_c + list(ext_pct_vals)
        bot_c = bot_c + list(ext_pct_vals)

    for xy in (trend_minmax_xy, enso_minmax_xy, seasonal_minmax_xy):
        if xy:
            top_c.append(max(mx for _, mx in xy.values()))
            bot_c.append(min(mn for mn, _ in xy.values()))

    y_bottom = min(0.0, min(bot_c)) - 10
    y_top = max(top_c) + 10

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.axhline(0, color='#333333', lw=0.8)
    ax.axhline(100, color='#999999', lw=0.6, ls=':')
    ax.set_ylabel("Share of total GMSTA anomaly (%)")
    ax.set_ylim(y_bottom, y_top)

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.13),
        ncol=2 if has_ext else 3,
        fontsize=10,
        framealpha=0.9
    )

    ax.grid(axis='y', alpha=0.3)

    fig.subplots_adjust(bottom=0.24 if has_ext else 0.20)

    subtitle = (
        "4 El Nino episodes (July(n) to June(n+1)) - the 3 signed shares "
        "sum to 100% by construction; ENSO = Total - Counterfactual "
        "(Trend+Seasonal)"
    )

    if ci_by_episode is not None:
        subtitle += " ; 90% CI (moving-block bootstrap, n=300)"

    if has_ext:
        subtitle += (
            " ; 4th bar = verification-model gap as % of total "
            "(n.a. = months not yet observed)"
        )

    if enso_minmax_xy or trend_minmax_xy or seasonal_minmax_xy:
        subtitle += (
            " ; triangles = min/max by component "
            "(ENSO: multi-model spread + model bootstrap; "
            "Trend/Seasonal: bootstrap P5/P95 bounds)"
        )

    header_in = _draw_header(
        fig,
        fig_h,
        "Percentage breakdown of the total anomaly: "
        "anthropogenic vs natural vs seasonal",
        subtitle
    )

    fig.subplots_adjust(top=1 - header_in / fig_h)

    fig.text(
        0.98,
        0.01,
        "Data: ECMWF/Copernicus C3S - ERA5; "
        "Nino3.4 ClimateReanalyzer (ref. 1850-1900)",
        fontsize=7.5,
        color='#666666',
        ha='right'
    )

    plt.savefig(filename, dpi=300)
    plt.close()
    plt.rcParams['font.family'] = 'sans-serif'

    return summary


# ----------------------------------------------------------------------
# 5. DISTINCT FUNCTION: real scenario vs counterfactual, 1 file / episode
#    (unchanged, not affected by the CI request for the bars)
# ----------------------------------------------------------------------
def plot_scenario_vs_contrefactuel_single(
    label,
    decomp,
    lag,
    verif=None,
    ci_monthly=None,
    envelope=None,
    filename=None
):
    """
    envelope : optional dict {'p25','p75','p05','p95','q0','q100'} ->
        {'ci_monthly': DataFrame, 'ci_mean': dict, 'decomp_central': DataFrame},
        as produced by build_pipeline()['enso_uncertainty'][label].

        IMPORTANT -- two clearly DISTINCT sources of uncertainty are displayed
        separately (they were previously stacked into the same band, which
        visually mixed them together):

          1. MULTI-MODEL ENSO SPREAD (solid blue bands + dotted lines):
             the CENTRAL model decomposition (without bootstrap,
             'decomp_central') applied to each ENSO scenario
             (q0/p05/p25/p75/p95/q100). This is a DETERMINISTIC quantity --
             "if this ENSO scenario occurs, this is what the central model
             predicts" -- NOT a statistical uncertainty. Three nested tiers:
             Q0-Q100 (total spread), Q5-Q95, Q25-Q75 (core of the
             distribution); the Q0/Q100 bounds are additionally plotted as
             dotted lines to identify the exact extreme envelope.

          2. TESR MODEL'S OWN UNCERTAINTY: moving-block bootstrap
             (Künsch 1989), displayed in TWO complementary ways --
             (a) as error bars on the central red/grey curve
                 (median ENSO scenario);
             (b) as a DOTTED FRINGE ("grid of dots", hatch, no solid fill)
                 extending EACH edge of the 3 tiers above to the P5/P95 tail
                 of the bootstrap specific to that scenario --
                 without it, the true uncertainty at each quantile would be
                 underestimated (optimistic bias in the displayed
                 probabilities); with solid fill, it would be mixed back
                 together with the ENSO spread (the flaw of a previous
                 version). The fringe is therefore visible and measurable,
                 but visually secondary relative to the solid bands.

        This is meaningful only for a projected scenario (multi-model ENSO);
        historical episodes (observed ENSO) do not provide this information.
    """
    if filename is None:
        safe = label.replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"gmst_scenario_vs_counterfactual_{safe}.png"

    counterfactual = decomp['trend'] + decomp['seasonal']
    idx = decomp.index

    plt.rcParams['font.family'] = 'serif'
    fig_w, fig_h = 14.5, 8.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # -- 1) MULTI-MODEL ENSO SPREAD ONLY (deterministic, decomp_central
    #    without bootstrap) -- 3 nested tiers from the widest (Q0-Q100)
    #    to the narrowest (Q25-Q75), with increasingly saturated blue
    #    tones toward the center. --
    env_bounds = None

    if envelope is not None:

        def _central(col):
            return envelope[col]['decomp_central']['total'].reindex(idx)

        def _tail(col, which):
            return envelope[col]['ci_monthly'][which].reindex(idx)

        # -- Colors: QUANTILE_BANDS (palette.py) -- 3 blue shades
        #    genuinely distinct in hue + brightness (not simply stacking
        #    alpha on the same color, which washed everything into an almost
        #    uniform navy blue -- see the documented gotcha in
        #    palette.py/light_tint: a Patch-level alpha applied IN ADDITION
        #    to an RGBA color overrides the alpha embedded in that RGBA,
        #    causing all 3 tiers to render at alpha~0.9 with the FULL hue
        #    of COUNTERFACT -- hence the poorly readable "dark blue shades"). --
        tiers = [
            (
                'q0',
                'q100',
                QUANTILE_BANDS[0],
                1.0,
                1,
                'Multi-model ENSO spread (Q0-Q100)'
            ),
            (
                'p05',
                'p95',
                QUANTILE_BANDS[1],
                1.0,
                2,
                'Multi-model ENSO spread (Q5-Q95)'
            ),
            (
                'p25',
                'p75',
                QUANTILE_BANDS[2],
                1.0,
                3,
                'Multi-model ENSO spread (Q25-Q75)'
            ),
        ]

        for lo_col, hi_col, color, alpha, z, lbl in tiers:
            ax.fill_between(
                idx,
                _central(lo_col),
                _central(hi_col),
                color=color,
                alpha=alpha,
                lw=0,
                zorder=z,
                label=lbl
            )

        ax.plot(
            idx,
            _central('q0'),
            color=COUNTERFACT,
            lw=1.1,
            ls=':',
            zorder=4
        )

        ax.plot(
            idx,
            _central('q100'),
            color=COUNTERFACT,
            lw=1.1,
            ls=':',
            zorder=4,
            label='Lowest / highest projection (Q0/Q100)'
        )

        # -- 1bis) MODEL'S OWN UNCERTAINTY, PROPAGATED TO EACH QUANTILE
        #    (not only to the median scenario): omitting it would bias the
        #    displayed probabilities (ENSO-only bands underestimate the
        #    true uncertainty at each bound); however, stacking it as a
        #    solid fill, as in a previous version, mixes it with the ENSO
        #    spread and makes the graph unreadable. Compromise: a DOTTED
        #    FRINGE ("grid of dots", hatch='....', no solid fill) extending
        #    each tier edge to the P5/P95 tail of the bootstrap FOR THIS
        #    SCENARIO -- visible and measurable if needed, but visually
        #    subordinate to the solid bands (ENSO spread = primary reading;
        #    dotted fringe = secondary statistical uncertainty). Only one
        #    tier -- Q25-Q75, the innermost one -- carries the legend label
        #    (all 3 tiers share the same style, so one legend entry per tier
        #    would be redundant). --
        fringe_kw = dict(
            facecolor='none',
            hatch='.',
            linewidth=0.0,
            alpha=0.38,
            zorder=3.6
        )

        _hatch_lw_saved = plt.rcParams['hatch.linewidth']
        plt.rcParams['hatch.linewidth'] = 0.5  # fine hatch lines -> "fringe" reading, not "texture"

        outer_lo, outer_hi = (
            _tail('q0', 'total_p5'),
            _tail('q100', 'total_p95')
        )

        ax.fill_between(
            idx,
            outer_lo,
            _central('q0'),
            edgecolor=COUNTERFACT,
            **fringe_kw
        )

        ax.fill_between(
            idx,
            _central('q100'),
            outer_hi,
            edgecolor=COUNTERFACT,
            **fringe_kw
        )

        mid_lo, mid_hi = (
            _tail('p05', 'total_p5'),
            _tail('p95', 'total_p95')
        )

        ax.fill_between(
            idx,
            mid_lo,
            _central('p05'),
            edgecolor=COUNTERFACT,
            **fringe_kw
        )

        ax.fill_between(
            idx,
            _central('p95'),
            mid_hi,
            edgecolor=COUNTERFACT,
            **fringe_kw
        )

        in_lo, in_hi = (
            _tail('p25', 'total_p5'),
            _tail('p75', 'total_p95')
        )

        ax.fill_between(
            idx,
            in_lo,
            _central('p25'),
            edgecolor=COUNTERFACT_DARK,
            **fringe_kw
        )

        ax.fill_between(
            idx,
            _central('p75'),
            in_hi,
            edgecolor=COUNTERFACT_DARK,
            **fringe_kw,
            label="Model's own uncertainty at each quantile\n"
                  "(bootstrap, dotted band)"
        )

        plt.rcParams['hatch.linewidth'] = _hatch_lw_saved

        env_bounds = (
            outer_lo,
            outer_hi
        )  # actual vertical extent = ENSO spread + model uncertainty

    # -- 2) MODEL'S OWN UNCERTAINTY (moving-block bootstrap), WITH THE
    #    MEDIAN ENSO SCENARIO FIXED -- error bars ONLY, never mixed back
    #    into an ENSO spread band. --
    has_ci = ci_monthly is not None

    if has_ci:
        ci_al = ci_monthly.reindex(idx)  # align in case the index differs

        lo_tot, hi_tot = _err_from_ci_series(
            decomp['total'],
            ci_al['total_p5'],
            ci_al['total_p95']
        )

        lo_cf, hi_cf = _err_from_ci_series(
            counterfactual,
            ci_al['counterfactual_p5'],
            ci_al['counterfactual_p95']
        )

        ax.errorbar(
            idx,
            decomp['total'],
            yerr=[lo_tot, hi_tot],
            color=COLOR_TREND,
            lw=2.2,
            marker='o',
            ms=5,
            capsize=3,
            elinewidth=1.1,
            ecolor=COLOR_TREND,
            alpha=0.95,
            zorder=6,
            label='Model prediction (with El Nino)'
        )

        ax.errorbar(
            idx,
            counterfactual,
            yerr=[lo_cf, hi_cf],
            color='#555555',
            lw=2.0,
            ls='--',
            marker='o',
            ms=5,
            capsize=3,
            elinewidth=1.1,
            ecolor='#555555',
            alpha=0.85,
            zorder=6,
            label='Counterfactual (ENSO-neutral)'
        )

    else:
        ax.plot(
            idx,
            decomp['total'],
            color=COLOR_TREND,
            lw=2.2,
            marker='o',
            ms=5,
            zorder=6,
            label='Central model prediction (with El Nino)'
        )

        ax.plot(
            idx,
            counterfactual,
            color='#555555',
            lw=2.0,
            ls='--',
            marker='o',
            ms=5,
            zorder=6,
            label='Counterfactual (ENSO-neutral)'
        )

    # -- 3) EL NINO CONTRIBUTION (median scenario) -- RED HATCHED AREA,
    #    so that it never becomes visually confused with the blue ENSO-spread
    #    bands above (nearly transparent fill + diagonal red hatching,
    #    as in IPCC attribution diagrams). --
    ax.fill_between(
        idx,
        counterfactual,
        decomp['total'],
        facecolor=COLOR_ENSO,
        alpha=0.12,
        zorder=4,
        lw=0
    )

    ax.fill_between(
        idx,
        counterfactual,
        decomp['total'],
        facecolor='none',
        edgecolor=COLOR_ENSO,
        hatch='///',
        linewidth=0.0,
        zorder=4.5,
        label='El Nino contribution (hatched area)'
    )

    # -- Verification line (actual measured ERA5/C3S observation),
    #    in black, only for months already observed (NaN otherwise --
    #    matplotlib skips these points, so the line simply stops). --
    ext_var_mean = None
    n_obs = 0
    verif_aligned = None

    if verif is not None:
        verif_aligned = verif.reindex(idx)
        n_obs = int(verif_aligned.notna().sum())

        if n_obs > 0:
            ax.plot(
                idx,
                verif_aligned,
                color=COLOR_VERIF,
                lw=2.0,
                ls='-',
                marker='s',
                ms=5,
                zorder=7,
                label='Verification (observed ERA5/C3S)'
            )

            ext_var = compute_external_variability(
                decomp,
                verif_aligned
            )

            ext_var_mean = ext_var.mean(skipna=True)

    # -- Reference thresholds (same styles as model.py): always plotted,
    #    but visible only if the episode approaches them. --
    threshold_styles = {
        1.5: (
            ':',
            '#888888',
            '+1.5 \u00b0C threshold (Paris Agreement)'
        ),
        2.0: (
            '--',
            '#555555',
            '+2 \u00b0C threshold'
        )
    }

    y_data_max = max(
        decomp['total'].max(),
        counterfactual.max()
    )

    shown_thresholds = []

    for th, (ls, col, lab) in threshold_styles.items():
        if th <= y_data_max * 1.25:
            ax.axhline(
                th,
                color=col,
                linestyle=ls,
                lw=1.1,
                label=lab
            )
            shown_thresholds.append(th)

    # -- EXPLICIT vertical extent: includes the curve, model CI,
    #    the 3 combined envelope bands and the verification -- with
    #    generous margins, so that the displayed extremes are never
    #    cropped. --
    y_series = [
        decomp['total'],
        counterfactual
    ]

    if has_ci:
        y_series += [
            ci_al['total_p5'],
            ci_al['total_p95'],
            ci_al['counterfactual_p5'],
            ci_al['counterfactual_p95']
        ]

    if env_bounds is not None:
        y_series += [
            env_bounds[0],
            env_bounds[1]
        ]

    if n_obs > 0:
        y_series.append(verif_aligned)

    all_y = np.concatenate([
        s.to_numpy(dtype=float)
        for s in y_series
    ])

    all_y = all_y[~np.isnan(all_y)]

    y_min, y_max = all_y.min(), all_y.max()

    if shown_thresholds:
        y_max = max(
            y_max,
            max(shown_thresholds)
        )

    y_span = max(
        y_max - y_min,
        0.2
    )

    ax.set_ylim(
        y_min - 0.15 * y_span,
        y_max + 0.15 * y_span
    )

    ax.set_ylabel(
        "Global mean surface temperature anomaly "
        "(\u00b0C) [1850-1900 baseline]"
    )

    ax.grid(alpha=0.3)

    ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=1)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter('%b %Y')
    )

    fig.autofmt_xdate(rotation=45)

    # -- Header (title + subtitle) anchored in INCHES from the top,
    #    not as a fraction of the figure -- independent of the number
    #    of subtitle lines (automatic wrapping) and figure size, so it
    #    never overflows regardless of the content. --
    # -- Title: ADAPTED TO EACH EPISODE (previously a single hard-coded
    #    text, "El Nino could push global temperature towards +2C in early
    #    2027", reused unchanged for all 4 calls in the loop -- therefore
    #    also displayed for 1982-1983/1997-1998/2015-2016, unrelated to
    #    their actual content). Neutral, factual wording without
    #    journalistic phrasing ("could push"), consistent with use in a
    #    research paper: model name, episode, nature (observed retrospective
    #    vs projection), and the metric actually shown on the figure
    #    (peak anomaly). --
    is_projection = "projection" in label.lower()
    episode_clean = label.replace(" (projection)", "").strip()
    peak_val = float(decomp['total'].max())
    peak_date = decomp['total'].idxmax()

    if is_projection:
        title_txt = (
            f"TESR-modelled global temperature anomaly under the projected "
            f"{episode_clean} El Nino ({peak_val:+.2f} \u00b0C peak, "
            f"{peak_date:%b %Y})"
        )
    else:
        title_txt = (
            f"TESR-modelled El Nino contribution to the global temperature "
            f"anomaly, {episode_clean} ({peak_val:+.2f} \u00b0C peak, "
            f"{peak_date:%b %Y})"
        )

    subtitle_ci = (
        "Projection initialized 1 August 2026"
        if (has_ci and is_projection)
        else ""
    )

    subtitle_txt = (
        f"Modelled anomaly with El Nino vs ENSO-neutral counterfactual - "
        f"Departure from preindustrial (1850-1900) baseline, \u00b0C - "
        f"TESR model, lag={lag} months          "
        f"{subtitle_ci}"
    )

    subtitle_wrapped = textwrap.fill(
        subtitle_txt,
        width=150
    )

    n_sub_lines = subtitle_wrapped.count("\n") + 1

    title_top_in = 0.40
    subtitle_top_in = title_top_in + 0.32
    ext_top_in = subtitle_top_in + 0.20 * n_sub_lines + 0.08
    header_in = ext_top_in + (
        0.28 if n_obs > 0 else 0.08
    )

    fig.suptitle(
        title_txt,
        fontsize=13.5,
        fontweight='bold',
        x=0.02,
        ha='left',
        va='top',
        y=1 - title_top_in / fig_h
    )

    fig.text(
        0.02,
        1 - subtitle_top_in / fig_h,
        subtitle_wrapped,
        fontsize=9.5,
        style='italic',
        color='#444444',
        ha='left',
        va='top'
    )

    if n_obs > 1 and ext_var_mean is not None:
        fig.text(
            0.02,
            1 - ext_top_in / fig_h,
            f"Mean verification \u2212 model gap over the observed period: "
            f"{ext_var_mean:+.2f} \u00b0C (n = {n_obs} months)",
            fontsize=9,
            fontweight='bold',
            color='#1a1a1a',
            ha='left',
            va='top'
        )

    # -- Legend OUTSIDE the frame (on the right) -- wider figure +
    #    dedicated right margin + reduced font size so that the 3 new
    #    envelope entries fit without being clipped. --
    fig.subplots_adjust(
        top=1 - header_in / fig_h,
        bottom=0.16,
        right=0.71
    )

    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8.3,
        framealpha=0.9
    )

    fig.text(
        0.98,
        0.01,
        "Data: ECMWF/Copernicus C3S - ERA5; "
        "Nino3.4 ClimateReanalyzer (1850-1900 baseline)",
        fontsize=7.5,
        color='#666666',
        ha='right'
    )

    plt.savefig(filename, dpi=300)
    plt.close()
    plt.rcParams['font.family'] = 'sans-serif'

    return filename
# ----------------------------------------------------------------------
# 6. DISTINCT FUNCTION: comparative bars AT ENSO PEAK + 90% CI
# ----------------------------------------------------------------------
def plot_bar_paroxysme_enso(decomps, labels, ci_by_episode_peak=None, peak_dates=None, verifs=None,
                             enso_uncertainty=None,
                             filename="gmst_episodes_enso_peak_bars.png"):
    """
    enso_uncertainty : optional dict {label: {'q0': {'ci_monthly': DataFrame}, 'q100': {...}, ...}},
        as produced by build_pipeline(). Used to derive the combined min/max
        (ENSO spread + model bootstrap uncertainty) OF THE SPECIFIC ENSO PEAK MONTH
        (not an episode average) -- q0/ci_monthly for the lower bound (P5),
        q100/ci_monthly for the upper bound (P95).
        The SAME logic (P5 under q0 / P95 under q100) is also applied to
        Trend and Seasonal: this is MODEL-INHERENT uncertainty
        (the bootstrap jointly refits trend + ENSO + seasonal components),
        not uncertainty specific to the ENSO component alone.
        Therefore, the min/max triangles are displayed ONLY for labels present
        in enso_uncertainty (in practice, only "2026-2027 (projection)"
        -- historical episodes use observed ENSO, without a
        multi-model scenario to propagate).
    """
    rows = []
    dates_found = {}
    enso_minmax_peak = {}
    trend_minmax_peak = {}
    seasonal_minmax_peak = {}
    for l in labels:
        d = decomps[l].copy()
        d['enso_pct'] = 100 * d['enso'] / d['total']
        idx_paroxysme = d['enso_pct'].idxmax()
        dates_found[l] = idx_paroxysme
        row = d.loc[idx_paroxysme]
        if enso_uncertainty is not None and l in enso_uncertainty:
            bands = enso_uncertainty[l]
            cm_q0, cm_q100 = bands['q0']['ci_monthly'], bands['q100']['ci_monthly']
            if idx_paroxysme in cm_q0.index and idx_paroxysme in cm_q100.index:
                enso_minmax_peak[l] = (cm_q0.loc[idx_paroxysme, 'enso_p5'],
                                        cm_q100.loc[idx_paroxysme, 'enso_p95'])
                # -- Propagation of ENSO min/max uncertainty to the 2 OTHER
                #    components (Trend, Seasonal): MODEL-INHERENT uncertainty,
                #    not uncertainty specific to these components -- the
                #    moving-block bootstrap refits the ENTIRE model
                #    (trend + ENSO + seasonal components together)
                #    on resampled residuals, so an extreme ENSO scenario
                #    (q0/q100) also produces a margin on Trend
                #    and Seasonal -- this is not visible when using only
                #    the bootstrap CI from the central scenario
                #    (ci_by_episode_peak).
                #    We therefore take, as for ENSO, the lower bound (P5) under
                #    the low ENSO scenario (q0) and the upper bound (P95) under
                #    the high ENSO scenario (q100). --
                trend_minmax_peak[l] = (cm_q0.loc[idx_paroxysme, 'trend_p5'],
                                         cm_q100.loc[idx_paroxysme, 'trend_p95'])
                seasonal_minmax_peak[l] = (cm_q0.loc[idx_paroxysme, 'seasonal_p5'],
                                            cm_q100.loc[idx_paroxysme, 'seasonal_p95'])
        # -- External variability AT THE PEAK MONTH (not the episode average):
        #    verification minus model for this specific month, NaN if the
        #    month has not yet been observed (e.g. end of the 2026-2027 projection) --
        ext_var = np.nan
        if verifs is not None and l in verifs:
            v_at = verifs[l].reindex(d.index).loc[idx_paroxysme]
            if pd.notna(v_at):
                ext_var = v_at - row['total']
        rows.append({
            'episode': l,
            'date_paroxysme_enso': idx_paroxysme,
            'total': row['total'], 'trend': row['trend'], 'enso': row['enso'],
            'seasonal': row['seasonal'],
            'trend_pct': 100 * row['trend'] / row['total'],
            'enso_pct': 100 * row['enso'] / row['total'],
            'seasonal_pct': 100 * row['seasonal'] / row['total'],
            'ext_var': ext_var,
            'ext_var_pct': 100 * ext_var / row['total'] if pd.notna(ext_var) else np.nan,
        })
    synth = pd.DataFrame(rows).set_index('episode')
    has_ext = verifs is not None and synth['ext_var'].notna().any()

    plt.rcParams['font.family'] = 'serif'
    fig_h = 8.4
    fig, ax = plt.subplots(figsize=(11.5, fig_h))

    x = np.arange(len(labels))
    width = 0.19 if has_ext else 0.26
    offsets = [-1.5, -0.5, 0.5, 1.5] if has_ext else [-1, 0, 1]

    b1 = ax.bar(x + offsets[0] * width, synth['trend'], width,
                color=COLOR_TREND, label="Trend (anthropogenic)")
    b2 = ax.bar(x + offsets[1] * width, synth['enso'], width,
                color=COLOR_ENSO, label="ENSO (natural)")
    b3 = ax.bar(x + offsets[2] * width, synth['seasonal'], width,
                color=COLOR_SEAS, label="Residual seasonal")
    b4 = None
    if has_ext:
        ext_plot = synth['ext_var'].fillna(0.0)
        b4 = ax.bar(x + offsets[3] * width, ext_plot, width,
                    color=COLOR_EXT,
                    label="ENSO-external variability (verification minus model, at peak)")

    if ci_by_episode_peak is not None:
        for bars, comp in ((b1, 'trend'), (b2, 'enso'), (b3, 'seasonal')):
            xs, ys, errs_lo, errs_hi = [], [], [], []
            for i, l in enumerate(labels):
                center = bars[i].get_height()
                ci = ci_by_episode_peak[l][comp]
                lo, hi = _err_from_ci(center, ci)
                xs.append(bars[i].get_x() + bars[i].get_width() / 2)
                ys.append(center)
                errs_lo.append(lo)
                errs_hi.append(hi)
            ax.errorbar(xs, ys, yerr=[errs_lo, errs_hi], fmt='none',
                        ecolor='#1a1a1a', elinewidth=1.3, capsize=4, capthick=1.3, zorder=5)

    # -- Min/max points, CENTERED on each bar, calculated BEFORE the
    #    labels (so that the text is pushed beyond the most extreme
    #    triangle) -- ENSO: combined uncertainty at the specific
    #    peak month; Trend/Seasonal: P5/P95 bounds from the bootstrap
    #    at the same month --
    # -- trend_minmax_peak / seasonal_minmax_peak are now built
    #    ABOVE (inside the rows loop), from the q0/q100 bands of
    #    enso_uncertainty -- therefore automatically restricted to labels
    #    for which ENSO spread exists (in practice, 2026-2027 (projection)),
    #    and propagating the ENSO min/max uncertainty to these 2 components
    #    rather than using the bootstrap CI of the central scenario. --
    trend_minmax_xy = _add_minmax_markers(ax, b1, labels, trend_minmax_peak)
    enso_minmax_xy = _add_minmax_markers(
        ax, b2, labels, enso_minmax_peak,
        legend_labels=('Max (ENSO spread + model uncertainty)',
                       'Min (ENSO spread + model uncertainty)'))
    seasonal_minmax_xy = _add_minmax_markers(ax, b3, labels, seasonal_minmax_peak)
    minmax_by_comp = {
        'trend': trend_minmax_xy,
        'enso': enso_minmax_xy,
        'seasonal': seasonal_minmax_xy
    }

    def label_bars(bars, pct_col, comp):
        for i, bar in enumerate(bars):
            h = bar.get_height()
            pct = synth[pct_col].iloc[i]
            is_nan = comp == 'ext_var' and pd.isna(synth['ext_var'].iloc[i])
            lo_err = hi_err = 0.0
            if ci_by_episode_peak is not None and comp in ci_by_episode_peak.get(labels[i], {}):
                lo_err, hi_err = _err_from_ci(h, ci_by_episode_peak[labels[i]][comp])
            top_extent = h + hi_err
            bottom_extent = h - lo_err
            mm = minmax_by_comp.get(comp)
            if mm is not None and labels[i] in mm:
                mn, mx = mm[labels[i]]
                top_extent = max(top_extent, mx)
                bottom_extent = min(bottom_extent, mn)
            va = 'bottom' if h >= 0 else 'top'
            y_anchor = (top_extent + 0.015) if h >= 0 else (bottom_extent - 0.015)
            txt = "n.d." if is_nan else f"{h:+.2f}°C\n({pct:+.0f}%)"
            ax.text(bar.get_x() + bar.get_width() / 2, y_anchor, txt, ha='center', va=va,
                    fontsize=8.3, fontweight='bold', color='#1a1a1a')

    label_bars(b1, 'trend_pct', 'trend')
    label_bars(b2, 'enso_pct', 'enso')
    label_bars(b3, 'seasonal_pct', 'seasonal')
    if b4 is not None:
        label_bars(b4, 'ext_var_pct', 'ext_var')

    if ci_by_episode_peak is not None:
        top_candidates = [
            synth['trend'].iloc[i] +
            _err_from_ci(synth['trend'].iloc[i], ci_by_episode_peak[l]['trend'])[1]
            for i, l in enumerate(labels)
        ]
        bottom_candidates = [
            synth['seasonal'].iloc[i] -
            _err_from_ci(synth['seasonal'].iloc[i], ci_by_episode_peak[l]['seasonal'])[0]
            for i, l in enumerate(labels)
        ]
    else:
        top_candidates = list(synth['trend'])
        bottom_candidates = list(synth['seasonal'])

    if has_ext:
        ext_vals = synth['ext_var'].fillna(0.0)
        top_candidates = [max(t, e) for t, e in zip(top_candidates, ext_vals)]
        bottom_candidates = [min(b, e) for b, e in zip(bottom_candidates, ext_vals)]

    for xy in (trend_minmax_xy, enso_minmax_xy, seasonal_minmax_xy):
        if xy:
            top_candidates.append(max(mx for _, mx in xy.values()))
            bottom_candidates.append(min(mn for mn, _ in xy.values()))

    y_top = max(top_candidates) * 1.50
    y_bottom = min(0.0, min(bottom_candidates)) - 0.18

    for i, l in enumerate(labels):
        date_str = synth['date_paroxysme_enso'].iloc[i].strftime('%b %Y')
        ax.text(x[i], y_top * 0.985,
                f"Total: {synth['total'].iloc[i]:+.2f} °C\n({date_str})",
                ha='center', va='top', fontsize=9, style='italic', color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.axhline(0, color='#333333', lw=0.8)
    ax.set_ylabel("Contribution to the GMSTA anomaly (°C, ref. 1850-1900)")
    ax.set_ylim(y_bottom, y_top)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2 if has_ext else 3,
              fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    fig.subplots_adjust(bottom=0.24 if has_ext else 0.20)

    # -- Title/subtitle: REFORMULATED for preprint use --
    #    the old title ("El Nino could add +0.36C to global warming in
    #    2027") contained a hard-coded number, not recalculated from
    #    `synth`, and used a journalistic tone -- replaced by a descriptive
    #    title reflecting the actual content of the figure (inter-episode
    #    comparison at the ENSO peak month). The old subtitle attempted
    #    to force a line break using a long sequence of spaces before
    #    "TESR model..." -- ineffective because _draw_header passes the
    #    text through textwrap.fill(), which normalizes all whitespace
    #    (including "\n") before wrapping according to wrap_width.
    #    The "TESR model..." line is now passed through the extra_line
    #    parameter of _draw_header, specifically designed for this purpose
    #    (already used for the same need in
    #    plot_scenario_vs_contrefactuel_single) -- displayed in bold on
    #    its own line below the subtitle. --
    sous_titre = (
        "Anthropogenic trend, ENSO and residual seasonal contributions at the month "
        "of peak ENSO share, four El Niño episodes (July(n)-June(n+1))"
    )
    if ci_by_episode_peak is not None:
        sous_titre += " ; 90% CI by moving-block bootstrap (n=300)"
    if has_ext:
        sous_titre += (
            " ; 4th bar = verification-model gap at peak month "
            "(n.d. = not yet observed)"
        )
    if enso_minmax_xy:
        sous_titre += (
            " ; triangles = min/max ENSO contribution at peak month "
            "(multi-model ENSO spread + model bootstrap uncertainty)"
        )

    extra_line = "TESR model - 1850-1900 baseline - projection initialized 1 August 2026"
    header_in = _draw_header(
        fig, fig_h,
        "Peak-month decomposition of the global temperature anomaly across El Niño episodes",
        sous_titre,
        extra_line=extra_line
    )
    fig.subplots_adjust(top=1 - header_in / fig_h)
    fig.text(
        0.98, 0.01,
        "Data: ECMWF/Copernicus C3S - ERA5; Nino3.4 ClimateReanalyzer (1850-1900 baseline)",
        fontsize=7.5, color='#666666', ha='right'
    )
    plt.savefig(filename, dpi=300)
    plt.close()
    plt.rcParams['font.family'] = 'sans-serif'
    return synth, dates_found


# ----------------------------------------------------------------------
# 6bis. DETAILED MONTHLY TABLE BY EPISODE (hindcast 82/83, 97/98,
#       15/16 + forecast 26/27 -- same visual approach as the monthly
#       summary table already used for 2026-2027 in enso_gmst_model.py,
#       extended to all episodes and with the new
#       "ENSO-external variability" column)
# ----------------------------------------------------------------------
def plot_monthly_attribution_table(label, decomp, verif=None, ci_monthly=None, filename=None):
    d = decomp.copy()
    d['enso_pct'] = 100 * d['enso'] / d['total']
    d['trend_pct'] = 100 * d['trend'] / d['total']
    d['seasonal_pct'] = 100 * d['seasonal'] / d['total']

    # -- ENSO-neutral counterfactual = Trend + Seasonal (= the dashed
    #    grey curve in the scenario-vs-counterfactual graph). This is
    #    THE total used as the reference to isolate ENSO: ENSO = Model -
    #    Counterfactual = Total - Trend - Seasonal (exact by construction
    #    of the linear decomposition, verifiable row by row). Trend alone
    #    is nearly flat over an episode (12 months); the Counterfactual,
    #    which includes the residual seasonal cycle, oscillates above/below. --
    d['contrefactuel'] = d['trend'] + d['seasonal']

    if verif is not None:
        d['verif'] = verif.reindex(d.index)
        d['ext_var'] = compute_external_variability(d, d['verif'])
        d['ext_var_pct'] = 100 * d['ext_var'] / d['total']
    else:
        d['verif'] = np.nan
        d['ext_var'] = np.nan
        d['ext_var_pct'] = np.nan

    has_ci = ci_monthly is not None
    if has_ci:
        d = d.join(ci_monthly[
            ['enso_p5', 'enso_p95', 'enso_pct_p5', 'enso_pct_p95',
             'trend_p5', 'trend_p95', 'trend_pct_p5', 'trend_pct_p95',
             'seasonal_p5', 'seasonal_p95', 'seasonal_pct_p5', 'seasonal_pct_p95',
             'total_p5', 'total_p95',
             'contrefactuel_p5', 'contrefactuel_p95']
        ])

        # -- CI of external variability, DERIVED from the total's CI
        #    (verification is a fixed observation and is not bootstrapped):
        #    sign is reversed because ext_var = verification - total -> lower
        #    bound of the difference when the model is at its UPPER bound,
        #    and vice versa. --
        d['ext_var_p5'] = d['verif'] - d['total_p95']
        d['ext_var_p95'] = d['verif'] - d['total_p5']

        # Percentage of external variability: derived in the same way from
        # ext_var_p5/p95 (not directly from the total CI, because we divide
        # by 'total' -- the point estimate -- rather than by a bootstrapped
        # quantity here).
        d['ext_var_pct_p5'] = 100 * d['ext_var_p5'] / d['total']
        d['ext_var_pct_p95'] = 100 * d['ext_var_p95'] / d['total']

    if filename is None:
        safe = label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
        filename = f"gmst_monthly_attribution_table_{safe}.png"

    n_rows = len(d)

    # -- header_inches covers the suptitle + up to 3 subtitle lines
    #    (description + CI + °† note) -- fixed to the worst-case rather
    #    than calculated afterwards, because the number of lines is only
    #    known after the cell_text loop (any_pct_flagged). Prevents overlap
    #    with the table. --
    header_inches = 1.55

    # -- 2-line cells (°C on the first line, % on the second) for the 4
    #    contribution columns -- slightly taller row than before. --
    fig_height = max(3.6, 0.44 * n_rows + 0.5) + header_inches

    # -- 8 columns instead of 10 (°C and % merged into a single cell):
    #    the figure can therefore be slightly narrower while remaining
    #    comfortable and without compressing the text (9pt font). --
    fig_width = 22 if has_ci else 15
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    plt.rcParams['font.family'] = 'serif'

    col_labels = [
        'Month',
        'Model (°C)',
        'Trend (°C | %)',
        'Seasonal (°C | %)',
        'Counterfactual\n(trend+seasonal, °C)',
        'ENSO (°C | %)',
        'Verification (°C)',
        'External variability (°C | %)'
    ]

    PCT_ALERT_THRESHOLD = 100  # above this threshold, the percentage share
                                # becomes an artifact of opposite signs
                                # (see figure footnote) rather than a direct
                                # interpretation of the contribution

    any_pct_flagged = [False]  # mutable so it can be updated from _fmt_pair

    def _fmt_pair(val, pct, p5=None, p95=None, pct_p5=None, pct_p95=None):
        """2-line cell: °C [CI] on top, % [CI] below.
        CI omitted if p5/p95 are None (no-bootstrap mode).
        A "†" flags a percentage whose magnitude exceeds
        PCT_ALERT_THRESHOLD -- a possible artifact when another
        component (often Seasonal) has the opposite sign to the Total,
        rather than a calculation error (see figure footnote).
        """
        flag = "†" if abs(pct) > PCT_ALERT_THRESHOLD else ""
        if flag:
            any_pct_flagged[0] = True
        if p5 is None:
            return f"{val:+.2f} °C\n{pct:+.0f}%{flag}"
        return (
            f"{val:+.2f} °C [{p5:+.2f};{p95:+.2f}] °C\n"
            f"{pct:+.0f}%{flag} [{pct_p5:+.0f};{pct_p95:+.0f}]%"
        )

    cell_text = []
    for m, row in d.iterrows():
        verif_str = f"{row['verif']:+.2f}" if pd.notna(row['verif']) else "—"

        if has_ci:
            total_str = (
                f"{row['total']:+.2f} °C "
                f"[{row['total_p5']:+.2f};{row['total_p95']:+.2f}] °C"
            )
            trend_str = _fmt_pair(
                row['trend'], row['trend_pct'],
                row['trend_p5'], row['trend_p95'],
                row['trend_pct_p5'], row['trend_pct_p95']
            )
            seas_str = _fmt_pair(
                row['seasonal'], row['seasonal_pct'],
                row['seasonal_p5'], row['seasonal_p95'],
                row['seasonal_pct_p5'], row['seasonal_pct_p95']
            )
            cf_str = (
                f"{row['contrefactuel']:+.2f} °C "
                f"[{row['contrefactuel_p5']:+.2f};{row['contrefactuel_p95']:+.2f}] °C"
            )
            enso_str = _fmt_pair(
                row['enso'], row['enso_pct'],
                row['enso_p5'], row['enso_p95'],
                row['enso_pct_p5'], row['enso_pct_p95']
            )
            if pd.notna(row['ext_var']):
                ext_str = _fmt_pair(
                    row['ext_var'], row['ext_var_pct'],
                    row['ext_var_p5'], row['ext_var_p95'],
                    row['ext_var_pct_p5'], row['ext_var_pct_p95']
                )
            else:
                ext_str = "—"
        else:
            total_str = f"{row['total']:+.2f} °C"
            trend_str = _fmt_pair(row['trend'], row['trend_pct'])
            seas_str = _fmt_pair(row['seasonal'], row['seasonal_pct'])
            cf_str = f"{row['contrefactuel']:+.2f} °C"
            enso_str = _fmt_pair(row['enso'], row['enso_pct'])
            ext_str = (
                _fmt_pair(row['ext_var'], row['ext_var_pct'])
                if pd.notna(row['ext_var']) else "—"
            )

        cell_text.append([
            m.strftime('%b %Y'),
            total_str,
            trend_str,
            seas_str,
            cf_str,
            enso_str,
            verif_str,
            ext_str,
        ])

    table_frac_height = min(
        0.95,
        (0.44 * n_rows + 0.15) / (fig_height - header_inches - 0.02)
    )

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        bbox=[0.0, 1.0 - table_frac_height, 1.0, table_frac_height],
        cellLoc='center',
        colLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)  # 2-line cells (°C + %) -> taller rows

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#dddddd')
        if r == 0:
            cell.set_facecolor('#1a1a1a')
            cell.set_text_props(color='white', fontweight='bold')
        elif c in (3, 4):  # Seasonal + Counterfactual columns --
                            # grey tint (Counterfactual = Trend + Seasonal)
            cell.set_facecolor('#e8e8e8' if r % 2 == 0 else '#f4f4f4')
        elif c in (6, 7):  # Verification / External variability columns --
                            # distinct tint
            cell.set_facecolor(light_tint(RECORD, 0.06) if r % 2 == 0 else 'white')
        else:
            cell.set_facecolor(light_tint(FACTUAL, 0.08) if r % 2 == 0 else 'white')

    n_obs = int(d['verif'].notna().sum())

    # -- Positions anchored at FIXED INCHES from the top (using va='top'),
    #    rather than as a fraction of fig_height: avoids recalibration if
    #    header_inches or the number of subtitle lines changes in the future. --
    TITLE_TOP_IN = 0.32
    SUBTITLE_TOP_IN = 0.62

    fig.suptitle(
        f"Detailed monthly table - {label}",
        fontsize=14.5,
        fontweight='bold',
        x=0.02,
        ha='left',
        va='top',
        y=1 - TITLE_TOP_IN / fig_height
    )

    subtitle_lines = [
        "TESR model - exact linear decomposition; ENSO = Model - Counterfactual "
        "(= Trend + Seasonal); Seasonal isolated separately from the Counterfactual to assess "
        "its own share; verification = ERA5/C3S observation "
        f"({n_obs}/{n_rows} months observed, ref. 1850-1900)"
    ]

    if has_ci:
        subtitle_lines.append(
            "90% CI by moving-block bootstrap (n=300, Kunsch 1989) on all columns "
            "except Verification (fixed observation); CI of External variability derived "
            "by symmetry of the Model's (sign reversed)"
        )

    if any_pct_flagged[0]:
        subtitle_lines.append(
            "†: |%| > 100 -- Trend+Seasonal+ENSO=Total is exact in °C, but an individual % "
            "can exceed 100 (or be negative) when another component has a sign opposite "
            "to the Total (often the Seasonal term); trust the °C value in that case, "
            "not the %"
        )

    if "projection" in label.lower():
        subtitle_lines.append("Official scenario, initialized 1 August 2026")

    sous_titre = "\n".join(subtitle_lines)

    fig.text(
        0.02,
        1 - SUBTITLE_TOP_IN / fig_height,
        sous_titre,
        fontsize=9.3,
        style='italic',
        color='#444444',
        ha='left',
        va='top'
    )

    fig.text(
        0.98,
        0.01,
        "Data: ECMWF/Copernicus C3S - ERA5; Nino3.4 ClimateReanalyzer (ref. 1850-1900)",
        fontsize=7.5,
        color='#666666',
        ha='right'
    )

    top_frac = 1 - header_inches / fig_height
    ax.set_position([0.015, 0.02, 0.97, top_frac - 0.02])

    plt.savefig(filename, dpi=300)
    plt.close()
    plt.rcParams['font.family'] = 'sans-serif'
    return d, filename


# ----------------------------------------------------------------------
# 7. EXECUTION
# ----------------------------------------------------------------------
if __name__ == "__main__":
    pipe = build_pipeline()
    decomps, labels, lag = pipe['decomps'], pipe['labels'], pipe['lag']
    model, feature_cols, dataset = pipe['model'], pipe['feature_cols'], pipe['dataset']
    X_train, y_train, calendars = pipe['X_train'], pipe['y_train'], pipe['calendars']
    gmst_df = pipe['gmst_df']

    # -- Verification (ERA5/C3S observation) by episode -- NaN for months
    #    not yet observed (end of the 2026-2027 projection) --
    verifs = {l: get_verification(gmst_df, decomps[l]) for l in labels}
    for l in labels:
        n_obs = int(verifs[l].notna().sum())
        print(f"  Verification {l}: {n_obs}/{len(decomps[l])} observed months")

    N_BOOT = 300  # 300 replicates: good compromise between precision and computation time

    print("\n=== Moving-block bootstrap (90% CI) -- one run per episode ===")
    raw_by_episode = {}

    for l in labels:
        print(f"  Bootstrap {l} (n={N_BOOT})...")
        raw_by_episode[l] = bootstrap_episode_raw(
            model, feature_cols, X_train, y_train, dataset, lag,
            calendars[l], n_boot=N_BOOT, block_size=12, seed=42
        )

    # -- CI for the episode average (comparative °C and % bars) --
    ci_mean_by_episode = {
        l: ci_mean_over_episode(raw_by_episode[l])
        for l in labels
    }

    # -- Month-by-month CI (detailed table) --
    ci_monthly_by_episode = {
        l: ci_all_dates(raw_by_episode[l])
        for l in labels
    }

    # -- COMBINED ENSO min/max points (multi-model ENSO spread -- q0/q100
    #    bounds -- COMBINED with the model's 90% bootstrap CI at these bounds),
    #    averaged over the episode -- for ENSO bars in the comparative
    #    attribution diagrams (°C) and percentages (%) --
    enso_uncertainty = pipe.get('enso_uncertainty', {})

    enso_minmax = {
        l: (
            bands['q0']['ci_mean']['enso'][0],
            bands['q100']['ci_mean']['enso'][2]
        )
        for l, bands in enso_uncertainty.items()
    }

    enso_minmax_pct = {
        l: (
            bands['q0']['ci_mean']['enso_pct'][0],
            bands['q100']['ci_mean']['enso_pct'][2]
        )
        for l, bands in enso_uncertainty.items()
    }

    summary_moy = plot_bar_comparatif_moyenne(
        decomps,
        labels,
        ci_by_episode=ci_mean_by_episode,
        verifs=verifs,
        enso_minmax=enso_minmax
    )

    print("\n=== Episode-average summary (with 90% CI) ===")

    for l in labels:
        ci = ci_mean_by_episode[l]
        ext_str = (
            f" ; external variability = {summary_moy.loc[l,'ext_var']:+.3f}°C "
            f"({summary_moy.loc[l,'ext_var_pct']:+.1f}%, "
            f"n={int(summary_moy.loc[l,'n_obs'])} months)"
            if pd.notna(summary_moy.loc[l, 'ext_var'])
            else " ; external variability = n.d."
        )

        print(
            f"{l}: ENSO = {summary_moy.loc[l,'enso']:+.3f}°C "
            f"[{ci['enso'][0]:+.3f}, {ci['enso'][2]:+.3f}] "
            f"({summary_moy.loc[l,'enso_pct']:+.1f}% "
            f"[{ci['enso_pct'][0]:+.1f}%, {ci['enso_pct'][2]:+.1f}%])"
            + ext_str
        )

    summary_pct = plot_bar_pourcentage(
        decomps,
        labels,
        ci_by_episode=ci_mean_by_episode,
        verifs=verifs,
        enso_minmax_pct=enso_minmax_pct
    )

    print("\n=== Generating the 4 separate scenario/counterfactual plots (+ verification) ===")

    for l in labels:
        fn = plot_scenario_vs_contrefactuel_single(
            l,
            decomps[l],
            lag,
            verif=verifs[l],
            ci_monthly=ci_monthly_by_episode[l],
            envelope=enso_uncertainty.get(l)
        )
        print(f"  -> {fn}")

    print("\n=== Generating detailed monthly tables (hindcast + forecast, with 90% ENSO CI) ===")

    for l in labels:
        _, fn = plot_monthly_attribution_table(
            l,
            decomps[l],
            verif=verifs[l],
            ci_monthly=ci_monthly_by_episode[l]
        )
        print(f"  -> {fn}")

    print("\n=== Bars at ENSO peak (maximum ENSO share) + 90% CI + external variability ===")

    # First determine the peak month (depends on the central scenario)
    # before calculating the CI at that specific date.
    tmp_synth, peak_dates = plot_bar_paroxysme_enso(
        decomps,
        labels,
        verifs=verifs,
        enso_uncertainty=enso_uncertainty
    )  # First pass without CI to determine the dates

    ci_peak_by_episode = {
        l: ci_at_date(raw_by_episode[l], peak_dates[l])
        for l in labels
    }

    synth_paroxysme, _ = plot_bar_paroxysme_enso(
        decomps,
        labels,
        ci_by_episode_peak=ci_peak_by_episode,
        peak_dates=peak_dates,
        verifs=verifs,
        enso_uncertainty=enso_uncertainty
    )

    print(synth_paroxysme.round(3))
    synth_paroxysme.to_csv("enso_peak_synthesis.csv")

    print("\nOK - all plots generated (with 90% moving-block bootstrap CI)")

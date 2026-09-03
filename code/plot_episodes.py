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
            # -- le texte est ancré au-delà du point le plus extrême entre la
            #    barre d'erreur ET le marqueur min/max (s'il existe pour ce
            #    label), pour ne jamais chevaucher le triangle centré --
            top_extent = h + hi_err
            bottom_extent = h - lo_err
            if minmax_by_label is not None and labels[i] in minmax_by_label:
                mn, mx = minmax_by_label[labels[i]]
                top_extent = max(top_extent, mx)
                bottom_extent = min(bottom_extent, mn)
            va = 'bottom' if h >= 0 else 'top'
            y_anchor = (top_extent + 0.015) if h >= 0 else (bottom_extent - 0.015)
            txt = "n.d." if is_nan else f"{h:+.2f}°C\n({pct:+.0f}%)"
            ax.text(bar.get_x() + bar.get_width() / 2, y_anchor, txt, ha='center', va=va,
                    fontsize=8.3, fontweight='bold', color='#1a1a1a')

    label_bars(b1, 'trend_pct', 'trend', trend_minmax_xy)
    label_bars(b2, 'enso_pct', 'enso', enso_minmax_xy)
    label_bars(b3, 'seasonal_pct', 'seasonal', seasonal_minmax_xy)
    if b4 is not None:
        label_bars(b4, 'ext_var_pct', 'ext_var')

    # -- Marges dynamiques : tiennent compte des barres d'erreur, du texte à
    #    2 lignes ET des marqueurs min/max, pour ne JAMAIS déborder du cadre --
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
        ax.text(x[i], y_top * 0.985, f"Total : {summary['total'].iloc[i]:+.2f}°C",
                ha='center', va='top', fontsize=9.5, style='italic', color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.axhline(0, color='#333333', lw=0.8)
    ax.set_ylabel("Mean contribution to the GMSTA anomaly (C, ref. 1850-1900)")
    ax.set_ylim(y_bottom, y_top)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2 if has_ext else 3,
              fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    fig.subplots_adjust(bottom=0.24 if has_ext else 0.20)
    sous_titre = ("4 El Nino episodes (July(n) to June(n+1)), episode average - TESR model, "
                  "exact linear decomposition (ridge); ENSO = Total - Counterfactual (Trend+Seasonal)")
    if ci_by_episode is not None:
        sous_titre += " ; barres d'erreur = IC 90% (bootstrap par blocs mobiles, n=300)"
    if has_ext:
        sous_titre += " ; 4e barre = écart verification-modèle (n.d. = mois pas encore observés)"
    if enso_minmax_xy or trend_minmax_xy or seasonal_minmax_xy:
        sous_titre += (" ; triangles = min/max par composante (ENSO : dispersion multi-modèles + bootstrap "
                        "of the model; Trend/Seasonal: bootstrap P5/P95 bounds)")
    header_in = _draw_header(fig, fig_h, "Attribution comparative : part anthropique vs part naturelle (ENSO) vs saisonnier",
                              sous_titre)
    fig.subplots_adjust(top=1 - header_in / fig_h)
    fig.text(0.98, 0.01, "Data: ECMWF/Copernicus C3S - ERA5; Nino3.4 ClimateReanalyzer (ref. 1850-1900)",
              fontsize=7.5, color='#666666', ha='right')
    plt.savefig(filename, dpi=300)
    plt.close()
    plt.rcParams['font.family'] = 'sans-serif'
    return summary


# ----------------------------------------------------------------------
# 4. GRAPHIQUE BARRES POURCENTAGE (moyenne épisode, %) + IC 90%
# ----------------------------------------------------------------------
def plot_bar_pourcentage(decomps, labels, ci_by_episode=None, verifs=None, enso_minmax_pct=None,
                          filename="gmst_episodes_barres_pourcentage.png"):
    """
    enso_minmax_pct : dict optionnel {label: (min_combiné_%, max_combiné_%)} --
        même principe que enso_minmax dans plot_bar_comparatif_moyenne, mais
        en % du total (IC 90% combiné dispersion ENSO + bootstrap du modèle).
    """
    summary = _make_summary(decomps, labels, verifs=verifs)
    has_ext = verifs is not None and summary['ext_var'].notna().any()

    plt.rcParams['font.family'] = 'serif'
    fig_h = 8.0
    fig, ax = plt.subplots(figsize=(11.5, fig_h))

    x = np.arange(len(labels))
    width = 0.19 if has_ext else 0.26
    offsets = [-1.5, -0.5, 0.5, 1.5] if has_ext else [-1, 0, 1]

    b1 = ax.bar(x + offsets[0] * width, summary['trend_pct'], width, color=COLOR_TREND, label="Tendance (anthropique)")
    b2 = ax.bar(x + offsets[1] * width, summary['enso_pct'], width, color=COLOR_ENSO, label="ENSO (naturel)")
    b3 = ax.bar(x + offsets[2] * width, summary['seasonal_pct'], width, color=COLOR_SEAS, label="Residual seasonal")
    bars_comp = [(b1, 'trend_pct'), (b2, 'enso_pct'), (b3, 'seasonal_pct')]
    b4 = None
    if has_ext:
        ext_pct_plot = summary['ext_var_pct'].fillna(0.0)
        b4 = ax.bar(x + offsets[3] * width, ext_pct_plot, width, color=COLOR_EXT,
                    label="ENSO-external variability (verification minus model)")
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
            ax.errorbar(xs, ys, yerr=[errs_lo, errs_hi], fmt='none',
                        ecolor='#1a1a1a', elinewidth=1.3, capsize=4, capthick=1.3, zorder=5)

    # -- Points min/max, CENTRÉS sur chaque barre concernée (calculés avant
    #    les étiquettes pour que le texte puisse être repoussé au-delà du
    #    triangle le plus extrême) -- ENSO : incertitude combinée dispersion
    #    multi-modèles + bootstrap ; Tendance/Saisonnier : bornes P5/P95 du
    #    bootstrap, en % --
    # -- Triangles min/max restreints aux labels présents dans enso_minmax_pct
    #    (en pratique, uniquement "2026-2027 (projection)") -- cf. même
    #    remarque que dans plot_bar_comparatif_moyenne. --
    minmax_labels = set(enso_minmax_pct.keys()) if enso_minmax_pct else set()
    trend_minmax_pct = ({l: (ci_by_episode[l]['trend_pct'][0], ci_by_episode[l]['trend_pct'][2]) for l in labels if l in minmax_labels}
                         if ci_by_episode is not None else None)
    seasonal_minmax_pct = ({l: (ci_by_episode[l]['seasonal_pct'][0], ci_by_episode[l]['seasonal_pct'][2]) for l in labels if l in minmax_labels}
                            if ci_by_episode is not None else None)
    trend_minmax_xy = _add_minmax_markers(ax, b1, labels, trend_minmax_pct)
    enso_minmax_xy = _add_minmax_markers(
        ax, b2, labels, enso_minmax_pct,
        legend_labels=('Max (ENSO spread + model uncertainty)',
                        'Min (ENSO spread + model uncertainty)'))
    seasonal_minmax_xy = _add_minmax_markers(ax, b3, labels, seasonal_minmax_pct)
    minmax_by_comp = {'trend_pct': trend_minmax_xy, 'enso_pct': enso_minmax_xy, 'seasonal_pct': seasonal_minmax_xy}

    for bars, comp in bars_comp:
        for i, bar in enumerate(bars):
            h = bar.get_height()
            is_nan = comp == 'ext_var_pct' and pd.isna(summary['ext_var_pct'].iloc[i])
            lo_err = hi_err = 0.0
            if ci_by_episode is not None and comp in ci_by_episode.get(labels[i], {}):
                lo_err, hi_err = _err_from_ci(h, ci_by_episode[labels[i]][comp])
            top_extent = h + hi_err
            bottom_extent = h - lo_err
            mm = minmax_by_comp.get(comp)
            if mm is not None and labels[i] in mm:
                mn, mx = mm[labels[i]]
                top_extent = max(top_extent, mx)
                bottom_extent = min(bottom_extent, mn)
            va = 'bottom' if h >= 0 else 'top'
            y_anchor = (top_extent + 1.5) if h >= 0 else (bottom_extent - 1.5)
            txt = "n.d." if is_nan else f"{h:+.0f}%"
            ax.text(bar.get_x() + bar.get_width() / 2, y_anchor, txt,
                    ha='center', va=va, fontsize=9, fontweight='bold', color='#1a1a1a')

    if ci_by_episode is not None:
        top_c = [max(summary['trend_pct'].iloc[i] + _err_from_ci(summary['trend_pct'].iloc[i], ci_by_episode[l]['trend_pct'])[1], 100)
                 for i, l in enumerate(labels)]
        bot_c = [summary['seasonal_pct'].iloc[i] - _err_from_ci(summary['seasonal_pct'].iloc[i], ci_by_episode[l]['seasonal_pct'])[0]
                 for i, l in enumerate(labels)]
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
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2 if has_ext else 3,
              fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    fig.subplots_adjust(bottom=0.24 if has_ext else 0.20)
    sous_titre = ("4 El Nino episodes (July(n) to June(n+1)) - the 3 signed shares sum to 100% by construction "
                  "; ENSO = Total - Contrefactuel (Tendance+Saisonnier)")
    if ci_by_episode is not None:
        sous_titre += " ; IC 90% (bootstrap par blocs mobiles, n=300)"
    if has_ext:
        sous_titre += " ; 4e barre = écart verification-modèle en % du total (n.d. = mois pas encore observés)"
    if enso_minmax_xy or trend_minmax_xy or seasonal_minmax_xy:
        sous_titre += (" ; triangles = min/max par composante (ENSO : dispersion multi-modèles + bootstrap "
                        "of the model; Trend/Seasonal: bootstrap P5/P95 bounds)")
    header_in = _draw_header(fig, fig_h, "Percentage breakdown of the total anomaly: anthropogenic vs natural vs seasonal",
                              sous_titre)
    fig.subplots_adjust(top=1 - header_in / fig_h)
    fig.text(0.98, 0.01, "Data: ECMWF/Copernicus C3S - ERA5; Nino3.4 ClimateReanalyzer (ref. 1850-1900)",
              fontsize=7.5, color='#666666', ha='right')
    plt.savefig(filename, dpi=300)
    plt.close()
    plt.rcParams['font.family'] = 'sans-serif'
    return summary


# ----------------------------------------------------------------------
# 5. FONCTION DISTINCTE : scénario réel vs contrefactuel, 1 fichier / épisode
#    (inchangé, pas concerné par la demande d'IC sur les barres)
# ----------------------------------------------------------------------
def plot_scenario_vs_contrefactuel_single(label, decomp, lag, verif=None, ci_monthly=None,
                                            envelope=None, filename=None):
    """
    envelope : dict optionnel {'p25','p75','p05','p95','q0','q100'} -> {'ci_monthly':
        DataFrame, 'ci_mean': dict, 'decomp_central': DataFrame}, tel que produit
        par build_pipeline()['enso_uncertainty'][label].

        IMPORTANT -- deux sources d'incertitude bien DISTINCTES, affichées
        séparément (elles étaient empilées dans une même bande dans une version
        précédente, ce qui les confondait visuellement) :

          1. DISPERSION MULTI-MODÈLES ENSO (bandes bleues pleines + lignes
             pointillées) : la décomposition CENTRALE du modèle (sans
             bootstrap, 'decomp_central') appliquée à chaque scénario ENSO
             (q0/p05/p25/p75/p95/q100). C'est une quantité DÉTERMINISTE --
             "si tel scénario ENSO se réalise, voici ce que prédit le modèle
             central" -- PAS une incertitude statistique. Trois paliers
             emboîtés : Q0-Q100 (dispersion totale), Q5-Q95, Q25-Q75 (cœur de
             la distribution) ; les bornes Q0/Q100 sont en plus tracées en
             pointillés pour repérer l'enveloppe extrême exacte.
          2. INCERTITUDE PROPRE DU MODÈLE TESR : bootstrap par blocs mobiles
             (Künsch 1989), affichée à DEUX endroits complémentaires --
             (a) en barres d'erreur sur la courbe centrale rouge/grise
                 (scénario ENSO median) ;
             (b) en frange POINTILLÉE ("grid of points", hatch, pas
                 d'aplat) qui prolonge CHAQUE bord des 3 paliers ci-dessus
                 jusqu'à la queue P5/P95 du bootstrap propre à ce scénario --
                 sans elle, l'incertitude réelle à chaque quantile serait
                 sous-estimée (biais optimiste des probabilités affichées) ;
                 en aplat plein elle se remélangerait avec la dispersion ENSO
                 (défaut d'une version précédente). La frange est donc
                 visible et mesurable, mais visuellement secondaire par
                 rapport aux bandes pleines.

        N'a de sens que pour un scénario projeté (ENSO multi-modèles) ; les
        épisodes historiques (ENSO observé) n'en fournissent pas.
    """
    if filename is None:
        safe = label.replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"gmst_scenario_vs_contrefactuel_{safe}.png"

    contrefactuel = decomp['trend'] + decomp['seasonal']
    idx = decomp.index

    plt.rcParams['font.family'] = 'serif'
    fig_w, fig_h = 14.5, 8.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # -- 1) DISPERSION MULTI-MODÈLES ENSO SEULE (déterministe, decomp_central
    #    sans bootstrap) -- 3 paliers emboîtés du plus large (Q0-Q100) au
    #    plus resserré (Q25-Q75), bleu de plus en plus saturé vers le centre --
    env_bounds = None
    if envelope is not None:
        def _central(col):
            return envelope[col]['decomp_central']['total'].reindex(idx)
        def _tail(col, which):
            return envelope[col]['ci_monthly'][which].reindex(idx)

        # -- Couleurs : QUANTILE_BANDS (palette.py) -- 3 teintes bleues
        #    réellement distinctes en teinte+luminosité (pas un simple
        #    empilement d'alpha sur une même couleur, qui washait tout en un
        #    bleu-marine quasi uniforme -- cf. gotcha documentée dans
        #    palette.py/light_tint : un alpha Patch-level passé EN PLUS d'une
        #    couleur RGBA écrase l'alpha embarqué dans cette RGBA, donc les 3
        #    paliers rendaient tous à alpha~0.9 avec la teinte COMPLÈTE de
        #    COUNTERFACT -- d'où les "nuances de bleu foncé" peu lisibles). --
        tiers = [
            ('q0',  'q100', QUANTILE_BANDS[0], 1.0, 1, 'Multi-model ENSO spread (Q0-Q100)'),
            ('p05', 'p95',  QUANTILE_BANDS[1], 1.0, 2, 'Multi-model ENSO spread (Q5-Q95)'),
            ('p25', 'p75',  QUANTILE_BANDS[2], 1.0, 3, 'Multi-model ENSO spread (Q25-Q75)'),
        ]
        for lo_col, hi_col, color, alpha, z, lbl in tiers:
            ax.fill_between(idx, _central(lo_col), _central(hi_col), color=color,
                             alpha=alpha, lw=0, zorder=z, label=lbl)

        ax.plot(idx, _central('q0'), color=COUNTERFACT, lw=1.1, ls=':', zorder=4)
        ax.plot(idx, _central('q100'), color=COUNTERFACT, lw=1.1, ls=':', zorder=4,
                label='Lowest / highest projection (Q0/Q100)')

        # -- 1bis) INCERTITUDE PROPRE DU MODÈLE, PROPAGÉE À CHAQUE QUANTILE
        #    (pas seulement au scénario median) : l'omettre biaiserait les
        #    probabilités affichées (les bandes ENSO seules sous-estiment le
        #    flou réel à chaque borne) ; mais l'empiler en aplat plein comme
        #    dans une version précédente la refond avec la dispersion ENSO et
        #    rend le graphique illisible. Compromis : une FRANGE POINTILLÉE
        #    ("grid of points", hatch='....', pas de remplissage plein) qui
        #    prolonge chaque bord de palier jusqu'à la queue P5/P95 du
        #    bootstrap DE CE SCÉNARIO -- visible, mesurable au besoin, mais
        #    visuellement subordonnée aux bandes pleines (dispersion ENSO =
        #    lecture principale ; frange pointillée = supplément d'incertitude
        #    statistique, secondaire). Un seul palier -- Q25-Q75, le plus
        #    interne -- porte l'étiquette de légende (les 3 paliers partagent
        #    le même style, une légende par palier serait redondante). --
        fringe_kw = dict(facecolor='none', hatch='.', linewidth=0.0, alpha=0.38, zorder=3.6)
        _hatch_lw_saved = plt.rcParams['hatch.linewidth']
        plt.rcParams['hatch.linewidth'] = 0.5  # traits de hachure fins -> lecture "frange", pas "texture"
        outer_lo, outer_hi = _tail('q0', 'total_p5'), _tail('q100', 'total_p95')
        ax.fill_between(idx, outer_lo, _central('q0'), edgecolor=COUNTERFACT, **fringe_kw)
        ax.fill_between(idx, _central('q100'), outer_hi, edgecolor=COUNTERFACT, **fringe_kw)
        mid_lo, mid_hi = _tail('p05', 'total_p5'), _tail('p95', 'total_p95')
        ax.fill_between(idx, mid_lo, _central('p05'), edgecolor=COUNTERFACT, **fringe_kw)
        ax.fill_between(idx, _central('p95'), mid_hi, edgecolor=COUNTERFACT, **fringe_kw)
        in_lo, in_hi = _tail('p25', 'total_p5'), _tail('p75', 'total_p95')
        ax.fill_between(idx, in_lo, _central('p25'), edgecolor=COUNTERFACT_DARK, **fringe_kw)
        ax.fill_between(idx, _central('p75'), in_hi, edgecolor=COUNTERFACT_DARK, **fringe_kw,
                         label="Model's own uncertainty at each quantile\n(bootstrap, dotted band)")
        plt.rcParams['hatch.linewidth'] = _hatch_lw_saved

        env_bounds = (outer_lo, outer_hi)  # étendue verticale réelle = dispersion ENSO + incertitude modèle

    # -- 2) INCERTITUDE PROPRE DU MODÈLE (bootstrap par blocs mobiles), À
    #    SCÉNARIO ENSO MÉDIAN FIXÉ -- barres d'erreur SEULEMENT, jamais
    #    remélangée dans une bande de dispersion ENSO --
    has_ci = ci_monthly is not None
    if has_ci:
        ci_al = ci_monthly.reindex(idx)  # aligne au cas où l'index diffère
        lo_tot, hi_tot = _err_from_ci_series(decomp['total'], ci_al['total_p5'], ci_al['total_p95'])
        lo_cf, hi_cf = _err_from_ci_series(contrefactuel, ci_al['contrefactuel_p5'], ci_al['contrefactuel_p95'])

        ax.errorbar(idx, decomp['total'], yerr=[lo_tot, hi_tot],
                    color=COLOR_TREND, lw=2.2, marker='o', ms=5, capsize=3, elinewidth=1.1,
                    ecolor=COLOR_TREND, alpha=0.95, zorder=6, label='Model prediction (with El Nino)')
        ax.errorbar(idx, contrefactuel, yerr=[lo_cf, hi_cf],
                    color='#555555', lw=2.0, ls='--', marker='o', ms=5, capsize=3, elinewidth=1.1,
                    ecolor='#555555', alpha=0.85, zorder=6, label='Counterfactual (ENSO-neutral)')
    else:
        ax.plot(idx, decomp['total'], color=COLOR_TREND, lw=2.2, marker='o', ms=5,
                zorder=6, label='Central model prediction (with El Nino)')
        ax.plot(idx, contrefactuel, color='#555555', lw=2.0, ls='--', marker='o', ms=5,
                zorder=6, label='Counterfactual (ENSO-neutral)')

    # -- 3) CONTRIBUTION EL NIÑO (scénario median) -- zone HACHURÉE rouge,
    #    pour ne jamais se confondre visuellement avec les bandes bleues de
    #    dispersion ENSO ci-dessus (remplissage quasi transparent + hachures
    #    obliques rouges, comme les diagrammes d'attribution IPCC) --
    ax.fill_between(idx, contrefactuel, decomp['total'], facecolor=COLOR_ENSO, alpha=0.12,
                     zorder=4, lw=0)
    ax.fill_between(idx, contrefactuel, decomp['total'], facecolor='none', edgecolor=COLOR_ENSO,
                     hatch='///', linewidth=0.0, zorder=4.5, label='El Nino contribution (hatched area)')

    # -- Ligne de verification (observation ERA5/C3S réellement mesurée),
    #    en noir, uniquement pour les mois déjà observés (NaN sinon --
    #    matplotlib saute alors ces points, la ligne "stops" simplement) --
    ext_var_mean = None
    n_obs = 0
    verif_aligned = None
    if verif is not None:
        verif_aligned = verif.reindex(idx)
        n_obs = int(verif_aligned.notna().sum())
        if n_obs > 0:
            ax.plot(idx, verif_aligned, color=COLOR_VERIF, lw=2.0, ls='-', marker='s', ms=5,
                    zorder=7, label='Verification (observed ERA5/C3S)')
            ext_var = compute_external_variability(decomp, verif_aligned)
            ext_var_mean = ext_var.mean(skipna=True)

    # -- Seuils de référence (mêmes styles que model.py) : toujours tracés,
    #    visibles seulement si l'épisode s'en approche --
    threshold_styles = {1.5: (':', '#888888', '+1.5 \u00b0C threshold (Paris Agreement)'),
                         2.0: ('--', '#555555', '+2 \u00b0C threshold')}
    y_data_max = max(decomp['total'].max(), contrefactuel.max())
    shown_thresholds = []
    for th, (ls, col, lab) in threshold_styles.items():
        if th <= y_data_max * 1.25:
            ax.axhline(th, color=col, linestyle=ls, lw=1.1, label=lab)
            shown_thresholds.append(th)

    # -- Étendue verticale EXPLICITE : inclut la courbe, l'IC du modèle, les
    #    3 bandes d'enveloppe combinées et la verification -- avec marge
    #    généreuse, pour ne plus jamais rogner les extrêmes affichés --
    y_series = [decomp['total'], contrefactuel]
    if has_ci:
        y_series += [ci_al['total_p5'], ci_al['total_p95'], ci_al['contrefactuel_p5'], ci_al['contrefactuel_p95']]
    if env_bounds is not None:
        y_series += [env_bounds[0], env_bounds[1]]
    if n_obs > 0:
        y_series.append(verif_aligned)
    all_y = np.concatenate([s.to_numpy(dtype=float) for s in y_series])
    all_y = all_y[~np.isnan(all_y)]
    y_min, y_max = all_y.min(), all_y.max()
    if shown_thresholds:
        y_max = max(y_max, max(shown_thresholds))
    y_span = max(y_max - y_min, 0.2)
    ax.set_ylim(y_min - 0.15 * y_span, y_max + 0.15 * y_span)

    ax.set_ylabel("Global mean surface temperature anomaly (\u00b0C) [1850-1900 baseline]")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate(rotation=45)

    # -- En-tête (titre + sous-titre) ancré en POUCES depuis le haut, pas en
    #    fraction de figure -- indépendant du nombre de lignes du sous-titre
    #    (auto-retour à la ligne) et de la taille de figure, donc ne déborde
    #    jamais quel que soit le contenu --
    # -- Titre : ADAPTÉ À CHAQUE ÉPISODE (auparavant un unique texte en dur,
    #    "El Nino could push global temperature towards +2C in early 2027",
    #    réutilisé tel quel pour les 4 appels de la boucle -- donc affiché
    #    aussi sur 1982-1983/1997-1998/2015-2016, sans rapport avec leur
    #    contenu). Formulation neutre, factuelle, sans registre journalistique
    #    ("could push"), cohérente avec un usage en revue de recherche : nom du
    #    modèle, épisode, nature (rétrospective observée vs projection), et la
    #    métrique réellement montrée sur la figure (anomalie au pic). --
    is_projection = "projection" in label.lower()
    episode_clean = label.replace(" (projection)", "").strip()
    peak_val = float(decomp['total'].max())
    peak_date = decomp['total'].idxmax()
    if is_projection:
        title_txt = (f"TESR-modelled global temperature anomaly under the projected "
                     f"{episode_clean} El Nino ({peak_val:+.2f} \u00b0C peak, {peak_date:%b %Y})")
    else:
        title_txt = (f"TESR-modelled El Nino contribution to the global temperature "
                     f"anomaly, {episode_clean} ({peak_val:+.2f} \u00b0C peak, {peak_date:%b %Y})")

    subtitle_ci = "Projection initialized 1 August 2026" if (has_ci and is_projection) else ""
    subtitle_txt = (
    f"Modelled anomaly with El Nino vs ENSO-neutral counterfactual - "
    f"Departure from preindustrial (1850-1900) baseline, \u00b0C - "
    f"TESR model, lag={lag} months          "
    f"{subtitle_ci}"
    )
    subtitle_wrapped = textwrap.fill(subtitle_txt, width=150)
    n_sub_lines = subtitle_wrapped.count("\n") + 1

    title_top_in = 0.40
    subtitle_top_in = title_top_in + 0.32
    ext_top_in = subtitle_top_in + 0.20 * n_sub_lines + 0.08
    header_in = ext_top_in + (0.28 if n_obs > 0 else 0.08)

    fig.suptitle(title_txt,
                 fontsize=13.5, fontweight='bold', x=0.02, ha='left', va='top',
                 y=1 - title_top_in / fig_h)
    fig.text(0.02, 1 - subtitle_top_in / fig_h, subtitle_wrapped,
              fontsize=9.5, style='italic', color='#444444', ha='left', va='top')
    if n_obs > 1 and ext_var_mean is not None:
        fig.text(0.02, 1 - ext_top_in / fig_h,
                  f"Mean verification \u2212 model gap over the observed period: "
                  f"{ext_var_mean:+.2f} \u00b0C (n = {n_obs} months)",
                  fontsize=9, fontweight='bold', color='#1a1a1a', ha='left', va='top')

    # -- Légende SORTIE du cadre (à droite) -- figure élargie + marge droite
    #    dédiée + police réduite pour que les 3 nouvelles entrées d'enveloppe
    #    tiennent sans être coupées --
    fig.subplots_adjust(top=1 - header_in / fig_h, bottom=0.16, right=0.71)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8.3, framealpha=0.9)

    fig.text(0.98, 0.01, "Data: ECMWF/Copernicus C3S - ERA5; Nino3.4 ClimateReanalyzer (1850-1900 baseline)",
              fontsize=7.5, color='#666666', ha='right')
    plt.savefig(filename, dpi=300)
    plt.close()
    plt.rcParams['font.family'] = 'sans-serif'
    return filename


# ----------------------------------------------------------------------
# 6. FONCTION DISTINCTE : barres comparatives AU PAROXYSME ENSO + IC 90%
# ----------------------------------------------------------------------
def plot_bar_paroxysme_enso(decomps, labels, ci_by_episode_peak=None, peak_dates=None, verifs=None,
                             enso_uncertainty=None,
                             filename="gmst_episodes_barres_paroxysme_enso.png"):
    """
    enso_uncertainty : dict optionnel {label: {'q0': {'ci_monthly': DataFrame}, 'q100': {...}, ...}},
        tel que produit par build_pipeline(). Utilisé pour dériver le min/max
        combiné (dispersion ENSO + bootstrap du modèle) DE LA CONTRIBUTION ENSO
        AU MOIS DE PAROXYSME précis (pas une moyenne épisode) -- q0/ci_monthly
        pour la borne basse (P5), q100/ci_monthly pour la borne haute (P95).
        La MÊME logique (P5 sous q0 / P95 sous q100) est aussi appliquée à
        Tendance et Saisonnier : c'est une incertitude INHÉRENTE AU MODÈLE
        (le bootstrap ré-ajuste tendance+ENSO+saisonnier ensemble), pas
        propre à la composante ENSO seule. Par construction, les triangles
        min/max ne sont donc affichés QUE pour les labels présents dans
        enso_uncertainty (en pratique, uniquement "2026-2027 (projection)"
        -- les épisodes historiques ont un ENSO observé, sans scénario
        multi-modèles à propager).
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
                # -- Propagation de l'incertitude min/max ENSO sur les 2
                #    AUTRES composantes (Tendance, Saisonnier) : incertitude
                #    INHÉRENTE AU MODÈLE, pas une dispersion propre à ces
                #    composantes -- le bootstrap par blocs mobiles ré-ajuste
                #    le modèle ENTIER (tendance + ENSO + saisonnier ensemble)
                #    sur des résidus ré-échantillonnés, donc un scénario ENSO
                #    extrême (q0/q100) entraîne aussi une marge sur Tendance
                #    et Saisonnier -- ce n'est pas visible en se limitant à
                #    l'IC bootstrap du seul scénario central (ci_by_episode_peak).
                #    On prend donc, comme pour ENSO, la borne basse (P5) sous
                #    le scénario ENSO bas (q0) et la borne haute (P95) sous
                #    le scénario ENSO haut (q100). --
                trend_minmax_peak[l] = (cm_q0.loc[idx_paroxysme, 'trend_p5'],
                                         cm_q100.loc[idx_paroxysme, 'trend_p95'])
                seasonal_minmax_peak[l] = (cm_q0.loc[idx_paroxysme, 'seasonal_p5'],
                                            cm_q100.loc[idx_paroxysme, 'seasonal_p95'])
        # -- variabilité externe AU mois de paroxysme (pas la moyenne
        #    épisode) : écart verification - modèle ce mois-là, NaN si le
        #    mois n'est pas encore observé (ex. fin de projection 26/27) --
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

    b1 = ax.bar(x + offsets[0] * width, synth['trend'], width, color=COLOR_TREND, label="Trend (anthropogenic)")
    b2 = ax.bar(x + offsets[1] * width, synth['enso'], width, color=COLOR_ENSO, label="ENSO (natural)")
    b3 = ax.bar(x + offsets[2] * width, synth['seasonal'], width, color=COLOR_SEAS, label="Residual seasonal")
    b4 = None
    if has_ext:
        ext_plot = synth['ext_var'].fillna(0.0)
        b4 = ax.bar(x + offsets[3] * width, ext_plot, width, color=COLOR_EXT,
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

    # -- Points min/max, CENTRÉS sur chaque barre, calculés AVANT les
    #    étiquettes (pour que le texte soit repoussé au-delà du triangle
    #    le plus extrême) -- ENSO : incertitude combinée au mois de
    #    paroxysme précis ; Tendance/Saisonnier : bornes P5/P95 du
    #    bootstrap à ce même mois --
    # -- trend_minmax_peak / seasonal_minmax_peak sont désormais construits
    #    PLUS HAUT (dans la boucle rows), à partir des bandes q0/q100 de
    #    enso_uncertainty -- donc automatiquement restreints aux labels
    #    pour lesquels cette dispersion ENSO existe (2026-2027 (projection)
    #    en pratique), et propageant l'incertitude min/max ENSO sur ces 2
    #    composantes plutôt que l'IC bootstrap du seul scénario central. --
    trend_minmax_xy = _add_minmax_markers(ax, b1, labels, trend_minmax_peak)
    enso_minmax_xy = _add_minmax_markers(
        ax, b2, labels, enso_minmax_peak,
        legend_labels=('Max (ENSO spread + model uncertainty)',
                        'Min (ENSO spread + model uncertainty)'))
    seasonal_minmax_xy = _add_minmax_markers(ax, b3, labels, seasonal_minmax_peak)
    minmax_by_comp = {'trend': trend_minmax_xy, 'enso': enso_minmax_xy, 'seasonal': seasonal_minmax_xy}

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
        top_candidates = [synth['trend'].iloc[i] + _err_from_ci(synth['trend'].iloc[i], ci_by_episode_peak[l]['trend'])[1]
                           for i, l in enumerate(labels)]
        bottom_candidates = [synth['seasonal'].iloc[i] - _err_from_ci(synth['seasonal'].iloc[i], ci_by_episode_peak[l]['seasonal'])[0]
                              for i, l in enumerate(labels)]
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
                f"Total: {synth['total'].iloc[i]:+.2f} \u00b0C\n({date_str})",
                ha='center', va='top', fontsize=9, style='italic', color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.axhline(0, color='#333333', lw=0.8)
    ax.set_ylabel("Contribution to the GMSTA anomaly (\u00b0C, ref. 1850-1900)")
    ax.set_ylim(y_bottom, y_top)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2 if has_ext else 3,
              fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    fig.subplots_adjust(bottom=0.24 if has_ext else 0.20)
    # -- Titre/sous-titre : REFORMULÉS pour un usage preprint --
    #    l'ancien titre ("El Nino could add +0.36C to global warming in
    #    2027") était un chiffre en dur, non recalculé depuis `synth`, au
    #    registre journalistique -- remplacé par un intitulé descriptif du
    #    contenu réel de la figure (comparaison inter-épisodes au mois de
    #    pic ENSO). L'ancien sous-titre tentait de forcer un retour à la
    #    ligne avec une longue suite d'espaces avant "TESR model..." --
    #    inefficace, car _draw_header passe le texte dans textwrap.fill(),
    #    qui normalise tous les espaces (y compris "\n") avant de
    #    ré-empaqueter selon wrap_width : un saut de ligne "manuel" dans la
    #    chaîne n'a donc aucun effet. La ligne "TESR model..." est
    #    maintenant portée par le paramètre extra_line de _draw_header,
    #    prévu pour ça (déjà utilisé pour ce même besoin dans
    #    plot_scenario_vs_contrefactuel_single) -- rendue en gras sur sa
    #    propre ligne, sous le sous-titre. --
    sous_titre = ("Anthropogenic trend, ENSO and residual seasonal contributions at the month of peak "
                  "ENSO share, four El Nino episodes (July(n)-June(n+1))")
    if ci_by_episode_peak is not None:
        sous_titre += " ; 90% CI by moving-block bootstrap (n=300)"
    if has_ext:
        sous_titre += " ; 4th bar = verification-model gap at peak month (n.d. = not yet observed)"
    if enso_minmax_xy:
        sous_titre += (" ; triangles = min/max ENSO contribution at peak month (multi-model ENSO "
                        "spread + model bootstrap uncertainty)")
    extra_line = "TESR model - 1850-1900 baseline - projection initialized 1 August 2026"
    header_in = _draw_header(fig, fig_h,
                              "Peak-month decomposition of the global temperature anomaly across El Nino episodes",
                              sous_titre, extra_line=extra_line)
    fig.subplots_adjust(top=1 - header_in / fig_h)
    fig.text(0.98, 0.01, "Data: ECMWF/Copernicus C3S - ERA5; Nino3.4 ClimateReanalyzer (1850-1900 baseline)",
              fontsize=7.5, color='#666666', ha='right')
    plt.savefig(filename, dpi=300)
    plt.close()
    plt.rcParams['font.family'] = 'sans-serif'
    return synth, dates_found


# ----------------------------------------------------------------------
# 6bis. TABLEAU MENSUEL DÉTAILLÉ PAR ÉPISODE (hindcast 82/83, 97/98,
#       15/16 + prévision 26/27 -- même esprit visuel que le tableau de
#       synthèse mensuelle déjà utilisé pour 2026-2027 dans
#       enso_gmst_model.py, étendu à tous les épisodes et à la nouvelle
#       colonne "ENSO-external variability")
# ----------------------------------------------------------------------
def plot_monthly_attribution_table(label, decomp, verif=None, ci_monthly=None, filename=None):
    d = decomp.copy()
    d['enso_pct'] = 100 * d['enso'] / d['total']
    d['trend_pct'] = 100 * d['trend'] / d['total']
    d['seasonal_pct'] = 100 * d['seasonal'] / d['total']
    # -- Contrefactuel ENSO neutre = Tendance + Saisonnier (= la courbe
    #    grise en tirets du graphique scénario-vs-contrefactuel). C'est
    #    CE total-là, et non la Tendance seule, qui sert de référence pour
    #    isoler ENSO : ENSO = Modèle - Contrefactuel = Total - Tendance -
    #    Saisonnier (exact par construction de la décomposition linéaire,
    #    vérifiable colonne par colonne). La Tendance seule est quasi
    #    plate sur un épisode (12 mois) ; c'est le Contrefactuel, avec le
    #    cycle saisonnier résiduel dedans, qui oscille au-dessus/dessous.
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
        d = d.join(ci_monthly[['enso_p5', 'enso_p95', 'enso_pct_p5', 'enso_pct_p95',
                                'trend_p5', 'trend_p95', 'trend_pct_p5', 'trend_pct_p95',
                                'seasonal_p5', 'seasonal_p95', 'seasonal_pct_p5', 'seasonal_pct_p95',
                                'total_p5', 'total_p95',
                                'contrefactuel_p5', 'contrefactuel_p95']])
        # -- IC de la variabilité externe, DÉRIVÉ de celui du total (verif est
        #    une observation fixe, pas une quantité bootstrappée) : signe inversé
        #    car ext_var = verif - total -> borne basse de l'écart quand le
        #    modèle est à sa borne HAUTE, et inversement. --
        d['ext_var_p5'] = d['verif'] - d['total_p95']
        d['ext_var_p95'] = d['verif'] - d['total_p5']
        # % de variabilité externe : dérivé de même à partir de ext_var_p5/p95
        # (pas de la CI du total directement, car on divise par 'total' -- point
        # estimate -- pas par une quantité elle-même bootstrappée ici)
        d['ext_var_pct_p5'] = 100 * d['ext_var_p5'] / d['total']
        d['ext_var_pct_p95'] = 100 * d['ext_var_p95'] / d['total']

    if filename is None:
        safe = label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
        filename = f"gmst_table_episode_{safe}.png"

    n_rows = len(d)
    # -- header_inches couvre suptitle + jusqu'à 3 lignes de sous-titre
    #    (description + IC + note °†) -- fixé au pire cas plutôt que calculé
    #    après coup, car le nombre de lignes n'est connu qu'après la boucle
    #    cell_text (any_pct_flagged). Évite tout chevauchement avec le tableau. --
    header_inches = 1.55
    # -- Cellules à 2 lignes (°C sur la 1ère, % sur la 2e) pour les 4
    #    colonnes de contribution -- ligne un peu plus haute qu'avant. --
    fig_height = max(3.6, 0.44 * n_rows + 0.5) + header_inches
    # -- 8 colonnes au lieu de 10 (°C et % fusionnés par cellule) : la
    #    figure peut donc être un peu moins large qu'avant tout en
    #    restant confortable, sans compresser le texte (police 9pt). --
    fig_width = 22 if has_ci else 15
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    plt.rcParams['font.family'] = 'serif'

    col_labels = ['Month', 'Model (\u00b0C)', 'Trend (\u00b0C | %)', 'Seasonal (\u00b0C | %)',
                  'Counterfactual\n(trend+seasonal, \u00b0C)', 'ENSO (\u00b0C | %)',
                  'Verification (\u00b0C)', 'External variability (\u00b0C | %)']

    PCT_ALERT_THRESHOLD = 100  # au-delà, la part en % devient un artefact de
                                # signes opposés (cf. note de bas de figure) plutôt
                                # qu'une lecture directe de la contribution
    any_pct_flagged = [False]  # mutable pour être mis à jour depuis _fmt_pair

    def _fmt_pair(val, pct, p5=None, p95=None, pct_p5=None, pct_p95=None):
        """Cellule 2 lignes : °C [IC] en haut, % [IC] en dessous. IC omis
        si p5/p95 valent None (mode sans bootstrap). Un "†" signale un % dont
        la magnitude dépasse PCT_ALERT_THRESHOLD -- artefact possible quand
        une autre composante (souvent le Saisonnier) est de signe opposé au
        Total, et non une erreur de calcul (cf. note de bas de figure)."""
        flag = "†" if abs(pct) > PCT_ALERT_THRESHOLD else ""
        if flag:
            any_pct_flagged[0] = True
        if p5 is None:
            return f"{val:+.2f} °C\n{pct:+.0f}%{flag}"
        return (f"{val:+.2f} °C [{p5:+.2f};{p95:+.2f}] °C\n"
                f"{pct:+.0f}%{flag} [{pct_p5:+.0f};{pct_p95:+.0f}]%")

    cell_text = []
    for m, row in d.iterrows():
        verif_str = f"{row['verif']:+.2f}" if pd.notna(row['verif']) else "—"
        if has_ci:
            total_str = f"{row['total']:+.2f} °C [{row['total_p5']:+.2f};{row['total_p95']:+.2f}] °C"
            trend_str = _fmt_pair(row['trend'], row['trend_pct'], row['trend_p5'], row['trend_p95'],
                                   row['trend_pct_p5'], row['trend_pct_p95'])
            seas_str = _fmt_pair(row['seasonal'], row['seasonal_pct'], row['seasonal_p5'], row['seasonal_p95'],
                                  row['seasonal_pct_p5'], row['seasonal_pct_p95'])
            cf_str = f"{row['contrefactuel']:+.2f} °C [{row['contrefactuel_p5']:+.2f};{row['contrefactuel_p95']:+.2f}] °C"
            enso_str = _fmt_pair(row['enso'], row['enso_pct'], row['enso_p5'], row['enso_p95'],
                                  row['enso_pct_p5'], row['enso_pct_p95'])
            if pd.notna(row['ext_var']):
                ext_str = _fmt_pair(row['ext_var'], row['ext_var_pct'], row['ext_var_p5'], row['ext_var_p95'],
                                     row['ext_var_pct_p5'], row['ext_var_pct_p95'])
            else:
                ext_str = "—"
        else:
            total_str = f"{row['total']:+.2f} °C"
            trend_str = _fmt_pair(row['trend'], row['trend_pct'])
            seas_str = _fmt_pair(row['seasonal'], row['seasonal_pct'])
            cf_str = f"{row['contrefactuel']:+.2f} °C"
            enso_str = _fmt_pair(row['enso'], row['enso_pct'])
            ext_str = _fmt_pair(row['ext_var'], row['ext_var_pct']) if pd.notna(row['ext_var']) else "—"
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

    table_frac_height = min(0.95, (0.44 * n_rows + 0.15) / (fig_height - header_inches - 0.02))
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                      bbox=[0.0, 1.0 - table_frac_height, 1.0, table_frac_height],
                      cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)  # cellules à 2 lignes (°C + %) -> lignes plus hautes qu'avant
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#dddddd')
        if r == 0:
            cell.set_facecolor('#1a1a1a')
            cell.set_text_props(color='white', fontweight='bold')
        elif c in (3, 4):  # colonnes Saisonnier + Contrefactuel -- teinte grise (Contrefactuel = Tendance+Saisonnier)
            cell.set_facecolor('#e8e8e8' if r % 2 == 0 else '#f4f4f4')
        elif c in (6, 7):  # colonnes verification / variabilité externe -- teinte distincte
            cell.set_facecolor(light_tint(RECORD, 0.06) if r % 2 == 0 else 'white')
        else:
            cell.set_facecolor(light_tint(FACTUAL, 0.08) if r % 2 == 0 else 'white')

    n_obs = int(d['verif'].notna().sum())
    # -- Positions ancrées en POUCES fixes depuis le haut (via va='top'), pas
    #    en fraction de fig_height : évite tout recalibrage manuel si
    #    header_inches ou le nombre de lignes du sous-titre change à l'avenir. --
    TITLE_TOP_IN = 0.32
    SUBTITLE_TOP_IN = 0.62
    fig.suptitle(f"Detailed monthly table - {label}",
                 fontsize=14.5, fontweight='bold', x=0.02, ha='left', va='top',
                 y=1 - TITLE_TOP_IN / fig_height)
    subtitle_lines = [
        "TESR model - exact linear decomposition; ENSO = Model - Counterfactual "
        "(= Trend + Seasonal); Seasonal isolated separately from the Counterfactual to assess "
        "its own share; verification = ERA5/C3S observation "
        f"({n_obs}/{n_rows} mois observés, réf. 1850-1900)"
    ]
    if has_ci:
        subtitle_lines.append(
            "90% CI by moving-block bootstrap (n=300, Kunsch 1989) on all columns "
            "except Verification (fixed observation); CI of the External variability derived by symmetry "
            "of the Model's (sign reversed)"
        )
    if any_pct_flagged[0]:
        subtitle_lines.append(
            "†: |%| > 100 -- Trend+Seasonal+ENSO=Total is exact in °C, but an individual % can "
            "exceed 100 (or be negative) when another component has a sign opposite to the Total "
            "(often the Seasonal term); trust the °C value in that case, not the %"
        )
    if "projection" in label.lower():
        subtitle_lines.append("Official scenario, initialized 1 August 2026")
    sous_titre = "\n".join(subtitle_lines)
    fig.text(0.02, 1 - SUBTITLE_TOP_IN / fig_height, sous_titre, fontsize=9.3, style='italic',
             color='#444444', ha='left', va='top')
    fig.text(0.98, 0.01, "Data: ECMWF/Copernicus C3S - ERA5; Nino3.4 ClimateReanalyzer (ref. 1850-1900)",
              fontsize=7.5, color='#666666', ha='right')

    top_frac = 1 - header_inches / fig_height
    ax.set_position([0.015, 0.02, 0.97, top_frac - 0.02])
    plt.savefig(filename, dpi=300)
    plt.close()
    plt.rcParams['font.family'] = 'sans-serif'
    return d, filename


# ----------------------------------------------------------------------
# 7. EXÉCUTION
# ----------------------------------------------------------------------
if __name__ == "__main__":
    pipe = build_pipeline()
    decomps, labels, lag = pipe['decomps'], pipe['labels'], pipe['lag']
    model, feature_cols, dataset = pipe['model'], pipe['feature_cols'], pipe['dataset']
    X_train, y_train, calendars = pipe['X_train'], pipe['y_train'], pipe['calendars']
    gmst_df = pipe['gmst_df']

    # -- Vérification (observation ERA5/C3S) par épisode -- NaN pour les
    #    mois pas encore observés (fin de la projection 2026-2027) --
    verifs = {l: get_verification(gmst_df, decomps[l]) for l in labels}
    for l in labels:
        n_obs = int(verifs[l].notna().sum())
        print(f"  Vérification {l} : {n_obs}/{len(decomps[l])} mois observés")

    N_BOOT = 300  # 300 répliques : bon compromis précision/temps de calcul

    print("\n=== Bootstrap par blocs mobiles (IC 90%) -- un run par épisode ===")
    raw_by_episode = {}
    for l in labels:
        print(f"  Bootstrap {l} (n={N_BOOT})...")
        raw_by_episode[l] = bootstrap_episode_raw(
            model, feature_cols, X_train, y_train, dataset, lag,
            calendars[l], n_boot=N_BOOT, block_size=12, seed=42
        )

    # -- IC pour la moyenne sur l'épisode (barres comparatif °C et %) --
    ci_mean_by_episode = {l: ci_mean_over_episode(raw_by_episode[l]) for l in labels}
    # -- IC mois par mois (tableau détaillé) --
    ci_monthly_by_episode = {l: ci_all_dates(raw_by_episode[l]) for l in labels}

    # -- Points min/max ENSO COMBINÉS (dispersion multi-modèles ENSO -- bornes
    #    q0/q100 -- EMPILÉE avec l'IC 90% bootstrap du modèle à ces bornes),
    #    moyennés sur l'épisode -- pour les barres ENSO des diagrammes
    #    d'attribution comparatif (°C) et pourcentage (%) --
    enso_uncertainty = pipe.get('enso_uncertainty', {})
    enso_minmax = {
        l: (bands['q0']['ci_mean']['enso'][0], bands['q100']['ci_mean']['enso'][2])
        for l, bands in enso_uncertainty.items()
    }
    enso_minmax_pct = {
        l: (bands['q0']['ci_mean']['enso_pct'][0], bands['q100']['ci_mean']['enso_pct'][2])
        for l, bands in enso_uncertainty.items()
    }

    summary_moy = plot_bar_comparatif_moyenne(decomps, labels, ci_by_episode=ci_mean_by_episode,
                                               verifs=verifs, enso_minmax=enso_minmax)
    print("\n=== Synthèse moyenne épisode (avec IC 90%) ===")
    for l in labels:
        ci = ci_mean_by_episode[l]
        ext_str = (f" ; variabilité externe = {summary_moy.loc[l,'ext_var']:+.3f}°C "
                    f"({summary_moy.loc[l,'ext_var_pct']:+.1f}%, n={int(summary_moy.loc[l,'n_obs'])} mois)"
                   if pd.notna(summary_moy.loc[l, 'ext_var']) else " ; variabilité externe = n.d.")
        print(f"{l} : ENSO = {summary_moy.loc[l,'enso']:+.3f}°C "
              f"[{ci['enso'][0]:+.3f}, {ci['enso'][2]:+.3f}] "
              f"({summary_moy.loc[l,'enso_pct']:+.1f}% [{ci['enso_pct'][0]:+.1f}%, {ci['enso_pct'][2]:+.1f}%])" + ext_str)

    summary_pct = plot_bar_pourcentage(decomps, labels, ci_by_episode=ci_mean_by_episode, verifs=verifs,
                                        enso_minmax_pct=enso_minmax_pct)

    print("\n=== Génération des 4 graphiques scénario/contrefactuel séparés (+ verification) ===")
    for l in labels:
        fn = plot_scenario_vs_contrefactuel_single(l, decomps[l], lag, verif=verifs[l],
                                                     ci_monthly=ci_monthly_by_episode[l],
                                                     envelope=enso_uncertainty.get(l))
        print(f"  -> {fn}")

    print("\n=== Génération des tableaux mensuels détaillés (hindcast + prévision, avec IC 90% ENSO) ===")
    for l in labels:
        _, fn = plot_monthly_attribution_table(l, decomps[l], verif=verifs[l], ci_monthly=ci_monthly_by_episode[l])
        print(f"  -> {fn}")

    print("\n=== Barres au paroxysme ENSO (part ENSO maximale) + IC 90% + variabilité externe ===")
    # Il faut d'abord connaître la date de paroxysme (dépend du scénario central)
    # avant de calculer l'IC à cette date précise
    tmp_synth, peak_dates = plot_bar_paroxysme_enso(decomps, labels, verifs=verifs,
                                                     enso_uncertainty=enso_uncertainty)  # 1er passage sans IC pour trouver les dates
    ci_peak_by_episode = {l: ci_at_date(raw_by_episode[l], peak_dates[l]) for l in labels}
    synth_paroxysme, _ = plot_bar_paroxysme_enso(decomps, labels, ci_by_episode_peak=ci_peak_by_episode,
                                                  peak_dates=peak_dates, verifs=verifs,
                                                  enso_uncertainty=enso_uncertainty)
    print(synth_paroxysme.round(3))
    synth_paroxysme.to_csv("synthese_paroxysme_enso.csv")

    print("\nOK - tous les graphiques générés (avec IC 90% par bootstrap par blocs mobiles)")

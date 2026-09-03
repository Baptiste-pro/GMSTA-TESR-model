"""
Author: Baptiste Boussemart

plot_extension_dec2027.py
Two functions to integrate into your pipeline (enso_gmst_model.py / plot_episodes.py):

  1. plot_temp_extended_dec2027()  -- TEMP figure (style of Fig. 12/13 in the manuscript),
     extended through December 2027, full range = union of the 90% CIs of the 3 H2 2027 hypotheses.

  2. plot_oni_extended_dec2027()   -- ONI figure (style of Fig. 9), extended through December 2027,
     with quantile bands retained over the entire official period (August 2026-April 2027)
     AND over the extension (May-December 2027), using a DISTINCT colour palette to
     clearly signal the non-official part (author hypotheses).

Dependencies: pandas, numpy, matplotlib. This file is now SELF-CONTAINED:

  - `model_fit`: built by `build_model_fit(model, X, dataset, enso_df_raw,
    gmst_df, lag)` (section 0 bis below) from the REAL objects produced
    by the main training block of enso_gmst_model.py (`model, X, y, ... =
    fit_model(dataset, ...)`) -- do NOT build this dictionary manually with
    the structure {model, mu, sd, feature_cols, df, nino, gmst, lag}: the column
    names ('t_index'/'t_index2'/'enso_x_t'/'m_2'..'m_12' on the pipeline side vs
    't'/'t2'/'inter'/'m2'..'m12' expected here) and the model alpha attribute
    ('.alpha_', not '.alpha') do not correspond term by term, hence the cascade
    of NameError/AttributeError/KeyError observed if this step is skipped ;
  - `h2_scenarios`: dict {'neutre':..., 'central':..., 'la_nina_forte':...} =
    output of build_h2_2027_scenarios() (section 0 below, embedded CSV) ;
  - `official_scenario_aug26_avr27`, `official_gmsta_table`: dicts from the
    official scenario (same objects as in the previous extension -- still
    supplied by the caller, not reconstructed here).

WARNING: reconstruction of the official quantile bands
(August 2026-April 2027) on the ONI figure is an APPROXIMATION (linear growth
of the width with forecast lead time, visually calibrated against your
original Fig. 9) -- to be replaced by the actual Q0/Q5/Q25/Q75/Q95/Q100 bounds
from the Climate Dashboard (member file) as soon as it is available in the
execution environment.

NOTE (figure titles/text): all text displayed in the figures
(titles, subtitles, legends, annotations) is in English, using neutral
descriptive labels rather than the "media" style of the initial version --
see plot_episodes_v7.py / extensions.py for the same convention.
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import Ridge
from palette import FACTUAL, light_tint


# ====================================================================
# 0. H2 2027 ONI HYPOTHESES -- EMBEDDED CSV
# ====================================================================
#
# Replaces the missing dependency on build_h2_2027_scenarios() / the
# enso_gmst_model_extension_2027.py file (never provided), so that this
# script is SELF-CONTAINED. Three monthly ONI trajectories, May-December
# 2027, constructed from the author's narrative description:
#
#   - neutre         : the decline slows down and stabilizes near
#                       0°C from September 2027 onward (the smoothest
#                       transition observed, 2016/2017, which produced
#                       only a weak La Niña) ;
#   - central         : steady decline towards a weak to moderate La Niña
#                       by the end of the year (-0.6 to -0.7°C
#                       in December 2027) ;
#   - la_nina_forte   : rapid decline crossing the conventional
#                       strong La Niña threshold (ONI <= -1.5°C)
#                       as early as October-November 2027, without exceeding
#                       the most extreme values recorded over
#                       1950-2026 (La Niña 1973/74, ONI ~ -2.1°C).
#
# WARNING: ILLUSTRATIVE values constructed to respect this qualitative
# trajectory (shape + described transition points), NOT official outputs
# from an ENSO model -- to be adjusted/recalibrated once the true
# April 2027 ONI value (the last month of the official scenario) is known.
# `build_h2_2027_scenarios(anchor_offset=...)` allows the 3 curves to be
# shifted together to connect to this point if necessary.
_H2_2027_ONI_CSV = """date,neutre,central,la_nina_forte
2027-05-01,0.85,0.75,0.65
2027-06-01,0.55,0.40,0.20
2027-07-01,0.25,0.05,-0.30
2027-08-01,0.05,-0.25,-0.75
2027-09-01,0.00,-0.40,-1.15
2027-10-01,0.00,-0.50,-1.55
2027-11-01,0.00,-0.60,-1.80
2027-12-01,0.00,-0.65,-1.95
"""


def build_h2_2027_scenarios(anchor_offset=0.0):
    """Builds `h2_scenarios` (dict {'neutre':..., 'central':...,
    'la_nina_forte':...}, each a dict {date_str: ONI value}) from
    the embedded `_H2_2027_ONI_CSV` above -- see the warning
    at the beginning of the section regarding the illustrative nature
    of these values.

    `anchor_offset`: additive offset (°C) applied to all 3 curves
    together, to recalibrate them against the actual official ONI value
    for April 2027 once known (e.g. anchor_offset = actual_Apr27_value - 1.0,
    if 1.0 is the value implicitly assumed for May 2027 by the
    shape above). Leave at 0.0 until this connection point has been
    verified.
    """
    df = pd.read_csv(io.StringIO(_H2_2027_ONI_CSV), parse_dates=['date']).set_index('date')
    df = df + anchor_offset
    return {col: df[col].to_dict() for col in df.columns}


# ------------------------------------------------------------------
# Decomposition utilities (identical to those already used for
# historical episodes -- see extensions_visualisation.py)
# ------------------------------------------------------------------

def _fit_ridge_raw_scale(X, y, alpha):
    """Fits a standardized Ridge model and then converts the coefficients
    back to the raw scale (beta, beta0), so that trend/ENSO/seasonal
    components can be decomposed term by term without reapplying
    standardization at each prediction."""
    m = Ridge(alpha=alpha)
    mu_r, sd_r = X.mean(0), X.std(0)
    m.fit((X - mu_r) / sd_r, y)
    beta = m.coef_ / sd_r
    beta0 = m.intercept_ - np.sum(m.coef_ * mu_r / sd_r)
    return beta, beta0


def _decompose(dates, enso_series_smoothed, beta, beta0, feature_cols, df, lag, t_last, last_date):
    idxmap = {c: i for i, c in enumerate(feature_cols)}
    rows = []
    for d in dates:
        t = t_last + (d.year - last_date.year) * 12 + (d.month - last_date.month)
        enso_lag = enso_series_smoothed.get(d - pd.DateOffset(months=lag), np.nan)
        trend = beta0 + beta[idxmap['t']] * t + beta[idxmap['t2']] * t * t
        enso = beta[idxmap['enso_lag']] * enso_lag + beta[idxmap['inter']] * enso_lag * t
        seasonal = 0.0 if d.month == 1 else beta[idxmap[f'm{d.month}']]
        rows.append(dict(date=d, trend=trend, enso=enso, seasonal=seasonal,
                          total=trend + enso + seasonal, contrefactuel=trend + seasonal))
    return pd.DataFrame(rows).set_index('date')


def _build_enso_series(nino_obs, official_scenario, extra_hypothesis):
    s = nino_obs.copy()
    for k, v in official_scenario.items():
        s.loc[pd.Timestamp(k)] = v
    for k, v in extra_hypothesis.items():
        s.loc[pd.Timestamp(k)] = v
    return s.sort_index()


def build_model_fit(model, X, dataset, enso_df_raw, gmst_df, lag):
    """Builds the `model_fit` dictionary expected by this file (_decompose,
    block_bootstrap_ci, plot_temp_extended_dec2027, plot_oni_extended_dec2027)
    from the REAL objects of the main enso_gmst_model.py pipeline
    (`model, X, y, ... = fit_model(dataset, ...)`, `dataset = build_dataset(...)`).

    Call once, immediately after fitting the model:

        model_fit = build_model_fit(model, X, dataset, enso_df_raw, gmst_df, lag)

    Two convention mismatches are corrected here (the source of the
    cascading NameError/AttributeError if `model_fit` is built manually
    with the structure described at the top of the file):

    1. Column renaming -- enso_gmst_model.py names its variables
       't_index'/'t_index2'/'enso_x_t' (see build_dataset), whereas
       _decompose()/_fit_ridge_raw_scale() in THIS file expect
       't'/'t2'/'inter'. `df` is built here from `X` (which already contains
       all columns used during fitting, including the seasonal dummy
       variables m2..m12), renamed, with the target added under the name
       'gmst' (dataset has 'gmst_anom_preind').

    2. mu=0 / sd=1 (no standardization) -- `model` (from
       fit_ridge_standardized) ALREADY has its coefficients converted to
       the raw scale (see its docstring in enso_gmst_model.py):
       `model.predict()` therefore expects RAW features, not
       standardized ones. Actual mu/sd values would make `(X-mu)/sd`
       non-neutral and would distort block_bootstrap_ci.
    """
    colmap = {'t_index': 't', 't_index2': 't2', 'enso_x_t': 'inter'}
    df = X.rename(columns=colmap).astype(float).copy()
    # pd.get_dummies(..., prefix='m') names its columns 'm_2'..'m_12'
    # (underscore, default separator), whereas _decompose() looks for
    # 'm2'..'m12' (idxmap[f'm{d.month}']) -- renamed here to make the two
    # conventions match.
    df = df.rename(columns={c: c.replace('m_', 'm') for c in df.columns if c.startswith('m_')})
    df['gmst'] = dataset.loc[df.index, 'gmst_anom_preind'].astype(float)
    feature_cols = [c for c in df.columns if c != 'gmst']
    return {
        'model': model,
        'mu': np.zeros(len(feature_cols)),
        'sd': np.ones(len(feature_cols)),
        'feature_cols': feature_cols,
        'df': df,
        'nino': {'ssta': enso_df_raw['enso_ssta']},
        'gmst': {'gmst': gmst_df['gmst_anom_preind']},
        'lag': lag,
    }


def block_bootstrap_ci(model_fit, dates, enso_series, n_boot=250, block=12, seed=42):
    """90% CI (P5/P95) using moving-block bootstrap (Künsch 1989), for
    `total` and `contrefactuel`, over the requested dates and a given
    ENSO calendar. Reproduces the methodology described in §2.2 of the manuscript."""
    model, mu, sd = model_fit['model'], model_fit['mu'], model_fit['sd']
    feature_cols, df, lag = model_fit['feature_cols'], model_fit['df'], model_fit['lag']
    # .alpha_ (underscore) = sklearn's "fitted" attribute -- see fit_model()
    # in enso_gmst_model.py; .alpha (without underscore) is accepted as
    # a fallback if another model object is passed here at some point.
    alpha = getattr(model, 'alpha_', getattr(model, 'alpha', None))
    X_all, y_all = df[feature_cols].values, df['gmst'].values
    pred_all = model.predict((X_all - mu) / sd)
    resid_all = y_all - pred_all
    t_last, last_date = df['t'].iloc[-1], df.index[-1]

    s_smooth = enso_series.rolling(3, min_periods=1).mean()
    n = len(y_all)
    possible_starts = n - block + 1
    n_blocks_needed = int(np.ceil(n / block))

    rng = np.random.RandomState(seed)
    tot_samples, cf_samples = [], []
    for _ in range(n_boot):
        starts = rng.randint(0, possible_starts, size=n_blocks_needed)
        resid_boot = np.concatenate([resid_all[s:s + block] for s in starts])[:n]
        beta_b, beta0_b = _fit_ridge_raw_scale(X_all, pred_all + resid_boot, alpha)
        dec_b = _decompose(dates, s_smooth, beta_b, beta0_b, feature_cols, df, lag, t_last, last_date)
        tot_samples.append(dec_b['total'].values)
        cf_samples.append(dec_b['contrefactuel'].values)

    tot_samples, cf_samples = np.array(tot_samples), np.array(cf_samples)
    return pd.DataFrame({
        'total_p5': np.percentile(tot_samples, 5, axis=0), 'total_p95': np.percentile(tot_samples, 95, axis=0),
        'cf_p5': np.percentile(cf_samples, 5, axis=0), 'cf_p95': np.percentile(cf_samples, 95, axis=0),
    }, index=dates)


# ====================================================================
# 1. EXTENDED TEMPERATURE FIGURE (July 2026 - December 2027)
# ====================================================================

# ====================================================================
# 0 ter. MONTHLY TABLE -- 3 H2 2027 HYPOTHESES (GMSTA)
# ====================================================================
#
# Same counterfactual decomposition as the official 2026-2027 table
# (plot_monthly_attribution_table, plot_episodes_v7.py): Model / Trend
# / Seasonal / Counterfactual (= Trend+Seasonal) / ENSO (= Model -
# Counterfactual) -- but repeated for each of the 3 H2 2027 hypotheses,
# stacked into 3 blocks a) neutral, b) central, c) strong La Niña in ONE
# single figure (instead of the 3 hypotheses in side-by-side columns).
# 90% CI available for Model and Counterfactual (moving-block bootstrap,
# see block_bootstrap_ci); Trend/Seasonal/ENSO shown as point estimates only
# (these 3 components are not individually bootstrapped by block_bootstrap_ci,
# unlike the detailed monthly CIs used for historical episodes).
_H2_PCT_ALERT_THRESHOLD = 100  # beyond this, the percentage share becomes an artifact of
                                # opposite signs rather than a direct interpretation


def _h2_fmt_pair(val, pct, flagged):
    flag = "\u2020" if abs(pct) > _H2_PCT_ALERT_THRESHOLD else ""
    if flag:
        flagged[0] = True
    return f"{val:+.2f} \u00b0C\n{pct:+.0f}%{flag}"


def plot_h2_2027_gmst_table(decomps, cis, ext_start, n_boot=250,
                             filename="gmst_h2_2027_hypotheses_table.png",
                             csv_filename="gmst_h2_2027_hypotheses_table.csv",
                             data_sources="Data: ECMWF/Copernicus C3S — ERA5; "
                                          "Nino3.4 ClimateReanalyzer (1850-1900 climatology)"):
    """
    `decomps`, `cis`: dicts {'neutre':..., 'central':..., 'la_nina_forte':...}
    already calculated by plot_temp_extended_dec2027() -- passed as is, not
    recalculated here. `decomps[label]` has the columns 'trend'/'seasonal'/
    'enso'/'total'/'contrefactuel' (same structure as the historical
    decompositions); `cis[label]` has 'total_p5'/'total_p95'/'cf_p5'/'cf_p95'.

    Produces ONE PNG figure with 3 stacked sub-tables (a/b/c, one per
    hypothesis), and exports the same values (all hypotheses combined,
    'hypothesis' column) in a single long-format CSV.
    """
    order = ['neutre', 'central', 'la_nina_forte']
    block_titles = {
        'neutre': 'a) Neutral ENSO hypothesis',
        'central': 'b) Central hypothesis (moderate La Niña)',
        'la_nina_forte': 'c) Strong La Niña hypothesis',
    }
    dates = [d for d in decomps['central'].index if d >= ext_start]
    n_rows = len(dates)
    col_labels = ['Month', 'Model (\u00b0C)', 'Trend (\u00b0C | %)', 'Seasonal (\u00b0C | %)',
                  'Counterfactual\n(trend+seasonal, \u00b0C)', 'ENSO (\u00b0C | %)']

    flagged = [False]
    export_records = []
    block_cell_text = {}
    for lbl in order:
        d = decomps[lbl].loc[dates].copy()
        d['trend_pct'] = 100 * d['trend'] / d['total']
        d['seasonal_pct'] = 100 * d['seasonal'] / d['total']
        d['enso_pct'] = 100 * d['enso'] / d['total']
        ci = cis[lbl].loc[dates]
        rows = []
        for m in dates:
            r, c = d.loc[m], ci.loc[m]
            model_str = f"{r['total']:+.2f} \u00b0C [{c['total_p5']:+.2f};{c['total_p95']:+.2f}] \u00b0C"
            trend_str = _h2_fmt_pair(r['trend'], r['trend_pct'], flagged)
            seas_str = _h2_fmt_pair(r['seasonal'], r['seasonal_pct'], flagged)
            cf_str = f"{r['contrefactuel']:+.2f} \u00b0C [{c['cf_p5']:+.2f};{c['cf_p95']:+.2f}] \u00b0C"
            enso_str = _h2_fmt_pair(r['enso'], r['enso_pct'], flagged)
            rows.append([m.strftime('%b %Y'), model_str, trend_str, seas_str, cf_str, enso_str])
            export_records.append(dict(
                date=m, hypothesis=lbl, model_C=r['total'], model_p5_C=c['total_p5'], model_p95_C=c['total_p95'],
                trend_C=r['trend'], trend_pct=r['trend_pct'], seasonal_C=r['seasonal'], seasonal_pct=r['seasonal_pct'],
                contrefactuel_C=r['contrefactuel'], contrefactuel_p5_C=c['cf_p5'], contrefactuel_p95_C=c['cf_p95'],
                enso_C=r['enso'], enso_pct=r['enso_pct']))
        block_cell_text[lbl] = rows

    # -- Layout: shared suptitle+subtitle at the top, followed by 3
    #    stacked blocks (a/b/c + table), with some spacing between each --
    header_inches = 1.05
    block_label_inches = 0.28
    row_inches = 0.40
    col_header_inches = 0.40
    gap_inches = 0.22
    block_inches = block_label_inches + col_header_inches + row_inches * n_rows
    fig_height = header_inches + 3 * block_inches + 2 * gap_inches + 0.35
    fig_width = 13

    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.rcParams['font.family'] = 'serif'

    y_cursor_in = header_inches  # distance from the TOP of the figure
    for lbl in order:
        ax = fig.add_axes([0.02, 1 - (y_cursor_in + block_label_inches + col_header_inches
                                       + row_inches * n_rows) / fig_height,
                            0.96, (col_header_inches + row_inches * n_rows) / fig_height])
        ax.axis('off')
        ax.text(0.0, 1.0, block_titles[lbl], transform=ax.transAxes, fontsize=11.5, fontweight='bold',
                ha='left', va='top', color='#1a1a1a')
        table_frac = (row_inches * n_rows) / (col_header_inches + row_inches * n_rows)
        table = ax.table(cellText=block_cell_text[lbl], colLabels=col_labels,
                          bbox=[0.0, 0.0, 1.0, table_frac], cellLoc='center', colLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.0)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor('#dddddd')
            if r == 0:
                cell.set_facecolor('#1a1a1a')
                cell.set_text_props(color='white', fontweight='bold')
            elif c in (2, 3, 4):  # Trend / Seasonal / Counterfactual -- grey tint
                cell.set_facecolor('#e8e8e8' if r % 2 == 0 else '#f4f4f4')
            else:
                cell.set_facecolor(light_tint(FACTUAL, 0.08) if r % 2 == 0 else 'white')
        y_cursor_in += block_inches + gap_inches

    TITLE_TOP_IN, SUBTITLE_TOP_IN = 0.32, 0.65
    fig.suptitle("Monthly GMSTA decomposition under the three H2 2027 ENSO hypotheses",
                 fontsize=14, fontweight='bold', x=0.02, ha='left', va='top',
                 y=1 - TITLE_TOP_IN / fig_height)
    subtitle_lines = [
    f"{dates[0]:%B}-December 2027 extension (author hypotheses), spliced onto the official scenario at "
    f"{ext_start - pd.DateOffset(months=1):%B %Y} - TESR model, exact linear decomposition: "
    "ENSO = Model - Counterfactual (= Trend + Seasonal) - departure from preindustrial baseline (1850-1900), °C",

    f"90% CI by moving-block bootstrap (n={n_boot}) on Model and Counterfactual only; "
    "Trend/Seasonal/ENSO shown as point estimates",
    ]
    
    if flagged[0]:
        subtitle_lines.append(
            "\u2020: |%| > 100 -- Trend+Seasonal+ENSO=Model is exact in \u00b0C, but an individual % can exceed "
            "100 (or be negative) when another component has a sign opposite to the Model; trust the \u00b0C value")
    
    subtitle = "\n".join(subtitle_lines)
    
    fig.text(0.02, 1 - SUBTITLE_TOP_IN / fig_height, subtitle,
              fontsize=9, style='italic', color='#444444', ha='left', va='top')
    fig.text(0.98, 0.01, data_sources, fontsize=7.5, color='#666666', ha='right')

    plt.savefig(filename, dpi=300)
    plt.close()
    plt.rcParams['font.family'] = 'sans-serif'

    export = pd.DataFrame.from_records(export_records).set_index(['hypothesis', 'date'])
    export.to_csv(csv_filename, float_format='%.3f')
    return filename, csv_filename



def plot_temp_extended_dec2027(model_fit, official_scenario_aug26_avr27, h2_scenarios,
                                official_gmsta_table, n_boot=250, seed=42,
                                filename="gmst_extended_synthesis_dec2027.png",
                                csv_filename="gmst_extended_synthesis_dec2027.csv",
                                table_filename="gmst_h2_2027_hypotheses_table.png",
                                table_csv_filename="gmst_h2_2027_hypotheses_table.csv",
                                data_sources="Data: ECMWF/Copernicus C3S — ERA5; "
                                             "Nino3.4 ClimateReanalyzer (1850-1900 climatology)"):
    """
    90% CI bounds are calculated SEPARATELY for each of the 3 H2 2027
    hypotheses (moving-block bootstrap, n_boot replicates each), then combined
    into a "full range" = [minimum P5, maximum P95] at each month -- this is the
    direct response to the request "add 90% confidence bounds for the 2027
    projections of each scenario and then obtain a full range".

    Layout follows the manuscript standard (Fig. 12/13): nested bands,
    El Niño contribution shown with hatching, error bars on the central
    scenario and counterfactual, +1.5°C/+2°C thresholds, legend centered
    vertically on the right, official/extension separator labelled in the
    middle of the graph.

    In addition to the PNG figure, exports the complete GMSTA trajectory
    (monthly, July 2026-December 2027) to CSV (`csv_filename`): central scenario
    (total + counterfactual + 90% CI), the 2 other ONI hypotheses (neutral /
    strong La Niña) and the combined range -- the data underlying the figure.
    """
    nino_obs, gmst = model_fit['nino'], model_fit['gmst']
    lag = model_fit['lag']

    full_dates = pd.date_range('2026-07-01', '2027-12-01', freq='MS')
    official_s = pd.Series(official_gmsta_table)
    official_s.index = pd.to_datetime(official_s.index)
    # -- Official -> H2 2027 hypothesis switch: derived from the LAST month
    #    actually present in `official_gmsta_table` (not fixed to June 30
    #    -- your current pipeline ends in April 2027, see enso envelope) --
    splice_month = official_s.index.max()
    ext_start = splice_month + pd.DateOffset(months=1)

    decomps, cis = {}, {}
    for label, hyp in h2_scenarios.items():
        enso_series = _build_enso_series(nino_obs['ssta'], official_scenario_aug26_avr27, hyp)
        beta_c, beta0_c = _fit_ridge_raw_scale(
            model_fit['df'][model_fit['feature_cols']].values, model_fit['df']['gmst'].values,
            getattr(model_fit['model'], 'alpha_', getattr(model_fit['model'], 'alpha', None)))
        dec = _decompose(full_dates, enso_series.rolling(3, min_periods=1).mean(), beta_c, beta0_c,
                          model_fit['feature_cols'], model_fit['df'], lag,
                          model_fit['df']['t'].iloc[-1], model_fit['df'].index[-1])
        # continuity: recalibrate to the last official month shared by all 3 hypotheses
        b = dec.loc[splice_month, 'total'] - official_gmsta_table[splice_month.strftime('%Y-%m-%d')]
        dec.loc[dec.index >= ext_start, ['trend', 'total', 'contrefactuel']] -= b
        dec.loc[official_s.index, 'total'] = official_s.values  # exact display from Table 4
        decomps[label] = dec

        ci = block_bootstrap_ci(model_fit, full_dates, enso_series, n_boot=n_boot, seed=seed)
        ci.loc[ci.index >= ext_start] -= b
        cis[label] = ci

    central = decomps['central']
    ci_c = cis['central']
    plage_p5 = pd.concat([cis[l]['total_p5'] for l in h2_scenarios], axis=1).min(axis=1)
    plage_p95 = pd.concat([cis[l]['total_p95'] for l in h2_scenarios], axis=1).max(axis=1)
    mask_off = full_dates < ext_start
    plage_p5[mask_off] = ci_c['total_p5'][mask_off]
    plage_p95[mask_off] = ci_c['total_p95'][mask_off]

    q5_95_low = plage_p5 + 0.35 * (central['total'] - plage_p5)
    q5_95_high = plage_p95 - 0.35 * (plage_p95 - central['total'])
    q25_75_low = plage_p5 + 0.65 * (central['total'] - plage_p5)
    q25_75_high = plage_p95 - 0.65 * (plage_p95 - central['total'])

    splice = splice_month + pd.Timedelta(days=15)
    fig, ax = plt.subplots(figsize=(19, 9.5))

    ax.fill_between(full_dates, plage_p5, plage_p95, color='#bcd7ee', alpha=0.65, lw=0, zorder=1,
                     label='Combined range (90% CI of the 3 hypotheses, pooled)')
    ax.fill_between(full_dates, q5_95_low, q5_95_high, color='#7fb2da', alpha=0.75, lw=0, zorder=2,
                     label='Narrower band (close to the central scenario 90% CI)')
    ax.fill_between(full_dates, q25_75_low, q25_75_high, color='#3f7cac', alpha=0.85, lw=0, zorder=3,
                     label='Distribution core (illustrative)')
    ax.plot(full_dates, plage_p5, color='#3f7cac', lw=1.0, ls=':', zorder=4)
    ax.plot(full_dates, plage_p95, color='#3f7cac', lw=1.0, ls=':', zorder=4,
            label='Combined lower/upper bound (90% CI, 3 hypotheses)')

    contrefactuel = central['contrefactuel']
    ax.fill_between(full_dates, contrefactuel, central['total'], where=(central['total'] >= contrefactuel),
                     facecolor='none', edgecolor='#e07b39', hatch='///', linewidth=0.0, zorder=5,
                     label='El Niño contribution (central scenario)')

    ax.errorbar(full_dates, central['total'],
                yerr=[(central['total'] - ci_c['total_p5']).clip(lower=0),
                      (ci_c['total_p95'] - central['total']).clip(lower=0)],
                color='#c0392b', lw=2.4, marker='o', ms=4.5, capsize=3, elinewidth=1.0, ecolor='#c0392b',
                zorder=7, label='Model prediction — central scenario (with El Niño)')
    ax.errorbar(full_dates, contrefactuel,
                yerr=[(contrefactuel - ci_c['cf_p5']).clip(lower=0), (ci_c['cf_p95'] - contrefactuel).clip(lower=0)],
                color='#555555', lw=1.8, ls='--', marker='o', ms=4, capsize=3, elinewidth=0.9, ecolor='#555555',
                zorder=6, label='Counterfactual (ENSO-neutral)')

    ax.plot(full_dates, decomps['neutre']['total'], color='#2980b9', lw=1.6, ls=':', zorder=6,
            label='Neutral ENSO hypothesis (H2 2027)')
    ax.plot(full_dates, decomps['la_nina_forte']['total'], color='#1a1aa6', lw=1.6, ls='-.', zorder=6,
            label='Strong La Niña hypothesis (H2 2027)')

    obs_pt = gmst['gmst'].loc['2026-07-01']
    ax.plot([pd.Timestamp('2026-07-01')], [obs_pt], marker='s', color='black', ms=8, zorder=8,
            label='Verification (observed ERA5/C3S)')

    ax.axhline(1.5, color='gray', ls=':', lw=1)
    ax.axhline(2.0, color='gray', ls='--', lw=1)
    ax.text(full_dates[0], 1.53, '+1.5 \u00b0C threshold (Paris Agreement)', fontsize=9, color='gray')
    ax.text(full_dates[0], 2.03, '+2 \u00b0C threshold', fontsize=9, color='gray')

    ylo, yhi = 1.2, 2.25
    ax.set_ylim(ylo, yhi)
    ax.axvline(splice, color='#555555', lw=1.1, ls=':', zorder=9)
    y_mid = (ylo + yhi) / 2
    ax.annotate('official\n(Table 4)', xy=(splice, y_mid), xytext=(-8, 0), textcoords='offset points',
                fontsize=9, color='#333333', ha='right', va='center', rotation=90,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#999999', alpha=0.9, linewidth=0.6),
                zorder=10)
    ax.annotate('H2 2027\nhypotheses', xy=(splice, y_mid), xytext=(8, 0), textcoords='offset points',
                fontsize=9, color='#333333', ha='left', va='center', rotation=90,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#999999', alpha=0.9, linewidth=0.6),
                zorder=10)

    ax.set_ylabel("Global mean surface temperature anomaly (\u00b0C) [1850-1900 baseline]", fontsize=11)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate(rotation=45)
    ax.tick_params(labelsize=10)

    fig.suptitle("TESR-modelled global temperature anomaly under three ENSO hypotheses: "
                 "extended projection through December 2027",
                 x=0.02, y=0.975, ha='left', fontsize=19, fontweight='bold')
    fig.text(0.02, 0.925,
              "Projection July 2026-December 2027 - with El Niño vs ENSO-neutral - Departure from preindustrial "
              "baseline (1850-1900), °C - TESR model, lag = 3 months\n"
              f"July 2026-{splice_month:%B %Y}: official scenario (Table 4, initialized 1 August 2026) - "
              f"{ext_start:%B}-December 2027: combined range across the "
              f"90% CI of the 3 ENSO hypotheses for H2 2027 (moving-block bootstrap, n={n_boot})",
              fontsize=11, style='italic', color='#555555', ha='left', va='top')

    ax.legend(loc='center left', fontsize=9.8, framealpha=0.95, ncol=1, bbox_to_anchor=(1.01, 0.5))
    fig.text(0.99, 0.008, data_sources, fontsize=8.5, color='#666', ha='right')

    plt.tight_layout(rect=[0, 0.02, 0.83, 0.90])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    # -- Export CSV of the GMSTA trajectory underlying the figure --
    export = pd.DataFrame({
        'central_total_C': central['total'],
        'central_contrefactuel_C': contrefactuel,
        'central_total_p5_C': ci_c['total_p5'],
        'central_total_p95_C': ci_c['total_p95'],
        'neutre_total_C': decomps['neutre']['total'],
        'la_nina_forte_total_C': decomps['la_nina_forte']['total'],
        'plage_combinee_p5_C': plage_p5,
        'plage_combinee_p95_C': plage_p95,
    })
    export.index.name = 'date'
    export.to_csv(csv_filename, float_format='%.3f')

    table_png, table_csv = plot_h2_2027_gmst_table(
        decomps, cis, ext_start, n_boot=n_boot,
        filename=table_filename, csv_filename=table_csv_filename,
        data_sources=data_sources)

    return filename, csv_filename, table_png, table_csv


# ====================================================================
# 2. EXTENDED ONI FIGURE (January 2026 - December 2027)
# ====================================================================

# ====================================================================
# 2 bis. ONI TABLES -- 3 H2 2027 HYPOTHESES (May-December) and
#        OFFICIAL SCENARIO (August 2026-April 2027), WITH CSV EXPORT
# ====================================================================

def _render_oni_table(col_labels, cell_text, title, subtitle, filename, csv_export=None,
                       csv_filename=None, fig_width=11, highlight_col=None,
                       data_sources="Data: The Climate Brink @ ENSO Dashboard"):
    n_rows = len(cell_text)
    header_inches = 1.15
    fig_height = max(3.0, 0.44 * n_rows + 0.5) + header_inches
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    plt.rcParams['font.family'] = 'serif'

    table_frac_height = min(0.95, (0.44 * n_rows + 0.15) / (fig_height - header_inches - 0.02))
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                      bbox=[0.0, 1.0 - table_frac_height, 1.0, table_frac_height],
                      cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 2.0)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#dddddd')
        if r == 0:
            cell.set_facecolor('#1a1a1a')
            cell.set_text_props(color='white', fontweight='bold')
        elif highlight_col is not None and c == highlight_col:
            cell.set_facecolor(light_tint(FACTUAL, 0.10) if r % 2 == 0 else light_tint(FACTUAL, 0.04))
        else:
            cell.set_facecolor('#f4f4f4' if r % 2 == 0 else 'white')

    TITLE_TOP_IN, SUBTITLE_TOP_IN = 0.32, 0.65
    fig.suptitle(title, fontsize=14, fontweight='bold', x=0.02, ha='left', va='top',
                 y=1 - TITLE_TOP_IN / fig_height)
    fig.text(0.02, 1 - SUBTITLE_TOP_IN / fig_height, subtitle,
              fontsize=9.3, style='italic', color='#444444', ha='left', va='top')
    fig.text(0.98, 0.01, data_sources, fontsize=7.5, color='#666666', ha='right')

    top_frac = 1 - header_inches / fig_height
    ax.set_position([0.02, 0.02, 0.96, top_frac - 0.02])
    plt.savefig(filename, dpi=300)
    plt.close()
    plt.rcParams['font.family'] = 'sans-serif'

    if csv_export is not None and csv_filename is not None:
        csv_export.to_csv(csv_filename, float_format='%.3f')
    return filename, csv_filename


def plot_oni_h2_2027_table(h2_scenarios,
                            filename="oni_h2_2027_hypotheses_table.png",
                            csv_filename="oni_h2_2027_hypotheses_table.csv"):
    """Monthly ONI table (May-December 2027) comparing the 3 H2 2027
    hypotheses, directly from `h2_scenarios` -- no recalculation."""
    order = ['neutre', 'central', 'la_nina_forte']
    disp_names = {'neutre': 'Neutral ENSO', 'central': 'Central (moderate La Niña)',
                  'la_nina_forte': 'Strong La Niña'}
    dates = sorted({pd.Timestamp(k) for k in h2_scenarios['central']})

    export_cols = {lbl: [] for lbl in order}
    cell_text = []
    for d in dates:
        row = [d.strftime('%b %Y')]
        for lbl in order:
            v = h2_scenarios[lbl][pd.Timestamp(d)]
            row.append(f"{v:+.2f} \u00b0C")
            export_cols[lbl].append(v)
        cell_text.append(row)

    col_labels = ['Month'] + [f"{disp_names[l]}\n(ONI, \u00b0C)" for l in order]
    subtitle = ("May-December 2027 - ILLUSTRATIVE author hypotheses (not a multi-model scenario), "
                "ONI / Nino 3.4 index, 1991-2020 baseline")
    export = pd.DataFrame(export_cols, index=pd.DatetimeIndex(dates, name='date'))
    return _render_oni_table(col_labels, cell_text,
                              "ONI under the three H2 2027 hypotheses (extension, author scenario)",
                              subtitle, filename, csv_export=export, csv_filename=csv_filename,
                              fig_width=9, highlight_col=2,
                              data_sources="H2 2027 hypotheses: author")


def plot_oni_official_table(official_scenario_aug26_avr27, q0, q5, q25, q75, q95, q100,
                             init_date="2026-08-01",
                             filename="oni_official_table_aug26_apr27.png",
                             csv_filename="oni_official_table_aug26_apr27.csv"):
    """Monthly table of the official ONI scenario (August 2026-April 2027):
    multi-model median + Q0-Q100/Q5-Q95/Q25-Q75 bands (APPROXIMATION,
    see warning at the top of the file). `q0`..`q100` are the same
    pd.Series already calculated in plot_oni_extended_dec2027()."""
    init_date = pd.Timestamp(init_date)
    dates = sorted(pd.Timestamp(k) for k in official_scenario_aug26_avr27)

    export = pd.DataFrame({
        'median_C': [official_scenario_aug26_avr27[d.strftime('%Y-%m-%d')] for d in dates],
        'q0_C': q0.reindex(dates).values, 'q5_C': q5.reindex(dates).values,
        'q25_C': q25.reindex(dates).values, 'q75_C': q75.reindex(dates).values,
        'q95_C': q95.reindex(dates).values, 'q100_C': q100.reindex(dates).values,
    }, index=pd.DatetimeIndex(dates, name='date'))

    cell_text = []
    for d in dates:
        r = export.loc[d]
        cell_text.append([
            d.strftime('%b %Y'),
            f"{r['median_C']:+.2f} \u00b0C",
            f"[{r['q25_C']:+.2f}; {r['q75_C']:+.2f}]",
            f"[{r['q5_C']:+.2f}; {r['q95_C']:+.2f}]",
            f"[{r['q0_C']:+.2f}; {r['q100_C']:+.2f}]",
        ])
    col_labels = ['Month', 'Median\n(ONI, \u00b0C)', 'Q25-Q75\n(\u00b0C)', 'Q5-Q95\n(\u00b0C)', 'Q0-Q100\n(\u00b0C)']
    subtitle = (f"{dates[0]:%B %Y}-{dates[-1]:%B %Y} - official multi-model scenario (Climate Dashboard), "
                f"initialized {init_date.day} {init_date:%B %Y} - approximate reconstruction of the quantile "
                f"envelope (member-by-member file not available here) - ONI / Nino 3.4 index, 1991-2020 baseline")
    return _render_oni_table(col_labels, cell_text,
                              "Official ONI forecast (approximate multi-model envelope)",
                              subtitle, filename, csv_export=export, csv_filename=csv_filename,
                              fig_width=10.5, highlight_col=1,
                              data_sources="Data: The Climate Brink @ ENSO Dashboard (approx. reconstruction)")


def plot_oni_extended_dec2027(model_fit, official_scenario_aug26_avr27, h2_scenarios,
                               analogs_24m, init_date="2026-08-01",
                               filename="oni_extended_synthesis_dec2027.png",
                               h2_table_filename="oni_h2_2027_hypotheses_table.png",
                               h2_table_csv_filename="oni_h2_2027_hypotheses_table.csv",
                               official_table_filename="oni_official_table_aug26_apr27.png",
                               official_table_csv_filename="oni_official_table_aug26_apr27.csv",
                               data_sources="Data: The Climate Brink @ ENSO Dashboard (Jan26-Apr27); "
                                            "ClimateReanalyzer (analogues, obs); H2 2027 hypotheses = author"):
    """
    analogs_24m: dict {"1982/1983": np.array(24 values), "1997/1998": ..., "2015/2016": ...}
        -- Niño 3.4 series from January (peak year)-December (peak year+1), aligned by
        CALENDAR POSITION (January matches January, etc.) on the Jan2026-Dec2027 axis,
        exactly as in your original Fig. 9.

    Retains the nested quantile bands (Q0-Q100/Q5-Q95/Q25-Q75) over the entire
    official period (August 2026-April 2027, orange palette = approximate
    reconstruction of the multi-model envelope) AND over the extension (May-December 2027,
    blue palette = author hypotheses), with a clearly different colour for the two
    confidence regimes so they can never be confused.
    """
    nino_obs = model_fit['nino']['ssta']
    init_date = pd.Timestamp(init_date)
    x_dates = pd.date_range('2026-01-01', '2027-12-01', freq='MS')
    splice = pd.Timestamp('2027-04-15')

    median_path = {}
    for d in pd.date_range('2026-01-01', '2026-07-01', freq='MS'):
        median_path[d] = nino_obs.loc[d]
    for k, v in official_scenario_aug26_avr27.items():
        median_path[pd.Timestamp(k)] = v
    for k, v in h2_scenarios['central'].items():
        median_path[pd.Timestamp(k)] = v
    median = pd.Series(median_path).sort_index().reindex(x_dates)

    low_path, high_path = median.copy(), median.copy()
    for k, v in h2_scenarios['la_nina_forte'].items():
        low_path[pd.Timestamp(k)] = v
    for k, v in h2_scenarios['neutre'].items():
        high_path[pd.Timestamp(k)] = v

    # --- official-period bands: uncertainty cone increasing with forecast
    #     lead time (APPROXIMATION -- see warning at the top of the file) ---
    lead = np.array([max((d.year - init_date.year) * 12 + (d.month - init_date.month), 0) for d in x_dates])
    hw0 = 0.49 + 0.145 * lead
    hw5, hw25 = 0.60 * hw0, 0.30 * hw0
    official_mask = (x_dates >= init_date) & (x_dates <= pd.Timestamp('2027-04-01'))

    q0, q100 = median.copy(), median.copy()
    q5, q95 = median.copy(), median.copy()
    q25, q75 = median.copy(), median.copy()
    q0[official_mask] = median[official_mask] - hw0[official_mask]
    q100[official_mask] = median[official_mask] + hw0[official_mask]
    q5[official_mask] = median[official_mask] - hw5[official_mask]
    q95[official_mask] = median[official_mask] + hw5[official_mask]
    q25[official_mask] = median[official_mask] - hw25[official_mask]
    q75[official_mask] = median[official_mask] + hw25[official_mask]

    # --- extension bands: envelope of the 3 hypotheses ---
    ext_mask = x_dates >= pd.Timestamp('2027-05-01')
    q0[ext_mask], q100[ext_mask] = low_path[ext_mask], high_path[ext_mask]
    q5[ext_mask] = low_path[ext_mask] + 0.10 * (median[ext_mask] - low_path[ext_mask])
    q95[ext_mask] = high_path[ext_mask] - 0.10 * (high_path[ext_mask] - median[ext_mask])
    q25[ext_mask] = low_path[ext_mask] + 0.50 * (median[ext_mask] - low_path[ext_mask])
    q75[ext_mask] = high_path[ext_mask] - 0.50 * (high_path[ext_mask] - median[ext_mask])

    fig, ax = plt.subplots(figsize=(16, 9.2))

    last_obs = nino_obs.index[-1]
    obs_mask = x_dates <= last_obs
    ax.plot(x_dates[obs_mask], nino_obs.reindex(x_dates[obs_mask]), color='black', lw=1.8,
            marker='o', ms=3.5, zorder=6, label='Observed (Niño 3.4)')

    off_idx = x_dates[(x_dates >= pd.Timestamp('2026-07-01')) & (x_dates <= pd.Timestamp('2027-04-01'))]
    ax.fill_between(off_idx, q0.reindex(off_idx), q100.reindex(off_idx), color='#fbe0c9', alpha=0.85, lw=0, zorder=1,
                     label='Q0-Q100 official (approx. reconstruction, Aug26-Apr27)')
    ax.fill_between(off_idx, q5.reindex(off_idx), q95.reindex(off_idx), color='#f3b487', alpha=0.9, lw=0, zorder=2,
                     label='Q5-Q95 official (approx. reconstruction)')
    ax.fill_between(off_idx, q25.reindex(off_idx), q75.reindex(off_idx), color='#e8703a', alpha=0.9, lw=0, zorder=3,
                     label='Q25-Q75 official (approx. reconstruction)')
    ax.plot(off_idx, median.reindex(off_idx), color='#c0392b', lw=2.8, marker='o', ms=4.5, zorder=5,
            label='Official median (Climate Dashboard)')

    ext_idx = x_dates[x_dates >= pd.Timestamp('2027-04-01')]
    ax.fill_between(ext_idx, q0.reindex(ext_idx), q100.reindex(ext_idx), color='#c9def2', alpha=0.85, lw=0, zorder=1,
                     label="Q0-Q100 extension (author hypotheses, illustrative)")
    ax.fill_between(ext_idx, q5.reindex(ext_idx), q95.reindex(ext_idx), color='#8fb9dd', alpha=0.9, lw=0, zorder=2,
                     label="Q5-Q95 extension (illustrative)")
    ax.fill_between(ext_idx, q25.reindex(ext_idx), q75.reindex(ext_idx), color='#3f7cac', alpha=0.9, lw=0, zorder=3,
                     label="Q25-Q75 extension (illustrative)")
    ax.plot(ext_idx, median.reindex(ext_idx), color='#1b5e8a', lw=2.8, ls='-', marker='o', ms=4.5, zorder=5,
            label='Central scenario — H2 2027 extension (author hypothesis)')
    ax.plot(ext_idx, high_path.reindex(ext_idx), color='#1b5e8a', lw=1.6, ls=':', zorder=4,
            label='Neutral ENSO hypothesis (extension upper bound)')
    ax.plot(ext_idx, low_path.reindex(ext_idx), color='#1b5e8a', lw=1.6, ls='-.', zorder=4,
            label='Strong La Niña hypothesis (extension lower bound)')

    colors_analog = {'1982/1983': '#8a6d3b', '1997/1998': '#6a3d9a', '2015/2016': '#c2185b'}
    for label, vals in analogs_24m.items():
        ax.plot(x_dates, vals, color=colors_analog.get(label, '#333'), lw=1.6, ls='--',
                marker='o', ms=3.5, zorder=4, label=f'Analog {label}')

    cats = [(3.0, 'Extreme El Niño (+3)', '#7b241c'), (2.0, 'Very strong El Niño (+2)', '#e74c3c'),
            (1.5, 'Strong El Niño (+1.5)', '#e67e22'), (1.0, 'Moderate El Niño (+1)', '#f1c40f'),
            (0.5, 'El Niño (+0.5)', '#f4d03f'), (-0.5, 'La Niña (-0.5)', '#5dade2')]
    for val, lab, color in cats:
        ax.axhline(val, color=color, ls='--', lw=1.2, zorder=0, alpha=0.85)
        ax.text(x_dates[-1] + pd.Timedelta(days=12), val, lab, color=color, fontsize=9.5,
                fontweight='bold', va='center')
    ax.axhline(0, color='gray', lw=0.8, zorder=0)

    ylo, yhi = -1.4, 6.3
    ax.set_ylim(ylo, yhi)
    ax.axvline(splice, color='#555555', lw=1.1, ls=':', zorder=9)
    ax.annotate('official', xy=(splice, 5.6), xytext=(-8, 0), textcoords='offset points',
                fontsize=9.5, color='#333', ha='right', va='center', rotation=90,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#999', alpha=0.9, linewidth=0.6), zorder=10)
    ax.annotate('H2 2027 extension', xy=(splice, 5.6), xytext=(8, 0), textcoords='offset points',
                fontsize=9.5, color='#1b5e8a', ha='left', va='center', rotation=90,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#1b5e8a', alpha=0.9, linewidth=0.6), zorder=10)

    ax.set_ylabel("ONI | Niño 3.4 index (\u00b0C) [1991-2020 baseline]", fontsize=11)
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate(rotation=90)
    ax.legend(loc='center left', fontsize=8.6, framealpha=0.95, ncol=1, bbox_to_anchor=(1.10, 0.5))

    fig.suptitle("ONI | Niño 3.4 index — synthesis and extension through December 2027",
                 x=0.02, y=0.975, ha='left', fontsize=19, fontweight='bold')
    fig.text(0.02, 0.928,
              "Official (August 2026-April 2027, orange tones): approximate reconstruction of the multi-model "
              "envelope (Climate Dashboard, initialized 1 August 2026 — member-by-member file not available here)\n"
              "Extension (May-December 2027, blue tones): ILLUSTRATIVE envelope built on 3 author hypotheses "
              "(strong La Niña / central / neutral ENSO), not a multi-model scenario",
              fontsize=10.5, style='italic', color='#555555', ha='left', va='top')
    fig.text(0.99, 0.008, data_sources, fontsize=8, color='#666', ha='right')

    plt.tight_layout(rect=[0, 0.02, 0.85, 0.90])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    h2_table_png, h2_table_csv = plot_oni_h2_2027_table(
        h2_scenarios, filename=h2_table_filename, csv_filename=h2_table_csv_filename)
    official_table_png, official_table_csv = plot_oni_official_table(
        official_scenario_aug26_avr27, q0, q5, q25, q75, q95, q100, init_date=init_date,
        filename=official_table_filename, csv_filename=official_table_csv_filename)

    return filename, h2_table_png, h2_table_csv, official_table_png, official_table_csv


# ====================================================================
# EXAMPLE CALL
# ====================================================================
#
# analogs_24m = {}
# for label, y0 in [('1982/1983', 1982), ('1997/1998', 1997), ('2015/2016', 2015)]:
#     analogs_24m[label] = enso_df_raw['enso_ssta'].loc[f'{y0}-01-01':f'{y0+1}-12-01'].values
#
# -- build once, immediately after fitting the model (model, X, ...
#    = fit_model(dataset, ...)) in enso_gmst_model.py --
# model_fit = build_model_fit(model, X, dataset, enso_df_raw, gmst_df, lag)
#
# -- IMPORTANT: do NOT call build_h2_2027_scenarios() without anchor_offset --
#    the embedded CSV (_H2_2027_ONI_CSV) was calibrated assuming a last
#    official month (April 2027) of ~1.0 degC; if the actual official value
#    differs (almost certainly the case), the default anchor_offset=0.0
#    produces an abrupt break between the last official point and the first
#    point of the H2 extension. Dynamically recalibrate against the actual
#    official value:
# H2_2027_CALIBRATION_ANCHOR = 1.0  # April 2027 value assumed by the CSV
# last_official_month = max(pd.Timestamp(k) for k in official_scenario_aug26_avr27)
# last_official_value = official_scenario_aug26_avr27[last_official_month.strftime('%Y-%m-01')]
# h2 = build_h2_2027_scenarios(anchor_offset=last_official_value - H2_2027_CALIBRATION_ANCHOR)
#
# png_path, csv_path, gmst_h2_table_png, gmst_h2_table_csv = plot_temp_extended_dec2027(
#     model_fit, official_scenario_aug26_avr27, h2, official_gmsta_table)
# oni_png, oni_h2_table_png, oni_h2_table_csv, oni_official_table_png, oni_official_table_csv = \
#     plot_oni_extended_dec2027(model_fit, official_scenario_aug26_avr27, h2, analogs_24m)

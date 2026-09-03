"""
enso_gmst_model.py (v3 -- non-linear model)
Monthly forecasting model for the global mean surface temperature
anomaly from the ENSO signal (Nino 3.4 SSTA, 1991-2020 base) and a
background trend, expressed as a preindustrial anomaly (consistent
with ERA5 / C3S).

Model (v3, non-linear):
    GMST_anom(t) = a + b*t + c*t^2 + d*ENSO(t-lag) + e*[ENSO(t-lag)*t] + resid(t)

- t, t^2: quadratic trend -> captures the recent acceleration of
  warming, clearly underestimated by a simple linear trend (measured:
  predicted std ~0.09C vs observed 0.24C with a pure linear fit ->
  0.17C with this quadratic + interaction model)
- ENSO(t-lag)*t: interaction term, lets GMST's sensitivity to ENSO grow
  with the background climate (stronger feedbacks in a warmer climate)
- Forms tested and discarded: cubic (no gain, unstable extrapolation
  risk), ENSO^2 alone (no gain on amplitude)

MODEL LIMITATIONS:
- captures neither volcanism (aerosols), nor the solar cycle, nor
  low-frequency variability such as PDO/AMO
- the future projection depends on an ENSO scenario supplied as input
  (IRI/CPC plumes, or a custom estimate such as CATL)
- remains a model with imposed functional forms (polynomial +
  interaction), not a true non-parametric model -> safer to
  extrapolate than a Random Forest/Gradient Boosting model (which
  plateau outside their training range, unsuited to a monotonically
  trending quantity such as GMST)
"""

import io
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from palette import (
    FACTUAL, FACTUAL_DARK, FACTUAL_DARKEST,
    COUNTERFACT, COUNTERFACT_2, COUNTERFACT_DARK, COUNTERFACT_LIGHT, COUNTERFACT_FILL,
    ALT_SCENARIO, HIGHLIGHT, RECORD, WARM_LIGHT, WARNING_BG,
    ANALOG_COLORS, ENSO_SCALE, light_tint,
    QUANTILE_BANDS, QUANTILE_MEDIAN, quantile_band_color, PROBABILITY_CMAP,
    overview_band_color,
)

# fit_ridge_standardized() ajuste volontairement sur des tableaux numpy nus
# (sans noms de colonnes) pour la standardisation ; les appels ultérieurs à
# .predict() sur des DataFrame déclenchent alors un avertissement sklearn
# purement cosmétique (aucun effet sur le résultat) -- on le masque ici.
warnings.filterwarnings("ignore", message="X has feature names, but .* was fitted without feature names")

# ----------------------------------------------------------------------
# 1. CHARGEMENT DES DONNÉES
# ----------------------------------------------------------------------

def load_era5_gmst_c3s(path_csv):
    """
    Charge un fichier bulletin C3S (ex : C3S_Bulletin_temp_*_DATA.csv).
    Ce format contient déjà l'anomalie préindustrielle calculée
    (colonne 'ano_pi'), donc aucun offset à appliquer manuellement.
    """
    df = pd.read_csv(path_csv, comment='#')
    df['month'] = pd.to_datetime(df['month'])
    df = df.set_index('month').sort_index()
    df = df.rename(columns={'ano_pi': 'gmst_anom_preind',
                             'ano_91-20': 'gmst_anom_9120'})
    return df


def load_era5_gmst(path_csv, base_offset_preindustrial=0.88):
    """
    Variante générique : charge un CSV ['date', 'gmst_anom_9120'] et
    applique un offset préindustriel fixe (moins précis que la fonction
    ci-dessus, à utiliser seulement si tu n'as pas le format C3S).
    """
    df = pd.read_csv(path_csv, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    df['gmst_anom_preind'] = df['gmst_anom_9120'] + base_offset_preindustrial
    return df


def load_enso(path_csv):
    """
    Charge l'indice ENSO (ex : Niño3.4 SSTA, base 1991-2020).
    Format attendu : colonnes ['date', 'enso_ssta']
    """
    df = pd.read_csv(path_csv, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    return df


def load_enso_climatereanalyzer(path_csv):
    """
    Charge un export Climate Reanalyzer (ClimateReanalyzer.org), ex :
    'era5-0p5deg_nino-3_4_sst_surface_anom_1991-2020.csv'.
    Format : en-tête texte de plusieurs lignes puis colonnes 'time,sst'
    avec time au format YYYYMM.
    """
    with open(path_csv, encoding='utf-8') as f:
        lines = f.readlines()
    header_idx = next(i for i, l in enumerate(lines) if l.strip().startswith('time,'))
    df = pd.read_csv(path_csv, skiprows=header_idx)
    df['date'] = pd.to_datetime(df['time'].astype(str), format='%Y%m')
    df = df.set_index('date').sort_index()
    df = df.rename(columns={'sst': 'enso_ssta'})
    return df[['enso_ssta']]


def _weighted_quantile(values, weights, quantiles):
    """
    Quantile empirique pondéré (interpolation linéaire sur la fonction de
    répartition cumulée pondérée). Utilisé pour donner un poids égal à
    chaque MODÈLE (indépendamment de la taille de son ensemble de
    membres), conformément à la méthodologie du Climate Dashboard
    (Hausfath et al., "each model weighted equally regardless of
    ensemble size").
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cw = np.cumsum(weights) - 0.5 * weights
    cw /= np.sum(weights)
    return np.interp(quantiles, cw, values)


def load_enso_dashboard_scenario(path_members, exclude_models=('SINTEX-F',)):
    """
    Reconstruit, à partir du fichier membre-par-membre du Climate
    Dashboard (enso_members_oni.csv, indice ONI/Niño 3.4), une enveloppe
    de scénario mensuelle EXCLUANT explicitement les modèles listés dans
    `exclude_models` (par défaut SINTEX-F, dont le biais froid marqué
    tire la borne basse et le percentile p05 vers des valeurs non
    représentatives des autres modèles -- cf. écarts observés modèle par
    modèle dans enso_members_oni.csv).

    Les percentiles (p05/p25/mediane/p75/p95) sont calculés en pondérant
    chaque MODÈLE également (et non chaque membre également), pour ne
    pas laisser un modèle à gros ensemble (ex. JMA, ~155 membres/mois)
    dominer un modèle à petit ensemble (ex. NASA-GEOS-S2S-2, 10 membres).
    Les bornes extrêmes (q0/q100) restent le min/max brut du pool de
    membres restant après exclusion -- c'est la définition la plus
    directe d'un "Q0-Q100" une fois l'outlier retiré.

    Retour
    ------
    DataFrame indexé par mois (Timestamp, jour=1), colonnes :
    ['median', 'p05', 'p25', 'p75', 'p95', 'q0', 'q100', 'n_models', 'n_members'].
    """
    members = pd.read_csv(path_members, comment='#')
    mex = members[~members['model'].isin(exclude_models)].copy()
    mex['weight'] = 1.0 / mex.groupby(['target_month', 'model'])['anomaly_c'].transform('count')

    rows = []
    for m in sorted(mex['target_month'].unique()):
        sub = mex[mex['target_month'] == m]
        p05, p25, med, p75, p95 = _weighted_quantile(
            sub['anomaly_c'].values, sub['weight'].values, [.05, .25, .5, .75, .95]
        )
        rows.append(dict(
            month=pd.Period(m, 'M').to_timestamp(),
            median=med, p05=p05, p25=p25, p75=p75, p95=p95,
            q0=sub['anomaly_c'].min(), q100=sub['anomaly_c'].max(),
            n_models=sub['model'].nunique(), n_members=len(sub),
        ))
    return pd.DataFrame(rows).set_index('month').sort_index()


def load_enso_dashboard_members(path_members, target_month, exclude_models=('SINTEX-F',)):
    """
    Retourne les membres bruts (valeur ONI + poids modèle-égal normalisé
    à somme 1) pour un seul mois cible (ex. '2026-11'), pour construire
    une distribution empirique (histogramme) -- typiquement le mois de
    paroxysme du scénario.
    """
    members = pd.read_csv(path_members, comment='#')
    sub = members[(members['target_month'] == target_month)
                  & (~members['model'].isin(exclude_models))].copy()
    sub['weight'] = 1.0 / sub.groupby('model')['anomaly_c'].transform('count')
    sub['weight'] /= sub['weight'].sum()
    return sub


# ----------------------------------------------------------------------
# 2. RECHERCHE DU LAG OPTIMAL ENSO -> GMST
# ----------------------------------------------------------------------

def smooth_enso(enso_df, window=3):
    """
    CAUSAL smoothing (backward-looking only, no future-information leak)
    of the ENSO index by a moving average over `window` months (current
    month + the previous (window-1) months).

    Why: a single month of raw ENSO is noisy (sub-seasonal variability);
    the real effect on GMST instead reflects an ocean-atmosphere forcing
    integrated over several months (thermal inertia). Smoothing BEFORE
    searching for the lag and fitting the model reduces predictor noise
    without introducing leakage (unlike a centred moving average, which
    would use future months).

    Measured gain (walk-forward validation, 12-month horizon, real, no
    leakage): RMSE 0.134 -> 0.127 C relative to raw ENSO, same features
    otherwise.

    Use this on the historical record (before fitting) AND on any future
    scenario supplied as forecast input (see smooth_enso_for_forecast),
    or training and forecasting will be inconsistent.
    """
    out = enso_df.copy()
    out['enso_ssta'] = enso_df['enso_ssta'].rolling(window, center=False, min_periods=1).mean()
    return out


def smooth_enso_for_forecast(enso_hist_df, enso_scenario, window=3):
    """
    Applies EXACTLY the same causal smoothing as smooth_enso(), but to a
    future scenario series (e.g. C3S/CFSv2 seasonal forecasts), using
    the most recent real historical ENSO values so that the first
    months of the scenario are smoothed correctly (otherwise the first
    1-2 scenario points would be under-smoothed, for lack of history
    within the scenario series alone).

    Parameters
    ----------
    enso_hist_df : historical raw ENSO DataFrame (column 'enso_ssta'),
                   as returned by load_enso_climatereanalyzer().
    enso_scenario : pd.Series {calendar_date: raw_enso_value} of the
                    (unsmoothed) future scenario, indexed by its own
                    calendar date (as for forecast_from_enso_calendar).

    Returns
    -------
    Smoothed pd.Series, same index as enso_scenario, ready to pass to
    forecast_from_enso_calendar().
    """
    combined = pd.concat([enso_hist_df['enso_ssta'], enso_scenario.rename('enso_ssta')])
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    smoothed = combined.rolling(window, center=False, min_periods=1).mean()
    return smoothed.loc[enso_scenario.index]


def detrend_linear(series):
    """
    Removes the linear trend from a series (residual = series - fitted
    trend). Necessary before searching for the ENSO lag: otherwise the
    secular warming trend (unrelated to ENSO) dominates the correlation
    and biases the search for the optimal lag.
    """
    t = np.arange(len(series))
    valid = ~series.isna()
    coeffs = np.polyfit(t[valid], series[valid], deg=1)
    trend = np.polyval(coeffs, t)
    return series - trend


def find_optimal_lag(enso, gmst, max_lag=12, detrend_gmst=True):
    """
    Searches for the lag (in months) that maximises the ENSO -> GMST
    correlation. By default the background trend of gmst is removed
    before the search (detrend_gmst=True), because the secular trend
    (anthropogenic forcing) has nothing to do with ENSO and would
    otherwise strongly dilute the correlation.
    """
    gmst_search = detrend_linear(gmst) if detrend_gmst else gmst
    best_lag, best_corr = 0, -np.inf
    for lag in range(0, max_lag + 1):
        shifted = enso.shift(lag)
        valid = pd.concat([shifted, gmst_search], axis=1, sort=False).dropna()
        if len(valid) < 24:
            continue
        corr = valid.iloc[:, 0].corr(valid.iloc[:, 1])
        if corr > best_corr:
            best_corr, best_lag = corr, lag
    return best_lag, best_corr


# ----------------------------------------------------------------------
# 3. DATASET AND MODEL CONSTRUCTION
# ----------------------------------------------------------------------

def build_dataset(enso_df, gmst_df, lag):
    df = gmst_df[['gmst_anom_preind']].join(enso_df[['enso_ssta']], how='inner')
    df['enso_lag'] = df['enso_ssta'].shift(lag)
    df['t_index'] = np.arange(len(df))
    df['t_index2'] = df['t_index'] ** 2          # quadratic trend (acceleration)
    df['enso_x_t'] = df['enso_lag'] * df['t_index']  # ENSO sensitivity growing with time
    df['month_num'] = df.index.month
    df = df.dropna(subset=['gmst_anom_preind', 'enso_lag'])
    return df


def fit_ridge_standardized(X, y, alpha):
    """
    Fits a Ridge(alpha) on STANDARDISED features (centred and scaled
    from X itself), then converts the resulting coefficients back to
    the raw scale before assigning them to the returned object.

    Why: the model's features have extremely heterogeneous scales
    (std of t_index2 ~ 3.2e5, vs ~0.84 for enso_lag and ~0.28 for the
    monthly dummies -- a factor of ~4e5). Ridge is not scale-invariant,
    so this leaves the problem numerically ill-conditioned: the
    solution becomes sensitive to the internal solver used by
    scikit-learn (a choice that can vary with the library version or
    platform), to the point of producing a measurably different
    intercept (~0.01C observed) for the very same alpha -- the root
    cause of the non-reproducibility issue reported after an
    environment update (Spyder/Anaconda/scikit-learn). Standardising
    makes the problem well-conditioned and the solution stable,
    independent of the environment.

    The result (.coef_, .intercept_) is then expressed back on the RAW
    feature scale (division by the std, intercept correction), so that
    the returned object behaves identically to a Ridge fitted directly
    on raw features -- no other function in this module (forecasting,
    attribution decomposition, etc.) needs to be changed.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    feat_mean = X_arr.mean(axis=0)
    feat_std = X_arr.std(axis=0, ddof=0)
    feat_std_safe = np.where(feat_std == 0, 1.0, feat_std)  # safeguard (constant column)
    X_std = (X_arr - feat_mean) / feat_std_safe

    model = Ridge(alpha=alpha)
    model.fit(X_std, y_arr)

    coef_std = model.coef_.copy()
    intercept_std = model.intercept_
    model.coef_ = coef_std / feat_std_safe
    model.intercept_ = intercept_std - np.sum(coef_std * feat_mean / feat_std_safe)
    return model


def select_ridge_alpha_loo(X, y, alphas):
    """
    Selects the Ridge hyperparameter alpha by EXACT leave-one-out cross-
    validation, via the closed-form leverage (hat matrix) formula --
    Stone (1974) for the LOO-CV principle, Golub, Heath & Wahba (1979)
    for the efficient closed-form computation without repeated
    refitting.

    Implemented here in plain numpy rather than via scikit-learn's
    RidgeCV(alphas=...): RidgeCV's internal search relies on an
    algorithm whose implementation has changed across library versions
    (the same issue already encountered and fixed for the walk-forward
    residual computation), which made the selected alpha -- and hence
    the whole downstream model -- dependent on the machine/version.
    This manual implementation, based only on basic linear algebra
    (matrix inversion), is strictly reproducible regardless of
    environment.

    For each alpha, the LOO error is computed without refitting the
    model n times: loo_residual_i = residual_i / (1 - H_ii), where
    H = X(X^T X + alpha*I)^-1 X^T is the projection ("hat") matrix
    (centred variables, so the intercept is treated as unpenalised,
    matching sklearn.linear_model.Ridge's default behaviour).

    Returns: (best alpha, corresponding LOO-MSE, dict {alpha: LOO-MSE})
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape

    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    Xc = X - X_mean
    yc = y - y_mean
    XtX = Xc.T @ Xc
    identity = np.eye(p)

    scores = {}
    best_alpha, best_mse = None, np.inf
    for alpha in alphas:
        A_inv = np.linalg.inv(XtX + alpha * identity)
        H = Xc @ A_inv @ Xc.T
        h_diag = np.clip(np.diag(H), None, 1 - 1e-10)  # évite une division par ~0
        resid = yc - H @ yc
        loo_resid = resid / (1 - h_diag)
        mse_loo = float(np.mean(loo_resid ** 2))
        scores[alpha] = mse_loo
        if mse_loo < best_mse:
            best_mse, best_alpha = mse_loo, alpha

    return best_alpha, best_mse, scores


def walk_forward_forecasts(X, y, fit_fn, horizon=12, min_train_frac=0.7, step=6):
    """
    Generic rolling-origin walk-forward evaluator, reusable for TESR and
    for any simpler benchmark model fitted the same way.

    For each origin (train size increasing by `step` months, starting at
    `min_train_frac` of the sample), fits `fit_fn(X_train, y_train)` on
    data available up to that origin only, then forecasts the next
    `horizon` months out of sample -- no leakage, matching a genuine
    pseudo-real-time forecasting setting.

    Parameters
    ----------
    X, y : pandas DataFrame / Series, chronologically ordered, same index.
    fit_fn : callable(X_train_values, y_train_values) -> object with
        a .predict(X_values) method. E.g. lambda Xt, yt:
        fit_ridge_standardized(Xt, yt, alpha) for TESR, or
        lambda Xt, yt: LinearRegression().fit(Xt, yt) for a plain
        benchmark.
    horizon : forecast length per origin, in months.
    min_train_frac : minimum fraction of the sample used for the first
        training window.
    step : number of months between successive origins.

    Returns
    -------
    DataFrame indexed by target date, columns ['obs', 'pred', 'residual'].
    """
    n = len(X)
    start = int(n * min_train_frac)
    rows = []
    for cut in range(start, n - horizon, step):
        model = fit_fn(X.iloc[:cut].values, y.iloc[:cut].values)
        pred = model.predict(X.iloc[cut:cut + horizon].values)
        obs = y.iloc[cut:cut + horizon].values
        dates = X.index[cut:cut + horizon]
        for d, o, p in zip(dates, obs, pred):
            rows.append({"date": d, "obs": o, "pred": p, "residual": o - p})
    return pd.DataFrame(rows).set_index("date").sort_index()


def fit_model(df, use_seasonal=True, regularize=True, test_size=0.2):
    """
    Non-linear model: quadratic trend (captures the acceleration of
    warming) + lagged ENSO + ENSO*time interaction (GMST's sensitivity
    to ENSO can grow with the background climate). Compared with a
    simple linear trend, this corrects a clear underestimation of
    recent amplitude (predicted std ~2x too low with a pure linear fit).

    Options (v4, validated by walk-forward, 12-month horizon, no
    leakage):
    - use_seasonal=True (default): adds monthly dummies. A residual
      seasonal cycle persists despite the anomaly transform -> on its
      own, this term delivers most of the gain (RMSE -4%).
    - regularize=True (default): RidgeCV instead of a plain linear
      regression. Stabilises the coefficients (t^2 and ENSO*t are
      correlated) and limits the risk of unstable extrapolation when
      forecasting. Modest but systematic additional gain on top of the
      dummies.
    Combined: walk-forward RMSE 0.134 -> 0.127 C, R^2 0.733 -> 0.759
    (measured over 1940-2026, ERA5/C3S + Nino3.4 ClimateReanalyzer).
    """
    features = ['t_index', 't_index2', 'enso_lag', 'enso_x_t']
    if use_seasonal:
        month_dummies = pd.get_dummies(df['month_num'], prefix='m', drop_first=True).astype(float)
        X = pd.concat([df[features], month_dummies], axis=1)
    else:
        X = df[features]
    y = df['gmst_anom_preind']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    if regularize:
        X_train_arr = X_train.values.astype(float)
        feat_mean = X_train_arr.mean(axis=0)
        feat_std = X_train_arr.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std == 0, 1.0, feat_std)
        X_train_std = (X_train_arr - feat_mean) / feat_std_safe

        best_alpha, best_mse_loo, _ = select_ridge_alpha_loo(X_train_std, y_train.values,
                                                              alphas=np.logspace(-3, 3, 50))
        model = fit_ridge_standardized(X_train.values, y_train.values, best_alpha)  # fit déjà fait en interne
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)

    print(f"Coefficients : {dict(zip(X.columns, model.coef_))}")
    print(f"Intercept : {model.intercept_:.3f}")
    if regularize:
        model.alpha_ = best_alpha  # pour compatibilité avec le reste du script (walk-forward, etc.)
        print(f"Selected ridge alpha (LOO-CV, reproducible manual computation, standardised features): "
              f"{model.alpha_:.4g} (MSE-LOO = {best_mse_loo:.5f})")

    return model, X, y, X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------
# 4. ÉVALUATION : R, R², RMSE
# ----------------------------------------------------------------------

def evaluate_model(model, X, y_obs, label="test"):
    y_pred = model.predict(X)
    r, _ = pearsonr(y_obs, y_pred)
    r2 = r2_score(y_obs, y_pred)
    rmse = np.sqrt(mean_squared_error(y_obs, y_pred))
    print(f"[{label}] R = {r:.3f} | R² = {r2:.3f} | RMSE = {rmse:.3f} °C")
    return y_pred, r, r2, rmse


def plot_model_vs_obs(df, model, X, y_obs,
                       data_sources="Data: ECMWF/Copernicus C3S ERA5 (ref. 1850-1900); Nino 3.4 SSTA: ClimateReanalyzer.org (ERA5, 1991-2020 base)",
                       title=None,
                       subtitle=None):
    """
    Confrontation modèle vs observations : série temporelle + nuage de
    points, habillage identique aux autres graphiques du script
    (typographie serif, crédits, seuils, palette de couleur cohérente).
    """
    y_pred, r, r2, rmse = evaluate_model(model, X, y_obs, label="ensemble complet")
    # -- Titre/sous-titre : REFORMULÉS -- l'ancien titre ("Model vs
    #    observations (ERA5)") ne disait rien de la période couverte ni de
    #    la performance du modèle ; construits ici dynamiquement (la
    #    période réelle de `df`, R² et RMSE déjà calculés ci-dessus) pour
    #    qu'ils portent l'information utile à un lecteur de preprint. --
    year_deb, year_fin = df.index[0].year, df.index[-1].year
    title = title or (f"Comparison of TESR-modelled and ERA5-observed global mean "
                       f"surface temperature anomalies, {year_deb}-{year_fin}")
    subtitle = subtitle or (f"Quadratic trend + Nino 3.4 ENSO term, full in-sample fit - "
                             f"R\u00b2 = {r2:.3f}, RMSE = {rmse:.3f} \u00b0C")

    plt.rcParams['font.family'] = 'serif'
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.6, 1]})

    # -- Panneau 1 : série temporelle --
    ax0 = axes[0]
    ax0.plot(df.index, y_obs, color='#1a1a1a', lw=1.1, label='Observed (ERA5, ref. 1850-1900)')
    ax0.plot(df.index, y_pred, color=FACTUAL, lw=1.1, alpha=0.85, label='Fitted model')
    ax0.axhline(1.5, color='#888888', linestyle=':', lw=1, label='+1.5 \u00b0C threshold (Paris Agreement)')
    ax0.set_ylabel("Temperature anomaly (\u00b0C, ref. 1850-1900)", fontsize=10.5)
    ax0.set_title("a) Time series", fontsize=11, loc='left', color='#333333')
    ax0.grid(True, alpha=0.25, lw=0.6)
    for spine in ['top', 'right']:
        ax0.spines[spine].set_visible(False)
    ax0.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=8.5)

    # -- Panneau 2 : nuage de points modèle vs obs --
    ax1 = axes[1]
    ax1.scatter(y_obs, y_pred, s=14, alpha=0.35, color=COUNTERFACT_2, edgecolor='none')
    lims = [min(y_obs.min(), y_pred.min()) - 0.05, max(y_obs.max(), y_pred.max()) + 0.05]
    ax1.plot(lims, lims, color='#1a1a1a', linestyle='--', lw=1.1, label='1:1')
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel("Observed (\u00b0C)", fontsize=10.5)
    ax1.set_ylabel("Modelled (\u00b0C)", fontsize=10.5)
    ax1.set_title("b) Model-observation correlation", fontsize=11, loc='left', color='#333333')
    ax1.grid(True, alpha=0.25, lw=0.6)
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=8.5)

    stats_txt = f"R = {r:.3f}\nR² = {r2:.3f}\nRMSE = {rmse:.3f} °C"
    ax1.text(0.04, 0.96, stats_txt, transform=ax1.transAxes, va='top', ha='left',
              fontsize=9, bbox=dict(boxstyle='round', facecolor='white',
                                     edgecolor='#888888', alpha=0.9))

    fig.suptitle(title, fontsize=15, fontweight='bold', x=0.02, ha='left', y=0.99)
    fig.text(0.02, 0.92, subtitle, fontsize=10, style='italic', color='#444444', ha='left')

    fig.text(0.98, 0.005, data_sources, fontsize=7.5, color='#666666', ha='right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.savefig("model_vs_obs.png", dpi=300)
    plt.show()
    plt.rcParams['font.family'] = 'sans-serif'
    return r, r2, rmse


# ----------------------------------------------------------------------
# 5. PROJECTION MANUELLE (saisie libre des valeurs ENSO)
# ----------------------------------------------------------------------

def forecast_manual(model, feature_cols, df, enso_values, start_date=None):
    """
    Projection mois par mois à partir de valeurs ENSO saisies manuellement.

    ATTENTION AU LAG (piège fréquent) :
    -----------------------------------
    enso_values[i] est utilisé TEL QUEL comme feature 'enso_lag' pour la
    date (start_date + i mois). Cette fonction n'applique AUCUN décalage
    automatique. Or le modèle a été entraîné avec 'enso_lag'(t) = ENSO
    réel observé à (t - lag) [cf. build_dataset()]. Donc enso_values[i]
    DOIT DÉJÀ être la valeur ENSO correspondant au mois (start_date+i-lag),
    et non la prévision ENSO du mois (start_date+i) lui-même.
    Exemple concret avec lag=4 : pour prédire le GMST de novembre 2026,
    il faut mettre en position correspondante la valeur ENSO de juillet
    2026 (novembre - 4 mois) -- pas la prévision ENSO de novembre.
    Se tromper ici décale tout le signal de `lag` mois et double
    grossièrement l'erreur (RMSE) sur un cas testé en interne.
    -> Si tu pars d'une série ENSO indexée par sa PROPRE date calendaire
    (ex. prévisions saisonnières C3S/CFSv2 mois par mois), utilise plutôt
    forecast_from_enso_calendar() ci-dessous, qui fait le décalage pour toi.

    Paramètres
    ----------
    enso_values : liste de floats, une valeur d'anomalie ENSO déjà décalée
                  du lag, une par mois cible
                  (ex : [0.4, 0.2, 0.0, -0.2, -0.4, -0.5])
    start_date  : date de départ de la projection (par défaut : le mois
                  suivant la dernière donnée du dataset). Permet aussi de
                  faire un backtest sur une période passée en choisissant
                  une date antérieure.

    Retour
    ------
    DataFrame indexé par date avec la colonne 'gmst_anom_pred_preind'
    """
    if start_date is None:
        start_date = df.index[-1] + pd.DateOffset(months=1)
        last_t = df['t_index'].iloc[-1]
    else:
        start_date = pd.to_datetime(start_date)
        # décalage en mois entre la DERNIÈRE date du dataset et start_date
        # (permet aussi bien une projection future qu'un backtest sur une
        # date antérieure, en restant cohérent avec l'axe t_index du fit)
        months_offset = (start_date.year - df.index[-1].year) * 12 + (start_date.month - df.index[-1].month)
        last_t = df['t_index'].iloc[-1] + months_offset - 1

    results = []
    for i, enso_val in enumerate(enso_values):
        date_i = start_date + pd.DateOffset(months=i)
        t_index = last_t + i + 1
        row = {
            't_index': t_index,
            't_index2': t_index ** 2,
            'enso_lag': enso_val,
            'enso_x_t': enso_val * t_index,
        }

        if any(c.startswith('m_') for c in feature_cols):
            for c in feature_cols:
                if c.startswith('m_'):
                    row[c] = 1 if c == f"m_{date_i.month}" else 0

        X_future = pd.DataFrame([row])[feature_cols]
        pred = model.predict(X_future)[0]
        results.append({'date': date_i, 'gmst_anom_pred_preind': pred, 'enso_input': enso_val})

    return pd.DataFrame(results).set_index('date')


def forecast_from_enso_calendar(model, feature_cols, df, lag, enso_calendar):
    """
    Variante sûre de forecast_manual() : tu donnes les valeurs ENSO indexées
    par LEUR PROPRE date calendaire (ex. prévision Niño3.4 de C3S/CFSv2
    mois par mois), et cette fonction applique elle-même le décalage de
    `lag` mois pour construire la bonne feature 'enso_lag' à la bonne date
    cible de GMST. Ça évite le piège documenté dans forecast_manual().

    Paramètres
    ----------
    lag : le lag (en mois) utilisé pour fit_model / build_dataset — donc
          celui retourné par find_optimal_lag() sur les vraies données.
    enso_calendar : dict ou pd.Series {date_calendaire: valeur_enso}.
          La date de chaque valeur est la date à laquelle cet ENSO est
          observé/prévu (pas la date du GMST cible).

    Logique
    -------
    Une valeur ENSO datée M sert à prédire le GMST du mois (M + lag).
    Donc si enso_calendar couvre [M0 .. M0+k], les GMST prédits couvrent
    [M0+lag .. M0+k+lag].

    Retour
    ------
    DataFrame indexé par date (mois cible du GMST, déjà décalés de +lag)
    avec la colonne 'gmst_anom_pred_preind'.
    """
    if isinstance(enso_calendar, dict):
        enso_calendar = pd.Series(enso_calendar)
    enso_calendar = enso_calendar.sort_index()
    enso_calendar.index = pd.to_datetime(enso_calendar.index)

    target_dates = enso_calendar.index + pd.DateOffset(months=lag)
    enso_values_aligned = list(enso_calendar.values)

    forecast_df = forecast_manual(model, feature_cols, df, enso_values_aligned,
                                   start_date=target_dates[0])
    # sécurité : vérifie que les dates cibles calculées correspondent bien
    assert list(forecast_df.index) == list(target_dates), \
        "Incohérence de dates dans le décalage du lag -- vérifier les entrées."
    return forecast_df


def decompose_forecast_enso_calendar(model, feature_cols, df, lag, enso_calendar):
    """
    Décomposition ADDITIVE et EXACTE de la prévision GMSTA en trois
    composantes (le modèle étant une régression linéaire -- Ridge --, la
    somme des trois est rigoureusement égale à la prédiction du modèle,
    sans approximation) :

      - trend    : contribution de la tendance temporelle (intercept +
                   β_t·t + β_t²·t²), i.e. la trajectoire que prédirait le
                   modèle hors de toute variabilité ENSO ou saisonnière ;
      - enso     : contribution du terme ENSO, incluant son interaction
                   avec le temps (β_enso·ENSO_lissé(t−τ) +
                   β_int·[ENSO_lissé(t−τ)·t]) -- c'est la grandeur
                   d'intérêt pour l'attribution ;
      - seasonal : contribution du cycle saisonnier résiduel (indicatrice
                   du mois).

    Équivaut à un calcul contrefactuel "ENSO neutre" (enso=0 partout) :
    total(t) - trend(t) - seasonal(t) = enso(t) par construction linéaire ;
    et total(t) évalué avec un enso_calendar à zéro redonnerait exactement
    trend(t) + seasonal(t). Les deux approches sont utilisées en parallèle
    dans le script (cette fonction pour la décomposition terme-à-terme,
    et un appel séparé à forecast_from_enso_calendar avec enso=0 pour la
    courbe contrefactuelle illustrative), à des fins de verification
    croisée.

    Retour
    ------
    DataFrame indexé par date cible (mois GMST, décalé de +lag) avec les
    colonnes 'trend', 'enso', 'seasonal', 'total' (= somme des trois,
    identique à la sortie de forecast_from_enso_calendar).
    """
    if isinstance(enso_calendar, dict):
        enso_calendar = pd.Series(enso_calendar)
    enso_calendar = enso_calendar.sort_index()
    enso_calendar.index = pd.to_datetime(enso_calendar.index)
    target_dates = enso_calendar.index + pd.DateOffset(months=lag)

    last_t = df['t_index'].iloc[-1]
    last_date = df.index[-1]
    coefs = dict(zip(feature_cols, model.coef_))
    intercept = model.intercept_

    rows = []
    for i, (enso_val, target_date) in enumerate(zip(enso_calendar.values, target_dates)):
        months_offset = (target_date.year - last_date.year) * 12 + (target_date.month - last_date.month)
        t_index = last_t + months_offset

        trend = intercept + coefs.get('t_index', 0.0) * t_index + coefs.get('t_index2', 0.0) * t_index ** 2
        enso_contrib = coefs.get('enso_lag', 0.0) * enso_val + coefs.get('enso_x_t', 0.0) * (enso_val * t_index)
        seasonal = coefs.get(f"m_{target_date.month}", 0.0)  # 0 pour janvier (mois de référence)
        total = trend + enso_contrib + seasonal
        rows.append({'date': target_date, 'trend': trend, 'enso': enso_contrib,
                      'seasonal': seasonal, 'total': total})

    return pd.DataFrame(rows).set_index('date')


def compare_historical_episodes(model, feature_cols, df, lag, enso_df, episodes,
                                 current_decomp=None, current_label=None):
    """
    Applique la MÊME décomposition d'attribution (trend/ENSO/saisonnier,
    modèle opérationnel unique) à plusieurs épisodes El Niño passés, pour
    comparer la part relative de la tendance ("anthropique") et de l'ENSO
    ("naturel") entre épisodes -- et avec le scénario en cours.

    IMPORTANT (à assumer explicitement dans le texte) : ceci est une
    attribution EX POST, avec le modèle entraîné sur l'ensemble des
    données disponibles aujourd'hui (jusqu'en 2026), appliqué
    rétrospectivement à des épisodes passés -- PAS une simulation de ce
    que le modèle aurait prédit en temps réel à l'époque (ce qui serait
    un exercice de validation de compétence, distinct, cf. §3.1). L'un
    répond à "avec notre compréhension actuelle de la relation
    tendance/ENSO, comment se répartit la chaleur observed lors de cet
    épisode ?", l'autre à "le modèle aurait-il pu l'anticiper à
    l'époque ?" -- deux questions différentes.

    episodes : dict {label: année_de_juillet_de_départ}, ex.
               {"1982-1983": 1982, "1997-1998": 1997, "2015-2016": 2015}
               La fenêtre couverte est juillet(année) -> juin(année+1),
               identique à celle utilisée pour le scénario 2026-2027.

    Retour : DataFrame indexé par label d'épisode, une ligne par épisode
    (moyenne et pic sur la fenêtre), + le scénario actuel si fourni.
    """
    rows = []
    for label, start_year in episodes.items():
        input_start = pd.Timestamp(f"{start_year}-04-01")
        input_end = pd.Timestamp(f"{start_year + 1}-03-01")
        calendar = enso_df['enso_ssta'].loc[input_start:input_end]
        if len(calendar) < 12:
            print(f"[compare_historical_episodes] {label} : données incomplètes "
                  f"({len(calendar)}/12 mois), épisode ignoré.")
            continue
        d = decompose_forecast_enso_calendar(model, feature_cols, df, lag, calendar)
        pic = d['total'].idxmax()
        rows.append({
            'episode': label,
            'total_moy': d['total'].mean(), 'total_pic': d.loc[pic, 'total'],
            'trend_moy': d['trend'].mean(), 'trend_pic': d.loc[pic, 'trend'],
            'enso_moy': d['enso'].mean(), 'enso_pic': d.loc[pic, 'enso'],
            'saisonnier_moy': d['seasonal'].mean(),
            'enso_pct_moy': 100 * d['enso'].mean() / d['total'].mean(),
            'enso_pct_pic': 100 * d.loc[pic, 'enso'] / d.loc[pic, 'total'],
            'mois_pic': pic,
        })

    if current_decomp is not None:
        pic = current_decomp['total'].idxmax()
        rows.append({
            'episode': current_label or "Ongoing scenario",
            'total_moy': current_decomp['total'].mean(), 'total_pic': current_decomp.loc[pic, 'total'],
            'trend_moy': current_decomp['trend'].mean(), 'trend_pic': current_decomp.loc[pic, 'trend'],
            'enso_moy': current_decomp['enso'].mean(), 'enso_pic': current_decomp.loc[pic, 'enso'],
            'saisonnier_moy': current_decomp['seasonal'].mean(),
            'enso_pct_moy': 100 * current_decomp['enso'].mean() / current_decomp['total'].mean(),
            'enso_pct_pic': 100 * current_decomp.loc[pic, 'enso'] / current_decomp.loc[pic, 'total'],
            'mois_pic': pic,
        })

    return pd.DataFrame(rows).set_index('episode')


def bootstrap_attribution_uncertainty(model, feature_cols, X_train, y_train, df, lag,
                                       enso_calendar, n_boot=1000, block_size=12, seed=42):
    """
    Incertitude autour de la décomposition d'attribution (trend/ENSO/saisonnier)
    par bootstrap par BLOCS MOBILES des résidus d'entraînement (Künsch, 1989),
    plus adapté qu'un bootstrap i.i.d. classique à des résidus mensuels
    autocorrélés (persistance ENSO/climatique d'un mois à l'autre).

    Principe (bootstrap résiduel, X fixé) :
      1. résidus = y_train - modèle.predict(X_train)  [résidus du modèle déjà
         entraîné, à alpha fixe -- cf. Reproductibilité]
      2. à chaque réplique : ré-échantillonnage des résidus par blocs de
         `block_size` mois consécutifs (préserve l'autocorrélation locale),
         nouvelle série y* = ŷ_train + résidus_rééchantillonnés
      3. ré-ajustement d'un Ridge à alpha FIXE (celui déjà sélectionné,
         model.alpha_ -- pas de nouvelle recherche RidgeCV, pour les mêmes
         raisons de reproductibilité qu'en walk-forward) sur (X_train, y*)
      4. décomposition trend/ENSO/saisonnier recalculée avec les nouveaux
         coefficients, pour le même scénario ENSO en entrée

    Le seed est fixé (défaut 42) : reproductible sur toute machine, malgré
    la composante aléatoire introduite ici (contrairement au reste du
    pipeline, volontairement déterministe -- cf. note en README/Méthodes).

    Retour
    ------
    DataFrame indexé par date cible, colonnes :
      enso_p5, enso_p50, enso_p95           (°C, contribution ENSO)
      enso_pct_p5, enso_pct_p50, enso_pct_p95   (%, part de ENSO dans total)
    """
    rng = np.random.default_rng(seed)
    n = len(X_train)
    y_hat_train = model.predict(X_train)
    resid_train = np.asarray(y_train) - y_hat_train

    target_dates = pd.to_datetime(enso_calendar.index) + pd.DateOffset(months=lag)
    n_dates = len(target_dates)
    enso_boot = np.empty((n_boot, n_dates))
    total_boot = np.empty((n_boot, n_dates))

    for b in range(n_boot):
        n_blocks = int(np.ceil(n / block_size))
        starts = rng.integers(0, max(n - block_size, 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        idx = np.clip(idx, 0, n - 1)
        y_star = y_hat_train + resid_train[idx]

        m_boot = fit_ridge_standardized(X_train, y_star, model.alpha_)

        decomp_b = decompose_forecast_enso_calendar(m_boot, feature_cols, df, lag, enso_calendar)
        enso_boot[b, :] = decomp_b['enso'].values
        total_boot[b, :] = decomp_b['total'].values

    pct_boot = 100 * enso_boot / total_boot

    result = pd.DataFrame({
        'enso_p5': np.percentile(enso_boot, 5, axis=0),
        'enso_p50': np.percentile(enso_boot, 50, axis=0),
        'enso_p95': np.percentile(enso_boot, 95, axis=0),
        'enso_pct_p5': np.percentile(pct_boot, 5, axis=0),
        'enso_pct_p50': np.percentile(pct_boot, 50, axis=0),
        'enso_pct_p95': np.percentile(pct_boot, 95, axis=0),
    }, index=target_dates)
    return result


def _combined_uncertainty_samples(central, residuals, bounds=None, n_samples=20000, seed=0):
    """
    Échantillon combinant DEUX sources d'incertitude indépendantes :
      1. l'incertitude propre du modèle TESR (résidus walk-forward réels,
         obs - prédit historique, ré-échantillonnés avec remise) ;
      2. si `bounds` est fourni, la dispersion multi-modèles du SCÉNARIO
         ENSO lui-même (bornes Q0/P05/P25/P75/P95/Q100 de la projection
         GMST, obtenues en repassant les quantiles ENSO du Climate
         Dashboard par le modèle central -- cf. enso_uncertainty /
         forecast_c3s_q0 etc.).

    Sans `bounds` (défaut), reproduit EXACTEMENT le comportement historique
    (central + résidus, un tirage par résidu, sans ré-échantillonnage) --
    rétrocompatible avec les appels existants.

    Avec `bounds` = dict/Series {'q0':, 'p05':, 'p95':, 'q100':, ...}
    (valeurs en anomalie GMST, °C), la dispersion ENSO est reconstruite
    par tirage à partir d'une fonction quantile linéaire par morceaux
    ancrée sur ces points + la valeur centrale (mediane), puis CHAQUE
    tirage ENSO est combiné à un résidu tiré indépendamment (avec remise)
    -- les deux sources ne sont donc jamais comptées deux fois l'une dans
    l'autre.

    Les clés 'p25'/'p75' sont OPTIONNELLES (rétrocompatibilité avec des
    `bounds` ne contenant que q0/p05/p95/q100) : si elles sont présentes,
    la fonction quantile utilise 7 points (Q0-P05-P25-mediane-P75-P95-
    Q100) au lieu de 5, ce qui resserre l'interpolation autour du corps
    de la distribution (là où l'essentiel de la masse de probabilité se
    trouve) plutôt que de relier P05 à la mediane par un seul segment
    linéaire.
    """
    if bounds is None:
        return central + residuals

    q0, p05, p95, q100 = bounds['q0'], bounds['p05'], bounds['p95'], bounds['q100']
    has_iqr = ('p25' in bounds) and ('p75' in bounds) and pd.notna(bounds['p25']) and pd.notna(bounds['p75'])
    if has_iqr:
        p25, p75 = bounds['p25'], bounds['p75']
        probs = np.array([0.0, 5.0, 25.0, 50.0, 75.0, 95.0, 100.0]) / 100.0
        vals = np.array([q0, p05, p25, central, p75, p95, q100])
    else:
        probs = np.array([0.0, 5.0, 50.0, 95.0, 100.0]) / 100.0
        vals = np.array([q0, p05, central, p95, q100])
    # -- garde-fou : impose la monotonie (au cas où le point central du
    #    modèle dévierait légèrement de la mediane multi-modèles, ce qui
    #    peut arriver car ce ne sont pas rigoureusement la même quantité) --
    vals = np.maximum.accumulate(vals)

    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, n_samples)
    enso_component = np.interp(u, probs, vals)
    resid_draw = rng.choice(np.asarray(residuals), size=n_samples, replace=True)
    return enso_component + resid_draw


def plot_probability_heatmap(forecast_df, residuals, thresholds=(1.5, 1.6, 1.7, 1.8, 1.9, 2.0),
                              enso_bounds_df=None,
                              data_sources="Data: ECMWF/Copernicus C3S - ERA5 (1850-1900 baseline)",
                              title="Exceedance probability of the Global Mean Surface Temperature\nAnomaly (GMSTA), by threshold and by month",
                              subtitle="TESR model - empirical probability",
                              filename="gmst_probability_heatmap.png"):
    """
    Tableau de probabilité de dépassement de seuils (ex. +1.5°C, +2°C, et
    étapes intermediaires) pour chaque mois d'un DataFrame de prévision.

    Méthode : pour chaque mois, on prend la prédiction centrale du modèle
    et on lui ajoute l'échantillon empirique des résidus walk-forward
    (observé - prédit, mesurés hors échantillon sur l'historique réel).
    La probabilité de dépassement d'un seuil T est simplement la
    proportion de cet échantillon (central + résidus) qui dépasse T.
    Plus robuste qu'une hypothèse gaussienne symétrique : les résidus
    réels sont ici modérément asymétriques (queue plus lourde vers le
    chaud), ce qui est repris fidèlement par cette approche empirique.

    Paramètres
    ----------
    forecast_df : DataFrame de prévision (colonne 'gmst_anom_pred_preind')
    residuals : array des résidus walk-forward (obs - prédit), issus par
                exemple d'une validation walk-forward sur l'historique.
    thresholds : liste des seuils (°C) à évaluer, en colonnes du tableau.
    """
    months = forecast_df.index
    table = np.zeros((len(months), len(thresholds)))
    has_enso_bounds = enso_bounds_df is not None
    for i, m in enumerate(months):
        central = forecast_df.loc[m, 'gmst_anom_pred_preind']
        bounds_m = enso_bounds_df.loc[m] if (has_enso_bounds and m in enso_bounds_df.index) else None
        samples = _combined_uncertainty_samples(central, residuals, bounds=bounds_m, seed=hash(str(m)) % (2**32))
        for j, t in enumerate(thresholds):
            table[i, j] = np.mean(samples > t) * 100

    if has_enso_bounds:
        subtitle += " - samples include multi-model ENSO spread uncertainty"

    plt.rcParams['font.family'] = 'serif'
    # header_inches doit couvrir le titre (2 lignes, 14.5pt bold) + le
    # sous-titre -- l'ancienne valeur (0.5in) sous-évaluait la hauteur réelle
    # du titre 2 lignes, et le sous-titre était positionné à une fraction de
    # figure FIXE (y=0.9) indépendante de header_inches/fig_height : les deux
    # se chevauchaient dès que le titre prenait sa pleine hauteur.
    header_inches = 0.95
    fig_height = max(5, 0.42 * len(months) + 2) + header_inches
    fig, ax = plt.subplots(figsize=(8.5, fig_height))

    im = ax.imshow(table, cmap=PROBABILITY_CMAP, vmin=0, vmax=100, aspect='auto')

    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"+{t:.1f}°C" for t in thresholds], fontsize=10)
    ax.set_yticks(range(len(months)))
    ax.set_yticklabels([_month_label(m).capitalize() for m in months], fontsize=9.5)

    # -- Couleur du texte : contraste calculé depuis la LUMINANCE RÉELLE de la
    #    couleur de cellule (via la colormap), pas un seuil de % arbitraire --
    #    l'ancienne règle ("white if val > 55 else '#1a1a1a'") ne blanchissait
    #    le texte QUE côté fort pourcentage, jamais côté proche de 0%, alors
    #    que ces cellules-là sont tout aussi sombres (bleu foncé). Cette
    #    version reste correcte quelle que soit la colormap utilisée. --
    cmap_probability = plt.get_cmap(PROBABILITY_CMAP)
    for i in range(len(months)):
        for j in range(len(thresholds)):
            val = table[i, j]
            r, g, b, _ = cmap_probability(val / 100.0)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            color = 'white' if luminance < 0.55 else '#1a1a1a'
            ax.text(j, i, f"{val:.0f}%", ha='center', va='center',
                     fontsize=9, color=color, fontweight='bold' if val > 90 or val < 10 else 'normal')

    ax.set_xticks(np.arange(-0.5, len(thresholds), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(months), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Exceedance probability (%)", fontsize=9.5)

    fig.suptitle(title, fontsize=14.5, fontweight='bold', x=0.02, ha='left', va='top',
                 y=1 - 0.08 / fig_height)
    fig.text(0.02, 1 - 0.72 / fig_height, subtitle, fontsize=9.5, style='italic',
             color='#444444', ha='left', va='top')
    fig.text(0.98, 0.003, data_sources, fontsize=7.5, color='#666666', ha='right')

    plt.tight_layout(rect=[0, 0.02, 1, 1 - header_inches / fig_height])
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.rcParams['font.family'] = 'sans-serif'
    return pd.DataFrame(table, index=months, columns=thresholds)


def plot_probability_distribution(central_value, residuals, month_label,
                                   thresholds=(1.5, 2.0),
                                   enso_bounds=None,
                                   data_sources="Data: ECMWF/Copernicus C3S - ERA5 (1850-1900 baseline)",
                                   title=None,
                                   subtitle=None,
                                   filename="gmst_probability_distribution.png"):
    """
    Distribution de probabilité (histogramme + densité lissée) de
    l'anomalie GMST pour un mois donné, construite en ajoutant à la
    prédiction centrale l'échantillon empirique des résidus walk-forward
    du modèle. Zones ombrées = probabilité de dépasser chaque seuil.
    """
    from scipy.stats import gaussian_kde

    samples = _combined_uncertainty_samples(central_value, residuals, bounds=enso_bounds, seed=0)
    title = title or (f"Modelled probability distribution of the global temperature anomaly, {month_label}\n"
                       f"Relative to the +1.5\u00b0C and +2\u00b0C warming thresholds")
    subtitle = subtitle or (
        f"Modelled projection in response to El Nino - departure from preindustrial (1850-1900), \u00b0C\n"
        f"{month_label}, warmest modelled month of the scenario - n={len(samples)} simulations\n"
        f"Projection initialized 1 August 2026"
    )
    if enso_bounds is not None:
        subtitle += " - includes multi-model ENSO spread uncertainty"

    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # On laisse la queue du KDE (noyau gaussien) se prolonger et retomber
    # naturellement vers 0 au-delà du min/max empirique, plutôt que de
    # couper l'affichage pile à la plage brute : une coupure nette à cet
    # endroit est visuellement incohérente (rupture brutale alors que la
    # densité y est encore non nulle). La grille est donc étendue d'une
    # marge basée sur la largeur de bande du KDE.
    emp_min, emp_max = samples.min(), samples.max()
    kde = gaussian_kde(samples)
    bw = kde.factor * samples.std(ddof=1)
    margin = 3 * bw
    x_grid = np.linspace(emp_min - margin, emp_max + margin, 500)
    density = kde(x_grid)
    # renormalise la densité affichée pour qu'elle intègre à 1 sur la
    # plage affichée (sinon l'extension de la grille biaiserait l'aire)
    dx = x_grid[1] - x_grid[0]
    density = density / (np.sum(density) * dx)

    ax.hist(samples, bins=30, density=True, color=FACTUAL, alpha=0.18, edgecolor='none')
    ax.plot(x_grid, density, color=FACTUAL_DARK, lw=1.8, label='Estimated density (KDE)')

    colors_fill = [WARM_LIGHT, FACTUAL]
    for t, col in zip(thresholds, colors_fill):
        mask = x_grid >= t
        ax.fill_between(x_grid[mask], 0, density[mask], color=col, alpha=0.35)
        p = np.mean(samples > t) * 100
        ax.axvline(t, color=col, linestyle='--', lw=1.3)
        ax.annotate(f"P(> +{t:.1f}°C) = {p:.0f}%",
                    xy=(t, kde(t)[0]), xytext=(t + 0.06, max(density) * 0.75),
                    fontsize=10, fontweight='bold', color=col,
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.1))

    ax.axvline(central_value, color='#1a1a1a', lw=1.3, linestyle='-',
               label=f'Central prediction ({central_value:+.2f}C)')
    n_label = f"n={len(samples)} draws (residuals + ENSO spread)" if enso_bounds is not None else f"n={len(residuals)} walk-forward residuals"
    ax.text(0.985, 0.95, f"Observed empirical maximum: {emp_max:+.2f}C\n({n_label})",
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5, color='#555555',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#cccccc', alpha=0.85))

    ax.set_xlabel("Global mean surface temperature anomaly (C) [1850-1900 baseline]", fontsize=10.5)
    ax.set_ylabel("Probability density", fontsize=10.5)
    ax.grid(True, alpha=0.25, lw=0.6)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=9)

    fig.suptitle(title, fontsize=15, fontweight='bold', x=0.02, ha='left', y=0.98)
    fig.text(0.02, 0.80, subtitle, fontsize=10, style='italic', color='#444444', ha='left')
    fig.text(0.98, 0.005, data_sources, fontsize=7.5, color='#666666', ha='right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.88])
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.rcParams['font.family'] = 'sans-serif'


def compute_forecast_synthesis(forecast_df, residuals, thresholds=(1.5, 2.0),
                                extreme_percentiles=(0, 100)):
    """
    Synthèse mois par mois : prédiction centrale, les DEUX EXTRÊMES de la
    prévision (percentiles bas/haut de la distribution empirique
    central + résidus walk-forward réels), et probabilité de dépassement
    de chaque seuil.

    "Extreme low/high" ici = percentiles empiriques (par défaut 0%/100%,
    soit le minimum et le maximum réellement observés dans l'échantillon
    walk-forward, c-à-d 100% du spectre), PAS une hypothèse gaussienne :
    ce sont les bornes réellement atteintes par l'échantillon empirique
    (central + résidus walk-forward réels, hors échantillon).
    """
    p_low, p_high = extreme_percentiles
    rows = []
    for m in forecast_df.index:
        central = forecast_df.loc[m, 'gmst_anom_pred_preind']
        samples = central + residuals
        row = {
            'mois': m,
            'central': central,
            f'extreme_bas_p{p_low}': np.percentile(samples, p_low),
            f'extreme_haut_p{p_high}': np.percentile(samples, p_high),
        }
        for t in thresholds:
            row[f'proba_gt_{t}'] = np.mean(samples > t) * 100
        rows.append(row)
    return pd.DataFrame(rows).set_index('mois')


def plot_enso_amplification(decomp, forecast_neutral, residuals, enso_bounds_df=None,
                             thresholds_marked=(1.5, 2.0), integral_range=(1.5, 2.0),
                             threshold_range=None, n_grid=400, peak_date=None,
                             data_sources="Data: ECMWF/Copernicus C3S - ERA5 (1850-1900 baseline)",
                             title=None,
                             subtitle="Empirical survival function P(anomaly > threshold) at the peak month of "
                                      "scenario - departure from preindustrial (1850-1900)\nCentral scenario "
                                      "(with El Nino) vs ENSO-neutral counterfactual (Nino 3.4 = 0) - projection initialized 1 August 2026",
                             filename="gmst_enso_amplification.png"):
    """
    Graphique synthétique demandé (1), version courbe de survie : trace,
    au mois de pic du scénario (ex. mars 2027), la PROBABILITÉ DE
    DÉPASSEMENT P(anomalie > seuil) en fonction du seuil -- une courbe
    décroissante par construction (plus le seuil est haut, moins il est
    probable de le dépasser) -- SOUS DEUX HYPOTHÈSES ENSO :
      - scénario central avec El Niño (`decomp['total']`, incertitude =
        résidus walk-forward + dispersion multi-modèles ENSO si
        `enso_bounds_df` fourni) ;
      - contrefactuel ENSO neutre (Niño 3.4 = 0 sur toute la période,
        `forecast_neutral`, incertitude = résidus walk-forward seuls --
        le contrefactuel est fixé par construction, sans dispersion de
        scénario ENSO à propager).

    La tendance de fond (réchauffement à long terme) est identique dans
    les deux courbes -- l'écart entre elles isole donc strictement la
    contribution d'El Niño. La zone hachurée entre les deux courbes
    visualise cet excès de risque de dépassement sur TOUTE la plage de
    seuils (pas seulement les deux seuils marqués) ; son aire, intégrée
    entre `integral_range` (par défaut 1,5°C-2,0°C), est calculée par
    trapèzes et affichée en légende -- c'est la version chiffrée de
    "the integral between the +1.5 and +2C probabilities" demandée : plus
    cette aire est grande, plus El Niño élève le risque cumulé de
    franchissement sur cette tranche, indépendamment du niveau exact du
    seuil retenu dans la tranche.

    Des traits verticaux marquent `thresholds_marked` (par défaut 1,5°C
    et 2,0°C) avec la probabilité de dépassement sous les deux
    hypothèses et le facteur d'amplification (avec El Niño / neutre).

    Retour : DataFrame indexé par seuil (`thresholds_marked`), colonnes
    ['proba_el_nino_%', 'proba_neutre_%', 'facteur_amplification'].
    """
    peak_date = peak_date or decomp['total'].idxmax()
    central_nino = decomp.loc[peak_date, 'total']
    central_neutral = forecast_neutral.loc[peak_date, 'gmst_anom_pred_preind']
    bounds_m = (enso_bounds_df.loc[peak_date]
                if (enso_bounds_df is not None and peak_date in enso_bounds_df.index) else None)

    samples_nino = _combined_uncertainty_samples(central_nino, residuals, bounds=bounds_m, seed=11)
    samples_neutral = _combined_uncertainty_samples(central_neutral, residuals, bounds=None, seed=12)

    # -- Plage de seuils par défaut : du bas de la distribution neutre
    #    (là où les deux courbes sont encore proches de 100%) jusqu'à un
    #    peu au-delà de la queue chaude d'El Niño (là où la courbe El
    #    Niño devient proche de 0%) -- couvre toute la zone où les deux
    #    courbes se distinguent visuellement. --
    if threshold_range is None:
        lo = np.percentile(samples_neutral, 2)
        hi = np.percentile(samples_nino, 99)
        threshold_range = (lo, hi)
    t_grid = np.linspace(threshold_range[0], threshold_range[1], n_grid)
    p_nino = np.array([np.mean(samples_nino > t) for t in t_grid]) * 100
    p_neutral = np.array([np.mean(samples_neutral > t) for t in t_grid]) * 100

    # -- Aire entre les deux courbes (intégrale par trapèzes), restreinte
    #    à `integral_range` -- unité "% points x C", proxy du risque
    #    cumulé additionnel dû à El Niño sur cette tranche de seuils. --
    mask_int = (t_grid >= integral_range[0]) & (t_grid <= integral_range[1])
    _trapz = getattr(np, 'trapezoid', None) or np.trapz  # np.trapz retiré en numpy>=2.0
    aire_excess = _trapz((p_nino - p_neutral)[mask_int], t_grid[mask_int])

    title = title or (f"El Nino-driven amplification of threshold-exceedance probability, "
                       f"{_month_label(peak_date)}")

    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.plot(t_grid, p_nino, color=FACTUAL_DARK, lw=2.3,
            label=f'Avec El Niño ({central_nino:+.2f}°C central)')
    ax.plot(t_grid, p_neutral, color='#333333', lw=2.0, ls='--',
            label=f'ENSO neutre, contrefactuel ({central_neutral:+.2f}°C central)')
    ax.fill_between(t_grid, p_neutral, p_nino, where=(p_nino >= p_neutral),
                     facecolor='none', edgecolor=HIGHLIGHT, hatch='////', linewidth=0.0, zorder=2,
                     label="Excess risk due to El Nino")
    ax.fill_between(t_grid, p_neutral, p_nino, where=(p_nino >= p_neutral),
                     color=HIGHLIGHT, alpha=0.14, lw=0, zorder=1.8)

    # -- Aire intégrée (valeur chiffrée), affichée à part de la légende
    #    pour ne pas la surcharger -- coin haut-droit, zone où les deux
    #    courbes sont déjà retombées près de 0% donc naturellement libre. --
    ax.text(0.985, 0.92,
            f"Cumulative excess risk {integral_range[0]:.1f}-{integral_range[1]:.1f}C:\n"
            f"area approx. {aire_excess:.0f} pts%*C between the two curves",
            transform=ax.transAxes, ha='right', va='top', fontsize=9, color=HIGHLIGHT,
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=HIGHLIGHT, alpha=0.92))

    # -- Boîtes d'annotation des seuils marqués, AVEC flèche (repère utile
    #    pour lire où sur la courbe le seuil est atteint) -- mais pointant
    #    toujours vers la courbe El Niño (rouge) et placées AU-DESSUS
    #    d'elle, jamais en diagonale à travers les deux courbes. On
    #    réserve pour ça une marge au-dessus de 100% (axe étendu jusqu'à
    #    `y_headroom`, sans graduation au-delà de 100 -- espace "hors
    #    courbe" garanti quelle que soit la forme des deux courbes). --
    y_headroom = 114
    colors_t = [WARM_LIGHT, FACTUAL]
    y_boxes = [108, 30]
    proba_rows = []
    for i, (t, col) in enumerate(zip(thresholds_marked, colors_t)):
        p_n = np.mean(samples_nino > t) * 100
        p_0 = np.mean(samples_neutral > t) * 100
        factor = (p_n / p_0) if p_0 >= 0.5 else np.nan
        proba_rows.append({'seuil': t, 'proba_el_nino_%': p_n, 'proba_neutre_%': p_0,
                            'facteur_amplification': factor})
        ax.axvline(t, color=col, linestyle=':', lw=1.3, zorder=1.5, ymax=1.0)
        ax.scatter([t, t], [p_n, p_0], color=[FACTUAL_DARK, '#333333'], s=32, zorder=6,
                   edgecolor='white', linewidth=0.6)
        factor_txt = f"soit ×{factor:.1f}" if np.isfinite(factor) else "quasi nulle sans El Niño"
        ax.annotate(
            f"+{t:.1f}°C : {p_n:.0f}% (El Niño) vs {p_0:.0f}% (neutre)\n{factor_txt}",
            xy=(t, p_n), xytext=(t + 0.05, y_boxes[i]),
            fontsize=9, fontweight='bold', color=col, ha='left', va='center', zorder=7,
            arrowprops=dict(arrowstyle='->', color=col, lw=1.1),
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=col, alpha=0.95),
            annotation_clip=False)

    ax.set_xlabel("Global mean surface temperature anomaly threshold (C) [1850-1900 baseline]",
                   fontsize=10.5)
    ax.set_ylabel("Threshold exceedance probability (%)", fontsize=10.5)
    ax.set_ylim(0, y_headroom)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlim(t_grid[0], t_grid[-1])
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.axhline(100, color='#cccccc', lw=0.8, zorder=0.5)  # rappel visuel du plafond 100%
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    # -- Légende sortie du graphique (bande dédiée sous l'axe), à distance
    #    resserrée pour limiter le blanc entre le graphique et la légende. --
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=3,
              frameon=True, framealpha=0.9, fontsize=9)

    fig.suptitle(title, fontsize=14.5, fontweight='bold', x=0.02, ha='left', y=0.985)
    fig.text(0.02, 0.9, subtitle, fontsize=9.5, style='italic', color='#444444', ha='left')
    fig.text(0.98, 0.018, data_sources, fontsize=7.5, color='#666666', ha='right')

    plt.tight_layout(rect=[0, 0.06, 1, 0.905])
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.rcParams['font.family'] = 'sans-serif'

    return pd.DataFrame(proba_rows).set_index('seuil')


def plot_climate_baseline_vs_peak(gmst_df, forecast_df, residuals, enso_bounds_df=None,
                                   baseline_period=("2016-01-01", "2026-12-31"),
                                   peak_date=None,
                                   data_sources="Data: ECMWF/Copernicus C3S - ERA5 (1850-1900 baseline)",
                                   title=None,
                                   subtitle=None,
                                   filename="gmst_climat_actuel_vs_pic.png"):
    """
    Graphique synthétique demandé (2) : distribution en cloche du
    "climat actuel" -- anomalies mensuelles OBSERVÉES sur
    `baseline_period` (par défaut 2016-2026), ajustées par une loi
    normale -- comparée à la distribution PROJETÉE du mois de pic du
    scénario (ex. mars 2027 : central + résidus walk-forward +
    dispersion multi-modèles ENSO si `enso_bounds_df` fourni), pour
    visualiser à quel point cette dernière s'écarte de la variabilité
    récente "normale".

    L'écart est quantifié de deux façons :
      - en écarts-types de la distribution de référence (z-score du
        centre de la distribution projetée par rapport à μ/σ du
        climat actuel) ;
      - en rang percentile EMPIRIQUE dans l'historique observé de la
        période de référence (pas une hypothèse gaussienne).

    Retour : dict avec mu/sigma de référence, valeur centrale projetée,
    z-score et percentiles (empirique + théorique sous loi normale).
    """
    from scipy.stats import gaussian_kde, norm, percentileofscore

    baseline = gmst_df['gmst_anom_preind'].loc[baseline_period[0]:baseline_period[1]].dropna()
    mu_b, sigma_b = baseline.mean(), baseline.std(ddof=1)

    peak_date = peak_date or forecast_df['gmst_anom_pred_preind'].idxmax()
    central_peak = forecast_df.loc[peak_date, 'gmst_anom_pred_preind']
    bounds_m = (enso_bounds_df.loc[peak_date]
                if (enso_bounds_df is not None and peak_date in enso_bounds_df.index) else None)
    samples_peak = _combined_uncertainty_samples(central_peak, residuals, bounds=bounds_m, seed=21)

    z_score = (central_peak - mu_b) / sigma_b
    pct_rank_obs = percentileofscore(baseline.values, central_peak, kind='mean')
    pct_rank_normal = norm.cdf(z_score) * 100

    year_deb = pd.Timestamp(baseline_period[0]).year
    year_fin = pd.Timestamp(baseline_period[1]).year
    title = title or (f"Departure of the projected {_month_label(peak_date)} anomaly from the "
                       f"{year_deb}-{year_fin} observed climate")
    subtitle = subtitle or (
        f"Current climate = observed monthly anomalies {year_deb}-{year_fin} - departure from preindustrial "
        f"preindustrial (1850-1900) - C\n"
        f"vs projected distribution in {_month_label(peak_date)}, at peak El Nino influence - projection initialized 1 August 2026")

    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # -- Climat actuel : histogramme des observations + cloche normale ajustée --
    ax.hist(baseline.values, bins=18, density=True, color=COUNTERFACT, alpha=0.22, edgecolor='none',
            label=f'Observations mensuelles {year_deb}-{year_fin} (n={len(baseline)})')
    x_lo = min(baseline.min(), samples_peak.min()) - 0.15
    x_hi = max(baseline.max(), samples_peak.max()) + 0.15
    x_grid = np.linspace(x_lo, x_hi, 500)
    bell = norm.pdf(x_grid, mu_b, sigma_b)
    ax.plot(x_grid, bell, color=COUNTERFACT, lw=2.0,
            label=f'Current climate, fitted normal (mu={mu_b:+.2f}C, sigma={sigma_b:.2f}C)')

    # -- Distribution projetée du mois de pic --
    kde = gaussian_kde(samples_peak)
    bw = kde.factor * samples_peak.std(ddof=1)
    grid_peak = np.linspace(samples_peak.min() - 3 * bw, samples_peak.max() + 3 * bw, 500)
    dens_peak = kde(grid_peak)
    dx = grid_peak[1] - grid_peak[0]
    dens_peak = dens_peak / (np.sum(dens_peak) * dx)
    ax.fill_between(grid_peak, 0, dens_peak, color=FACTUAL, alpha=0.20, lw=0)
    ax.plot(grid_peak, dens_peak, color=FACTUAL_DARK, lw=2.2,
            label=f'{_month_label(peak_date)}, projected ({central_peak:+.2f}C central)')

    ax.axvline(mu_b, color=COUNTERFACT, lw=1.1, ls=':')
    ax.axvline(central_peak, color=FACTUAL_DARK, lw=1.3, ls='-')

    ax.set_ylim(0, 3.8)
    y_top = 3.8
    #y_top = max(bell.max(), dens_peak.max())
    ax.annotate(
        f"{central_peak:+.2f}C\n= {z_score:+.1f} sigma above the current climate\n"
        f"(percentile approx. {pct_rank_obs:.0f}% of the {year_deb}-{year_fin} historical record)",
        xy=(central_peak, kde(central_peak)[0]), xytext=(central_peak + 0.08, y_top * 0.85),
        fontsize=9.5, fontweight='bold', color=FACTUAL_DARK, ha='left',
        arrowprops=dict(arrowstyle='->', color=FACTUAL_DARK, lw=1.1),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=FACTUAL_DARK, alpha=0.9))

    ax.set_xlabel("Global mean surface temperature anomaly (C) [1850-1900 baseline]", fontsize=10.5)
    ax.set_ylabel("Probability density", fontsize=10.5)
    ax.grid(True, alpha=0.25, lw=0.6)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=8.8)

    fig.suptitle(title, fontsize=14.5, fontweight='bold', x=0.02, ha='left', y=0.98)
    fig.text(0.02, 0.865, subtitle, fontsize=9.3, style='italic', color='#444444', ha='left')
    fig.text(0.98, 0.005, data_sources, fontsize=7.5, color='#666666', ha='right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.84])
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.rcParams['font.family'] = 'sans-serif'

    return {
        'mu_baseline': mu_b, 'sigma_baseline': sigma_b,
        'central_peak': central_peak, 'z_score': z_score,
        'percentile_empirique_baseline': pct_rank_obs,
        'percentile_normal_baseline': pct_rank_normal,
    }


def plot_fan_chart(df, forecast_df, residuals, zoom=None,
                    percentile_bands=((5, 95), (25, 75)),
                    thresholds=(1.5, 2.0),
                    data_sources="Données : ECMWF/Copernicus C3S — ERA5 (réf. 1850-1900)",
                    title="Global Mean Surface Temperature Anomaly (GMSTA) - probability fan chart",
                    subtitle="TESR model - empirical percentile bands (actual walk-forward residuals)",
                    filename="gmst_fan_chart.png"):
    """
    "Fan chart" : historique observé + prédiction centrale + bandes de
    percentiles emboîtées (ex. 5-95% et 25-75%) montrant visuellement les
    DEUX EXTRÊMES plausibles de la prévision à chaque mois, construites à
    partir des résidus walk-forward réels du modèle (pas une hypothèse
    gaussienne symétrique).
    """
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(13, 6.5))

    ax.plot(df.index, df['gmst_anom_preind'], color='#1a1a1a', lw=1.2,
            label='Observed (ERA5, ref. 1850-1900)')

    last_obs_date = df.index[-1]
    last_obs_val = df['gmst_anom_preind'].iloc[-1]
    fc_dates = [last_obs_date] + list(forecast_df.index)
    central_vals = np.array([last_obs_val] + list(forecast_df['gmst_anom_pred_preind']))

    # percentile_bands is given widest-first (e.g. 5-95 then 25-75), so
    # colours go palest-first -> most saturated for the narrowest, innermost
    # band -- a real graded ramp, not the same hue at different alphas.
    n_bands = len(percentile_bands)
    band_colors = [quantile_band_color(i, n_bands) for i in range(n_bands)]
    for (p_low, p_high), color in zip(percentile_bands, band_colors):
        low = np.array([last_obs_val] + [np.percentile(v + residuals, p_low)
                                          for v in forecast_df['gmst_anom_pred_preind']])
        high = np.array([last_obs_val] + [np.percentile(v + residuals, p_high)
                                           for v in forecast_df['gmst_anom_pred_preind']])
        ax.fill_between(fc_dates, low, high, color=color, alpha=1.0,
                         label=f'Bande {p_low}-{p_high}%', lw=0)

    ax.plot(fc_dates, central_vals, 'o--', color=FACTUAL, lw=1.8, ms=4.5,
            label='Central prediction')

    threshold_styles = {1.5: (':', '#888888', '+1.5 C threshold (Paris Agreement)'),
                         2.0: ('--', '#555555', 'Seuil +2 °C')}
    for th in thresholds:
        ls, col, lab = threshold_styles.get(th, ('-.', '#999999', f'Seuil +{th} °C'))
        ax.axhline(th, color=col, linestyle=ls, lw=1.1, label=lab)

    if zoom is not None:
        z0, z1 = pd.to_datetime(zoom[0]), pd.to_datetime(zoom[1])
        ax.set_xlim(z0, z1)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(FuncFormatter(_month_axis_formatter))
        plt.setp(ax.get_xticklabels(), rotation=90, ha='center', fontsize=7.5)

    ax.set_ylabel("Temperature anomaly (C, ref. 1850-1900)", fontsize=11)
    ax.grid(True, alpha=0.25, lw=0.6)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=8.5, ncol=1)

    fig.suptitle(title, fontsize=15, fontweight='bold', x=0.02, ha='left', y=0.98)
    fig.text(0.02, 0.90, subtitle, fontsize=10, style='italic', color='#444444', ha='left')
    fig.text(0.98, 0.005, data_sources, fontsize=7.5, color='#666666', ha='right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.88])
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.rcParams['font.family'] = 'sans-serif'


def plot_monthly_table(synthesis_df,
                        data_sources="Données : ECMWF/Copernicus C3S — ERA5 (réf. 1850-1900)",
                        title="Global Mean Surface Temperature Anomaly (GMSTA)\nMonthly summary of the ENSO scenario",
                        subtitle="TESR model - extremes = empirical minimum/maximum, full spread (actual walk-forward residuals)",
                        filename="gmst_monthly_table.png"):
    """
    Rendu "tableau" de la synthèse mensuelle (compute_forecast_synthesis) :
    mois, prédiction centrale, les deux extrêmes (bas/haut), probabilités
    de dépassement des seuils. Plus lisible qu'un DataFrame brut pour une
    présentation.
    """
    cols = list(synthesis_df.columns)
    n_rows = len(synthesis_df)
    header_inches = 1.05
    fig_height = max(3.2, 0.34 * n_rows + 0.5) + header_inches
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.axis('off')

    col_labels = ['Mois', 'Central (°C)', 'Extreme low (C)', 'Extreme high (C)'] + \
                 [f'P(> +{c.split("_")[-1]}°C)' for c in cols if c.startswith('proba_gt_')]
    cell_text = []
    for m, row in synthesis_df.iterrows():
        line = [_month_label(m).capitalize(), f"{row['central']:+.2f}"]
        line += [f"{row[c]:+.2f}" for c in cols if c.startswith('extreme_')]
        line += [f"{row[c]:.0f}%" for c in cols if c.startswith('proba_gt_')]
        cell_text.append(line)

    # Table ancrée en haut de la zone disponible (juste sous le sous-titre),
    # pas centrée verticalement -> évite le grand vide au-dessus.
    # NB : bbox est en coordonnées LOCALES de l'axe (0=bas, 1=haut de l'axe).
    # L'axe est positionné directement via set_position() (voir plus bas)
    # plutôt que via tight_layout(), qui insérait une marge supplémentaire
    # non désirée entre le sous-titre et le tableau.
    table_frac_height = min(0.95, (0.34 * n_rows + 0.15) / (fig_height - header_inches - 0.02))
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                      bbox=[0.0, 1.0 - table_frac_height, 1.0, table_frac_height],
                      cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#dddddd')
        if r == 0:
            cell.set_facecolor('#1a1a1a')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor(light_tint(FACTUAL, 0.08) if r % 2 == 0 else 'white')

    plt.rcParams['font.family'] = 'serif'
    fig.suptitle(title, fontsize=14.5, fontweight='bold', x=0.02, ha='left',
                 y=1 - 0.30 / fig_height)
    fig.text(0.02, 1 - 0.95 / fig_height, subtitle, fontsize=9.5, style='italic',
             color='#444444', ha='left')
    fig.text(0.98, 0.003, data_sources, fontsize=7.5, color='#666666', ha='right')

    # Positionnement direct de l'axe (au lieu de tight_layout) : évite la
    # marge automatique supplémentaire que tight_layout insérait entre le
    # sous-titre et le haut du tableau.
    top_frac = 1 - header_inches / fig_height
    ax.set_position([0.015, 0.02, 0.97, top_frac - 0.02])
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.rcParams['font.family'] = 'sans-serif'


def plot_forecast(df, forecast_df, obs_future=None, title="GMST projection"):
    """
    Affiche l'historique + la projection. Si obs_future est fourni
    (série observed couvrant la période de projection, utile pour un
    backtest), elle est superposée et R/RMSE sont calculés sur cette
    période spécifique.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df.index, df['gmst_anom_preind'], label="Observed (historical)", color="black", lw=1.1)
    ax.plot(forecast_df.index, forecast_df['gmst_anom_pred_preind'],
            'o--', label="Projection (ENSO saisi manuellement)", color=FACTUAL)

    if obs_future is not None:
        common_idx = forecast_df.index.intersection(obs_future.index)
        if len(common_idx) > 0:
            y_obs = obs_future.loc[common_idx]
            y_pred = forecast_df.loc[common_idx, 'gmst_anom_pred_preind']
            r, _ = pearsonr(y_obs, y_pred)
            rmse = np.sqrt(mean_squared_error(y_obs, y_pred))
            ax.plot(common_idx, y_obs, 's-', label="Observed (test period)", color=COUNTERFACT)
            ax.text(0.02, 0.95, f"R = {r:.3f}\nRMSE = {rmse:.3f} °C",
                    transform=ax.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    ax.axhline(1.5, color='gray', linestyle=':', label='Seuil 1.5°C')
    ax.set_ylabel("GMST anomaly (C vs 1850-1900)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig("gmst_forecast.png", dpi=150)
    plt.show()


def plot_forecast_academic(df, forecast_df, obs_future=None,
                            data_sources="Data: ECMWF/Copernicus C3S ERA5 (ref. 1850-1900); Nino 3.4 SSTA: ClimateReanalyzer.org (ERA5, 1991-2020 base)",
                            title="Global mean surface temperature anomaly",
                            subtitle="ERA5 observations and statistical projection (quadratic trend + Nino 3.4 ENSO, optimal lag)",
                            scenario_warning=None):
    """
    Version présentation/publication du graphique de projection :
    - typographie serif, habillage sobre
    - titre + sous-titre méthodologique
    - crédit auteur et source des données en pied de figure
    - annotation optionnelle si le scénario ENSO saisi sort de la plage
      historiquement observed (scenario_warning : str ou None)
    """
    plt.rcParams['font.family'] = 'serif'

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(df.index, df['gmst_anom_preind'], color='#1a1a1a', lw=1.1,
            label='Observed (ERA5, 1850-1900 base)')
    ax.plot(forecast_df.index, forecast_df['gmst_anom_pred_preind'],
            'o--', color=FACTUAL, lw=1.4, ms=4,
            label='Projection (ENSO model + trend)')

    if obs_future is not None:
        common_idx = forecast_df.index.intersection(obs_future.index)
        if len(common_idx) > 0:
            y_obs = obs_future.loc[common_idx]
            y_pred = forecast_df.loc[common_idx, 'gmst_anom_pred_preind']
            r, _ = pearsonr(y_obs, y_pred)
            rmse = np.sqrt(mean_squared_error(y_obs, y_pred))
            ax.plot(common_idx, y_obs, 's-', color=COUNTERFACT_2, ms=4,
                    label='Observed (test period)')
            ax.text(0.015, 0.96, f"R = {r:.3f}\nRMSE = {rmse:.3f} °C",
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white',
                              edgecolor='#888888', alpha=0.9))

    ax.axhline(1.5, color='#888888', linestyle=':', lw=1, label='Seuil +1.5°C (Accord de Paris)')

    if scenario_warning:
        ax.text(0.985, 0.04, scenario_warning, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8.5, style='italic',
                color=FACTUAL_DARK,
                bbox=dict(boxstyle='round', facecolor=WARNING_BG,
                          edgecolor=FACTUAL, linewidth=0.8))

    ax.set_ylabel("Temperature anomaly (C, ref. 1850-1900)", fontsize=11)
    ax.grid(True, alpha=0.25, lw=0.6)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.legend(loc='upper left', frameon=False, fontsize=9.5)

    fig.suptitle(title, fontsize=15, fontweight='bold', x=0.02, ha='left', y=0.98)
    ax.set_title(subtitle, fontsize=10, style='italic', color='#444444', loc='left', pad=10)

    fig.text(0.98, 0.005, data_sources, fontsize=7.5, color='#666666', ha='right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("gmst_forecast_academic.png", dpi=300)
    plt.show()
    plt.rcParams['font.family'] = 'sans-serif'


def load_cfsv2_forecast(path_nc, target_start=None):
    """
    Charge un fichier CPC CFSv2 'nino34Mon.nc' (monitoring/prévision ENSO).
    Structure : anom(ens, time, lev, lat, lon), ens=1..40 = runs du modèle
    (du plus récent au plus ancien), ens=41 = observation ; time couvre
    à la fois l'historique récent et la période de prévision.
    Retourne un DataFrame [date, ens_mean, ens_std] limité à la période
    de prévision (target_start, par défaut : aujourd'hui + 1 mois).
    """
    import netCDF4 as nc
    from datetime import datetime, timedelta

    ds = nc.Dataset(path_nc)
    time_units = ds.variables['time'].units  # ex : 'days since 2021-01-01, 00:00:00'
    base_str = time_units.split('since')[1].strip().split(',')[0].strip()
    base_date = datetime.strptime(base_str, '%Y-%m-%d')
    dates = [base_date + timedelta(days=float(t)) for t in ds.variables['time'][:]]

    anom = np.squeeze(ds.variables['anom'][:])  # (ens, time)
    ens_mean = anom.mean(axis=0)
    ens_std = anom.std(axis=0)

    df = pd.DataFrame({'date': dates, 'ens_mean': ens_mean, 'ens_std': ens_std})
    df['date'] = pd.to_datetime(df['date']).dt.to_period('M').dt.to_timestamp()

    if target_start is None:
        target_start = pd.Timestamp.today().normalize().replace(day=1) + pd.DateOffset(months=1)
    df = df[df['date'] >= pd.to_datetime(target_start)].reset_index(drop=True)
    return df


def build_bias_corrected_scenario(cfs_df, target_peak):
    """
    Construit un scénario ENSO à partir de la forme du forecast CFSv2 brut,
    mais rééchelonné pour atteindre un pic donné (target_peak). Utile pour
    représenter la correction de biais connue du CFSv2 (le brut surestime
    généralement l'amplitude Niño3.4 par rapport aux versions post-traitées
    du CPC/NOAA).
    """
    scale = target_peak / cfs_df['ens_mean'].max()
    return (cfs_df['ens_mean'] * scale).tolist()


def plot_enso_peak_distribution(members_weighted, target_month_label, peak_median,
                                 bin_width=0.2, bin_range=None, init_date="2026-08-01",
                                 data_sources="Data: Climate Dashboard (multi-model ONI/Nino 3.4), SINTEX-F excluded",
                                 filename="enso_distribution_peak.png"):
    """
    Histogramme pondéré (poids égal par modèle) de la distribution ONI/
    Niño 3.4 au mois de paroxysme du scénario, en classes de largeur
    `bin_width` (0.2°C par défaut). SINTEX-F déjà exclu en amont
    (load_enso_dashboard_members).

    `init_date` : date d'initialisation du scénario multi-modèle
    (Climate Dashboard), reportée dans le sous-titre descriptif de la
    figure pour qu'un lecteur puisse situer la prévision sans se reporter
    au texte principal.
    """
    vals = members_weighted['anomaly_c'].values
    wts = members_weighted['weight'].values
    n_models = members_weighted['model'].nunique() if 'model' in members_weighted.columns else None
    n_members = len(members_weighted)
    init_date = pd.Timestamp(init_date)

    if bin_range is None:
        lo = np.floor(vals.min() / bin_width) * bin_width
        hi = np.ceil(vals.max() / bin_width) * bin_width
    else:
        lo, hi = bin_range
    edges = np.arange(lo, hi + 1e-9, bin_width)

    idx = np.clip(np.digitize(vals, edges) - 1, 0, len(edges) - 2)
    pct = np.zeros(len(edges) - 1)
    for i, w in zip(idx, wts):
        pct[i] += w
    pct *= 100

    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(10, 6))
    centers = (edges[:-1] + edges[1:]) / 2
    bars = ax.bar(centers, pct, width=bin_width * 0.92, color=FACTUAL, alpha=0.85,
                   edgecolor=FACTUAL_DARKEST, linewidth=0.6)
    for c, p in zip(centers, pct):
        if p > 0.05:
            ax.text(c, p + max(pct) * 0.015, f"{p:.1f}%", ha='center', va='bottom', fontsize=7.5)

    ax.axvline(peak_median, color='#1a1a1a', ls='--', lw=1.3,
               label=f"Multi-model median: {peak_median:+.2f}C")
    ax.set_xlabel(f"Indice ONI / Niño 3.4 ({target_month_label}, °C)", fontsize=11)
    ax.set_ylabel("Share of members (%, equal weight per model)", fontsize=11)
    ax.grid(True, axis='y', alpha=0.25, lw=0.6)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.legend(loc='upper right', frameon=False, fontsize=9.5)

    fig.suptitle(f"Multi-model distribution of the ONI index at peak ({target_month_label})",
                 fontsize=14, fontweight='bold', x=0.02, ha='left', y=0.98)
    n_models_str = f"{n_models} models" if n_models is not None else "multiple models"
    subtitle = (f"Climate Dashboard multi-model ensemble initialized {init_date.day} {init_date:%B %Y} - "
                f"{n_models_str}, {n_members} members (equal weight per model), SINTEX-F excluded\n"
                f"ONI / Nino 3.4 index, 1991-2020 baseline")
    fig.text(
    0.02, 0.925, subtitle,
    fontsize=9.5,
    style='italic',
    color='#444444',
    ha='left',
    va='top'
    )

    fig.text(
        0.02, 0.845,
        f"Bin width {bin_width:.1f}\u00b0C",
            fontsize=8.5,
            color='#666666',
            ha='left',
            va='top'
            )

    fig.text(0.98, 0.005, data_sources, fontsize=7.5, color='#666666', ha='right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.87])
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.rcParams['font.family'] = 'sans-serif'
    return pd.DataFrame({'bin_low': edges[:-1], 'bin_high': edges[1:], 'pct': pct})


def plot_multi_scenario(df, scenarios,                         data_sources="Data: ECMWF/Copernicus C3S - ERA5 (ref. 1850-1900); ENSO: NCEP CFSv2 (CPC/NOAA)",
                         title="Global mean surface temperature anomaly - several ENSO hypotheses",
                         subtitle="ERA5 observations and statistical projections (quadratic trend + Nino 3.4 ENSO)",
                         scenario_warning=None):
    """
    Affiche l'historique + plusieurs scénarios de projection en courbes
    continues (pas de marqueurs) sur le même graphique.

    scenarios : dict {label: (forecast_df, couleur)}
    """
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(12, 6.5))

    ax.plot(df.index, df['gmst_anom_preind'], color='#1a1a1a', lw=1.1,
            label='Observed (ERA5, ref. 1850-1900)')

    for label, (fc, color) in scenarios.items():
        ax.plot(fc.index, fc['gmst_anom_pred_preind'], '-', color=color, lw=2,
                label=label)

    ax.axhline(1.5, color='#888888', linestyle=':', lw=1, label='Seuil +1.5°C (Accord de Paris)')

    if scenario_warning:
        ax.text(0.985, 0.03, scenario_warning, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8.5, style='italic',
                color=FACTUAL_DARK,
                bbox=dict(boxstyle='round', facecolor=WARNING_BG,
                          edgecolor=FACTUAL, linewidth=0.8))

    ax.set_ylabel("Temperature anomaly (C, ref. 1850-1900)", fontsize=11)
    ax.grid(True, alpha=0.25, lw=0.6)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.legend(loc='upper left', frameon=False, fontsize=9.5)

    fig.suptitle(title, fontsize=15, fontweight='bold', x=0.02, ha='left', y=0.98)
    ax.set_title(subtitle, fontsize=10, style='italic', color='#444444', loc='left', pad=10)

    fig.text(0.98, 0.005, data_sources, fontsize=7.5, color='#666666', ha='right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("gmst_forecast_multiscenario.png", dpi=300)
    plt.show()
    plt.rcParams['font.family'] = 'sans-serif'


# ----------------------------------------------------------------------
# 5bis. GRAPHIQUE AVEC ZOOM TEMPOREL + JONCTION SANS SAUT
#       (vérif. seule, prévision seule, ou mix vérif/prévision)
# ----------------------------------------------------------------------

import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

_MONTH_ABBR = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
               7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
_MONTH_LONG = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May',
               6: 'June', 7: 'July', 8: 'August', 9: 'September',
               10: 'October', 11: 'November', 12: 'December'}


def _month_axis_formatter(x, pos=None):
    d = mdates.num2date(x)
    return f"{_MONTH_ABBR[d.month]} {d.year}"


def _month_label(ts):
    return f"{_MONTH_LONG[ts.month]} {ts.year}"


def plot_forecast_zoom(df, forecast_df=None, obs_future=None, zoom=None,
                        model_fit=None, forecast_std=None,
                        residuals=None, percentile_bands=((5, 95), (25, 75)),
                        forecast_df_alt=None, label_alt='Alternative scenario',
                        filename="gmst_forecast_zoom.png",
                        data_sources="Data: ECMWF/Copernicus C3S - ERA5 (1850-1900 baseline)",
                        title="Global mean surface temperature anomaly",
                        subtitle=None,
                        thresholds=(1.5, 2.0),
                        legend_loc='lower left',
                        scenario_warning=None):
    """
    Graphique combiné verification / prévision, avec zoom temporel optionnel
    et jonction sans saut entre les courbes.

    Paramètres
    ----------
    df : DataFrame historique observé, colonne 'gmst_anom_preind' (obligatoire)
    forecast_df : DataFrame de projection (colonne 'gmst_anom_pred_preind'),
                  ou None si on ne veut que la verification
    obs_future : Series des observations réelles couvrant tout ou partie de
                 la période de forecast_df (mode "verification"). Si fournie
                 en même temps que forecast_df, les deux courbes sont
                 superposées sur le même graphique (mode "mix").
    model_fit : Series optionnelle des valeurs du modèle AJUSTÉ sur
                l'historique (model.predict(X), même index que df).
                Superpose "model vs observed" sur les mois passés pour une
                verification directe (indépendante de la prévision future).
    forecast_std : écart-type (float, ex. le RMSE du modèle) pour tracer une
                   enveloppe d'incertitude ombrée simple (±1 écart-type)
                   autour de forecast_df. Ignoré si `residuals` est fourni
                   (les bandes de percentiles empiriques sont préférées,
                   plus honnêtes qu'une hypothèse gaussienne symétrique).
    residuals : array des résidus walk-forward réels (obs - prédit). Si
                fourni, remplace l'enveloppe simple par des bandes de
                percentiles empiriques (voir percentile_bands) -- montre
                les deux extrêmes réels de la prévision, pas juste ±1σ.
    percentile_bands : tuple de paires (p_bas, p_haut) à tracer en bandes
                        emboîtées si `residuals` est fourni. Par défaut
                        (5,95) et (25,75).
    forecast_df_alt : DataFrame optionnel d'un second scénario (ex. version
                      révisée) à superposer en trait fin pour comparaison
                      directe avec forecast_df (label_alt : sa légende).
    zoom : tuple (date_debut, date_fin) ex. ("2022-01-01", "2027-03-01").
           Si fourni, l'axe x est zoomé sur cette période avec un repère
           mensuel explicite (mois par mois).
    thresholds : seuils horizontaux à tracer, par défaut +1.5°C et +2°C.
    legend_loc : position de la légende (par défaut 'lower left'). Valeur
                 spéciale 'between_thresholds' : cale la légende à gauche,
                 verticalement centrée entre les deux seuils (utile quand
                 les seuils +1.5/+2°C sont hauts dans le cadrage du zoom).
    scenario_warning : texte d'avertissement optionnel (ex. scénario ENSO
                       hors de la plage historique -> extrapolation).

    Jonction sans saut
    ------------------
    Le dernier point observé de `df` est répété comme premier point de
    chaque courbe de projection/verification, ce qui relie visuellement
    l'historique et la suite (plus de "branche vide" entre les deux).
    """
    plt.rcParams['font.family'] = 'serif'

    # largeur adaptative si zoom mensuel sur une longue période (lisibilité)
    fig_width = 12
    if zoom is not None:
        n_months_zoom = (pd.to_datetime(zoom[1]).year - pd.to_datetime(zoom[0]).year) * 12 + \
                         (pd.to_datetime(zoom[1]).month - pd.to_datetime(zoom[0]).month)
        fig_width = max(12, min(24, n_months_zoom * 0.28))
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))

    ax.plot(df.index, df['gmst_anom_preind'], color='#1a1a1a', lw=1.3,
            label='Observed (ERA5, 1850-1900 baseline)')

    # -- Vérification modèle vs obs sur l'historique (indépendant du futur) --
    if model_fit is not None:
        ax.plot(model_fit.index, model_fit.values, color=HIGHLIGHT, lw=1.2,
                linestyle='-', alpha=0.85,
                label='Fitted model (historical verification)')
        common_hist = df.index.intersection(model_fit.index)
        if len(common_hist) > 0:
            r_hist, _ = pearsonr(df.loc[common_hist, 'gmst_anom_preind'], model_fit.loc[common_hist])
            rmse_hist = np.sqrt(mean_squared_error(df.loc[common_hist, 'gmst_anom_preind'], model_fit.loc[common_hist]))
            ax.text(0.015, 0.97, f"Historical verification: R = {r_hist:.3f} | RMSE = {rmse_hist:.3f} C",
                    transform=ax.transAxes, va='top', fontsize=8.5,
                    bbox=dict(boxstyle='round', facecolor='white',
                              edgecolor=HIGHLIGHT, alpha=0.9))

    last_obs_date = df.index[-1]
    last_obs_val = df['gmst_anom_preind'].iloc[-1]

    warmest_date, warmest_val = None, -np.inf

    # -- Courbe de prévision, jointe au dernier point observé --
    if forecast_df is not None:
        fc_dates = [last_obs_date] + list(forecast_df.index)
        fc_vals = [last_obs_val] + list(forecast_df['gmst_anom_pred_preind'])

        if residuals is not None:
            # Bandes de percentiles empiriques (résidus walk-forward réels)
            # -- montre les DEUX EXTRÊMES réels, pas une hypothèse gaussienne.
            # Part à largeur nulle sur le dernier point observé (pas de saut).
            n_bands = len(percentile_bands)
            band_colors = [overview_band_color(i, n_bands) for i in range(n_bands)]
            for (p_low, p_high), color in zip(percentile_bands, band_colors):
                low = [last_obs_val] + [np.percentile(v + residuals, p_low)
                                         for v in forecast_df['gmst_anom_pred_preind']]
                high = [last_obs_val] + [np.percentile(v + residuals, p_high)
                                          for v in forecast_df['gmst_anom_pred_preind']]
                ax.fill_between(fc_dates, low, high, color=color, alpha=1.0, lw=0,
                                 label=f'Band {p_low}-{p_high}% (empirical extremes)')
        elif forecast_std is not None:
            # l'enveloppe part à largeur nulle sur le dernier point observé
            # (jonction sans saut), puis s'élargit à ±forecast_std
            std_arr = np.array([0.0] + [forecast_std] * len(forecast_df))
            fc_vals_arr = np.array(fc_vals)
            ax.fill_between(fc_dates, fc_vals_arr - std_arr, fc_vals_arr + std_arr,
                             color=FACTUAL, alpha=0.15, lw=0,
                             label=f'Envelope +/-1 std dev ({forecast_std:.2f} C)')

        ax.plot(fc_dates, fc_vals, 'o--', color=FACTUAL, lw=1.7, ms=4.5,
                label='Projection (ENSO scenario)',
                path_effects=[pe.withStroke(linewidth=3.2, foreground='white')] if residuals is not None else None)
        i_max = int(np.argmax(forecast_df['gmst_anom_pred_preind'].values))
        if forecast_df['gmst_anom_pred_preind'].iloc[i_max] > warmest_val:
            warmest_val = forecast_df['gmst_anom_pred_preind'].iloc[i_max]
            warmest_date = forecast_df.index[i_max]

    # -- Second scénario optionnel (ex. version révisée), pour comparaison --
    if forecast_df_alt is not None:
        fc_alt_dates = [last_obs_date] + list(forecast_df_alt.index)
        fc_alt_vals = [last_obs_val] + list(forecast_df_alt['gmst_anom_pred_preind'])
        ax.plot(fc_alt_dates, fc_alt_vals, '--', color=FACTUAL_DARKEST, lw=1.3, alpha=0.6,
                label=label_alt)
        i_max_alt = int(np.argmax(forecast_df_alt['gmst_anom_pred_preind'].values))
        if forecast_df_alt['gmst_anom_pred_preind'].iloc[i_max_alt] > warmest_val:
            warmest_val = forecast_df_alt['gmst_anom_pred_preind'].iloc[i_max_alt]
            warmest_date = forecast_df_alt.index[i_max_alt]

    # -- Courbe d'observations réelles sur la période de projection --
    #    (mode verification seule, ou mix vérif/prévision sur le même graph)
    if obs_future is not None and len(obs_future) > 0:
        of_dates = [last_obs_date] + list(obs_future.index)
        of_vals = [last_obs_val] + list(obs_future.values)
        ax.plot(of_dates, of_vals, 's-', color=COUNTERFACT_2, lw=1.6, ms=4.5,
                label='Observed (verification)')
        i_max = int(np.argmax(obs_future.values))
        if obs_future.values[i_max] > warmest_val:
            warmest_val = obs_future.values[i_max]
            warmest_date = obs_future.index[i_max]

        if forecast_df is not None:
            common_idx = forecast_df.index.intersection(obs_future.index)
            if len(common_idx) > 0:
                y_obs = obs_future.loc[common_idx]
                y_pred = forecast_df.loc[common_idx, 'gmst_anom_pred_preind']
                r, _ = pearsonr(y_obs, y_pred)
                rmse = np.sqrt(mean_squared_error(y_obs, y_pred))
                ax.text(0.015, 0.97 if model_fit is None else 0.90,
                        f"Forecast vs obs: R = {r:.3f} | RMSE = {rmse:.3f} C",
                        transform=ax.transAxes, va='top', fontsize=8.5,
                        bbox=dict(boxstyle='round', facecolor='white',
                                  edgecolor='#888888', alpha=0.9))

    # -- Mois le plus chaud (placé AU-DESSUS de l'enveloppe d'incertitude,
    #    pas juste au-dessus du point, pour ne pas gêner la lecture) --
    annotation_top_y = None
    if warmest_date is not None:
        if residuals is not None:
            p_high_max = max(p[1] for p in percentile_bands)
            env_top = np.percentile(warmest_val + residuals, p_high_max)
        else:
            env_top = warmest_val + (forecast_std if forecast_std is not None else 0.05)
        text_y = env_top + 0.14
        annotation_top_y = text_y + 0.08
        ax.annotate(
            f"Warmest month: {_month_label(warmest_date)}\n"
            f"({warmest_val:+.2f} °C)",
            xy=(warmest_date, warmest_val),
            xytext=(warmest_date, text_y), textcoords='data',
            ha='center', va='bottom', fontsize=9, fontweight='bold', color=FACTUAL_DARK,
            arrowprops=dict(arrowstyle='->', color=FACTUAL_DARK, lw=1.2),
            bbox=dict(boxstyle='round', facecolor=WARNING_BG,
                          edgecolor=FACTUAL, linewidth=0.8))

    # -- Seuils --
    threshold_styles = {1.5: (':', '#888888', '+1.5 C threshold (Paris Agreement)'),
                         2.0: ('--', '#555555', 'Seuil +2 °C')}
    for th in thresholds:
        ls, col, lab = threshold_styles.get(th, ('-.', '#999999', f'Seuil +{th} °C'))
        ax.axhline(th, color=col, linestyle=ls, lw=1.1, label=lab)

    if scenario_warning:
        ax.text(0.985, 0.03, scenario_warning, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8.5, style='italic',
                color=FACTUAL_DARK,
                bbox=dict(boxstyle='round', facecolor=WARNING_BG,
                          edgecolor=FACTUAL, linewidth=0.8))

    # -- Zoom + repère mensuel explicite --
    if zoom is not None:
        z0, z1 = pd.to_datetime(zoom[0]), pd.to_datetime(zoom[1])
        ax.set_xlim(z0, z1)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(FuncFormatter(_month_axis_formatter))
        tick_fontsize = 8 if n_months_zoom <= 24 else 7
        plt.setp(ax.get_xticklabels(), rotation=90, ha='center', fontsize=tick_fontsize)
        # ré-échelonne l'axe y sur la fenêtre zoomée uniquement
        mask_hist = (df.index >= z0) & (df.index <= z1)
        y_vals = list(df.loc[mask_hist, 'gmst_anom_preind'].values)
        if forecast_df is not None:
            mask_fc = (forecast_df.index >= z0) & (forecast_df.index <= z1)
            fc_y = forecast_df.loc[mask_fc, 'gmst_anom_pred_preind'].values
            y_vals += list(fc_y)
            if residuals is not None and len(fc_y) > 0:
                p_low_min = min(p[0] for p in percentile_bands)
                p_high_max = max(p[1] for p in percentile_bands)
                y_vals += [np.percentile(v + residuals, p_low_min) for v in fc_y]
                y_vals += [np.percentile(v + residuals, p_high_max) for v in fc_y]
            elif forecast_std is not None and len(fc_y) > 0:
                y_vals += list(fc_y + forecast_std) + list(fc_y - forecast_std)
        if forecast_df_alt is not None:
            mask_alt = (forecast_df_alt.index >= z0) & (forecast_df_alt.index <= z1)
            y_vals += list(forecast_df_alt.loc[mask_alt, 'gmst_anom_pred_preind'].values)
        if annotation_top_y is not None:
            y_vals.append(annotation_top_y)
        if obs_future is not None and len(obs_future) > 0:
            mask_of = (obs_future.index >= z0) & (obs_future.index <= z1)
            y_vals += list(obs_future.loc[mask_of].values)
        if y_vals:
            ymin, ymax = min(y_vals + [min(thresholds) - 0.1]), max(y_vals + [max(thresholds) + 0.1])
            pad = 0.08 * (ymax - ymin if ymax > ymin else 1)
            ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_ylabel("Temperature anomaly (C) [1850-1900 baseline]", fontsize=11)
    ax.grid(True, alpha=0.25, lw=0.6)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    if legend_loc == 'between_thresholds' and len(thresholds) >= 2:
        y0, y1 = ax.get_ylim()
        y_mid = (min(thresholds) + max(thresholds)) / 2
        y_frac = (y_mid - y0) / (y1 - y0)
        ax.legend(loc='center left', bbox_to_anchor=(0.012, y_frac),
                  frameon=True, framealpha=0.9, fontsize=9)
    else:
        ax.legend(loc=legend_loc, frameon=True, framealpha=0.9, fontsize=9)

    fig.suptitle(title, fontsize=15, fontweight='bold', x=0.02, ha='left', y=0.98)
    if subtitle:
        ax.set_title(subtitle, fontsize=10, style='italic', color='#444444',
                     loc='left', pad=10)

    fig.text(0.98, 0.005, data_sources, fontsize=7.5, color='#666666', ha='right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.rcParams['font.family'] = 'sans-serif'


def plot_forecast_overview_media(df, forecast_df, residuals=None,
                                  percentile_bands=((5, 95), (25, 75)),
                                  thresholds=(1.5, 2.0),
                                  filename="gmst_forecast_overview_media.png",
                                  data_sources="Data: ECMWF/Copernicus C3S - ERA5 (1850-1900 baseline)",
                                  title="Global mean surface temperature anomaly",
                                  subtitle="Departure from preindustrial (1850-1900) - C",
                                  scenario_warning=None):
    """
    Version "media" de plot_forecast_zoom(), pensée pour la reprise dans la
    presse ou sur les réseaux -- MÊMES DONNÉES, MÊME NIVEAU D'INFORMATION,
    mais mise en page épurée façon dataviz éditoriale (Reuters Graphics,
    Our World in Data, The Climate Brink...) : pas de cadre de légende à
    décoder, étiquettes posées directement sur les courbes, hiérarchie
    typographique nette titre/sous-titre/source, grille minimale.

    Différence assumée avec la référence qui a inspiré ce style : la bande
    d'incertitude ne court PAS sur toute la période historique. Ici,
    l'historique est une observation directe (ERA5), pas un ensemble de
    modèles -- lui ajouter une bande grise imiterait une incertitude qui
    n'existe pas dans les données. Elle n'apparaît donc que sur la partie
    projetée, où elle correspond aux bandes de percentiles empiriques
    (résidus walk-forward réels du modèle TESR).

    Paramètres : mêmes conventions que plot_forecast_zoom() (voir sa
    docstring) pour df / forecast_df / residuals / percentile_bands /
    thresholds / scenario_warning -- réutilisable tel quel avec les mêmes
    objets déjà calculés dans le pipeline.
    """
    plt.rcParams['font.family'] = 'sans-serif'
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(13, 7.2))

    last_obs_date = df.index[-1]
    last_obs_val = df['gmst_anom_preind'].iloc[-1]

    # -- Historique observé -- trait fin gris anthracite, pas noir pur
    #    (moins agressif visuellement, laisse le rouge de la projection
    #    porter l'attention) --
    ax.plot(df.index, df['gmst_anom_preind'], color='#4d4d4d', lw=1.1, zorder=3)

    # -- Bande d'incertitude sur la projection uniquement (cf. docstring) --
    fc_dates = [last_obs_date] + list(forecast_df.index)
    fc_vals = [last_obs_val] + list(forecast_df['gmst_anom_pred_preind'])
    n_bands = len(percentile_bands)
    band_colors = [overview_band_color(i, n_bands) for i in range(n_bands)]
    if residuals is not None:
        for (p_low, p_high), color in zip(percentile_bands, band_colors):
            low = [last_obs_val] + [np.percentile(v + residuals, p_low)
                                     for v in forecast_df['gmst_anom_pred_preind']]
            high = [last_obs_val] + [np.percentile(v + residuals, p_high)
                                      for v in forecast_df['gmst_anom_pred_preind']]
            ax.fill_between(fc_dates, low, high, color=color, alpha=1.0, lw=0, zorder=2)

    # -- Projection -- rouge brique, cohérent avec le reste des figures --
    #    halo blanc pour rester lisible par-dessus la bande turbo rouge --
    ax.plot(fc_dates, fc_vals, color=FACTUAL, lw=2.2, zorder=4,
            path_effects=[pe.withStroke(linewidth=4.0, foreground='white')])
    ax.scatter([fc_dates[-1]], [fc_vals[-1]], color=FACTUAL, s=22, zorder=5)

    # -- Record historique observé (mois le plus chaud AVANT la projection) --
    #    ligne de référence tiret-point, distincte des seuils (pointillés) et
    #    de la projection (plein rouge vif) --
    record_date = df['gmst_anom_preind'].idxmax()
    record_val = df['gmst_anom_preind'].max()
    ax.axhline(record_val, color=HIGHLIGHT, linestyle='--', lw=1.0, zorder=1, alpha=0.75)
    ax.scatter([record_date], [record_val], color=HIGHLIGHT, s=28, zorder=5,
               edgecolor='white', linewidth=0.6)

    # -- Limites Y calculées AVANT le placement des textes, pour pouvoir
    #    convertir des coordonnées données en fraction d'axes (légende,
    #    étiquettes) de façon fiable --
    y_top = max(fc_vals + [max(thresholds), record_val])
    y_bot = min(df['gmst_anom_preind'].min(), min(fc_vals))
    span = y_top - y_bot
    ylim_bot, ylim_top = y_bot - 0.08 * span, y_top + 0.16 * span
    ax.set_ylim(ylim_bot, ylim_top)

    def _to_frac(y_data):
        return (y_data - ylim_bot) / (ylim_top - ylim_bot)

    # -- Seuils, étiquetés à GAUCHE de la ligne (comme la référence media) --
    #    plutôt qu'à droite : la zone de droite est déjà dense (fin de
    #    l'historique + toute la projection compressée sur ~1% de l'axe
    #    des dates, cf. note plus bas) --
    threshold_labels = {1.5: '+1.5 C threshold (Paris Agreement)', 2.0: 'Seuil +2 °C'}
    x_left = df.index[int(len(df) * 0.012)]
    for th in thresholds:
        ax.axhline(th, color='#999999', linestyle=(0, (1, 2)), lw=1.0, zorder=1)
        ax.text(x_left, th + 0.025, threshold_labels.get(th, f'Seuil +{th} °C'),
                ha='left', va='bottom', fontsize=9.5, color='#777777')
    ax.text(x_left, record_val + 0.025,
            f"Observed record: {_month_label(record_date)} ({record_val:+.2f} C)",
            ha='left', va='bottom', fontsize=9.5, color=HIGHLIGHT)

    # -- Légende compacte (remplace les étiquettes flottantes qui rentraient
    #    dans les données) -- ancrée juste sous la ligne de seuil +1,5°C,
    #    à gauche, comme demandé --
    legend_handles = [
        Line2D([0], [0], color='#4d4d4d', lw=1.6, label='Observed (ERA5)'),
        Line2D([0], [0], color=FACTUAL, lw=2.2, label='Projection (ENSO scenario)'),
        Line2D([0], [0], color=HIGHLIGHT, lw=1.0, ls='--', label='Record observé avant projection'),
    ]
    if residuals is not None and percentile_bands:
        p_low0, p_high0 = percentile_bands[0]
        legend_handles.append(
            Line2D([0], [0], color=overview_band_color(0, len(percentile_bands)), lw=7, alpha=1.0,
                   label=f'Uncertainty {p_low0}-{p_high0}% (model residuals)'))
    ax.legend(handles=legend_handles, loc='upper left',
              bbox_to_anchor=(0.012, _to_frac(1.5) - 0.03), fontsize=9.3,
              frameon=True, framealpha=0.92, edgecolor='#e2e2e2', facecolor='white',
              handlelength=1.8, borderpad=0.6, labelspacing=0.55)

    # -- NOTE SUR LA COMPRESSION VISUELLE : la projection ne représente
    #    qu'1 an sur les ~87 ans de l'axe (~1% de la largeur du graphique).
    #    L'étiquette du pic est donc ancrée en HAUT du graphique, en
    #    coordonnées de FIGURE (indépendantes de cette compression), bien
    #    au-dessus de la courbe la plus haute -- jamais dans les données --
    #    et reliée par une fine flèche au point réel. --
    warmest_date = forecast_df['gmst_anom_pred_preind'].idxmax()
    warmest_val = forecast_df['gmst_anom_pred_preind'].max()
    ax.scatter([warmest_date], [warmest_val], color=FACTUAL, s=45, zorder=6,
               edgecolor='white', linewidth=0.8)
    ax.annotate(
        f"Projection - {_month_label(warmest_date)}\n{warmest_val:+.2f} C, warmest month of the scenario",
        xy=(warmest_date, warmest_val), xycoords='data',
        xytext=(0.60, 0.97), textcoords='axes fraction',
        ha='left', va='top', fontsize=10.5, fontweight='bold', color=FACTUAL, linespacing=1.4,
        arrowprops=dict(arrowstyle='-', color=FACTUAL, lw=1.0, shrinkA=0, shrinkB=6))

    # -- Avertissement scénario -- encadré léger, en bas à droite du tracé,
    #    même emplacement que la version précédente --
    if scenario_warning:
        ax.text(0.985, 0.025, scenario_warning, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8.5, style='italic', color=FACTUAL_DARK,
                bbox=dict(boxstyle='round,pad=0.35', facecolor=WARNING_BG,
                          edgecolor=FACTUAL, linewidth=0.7))

    # -- Grille horizontale seule, minimale ; pas de cadre --
    ax.grid(True, axis='y', alpha=0.18, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.tick_params(axis='both', length=0, labelsize=10.5, colors='#333333')
    # -- Pas de titre d'axe X/Y : les dates sont lisibles directement sur les
    #    graduations, et l'unité (°C, réf. préindustrielle) est déjà donnée
    #    dans le sous-titre -- cohérent avec la référence media qui s'en
    #    passe aussi. Réactiver si le graphique circule sans son sous-titre. --
    ax.set_ylabel("")
    ax.set_xlabel("")

    fig.suptitle(title, x=0.045, y=0.985, ha='left', fontsize=19, fontweight='bold', color='#1a1a1a')
    fig.text(0.045, 0.925, subtitle, fontsize=12.5, color='#666666', ha='left')

    fig.text(0.98, 0.01, data_sources, fontsize=8, color='#888888', ha='right')

    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.045, right=0.97)
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.rcParams['font.family'] = 'sans-serif'


def _analog_enso_series(nino_hist_df, start_year, target_dec_year, months=16):
    """
    Extrait, mois par mois, un épisode El Niño historique de référence
    de décembre `start_year` à mars `start_year+1` (4 mois), et le
    repositionne sur l'axe du graphique en cours en l'alignant sur
    décembre `target_dec_year` (même position calendaire mois-à-mois,
    années différentes -- comparaison de phase du cycle, pas une
    superposition de dates réelles).
    """
    src_dates = pd.date_range(f"{start_year}-01-01", periods=months, freq='MS')
    tgt_dates = pd.date_range(f"{target_dec_year}-01-01", periods=months, freq='MS')
    vals = nino_hist_df['enso_ssta'].reindex(src_dates)
    return pd.Series(vals.values, index=tgt_dates)


def plot_enso_scenario_envelope(enveloppe_enso, enso_hist_df=None, obs_junction=None,
                                 hist_months=6, end_date=None,
                                 data_sources="Data: The Climate Brink @ ENSO Dashboard (multi-model ONI/Nino 3.4); analogues ERA5",
                                 title="ONI | Nino 3.4 index - multi model synthesis through April 2027",
                                 subtitle="Nested multi-model percentile envelopes (equal weight per model)\nModels considered: BOM, CFSv2, CMCC, CanSIPS-CanESM5, CanSIPS-GEM-NEMO, DWD, ECMWF, JMA, Meteo-France,\nNASA-GEOS-S2S-2, NCAR-CCSM4, NCAR-CESM1, UKMO | Initialized 1 August 2026",
                                 filename="enso_scenario_envelope.png",
                                 show_thresholds=True,
                                 analog_hist_df=None,
                                 analog_events=(("1982/1983", 1982, ANALOG_COLORS[0]),
                                                ("1997/1998", 1997, ANALOG_COLORS[1]),
                                                ("2015/2016", 2015, ANALOG_COLORS[2])),
                                 legend_loc='upper left'):
    """
    Trace le scénario ENSO retenu (mediane multi-modèles, telle que
    produite par load_enso_dashboard_scenario) avec ses enveloppes de
    percentiles emboîtées :
        - Q0-Q100  : min/max bruts du pool de membres (SINTEX-F exclu)
        - Q5-Q95   : bande p05-p95 pondérée par modèle
        - Q25-Q75  : bande p25-p75 pondérée par modèle
        - Médiane  : courbe centrale pondérée par modèle
    Style cohérent avec plot_fan_chart / plot_forecast_zoom (mêmes
    couleurs, mise en page, police).

    Paramètres
    ----------
    enveloppe_enso : DataFrame retourné par load_enso_dashboard_scenario
        (colonnes 'median', 'p05', 'p25', 'p75', 'p95', 'q0', 'q100',
        indexé par mois).
    enso_hist_df : DataFrame ENSO observé historique optionnel (colonne
        'enso_ssta', ex. enso_df_raw), pour afficher les derniers mois
        observés avant le scénario. Optionnel -- omis si None.
    obs_junction : tuple optionnel (date, valeur) -- point de jonction
        observé déterministe (ex. dernier mois ONI observé, sans bande
        d'incertitude) inséré entre l'historique et le scénario multi-
        modèles, pour éviter un saut visuel. Optionnel.
    hist_months : nombre de mois d'historique affichés avant la jonction
        (si enso_hist_df est fourni).
    end_date : borne temporelle finale du graphique, incluse (ex.
        "2027-03-01" pour arrêter le scénario à mars 2027 même si
        enveloppe_enso va au-delà). None = jusqu'au dernier mois
        disponible dans enveloppe_enso.
    """
    env = enveloppe_enso.copy()
    if end_date is not None:
        env = env.loc[:pd.to_datetime(end_date)]

    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(12, 6.5))

    # -- Historique observé récent + point(s) de jonction déterministe(s) --
    # IMPORTANT : l'historique et la jonction sont tracés comme UNE SEULE
    # ligne continue (et non deux artistes séparés comme dans la version
    # précédente). Avant ce correctif, le point de jonction (ex. juillet,
    # seule valeur "observed" fournie par la source du scénario) était
    # dessiné en marqueur isolé, sans segment le reliant au dernier point
    # historique -- d'où le "trou" visuel dans le mois qui les sépare
    # (et un trou plus large encore si ce mois-là est lui-même absent du
    # fichier CSV d'historique, ce qui est le cas ici : la série ERA5
    # utilisée pour l'historique stops un mois avant la jonction).
    hist = pd.Series(dtype=float)
    if enso_hist_df is not None:
        hist = enso_hist_df['enso_ssta'].dropna()
        hist = hist.loc[:env.index[0]].iloc[-hist_months:]

    junction_dates, junction_vals = [], []
    if obs_junction is not None:
        jdate, jval = obs_junction
        jdate = pd.to_datetime(jdate)
        junction_dates, junction_vals = [jdate], [jval]

    hist_dates_all = list(hist.index) + junction_dates
    hist_vals_all = list(hist.values) + junction_vals
    if hist_dates_all:
        ax.plot(hist_dates_all, hist_vals_all, 'o-', color='#1a1a1a', lw=1.2, ms=3.5,
                label='Observed (Nino 3.4 / ONI)', zorder=5)

    fc_dates = junction_dates + list(env.index)

    def _series(col):
        return np.array(junction_vals + list(env[col]))

    # -- Enveloppes emboîtées, de la plus large (claire) à la plus étroite
    #    (foncée) -- QUANTILE_BANDS est une vraie rampe teinte+luminosité
    #    (bleu pâle -> bleu saturé), pas la même teinte orange à trois
    #    niveaux d'alpha (les trois bandes étaient auparavant quasi
    #    indiscernables une fois superposées) --
    ax.fill_between(fc_dates, _series('q0'), _series('q100'), color=QUANTILE_BANDS[0],
                     alpha=1.0, lw=0, label='Q0-Q100 (raw extremes)')
    ax.fill_between(fc_dates, _series('p05'), _series('p95'), color=QUANTILE_BANDS[1],
                     alpha=1.0, lw=0, label='Q5-Q95')
    ax.fill_between(fc_dates, _series('p25'), _series('p75'), color=QUANTILE_BANDS[2],
                     alpha=1.0, lw=0, label='Q25-Q75')

    ax.plot(fc_dates, _series('median'), 'o-', color=FACTUAL, lw=2, ms=4,
            label='Multi-model median')

    # -- Analogues historiques : les 3 El Niño les plus forts précédents,
    #    alignés phase à phase (déc. N-1 -> mars N+1) sur le même axe --
    if analog_hist_df is not None and analog_events:
        dec_years = sorted({d.year for d in env.index if d.month == 12})
        if dec_years:
            target_dec_year = dec_years[0]
            for label, start_year, color in analog_events:
                serie = _analog_enso_series(analog_hist_df, start_year, target_dec_year)
                serie = serie.dropna()
                if serie.empty:
                    continue
                ax.plot(serie.index, serie.values, 'o--', color=color, lw=1.6, ms=3.5,
                        alpha=0.85, zorder=4, label=f'Analogue {label}')

    # -- Seuils ENSO / catégories d'intensité (étiquettes au bord droit,
    #    pas dans la légende, pour ne pas la surcharger) --
    if show_thresholds:
        thresholds = [
            (-0.5, ENSO_SCALE[-0.5], ':',  'La Niña (-0.5)'),
            (0.5,  ENSO_SCALE[0.5], ':',  'El Niño (+0.5)'),
            (1.0,  ENSO_SCALE[1.0], '--', 'Moderate El Nino (+1)'),
            (1.5,  ENSO_SCALE[1.5], '--', 'El Niño fort (+1.5)'),
            (2.0,  ENSO_SCALE[2.0], '--', 'Very strong El Nino (+2)'),
            (3.0,  ENSO_SCALE[3.0], '--', 'Extreme El Nino (+3)'),
        ]
        for y, color, ls, label in thresholds:
            ax.axhline(y, color=color, ls=ls, lw=1.3, zorder=1, alpha=0.9)
            ax.text(1.002, y, label, transform=ax.get_yaxis_transform(),
                    fontsize=7.5, color=color, va='center', ha='left',
                    fontweight='bold', clip_on=False)
        ax.axhline(0, color='#bbbbbb', ls='-', lw=0.7, zorder=1)

    ax.set_ylabel("ONI | Nino 3.4 index (C) [1991-2020 baseline]", fontsize=11)
    ax.grid(True, alpha=0.25, lw=0.6)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(FuncFormatter(_month_axis_formatter))
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center', fontsize=7.5)

    # Légende en haut à gauche : les valeurs les plus hautes du scénario
    # (pic de l'épisode El Niño) sont concentrées en fin de série (à
    # droite), donc le coin supérieur gauche reste dégagé.
    ax.legend(loc=legend_loc, frameon=True, framealpha=0.9, fontsize=8, ncol=1)

    fig.suptitle(title, fontsize=15, fontweight='bold', x=0.02, ha='left', y=0.98)
    fig.text(0.02, 0.85, subtitle, fontsize=10, style='italic', color='#444444', ha='left')
    fig.text(0.91, 0.005, data_sources, fontsize=7.5, color='#666666', ha='right')

    # Marge à droite pour les étiquettes de seuils hors zone de tracé
    plt.tight_layout(rect=[0, 0.03, 0.93, 0.88])
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.rcParams['font.family'] = 'sans-serif'


# ----------------------------------------------------------------------
# 6. EXAMPLE USAGE
# ----------------------------------------------------------------------

# ========================================================================
# EXTENSION -- projection étendue à Décembre 2027 (3 hypothèses ENSO H2 2027)
# Intégrée directement ici (auparavant plot_extension_dec2027.py, un fichier
# séparé jamais appelé depuis __main__) pour que ces figures soient générées
# automatiquement à l'exécution du script, sans étape manuelle -- cf. l'appel
# dans le bloc principal, juste après plot_enso_peak_distribution().
# ========================================================================

# ====================================================================
# 0. HYPOTHÈSES ONI H2 2027 -- CSV embarqué
# ====================================================================
#
# Remplace la dépendance manquante à build_h2_2027_scenarios() / au
# fichier enso_gmst_model_extension_2027.py (jamais fourni), pour que ce
# script soit AUTONOME. Trois trajectoires ONI mensuelles, Mai-Décembre
# 2027, construites à partir de la description narrative de l'auteur :
#
#   - neutre         : la décroissance ralentit et se stabilise près de
#                       0°C dès septembre 2027 (transition la plus douce
#                       observée, 2016/2017, qui n'a produit qu'une
#                       faible La Niña) ;
#   - central         : décroissance régulière vers une La Niña
#                       faible à modérée en fin d'année (-0.6 à -0.7°C
#                       en décembre 2027) ;
#   - la_nina_forte   : décroissance rapide franchissant le seuil
#                       conventionnel de La Niña forte (ONI <= -1.5°C)
#                       dès octobre-novembre 2027, sans dépasser les
#                       valeurs les plus extrêmes enregistrées sur
#                       1950-2026 (La Niña 1973/74, ONI ~ -2.1°C).
#
# AVERTISSEMENT : valeurs ILLUSTRATIVES construites pour respecter cette
# trajectoire qualitative (forme + points de passage décrits), PAS des
# sorties officielles d'un modèle ENSO -- à ajuster/recaler dès que la
# vraie valeur ONI d'avril 2027 (dernier mois du scénario officiel) est
# connue. `build_h2_2027_scenarios(anchor_offset=...)` permet de décaler
# les 3 courbes en bloc pour les raccorder à ce point si besoin.
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
    """Construit `h2_scenarios` (dict {'neutre':..., 'central':...,
    'la_nina_forte':...}, chacun un dict {date_str: valeur ONI}) à partir
    du CSV `_H2_2027_ONI_CSV` embarqué ci-dessus -- voir l'avertissement
    en tête de section sur la nature illustrative de ces valeurs.

    `anchor_offset` : décalage additif (°C) appliqué aux 3 courbes en
    bloc, pour les recaler sur la vraie valeur ONI officielle d'avril
    2027 une fois connue (ex. anchor_offset = valeur_reelle_avr27 - 1.0,
    si 1.0 est la valeur de mai 2027 implicitement supposée par la forme
    ci-dessus). Laisser à 0.0 tant que ce point de raccord n'est pas
    vérifié.
    """
    df = pd.read_csv(io.StringIO(_H2_2027_ONI_CSV), parse_dates=['date']).set_index('date')
    df = df + anchor_offset
    return {col: df[col].to_dict() for col in df.columns}


# ------------------------------------------------------------------
# Utilitaires de décomposition (identiques à ceux déjà utilisés pour
# les épisodes historiques -- cf. extensions_visualisation.py)
# ------------------------------------------------------------------

def _fit_ridge_raw_scale(X, y, alpha):
    """Ajuste un Ridge standardisé puis reconvertit les coefficients à
    l'échelle brute (beta, beta0), pour pouvoir décomposer trend/enso/
    seasonal terme à terme sans repasser par la standardisation à chaque
    prédiction."""
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
    """Construit le dict `model_fit` attendu par ce fichier (_decompose,
    block_bootstrap_ci, plot_temp_extended_dec2027, plot_oni_extended_dec2027)
    à partir des objets RÉELS du pipeline principal de enso_gmst_model.py
    (`model, X, y, ... = fit_model(dataset, ...)`, `dataset = build_dataset(...)`).

    À appeler une fois, juste après l'ajustement du modèle :

        model_fit = build_model_fit(model, X, dataset, enso_df_raw, gmst_df, lag)

    Deux incompatibilités de convention corrigées ici (source du
    NameError/AttributeError en cascade si `model_fit` est construit "à la
    main" avec la structure décrite en tête de fichier) :

    1. Renommage de colonnes -- enso_gmst_model.py nomme ses variables
       't_index'/'t_index2'/'enso_x_t' (cf. build_dataset), alors que
       _decompose()/_fit_ridge_raw_scale() de CE fichier attendent
       't'/'t2'/'inter'. `df` est construit ici à partir de `X` (qui
       contient déjà toutes les colonnes utilisées à l'ajustement, y
       compris les indicatrices saisonnières m2..m12), renommé, avec la
       cible rajoutée sous le nom 'gmst' (dataset a 'gmst_anom_preind').
    2. mu=0 / sd=1 (pas de standardisation) -- `model` (issu de
       fit_ridge_standardized) a DÉJÀ ses coefficients reconvertis à
       l'échelle brute (cf. sa docstring dans enso_gmst_model.py) :
       `model.predict()` attend donc des features BRUTES, pas
       standardisées. mu/sd réels rendraient `(X-mu)/sd` non neutre et
       fausseraient block_bootstrap_ci.
    """
    colmap = {'t_index': 't', 't_index2': 't2', 'enso_x_t': 'inter'}
    df = X.rename(columns=colmap).astype(float).copy()
    # pd.get_dummies(..., prefix='m') nomme ses colonnes 'm_2'..'m_12' (tiret
    # bas, séparateur par défaut), alors que _decompose() cherche 'm2'..'m12'
    # (idxmap[f'm{d.month}']) -- renommées ici pour faire correspondre les deux.
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
    """IC 90% (P5/P95) par bootstrap par blocs mobiles (Künsch 1989), pour
    `total` et `contrefactuel`, sur les dates demandées et un calendrier
    ENSO donné. Reproduit la méthodologie §2.2 du manuscrit."""
    model, mu, sd = model_fit['model'], model_fit['mu'], model_fit['sd']
    feature_cols, df, lag = model_fit['feature_cols'], model_fit['df'], model_fit['lag']
    # .alpha_ (tiret bas) = attribut sklearn "fitted" -- cf. fit_model() dans
    # enso_gmst_model.py ; .alpha (sans tiret) accepté en repli si un autre
    # objet modèle est un jour passé ici.
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
# 0 quater. UTILITAIRES ADDITIONNELS -- import depuis
# enso_gmst_model_extension_2027.py
# ====================================================================
#
# NB : ce fichier intègre déjà, ci-dessus, sa PROPRE version (autonome,
# via _H2_2027_ONI_CSV) de build_h2_2027_scenarios(), _fit_ridge_raw_scale(),
# _decompose(), _build_enso_series() et block_bootstrap_ci() -- ces
# fonctions-là ne sont donc PAS réimportées ici pour éviter d'écraser
# silencieusement les correctifs déjà en place (repli .alpha_/.alpha,
# renommage de colonnes via build_model_fit, etc.). Seules les fonctions
# de enso_gmst_model_extension_2027.py qui n'avaient pas encore
# d'équivalent ici sont ajoutées : deux jeux de valeurs par défaut
# (scénario ONI officiel + tableau GMSTA officiel), la décomposition
# ponctuelle decompose_scenario(), l'assemblage complet par hypothèse
# forecast_full_2027(), et les deux agrégats period_averages() /
# annual_enso_attribution_2027(). Toutes réutilisent les utilitaires du
# fichier (model_fit, _fit_ridge_raw_scale, _decompose) déjà définis
# plus haut.

def default_official_scenario_aug26_avr27():
    """Interpolation entre les points d'ancrage cités dans le manuscrit
    (Juil26 obs. +2,03°C ; pic Nov26 +3,96°C ; Dec26 +3,92°C ; Avr27 +2,29°C).
    À REMPLACER par les valeurs mensuelles réelles du Climate Dashboard
    dès qu'elles sont disponibles."""
    return {
        '2026-08-01': 2.03 + (3.96 - 2.03) * 1 / 4,
        '2026-09-01': 2.03 + (3.96 - 2.03) * 2 / 4,
        '2026-10-01': 2.03 + (3.96 - 2.03) * 3 / 4,
        '2026-11-01': 3.96,
        '2026-12-01': 3.92,
        '2027-01-01': 3.92 + (2.29 - 3.92) * 1 / 4,
        '2027-02-01': 3.92 + (2.29 - 3.92) * 2 / 4,
        '2027-03-01': 3.92 + (2.29 - 3.92) * 3 / 4,
        '2027-04-01': 2.29,
    }


def default_official_gmsta_table():
    """Tableau mensuel détaillé du manuscrit (°C), Juillet 2026-Juin 2027."""
    return {
        '2026-07-01': 1.47, '2026-08-01': 1.49, '2026-09-01': 1.51,
        '2026-10-01': 1.57, '2026-11-01': 1.65, '2026-12-01': 1.79,
        '2027-01-01': 1.90, '2027-02-01': 1.90, '2027-03-01': 1.91,
        '2027-04-01': 1.90, '2027-05-01': 1.89, '2027-06-01': 1.80,
    }


def decompose_scenario(model_fit, dates, enso_series):
    """Décomposition centrale (non bootstrapée) trend/enso/seasonal/total
    pour un calendrier ENSO donné, sur les dates demandées."""
    model, df = model_fit['model'], model_fit['df']
    feature_cols, lag = model_fit['feature_cols'], model_fit['lag']
    beta, beta0 = _fit_ridge_raw_scale(df[feature_cols].values, df['gmst'].values, model.alpha)
    t_last, last_date = df['t'].iloc[-1], df.index[-1]
    s_smooth = enso_series.rolling(3, min_periods=1).mean()
    return _decompose(dates, s_smooth, beta, beta0, feature_cols, df, lag, t_last, last_date)


def forecast_full_2027(model_fit, official_scenario_aug26_avr27, h2_scenario,
                        official_gmsta_table, bias_correction_month='2027-06-01'):
    """Assemble Juillet 2026-Décembre 2027 pour UNE hypothèse H2 :
    Juillet 2026-Juin 2027 = valeurs officielles du Tableau 4 (affichage exact) ;
    Juillet-Décembre 2027 = décomposition du modèle sous l'hypothèse, recalée
    par une correction de biais additive constante estimée au mois de raccord
    (défaut : juin 2027), pour la continuité avec le corps du manuscrit."""
    nino_obs = model_fit['nino']['ssta']
    full_dates = pd.date_range('2026-07-01', '2027-12-01', freq='MS')
    enso_series = _build_enso_series(nino_obs, official_scenario_aug26_avr27, h2_scenario)
    dec = decompose_scenario(model_fit, full_dates, enso_series)

    bias = dec.loc[bias_correction_month, 'total'] - official_gmsta_table[bias_correction_month]
    mask = dec.index >= '2027-07-01'
    dec.loc[mask, ['trend', 'total', 'contrefactuel']] -= bias

    official_s = pd.Series(official_gmsta_table)
    official_s.index = pd.to_datetime(official_s.index)
    dec.loc[official_s.index, 'total'] = official_s.values
    return dec


def period_averages(full_decomp, gmst_obs_2026_h1):
    """S1 2027 (Jan-Juin), année 2026 (obs Jan-Juin + projeté Juil-Déc),
    année 2027 complète (pour l'hypothèse de `full_decomp`)."""
    s1_2027 = full_decomp.loc['2027-01-01':'2027-06-01', 'total'].mean()
    annee_2026 = pd.concat([gmst_obs_2026_h1, full_decomp.loc['2026-07-01':'2026-12-01', 'total']]).mean()
    annee_2027 = full_decomp.loc['2027-01-01':'2027-12-01', 'total'].mean()
    return dict(s1_2027=s1_2027, annee_2026=annee_2026, annee_2027=annee_2027)


def annual_enso_attribution_2027(model_fit, official_scenario_aug26_avr27, h2_scenarios,
                                  official_gmsta_table):
    """Part ENSO (°C et %) dans la moyenne annuelle GMSTA 2027, pour CHACUNE
    des 3 hypothèses H2 2027 (pas seulement le scénario central)."""
    nino_obs = model_fit['nino']['ssta']
    jan_jun = pd.date_range('2027-01-01', '2027-06-01', freq='MS')
    enso_common = _build_enso_series(nino_obs, official_scenario_aug26_avr27, {})
    jj_decomp = decompose_scenario(model_fit, jan_jun, enso_common)
    b = (jj_decomp['total'] - pd.Series(official_gmsta_table).rename(index=pd.Timestamp)).mean()
    jj_decomp[['trend', 'total', 'contrefactuel']] -= b

    results = {}
    for label, hyp in h2_scenarios.items():
        full = forecast_full_2027(model_fit, official_scenario_aug26_avr27, hyp, official_gmsta_table)
        dec_h2 = full.loc['2027-07-01':'2027-12-01']
        full_year = pd.concat([jj_decomp, dec_h2]).sort_index()
        ann = full_year[['trend', 'enso', 'seasonal', 'total']].mean()
        ann['enso_pct'] = 100 * ann['enso'] / ann['total']
        results[label] = ann
    return results


# ====================================================================
# 1. FIGURE TEMPÉRATURE ÉTENDUE (Juillet 2026 - Décembre 2027)
# ====================================================================

# ====================================================================
# 0 ter. TABLEAU MENSUEL -- 3 HYPOTHÈSES H2 2027 (GMSTA)
# ====================================================================
#
# Même décomposition contrefactuelle que le tableau officiel 2026-2027
# (plot_monthly_attribution_table, plot_episodes_v7.py) : Modèle / Tendance
# / Saisonnier / Contrefactuel (= Tendance+Saisonnier) / ENSO (= Modèle -
# Contrefactuel) -- mais répétée pour chacune des 3 hypothèses H2 2027,
# empilée en 3 blocs a) neutre, b) central, c) La Niña forte dans UNE
# seule figure (au lieu des 3 hypothèses en colonnes côte à côte).
# IC 90% disponible pour Modèle et Contrefactuel (bootstrap par blocs
# mobiles, cf. block_bootstrap_ci) ; Tendance/Saisonnier/ENSO en valeur
# ponctuelle seulement (ces 3 composantes ne sont pas individuellement
# bootstrappées par block_bootstrap_ci, à la différence des IC mensuels
# détaillés utilisés pour les épisodes historiques).
_H2_PCT_ALERT_THRESHOLD = 100  # au-delà, la part en % devient un artefact de
                                # signes opposés plutôt qu'une lecture directe


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
    `decomps`, `cis` : dicts {'neutre':..., 'central':..., 'la_nina_forte':...}
    déjà calculés par plot_temp_extended_dec2027() -- passés tels quels, pas
    recalculés ici. `decomps[label]` a les colonnes 'trend'/'seasonal'/
    'enso'/'total'/'contrefactuel' (même structure que les décompositions
    historiques) ; `cis[label]` a 'total_p5'/'total_p95'/'cf_p5'/'cf_p95'.

    Produit UNE figure PNG avec 3 sous-tableaux empilés (a/b/c, un par
    hypothèse), et exporte les mêmes valeurs (toutes hypothèses confondues,
    colonne 'hypothesis') en un seul CSV long-format.
    """
    order = ['neutre', 'central', 'la_nina_forte']
    block_titles = {
        'neutre': 'a) Neutral ENSO hypothesis',
        'central': 'b) Central hypothesis (moderate La Nina)',
        'la_nina_forte': 'c) Strong La Nina hypothesis',
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

    # -- Mise en page : suptitle+sous-titre communs en haut, puis 3 blocs
    #    empilés (label a/b/c + tableau), un peu d'air entre chaque --
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

    y_cursor_in = header_inches  # distance depuis le HAUT de la figure
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
            elif c in (2, 3, 4):  # Tendance / Saisonnier / Contrefactuel -- teinte grise
                cell.set_facecolor('#e8e8e8' if r % 2 == 0 else '#f4f4f4')
            else:
                cell.set_facecolor(light_tint(FACTUAL, 0.08) if r % 2 == 0 else 'white')
        y_cursor_in += block_inches + gap_inches

    TITLE_TOP_IN, SUBTITLE_TOP_IN = 0.32, 0.65
    fig.suptitle("Monthly GMSTA decomposition under the three H2 2027 ENSO hypotheses",
                 fontsize=14, fontweight='bold', x=0.02, ha='left', va='top',
                 y=1 - TITLE_TOP_IN / fig_height)
    subtitle_lines = [
    f"{dates[0]:%B}-December 2027 extension (author hypotheses), spliced onto the official scenario "
    f"at {ext_start - pd.DateOffset(months=1):%B %Y} - TESR model, exact linear decomposition:",
    
    "ENSO = Model - Counterfactual (= Trend + Seasonal) - departure from preindustrial baseline "
    "(1850-1900), °C",

    f"90% CI by moving-block bootstrap (n={n_boot}) on Model and Counterfactual only; "
    "Trend/Seasonal/ENSO shown as point estimates",
    ]

    if flagged[0]:
        subtitle_lines.append(
            "†: |%| > 100 -- Trend+Seasonal+ENSO=Model is exact in °C, but an individual % can exceed "
            "100 (or be negative) when another component has a sign opposite to the Model; trust the °C value"
            )

    subtitle = "\n".join(subtitle_lines)

    fig.text(
        0.02,
        1 - SUBTITLE_TOP_IN / fig_height,
        subtitle,
        fontsize=9,
        style='italic',
        color='#444444',
        ha='left',
        va='top'
        )
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
    Bornes IC 90% calculées SÉPARÉMENT pour chacune des 3 hypothèses H2 2027
    (bootstrap par blocs mobiles, n_boot réplicats chacune), puis combinées en
    une "plage complète" = [min des P5, max des P95] à chaque mois -- c'est la
    réponse directe à la demande "ajouter les bornes de confiance à 90% pour
    les projections 2027 de chaque scénario puis obtenir une plage complète".

    Mise en page calée sur le standard du manuscrit (Fig. 12/13) : bandes
    emboîtées, contribution El Niño hachurée, barres d'erreur sur le scénario
    central et le contrefactuel, seuils +1,5°C/+2°C, légende à droite centrée
    verticalement, séparateur officiel/extension étiqueté au milieu du graphe.

    En plus de la figure PNG, exporte la trajectoire GMSTA complète (mensuelle,
    Juillet 2026-Décembre 2027) en CSV (`csv_filename`) : scénario central
    (total + contrefactuel + IC 90%), les 2 autres hypothèses ONI (neutre /
    La Niña forte) et la plage combinée -- la donnée qui sous-tend la figure.
    """
    nino_obs, gmst = model_fit['nino'], model_fit['gmst']
    lag = model_fit['lag']

    full_dates = pd.date_range('2026-07-01', '2027-12-01', freq='MS')
    official_s = pd.Series(official_gmsta_table)
    official_s.index = pd.to_datetime(official_s.index)
    # -- Bascule officiel -> hypothèses H2 2027 : dérivée du DERNIER mois
    #    réellement présent dans `official_gmsta_table` (pas figée au 30 juin
    #    -- votre pipeline actuel s'arrête en avril 2027, cf. enveloppe_enso) --
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
        # continuité : recaler sur le dernier mois officiel commun aux 3 hypothèses
        b = dec.loc[splice_month, 'total'] - official_gmsta_table[splice_month.strftime('%Y-%m-%d')]
        dec.loc[dec.index >= ext_start, ['trend', 'total', 'contrefactuel']] -= b
        dec.loc[official_s.index, 'total'] = official_s.values  # affichage exact Tableau 4
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
                     label='El Nino contribution (central scenario)')

    ax.errorbar(full_dates, central['total'],
                yerr=[(central['total'] - ci_c['total_p5']).clip(lower=0),
                      (ci_c['total_p95'] - central['total']).clip(lower=0)],
                color='#c0392b', lw=2.4, marker='o', ms=4.5, capsize=3, elinewidth=1.0, ecolor='#c0392b',
                zorder=7, label='Model prediction — central scenario (with El Nino)')
    ax.errorbar(full_dates, contrefactuel,
                yerr=[(contrefactuel - ci_c['cf_p5']).clip(lower=0), (ci_c['cf_p95'] - contrefactuel).clip(lower=0)],
                color='#555555', lw=1.8, ls='--', marker='o', ms=4, capsize=3, elinewidth=0.9, ecolor='#555555',
                zorder=6, label='Counterfactual (ENSO-neutral)')

    ax.plot(full_dates, decomps['neutre']['total'], color='#2980b9', lw=1.6, ls=':', zorder=6,
            label='Neutral ENSO hypothesis (H2 2027)')
    ax.plot(full_dates, decomps['la_nina_forte']['total'], color='#1a1aa6', lw=1.6, ls='-.', zorder=6,
            label='Strong La Nina hypothesis (H2 2027)')

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
              "Projection July 2026-December 2027 - with El Nino vs ENSO-neutral - Departure from preindustrial "
              "baseline (1850-1900), \u00b0C - TESR model, lag = 3 months\n"
              f"July 2026-{splice_month:%B %Y}: official scenario (Table 4, initialized 1 August 2026) - "
              f"{ext_start:%B}-December 2027: combined range across the "
              f"90% CI of the 3 ENSO hypotheses for H2 2027 (moving-block bootstrap, n={n_boot})",
              fontsize=11, style='italic', color='#555555', ha='left', va='top')

    ax.legend(loc='center left', fontsize=9.8, framealpha=0.95, ncol=1, bbox_to_anchor=(1.01, 0.5))
    fig.text(0.99, 0.008, data_sources, fontsize=8.5, color='#666', ha='right')

    plt.tight_layout(rect=[0, 0.02, 0.83, 0.90])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    # -- Export CSV de la trajectoire GMSTA sous-jacente à la figure --
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

    # -- Tableau mensuel comparant les 3 hypothèses H2 2027 (même esprit
    #    visuel que le tableau d'attribution mensuel déjà utilisé pour les
    #    épisodes 2026-2027), à partir des `decomps`/`cis` déjà calculés
    #    ci-dessus (pas de recalcul) --
    table_png, table_csv = plot_h2_2027_gmst_table(
        decomps, cis, ext_start, n_boot=n_boot,
        filename=table_filename, csv_filename=table_csv_filename,
        data_sources=data_sources)

    return filename, csv_filename, table_png, table_csv


# ====================================================================
# 2. FIGURE ONI ÉTENDUE (Janvier 2026 - Décembre 2027)
# ====================================================================

# ====================================================================
# 2 bis. TABLEAUX ONI -- 3 hypothèses H2 2027 (Mai-Décembre) et
#        scénario officiel (Août 2026-Avril 2027), avec export CSV
# ====================================================================

def _render_oni_table(col_labels, cell_text, title, subtitle, filename, csv_export=None,
                       csv_filename=None, fig_width=11, highlight_col=None,
                       data_sources="Data: The Climate Brink @ ENSO Dashboard"):
    """Rendu matplotlib partagé par plot_oni_h2_2027_table() et
    plot_oni_official_table() -- même esprit visuel que les autres
    tableaux du pipeline (plot_monthly_attribution_table, plot_h2_2027_gmst_table)."""
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
    """
    Tableau mensuel ONI (Mai-Décembre 2027) comparant les 3 hypothèses
    d'auteur H2 2027 (neutre / central / La Niña forte) -- valeurs
    directement issues de `h2_scenarios` (build_h2_2027_scenarios()),
    sans recalcul. Exporte les mêmes valeurs en CSV.
    """
    order = ['neutre', 'central', 'la_nina_forte']
    disp_names = {'neutre': 'Neutral ENSO', 'central': 'Central (moderate La Nina)',
                  'la_nina_forte': 'Strong La Nina'}
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
    """
    Tableau mensuel du scénario ONI officiel (Août 2026-Avril 2027) :
    médiane multi-modèle + bandes de quantiles emboîtées Q0-Q100/Q5-Q95/
    Q25-Q75 (APPROXIMATION -- même avertissement que la figure, cf. tête
    de fichier : cône reconstruit faute du fichier membre-par-membre).
    `q0`..`q100` sont les mêmes pd.Series (indexées sur x_dates) déjà
    calculées dans plot_oni_extended_dec2027(), passées telles quelles.
    Exporte les mêmes valeurs en CSV.
    """
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
    analogs_24m : dict {"1982/1983": np.array(24 valeurs), "1997/1998": ..., "2015/2016": ...}
        -- séries Niño 3.4 Janvier(année pic)-Décembre(année pic+1), alignées par POSITION
        calendaire (Jan matche Jan, etc.) sur l'axe Jan2026-Déc2027, exactement comme
        dans votre Fig. 9 originale.

    Conserve les bandes de quantiles emboîtées (Q0-Q100/Q5-Q95/Q25-Q75) sur toute la
    période officielle (Août 2026-Avril 2027, palette orange = reconstruction
    approximative de l'enveloppe multi-modèles) ET sur l'extension (Mai-Décembre 2027,
    palette bleue = hypothèses d'auteur), avec une couleur nettement différente pour
    ne jamais confondre les deux régimes de confiance.
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

    # --- bandes période officielle : cône d'incertitude croissant avec le délai de
    #     prévision (APPROXIMATION -- cf. avertissement en tête de fichier) ---
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

    # --- bandes extension : enveloppe des 3 hypothèses ---
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
            marker='o', ms=3.5, zorder=6, label='Observed (Nino 3.4)')

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
            label='Strong La Nina hypothesis (extension lower bound)')

    colors_analog = {'1982/1983': '#8a6d3b', '1997/1998': '#6a3d9a', '2015/2016': '#c2185b'}
    for label, vals in analogs_24m.items():
        ax.plot(x_dates, vals, color=colors_analog.get(label, '#333'), lw=1.6, ls='--',
                marker='o', ms=3.5, zorder=4, label=f'Analog {label}')

    cats = [(3.0, 'Extreme El Nino (+3)', '#7b241c'), (2.0, 'Very strong El Nino (+2)', '#e74c3c'),
            (1.5, 'Strong El Nino (+1.5)', '#e67e22'), (1.0, 'Moderate El Nino (+1)', '#f1c40f'),
            (0.5, 'El Nino (+0.5)', '#f4d03f'), (-0.5, 'La Nina (-0.5)', '#5dade2')]
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

    ax.set_ylabel("ONI | Nino 3.4 index (\u00b0C) [1991-2020 baseline]", fontsize=11)
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate(rotation=90)
    ax.legend(loc='center left', fontsize=8.6, framealpha=0.95, ncol=1, bbox_to_anchor=(1.10, 0.5))

    fig.suptitle("ONI | Nino 3.4 index — synthesis and extension through December 2027",
                 x=0.02, y=0.975, ha='left', fontsize=19, fontweight='bold')
    fig.text(0.02, 0.928,
              "Official (August 2026-April 2027, orange tones): approximate reconstruction of the multi-model "
              "envelope (Climate Dashboard, initialized 1 August 2026 — member-by-member file not available here)\n"
              "Extension (May-December 2027, blue tones): ILLUSTRATIVE envelope built on 3 author hypotheses "
              "(strong La Nina / central / neutral ENSO), not a multi-model scenario",
              fontsize=10.5, style='italic', color='#555555', ha='left', va='top')
    fig.text(0.99, 0.008, data_sources, fontsize=8, color='#666', ha='right')

    plt.tight_layout(rect=[0, 0.02, 0.85, 0.90])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    # -- Tableau ONI : 3 hypothèses H2 2027 (Mai-Décembre), à partir de
    #    `h2_scenarios` directement (pas de recalcul) --
    h2_table_png, h2_table_csv = plot_oni_h2_2027_table(
        h2_scenarios, filename=h2_table_filename, csv_filename=h2_table_csv_filename)

    # -- Tableau ONI officiel (Août 2026-Avril 2027), à partir des bandes
    #    q0..q100 déjà calculées ci-dessus pour la figure (même
    #    reconstruction approximative, pas de recalcul) --
    official_table_png, official_table_csv = plot_oni_official_table(
        official_scenario_aug26_avr27, q0, q5, q25, q75, q95, q100, init_date=init_date,
        filename=official_table_filename, csv_filename=official_table_csv_filename)

    return filename, h2_table_png, h2_table_csv, official_table_png, official_table_csv




if __name__ == "__main__":
    # ==================================================================
    # Data paths: override via environment variables, or edit the
    # defaults below to point at your own local CSV files (see README
    # for the expected format of each file).
    # ==================================================================
    import os

    PATH_GMST = os.environ.get("TESR_PATH_GMST", "C:/Users/lolma/Documents/B/Climat temp autres/previ et fiab/Tendance saiso/ENSOT/era5_gmst_c3s.csv")
    PATH_ENSO = os.environ.get("TESR_PATH_ENSO", "C:/Users/lolma/Documents/B/Climat temp autres/previ et fiab/Tendance saiso/ENSOT/nino34_anomaly.csv")
    # Member-by-member Climate Dashboard file (multi-model ONI), used
    # below for the ENSO scenario and the peak-month distribution.
    PATH_ENSO_MEMBERS = os.environ.get("TESR_PATH_ENSO_MEMBERS", "C:/Users/lolma/Documents/B/Climat temp autres/previ et fiab/Tendance saiso/ENSOT/enso_members_oni.csv")

    gmst_df = load_era5_gmst_c3s(PATH_GMST)
    enso_df_raw = load_enso_climatereanalyzer(PATH_ENSO)

    # ------------------------------------------------------------------
    # MISE À JOUR MANUELLE : mois observés plus récents que les CSV
    # statiques (à compléter/retirer à chaque fois que tu régénères les
    # fichiers sources avec les nouvelles données officielles).
    # ------------------------------------------------------------------
    observations_recentes = {
        # date : (gmst_anom_preind, enso_ssta_brut)
        "2026-06-01": (1.39, 1.45),
        "2026-07-01": (1.47, 2.03),
    }
    for date_str, (gmst_val, enso_val) in observations_recentes.items():
        d = pd.Timestamp(date_str)
        gmst_df.loc[d, 'gmst_anom_preind'] = gmst_val
        enso_df_raw.loc[d, 'enso_ssta'] = enso_val
    gmst_df = gmst_df.sort_index()
    enso_df_raw = enso_df_raw.sort_index()

    # ==================================================================
    # AJUSTEMENT DU MODÈLE (TESR : Tendance quadratique + ENSO lissé causal + Saisonnier + Ridge)
    # Gain validé par walk-forward (horizon 12 mois, sans fuite) :
    # RMSE 0.134 -> 0.127 °C, R² 0.733 -> 0.759, vs la v3 (ENSO brut, OLS).
    # ==================================================================
    enso_df = smooth_enso(enso_df_raw, window=3)
    lag, corr = find_optimal_lag(enso_df['enso_ssta'], gmst_df['gmst_anom_preind'])
    print(f"Optimal lag (3-month-smoothed ENSO): {lag} months (correlation = {corr:.3f})")

    dataset = build_dataset(enso_df, gmst_df, lag)
    model, X, y, X_train, X_test, y_train, y_test = fit_model(
        dataset, use_seasonal=True, regularize=True
    )

    # -- Évaluation out-of-sample (dernier 20%, chronologique) --
    _, r_test, r2_test, rmse_test = evaluate_model(
        model, X_test, y_test, label="test (chronologique, 20% final)"
    )

    # -- Résidus walk-forward réels (horizon 12 mois, sans fuite) : servent
    #    de base empirique à toutes les bandes d'incertitude / probabilités
    #    (préféré à une hypothèse gaussienne symétrique) --
    #
    # IMPORTANT (reproductibilité) : on utilise ici un Ridge à alpha FIXE
    # (celui déjà validé par RidgeCV sur l'ensemble complet, model.alpha_),
    # plutôt que de relancer une recherche RidgeCV complète à chaque
    # sous-fenêtre. Relancer RidgeCV(alphas=...) à chaque étape rendait le
    # résultat dépendant de la version de scikit-learn installée (son
    # algorithme de LOO-CV interne a changé d'implémentation au fil des
    # versions), ce qui produisait des résidus -- et donc des distributions
    # de probabilité -- différents d'une machine à l'autre pour un même
    # jeu de données. Un alpha fixe rend le calcul déterministe et stable
    # quelle que soit la version de scikit-learn.
    _wf = walk_forward_forecasts(
        X, y, fit_fn=lambda Xt, yt: fit_ridge_standardized(Xt, yt, model.alpha_)
    )
    residuals = _wf["residual"].values
    print(f"\nRésidus walk-forward : n={len(residuals)}, "
          f"mean={residuals.mean():+.3f}, std={residuals.std():.3f}, "
          f"skew={pd.Series(residuals).skew():.2f}")

    # ==================================================================
    # IMAGE 1 : confrontation modèle vs observations sur tout l'historique
    # ==================================================================
    plot_model_vs_obs(
        dataset, model, X, y,
        title="Global Mean Surface Temperature Anomaly (GMSTA) — TESR model",
        subtitle="Historical model fit and model–observation agreement, 1940–2026 (1850–1900 reference period)"
    )

    # Vérification modèle vs obs, réutilisée dans les graphiques suivants
    model_fit_full = pd.Series(model.predict(X), index=dataset.index)

    # ==================================================================
    # IMAGE 2 : projection à partir de valeurs ENSO saisies manuellement
    #           (courte démo, ENSO déjà lissé/décalé -> voir docstring)
    # ==================================================================
    mes_valeurs_enso = [0.4, 0.3, 0.2, 0.1, 0.0, -0.1]  # à adapter (déjà décalées du lag)
    forecast_df = forecast_manual(model, X.columns.tolist(), dataset, mes_valeurs_enso)
    print(forecast_df)

    plot_forecast_zoom(
        dataset, forecast_df,
        model_fit=model_fit_full,
        residuals=residuals,
        zoom=("2022-01-01", forecast_df.index[-1].strftime("%Y-%m-%d")),
        filename="gmst_forecast_manual_demo.png",
        title="Global Mean Surface Temperature Anomaly (GMSTA) - projection from manually entered ENSO values",
        subtitle=f"TESR model (quadratic trend + smoothed ENSO + seasonal + ridge), lag={lag} months",
        legend_loc='lower left',
    )

    # ==================================================================
    # SCÉNARIO ENSO MÉDIAN (Niño 3.4/ONI prévu, août 2026 -> avril 2027)
    # Source : Climate Dashboard multi-modèles, membre par membre
    # (enso_members_oni.csv), pondération égale par modèle, SINTEX-F
    # explicitement exclu (biais froid marqué, tire p05/q0 vers le bas
    # de façon non représentative des autres modèles -- cf. écart type
    # par modèle dans le fichier source).
    #
    # NB méthodologique (à reporter en Limites du manuscrit) : l'indice
    # utilisé pour l'entraînement (nino34_anomaly.csv, ERA5, base 1991-2020)
    # diffère de la source du scénario ici (ONI, CPC/ERSSTv5, moyenne
    # mobile centrée sur 3 mois). Juillet 2026 est la seule valeur
    # "observed" fournie dans cette source (ONI=+2.03°C) ; elle est
    # utilisée ici comme point de jonction déterministe (sans bande
    # d'incertitude) avant le scénario multi-modèles proprement dit
    # (août 2026 -> avril 2027), plutôt que d'être mélangée aux mois
    # ERA5 historiques dans enso_df_raw.
    # ==================================================================
    EXCLUDE_MODELS = ('SINTEX-F',)

    enveloppe_enso = load_enso_dashboard_scenario(PATH_ENSO_MEMBERS, exclude_models=EXCLUDE_MODELS)
    print("\n--- Enveloppe ENSO (ONI, pondération égale par modèle, SINTEX-F exclu) ---")
    with pd.option_context('display.float_format', '{:+.3f}'.format):
        print(enveloppe_enso)
    enveloppe_enso.to_csv("scenario_enso_enveloppe_exSTX.csv")

    JUILLET_OBS_ONI = 2.03  # ONI observé juillet 2026 (source Climate Dashboard)

    # -- Graphique de l'hypothèse ENSO retenue (enveloppes emboîtées,
    #    jusqu'à mars 2027 même si le fichier source va jusqu'en avril) --
    plot_enso_scenario_envelope(
        enveloppe_enso,
        enso_hist_df=enso_df_raw,
        obs_junction=(pd.Timestamp("2026-07-01"), JUILLET_OBS_ONI),
        end_date="2027-04-01",
        filename="enso_scenario_envelope_mai2027.png",
        analog_hist_df=enso_df_raw,
    )
    enso_c3s_median_brut = pd.concat([
        pd.Series([JUILLET_OBS_ONI], index=pd.to_datetime(["2026-07-01"])),
        enveloppe_enso['median'],
    ])
    enso_c3s_median = smooth_enso_for_forecast(enso_df_raw, enso_c3s_median_brut, window=3)

    forecast_c3s = forecast_from_enso_calendar(model, X.columns.tolist(),
                                                dataset, lag, enso_c3s_median)
    print(f"Applied lag: {lag} months -> the Jul.26-Apr.27 ENSO scenario "
          f"predicts GMST from {forecast_c3s.index[0].strftime('%B %Y')} "
          f"to {forecast_c3s.index[-1].strftime('%B %Y')}")
    print(forecast_c3s)

    # ------------------------------------------------------------------
    # Le scénario Climate Dashboard du 10/08/2026 est désormais la donnée
    # définitive (hypothèse centrale) ; plus de révision provisoire.
    # ------------------------------------------------------------------
    enso_c3s_revise_brut = enso_c3s_median_brut.copy()
    enso_c3s_revise = smooth_enso_for_forecast(enso_df_raw, enso_c3s_revise_brut, window=3)
    forecast_c3s_revise = forecast_from_enso_calendar(model, X.columns.tolist(),
                                                       dataset, lag, enso_c3s_revise)
    print(f"\nPic scénario (hypothèse centrale) : "
          f"{forecast_c3s_revise['gmst_anom_pred_preind'].max():.3f}°C "
          f"en {forecast_c3s_revise['gmst_anom_pred_preind'].idxmax():%B %Y}")

    # -- Scénarios alternatifs hauts/bas (enveloppe P5/P95 mensuelle,
    #    passée telle quelle dans le modèle GMST -- ne pas confondre avec
    #    l'incertitude propre au modèle GMST lui-même, qui vient des
    #    résidus walk-forward et s'ajoute par-dessus) --
    enso_c3s_haut_brut = pd.concat([
        pd.Series([JUILLET_OBS_ONI], index=pd.to_datetime(["2026-07-01"])),
        enveloppe_enso['p95'],
    ])
    enso_c3s_bas_brut = pd.concat([
        pd.Series([JUILLET_OBS_ONI], index=pd.to_datetime(["2026-07-01"])),
        enveloppe_enso['p05'],
    ])
    forecast_c3s_haut = forecast_from_enso_calendar(
        model, X.columns.tolist(), dataset, lag,
        smooth_enso_for_forecast(enso_df_raw, enso_c3s_haut_brut, window=3))
    forecast_c3s_bas = forecast_from_enso_calendar(
        model, X.columns.tolist(), dataset, lag,
        smooth_enso_for_forecast(enso_df_raw, enso_c3s_bas_brut, window=3))
    print(f"High alternative scenario (monthly P95): peak "
          f"{forecast_c3s_haut['gmst_anom_pred_preind'].max():.3f}°C en "
          f"{forecast_c3s_haut['gmst_anom_pred_preind'].idxmax():%B %Y}")
    print(f"Low alternative scenario (monthly P05): peak "
          f"{forecast_c3s_bas['gmst_anom_pred_preind'].max():.3f}°C en "
          f"{forecast_c3s_bas['gmst_anom_pred_preind'].idxmax():%B %Y}")

    # -- Scénarios INTERQUARTILES (P25/P75 -- bande resserrée autour de la
    #    mediane, même logique que P05/P95 ci-dessus). Ajoutés pour que la
    #    fonction quantile utilisée par _combined_uncertainty_samples (et
    #    donc la heatmap + la distribution de probabilité) s'ancre sur le
    #    corps de la distribution ENSO, pas seulement sur ses extrêmes. --
    enso_c3s_p75_brut = pd.concat([
        pd.Series([JUILLET_OBS_ONI], index=pd.to_datetime(["2026-07-01"])),
        enveloppe_enso['p75'],
    ])
    enso_c3s_p25_brut = pd.concat([
        pd.Series([JUILLET_OBS_ONI], index=pd.to_datetime(["2026-07-01"])),
        enveloppe_enso['p25'],
    ])
    forecast_c3s_p75 = forecast_from_enso_calendar(
        model, X.columns.tolist(), dataset, lag,
        smooth_enso_for_forecast(enso_df_raw, enso_c3s_p75_brut, window=3))
    forecast_c3s_p25 = forecast_from_enso_calendar(
        model, X.columns.tolist(), dataset, lag,
        smooth_enso_for_forecast(enso_df_raw, enso_c3s_p25_brut, window=3))
    print(f"Interquartile high scenario (monthly P75): peak "
          f"{forecast_c3s_p75['gmst_anom_pred_preind'].max():.3f}°C en "
          f"{forecast_c3s_p75['gmst_anom_pred_preind'].idxmax():%B %Y}")
    print(f"Interquartile low scenario (monthly P25): peak "
          f"{forecast_c3s_p25['gmst_anom_pred_preind'].max():.3f}°C en "
          f"{forecast_c3s_p25['gmst_anom_pred_preind'].idxmax():%B %Y}")

    # -- Scénarios EXTRÊMES (bornes Q0/Q100 -- min/max bruts du pool de
    #    membres après exclusion de SINTEX-F, cf. load_enso_dashboard_scenario) :
    #    même logique que P05/P95 ci-dessus, mais pour la projection GMST la
    #    plus basse / la plus haute (pas juste P5-P95). Utilisé pour habiller
    #    le graphique scénario-vs-contrefactuel d'une enveloppe de dispersion
    #    ENSO, et pour élargir la distribution probabiliste (heatmap +
    #    distribution au paroxysme) à cette 2e source d'incertitude. --
    enso_c3s_q100_brut = pd.concat([
        pd.Series([JUILLET_OBS_ONI], index=pd.to_datetime(["2026-07-01"])),
        enveloppe_enso['q100'],
    ])
    enso_c3s_q0_brut = pd.concat([
        pd.Series([JUILLET_OBS_ONI], index=pd.to_datetime(["2026-07-01"])),
        enveloppe_enso['q0'],
    ])
    forecast_c3s_q100 = forecast_from_enso_calendar(
        model, X.columns.tolist(), dataset, lag,
        smooth_enso_for_forecast(enso_df_raw, enso_c3s_q100_brut, window=3))
    forecast_c3s_q0 = forecast_from_enso_calendar(
        model, X.columns.tolist(), dataset, lag,
        smooth_enso_for_forecast(enso_df_raw, enso_c3s_q0_brut, window=3))
    print(f"Extreme high scenario (Q100): peak "
          f"{forecast_c3s_q100['gmst_anom_pred_preind'].max():.3f}°C en "
          f"{forecast_c3s_q100['gmst_anom_pred_preind'].idxmax():%B %Y}")
    print(f"Extreme low scenario (Q0): peak "
          f"{forecast_c3s_q0['gmst_anom_pred_preind'].max():.3f}°C en "
          f"{forecast_c3s_q0['gmst_anom_pred_preind'].idxmax():%B %Y}")

    # -- Distribution au paroxysme (résolution 0.2°C) --
    mois_paroxysme = enveloppe_enso['median'].idxmax()
    mois_paroxysme_label = f"{mois_paroxysme:%B %Y}"
    membres_paroxysme = load_enso_dashboard_members(
        PATH_ENSO_MEMBERS, mois_paroxysme.strftime("%Y-%m"), exclude_models=EXCLUDE_MODELS
    )
    hist_paroxysme = plot_enso_peak_distribution(
        membres_paroxysme, mois_paroxysme_label,
        enveloppe_enso.loc[mois_paroxysme, 'median'],
        bin_range=(1.9, 5.3),
    )
    print(f"\n--- Distribution ONI au paroxysme ({mois_paroxysme_label}), résolution 0.2°C ---")
    print(hist_paroxysme.round(1))

    # ==================================================================
    # EXTENSION -- projection étendue à Décembre 2027 (3 hypothèses ENSO
    # H2 2027), branchée directement sur les objets déjà calculés ci-dessus
    # (pas de données inventées : `enveloppe_enso['median']` = scénario
    # officiel ONI, `forecast_c3s_revise` = sa traduction GMSTA par CE
    # modèle -- le point de bascule officiel/hypothèses est dérivé du
    # dernier mois réellement disponible, actuellement avril 2027).
    # ==================================================================
    model_fit = build_model_fit(model, X, dataset, enso_df_raw, gmst_df, lag)
    official_scenario_aug26_avr27 = {d.strftime('%Y-%m-01'): v
                                      for d, v in enveloppe_enso['median'].items()}
    official_gmsta_table = {d.strftime('%Y-%m-01'): v
                             for d, v in forecast_c3s_revise['gmst_anom_pred_preind'].items()}

    # -- Recalage des 3 hypothèses ONI H2 2027 sur la vraie valeur officielle --
    # `_H2_2027_ONI_CSV` (voir build_h2_2027_scenarios) a été construit en
    # supposant un dernier mois officiel (avril 2027) à ~1.0 degC. Appeler
    # build_h2_2027_scenarios() sans anchor_offset (comme précédemment)
    # laisse ce décalage à 0.0 : si la vraie valeur officielle d'avril 2027
    # diffère de cette hypothèse de calibration -- ce qui est le cas ici --
    # l'extension (Mai 2027) démarre alors depuis la valeur BRUTE du CSV et
    # non depuis la trajectoire officielle, d'où la cassure abrupte début H2
    # observée entre le dernier point officiel et le premier point Mai 2027.
    # On recale ici les 3 courbes en bloc pour repartir exactement de la
    # vraie valeur officielle du dernier mois disponible.
    H2_2027_CALIBRATION_ANCHOR = 1.0  # valeur d'avril 2027 supposée par le CSV (voir avertissement ci-dessus)
    last_official_month = max(pd.Timestamp(k) for k in official_scenario_aug26_avr27)
    last_official_value = official_scenario_aug26_avr27[last_official_month.strftime('%Y-%m-01')]
    h2_anchor_offset = last_official_value - H2_2027_CALIBRATION_ANCHOR
    h2_scenarios = build_h2_2027_scenarios(anchor_offset=h2_anchor_offset)

    png_temp_ext, csv_temp_ext, png_gmst_h2_table, csv_gmst_h2_table = plot_temp_extended_dec2027(
        model_fit, official_scenario_aug26_avr27, h2_scenarios, official_gmsta_table)
    print(f"\n--- Projection étendue Décembre 2027 (3 hypothèses ENSO H2 2027) ---")
    print(f"  -> {png_temp_ext} (figure) / {csv_temp_ext} (trajectoire GMSTA mensuelle)")
    print(f"  -> {png_gmst_h2_table} (tableau) / {csv_gmst_h2_table} (csv)")

    analogs_24m = {}
    for lbl, y0 in [('1982/1983', 1982), ('1997/1998', 1997), ('2015/2016', 2015)]:
        analogs_24m[lbl] = enso_df_raw['enso_ssta'].loc[f'{y0}-01-01':f'{y0+1}-12-01'].values
    (png_oni_ext, png_oni_h2_table, csv_oni_h2_table,
     png_oni_official_table, csv_oni_official_table) = plot_oni_extended_dec2027(
        model_fit, official_scenario_aug26_avr27, h2_scenarios, analogs_24m)
    print(f"  -> {png_oni_ext}")
    print(f"  -> {png_oni_h2_table} (tableau ONI H2 2027) / {csv_oni_h2_table} (csv)")
    print(f"  -> {png_oni_official_table} (tableau ONI officiel) / {csv_oni_official_table} (csv)")

    # Court terme (juil.-sept. 2026), à partir de l'ENSO déjà observé
    near_term_enso_brut = enso_df_raw['enso_ssta'].loc["2026-04-01":"2026-06-01"]
    near_term_enso = smooth_enso_for_forecast(enso_df_raw, near_term_enso_brut, window=3)
    forecast_near = forecast_from_enso_calendar(model, X.columns.tolist(), dataset, lag, near_term_enso)

    forecast_all = pd.concat([forecast_near, forecast_c3s])
    forecast_all_revise = pd.concat([forecast_near, forecast_c3s_revise])

    # -- Table des bornes ENSO (Q0/P05/P25/P75/P95/Q100) alignée sur l'index
    #    complet de forecast_all_revise -- 0 d'écart sur les mois déjà
    #    observés (near-term, pas d'incertitude ENSO puisque l'ENSO est
    #    connu), bornes réelles sur la période projetée (C3S). Sert à la
    #    fois au graphique scénario-vs-contrefactuel et aux visualisations
    #    probabilistes (heatmap + distribution) -- P25/P75 inclus pour que
    #    _combined_uncertainty_samples ancre sa fonction quantile sur 7
    #    points (Q0-P05-P25-mediane-P75-P95-Q100) et pas seulement 5 --
    central_col = forecast_all_revise['gmst_anom_pred_preind']
    enso_bounds_df = pd.DataFrame(index=forecast_all_revise.index)
    for name, fdf in (('q0', forecast_c3s_q0), ('p05', forecast_c3s_bas),
                       ('p25', forecast_c3s_p25), ('p75', forecast_c3s_p75),
                       ('p95', forecast_c3s_haut), ('q100', forecast_c3s_q100)):
        col = fdf['gmst_anom_pred_preind'].reindex(forecast_all_revise.index)
        col = col.fillna(central_col)  # mois near-term (hors C3S) -> pas d'incertitude ENSO
        enso_bounds_df[name] = col

    # ==================================================================
    # ATTRIBUTION : décomposition trend / ENSO / saisonnier de la
    # prévision (§ 3.3 du manuscrit), + contrefactuel "ENSO neutre"
    # (Niño 3.4 = 0 sur toute la période) pour comparaison directe.
    # Le modèle étant linéaire (Ridge), la décomposition est EXACTE :
    # trend + enso + seasonal = total, à l'arrondi flottant près.
    # ==================================================================
    enso_full_calendar = pd.concat([near_term_enso, enso_c3s_revise])
    decomp = decompose_forecast_enso_calendar(model, X.columns.tolist(), dataset, lag, enso_full_calendar)

    enso_neutral_calendar = enso_full_calendar * 0.0
    forecast_neutral = forecast_from_enso_calendar(model, X.columns.tolist(), dataset, lag, enso_neutral_calendar)

    # Vérification croisée : total(t) - trend(t) - seasonal(t) == enso(t)
    # et total(t) avec ENSO=0 (contrefactuel) == trend(t) + seasonal(t)
    check_a = (decomp['total'] - decomp['trend'] - decomp['seasonal'] - decomp['enso']).abs().max()
    check_b = (forecast_neutral['gmst_anom_pred_preind'] - (decomp['trend'] + decomp['seasonal'])).abs().max()
    print(f"\n[Vérification décomposition] écart max décomposition/total : {check_a:.2e}°C "
          f"| max discrepancy counterfactual/(trend+seasonal): {check_b:.2e}C (should be ~0)")

    pic_date = decomp['total'].idxmax()
    print(f"\n--- Attribution au pic du scénario ({pic_date:%B %Y}) ---")
    print(f"Prédiction totale         : {decomp.loc[pic_date, 'total']:+.3f}°C")
    print(f"  dont tendance           : {decomp.loc[pic_date, 'trend']:+.3f}°C "
          f"({100*decomp.loc[pic_date, 'trend']/decomp.loc[pic_date, 'total']:.0f}%)")
    print(f"  dont ENSO               : {decomp.loc[pic_date, 'enso']:+.3f}°C "
          f"({100*decomp.loc[pic_date, 'enso']/decomp.loc[pic_date, 'total']:.0f}%)")
    print(f"  dont cycle saisonnier   : {decomp.loc[pic_date, 'seasonal']:+.3f}°C "
          f"({100*decomp.loc[pic_date, 'seasonal']/decomp.loc[pic_date, 'total']:.0f}%)")
    print(f"Contrefactuel ENSO neutre : {forecast_neutral.loc[pic_date, 'gmst_anom_pred_preind']:+.3f}°C "
          f"(écart au scénario : {decomp.loc[pic_date, 'total'] - forecast_neutral.loc[pic_date, 'gmst_anom_pred_preind']:+.3f}°C, "
          f"= contribution ENSO)")

    # -- Incertitude autour de la décomposition : bootstrap par blocs mobiles
    #    des résidus d'entraînement (Künsch, 1989), n=1000 répliques,
    #    blocs de 12 mois (préserve l'autocorrélation), alpha fixe (cf. la
    #    même logique que pour les résidus walk-forward). Seed fixé = 42
    #    pour la reproductibilité malgré la composante aléatoire du bootstrap.
    boot = bootstrap_attribution_uncertainty(model, X.columns.tolist(), X_train, y_train,
                                              dataset, lag, enso_full_calendar,
                                              n_boot=1000, block_size=12, seed=42)

    # -- Synthèse mensuelle complète (§ demandé) : décomposition + IC 90% --
    synth_attrib = decomp.join(boot)
    synth_attrib['enso_pct'] = 100 * synth_attrib['enso'] / synth_attrib['total']
    synth_attrib = synth_attrib[['total', 'trend', 'enso', 'enso_pct', 'enso_p5', 'enso_p95',
                                  'enso_pct_p5', 'enso_pct_p95', 'seasonal']]
    print("\n--- Synthèse mensuelle de l'attribution (juillet 2026 - juin 2027) ---")
    print("(ENSO in C and % with 90% interval [P5-P95], block bootstrap, n=1000)")
    with pd.option_context('display.float_format', '{:+.3f}'.format):
        print(synth_attrib)

    synth_attrib.to_csv("attribution_synthese_mensuelle.csv")
    decomp.to_csv("attribution_decomposition.csv")
    forecast_neutral.to_csv("attribution_contrefactuel_enso_neutre.csv")

    # -- Tableau visuel de synthèse mensuelle (même style que plot_monthly_table) --
    n_rows_at = len(synth_attrib)
    header_inches_at = 1.05
    fig_height_at = max(3.2, 0.34 * n_rows_at + 0.5) + header_inches_at
    fig, ax = plt.subplots(figsize=(13, fig_height_at))
    ax.axis('off')
    plt.rcParams['font.family'] = 'serif'

    col_labels_at = ['Mois', 'Total (°C)', 'Tendance (°C)', 'ENSO (°C) [IC 90%]', 'ENSO (%) [IC 90%]']
    cell_text_at = []
    for m, row in synth_attrib.iterrows():
        cell_text_at.append([
            _month_label(m).capitalize(),
            f"{row['total']:+.2f}",
            f"{row['trend']:+.2f}",
            f"{row['enso']:+.2f} [{row['enso_p5']:+.2f}, {row['enso_p95']:+.2f}]",
            f"{row['enso_pct']:.0f}% [{row['enso_pct_p5']:.0f}%, {row['enso_pct_p95']:.0f}%]",
        ])

    table_frac_height_at = min(0.95, (0.34 * n_rows_at + 0.15) / (fig_height_at - header_inches_at - 0.02))
    table_at = ax.table(cellText=cell_text_at, colLabels=col_labels_at,
                         bbox=[0.0, 1.0 - table_frac_height_at, 1.0, table_frac_height_at],
                         cellLoc='center', colLoc='center')
    table_at.auto_set_font_size(False)
    table_at.set_fontsize(9.5)
    for (r, c), cell in table_at.get_celld().items():
        cell.set_edgecolor('#dddddd')
        if r == 0:
            cell.set_facecolor('#1a1a1a')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor(light_tint(FACTUAL, 0.08) if r % 2 == 0 else 'white')

    fig.suptitle("Monthly attribution: trend share vs share of El Niño",
                  fontsize=14.5, fontweight='bold', x=0.02, ha='left', y=1 - 0.30 / fig_height_at)
    fig.text(0.02, 1 - 0.95 / fig_height_at,
              "TESR model - exact linear decomposition + 90% interval by moving-block bootstrap (n=1000, Kunsch 1989)",
              fontsize=9.5, style='italic', color='#444444', ha='left')
    fig.text(0.98, 0.01, "Données : ECMWF/Copernicus C3S — ERA5 (réf. 1850-1900)", fontsize=7.5, color='#666666', ha='right')

    top_frac_at = 1 - header_inches_at / fig_height_at
    ax.set_position([0.015, 0.02, 0.97, top_frac_at - 0.02])
    plt.savefig("gmst_attribution_monthly_table.png", dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.show()
    plt.rcParams['font.family'] = 'sans-serif'

    # -- Figure : scénario (avec El Niño) vs contrefactuel (ENSO neutre) --
    plt.rcParams['font.family'] = 'serif'
    header_inches_fig = 1.35  # espace réservé au titre (2 lignes) + sous-titre (2 lignes)
    fig_height_fig = 7.4 + header_inches_fig  # agrandi pour loger l'enveloppe ENSO + IC
    fig, ax = plt.subplots(figsize=(11.5, fig_height_fig))

    # -- Enveloppe de dispersion multi-modèles ENSO (mêmes couleurs/logique
    #    que le graphique par épisode de plot_episodes_v7.py : famille
    #    bleu/gris, disjointe du rouge/orange utilisé pour la contribution
    #    El Niño, + bornes Q0/Q100 tracées comme lignes individualisées
    #    (projection la plus basse / la plus haute) --
    env = enso_bounds_df.reindex(decomp.index)
    ax.fill_between(decomp.index, env['q0'], env['q100'], color=COUNTERFACT_FILL, alpha=0.9, lw=0,
                     zorder=1, label='Multi-model ENSO spread (Q0-Q100)')
    ax.fill_between(decomp.index, env['p05'], env['p95'], color=COUNTERFACT_LIGHT, alpha=0.75, lw=0,
                     zorder=2, label='Multi-model ENSO spread (Q5-Q95)')
    ax.plot(decomp.index, env['q0'], color=COUNTERFACT, lw=1.0, ls=':', zorder=2.5, alpha=0.9,
            label='Lowest / highest projection (Q0 / Q100)')
    ax.plot(decomp.index, env['q100'], color=COUNTERFACT, lw=1.0, ls=':', zorder=2.5, alpha=0.9)
    # -- Annotation explicite du minimum et du maximum absolus de l'enveloppe --
    i_min, i_max = env['q0'].idxmin(), env['q100'].idxmax()
    ax.scatter([i_min], [env['q0'].min()], marker='v', s=60, color=COUNTERFACT,
               edgecolor='#1a1a1a', linewidth=0.6, zorder=7)
    ax.scatter([i_max], [env['q100'].max()], marker='^', s=60, color=RECORD,
               edgecolor='#1a1a1a', linewidth=0.6, zorder=7)
    ax.annotate(f"Projected min: {env['q0'].min():+.2f}C", xy=(i_min, env['q0'].min()),
                xytext=(0, 10), textcoords='offset points', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold', color=COUNTERFACT,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.75))
    ax.annotate(f"Projected max: {env['q100'].max():+.2f}C", xy=(i_max, env['q100'].max()),
                xytext=(0, 8), textcoords='offset points', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold', color=RECORD,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.75))

    ax.plot(decomp.index, decomp['total'], color=FACTUAL_DARK, lw=2.2, marker='o', ms=4, zorder=5,
            label='Central scenario (record El Nino)')
    ax.plot(forecast_neutral.index, forecast_neutral['gmst_anom_pred_preind'], color='#555555',
            lw=2.0, ls='--', marker='o', ms=4, zorder=5, label='Contrefactuel (ENSO neutre, Niño 3.4 = 0)')
    # -- Contribution El Niño en hachures (pas un aplat semi-transparent) :
    #    reste identique quel que soit le fond, donc ne se confond jamais
    #    avec les teintes de l'enveloppe ni ne les fait paraître plus ou
    #    moins saturées selon la zone recouverte --
    ax.fill_between(decomp.index, forecast_neutral['gmst_anom_pred_preind'], decomp['total'],
                     facecolor='none', edgecolor=HIGHLIGHT, hatch='////', linewidth=0.0,
                     zorder=3, label='El Nino contribution (hatched area)')
    ax.fill_between(decomp.index, forecast_neutral['gmst_anom_pred_preind'], decomp['total'],
                     color=HIGHLIGHT, alpha=0.12, lw=0, zorder=2.8)
    ax.axhline(1.5, color='gray', ls=':', lw=1, label='+1.5C threshold (Paris Agreement)')
    ax.axhline(2.0, color='gray', ls='--', lw=1, label='Seuil +2°C')
    # -- Marge explicite incluant l'enveloppe ENSO --
    y_top_data = max(decomp['total'].max(), 2.0, env['q100'].max())
    y_bottom_data = min(forecast_neutral['gmst_anom_pred_preind'].min(), env['q0'].min())
    y_span = y_top_data - y_bottom_data
    ax.set_ylim(y_bottom_data - 0.12 * y_span, y_top_data + 0.10 * y_span)
    ax.set_ylabel("Global mean surface temperature anomaly (C) [1850-1900 baseline]")
    ax.xaxis.set_major_formatter(FuncFormatter(_month_axis_formatter))
    fig.autofmt_xdate(rotation=45)
    ax.legend(loc='upper left', fontsize=8.5, framealpha=0.9, ncol=2)
    ax.grid(alpha=0.3)
    fig.suptitle("Attribution: El Nino contribution to the projected GMSTA anomaly\n"
                 "July 2026 - June 2027, TESR model", fontsize=13.5, fontweight='bold',
                 x=0.02, ha='left', y=1 - 0.35 / fig_height_fig)
    fig.text(0.02, 1 - 1.05 / fig_height_fig,
              "Exact linear decomposition (ridge): central scenario vs ENSO-neutral counterfactual;\n"
              "dotted envelope = Q0/Q100 bounds (multi-model ENSO spread, most "
              "basse/haute) ; zones bleues = Q0-Q100 / Q5-Q95",
              fontsize=9, style='italic', color='#444444', ha='left', va='top')
    fig.text(0.98, 0.01, "Data: ECMWF/Copernicus C3S - ERA5 (1850-1900 baseline)", fontsize=7.5, color='#666666', ha='right')
    fig.subplots_adjust(top=1 - header_inches_fig / fig_height_fig, bottom=0.14)
    plt.savefig("gmst_attribution_enso_vs_neutre.png", dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.show()
    plt.rcParams['font.family'] = 'sans-serif'

    zoom = ("2022-01-01", "2027-07-31")
    debut, fin = pd.to_datetime(zoom[0]).year, pd.to_datetime(zoom[1]).year
    title = (f"Evolution of the global mean surface temperature anomaly relative to "
             f"preindustrial over the period {debut}-{fin},\nmedian ENSO scenario "
             f"multi-model (Climate Dashboard, init. 10/08/2026, SINTEX-F excluded)")
    avertissement = ("Hypothesis of an El Nino outside the range of variability tested and validated by the model")

    plot_forecast_zoom(
        dataset, forecast_all_revise,
        model_fit=model_fit_full,
        residuals=residuals,
        zoom=zoom,
        filename="gmst_forecast_c3s_scenario.png",
        title=title,
        subtitle=f"TESR model (quadratic trend + smoothed ENSO + seasonal + ridge), lag={lag} months; ERA5/Climate Dashboard, ref. 1850-1900",
        legend_loc='between_thresholds',
        scenario_warning=avertissement,
    )

    # ==================================================================
    # IMAGE : VUE D'ENSEMBLE -- toute la série observed (1940 -> auj.)
    # + la prévision, sans zoom, pour voir la trajectoire complète en un
    # coup d'oeil (contexte long terme du réchauffement + scénario ENSO).
    # ==================================================================
    debut_full = dataset.index[0].year
    plot_forecast_zoom(
        dataset, forecast_all_revise,
        residuals=residuals,
        zoom=None,
        filename="gmst_forecast_overview_full.png",
        title=(f"Global mean surface temperature anomaly relative to "
               f"preindustrial,\nover the period {debut_full}-{fin} and projections through July 2027\n\nModelling based on seasonal forecasts initialized 1 August 2026"),
        subtitle=f"TESR model, lag={lag} months; ERA5/C3S, 1850-1900 baseline",
        legend_loc='between_thresholds',
        scenario_warning=avertissement,
    )

    # -- Même vue d'ensemble, déclinée en version "media" (cf.
    #    plot_forecast_overview_media) -- pensée pour une reprise presse ou
    #    réseaux sociaux : sans perte d'information par rapport à la version
    #    ci-dessus, juste une mise en page épurée (étiquettes directes,
    #    pas de cadre de légende, hiérarchie titre/sous-titre plus nette) --
    plot_forecast_overview_media(
        dataset, forecast_all_revise,
        residuals=residuals,
        filename="gmst_forecast_overview_media.png",
        title="Global mean surface temperature anomaly",
        subtitle=(f"Departure from preindustrial (1850-1900) - C - TESR model, lag={lag} months; "
                  f"ERA5/C3S - projection initialized 1 August 2026"),
        scenario_warning=avertissement,
    )

    # ==================================================================
    # VISUALISATIONS PROBABILISTES : tableau de dépassement de seuils +
    # distribution pour le mois le plus chaud. Utilise les résidus
    # walk-forward réels du modèle (déjà calculés plus haut).
    # ==================================================================
    proba_table = plot_probability_heatmap(
        forecast_all_revise, residuals, enso_bounds_df=enso_bounds_df,
        title="Probability of exceeding +1.5C and +2C of global warming",
        subtitle="TESR model - empirical probability - multi-model ENSO spread - departure from preindustrial (1850-1900) - C\nJuly 2026 - August 2027 - projection initialized 1 August 2026",
    )
    print("\nTableau de probabilité (%) :")
    print(proba_table.round(1))

    warmest_month_revise = forecast_all_revise['gmst_anom_pred_preind'].idxmax()
    warmest_val_revise = forecast_all_revise['gmst_anom_pred_preind'].max()
    warmest_bounds = (enso_bounds_df.loc[warmest_month_revise]
                       if warmest_month_revise in enso_bounds_df.index else None)
    plot_probability_distribution(
        warmest_val_revise, residuals,
        month_label=_month_label(warmest_month_revise),
        enso_bounds=warmest_bounds,
    )

    # ==================================================================
    # (1) SYNTHÈSE VISUELLE : à quel point El Niño 2027 accentue la
    # probabilité de franchir +1,5°C / +2°C, par rapport à un scénario
    # ENSO neutre -- réutilise `decomp`/`forecast_neutral`/`pic_date`
    # déjà calculés plus haut (§ attribution).
    # ==================================================================
    proba_amplification = plot_enso_amplification(
        decomp, forecast_neutral, residuals,
        enso_bounds_df=enso_bounds_df, peak_date=pic_date,
    )
    print("\nAmplification El Niño vs ENSO neutre (probabilité de dépassement, %) :")
    print(proba_amplification.round(1))

    # ==================================================================
    # (2) SYNTHÈSE VISUELLE : distribution en cloche du climat actuel
    # (2016-2026, observations réelles) comparée à la distribution
    # projetée du mois de pic (mars 2027) -- mesure directe du caractère
    # inhabituel du pic par rapport à la variabilité récente.
    # ==================================================================
    climat_shift = plot_climate_baseline_vs_peak(
        gmst_df, forecast_all_revise, residuals,
        enso_bounds_df=enso_bounds_df, baseline_period=("2016-01-01", "2026-12-31"),
        peak_date=pic_date,
    )
    print(f"\nMars 2027 (pic) : {climat_shift['central_peak']:+.2f}°C, soit "
          f"{climat_shift['z_score']:+.1f} sigma above the 2016-2026 current climate "
          f"(percentile empirique ~{climat_shift['percentile_empirique_baseline']:.0f}%).")

    # -- Synthèse mensuelle (extrêmes + probabilités) et rendu tableau --
    synthesis = compute_forecast_synthesis(forecast_all_revise, residuals)
    print("\nSynthèse mensuelle (extrêmes p5/p95 + probabilités) :")
    print(synthesis.round(2))
    plot_monthly_table(synthesis)

    # NB : les bandes de percentiles (extrêmes) sont désormais intégrées
    # directement dans le graphique du scénario C3S ci-dessus (paramètre
    # residuals=...) plutôt que dans un fan chart séparé -- redondant avec
    # ce graphique. plot_fan_chart() reste disponible dans le module si
    # besoin d'une vue autonome.

    # Mode "mix" vérif/prévision : si tu as déjà les observations réelles
    # pour une partie de la période projetée, superpose-les pour comparer :
    #
    # obs_reelles = gmst_df['gmst_anom_preind'].loc["2026-07-01":"2026-09-30"]
    # plot_forecast_zoom(dataset, forecast_all, obs_future=obs_reelles,
    #                     zoom=zoom, title="GMST - verification vs forecast (mixed)")
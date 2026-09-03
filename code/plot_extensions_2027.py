"""
plot_extension_dec2027.py
Deux fonctions à intégrer à votre pipeline (enso_gmst_model.py / plot_episodes_v7.py) :

  1. plot_temp_extended_dec2027()  -- figure TEMP (style Fig. 12/13 du manuscrit),
     étendue à Décembre 2027, plage complète = union des IC 90% des 3 hypothèses H2 2027.

  2. plot_oni_extended_dec2027()   -- figure ONI (style Fig. 9), étendue à Décembre 2027,
     bandes de quantiles conservées sur toute la période officielle (Août 2026-Avril 2027)
     ET sur l'extension (Mai-Décembre 2027), avec palette de couleur DISTINCTE pour
     bien signaler la partie non officielle (hypothèses d'auteur).

Dépendances : pandas, numpy, matplotlib. Ce fichier est maintenant AUTONOME :

  - `model_fit` : construit par `build_model_fit(model, X, dataset, enso_df_raw,
    gmst_df, lag)` (section 0 bis ci-dessous) à partir des objets RÉELS produits
    par le bloc d'entraînement principal de enso_gmst_model.py (`model, X, y, ... =
    fit_model(dataset, ...)`) -- ne PAS construire ce dict "à la main" avec la
    structure {model, mu, sd, feature_cols, df, nino, gmst, lag} : les noms de
    colonnes ('t_index'/'t_index2'/'enso_x_t'/'m_2'..'m_12' côté pipeline vs
    't'/'t2'/'inter'/'m2'..'m12' attendus ici) et l'attribut alpha du modèle
    ('.alpha_', pas '.alpha') ne correspondent pas terme à terme, d'où le
    NameError/AttributeError/KeyError en cascade observé si on saute cette
    étape ;
  - `h2_scenarios` : dict {'neutre':..., 'central':..., 'la_nina_forte':...} =
    sortie de build_h2_2027_scenarios() (section 0 ci-dessous, CSV embarqué) ;
  - `official_scenario_aug26_avr27`, `official_gmsta_table` : dicts du scénario
    officiel (mêmes objets que dans l'extension précédente -- toujours à fournir
    par l'appelant, pas reconstruits ici).

AVERTISSEMENT : la reconstruction des bandes de quantiles officielles
(Août 2026-Avril 2027) sur la figure ONI est une APPROXIMATION (croissance
linéaire de la largeur avec le délai de prévision, calée à l'œil sur votre
Fig. 9 originale) -- à remplacer par les vraies bornes Q0/Q5/Q25/Q75/Q95/Q100
du Climate Dashboard (fichier membres) dès qu'il est disponible dans
l'environnement d'exécution.

NOTE (titres/textes de figure) : tous les textes affichés dans les figures
(titres, sous-titres, légendes, annotations) sont en anglais, avec des
intitulés descriptifs neutres plutôt que le registre "média" de la version
initiale -- cf. plot_episodes_v7.py / extensions.py pour la même convention.
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import Ridge
from palette import FACTUAL, light_tint


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
    """Tableau mensuel ONI (Mai-Décembre 2027) comparant les 3 hypothèses
    H2 2027, directement à partir de `h2_scenarios` -- pas de recalcul."""
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
    """Tableau mensuel du scénario ONI officiel (Août 2026-Avril 2027) :
    médiane multi-modèle + bandes Q0-Q100/Q5-Q95/Q25-Q75 (APPROXIMATION,
    cf. avertissement en tête de fichier). `q0`..`q100` = les mêmes
    pd.Series déjà calculées dans plot_oni_extended_dec2027()."""
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

    h2_table_png, h2_table_csv = plot_oni_h2_2027_table(
        h2_scenarios, filename=h2_table_filename, csv_filename=h2_table_csv_filename)
    official_table_png, official_table_csv = plot_oni_official_table(
        official_scenario_aug26_avr27, q0, q5, q25, q75, q95, q100, init_date=init_date,
        filename=official_table_filename, csv_filename=official_table_csv_filename)

    return filename, h2_table_png, h2_table_csv, official_table_png, official_table_csv


# ====================================================================
# EXEMPLE D'APPEL
# ====================================================================
#
# analogs_24m = {}
# for label, y0 in [('1982/1983', 1982), ('1997/1998', 1997), ('2015/2016', 2015)]:
#     analogs_24m[label] = enso_df_raw['enso_ssta'].loc[f'{y0}-01-01':f'{y0+1}-12-01'].values
#
# -- construit une seule fois, juste après l'ajustement du modèle (model, X, ...
#    = fit_model(dataset, ...)) dans enso_gmst_model.py --
# model_fit = build_model_fit(model, X, dataset, enso_df_raw, gmst_df, lag)
#
# -- IMPORTANT : ne PAS appeler build_h2_2027_scenarios() sans anchor_offset --
#    le CSV embarqué (_H2_2027_ONI_CSV) a été calibré en supposant un dernier
#    mois officiel (avril 2027) à ~1.0 degC ; si la vraie valeur officielle
#    diffère (cas quasi certain), un anchor_offset=0.0 par défaut produit une
#    cassure abrupte entre le dernier point officiel et le premier point de
#    l'extension H2. Recaler dynamiquement sur la vraie valeur officielle :
# H2_2027_CALIBRATION_ANCHOR = 1.0  # valeur d'avril 2027 supposée par le CSV
# last_official_month = max(pd.Timestamp(k) for k in official_scenario_aug26_avr27)
# last_official_value = official_scenario_aug26_avr27[last_official_month.strftime('%Y-%m-01')]
# h2 = build_h2_2027_scenarios(anchor_offset=last_official_value - H2_2027_CALIBRATION_ANCHOR)
#
# png_path, csv_path, gmst_h2_table_png, gmst_h2_table_csv = plot_temp_extended_dec2027(
#     model_fit, official_scenario_aug26_avr27, h2, official_gmsta_table)
# oni_png, oni_h2_table_png, oni_h2_table_csv, oni_official_table_png, oni_official_table_csv = \
#     plot_oni_extended_dec2027(model_fit, official_scenario_aug26_avr27, h2, analogs_24m)
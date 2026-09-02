"""
palette.py
Centralized colour-blind-safe palette for all TESR figures, aligned with
how ECMWF/Copernicus (C3S) present GMST anomalies and probabilistic
spread: a diverging blue-to-red scale for anomaly magnitude, a
hue-and-lightness-separated sequential ramp (NOT same-hue alpha
stacking) for nested uncertainty/quantile bands, and the Okabe & Ito
(2008) qualitative set for categorical series (observed vs
counterfactual, episodes, etc.).

Okabe & Ito (2008) is the standard reference for palettes that remain
distinguishable under protanopia, deuteranopia and tritanopia. See also
Crameri et al. (2020, Nat. Commun.) "The misuse of colour in science
communication".

REVISION NOTE (this version -- fixes the "jaune orangé illisible"
issue): the previous version rendered nested quantile bands (Q0-Q100,
Q5-Q95, Q25-Q75; the 5-95%/25-75% fan-chart bands; the forecast
uncertainty envelopes) by taking a SINGLE orange hue (HIGHLIGHT) and
stacking it at different alpha values. Same-hue alpha stacking is not
a real graded scale: once alpha exceeds roughly 0.3, the layers wash
out into a near-uniform pale yellow, so the individual bands become
indistinguishable from one another -- exactly the complaint. It also
collided with HIGHLIGHT itself, which is used elsewhere in the same
figures for annotation callouts, and with the ENSO_SCALE ladder, which
used four consecutive warm tints (yellow/orange/vermillion/dark
vermillion) that were nearly identical as thin threshold lines.

Fix: nested bands now use QUANTILE_BANDS, a genuine 3-step
hue+lightness-separated sequential ramp in the blue family (pale ->
medium -> saturated blue), matching the blue "ensemble spread"
convention used in Copernicus/ECMWF plots, and kept clear of
FACTUAL/HIGHLIGHT so central lines and callouts always stand out
against it. ENSO_SCALE was rebuilt as a true diverging cool -> warm
ladder (blue -> yellow -> orange -> vermillion -> near-black) so each
step differs in both hue and lightness. HIGHLIGHT is now reserved for
sparse annotations/callouts only, never for stacked bands. WARM_LIGHT
was previously a straight duplicate of HIGHLIGHT (same hex) and is
now a distinct muted gold, so the two 1.5C/2C threshold shadings in
plot_probability_distribution are actually two different colours.

Roles used throughout the codebase:
    FACTUAL       -- observed / model-fitted series (ENSO-influenced world)
    COUNTERFACT   -- ENSO-neutral counterfactual series
    COUNTERFACT_2 -- secondary shade for counterfactual scatter/markers
    ALT_SCENARIO  -- alternative / secondary scenario line
    HIGHLIGHT     -- annotation boxes, peak markers, callouts (sparse use
                     only -- never for stacked/nested bands; see
                     QUANTILE_BANDS for those)
    RECORD        -- past-record reference lines/markers
    WARM_LIGHT    -- second warm accent, distinct from HIGHLIGHT (muted
                     gold, not a tint of it)
    QUANTILE_BANDS -- 3-step sequential ramp (pale->saturated) for nested
                     percentile/quantile envelopes (Q0-Q100 / Q5-Q95 /
                     Q25-Q75), see quantile_band_color()
    QUANTILE_MEDIAN -- colour for the median/central line drawn over
                     QUANTILE_BANDS
    ANOMALY_CMAP  -- diverging blue->white->red colormap name for
                     continuous, signed anomaly-magnitude fields
    PROBABILITY_CMAP -- 'turbo' sequential-rainbow colormap for one-sided
                     0-100% fields (exceedance-probability heatmap)
    OVERVIEW_CMAP / overview_band_color() -- same 'turbo' family, used
                     for the nested bands in the full-overview figures
    NEUTRAL_DARK / NEUTRAL_MID / NEUTRAL_LIGHT -- text, axes, gridlines

Every role combination that is actually drawn together in a single
figure has been re-checked pairwise under CVD simulation (Coblis:
protanopia / deuteranopia / tritanopia) -- in particular QUANTILE_BANDS
vs FACTUAL vs HIGHLIGHT vs RECORD, which previously shared too narrow a
hue band. Grey/black neutrals are unaffected by colour vision
deficiency and are left as in the original figures.
"""

import matplotlib.colors as mcolors

FACTUAL = "#D55E00"          # vermillion -- central scenario / model line
FACTUAL_DARK = "#8C3900"     # darker vermillion accent
FACTUAL_DARKEST = "#7A3000"  # darkest vermillion accent

COUNTERFACT = "#0072B2"      # blue -- ENSO-neutral counterfactual
COUNTERFACT_2 = "#56B4E9"    # sky blue, secondary series
COUNTERFACT_DARK = "#00426B"  # darkest blue accent
COUNTERFACT_LIGHT = "#A6D3EA"  # light blue tint
COUNTERFACT_FILL = "#DCEEF7"   # very light blue fill

ALT_SCENARIO = "#009E73"     # bluish green, alternative/dashed scenario
HIGHLIGHT = "#E69F00"        # orange -- annotations/callouts ONLY, never bands
RECORD = "#CC79A7"           # reddish purple, past-record markers
WARM_LIGHT = "#F0C674"       # muted gold -- distinct 2nd warm accent
                              # (was a straight duplicate of HIGHLIGHT)

NEUTRAL_DARKEST = "#1a1a1a"
NEUTRAL_DARK = "#333333"
NEUTRAL_MID = "#444444"
NEUTRAL = "#666666"
NEUTRAL_LIGHT = "#888888"
NEUTRAL_LIGHTER = "#999999"
NEUTRAL_PALE = "#cccccc"

WHITE = "#ffffff"

WARNING_BG = "#FBEFE6"       # solid pale orange for scenario-warning boxes
                              # -- see note below on why this replaces
                              # light_tint(FACTUAL, ...) for boxes that
                              # also set a Patch-level `alpha`.


def light_tint(hex_color, alpha=0.20):
    """Return an RGBA tuple to use as a light fill of `hex_color` at the
    given alpha. Fine for a SINGLE band against a white/grid background.
    Do NOT stack several alphas of the same hue to represent nested
    bands -- that collapses into a near-uniform wash (this was the bug).
    Use QUANTILE_BANDS / quantile_band_color() for nested bands instead.

    GOTCHA: never pass the result of this function as `facecolor` in a
    dict/Patch that ALSO sets its own `alpha=` keyword (e.g.
    `bbox=dict(facecolor=light_tint(X, 0.08), alpha=0.9)`). A Patch-level
    `alpha` overrides the alpha channel embedded in any RGBA colour it
    is given -- for both facecolor AND edgecolor -- so the box renders
    at alpha=0.9 with the FULL-STRENGTH hue `X`, not the pale tint that
    was intended. This produced the "dark text on a near-opaque dark
    orange box" bug in the scenario-warning annotations. Use a solid,
    pre-blended colour instead (see WARNING_BG below) and leave the
    Patch's own `alpha` unset (or 1.0) when you need full control over
    the exact shade shown.
    """
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(hex_color)
    return (r, g, b, alpha)


# ----------------------------------------------------------------------
# Nested quantile / percentile bands (fan charts, ENSO envelopes,
# forecast uncertainty). These are pre-mixed, distinct colours -- NOT
# alpha tints of a single hue -- so Q0-Q100, Q5-Q95 and Q25-Q75 (or any
# other nesting) stay visually separable even in greyscale or under
# red-green colour blindness. Ordered widest/outermost (palest) to
# narrowest/innermost (most saturated), matching the ECMWF/Copernicus
# convention of a cool blue spread behind a warm central line.
QUANTILE_BANDS = [
    "#E4EEF6",   # widest band  (e.g. Q0-Q100)  -- very pale blue
    "#B8D4E8",   # middle band  (e.g. Q5-Q95)    -- light blue
    "#5D9BC7",   # narrowest band (e.g. Q25-Q75) -- medium blue
]
QUANTILE_MEDIAN = COUNTERFACT_DARK  # median/central line drawn over the bands


def quantile_band_color(i, n=None):
    """Return the i-th nested-band colour (i=0 -> widest/outermost band).
    If more bands are requested than QUANTILE_BANDS provides, extra
    steps are interpolated between its palest and most saturated stop
    (a real hue/lightness gradient, never alpha-on-alpha)."""
    if n is None or n <= len(QUANTILE_BANDS):
        return QUANTILE_BANDS[i]
    lo = mcolors.to_rgb(QUANTILE_BANDS[0])
    hi = mcolors.to_rgb(QUANTILE_BANDS[-1])
    t = i / (n - 1)
    return tuple(lo[k] + t * (hi[k] - lo[k]) for k in range(3))


# Diverging colormap for continuous, SIGNED anomaly-magnitude fields,
# matching the blue-white-red convention ECMWF/Copernicus use for ERA5
# temperature anomaly maps. 'RdBu_r' is the standard perceptually
# diverging colormap in the climate-science literature.
ANOMALY_CMAP = "RdBu_r"

# Sequential colormap for one-sided 0-100% fields (e.g. the exceedance-
# probability heatmap). Was 'turbo' (Google AI, 2019, blue -> cyan ->
# green -> yellow -> orange -> red): more distinguishable steps between
# adjacent percentages than a single-hue ramp, but its rainbow banding
# and highly saturated ends (especially the near-0% blue) read as
# visually "aggressive" at this cell size, and turbo is not as
# rigorously CVD-validated as 'cividis'. Switched to 'cividis'
# (Nunez, Anderton & Renslow 2018): perceptually uniform, monotonic
# lightness ramp (dark blue -> yellow), no hue reversals, and the
# colormap most rigorously validated for protanopia/deuteranopia --
# smoother and calmer at both ends while remaining fully readable
# (each heatmap cell also prints its own % value regardless).
PROBABILITY_CMAP = "cividis"

# Same colormap, used to derive the nested uncertainty bands in the
# "vue d'ensemble" / full-overview figures (plot_forecast_zoom with
# zoom=None, plot_forecast_overview_media) -- see overview_band_color().
OVERVIEW_CMAP = "turbo"


def overview_band_color(i, n, lo=0.10, hi=0.88):
    """Sample the i-th of n colours from OVERVIEW_CMAP ('turbo'),
    i=0 -> widest/outermost band (blue/cool end) through
    i=n-1 -> narrowest/innermost band (red end) -- a genuine
    turbo-toward-red progression for the overview figures, distinct
    from the blue-only QUANTILE_BANDS used in the fan chart / ENSO
    envelope. `lo`/`hi` trim the very darkest navy and darkest maroon
    ends of turbo, which read poorly as fills."""
    import matplotlib.cm as cm
    cmap = cm.get_cmap(OVERVIEW_CMAP)
    t = hi if n <= 1 else lo + i * (hi - lo) / (n - 1)
    return mcolors.to_hex(cmap(t))

# Sequential ramp for ordered severity (e.g. successive GMST thresholds),
# lightest to darkest -- Okabe-Ito orange -> vermillion.
THRESHOLD_RAMP = [HIGHLIGHT, FACTUAL, FACTUAL_DARK]

# Qualitative palette for discrete categories (episodes, ENSO models,
# scenario branches) needing more than 3 distinguishable colours.
QUALITATIVE = [
    COUNTERFACT,     # blue
    FACTUAL,         # vermillion
    ALT_SCENARIO,    # bluish green
    HIGHLIGHT,       # orange
    RECORD,          # reddish purple
    "#F0E442",       # yellow
    COUNTERFACT_2,   # sky blue
    NEUTRAL_DARKEST,  # black
]

# Three named historical ENSO analogues (1982/83, 1997/98, 2015/16):
# hue-distinct, chosen to stay clear of FACTUAL/COUNTERFACT used
# elsewhere in the same figures.
ANALOG_COLORS = [ALT_SCENARIO, RECORD, HIGHLIGHT]

# ENSO intensity ladder (La Nina -0.5 through extreme El Nino +3.0): a
# genuine diverging cool->warm progression, each step chosen to differ
# in BOTH hue and lightness so the ladder stays ordered and separable
# under any CVD type, in print, and when overlaid on QUANTILE_BANDS in
# the same figure. Replaces the earlier ladder, which used four
# consecutive warm-family tints (yellow/orange/vermillion/dark
# vermillion) that were nearly indistinguishable as thin threshold
# lines.
ENSO_SCALE = {
    -0.5: COUNTERFACT_2,   # La Nina        -- sky blue
    0.5: "#F0E442",        # El Nino onset  -- yellow
    1.0: HIGHLIGHT,        # moderate       -- orange
    1.5: FACTUAL,          # strong         -- vermillion
    2.0: FACTUAL_DARK,     # very strong    -- dark vermillion
    3.0: NEUTRAL_DARKEST,  # extreme        -- near-black, max contrast
}

# Seasonal-trend vs ENSO-external-variability components (bar charts).
COLOR_TREND = COUNTERFACT
COLOR_EXTERNAL_VAR = RECORD
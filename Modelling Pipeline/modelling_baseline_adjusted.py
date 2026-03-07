import re

import pandas as pd
import pycountry

# ── Step 1: PTA aggregation from Chemicals & Allied Industries tariff data ────
# Each row is a reporter-partner-product tariff record.
# We aggregate to the reporter (country) level to get a country-level PTA indicator.

trade = pd.read_csv(
    r"C:\Users\srima\OneDrive\Desktop\INSY674\Individual Project 1\Data\Chemicals_Allied_Industries.csv"
)

pta_agg = (
    trade.groupby("reporteriso3")["PTA"]
    .agg(pta_share="mean", has_pta="max")   # share of partners with PTA; binary flag
    .reset_index()
    .rename(columns={"reporteriso3": "country_iso3"})
)
pta_agg["has_pta"] = pta_agg["has_pta"].astype(int)

print("Step 1 — PTA aggregation (by reporter country):")
print(pta_agg.shape)
print(pta_agg.head(10).to_string())


# ── Step 2: Pharma-specific PTA start year from WTO-X dataset ────────────────
# The WTO-X dataset codes which PTAs contain health, IPR, consumer protection,
# and data protection provisions — i.e., provisions directly relevant to pharma.
# We filter to agreements with any of these provisions, parse the member countries
# from agreement names, and take the earliest entry-into-force year per country.
# Countries not matched (no pharma-relevant PTA) are treated as never-treated in DiD.

WTO_PLUS_PATH = (
    r"C:\Users\srima\OneDrive\Desktop\INSY674\Individual Project 1\Data\pta-agreements_1.xls"
)
PHARMA_COLS = ["Health", "IPR", "ConsumerProtection", "DataProtection"]

wto_x = pd.read_excel(WTO_PLUS_PATH, sheet_name="WTO-X AC")
pharma_mask = wto_x[PHARMA_COLS].ge(1).any(axis=1)
pharma = wto_x[pharma_mask][["Agreement", "year"]].copy()
pharma["year"] = pd.to_numeric(pharma["year"], errors="coerce")
pharma = pharma.dropna(subset=["year"])
pharma["year"] = pharma["year"].astype(int)

# Multilateral blocs → member ISO3 lists
BLOC_MAP = {
    "asean":     ["BRN", "KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP", "THA", "VNM"],
    "cefta":     ["ALB", "BIH", "MKD", "MDA", "MNE", "SRB"],
    "comesa":    ["BDI", "COM", "COD", "DJI", "EGY", "ERI", "ETH", "KEN", "MDG", "MWI",
                  "MUS", "MOZ", "RWA", "SOM", "SDN", "TZA", "UGA", "ZMB", "ZWE"],
    "cafta":     ["CRI", "SLV", "GTM", "HND", "NIC", "DOM", "USA"],
    "cez":       ["CHN", "HKG"],
    "ec":        ["AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
                  "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
                  "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"],
    "efta":      ["ISL", "LIE", "NOR", "CHE"],
    "gcc":       ["BHR", "KWT", "OMN", "QAT", "SAU", "ARE"],
    "cariforum": ["ATG", "BHS", "BRB", "BLZ", "DMA", "DOM", "GRD", "GUY", "HTI", "JAM",
                  "KNA", "LCA", "VCT", "SUR", "TTO"],
    "sadc":      ["AGO", "BWA", "COD", "LSO", "MDG", "MWI", "MUS", "MOZ", "NAM", "ZAF",
                  "SWZ", "TZA", "ZMB", "ZWE"],
    "mercosur":  ["ARG", "BRA", "PRY", "URY"],
    "andean":    ["BOL", "COL", "ECU", "PER"],
}

NAME_OVERRIDES = {
    "korea": "KOR", "uk": "GBR", "usa": "USA", "uae": "ARE",
    "hong kong": "HKG", "macao": "MAC", "taiwan": "TWN",
    "russia": "RUS", "vietnam": "VNM", "iran": "IRN",
    "dr": "DOM", "turkey": "TUR",
}


def name_to_iso3(token: str) -> str | None:
    token = token.strip().lower()
    if token in NAME_OVERRIDES:
        return NAME_OVERRIDES[token]
    try:
        return pycountry.countries.search_fuzzy(token)[0].alpha_3
    except Exception:
        return None


rows = []
for _, row in pharma.iterrows():
    agr = row["Agreement"].lower().strip()
    year = row["year"]
    parts = re.split(r"\s*[-–]\s*", agr)
    countries_found = set()

    for part in parts:
        part = part.strip()
        matched_bloc = False
        for bloc, members in BLOC_MAP.items():
            if bloc in part:
                countries_found.update(members)
                matched_bloc = True
                break
        if not matched_bloc:
            iso = name_to_iso3(part)
            if iso:
                countries_found.add(iso)

    for iso in countries_found:
        rows.append({"country_iso3": iso, "pta_pharma_year": year})

pta_pharma_start = (
    pd.DataFrame(rows)
    .groupby("country_iso3")["pta_pharma_year"]
    .min()
    .reset_index()
    .rename(columns={"pta_pharma_year": "pta_pharma_start_year"})
)

print("\nStep 2 — Pharma-specific PTA start year per country (WTO-X: Health/IPR/ConsumerProtection/DataProtection):")
print(pta_pharma_start.shape)
print(pta_pharma_start.sort_values("pta_pharma_start_year").head(10).to_string())

# ── Step 3: Merge PTA variables into pivot_dataset ───────────────────────────
# country_iso3 is kept in feature_engineering.py (removed from the drop list).
# Left-join so all panel rows are preserved; unmatched countries get NaN.

df = pd.read_csv(
    r"C:\Users\srima\OneDrive\Desktop\INSY674\Individual Project 1\pivot_dataset_fe.csv"
)

df = (
    df
    .merge(pta_agg,         on="country_iso3", how="left")   # has_pta, pta_share
    .merge(pta_pharma_start, on="country_iso3", how="left")  # pta_pharma_start_year
)

# Countries with no pharma PTA in WTO-X → never treated; fill start year with 0
df["pta_pharma_start_year"] = df["pta_pharma_start_year"].fillna(0).astype(int)

# DiD treatment indicator: 1 if the country has a pharma PTA *and* the current year
# is at or after the agreement's entry-into-force year
df["pta_treated"] = (
    (df["has_pta"] == 1) &
    (df["pta_pharma_start_year"] > 0) &
    (df["year"] >= df["pta_pharma_start_year"])
).astype(int)

# Years since pharma PTA entry into force (analogous to years_since_intro for vaccines)
# Pre-treatment and never-treated countries get 0; clipped at 0 so no negatives
df["years_since_pta"] = (
    (df["year"] - df["pta_pharma_start_year"])
    .clip(lower=0)
)
# Never-treated countries (start year == 0) should stay 0, not get a spurious count
df.loc[df["pta_pharma_start_year"] == 0, "years_since_pta"] = 0

print("\nStep 3 — Merged pivot dataset:")
print(df.shape)
print(df[["country_iso3", "year", "has_pta", "pta_share", "pta_pharma_start_year", "pta_treated"]].head(15).to_string())
print("\npta_treated distribution:", df["pta_treated"].value_counts().to_dict())
print("Missing has_pta:", df["has_pta"].isna().sum(), "rows")


#Negative relative time values are pre-treatment periods, zero is the treatment year, and positive values are post-treatment periods. This variable can be used to estimate event-study style DiD models, where we can flexibly estimate the treatment effect at each relative time period compared to the baseline (relative_time=0). It also allows us to check for pre-trends by examining the coefficients on negative relative time periods.
df['relative_time'] = df['year'] - df['pta_pharma_start_year']
df.loc[df['pta_pharma_start_year'] == 0, 'relative_time'] = float('nan')

# ── Step 4: Staggered DiD — Athey-Imbens three-group design ─────────────────
import matplotlib.pyplot as plt
import pyfixest as pf

PANEL_START = 2001

# ── 4a: Classify treatment groups ────────────────────────────────────────────
# Always-treated  (G_i < 2001): PTA predates panel → no pre-period observable
#                               → pooled into control group (gname = 0)
# Staggered       (G_i 2001-2023): PTA adopted within panel → main identification
# Never-treated   (G_i = 0):  no pharma PTA → pure control (gname = 0)
df['treatment_group'] = 'never_treated'
df.loc[
    (df['pta_pharma_start_year'] > 0) &
    (df['pta_pharma_start_year'] < PANEL_START),
    'treatment_group'
] = 'always_treated'
df.loc[
    df['pta_pharma_start_year'] >= PANEL_START,
    'treatment_group'
] = 'staggered'

print('\nStep 4a — Treatment group counts (unique countries):')
print(df.groupby('treatment_group')['country_iso3'].nunique())

# ── 4b: Aggregate to country-year ────────────────────────────────────────────
# event_study / lpdid require one row per (unit, time).
# Treatment is country-level so average immunization coverage across antigens.
COVARS = ['gdp_per_capita_usd', 'health_exp_pct_gdp', 'oop_health_exp_pct',
          'population_total']

panel = (
    df
    .groupby(['country_iso3', 'year', 'treatment_group', 'pta_pharma_start_year'])
    .agg(
        immunization_coverage=('immunization_coverage', 'mean'),
        **{c: (c, 'mean') for c in COVARS}
    )
    .reset_index()
)

# Drop always-treated: no observable pre-period, cannot identify CATT
panel = panel[panel['treatment_group'] != 'always_treated'].copy()

# gname: staggered adopters get actual adoption year; never-treated = 0
panel['gname'] = panel['pta_pharma_start_year']

# ── Window restriction ─────────────────────────────────────────────────────────
# pyfixest builds a full (cohort × relative_time) grid from the unique values
# present in the data.  If cohort A has relative_time up to -10 but cohort B
# only goes to -9, it still tries to look up (cohort_B, -10) → KeyError.
# Trimming the panel to the desired window here ensures the global relative_time
# range is consistent across all cohorts before any model is fitted.
_DESIRED_PRE, _POST = -5, 10

# ── Effective pre-window: per-cohort diagnostic ────────────────────────────────
# For each cohort g, find the earliest year ANY country in it has data.
# The cohort's effective pre-window = that year − g  (a negative integer).
# The GLOBAL _PRE must be the least ambitious (max / closest to 0) across all
# cohorts, so that every cohort has at least one observation at the boundary.
_treated = panel[panel['gname'] > 0]

# For each (cohort, country) find the earliest year that country has data.
# Then take the MAX across countries per cohort: the latest-starting country
# is the binding constraint.  Using .min() was wrong — it kept cohorts where
# SOME countries start early but others don't, leaving empty cells in pyfixest's
# (cohort × relative_time) grid and triggering KeyError.
_country_cohort_min = _treated.groupby(['gname', 'country_iso3'])['year'].min()
_cohort_min_year    = _country_cohort_min.groupby('gname').max()   # binding = latest starter

# .to_numpy() on both sides to avoid pandas Series - Index alignment bug.
_cohort_eff_pre = pd.Series(
    (_cohort_min_year.to_numpy() - _cohort_min_year.index.to_numpy()).clip(max=0),
    index=_cohort_min_year.index,
)

# LP-DiD uses gname-1 as the base period for every horizon computation.
# A cohort with no observation at gname-1 will produce empty cells regardless
# of how many pre-treatment years are present → drop it.
_cohort_has_base = (
    _treated
    .assign(_base=lambda d: d['year'] == d['gname'] - 1)
    .groupby('gname')['_base']
    .any()
)

print('\nPer-cohort effective pre-window:')
for _g in sorted(_cohort_eff_pre.index):
    _eff  = int(_cohort_eff_pre[_g])
    _miny = int(_cohort_min_year[_g])
    _base = bool(_cohort_has_base.get(_g, False))
    _tags = []
    if _eff == 0:   _tags.append('no pre-data')
    if not _base:   _tags.append('no base period (gname-1) → will drop')
    print(f'  gname={_g}: min_year={_miny}, effective_pre={_eff}'
          + (f'  ⚠ {", ".join(_tags)}' if _tags else ''))

# A cohort is valid only if it has pre-treatment data AND the LP-DiD base period
_valid_cohorts   = [g for g in _cohort_eff_pre.index
                    if _cohort_eff_pre[g] < 0 and _cohort_has_base.get(g, False)]
_dropped_cohorts = [g for g in _cohort_eff_pre.index if g not in _valid_cohorts]

if _dropped_cohorts:
    print(f'\nDropping cohorts: {sorted(_dropped_cohorts)}')
    panel = panel[(panel['gname'] == 0) | (panel['gname'].isin(_valid_cohorts))].copy()

# Global _PRE from surviving cohorts only
_eff_valid = _cohort_eff_pre.loc[_valid_cohorts]
_binding   = int(_eff_valid.max()) if not _eff_valid.empty else _DESIRED_PRE
_PRE       = max(_binding, _DESIRED_PRE)
print(f'\nDesired pre-window : {_DESIRED_PRE}')
print(f'Binding constraint : {_binding}')
print(f'Effective _PRE used: {_PRE}')

# Apply window filter using the resolved _PRE
_rel = panel['year'] - panel['gname']
panel = panel[
    (panel['gname'] == 0) |
    ((_rel >= _PRE) & (_rel <= _POST))
].copy()

# LP-DiD validates post_window against the unique relative times actually present
# in the filtered data.  If the latest-gname cohort can only reach (panel_end - gname)
# < _POST, the observed max relative time will be less than _POST → ValueError.
# Recompute _POST from the filtered panel so lpdid gets a valid value.
_actual_post = int(
    (panel.loc[panel['gname'] > 0, 'year'] - panel.loc[panel['gname'] > 0, 'gname']).max()
)
_POST = min(_actual_post, _POST)
print(f'Actual max relative time in filtered data: {_actual_post}  →  _POST set to {_POST}')

# Re-factorize after filtering
panel['country_id'] = pd.factorize(panel['country_iso3'])[0] + 1

print('\nStep 4b — Country-year panel:')
print(panel.shape)
print('\nStaggered adoption cohorts:')
print(panel[panel['gname'] > 0][['country_iso3', 'gname']]
      .drop_duplicates().sort_values('gname').to_string())

# ── 4c: TWFE — biased benchmark ──────────────────────────────────────────────
print('\nStep 4c — TWFE (biased benchmark)...')
fit_twfe = pf.event_study(
    data=panel,
    yname='immunization_coverage',
    idname='country_id',
    tname='year',
    gname='gname',
    estimator='twfe',
    att=False,
    cluster='country_id',
)
print(fit_twfe.tidy().to_string())

# ── 4d: Saturated event study — Sun & Abraham (main estimator) ────────────────
print('\nStep 4d — Saturated event study / Sun & Abraham (main)...')
fit_sa = pf.event_study(
    data=panel,
    yname='immunization_coverage',
    idname='country_id',
    tname='year',
    gname='gname',
    estimator='saturated',
    cluster='country_id',
)
es_sa = fit_sa.aggregate()
print(es_sa.to_string())

# ── 4e: LP-DiD — local projections (robustness) ───────────────────────────────
print('\nStep 4e — LP-DiD (robustness)...')
fit_lp = pf.lpdid(
    data=panel,
    yname='immunization_coverage',
    idname='country_id',
    tname='year',
    gname='gname',
    vcov={'CRV1': 'country_id'},
    pre_window=_PRE,
    post_window=_POST,
    never_treated=0,
    att=False,
)
es_lp = fit_lp.tidy()
print(es_lp.to_string())

# ── Sample-size diagnostic ─────────────────────────────────────────────────────
_tr = panel[panel['gname'] > 0]
print('\n── Sample composition ──')
print(f"  Treated countries  : {_tr['country_iso3'].nunique()}")
print(f"  Never-treated      : {panel[panel['gname']==0]['country_iso3'].nunique()}")
print(f"  Cohorts retained   : {sorted(_tr['gname'].unique().tolist())}")
print(f"  Pre-window _PRE    : {_PRE}")
print(f"  Post-window _POST  : {_POST}")
print(f"  Obs (treated rows) : {len(_tr)}")

# ── 4f: Shared plotting helpers ───────────────────────────────────────────────
import numpy as np

def _normalize_tidy(df):
    """
    Normalise column names from pyfixest tidy()/aggregate() variants to a
    common schema used by the plotting helpers:
        estimate   — point estimate
        std error  — standard error
    Different pyfixest functions (event_study, lpdid, aggregate) use different
    column names across versions; this maps the most common alternatives.
    """
    df = df.copy()
    if 'estimate' not in df.columns:
        for c in ['att', 'Estimate', 'coef', 'coefficient']:
            if c in df.columns:
                df = df.rename(columns={c: 'estimate'})
                break
    if 'std error' not in df.columns:
        for c in ['se', 'std_error', 'Std. Error', 'Std Error', 'stderr']:
            if c in df.columns:
                df = df.rename(columns={c: 'std error'})
                break
    # Raise early with informative message if still missing
    for col in ('estimate', 'std error'):
        if col not in df.columns:
            raise KeyError(
                f"Cannot find '{col}' in tidy DataFrame. "
                f"Actual columns: {list(df.columns)}"
            )
    # Force numeric dtype — object arrays cause np.isfinite to fail in matplotlib
    df['estimate']  = pd.to_numeric(df['estimate'],   errors='coerce')
    df['std error'] = pd.to_numeric(df['std error'],  errors='coerce')
    return df

def _pval_col(df):
    """Return the p-value column name, handling pyfixest naming variants."""
    for c in ['Pr(>|t|)', 'pvalue', 'p-value', 'p_value']:
        if c in df.columns:
            return c
    return None

def _sig_stars(p):
    """Convert p-value to significance stars."""
    if p < 0.01:   return '***'
    elif p < 0.05: return '**'
    elif p < 0.10: return '*'
    return ''

def _event_times(df):
    """Extract numeric event-time values from pyfixest tidy output."""
    # lpdid stores event time in a 't' column directly
    if 't' in df.columns:
        return df['t'].values
    if 'term' in df.columns:
        extracted = df['term'].str.extract(r'(-?\d+)$')[0]
        if extracted.notna().all():
            return extracted.astype(int).values
    return np.arange(len(df))  # fallback: sequential integers

def plot_single_es(tidy_df, title, color, filename):
    """
    Individual event-study plot with 95% CI error bars, significance stars,
    and a vertical dashed line at t=0 (treatment onset).
    Stars are drawn above each estimate that clears the threshold:
        * p<0.10   ** p<0.05   *** p<0.01
    """
    tidy_df = _normalize_tidy(tidy_df)
    times = _event_times(tidy_df)
    pc    = _pval_col(tidy_df)
    ests  = tidy_df['estimate'].values
    ses   = tidy_df['std error'].values

    fig, ax = plt.subplots(figsize=(11, 5))

    # Zero line and treatment-onset marker
    ax.axhline(0, color='black', lw=0.8, linestyle='--')
    ax.axvline(-0.5, color='grey', lw=0.6, linestyle=':')

    # Error bars (95% CI)
    ax.errorbar(times, ests, yerr=1.96 * ses,
                fmt='o-', capsize=4, color=color, linewidth=1.5, zorder=3)

    # Significance stars above each point
    if pc is not None:
        for t, est, se, pv in zip(times, ests, ses, tidy_df[pc].values):
            stars = _sig_stars(pv)
            if stars:
                ax.text(t, est + 1.96 * se + 0.3, stars,
                        ha='center', va='bottom', fontsize=9,
                        fontweight='bold', color=color)

    # Star legend
    ax.text(0.01, 0.99, '* p<0.10   ** p<0.05   *** p<0.01',
            transform=ax.transAxes, fontsize=8, va='top', color='dimgrey')

    ax.set_xticks(times)
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel('Event time (years relative to PTA)')
    ax.set_ylabel('ATT — immunization coverage (pp)')
    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved → {filename}')

# ── 4g: Individual plots (one per estimator) ─────────────────────────────────
plot_single_es(fit_twfe.tidy(), 'TWFE — biased benchmark',          'tomato',   'es_twfe.png')
plot_single_es(es_sa,           'Saturated / Sun & Abraham — main', 'steelblue','es_sa.png')
plot_single_es(es_lp,           'LP-DiD — robustness check',        'seagreen', 'es_lpdid.png')
1
# ── 4h: Comparison overlay — all three estimators on one axes ─────────────────
# Shaded 95% CI ribbons + lines. Useful for spotting where TWFE diverges from
# the heterogeneity-robust estimators (Sun & Abraham, LP-DiD).
fig, ax = plt.subplots(figsize=(13, 6))

overlay_configs = [
    (fit_twfe.tidy(), 'TWFE (benchmark)',      'tomato',   's--', 0.55),
    (es_sa,           'Sun & Abraham (main)',  'steelblue','o-',  1.00),
    (es_lp,           'LP-DiD (robustness)',   'seagreen', '^:',  0.80),
]

for tidy_df, label, color, fmt, alpha in overlay_configs:
    tidy_df = _normalize_tidy(tidy_df)
    times = _event_times(tidy_df)
    ests  = tidy_df['estimate'].values
    ses   = tidy_df['std error'].values
    # 95% CI ribbon
    ax.fill_between(times, ests - 1.96 * ses, ests + 1.96 * ses,
                    color=color, alpha=0.12)
    ax.plot(times, ests, fmt, color=color,
            label=label, linewidth=1.8, markersize=5, alpha=alpha)

ax.axhline(0, color='black', lw=0.8, linestyle='--')
ax.axvline(-0.5, color='grey', lw=0.6, linestyle=':', label='treatment onset')
ax.set_xlabel('Event time (years relative to PTA)')
ax.set_ylabel('ATT — immunization coverage (pp)')
ax.set_title(
    'Estimator Comparison: TWFE vs Sun & Abraham vs LP-DiD\n'
    'Pharma PTA effect on immunization coverage (95% CI shaded)',
    fontsize=12, fontweight='bold',
)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('es_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('\nComparison plot saved → es_comparison.png')

# ── Step 5: Heterogeneous Treatment Effects + Country Clustering ──────────────
# (commented out — run after Step 4 results are confirmed)
#
# Goal: estimate country-level CATEs via Causal Forest DML, then cluster
# countries by [CATE + covariates] to identify groups with similar treatment
# effect profiles.
#
# Method: Causal Forest DML (EconML / Chernozhukov et al. 2018)
#   Y  = immunization_coverage
#   T  = pta_active (binary: 1 once pharma PTA is in force)
#   X  = country-level mean covariates (MODERATORS — what drives heterogeneity)
#   W  = year dummies (nuisance time FE, partialled out by the DML step)
#
# DML cross-fits nuisance models for E[Y|W,X] and E[T|W,X], then fits a causal
# forest on residuals (ε_Y ~ ε_T × X) to recover CATE(X_i) non-parametrically.
# K-means then clusters countries on [CATE + covariates]; PCA projects to 2-D.
#
# from econml.dml import CausalForestDML
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
#
# # ── 5a: Prepare data ──────────────────────────────────────────────────────────
# # pta_active = 1 once a staggered adopter's PTA is in force; 0 for never-treated
# panel['pta_active'] = (
#     (panel['gname'] > 0) & (panel['year'] >= panel['gname'])
# ).astype(float)
#
# ml = panel.dropna(subset=['immunization_coverage'] + COVARS).copy()
#
# # Country-level mean covariates as moderators (time-invariant profile per country)
# country_avg = ml.groupby('country_iso3')[COVARS].mean()
# ml = ml.join(
#     country_avg.rename(columns={c: f'{c}_avg' for c in COVARS}),
#     on='country_iso3'
# )
# X_mod_cols = [f'{c}_avg' for c in COVARS]
#
# Y = ml['immunization_coverage'].values
# T = ml['pta_active'].values
# X = ml[X_mod_cols].values                                          # moderators
# W = pd.get_dummies(ml['year'], drop_first=True).astype(float).values  # time FE
#
# # ── 5b: Causal Forest DML ─────────────────────────────────────────────────────
# # StandardScaler on X so forest splits aren't distorted by variable scale.
# # discrete_treatment=True → EconML uses a classifier for the propensity model.
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# cf = CausalForestDML(
#     n_estimators=1000,
#     min_samples_leaf=5,
#     max_depth=4,
#     discrete_treatment=True,
#     random_state=42,
#     cv=3,       # 3-fold cross-fitting for nuisance models
#     n_jobs=-1,
# )
# cf.fit(Y, T, X=X_scaled, W=W)
# print('\nStep 5b — Overall ATE (Causal Forest):', cf.ate(X_scaled).round(4))
#
# # ── 5c: Country-level CATEs ───────────────────────────────────────────────────
# # Evaluate forest at each country's mean covariate vector.
# # effect_interval uses jackknife-after-bootstrap for 90% CIs.
# country_Xmat     = scaler.transform(country_avg[COVARS].values)
# cate             = cf.effect(country_Xmat)
# cate_lb, cate_ub = cf.effect_interval(country_Xmat, alpha=0.10)
#
# country_res = country_avg.reset_index().copy()
# country_res['cate']    = cate
# country_res['cate_lb'] = cate_lb
# country_res['cate_ub'] = cate_ub
#
# print('\nStep 5c — Top 10 countries by CATE:')
# print(country_res[['country_iso3', 'cate', 'cate_lb', 'cate_ub']]
#       .sort_values('cate', ascending=False).head(10).to_string())
#
# # ── 5d: K-means clustering on [CATE + covariates] ────────────────────────────
# # K=3 → low / medium / high responders.
# # Re-standardise before clustering so no single variable dominates the metric.
# K = 3
# X_clust        = country_res[['cate'] + COVARS].dropna()
# X_clust_scaled = StandardScaler().fit_transform(X_clust)
#
# kmeans = KMeans(n_clusters=K, random_state=42, n_init=20)
# country_res.loc[X_clust.index, 'cluster'] = kmeans.fit_predict(X_clust_scaled).astype(float)
#
# pca   = PCA(n_components=2)
# X_pca = pca.fit_transform(X_clust_scaled)
#
# # ── 5e: Plots ─────────────────────────────────────────────────────────────────
# fig, axes = plt.subplots(1, 3, figsize=(20, 6))
#
# # Panel A: caterpillar — one dot per country, sorted by CATE, with 90% CI
# s = country_res.sort_values('cate').reset_index(drop=True)
# axes[0].errorbar(
#     range(len(s)), s['cate'],
#     yerr=[s['cate'] - s['cate_lb'], s['cate_ub'] - s['cate']],
#     fmt='o', capsize=2, markersize=4, linewidth=0.6, color='steelblue', alpha=0.8,
# )
# axes[0].axhline(0, color='black', linestyle='--', lw=0.8)
# axes[0].set(xlabel='Country (sorted by CATE)',
#             ylabel='CATE (pp immunization coverage)',
#             title='Country-Level CATEs\n(Causal Forest DML, 90% CI)')
#
# # Panel B: CATE vs GDP — tests whether richer countries respond differently
# sc = axes[1].scatter(
#     country_res['gdp_per_capita_usd'], country_res['cate'],
#     c=country_res['cluster'], cmap='Set1', s=60, alpha=0.8,
# )
# axes[1].axhline(0, color='black', linestyle='--', lw=0.8)
# axes[1].set(xlabel='Mean GDP per Capita (USD)', ylabel='CATE',
#             title='CATE vs GDP per Capita\n(coloured by cluster)')
# plt.colorbar(sc, ax=axes[1], label='Cluster')
#
# # Panel C: PCA projection of clustering space — countries close together share
# # similar CATE + covariate profiles
# cluster_colors = ['tomato', 'steelblue', 'seagreen']
# for k in range(K):
#     mask = kmeans.labels_ == k
#     axes[2].scatter(X_pca[mask, 0], X_pca[mask, 1],
#                     c=cluster_colors[k], label=f'Cluster {k}', s=60, alpha=0.8)
# isos = country_res.loc[X_clust.index, 'country_iso3'].values
# for i, iso in enumerate(isos):
#     axes[2].annotate(iso, (X_pca[i, 0], X_pca[i, 1]), fontsize=6, alpha=0.6)
# axes[2].set(xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
#             ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
#             title='Country Clusters\n(CATE + Covariates, PCA)')
# axes[2].legend(fontsize=8)
#
# fig.suptitle('Treatment Effect Heterogeneity — Pharma PTA on Immunization Coverage',
#              fontsize=12, fontweight='bold')
# plt.tight_layout()
# plt.savefig('heterogeneity_clusters.png', dpi=150, bbox_inches='tight')
# plt.show()
# print('\nHeterogeneity plot saved → heterogeneity_clusters.png')
#
# # ── 5f: Cluster summaries ─────────────────────────────────────────────────────
# # Mean CATE + covariate profile per cluster → tells you what kind of country
# # belongs to each group (e.g., low-income small-effect vs middle-income large-effect)
# print('\n── Cluster mean characteristics ──')
# print(country_res.groupby('cluster')[['cate'] + COVARS].mean().round(3).to_string())
# print('\n── Countries per cluster ──')
# for k in range(K):
#     cs = sorted(country_res[country_res['cluster'] == k]['country_iso3'].tolist())
#     print(f'Cluster {k}: {cs}')

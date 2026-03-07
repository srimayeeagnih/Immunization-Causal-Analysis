import os
import re

import pandas as pd
import pycountry

# ── Step 1: PTA aggregation from Chemicals & Allied Industries tariff data ────
# Each row is a reporter-partner-product tariff record.
# We aggregate to the reporter (country) level to get a country-level PTA indicator.

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

trade = pd.read_csv(
    os.path.join(_ROOT, "data", "raw", "Chemicals_Allied_Industries.csv")
)

pta_agg = (
    trade.groupby("reporteriso3")["PTA"]
    .agg(pta_share="mean", has_pta="max")   # share of partners with PTA; binary flag
    .reset_index()
    .rename(columns={"reporteriso3": "country_iso3"})
)
pta_agg["has_pta"] = pta_agg["has_pta"].astype(int)

print("Step 1 -- PTA aggregation (by reporter country):")
print(pta_agg.shape)
print(pta_agg.head(10).to_string())


# ── Step 2: Pharma-specific PTA start year from WTO-X dataset ────────────────
# The WTO-X dataset codes which PTAs contain health provisions directly
# relevant to pharma access and regulation.
# We filter to agreements with Health provisions only, parse the member countries
# from agreement names, and take the earliest entry-into-force year per country.
# Countries not matched (no health PTA) are treated as never-treated in DiD.

WTO_PLUS_PATH = os.path.join(_ROOT, "data", "raw", "pta-agreements_1.xls")

wto_x = pd.read_excel(WTO_PLUS_PATH, sheet_name="WTO-X AC")
pharma_mask = wto_x["Health"].ge(1)   # Health-provision PTAs only
pharma = wto_x[pharma_mask][["Agreement", "year"]].copy()
pharma["year"] = pd.to_numeric(pharma["year"], errors="coerce")
pharma = pharma.dropna(subset=["year"])
pharma["year"] = pharma["year"].astype(int)

# Multilateral blocs -> member ISO3 lists
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
    parts = re.split(r"\s*[-\u2013]\s*", agr)
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

print("\nStep 2 -- Pharma-specific PTA start year per country (WTO-X: Health provisions only):")
print(pta_pharma_start.shape)
print(pta_pharma_start.sort_values("pta_pharma_start_year").head(10).to_string())

# ── Step 3: Merge PTA variables into pivot_dataset ───────────────────────────
# country_iso3 is kept in feature_engineering.py (removed from the drop list).
# Left-join so all panel rows are preserved; unmatched countries get NaN.

df = pd.read_csv(os.path.join(_ROOT, "data", "processed", "pivot_dataset_fe.csv"))

df = (
    df
    .merge(pta_agg,          on="country_iso3", how="left")   # has_pta, pta_share
    .merge(pta_pharma_start, on="country_iso3", how="left")   # pta_pharma_start_year
)

# Countries with no health PTA in WTO-X -> never treated; fill start year with 0
df["pta_pharma_start_year"] = df["pta_pharma_start_year"].fillna(0).astype(int)

# DiD treatment indicator: 1 if the country has a health PTA *and* the current year
# is at or after the agreement's entry-into-force year
df["pta_treated"] = (
    (df["has_pta"] == 1) &
    (df["pta_pharma_start_year"] > 0) &
    (df["year"] >= df["pta_pharma_start_year"])
).astype(int)

# Years since health PTA entry into force
df["years_since_pta"] = (
    (df["year"] - df["pta_pharma_start_year"])
    .clip(lower=0)
)
df.loc[df["pta_pharma_start_year"] == 0, "years_since_pta"] = 0

print("\nStep 3 -- Merged pivot dataset:")
print(df.shape)
print(df[["country_iso3", "year", "has_pta", "pta_share", "pta_pharma_start_year", "pta_treated"]].head(15).to_string())
print("\npta_treated distribution:", df["pta_treated"].value_counts().to_dict())
print("Missing has_pta:", df["has_pta"].isna().sum(), "rows")

df['relative_time'] = df['year'] - df['pta_pharma_start_year']
df.loc[df['pta_pharma_start_year'] == 0, 'relative_time'] = float('nan')

# ── Step 4: Staggered DiD -- Two-scenario design ──────────────────────────────
import os
import matplotlib.pyplot as plt
import numpy as np
import pyfixest as pf

PANEL_START = 2001
OUT_DIR = os.path.join(_ROOT, "outputs", "visualization")

# ── 4a: Classify treatment groups ────────────────────────────────────────────
# Always-treated  (G_i < 2001): PTA predates panel -> EU bloc as upper-bound controls
# Staggered       (G_i >= 2001): PTA adopted within panel -> main identification
# Never-treated   (G_i = 0):  no health PTA -> conditional parallel trends controls
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

print('\nStep 4a -- Treatment group counts (unique countries):')
print(df.groupby('treatment_group')['country_iso3'].nunique())

# ── 4b: Country group definitions ─────────────────────────────────────────────
# Scenario 1 controls: EU-27 (always-treated, relabelled as gname=0)
# Scenario 2 controls: never-treated whitelist (high- and upper-middle-income countries)
EU_BLOC = {
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", "DEU", "GRC",
    "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD", "POL", "PRT", "ROU", "SVK",
    "SVN", "ESP", "SWE"
}

# High income: BHR KWT OMN QAT SAU ARE CHE ISL NOR GBR ISR NZL PAN URY
# Upper-middle income: ALB ARM AZE BIH BLR MKD MNE SRB ARG BOL BRA ECU PER PRY IDN THA LKA KAZ DZA LBN TUN
NEVER_TREATED_KEEP = {
    "BHR", "KWT", "OMN", "QAT", "SAU", "ARE", "CHE", "ISL", "NOR", "GBR", "ISR", "NZL", "PAN", "URY",
    "ALB", "ARM", "AZE", "BIH", "BLR", "MKD", "MNE", "SRB",
    "ARG", "BOL", "BRA", "ECU", "PER", "PRY",
    "IDN", "THA", "LKA", "KAZ", "DZA", "LBN", "TUN",
}

COVARS     = ['gdp_per_capita_usd', 'health_exp_pct_gdp', 'oop_health_exp_pct', 'population_total']
ADJ_COVARS = ['gdp_per_capita_usd', 'health_exp_pct_gdp']
_DESIRED_PRE  = -3
_DESIRED_POST = 10

# ── Shared plotting helpers ────────────────────────────────────────────────────

def _normalize_tidy(tidy_df):
    """
    Normalise column names from pyfixest tidy()/aggregate() variants to a
    common schema: estimate, std error.
    """
    tidy_df = tidy_df.copy()
    if 'estimate' not in tidy_df.columns:
        for c in ['att', 'Estimate', 'coef', 'coefficient']:
            if c in tidy_df.columns:
                tidy_df = tidy_df.rename(columns={c: 'estimate'})
                break
    if 'std error' not in tidy_df.columns:
        for c in ['se', 'std_error', 'Std. Error', 'Std Error', 'stderr']:
            if c in tidy_df.columns:
                tidy_df = tidy_df.rename(columns={c: 'std error'})
                break
    for col in ('estimate', 'std error'):
        if col not in tidy_df.columns:
            raise KeyError(
                f"Cannot find '{col}' in tidy DataFrame. "
                f"Actual columns: {list(tidy_df.columns)}"
            )
    tidy_df['estimate']  = pd.to_numeric(tidy_df['estimate'],  errors='coerce')
    tidy_df['std error'] = pd.to_numeric(tidy_df['std error'], errors='coerce')
    return tidy_df


def _pval_col(tidy_df):
    for c in ['Pr(>|t|)', 'pvalue', 'p-value', 'p_value']:
        if c in tidy_df.columns:
            return c
    return None


def _sig_stars(p):
    if p < 0.01:   return '***'
    elif p < 0.05: return '**'
    elif p < 0.10: return '*'
    return ''


def _event_times(tidy_df):
    if 't' in tidy_df.columns:
        return tidy_df['t'].values
    if 'period' in tidy_df.columns:        # pyfixest aggregate() uses 'period'
        return tidy_df['period'].values
    if 'term' in tidy_df.columns:
        extracted = tidy_df['term'].str.extract(r'(-?\d+)$')[0]
        if extracted.notna().all():
            return extracted.astype(int).values
    return np.arange(len(tidy_df))


def plot_single_es(tidy_df, title, color, filename, star_offset=None, xticklabels=None):
    """
    Individual event-study plot with 95% CI error bars and significance stars.
    star_offset:  fixed y-offset above the point estimate for significance stars.
                  Defaults to top of 95% CI bar (1.96*se + 0.3) if None.
    xticklabels:  list of strings to use as x-tick labels (overrides raw axis values).
                  Use this when the underlying positions are 0..N but the true event
                  times are e.g. -3, -2, 0, 1, ..., 10 (skipping base period -1).
    """
    tidy_df = _normalize_tidy(tidy_df)
    times = _event_times(tidy_df)
    pc    = _pval_col(tidy_df)
    ests  = tidy_df['estimate'].values
    ses   = tidy_df['std error'].values

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(0, color='black', lw=0.8, linestyle='--')
    ax.axvline(-0.5, color='grey', lw=0.6, linestyle=':')
    ax.errorbar(times, ests, yerr=1.96 * ses,
                fmt='o-', capsize=4, color=color, linewidth=1.5, zorder=3)

    if pc is not None:
        for t, est, se, pv in zip(times, ests, ses, tidy_df[pc].values):
            stars = _sig_stars(pv)
            if stars:
                y_pos = est + (star_offset if star_offset is not None else 1.96 * se + 0.3)
                ax.text(t, y_pos, stars,
                        ha='center', va='bottom', fontsize=9,
                        fontweight='bold', color=color)

    ax.text(0.01, 0.99, '* p<0.10   ** p<0.05   *** p<0.01',
            transform=ax.transAxes, fontsize=8, va='top', color='dimgrey')
    ax.set_xticks(times)
    if xticklabels is not None and len(xticklabels) == len(times):
        ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    else:
        ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel('Event time (years relative to PTA)')
    ax.set_ylabel('ATT -- immunization coverage (pp)')
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved -> {filename}')


# ── build_panel() helper ───────────────────────────────────────────────────────
def build_panel(df_in, label):
    """
    Aggregate df_in to country-year, apply country-level window cleaning
    (>= 3 pre-period years AND base period gname-1 present per country),
    and return the filtered panel ready for pyfixest estimators.

    Country-level cleaning (not cohort-level) retains clean countries even when
    a cohort has some countries with short pre-windows, preventing silent
    observation loss from cohort-wide dropping.
    """
    panel = (
        df_in
        .groupby(['country_iso3', 'year', 'treatment_group', 'pta_pharma_start_year'])
        .agg(
            immunization_coverage=('immunization_coverage', 'mean'),
            **{c: (c, 'mean') for c in COVARS}
        )
        .reset_index()
    )
    panel['gname'] = panel['pta_pharma_start_year']

    _tr = panel[panel['gname'] > 0]

    # Per (cohort, country): check if >= 3 pre-periods exist AND base period (gname-1)
    _cc = (
        _tr.assign(_b=lambda d: d['year'] == d['gname'] - 1)
        .groupby(['gname', 'country_iso3'])
        .agg(min_year=('year', 'min'), has_base=('_b', 'any'))
        .reset_index()
    )
    _cc['req_year'] = _cc['gname'] + _DESIRED_PRE   # gname - 3
    _cc['is_clean'] = (_cc['min_year'] <= _cc['req_year']) & _cc['has_base']

    # Keep cohorts that have at least 1 clean country
    _vc = _cc[_cc['is_clean']].groupby('gname').size()
    _vc = _vc[_vc >= 1].index.tolist()

    _ck = _cc.loc[_cc['is_clean'] & _cc['gname'].isin(_vc), ['gname', 'country_iso3']]
    _tc = _tr.merge(_ck, on=['gname', 'country_iso3'], how='inner')

    panel = pd.concat([panel[panel['gname'] == 0], _tc], ignore_index=True).copy()

    # Apply event-time window
    _rel = panel['year'] - panel['gname']
    panel = panel[
        (panel['gname'] == 0) | ((_rel >= _DESIRED_PRE) & (_rel <= _DESIRED_POST))
    ].copy()

    # Recompute actual post window from filtered data
    _actual_post = int(
        (panel.loc[panel['gname'] > 0, 'year'] - panel.loc[panel['gname'] > 0, 'gname']).max()
    )
    post = min(_actual_post, _DESIRED_POST)

    panel['country_id'] = pd.factorize(panel['country_iso3'])[0] + 1

    _tr2 = panel[panel['gname'] > 0]
    print(f'\n[{label}] Panel built:')
    print(f'  Treated countries : {sorted(_tr2["country_iso3"].unique().tolist())}')
    print(f'  Control countries : {panel[panel["gname"]==0]["country_iso3"].nunique()}')
    print(f'  Cohorts           : {sorted(_tr2["gname"].unique().tolist())}')
    print(f'  Treated rows      : {len(_tr2)}  |  Control rows: {len(panel[panel["gname"]==0])}')
    print(f'  Pre-window        : {_DESIRED_PRE}  |  Post-window: {post}')

    return panel, post


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 1: Treated vs EU controls (convergence framing)
# ══════════════════════════════════════════════════════════════════════════════
# EU-27 are always-treated (health PTAs since pre-2001) and serve as an
# upper-bound counterfactual: would staggered adopters converge toward the
# EU immunization baseline?
# EU countries are relabelled as gname=0 (no staggered adoption in window).
# No covariate adjustment -- parallel trends relies on EU structural similarity.

print('\n' + '='*60)
print('SCENARIO 1: Treated vs EU controls (convergence framing)')
print('='*60)

df_s1 = df[
    df['country_iso3'].isin(EU_BLOC) | (df['treatment_group'] == 'staggered')
].copy()
df_s1.loc[df_s1['country_iso3'].isin(EU_BLOC), 'treatment_group']       = 'never_treated'
df_s1.loc[df_s1['country_iso3'].isin(EU_BLOC), 'pta_pharma_start_year'] = 0

panel_s1, post_s1 = build_panel(df_s1, 'S1')

print('\nS1 TWFE (biased benchmark)...')
fit_twfe_s1 = pf.event_study(
    data=panel_s1, yname='immunization_coverage',
    idname='country_id', tname='year', gname='gname',
    estimator='twfe', att=False, cluster='country_id',
)
print(fit_twfe_s1.tidy().to_string())

print('\nS1 Sun & Abraham (main)...')
fit_sa_s1 = pf.event_study(
    data=panel_s1, yname='immunization_coverage',
    idname='country_id', tname='year', gname='gname',
    estimator='saturated', cluster='country_id',
)
es_sa_s1 = fit_sa_s1.aggregate()
print(es_sa_s1.to_string())

print('\nS1 LP-DiD (robustness)...')
fit_lp_s1 = pf.lpdid(
    data=panel_s1, yname='immunization_coverage',
    idname='country_id', tname='year', gname='gname',
    vcov={'CRV1': 'country_id'},
    pre_window=_DESIRED_PRE, post_window=post_s1,
    never_treated=0, att=False,
)
es_lp_s1 = fit_lp_s1.tidy()
print(es_lp_s1.to_string())

print('\nS1 -- Sample composition:')
_tr1 = panel_s1[panel_s1['gname'] > 0]
print(f"  Treated: {_tr1['country_iso3'].nunique()} countries, "
      f"{len(_tr1)} rows, cohorts: {sorted(_tr1['gname'].unique().tolist())}")
print(f"  EU controls: {panel_s1[panel_s1['gname']==0]['country_iso3'].nunique()} countries")

plot_single_es(fit_twfe_s1.tidy(),
               'S1: TWFE -- EU controls (benchmark)', 'tomato', os.path.join(OUT_DIR, 'es_s1_twfe.png'))
_sa_labels_s1 = [str(t) for t in sorted(list(range(_DESIRED_PRE, 0)) + list(range(0, post_s1 + 1))) if t != -1]
plot_single_es(es_sa_s1,
               'S1: Sun & Abraham -- EU controls', 'steelblue', os.path.join(OUT_DIR, 'es_s1_sa.png'),
               star_offset=0.05, xticklabels=_sa_labels_s1)
plot_single_es(es_lp_s1,
               'S1: LP-DiD -- EU controls', 'seagreen', os.path.join(OUT_DIR, 'es_s1_lpdid.png'))


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 2: Treated vs Never-treated (covariate-adjusted LP-DiD)
# ══════════════════════════════════════════════════════════════════════════════
# Never-treated whitelist (high- and upper-middle-income countries with no
# health PTA) serves as the causal counterfactual.
# Covariate adjustment (GDP per capita + health expenditure % GDP) achieves
# conditional parallel trends without propensity score instability at n=11.
# Question: absent a pharma PTA, would treated countries have tracked never-
# treated trends conditional on economic characteristics?

print('\n' + '='*60)
print('SCENARIO 2: Treated vs Never-treated (covariate-adjusted)')
print('='*60)

df_s2 = df[
    (df['country_iso3'].isin(NEVER_TREATED_KEEP) & (df['treatment_group'] == 'never_treated')) |
    (df['treatment_group'] == 'staggered')
].copy()

panel_s2, post_s2 = build_panel(df_s2, 'S2')

print('\nS2 TWFE (biased benchmark)...')
fit_twfe_s2 = pf.event_study(
    data=panel_s2, yname='immunization_coverage',
    idname='country_id', tname='year', gname='gname',
    estimator='twfe', att=False, cluster='country_id',
)
print(fit_twfe_s2.tidy().to_string())

print('\nS2 Sun & Abraham (main)...')
fit_sa_s2 = pf.event_study(
    data=panel_s2, yname='immunization_coverage',
    idname='country_id', tname='year', gname='gname',
    estimator='saturated', cluster='country_id',
)
es_sa_s2 = fit_sa_s2.aggregate()
print(es_sa_s2.to_string())

print('\nS2 LP-DiD (robustness) ...')
print('  Note: pyfixest 0.40.1 xfml not yet supported in lpdid -- running unadjusted.')
print('  Conditional parallel trends addressed via income-matched control group selection.')
fit_lp_s2 = pf.lpdid(
    data=panel_s2, yname='immunization_coverage',
    idname='country_id', tname='year', gname='gname',
    vcov={'CRV1': 'country_id'},
    pre_window=_DESIRED_PRE, post_window=post_s2,
    never_treated=0, att=False,
)
es_lp_s2 = fit_lp_s2.tidy()
print(es_lp_s2.to_string())

print('\nS2 -- Sample composition:')
_tr2 = panel_s2[panel_s2['gname'] > 0]
print(f"  Treated: {_tr2['country_iso3'].nunique()} countries, "
      f"{len(_tr2)} rows, cohorts: {sorted(_tr2['gname'].unique().tolist())}")
print(f"  Never-treated: {panel_s2[panel_s2['gname']==0]['country_iso3'].nunique()} countries")

plot_single_es(fit_twfe_s2.tidy(),
               'S2: TWFE -- Never-treated controls (benchmark)', 'tomato', os.path.join(OUT_DIR, 'es_s2_twfe.png'))
_sa_labels_s2 = [str(t) for t in sorted(list(range(_DESIRED_PRE, 0)) + list(range(0, post_s2 + 1))) if t != -1]
plot_single_es(es_sa_s2,
               'S2: Sun & Abraham -- Never-treated controls', 'steelblue', os.path.join(OUT_DIR, 'es_s2_sa.png'),
               star_offset=0.05, xticklabels=_sa_labels_s2)
plot_single_es(es_lp_s2,
               'S2: LP-DiD -- Never-treated controls', 'seagreen',
               os.path.join(OUT_DIR, 'es_s2_lpdid.png'))


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-SCENARIO COMPARISON: LP-DiD S1 vs S2
# ══════════════════════════════════════════════════════════════════════════════
# Overlay S1 (EU controls, no covariates) and S2 (never-treated, covariate-adjusted)
# on the same axes. Agreement between scenarios strengthens credibility;
# divergence reveals sensitivity to the choice of control group.

print('\n' + '='*60)
print('CROSS-SCENARIO: LP-DiD S1 (EU) vs S2 (Never-treated, adj)')
print('='*60)

fig, ax = plt.subplots(figsize=(13, 6))

overlay_configs = [
    (es_lp_s1, 'S1: LP-DiD vs EU controls (convergence)',     'steelblue', 'o-',  1.0),
    (es_lp_s2, 'S2: LP-DiD vs Never-treated controls',        'seagreen',  's--', 0.9),
]

for tidy_df, label, color, fmt, alpha in overlay_configs:
    tidy_df = _normalize_tidy(tidy_df)
    times = _event_times(tidy_df)
    ests  = tidy_df['estimate'].values
    ses   = tidy_df['std error'].values
    ax.fill_between(times, ests - 1.96 * ses, ests + 1.96 * ses, color=color, alpha=0.12)
    ax.plot(times, ests, fmt, color=color,
            label=label, linewidth=1.8, markersize=5, alpha=alpha)

ax.axhline(0, color='black', lw=0.8, linestyle='--')
ax.axvline(-0.5, color='grey', lw=0.6, linestyle=':', label='treatment onset')
ax.set_xlabel('Event time (years relative to PTA)')
ax.set_ylabel('ATT -- immunization coverage (pp)')
ax.set_title(
    'LP-DiD: S1 (EU controls) vs S2 (Never-treated, covariate-adjusted)\n'
    'Pharma PTA effect on immunization coverage (95% CI shaded)',
    fontsize=12, fontweight='bold',
)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'es_cross_scenario.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Cross-scenario comparison plot saved -> es_cross_scenario.png')


# ── Step 5: Heterogeneous Treatment Effects + Country Clustering ──────────────
# Goal: estimate country-level CATEs via Linear DML, then cluster countries by
# [CATE + covariates] to identify groups with similar treatment effect profiles.
#
# Method: Linear DML (EconML / Chernozhukov et al. 2018)
#   Y  = immunization_coverage
#   T  = pta_active (binary: 1 once pharma PTA is in force)
#   X  = country-level mean covariates (MODERATORS -- what drives heterogeneity)
#   W  = year dummies (nuisance time FE, partialled out by the DML step)

from econml.dml import LinearDML
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

model_y = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge',  RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]))
])

model_t = Pipeline([
    ('scaler',   StandardScaler()),
    ('logistic', LogisticRegressionCV(cv=3, penalty='l2', solver='lbfgs'))
])

# ── 5a: Prepare data ──────────────────────────────────────────────────────────
# pta_active = 1 once a staggered adopter's PTA is in force; 0 for never-treated
# Use panel_s2 (never-treated controls) as the main identification sample
panel = panel_s2.copy()
panel['pta_active'] = (
    (panel['gname'] > 0) & (panel['year'] >= panel['gname'])
).astype(float)

ml = panel.dropna(subset=['immunization_coverage'] + COVARS).copy()

# Country-level mean covariates as moderators (time-invariant profile per country)
country_avg = ml.groupby('country_iso3')[COVARS].mean()
ml = ml.join(
    country_avg.rename(columns={c: f'{c}_avg' for c in COVARS}),
    on='country_iso3'
)
X_mod_cols = [f'{c}_avg' for c in COVARS]

Y = ml['immunization_coverage'].values
T = ml['pta_active'].values
X = ml[X_mod_cols].values                                           # moderators

# Pre-partial out global time trend from Y using year dummies.
# Year dummies are NOT passed to model_t — signing a health PTA is driven by
# country characteristics, not by what year it is. Passing year dummies to
# model_t would let it predict treatment almost perfectly (treatment is a
# deterministic step function of time), collapsing T residuals to near-zero
# and destabilising the second-stage CATE estimation.
# Year dummies are used here only to remove secular coverage increases
# (global immunisation trends) that are unrelated to PTAs.
year_dummies = pd.get_dummies(ml['year'], drop_first=True).astype(float).values
_time_model = Pipeline([('scaler', StandardScaler()), ('ridge', RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]))])
_time_model.fit(year_dummies, Y)
Y_detrended = Y - _time_model.predict(year_dummies)

# ── 5b: Linear DML ────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dml = LinearDML(
    model_y=model_y,      # fits E[Y_detrended | X] — country covariates only
    model_t=model_t,      # fits E[T | X]           — country covariates only, no year dummies
    discrete_treatment=True,
    random_state=42,
    cv=3,
)
dml.fit(Y_detrended, T, X=X_scaled, W=None)
print('\nStep 5b -- Overall ATE (Linear DML):', dml.ate(X_scaled).round(4))

# ── 5c: Country-level CATEs ───────────────────────────────────────────────────
country_Xmat     = scaler.transform(country_avg[COVARS].values)
cate             = dml.effect(country_Xmat)
cate_lb, cate_ub = dml.effect_interval(country_Xmat, alpha=0.10)

country_res = country_avg.reset_index().copy()
country_res['cate']    = cate
country_res['cate_lb'] = cate_lb
country_res['cate_ub'] = cate_ub

print('\nStep 5c -- Top 10 countries by CATE:')
print(country_res[['country_iso3', 'cate', 'cate_lb', 'cate_ub']]
      .sort_values('cate', ascending=False).head(10).to_string())

# ── 5d: K-means clustering on [CATE + covariates] ────────────────────────────
K = 3
X_clust        = country_res[['cate'] + COVARS].dropna()
X_clust_scaled = StandardScaler().fit_transform(X_clust)

kmeans = KMeans(n_clusters=K, random_state=42, n_init=20)
country_res.loc[X_clust.index, 'cluster'] = kmeans.fit_predict(X_clust_scaled).astype(float)

pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_clust_scaled)

# Save intermediate objects so plot_heterogeneity.py can regenerate the figure
# without re-running the full pipeline.
import pickle as _pickle
_cache = dict(
    country_res = country_res,
    X_pca       = X_pca,
    pca         = pca,
    kmeans      = kmeans,
    X_clust_idx = X_clust.index.tolist(),
    K           = K,
    COVARS      = COVARS,
    OUT_DIR     = OUT_DIR,
)
_cache_path = os.path.join(_ROOT, 'data', 'processed', 'plot_cache.pkl')
with open(_cache_path, 'wb') as _f:
    _pickle.dump(_cache, _f)
print(f'Plot cache saved -> {_cache_path}')

# ── 5e: Plots ─────────────────────────────────────────────────────────────────
# WHO region lookup (used to group countries in plot 1)
_WHO = {
    'AFG':'EMRO','ALB':'EURO','DZA':'EMRO','AGO':'AFRO','ARG':'AMRO',
    'ARM':'EURO','AUS':'WPRO','AUT':'EURO','AZE':'EURO','BHR':'EMRO',
    'BGD':'SEARO','BLR':'EURO','BEL':'EURO','BLZ':'AMRO','BEN':'AFRO',
    'BIH':'EURO','BOL':'AMRO','BRA':'AMRO','BRN':'WPRO','BGR':'EURO',
    'BFA':'AFRO','BDI':'AFRO','KHM':'WPRO','CMR':'AFRO','CAN':'AMRO',
    'CHL':'AMRO','CHN':'WPRO','COL':'AMRO','CRI':'AMRO','HRV':'EURO',
    'CUB':'AMRO','CYP':'EURO','CZE':'EURO','DNK':'EURO','DJI':'EMRO',
    'DOM':'AMRO','ECU':'AMRO','EGY':'EMRO','SLV':'AMRO','EST':'EURO',
    'ETH':'AFRO','FIN':'EURO','FRA':'EURO','DEU':'EURO','GHA':'AFRO',
    'GRC':'EURO','GTM':'AMRO','HUN':'EURO','ISL':'EURO','IND':'SEARO',
    'IDN':'SEARO','IRN':'EMRO','IRQ':'EMRO','IRL':'EURO','ISR':'EURO',
    'ITA':'EURO','JAM':'AMRO','JPN':'WPRO','JOR':'EMRO','KAZ':'EURO',
    'KEN':'AFRO','KOR':'WPRO','KWT':'EMRO','KGZ':'EURO','LAO':'WPRO',
    'LVA':'EURO','LBN':'EMRO','LTU':'EURO','LUX':'EURO','MYS':'WPRO',
    'MDV':'SEARO','MLT':'EURO','MDA':'EURO','MNG':'WPRO','MNE':'EURO',
    'MAR':'EMRO','MOZ':'AFRO','MMR':'SEARO','NPL':'SEARO','NLD':'EURO',
    'NZL':'WPRO','NIC':'AMRO','MKD':'EURO','NOR':'EURO','OMN':'EMRO',
    'PAK':'EMRO','PAN':'AMRO','PRY':'AMRO','PER':'AMRO','PHL':'WPRO',
    'POL':'EURO','PRT':'EURO','QAT':'EMRO','ROU':'EURO','RUS':'EURO',
    'RWA':'AFRO','SAU':'EMRO','SEN':'AFRO','SRB':'EURO','SGP':'WPRO',
    'SVK':'EURO','SVN':'EURO','ZAF':'AFRO','ESP':'EURO','LKA':'SEARO',
    'SDN':'EMRO','SWE':'EURO','CHE':'EURO','TJK':'EURO','TZA':'AFRO',
    'THA':'SEARO','TGO':'AFRO','TTO':'AMRO','TUN':'EMRO','TUR':'EURO',
    'TKM':'EURO','UGA':'AFRO','UKR':'EURO','ARE':'EMRO','GBR':'EURO',
    'USA':'AMRO','URY':'AMRO','UZB':'EURO','VNM':'WPRO','YEM':'EMRO',
    'ZMB':'AFRO','ZWE':'AFRO',
}

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

# ── Plot 1: Mean CATE by WHO region (horizontal dot plot) ─────────────────────
country_res['who_region'] = country_res['country_iso3'].map(_WHO).fillna('Other')
_reg = (
    country_res.groupby('who_region')['cate']
    .agg(['mean', 'std', 'count'])
    .reset_index()
    .rename(columns={'mean': 'mean_cate', 'std': 'std_cate', 'count': 'n'})
)
_reg['ci90'] = 1.645 * _reg['std_cate'] / np.sqrt(_reg['n'])
_reg = _reg.sort_values('mean_cate').reset_index(drop=True)

axes[0].errorbar(
    _reg['mean_cate'], _reg['who_region'],
    xerr=_reg['ci90'],
    fmt='o', capsize=4, markersize=8, color='steelblue',
    linewidth=1.3, elinewidth=1.3, alpha=0.85,
)
axes[0].axvline(0, color='black', linestyle='--', lw=0.8)
for _, row in _reg.iterrows():
    axes[0].text(
        row['mean_cate'] + row['ci90'] + 0.005, row['who_region'],
        f"n={int(row['n'])}", va='center', fontsize=8, color='dimgrey'
    )
axes[0].set_xlabel('Mean CATE (pp immunization coverage)')
axes[0].set_ylabel('WHO Region')
axes[0].set_title('Mean CATE by WHO Region\n(Linear DML, 90% CI)', fontweight='bold')
axes[0].tick_params(axis='y', labelsize=9)

# ── Plot 2: CATE vs GDP per capita (coloured by cluster) ──────────────────────
_plot_cr = country_res.dropna(subset=['cluster'])
sc = axes[1].scatter(
    _plot_cr['gdp_per_capita_usd'], _plot_cr['cate'],
    c=_plot_cr['cluster'], cmap='Set1', s=60, alpha=0.8, vmin=0, vmax=2,
)
axes[1].axhline(0, color='black', linestyle='--', lw=0.8)
axes[1].set_xlabel('Mean GDP per Capita (USD)')
axes[1].set_ylabel('CATE (pp immunization coverage)')
axes[1].set_title('CATE vs GDP per Capita\n(coloured by cluster)', fontweight='bold')
fig.colorbar(sc, ax=axes[1], label='Cluster', ticks=[0, 1, 2])

# ── Plot 3: PCA country clusters with loadings annotation ─────────────────────
cluster_colors = ['tomato', 'steelblue', 'seagreen']
for k in range(K):
    mask = kmeans.labels_ == k
    axes[2].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=cluster_colors[k], label=f'Cluster {k}', s=65, alpha=0.85)
isos = country_res.loc[X_clust.index, 'country_iso3'].values
for i, iso in enumerate(isos):
    axes[2].annotate(iso, (X_pca[i, 0], X_pca[i, 1]),
                     fontsize=6.5, alpha=0.75,
                     xytext=(3, 3), textcoords='offset points')

# PC loadings annotation — explains what each axis represents
_feat_names = ['CATE', 'GDP/cap', 'Health Exp%', 'OOP%', 'Population']
_pc1 = sorted(zip(_feat_names, pca.components_[0]), key=lambda x: abs(x[1]), reverse=True)[:3]
_pc2 = sorted(zip(_feat_names, pca.components_[1]), key=lambda x: abs(x[1]), reverse=True)[:3]
_loading_txt = (
    'PC1 drivers: ' + ' | '.join(f'{n} ({v:+.2f})' for n, v in _pc1) +
    '\nPC2 drivers: ' + ' | '.join(f'{n} ({v:+.2f})' for n, v in _pc2)
)
axes[2].text(0.01, 0.01, _loading_txt, transform=axes[2].transAxes,
             fontsize=6.5, va='bottom', color='dimgrey',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                       alpha=0.8, edgecolor='silver'))
axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance explained)')
axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance explained)')
axes[2].set_title('Country Clusters\n(CATE + Covariates, PCA)', fontweight='bold')
axes[2].legend(fontsize=8, loc='upper right', framealpha=0.9, edgecolor='silver')

fig.suptitle('Treatment Effect Heterogeneity — Pharma PTA on Immunization Coverage',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 'heterogeneity_clusters.png'), dpi=150, bbox_inches='tight')
plt.show()
print('\nHeterogeneity plot saved -> heterogeneity_clusters.png')

# ── 5f: Cluster summaries ─────────────────────────────────────────────────────
print('\n-- Cluster mean characteristics --')
print(country_res.groupby('cluster')[['cate'] + COVARS].mean().round(3).to_string())
print('\n-- Countries per cluster --')
for k in range(K):
    cs = sorted(country_res[country_res['cluster'] == k]['country_iso3'].tolist())
    print(f'Cluster {k}: {cs}')

"""
Standalone DiD script — Steps 1-4 of the causal pipeline.

Runs:
  - Step 1: PTA aggregation (pharma tariff data)
  - Step 2: Pharma-specific PTA start years (WTO-X health provisions)
  - Step 3: Merge into pivot_dataset_fe + construct DiD indicators
  - Step 4: Staggered DiD (Scenario 1: EU controls, Scenario 2: never-treated)
              + cross-scenario LP-DiD comparison

Saves:
  - data/processed/panel_s2.parquet   (consumed by run_dml.py)
  - outputs/visualization/es_*.png    (event study plots)

Usage:
    python src/run_did.py
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycountry
import pyfixest as pf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Step 1: PTA aggregation ───────────────────────────────────────────────────
trade = pd.read_csv(os.path.join(_ROOT, "data", "raw", "Chemicals_Allied_Industries.csv"))

pta_agg = (
    trade.groupby("reporteriso3")["PTA"]
    .agg(pta_share="mean", has_pta="max")
    .reset_index()
    .rename(columns={"reporteriso3": "country_iso3"})
)
pta_agg["has_pta"] = pta_agg["has_pta"].astype(int)
print("Step 1 -- PTA aggregation:", pta_agg.shape)

# ── Step 2: Pharma-specific PTA start year (WTO-X Health provisions only) ────
WTO_PLUS_PATH = os.path.join(_ROOT, "data", "raw", "pta-agreements_1.xls")
wto_x = pd.read_excel(WTO_PLUS_PATH, sheet_name="WTO-X AC")
pharma_mask = wto_x["Health"].ge(1)
pharma = wto_x[pharma_mask][["Agreement", "year"]].copy()
pharma["year"] = pd.to_numeric(pharma["year"], errors="coerce")
pharma = pharma.dropna(subset=["year"])
pharma["year"] = pharma["year"].astype(int)

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
print("Step 2 -- PTA start years:", pta_pharma_start.shape)

# ── Step 3: Merge PTA variables into pivot_dataset_fe ────────────────────────
df = pd.read_csv(os.path.join(_ROOT, "data", "processed", "pivot_dataset_fe.csv"))
df = (
    df
    .merge(pta_agg,          on="country_iso3", how="left")
    .merge(pta_pharma_start, on="country_iso3", how="left")
)
df["pta_pharma_start_year"] = df["pta_pharma_start_year"].fillna(0).astype(int)
df["pta_treated"] = (
    (df["has_pta"] == 1) &
    (df["pta_pharma_start_year"] > 0) &
    (df["year"] >= df["pta_pharma_start_year"])
).astype(int)
df["years_since_pta"] = (df["year"] - df["pta_pharma_start_year"]).clip(lower=0)
df.loc[df["pta_pharma_start_year"] == 0, "years_since_pta"] = 0
df["relative_time"] = df["year"] - df["pta_pharma_start_year"]
df.loc[df["pta_pharma_start_year"] == 0, "relative_time"] = float("nan")
print("Step 3 -- Merged dataset:", df.shape)

# ── Step 4: Staggered DiD setup ───────────────────────────────────────────────
OUT_DIR = os.path.join(_ROOT, "outputs", "visualization")
PANEL_START   = 2001
COVARS        = ["gdp_per_capita_usd", "health_exp_pct_gdp", "oop_health_exp_pct", "population_total"]
ADJ_COVARS    = ["gdp_per_capita_usd", "health_exp_pct_gdp"]
_DESIRED_PRE  = -3
_DESIRED_POST = 10

df["treatment_group"] = "never_treated"
df.loc[(df["pta_pharma_start_year"] > 0) & (df["pta_pharma_start_year"] < PANEL_START), "treatment_group"] = "always_treated"
df.loc[df["pta_pharma_start_year"] >= PANEL_START, "treatment_group"] = "staggered"
print("\nStep 4a -- Treatment group counts:")
print(df.groupby("treatment_group")["country_iso3"].nunique())

EU_BLOC = {
    "AUT","BEL","BGR","HRV","CYP","CZE","DNK","EST","FIN","FRA","DEU","GRC",
    "HUN","IRL","ITA","LVA","LTU","LUX","MLT","NLD","POL","PRT","ROU","SVK",
    "SVN","ESP","SWE",
}
NEVER_TREATED_KEEP = {
    "BHR","KWT","OMN","QAT","SAU","ARE","CHE","ISL","NOR","GBR","ISR","NZL","PAN","URY",
    "ALB","ARM","AZE","BIH","BLR","MKD","MNE","SRB",
    "ARG","BOL","BRA","ECU","PER","PRY",
    "IDN","THA","LKA","KAZ","DZA","LBN","TUN",
}

# ── Plotting helpers ───────────────────────────────────────────────────────────
def _normalize_tidy(tidy_df):
    tidy_df = tidy_df.copy()
    if "estimate" not in tidy_df.columns:
        for c in ["att", "Estimate", "coef", "coefficient"]:
            if c in tidy_df.columns:
                tidy_df = tidy_df.rename(columns={c: "estimate"})
                break
    if "std error" not in tidy_df.columns:
        for c in ["se", "std_error", "Std. Error", "Std Error", "stderr"]:
            if c in tidy_df.columns:
                tidy_df = tidy_df.rename(columns={c: "std error"})
                break
    tidy_df["estimate"]  = pd.to_numeric(tidy_df["estimate"],  errors="coerce")
    tidy_df["std error"] = pd.to_numeric(tidy_df["std error"], errors="coerce")
    return tidy_df

def _pval_col(tidy_df):
    for c in ["Pr(>|t|)", "pvalue", "p-value", "p_value"]:
        if c in tidy_df.columns:
            return c
    return None

def _sig_stars(p):
    if p < 0.01:   return "***"
    elif p < 0.05: return "**"
    elif p < 0.10: return "*"
    return ""

def _event_times(tidy_df):
    if "t" in tidy_df.columns:
        return tidy_df["t"].values
    if "period" in tidy_df.columns:
        return tidy_df["period"].values
    if "term" in tidy_df.columns:
        extracted = tidy_df["term"].str.extract(r"(-?\d+)$")[0]
        if extracted.notna().all():
            return extracted.astype(int).values
    return np.arange(len(tidy_df))

def plot_single_es(tidy_df, title, color, filename, star_offset=None, xticklabels=None):
    tidy_df = _normalize_tidy(tidy_df)
    times = _event_times(tidy_df)
    pc    = _pval_col(tidy_df)
    ests  = tidy_df["estimate"].values
    ses   = tidy_df["std error"].values
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.axvline(-0.5, color="grey", lw=0.6, linestyle=":")
    ax.errorbar(times, ests, yerr=1.96 * ses,
                fmt="o-", capsize=4, color=color, linewidth=1.5, zorder=3)
    if pc is not None:
        for t, est, se, pv in zip(times, ests, ses, tidy_df[pc].values):
            stars = _sig_stars(pv)
            if stars:
                y_pos = est + (star_offset if star_offset is not None else 1.96 * se + 0.3)
                ax.text(t, y_pos, stars, ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color=color)
    ax.text(0.01, 0.99, "* p<0.10   ** p<0.05   *** p<0.01",
            transform=ax.transAxes, fontsize=8, va="top", color="dimgrey")
    ax.set_xticks(times)
    if xticklabels is not None and len(xticklabels) == len(times):
        ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    else:
        ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Event time (years relative to PTA)")
    ax.set_ylabel("ATT -- immunization coverage (pp)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {filename}")

def build_panel(df_in, label):
    panel = (
        df_in
        .groupby(["country_iso3", "year", "treatment_group", "pta_pharma_start_year"])
        .agg(
            immunization_coverage=("immunization_coverage", "mean"),
            **{c: (c, "mean") for c in COVARS}
        )
        .reset_index()
    )
    panel["gname"] = panel["pta_pharma_start_year"]
    _tr = panel[panel["gname"] > 0]
    _cc = (
        _tr.assign(_b=lambda d: d["year"] == d["gname"] - 1)
        .groupby(["gname", "country_iso3"])
        .agg(min_year=("year", "min"), has_base=("_b", "any"))
        .reset_index()
    )
    _cc["req_year"] = _cc["gname"] + _DESIRED_PRE
    _cc["is_clean"] = (_cc["min_year"] <= _cc["req_year"]) & _cc["has_base"]
    _vc = _cc[_cc["is_clean"]].groupby("gname").size()
    _vc = _vc[_vc >= 1].index.tolist()
    _ck = _cc.loc[_cc["is_clean"] & _cc["gname"].isin(_vc), ["gname", "country_iso3"]]
    _tc = _tr.merge(_ck, on=["gname", "country_iso3"], how="inner")
    panel = pd.concat([panel[panel["gname"] == 0], _tc], ignore_index=True).copy()
    _rel = panel["year"] - panel["gname"]
    panel = panel[
        (panel["gname"] == 0) | ((_rel >= _DESIRED_PRE) & (_rel <= _DESIRED_POST))
    ].copy()
    _actual_post = int(
        (panel.loc[panel["gname"] > 0, "year"] - panel.loc[panel["gname"] > 0, "gname"]).max()
    )
    post = min(_actual_post, _DESIRED_POST)
    panel["country_id"] = pd.factorize(panel["country_iso3"])[0] + 1
    _tr2 = panel[panel["gname"] > 0]
    print(f"\n[{label}] Treated: {sorted(_tr2['country_iso3'].unique().tolist())}")
    print(f"[{label}] Controls: {panel[panel['gname']==0]['country_iso3'].nunique()}, Post-window: {post}")
    return panel, post


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 1: Treated vs EU controls
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SCENARIO 1: Treated vs EU controls (convergence framing)")
print("="*60)

df_s1 = df[df["country_iso3"].isin(EU_BLOC) | (df["treatment_group"] == "staggered")].copy()
df_s1.loc[df_s1["country_iso3"].isin(EU_BLOC), "treatment_group"]       = "never_treated"
df_s1.loc[df_s1["country_iso3"].isin(EU_BLOC), "pta_pharma_start_year"] = 0

panel_s1, post_s1 = build_panel(df_s1, "S1")

fit_twfe_s1 = pf.event_study(data=panel_s1, yname="immunization_coverage",
    idname="country_id", tname="year", gname="gname",
    estimator="twfe", att=False, cluster="country_id")

fit_sa_s1 = pf.event_study(data=panel_s1, yname="immunization_coverage",
    idname="country_id", tname="year", gname="gname",
    estimator="saturated", cluster="country_id")
es_sa_s1 = fit_sa_s1.aggregate()

fit_lp_s1 = pf.lpdid(data=panel_s1, yname="immunization_coverage",
    idname="country_id", tname="year", gname="gname",
    vcov={"CRV1": "country_id"},
    pre_window=_DESIRED_PRE, post_window=post_s1, never_treated=0, att=False)
es_lp_s1 = fit_lp_s1.tidy()

_sa_labels_s1 = [str(t) for t in sorted(list(range(_DESIRED_PRE, 0)) + list(range(0, post_s1 + 1))) if t != -1]
plot_single_es(fit_twfe_s1.tidy(), "S1: TWFE -- EU controls (benchmark)", "tomato", os.path.join(OUT_DIR, "es_s1_twfe.png"))
plot_single_es(es_sa_s1, "S1: Sun & Abraham -- EU controls", "steelblue",
               os.path.join(OUT_DIR, "es_s1_sa.png"), star_offset=0.05, xticklabels=_sa_labels_s1)
plot_single_es(es_lp_s1, "S1: LP-DiD -- EU controls", "seagreen", os.path.join(OUT_DIR, "es_s1_lpdid.png"))


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 2: Treated vs Never-treated
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SCENARIO 2: Treated vs Never-treated (covariate-adjusted)")
print("="*60)

df_s2 = df[
    (df["country_iso3"].isin(NEVER_TREATED_KEEP) & (df["treatment_group"] == "never_treated")) |
    (df["treatment_group"] == "staggered")
].copy()

panel_s2, post_s2 = build_panel(df_s2, "S2")

fit_twfe_s2 = pf.event_study(data=panel_s2, yname="immunization_coverage",
    idname="country_id", tname="year", gname="gname",
    estimator="twfe", att=False, cluster="country_id")

fit_sa_s2 = pf.event_study(data=panel_s2, yname="immunization_coverage",
    idname="country_id", tname="year", gname="gname",
    estimator="saturated", cluster="country_id")
es_sa_s2 = fit_sa_s2.aggregate()

fit_lp_s2 = pf.lpdid(data=panel_s2, yname="immunization_coverage",
    idname="country_id", tname="year", gname="gname",
    vcov={"CRV1": "country_id"},
    pre_window=_DESIRED_PRE, post_window=post_s2, never_treated=0, att=False)
es_lp_s2 = fit_lp_s2.tidy()

_sa_labels_s2 = [str(t) for t in sorted(list(range(_DESIRED_PRE, 0)) + list(range(0, post_s2 + 1))) if t != -1]
plot_single_es(fit_twfe_s2.tidy(), "S2: TWFE -- Never-treated controls (benchmark)", "tomato", os.path.join(OUT_DIR, "es_s2_twfe.png"))
plot_single_es(es_sa_s2, "S2: Sun & Abraham -- Never-treated controls", "steelblue",
               os.path.join(OUT_DIR, "es_s2_sa.png"), star_offset=0.05, xticklabels=_sa_labels_s2)
plot_single_es(es_lp_s2, "S2: LP-DiD -- Never-treated controls", "seagreen", os.path.join(OUT_DIR, "es_s2_lpdid.png"))


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-SCENARIO COMPARISON: LP-DiD S1 vs S2
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 6))
for tidy_df, label, color, fmt, alpha in [
    (es_lp_s1, "S1: LP-DiD vs EU controls (convergence)", "steelblue", "o-",  1.0),
    (es_lp_s2, "S2: LP-DiD vs Never-treated controls",    "seagreen",  "s--", 0.9),
]:
    tidy_df = _normalize_tidy(tidy_df)
    times = _event_times(tidy_df)
    ests  = tidy_df["estimate"].values
    ses   = tidy_df["std error"].values
    ax.fill_between(times, ests - 1.96 * ses, ests + 1.96 * ses, color=color, alpha=0.12)
    ax.plot(times, ests, fmt, color=color, label=label, linewidth=1.8, markersize=5, alpha=alpha)

ax.axhline(0, color="black", lw=0.8, linestyle="--")
ax.axvline(-0.5, color="grey", lw=0.6, linestyle=":", label="treatment onset")
ax.set_xlabel("Event time (years relative to PTA)")
ax.set_ylabel("ATT -- immunization coverage (pp)")
ax.set_title("LP-DiD: S1 (EU controls) vs S2 (Never-treated, covariate-adjusted)\n"
             "Pharma PTA effect on immunization coverage (95% CI shaded)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "es_cross_scenario.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Cross-scenario comparison saved -> es_cross_scenario.png")

# ── Save panel_s2 for run_dml.py ──────────────────────────────────────────────
_panel_path = os.path.join(_ROOT, "data", "processed", "panel_s2.parquet")
panel_s2.to_parquet(_panel_path, index=False)
print(f"\nSaved panel_s2 -> {_panel_path}")
print("DiD pipeline complete. Run src/run_dml.py next for heterogeneous effects.")

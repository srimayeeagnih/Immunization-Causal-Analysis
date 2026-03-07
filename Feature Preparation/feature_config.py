"""
feature_config.py
─────────────────
Central governance module for the immunization-tariff project.
Import this from EDA.py, feature engineering, and modelling scripts
to keep naming conventions, metadata, and covariate lists consistent.

Usage
-----
    from feature_config import (
        ANTIGEN_REGISTRY, COVARIATE_REGISTRY, ISO3_COL,
        get_active_covariates, get_antigen_family, print_data_dictionary,
    )
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
# ANTIGEN REGISTRY
# Update here when WHO renames or adds antigen codes.
# Keys  : WHO/UNICEF WUENIC vaccine codes (uppercase).
# Values: family    – grouped label used in analysis
#         display   – human-readable name for plots / reports
#         who_code  – canonical WHO code (mirrors key, kept for explicitness)
#         intro_year– year the WHO global recommendation was issued
# ══════════════════════════════════════════════════════════════════════════════
ANTIGEN_REGISTRY: dict[str, dict] = {
    "HIB3": {
        "family":     "Hib",
        "display":    "Haemophilus influenzae type b – 3rd dose",
        "who_code":   "HIB3",
        "intro_year": 1998,   # WHO position paper
    },
    "PCV3": {
        "family":     "PCV",
        "display":    "Pneumococcal conjugate vaccine – 3rd dose",
        "who_code":   "PCV3",
        "intro_year": 2007,   # WHO position paper
    },
    "ROTAC": {
        "family":     "Rotavirus",
        "display":    "Rotavirus vaccine – completed series",
        "who_code":   "ROTAC",
        "intro_year": 2006,   # WHO position paper
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# ISO3 CONVENTIONS
# ══════════════════════════════════════════════════════════════════════════════
ISO3_COL      = "country_iso3"      # canonical column name in every dataset
ISO3_STANDARD = "ISO 3166-1 alpha-3"

# Known ISO3 quirks / legacy codes that may appear in raw source files.
# Map non-standard codes → standard ISO3.  Add entries as encountered.
ISO3_ALIASES: dict[str, str] = {
    "ROM": "ROU",   # Romania (legacy UN code)
    "ZAR": "COD",   # DR Congo (pre-1997 code)
    "TMP": "TLS",   # Timor-Leste (transitional code)
}

def resolve_iso3(code: str) -> str:
    """Return the canonical ISO3 code, resolving known aliases."""
    code = code.upper().strip()
    return ISO3_ALIASES.get(code, code)


# ══════════════════════════════════════════════════════════════════════════════
# COVARIATE REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
# role options:
#   "target"   – outcome variable
#   "feature"  – active predictor
#   "metadata" – identifier / bookkeeping, not modelled
#   "dropped"  – excluded with documented reason
#
# available_from / available_until: earliest / latest year the SOURCE reliably
#   provides this variable across the sample countries.  Per-country gaps
#   within that range are handled separately (see imputation_strategy below).
#
# imputation_strategy:
#   None        – no imputation applied yet
#   "interp"    – linear interpolation within country time series
#   "region_median" – fill with UNICEF-region median for that year
#   "listwise"  – drop rows with missing values for this column
# ══════════════════════════════════════════════════════════════════════════════
COVARIATE_REGISTRY: dict[str, dict] = {

    # ── Target ────────────────────────────────────────────────────────────────
    "immunization_coverage": {
        "label":                "Immunization Coverage (%)",
        "source":               "WHO/UNICEF WUENIC 2024 revision",
        "dtype":                "float",
        "expected_range":       (0, 100),
        "role":                 "target",
        "available_from":       2001,
        "available_until":      2023,
        "imputation_strategy":  None,
        "dropped":              False,
        "drop_reason":          None,
        "eda_notes":            "Left-skewed (heavy mass 80–99%); no missing values in pivot dataset.",
    },

    # ── Active features ────────────────────────────────────────────────────────
    "pharma_tariff_rate": {
        "label":                "Pharmaceutical Tariff Rate (%)",
        "source":               "WITS / WTO Chemicals & Allied Industries",
        "dtype":                "float",
        "expected_range":       (0, None),
        "role":                 "feature",
        "available_from":       2001,
        "available_until":      2023,
        "imputation_strategy":  None,
        "dropped":              False,
        "drop_reason":          None,
        "eda_notes":            "Right-skewed; log1p transform recommended. Spearman rho=-0.13 with coverage.",
    },
    "gdp_per_capita_usd": {
        "label":                "GDP per Capita (current USD)",
        "source":               "World Bank WDI (NY.GDP.PCAP.CD)",
        "dtype":                "float",
        "expected_range":       (0, None),
        "role":                 "feature",
        "available_from":       2001,
        "available_until":      2023,
        "imputation_strategy":  None,
        "dropped":              False,
        "drop_reason":          None,
        "eda_notes":            "Spearman rho=0.22 with coverage. Strongly collinear with gni_per_capita_usd (rho=0.997).",
    },
    "health_exp_pct_gdp": {
        "label":                "Current Health Expenditure (% of GDP)",
        "source":               "World Bank WDI (SH.XPD.CHEX.GD.ZS)",
        "dtype":                "float",
        "expected_range":       (0, 100),
        "role":                 "feature",
        "available_from":       2001,
        "available_until":      2023,
        "imputation_strategy":  None,   # TODO: interp for MNE 2006-2010
        "dropped":              False,
        "drop_reason":          None,
        "eda_notes":            "5 missing cells: MNE 2006-2010. Verdict: MAR (predicted by year). "
                                "Spearman rho=-0.06 with coverage.",
    },
    "population_total": {
        "label":                "Total Population",
        "source":               "World Bank WDI (SP.POP.TOTL)",
        "dtype":                "int",
        "expected_range":       (0, None),
        "role":                 "feature",
        "available_from":       2001,
        "available_until":      2023,
        "imputation_strategy":  None,
        "dropped":              False,
        "drop_reason":          None,
        "eda_notes":            "Spearman rho=-0.12 with coverage. Consider log-transform.",
    },
    "oop_health_exp_pct": {
        "label":                "Out-of-Pocket Health Expenditure (% of current health exp.)",
        "source":               "World Bank WDI (SH.XPD.OOPC.CH.ZS)",
        "dtype":                "float",
        "expected_range":       (0, 100),
        "role":                 "feature",
        "available_from":       2001,
        "available_until":      2023,
        "imputation_strategy":  None,   # TODO: interp for MNE 2006-2010
        "dropped":              False,
        "drop_reason":          None,
        "eda_notes":            "5 missing cells: MNE 2006-2010. Verdict: MAR (predicted by year). "
                                "Spearman rho=-0.14 with coverage.",
    },
    "tariff_x_oop": {
        "label":                "Pharma Tariff × OOP Interaction",
        "source":               "Derived (pharma_tariff_rate * oop_health_exp_pct / 100)",
        "dtype":                "float",
        "expected_range":       (0, None),
        "role":                 "feature",
        "available_from":       2001,
        "available_until":      2023,
        "imputation_strategy":  None,   # propagates MNE missingness from oop_health_exp_pct
        "dropped":              False,
        "drop_reason":          None,
        "eda_notes":            "Interaction term. Collinear with pharma_tariff_rate (rho=0.78). "
                                "Monitor VIF; may be dropped pre-modelling.",
    },

    # ── Metadata (kept in pv_plot, not modelled) ──────────────────────────────
    "reporter_flag": {
        "label":                "Tariff Reporter Flag",
        "source":               "WITS",
        "dtype":                "int",
        "expected_range":       (0, 1),
        "role":                 "metadata",
        "available_from":       2001,
        "available_until":      2023,
        "imputation_strategy":  None,
        "dropped":              False,
        "drop_reason":          None,
        "eda_notes":            "1 = country reported tariff data for that year.",
    },
    "unicef_region": {
        "label":                "UNICEF Regional Classification",
        "source":               "UNICEF",
        "dtype":                "str",
        "expected_range":       None,
        "role":                 "metadata",
        "available_from":       2001,
        "available_until":      2023,
        "imputation_strategy":  None,
        "dropped":              False,
        "drop_reason":          None,
        "eda_notes":            "Used for regional cluster labels in Plot P4.",
    },
    "country": {
        "label":                "Country Name",
        "source":               "ISO 3166-1",
        "dtype":                "str",
        "expected_range":       None,
        "role":                 "metadata",
        "available_from":       2001,
        "available_until":      2023,
        "imputation_strategy":  None,
        "dropped":              False,
        "drop_reason":          None,
        "eda_notes":            None,
    },
    "vaccine": {
        "label":                "Vaccine / Antigen Code",
        "source":               "WHO/UNICEF WUENIC",
        "dtype":                "str",
        "expected_range":       None,
        "role":                 "metadata",
        "available_from":       2001,
        "available_until":      2023,
        "imputation_strategy":  None,
        "dropped":              False,
        "drop_reason":          None,
        "eda_notes":            "Values: HIB3, PCV3, ROTAC. See ANTIGEN_REGISTRY.",
    },
    "antigen_family": {
        "label":                "Antigen Family",
        "source":               "Derived from vaccine column",
        "dtype":                "str",
        "expected_range":       None,
        "role":                 "metadata",
        "available_from":       2001,
        "available_until":      2023,
        "imputation_strategy":  None,
        "dropped":              False,
        "drop_reason":          None,
        "eda_notes":            "Values: Hib, PCV, Rotavirus.",
    },
    "year": {
        "label":                "Observation Year",
        "source":               "All sources",
        "dtype":                "int",
        "expected_range":       (2001, 2023),
        "role":                 "metadata",
        "available_from":       2001,
        "available_until":      2023,
        "imputation_strategy":  None,
        "dropped":              False,
        "drop_reason":          None,
        "eda_notes":            None,
    },

    # ── Dropped columns ────────────────────────────────────────────────────────
    "gavi_eligible": {
        "label":                "GAVI Eligibility Flag",
        "source":               "GAVI Alliance",
        "dtype":                "int",
        "expected_range":       (0, 1),
        "role":                 "dropped",
        "available_from":       2001,
        "available_until":      2023,
        "imputation_strategy":  None,
        "dropped":              True,
        "drop_reason":          "EDA step P5: sample is already filtered to non-GAVI countries; "
                                "column is constant (=0) and uninformative.",
        "eda_notes":            None,
    },
    "gni_per_capita_usd": {
        "label":                "GNI per Capita (current USD)",
        "source":               "World Bank WDI (NY.GNP.PCAP.CD)",
        "dtype":                "float",
        "expected_range":       (0, None),
        "role":                 "dropped",
        "available_from":       2001,
        "available_until":      2023,
        "imputation_strategy":  None,
        "dropped":              True,
        "drop_reason":          "EDA step P9: Spearman rho=0.997 with gdp_per_capita_usd; "
                                "near-perfect collinearity. Retaining GDP is sufficient.",
        "eda_notes":            None,
    },
    "country_iso3": {
        "label":                "ISO3 Country Code",
        "source":               "ISO 3166-1",
        "dtype":                "str",
        "expected_range":       None,
        "role":                 "dropped",
        "available_from":       2001,
        "available_until":      2023,
        "imputation_strategy":  None,
        "dropped":              True,
        "drop_reason":          "EDA step P6: dropped from modelling matrix; "
                                "retained in pv_plot for joins and plots.",
        "eda_notes":            None,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_active_covariates(
    year: int | None = None,
    roles: tuple[str, ...] = ("feature",),
) -> list[str]:
    """
    Return covariate column names that are active (not dropped) and match
    the requested roles.  Optionally filter to those available in a given year.

    Parameters
    ----------
    year  : int | None
        Observation year to filter against available_from / available_until.
        Pass None to skip year filtering.
    roles : tuple[str, ...]
        Subset of {"target", "feature", "metadata"} to include.

    Examples
    --------
    >>> get_active_covariates()                  # all active features
    >>> get_active_covariates(year=2005)         # features available in 2005
    >>> get_active_covariates(roles=("target",)) # just the outcome
    """
    result = []
    for col, meta in COVARIATE_REGISTRY.items():
        if meta["dropped"]:
            continue
        if meta["role"] not in roles:
            continue
        if year is not None:
            if not (meta["available_from"] <= year <= meta["available_until"]):
                continue
        result.append(col)
    return result


def normalize_antigen(code: str) -> dict:
    """
    Return full metadata for a WHO antigen code.
    Raises KeyError for unknown codes so callers fail loudly.
    """
    code = code.upper().strip()
    if code not in ANTIGEN_REGISTRY:
        known = list(ANTIGEN_REGISTRY)
        raise KeyError(f"Unknown antigen '{code}'. Known codes: {known}")
    return ANTIGEN_REGISTRY[code]


def get_antigen_family(code: str) -> str:
    """Convenience wrapper: antigen code -> family string."""
    return normalize_antigen(code)["family"]


def print_data_dictionary(include_dropped: bool = True) -> None:
    """Pretty-print the full data dictionary to stdout."""
    width = 76

    print("=" * width)
    print("  DATA DICTIONARY")
    print("=" * width)

    for col, meta in COVARIATE_REGISTRY.items():
        if not include_dropped and meta["dropped"]:
            continue
        status = "[DROPPED]" if meta["dropped"] else f"[{meta['role'].upper()}]"
        print(f"\n  {col}  {status}")
        print(f"    Label    : {meta['label']}")
        print(f"    Source   : {meta['source']}")
        print(f"    Type     : {meta['dtype']}")
        print(f"    Range    : {meta['expected_range']}")
        print(f"    Years    : {meta['available_from']} – {meta['available_until']}")
        if meta["dropped"] and meta["drop_reason"]:
            print(f"    Dropped  : {meta['drop_reason']}")
        if meta["imputation_strategy"]:
            print(f"    Impute   : {meta['imputation_strategy']}")
        if meta["eda_notes"]:
            print(f"    EDA note : {meta['eda_notes']}")

    print()
    print("=" * width)
    print("  ANTIGEN REGISTRY")
    print("=" * width)
    for code, meta in ANTIGEN_REGISTRY.items():
        print(f"  {code:8s}  family={meta['family']:12s}  "
              f"intro={meta['intro_year']}  {meta['display']}")

    print()
    print("=" * width)
    print("  ISO3 CONVENTIONS")
    print("=" * width)
    print(f"  Standard    : {ISO3_STANDARD}")
    print(f"  Column name : {ISO3_COL}")
    if ISO3_ALIASES:
        print("  Known aliases (non-standard -> canonical):")
        for alias, canon in ISO3_ALIASES.items():
            print(f"    {alias} -> {canon}")
    print()


# ── Quick self-test when run directly ────────────────────────────────────────
if __name__ == "__main__":
    print_data_dictionary()

    print("Active features (all years):")
    print(" ", get_active_covariates())

    print("\nActive features available in year 2005:")
    print(" ", get_active_covariates(year=2005))

    print("\nAntigen lookup: ROTAC")
    print(" ", normalize_antigen("ROTAC"))

    print("\nISO3 resolve: ROM ->", resolve_iso3("ROM"))

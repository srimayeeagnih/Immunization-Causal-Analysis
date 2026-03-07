
"""Business Problem:
How can this model be productionized?:
Who are the key stakeholders?: Immunization programs. A low-code """




"""
data_strategy.py
Vaccine Causal Project — Data Loading, Naming Harmonisation, and Merge Pipeline

Causal chain:
    Pharma Tariff (country, static/annual)
        → Vaccine Price (region, mediator)
        → Immunization Coverage (country × year × vaccine type)

Naming strategy:
    WUENIC uses dose-level program codes  (e.g. MCV1, DTP3, HEPB3)
    MI4A   uses product/formulation names (e.g. MMR, DTwP-HepB-Hib, HepB (ped.))
    Solution: introduce a canonical ANTIGEN_FAMILY as the join key for both.
"""

import pandas as pd
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent if "__file__" in dir() else Path("c:/Users/srima/OneDrive/Desktop/INSY674/Individual Project 1")
IMMUNIZ    = ROOT / "wuenic2024rev_web-update.xlsx" # WUENIC immunization coverage data
PRICES     = ROOT / "Data" / "who-mi4a-dataset-final-september-2025.xlsx" # MI4A vaccine price data (multiple sheets)
TARIFF_CSV = ROOT / "1.-preferential-tariffs.csv (1)" / "1. Preferential Tariffs.csv"  #Preferential tariff data
TARIFF_PKL = ROOT / "1.-preferential-tariffs.csv (1)" / "1. Preferential Tariffs.parquet" # Parquet version for faster loading
PHARMA     = ROOT / "Data" / "Chemicals_Allied_Industries.csv" # Chemicals & Allied Industries tariff data (HS chapter 30 proxy taken from preferential tariff dataset)


# ── 1. Vaccine Type Naming Harmonisation ──────────────────────────────────────
# Strategy: map both WUENIC codes and MI4A vaccine types to a shared
# ANTIGEN_FAMILY string, then join on that key.
#
# Design rules:
#   - ANTIGEN_FAMILY is disease-target focused, not formulation focused
#   - Combination vaccines (e.g. DTwP-HepB-Hib) map to ALL constituent families
#   - Dose suffixes (1, 2, 3, B) are stripped from WUENIC codes
#   - Many-to-one: multiple MI4A types → one family (aggregation needed on price side)

# WUENIC code → canonical antigen family
WUENIC_TO_ANTIGEN: dict[str, str] = {
    "BCG"    : "BCG",
    "DTP1"   : "DTP",
    "DTP3"   : "DTP",
    "HEPB3"  : "HepB",
    "HEPBB"  : "HepB",           # birth dose
    "HIB3"   : "Hib",
    "IPV1"   : "Polio",
    "IPV2"   : "Polio",
    "MCV1"   : "Measles",
    "MCV2"   : "Measles",
    "MENGA"  : "MenA",
    "PCV3"   : "PCV",
    "POL3"   : "Polio",
    "RCV1"   : "Rubella",
    "ROTAC"  : "Rotavirus",
    "YFV"    : "YellowFever",
}

# MI4A vaccine type → canonical antigen family (or list if combination)
# Combination vaccines contribute to multiple families — handled below.
MI4A_TO_ANTIGEN: dict[str, list[str]] = {
    # Single-antigen
    "BCG"                             : ["BCG"],
    "Measles"                         : ["Measles"],
    "Rubella"                         : ["Rubella"],
    "Mumps"                           : ["Mumps"],
    "MM"                              : ["Measles", "Mumps"],
    "MR"                              : ["Measles", "Rubella"],
    "MMR"                             : ["Measles", "Mumps", "Rubella"],
    "MMRV"                            : ["Measles", "Mumps", "Rubella"],
    "HepB (ped.)"                     : ["HepB"],
    "HepB (adult)"                    : ["HepB"],
    "Hib"                             : ["Hib"],
    "Hib-MenC conj."                  : ["Hib"],
    "IPV"                             : ["Polio"],
    "IPV fractional dose"             : ["Polio"],
    "sIPV"                            : ["Polio"],
    "tOPV"                            : ["Polio"],
    "bOPV"                            : ["Polio"],
    "mOPV1"                           : ["Polio"],
    "mOPV2"                           : ["Polio"],
    "mOPV3"                           : ["Polio"],
    "nOPV2"                           : ["Polio"],
    "PCV7"                            : ["PCV"],
    "PCV10"                           : ["PCV"],
    "PCV13"                           : ["PCV"],
    "PCV15"                           : ["PCV"],
    "PCV20"                           : ["PCV"],
    "PPSV23"                          : ["PCV"],
    "RV1"                             : ["Rotavirus"],
    "RV5"                             : ["Rotavirus"],
    "Rota"                            : ["Rotavirus"],
    "YF"                              : ["YellowFever"],
    "MenA conj."                      : ["MenA"],
    "MenA Ps"                         : ["MenA"],
    "MenAC conj."                     : ["MenA"],
    "MenAC Ps"                        : ["MenA"],
    "MenACW-135 Ps"                   : ["MenA"],
    "MenACYW-135 conj."               : ["MenA"],
    "MenACYW-135 Ps"                  : ["MenA"],
    "MenACYWX conj."                  : ["MenA"],
    "MenB"                            : ["MenB"],
    "MenBC"                           : ["MenB"],
    "MenC conj."                      : ["MenC"],
    # DTP combinations — all map to DTP family for price proxy
    "DTwP"                            : ["DTP"],
    "DTaP"                            : ["DTP"],
    "DT"                              : ["DTP"],
    "DTwP-HepB"                       : ["DTP", "HepB"],
    "DTwP-Hib"                        : ["DTP", "Hib"],
    "DTaP-Hib"                        : ["DTP", "Hib"],
    "DTwP-HepB-Hib"                   : ["DTP", "HepB", "Hib"],
    "DTwP-HepB-Hib-IPV"              : ["DTP", "HepB", "Hib", "Polio"],
    "DTaP-HepB-Hib-IPV"              : ["DTP", "HepB", "Hib", "Polio"],
    "DTaP-HepB-IPV"                   : ["DTP", "HepB", "Polio"],
    "DTaP-HepB-_x000D_\nIPV"         : ["DTP", "HepB", "Polio"],   # data artifact = same as above
    "DTaP-Hib-IPV"                    : ["DTP", "Hib", "Polio"],
    "DTaP-IPV"                        : ["DTP", "Polio"],
    "DT-IPV"                          : ["DTP", "Polio"],
    "Td"                              : ["DTP"],
    "Tdap"                            : ["DTP"],
    "Tdap-IPV"                        : ["DTP", "Polio"],
    "Td-IPV"                          : ["DTP", "Polio"],
    # Outside WUENIC scope — kept for completeness
    "HPV2"                            : ["HPV"],
    "HPV4"                            : ["HPV"],
    "HPV9"                            : ["HPV"],
    "HepA (adult)"                    : ["HepA"],
    "HepA (ped.)"                     : ["HepA"],
    "HepA+B"                          : ["HepA", "HepB"],
    "HepA-Typhoid"                    : ["HepA", "Typhoid"],
    "TCV"                             : ["Typhoid"],
    "Typhoid Ps"                      : ["Typhoid"],
    "Typhoid-Tetanus"                 : ["Typhoid"],
    "TT"                              : ["Tetanus"],
    "Varicella"                       : ["Varicella"],
    "Shingles"                        : ["Varicella"],
    "COVID-19"                        : ["COVID19"],
    "Dengue"                          : ["Dengue"],
    "Malaria"                         : ["Malaria"],
    "Rabies"                          : ["Rabies"],
    "RSV"                             : ["RSV"],
    "RSV mAb"                         : ["RSV"],
    "OCV"                             : ["Cholera"],
    "TBE (adult)"                     : ["TBE"],
    "TBE (ped.)"                      : ["TBE"],
    "JE inactivated"                  : ["JE"],
    "JE inactivated (ped.)"           : ["JE"],
    "JE live attenuated"              : ["JE"],
    "JE live attenuated (adult)"      : ["JE"],
    "JE live attenuated (ped.)"       : ["JE"],
    "influenza adjuvant vaccine"      : ["Influenza"],
    "influenza high dose vaccine"     : ["Influenza"],
    "influenza inactivated vaccine"   : ["Influenza"],
    "influenza quadrivalent inactivated vaccine"    : ["Influenza"],
    "influenza quadrivalent live attenuated vaccine": ["Influenza"],
    "influenza trivalent inactivated vaccine"       : ["Influenza"],
    "influenza trivalent live attenuated vaccine"   : ["Influenza"],
    "Anthrax"                         : ["Anthrax"],
    "CCHF"                            : ["CCHF"],
    "Mpox"                            : ["Mpox"],
    "Tularemia"                       : ["Tularemia"],
    "Diphtheria"                      : ["Diphtheria"],
    "Other"                           : ["Other"],
}

# Antigen families covered by WUENIC (used to filter MI4A prices to relevant subset)
WUENIC_ANTIGEN_FAMILIES = set(WUENIC_TO_ANTIGEN.values())


# ── 2. Data Loading ───────────────────────────────────────────────────────────

def load_immunization() -> pd.DataFrame:
    """
    Load WUENIC data. The file has one sheet per vaccine code (BCG, DTP1, ...)
    plus a combined 'regional_global' summary sheet. We load the summary sheet
    and add antigen_family. If the summary sheet is missing, we fall back to
    reading and concatenating all individual vaccine sheets.
    """
    xl = pd.ExcelFile(IMMUNIZ)
    # Note: the regional_global sheet contains only regional aggregates (no iso3 / country rows)
    # and uses "region" instead of "unicef_region" — always use individual vaccine sheets instead.
    frames = []
    for sheet in xl.sheet_names:
        if sheet in WUENIC_TO_ANTIGEN:
            tmp = xl.parse(sheet)
            tmp["vaccine"] = sheet
            frames.append(tmp)
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()
    df["antigen_family"] = df["vaccine"].map(WUENIC_TO_ANTIGEN)
    unmapped = df.loc[df["antigen_family"].isna(), "vaccine"].unique()
    if len(unmapped):
        print(f"[immunization] Unmapped WUENIC codes: {unmapped}")

    # Melt from wide (year columns) to long — creates Year + immunization_coverage columns
    year_cols = [c for c in df.columns if str(c).isdigit() and int(str(c)) >= 1980]
    df = df.drop(columns=["0"], errors="ignore")
    df = df.melt(
        id_vars=["unicef_region", "iso3", "country", "vaccine", "antigen_family"],
        value_vars=year_cols,
        var_name="year",
        value_name="immunization_coverage",
    )
    df["year"] = df["year"].astype(int)
    df = df.rename(columns={"iso3": "country_iso3"})
    return df


def load_vaccine_prices() -> pd.DataFrame:
    """
    Load MI4A price data and explode combination vaccines into constituent
    antigen families so they can join with WUENIC coverage data.
    """
    df = pd.read_excel(PRICES, sheet_name=None)          # load all sheets first
    # Identify the main price sheet (exclude glossary and instructions sheets)
    price_sheet = [s for s in df.keys() if "glossary" not in s.lower() and "instruction" not in s.lower()][0]
    prices = df[price_sheet].copy()
    prices.columns = prices.columns.str.strip()

    # Map to antigen families (one row may expand to multiple families)
    prices["antigen_family"] = prices["Vaccine"].map(MI4A_TO_ANTIGEN) #The Vaccine type column is coming from 
    prices = prices.explode("antigen_family")

    # Restrict to antigen families that appear in WUENIC
    prices = prices[prices["antigen_family"].isin(WUENIC_ANTIGEN_FAMILIES)] 
    return prices


def load_tariffs() -> pd.DataFrame: #This
    """Load static tariff data (preferential rates)."""
    if TARIFF_PKL.exists():
        df = pd.read_parquet(TARIFF_PKL)
    else:
        df = pd.read_csv(TARIFF_CSV)
    df.columns = df.columns.str.strip()
    return df


def load_pharma_tariffs() -> pd.DataFrame: 
    """Better to just load this is since it's the most relevant subset of the preferential tariff data for our analysis, 
    and the CSV is small enough to load quickly. But let's give the option to load the full preferential tariff data if needed for future exploration."""
    df = pd.read_csv(PHARMA)
    df.columns = df.columns.str.strip()
    return df


# ── 3. Reporter Country Extraction ───────────────────────────────────────────

def get_reporter_countries(tariffs: pd.DataFrame) -> list[str]:
    """Extract unique reporter ISO3 codes — Phase 1 analysis countries."""
    return sorted(tariffs["reporteriso3"].dropna().unique().tolist())


def get_partner_only_countries(tariffs: pd.DataFrame) -> list[str]:
    """Countries that appear only as partneriso3, never as reporteriso3 — Phase 2."""
    reporters = set(tariffs["reporteriso3"].dropna().unique())
    partners  = set(tariffs["partneriso3"].dropna().unique())
    return sorted(partners - reporters)


# ── 4. WITS/TRAINS Annual Tariff Pull ─────────────────────────────────────────

def _iso3_to_wits_numeric(iso3_codes: list[str]) -> dict[str, str]:
    """
    Convert ISO3 alpha codes to WITS ISO numeric codes using the
    world_trade_data country list. WITS rejects alpha codes with
    Invalid_Reporter — numeric codes are required.
    """
    import world_trade_data as wits
    countries = wits.get_countries()
    # Column names vary by library version — find the code and name cols
    code_col = next(c for c in countries.columns if "iso" in c.lower() or "code" in c.lower())
    name_col = next((c for c in countries.columns if "name" in c.lower()), None)

    # Build alpha3 → numeric lookup via country_converter if available
    try:
        import country_converter as coco
        names = countries[name_col].tolist() if name_col else []
        numeric_codes = countries[code_col].astype(str).str.zfill(3).tolist()
        alpha3_raw = coco.convert(names, to="ISO3", not_found=None)
        alpha3_list = [v[0] if isinstance(v, list) and len(v) == 1 else (None if isinstance(v, list) else v) for v in alpha3_raw]
        lookup = {a: n for a, n in zip(alpha3_list, numeric_codes) if a is not None}
    except ImportError:
        lookup = {}

    result = {}
    for code in iso3_codes:
        if code in lookup:
            result[code] = lookup[code]
        else:
            print(f"  [warn] No WITS numeric code found for {code} — skipping")
    return result


def pull_annual_tariffs(
    reporter_countries: list[str],   # ISO3 alpha codes (e.g. "USA", "DEU")
    years: range = range(2000, 2024),
    product: str = "ALLPRODUCTS",    # ALLPRODUCTS confirmed working; HS chapter "30" is invalid
    sleep_s: float = 0.5,
    out_csv: Path | None = None,     # if set, save raw pull to this CSV path
) -> pd.DataFrame:
    """
    Pull annual MFN tariff time series from WITS/TRAINS for reporter countries.

    API notes (confirmed by smoke-test):
      - Country codes  : WITS ISO numeric (e.g. "840" for USA), NOT ISO3 alpha.
                         Conversion is handled automatically via _iso3_to_wits_numeric().
      - Product codes  : "ALLPRODUCTS" works. HS 2-digit chapter codes (e.g. "30")
                         return Invalid_Product. HS6 codes untested.
      - Returned data  : tariff rate distribution — each row is a rate value with
                         counts of HS lines at that rate (Value, TotalNoOfLines, etc.)

    To get a single per-country/year treatment variable, aggregate with
    aggregate_mfn_tariff_rate() after pulling.

    Requires: pip install world_trade_data country_converter
    """
    try:
        import world_trade_data as wits
    except ImportError:
        raise ImportError("Run: pip install world_trade_data")

    # Convert ISO3 alpha → WITS numeric
    code_map = _iso3_to_wits_numeric(reporter_countries)
    if not code_map:
        raise ValueError("Could not map any reporter countries to WITS numeric codes.")

    results = []
    total = len(code_map) * len(years)
    done  = 0

    for iso3, numeric in code_map.items():
        for year in years:
            try:
                df = wits.get_tariff_reported(
                    reporter=numeric,
                    year=str(year),
                    product=product,
                )
                df["reporter_iso3"]    = iso3
                df["reporter_numeric"] = numeric
                df["year"]             = year
                results.append(df)
            except Exception as e:
                print(f"  Missing: {iso3} ({numeric}) {year} — {e}")
            done += 1
            if done % 50 == 0:
                print(f"  Progress: {done}/{total}")
            time.sleep(sleep_s)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)

    if out_csv is not None:
        combined.to_csv(out_csv, index=False)
        print(f"  Saved raw pull → {out_csv}")

    return combined


def aggregate_mfn_tariff_rate(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the raw WITS pull (distribution rows) to one
    simple-average MFN tariff rate per reporter_iso3 × year.

    Logic: filter to MFN rows, then compute weighted mean of
    Value using TotalNoOfLines as weights.
    """
    mfn = raw[raw["TariffType"].str.contains("Most Favoured", na=False)].copy()
    mfn["Value"] = pd.to_numeric(mfn["Value"], errors="coerce")
    mfn["TotalNoOfLines"] = pd.to_numeric(mfn["TotalNoOfLines"], errors="coerce")

    def weighted_mean(g):
        w = g["TotalNoOfLines"].fillna(1)
        return (g["Value"] * w).sum() / w.sum()

    agg = (
        mfn.groupby(["reporter_iso3", "year"])
        .apply(weighted_mean)
        .reset_index()
        .rename(columns={0: "mfn_tariff_rate_avg"})
    )
    return agg


# ── 4.5. Country Covariates (World Bank · GAVI Eligibility) ───────────────────

# GAVI GNI per capita eligibility thresholds (historical, Atlas method, USD)
# Source: GAVI Eligibility and Transition Policy documents
GAVI_GNI_THRESHOLDS: list[tuple[int, int, float]] = [
    (2001, 2010, 1_000.0),
    (2011, 2015, 1_520.0),
    (2016, 2099, 1_580.0),   # frozen through COVID and beyond
]

# World Bank WDI indicator codes → column names
# Aligned to WUENIC coverage years (1980–present)
WB_INDICATORS: dict[str, str] = {
    "NY.GDP.PCAP.CD":    "gdp_per_capita_usd",    # GDP per capita, current USD
    "SH.XPD.CHEX.GD.ZS": "health_exp_pct_gdp",   # Current health expenditure (% of GDP)
    "SP.POP.TOTL":        "population_total",      # Total population
    "NY.GNP.PCAP.CD":     "gni_per_capita_usd",   # GNI per capita, Atlas — used to derive GAVI eligibility
}


def _gavi_threshold(year: int) -> float | None:
    """Return the GAVI GNI eligibility threshold for a given year."""
    for start, end, threshold in GAVI_GNI_THRESHOLDS:
        if start <= year <= end:
            return threshold
    return None


def load_world_bank_covariates(
    years: range = range(1980, 2025),   # matched to WUENIC immunization coverage range
    cache_parquet: Path | None = None,
) -> pd.DataFrame:
    """
    Pull GDP per capita, health expenditure, total population, and GNI per capita
    for all countries from the World Bank Development Indicators API.

    Year range defaults to 1980–2024 to align with WUENIC coverage timestamps.
    Merge key: country_iso3 × year (joins directly onto the immunization panel).

    Derives gavi_eligible (0/1) using GAVI's historical GNI per capita thresholds:
        2001–2010 : GNI < $1,000  → eligible
        2011–2015 : GNI < $1,520  → eligible
        2016+     : GNI < $1,580  → eligible (frozen through COVID)
    Years before 2001 get NaN (GAVI eligibility not applicable).

    Results are cached to parquet on first run (~30–60s API call).
    Requires: pip install wbdata country_converter
    """
    if cache_parquet and Path(cache_parquet).exists():
        print(f"  Loading WB covariates from cache: {cache_parquet}")
        return pd.read_parquet(cache_parquet)

    try:
        import wbdata
    except ImportError:
        raise ImportError("Run: pip install wbdata")
    try:
        import country_converter as coco
    except ImportError:
        raise ImportError("Run: pip install country_converter")
    import datetime

    date_range = (
        datetime.datetime(min(years), 1, 1),
        datetime.datetime(max(years), 12, 31),
    )

    print("  Pulling World Bank indicators (this may take ~30–60s)...")
    raw = wbdata.get_dataframe(WB_INDICATORS, country="all", date=date_range)
    raw = raw.reset_index()
    raw.columns.name = None

    # Normalise column names across wbdata versions
    col_map = {}
    for c in raw.columns:
        if c.lower() in ("country", "economy"):
            col_map[c] = "country_name"
        elif c.lower() in ("date", "year"):
            col_map[c] = "year"
    raw = raw.rename(columns=col_map)
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")

    # Convert country names → ISO3; drops World Bank regional/income aggregates
    iso3_values = coco.convert(raw["country_name"].tolist(), to="ISO3", not_found=None)
    # coco.convert can return lists for ambiguous matches; flatten to scalar or None
    def _to_scalar(v):
        if isinstance(v, list):
            return v[0] if len(v) >= 1 else None  # pick first for ambiguous; None for empty
        return v
    raw["country_iso3"] = [_to_scalar(v) for v in iso3_values]
    raw = raw[raw["country_iso3"].notna() & (raw["country_iso3"] != "not found")].copy()

    # Derive GAVI eligibility from GNI per capita threshold
    def _is_gavi_eligible(row) -> float:
        threshold = _gavi_threshold(int(row["year"]))
        if threshold is None or pd.isna(row["gni_per_capita_usd"]):
            return float("nan")
        return float(int(row["gni_per_capita_usd"] < threshold))

    raw["gavi_eligible"] = raw.apply(_is_gavi_eligible, axis=1)

    result = raw[[
        "country_iso3", "year",
        "gdp_per_capita_usd", "health_exp_pct_gdp",
        "population_total", "gni_per_capita_usd", "gavi_eligible",
    ]].copy()

    if cache_parquet:
        result.to_parquet(cache_parquet, index=False)
        print(f"  Saved WB covariates → {cache_parquet}")

    return result


# ── 5. Merge Pipeline ─────────────────────────────────────────────────────────

def build_analysis_dataset(
    immunization : pd.DataFrame,
    tariffs      : pd.DataFrame,
    covariates   : pd.DataFrame | None = None,   # from load_world_bank_covariates()
) -> pd.DataFrame:
    """
    Merge immunization coverage with pharma tariff data and World Bank covariates.

    Final structure:
        country_iso3 × year × antigen_family
            - immunization_coverage  (outcome)
            - pharma_tariff_rate     (treatment: mean prf across HS-30 lines per country)
            - reporter_flag          (1 = country is a tariff reporter in the dataset)
            - gdp_per_capita_usd     (covariate, if covariates provided)
            - health_exp_pct_gdp     (covariate, if covariates provided)
            - population_total       (covariate, if covariates provided)
            - gni_per_capita_usd     (covariate, if covariates provided)
            - gavi_eligible          (0/1, derived from GNI threshold — enables non_gavi_sample())
    """
    # Step 1 — aggregate tariff to one rate per reporter country
    tariff_agg = (
        tariffs
        .groupby("reporteriso3")["prf"]
        .mean()
        .reset_index()
        .rename(columns={"reporteriso3": "country_iso3", "prf": "pharma_tariff_rate"})
    )

    # Step 2 — left-join onto immunization (country_iso3 is the shared key)
    merged = immunization.merge(tariff_agg, on="country_iso3", how="left")

    # Step 3 — flag reporter countries (have direct tariff data) vs rest
    reporter_set = set(tariffs["reporteriso3"].dropna().unique())
    merged["reporter_flag"] = merged["country_iso3"].isin(reporter_set).astype(int)

    # Step 4 — merge World Bank covariates + GAVI eligibility (country_iso3 × year)
    if covariates is not None:
        merged = merged.merge(covariates, on=["country_iso3", "year"], how="left")

    return merged


# ── 6. Sample Split Helpers ───────────────────────────────────────────────────

def phase1_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Reporter countries only — direct import tariff mechanism."""
    return df[df["reporter_flag"] == 1].copy()


def phase2_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Partner-only countries — export tariff / indirect mechanism."""
    return df[df["reporter_flag"] == 0].copy()


def non_gavi_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict to non-GAVI countries only.

    Design rationale: GAVI-eligible countries receive heavily subsidised
    vaccines through pooled procurement, breaking the tariff → price →
    coverage causal chain we aim to identify. Keeping only non-GAVI
    countries lets us estimate the mechanism in the unsubsidised market
    without needing to control for GAVI status as a covariate.
    """
    return df[df["gavi_eligible"] == 0].copy()


# ── 7. Quick Diagnostic ───────────────────────────────────────────────────────

def dataset_overview(df: pd.DataFrame) -> None:
    print(f"Rows          : {len(df):,}")
    print(f"Countries     : {df['country_iso3'].nunique()}")
    print(f"Years         : {sorted(df['year'].unique())}")
    print(f"Antigen fams  : {sorted(df['antigen_family'].dropna().unique())}")
    print(f"Reporter flag : {df['reporter_flag'].value_counts().to_dict()}")
    if "gavi_eligible" in df.columns:
        print(f"GAVI eligible : {df['gavi_eligible'].value_counts().to_dict()}")
    print(f"\nMissingness:\n{df.isna().mean().round(3)}")


# ── Entrypoint (original pipeline — commented out, superseded by pivot below)
#
# if __name__ == "__main__":
#     print("Loading data...")
#     immunization = load_immunization()
#     tariffs      = load_pharma_tariffs()
#
#     reporters    = get_reporter_countries(tariffs)
#     partner_only = get_partner_only_countries(tariffs)
#     print(f"Reporter countries    : {len(reporters)}")
#     print(f"Partner-only countries: {len(partner_only)}")
#
#     wb_cache   = ROOT / "wb_covariates.parquet"
#     covariates = load_world_bank_covariates(cache_parquet=wb_cache)
#
#     analysis_df = build_analysis_dataset(immunization, tariffs, covariates=covariates)
#     dataset_overview(analysis_df)
#
#     non_gavi_df = non_gavi_sample(analysis_df)
#     dataset_overview(non_gavi_df)
#
#     out_full     = ROOT / "analysis_dataset.csv"
#     out_non_gavi = ROOT / "analysis_dataset_non_gavi.csv"
#     analysis_df.to_csv(out_full,     index=False)
#     non_gavi_df.to_csv(out_non_gavi, index=False)


# ════════════════════════════════════════════════════════════════════════════
# PIVOT — Treatment Redesign
# Treatment  : pharma_tariff_rate x oop_health_exp_pct (interaction)
# Rationale  : Tariffs only transmit to coverage where health spending is
#              privately borne (high OOP). The interaction captures the
#              conditional effect: tariff bite is proportional to OOP share.
# New data   : Out-of-pocket health expenditure (% of current health exp.)
#              World Bank WDI indicator SH.XPD.OOPC.CH.ZS
# ════════════════════════════════════════════════════════════════════════════

def load_oop_expenditure(
    years: range = range(1990, 2025),
    cache_parquet: Path | None = None,
) -> pd.DataFrame:
    """
    Pull out-of-pocket health expenditure (% of current health expenditure)
    for all countries from the World Bank WDI API.

    Indicator : SH.XPD.OOPC.CH.ZS
    Granularity: country_iso3 x year (annual)
    Coverage  : ~180 countries, 2000-2022 (WB data availability)

    Returns DataFrame with columns:
        country_iso3, year, oop_health_exp_pct
    """
    if cache_parquet and Path(cache_parquet).exists():
        print(f"  Loading OOP data from cache: {cache_parquet}")
        return pd.read_parquet(cache_parquet)

    try:
        import wbdata
    except ImportError:
        raise ImportError("Run: pip install wbdata")
    try:
        import country_converter as coco
    except ImportError:
        raise ImportError("Run: pip install country_converter")
    import datetime

    date_range = (
        datetime.datetime(min(years), 1, 1),
        datetime.datetime(max(years), 12, 31),
    )

    print("  Pulling WB OOP expenditure data...")
    raw = wbdata.get_dataframe(
        {"SH.XPD.OOPC.CH.ZS": "oop_health_exp_pct"},
        country="all",
        date=date_range,
    )
    raw = raw.reset_index()
    raw.columns.name = None

    col_map = {}
    for c in raw.columns:
        if c.lower() in ("country", "economy"):
            col_map[c] = "country_name"
        elif c.lower() in ("date", "year"):
            col_map[c] = "year"
    raw = raw.rename(columns=col_map)
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")

    iso3_values = coco.convert(raw["country_name"].tolist(), to="ISO3", not_found=None)
    def _to_scalar(v):
        if isinstance(v, list):
            return v[0] if len(v) >= 1 else None
        return v
    raw["country_iso3"] = [_to_scalar(v) for v in iso3_values]
    raw = raw[raw["country_iso3"].notna() & (raw["country_iso3"] != "not found")].copy()

    result = raw[["country_iso3", "year", "oop_health_exp_pct"]].dropna(
        subset=["oop_health_exp_pct"]
    ).copy()

    if cache_parquet:
        result.to_parquet(cache_parquet, index=False)
        print(f"  Saved OOP data -> {cache_parquet}")

    print(f"  OOP data: {result['country_iso3'].nunique()} countries, "
          f"{result['year'].min()}-{result['year'].max()}")
    return result


def load_non_epi_coverage(
    wuenic_path: Path = IMMUNIZ,
    year_range: tuple[int, int] = (1980, 2023),
) -> pd.DataFrame:
    """
    Load non-EPI vaccine coverage from the WUENIC Excel file.

    Sheets extracted : PCV3 (pneumococcal), ROTAC (rotavirus), HIB3 (Hib)
    These are the vaccines most likely to have market-driven price sensitivity
    in non-GAVI middle-income countries.

    Returns long-format DataFrame with columns:
        unicef_region, country_iso3, country, antigen_family, year,
        immunization_coverage
    Year range restricted to year_range (default 1980-2023).
    """
    NON_EPI_SHEETS = {
        "PCV3"  : "PCV",
        "ROTAC" : "Rotavirus",
        "HIB3"  : "Hib",
    }

    frames = []
    xl = pd.ExcelFile(wuenic_path)

    for sheet, antigen in NON_EPI_SHEETS.items():
        if sheet not in xl.sheet_names:
            print(f"  [warn] Sheet {sheet} not found in WUENIC file — skipping")
            continue
        df = xl.parse(sheet)
        df = df.drop(columns=["0"], errors="ignore")

        year_cols = [
            c for c in df.columns
            if str(c).isdigit()
            and year_range[0] <= int(str(c)) <= year_range[1]
        ]
        df = df.melt(
            id_vars=["unicef_region", "iso3", "country", "vaccine"],
            value_vars=year_cols,
            var_name="year",
            value_name="immunization_coverage",
        )
        df["year"]           = df["year"].astype(int)
        df["antigen_family"] = antigen
        df = df.rename(columns={"iso3": "country_iso3"})
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    result = result.dropna(subset=["immunization_coverage"])
    print(f"  Non-EPI coverage: {result['country_iso3'].nunique()} countries, "
          f"{result['year'].min()}-{result['year'].max()}, "
          f"antigens: {sorted(result['antigen_family'].unique())}")
    return result


def build_pivot_dataset(
    non_epi_df  : pd.DataFrame,   # from load_non_epi_coverage()
    covariates  : pd.DataFrame,   # from load_world_bank_covariates()
    tariffs     : pd.DataFrame,   # from load_pharma_tariffs()
    oop_df      : pd.DataFrame,   # from load_oop_expenditure()
    year_range  : tuple[int, int] = (1980, 2023),
) -> pd.DataFrame:
    """
    Build the pivot analysis dataset:

    Base     : WUENIC non-EPI coverage (PCV, Rotavirus, Hib)
               country_iso3 x year x antigen_family
    Step 1   : Merge WB covariates (GDP, health exp, population, GNI,
               gavi_eligible) on country_iso3 x year
    Step 2   : Filter to non-GAVI countries (gavi_eligible == 0)
    Step 3   : Merge pharma tariff rate on country_iso3 (static, no year dim)
               and add reporter_flag
    Step 4   : Filter to reporter countries only (reporter_flag == 1)
    Step 5   : Merge OOP expenditure on country_iso3 x year
    Step 6   : Construct interaction treatment: tariff_x_oop
    Step 7   : Restrict years to year_range

    Final columns:
        unicef_region, country_iso3, country, antigen_family, year,
        immunization_coverage, gdp_per_capita_usd, health_exp_pct_gdp,
        population_total, gni_per_capita_usd, gavi_eligible,
        pharma_tariff_rate, reporter_flag, oop_health_exp_pct, tariff_x_oop
    """
    df = non_epi_df.copy()

    # Step 1 — merge WB covariates
    cov_cols = ["country_iso3", "year", "gdp_per_capita_usd",
                "health_exp_pct_gdp", "population_total",
                "gni_per_capita_usd", "gavi_eligible"]
    df = df.merge(
        covariates[[c for c in cov_cols if c in covariates.columns]],
        on=["country_iso3", "year"], how="left",
    )

    # Step 2 — filter non-GAVI
    if "gavi_eligible" in df.columns:
        df = df[df["gavi_eligible"] == 0].copy()
        print(f"  After non-GAVI filter : {df['country_iso3'].nunique()} countries, "
              f"{len(df):,} rows")

    # Step 3 — merge tariff rate (country-level, aggregate mean across HS-30 lines)
    tariff_agg = (
        tariffs
        .groupby("reporteriso3")["prf"]
        .mean()
        .reset_index()
        .rename(columns={"reporteriso3": "country_iso3", "prf": "pharma_tariff_rate"})
    )
    reporter_set = set(tariffs["reporteriso3"].dropna().unique())
    df = df.merge(tariff_agg, on="country_iso3", how="left")
    df["reporter_flag"] = df["country_iso3"].isin(reporter_set).astype(int)

    # Step 4 — filter to reporter countries
    df = df[df["reporter_flag"] == 1].copy()
    print(f"  After reporter filter : {df['country_iso3'].nunique()} countries, "
          f"{len(df):,} rows")

    # Step 5 — merge OOP
    df = df.merge(oop_df, on=["country_iso3", "year"], how="left")

    # Step 6 — interaction treatment
    df["tariff_x_oop"] = df["pharma_tariff_rate"] * df["oop_health_exp_pct"] / 100

    # Step 7 — restrict year range
    df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])].copy()
    print(f"  Final pivot dataset   : {df['country_iso3'].nunique()} countries, "
          f"{len(df):,} rows, {df['year'].min()}-{df['year'].max()}")

    return df


# ── Entrypoint (pivot pipeline) ───────────────────────────────────────────────

if __name__ == "__main__":
    print("Pivot pipeline starting...\n")

    print("Step 1 — loading non-EPI WUENIC coverage (PCV, Rotavirus, Hib)...")
    non_epi_df = load_non_epi_coverage()

    print("\nStep 2 — loading WB covariates (from cache)...")
    wb_cache   = ROOT / "wb_covariates.parquet"
    covariates = load_world_bank_covariates(cache_parquet=wb_cache)

    print("\nStep 3 — loading pharma tariffs...")
    tariffs = load_pharma_tariffs()

    print("\nStep 4 — loading OOP expenditure (from cache)...")
    oop_cache = ROOT / "oop_expenditure.parquet"
    oop_df    = load_oop_expenditure(cache_parquet=oop_cache)

    print("\nStep 5 — building pivot dataset...")
    pivot_df = build_pivot_dataset(non_epi_df, covariates, tariffs, oop_df)

    out_pivot = ROOT / "pivot_dataset.csv"
    pivot_df.to_csv(out_pivot, index=False)
    print(f"\nSaved -> {out_pivot}")
    print(f"Shape      : {pivot_df.shape}")
    print(f"Columns    : {pivot_df.columns.tolist()}")
    print(f"\nMissingness:\n{pivot_df.isna().mean().round(3)}")
    print(f"\nAntigen breakdown:\n{pivot_df['antigen_family'].value_counts().to_string()}")

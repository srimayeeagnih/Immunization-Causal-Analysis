# Data Strategy — Vaccine Causal Project

## Research Question
Does the pharmaceutical tariff rate a country imposes affect immunization coverage, mediated through vaccine prices?

**Causal Chain:**
```
Pharma Tariff (country, static) → Vaccine Price (regional, mediator) → Immunization Coverage (country × year × vaccine type)
```

---

## Data Assets

| Dataset | File | Granularity | Time variation |
|---|---|---|---|
| Immunization coverage | `wuenic2024rev_web-update.xlsx` (sheet: `region_global`) | Country × year × vaccine | Yes |
| Vaccine prices | `Data/who-mi4a-dataset-final-september-2025.xlsx` | Region × year × vaccine type | Yes |
| Tariff rates | `1.-preferential-tariffs.csv (1)/1. Preferential Tariffs.csv` | Country (static) | No (single snapshot) |
| Pharma tariffs | `Data/Chemicals_Allied_Industries.csv` | Country (static) | No |
| Trade agreements | `Data/pta-agreements_1.xls` | Country-pair | Varies |
| Processed regional | `region_vaccine_df.csv` | Region × vaccine | — |

---

## Key Data Structure Facts

### Tariff Data (reporterISO3 / partnerISO3)
- **reporterISO3** = country imposing the tariff (importer/tariff-setter)
- **partnerISO3** = country of origin of imports (exporter)
- More partner countries than reporter countries → **data reporting gap**, not economic asymmetry
  - Partner-only countries tend to be poorer/less institutionally capable
  - They appear as targets of others' tariffs but never submitted their own schedules

### Vaccine Type Mismatch (Critical Finding)
- WUENIC uses **dose-level program codes**: `MCV1`, `MCV2`, `DTP3`, `HEPB3`, etc. (16 unique)
- MI4A uses **product/formulation names**: `MMR`, `Measles`, `DTwP-HepB-Hib`, `PCV13`, etc. (100 unique)
- **Exact overlap: only 1 (BCG)**
- A **manual crosswalk table** is required to join the two datasets
- Example mappings needed:
  - `MCV1`, `MCV2` → `Measles`, `MMR`, `MR`, `MMRV`
  - `DTP3` → `DTwP`, `DTaP`, `DTwP-HepB-Hib`, etc.
  - `HEPB3`, `HEPBB` → `HepB (ped.)`
  - `HIB3` → `Hib`, various combination vaccines
  - `PCV3` → `PCV10`, `PCV13`
  - `POL3` → `tOPV`, `bOPV`, `IPV`
  - `ROTAC` → `RV1`, `RV5`
  - `YFV` → `YF`

---

## Research Design

### Phase 1 — Primary Analysis (Reporter Countries)
- Countries that submitted tariff data (appear as `reporterISO3`)
- **Causal chain is intact**: import tariff directly affects vaccine import costs
- Control for GAVI eligibility (GAVI breaks the tariff→price chain)
- Identification: **cross-sectional** (tariff is static — no within-country time variation)
- Cannot use country fixed effects (would absorb static tariff)

### Phase 2 — Secondary Analysis (Partner-Only Countries)
- Countries appearing only as `partnerISO3`, never as `reporterISO3`
- Different mechanism: tariffs on their *exports* → economic/fiscal pressure → health system capacity
- More distal, more confounded — frame explicitly as a separate pathway
- Likely poorer countries — important for policy generalizability

---

## GAVI Control (Critical)
- GAVI eligibility **breaks the tariff→price→immunization chain** for eligible countries
  - GAVI negotiates heavily subsidized prices, decoupled from market tariffs
- GAVI support varies by **country AND vaccine type** (not just country)
- Control should ideally be at **country × vaccine type × year** level
- Treat GAVI status as a **moderator**, not just a confounder
  - Consider: `tariff × GAVI` interaction term, or stratify by GAVI status

---

## Identification Strategy

| Dimension | Can exploit for tariff ID? | Notes |
|---|---|---|
| Cross-country variation | Yes (primary) | Main source of tariff identification |
| Within-country over time | No (tariff is static) | Blocked unless TRAINS annual data added |
| Within vaccine type | No | All vaccines share same country tariff |
| Within region | Limited | May absorb tariff if countries cluster regionally |

**Recommended FE structure:**
- Year FEs — absorb global immunization trends
- Vaccine type FEs — absorb vaccine-specific baseline coverage
- Region FEs — use carefully (may absorb tariff variation)
- **No country FEs** with static tariff

---

## Data Pipeline Steps

### Step 1 — Extract Reporter Countries
```python
import pandas as pd
tariffs = pd.read_parquet("1.-preferential-tariffs.csv (1)/1. Preferential Tariffs.parquet")
reporter_countries = tariffs['reporterISO3'].dropna().unique().tolist()
```

### Step 2 — Pull Annual Tariff Time Series (WITS/TRAINS)
```python
import world_trade_data as wits
import time

results = []
for country in reporter_countries:
    for year in range(2000, 2024):
        try:
            df = wits.get_tariff_reported(reporter=country, year=year, product='30')  # HS chapter 30 = pharma
            df['reporterISO3'] = country
            df['year'] = year
            results.append(df)
            time.sleep(0.5)  # avoid rate limiting
        except Exception as e:
            print(f"Missing: {country} {year} — {e}")

annual_tariffs = pd.concat(results)
```

### Step 3 — Build Vaccine Type Crosswalk
Manually map WUENIC codes → MI4A vaccine type families:

| WUENIC code | MI4A antigen family |
|---|---|
| BCG | BCG |
| MCV1, MCV2 | Measles / MMR / MR / MMRV |
| DTP1, DTP3 | DTwP / DTaP / DTP combinations |
| HEPB3, HEPBB | HepB (ped.) |
| HIB3 | Hib / combination vaccines |
| PCV3 | PCV10 / PCV13 |
| POL3 | tOPV / bOPV / IPV |
| ROTAC | RV1 / RV5 |
| YFV | YF |
| IPV1, IPV2 | IPV |
| RCV1 | Rubella / MR / MMR |
| MENGA | MenA conjugate |

### Step 4 — Merge Strategy
```
immunization (country × year × vaccine)
    ↓ join on [country → region mapping]
vaccine_price (region × year × vaccine_family)  ← via crosswalk
    ↓ join on [country]
tariff (country × year)  ← from TRAINS pull
    ↓ join on [country × year]
GAVI (country × vaccine_type × year)  ← external lookup
```

### Step 5 — Add GAVI Controls
- Source: GAVI eligibility list (historical, by country and year)
- Merge at country × vaccine type × year level
- Create binary `gavi_eligible` and optionally `gavi_supported` (for specific vaccines)

---

## Final Analysis-Ready Dataset Structure

| Column | Level | Source |
|---|---|---|
| country_iso3 | Country | WUENIC |
| year | Year | WUENIC |
| vaccine_family | Vaccine type (crosswalked) | Crosswalk |
| immunization_coverage | Outcome | WUENIC |
| vaccine_price_regional | Mediator | MI4A |
| pharma_tariff_rate | Treatment | TRAINS/WITS |
| gavi_eligible | Control | GAVI list |
| gavi_supported_vaccine | Control | GAVI list |
| region | Control/FE | Lookup |
| reporter_flag | Sample split | Tariff data |

---

## Key Limitations (for paper)
1. Pharma tariff ≠ vaccine-specific tariff → assumes broad pharma tariffs proxy vaccine tariff burden
2. Vaccine prices at regional level only → mediation pathway is coarse, not country-specific
3. Static tariff data → purely cross-sectional identification; panel dimension used for controls only
4. Non-random missingness: partner-only countries are systematically poorer (potential selection bias)
5. Vaccine type crosswalk introduces measurement error (many-to-many mapping)
6. GAVI pricing may not perfectly correspond to GAVI eligibility dates

---

## Causal Soundness — What Would Strengthen This (Future Work)
- Time-varying tariff data from TRAINS (annual since 1988)
- Instrument for tariffs: WTO accession timing, trade agreement phase-ins, GSP changes
- Country-level vaccine prices (currently only regional)
- Controls: GDP per capita, govt health expenditure, health system capacity, disease burden

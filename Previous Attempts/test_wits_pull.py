"""
test_wits_pull.py
-----------------
Standalone smoke-test for the WITS/TRAINS annual tariff API.

Country codes  : ISO numeric (076=Brazil, 276=Germany, 840=USA) ✓
Product codes  : probing valid formats — "30" (HS chapter) is invalid.

WITS vaccine-relevant HS codes:
  3002   = Human/animal blood, vaccines, toxins (HS4 chapter heading)
  300220 = Vaccines for human medicine (HS6)

Usage:
    pip install world_trade_data
    python test_wits_pull.py
"""

import time
import pandas as pd

SLEEP_S = 0.5

# ISO numeric codes confirmed working from Step 1
TEST_COUNTRIES = ["076", "276", "840"]   # Brazil, Germany, USA
TEST_YEARS     = [2018, 2021]

# Candidate product codes to probe — first one to succeed wins
PRODUCT_CANDIDATES = [
    ("ALLPRODUCTS",  "all-product aggregate keyword"),
    ("ALL",          "all-product alternate keyword"),
    ("300220",       "HS6: vaccines for human medicine"),
    ("300200",       "HS6: blood/vaccine heading padded"),
    ("3002",         "HS4: blood products & vaccines"),
    ("300210",       "HS6: antisera"),
    ("300215",       "HS6: immunological products"),
]


def probe_product_codes(wits) -> str | None:
    """Try each candidate product code against one country/year. Return first that works."""
    print("── Probing valid product code formats ──")
    for code, label in PRODUCT_CANDIDATES:
        try:
            df = wits.get_tariff_reported(
                reporter="840",      # USA
                year="2018",
                product=code,
            )
            print(f"  OK   product='{code}'  ({label}) — {len(df):,} rows")
            print(f"       cols: {df.columns.tolist()}")
            print(f"       sample:\n{df.head(3).to_string()}\n")
            return code
        except Exception as e:
            short = str(e).split("url:")[-1].strip() if "url:" in str(e) else str(e)
            print(f"  FAIL product='{code}'  ({label}) — {short}")
        time.sleep(SLEEP_S)

    # Also try the library's own product listing
    for fn_name in ["get_products", "get_product_list", "get_nomenclature"]:
        fn = getattr(wits, fn_name, None)
        if fn is not None:
            try:
                products = fn()
                print(f"\nwits.{fn_name}() returned:\n{products.head(10).to_string()}")
            except Exception as e:
                print(f"  wits.{fn_name}() failed: {e}")

    return None


def test_pull(product: str) -> pd.DataFrame:
    try:
        import world_trade_data as wits
    except ImportError:
        raise ImportError("Run: pip install world_trade_data")

    print(f"\n── Pulling tariffs: product={product}, countries={TEST_COUNTRIES}, years={TEST_YEARS} ──\n")

    results = []
    for country in TEST_COUNTRIES:
        for year in TEST_YEARS:
            try:
                df = wits.get_tariff_reported(
                    reporter=country,
                    year=str(year),
                    product=product,
                )
                df["reporter_code"] = country
                df["pull_year"]     = year
                results.append(df)
                print(f"  OK   {country} {year}  — {len(df):,} rows")
            except Exception as e:
                print(f"  FAIL {country} {year}  — {e}")
            time.sleep(SLEEP_S)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    print(f"\nTotal rows : {len(combined):,}")
    print(f"Columns    : {combined.columns.tolist()}")
    print(f"\nMissingness:\n{combined.isna().mean().round(3)}")
    print(f"\nFirst 5 rows:\n{combined.head().to_string()}")

    rate_cols = [c for c in combined.columns if any(
        kw in c.lower() for kw in ["rate", "mfn", "prf", "applied", "bound", "tariff", "duty"]
    )]
    print(f"\nTariff-rate columns: {rate_cols}")
    return combined


if __name__ == "__main__":
    import world_trade_data as wits

    valid_product = probe_product_codes(wits)
    OUT_DIR = r"C:\Users\srima\OneDrive\Desktop\INSY674\Individual Project 1"

    if valid_product:
        df = test_pull(valid_product)
        if not df.empty:
            out = f"{OUT_DIR}\\wits_test_sample.csv"
            df.to_csv(out, index=False)
            print(f"\nSaved → {out}")
    else:
        print("\nNo valid product code found. Check wits library docs or try get_products().")
        fns = [a for a in dir(wits) if not a.startswith("_")]
        print(f"Available wits functions: {fns}")

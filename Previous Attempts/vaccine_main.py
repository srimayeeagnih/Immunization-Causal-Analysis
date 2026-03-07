
#DATA MERGING

import os
import pandas as pd
import country_converter as coco
from itertools import permutations
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from econml.dml import CausalForestDML
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

df = pd.read_csv(os.path.join(DATA_DIR, 'Chemicals_Allied_Industries.csv'))

df = df.rename(columns={'mfn': 'standard_rate', 'prf': 'preferential_rate'})

pharma_tariffs = len(df[df['hs2']==30])
print(pharma_tariffs)

pharma_trade_df = df[df['hs2']==30]

wto_x_ac = pd.read_excel(r'C:\Users\srima\Downloads\pta-agreements_1.xls', sheet_name='WTO-X AC')
wto_x_le = pd.read_excel(r'C:\Users\srima\Downloads\pta-agreements_1.xls', sheet_name='WTO-X LE')

vaccine_df = pd.read_excel(r'C:\Users\srima\OneDrive\Desktop\INSY674\Individual Project 1\who-mi4a-dataset-final-september-2025.xlsx', sheet_name='Vaccine purchase database')

# Split on em-dash, en-dash, or hyphen with optional surrounding whitespace
partners = wto_x_ac['Agreement'].str.split(r'\s*[—–-]\s*', regex=True)

# Find the max number of partners across all rows
max_partners = partners.str.len().max()

# Build Partner 1, Partner 2, ... Partner N columns
for i in range(max_partners):
    col = f'Partner {i + 1}'
    wto_x_ac[col] = partners.str[i]

# Rows with no dash (only 1 element after split) -> set all partner cols to 'N/A'
partner_cols = [f'Partner {i + 1}' for i in range(max_partners)]
no_dash = partners.str.len() == 1
wto_x_ac.loc[no_dash, partner_cols] = 'N/A'

# Convert partner names to ISO3 codes ('N/A' for trade blocs like ASEAN, Pacific Partnership, etc.)
for col in partner_cols:
    converted = coco.convert(wto_x_ac[col].fillna('N/A').tolist(), to='ISO3', not_found='N/A')
    # If coco returns a list for ambiguous names, take the first match
    wto_x_ac[f'{col} ISO3'] = [v[0] if isinstance(v, list) else v for v in converted]

# Remove duplicate columns (can happen if cells above were re-run)
wto_x_ac = wto_x_ac.loc[:, ~wto_x_ac.columns.duplicated(keep='last')]

# Build all ordered partner pairs per agreement (so both directions are covered)
iso3_cols = [f'Partner {i+1} ISO3' for i in range(max_partners)]
pair_rows = []
for idx, row in wto_x_ac.iterrows():
    codes = [row[c] for c in iso3_cols if isinstance(row[c], str) and row[c] != 'N/A']
    for a, b in permutations(codes, 2):
        pair_rows.append({**row.to_dict(), 'reporter_iso3': a, 'partner_iso3': b})

wto_ac_pairs = pd.DataFrame(pair_rows)

# Do same with WTO-X LE dataset
# Split Agreement column into partner columns (same logic as wto_x_ac)
le_partners = wto_x_le['Agreement'].str.split(r'\s*[—–-]\s*', regex=True)
for i in range(max_partners):
    col = f'Partner {i + 1}'
    wto_x_le[col] = le_partners.str[i]

le_no_dash = le_partners.str.len() == 1
wto_x_le.loc[le_no_dash, partner_cols] = 'N/A'

# Convert partner names to ISO3 codes
for col in partner_cols:
    converted = coco.convert(wto_x_le[col].fillna('N/A').tolist(), to='ISO3', not_found='N/A')
    # If coco returns a list for ambiguous names, take the first match
    wto_x_le[f'{col} ISO3'] = [v[0] if isinstance(v, list) else v for v in converted]

# Remove duplicate columns (can happen if cells above were re-run)
wto_x_le = wto_x_le.loc[:, ~wto_x_le.columns.duplicated(keep='last')]

le_pair_rows = []
for idx, row in wto_x_le.iterrows():
    codes = [row[c] for c in iso3_cols if isinstance(row[c], str) and row[c] != 'N/A']
    for a, b in permutations(codes, 2):
        le_pair_rows.append({**row.to_dict(), 'reporter_iso3': a, 'partner_iso3': b})

wto_le_pairs = pd.DataFrame(le_pair_rows)

# Merge WTO-X AC and WTO-X LE pair DataFrames on reporter/partner ISO3
wto_merged = wto_ac_pairs.merge(
    wto_le_pairs,
    on=['reporter_iso3', 'partner_iso3'],
    how='outer',
    suffixes=('_ac', '_le')
)

# Merge WTO combined data with pharma trade data
final_merged = pharma_trade_df.merge(
    wto_merged,
    left_on=['reporteriso3', 'partneriso3'],
    right_on=['reporter_iso3', 'partner_iso3'],
    how='outer'
)

print(final_merged.head())

final_merged['partneriso3'].value_counts()

#DATA CLEANING

final_merged['Terrorism_ac'].value_counts()
final_merged['Terrorism_le'].value_counts()


final_merged['Health_ac'].value_counts()
final_merged['Health_le'].value_counts()

#Since the WTO-X_LE dataset has more provisions and also a more granular breakdown between mentioned agreements
# vs mentioned legally enforceable agreements, we will use the columns derived from the WTO-X_LE dataset

# Drop all columns with the _ac suffix
final_merged = final_merged.drop(columns=[c for c in final_merged.columns if c.endswith('_ac')])

final_merged.columns

#Keeping covariates related to healthcare related trade agreements--> illicit drugs, R&D, IPR, 
# Consumer Protection, Innovation, EnvironmentalLaws, SocialMatters

cols_to_exclude = ['Agreement_le', 'Date_le', 'year_le', 'Type_le',
                   'Partner 1_le', 'Partner 2_le', 'Partner 3_le',
                   'Partner 1 ISO3_le', 'Partner 2 ISO3_le', 'Partner 3 ISO3_le']
cols_to_exclude_2 = ['EnvironmentalLaws_le','IPR_le','ConsumerProtection_le','InnovationPolicies_le','Health_le','IllicitDrugs_le','SocialMatters_le']
final_merged = final_merged.drop(columns=[c for c in final_merged.columns if c.endswith('_le') and c not in cols_to_exclude+cols_to_exclude_2])

final_merged.columns

final_merged[['reporteriso3','partneriso3','reporter_iso3','partner_iso3','Partner 1_le', 'Partner 2_le',
       'Partner 3_le', 'Partner 1 ISO3_le', 'Partner 2 ISO3_le',
       'Partner 3 ISO3_le']]

final_merged['Partner 1 ISO3_le'].value_counts()

# Check how many rows have matching Partner 1 ISO3_le and reporteriso3
match_count = (final_merged['Partner 1 ISO3_le'] == final_merged['reporteriso3']).sum()
total_count = final_merged[['Partner 1 ISO3_le', 'reporteriso3']].dropna().shape[0]
print(f"Matching rows: {match_count} / {total_count} ({match_count/total_count*100:.1f}%)")

#1/3 of the columns match...maybe the provisions under the current mandate of WTO are just as important
#For the reasoning previously mentioned, we will only keep the WTO+LE sheet

wto_plus =pd.read_excel(r"C:\Users\srima\OneDrive\Desktop\INSY674\Individual Project 1\pta-agreements_1.xls", sheet_name='WTO+ LE')

partners_plus = wto_plus['Agreement'].str.split(r'\s*[—–-]\s*', regex=True)

max_partners_plus = partners_plus.str.len().max()

# Build Partner 1, Partner 2, ... Partner N columns
for i in range(max_partners_plus):
    col = f'Partner {i + 1}'
    wto_plus[col] = partners_plus.str[i]

# Rows with no dash (only 1 element after split) -> set all partner cols to 'N/A'
partner_cols = [f'Partner {i + 1}' for i in range(max_partners_plus)]
no_dash = partners_plus.str.len() == 1
wto_plus.loc[no_dash, partner_cols] = 'N/A'

for col in partner_cols:
    converted = coco.convert(wto_plus[col].fillna('N/A').tolist(), to='ISO3', not_found='N/A')
    # If coco returns a list for ambiguous names, take the first match
    wto_plus[f'{col} ISO3'] = [v[0] if isinstance(v, list) else v for v in converted]

# Remove duplicate columns (can happen if cells above were re-run)
wto_plus = wto_plus.loc[:, ~wto_plus.columns.duplicated(keep='last')]

# Build all ordered partner pairs per agreement (so both directions are covered)
iso3_cols = [f'Partner {i+1} ISO3' for i in range(max_partners_plus)]
pair_rows = []
for idx, row in wto_plus.iterrows():
    codes = [row[c] for c in iso3_cols if isinstance(row[c], str) and row[c] != 'N/A']
    for a, b in permutations(codes, 2):
        pair_rows.append({**row.to_dict(), 'reporter_iso3': a, 'partner_iso3': b})

wto_plus_pairs = pd.DataFrame(pair_rows)

wto_plus.columns

wto_plus_pairs.columns

# Merge WTO+ LE pairs with final_merged on reporter/partner ISO3
print(f"Before merge: final_merged has {final_merged.shape[1]} cols, wto_plus_pairs has {wto_plus_pairs.shape[1]} cols")
final_merged = final_merged.merge(
    wto_plus_pairs,
    on=['reporter_iso3', 'partner_iso3'],
    how='outer',
    suffixes=('', '_plus')
)
print(f"After merge: final_merged has {final_merged.shape[1]} cols, {final_merged.shape[0]} rows")
print([c for c in final_merged.columns if 'plus' in c or 'ISO3' in c])

print(final_merged.columns.tolist())

#Checking all the country row matches 

match_count = (final_merged['Partner 1 ISO3_le'] == final_merged['reporteriso3']).sum()
total_count = final_merged[['Partner 1 ISO3_le', 'reporteriso3']].dropna().shape[0]
print(f"Matching rows: {match_count} / {total_count} ({match_count/total_count*100:.1f}%)")

match2_count = (final_merged['Partner 2 ISO3_le'] == final_merged['partneriso3']).sum()
total2_count = final_merged[['Partner 2 ISO3_le', 'partneriso3']].dropna().shape[0]
print(f"Matching rows: {match2_count} / {total2_count} ({match2_count/total2_count*100:.1f}%)")

match_plus_count = (final_merged['Partner 1 ISO3'] == final_merged['reporteriso3']).sum()
total_plus_count = final_merged[['Partner 1 ISO3', 'reporteriso3']].dropna().shape[0]
print(f"Matching rows: {match_plus_count} / {total_plus_count} ({match_plus_count/total_plus_count*100:.1f}%)")

match_plus2_count = (final_merged['Partner 2 ISO3'] == final_merged['partneriso3']).sum()
total_plus2_count = final_merged[['Partner 2 ISO3', 'partneriso3']].dropna().shape[0]
print(f"Matching rows: {match_plus2_count} / {total_plus2_count} ({match_plus2_count/total_plus2_count*100:.1f}%)")

print(wto_plus_pairs.shape)
print(wto_plus_pairs.columns.tolist())

wto_plus_pairs['Partner 1 ISO3'].value_counts()


#DATA CLEANING (CONT.)

final_merged.columns

#Check for columns with duplicate rows

final_merged[['Partner 2 ISO3','Partner 2 ISO3_le']] #identical, drop one of them

final_merged = final_merged.drop(columns=['Partner 1 ISO3_le', 'Partner 2 ISO3_le', 'Partner 3 ISO3_le'])

final_merged.columns

final_merged[['Partner 1 ISO3', 'Partner 1']]
final_merged[['Partner 2 ISO3', 'Partner 2']]
final_merged[['Partner 3 ISO3', 'Partner 3']].value_counts() #Partner 3 ISO3 is all N/A, drop it

# Where Partner 3 ISO3 is 'N/A' but Partner 3 has a name (e.g., ASEAN, Central America), copy the name over
mask = (final_merged['Partner 3 ISO3'] == 'N/A') & (final_merged['Partner 3'].notna()) & (final_merged['Partner 3'] != 'N/A')
final_merged.loc[mask, 'Partner 3 ISO3'] = final_merged.loc[mask, 'Partner 3']

vaccine_df=pd.read_excel(r"C:\Users\srima\OneDrive\Desktop\INSY674\Individual Project 1\who-mi4a-dataset-final-september-2025.xlsx", sheet_name='Vaccine purchase database')

print(vaccine_df.columns.tolist())

vaccine_df['Region']

final_merged.head()

vaccine_df.head()

#Vaccine data cleaning

#EDA on the merged dataset..? 

AFRO = ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon', 'Cape Verde', 'Central African Republic', 'Chad', 'Comoros', "Côte d'Ivoire", 'Democratic Republic of the Congo', 'Equatorial Guinea', 'Eritrea', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Kenya', 'Lesotho', 'Liberia', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', "Republic of the Congo", "Rwanda", "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone", "South Africa", "Swaziland", "Togo", "Uganda", "United Republic of Tanzania", "Zambia", "Zimbabwe"]

AMRO = ['Antigua and Barbuda', 'Anguilla', 'Argentina', 'Bahamas', 'Barbados', 'Belize', 'Bermuda', 'Bolivia (Plurinational State of)', 'Brazil', 'British Virgin Islands', 'Canada', 'Cayman Islands', 'Chile', 'Colombia', 'Costa Rica', 'Cuba', 'Dominica', 'Dominican Republic', 'Ecuador', 'El Salvador', 'Grenada', 'Guatemala', 'Guyana', 'Haiti', 'Honduras', 'Jamaica', 'Mexico', 'Nicaragua', 'Panama', 'Paraguay', 'Peru', "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Suriname", "Trinidad and Tobago", "Turks and Caicos", "United States of America", "Uruguay", "Venezuela (Bolivarian Republic of)"]

EMRO = ['Afghanistan', 'Bahrain', 'Djibouti', 'Egypt', 'Iran (Islamic Republic of)', 'Iraq', 'Jordan', 'Kuwait', 'Lebanon', 'Libyan Arab Jamahiriya', 'Morocco', 'Oman', 'Pakistan', 'Palestine', 'Qatar', 'Saudi Arabia', 'Somalia', 'Sudan', 'South Sudan', 'Syrian Arab Republic', 'Tunisia', 'United Arab Emirates', 'Yemen']

EURO = ['Albania', 'Andorra', 'Armenia', 'Austria', 'Azerbaijan', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Georgia', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Israel', 'Italy', 'Kazakhstan', 'Kosovo', 'Kyrgyzstan', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 'Poland', 'Portugal','Republic of Moldova','Romania','Russian Federation','San Marino','Serbia','Slovakia','Slovenia','Spain','Sweden','Switzerland','Tajikistan','Turkey','Turkmenistan','Ukraine','United Kingdom','Uzbekistan']

SEARO = ['Bangladesh', 'Bhutan', 'Democratic People\'s Republic of Korea', 'India', 'Indonesia', 'Maldives', 'Myanmar', 'Nepal', 'Sri Lanka', 'Thailand', 'Timor-Leste']

WPRO = ['Australia', 'Brunei Darussalam', 'Cambodia', 'China', 'Cook Islands', 'Fiji', 'Japan', 'Kiribati', 'Lao People\'s Democratic Republic', 'Malaysia', 'Marshall Islands', 'Micronesia (Federated States of)', 'Mongolia', 'Nauru', 'New Caledonia', 'New Zealand', 'Niue', 'Palau', 'Papua New Guinea', 'Philippines', 'Republic of Korea', 'Samoa', 'Singapore', 'Solomon Islands', 'Taiwan', 'Tonga', 'Tuvalu', 'Vanuatu', 'Viet Nam']

# Build a country -> WHO region lookup, then convert to ISO3 -> region
region_lists = {
    'AFRO': AFRO,
    'AMRO': AMRO,
    'EMRO': EMRO,
    'EURO': EURO,
    'SEARO': SEARO,
    'WPRO': WPRO,
}

iso3_to_region = {}
for region, countries in region_lists.items():
    iso3_codes = coco.convert(countries, to='ISO3', not_found=None)
    for code in iso3_codes:
        if code is not None:
            iso3_to_region[code] = region

# Create WHO region columns for reporter and partner
final_merged['reporter_WHO_region'] = final_merged['reporteriso3'].map(iso3_to_region)
final_merged['partner_WHO_region'] = final_merged['partneriso3'].map(iso3_to_region)

# Check for unmatched
unmatched_reporters = final_merged[final_merged['reporter_WHO_region'].isna()]['reporteriso3'].unique()
unmatched_partners = final_merged[final_merged['partner_WHO_region'].isna()]['partneriso3'].unique()
if len(unmatched_reporters) > 0:
    print(f"Unmatched reporter ISO3 codes: {unmatched_reporters}")
if len(unmatched_partners) > 0:
    print(f"Unmatched partner ISO3 codes: {unmatched_partners}")

print(final_merged[['reporteriso3', 'reporter_WHO_region', 'partneriso3', 'partner_WHO_region']].head(10))

#Further inspect country codes


final_merged

unmatched_codes = list(set(unmatched_reporters).union(set(unmatched_partners)))
unmatched_names = coco.convert(unmatched_codes, to='name_short', not_found='Unknown')

print(pd.DataFrame({'ISO3': unmatched_codes, 'Country': unmatched_names}))

#Manually assigning them to a WHO region based on continent

unmatched_continents = coco.convert(unmatched_codes, to='continent', not_found='Unknown')

print(pd.DataFrame({'ISO3': unmatched_codes, 'Country': unmatched_names, 'Continent': unmatched_continents}))

# Assign WHO region based on continent
for code, continent in zip(unmatched_codes, unmatched_continents):
    if continent == 'Africa':
        iso3_to_region[code] = 'AFRO'
    elif continent == 'America':
        iso3_to_region[code] = 'AMRO'
    elif continent == 'Europe':
        iso3_to_region[code] = 'EMRO'
    elif continent in ('Asia', 'Oceania'):
        iso3_to_region[code] = 'WPRO'

# Re-map with the updated dictionary
final_merged['reporter_WHO_region'] = final_merged['reporteriso3'].map(iso3_to_region)
final_merged['partner_WHO_region'] = final_merged['partneriso3'].map(iso3_to_region)

# Verify no remaining unmatched
still_missing = final_merged[final_merged['reporter_WHO_region'].isna() | final_merged['partner_WHO_region'].isna()].shape[0]
print(f"Rows still missing a WHO region: {still_missing}")

final_merged.shape

vaccine_df.dtypes
final_merged.dtypes

#EDA on treatment variable: trade deal enactement

# Step 1: Check vaccine data year range

vaccine_df['Year'] = pd.to_numeric(vaccine_df['Year'], errors='coerce')
vaccine_df = vaccine_df.dropna(subset=['Year'])
vaccine_df['Year'] = vaccine_df['Year'].astype(int)

print(f"Vaccine Year range: {vaccine_df['Year'].min()} - {vaccine_df['Year'].max()}")


# Step 2: Check trade deal year range 
final_merged['year']
final_merged['year'] = final_merged['year'].astype('Int64')
print(f'Trade deal range: {final_merged["year"].min()} - {final_merged["year"].max()}')

#Step 3: Perform EDA on trade dataset 

#Look at the distribution of the year variables to see if there's a breaking point to cluster countries into recency in which trade deals were enacted

final_merged['year_le'].value_counts().sort_index().plot(kind='bar')
plt.title('Trade Deals by Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

#Trade deals by country

final_merged['reporter_WHO_region'].value_counts().sort_index().plot(kind='bar')
plt.title('Trade Deals by Country 1 WHO Region')
plt.xlabel('Country 1 WHO Region')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

final_merged['partner_WHO_region'].value_counts().sort_index().plot(kind='bar')
plt.title('Trade Deals by Country 2 WHO Region')
plt.xlabel('Country 2 WHO Region')
plt.ylabel('Count')
plt.tight_layout()
plt.show()   #European countries have the most number of trade deals


#Countries/Affiliations that have more than 1 trade deal

# Count how often each country appears across Partner 1/2/3
country_counts = (
    final_merged[['Partner 1','Partner 2','Partner 3']]
    .stack()
    .value_counts()
)

country_counts.max()

# Filter to only those with more than 1 deal

# Plot
country_counts.nlargest(10).plot(kind='bar', figsize=(10,5), color='steelblue')

plt.title('Countries with the Highest Number of Trade Deals')
plt.xlabel('Country')
plt.ylabel('Number of Deals')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

final_merged.columns

#Perform regional analysis of trade deal provisions to capture intra-regional dynamics

#1. Define column groups for weighting

# WTO+ LE provisions (sectoral/trade liberalization)
sectoral_cols = ['FTAIndustrial', 'FTAAgriculture']
trade_lib_cols = ['Customs', 'ExportTaxes', 'SPS', 'TBT', 'STE', 'AD',
                  'CVM', 'StateAid', 'PublicProcurement', 'TRIMs', 'GATS', 'TRIPs']

# WTO-X LE provisions (behind-the-border / regulatory depth)
regulatory_cols = ['EnvironmentalLaws_le', 'IPR_le', 'ConsumerProtection_le',
                   'InnovationPolicies_le', 'Health_le', 'IllicitDrugs_le', 'SocialMatters_le']

#  2. Inverse-frequency weights (rarer provisions → higher weight) 
# This captures how "unusual" or ambitious a provision is across all deals

import numpy as np

def inverse_freq_weights(df, cols):
    """Compute weights = 1 / proportion for each binary column (rarer = heavier)."""
    freq = df[cols].mean()                       # proportion of 1s per column
    freq = freq.replace(0, np.nan)               # avoid division by zero
    raw_weights = 1.0 / freq
    return raw_weights / raw_weights.sum()        # normalise to sum to 1

sectoral_weights   = inverse_freq_weights(final_merged, sectoral_cols)
trade_lib_weights  = inverse_freq_weights(final_merged, trade_lib_cols)
regulatory_weights = inverse_freq_weights(final_merged, regulatory_cols)

print("Sectoral weights:\n", sectoral_weights)
print("\nTrade liberalization weights:\n", trade_lib_weights)
print("\nRegulatory depth weights:\n", regulatory_weights)

#  3. Row-level weighted scores 
final_merged['sectoral_score']   = final_merged[sectoral_cols].multiply(sectoral_weights).sum(axis=1)
final_merged['trade_lib_score']  = final_merged[trade_lib_cols].multiply(trade_lib_weights).sum(axis=1)
final_merged['regulatory_score'] = final_merged[regulatory_cols].multiply(regulatory_weights).sum(axis=1)

#  4. Deal-type weight (ordinal encoding by depth of integration) 
deal_type_weights = {
    'PSA':       1,  # Partial Scope Agreement – narrowest
    'PSA & EIA': 2,  # PSA + Economic Integration Agreement
    'FTA':       3,  # Free Trade Agreement – broader than PSA
    'FTA & EIA': 4,  # FTA + Economic Integration Agreement
    'CU':        5,  # Customs Union – common external tariff
    'CU & EIA':  6,  # Customs Union + EIA – deepest integration
}
final_merged['deal_type_weight'] = final_merged['Type'].map(deal_type_weights).fillna(1)

#  5. Intra-regional binary flag 
final_merged['intra_regional'] = (
    final_merged['reporter_WHO_region'] == final_merged['partner_WHO_region']
).astype(int)

#  6. Regional aggregations (mean scores within each WHO region pair) 
region_pair_agg = (
    final_merged
    .groupby(['reporter_WHO_region', 'partner_WHO_region'])
    .agg(
        region_sectoral_mean   = ('sectoral_score', 'mean'),
        region_trade_lib_mean  = ('trade_lib_score', 'mean'),
        region_regulatory_mean = ('regulatory_score', 'mean'),
        region_deal_depth_mean = ('deal_type_weight', 'mean'),
        region_deal_count      = ('sectoral_score', 'size'),
    )
    .reset_index()
)

# Merge regional aggregations back to row level
final_merged = final_merged.merge(
    region_pair_agg,
    on=['reporter_WHO_region', 'partner_WHO_region'],
    how='left'
)

print(final_merged[['reporter_WHO_region', 'partner_WHO_region', 'intra_regional',
                     'sectoral_score', 'trade_lib_score', 'regulatory_score',
                     'deal_type_weight', 'region_sectoral_mean', 'region_trade_lib_mean',
                     'region_regulatory_mean', 'region_deal_depth_mean', 'region_deal_count']].head(10))


# --- Merge vaccine data at the WHO Region + Vaccine type level ---

vaccine_agg = (
    vaccine_df
    .groupby(['Region', 'Vaccine'])
    .agg(
        avg_price_per_dose=('Price per Dose in USD', 'mean'),
        num_manufacturers=('Manufacturer', 'nunique'),         # market competition proxy
        total_annual_volume=('Total Annual Quantity', 'sum'),   # demand/bargaining power proxy
    )
    .reset_index()
)

print(vaccine_agg.head(10))

# Merge to final_merged on reporter WHO region
final_merged = final_merged.merge(
    vaccine_agg,
    left_on='reporter_WHO_region',
    right_on='Region',
    how='left',
)
final_merged = final_merged.drop(columns=['WHO Region'], errors='ignore')

print(f"After vaccine merge: {final_merged.shape[0]} rows, {final_merged.shape[1]} cols")

final_merged.columns

#Further Data Preprocessing + EDA on feature engineered variables

eda_cols = ['sectoral_score', 'trade_lib_score', 'regulatory_score',
            'deal_type_weight', 'intra_regional', 'region_sectoral_mean',
            'region_trade_lib_mean', 'region_regulatory_mean',
            'region_deal_depth_mean', 'region_deal_count', 'avg_price_per_dose']

eda_df = final_merged[eda_cols].copy()

# --- 1. Descriptive Statistics ---
print("=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
print(eda_df.describe().round(3).to_string())
print(f"\nMissing values:\n{eda_df.isnull().sum()}")

# --- 2. Distribution plots (histograms + KDE) ---
continuous_cols = [c for c in eda_cols if c not in ['intra_regional', 'deal_type_weight']]

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(continuous_cols):
    ax = axes[i]
    eda_df[col].dropna().hist(bins=30, ax=ax, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency')
fig.suptitle('Distributions of Continuous EDA Variables', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()

# --- 3. Deal type weight & intra-regional (categorical/ordinal) bar charts ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

eda_df['deal_type_weight'].value_counts().sort_index().plot(
    kind='bar', ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title('Deal Type Weight Distribution', fontweight='bold')
axes[0].set_xlabel('Deal Type Weight')
axes[0].set_ylabel('Count')

eda_df['intra_regional'].value_counts().sort_index().plot(
    kind='bar', ax=axes[1], color=['salmon', 'steelblue'], edgecolor='white')
axes[1].set_title('Intra-Regional Flag Distribution', fontweight='bold')
axes[1].set_xlabel('Intra-Regional (0=No, 1=Yes)')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.show()   #why is the second deal type not showing up? Most deal types are PSAs, so not many deeply integrated deals such as customes unions.

# # --- 4. Correlation Heatmap ---
# corr_matrix = eda_df[continuous_cols + ['deal_type_weight']].corr()
# fig, ax = plt.subplots(figsize=(12, 9))
# mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
# sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
#             center=0, square=True, linewidths=0.5, ax=ax,
#             cbar_kws={'shrink': 0.8})
# ax.set_title('Correlation Heatmap of EDA Variables', fontsize=14, fontweight='bold')
# plt.tight_layout()
# plt.show()

# # --- 5. Box plots by intra-regional flag ---
# #score_cols = ['sectoral_score', 'trade_lib_score', 'regulatory_score']
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# for i, col in enumerate(score_cols):
#     sns.boxplot(data=eda_df, x='intra_regional', y=col, ax=axes[i],
#                 palette=['salmon', 'steelblue'])
#     axes[i].set_title(f'{col} by Intra-Regional', fontweight='bold')
#     axes[i].set_xlabel('Intra-Regional (0=No, 1=Yes)')
# fig.suptitle('Score Distributions: Intra vs Inter-Regional Deals', fontsize=14, fontweight='bold', y=1.02)
# plt.tight_layout()
# plt.show()

# # --- 6. Regional mean scores heatmap (reporter WHO region) ---
# region_means = (
#     final_merged
#     .groupby('reporter_WHO_region')[score_cols + ['avg_price_per_dose']]
#     .mean()
# )
# fig, ax = plt.subplots(figsize=(10, 5))
# sns.heatmap(region_means, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, linewidths=0.5)
# ax.set_title('Mean Scores & Avg Price per Dose by Reporter WHO Region', fontsize=13, fontweight='bold')
# plt.tight_layout()
# plt.show()

# # --- 7. Scatter: trade_lib_score vs avg_price_per_dose ---
# fig, ax = plt.subplots(figsize=(8, 6))
# scatter = ax.scatter(
#     eda_df['trade_lib_score'], eda_df['avg_price_per_dose'],
#     c=eda_df['intra_regional'], cmap='coolwarm', alpha=0.4, s=15)
# ax.set_xlabel('Trade Liberalization Score', fontsize=11)
# ax.set_ylabel('Avg Price per Dose (USD)', fontsize=11)
# ax.set_title('Trade Liberalization vs Vaccine Price', fontsize=13, fontweight='bold')
# plt.colorbar(scatter, label='Intra-Regional')
# plt.tight_layout()
# plt.show()

# # --- 8. Region deal count vs avg price per dose ---
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.scatter(eda_df['region_deal_count'], eda_df['avg_price_per_dose'],
#            alpha=0.3, s=15, color='steelblue')
# ax.set_xlabel('Region Deal Count', fontsize=11)
# ax.set_ylabel('Avg Price per Dose (USD)', fontsize=11)
# ax.set_title('Region Deal Count vs Vaccine Price', fontsize=13, fontweight='bold')
# plt.tight_layout()
# plt.show()

# print("=" * 60)
# print("EDA COMPLETE")
# print("=" * 60)

# --- Aggregate to Region × Vaccine level ---

# Aggregate deal-level scores and flags to reporter WHO region
region_vaccine_df = (
    final_merged
    .groupby(['reporter_WHO_region', 'Vaccine'])
    .agg(
        avg_price_per_dose     = ('avg_price_per_dose', 'mean'),
        sectoral_score_mean    = ('sectoral_score', 'mean'),
        trade_lib_score_mean   = ('trade_lib_score', 'mean'),
        regulatory_score_mean  = ('regulatory_score', 'mean'),
        deal_type_weight_mean  = ('deal_type_weight', 'mean'),
        intra_regional_share   = ('intra_regional', 'mean'),   # proportion of intra-regional deals
        deal_count             = ('intra_regional', 'size'),    # total deals in this region
        # Provision coverage rates (share of deals with each provision)
        health_coverage        = ('Health_le', 'mean'),
        ipr_coverage           = ('IPR_le', 'mean'),
        env_coverage           = ('EnvironmentalLaws_le', 'mean'),
        consumer_prot_coverage = ('ConsumerProtection_le', 'mean'),
        innovation_coverage    = ('InnovationPolicies_le', 'mean'),
        illicit_drugs_coverage = ('IllicitDrugs_le', 'mean'),
        social_matters_coverage= ('SocialMatters_le', 'mean'),
        # Vaccine market covariates
        num_manufacturers      = ('num_manufacturers', 'first'),  # already at region×vaccine level
        total_annual_volume    = ('total_annual_volume', 'first'),
    )
    .reset_index()
)

print(f"Region × Vaccine dataset: {region_vaccine_df.shape[0]} rows, {region_vaccine_df.shape[1]} cols")
print(region_vaccine_df.head(10))
print(f"\nMissing values:\n{region_vaccine_df.isnull().sum()}")

#Apply Causal ML techniques



#What is our treatment variable?

print(final_merged[eda_cols].isnull().sum())
final_merged.columns

#Treatment variable: trade liberalization score 
#Outcome variable: avg price per dose at the WHO region + vaccine level
#Covariates: 'num_manufacturers', 'total_annual_volume'
#Confounders: 'year','reporter_WHO_region','partner_WHO_region','environmental_laws','IPR','Consumer Protection','Innovation Policies','Health','Illicit Drugs','Social Matters'


#Interpretation

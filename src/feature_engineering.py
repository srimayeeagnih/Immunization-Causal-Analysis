import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(os.path.join(_ROOT, "data", "processed", "pivot_dataset.csv"))

df.columns

df.drop(columns=['gni_per_capita_usd','tariff_x_oop'], axis=1, inplace=True)

#Step 1: Create an interaction term between pharma tariff and health expenditure per capita
df['tariff_health'] = df['pharma_tariff_rate'] * df['health_exp_pct_gdp']

#Step 2: Create another interaction term between health expenditure per capita and out of pocket expenditure per capita. This can help
#Capture the effect of health expenditure on out of pocket expenditure, which may be an important factor in determining the overall cost burden on individuals.
df['health_exp_oop_interaction'] = df['health_exp_pct_gdp'] * df['oop_health_exp_pct']

#Step 3: Missing Value Inspection
cols = ['health_exp_pct_gdp', 'oop_health_exp_pct']

# Overall missing counts and %
print(df[cols].isna().sum().rename('missing_count').to_frame()
      .assign(missing_pct=lambda x: (x['missing_count'] / len(df) * 100).round(2)))

# Missing % by country
print(df.groupby('country')[cols].apply(lambda g: g.isna().mean() * 100).round(2))

#Step 4: MICE imputation (linear regression, health_exp_pct_gdp <-> oop_health_exp_pct)
mice = IterativeImputer(
    estimator=LinearRegression(),
    max_iter=10,
    random_state=42,
    verbose=0
)
df[cols] = mice.fit_transform(df[cols])

#Step 5: Encode categorical variables
# Bin rare countries (< 0.5% frequency) into 'Other'
freq = df['country'].value_counts(normalize=True)
rare_countries = freq[freq < 0.005].index
df['country'] = df['country'].where(~df['country'].isin(rare_countries), other='Other')

# Label encode all categorical columns except reporter_flag and gavi_eligible
cat_cols = ['unicef_region', 'country', 'vaccine', 'antigen_family']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

#Step 6: Log-transform immunization coverage
df['immunization_coverage'] = np.log1p(df['immunization_coverage'])

#Step 7: Bin GDP per capita into World Bank income groups (arbitrary thresholds)
bins   = [0, 1135, 4465, 13845, np.inf]
labels = ['Low', 'Lower-middle', 'Upper-middle', 'High']
df['income_group'] = pd.cut(df['gdp_per_capita_usd'], bins=bins, labels=labels)
print(df['income_group'].value_counts())

# Label encode income group (ordinal — preserves order)
income_order = {'Low': 0, 'Lower-middle': 1, 'Upper-middle': 2, 'High': 3}
df['income_group'] = df['income_group'].map(income_order)

#Step 8: Violin plots for anomaly inspection
numeric_cols = [
    'immunization_coverage', 'pharma_tariff_rate',
    'gdp_per_capita_usd', 'health_exp_pct_gdp',
    'oop_health_exp_pct', 'population_total'
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.violinplot(y=df[col].dropna(), ax=axes[i], color='steelblue', inner='quartile')
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].set_ylabel('')

fig.suptitle('Violin Plots — Anomaly Inspection', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#Step 9: Recency control — years since antigen introduction per country
# For each country-antigen pair, find first nonzero year and minimum coverage
nonzero = df[df['immunization_coverage'] > 0]
first_intro = (
    nonzero.groupby(['country', 'antigen_family'])['year']
    .min()
    .rename('first_intro_year')
)
min_coverage = (
    nonzero.groupby(['country', 'antigen_family'])['immunization_coverage']
    .min()
    .rename('min_coverage')
)
intro_meta = pd.concat([first_intro, min_coverage], axis=1)

# Flag established programs: min coverage >= 50% means data window doesn't capture true intro
ESTABLISHED_THRESHOLD = 50
intro_meta['established'] = intro_meta['min_coverage'] >= ESTABLISHED_THRESHOLD

df = df.join(intro_meta, on=['country', 'antigen_family'])
df['years_since_intro'] = (df['year'] - df['first_intro_year']).clip(lower=0)

# Established programs: years_since_intro = 0 (unknown), flag separately
df.loc[df['established'] == True, 'years_since_intro'] = np.nan
df['years_since_intro'] = df['years_since_intro'].fillna(0)
df['is_established_program'] = df['established'].astype(int)
df = df.drop(columns=['first_intro_year', 'min_coverage', 'established'])
print(df.groupby('antigen_family')['years_since_intro'].describe().round(1))

df.to_csv(os.path.join(_ROOT, "data", "processed", "pivot_dataset_fe.csv"), index=False)
print(f"Saved -> data/processed/pivot_dataset_fe.csv")
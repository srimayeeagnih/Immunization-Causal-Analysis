# IMPORTS

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import RFECV, SelectFromModel

# Scipy
from scipy.stats import skew, kurtosis, shapiro, spearmanr, ks_2samp

# SHAP for model interpretability
import shap

# Load the dataset with encoding specified
df = pd.read_csv(r"C:/Users/srima/OneDrive/Desktop/INSY674/Individual Project 1/dataset.csv", encoding='latin-1')

# Display basic information about the dataset
print("Dataset loaded successfully!")
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nColumn names:")
print(df.columns.tolist())
print(f"\nData types:")
print(df.dtypes)

df.columns

#DATA CLEANING

#Deleting unusable columns
df1 = df.drop(['Country Code','Industry Code', 'Series Code'], axis=1)

df1.columns
#Transposing 'Series Name' values into separate headers/columns

# First, identify the year columns (adjust this based on your actual column names)
# For example, if your years are from 2000 to 2023:
years = [col for col in df1.columns if col not in ['Country Name', 'Series Name', 'Industry Name']]

df1_melted = df1.melt(
    id_vars=['Country Name', 'Series Name', 'Industry Name'],
    value_vars=years,
    var_name='Year',
    value_name='Value'
)

df1_pivot = df1_melted.pivot_table(
    index=['Country Name', 'Industry Name', 'Year'],
    columns='Series Name',
    values='Value',
    aggfunc='first'
).reset_index() 

df1_pivot.head()

df1_pivot.columns

#Keep the mean value of each series' column

# Define key columns to always keep
key_columns = ['Country Name', 'Industry Name', 'Year']

# Keep key columns + columns containing 'Mean'
columns_to_keep = key_columns + [col for col in df1_pivot.columns if 'Mean' in str(col)]

df1_pivot = df1_pivot[columns_to_keep]

df1_pivot.columns


# Get feature columns (exclude key columns)
feature_cols = [col for col in df1_pivot.columns if col not in key_columns]

# Replace '..' with NaN (your dataset uses '..' for missing values)
df1_pivot[feature_cols] = df1_pivot[feature_cols].replace('..', np.nan)

# ============================================
# MISSING DATA PATTERN DIAGNOSTIC
# Check if pattern is Monotone or Arbitrary
# ============================================

def check_monotone_pattern(df, cols):
    """
    Check if missing data follows a monotone pattern.
    Monotone: If col_j is missing, all cols after j are also missing.
    Returns: percentage of rows that follow monotone pattern
    """
    missing_matrix = df[cols].isnull()
    monotone_count = 0

    for idx, row in missing_matrix.iterrows():
        first_missing = None
        for i, val in enumerate(row):
            if val:
                first_missing = i
                break
        if first_missing is None:
            monotone_count += 1
        elif all(row[first_missing:]):
            monotone_count += 1

    return monotone_count / len(df) * 100

# 1. Count unique missing patterns
missing_patterns = df1_pivot[feature_cols].isnull().astype(int).apply(tuple, axis=1)
unique_patterns = missing_patterns.value_counts()
print("\n" + "="*50)
print("MISSING DATA PATTERN DIAGNOSTIC")
print("="*50)
print(f"\nNumber of unique missing patterns: {len(unique_patterns)}")
print(f"Top 10 most common patterns (1=missing, 0=present):")
print(unique_patterns.head(10))

# 2. Check monotonicity
monotone_pct = check_monotone_pattern(df1_pivot, feature_cols)
print(f"\n% of rows with monotone pattern: {monotone_pct:.2f}%")

if monotone_pct > 80:
    print("-> Pattern is likely MONOTONE - use monotone imputation methods")
elif monotone_pct < 20:
    print("-> Pattern is likely ARBITRARY - use MCMC or FCS imputation")
else:
    print("-> Pattern is MIXED - consider hybrid approaches")

# 3. Missing value correlation matrix
missing_corr = df1_pivot[feature_cols].isnull().corr()
print(f"\nMissing value correlation (do variables tend to be missing together?):")
print(missing_corr.round(2))

# 4. Per-variable missing percentage
print(f"\nMissing % per variable:")
for col in feature_cols:
    miss_pct = df1_pivot[col].isnull().sum() / len(df1_pivot) * 100
    print(f"  {col}: {miss_pct:.1f}%")

print("="*50 + "\n")

# ============================================
# TRAIN/VALIDATION/TEST SPLIT (80/10/10)
# Temporal split based on Year
# ============================================

# Extract numeric year for sorting
df1_pivot['Year_Num'] = df1_pivot['Year'].str.extract(r'(\d{4})').astype(int)

# Check year distribution
print("\n" + "="*50)
print("TEMPORAL SPLIT")
print("="*50)
print("\nYear distribution in dataset:")
print(df1_pivot['Year_Num'].value_counts().sort_index())

# Define temporal boundaries
# Years: 1997-2014 (18 years)
# Train: 1997-2010 (14 years, ~78%)
# Validation: 2011-2012 (2 years, ~11%)
# Test: 2013-2014 (2 years, ~11%)

TRAIN_END = 2010
VAL_END = 2012

train_df = df1_pivot[df1_pivot['Year_Num'] <= TRAIN_END].copy()
val_df = df1_pivot[(df1_pivot['Year_Num'] > TRAIN_END) & (df1_pivot['Year_Num'] <= VAL_END)].copy()
test_df = df1_pivot[df1_pivot['Year_Num'] > VAL_END].copy()

# Remove helper column from all dataframes
train_df = train_df.drop(['Year_Num'], axis=1)
val_df = val_df.drop(['Year_Num'], axis=1)
test_df = test_df.drop(['Year_Num'], axis=1)
df1_pivot = df1_pivot.drop(['Year_Num'], axis=1)

# Verify the splits
print(f"\nTotal rows: {len(df1_pivot)}")
print(f"Train (≤{TRAIN_END}): {len(train_df)} ({len(train_df)/len(df1_pivot)*100:.1f}%)")
print(f"Validation ({TRAIN_END+1}-{VAL_END}): {len(val_df)} ({len(val_df)/len(df1_pivot)*100:.1f}%)")
print(f"Test (>{VAL_END}): {len(test_df)} ({len(test_df)/len(df1_pivot)*100:.1f}%)")

# Show year ranges in each split
print(f"\nYear ranges:")
print(f"  Train: {train_df['Year'].min()} to {train_df['Year'].max()}")
print(f"  Validation: {val_df['Year'].min()} to {val_df['Year'].max()}")
print(f"  Test: {test_df['Year'].min()} to {test_df['Year'].max()}")

# Missing data ratio verification (will differ due to temporal patterns, which is expected)
print(f"\nMissing data ratio by split:")
print(f"  Train missing %: {train_df[feature_cols].isnull().sum().sum() / (len(train_df) * len(feature_cols)) * 100:.2f}%")
print(f"  Val missing %: {val_df[feature_cols].isnull().sum().sum() / (len(val_df) * len(feature_cols)) * 100:.2f}%")
print(f"  Test missing %: {test_df[feature_cols].isnull().sum().sum() / (len(test_df) * len(feature_cols)) * 100:.2f}%")

print(f"\n⚠️  Note: Missing data ratios may differ across splits due to temporal patterns.")
print(f"   This is expected and reflects real-world forecasting conditions.")

# ============================================
# DISTRIBUTION SHIFT DETECTION & MITIGATION
# ============================================

def detect_distribution_shift(train_data, test_data, col_name, alpha=0.05):
    """Use Kolmogorov-Smirnov test to detect distribution shift."""
    train_clean = pd.to_numeric(train_data, errors='coerce').dropna()
    test_clean = pd.to_numeric(test_data, errors='coerce').dropna()

    if len(train_clean) < 5 or len(test_clean) < 5:
        return None, None

    stat, p_value = ks_2samp(train_clean, test_clean)
    return stat, p_value

print("\n" + "="*50)
print("DISTRIBUTION SHIFT DETECTION & MITIGATION")
print("="*50)

# Check distribution shift for each feature
shift_results = []
shifted_features = []

print("\nChecking distribution shift (KS test)...")
for col in feature_cols:
    if col not in train_df.columns:
        continue
    stat, p_value = detect_distribution_shift(train_df[col], test_df[col], col)
    if stat is not None:
        shift_results.append({
            'feature': col,
            'ks_statistic': stat,
            'p_value': p_value,
            'shifted': p_value < 0.05
        })
        if p_value < 0.05:
            shifted_features.append(col)

shift_df = pd.DataFrame(shift_results)

if len(shift_df) > 0:
    n_shifted = shift_df['shifted'].sum()
    print(f"\nFeatures with distribution shift: {n_shifted}/{len(shift_df)} ({n_shifted/len(shift_df)*100:.1f}%)")

    # Show top shifted features
    top_shifted = shift_df.nlargest(5, 'ks_statistic')
    print("\nTop 5 features with largest shift:")
    for _, row in top_shifted.iterrows():
        status = "SHIFTED" if row['shifted'] else "OK"
        print(f"  {row['feature'][:40]}...: KS={row['ks_statistic']:.3f}, p={row['p_value']:.4f} [{status}]")

# Mitigation strategies
print("\n" + "="*50)
print("MITIGATION STRATEGIES")
print("="*50)

print("\n1. MONITORING FEATURES IDENTIFIED:")
print("   - Track shifted features during production")
print("   - Retrain model when shift exceeds threshold")

print("\n2. SAMPLE REWEIGHTING (Conceptual):")
print("   - Weight samples by density ratio: P(test)/P(train)")
print("   - Helps model generalize to test distribution")

print("\n3. FEATURE STANDARDIZATION:")
print("   - Apply consistent scaling across all splits")
print("   - Use train statistics for val/test normalization")

# Calculate distribution statistics for monitoring
train_stats = {}
for col in feature_cols[:10]:  # Sample of features
    if col in train_df.columns:
        train_clean = pd.to_numeric(train_df[col], errors='coerce').dropna()
        if len(train_clean) > 0:
            train_stats[col] = {
                'mean': train_clean.mean(),
                'std': train_clean.std(),
                'q25': train_clean.quantile(0.25),
                'q75': train_clean.quantile(0.75)
            }

# Plot distribution comparison for top shifted feature
if len(shifted_features) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Most shifted feature
    top_shifted_col = shift_df.nlargest(1, 'ks_statistic')['feature'].values[0]
    train_vals = pd.to_numeric(train_df[top_shifted_col], errors='coerce').dropna()
    test_vals = pd.to_numeric(test_df[top_shifted_col], errors='coerce').dropna()

    axes[0].hist(train_vals, bins=30, alpha=0.5, label='Train', density=True)
    axes[0].hist(test_vals, bins=30, alpha=0.5, label='Test', density=True)
    axes[0].set_title(f'Distribution Shift: {top_shifted_col[:30]}...')
    axes[0].legend()
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')

    # KS statistics distribution
    axes[1].hist(shift_df['ks_statistic'], bins=20, edgecolor='black')
    axes[1].axvline(x=0.1, color='r', linestyle='--', label='Threshold (0.1)')
    axes[1].set_title('Distribution of KS Statistics Across Features')
    axes[1].set_xlabel('KS Statistic')
    axes[1].set_ylabel('Count')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('distribution_shift_analysis.png', dpi=100)
    plt.close()
    print("\nSaved: distribution_shift_analysis.png")

print(f"\nDistribution shift analysis complete.")

# ============================================
# HANDLING MISSING DATA (Imputation)
# Adaptive: Linear Regression (normal) vs PMM (skewed)
# ============================================
def check_skewness(series):
    """Return True if distribution is skewed (|skew| > 1)"""
    clean = series.dropna()
    if len(clean) < 3:
        return False  # Not enough data, default to linear
    return abs(skew(clean)) > 1

def pmm_impute(train_obs, train_miss, predictors_obs, predictors_miss, k=5):
    """
    Predictive Mean Matching imputation
    - Fit model on observed
    - Find k donors with closest predicted values
    - Return actual values from donors
    """
    model = LinearRegression()
    model.fit(predictors_obs, train_obs)

    pred_obs = model.predict(predictors_obs)
    pred_miss = model.predict(predictors_miss)

    imputed_values = []
    for pred in pred_miss:
        # Find k nearest donors
        distances = np.abs(pred_obs - pred)
        donor_indices = np.argsort(distances)[:k]
        # Randomly select one donor
        donor_idx = np.random.choice(donor_indices)
        imputed_values.append(train_obs.iloc[donor_idx])

    return np.array(imputed_values)

# Convert feature columns to numeric (in case they're still strings)
for df in [train_df, val_df, test_df]:
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Sort columns by missingness (least → most) for monotone approach
missing_counts = train_df[feature_cols].isnull().sum()
sorted_cols = missing_counts.sort_values().index.tolist()

print("\n" + "="*50)
print("IMPUTATION PROCESS")
print("="*50)

np.random.seed(42)  # For reproducibility

for col in sorted_cols:
    if train_df[col].isnull().sum() == 0:
        continue  # Skip columns with no missing values

    # Check if skewed or normal
    is_skewed = check_skewness(train_df[col])
    method = "PMM" if is_skewed else "Linear Regression"

    # Get previously imputed columns as predictors
    prev_cols = sorted_cols[:sorted_cols.index(col)]

    if len(prev_cols) == 0:
        # No predictors available, use mean/median
        if is_skewed:
            fill_val = train_df[col].median()
        else:
            fill_val = train_df[col].mean()
        train_df[col] = train_df[col].fillna(fill_val)
        val_df[col] = val_df[col].fillna(fill_val)
        test_df[col] = test_df[col].fillna(fill_val)
        print(f"  {col}: {method} (no predictors, used {'median' if is_skewed else 'mean'})")
    else:
        # Get complete cases for fitting
        train_complete = train_df[prev_cols + [col]].dropna()

        if len(train_complete) < 10:
            # Not enough complete cases, use mean/median
            fill_val = train_df[col].median() if is_skewed else train_df[col].mean()
            train_df[col] = train_df[col].fillna(fill_val)
            val_df[col] = val_df[col].fillna(fill_val)
            test_df[col] = test_df[col].fillna(fill_val)
            print(f"  {col}: {method} (insufficient data, used {'median' if is_skewed else 'mean'})")
            continue

        if is_skewed:
            # PMM imputation
            for df in [train_df, val_df, test_df]:
                mask = df[col].isnull()
                if mask.sum() > 0:
                    # Need complete predictors for missing rows
                    miss_predictors = df.loc[mask, prev_cols]
                    # Handle any missing predictors (should be imputed already)
                    if miss_predictors.isnull().any().any():
                        miss_predictors = miss_predictors.fillna(miss_predictors.mean())

                    imputed = pmm_impute(
                        train_complete[col],
                        df.loc[mask, col],
                        train_complete[prev_cols],
                        miss_predictors,
                        k=5
                    )
                    df.loc[mask, col] = imputed
        else:
            # Linear Regression imputation
            model = LinearRegression()
            model.fit(train_complete[prev_cols], train_complete[col])

            for df in [train_df, val_df, test_df]:
                mask = df[col].isnull()
                if mask.sum() > 0:
                    predictors = df.loc[mask, prev_cols]
                    if predictors.isnull().any().any():
                        predictors = predictors.fillna(predictors.mean())
                    df.loc[mask, col] = model.predict(predictors)

        print(f"  {col}: {method} (skew={skew(train_df[col].dropna()):.2f})")

print("="*50)
print("\nImputation complete!")
print(f"Train missing after: {train_df[feature_cols].isnull().sum().sum()}")
print(f"Val missing after: {val_df[feature_cols].isnull().sum().sum()}")
print(f"Test missing after: {test_df[feature_cols].isnull().sum().sum()}")


# ============================================
# GENERATING TARGET VARIABLE
# Net Active Export = Exporter Value - Exiter Value
# ============================================

# Define the columns for the latent variable
exporter_col = '006.Export Value per Exporter: Mean'
exiter_col = '016.Export Value per Exiter: Mean'

# Verify columns exist
candidate_columns = [col for col in df1_pivot.columns if 'Export Value' in str(col)]
print("Export Value columns found:")
print(candidate_columns)

# Create the Net Active Export variable (captures net growth after exits)
# Higher value = more export value retained (good)
# Lower/negative = losing more to exits than gaining

def create_net_export_target(df, exporter_col, exiter_col):
    """
    Create Net Active Export latent variable.
    Net Export = Export Value per Exporter - Export Value per Exiter

    Interpretation:
    - Positive: Active exporters generate more value than is lost to exits
    - Negative: Exit losses exceed active exporter value
    """
    return df[exporter_col] - df[exiter_col]

# Apply to train, val, test sets
for df in [train_df, val_df, test_df]:
    df['Average_Net_Active_Export'] = create_net_export_target(df, exporter_col, exiter_col)

# Set target column
target_col = 'Average_Net_Active_Export'

# Quick summary of the new target variable
print(f"\n{'='*50}")
print("TARGET VARIABLE CREATED: 'Average_Net_Active_Export'")
print(f"{'='*50}")
print(f"Formula: {exporter_col} - {exiter_col}")
print(f"\nTrain set summary:")
print(f"  Mean: {train_df[target_col].mean():.2f}")
print(f"  Median: {train_df[target_col].median():.2f}")
print(f"  Std: {train_df[target_col].std():.2f}")
print(f"  Min: {train_df[target_col].min():.2f}")
print(f"  Max: {train_df[target_col].max():.2f}")
print(f"  % Positive (net gain): {(train_df[target_col] > 0).mean()*100:.1f}%")
print(f"  % Negative (net loss): {(train_df[target_col] < 0).mean()*100:.1f}%")

# Update feature_cols to exclude the target and its components
predictor_cols = [col for col in feature_cols if col not in [exporter_col, exiter_col, target_col]]
print(f"\nPredictor columns (excluding target components): {len(predictor_cols)}")



# FEATURE EDA - Pre-Model Selection Analysis

def analyze_target(df, target_col):
    """
    Analyze target variable distribution.
    Returns: dict with skewness, normality test, outlier count
    """
    y = df[target_col].dropna()

    print(f"\n{'='*50}")
    print(f"TARGET VARIABLE ANALYSIS: {target_col}")
    print(f"{'='*50}")

    # Basic stats
    print(f"Mean: {y.mean():.2f}")
    print(f"Median: {y.median():.2f}")
    print(f"Std Dev: {y.std():.2f}")

    # Distribution shape
    skewness = skew(y)
    kurt = kurtosis(y)
    print(f"\nSkewness: {skewness:.2f}", end=" ")
    if abs(skewness) < 0.5:
        print("(approximately symmetric)")
    elif abs(skewness) < 1:
        print("(moderately skewed)")
    else:
        print("(highly skewed - consider transformation)")

    print(f"Kurtosis: {kurt:.2f}")

    # Normality test (Shapiro-Wilk, use sample if large)
    sample = y.sample(min(5000, len(y)), random_state=42)
    stat, p_value = shapiro(sample)
    print(f"\nShapiro-Wilk test p-value: {p_value:.4f}", end=" ")
    if p_value < 0.05:
        print("(NOT normal)")
    else:
        print("(approximately normal)")

    # Outliers (IQR method)
    Q1, Q3 = y.quantile(0.25), y.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((y < Q1 - 1.5*IQR) | (y > Q3 + 1.5*IQR)).sum()
    print(f"\nOutliers (IQR method): {outliers} ({outliers/len(y)*100:.1f}%)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(y, bins=50, edgecolor='black')
    axes[0].set_title(f'Distribution of {target_col}')
    axes[0].set_xlabel(target_col)
    axes[1].boxplot(y)
    axes[1].set_title(f'Boxplot of {target_col}')
    plt.tight_layout()
    plt.savefig('target_distribution.png', dpi=100)
    plt.close()
    print("\nSaved: target_distribution.png")

    return {'skewness': skewness, 'p_value': p_value, 'outliers_pct': outliers/len(y)*100}


def analyze_feature_target_relationships(df, feature_cols, target_col, top_n=10):
    """
    Analyze relationships between features and target.
    Returns: correlation series sorted by absolute value
    """
    print(f"\n{'='*50}")
    print("FEATURE-TARGET RELATIONSHIPS")
    print(f"{'='*50}")

    # Correlations with target
    correlations = df[feature_cols].corrwith(df[target_col]).sort_values(key=abs, ascending=False)

    print(f"\nTop {top_n} correlations with {target_col}:")
    for feat, corr in correlations.head(top_n).items():
        direction = "+" if corr > 0 else "-"
        strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
        print(f"  {feat}: {corr:+.3f} ({strength} {direction})")

    # Check for non-linear relationships (compare correlation vs rank correlation)
    print(f"\nLinear vs Non-linear check (Pearson vs Spearman):")
    for feat in correlations.head(5).index:
        pearson = correlations[feat]
        spearman, _ = spearmanr(df[feat].dropna(), df.loc[df[feat].notna(), target_col])
        diff = abs(spearman) - abs(pearson)
        if abs(diff) > 0.1:
            print(f"  {feat}: Pearson={pearson:.3f}, Spearman={spearman:.3f} -> likely NON-LINEAR")
        else:
            print(f"  {feat}: Pearson={pearson:.3f}, Spearman={spearman:.3f} -> likely linear")

    return correlations


def analyze_multicollinearity(df, feature_cols, threshold=0.8):
    """
    Check for multicollinearity among features.
    Returns: pairs of highly correlated features
    """
    print(f"\n{'='*50}")
    print("MULTICOLLINEARITY CHECK")
    print(f"{'='*50}")

    corr_matrix = df[feature_cols].corr()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i, col1 in enumerate(feature_cols):
        for col2 in feature_cols[i+1:]:
            corr = corr_matrix.loc[col1, col2]
            if abs(corr) > threshold:
                high_corr_pairs.append((col1, col2, corr))

    if high_corr_pairs:
        print(f"\nHighly correlated feature pairs (|r| > {threshold}):")
        for col1, col2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {col1} <-> {col2}: {corr:.3f}")
        print("\n-> Consider removing one from each pair for linear models")
    else:
        print(f"\nNo feature pairs with |correlation| > {threshold}")

    # Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=100)
    plt.close()
    print("\nSaved: correlation_heatmap.png")

    return high_corr_pairs


def analyze_feature_distributions(df, feature_cols):
    """
    Analyze distributions of all features.
    Returns: dict with skewness and outlier info per feature
    """
    print(f"\n{'='*50}")
    print("FEATURE DISTRIBUTIONS")
    print(f"{'='*50}")

    results = {}
    skewed_features = []
    high_outlier_features = []

    for col in feature_cols:
        data = df[col].dropna()
        sk = skew(data)
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        outlier_pct = ((data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)).sum() / len(data) * 100

        results[col] = {'skewness': sk, 'outlier_pct': outlier_pct}

        if abs(sk) > 1:
            skewed_features.append((col, sk))
        if outlier_pct > 5:
            high_outlier_features.append((col, outlier_pct))

    print(f"\nSkewed features (|skew| > 1): {len(skewed_features)}/{len(feature_cols)}")
    for col, sk in sorted(skewed_features, key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {col}: skew={sk:.2f}")

    print(f"\nFeatures with many outliers (>5%): {len(high_outlier_features)}/{len(feature_cols)}")
    for col, pct in sorted(high_outlier_features, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {col}: {pct:.1f}% outliers")

    return results


def check_dimensionality(df, feature_cols):
    """
    Check sample size vs feature count ratio.
    """
    print(f"\n{'='*50}")
    print("DIMENSIONALITY CHECK")
    print(f"{'='*50}")

    n_samples = len(df)
    n_features = len(feature_cols)
    ratio = n_samples / n_features

    print(f"\nSamples (n): {n_samples}")
    print(f"Features (p): {n_features}")
    print(f"Ratio (n/p): {ratio:.1f}")

    if ratio < 5:
        print("\n-> LOW ratio: High risk of overfitting")
        print("   Recommendations: Regularization (Lasso/Ridge), PCA, or feature selection")
    elif ratio < 20:
        print("\n-> MODERATE ratio: Some risk of overfitting")
        print("   Recommendations: Consider regularization or simpler models")
    else:
        print("\n-> GOOD ratio: Lower risk of overfitting")
        print("   Most algorithms should work well")


def summarize_model_recommendations(target_analysis, feature_distributions, high_corr_pairs, n_samples, n_features):
    """
    Provide model recommendations based on EDA findings.
    """
    print(f"\n{'='*50}")
    print("MODEL RECOMMENDATIONS")
    print(f"{'='*50}")

    recommendations = []

    # Based on target skewness
    if abs(target_analysis['skewness']) > 1:
        recommendations.append("- Target is skewed: Consider log-transform or tree-based models")

    # Based on outliers
    if target_analysis['outliers_pct'] > 5:
        recommendations.append("- Many outliers: Tree-based models (RF, XGBoost) are more robust")

    # Based on multicollinearity
    if len(high_corr_pairs) > 3:
        recommendations.append("- High multicollinearity: Avoid plain linear regression, use Ridge/Lasso or trees")

    # Based on dimensionality
    ratio = n_samples / n_features
    if ratio < 10:
        recommendations.append("- Low n/p ratio: Use regularization (Lasso/Ridge) or dimensionality reduction")

    # Count skewed features
    skewed_count = sum(1 for v in feature_distributions.values() if abs(v['skewness']) > 1)
    if skewed_count > n_features * 0.5:
        recommendations.append("- Many skewed features: Tree-based models handle this well without transformation")

    print("\nBased on your data characteristics:")
    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("- Data looks well-behaved, most algorithms should work")

    print("\nSuggested models to try:")
    print("  1. Random Forest (robust baseline)")
    print("  2. XGBoost/LightGBM (usually best performance)")
    print("  3. Ridge/Lasso Regression (if interpretability matters)")


# RUN EDA
target_col = 'Average_Net_Active_Export'

target_results = analyze_target(train_df, target_col)
correlations = analyze_feature_target_relationships(train_df, predictor_cols, target_col)
high_corr = analyze_multicollinearity(train_df, predictor_cols)
feat_dist = analyze_feature_distributions(train_df, predictor_cols)
check_dimensionality(train_df, predictor_cols)
summarize_model_recommendations(target_results, feat_dist, high_corr, len(train_df), len(predictor_cols))

#Final Choice: Random Forest Regressor + SHAP

#Rationale: 
# RF handles nonlinear patterns and relationships in the predictors
# Offers moderate interpretability with SHAP analysis and feature importance  

# Remove redundant inverse-ratio features

print(f"\n{'='*50}")
print("FEATURE ENGINEERING: Removing Redundant Features")
print(f"{'='*50}")

# Identify inverse-ratio pairs
inverse_pairs = [
    ('045.Number of HS6 Products per Exporter: Mean',
     '051.Number of Exporters per HS6 Product: Mean'),
    ('048.Number of Destinations per Exporter: Mean',
     '054.Number of Exporters per Destination: Mean')
]

# Decide which to keep based on correlation with target
cols_to_drop = []

for col1, col2 in inverse_pairs:
    # Check if both columns exist in predictor_cols
    if col1 in predictor_cols and col2 in predictor_cols:
        corr1 = abs(train_df[col1].corr(train_df[target_col]))
        corr2 = abs(train_df[col2].corr(train_df[target_col]))

        print(f"\nPair comparison:")
        print(f"  {col1[:40]}... : |r| = {corr1:.3f}")
        print(f"  {col2[:40]}... : |r| = {corr2:.3f}")

        # Keep the one with higher correlation, drop the other
        if corr1 >= corr2:
            cols_to_drop.append(col2)
            print(f"  -> Dropping: {col2[:40]}...")
        else:
            cols_to_drop.append(col1)
            print(f"  -> Dropping: {col1[:40]}...")

# Update predictor_cols
predictor_cols_clean = [c for c in predictor_cols if c not in cols_to_drop]

print(f"\nFeatures before: {len(predictor_cols)}")
print(f"Features after:  {len(predictor_cols_clean)}")
print(f"Removed: {len(cols_to_drop)} redundant features")

# Update predictor_cols for modeling
predictor_cols = predictor_cols_clean

# ============================================
# DIMENSIONALITY REDUCTION & FEATURE SELECTION
# ============================================

print(f"\n{'='*50}")
print("DIMENSIONALITY REDUCTION & FEATURE SELECTION")
print(f"{'='*50}")

# Prepare data for feature selection
X_train_fs = train_df[predictor_cols]
y_train_fs = train_df[target_col]
X_val_fs = val_df[predictor_cols]
X_test_fs = test_df[predictor_cols]

# Store results for comparison
fs_results = {}

# -----------------------------------------
# 1. TREE-BASED FEATURE SELECTION
# -----------------------------------------
print("\n" + "-"*40)
print("1. TREE-BASED FEATURE SELECTION")
print("-"*40)

# Fit a Random Forest for feature importance
rf_selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_selector.fit(X_train_fs, y_train_fs)

# Get feature importances
importances = rf_selector.feature_importances_
importance_df = pd.DataFrame({
    'feature': predictor_cols,
    'importance': importances
}).sort_values('importance', ascending=False)

# Select features with importance > mean importance
threshold = np.mean(importances)
tree_selected_features = importance_df[importance_df['importance'] > threshold]['feature'].tolist()

print(f"Threshold (mean importance): {threshold:.4f}")
print(f"Features selected: {len(tree_selected_features)}/{len(predictor_cols)}")
print("\nTop 10 features by tree importance:")
for i, row in importance_df.head(10).iterrows():
    print(f"  {row['feature'][:45]}: {row['importance']:.4f}")

# Using SelectFromModel for formal selection
sfm = SelectFromModel(rf_selector, threshold='mean', prefit=True)
X_train_tree = sfm.transform(X_train_fs)
X_val_tree = sfm.transform(X_val_fs)
X_test_tree = sfm.transform(X_test_fs)

fs_results['tree_based'] = {
    'n_features': X_train_tree.shape[1],
    'features': tree_selected_features
}


# Plot 1: Tree-based feature importance

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

top_n = 15
axes[0, 0].barh(range(top_n), importance_df.head(top_n)['importance'].values)
axes[0, 0].set_yticks(range(top_n))
axes[0, 0].set_yticklabels([f[:25] for f in importance_df.head(top_n)['feature'].values])
axes[0, 0].set_xlabel('Importance')
axes[0, 0].set_title('Tree-Based Feature Importance (Top 15)')
axes[0, 0].invert_yaxis()


# Plot 4: Comparison bar chart
methods = list(fs_results.keys())
n_features_list = [fs_results[m]['n_features'] for m in methods]
colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple']
axes[1, 1].bar(methods, n_features_list, color=colors)
axes[1, 1].set_ylabel('Number of Features')
axes[1, 1].set_title('Feature Selection Methods Comparison')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('feature_selection_comparison.png', dpi=100)
plt.close()
print("\nSaved: feature_selection_comparison.png")

print("\nRECOMMENDATION: Using Tree-Based selection for model training")
print("(Preserves interpretability while reducing dimensionality)")

selected_features_final = tree_selected_features

#Encoding the variables

encoder_industry = LabelEncoder()
encoder_country = LabelEncoder()

df1_pivot['Encoded_Industry'] = encoder_industry.fit_transform(df1_pivot['Industry Name'])
df1_pivot['Encoded_Country'] = encoder_country.fit_transform(df1_pivot['Country Name'])
print("\nDataFrame after Label Encoding:")
print(df1_pivot)


# --------------------------------------------
# STAGE 1: RandomizedSearchCV (Broad Search)
# --------------------------------------------
print("\n--- STAGE 1: Randomized Search (Broad Exploration) ---")

# Define broad parameter distributions
rf_param_distributions = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [None, 5, 10, 15, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
    'bootstrap': [True, False]
}

# Initialize Random Forest
rf = RandomForestRegressor(random_state=42)

X_train = train_df[predictor_cols]
y_train = train_df[target_col]

X_val = val_df[predictor_cols]
y_val = val_df[target_col]

X_test = test_df[predictor_cols]
y_test = test_df[target_col]


# Randomized Search
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_param_distributions,
    n_iter=20,  # Reduced for speed (was 50)
    cv=3,       # 3-fold CV for speed (was 5)
    scoring='neg_mean_squared_error',
    n_jobs=-1,  # Use all CPU cores
    random_state=42,
    verbose=1
)

print("Running RandomizedSearchCV...")
random_search.fit(X_train, y_train)

print(f"\nBest parameters from Randomized Search:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"Best CV Score (neg MSE): {random_search.best_score_:.4f}")
print(f"Best CV Score (RMSE): {np.sqrt(-random_search.best_score_):.4f}")

# Skip Stage 2 - Stage 1 results are sufficient for Random Forest

# --------------------------------------------
# FINAL MODEL EVALUATION
# --------------------------------------------
print(f"\n{'='*50}")
print("FINAL MODEL EVALUATION")
print(f"{'='*50}")

# Get the best model from Stage 1
best_rf = random_search.best_estimator_

# Predictions
y_train_pred = best_rf.predict(X_train)
y_val_pred = best_rf.predict(X_val)
y_test_pred = best_rf.predict(X_test)

# Evaluation metrics
def evaluate_model(y_true, y_pred, set_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{set_name} Set:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

train_metrics = evaluate_model(y_train, y_train_pred, "Training")
val_metrics = evaluate_model(y_val, y_val_pred, "Validation")

# --------------------------------------------
# STAGE 2: Fine-Tuning on Train+Val Combined
# --------------------------------------------
print(f"\n{'='*50}")
print("FINE-TUNING (Stage 2)")
print(f"{'='*50}")

# Combine train and validation sets for final tuning
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

# Narrow parameter space around best params from Stage 1
best_params = random_search.best_params_
rf_param_refined = {
    'n_estimators': [max(50, best_params['n_estimators'] - 50),
                     best_params['n_estimators'],
                     best_params['n_estimators'] + 50],
    'max_depth': [best_params['max_depth']],
    'min_samples_split': [best_params['min_samples_split']],
    'min_samples_leaf': [best_params['min_samples_leaf']],
    'max_features': [best_params['max_features']],
    'bootstrap': [best_params['bootstrap']]
}

fine_tune_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=rf_param_refined,
    n_iter=10,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("Running fine-tuning search...")
fine_tune_search.fit(X_train_val, y_train_val)

print(f"\nBest parameters after fine-tuning:")
for param, value in fine_tune_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"Best CV Score (RMSE): {np.sqrt(-fine_tune_search.best_score_):.4f}")

# --------------------------------------------
# FINAL TEST SET EVALUATION
# --------------------------------------------
print(f"\n{'='*50}")
print("TEST SET EVALUATION")
print(f"{'='*50}")

final_model = fine_tune_search.best_estimator_
y_test_pred_final = final_model.predict(X_test)
test_metrics = evaluate_model(y_test, y_test_pred_final, "Test")

# --------------------------------------------
# FEATURE IMPORTANCE (Random Forest)
# --------------------------------------------
print(f"\n{'='*50}")
print("FEATURE IMPORTANCE")
print(f"{'='*50}")

# Get feature importances from the model
feature_importance = pd.DataFrame({
    'feature': predictor_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature'][:50]}: {row['importance']:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'].values)
plt.yticks(range(len(top_features)), [f[:40] for f in top_features['feature'].values])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importances (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100)
plt.close()
print("\nSaved: feature_importance.png")

# --------------------------------------------
# SHAP ANALYSIS
# --------------------------------------------
print(f"\n{'='*50}")
print("SHAP ANALYSIS")
print(f"{'='*50}")

# Use a sample for SHAP (faster computation)
sample_size = min(500, len(X_test))
X_sample = X_test.sample(n=sample_size, random_state=42)

print(f"Computing SHAP values for {sample_size} samples...")

# Create SHAP explainer
explainer = shap.Explainer(final_model, X_sample)
shap_values = explainer(X_sample).values


# SHAP Summary Plot (Bar)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
plt.tight_layout()
plt.savefig('shap_importance.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: shap_importance.png")

# SHAP Summary Plot (Beeswarm)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
plt.tight_layout()
plt.savefig('shap_beeswarm.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: shap_beeswarm.png")

# SHAP feature importance (mean absolute SHAP values)
shap_importance = pd.DataFrame({
    'feature': predictor_cols,
    'shap_importance': np.abs(shap_values).mean(axis=0)
}).sort_values('shap_importance', ascending=False)

print("\nTop 10 Features by SHAP Importance:")
for i, row in shap_importance.head(10).iterrows():
    print(f"  {row['feature'][:50]}: {row['shap_importance']:.4f}")

# ============================================
# MODEL OUTPUT VISUALIZATION & EXPLANATIONS
# ============================================

print(f"\n{'='*50}")
print("MODEL OUTPUT VISUALIZATION & EXPLANATIONS")
print(f"{'='*50}")

# Get predictions
y_pred = final_model.predict(X_test)

# Create visualization figure
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Predicted vs Actual
axes[0, 0].scatter(y_test, y_pred, alpha=0.5, s=20)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Values')
axes[0, 0].set_ylabel('Predicted Values')
axes[0, 0].set_title('Predicted vs Actual Values')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Residual Analysis
residuals = y_test - y_pred

# Residual distribution
axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Residual (Actual - Predicted)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Residual Distribution')
axes[0, 1].grid(True, alpha=0.3)

# Residuals vs Predicted
axes[0, 2].scatter(y_pred, residuals, alpha=0.5, s=20)
axes[0, 2].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 2].set_xlabel('Predicted Values')
axes[0, 2].set_ylabel('Residuals')
axes[0, 2].set_title('Residuals vs Predicted Values')
axes[0, 2].grid(True, alpha=0.3)

# 3. Prediction Distribution
axes[1, 0].hist(y_test, bins=50, alpha=0.5, label='Actual', edgecolor='black')
axes[1, 0].hist(y_pred, bins=50, alpha=0.5, label='Predicted', edgecolor='black')
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution: Actual vs Predicted')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Error by Magnitude
y_test_np = np.array(y_test)
percentiles = np.percentile(y_test_np, [0, 20, 40, 60, 80, 100])
bin_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
bins = pd.cut(y_test, bins=percentiles, labels=bin_labels, include_lowest=True)

error_by_bin = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred,
    'abs_error': np.abs(residuals),
    'bin': bins
}).groupby('bin')['abs_error'].mean()

axes[1, 1].bar(error_by_bin.index, error_by_bin.values, color='steelblue', edgecolor='black')
axes[1, 1].set_xlabel('Actual Value Range')
axes[1, 1].set_ylabel('Mean Absolute Error')
axes[1, 1].set_title('Prediction Error by Value Range')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3)

# 5. Cumulative Error Plot
sorted_abs_errors = np.sort(np.abs(residuals))
cumulative_pct = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors) * 100

axes[1, 2].plot(sorted_abs_errors, cumulative_pct, 'b-', lw=2)
axes[1, 2].axhline(y=90, color='r', linestyle='--', alpha=0.7, label='90th percentile')
axes[1, 2].set_xlabel('Absolute Error')
axes[1, 2].set_ylabel('Cumulative % of Predictions')
axes[1, 2].set_title('Cumulative Error Distribution')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_output_visualization.png', dpi=100)
plt.close()
print("\nSaved: model_output_visualization.png")

# Decision Making Metrics
print("\n" + "-"*40)
print("DECISION MAKING METRICS")
print("-"*40)

mae = np.mean(np.abs(residuals))
mape = np.mean(np.abs(residuals / (y_test + 1e-10))) * 100
within_10pct = np.mean(np.abs(residuals / (y_test + 1e-10)) < 0.10) * 100
within_20pct = np.mean(np.abs(residuals / (y_test + 1e-10)) < 0.20) * 100

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Absolute Percentage Error: {mape:.1f}%")
print(f"Predictions within 10% of actual: {within_10pct:.1f}%")
print(f"Predictions within 20% of actual: {within_20pct:.1f}%")

# Identify high-error cases
error_threshold = np.percentile(np.abs(residuals), 95)
high_error_mask = np.abs(residuals) > error_threshold
n_high_error = high_error_mask.sum()

print(f"\nHigh-error cases (>95th percentile): {n_high_error}")
print(f"Error threshold: {error_threshold:.2f}")

# ============================================
# INDIVIDUAL PREDICTION EXPLANATIONS
# ============================================

print(f"\n{'='*50}")
print("INDIVIDUAL PREDICTION EXPLANATIONS")
print(f"{'='*50}")

# Select sample instances for explanation
sample_indices = [0, len(X_test)//4, len(X_test)//2, 3*len(X_test)//4, len(X_test)-1]
sample_indices = [i for i in sample_indices if i < len(X_test)]

X_explain = X_test.iloc[sample_indices]
shap_values_explain = explainer.shap_values(X_explain)
predictions_explain = final_model.predict(X_explain)

print(f"\nExplaining {len(sample_indices)} sample predictions...")

# Create explanation summary
for idx, (sample_idx, pred) in enumerate(zip(sample_indices, predictions_explain)):
    print(f"\n--- Sample {idx + 1} (Index: {sample_idx}) ---")
    print(f"Predicted Value: {pred:.2f}")

    # Get top contributing features for this prediction
    shap_vals = shap_values_explain[idx]
    feature_contributions = pd.DataFrame({
        'feature': predictor_cols,
        'shap_value': shap_vals,
        'feature_value': X_explain.iloc[idx].values
    }).sort_values('shap_value', key=abs, ascending=False)

    print("Top 5 contributing features:")
    for _, row in feature_contributions.head(5).iterrows():
        direction = "+" if row['shap_value'] > 0 else "-"
        print(f"  {direction} {row['feature'][:40]}: SHAP={row['shap_value']:+.2f}")

# Visualize explanations for first sample
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Waterfall plot for first sample
plt.sca(axes[0])
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_explain[0],
        base_values=explainer.expected_value,
        data=X_explain.iloc[0].values,
        feature_names=predictor_cols
    ),
    max_display=10,
    show=False
)
axes[0].set_title('SHAP Waterfall: Sample Prediction Breakdown')

# Feature importance for the explained samples
mean_abs_shap = np.abs(shap_values_explain).mean(axis=0)
top_indices = np.argsort(mean_abs_shap)[-10:][::-1]

axes[1].barh(
    range(len(top_indices)),
    mean_abs_shap[top_indices],
    color='steelblue'
)
axes[1].set_yticks(range(len(top_indices)))
axes[1].set_yticklabels([predictor_cols[i][:30] for i in top_indices])
axes[1].set_xlabel('Mean |SHAP Value|')
axes[1].set_title('Top 10 Features (Sample Explanations)')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('individual_explanations.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: individual_explanations.png")

print("\n" + "="*50)
print("MODEL INTERPRETATION GUIDELINES")
print("="*50)
print("""
How to interpret SHAP values:
- Positive SHAP: Feature pushes prediction HIGHER
- Negative SHAP: Feature pushes prediction LOWER
- Magnitude: Strength of feature's influence

For decision making:
- Focus on top contributing features
- Check if feature directions make business sense
- Monitor features with high variability in impact
""")

# ============================================
# MODEL EXPORT FOR PRODUCTION DEPLOYMENT
# ============================================

print(f"\n{'='*50}")
print("MODEL EXPORT FOR PRODUCTION DEPLOYMENT")
print(f"{'='*50}")

# 1. Save the trained model
model_path = "my_california_housing_model.pkl"
joblib.dump(final_model, model_path)
print(f"\n1. Model saved: {model_path}")

# 2. Save preprocessing artifacts
scaler_path = "feature_scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"2. Scaler saved: {scaler_path}")

# Save feature names for validation
feature_config = {
    'feature_names': predictor_cols,
    'n_features': len(predictor_cols),
    'train_stats': train_stats
}
feature_config_path = "feature_config.pkl"
joblib.dump(feature_config, feature_config_path)
print(f"3. Feature config saved: {feature_config_path}")

# 3. Create production artifacts bundle
production_bundle = {
    'model': final_model,
    'scaler': scaler,
    'feature_names': predictor_cols,
    'feature_selection_results': {
        method: {
            'n_features': results['n_features'],
            'features': results.get('features', None)
        }
        for method, results in fs_results.items()
        if method != 'pca'
    },
    'train_statistics': train_stats,
    'model_metadata': {
        'model_type': 'RandomForestRegressor',
        'n_estimators': final_model.n_estimators,
        'max_depth': final_model.max_depth,
        'n_features': len(predictor_cols)
    }
}

bundle_path = "production_model_bundle.pkl"
joblib.dump(production_bundle, bundle_path)
print(f"4. Production bundle saved: {bundle_path}")

# 4. Print deployment instructions
print("\n" + "="*50)
print("DEPLOYMENT INSTRUCTIONS")
print("="*50)

print("""
To deploy this model to production, use the following code:

```python
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

# Load the production bundle
bundle = joblib.load("production_model_bundle.pkl")

# Extract components
model = bundle['model']
scaler = bundle['scaler']
feature_names = bundle['feature_names']
train_stats = bundle['train_statistics']

# Function to validate and preprocess new data
def preprocess_new_data(new_data, feature_names, train_stats):
    '''Validate and preprocess new data for prediction.'''
    # Ensure all required features are present
    missing_features = set(feature_names) - set(new_data.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Select and order features
    X = new_data[feature_names].copy()

    # Check for distribution shift (optional monitoring)
    for col in list(train_stats.keys())[:5]:
        if col in X.columns:
            col_mean = X[col].mean()
            train_mean = train_stats[col]['mean']
            train_std = train_stats[col]['std']
            if abs(col_mean - train_mean) > 3 * train_std:
                print(f"WARNING: Distribution shift detected in {col}")

    return X

# Make predictions
def predict(new_data):
    X = preprocess_new_data(new_data, feature_names, train_stats)
    predictions = model.predict(X)
    return predictions

# Example usage:
# new_data = pd.read_csv("new_districts.csv")
# predictions = predict(new_data)
```
""")

# 5. Verification
print("\n" + "-"*40)
print("VERIFICATION: Loading saved model...")
print("-"*40)

# Reload and verify
final_model_reloaded = joblib.load(model_path)
bundle_reloaded = joblib.load(bundle_path)

print(f"Model type: {type(final_model_reloaded).__name__}")
print(f"Number of trees: {final_model_reloaded.n_estimators}")
print(f"Features expected: {len(bundle_reloaded['feature_names'])}")
print(f"Model metadata: {bundle_reloaded['model_metadata']}")

print("\n Model successfully exported and verified for production!")

# ============================================
# PRODUCTION INFERENCE EXAMPLE
# ============================================

print(f"\n{'='*50}")
print("PRODUCTION INFERENCE EXAMPLE")
print(f"{'='*50}")

# Simulate new data (using test data as example)
new_data = test_df.iloc[:5].copy()

print("\nNew data sample (5 records):")
print(new_data[['Country Name', 'Industry Name', 'Year']].to_string())

# Make predictions
X_new = new_data[predictor_cols]
predictions = final_model_reloaded.predict(X_new)

print("\nPredictions (Average Net Active Export):")
for i, pred in enumerate(predictions):
    print(f"  Record {i+1}: {pred:,.2f}")

# Create results dataframe
results_df = pd.DataFrame({
    'Country': new_data['Country Name'].values,
    'Industry': new_data['Industry Name'].values,
    'Year': new_data['Year'].values,
    'Predicted_Net_Active_Export': predictions
})

print("\nResults DataFrame:")
print(results_df.to_string())

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)



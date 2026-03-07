# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "seaborn",
#     "scikit-learn",
#     "scipy",
#     "shap",
#     "marimo",
#     "joblib",
# ]
# ///

import marimo

__generated_with = "0.19.5"
app = marimo.App(width="medium")


@app.cell
def imports():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import skew, kurtosis, shapiro, spearmanr, ks_2samp
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import RFECV, SelectFromModel
    import shap
    import joblib
    return (
        LinearRegression,
        PCA,
        RFECV,
        RandomForestRegressor,
        RandomizedSearchCV,
        SelectFromModel,
        StandardScaler,
        joblib,
        ks_2samp,
        kurtosis,
        mean_absolute_error,
        mean_squared_error,
        np,
        pd,
        plt,
        r2_score,
        shap,
        shapiro,
        skew,
        sns,
        train_test_split,
    )


@app.cell
def load_dataset(pd):
    def _():
        """Load the raw dataset from CSV."""
        df = pd.read_csv(
            r"C:/Users/srima/OneDrive/Desktop/INSY674/Individual Project 1/dataset.csv",
            encoding='latin-1'
        )

        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"\nFirst few rows:")
        return print(df.head())


    _()
    return


@app.cell
def clean_data(df, np):
    """Clean and reshape the dataset."""
    # Drop unusable columns
    df1 = df.drop(['Country Code', 'Industry Code', 'Series Code'], axis=1)

    # Identify year columns
    key_columns = ['Country Name', 'Series Name', 'Industry Name']
    years = [col for col in df1.columns if col not in key_columns]

    # Melt and pivot to reshape data
    df1_melted = df1.melt(
        id_vars=key_columns,
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

    # Keep only Mean columns
    key_cols = ['Country Name', 'Industry Name', 'Year']
    columns_to_keep = key_cols + [col for col in df1_pivot.columns if 'Mean' in str(col)]
    df1_pivot = df1_pivot[columns_to_keep]

    # Get feature columns
    feature_cols = [col for col in df1_pivot.columns if col not in key_cols]

    # Replace '..' with NaN
    df1_pivot[feature_cols] = df1_pivot[feature_cols].replace('..', np.nan)

    print(f"Cleaned dataset shape: {df1_pivot.shape}")
    print(f"Feature columns: {len(feature_cols)}")
    return df1_pivot, feature_cols


@app.cell
def missing_data_diagnostic(df1_pivot, feature_cols):
    """Analyze missing data patterns."""

    def check_monotone_pattern(df, cols):
        """Check if missing data follows a monotone pattern."""
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

    # Count unique missing patterns
    missing_patterns = df1_pivot[feature_cols].isnull().astype(int).apply(tuple, axis=1)
    unique_patterns = missing_patterns.value_counts()

    print("=" * 50)
    print("MISSING DATA PATTERN DIAGNOSTIC")
    print("=" * 50)
    print(f"\nNumber of unique missing patterns: {len(unique_patterns)}")
    print(f"Top 5 most common patterns:")
    print(unique_patterns.head(5))

    # Check monotonicity
    monotone_pct = check_monotone_pattern(df1_pivot, feature_cols)
    print(f"\n% of rows with monotone pattern: {monotone_pct:.2f}%")

    if monotone_pct > 80:
        pattern_type = "MONOTONE"
    elif monotone_pct < 20:
        pattern_type = "ARBITRARY"
    else:
        pattern_type = "MIXED"

    print(f"-> Pattern is likely {pattern_type}")

    # Per-variable missing percentage
    missing_pct = {col: df1_pivot[col].isnull().sum() / len(df1_pivot) * 100
                   for col in feature_cols}
    return


@app.cell
def split_data(df1_pivot, feature_cols, pd, train_test_split):
    """Split data into train/validation/test sets (80/10/10)."""
    # Calculate missing data ratio for stratification
    df_temp = df1_pivot.copy()
    df_temp['missing_ratio'] = df_temp[feature_cols].isnull().sum(axis=1) / len(feature_cols)
    df_temp['missing_bin'] = pd.cut(df_temp['missing_ratio'], bins=10, labels=False, duplicates='drop')
    df_temp['missing_bin'] = df_temp['missing_bin'].fillna(0).astype(int)

    # First split: 80% train, 20% temp
    train_df, temp_df = train_test_split(
        df_temp, test_size=0.2, random_state=42, stratify=df_temp['missing_bin']
    )

    # Second split: 50/50 of temp -> 10% val, 10% test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['missing_bin']
    )

    # Remove helper columns
    for df in [train_df, val_df, test_df]:
        df.drop(['missing_ratio', 'missing_bin'], axis=1, inplace=True)

    print(f"Train: {len(train_df)} ({len(train_df)/len(df1_pivot)*100:.1f}%)")
    print(f"Validation: {len(val_df)} ({len(val_df)/len(df1_pivot)*100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df)/len(df1_pivot)*100:.1f}%)")
    return df, test_df, train_df, val_df


@app.cell
def distribution_shift_detection(feature_cols, ks_2samp, np, pd, plt, test_df, train_df, val_df):
    """Detect and mitigate distribution shift between train/val/test sets."""
    print("=" * 50)
    print("DISTRIBUTION SHIFT DETECTION & MITIGATION")
    print("=" * 50)

    def detect_shift(train_data, test_data, col_name, alpha=0.05):
        """Use Kolmogorov-Smirnov test to detect distribution shift."""
        train_clean = pd.to_numeric(train_data, errors='coerce').dropna()
        test_clean = pd.to_numeric(test_data, errors='coerce').dropna()

        if len(train_clean) < 5 or len(test_clean) < 5:
            return None, None

        stat, p_value = ks_2samp(train_clean, test_clean)
        return stat, p_value

    # Check distribution shift for each feature
    shift_results = []
    shifted_features = []

    print("\nChecking distribution shift (KS test)...")
    for col in feature_cols:
        if col not in train_df.columns:
            continue
        stat, p_value = detect_shift(train_df[col], test_df[col], col)
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
    print("\n" + "=" * 50)
    print("MITIGATION STRATEGIES")
    print("=" * 50)

    mitigation_applied = []

    # Strategy 1: Identify temporal/domain features for monitoring
    print("\n1. MONITORING FEATURES IDENTIFIED:")
    print("   - Track shifted features during production")
    print("   - Retrain model when shift exceeds threshold")

    # Strategy 2: Importance weighting concept
    print("\n2. SAMPLE REWEIGHTING (Conceptual):")
    print("   - Weight samples by density ratio: P(test)/P(train)")
    print("   - Helps model generalize to test distribution")

    # Strategy 3: Feature standardization check
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

    print(f"\nDistribution shift analysis complete.")
    return shift_df, shifted_features, train_stats


@app.cell
def imputation_helpers(LinearRegression, np, skew):
    """Helper functions for imputation."""

    def check_skewness(series):
        """Return True if distribution is skewed (|skew| > 1)."""
        clean = series.dropna()
        if len(clean) < 3:
            return False
        return abs(skew(clean)) > 1

    def pmm_impute(train_obs, train_miss, predictors_obs, predictors_miss, k=5):
        """Predictive Mean Matching imputation."""
        model = LinearRegression()
        model.fit(predictors_obs, train_obs)

        pred_obs = model.predict(predictors_obs)
        pred_miss = model.predict(predictors_miss)

        imputed_values = []
        for pred in pred_miss:
            distances = np.abs(pred_obs - pred)
            donor_indices = np.argsort(distances)[:k]
            donor_idx = np.random.choice(donor_indices)
            imputed_values.append(train_obs.iloc[donor_idx])

        return np.array(imputed_values)
    return check_skewness, pmm_impute


@app.cell
def perform_imputation(
    LinearRegression,
    check_skewness,
    feature_cols,
    np,
    pd,
    pmm_impute,
    test_df,
    train_df,
    val_df,
):
    """Perform adaptive imputation on all datasets."""
    # Make copies to avoid modifying originals
    train_imputed = train_df.copy()
    val_imputed = val_df.copy()
    test_imputed = test_df.copy()

    # Convert to numeric
    for df in [train_imputed, val_imputed, test_imputed]:
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sort columns by missingness
    missing_counts = train_imputed[feature_cols].isnull().sum()
    sorted_cols = missing_counts.sort_values().index.tolist()

    np.random.seed(42)

    print("=" * 50)
    print("IMPUTATION PROCESS")
    print("=" * 50)

    for col in sorted_cols:
        if train_imputed[col].isnull().sum() == 0:
            continue

        is_skewed = check_skewness(train_imputed[col])
        method = "PMM" if is_skewed else "Linear Regression"
        prev_cols = sorted_cols[:sorted_cols.index(col)]

        if len(prev_cols) == 0:
            fill_val = train_imputed[col].median() if is_skewed else train_imputed[col].mean()
            for df in [train_imputed, val_imputed, test_imputed]:
                df[col] = df[col].fillna(fill_val)
            print(f"  {col[:40]}...: {method} (mean/median)")
        else:
            train_complete = train_imputed[prev_cols + [col]].dropna()

            if len(train_complete) < 10:
                fill_val = train_imputed[col].median() if is_skewed else train_imputed[col].mean()
                for df in [train_imputed, val_imputed, test_imputed]:
                    df[col] = df[col].fillna(fill_val)
                continue

            if is_skewed:
                for df in [train_imputed, val_imputed, test_imputed]:
                    mask = df[col].isnull()
                    if mask.sum() > 0:
                        miss_predictors = df.loc[mask, prev_cols].fillna(df[prev_cols].mean())
                        imputed = pmm_impute(
                            train_complete[col], df.loc[mask, col],
                            train_complete[prev_cols], miss_predictors, k=5
                        )
                        df.loc[mask, col] = imputed
            else:
                model = LinearRegression()
                model.fit(train_complete[prev_cols], train_complete[col])
                for df in [train_imputed, val_imputed, test_imputed]:
                    mask = df[col].isnull()
                    if mask.sum() > 0:
                        predictors = df.loc[mask, prev_cols].fillna(df[prev_cols].mean())
                        df.loc[mask, col] = model.predict(predictors)

            print(f"  {col[:40]}...: {method}")

    print(f"\nImputation complete!")
    print(f"Train missing after: {train_imputed[feature_cols].isnull().sum().sum()}")
    return df, test_imputed, train_imputed, val_imputed


@app.cell
def create_target(feature_cols, test_imputed, train_imputed, val_imputed):
    """Create the target variable: Net Active Export."""
    exporter_col = '006.Export Value per Exporter: Mean'
    exiter_col = '016.Export Value per Exiter: Mean'
    target_col = 'Average_Net_Active_Export'

    # Create target variable
    for df in [train_imputed, val_imputed, test_imputed]:
        df[target_col] = df[exporter_col] - df[exiter_col]

    # Define predictor columns (exclude target components)
    predictor_cols = [col for col in feature_cols
                      if col not in [exporter_col, exiter_col, target_col]]

    print("=" * 50)
    print(f"TARGET VARIABLE: {target_col}")
    print("=" * 50)
    print(f"Formula: {exporter_col} - {exiter_col}")
    print(f"\nTrain set summary:")
    print(f"  Mean: {train_imputed[target_col].mean():.2f}")
    print(f"  Median: {train_imputed[target_col].median():.2f}")
    print(f"  Std: {train_imputed[target_col].std():.2f}")
    print(f"  % Positive: {(train_imputed[target_col] > 0).mean()*100:.1f}%")
    print(f"\nPredictor columns: {len(predictor_cols)}")
    return df, predictor_cols, target_col


@app.cell
def analyze_target(kurtosis, plt, shapiro, skew, target_col, train_imputed):
    """Analyze target variable distribution."""
    y = train_imputed[target_col].dropna()

    print("=" * 50)
    print(f"TARGET VARIABLE ANALYSIS")
    print("=" * 50)

    skewness = skew(y)
    kurt = kurtosis(y)

    print(f"Mean: {y.mean():.2f}")
    print(f"Median: {y.median():.2f}")
    print(f"Std Dev: {y.std():.2f}")
    print(f"Skewness: {skewness:.2f}")
    print(f"Kurtosis: {kurt:.2f}")

    # Normality test
    sample = y.sample(min(5000, len(y)), random_state=42)
    stat, p_value = shapiro(sample)
    print(f"Shapiro-Wilk p-value: {p_value:.4f} ({'NOT normal' if p_value < 0.05 else 'normal'})")

    # Outliers
    Q1, Q3 = y.quantile(0.25), y.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((y < Q1 - 1.5*IQR) | (y > Q3 + 1.5*IQR)).sum()
    outliers_pct = outliers / len(y) * 100
    print(f"Outliers: {outliers} ({outliers_pct:.1f}%)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(y, bins=50, edgecolor='black')
    axes[0].set_title(f'Distribution of {target_col}')
    axes[1].boxplot(y)
    axes[1].set_title(f'Boxplot of {target_col}')
    plt.tight_layout()

    target_analysis = {'skewness': skewness, 'p_value': p_value, 'outliers_pct': outliers_pct}
    return (target_analysis,)


@app.cell
def analyze_correlations(predictor_cols, target_col, train_imputed):
    """Analyze correlations between features and target."""
    correlations = train_imputed[predictor_cols].corrwith(
        train_imputed[target_col]
    ).sort_values(key=abs, ascending=False)

    print("=" * 50)
    print("FEATURE-TARGET CORRELATIONS")
    print("=" * 50)
    print("\nTop 10 correlations:")
    for feat, corr in correlations.head(10).items():
        strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
        print(f"  {feat[:45]}: {corr:+.3f} ({strength})")
    return


@app.cell
def check_multicollinearity(plt, predictor_cols, sns, train_imputed):
    """Check for multicollinearity among features."""
    corr_matrix = train_imputed[predictor_cols].corr()

    # Find highly correlated pairs
    high_corr_pairs = []
    threshold = 0.8
    for i, col1 in enumerate(predictor_cols):
        for col2 in predictor_cols[i+1:]:
            corr = corr_matrix.loc[col1, col2]
            if abs(corr) > threshold:
                high_corr_pairs.append((col1, col2, corr))

    print("=" * 50)
    print("MULTICOLLINEARITY CHECK")
    print("=" * 50)

    if high_corr_pairs:
        print(f"\nHighly correlated pairs (|r| > {threshold}): {len(high_corr_pairs)}")
        for col1, col2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]:
            print(f"  {col1[:30]} <-> {col2[:30]}: {corr:.3f}")
    else:
        print(f"\nNo pairs with |correlation| > {threshold}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=ax)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    return (high_corr_pairs,)


@app.cell
def analyze_distributions(predictor_cols, skew, train_imputed):
    """Analyze distributions of all features."""
    results = {}
    skewed_features = []

    for col in predictor_cols:
        data = train_imputed[col].dropna()
        sk = skew(data)
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        outlier_pct = ((data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)).sum() / len(data) * 100

        results[col] = {'skewness': sk, 'outlier_pct': outlier_pct}
        if abs(sk) > 1:
            skewed_features.append((col, sk))

    print("=" * 50)
    print("FEATURE DISTRIBUTIONS")
    print("=" * 50)
    print(f"\nSkewed features (|skew| > 1): {len(skewed_features)}/{len(predictor_cols)}")

    for col, sk in sorted(skewed_features, key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {col[:45]}: skew={sk:.2f}")
    return


@app.cell
def model_recommendations(
    high_corr_pairs,
    predictor_cols,
    target_analysis,
    train_imputed,
):
    """Provide model recommendations based on EDA."""
    print("=" * 50)
    print("MODEL RECOMMENDATIONS")
    print("=" * 50)

    recommendations = []

    if abs(target_analysis['skewness']) > 1:
        recommendations.append("- Target is skewed: Consider tree-based models")
    if target_analysis['outliers_pct'] > 5:
        recommendations.append("- Many outliers: Tree-based models are more robust")
    if len(high_corr_pairs) > 3:
        recommendations.append("- High multicollinearity: Use Ridge/Lasso or trees")

    n_samples = len(train_imputed)
    n_features = len(predictor_cols)
    if n_samples / n_features < 10:
        recommendations.append("- Low n/p ratio: Use regularization")

    print("\nBased on your data:")
    for rec in recommendations:
        print(rec)

    print("\nSuggested models:")
    print("  1. Random Forest (robust baseline)")
    print("  2. XGBoost/LightGBM (best performance)")
    print("  3. Ridge/Lasso (interpretability)")
    return


@app.cell
def remove_redundant_features(predictor_cols, target_col, train_imputed):
    """Remove redundant inverse-ratio features."""
    inverse_pairs = [
        ('045.Number of HS6 Products per Exporter: Mean',
         '051.Number of Exporters per HS6 Product: Mean'),
        ('048.Number of Destinations per Exporter: Mean',
         '054.Number of Exporters per Destination: Mean')
    ]

    cols_to_drop = []

    print("=" * 50)
    print("REMOVING REDUNDANT FEATURES")
    print("=" * 50)

    for col1, col2 in inverse_pairs:
        if col1 in predictor_cols and col2 in predictor_cols:
            corr1 = abs(train_imputed[col1].corr(train_imputed[target_col]))
            corr2 = abs(train_imputed[col2].corr(train_imputed[target_col]))

            if corr1 >= corr2:
                cols_to_drop.append(col2)
                print(f"Dropping: {col2[:40]}... (|r|={corr2:.3f})")
            else:
                cols_to_drop.append(col1)
                print(f"Dropping: {col1[:40]}... (|r|={corr1:.3f})")

    predictor_cols_clean = [c for c in predictor_cols if c not in cols_to_drop]

    print(f"\nFeatures: {len(predictor_cols)} -> {len(predictor_cols_clean)}")
    return (predictor_cols_clean,)


@app.cell
def prepare_model_data(
    predictor_cols_clean,
    target_col,
    test_imputed,
    train_imputed,
    val_imputed,
):
    """Prepare X and y for modeling."""
    X_train = train_imputed[predictor_cols_clean]
    y_train = train_imputed[target_col]
    X_val = val_imputed[predictor_cols_clean]
    y_val = val_imputed[target_col]
    X_test = test_imputed[predictor_cols_clean]
    y_test = test_imputed[target_col]

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    return X_test, X_train, X_val, y_test, y_train, y_val


@app.cell
def feature_selection_techniques(
    PCA,
    RFECV,
    RandomForestRegressor,
    SelectFromModel,
    StandardScaler,
    X_train,
    X_val,
    X_test,
    np,
    pd,
    plt,
    predictor_cols_clean,
    y_train,
):
    """Dimensionality Reduction and Feature Subset Selection Techniques."""
    print("=" * 50)
    print("DIMENSIONALITY REDUCTION & FEATURE SELECTION")
    print("=" * 50)

    # Store results for comparison
    fs_results = {}

    # =========================================
    # 1. TREE-BASED FEATURE SELECTION
    # =========================================
    print("\n" + "-" * 40)
    print("1. TREE-BASED FEATURE SELECTION")
    print("-" * 40)

    # Fit a Random Forest for feature importance
    rf_selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X_train, y_train)

    # Get feature importances
    importances = rf_selector.feature_importances_
    importance_df = pd.DataFrame({
        'feature': predictor_cols_clean,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Select features with importance > mean importance
    threshold = np.mean(importances)
    tree_selected_features = importance_df[importance_df['importance'] > threshold]['feature'].tolist()

    print(f"Threshold (mean importance): {threshold:.4f}")
    print(f"Features selected: {len(tree_selected_features)}/{len(predictor_cols_clean)}")
    print("\nTop 10 features by tree importance:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature'][:45]}: {row['importance']:.4f}")

    # Using SelectFromModel for formal selection
    sfm = SelectFromModel(rf_selector, threshold='mean', prefit=True)
    X_train_tree = sfm.transform(X_train)
    X_val_tree = sfm.transform(X_val)
    X_test_tree = sfm.transform(X_test)

    fs_results['tree_based'] = {
        'n_features': X_train_tree.shape[1],
        'features': tree_selected_features,
        'X_train': X_train_tree,
        'X_val': X_val_tree,
        'X_test': X_test_tree
    }

    # =========================================
    # 2. RECURSIVE FEATURE ELIMINATION (RFECV)
    # =========================================
    print("\n" + "-" * 40)
    print("2. RECURSIVE FEATURE ELIMINATION (RFECV)")
    print("-" * 40)

    # Use a smaller estimator for RFECV (faster)
    rf_for_rfe = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

    # RFECV with cross-validation
    rfecv = RFECV(
        estimator=rf_for_rfe,
        step=1,
        cv=3,
        scoring='neg_mean_squared_error',
        min_features_to_select=5,
        n_jobs=-1
    )

    print("Running RFECV (this may take a moment)...")
    rfecv.fit(X_train, y_train)

    rfe_selected_features = [f for f, s in zip(predictor_cols_clean, rfecv.support_) if s]

    print(f"Optimal number of features: {rfecv.n_features_}")
    print(f"Features selected: {len(rfe_selected_features)}/{len(predictor_cols_clean)}")

    X_train_rfe = rfecv.transform(X_train)
    X_val_rfe = rfecv.transform(X_val)
    X_test_rfe = rfecv.transform(X_test)

    fs_results['rfecv'] = {
        'n_features': rfecv.n_features_,
        'features': rfe_selected_features,
        'X_train': X_train_rfe,
        'X_val': X_val_rfe,
        'X_test': X_test_rfe,
        'cv_results': rfecv.cv_results_
    }

    # =========================================
    # 3. PCA (PRINCIPAL COMPONENT ANALYSIS)
    # =========================================
    print("\n" + "-" * 40)
    print("3. PCA (PRINCIPAL COMPONENT ANALYSIS)")
    print("-" * 40)

    # Standardize features before PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Fit PCA to find optimal components (95% variance)
    pca_full = PCA(random_state=42)
    pca_full.fit(X_train_scaled)

    # Calculate cumulative variance
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

    print(f"Components for 95% variance: {n_components_95}/{len(predictor_cols_clean)}")
    print(f"Variance explained by first 5 components: {cumulative_variance[4]*100:.1f}%")
    print(f"Variance explained by first 10 components: {cumulative_variance[min(9, len(cumulative_variance)-1)]*100:.1f}%")

    # Apply PCA with optimal components
    pca = PCA(n_components=n_components_95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    fs_results['pca'] = {
        'n_features': n_components_95,
        'explained_variance': cumulative_variance[n_components_95-1],
        'X_train': X_train_pca,
        'X_val': X_val_pca,
        'X_test': X_test_pca,
        'pca_model': pca,
        'scaler': scaler
    }

    # =========================================
    # 4. CORRELATION-BASED SELECTION
    # =========================================
    print("\n" + "-" * 40)
    print("4. CORRELATION-BASED FEATURE SELECTION")
    print("-" * 40)

    # Select features with |correlation| > threshold with target
    correlations = X_train.corrwith(y_train).abs()
    corr_threshold = 0.1
    corr_selected_features = correlations[correlations > corr_threshold].index.tolist()

    print(f"Correlation threshold: {corr_threshold}")
    print(f"Features selected: {len(corr_selected_features)}/{len(predictor_cols_clean)}")

    X_train_corr = X_train[corr_selected_features]
    X_val_corr = X_val[corr_selected_features]
    X_test_corr = X_test[corr_selected_features]

    fs_results['correlation'] = {
        'n_features': len(corr_selected_features),
        'features': corr_selected_features,
        'X_train': X_train_corr,
        'X_val': X_val_corr,
        'X_test': X_test_corr
    }

    # =========================================
    # COMPARISON SUMMARY
    # =========================================
    print("\n" + "=" * 50)
    print("FEATURE SELECTION SUMMARY")
    print("=" * 50)

    print(f"\n{'Method':<25} {'Features Selected':<20} {'Reduction %':<15}")
    print("-" * 60)
    original_n = len(predictor_cols_clean)
    for method, results in fs_results.items():
        n_feat = results['n_features']
        reduction = (1 - n_feat / original_n) * 100
        print(f"{method:<25} {n_feat:<20} {reduction:.1f}%")

    # =========================================
    # VISUALIZATION
    # =========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Tree-based feature importance
    top_n = 15
    axes[0, 0].barh(range(top_n), importance_df.head(top_n)['importance'].values)
    axes[0, 0].set_yticks(range(top_n))
    axes[0, 0].set_yticklabels([f[:25] for f in importance_df.head(top_n)['feature'].values])
    axes[0, 0].set_xlabel('Importance')
    axes[0, 0].set_title('Tree-Based Feature Importance (Top 15)')
    axes[0, 0].invert_yaxis()

    # Plot 2: RFECV cross-validation scores
    cv_results = fs_results['rfecv']['cv_results']
    n_features_range = range(1, len(cv_results['mean_test_score']) + 1)
    axes[0, 1].plot(n_features_range, -np.array(cv_results['mean_test_score']), 'b-')
    axes[0, 1].fill_between(
        n_features_range,
        -np.array(cv_results['mean_test_score']) - np.array(cv_results['std_test_score']),
        -np.array(cv_results['mean_test_score']) + np.array(cv_results['std_test_score']),
        alpha=0.2
    )
    axes[0, 1].axvline(x=rfecv.n_features_, color='r', linestyle='--', label=f'Optimal: {rfecv.n_features_}')
    axes[0, 1].set_xlabel('Number of Features')
    axes[0, 1].set_ylabel('MSE (CV)')
    axes[0, 1].set_title('RFECV: Cross-Validation Score vs Features')
    axes[0, 1].legend()

    # Plot 3: PCA cumulative variance
    axes[1, 0].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-o', markersize=3)
    axes[1, 0].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    axes[1, 0].axvline(x=n_components_95, color='g', linestyle='--', label=f'n={n_components_95}')
    axes[1, 0].set_xlabel('Number of Components')
    axes[1, 0].set_ylabel('Cumulative Explained Variance')
    axes[1, 0].set_title('PCA: Cumulative Explained Variance')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Comparison bar chart
    methods = list(fs_results.keys())
    n_features_list = [fs_results[m]['n_features'] for m in methods]
    colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple']
    axes[1, 1].bar(methods, n_features_list, color=colors)
    axes[1, 1].axhline(y=original_n, color='r', linestyle='--', label=f'Original: {original_n}')
    axes[1, 1].set_ylabel('Number of Features')
    axes[1, 1].set_title('Feature Selection Methods Comparison')
    axes[1, 1].legend()

    plt.tight_layout()

    # Determine best method (use tree-based as default recommendation)
    print("\nRECOMMENDATION: Using Tree-Based selection for model training")
    print("(Preserves interpretability while reducing dimensionality)")

    selected_features_final = tree_selected_features
    return fs_results, importance_df, pca, rfecv, scaler, selected_features_final, sfm


@app.cell
def hyperparameter_tuning(
    RandomForestRegressor,
    RandomizedSearchCV,
    X_train,
    np,
    y_train,
):
    """Stage 1: Broad hyperparameter search."""
    rf_param_distributions = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [None, 5, 10, 15, 20, 30],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
        'bootstrap': [True, False]
    }

    rf = RandomForestRegressor(random_state=42)

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_param_distributions,
        n_iter=20,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    print("=" * 50)
    print("HYPERPARAMETER TUNING (Stage 1)")
    print("=" * 50)
    print("\nRunning RandomizedSearchCV...")

    random_search.fit(X_train, y_train)

    print(f"\nBest parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Best CV RMSE: {np.sqrt(-random_search.best_score_):.4f}")
    return (random_search,)


@app.cell
def initial_evaluation(
    X_train,
    X_val,
    mean_absolute_error,
    mean_squared_error,
    np,
    r2_score,
    random_search,
    y_train,
    y_val,
):
    """Evaluate model on train and validation sets."""

    def evaluate_model(y_true, y_pred, set_name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"\n{set_name} Set:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R2:   {r2:.4f}")
        return {'rmse': rmse, 'mae': mae, 'r2': r2}

    best_rf = random_search.best_estimator_

    print("=" * 50)
    print("INITIAL MODEL EVALUATION")
    print("=" * 50)

    train_metrics = evaluate_model(y_train, best_rf.predict(X_train), "Training")
    val_metrics = evaluate_model(y_val, best_rf.predict(X_val), "Validation")
    return (evaluate_model,)


@app.cell
def fine_tuning(
    RandomForestRegressor,
    RandomizedSearchCV,
    X_train,
    X_val,
    np,
    pd,
    random_search,
    y_train,
    y_val,
):
    """Stage 2: Fine-tune on combined train+val."""
    # Combine train and validation
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

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

    print("=" * 50)
    print("FINE-TUNING (Stage 2)")
    print("=" * 50)
    print("\nRunning fine-tuning search...")

    fine_tune_search.fit(X_train_val, y_train_val)

    print(f"\nBest parameters after fine-tuning:")
    for param, value in fine_tune_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Best CV RMSE: {np.sqrt(-fine_tune_search.best_score_):.4f}")

    final_model = fine_tune_search.best_estimator_
    return (final_model,)


@app.cell
def test_evaluation(X_test, evaluate_model, final_model, y_test):
    """Final evaluation on test set."""
    print("=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)

    y_test_pred = final_model.predict(X_test)
    test_metrics = evaluate_model(y_test, y_test_pred, "Test")
    return


@app.cell
def feature_importance(final_model, pd, plt, predictor_cols_clean):
    """Extract and visualize feature importance."""
    feature_importance_df = pd.DataFrame({
        'feature': predictor_cols_clean,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("=" * 50)
    print("FEATURE IMPORTANCE")
    print("=" * 50)
    print("\nTop 10 Most Important Features:")

    for i, row in feature_importance_df.head(10).iterrows():
        print(f"  {row['feature'][:45]}: {row['importance']:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = feature_importance_df.head(15)
    ax.barh(range(len(top_features)), top_features['importance'].values)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels([f[:35] for f in top_features['feature'].values])
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Feature Importances (Random Forest)')
    ax.invert_yaxis()
    plt.tight_layout()
    return


@app.cell
def shap_analysis(
    X_test,
    final_model,
    np,
    pd,
    plt,
    predictor_cols_clean,
    shap,
):
    """SHAP analysis for model interpretability."""
    print("=" * 50)
    print("SHAP ANALYSIS")
    print("=" * 50)

    # Sample for faster computation
    sample_size = min(500, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)

    print(f"Computing SHAP values for {sample_size} samples...")

    # Create explainer and compute SHAP values
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_sample)

    # SHAP importance
    shap_importance_df = pd.DataFrame({
        'feature': predictor_cols_clean,
        'shap_importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('shap_importance', ascending=False)

    print("\nTop 10 Features by SHAP Importance:")
    for i, row in shap_importance_df.head(10).iterrows():
        print(f"  {row['feature'][:45]}: {row['shap_importance']:.4f}")

    # Summary plot (bar)
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()

    # Summary plot (beeswarm)
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.tight_layout()

    return X_sample, explainer, shap_importance_df, shap_values


@app.cell
def model_output_visualization(
    X_test,
    final_model,
    np,
    pd,
    plt,
    predictor_cols_clean,
    shap,
    y_test,
):
    """Visualize model output, decision making information, and explanations."""
    print("=" * 50)
    print("MODEL OUTPUT VISUALIZATION & EXPLANATIONS")
    print("=" * 50)

    # Get predictions
    y_pred = final_model.predict(X_test)

    # =========================================
    # 1. PREDICTED VS ACTUAL PLOT
    # =========================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Scatter plot: Predicted vs Actual
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5, s=20)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Predicted vs Actual Values')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # =========================================
    # 2. RESIDUAL ANALYSIS
    # =========================================
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

    # =========================================
    # 3. PREDICTION DISTRIBUTION
    # =========================================
    axes[1, 0].hist(y_test, bins=50, alpha=0.5, label='Actual', edgecolor='black')
    axes[1, 0].hist(y_pred, bins=50, alpha=0.5, label='Predicted', edgecolor='black')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution: Actual vs Predicted')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # =========================================
    # 4. ERROR BY MAGNITUDE
    # =========================================
    # Bin predictions and show error by bin
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

    # =========================================
    # 5. CUMULATIVE ERROR PLOT
    # =========================================
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

    # =========================================
    # DECISION MAKING INFORMATION
    # =========================================
    print("\n" + "-" * 40)
    print("DECISION MAKING METRICS")
    print("-" * 40)

    # Calculate key decision metrics
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

    return residuals, y_pred


@app.cell
def individual_predictions_explanation(
    X_test,
    final_model,
    np,
    pd,
    plt,
    predictor_cols_clean,
    shap,
):
    """Explain individual predictions using SHAP waterfall and force plots."""
    print("=" * 50)
    print("INDIVIDUAL PREDICTION EXPLANATIONS")
    print("=" * 50)

    # Create explainer
    explainer = shap.TreeExplainer(final_model)

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
            'feature': predictor_cols_clean,
            'shap_value': shap_vals,
            'feature_value': X_explain.iloc[idx].values
        }).sort_values('shap_value', key=abs, ascending=False)

        print("Top 5 contributing features:")
        for _, row in feature_contributions.head(5).iterrows():
            direction = "↑" if row['shap_value'] > 0 else "↓"
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
            feature_names=predictor_cols_clean
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
    axes[1].set_yticklabels([predictor_cols_clean[i][:30] for i in top_indices])
    axes[1].set_xlabel('Mean |SHAP Value|')
    axes[1].set_title('Top 10 Features (Sample Explanations)')
    axes[1].invert_yaxis()

    plt.tight_layout()

    print("\n" + "=" * 50)
    print("MODEL INTERPRETATION GUIDELINES")
    print("=" * 50)
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

    return


@app.cell
def _():
    #Interpretation
    return


@app.cell
def export_model_for_production(
    final_model,
    joblib,
    np,
    pd,
    predictor_cols_clean,
    scaler,
    fs_results,
    train_stats,
):
    """Export model and artifacts for production deployment."""
    print("=" * 50)
    print("MODEL EXPORT FOR PRODUCTION DEPLOYMENT")
    print("=" * 50)

    # =========================================
    # 1. SAVE THE TRAINED MODEL
    # =========================================
    model_path = "my_california_housing_model.pkl"
    joblib.dump(final_model, model_path)
    print(f"\n1. Model saved: {model_path}")

    # =========================================
    # 2. SAVE PREPROCESSING ARTIFACTS
    # =========================================
    # Save scaler for consistent preprocessing
    scaler_path = "feature_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"2. Scaler saved: {scaler_path}")

    # Save feature names for validation
    feature_config = {
        'feature_names': predictor_cols_clean,
        'n_features': len(predictor_cols_clean),
        'train_stats': train_stats
    }
    feature_config_path = "feature_config.pkl"
    joblib.dump(feature_config, feature_config_path)
    print(f"3. Feature config saved: {feature_config_path}")

    # =========================================
    # 3. CREATE PRODUCTION ARTIFACTS BUNDLE
    # =========================================
    production_bundle = {
        'model': final_model,
        'scaler': scaler,
        'feature_names': predictor_cols_clean,
        'feature_selection_results': {
            method: {
                'n_features': results['n_features'],
                'features': results.get('features', None)
            }
            for method, results in fs_results.items()
            if method != 'pca'  # PCA features are components, not original features
        },
        'train_statistics': train_stats,
        'model_metadata': {
            'model_type': 'RandomForestRegressor',
            'n_estimators': final_model.n_estimators,
            'max_depth': final_model.max_depth,
            'n_features': len(predictor_cols_clean)
        }
    }

    bundle_path = "production_model_bundle.pkl"
    joblib.dump(production_bundle, bundle_path)
    print(f"4. Production bundle saved: {bundle_path}")

    # =========================================
    # 4. PRINT DEPLOYMENT INSTRUCTIONS
    # =========================================
    print("\n" + "=" * 50)
    print("DEPLOYMENT INSTRUCTIONS")
    print("=" * 50)

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

    # =========================================
    # 5. VERIFICATION
    # =========================================
    print("\n" + "-" * 40)
    print("VERIFICATION: Loading saved model...")
    print("-" * 40)

    # Reload and verify
    final_model_reloaded = joblib.load(model_path)
    bundle_reloaded = joblib.load(bundle_path)

    print(f"Model type: {type(final_model_reloaded).__name__}")
    print(f"Number of trees: {final_model_reloaded.n_estimators}")
    print(f"Features expected: {len(bundle_reloaded['feature_names'])}")
    print(f"Model metadata: {bundle_reloaded['model_metadata']}")

    print("\n✓ Model successfully exported and verified for production!")

    return bundle_path, feature_config_path, model_path, production_bundle, scaler_path


@app.cell
def production_inference_example(
    final_model,
    joblib,
    pd,
    predictor_cols_clean,
    test_imputed,
):
    """Demonstrate production inference with the saved model."""
    print("=" * 50)
    print("PRODUCTION INFERENCE EXAMPLE")
    print("=" * 50)

    # Reload the model (simulating production environment)
    final_model_reloaded = joblib.load("my_california_housing_model.pkl")

    # Simulate new data (using test data as example)
    new_data = test_imputed.iloc[:5].copy()

    print("\nNew data sample (5 records):")
    print(new_data[['Country Name', 'Industry Name', 'Year']].to_string())

    # Make predictions
    X_new = new_data[predictor_cols_clean]
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

    return predictions, results_df


if __name__ == "__main__":
    app.run()

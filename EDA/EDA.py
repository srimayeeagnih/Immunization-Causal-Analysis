import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = Path(r"C:\Users\srima\OneDrive\Desktop\INSY674\Individual Project 1\analysis_dataset_non_gavi.csv")
FIG_DIR   = DATA_PATH.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


# ── OLD PIPELINE (commented out — see pivot EDA below) ──────────────────────
# # ── Load ──────────────────────────────────────────────────────────────────────
# df = pd.read_csv(DATA_PATH)
# print(f"Loaded  : {df.shape[0]:,} rows × {df.shape[1]} columns")
# print(f"Columns : {df.columns.tolist()}\n")
# 
# # ════════════════════════════════════════════════════════════════════════════
# # 1. Total Missing Values
# # ════════════════════════════════════════════════════════════════════════════
# print("=" * 60)
# print("1. TOTAL MISSING VALUES")
# print("=" * 60)
# total_missing = df.isna().sum().sum()
# total_cells   = df.size
# print(f"  Missing cells : {total_missing:,} / {total_cells:,}  ({total_missing / total_cells:.2%})")
# 
# # ════════════════════════════════════════════════════════════════════════════
# # 2. Missing Values per Column
# # ════════════════════════════════════════════════════════════════════════════
# print("\n" + "=" * 60)
# print("2. MISSING VALUES PER COLUMN")
# print("=" * 60)
# miss_col = df.isna().sum().rename("count")
# miss_pct  = (miss_col / len(df) * 100).round(2).rename("pct_%")
# miss_summary = pd.concat([miss_col, miss_pct], axis=1).sort_values("count", ascending=False)
# print(miss_summary.to_string())
# 
# # ════════════════════════════════════════════════════════════════════════════
# # 3. Missing Values per Country
# # ════════════════════════════════════════════════════════════════════════════
# print("\n" + "=" * 60)
# print("3. MISSING VALUES PER COUNTRY")
# print("=" * 60)
# rows_per_country = df.groupby("country_iso3").size().rename("n_rows")
# miss_per_country = (
#     df.groupby("country_iso3")
#     .apply(lambda g: g.isna().sum().sum(), include_groups=False)
#     .rename("missing_cells")
# )
# miss_country_pct = (
#     miss_per_country / (rows_per_country * df.shape[1]) * 100
# ).round(2).rename("missing_pct_%")
# country_summary = (
#     pd.concat([rows_per_country, miss_per_country, miss_country_pct], axis=1)
#     .sort_values("missing_cells", ascending=False)
# )
# has_missing = country_summary[country_summary["missing_cells"] > 0]
# print(f"  {len(has_missing)} / {len(country_summary)} countries have at least one missing cell\n")
# print(has_missing.to_string())
# 
# # ════════════════════════════════════════════════════════════════════════════
# # 4. MAR / MNAR Tests
# # ════════════════════════════════════════════════════════════════════════════
# print("\n" + "=" * 60)
# print("4. MAR / MNAR TESTS")
# print("=" * 60)
# print(
#     "  Method: point-biserial correlation between each column's missingness\n"
#     "  indicator (1=missing) and all other observed numeric variables.\n"
#     "  MAR  -> missingness is significantly predicted by observed variables (p<0.05).\n"
#     "  MNAR -> cannot be verified statistically; flagged when MAR evidence is weak\n"
#     "         but missingness rate is high and domain knowledge suggests self-selection.\n"
#     "  MCAR -> no observed variable predicts missingness (inconclusive baseline).\n"
# )
# 
# numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
# cols_with_miss   = [c for c in numeric_cols if df[c].isna().any()]
# 
# for target in cols_with_miss:
#     indicator   = df[target].isna().astype(int)
#     miss_rate   = indicator.mean()
#     predictors  = [c for c in numeric_cols if c != target and df[c].notna().sum() >= 30]
#     sig_preds   = []
# 
#     for pred in predictors:
#         mask = df[pred].notna()
#         if mask.sum() < 10:
#             continue
#         r, p = stats.pointbiserialr(indicator[mask], df[pred][mask])
#         if p < 0.05:
#             sig_preds.append((pred, r, p))
# 
#     if sig_preds:
#         verdict = "MAR"
#         note    = f"Missingness predicted by {len(sig_preds)} variable(s)"
#     elif miss_rate > 0.30:
#         verdict = "Likely MNAR"
#         note    = f"High miss rate ({miss_rate:.0%}) with no observed predictor — self-selection plausible"
#     else:
#         verdict = "MCAR / Inconclusive"
#         note    = "No observed variable predicts missingness"
# 
#     print(f"  Column : {target}  (missing: {miss_rate:.1%})")
#     print(f"  Verdict: {verdict}")
#     print(f"  Note   : {note}")
#     if sig_preds:
#         for pred, r, p in sorted(sig_preds, key=lambda x: x[2]):
#             print(f"           {pred:<30s}  r={r:+.3f}  p={p:.4f}")
#     print()
# 
# # ════════════════════════════════════════════════════════════════════════════
# # 5. Drop 'gavi_eligible'
# # ════════════════════════════════════════════════════════════════════════════
# print("=" * 60)
# print("5. DROPPING 'gavi_eligible'")
# print("=" * 60)
# if "gavi_eligible" in df.columns:
#     df = df.drop(columns=["gavi_eligible"])
#     print("  Dropped: gavi_eligible")
# else:
#     print("  Column not present — skipping")
# 
# # keep a copy with identifiers for plots
# df_plot = df.copy()
# 
# # ════════════════════════════════════════════════════════════════════════════
# # 6. Drop ISO3 Columns
# # ════════════════════════════════════════════════════════════════════════════
# print("\n" + "=" * 60)
# print("6. DROPPING ISO3 COLUMNS")
# print("=" * 60)
# iso3_cols = [c for c in df.columns if "iso3" in c.lower()]
# df = df.drop(columns=iso3_cols)
# print(f"  Dropped: {iso3_cols}")
# print(f"  Remaining columns: {df.columns.tolist()}")
# 
# # ════════════════════════════════════════════════════════════════════════════
# # 7. Spearman Correlation
# # ════════════════════════════════════════════════════════════════════════════
# print("\n" + "=" * 60)
# print("7. SPEARMAN CORRELATION BETWEEN COVARIATES")
# print("=" * 60)
# covariate_cols = [c for c in [
#     "immunization_coverage", "pharma_tariff_rate",
#     "gdp_per_capita_usd", "health_exp_pct_gdp",
#     "population_total",
#     # gni_per_capita_usd excluded: used as GAVI eligibility proxy;
#     # already filtered to non-GAVI sample, retaining it would be redundant
#     # and collinear with gdp_per_capita_usd (rho=0.993)
# ] if c in df.columns]
# 
# spearman_matrix = df[covariate_cols].corr(method="spearman")
# print(spearman_matrix.round(3).to_string())
# 
# # ════════════════════════════════════════════════════════════════════════════
# # 8. Eta-Squared (country groups -> immunization coverage)
# # ════════════════════════════════════════════════════════════════════════════
# print("\n" + "=" * 60)
# print("8. ETA-SQUARED  (country grouping -> immunization coverage)")
# print("=" * 60)
# print("  One-way ANOVA: immunization_coverage ~ country_iso3\n")
# 
# groups = [
#     g["immunization_coverage"].dropna().values
#     for _, g in df_plot.groupby("country_iso3")
#     if g["immunization_coverage"].dropna().shape[0] > 1
# ]
# 
# if groups:
#     f_stat, p_val  = stats.f_oneway(*groups)
#     valid_coverage = df_plot["immunization_coverage"].dropna()
#     grand_mean     = valid_coverage.mean()
#     ss_total       = ((valid_coverage - grand_mean) ** 2).sum()
#     ss_between     = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
#     eta_sq         = ss_between / ss_total
# 
#     if eta_sq < 0.01:
#         effect = "negligible"
#     elif eta_sq < 0.06:
#         effect = "small"
#     elif eta_sq < 0.14:
#         effect = "medium"
#     else:
#         effect = "large"
# 
#     print(f"  F-statistic : {f_stat:.4f}")
#     print(f"  p-value     : {p_val:.4e}")
#     print(f"  Eta-squared : {eta_sq:.4f}  ({effect} effect)")
#     print(f"  Interpretation: {eta_sq:.1%} of variance in immunization coverage is")
#     print(f"                  explained by country membership.")
# 
# # ════════════════════════════════════════════════════════════════════════════
# # PLOTS
# # ════════════════════════════════════════════════════════════════════════════
# 
# # ── Plot 1: Immunization Coverage by Vaccine Type ────────────────────────────
# if "antigen_family" in df_plot.columns:
#     order = (
#         df_plot.groupby("antigen_family")["immunization_coverage"]
#         .median().sort_values(ascending=False).index
#     )
#     fig, ax = plt.subplots(figsize=(13, 6))
#     sns.boxplot(
#         data=df_plot, x="antigen_family", y="immunization_coverage",
#         order=order, palette="Set2", ax=ax, width=0.55,
#     )
#     ax.set_title("Immunization Coverage by Vaccine Type", fontsize=14, fontweight="bold")
#     ax.set_xlabel("Antigen Family")
#     ax.set_ylabel("Coverage (%)")
#     ax.tick_params(axis="x", rotation=30)
#     plt.tight_layout()
#     plt.savefig(FIG_DIR / "01_coverage_by_vaccine.png", dpi=150)
#     plt.show()
#     print("\nSaved: 01_coverage_by_vaccine.png")
# 
# # ── Plot 2: Pharma Tariffs over the Years ────────────────────────────────────
# if {"pharma_tariff_rate", "year"}.issubset(df_plot.columns):
#     t = (
#         df_plot.dropna(subset=["pharma_tariff_rate"])
#         .groupby("year")["pharma_tariff_rate"]
#         .agg(
#             mean   ="mean",
#             median ="median",
#             q25    =lambda x: x.quantile(0.25),
#             q75    =lambda x: x.quantile(0.75),
#         )
#     )
#     fig, ax = plt.subplots(figsize=(12, 5))
#     ax.fill_between(t.index, t["q25"], t["q75"], alpha=0.20, color="steelblue", label="IQR (25–75%)")
#     ax.plot(t.index, t["mean"],   "o-",  color="steelblue",  linewidth=2,   label="Mean")
#     ax.plot(t.index, t["median"], "s--", color="darkorange",  linewidth=1.5, label="Median")
#     ax.set_title("Pharma Tariff Rates over the Years", fontsize=14, fontweight="bold")
#     ax.set_xlabel("Year")
#     ax.set_ylabel("Pharma Tariff Rate")
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig(FIG_DIR / "02_pharma_tariffs_over_years.png", dpi=150)
#     plt.show()
#     print("Saved: 02_pharma_tariffs_over_years.png")
# 
# # ── Plot 3: Immunization Coverage over the Years (by country) ────────────────
# if {"year", "country_iso3", "immunization_coverage"}.issubset(df_plot.columns):
#     top_countries = (
#         df_plot.groupby("country_iso3")["immunization_coverage"]
#         .std().nlargest(15).index
#     )
#     trend = (
#         df_plot[df_plot["country_iso3"].isin(top_countries)]
#         .groupby(["country_iso3", "year"])["immunization_coverage"]
#         .mean().reset_index()
#     )
#     fig, ax = plt.subplots(figsize=(14, 7))
#     for country, grp in trend.groupby("country_iso3"):
#         ax.plot(grp["year"], grp["immunization_coverage"], marker="o", linewidth=1.5, label=country)
#     ax.set_title(
#         "Immunization Coverage over the Years\n(Top 15 countries by coverage variability)",
#         fontsize=14, fontweight="bold",
#     )
#     ax.set_xlabel("Year")
#     ax.set_ylabel("Coverage (%)")
#     ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, ncol=1)
#     plt.tight_layout()
#     plt.savefig(FIG_DIR / "03_coverage_over_years_by_country.png", dpi=150)
#     plt.show()
#     print("Saved: 03_coverage_over_years_by_country.png")
# 
# # ── Plot 4: Cluster of Coverage Scores with Regional Labels ──────────────────
# try:
#     import country_converter as coco
#     region_map = {
#         iso: coco.convert(iso, to="continent", not_found="Unknown")
#         for iso in df_plot["country_iso3"].dropna().unique()
#     }
# except ImportError:
#     region_map = {}
# 
# cluster_df = (
#     df_plot.groupby("country_iso3")["immunization_coverage"]
#     .agg(mean_coverage="mean", std_coverage="std")
#     .dropna()
#     .reset_index()
# )
# cluster_df["region"] = cluster_df["country_iso3"].map(region_map).fillna("Unknown")
# 
# X_scaled = StandardScaler().fit_transform(cluster_df[["mean_coverage", "std_coverage"]])
# cluster_df["cluster"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_scaled)
# 
# regions  = cluster_df["region"].unique()
# palette  = sns.color_palette("tab10", n_colors=len(regions))
# reg_color = dict(zip(regions, palette))
# markers   = ["o", "s", "^", "D"]
# 
# fig, ax = plt.subplots(figsize=(13, 8))
# for _, row in cluster_df.iterrows():
#     ax.scatter(
#         row["mean_coverage"], row["std_coverage"],
#         color=reg_color[row["region"]],
#         marker=markers[int(row["cluster"]) % len(markers)],
#         s=90, alpha=0.85, edgecolors="white", linewidths=0.4,
#     )
#     ax.annotate(
#         row["country_iso3"],
#         (row["mean_coverage"], row["std_coverage"]),
#         fontsize=6, alpha=0.75,
#         textcoords="offset points", xytext=(4, 2),
#     )
# 
# legend_handles = [mpatches.Patch(color=c, label=r) for r, c in reg_color.items()]
# ax.legend(handles=legend_handles, title="Region", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
# ax.set_title("Cluster of Immunization Coverage Scores with Regional Labels", fontsize=14, fontweight="bold")
# ax.set_xlabel("Mean Coverage (%)")
# ax.set_ylabel("Std Dev of Coverage")
# plt.tight_layout()
# plt.savefig(FIG_DIR / "04_coverage_clusters_regional.png", dpi=150)
# plt.show()
# print("Saved: 04_coverage_clusters_regional.png")
# 
# # ── Plot 5: Joint Distribution of Immunization Coverage & GDP per Capita ──────
# joint_data = df_plot[["immunization_coverage", "gdp_per_capita_usd"]].dropna()
# 
# fig = plt.figure(figsize=(10, 9))
# g = sns.JointGrid(
#     data=joint_data,
#     x="gdp_per_capita_usd",
#     y="immunization_coverage",
#     height=9,
# )
# g.plot_joint(sns.scatterplot, alpha=0.25, s=18, color="steelblue", edgecolors="none")
# g.plot_joint(sns.kdeplot,     levels=6,   color="navy", linewidths=0.8)
# g.plot_marginals(sns.histplot, kde=True, color="steelblue", bins=35)
# 
# g.ax_joint.set_xlabel("GDP per Capita (USD)", fontsize=11)
# g.ax_joint.set_ylabel("Immunization Coverage (%)", fontsize=11)
# g.figure.suptitle(
#     "Joint Distribution: Immunization Coverage vs GDP per Capita",
#     fontsize=13, fontweight="bold", y=1.01,
# )
# g.figure.tight_layout()
# g.figure.savefig(FIG_DIR / "05_coverage_gdp_joint.png", dpi=150, bbox_inches="tight")
# plt.show()
# print("Saved: 05_coverage_gdp_joint.png")
# 
# # ── Plot 5b: Coverage vs GDP faceted by Antigen Family ───────────────────────
# facet_data = df_plot[["immunization_coverage", "gdp_per_capita_usd", "antigen_family"]].dropna()
# antigens   = sorted(facet_data["antigen_family"].unique())
# ncols      = 4
# nrows      = -(-len(antigens) // ncols)   # ceiling division
# 
# fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5), sharex=False, sharey=True)
# axes_flat = axes.flatten()
# 
# for i, antigen in enumerate(antigens):
#     ax  = axes_flat[i]
#     sub = facet_data[facet_data["antigen_family"] == antigen]
#     ax.scatter(sub["gdp_per_capita_usd"], sub["immunization_coverage"],
#                alpha=0.20, s=12, color="steelblue", edgecolors="none")
#     # lowess trend line
#     sorted_sub = sub.sort_values("gdp_per_capita_usd")
#     lowess = stats.siegelslopes(
#         sorted_sub["immunization_coverage"].values,
#         sorted_sub["gdp_per_capita_usd"].values,
#     )
#     x_range = np.linspace(sorted_sub["gdp_per_capita_usd"].min(),
#                           sorted_sub["gdp_per_capita_usd"].max(), 200)
#     ax.plot(x_range, lowess.slope * x_range + lowess.intercept,
#             color="crimson", linewidth=1.5)
#     rho, p = stats.spearmanr(sub["gdp_per_capita_usd"], sub["immunization_coverage"])
#     ax.set_title(f"{antigen}\nr={rho:.2f}, p={p:.3f}", fontsize=9, fontweight="bold")
#     ax.set_xlabel("")
#     ax.set_ylabel("Coverage (%)" if i % ncols == 0 else "")
#     ax.tick_params(labelsize=7)
# 
# # hide unused panels
# for j in range(len(antigens), len(axes_flat)):
#     axes_flat[j].set_visible(False)
# 
# fig.supxlabel("GDP per Capita (USD)", fontsize=11)
# fig.suptitle("Immunization Coverage vs GDP per Capita by Antigen Family\n(red line = Siegel robust trend)",
#              fontsize=13, fontweight="bold")
# plt.tight_layout()
# plt.savefig(FIG_DIR / "05b_coverage_gdp_by_antigen.png", dpi=150)
# plt.show()
# print("Saved: 05b_coverage_gdp_by_antigen.png")
# 
# # ── Plot 6: Log-Transform of Pharma Tariff Rates ─────────────────────────────
# if "pharma_tariff_rate" in df_plot.columns:
#     tariff_data = df_plot["pharma_tariff_rate"].dropna()
#     # log1p handles zero values; shift if any negatives exist
#     shift      = max(0.0, -tariff_data.min() + 1e-9) if (tariff_data <= 0).any() else 0.0
#     log_tariff = np.log1p(tariff_data + shift)
# 
#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#     sns.histplot(tariff_data, bins=40, kde=True, color="coral",     ax=axes[0])
#     sns.histplot(log_tariff,  bins=40, kde=True, color="steelblue", ax=axes[1])
#     axes[0].set_title("Original",          fontsize=12, fontweight="bold")
#     axes[0].set_xlabel("Pharma Tariff Rate")
#     axes[1].set_title("Log-Transformed",   fontsize=12, fontweight="bold")
#     axes[1].set_xlabel("log(1 + Tariff Rate)")
#     for ax_ in axes:
#         ax_.set_ylabel("Count")
#     fig.suptitle("Pharma Tariff Rate: Original vs Log-Transformed", fontsize=14, fontweight="bold")
#     plt.tight_layout()
#     plt.savefig(FIG_DIR / "06_pharma_tariff_log_transform.png", dpi=150)
#     plt.show()
#     print("Saved: 06_pharma_tariff_log_transform.png")
# 
# # ── Plot 7: Spearman Correlation Heatmap ─────────────────────────────────────
# fig, ax = plt.subplots(figsize=(9, 7))
# mask = np.triu(np.ones_like(spearman_matrix, dtype=bool))
# sns.heatmap(
#     spearman_matrix, mask=mask, annot=True, fmt=".2f",
#     cmap="RdBu_r", center=0, vmin=-1, vmax=1,
#     square=True, linewidths=0.5, ax=ax,
#     annot_kws={"size": 10},
# )
# ax.set_title("Spearman Correlation Heatmap (Covariates)", fontsize=14, fontweight="bold")
# plt.tight_layout()
# plt.savefig(FIG_DIR / "07_spearman_correlation.png", dpi=150)
# plt.show()
# print("Saved: 07_spearman_correlation.png")
# 
# print(f"\nAll figures saved -> {FIG_DIR}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        PIVOT DATASET  EDA                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

PIVOT_PATH    = DATA_PATH.parent / "pivot_dataset.csv"
PIVOT_FIG_DIR = DATA_PATH.parent / "figures" / "pivot"
PIVOT_FIG_DIR.mkdir(parents=True, exist_ok=True)

pv = pd.read_csv(PIVOT_PATH)
print("\n" + "#" * 60)
print("#  PIVOT DATASET")
print("#" * 60)
print(f"Loaded  : {pv.shape[0]:,} rows × {pv.shape[1]} columns")
print(f"Columns : {pv.columns.tolist()}\n")

# ════════════════════════════════════════════════════════════════════════════
# P1. Total Missing Values
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("P1. TOTAL MISSING VALUES  [pivot_dataset]")
print("=" * 60)
pv_total_missing = pv.isna().sum().sum()
pv_total_cells   = pv.size
print(f"  Missing cells : {pv_total_missing:,} / {pv_total_cells:,}  ({pv_total_missing / pv_total_cells:.2%})")

# ════════════════════════════════════════════════════════════════════════════
# P2. Missing Values per Column
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("P2. MISSING VALUES PER COLUMN  [pivot_dataset]")
print("=" * 60)
pv_miss_col     = pv.isna().sum().rename("count")
pv_miss_pct     = (pv_miss_col / len(pv) * 100).round(2).rename("pct_%")
pv_miss_summary = pd.concat([pv_miss_col, pv_miss_pct], axis=1).sort_values("count", ascending=False)
print(pv_miss_summary.to_string())

# ════════════════════════════════════════════════════════════════════════════
# P3. Missing Values per Country
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("P3. MISSING VALUES PER COUNTRY  [pivot_dataset]")
print("=" * 60)
pv_rows_per_country = pv.groupby("country_iso3").size().rename("n_rows")
pv_miss_per_country = (
    pv.groupby("country_iso3")
    .apply(lambda g: g.isna().sum().sum(), include_groups=False)
    .rename("missing_cells")
)
pv_miss_country_pct = (
    pv_miss_per_country / (pv_rows_per_country * pv.shape[1]) * 100
).round(2).rename("missing_pct_%")
pv_country_summary = (
    pd.concat([pv_rows_per_country, pv_miss_per_country, pv_miss_country_pct], axis=1)
    .sort_values("missing_cells", ascending=False)
)
pv_has_missing = pv_country_summary[pv_country_summary["missing_cells"] > 0]
print(f"  {len(pv_has_missing)} / {len(pv_country_summary)} countries have at least one missing cell\n")
print(pv_has_missing.to_string())

# ════════════════════════════════════════════════════════════════════════════
# P4. MAR / MNAR Tests
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("P4. MAR / MNAR TESTS  [pivot_dataset]")
print("=" * 60)
print(
    "  Method: point-biserial correlation between each column's missingness\n"
    "  indicator (1=missing) and all other observed numeric variables.\n"
    "  MAR  -> missingness is significantly predicted by observed variables (p<0.05).\n"
    "  MNAR -> cannot be verified statistically; flagged when MAR evidence is weak\n"
    "         but missingness rate is high and domain knowledge suggests self-selection.\n"
    "  MCAR -> no observed variable predicts missingness (inconclusive baseline).\n"
)

pv_numeric_cols   = pv.select_dtypes(include=[np.number]).columns.tolist()
pv_cols_with_miss = [c for c in pv_numeric_cols if pv[c].isna().any()]

for target in pv_cols_with_miss:
    indicator  = pv[target].isna().astype(int)
    miss_rate  = indicator.mean()
    predictors = [c for c in pv_numeric_cols if c != target and pv[c].notna().sum() >= 30]
    sig_preds  = []

    for pred in predictors:
        mask = pv[pred].notna()
        if mask.sum() < 10:
            continue
        r, p = stats.pointbiserialr(indicator[mask], pv[pred][mask])
        if p < 0.05:
            sig_preds.append((pred, r, p))

    if sig_preds:
        pv_verdict = "MAR"
        pv_note    = f"Missingness predicted by {len(sig_preds)} variable(s)"
    elif miss_rate > 0.30:
        pv_verdict = "Likely MNAR"
        pv_note    = f"High miss rate ({miss_rate:.0%}) with no observed predictor — self-selection plausible"
    else:
        pv_verdict = "MCAR / Inconclusive"
        pv_note    = "No observed variable predicts missingness"

    print(f"  Column : {target}  (missing: {miss_rate:.1%})")
    print(f"  Verdict: {pv_verdict}")
    print(f"  Note   : {pv_note}")
    if sig_preds:
        for pred, r, p in sorted(sig_preds, key=lambda x: x[2]):
            print(f"           {pred:<30s}  r={r:+.3f}  p={p:.4f}")
    print()

# ════════════════════════════════════════════════════════════════════════════
# P5. Drop 'gavi_eligible'
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("P5. DROPPING 'gavi_eligible'  [pivot_dataset]")
print("=" * 60)
if "gavi_eligible" in pv.columns:
    pv = pv.drop(columns=["gavi_eligible"])
    print("  Dropped: gavi_eligible")
else:
    print("  Column not present — skipping")

# keep a copy with identifiers for plots
pv_plot = pv.copy()

# ════════════════════════════════════════════════════════════════════════════
# P6. Drop ISO3 Columns
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("P6. DROPPING ISO3 COLUMNS  [pivot_dataset]")
print("=" * 60)
pv_iso3_cols = [c for c in pv.columns if "iso3" in c.lower()]
pv = pv.drop(columns=pv_iso3_cols)
print(f"  Dropped: {pv_iso3_cols}")
print(f"  Remaining columns: {pv.columns.tolist()}")

# ════════════════════════════════════════════════════════════════════════════
# P7. Spearman Correlation
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("P7. SPEARMAN CORRELATION BETWEEN COVARIATES  [pivot_dataset]")
print("=" * 60)
pv_covariate_cols = [c for c in [
    "immunization_coverage", "pharma_tariff_rate",
    "gdp_per_capita_usd", "health_exp_pct_gdp",
    "population_total", "oop_health_exp_pct", "tariff_x_oop",
] if c in pv.columns]

pv_spearman_matrix = pv[pv_covariate_cols].corr(method="spearman")
print(pv_spearman_matrix.round(3).to_string())

# ════════════════════════════════════════════════════════════════════════════
# P8. Eta-Squared (country groups -> immunization coverage)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("P8. ETA-SQUARED  (country grouping -> immunization coverage)  [pivot_dataset]")
print("=" * 60)
print("  One-way ANOVA: immunization_coverage ~ country_iso3\n")

pv_groups = [
    g["immunization_coverage"].dropna().values
    for _, g in pv_plot.groupby("country_iso3")
    if g["immunization_coverage"].dropna().shape[0] > 1
]

if pv_groups:
    pv_f_stat, pv_p_val  = stats.f_oneway(*pv_groups)
    pv_valid_coverage    = pv_plot["immunization_coverage"].dropna()
    pv_grand_mean        = pv_valid_coverage.mean()
    pv_ss_total          = ((pv_valid_coverage - pv_grand_mean) ** 2).sum()
    pv_ss_between        = sum(len(g) * (g.mean() - pv_grand_mean) ** 2 for g in pv_groups)
    pv_eta_sq            = pv_ss_between / pv_ss_total

    if pv_eta_sq < 0.01:
        pv_effect = "negligible"
    elif pv_eta_sq < 0.06:
        pv_effect = "small"
    elif pv_eta_sq < 0.14:
        pv_effect = "medium"
    else:
        pv_effect = "large"

    print(f"  F-statistic : {pv_f_stat:.4f}")
    print(f"  p-value     : {pv_p_val:.4e}")
    print(f"  Eta-squared : {pv_eta_sq:.4f}  ({pv_effect} effect)")
    print(f"  Interpretation: {pv_eta_sq:.1%} of variance in immunization coverage is")
    print(f"                  explained by country membership.")

# ════════════════════════════════════════════════════════════════════════════
# P9. Drop 'gni_per_capita_usd'
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("P9. DROPPING 'gni_per_capita_usd'  [pivot_dataset]")
print("=" * 60)
if "gni_per_capita_usd" in pv.columns:
    pv = pv.drop(columns=["gni_per_capita_usd"])
    print("  Dropped: gni_per_capita_usd  (collinear with gdp_per_capita_usd; GAVI proxy redundant)")
else:
    print("  Column not present — skipping")
print(f"  Remaining columns: {pv.columns.tolist()}")

# ════════════════════════════════════════════════════════════════════════════
# PIVOT DATASET PLOTS
# ════════════════════════════════════════════════════════════════════════════

# ── Plot P1: Immunization Coverage by Vaccine Type ───────────────────────────
if "antigen_family" in pv_plot.columns:
    pv_order = (
        pv_plot.groupby("antigen_family")["immunization_coverage"]
        .median().sort_values(ascending=False).index
    )
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(
        data=pv_plot, x="antigen_family", y="immunization_coverage",
        order=pv_order, palette="Set2", ax=ax, width=0.55,
    )
    ax.set_title("Immunization Coverage by Vaccine Type  [pivot_dataset]", fontsize=14, fontweight="bold")
    ax.set_xlabel("Antigen Family")
    ax.set_ylabel("Coverage (%)")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(PIVOT_FIG_DIR / "p01_coverage_by_vaccine.png", dpi=150)
    plt.show()
    print("\nSaved: p01_coverage_by_vaccine.png")

# ── Plot P2: Pharma Tariffs over the Years ───────────────────────────────────
if {"pharma_tariff_rate", "year"}.issubset(pv_plot.columns):
    pv_t = (
        pv_plot.dropna(subset=["pharma_tariff_rate"])
        .groupby("year")["pharma_tariff_rate"]
        .agg(
            mean   ="mean",
            median ="median",
            q25    =lambda x: x.quantile(0.25),
            q75    =lambda x: x.quantile(0.75),
        )
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(pv_t.index, pv_t["q25"], pv_t["q75"], alpha=0.20, color="steelblue", label="IQR (25–75%)")
    ax.plot(pv_t.index, pv_t["mean"],   "o-",  color="steelblue",  linewidth=2,   label="Mean")
    ax.plot(pv_t.index, pv_t["median"], "s--", color="darkorange",  linewidth=1.5, label="Median")
    ax.set_title("Pharma Tariff Rates over the Years  [pivot_dataset]", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Pharma Tariff Rate")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PIVOT_FIG_DIR / "p02_pharma_tariffs_over_years.png", dpi=150)
    plt.show()
    print("Saved: p02_pharma_tariffs_over_years.png")

# ── Plot P3: Immunization Coverage over the Years (by country) ───────────────
if {"year", "country_iso3", "immunization_coverage"}.issubset(pv_plot.columns):
    pv_top_countries = (
        pv_plot.groupby("country_iso3")["immunization_coverage"]
        .std().nlargest(15).index
    )
    pv_trend = (
        pv_plot[pv_plot["country_iso3"].isin(pv_top_countries)]
        .groupby(["country_iso3", "year"])["immunization_coverage"]
        .mean().reset_index()
    )
    fig, ax = plt.subplots(figsize=(14, 7))
    for country, grp in pv_trend.groupby("country_iso3"):
        ax.plot(grp["year"], grp["immunization_coverage"], marker="o", linewidth=1.5, label=country)
    ax.set_title(
        "Immunization Coverage over the Years  [pivot_dataset]\n(Top 15 countries by coverage variability)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Coverage (%)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, ncol=1)
    plt.tight_layout()
    plt.savefig(PIVOT_FIG_DIR / "p03_coverage_over_years_by_country.png", dpi=150)
    plt.show()
    print("Saved: p03_coverage_over_years_by_country.png")

# ── Plot P4: Cluster of Coverage Scores with Regional Labels ─────────────────
# pivot_dataset has 'unicef_region' directly — no external library needed
pv_cluster_df = (
    pv_plot.groupby("country_iso3")
    .agg(
        mean_coverage=("immunization_coverage", "mean"),
        std_coverage =("immunization_coverage", "std"),
        region       =("unicef_region",          "first"),
    )
    .dropna(subset=["mean_coverage", "std_coverage"])
    .reset_index()
)

pv_X_scaled = StandardScaler().fit_transform(pv_cluster_df[["mean_coverage", "std_coverage"]])
pv_cluster_df["cluster"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(pv_X_scaled)

pv_regions  = pv_cluster_df["region"].unique()
pv_palette  = sns.color_palette("tab10", n_colors=len(pv_regions))
pv_reg_color = dict(zip(pv_regions, pv_palette))
pv_markers   = ["o", "s", "^", "D"]

fig, ax = plt.subplots(figsize=(13, 8))
for _, row in pv_cluster_df.iterrows():
    ax.scatter(
        row["mean_coverage"], row["std_coverage"],
        color=pv_reg_color[row["region"]],
        marker=pv_markers[int(row["cluster"]) % len(pv_markers)],
        s=90, alpha=0.85, edgecolors="white", linewidths=0.4,
    )
    ax.annotate(
        row["country_iso3"],
        (row["mean_coverage"], row["std_coverage"]),
        fontsize=6, alpha=0.75,
        textcoords="offset points", xytext=(4, 2),
    )

pv_legend_handles = [mpatches.Patch(color=c, label=r) for r, c in pv_reg_color.items()]
ax.legend(handles=pv_legend_handles, title="UNICEF Region", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
ax.set_title("Cluster of Immunization Coverage Scores with Regional Labels  [pivot_dataset]",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Mean Coverage (%)")
ax.set_ylabel("Std Dev of Coverage")
plt.tight_layout()
plt.savefig(PIVOT_FIG_DIR / "p04_coverage_clusters_regional.png", dpi=150)
plt.show()
print("Saved: p04_coverage_clusters_regional.png")

# ── Plot P5: Distribution of Immunization Coverage ───────────────────────────
pv_cov_data = pv_plot["immunization_coverage"].dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# overall histogram + KDE
sns.histplot(pv_cov_data, bins=40, kde=True, color="steelblue", ax=axes[0])
axes[0].axvline(pv_cov_data.mean(),   color="crimson",    linestyle="--", linewidth=1.5, label=f"Mean {pv_cov_data.mean():.1f}%")
axes[0].axvline(pv_cov_data.median(), color="darkorange", linestyle=":" , linewidth=1.5, label=f"Median {pv_cov_data.median():.1f}%")
axes[0].legend(fontsize=9)
axes[0].set_title("Overall Distribution", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Immunization Coverage (%)")
axes[0].set_ylabel("Count")

# by UNICEF region (violin)
if "unicef_region" in pv_plot.columns:
    pv_region_order = (
        pv_plot.groupby("unicef_region")["immunization_coverage"]
        .median().sort_values(ascending=False).index
    )
    sns.violinplot(
        data=pv_plot, x="unicef_region", y="immunization_coverage",
        order=pv_region_order, palette="Set3", ax=axes[1], inner="quartile",
    )
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_title("By UNICEF Region", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("UNICEF Region")
    axes[1].set_ylabel("Coverage (%)")

fig.suptitle("Distribution of Immunization Coverage  [pivot_dataset]", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PIVOT_FIG_DIR / "p05_coverage_distribution.png", dpi=150)
plt.show()
print("Saved: p05_coverage_distribution.png")

# ── Plot P6: Log-Transform of Pharma Tariff Rates ────────────────────────────
if "pharma_tariff_rate" in pv_plot.columns:
    pv_tariff_data = pv_plot["pharma_tariff_rate"].dropna()
    pv_shift       = max(0.0, -pv_tariff_data.min() + 1e-9) if (pv_tariff_data <= 0).any() else 0.0
    pv_log_tariff  = np.log1p(pv_tariff_data + pv_shift)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(pv_tariff_data, bins=40, kde=True, color="coral",     ax=axes[0])
    sns.histplot(pv_log_tariff,  bins=40, kde=True, color="steelblue", ax=axes[1])
    axes[0].set_title("Original",        fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Pharma Tariff Rate")
    axes[1].set_title("Log-Transformed", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("log(1 + Tariff Rate)")
    for ax_ in axes:
        ax_.set_ylabel("Count")
    fig.suptitle("Pharma Tariff Rate: Original vs Log-Transformed  [pivot_dataset]",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PIVOT_FIG_DIR / "p06_pharma_tariff_log_transform.png", dpi=150)
    plt.show()
    print("Saved: p06_pharma_tariff_log_transform.png")

# ── Plot P7: Spearman Correlation Heatmap ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
pv_mask = np.triu(np.ones_like(pv_spearman_matrix, dtype=bool))
sns.heatmap(
    pv_spearman_matrix, mask=pv_mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    square=True, linewidths=0.5, ax=ax,
    annot_kws={"size": 9},
)
ax.set_title("Spearman Correlation Heatmap (Covariates)  [pivot_dataset]", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PIVOT_FIG_DIR / "p07_spearman_correlation.png", dpi=150)
plt.show()
print("Saved: p07_spearman_correlation.png")

print(f"\nAll pivot figures saved -> {PIVOT_FIG_DIR}")

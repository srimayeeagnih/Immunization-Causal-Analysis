"""
Standalone DML script — Step 5 of the causal pipeline.

Loads panel_s2.parquet saved by run_did.py and runs:
  - LinearDML to estimate country-level CATEs
  - K-means clustering on [CATE + covariates]
  - Heterogeneity visualizations (WHO region, CATE vs GDP, PCA clusters)

Saves:
  - data/processed/plot_cache.pkl           (consumed by src/plot_heterogeneity.py)
  - outputs/visualization/heterogeneity_clusters.png

Usage:
    python src/run_dml.py
    (requires data/processed/panel_s2.parquet — run src/run_did.py first)
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from econml.dml import LinearDML
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import BoundaryNorm, ListedColormap

_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, "outputs", "visualization")
COVARS  = ["gdp_per_capita_usd", "health_exp_pct_gdp", "oop_health_exp_pct", "population_total"]

# ── Load panel_s2 ─────────────────────────────────────────────────────────────
_panel_path = os.path.join(_ROOT, "data", "processed", "panel_s2.parquet")
if not os.path.exists(_panel_path):
    raise FileNotFoundError(
        f"panel_s2.parquet not found at {_panel_path}.\n"
        "Run src/run_did.py first."
    )
panel_s2 = pd.read_parquet(_panel_path)
print(f"Loaded panel_s2: {panel_s2.shape}")

# ── 5a: Prepare data ──────────────────────────────────────────────────────────
panel = panel_s2.copy()
panel["pta_active"] = (
    (panel["gname"] > 0) & (panel["year"] >= panel["gname"])
).astype(float)

ml = panel.dropna(subset=["immunization_coverage"] + COVARS).copy()
country_avg = ml.groupby("country_iso3")[COVARS].mean()
ml = ml.join(
    country_avg.rename(columns={c: f"{c}_avg" for c in COVARS}),
    on="country_iso3"
)
X_mod_cols = [f"{c}_avg" for c in COVARS]

Y = ml["immunization_coverage"].values
T = ml["pta_active"].values
X = ml[X_mod_cols].values

# Pre-partial out global time trend from Y
year_dummies  = pd.get_dummies(ml["year"], drop_first=True).astype(float).values
_time_model   = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]))])
_time_model.fit(year_dummies, Y)
Y_detrended   = Y - _time_model.predict(year_dummies)

# ── 5b: Linear DML ────────────────────────────────────────────────────────────
model_y = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]))])
model_t = Pipeline([("scaler", StandardScaler()), ("logistic", LogisticRegressionCV(cv=3, penalty="l2", solver="lbfgs"))])

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

dml = LinearDML(model_y=model_y, model_t=model_t, discrete_treatment=True, random_state=42, cv=3)
dml.fit(Y_detrended, T, X=X_scaled, W=None)
print("\nStep 5b -- Overall ATE (Linear DML):", dml.ate(X_scaled).round(4))

# ── 5c: Country-level CATEs ───────────────────────────────────────────────────
country_Xmat     = scaler.transform(country_avg[COVARS].values)
cate             = dml.effect(country_Xmat)
cate_lb, cate_ub = dml.effect_interval(country_Xmat, alpha=0.10)

country_res = country_avg.reset_index().copy()
country_res["cate"]    = cate
country_res["cate_lb"] = cate_lb
country_res["cate_ub"] = cate_ub

print("\nStep 5c -- Top 10 countries by CATE:")
print(country_res[["country_iso3", "cate", "cate_lb", "cate_ub"]]
      .sort_values("cate", ascending=False).head(10).to_string())

# ── 5d: K-means clustering ────────────────────────────────────────────────────
K = 3
X_clust        = country_res[["cate"] + COVARS].dropna()
X_clust_scaled = StandardScaler().fit_transform(X_clust)

kmeans = KMeans(n_clusters=K, random_state=42, n_init=20)
country_res.loc[X_clust.index, "cluster"] = kmeans.fit_predict(X_clust_scaled).astype(float)

pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_clust_scaled)

# Save cache for plot_heterogeneity.py
_cache_path = os.path.join(_ROOT, "data", "processed", "plot_cache.pkl")
with open(_cache_path, "wb") as f:
    pickle.dump(dict(
        country_res = country_res,
        X_pca       = X_pca,
        pca         = pca,
        kmeans      = kmeans,
        X_clust_idx = X_clust.index.tolist(),
        K           = K,
        COVARS      = COVARS,
        OUT_DIR     = OUT_DIR,
    ), f)
print(f"Plot cache saved -> {_cache_path}")

# ── 5e: Heterogeneity plots ───────────────────────────────────────────────────
_WHO = {
    "AFG":"EMRO","ALB":"EURO","DZA":"EMRO","AGO":"AFRO","ARG":"AMRO","ARM":"EURO",
    "AUS":"WPRO","AUT":"EURO","AZE":"EURO","BHR":"EMRO","BGD":"SEARO","BLR":"EURO",
    "BEL":"EURO","BLZ":"AMRO","BEN":"AFRO","BIH":"EURO","BOL":"AMRO","BRA":"AMRO",
    "BRN":"WPRO","BGR":"EURO","BFA":"AFRO","BDI":"AFRO","KHM":"WPRO","CMR":"AFRO",
    "CAN":"AMRO","CHL":"AMRO","CHN":"WPRO","COL":"AMRO","CRI":"AMRO","HRV":"EURO",
    "CUB":"AMRO","CYP":"EURO","CZE":"EURO","DNK":"EURO","DJI":"EMRO","DOM":"AMRO",
    "ECU":"AMRO","EGY":"EMRO","SLV":"AMRO","EST":"EURO","ETH":"AFRO","FIN":"EURO",
    "FRA":"EURO","DEU":"EURO","GHA":"AFRO","GRC":"EURO","GTM":"AMRO","HUN":"EURO",
    "ISL":"EURO","IND":"SEARO","IDN":"SEARO","IRN":"EMRO","IRQ":"EMRO","IRL":"EURO",
    "ISR":"EURO","ITA":"EURO","JAM":"AMRO","JPN":"WPRO","JOR":"EMRO","KAZ":"EURO",
    "KEN":"AFRO","KOR":"WPRO","KWT":"EMRO","KGZ":"EURO","LAO":"WPRO","LVA":"EURO",
    "LBN":"EMRO","LTU":"EURO","LUX":"EURO","MYS":"WPRO","MDV":"SEARO","MLT":"EURO",
    "MDA":"EURO","MNG":"WPRO","MNE":"EURO","MAR":"EMRO","MOZ":"AFRO","MMR":"SEARO",
    "NPL":"SEARO","NLD":"EURO","NZL":"WPRO","NIC":"AMRO","MKD":"EURO","NOR":"EURO",
    "OMN":"EMRO","PAK":"EMRO","PAN":"AMRO","PRY":"AMRO","PER":"AMRO","PHL":"WPRO",
    "POL":"EURO","PRT":"EURO","QAT":"EMRO","ROU":"EURO","RUS":"EURO","RWA":"AFRO",
    "SAU":"EMRO","SEN":"AFRO","SRB":"EURO","SGP":"WPRO","SVK":"EURO","SVN":"EURO",
    "ZAF":"AFRO","ESP":"EURO","LKA":"SEARO","SDN":"EMRO","SWE":"EURO","CHE":"EURO",
    "TJK":"EURO","TZA":"AFRO","THA":"SEARO","TGO":"AFRO","TTO":"AMRO","TUN":"EMRO",
    "TUR":"EURO","TKM":"EURO","UGA":"AFRO","UKR":"EURO","ARE":"EMRO","GBR":"EURO",
    "USA":"AMRO","URY":"AMRO","UZB":"EURO","VNM":"WPRO","YEM":"EMRO","ZMB":"AFRO","ZWE":"AFRO",
}

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.patch.set_facecolor("#f8f8f8")
for ax in axes:
    ax.set_facecolor("#f8f8f8")

# Plot 1: Mean CATE by WHO region
country_res["who_region"] = country_res["country_iso3"].map(_WHO).fillna("Other")
_reg = (
    country_res.groupby("who_region")["cate"]
    .agg(["mean", "std", "count"]).reset_index()
    .rename(columns={"mean": "mean_cate", "std": "std_cate", "count": "n"})
)
_reg["ci90"] = 1.645 * _reg["std_cate"] / np.sqrt(_reg["n"])
_reg = _reg.sort_values("mean_cate").reset_index(drop=True)

axes[0].errorbar(_reg["mean_cate"] + 0.0015, _reg["who_region"], xerr=_reg["ci90"],
                 fmt="o", capsize=4, markersize=9, color="#b0b0b0", linewidth=1.5, elinewidth=1.5, alpha=0.4, zorder=1)
axes[0].errorbar(_reg["mean_cate"], _reg["who_region"], xerr=_reg["ci90"],
                 fmt="o", capsize=4, markersize=8, color="steelblue", linewidth=1.3, elinewidth=1.3, alpha=0.9, zorder=2)
axes[0].axvline(0, color="black", linestyle="--", lw=0.8, alpha=0.6)
for _, row in _reg.iterrows():
    axes[0].text(row["mean_cate"] + row["ci90"] + 0.005, row["who_region"],
                 f"n={int(row['n'])}", va="center", fontsize=8, color="dimgrey")
axes[0].set_xlabel("Mean CATE (pp immunization coverage)", labelpad=8)
axes[0].set_ylabel("WHO Region", labelpad=14)
axes[0].set_title("Mean CATE by WHO Region\n(Linear DML, 90% CI)", fontweight="bold", pad=10)
axes[0].tick_params(axis="y", labelsize=9)

# Plot 2: CATE vs GDP (discrete colorbar)
_plot_cr = country_res.dropna(subset=["cluster"])
axes[1].scatter(_plot_cr["gdp_per_capita_usd"] + 120, _plot_cr["cate"] - 0.003,
                c="#a0a0a0", s=70, alpha=0.25, zorder=1, linewidths=0)
_cmap3 = ListedColormap(["tomato", "steelblue", "seagreen"])
_norm3 = BoundaryNorm([0, 1, 2, 3], _cmap3.N)
sc = axes[1].scatter(_plot_cr["gdp_per_capita_usd"], _plot_cr["cate"],
                     c=_plot_cr["cluster"], cmap=_cmap3, norm=_norm3, s=65,
                     alpha=0.55, edgecolors="white", linewidths=0.5, zorder=2)
axes[1].axhline(0, color="black", linestyle="--", lw=0.8, alpha=0.6)
axes[1].set_xlabel("Mean GDP per Capita (USD)", labelpad=8)
axes[1].set_ylabel("CATE (pp immunization coverage)", labelpad=8)
axes[1].set_title("CATE vs GDP per Capita\n(coloured by cluster)", fontweight="bold", pad=10)
cb = fig.colorbar(sc, ax=axes[1], label="Cluster", ticks=[0.5, 1.5, 2.5])
cb.ax.set_yticklabels(["Cluster 0", "Cluster 1", "Cluster 2"])

# Plot 3: PCA clusters
cluster_colors = ["tomato", "steelblue", "seagreen"]
for k in range(K):
    mask = kmeans.labels_ == k
    axes[2].scatter(X_pca[mask, 0] + 0.04, X_pca[mask, 1] - 0.04,
                    c="#a0a0a0", s=75, alpha=0.2, zorder=1, linewidths=0)
    axes[2].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=cluster_colors[k], label=f"Cluster {k}",
                    s=65, alpha=0.5, edgecolors="white", linewidths=0.6, zorder=2)

isos = country_res.loc[X_clust.index, "country_iso3"].values
for i, iso in enumerate(isos):
    axes[2].annotate(iso, (X_pca[i, 0], X_pca[i, 1]),
                     fontsize=6.5, alpha=0.75, xytext=(3, 3), textcoords="offset points")

_feat_names = ["CATE", "GDP/cap", "Health Exp%", "OOP%", "Population"]
_pc1 = sorted(zip(_feat_names, pca.components_[0]), key=lambda x: abs(x[1]), reverse=True)[:3]
_pc2 = sorted(zip(_feat_names, pca.components_[1]), key=lambda x: abs(x[1]), reverse=True)[:3]
_loading_txt = ("PC1 drivers: " + " | ".join(f"{n} ({v:+.2f})" for n, v in _pc1) +
                "\nPC2 drivers: " + " | ".join(f"{n} ({v:+.2f})" for n, v in _pc2))
axes[2].text(0.01, 0.01, _loading_txt, transform=axes[2].transAxes,
             fontsize=6.5, va="bottom", color="dimgrey",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85, edgecolor="silver"))
axes[2].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance explained)", labelpad=8)
axes[2].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance explained)", labelpad=8)
axes[2].set_title("Country Clusters\n(CATE + Covariates, PCA)", fontweight="bold", pad=10)
axes[2].legend(fontsize=8, loc="upper right", framealpha=0.9, edgecolor="silver", shadow=True)

fig.suptitle("Treatment Effect Heterogeneity: Pharma PTA on Immunization Coverage",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
_out = os.path.join(OUT_DIR, "heterogeneity_clusters.png")
plt.savefig(_out, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nHeterogeneity plot saved -> {_out}")

# ── 5f: Cluster summaries ─────────────────────────────────────────────────────
print("\n-- Cluster mean characteristics --")
print(country_res.groupby("cluster")[["cate"] + COVARS].mean().round(3).to_string())
print("\n-- Countries per cluster --")
for k in range(K):
    cs = sorted(country_res[country_res["cluster"] == k]["country_iso3"].tolist())
    print(f"Cluster {k}: {cs}")
print("\nDML pipeline complete.")

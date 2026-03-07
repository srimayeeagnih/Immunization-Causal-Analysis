"""
Standalone script to regenerate the heterogeneity clusters figure.

Run this AFTER running modelling_baseline_and_scenario.py at least once
(which saves plot_cache.pkl). Subsequent aesthetic tweaks can be iterated
here without re-running the full pipeline.

Usage:
    python "Modelling Pipeline/plot_heterogeneity.py"
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, ListedColormap

# ── Load cached objects ────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cache_path = os.path.join(_ROOT, 'data', 'processed', 'plot_cache.pkl')
if not os.path.exists(_cache_path):
    raise FileNotFoundError(
        f'Cache not found at {_cache_path}.\n'
        'Run modelling_baseline_and_scenario.py first to generate plot_cache.pkl.'
    )

with open(_cache_path, 'rb') as f:
    cache = pickle.load(f)

country_res  = cache['country_res']
X_pca        = cache['X_pca']
pca          = cache['pca']
kmeans       = cache['kmeans']
X_clust_idx  = cache['X_clust_idx']
K            = cache['K']
COVARS       = cache['COVARS']
OUT_DIR      = os.path.join(_ROOT, 'outputs', 'visualization')

# ── WHO region lookup ──────────────────────────────────────────────────────────
_WHO = {
    'AFG':'EMRO','ALB':'EURO','DZA':'EMRO','AGO':'AFRO','ARG':'AMRO',
    'ARM':'EURO','AUS':'WPRO','AUT':'EURO','AZE':'EURO','BHR':'EMRO',
    'BGD':'SEARO','BLR':'EURO','BEL':'EURO','BLZ':'AMRO','BEN':'AFRO',
    'BIH':'EURO','BOL':'AMRO','BRA':'AMRO','BRN':'WPRO','BGR':'EURO',
    'BFA':'AFRO','BDI':'AFRO','KHM':'WPRO','CMR':'AFRO','CAN':'AMRO',
    'CHL':'AMRO','CHN':'WPRO','COL':'AMRO','CRI':'AMRO','HRV':'EURO',
    'CUB':'AMRO','CYP':'EURO','CZE':'EURO','DNK':'EURO','DJI':'EMRO',
    'DOM':'AMRO','ECU':'AMRO','EGY':'EMRO','SLV':'AMRO','EST':'EURO',
    'ETH':'AFRO','FIN':'EURO','FRA':'EURO','DEU':'EURO','GHA':'AFRO',
    'GRC':'EURO','GTM':'AMRO','HUN':'EURO','ISL':'EURO','IND':'SEARO',
    'IDN':'SEARO','IRN':'EMRO','IRQ':'EMRO','IRL':'EURO','ISR':'EURO',
    'ITA':'EURO','JAM':'AMRO','JPN':'WPRO','JOR':'EMRO','KAZ':'EURO',
    'KEN':'AFRO','KOR':'WPRO','KWT':'EMRO','KGZ':'EURO','LAO':'WPRO',
    'LVA':'EURO','LBN':'EMRO','LTU':'EURO','LUX':'EURO','MYS':'WPRO',
    'MDV':'SEARO','MLT':'EURO','MDA':'EURO','MNG':'WPRO','MNE':'EURO',
    'MAR':'EMRO','MOZ':'AFRO','MMR':'SEARO','NPL':'SEARO','NLD':'EURO',
    'NZL':'WPRO','NIC':'AMRO','MKD':'EURO','NOR':'EURO','OMN':'EMRO',
    'PAK':'EMRO','PAN':'AMRO','PRY':'AMRO','PER':'AMRO','PHL':'WPRO',
    'POL':'EURO','PRT':'EURO','QAT':'EMRO','ROU':'EURO','RUS':'EURO',
    'RWA':'AFRO','SAU':'EMRO','SEN':'AFRO','SRB':'EURO','SGP':'WPRO',
    'SVK':'EURO','SVN':'EURO','ZAF':'AFRO','ESP':'EURO','LKA':'SEARO',
    'SDN':'EMRO','SWE':'EURO','CHE':'EURO','TJK':'EURO','TZA':'AFRO',
    'THA':'SEARO','TGO':'AFRO','TTO':'AMRO','TUN':'EMRO','TUR':'EURO',
    'TKM':'EURO','UGA':'AFRO','UKR':'EURO','ARE':'EMRO','GBR':'EURO',
    'USA':'AMRO','URY':'AMRO','UZB':'EURO','VNM':'WPRO','YEM':'EMRO',
    'ZMB':'AFRO','ZWE':'AFRO',
}

# ── Figure setup ───────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.patch.set_facecolor('#f8f8f8')
for ax in axes:
    ax.set_facecolor('#f8f8f8')

# ── Plot 1: Mean CATE by WHO region ───────────────────────────────────────────
country_res['who_region'] = country_res['country_iso3'].map(_WHO).fillna('Other')
_reg = (
    country_res.groupby('who_region')['cate']
    .agg(['mean', 'std', 'count'])
    .reset_index()
    .rename(columns={'mean': 'mean_cate', 'std': 'std_cate', 'count': 'n'})
)
_reg['ci90'] = 1.645 * _reg['std_cate'] / np.sqrt(_reg['n'])
_reg = _reg.sort_values('mean_cate').reset_index(drop=True)

# Shadow effect: draw a slightly offset copy in grey first
axes[0].errorbar(
    _reg['mean_cate'] + 0.0015, _reg['who_region'],
    xerr=_reg['ci90'],
    fmt='o', capsize=4, markersize=9,
    color='#b0b0b0', linewidth=1.5, elinewidth=1.5, alpha=0.4, zorder=1,
)
axes[0].errorbar(
    _reg['mean_cate'], _reg['who_region'],
    xerr=_reg['ci90'],
    fmt='o', capsize=4, markersize=8,
    color='steelblue', linewidth=1.3, elinewidth=1.3, alpha=0.9, zorder=2,
)
axes[0].axvline(0, color='black', linestyle='--', lw=0.8, alpha=0.6)
for _, row in _reg.iterrows():
    axes[0].text(
        row['mean_cate'] + row['ci90'] + 0.005, row['who_region'],
        f"n={int(row['n'])}", va='center', fontsize=8, color='dimgrey'
    )
axes[0].set_xlabel('Mean CATE (pp immunization coverage)', labelpad=8)
axes[0].set_ylabel('WHO Region', labelpad=14)
axes[0].set_title('Mean CATE by WHO Region\n(Linear DML, 90% CI)', fontweight='bold', pad=10)
axes[0].tick_params(axis='y', labelsize=9)

# ── Plot 2: CATE vs GDP per capita ────────────────────────────────────────────
_plot_cr = country_res.dropna(subset=['cluster'])

# Shadow layer
axes[1].scatter(
    _plot_cr['gdp_per_capita_usd'] + 120, _plot_cr['cate'] - 0.003,
    c='#a0a0a0', s=70, alpha=0.25, zorder=1, linewidths=0,
)
# Main scatter — discrete 3-color mapping (no gradient bleed)
_cmap3 = ListedColormap(['tomato', 'steelblue', 'seagreen'])
_norm3 = BoundaryNorm([0, 1, 2, 3], _cmap3.N)
sc = axes[1].scatter(
    _plot_cr['gdp_per_capita_usd'], _plot_cr['cate'],
    c=_plot_cr['cluster'], cmap=_cmap3, norm=_norm3, s=65,
    alpha=0.55,
    edgecolors='white', linewidths=0.5,
    zorder=2,
)
axes[1].axhline(0, color='black', linestyle='--', lw=0.8, alpha=0.6)
axes[1].set_xlabel('Mean GDP per Capita (USD)', labelpad=8)
axes[1].set_ylabel('CATE (pp immunization coverage)', labelpad=8)
axes[1].set_title('CATE vs GDP per Capita\n(coloured by cluster)', fontweight='bold', pad=10)
cb = fig.colorbar(sc, ax=axes[1], label='Cluster', ticks=[0.5, 1.5, 2.5])
cb.ax.set_yticklabels(['Cluster 0', 'Cluster 1', 'Cluster 2'])

# ── Plot 3: PCA country clusters ──────────────────────────────────────────────
cluster_colors = ['tomato', 'steelblue', 'seagreen']

for k in range(K):
    mask = kmeans.labels_ == k
    # Shadow layer
    axes[2].scatter(
        X_pca[mask, 0] + 0.04, X_pca[mask, 1] - 0.04,
        c='#a0a0a0', s=75, alpha=0.2, zorder=1, linewidths=0,
    )
    # Main scatter — transparent fill with white edge
    axes[2].scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        c=cluster_colors[k], label=f'Cluster {k}',
        s=65,
        alpha=0.5,        # <── transparent fill
        edgecolors='white', linewidths=0.6,
        zorder=2,
    )

# ISO3 country labels
isos = country_res.loc[X_clust_idx, 'country_iso3'].values
for i, iso in enumerate(isos):
    axes[2].annotate(iso, (X_pca[i, 0], X_pca[i, 1]),
                     fontsize=6.5, alpha=0.75,
                     xytext=(3, 3), textcoords='offset points')

# PC loadings annotation
_feat_names = ['CATE', 'GDP/cap', 'Health Exp%', 'OOP%', 'Population']
_pc1 = sorted(zip(_feat_names, pca.components_[0]), key=lambda x: abs(x[1]), reverse=True)[:3]
_pc2 = sorted(zip(_feat_names, pca.components_[1]), key=lambda x: abs(x[1]), reverse=True)[:3]
_loading_txt = (
    'PC1 drivers: ' + ' | '.join(f'{n} ({v:+.2f})' for n, v in _pc1) +
    '\nPC2 drivers: ' + ' | '.join(f'{n} ({v:+.2f})' for n, v in _pc2)
)
axes[2].text(0.01, 0.01, _loading_txt, transform=axes[2].transAxes,
             fontsize=6.5, va='bottom', color='dimgrey',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                       alpha=0.85, edgecolor='silver'))

axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance explained)', labelpad=8)
axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance explained)', labelpad=8)
axes[2].set_title('Country Clusters\n(CATE + Covariates, PCA)', fontweight='bold', pad=10)

# Legend — upper right to avoid overlapping country labels at bottom-left
axes[2].legend(fontsize=8, loc='upper right',
               framealpha=0.9, edgecolor='silver',
               shadow=True)   # <── legend shadow

# ── Save ──────────────────────────────────────────────────────────────────────
fig.suptitle('Treatment Effect Heterogeneity: Pharma PTA on Immunization Coverage',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])

out_path = os.path.join(OUT_DIR, 'heterogeneity_clusters.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved -> {out_path}')

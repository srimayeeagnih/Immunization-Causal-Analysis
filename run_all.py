"""
Full pipeline runner — executes all steps in order.

Steps:
  1. src/data_processing.py     — load + merge raw data -> data/processed/
  2. src/feature_engineering.py — feature engineering  -> data/processed/pivot_dataset_fe.csv
  3. src/run_did.py             — staggered DiD        -> outputs/visualization/es_*.png
                                                          data/processed/panel_s2.parquet
  4. src/run_dml.py             — LinearDML + clusters -> outputs/visualization/heterogeneity_clusters.png
                                                          data/processed/plot_cache.pkl

Usage:
    python run_all.py

Individual steps can also be run directly:
    python src/data_processing.py
    python src/feature_engineering.py
    python src/run_did.py
    python src/run_dml.py
    python src/plot_heterogeneity.py   # re-plot from cache (instant)
"""

import os
import runpy
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("Step 1: Data processing",    "src/data_processing.py"),
    ("Step 2: Feature engineering", "src/feature_engineering.py"),
    ("Step 3: DiD estimation",      "src/run_did.py"),
    ("Step 4: DML + clustering",    "src/run_dml.py"),
]

for label, rel_path in STEPS:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    script = os.path.join(ROOT, rel_path)
    if not os.path.exists(script):
        print(f"ERROR: {script} not found. Aborting.", file=sys.stderr)
        sys.exit(1)
    runpy.run_path(script, run_name="__main__")

print("\n" + "="*60)
print("  Pipeline complete.")
print("="*60)

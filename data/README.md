# Data Directory

This directory is intended to hold the raw and processed data for the FreshBasket Store Segmentation project.

## Why is this directory empty of raw data?

The original raw dataset consists of 31M+ transactional records spanning multiple years (stored as weekly `transactions_YYYYWW.csv` files), totaling several gigabytes. 
To keep the repository lightweight and prevent pushing massive files to GitHub, the raw transaction data has been excluded via `.gitignore`.

## Processed Features
The output of the data preprocessing phase is an aggregated, store-level dataset stored in the `processed/` directory (e.g., `store_features_base.parquet`). This processed dataset contains the behavioral features for the 761 stores, which is sufficient to run the clustering pipeline (`src/a2_segmentation.py`).

## How to reproduce from scratch
If you have access to the original raw transactional chunks:
1. Place all `transactions_YYYYWW.csv` files into this `data/` folder.
2. Run the preprocessing and aggregation scripts (Phase 1, external to this assessment) to generate the store-level feature dataset.
3. Once `data/processed/store_features_base.parquet` is available, you can run `python run_pipeline.py`.

"""
================================================================================
Smoke Test — Phase 1: Data Pipeline
================================================================================

Validates all pipeline components using synthetic M5-format data.
No real M5 dataset required — we generate small CSVs that mirror the
exact schema (column names, dtypes, format) of the real Kaggle files.

Run:
    python test_pipeline.py

Expected: all assertions pass, no exceptions.
================================================================================
"""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ── Import our pipeline classes ──────────────────────────────────────────
from data_pipeline import (
    DemandForecastingPipeline,
    FeatureEngineer,
    M5DataLoader,
    MemoryOptimizer,
    PipelineConfig,
    TimeSeriesSplitter,
)


# ════════════════════════════════════════════════════════════════════════════
# §1  SYNTHETIC DATA GENERATOR
# ════════════════════════════════════════════════════════════════════════════

def generate_synthetic_m5(data_dir: Path, n_items: int = 5, n_days: int = 120) -> None:
    """
    Create synthetic M5-format CSVs for testing.

    Parameters
    ----------
    data_dir : Path
        Where to write the CSVs.
    n_items : int
        Number of unique item-store combinations.
    n_days : int
        Number of daily time steps.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    # ── 1. Calendar ──────────────────────────────────────────────────────
    start_date = pd.Timestamp("2016-01-01")
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    cal = pd.DataFrame({
        "date": dates,
        "wm_yr_wk": [11101 + (i // 7) for i in range(n_days)],
        "weekday": [d.strftime("%A") for d in dates],
        "wday": [d.weekday() + 1 for d in dates],
        "month": [d.month for d in dates],
        "year": [d.year for d in dates],
        "d": [f"d_{i+1}" for i in range(n_days)],
        "event_name_1": pd.array([np.nan] * n_days, dtype="object"),
        "event_type_1": pd.array([np.nan] * n_days, dtype="object"),
        "event_name_2": pd.array([np.nan] * n_days, dtype="object"),
        "event_type_2": pd.array([np.nan] * n_days, dtype="object"),
        "snap_CA": rng.randint(0, 2, n_days),
        "snap_TX": rng.randint(0, 2, n_days),
        "snap_WI": rng.randint(0, 2, n_days),
    })

    # Sprinkle some events
    event_indices = rng.choice(n_days, size=min(5, n_days), replace=False)
    for idx in event_indices:
        cal.loc[idx, "event_name_1"] = "SuperBowl"
        cal.loc[idx, "event_type_1"] = "Sporting"

    cal.to_csv(data_dir / "calendar.csv", index=False)

    # ── 2. Sales (wide format) ───────────────────────────────────────────
    stores = ["CA_1", "TX_1"]
    items = [f"FOODS_1_{str(i).zfill(3)}" for i in range(1, n_items + 1)]
    day_cols = [f"d_{i+1}" for i in range(n_days)]

    rows = []
    for store in stores:
        state = store.split("_")[0]
        for item in items:
            sales = rng.poisson(lam=5, size=n_days).tolist()
            row = {
                "id": f"{item}_{store}_validation",
                "item_id": item,
                "dept_id": "FOODS_1",
                "cat_id": "FOODS",
                "store_id": store,
                "state_id": state,
            }
            for j, d in enumerate(day_cols):
                row[d] = sales[j]
            rows.append(row)

    sales_df = pd.DataFrame(rows)
    sales_df.to_csv(
        data_dir / "sales_train_validation.csv", index=False
    )

    # ── 3. Sell prices ───────────────────────────────────────────────────
    price_rows = []
    unique_weeks = sorted(cal["wm_yr_wk"].unique())
    for store in stores:
        for item in items:
            base_price = rng.uniform(1.0, 10.0)
            for wk in unique_weeks:
                # Small random price variation
                price = round(base_price + rng.uniform(-0.5, 0.5), 2)
                price_rows.append({
                    "store_id": store,
                    "item_id": item,
                    "wm_yr_wk": wk,
                    "sell_price": price,
                })

    prices_df = pd.DataFrame(price_rows)
    prices_df.to_csv(data_dir / "sell_prices.csv", index=False)

    print(f"[OK] Generated synthetic M5 data in {data_dir}")
    print(f"  - calendar.csv:               {len(cal)} rows")
    print(f"  - sales_train_validation.csv: {len(sales_df)} rows (wide)")
    print(f"  - sell_prices.csv:            {len(prices_df)} rows")


# ============================================================================
# S2  TESTS
# ============================================================================

def test_memory_optimizer():
    """Verify dtype downcasting works correctly."""
    print("\n--- Test: MemoryOptimizer ---")

    # Use 1000 rows so that downcasting savings outweigh category metadata overhead
    rng = np.random.RandomState(0)
    n = 1000
    df = pd.DataFrame({
        "a": rng.randint(0, 100, size=n).astype(np.int64),
        "b": rng.uniform(0, 10, size=n).astype(np.float64),
        "c": rng.choice(["cat", "dog", "bird"], size=n),
    })

    mem_before = df.memory_usage(deep=True).sum()
    df = MemoryOptimizer.optimize(df, categorical_cols=["c"])
    mem_after = df.memory_usage(deep=True).sum()

    assert df["a"].dtype in [np.int8, np.int16], f"Unexpected int dtype: {df['a'].dtype}"
    assert df["b"].dtype == np.float32, f"Unexpected float dtype: {df['b'].dtype}"
    assert df["c"].dtype.name == "category", f"Unexpected cat dtype: {df['c'].dtype}"
    assert mem_after < mem_before, f"Memory should decrease: {mem_before} -> {mem_after}"

    print(f"  [OK] Memory reduced: {mem_before} -> {mem_after} bytes ({100*(1 - mem_after/mem_before):.0f}% reduction)")


def test_data_loading(data_dir: Path):
    """Verify loading and melting produces expected shapes."""
    print("\n--- Test: Data Loading ---")

    config = PipelineConfig(data_dir=data_dir)
    loader = M5DataLoader(config)

    cal = loader.load_calendar()
    assert "has_event" in cal.columns, "Missing has_event column"
    assert cal["has_event"].dtype == np.int8, "has_event should be int8"
    print(f"  [OK] Calendar: {cal.shape}")

    prices = loader.load_prices()
    assert "sell_price" in prices.columns, "Missing sell_price column"
    print(f"  [OK] Prices: {prices.shape}")

    df = loader.load_and_merge()
    assert "date" in df.columns, "Missing date column after merge"
    assert "sell_price" in df.columns, "Missing sell_price after merge"
    assert "sales" in df.columns, "Missing sales column after melt"
    print(f"  [OK] Merged: {df.shape}")

    return df


def test_feature_engineering(df: pd.DataFrame):
    """Verify all feature groups are created."""
    print("\n--- Test: Feature Engineering ---")

    config = PipelineConfig()
    engineer = FeatureEngineer(config)
    df_feat = engineer.engineer_features(df.copy())

    # -- Lag features -----------------------------------------------------
    for lag in config.lag_days:
        col = f"sales_lag_{lag}"
        assert col in df_feat.columns, f"Missing {col}"
    print(f"  [OK] Lag features present: {config.lag_days}")

    # -- Rolling features -------------------------------------------------
    for w in config.rolling_windows:
        for suffix in ["rmean", "rstd", "rmin", "rmax"]:
            col = f"sales_{suffix}_{w}"
            assert col in df_feat.columns, f"Missing {col}"
    print(f"  [OK] Rolling features present: {config.rolling_windows}")

    # -- Calendar features ------------------------------------------------
    cal_cols = ["day_of_week", "month", "is_weekend", "year", "week_of_year"]
    for col in cal_cols:
        assert col in df_feat.columns, f"Missing calendar feature {col}"
    print(f"  [OK] Calendar features present")

    # -- Price features ---------------------------------------------------
    price_cols = ["price_change_abs", "price_change_pct", "price_momentum_7d"]
    for col in price_cols:
        assert col in df_feat.columns, f"Missing price feature {col}"
    print(f"  [OK] Price features present")

    # -- SNAP feature -----------------------------------------------------
    assert "snap_eligible" in df_feat.columns, "Missing snap_eligible"
    print(f"  [OK] SNAP feature present")

    # -- No NaN in numeric columns after warmup cutoff --------------------
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns
    nan_counts = df_feat[numeric_cols].isna().sum()
    cols_with_nans = nan_counts[nan_counts > 0]
    assert len(cols_with_nans) == 0, f"NaNs remaining in: {cols_with_nans.to_dict()}"
    print(f"  [OK] No NaN values in numeric columns")

    return df_feat


def test_time_series_splitter(df: pd.DataFrame):
    """Verify temporal splitting prevents leakage."""
    print("\n--- Test: TimeSeriesSplitter ---")

    splitter = TimeSeriesSplitter(
        date_col="date",
        test_days=14,
        n_folds=1,
        embargo_days=1,
    )

    # -- Single split -----------------------------------------------------
    train, test = splitter.single_split(df)

    assert len(train) > 0, "Train set is empty"
    assert len(test) > 0, "Test set is empty"

    # *** THE CRITICAL ASSERTION: no temporal leakage ***
    train_max = train["date"].max()
    test_min = test["date"].min()
    assert train_max < test_min, (
        f"DATA LEAKAGE! Train max date ({train_max}) >= "
        f"Test min date ({test_min})"
    )
    print(f"  [OK] No data leakage: train ends {train_max.date()}, test starts {test_min.date()}")
    print(f"  [OK] Train: {len(train):,} rows | Test: {len(test):,} rows")

    # -- Rolling split ----------------------------------------------------
    rolling_splitter = TimeSeriesSplitter(
        date_col="date",
        test_days=14,
        n_folds=3,
        step_days=14,
        embargo_days=1,
    )
    folds = rolling_splitter.rolling_split(df)
    assert len(folds) >= 1, "No valid folds generated"

    print(f"  [OK] Rolling split: {len(folds)} folds generated")

    # Verify no overlap within any fold
    for fold_idx, fold_train, fold_test in folds:
        ft_max = fold_train["date"].max()
        ft_min = fold_test["date"].min()
        assert ft_max < ft_min, f"Leakage in fold {fold_idx}"
    print(f"  [OK] All folds pass leakage check")

    # -- Summary ----------------------------------------------------------
    summary = splitter.get_split_summary(df)
    assert "total_days" in summary
    print(f"  [OK] Summary: {summary}")


def test_full_pipeline(data_dir: Path, output_dir: Path):
    """End-to-end pipeline test."""
    print("\n--- Test: Full Pipeline (End-to-End) ---")

    config = PipelineConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        sample_frac=None,
        warmup_days=30,
    )

    pipeline = DemandForecastingPipeline(config)
    train_df, test_df = pipeline.run(save=True)

    # -- Verify outputs exist ---------------------------------------------
    assert (output_dir / "train.parquet").exists(), "train.parquet not saved"
    assert (output_dir / "test.parquet").exists(), "test.parquet not saved"
    assert (output_dir / "full_featured.parquet").exists(), "full_featured.parquet not saved"
    print(f"  [OK] Parquet files saved to {output_dir}")

    # -- Verify roundtrip: read back and check shape ----------------------
    train_reloaded = pd.read_parquet(output_dir / "train.parquet")
    assert train_reloaded.shape == train_df.shape, "Shape mismatch on reload"
    print(f"  [OK] Parquet roundtrip verified")

    # -- Verify no leakage in saved splits --------------------------------
    assert train_df["date"].max() < test_df["date"].min(), "Leakage in saved splits!"
    print(f"  [OK] Saved splits pass leakage check")

    print(f"\n  Train shape: {train_df.shape}")
    print(f"  Test shape:  {test_df.shape}")
    print(f"  Features:    {list(train_df.columns)}")


# ============================================================================
# S3  MAIN
# ============================================================================

def main():
    """Run all smoke tests."""
    print("=" * 64)
    print("  SMOKE TEST — Phase 1: Data Pipeline")
    print("=" * 64)

    # -- Create temp directories ------------------------------------------
    tmp_root = Path(tempfile.mkdtemp(prefix="m5_test_"))
    data_dir = tmp_root / "raw"
    output_dir = tmp_root / "processed"

    try:
        # -- Generate synthetic data --------------------------------------
        generate_synthetic_m5(data_dir, n_items=5, n_days=120)

        # -- Run tests ----------------------------------------------------
        test_memory_optimizer()
        df = test_data_loading(data_dir)
        df_feat = test_feature_engineering(df)
        test_time_series_splitter(df_feat)
        test_full_pipeline(data_dir, output_dir)

        print("")
        print("=" * 64)
        print("  ALL TESTS PASSED")
        print("=" * 64)

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        raise

    finally:
        # -- Cleanup ------------------------------------------------------
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"\nCleaned up temp directory: {tmp_root}")


if __name__ == "__main__":
    main()

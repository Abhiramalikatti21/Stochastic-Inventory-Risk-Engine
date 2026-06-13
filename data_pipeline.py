"""
================================================================================
Phase 1: Data Preprocessing & Feature Engineering Pipeline
================================================================================

Prescriptive Demand Forecasting Engine
──────────────────────────────────────
Transforms raw M5 Kaggle dataset (Walmart retail data) into a model-ready
feature matrix for downstream consumption by:
  • Phase 2 — Temporal Fusion Transformer (TFT) quantile forecaster
  • Phase 3 — CausalForestDML price-elasticity estimator

Key Design Decisions:
  1. Memory efficiency: aggressive downcasting of int/float columns and
     categorical encoding to handle the ~30K items × 1,913 days matrix
     on machines with ≤16 GB RAM.
  2. Strict temporal ordering: all lag/rolling features are computed per
     (item_id, store_id) group with chronological sort to prevent leakage.
  3. NaN handling: lag/rolling features naturally produce leading NaNs;
     we forward-fill then zero-fill as a final safety net, and expose a
     configurable warm-up cutoff to drop the unstable initial window.
  4. The TimeSeriesSplitter class enforces a hard calendar cutoff —
     no random shuffling — and supports rolling-window evaluation for
     Phase 4 backtesting.

Usage:
    python data_pipeline.py --data-dir data/raw --output-dir data/processed

Author: Prescriptive Demand Forecasting Team
================================================================================
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─── Logging Configuration ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress pandas performance warnings for chained assignment (we use .loc)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# ════════════════════════════════════════════════════════════════════════════
# §1  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """
    Centralised configuration for every knob in the data pipeline.

    Attributes
    ----------
    data_dir : Path
        Directory containing the three raw M5 CSVs.
    output_dir : Path
        Directory where processed parquet files will be written.
    lag_days : list[int]
        Which daily lags of the target variable to construct.
        Default: [1, 7, 14, 30] — captures short-term momentum, weekly
        seasonality, bi-weekly patterns, and monthly trends.
    rolling_windows : list[int]
        Window sizes for rolling mean / std features.
        Default: [7, 30] — weekly and monthly smoothed demand and volatility.
    warmup_days : int
        Number of initial time-steps to drop per series after feature
        engineering (these rows have NaN lags/rolling stats).
        Should be ≥ max(lag_days + rolling_windows) = 30.
    target_col : str
        Name of the target variable after melting.
    sample_frac : float | None
        If set (0 < sample_frac ≤ 1.0), randomly sample this fraction of
        unique item-store combinations for rapid prototyping. Set to None
        for full-scale runs.
    """

    data_dir: Path = Path("data/raw")
    output_dir: Path = Path("data/processed")

    # ── Feature engineering knobs ────────────────────────────────────────
    lag_days: List[int] = field(default_factory=lambda: [1, 7, 14, 30])
    rolling_windows: List[int] = field(default_factory=lambda: [7, 30])
    warmup_days: int = 30  # drop first N rows per series (NaN-tainted)

    # ── Column naming conventions ────────────────────────────────────────
    target_col: str = "sales"
    date_col: str = "date"
    item_col: str = "item_id"
    store_col: str = "store_id"

    # ── Prototyping ──────────────────────────────────────────────────────
    sample_frac: Optional[float] = None  # e.g. 0.01 for 1% of items

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)


# ════════════════════════════════════════════════════════════════════════════
# §2  MEMORY UTILITIES
# ════════════════════════════════════════════════════════════════════════════

class MemoryOptimizer:
    """
    Static utility class for aggressive dtype downcasting.

    On the full M5 dataset the melted long-format frame can exceed 12 GB
    in naive int64/float64.  Downcasting to int16/float32 + categoricals
    typically reduces this to ~2–3 GB.
    """

    @staticmethod
    def downcast_integers(df: pd.DataFrame) -> pd.DataFrame:
        """
        Downcast every integer column to the smallest subtype that can
        represent its range (int8 → int16 → int32).
        """
        int_cols = df.select_dtypes(include=["int64", "int32", "int16"]).columns
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], downcast="integer")
        return df

    @staticmethod
    def downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Downcast float64 → float32.  We avoid float16 because it has
        only ~3 decimal digits of precision, which can corrupt rolling
        std calculations on small counts.
        """
        float_cols = df.select_dtypes(include=["float64"]).columns
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast="float")
        return df

    @staticmethod
    def convert_categoricals(
        df: pd.DataFrame, columns: List[str]
    ) -> pd.DataFrame:
        """
        Convert object/string columns to pandas Categorical dtype,
        which uses integer codes internally and is far more compact.
        """
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    @classmethod
    def optimize(
        cls, df: pd.DataFrame, categorical_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Full optimization pass: ints → floats → categoricals."""
        df = cls.downcast_integers(df)
        df = cls.downcast_floats(df)
        if categorical_cols:
            df = cls.convert_categoricals(df, categorical_cols)
        return df


# ════════════════════════════════════════════════════════════════════════════
# §3  DATA LOADER
# ════════════════════════════════════════════════════════════════════════════

class M5DataLoader:
    """
    Responsible for reading the three raw M5 CSV files and performing
    the expensive wide → long melt on the sales matrix.

    Expected files in `config.data_dir`:
        • sales_train_validation.csv   (or sales_train_evaluation.csv)
        • calendar.csv
        • sell_prices.csv
    """

    # Column prefixes in the wide sales file (d_1, d_2, …, d_1913)
    _DAY_PREFIX = "d_"

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    # ── File discovery ───────────────────────────────────────────────────

    def _find_sales_file(self) -> Path:
        """
        Auto-detect whether the user has the 'validation' or 'evaluation'
        variant of the sales training file.
        """
        candidates = [
            "sales_train_validation.csv",
            "sales_train_evaluation.csv",
        ]
        for name in candidates:
            path = self.config.data_dir / name
            if path.exists():
                logger.info(f"Found sales file: {path}")
                return path
        raise FileNotFoundError(
            f"No sales CSV found in {self.config.data_dir}. "
            f"Expected one of: {candidates}"
        )

    # ── Core loaders ─────────────────────────────────────────────────────

    def load_calendar(self) -> pd.DataFrame:
        """
        Load calendar.csv and parse dates.

        Returns a DataFrame indexed by the 'd' column (d_1 … d_1969)
        with date, weekday, month, year, event columns, and SNAP flags.
        """
        path = self.config.data_dir / "calendar.csv"
        logger.info(f"Loading calendar from {path}")

        cal = pd.read_csv(path, parse_dates=["date"])

        # ── Encode event columns: fill missing → 'NoEvent', then categorise
        event_cols = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
        for col in event_cols:
            cal[col] = cal[col].fillna("NoEvent")

        # ── Create binary event indicator (any event that day)
        cal["has_event"] = (
            (cal["event_name_1"] != "NoEvent") | (cal["event_name_2"] != "NoEvent")
        ).astype(np.int8)

        # ── Memory: categorise string columns
        cat_cols = ["weekday"] + event_cols
        cal = MemoryOptimizer.convert_categoricals(cal, cat_cols)
        cal = MemoryOptimizer.downcast_integers(cal)

        logger.info(f"Calendar loaded: {cal.shape[0]} rows, {cal.shape[1]} cols")
        return cal

    def load_prices(self) -> pd.DataFrame:
        """
        Load sell_prices.csv.

        Each row is (store_id, item_id, wm_yr_wk, sell_price).
        We'll merge this onto the main frame via (store_id, item_id, wm_yr_wk).
        """
        path = self.config.data_dir / "sell_prices.csv"
        logger.info(f"Loading prices from {path}")

        prices = pd.read_csv(path)

        # ── Memory: sell_price is float64 by default → float32
        prices = MemoryOptimizer.downcast_floats(prices)
        prices = MemoryOptimizer.convert_categoricals(
            prices, ["store_id", "item_id"]
        )
        prices = MemoryOptimizer.downcast_integers(prices)

        logger.info(
            f"Prices loaded: {prices.shape[0]:,} rows, "
            f"mem = {prices.memory_usage(deep=True).sum() / 1e6:.1f} MB"
        )
        return prices

    def load_and_melt_sales(self) -> pd.DataFrame:
        """
        Load the wide-format sales CSV and melt into long format.

        Wide format:
            id | item_id | dept_id | cat_id | store_id | state_id | d_1 | d_2 | … | d_1913
        Long format (output):
            item_id | store_id | dept_id | cat_id | state_id | d | sales

        This is the most memory-intensive step.  We:
          1. Read only necessary columns (drop the composite 'id' col).
          2. Optionally subsample items for prototyping.
          3. Melt in one vectorised call.
          4. Downcast the sales column to int16 (max observed sales ≈ 760).
        """
        sales_path = self._find_sales_file()
        logger.info(f"Loading sales from {sales_path} (this may take a moment)…")

        sales_wide = pd.read_csv(sales_path)

        # ── Identify day columns (d_1, d_2, …)
        day_cols = [c for c in sales_wide.columns if c.startswith(self._DAY_PREFIX)]
        id_cols = ["item_id", "store_id", "dept_id", "cat_id", "state_id"]

        # ── Optional subsampling for prototyping ─────────────────────────
        if self.config.sample_frac is not None and self.config.sample_frac < 1.0:
            n_total = len(sales_wide)
            sales_wide = sales_wide.sample(
                frac=self.config.sample_frac, random_state=42
            )
            logger.info(
                f"Subsampled {len(sales_wide):,}/{n_total:,} item-stores "
                f"({self.config.sample_frac:.1%})"
            )

        # ── Melt: wide → long ───────────────────────────────────────────
        logger.info(f"Melting {len(sales_wide):,} rows × {len(day_cols)} days…")
        t0 = time.perf_counter()

        sales_long = sales_wide.melt(
            id_vars=id_cols,
            value_vars=day_cols,
            var_name="d",                     # e.g. "d_1"
            value_name=self.config.target_col,  # "sales"
        )

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Melt complete: {sales_long.shape[0]:,} rows in {elapsed:.1f}s"
        )

        # ── Memory optimisation ──────────────────────────────────────────
        sales_long = MemoryOptimizer.convert_categoricals(sales_long, id_cols)
        sales_long = MemoryOptimizer.downcast_integers(sales_long)

        logger.info(
            f"Sales (long) memory: "
            f"{sales_long.memory_usage(deep=True).sum() / 1e6:.1f} MB"
        )
        return sales_long

    def load_and_merge(self) -> pd.DataFrame:
        """
        Master loader: melt sales, join calendar + prices, return a single
        unified DataFrame with date, sales, price, and calendar metadata.

        Merge strategy:
            sales ──(d)──▶ calendar   → brings in date, weekday, events, SNAP
            merged ──(store_id, item_id, wm_yr_wk)──▶ prices → brings in sell_price
        """
        sales = self.load_and_melt_sales()
        calendar = self.load_calendar()
        prices = self.load_prices()

        # ── Merge 1: sales ↔ calendar on 'd' column ─────────────────────
        logger.info("Merging sales ↔ calendar on 'd'…")
        # Calendar has a 'd' column like "d_1", matching the melted var_name
        df = sales.merge(calendar, on="d", how="left")

        # ── Merge 2: result ↔ prices on (store_id, item_id, wm_yr_wk) ───
        logger.info("Merging ↔ prices on (store_id, item_id, wm_yr_wk)…")

        # Need to align category dtypes for merge keys
        for col in ["store_id", "item_id"]:
            if df[col].dtype.name == "category" and prices[col].dtype.name == "category":
                # Unify category codes so the merge doesn't silently drop rows
                common_cats = df[col].cat.categories.union(prices[col].cat.categories)
                df[col] = df[col].cat.set_categories(common_cats)
                prices[col] = prices[col].cat.set_categories(common_cats)

        df = df.merge(
            prices,
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left",
        )

        # ── Sort chronologically per series (critical for lag features) ──
        logger.info("Sorting by (item_id, store_id, date)…")
        df = df.sort_values(
            ["item_id", "store_id", self.config.date_col]
        ).reset_index(drop=True)

        # ── Final memory pass ────────────────────────────────────────────
        df = MemoryOptimizer.optimize(df)

        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        logger.info(
            f"Merged dataset: {df.shape[0]:,} rows × {df.shape[1]} cols "
            f"({mem_mb:.1f} MB)"
        )
        return df


# ════════════════════════════════════════════════════════════════════════════
# §4  FEATURE ENGINEER
# ════════════════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """
    Constructs all time-series features required by the TFT and DML engines.

    Feature groups
    ──────────────
    1. **Lag features** (sales_lag_1, …, sales_lag_30)
       Capture autoregressive demand signal at multiple horizons.

    2. **Rolling statistics** (sales_rmean_7, sales_rstd_7, …)
       Smooth demand trends and local volatility — crucial for the
       quantile/uncertainty modelling in the TFT and for the quant
       risk dashboard.

    3. **Calendar / categorical embeddings**
       day_of_week, day_of_month, month, week_of_year, is_weekend,
       event_type_*, SNAP flags — all known future covariates for TFT.

    4. **Price dynamics**
       sell_price, price_change_abs, price_change_pct, price_momentum_7d
       — the treatment variable for the causal engine and a known future
       covariate for the TFT.

    All features are computed per (item_id, store_id) group with proper
    chronological ordering to prevent information leakage.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    # ── §4a  Lag features ────────────────────────────────────────────────

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lagged versions of the target variable.

        For each lag L in config.lag_days, we compute:
            sales_lag_{L} = sales(t − L)

        Implementation uses groupby + shift, which respects the
        chronological sort order established in load_and_merge().

        Note: The first L rows of each group will be NaN — these are
        handled in the warmup cutoff later.
        """
        logger.info(f"Adding lag features: {self.config.lag_days}")
        group_keys = [self.config.item_col, self.config.store_col]
        target = self.config.target_col

        for lag in self.config.lag_days:
            col_name = f"{target}_lag_{lag}"
            df[col_name] = df.groupby(group_keys)[target].shift(lag)
            logger.debug(f"  Created {col_name}")

        return df

    # ── §4b  Rolling window features ─────────────────────────────────────

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling mean and rolling standard deviation of the target.

        For each window W in config.rolling_windows:
            sales_rmean_{W}  = mean of sales over the past W days
            sales_rstd_{W}   = std dev of sales over the past W days
            sales_rmin_{W}   = min of sales over the past W days
            sales_rmax_{W}   = max of sales over the past W days

        We use `min_periods=1` so that partial windows at the start of
        a series still produce values (albeit noisier), then rely on the
        warmup cutoff to remove the most unreliable rows.

        The rolling std (volatility) is the single most important feature
        for the uncertainty / quantile aspect of the forecasting engine.
        """
        logger.info(f"Adding rolling features: windows = {self.config.rolling_windows}")
        group_keys = [self.config.item_col, self.config.store_col]
        target = self.config.target_col

        for window in self.config.rolling_windows:
            # ── Rolling mean ─────────────────────────────────────────────
            col_mean = f"{target}_rmean_{window}"
            df[col_mean] = (
                df.groupby(group_keys)[target]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )

            # ── Rolling std (volatility) ─────────────────────────────────
            col_std = f"{target}_rstd_{window}"
            df[col_std] = (
                df.groupby(group_keys)[target]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
            )

            # ── Rolling min (demand floor) ───────────────────────────────
            col_min = f"{target}_rmin_{window}"
            df[col_min] = (
                df.groupby(group_keys)[target]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).min())
            )

            # ── Rolling max (demand ceiling) ─────────────────────────────
            col_max = f"{target}_rmax_{window}"
            df[col_max] = (
                df.groupby(group_keys)[target]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).max())
            )

            logger.debug(f"  Created {col_mean}, {col_std}, {col_min}, {col_max}")

        return df

    # ── §4c  Calendar / temporal features ────────────────────────────────

    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract cyclical and categorical time features from the date column.

        Features created:
            day_of_week    (0=Mon … 6=Sun)  — weekly seasonality
            day_of_month   (1–31)           — monthly effects
            week_of_year   (1–53)           — annual seasonality
            month          (1–12)           — seasonal macro-trends
            year           (2011–2016)      — long-term trend
            is_weekend     (0/1)            — binary weekend flag
            is_month_start (0/1)
            is_month_end   (0/1)

        These are all *known future covariates* in the TFT framework:
        we know the calendar at prediction time.
        """
        logger.info("Adding calendar/temporal features")
        dt = df[self.config.date_col]

        df["day_of_week"] = dt.dt.dayofweek.astype(np.int8)
        df["day_of_month"] = dt.dt.day.astype(np.int8)
        df["week_of_year"] = dt.dt.isocalendar().week.astype(np.int8)
        df["month"] = dt.dt.month.astype(np.int8)
        df["year"] = dt.dt.year.astype(np.int16)
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(np.int8)
        df["is_month_start"] = dt.dt.is_month_start.astype(np.int8)
        df["is_month_end"] = dt.dt.is_month_end.astype(np.int8)

        return df

    # ── §4d  Price dynamics features ─────────────────────────────────────

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct price-related features that are critical for both the TFT
        (known future covariate) and the Causal Engine (treatment variable).

        Features:
            sell_price           — already merged from sell_prices.csv
            price_change_abs     — absolute Δ from previous week's price
            price_change_pct     — percentage Δ (the ΔP in D_final = D_base × (1 + τ·ΔP))
            price_momentum_7d    — 7-day rolling mean of price (smooth trend)
            price_rel_to_dept    — item price / dept mean price (relative positioning)

        Missing prices (items not yet on shelf) are forward-filled within
        each (item, store) group, then zero-filled as a last resort.
        """
        logger.info("Adding price dynamics features")
        group_keys = [self.config.item_col, self.config.store_col]

        # ── Forward-fill missing prices within each item-store series ────
        df["sell_price"] = df.groupby(group_keys)["sell_price"].transform(
            lambda x: x.ffill()
        )

        # ── Absolute price change ────────────────────────────────────────
        df["price_change_abs"] = df.groupby(group_keys)["sell_price"].diff()

        # ── Percentage price change (guard against division by zero) ─────
        prev_price = df.groupby(group_keys)["sell_price"].shift(1)
        df["price_change_pct"] = df["price_change_abs"] / prev_price.replace(0, np.nan)

        # ── 7-day price momentum (smoothed price trend) ──────────────────
        df["price_momentum_7d"] = df.groupby(group_keys)["sell_price"].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )

        # ── Relative price: item price vs. department average ────────────
        #    This captures whether an item is premium or value within its
        #    department — a powerful demand driver.
        if "dept_id" in df.columns:
            dept_mean = df.groupby(["dept_id", self.config.date_col])[
                "sell_price"
            ].transform("mean")
            df["price_rel_to_dept"] = df["sell_price"] / dept_mean.replace(0, np.nan)

        return df

    # ── §4e  SNAP features ──────────────────────────────────────────────

    def add_snap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a unified SNAP (food stamps) indicator per row.

        The raw calendar has separate snap_CA, snap_TX, snap_WI columns.
        We map each row to the appropriate state's SNAP flag.
        """
        logger.info("Adding unified SNAP indicator")
        snap_map = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}

        # Default to 0 (no SNAP)
        df["snap_eligible"] = np.int8(0)

        for state, col in snap_map.items():
            if col in df.columns and "state_id" in df.columns:
                mask = df["state_id"] == state
                df.loc[mask, "snap_eligible"] = df.loc[mask, col].astype(np.int8)

        return df

    # ── §4f  NaN handling & warmup cutoff ────────────────────────────────

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Multi-stage NaN treatment:

        1. Forward-fill within groups for price and rolling features
           (reasonable: carry the last known value forward).
        2. Fill remaining NaNs with 0 for numeric columns (safe default
           for sales-related features — zero demand is a valid state).
        3. Drop the warmup window (first `warmup_days` rows per group)
           where lag features are structurally undefined.

        This ordering ensures we never train on fabricated data while
        still maximising usable rows.
        """
        logger.info("Handling missing values")

        # ── Stage 1: forward-fill price columns within each series ───────
        price_cols = [c for c in df.columns if "price" in c.lower()]
        group_keys = [self.config.item_col, self.config.store_col]
        for col in price_cols:
            df[col] = df.groupby(group_keys)[col].transform(lambda x: x.ffill())

        # ── Stage 2: fill remaining numeric NaNs with 0 ─────────────────
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # ── Stage 3: fill remaining categorical/string NaNs ──────────────
        cat_cols = df.select_dtypes(include=["category"]).columns
        for col in cat_cols:
            if df[col].isna().any():
                df[col] = df[col].cat.add_categories("Unknown").fillna("Unknown")

        # ── Stage 4: drop warmup window ──────────────────────────────────
        if self.config.warmup_days > 0:
            logger.info(
                f"Dropping first {self.config.warmup_days} days per series (warmup)"
            )
            # Create a within-group row counter
            df["_row_num"] = df.groupby(group_keys).cumcount()
            before = len(df)
            df = df[df["_row_num"] >= self.config.warmup_days].copy()
            df.drop(columns=["_row_num"], inplace=True)
            logger.info(f"Dropped {before - len(df):,} warmup rows")

        return df

    # ── §4g  Master feature engineering pipeline ─────────────────────────

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrate all feature engineering steps in the correct order.

        Order matters:
            1. Calendar features (no dependencies)
            2. SNAP features (depends on state_id + calendar merge)
            3. Price features (depends on sell_price merge)
            4. Lag features (depends on target, must come before rolling)
            5. Rolling features (operates on target, uses shift(1) internally)
            6. NaN handling (must come last to catch all generated NaNs)
        """
        t0 = time.perf_counter()
        logger.info("═" * 60)
        logger.info("Starting feature engineering pipeline")
        logger.info("═" * 60)

        df = self.add_calendar_features(df)
        df = self.add_snap_features(df)
        df = self.add_price_features(df)
        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)
        df = self.handle_missing_values(df)

        # ── Final memory optimisation ────────────────────────────────────
        df = MemoryOptimizer.optimize(df)

        elapsed = time.perf_counter() - t0
        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        logger.info("═" * 60)
        logger.info(
            f"Feature engineering complete: "
            f"{df.shape[0]:,} rows × {df.shape[1]} cols "
            f"({mem_mb:.1f} MB) in {elapsed:.1f}s"
        )
        logger.info("═" * 60)

        return df


# ════════════════════════════════════════════════════════════════════════════
# §5  TIME-SERIES TRAIN/TEST SPLITTER
# ════════════════════════════════════════════════════════════════════════════

class TimeSeriesSplitter:
    """
    Strict temporal train/test splitter for time-series data.

    ╔═══════════════════════════════════════════════════════════════╗
    ║  *** NO RANDOM SPLITTING — EVER ***                         ║
    ║                                                             ║
    ║  Random splits in time-series cause catastrophic data       ║
    ║  leakage: the model sees future sales in training and       ║
    ║  produces artificially inflated accuracy metrics.           ║
    ║                                                             ║
    ║  We enforce a hard calendar cutoff:                         ║
    ║    train = all rows with date  < cutoff_date                ║
    ║    test  = all rows with date >= cutoff_date                ║
    ╚═══════════════════════════════════════════════════════════════╝

    Supports two modes:
        1. Single split:    one train/test pair at a fixed cutoff.
        2. Rolling window:  multiple folds for time-series cross-validation,
                            each fold advancing the cutoff forward by
                            `step_days`.  Used in Phase 4 backtesting.

    Parameters
    ----------
    date_col : str
        Name of the datetime column.
    test_days : int
        Number of days in each test window.
    n_folds : int
        Number of rolling-window folds (1 = single split).
    step_days : int
        How many days to advance the cutoff between folds.
    min_train_days : int
        Minimum number of training days required.  If a fold would
        produce fewer training days, it is skipped with a warning.
    embargo_days : int
        Gap between train and test to prevent label leakage from
        features that span across the cutoff (e.g., rolling windows).
        Similar to the "purge" concept in financial ML (de Prado).
    """

    def __init__(
        self,
        date_col: str = "date",
        test_days: int = 28,
        n_folds: int = 1,
        step_days: int = 28,
        min_train_days: int = 365,
        embargo_days: int = 0,
    ) -> None:
        self.date_col = date_col
        self.test_days = test_days
        self.n_folds = n_folds
        self.step_days = step_days
        self.min_train_days = min_train_days
        self.embargo_days = embargo_days

    def _validate_dates(self, df: pd.DataFrame) -> None:
        """Ensure the date column exists and is datetime."""
        if self.date_col not in df.columns:
            raise ValueError(f"Date column '{self.date_col}' not found in DataFrame")
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
            raise TypeError(
                f"Column '{self.date_col}' must be datetime64, "
                f"got {df[self.date_col].dtype}"
            )

    def single_split(
        self,
        df: pd.DataFrame,
        cutoff_date: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform a single temporal train/test split.

        Parameters
        ----------
        df : DataFrame
            The full feature-engineered dataset.
        cutoff_date : str or None
            ISO date string (e.g., '2016-04-25').  If None, the cutoff is
            automatically set to (max_date − test_days).

        Returns
        -------
        (train_df, test_df) : tuple of DataFrames
        """
        self._validate_dates(df)

        if cutoff_date is None:
            max_date = df[self.date_col].max()
            cutoff = max_date - pd.Timedelta(days=self.test_days)
        else:
            cutoff = pd.Timestamp(cutoff_date)

        # ── Apply embargo: remove `embargo_days` before the cutoff ───────
        train_end = cutoff - pd.Timedelta(days=self.embargo_days)

        train_mask = df[self.date_col] < train_end
        test_mask = (df[self.date_col] >= cutoff) & (
            df[self.date_col] < cutoff + pd.Timedelta(days=self.test_days)
        )

        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()

        # ── Validate minimum training window ─────────────────────────────
        actual_train_days = (
            train_df[self.date_col].max() - train_df[self.date_col].min()
        ).days
        if actual_train_days < self.min_train_days:
            logger.warning(
                f"Training window ({actual_train_days}d) is shorter than "
                f"min_train_days ({self.min_train_days}d)"
            )

        logger.info(
            f"Split @ {cutoff.date()} | "
            f"Train: {len(train_df):,} rows "
            f"({train_df[self.date_col].min().date()} to {train_df[self.date_col].max().date()}) | "
            f"Test: {len(test_df):,} rows "
            f"({test_df[self.date_col].min().date()} to {test_df[self.date_col].max().date()})"
        )

        return train_df, test_df

    def rolling_split(
        self, df: pd.DataFrame
    ) -> List[Tuple[int, pd.DataFrame, pd.DataFrame]]:
        """
        Generate multiple train/test folds by advancing the cutoff.

        Yields (fold_index, train_df, test_df) tuples.

        The last fold's test window ends at the dataset's max date.
        Earlier folds are obtained by stepping backward.

        Timeline visualisation (n_folds=3, step=28d, test=28d):

        ├──────── Train 1 ─────────┤ emb ├─ Test 1 ─┤
        ├────────── Train 2 ────────────┤ emb ├─ Test 2 ─┤
        ├──────────── Train 3 ──────────────┤ emb ├─ Test 3 ─┤
        """
        self._validate_dates(df)

        max_date = df[self.date_col].max()
        folds = []

        for i in range(self.n_folds):
            # Walk backward from the end of the dataset
            offset = (self.n_folds - 1 - i) * self.step_days
            fold_cutoff = max_date - pd.Timedelta(
                days=self.test_days + offset
            )

            logger.info(f"─── Fold {i + 1}/{self.n_folds} ───")
            train_df, test_df = self.single_split(
                df, cutoff_date=str(fold_cutoff.date())
            )

            if len(test_df) == 0:
                logger.warning(f"Fold {i + 1} has empty test set — skipping")
                continue

            folds.append((i + 1, train_df, test_df))

        logger.info(f"Generated {len(folds)} valid folds")
        return folds

    def get_split_summary(self, df: pd.DataFrame) -> Dict:
        """
        Return a diagnostic summary of the split configuration
        without actually performing the split.
        """
        self._validate_dates(df)

        min_date = df[self.date_col].min()
        max_date = df[self.date_col].max()
        total_days = (max_date - min_date).days

        return {
            "date_range": f"{min_date.date()} to {max_date.date()}",
            "total_days": total_days,
            "test_days": self.test_days,
            "n_folds": self.n_folds,
            "step_days": self.step_days,
            "embargo_days": self.embargo_days,
            "min_train_days": self.min_train_days,
            "estimated_train_days": total_days - self.test_days - self.embargo_days,
        }


# ════════════════════════════════════════════════════════════════════════════
# §6  MASTER PIPELINE ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════

class DemandForecastingPipeline:
    """
    Top-level orchestrator that chains:
        DataLoader → FeatureEngineer → TimeSeriesSplitter → save

    This is the single entry point for Phase 1.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.loader = M5DataLoader(self.config)
        self.engineer = FeatureEngineer(self.config)
        self.splitter = TimeSeriesSplitter(
            date_col=self.config.date_col,
            test_days=28,       # M5 competition uses 28-day horizon
            n_folds=1,          # single split for Phase 1; rolling in Phase 4
            embargo_days=1,     # 1-day gap to avoid rolling-window bleed
        )

    def run(self, save: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the full Phase 1 pipeline.

        Returns
        -------
        (train_df, test_df) : tuple of DataFrames
            Ready for Phase 2 (TFT) and Phase 3 (DML).
        """
        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║  PHASE 1: Data Preprocessing & Feature Engineering        ║")
        logger.info("╚" + "═" * 58 + "╝")
        t_start = time.perf_counter()

        # ── Step 1: Load and merge raw data ──────────────────────────────
        df = self.loader.load_and_merge()

        # ── Step 2: Engineer features ────────────────────────────────────
        df = self.engineer.engineer_features(df)

        # ── Step 3: Print feature inventory ──────────────────────────────
        self._print_feature_inventory(df)

        # ── Step 4: Train/test split ─────────────────────────────────────
        train_df, test_df = self.splitter.single_split(df)

        # ── Step 5: Save to parquet ──────────────────────────────────────
        if save:
            self._save(train_df, test_df, df)

        total_time = time.perf_counter() - t_start
        logger.info(f"Phase 1 complete in {total_time:.1f}s")
        logger.info(
            f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows"
        )

        return train_df, test_df

    def _print_feature_inventory(self, df: pd.DataFrame) -> None:
        """Log a structured summary of all engineered features."""
        logger.info("\n┌─ Feature Inventory ─────────────────────────────────┐")

        # Group features by type
        lag_feats = [c for c in df.columns if "_lag_" in c]
        roll_feats = [c for c in df.columns if "_rmean_" in c or "_rstd_" in c
                      or "_rmin_" in c or "_rmax_" in c]
        price_feats = [c for c in df.columns if "price" in c.lower()]
        cal_feats = [
            "day_of_week", "day_of_month", "week_of_year", "month",
            "year", "is_weekend", "is_month_start", "is_month_end",
        ]
        cal_feats = [c for c in cal_feats if c in df.columns]

        logger.info(f"│  Target:       {self.config.target_col}")
        logger.info(f"│  Lag features: {len(lag_feats):>3}  {lag_feats}")
        logger.info(f"│  Rolling feat: {len(roll_feats):>3}  {roll_feats}")
        logger.info(f"│  Price feat:   {len(price_feats):>3}  {price_feats}")
        logger.info(f"│  Calendar:     {len(cal_feats):>3}  {cal_feats}")
        logger.info(f"│  Total cols:   {len(df.columns):>3}")
        logger.info("└─────────────────────────────────────────────────────┘")

    def _save(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        full_df: pd.DataFrame,
    ) -> None:
        """
        Persist DataFrames to compressed Parquet files.

        Parquet is chosen over CSV because:
          • 5–10× smaller on disk (columnar + snappy compression)
          • Preserves dtypes (no re-parsing categoricals / dates)
          • 10× faster to read back
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "train": self.config.output_dir / "train.parquet",
            "test": self.config.output_dir / "test.parquet",
            "full": self.config.output_dir / "full_featured.parquet",
        }

        for name, path in paths.items():
            data = {"train": train_df, "test": test_df, "full": full_df}[name]
            data.to_parquet(path, engine="pyarrow", compression="snappy")
            size_mb = path.stat().st_size / 1e6
            logger.info(f"Saved {name}: {path} ({size_mb:.1f} MB)")


# ════════════════════════════════════════════════════════════════════════════
# §7  CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1: M5 Data Preprocessing & Feature Engineering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full dataset
  python data_pipeline.py --data-dir data/raw --output-dir data/processed

  # Quick prototype with 1%% of items
  python data_pipeline.py --data-dir data/raw --sample-frac 0.01

  # Custom lags
  python data_pipeline.py --data-dir data/raw --lags 1 7 14 28 56
        """,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw M5 CSVs (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed parquet files (default: data/processed)",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Fraction of item-stores to sample for prototyping (e.g., 0.01)",
    )
    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=[1, 7, 14, 30],
        help="Lag days for target variable (default: 1 7 14 30)",
    )
    parser.add_argument(
        "--rolling-windows",
        type=int,
        nargs="+",
        default=[7, 30],
        help="Rolling window sizes (default: 7 30)",
    )
    parser.add_argument(
        "--warmup-days",
        type=int,
        default=30,
        help="Warmup days to drop per series (default: 30)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    config = PipelineConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        sample_frac=args.sample_frac,
        lag_days=args.lags,
        rolling_windows=args.rolling_windows,
        warmup_days=args.warmup_days,
    )

    pipeline = DemandForecastingPipeline(config)
    pipeline.run(save=True)


if __name__ == "__main__":
    main()

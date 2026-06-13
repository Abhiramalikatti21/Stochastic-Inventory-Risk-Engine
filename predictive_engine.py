"""
================================================================================
Phase 2: Predictive Engine -- Temporal Fusion Transformer (TFT)
================================================================================

Prescriptive Demand Forecasting Engine
--------------------------------------
Builds, trains, and runs inference with a Temporal Fusion Transformer that
produces multi-horizon **quantile** forecasts (p10, p50, p90) for unit sales.

Architecture Overview:
    Phase 1 (data_pipeline.py)
        |
        v  full_featured.parquet (45 cols)
    Phase 2 (this file)
        |
        +-- DatasetBuilder   : DataFrame -> TimeSeriesDataSet + DataLoaders
        +-- ModelBuilder     : TimeSeriesDataSet -> TFT(QuantileLoss)
        +-- TrainingPipeline : LR Finder + Trainer(EarlyStopping, TensorBoard)
        +-- TFTPredictor     : model.predict() -> p10/p50/p90 DataFrame
        +-- PredictiveEngine : Orchestrator
        |
        v  predictions.parquet  (item, store, date, p10, p50, p90)
           -> consumed by Phase 3 (Causal Engine) as D_base

Quantile Loss Rationale:
    QuantileLoss([0.1, 0.5, 0.9]) forces the network to output three
    prediction heads per horizon step.  The resulting fan chart
    (p10 = downside, p50 = central, p90 = upside) directly feeds:
      - Phase 3's D_base term in  D_final = D_base * (1 + tau * dP)
      - Phase 5's risk dashboard with VaR-style uncertainty bands

Usage:
    python predictive_engine.py --data-path data/processed/full_featured.parquet

Dependencies:
    pip install torch pytorch-forecasting pytorch-lightning tensorboard

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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---- Suppress noisy warnings before heavy imports --------------------------
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")

# ---- Deep-learning imports with graceful fallback --------------------------
try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is required for Phase 2.  Install via:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "  (or the CUDA variant for GPU acceleration)"
    )

try:
    # pytorch-lightning 2.x preferred import path
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.tuner import Tuner
except ImportError:
    try:
        # Fallback to older pytorch_lightning namespace
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import (
            EarlyStopping,
            LearningRateMonitor,
            ModelCheckpoint,
        )
        from pytorch_lightning.loggers import TensorBoardLogger
        from pytorch_lightning.tuner import Tuner
    except ImportError:
        raise ImportError(
            "PyTorch Lightning is required.  Install via:\n"
            "  pip install lightning   (or)   pip install pytorch-lightning"
        )

try:
    from pytorch_forecasting import (
        Baseline,
        TemporalFusionTransformer,
        TimeSeriesDataSet,
    )
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
except ImportError:
    raise ImportError(
        "pytorch-forecasting is required.  Install via:\n"
        "  pip install pytorch-forecasting"
    )

# ---- Logging ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# S1  CONFIGURATION
# ============================================================================

@dataclass
class TFTConfig:
    """
    Centralised configuration for every knob in the TFT pipeline.

    Groups
    ------
    Data:   paths, column definitions, sample controls
    Model:  TFT architecture hyperparameters
    Train:  optimiser, scheduler, early-stopping, hardware
    """

    # ---- Data paths --------------------------------------------------------
    data_path: Path = Path("data/processed/full_featured.parquet")
    output_dir: Path = Path("models/tft")
    log_dir: Path = Path("logs/tft")

    # ---- Forecasting horizons ----------------------------------------------
    max_prediction_length: int = 30   # days ahead to forecast
    max_encoder_length: int = 90      # days of history the encoder sees
    min_encoder_length: int = 30      # allow shorter sequences at series start

    # ---- Column taxonomy (must match Phase 1 output) -----------------------
    target: str = "sales"
    time_idx: str = "time_idx"        # will be created from 'd' column
    group_ids: List[str] = field(
        default_factory=lambda: ["item_id", "store_id"]
    )

    static_categoricals: List[str] = field(
        default_factory=lambda: [
            "item_id", "store_id", "dept_id", "cat_id", "state_id",
        ]
    )

    time_varying_known_categoricals: List[str] = field(
        default_factory=lambda: [
            "event_type_1", "event_type_2",
        ]
    )

    time_varying_known_reals: List[str] = field(
        default_factory=lambda: [
            "sell_price",
            "price_change_abs", "price_change_pct",
            "price_momentum_7d", "price_rel_to_dept",
            "day_of_week", "day_of_month", "week_of_year",
            "month", "year",
            "is_weekend", "is_month_start", "is_month_end",
            "has_event", "snap_eligible",
        ]
    )

    # NOTE: The target ("sales") is automatically added by TimeSeriesDataSet.
    # Lag/rolling features are "unknown" -- only available in the encoder
    # (historical) window.  The TFT masks them in the decoder (future) window,
    # so no leakage occurs even though the DataFrame has values for these
    # columns in the test period.
    time_varying_unknown_reals: List[str] = field(
        default_factory=lambda: [
            "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_30",
            "sales_rmean_7", "sales_rstd_7", "sales_rmin_7", "sales_rmax_7",
            "sales_rmean_30", "sales_rstd_30", "sales_rmin_30", "sales_rmax_30",
        ]
    )

    # ---- Model architecture ------------------------------------------------
    hidden_size: int = 64             # width of the TFT's GRN blocks
    attention_head_size: int = 4      # multi-head attention heads
    dropout: float = 0.1
    hidden_continuous_size: int = 32  # embedding dim for continuous vars
    output_size: int = 3              # one per quantile (p10, p50, p90)

    # ---- Quantiles ---------------------------------------------------------
    quantiles: List[float] = field(
        default_factory=lambda: [0.1, 0.5, 0.9]
    )

    # ---- Training ----------------------------------------------------------
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 50
    gradient_clip_val: float = 0.1
    patience: int = 5                 # early-stopping patience (val loss)
    reduce_on_plateau_patience: int = 3
    num_workers: int = 0              # 0 for Windows compat; increase on Linux

    # ---- Hardware ----------------------------------------------------------
    accelerator: str = "auto"         # "auto" picks GPU if available
    precision: str = "32-true"        # "16-mixed" for faster GPU training

    # ---- Prototyping -------------------------------------------------------
    sample_n_items: Optional[int] = None   # subsample N item-store combos

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        self.output_dir = Path(self.output_dir)
        self.log_dir = Path(self.log_dir)


# ============================================================================
# S2  DATASET BUILDER
# ============================================================================

class DatasetBuilder:
    """
    Converts a Phase-1-processed pandas DataFrame into pytorch-forecasting
    TimeSeriesDataSet objects and DataLoaders.

    Responsibilities
    ----------------
    1. Create a monotonic integer `time_idx` from the M5 'd' column.
    2. Cast all categorical columns to `str` (required by TimeSeriesDataSet).
    3. Upcast numerical columns to float32 (PyTorch requirement).
    4. Split training data into fit/validation by temporal cutoff.
    5. Construct test dataset with proper encoder context.
    """

    def __init__(self, config: TFTConfig) -> None:
        self.config = config

    # ---- S2a  DataFrame preparation ----------------------------------------

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform Phase 1 output into the format TimeSeriesDataSet expects.

        Steps:
            1. Create integer time_idx from the 'd' column (d_31 -> 31).
            2. Convert categorical columns to plain strings.
            3. Ensure all numeric feature columns are float32.
            4. Sort by group + time (already done in Phase 1, but safety check).
            5. Optionally subsample items for fast prototyping.
        """
        logger.info("Preparing DataFrame for TimeSeriesDataSet")
        df = df.copy()

        # ---- Step 1: time_idx from 'd' column ------------------------------
        if "d" in df.columns:
            df[self.config.time_idx] = (
                df["d"].astype(str).str.replace("d_", "", regex=False).astype(int)
            )
            logger.info(
                f"Created time_idx: range [{df[self.config.time_idx].min()}, "
                f"{df[self.config.time_idx].max()}]"
            )
        elif self.config.time_idx not in df.columns:
            # Fallback: create from date
            min_date = df["date"].min()
            df[self.config.time_idx] = (df["date"] - min_date).dt.days
            logger.info("Created time_idx from date column (fallback)")

        # ---- Step 2: categoricals -> str ------------------------------------
        all_categoricals = (
            self.config.static_categoricals
            + self.config.time_varying_known_categoricals
        )
        for col in all_categoricals:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # ---- Step 3: numerics -> float32 ------------------------------------
        all_reals = (
            self.config.time_varying_known_reals
            + self.config.time_varying_unknown_reals
            + [self.config.target]
        )
        for col in all_reals:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)

        # ---- Step 4: sort ---------------------------------------------------
        df = df.sort_values(
            self.config.group_ids + [self.config.time_idx]
        ).reset_index(drop=True)

        # ---- Step 5: subsample for prototyping ------------------------------
        if self.config.sample_n_items is not None:
            df = self._subsample(df, self.config.sample_n_items)

        logger.info(
            f"Prepared: {df.shape[0]:,} rows, {df.shape[1]} cols, "
            f"{df[self.config.group_ids[0]].nunique()} unique items"
        )
        return df

    def _subsample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Randomly select n unique item-store combinations."""
        groups = df.groupby(self.config.group_ids).ngroups
        if n >= groups:
            return df

        # Get unique group tuples, sample, filter
        unique_groups = (
            df[self.config.group_ids]
            .drop_duplicates()
            .sample(n=n, random_state=42)
        )
        df = df.merge(unique_groups, on=self.config.group_ids, how="inner")
        logger.info(f"Subsampled to {n}/{groups} item-store groups")
        return df

    # ---- S2b  Filter to available columns ----------------------------------

    def _filter_columns(
        self, df: pd.DataFrame, col_list: List[str]
    ) -> List[str]:
        """Return only the columns that exist in the DataFrame."""
        available = [c for c in col_list if c in df.columns]
        missing = set(col_list) - set(available)
        if missing:
            logger.warning(f"Columns not in DataFrame (skipped): {missing}")
        return available

    # ---- S2c  Build TimeSeriesDataSets -------------------------------------

    def build_training_dataset(
        self, df: pd.DataFrame
    ) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """
        Build training and validation TimeSeriesDataSets from the training
        DataFrame.

        The validation set is carved from the TAIL of the training data:
            training_cutoff = max(time_idx) - max_prediction_length

        All series with time_idx <= training_cutoff go to fit;
        the remaining form the validation target window.

        Returns
        -------
        (training_dataset, validation_dataset)
        """
        cfg = self.config

        # ---- Temporal cutoff for internal validation ------------------------
        training_cutoff = df[cfg.time_idx].max() - cfg.max_prediction_length
        logger.info(
            f"Training cutoff at time_idx={training_cutoff} "
            f"(max={df[cfg.time_idx].max()}, "
            f"pred_len={cfg.max_prediction_length})"
        )

        # ---- Filter column lists to what's actually available ---------------
        static_cats = self._filter_columns(df, cfg.static_categoricals)
        tv_known_cats = self._filter_columns(df, cfg.time_varying_known_categoricals)
        tv_known_reals = self._filter_columns(df, cfg.time_varying_known_reals)
        tv_unknown_reals = self._filter_columns(df, cfg.time_varying_unknown_reals)

        # ---- Build training dataset -----------------------------------------
        logger.info("Building training TimeSeriesDataSet...")
        training = TimeSeriesDataSet(
            df[df[cfg.time_idx] <= training_cutoff],
            time_idx=cfg.time_idx,
            target=cfg.target,
            group_ids=cfg.group_ids,
            max_encoder_length=cfg.max_encoder_length,
            min_encoder_length=cfg.min_encoder_length,
            max_prediction_length=cfg.max_prediction_length,
            static_categoricals=static_cats,
            time_varying_known_categoricals=tv_known_cats,
            time_varying_known_reals=tv_known_reals,
            time_varying_unknown_reals=tv_unknown_reals,
            target_normalizer=GroupNormalizer(
                groups=cfg.group_ids,
                transformation="softplus",  # handles zeros in sales
            ),
            add_relative_time_idx=True,      # relative position feature
            add_target_scales=True,           # exposes group-level scale
            add_encoder_length=True,          # tells model how much context it has
            allow_missing_timesteps=True,     # robust to gaps
        )
        logger.info(
            f"Training dataset: {len(training)} samples, "
            f"encoder=[{cfg.min_encoder_length}..{cfg.max_encoder_length}], "
            f"decoder={cfg.max_prediction_length}"
        )

        # ---- Build validation dataset from the same schema ------------------
        logger.info("Building validation TimeSeriesDataSet...")
        validation = TimeSeriesDataSet.from_dataset(
            training,
            df,
            min_prediction_idx=training_cutoff + 1,
        )
        logger.info(f"Validation dataset: {len(validation)} samples")

        return training, validation

    def build_test_dataset(
        self,
        training_dataset: TimeSeriesDataSet,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> TimeSeriesDataSet:
        """
        Build a test TimeSeriesDataSet for out-of-sample inference.

        We need encoder context from the TAIL of the training data
        (last max_encoder_length rows per group) concatenated with the
        test data.  This ensures the model has historical context for
        generating predictions over the test horizon.
        """
        cfg = self.config

        # Grab encoder context from training tail
        max_train_idx = train_df[cfg.time_idx].max()
        encoder_start = max_train_idx - cfg.max_encoder_length
        encoder_context = train_df[
            train_df[cfg.time_idx] > encoder_start
        ].copy()

        # Concatenate encoder context + test data
        test_with_context = pd.concat(
            [encoder_context, test_df], ignore_index=True
        )
        test_with_context = test_with_context.sort_values(
            cfg.group_ids + [cfg.time_idx]
        ).reset_index(drop=True)

        # Build dataset from training schema
        test_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset,
            test_with_context,
            predict=True,
            stop_randomization=True,
        )
        logger.info(f"Test dataset: {len(test_dataset)} samples")
        return test_dataset

    def create_dataloaders(
        self,
        training: TimeSeriesDataSet,
        validation: TimeSeriesDataSet,
    ) -> Tuple[Any, Any]:
        """Create training and validation DataLoaders."""
        cfg = self.config

        train_dl = training.to_dataloader(
            train=True,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            persistent_workers=cfg.num_workers > 0,
        )
        val_dl = validation.to_dataloader(
            train=False,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            persistent_workers=cfg.num_workers > 0,
        )

        logger.info(
            f"DataLoaders created: batch_size={cfg.batch_size}, "
            f"train_batches={len(train_dl)}, val_batches={len(val_dl)}"
        )
        return train_dl, val_dl


# ============================================================================
# S3  MODEL BUILDER
# ============================================================================

class ModelBuilder:
    """
    Instantiates the Temporal Fusion Transformer with QuantileLoss.

    The TFT architecture (Lim et al., 2021) combines:
      - Variable Selection Networks   (automatic feature importance)
      - Gated Residual Networks        (non-linear feature processing)
      - Multi-Head Attention           (long-range temporal dependencies)
      - Quantile output heads          (probabilistic forecasts)

    QuantileLoss([0.1, 0.5, 0.9]) produces three prediction bands:
      p10 = 10th percentile (downside / VaR-like risk bound)
      p50 = 50th percentile (central forecast / median)
      p90 = 90th percentile (upside / demand ceiling)
    """

    def __init__(self, config: TFTConfig) -> None:
        self.config = config

    def build(self, dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
        """
        Create a TFT model from the training dataset's metadata.

        TimeSeriesDataSet.from_dataset() embeds the full variable schema,
        so TemporalFusionTransformer.from_dataset() can auto-configure
        embedding dimensions, input sizes, and output heads.
        """
        cfg = self.config

        model = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=cfg.learning_rate,
            hidden_size=cfg.hidden_size,
            attention_head_size=cfg.attention_head_size,
            dropout=cfg.dropout,
            hidden_continuous_size=cfg.hidden_continuous_size,
            output_size=cfg.output_size,  # 3 quantiles
            loss=QuantileLoss(quantiles=cfg.quantiles),
            reduce_on_plateau_patience=cfg.reduce_on_plateau_patience,
            log_interval=10,
            log_val_interval=1,
        )

        # ---- Log model summary ----------------------------------------------
        n_params = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"TFT instantiated: {n_params:,} params ({n_train:,} trainable)"
        )
        logger.info(f"  Loss:       QuantileLoss({cfg.quantiles})")
        logger.info(f"  Hidden:     {cfg.hidden_size}")
        logger.info(f"  Heads:      {cfg.attention_head_size}")
        logger.info(f"  Dropout:    {cfg.dropout}")
        logger.info(f"  Output:     {cfg.output_size} heads (p10/p50/p90)")

        return model

    @staticmethod
    def compute_baseline(validation_dataset: TimeSeriesDataSet, val_dl) -> float:
        """
        Compute a naive baseline (last-value repeat) for comparison.

        This gives us a sanity-check metric: the TFT should beat this
        trivially if training is working correctly.
        """
        try:
            baseline = Baseline()
            baseline_preds = baseline.predict(val_dl, return_y=True)
            baseline_mae = (
                (baseline_preds.output - baseline_preds.y[0])
                .abs()
                .mean()
                .item()
            )
            logger.info(f"Naive baseline MAE: {baseline_mae:.4f}")
            return baseline_mae
        except Exception as e:
            logger.warning(f"Baseline computation failed: {e}")
            return float("nan")


# ============================================================================
# S4  TRAINING PIPELINE
# ============================================================================

class TrainingPipeline:
    """
    Manages the full training lifecycle:
        1. Hardware detection (GPU/CPU/MPS)
        2. Callback setup (EarlyStopping, LR Monitor, Checkpointing)
        3. TensorBoard logger
        4. Optional LR Finder (Tuner)
        5. model.fit() via pl.Trainer
    """

    def __init__(self, config: TFTConfig) -> None:
        self.config = config

    # ---- S4a  Hardware detection -------------------------------------------

    @staticmethod
    def detect_accelerator() -> Tuple[str, int]:
        """
        Auto-detect the best available accelerator.

        Returns (accelerator_str, device_count).
        """
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1e9
            logger.info(f"GPU detected: {name} ({vram:.1f} GB VRAM)")
            return "gpu", torch.cuda.device_count()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple MPS detected")
            return "mps", 1
        else:
            logger.info("No GPU detected -- using CPU (training will be slow)")
            return "cpu", 1

    # ---- S4b  Callbacks ----------------------------------------------------

    def _build_callbacks(self) -> List:
        """Construct the callback stack."""
        cfg = self.config

        callbacks = [
            # ---- Early Stopping: halt when val_loss plateaus ----------------
            EarlyStopping(
                monitor="val_loss",
                patience=cfg.patience,
                verbose=True,
                mode="min",
                min_delta=1e-4,
            ),

            # ---- LR Monitor: log learning rate to TensorBoard ---------------
            LearningRateMonitor(logging_interval="epoch"),

            # ---- Model Checkpoint: save best model by val_loss --------------
            ModelCheckpoint(
                dirpath=str(cfg.output_dir / "checkpoints"),
                filename="tft-{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                verbose=True,
            ),
        ]
        return callbacks

    # ---- S4c  Trainer setup ------------------------------------------------

    def setup_trainer(self) -> pl.Trainer:
        """
        Configure the PyTorch Lightning Trainer.

        Key settings:
            - gradient_clip_val: prevents exploding gradients in attention
            - enable_model_summary: prints layer shapes at start
            - deterministic: reproducibility (slight speed cost)
        """
        cfg = self.config
        accelerator, devices = self.detect_accelerator()

        # Override if user specified
        if cfg.accelerator != "auto":
            accelerator = cfg.accelerator

        # Create log directory
        tb_logger = TensorBoardLogger(
            save_dir=str(cfg.log_dir),
            name="tft_experiment",
        )

        trainer = pl.Trainer(
            max_epochs=cfg.max_epochs,
            accelerator=accelerator,
            devices=1,  # single device for simplicity
            gradient_clip_val=cfg.gradient_clip_val,
            callbacks=self._build_callbacks(),
            logger=tb_logger,
            enable_model_summary=True,
            precision=cfg.precision,
            log_every_n_steps=10,
        )

        logger.info(
            f"Trainer ready: max_epochs={cfg.max_epochs}, "
            f"accelerator={accelerator}, "
            f"precision={cfg.precision}, "
            f"grad_clip={cfg.gradient_clip_val}"
        )
        return trainer

    # ---- S4d  Learning Rate Finder -----------------------------------------

    def find_learning_rate(
        self,
        model: TemporalFusionTransformer,
        train_dl,
        val_dl,
    ) -> float:
        """
        Use PyTorch Lightning's Tuner to sweep learning rates and find
        the optimal starting LR (the steepest point on the loss-vs-LR curve).

        Falls back to the configured default if the finder fails.
        """
        logger.info("=" * 60)
        logger.info("Running Learning Rate Finder...")
        logger.info("=" * 60)

        try:
            # Create a temporary trainer for LR finding
            temp_trainer = pl.Trainer(
                accelerator=self.detect_accelerator()[0]
                if self.config.accelerator == "auto"
                else self.config.accelerator,
                devices=1,
                gradient_clip_val=self.config.gradient_clip_val,
                max_epochs=1,
                enable_model_summary=False,
                logger=False,
                enable_checkpointing=False,
            )

            tuner = Tuner(temp_trainer)
            lr_finder = tuner.lr_find(
                model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl,
                min_lr=1e-6,
                max_lr=1.0,
                num_training=100,
            )

            suggested_lr = lr_finder.suggestion()

            if suggested_lr is None or suggested_lr <= 0:
                logger.warning(
                    "LR finder returned invalid result; "
                    f"using default lr={self.config.learning_rate}"
                )
                return self.config.learning_rate

            logger.info(f"Suggested learning rate: {suggested_lr:.6f}")

            # Save the LR finder plot if possible
            try:
                fig = lr_finder.plot(suggest=True)
                plot_path = self.config.output_dir / "lr_finder.png"
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
                logger.info(f"LR finder plot saved: {plot_path}")
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass  # plotting is optional

            return suggested_lr

        except Exception as e:
            logger.warning(
                f"LR finder failed ({e}); "
                f"using default lr={self.config.learning_rate}"
            )
            return self.config.learning_rate

    # ---- S4e  Full training loop -------------------------------------------

    def train(
        self,
        model: TemporalFusionTransformer,
        train_dl,
        val_dl,
        find_lr: bool = True,
    ) -> Tuple[TemporalFusionTransformer, pl.Trainer]:
        """
        Execute the full training pipeline:
            1. (Optional) Find optimal learning rate
            2. Update model LR
            3. Train with Trainer.fit()
            4. Load best checkpoint
            5. Return best model + trainer

        Parameters
        ----------
        model : TemporalFusionTransformer
            Uninitialised (freshly built) TFT model.
        train_dl, val_dl : DataLoader
            Training and validation dataloaders.
        find_lr : bool
            Whether to run the LR finder before training.

        Returns
        -------
        (best_model, trainer) tuple
        """
        cfg = self.config

        # ---- Step 1: LR Finder (optional) ----------------------------------
        if find_lr:
            suggested_lr = self.find_learning_rate(model, train_dl, val_dl)
            model.hparams.learning_rate = suggested_lr
            logger.info(f"Model LR set to: {suggested_lr:.6f}")

        # ---- Step 2: Create production trainer -----------------------------
        trainer = self.setup_trainer()

        # ---- Step 3: Train -------------------------------------------------
        logger.info("=" * 60)
        logger.info("Starting TFT Training")
        logger.info("=" * 60)
        t0 = time.perf_counter()

        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

        elapsed = time.perf_counter() - t0
        logger.info(f"Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

        # ---- Step 4: Load best checkpoint ----------------------------------
        best_path = trainer.checkpoint_callback.best_model_path
        if best_path and os.path.exists(best_path):
            logger.info(f"Loading best model from: {best_path}")
            best_model = TemporalFusionTransformer.load_from_checkpoint(best_path)
        else:
            logger.warning("No checkpoint found; using last model state")
            best_model = model

        return best_model, trainer


# ============================================================================
# S5  PREDICTOR & EVALUATION
# ============================================================================

class TFTPredictor:
    """
    Generates and structures quantile predictions from a trained TFT.

    Output format (DataFrame):
        item_id | store_id | horizon_step | p10 | p50 | p90

    Where:
        p10 = 10th percentile (downside demand bound)
        p50 = 50th percentile (central / median forecast)
        p90 = 90th percentile (upside demand bound)

    These feed directly into:
        - Phase 3: D_base in  D_final = D_base * (1 + tau * dP)
        - Phase 5: Fan chart visualisation on the Streamlit dashboard
    """

    def __init__(self, config: TFTConfig) -> None:
        self.config = config

    def predict(
        self,
        model: TemporalFusionTransformer,
        dataset: TimeSeriesDataSet,
        dataloader=None,
    ) -> pd.DataFrame:
        """
        Run inference and return structured quantile predictions.

        Parameters
        ----------
        model : TemporalFusionTransformer
            Trained TFT model.
        dataset : TimeSeriesDataSet
            Dataset to predict on.
        dataloader : optional
            Pre-built dataloader; if None, one is created from dataset.

        Returns
        -------
        pd.DataFrame with columns: [group_ids..., horizon_step, p10, p50, p90]
        """
        cfg = self.config

        if dataloader is None:
            dataloader = dataset.to_dataloader(
                train=False,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
            )

        logger.info("Generating quantile predictions...")
        t0 = time.perf_counter()

        # ---- Raw predictions: shape (n_samples, horizon, n_quantiles) ------
        raw_predictions = model.predict(
            dataloader,
            mode="raw",
            return_index=True,
            return_decoder_lengths=True,
        )

        elapsed = time.perf_counter() - t0
        logger.info(f"Inference complete in {elapsed:.1f}s")

        # ---- Extract and structure the quantile outputs --------------------
        predictions_df = self._extract_quantiles(raw_predictions, dataset)

        return predictions_df

    def _extract_quantiles(
        self,
        raw_predictions: Any,
        dataset: TimeSeriesDataSet,
    ) -> pd.DataFrame:
        """
        Convert raw TFT output tensors into a tidy DataFrame.

        The raw prediction tensor has shape:
            (n_samples, max_prediction_length, n_quantiles)

        We reshape this into:
            n_samples * max_prediction_length rows, with columns:
            [item_id, store_id, horizon_step, p10, p50, p90]
        """
        cfg = self.config

        # Extract prediction tensor
        pred_tensor = getattr(raw_predictions, "output", raw_predictions[0])  # (N, H, Q)

        import torch
        # Handle list/tuple of tensors (batch outputs)
        if isinstance(pred_tensor, (list, tuple)):
            try:
                pred_tensor = torch.cat(pred_tensor, dim=0)
            except Exception as e:
                logger.warning(f"Failed to concatenate prediction tensors: {e}")
                # Fallback to first batch if concatenation fails due to inhomogeneous shapes
                pred_tensor = pred_tensor[0]

        # Handle torch tensor -> numpy
        if hasattr(pred_tensor, "cpu"):
            pred_np = pred_tensor.cpu().numpy()
        else:
            pred_np = np.array(pred_tensor)

        n_samples, horizon, n_quantiles = pred_np.shape
        logger.info(
            f"Prediction tensor: {n_samples} samples x "
            f"{horizon} steps x {n_quantiles} quantiles"
        )

        # Extract the index (group identifiers for each sample)
        index_df = getattr(raw_predictions, "index", raw_predictions[2] if len(raw_predictions) > 2 else None)
        
        if isinstance(index_df, (list, tuple)):
            try:
                index_df = pd.concat(index_df, ignore_index=True)
            except Exception as e:
                logger.warning(f"Failed to concatenate index dataframes: {e}")
                index_df = index_df[0]

        # Build the output DataFrame
        records = []
        for i in range(n_samples):
            # Get group identifiers for this sample
            group_info = {}
            for col in cfg.group_ids:
                if col in index_df.columns:
                    group_info[col] = index_df.iloc[i][col]

            # Each sample produces `horizon` prediction rows
            for h in range(horizon):
                row = {**group_info, "horizon_step": h + 1}
                for q_idx, q_val in enumerate(cfg.quantiles):
                    col_name = f"p{int(q_val * 100)}"
                    row[col_name] = float(pred_np[i, h, q_idx])
                records.append(row)

        predictions_df = pd.DataFrame(records)

        # ---- Ensure non-negative predictions (sales can't be < 0) ----------
        for col in ["p10", "p50", "p90"]:
            if col in predictions_df.columns:
                predictions_df[col] = predictions_df[col].clip(lower=0.0)

        logger.info(
            f"Structured predictions: {len(predictions_df):,} rows "
            f"({n_samples} series x {horizon} steps)"
        )
        return predictions_df

    def evaluate(
        self,
        predictions_df: pd.DataFrame,
        actuals: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics on the predictions.

        Metrics:
            - Median Absolute Error (MAE) on p50 vs actuals
            - Weighted Quantile Loss (WQL) across all quantiles
            - Empirical Coverage (fraction of actuals within [p10, p90])

        Parameters
        ----------
        predictions_df : DataFrame
            Output from self.predict().
        actuals : DataFrame or None
            If provided, must have a 'sales' column aligned with predictions.

        Returns
        -------
        dict of metric_name -> value
        """
        metrics = {}

        if actuals is not None and "sales" in actuals.values:
            # Align actuals with predictions
            actual_values = actuals["sales"].values[:len(predictions_df)]
            p50 = predictions_df["p50"].values[:len(actual_values)]

            # MAE
            metrics["mae_p50"] = float(np.mean(np.abs(actual_values - p50)))

            # Empirical coverage (should be ~80% for [p10, p90])
            p10 = predictions_df["p10"].values[:len(actual_values)]
            p90 = predictions_df["p90"].values[:len(actual_values)]
            covered = (actual_values >= p10) & (actual_values <= p90)
            metrics["coverage_p10_p90"] = float(np.mean(covered))

            logger.info(f"Evaluation metrics: {metrics}")

        # Always compute prediction interval width (uncertainty measure)
        if "p10" in predictions_df.columns and "p90" in predictions_df.columns:
            metrics["mean_interval_width"] = float(
                (predictions_df["p90"] - predictions_df["p10"]).mean()
            )
            metrics["median_p50"] = float(predictions_df["p50"].median())

        return metrics

    def save_predictions(
        self, predictions_df: pd.DataFrame, path: Optional[Path] = None
    ) -> Path:
        """Save predictions to Parquet for downstream phases."""
        if path is None:
            path = self.config.output_dir / "predictions.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_parquet(path, engine="pyarrow", compression="snappy")
        logger.info(f"Predictions saved: {path} ({path.stat().st_size / 1e6:.1f} MB)")
        return path


# ============================================================================
# S6  MASTER ORCHESTRATOR
# ============================================================================

class PredictiveEngine:
    """
    Top-level orchestrator for Phase 2.

    Chains: load data -> build dataset -> build model -> train -> predict

    Usage:
        engine = PredictiveEngine(config)
        predictions = engine.run()        # full pipeline
        predictions = engine.run_from_dataframe(df)  # from Phase 1 output
    """

    def __init__(self, config: Optional[TFTConfig] = None) -> None:
        self.config = config or TFTConfig()
        self.dataset_builder = DatasetBuilder(self.config)
        self.model_builder = ModelBuilder(self.config)
        self.training_pipeline = TrainingPipeline(self.config)
        self.predictor = TFTPredictor(self.config)

        # Stored references after run
        self.model: Optional[TemporalFusionTransformer] = None
        self.training_dataset: Optional[TimeSeriesDataSet] = None

    def run(self, find_lr: bool = True) -> pd.DataFrame:
        """
        Full pipeline: load parquet -> train -> predict.

        Returns the structured predictions DataFrame.
        """
        logger.info("=" * 60)
        logger.info("  PHASE 2: Predictive Engine (TFT)")
        logger.info("=" * 60)

        # ---- Load data from Phase 1 output ---------------------------------
        data_path = self.config.data_path
        if not data_path.exists():
            raise FileNotFoundError(
                f"Phase 1 output not found: {data_path}\n"
                "Run data_pipeline.py first to generate the processed data."
            )

        logger.info(f"Loading data from {data_path}")
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

        return self.run_from_dataframe(df, find_lr=find_lr)

    def run_from_dataframe(
        self, df: pd.DataFrame, find_lr: bool = True
    ) -> pd.DataFrame:
        """
        Run the full pipeline from an in-memory DataFrame.

        Steps:
            1. Prepare DataFrame (time_idx, dtypes)
            2. Build train/val TimeSeriesDataSets
            3. Create DataLoaders
            4. Build TFT model
            5. (Optional) Compute naive baseline
            6. (Optional) Find learning rate
            7. Train with early stopping
            8. Generate quantile predictions on validation set
            9. Save model + predictions
        """
        cfg = self.config
        t_start = time.perf_counter()

        # ---- 1. Prepare data -----------------------------------------------
        df = self.dataset_builder.prepare_dataframe(df)

        # ---- 2. Build datasets ---------------------------------------------
        training, validation = self.dataset_builder.build_training_dataset(df)
        self.training_dataset = training

        # ---- 3. Create dataloaders -----------------------------------------
        train_dl, val_dl = self.dataset_builder.create_dataloaders(
            training, validation
        )

        # ---- 4. Build model ------------------------------------------------
        model = self.model_builder.build(training)

        # ---- 5. Naive baseline (for comparison) ----------------------------
        ModelBuilder.compute_baseline(validation, val_dl)

        # ---- 6+7. Train (includes LR finder if enabled) --------------------
        best_model, trainer = self.training_pipeline.train(
            model, train_dl, val_dl, find_lr=find_lr
        )
        self.model = best_model

        # ---- 8. Predict on validation set ----------------------------------
        predictions_df = self.predictor.predict(
            best_model, validation, val_dl
        )

        # ---- 9. Save -------------------------------------------------------
        self.predictor.save_predictions(predictions_df)

        # ---- Save model for Phase 3/4 reuse --------------------------------
        model_path = cfg.output_dir / "tft_model.ckpt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(model_path))
        logger.info(f"Model checkpoint saved: {model_path}")

        total_time = time.perf_counter() - t_start
        logger.info(f"Phase 2 complete in {total_time:.1f}s ({total_time/60:.1f} min)")

        # ---- Summary stats -------------------------------------------------
        metrics = self.predictor.evaluate(predictions_df)
        logger.info(f"Prediction summary: {metrics}")

        return predictions_df


# ============================================================================
# S7  CLI ENTRY POINT
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 2: TFT Predictive Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run on Phase 1 output
  python predictive_engine.py --data-path data/processed/full_featured.parquet

  # Quick prototype with 50 items, no LR finder
  python predictive_engine.py --data-path data/processed/full_featured.parquet \\
      --sample-n-items 50 --no-lr-finder --max-epochs 5

  # GPU with mixed precision
  python predictive_engine.py --data-path data/processed/full_featured.parquet \\
      --precision 16-mixed --batch-size 128
        """,
    )
    parser.add_argument(
        "--data-path", type=str,
        default="data/processed/full_featured.parquet",
        help="Path to Phase 1 output parquet",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/tft",
        help="Directory for model checkpoints and predictions",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs/tft",
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=50,
        help="Maximum training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3,
        help="Initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=64,
        help="TFT hidden size (default: 64)",
    )
    parser.add_argument(
        "--max-encoder-length", type=int, default=90,
        help="Encoder lookback window in days (default: 90)",
    )
    parser.add_argument(
        "--max-prediction-length", type=int, default=30,
        help="Prediction horizon in days (default: 30)",
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Early stopping patience (default: 5)",
    )
    parser.add_argument(
        "--sample-n-items", type=int, default=None,
        help="Subsample N item-store combinations for prototyping",
    )
    parser.add_argument(
        "--precision", type=str, default="32-true",
        choices=["32-true", "16-mixed", "bf16-mixed"],
        help="Training precision (default: 32-true)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers (default: 0 for Windows)",
    )
    parser.add_argument(
        "--no-lr-finder", action="store_true",
        help="Skip the learning rate finder step",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for Phase 2."""
    args = parse_args()

    config = TFTConfig(
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        log_dir=Path(args.log_dir),
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        max_encoder_length=args.max_encoder_length,
        max_prediction_length=args.max_prediction_length,
        patience=args.patience,
        sample_n_items=args.sample_n_items,
        precision=args.precision,
        num_workers=args.num_workers,
    )

    engine = PredictiveEngine(config)
    predictions = engine.run(find_lr=not args.no_lr_finder)

    # Print final summary
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Columns: {list(predictions.columns)}")
    print(f"\nSample predictions (first 10 rows):")
    print(predictions.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

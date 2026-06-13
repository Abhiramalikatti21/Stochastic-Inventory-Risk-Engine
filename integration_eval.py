"""
================================================================================
Phase 4: Integration & Evaluation Engine
================================================================================

This module connects the Temporal Fusion Transformer (TFT) baseline forecasts
(Phase 2) with the Double Machine Learning (DML) causal elasticities (Phase 3).
It outputs final prescriptive forecasts adjusted for hypothetical pricing strategies.

It also contains a rigorous Quant Evaluation module computing:
1. WQL (Weighted Quantile Loss) for probabilistic bounds
2. Empirical Coverage (verifying the 80% prediction interval)
3. MASE (Mean Absolute Scaled Error) against a 7-day naive baseline

Output is serialized to a JSON report for downstream dashboarding.
================================================================================
"""

import json
import logging
from typing import Dict, List, Optional, Union
import warnings

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# Suppress pandas chained assignment warnings for clean execution
pd.options.mode.chained_assignment = None


class PrescriptiveEngine:
    """
    Integrates TFT base forecasts with DML causal elasticities to generate
    prescriptive, price-aware demand forecasts.
    """

    @staticmethod
    def generate_prescriptive_forecast(
        df_tft_predictions: pd.DataFrame,
        df_causal_elasticities: pd.DataFrame,
        delta_p: Union[float, pd.Series, np.ndarray],
    ) -> pd.DataFrame:
        """
        Combines baseline forecasts with causal elasticities to simulate demand
        under a new pricing strategy.

        Formula: D_final[q] = D_base[q] * max(0, 1 + (tau * delta_p))
        *We floor the multiplier at 0 to prevent negative demand.

        Args:
            df_tft_predictions: DataFrame with ['item_id', 'store_id', 'horizon_step', 'p10', 'p50', 'p90']
            df_causal_elasticities: DataFrame with ['item_id', 'store_id', 'cat_id', 'tau']
            delta_p: Percentage change in price (e.g., -0.10 for a 10% discount). Can be a scalar
                     or aligned array matching the predictions.

        Returns:
            DataFrame with final prescriptive quantiles ['final_p10', 'final_p50', 'final_p90']
        """
        logger.info("Merging TFT predictions with Causal Elasticities...")

        # Create a working copy
        pred_df = df_tft_predictions.copy()

        # Merge elasticities based on item and store
        merged = pd.merge(
            pred_df,
            df_causal_elasticities,
            on=["item_id", "store_id"],
            how="left"
        )

        # Handle missing elasticities:
        # 1. Fallback to category average if item/store tau is missing
        if merged["tau"].isna().any():
            missing_count = merged["tau"].isna().sum()
            logger.warning(f"Found {missing_count} rows missing specific elasticities. Falling back to category averages.")
            
            cat_avg = df_causal_elasticities.groupby("cat_id")["tau"].mean().to_dict()
            
            # Map category averages where missing (if cat_id is in the predictions dataframe)
            if "cat_id" in merged.columns:
                merged["tau"] = merged["tau"].fillna(merged["cat_id"].map(cat_avg))
            
            # 2. Global fallback for anything remaining
            global_tau = df_causal_elasticities["tau"].mean()
            merged["tau"] = merged["tau"].fillna(global_tau)
            logger.info(f"Global fallback tau applied: {global_tau:.4f}")

        # Vectorized application of the causal effect formula
        # Demand = Base * (1 + (elasticity * %_price_change))
        # We use np.maximum to ensure the multiplier doesn't drive demand below 0
        multiplier = np.maximum(0.0, 1.0 + (merged["tau"] * delta_p))

        merged["final_p10"] = merged["p10"] * multiplier
        merged["final_p50"] = merged["p50"] * multiplier
        merged["final_p90"] = merged["p90"] * multiplier

        logger.info("Prescriptive forecasting complete.")
        return merged


class EvaluationMetrics:
    """
    Strict quant-level backtesting metrics for probabilistic and point forecasts.
    """

    @staticmethod
    def wql(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
        """
        Weighted Quantile Loss (WQL) / Pinball Loss.
        Evaluates the accuracy of a specific quantile prediction.
        """
        error = y_true - y_pred
        loss = np.where(error >= 0, q * error, (q - 1) * error)
        
        # Weighted by the absolute sum of the true values to normalize across series
        denominator = np.sum(np.abs(y_true))
        if denominator == 0:
            return 0.0
        
        return 2.0 * np.sum(loss) / denominator

    @staticmethod
    def empirical_coverage(y_true: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> Dict[str, float]:
        """
        Calculates the percentage of true observations falling within the p10-p90 bounds.
        For a perfectly calibrated model, this should be exactly 80%.
        """
        in_bounds = (y_true >= p10) & (y_true <= p90)
        coverage_pct = np.mean(in_bounds)
        
        # Error is absolute deviation from 80%
        coverage_error = abs(coverage_pct - 0.80)
        
        return {
            "coverage_pct": float(coverage_pct),
            "coverage_error": float(coverage_error)
        }

    @staticmethod
    def mase(y_true: np.ndarray, y_pred: np.ndarray, y_hist: np.ndarray, seasonality: int = 7) -> float:
        """
        Mean Absolute Scaled Error (MASE).
        Compares the forecast's MAE against a naive historical seasonal baseline (e.g., lag 7).
        MASE < 1 means the model is better than simply repeating last week's sales.
        """
        # Forecast Mean Absolute Error
        mae_forecast = np.mean(np.abs(y_true - y_pred))
        
        # Naive Seasonal Mean Absolute Error
        if len(y_hist) <= seasonality:
            logger.warning("History too short for seasonal MASE. Defaulting denominator to 1.0")
            mae_naive = 1.0
        else:
            diffs = np.abs(y_hist[seasonality:] - y_hist[:-seasonality])
            mae_naive = np.mean(diffs)
            
            if mae_naive == 0:
                mae_naive = 1e-6 # prevent division by zero
                
        return float(mae_forecast / mae_naive)


class BacktestingRunner:
    """
    Executes the evaluation pipeline, slicing data by category and store, 
    and serializing a final JSON report.
    """

    def __init__(self, metrics: EvaluationMetrics):
        self.metrics = metrics

    def run_evaluation(
        self,
        df_forecasts: pd.DataFrame,
        df_history: pd.DataFrame,
        output_path: str = "evaluation_report.json"
    ) -> Dict:
        """
        Runs comprehensive evaluation.
        
        Args:
            df_forecasts: DataFrame with ['item_id', 'store_id', 'cat_id', 'sales', 'p10', 'p50', 'p90']
                          where 'sales' is the true outcome.
            df_history: DataFrame with historical ['item_id', 'store_id', 'sales'] for MASE calc.
            output_path: Path to save the JSON report.
        """
        logger.info("Starting Backtesting Evaluation...")
        
        report = {
            "global_metrics": {},
            "by_category": {},
            "by_store": {}
        }
        
        # --- 1. Global Metrics ---
        y_true = df_forecasts["sales"].values
        p10 = df_forecasts["p10"].values
        p50 = df_forecasts["p50"].values
        p90 = df_forecasts["p90"].values
        
        coverage_stats = self.metrics.empirical_coverage(y_true, p10, p90)
        
        report["global_metrics"] = {
            "WQL_10": self.metrics.wql(y_true, p10, 0.1),
            "WQL_50": self.metrics.wql(y_true, p50, 0.5), # Effectively Normalized MAE
            "WQL_90": self.metrics.wql(y_true, p90, 0.9),
            "empirical_coverage_pct": coverage_stats["coverage_pct"],
            "coverage_error": coverage_stats["coverage_error"],
            # Global MASE calculated later via aggregation for accuracy
        }

        # --- 2. MASE & Aggregations by Slice ---
        def evaluate_slice(slice_df: pd.DataFrame) -> Dict:
            slice_y = slice_df["sales"].values
            
            # Find matching history for this slice
            # To be precise, we flatten all history for these items/stores
            if len(slice_y) == 0:
                return {}
                
            # Naive MASE calc - approximation by combining histories
            keys = slice_df[["item_id", "store_id"]].drop_duplicates()
            hist_subset = pd.merge(df_history, keys, on=["item_id", "store_id"], how="inner")
            hist_y = hist_subset.sort_values(["item_id", "store_id"])["sales"].values
            
            slice_mase = self.metrics.mase(slice_y, slice_df["p50"].values, hist_y, seasonality=7)
            
            return {
                "WQL_50": self.metrics.wql(slice_y, slice_df["p50"].values, 0.5),
                "MASE": slice_mase,
                "coverage_pct": self.metrics.empirical_coverage(
                    slice_y, slice_df["p10"].values, slice_df["p90"].values
                )["coverage_pct"]
            }

        # Slicing by Category
        if "cat_id" in df_forecasts.columns:
            for cat in df_forecasts["cat_id"].unique():
                subset = df_forecasts[df_forecasts["cat_id"] == cat]
                report["by_category"][str(cat)] = evaluate_slice(subset)
                
        # Slicing by Store
        if "store_id" in df_forecasts.columns:
            for store in df_forecasts["store_id"].unique():
                subset = df_forecasts[df_forecasts["store_id"] == store]
                report["by_store"][str(store)] = evaluate_slice(subset)

        # Global MASE (average of category MASEs)
        cat_mases = [v["MASE"] for v in report["by_category"].values() if "MASE" in v]
        if cat_mases:
            report["global_metrics"]["MASE"] = float(np.mean(cat_mases))

        # --- Serialization ---
        with open(output_path, "w") as f:
            json.dump(report, f, indent=4)
            
        logger.info(f"Evaluation Report written to {output_path}")
        self._print_summary(report)
        
        return report

    def _print_summary(self, report: Dict):
        """Prints a clean statistical summary to the console."""
        g = report.get("global_metrics", {})
        
        print("\n" + "="*50)
        print("  BACKTEST EVALUATION SUMMARY")
        print("="*50)
        print(f"  MASE (vs 7d naive):       {g.get('MASE', 0):.4f}  (< 1 is good)")
        print(f"  WQL (50th percentile):    {g.get('WQL_50', 0):.4f}")
        print(f"  WQL (10th / 90th):        {g.get('WQL_10', 0):.4f} / {g.get('WQL_90', 0):.4f}")
        print("-" * 50)
        
        cov = g.get('empirical_coverage_pct', 0) * 100
        target = 80.0
        err = g.get('coverage_error', 0) * 100
        print(f"  Empirical Coverage:       {cov:.1f}%")
        print(f"  Target Coverage:          {target:.1f}%")
        print(f"  Coverage Error:           {err:.1f} pp")
        
        if err < 5.0:
            print("  [OK] Model is well-calibrated probabilistically.")
        else:
            print("  [WARN] Model calibration is drifting. Consider retraining.")
        print("="*50 + "\n")

"""
================================================================================
Smoke Test -- Phase 4: Integration & Evaluation Engine
================================================================================

Validates the fusion of TFT baseline predictions with DML elasticities.
Verifies the calculation of WQL, Empirical Coverage, and MASE.
"""

import os
import numpy as np
import pandas as pd

from integration_eval import PrescriptiveEngine, EvaluationMetrics, BacktestingRunner

def test_prescriptive_generation():
    print("--- Test: Prescriptive Forecast Generation ---")
    
    # Synthetic TFT Output
    tft_df = pd.DataFrame({
        'item_id': ['A', 'A', 'B', 'B', 'C'],
        'store_id': [1, 1, 1, 1, 1],
        'horizon_step': [1, 2, 1, 2, 1],
        'p10': [8.0, 9.0, 15.0, 14.0, 5.0],
        'p50': [10.0, 12.0, 20.0, 18.0, 8.0],
        'p90': [12.0, 15.0, 25.0, 22.0, 11.0],
        'cat_id': ['Cat1', 'Cat1', 'Cat2', 'Cat2', 'Cat3']
    })
    
    # Synthetic Causal Elasticities (Item C is missing to test fallback)
    causal_df = pd.DataFrame({
        'item_id': ['A', 'B'],
        'store_id': [1, 1],
        'cat_id': ['Cat1', 'Cat2'],
        'tau': [-2.0, -1.0]
    })
    
    # Let's apply a 10% price drop (delta_p = -0.10)
    delta_p = -0.10
    
    final_df = PrescriptiveEngine.generate_prescriptive_forecast(
        tft_df, causal_df, delta_p
    )
    
    assert len(final_df) == len(tft_df), "Row count mismatch"
    assert "final_p50" in final_df.columns, "Missing prescriptive forecast"
    
    # Check Item A: Base 10.0, Tau -2.0, dP -0.10 -> Multiplier = 1 + (-2 * -0.1) = 1.2
    # final_p50 should be 12.0
    item_a_p50 = final_df[(final_df['item_id'] == 'A') & (final_df['horizon_step'] == 1)]['final_p50'].values[0]
    assert np.isclose(item_a_p50, 12.0), f"Item A multiplier failed, got {item_a_p50}"
    
    # Check Item C (fallback): Base 8.0. Missing tau. Cat fallback missing, global avg tau is -1.5
    # Multiplier = 1 + (-1.5 * -0.1) = 1.15
    # final_p50 should be 8.0 * 1.15 = 9.2
    item_c_p50 = final_df[(final_df['item_id'] == 'C') & (final_df['horizon_step'] == 1)]['final_p50'].values[0]
    assert np.isclose(item_c_p50, 9.2), f"Fallback logic failed, got {item_c_p50}"
    
    print("  [OK] Prescriptive multiplier logic and fallback passed.")

def test_evaluation_metrics():
    print("\n--- Test: Evaluation Metrics & Backtest Runner ---")
    
    y_true = np.array([10.0, 15.0, 20.0, 25.0])
    p10    = np.array([8.0,  12.0, 16.0, 20.0])
    p50    = np.array([10.0, 14.0, 21.0, 24.0])
    p90    = np.array([12.0, 18.0, 24.0, 28.0])
    
    metrics = EvaluationMetrics()
    
    # Test WQL 0.5 (should be sum(|y_true - p50|) / sum(|y_true|)
    # Errors: 0, 1, -1, 1 -> absolute sum = 3
    # Denominator sum(y) = 70
    # WQL50 = 2 * (3 * 0.5) / 70 = 3 / 70 ~ 0.0428
    wql_50 = metrics.wql(y_true, p50, 0.5)
    assert np.isclose(wql_50, 3 / 70), f"WQL 50 failed, got {wql_50}"
    
    # Test coverage
    cov = metrics.empirical_coverage(y_true, p10, p90)
    assert cov['coverage_pct'] == 1.0, "All true values are within bounds, coverage should be 1.0"
    assert np.isclose(cov['coverage_error'], 0.2), "Expected error 0.2 vs 0.8 target"
    
    # Setup test runner
    df_forecasts = pd.DataFrame({
        'item_id': ['A', 'A', 'B', 'B'],
        'store_id': [1, 1, 1, 1],
        'cat_id': ['Cat1', 'Cat1', 'Cat2', 'Cat2'],
        'sales': y_true,
        'p10': p10,
        'p50': p50,
        'p90': p90
    })
    
    # History for MASE (needs 7 days minimum, let's provide 10 days of flat history)
    df_history = pd.DataFrame({
        'item_id': ['A']*10 + ['B']*10,
        'store_id': [1]*20,
        'sales': [10.0]*10 + [20.0]*10
    })
    
    runner = BacktestingRunner(metrics)
    report_file = "test_eval_report.json"
    
    report = runner.run_evaluation(df_forecasts, df_history, output_path=report_file)
    
    assert os.path.exists(report_file), "JSON report was not created"
    assert "global_metrics" in report
    assert "by_category" in report
    assert "Cat1" in report["by_category"]
    
    print("  [OK] Metrics calc and runner passed.")
    os.remove(report_file)

def main():
    print("=" * 64)
    print("  SMOKE TEST -- Phase 4: Integration Engine")
    print("=" * 64)
    
    test_prescriptive_generation()
    test_evaluation_metrics()
    
    print("\n" + "=" * 64)
    print("  PHASE 4 TESTS PASSED")
    print("=" * 64)

if __name__ == "__main__":
    main()

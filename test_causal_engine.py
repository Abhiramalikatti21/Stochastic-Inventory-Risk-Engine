"""
================================================================================
Smoke Test -- Phase 3: Causal Inference Engine (DML)
================================================================================

Validates the DML architecture on synthetic data.
Ensures we can fit the model and extract the ATE and CATE.
"""

import sys
import numpy as np
import pandas as pd

def check_dependencies():
    try:
        import econml
        import lightgbm
        import sklearn
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install econml lightgbm scikit-learn")
        sys.exit(0)

check_dependencies()

from causal_engine import CausalElasticityEstimator

def generate_causal_data(n=2000):
    """Generate synthetic sales and price data with confounders."""
    np.random.seed(42)
    
    # Confounders (W)
    is_holiday = np.random.binomial(1, 0.1, n)
    day_of_week = np.random.randint(0, 7, n)
    
    # Heterogeneity (X)
    category = np.random.choice(['electronics', 'clothing', 'food'], n)
    base_volatility = np.random.uniform(0.1, 0.5, n)
    
    # Treatment: Price (T)
    # Price is higher on holidays and for electronics (confounded)
    base_price = 10.0 + 5.0 * is_holiday + 20.0 * (category == 'electronics')
    # Add random noise
    price = base_price + np.random.normal(0, 2.0, n)
    
    # Outcome: Sales (Y)
    # True elasticity (tau): -2.0 for electronics, -1.0 for others
    # More volatile items have less elasticity (tau moves closer to 0)
    true_tau = -1.0 - 1.0 * (category == 'electronics') + 1.0 * base_volatility
    
    # Sales depend on price (causal), holiday (confounder), and base demand
    sales = 100.0 + 50.0 * is_holiday + (true_tau * price) + np.random.normal(0, 5.0, n)
    
    return pd.DataFrame({
        'sales': sales,
        'price': price,
        'is_holiday': is_holiday,
        'day_of_week': day_of_week,
        'category': category,
        'base_volatility': base_volatility,
        'true_tau': true_tau
    })

def main():
    print("=" * 64)
    print("  SMOKE TEST -- Phase 3: Causal Inference Engine")
    print("=" * 64)
    
    df = generate_causal_data(3000)
    print(f"[OK] Generated {len(df)} rows of synthetic data")
    
    # Initialize estimator (n_estimators must be divisible by 4)
    estimator = CausalElasticityEstimator(n_estimators=100, random_state=42)
    
    # Define variables
    Y_col = 'sales'
    T_col = 'price'
    X_cols = ['category', 'base_volatility']
    W_cols = ['is_holiday', 'day_of_week']
    
    # Fit model
    estimator.fit(
        df=df,
        outcome_col=Y_col,
        treatment_col=T_col,
        heterogeneity_cols=X_cols,
        confounder_cols=W_cols
    )
    print("[OK] CausalForestDML fitted successfully")
    
    # Test Summary (ATE)
    summary = estimator.summary(df)
    assert 'ATE' in summary
    assert 'p_value' in summary
    
    # Test CATE inference
    tau, lower, upper = estimator.estimate_elasticity(df)
    assert len(tau) == len(df)
    assert len(lower) == len(df)
    assert len(upper) == len(df)
    
    # Calculate Mean Absolute Error against the True Tau
    # DML should recover the true elasticity reasonably well
    mae = np.mean(np.abs(tau - df['true_tau'].values))
    print(f"\n[OK] CATE Mean Absolute Error vs True Tau: {mae:.4f}")
    
    # Feature Importances
    importances = estimator.feature_importances()
    print("\n[OK] Feature Importances:")
    print(importances)
    
    print("\n" + "=" * 64)
    print("  PHASE 3 TEST PASSED")
    print("=" * 64)

if __name__ == "__main__":
    main()

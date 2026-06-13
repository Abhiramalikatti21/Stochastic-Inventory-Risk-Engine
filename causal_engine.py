"""
================================================================================
Phase 3: Causal Inference Engine (DML)
================================================================================

This module implements Double Machine Learning (DML) using Microsoft's EconML
to estimate the Conditional Average Treatment Effect (CATE). Specifically, we
are estimating Price Elasticity: how a change in price (Treatment, T) causally
impacts sales volume (Outcome, Y), conditioned on product characteristics (X),
while controlling for confounders like seasonality and holidays (W).

Statistical Justification for DML:
Standard machine learning models (like our TFT in Phase 2) are optimized for
predictive accuracy, not causal estimation. If we simply look at the correlation
between price and sales, we suffer from omitted variable bias (e.g., prices
might be lowered during holidays when demand is already high, making it look
like the price drop caused a massive spike, confounding the true elasticity).
DML isolates the orthogonal, unconfounded variance in price to estimate the
true marginal effect (tau).

Architecture:
    Y: Demand / Sales volume
    T: Price (log_price or price_change)
    X: Heterogeneity features (item characteristics, rolling volatility)
    W: Confounders (day of week, events, SNAP days)
    Model: CausalForestDML with LightGBM nuisance estimators.
================================================================================
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Suppress EconML/LightGBM verbose warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from econml.dml import CausalForestDML
    from lightgbm import LGBMRegressor
    from sklearn.compose import ColumnTransformer
    from sklearn.exceptions import NotFittedError
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler
except ImportError as e:
    raise ImportError(
        "Missing required dependencies for Causal Engine. "
        "Please run: pip install econml lightgbm scikit-learn"
    ) from e

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


class CausalElasticityEstimator:
    """
    Wrapper for EconML's CausalForestDML to estimate price elasticity.
    """

    def __init__(
        self,
        discrete_treatment: bool = False,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        """
        Initialize the Causal Elasticity Estimator.

        Args:
            discrete_treatment: True if treatment is categorical (e.g. promo flag).
                                False if treatment is continuous (e.g. price).
            n_estimators: Number of trees in the Causal Forest.
            random_state: Seed for reproducibility.
        """
        self.discrete_treatment = discrete_treatment
        self.random_state = random_state
        
        # Nuisance models: LightGBM is chosen for its speed and ability to handle
        # complex, non-linear confounding structures without extensive tuning.
        self.model_y = LGBMRegressor(
            n_estimators=100, max_depth=5, random_state=random_state, n_jobs=-1
        )
        self.model_t = LGBMRegressor(
            n_estimators=100, max_depth=5, random_state=random_state, n_jobs=-1
        )

        # The Causal Forest DML estimator
        self.estimator = CausalForestDML(
            model_y=self.model_y,
            model_t=self.model_t,
            discrete_treatment=self.discrete_treatment,
            n_estimators=n_estimators,
            random_state=random_state,
            # honest=True ensures that trees are grown and evaluated on different
            # subsamples, producing asymptotically normal, unbiased estimates.
            honest=True,
            cv=3, # 3-fold cross-fitting
        )

        self.preprocessor_X: Optional[ColumnTransformer] = None
        self.preprocessor_W: Optional[ColumnTransformer] = None
        
        self._is_fitted = False
        self._feature_names_X: List[str] = []

    def _build_preprocessor(self, df: pd.DataFrame, features: List[str]) -> ColumnTransformer:
        """
        Builds a scikit-learn ColumnTransformer that standardizes numeric 
        features and ordinally encodes categorical features.
        """
        numeric_features = []
        categorical_features = []
        
        for col in features:
            if df[col].dtype.name in ['category', 'object', 'bool']:
                categorical_features.append(col)
            else:
                numeric_features.append(col)
                
        transformers = []
        if numeric_features:
            # We scale numeric variables for stable gradient calculations in nuisance models
            transformers.append(("num", StandardScaler(), numeric_features))
        if categorical_features:
            # LightGBM can handle ordinal integers natively as categoricals
            transformers.append((
                "cat", 
                OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), 
                categorical_features
            ))
            
        return ColumnTransformer(transformers, remainder='drop')

    def fit(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        heterogeneity_cols: List[str],
        confounder_cols: List[str],
    ) -> "CausalElasticityEstimator":
        """
        Fit the Double Machine Learning models.

        Args:
            df: Input dataframe containing all required columns.
            outcome_col: (Y) The target metric, e.g., 'sales'.
            treatment_col: (T) The causal lever, e.g., 'sell_price'.
            heterogeneity_cols: (X) Features that modify elasticity.
            confounder_cols: (W) Features that affect both Y and T.
            
        Returns:
            Self.
        """
        logger.info(f"Preparing DML matrices: Y='{outcome_col}', T='{treatment_col}'")
        logger.info(f"Heterogeneity features (X): {len(heterogeneity_cols)}")
        logger.info(f"Confounders (W): {len(confounder_cols)}")

        # Drop rows with NaNs in required columns
        req_cols = [outcome_col, treatment_col] + heterogeneity_cols + confounder_cols
        clean_df = df[req_cols].dropna().copy()
        
        if len(clean_df) < len(df):
            logger.warning(f"Dropped {len(df) - len(clean_df)} rows containing NaNs.")

        Y = clean_df[outcome_col].values
        T = clean_df[treatment_col].values
        
        # Build and fit preprocessors
        self.preprocessor_X = self._build_preprocessor(clean_df, heterogeneity_cols)
        self.preprocessor_W = self._build_preprocessor(clean_df, confounder_cols)
        
        X_proc = self.preprocessor_X.fit_transform(clean_df[heterogeneity_cols])
        W_proc = self.preprocessor_W.fit_transform(clean_df[confounder_cols])
        
        # Save feature names for interpretability later
        self._feature_names_X = heterogeneity_cols

        logger.info("Fitting CausalForestDML (this may take a moment)...")
        try:
            self.estimator.fit(Y=Y, T=T, X=X_proc, W=W_proc)
            self._is_fitted = True
            logger.info("Causal model fitted successfully.")
        except np.linalg.LinAlgError as e:
            logger.error("Matrix singularity encountered during fitting. Check for highly collinear controls.")
            raise e
        except Exception as e:
            logger.error(f"Failed to fit causal model: {e}")
            raise e

        return self

    def estimate_elasticity(self, df_X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate the Conditional Average Treatment Effect (CATE) for new samples.
        In the context of price, CATE represents the price elasticity multiplier (tau).

        Args:
            df_X: DataFrame containing the heterogeneity features (X).
            
        Returns:
            Tuple of (point_estimates, lower_bound, upper_bound)
            representing tau and its 95% confidence intervals.
        """
        if not self._is_fitted:
            raise NotFittedError("The causal model must be fitted before estimating effects.")
            
        # Process X features identically to training
        X_proc = self.preprocessor_X.transform(df_X[self._feature_names_X])
        
        logger.info(f"Estimating treatment effects for {len(df_X)} samples...")
        
        # Predict CATE and confidence intervals (alpha=0.05 for 95% CI)
        effects = self.estimator.effect(X_proc)
        lower_bound, upper_bound = self.estimator.effect_interval(X_proc, alpha=0.05)
        
        return effects, lower_bound, upper_bound

    def summary(self, df_X: pd.DataFrame) -> Dict[str, float]:
        """
        Compute the overall Average Treatment Effect (ATE) to evaluate 
        the global statistical significance of price changes.
        
        Args:
            df_X: DataFrame containing the heterogeneity features (X) to evaluate ATE over.
            
        Returns:
            Dictionary containing ATE, p-value, and confidence bounds.
        """
        if not self._is_fitted:
            raise NotFittedError("The causal model must be fitted to generate a summary.")
            
        if self.discrete_treatment:
            summary_df = self.estimator.ate__inference().summary_frame()
            ate = summary_df["point_estimate"].values[0]
            p_val = summary_df["pvalue"].values[0]
            ci_lower = summary_df["ci_lower"].values[0]
            ci_upper = summary_df["ci_upper"].values[0]
        else:
            # For continuous treatments in CausalForestDML, we estimate the empirical ATE
            # over the provided population df_X.
            effects, lower, upper = self.estimate_elasticity(df_X)
            ate = np.mean(effects)
            ci_lower = np.mean(lower)
            ci_upper = np.mean(upper)
            # Approximate p-value based on whether 0 is in the confidence interval
            p_val = 0.001 if (ci_lower > 0 or ci_upper < 0) else 0.5
        
        logger.info("-" * 50)
        logger.info("CAUSAL INFERENCE SUMMARY (ATE)")
        logger.info("-" * 50)
        logger.info(f"Average Treatment Effect: {ate:.4f}")
        logger.info(f"P-value (approximate):    {p_val:.4e}")
        logger.info(f"95% Confidence Interval:  [{ci_lower:.4f}, {ci_upper:.4f}]")
        logger.info("-" * 50)
        
        if p_val < 0.05:
            logger.info("Conclusion: The treatment effect is statistically significant (p < 0.05).")
        else:
            logger.warning("Conclusion: The treatment effect is NOT statistically significant (p >= 0.05).")
            
        return {
            "ATE": float(ate),
            "p_value": float(p_val),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper)
        }

    def feature_importances(self) -> pd.DataFrame:
        """
        Extract the importance of heterogeneity features (X) in driving 
        variance in the treatment effect. Which features make a product more elastic?
        """
        if not self._is_fitted:
            raise NotFittedError("Model not fitted.")
            
        # CausalForest provides direct feature importances for heterogeneity
        importances = self.estimator.feature_importances()
        
        df_imp = pd.DataFrame({
            "Feature": self._feature_names_X,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        
        return df_imp

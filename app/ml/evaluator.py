"""
Evaluation metrics for regression-based stock prediction.

Provides RMSE, MAE, MAPE, and R² calculations with clean
reporting and optional comparison between multiple models.

Usage:
    from app.ml.evaluator import RegressionMetrics, compare_models

    metrics = RegressionMetrics(y_true, y_pred)
    print(metrics.report())

    results = compare_models({"ANN": (y_ann, y_pred_ann), "LSTM": ...})
    print(results)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegressionMetrics:
    """
    Calculate common regression metrics for model evaluation.

    Metrics:
        - RMSE:  Root Mean Squared Error (penalises large errors).
        - MAE:   Mean Absolute Error (interpretable in original units).
        - MAPE:  Mean Absolute Percentage Error (unit-free, relative).
        - R²:    Coefficient of determination (1.0 = perfect fit).
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, decimals: int = 4):
        """
        Args:
            y_true: Ground-truth target values.
            y_pred: Predicted target values.
            decimals: Rounding precision for metrics.
        """
        self.y_true = np.asarray(y_true, dtype=np.float64).flatten()
        self.y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

        if len(self.y_true) != len(self.y_pred):
            raise ValueError(
                f"y_true ({len(self.y_true)}) and y_pred "
                f"({len(self.y_pred)}) must have the same length."
            )

        self.decimals = decimals
        self._compute()

    # ------------------------------------------------------------------
    # Public computed properties
    # ------------------------------------------------------------------

    @property
    def rmse(self) -> float:
        """Root Mean Squared Error."""
        return round(float(self._rmse), self.decimals)

    @property
    def mae(self) -> float:
        """Mean Absolute Error."""
        return round(float(self._mae), self.decimals)

    @property
    def mape(self) -> float:
        """Mean Absolute Percentage Error (as a percentage, e.g. 2.5 for 2.5%)."""
        return round(float(self._mape), self.decimals)

    @property
    def r2(self) -> float:
        """R² score (coefficient of determination)."""
        return round(float(self._r2), self.decimals)

    # ------------------------------------------------------------------
    # Internal calculation
    # ------------------------------------------------------------------

    def _compute(self) -> None:
        """Calculate all metrics from the stored arrays."""
        residuals = self.y_true - self.y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((self.y_true - np.mean(self.y_true)) ** 2)

        self._rmse = np.sqrt(np.mean(residuals ** 2))
        self._mae = np.mean(np.abs(residuals))

        denominator = np.abs(self.y_true) + 1e-10  # avoid division by zero
        self._mape = np.mean(np.abs(residuals) / denominator) * 100.0

        self._r2 = 1 - (ss_res / (ss_tot + 1e-10))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, float]:
        """Return all metrics as a dictionary."""
        return {
            "RMSE": self.rmse,
            "MAE": self.mae,
            "MAPE": self.mape,
            "R²": self.r2,
        }

    def report(self) -> str:
        """Return a human-readable multi-line report."""
        lines = [
            "── Regression Metrics ──────────────────────",
            f"  RMSE:  {self.rmse:>{self.decimals + 4}.{self.decimals}f}",
            f"  MAE:   {self.mae:>{self.decimals + 4}.{self.decimals}f}",
            f"  MAPE:  {self.mape:>{self.decimals + 4}.{self.decimals}f} %",
            f"  R²:    {self.r2:>{self.decimals + 4}.{self.decimals}f}",
            "────────────────────────────────────────────",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"RegressionMetrics(RMSE={self.rmse}, MAE={self.mae}, "
            f"MAPE={self.mape}%, R²={self.r2})"
        )


def compare_models(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    decimals: int = 4,
) -> pd.DataFrame:
    """
    Compare multiple models side-by-side.

    Args:
        results: Map of ``{model_name: (y_true, y_pred)}``.
        decimals: Rounding precision.

    Returns:
        DataFrame with one row per model and columns for each metric.

    Example:
        >>> y_true = np.array([100, 102, 101])
        >>> compare_models({
        ...     "ANN":  (y_true, np.array([99, 103, 100])),
        ...     "LSTM": (y_true, np.array([100, 101, 102])),
        ... })
          Model   RMSE    MAE   MAPE     R²
        0   ANN  1.290  1.000  0.996  0.250
        1  LSTM  0.816  0.667  0.663  0.700
    """
    if not results:
        return pd.DataFrame(columns=["Model", "RMSE", "MAE", "MAPE", "R²"]).set_index("Model")

    rows = []
    for name, (y_true, y_pred) in results.items():
        m = RegressionMetrics(y_true, y_pred, decimals=decimals)
        rows.append(
            {
                "Model": name,
                **m.to_dict(),
            }
        )

    df = pd.DataFrame(rows).set_index("Model")
    logger.info("Model comparison:\n%s", df.to_string())
    return df

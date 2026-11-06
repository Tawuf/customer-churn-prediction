"""Turn churn probabilities into a retention campaign decision.

A model is only useful if it changes a business decision. Here the decision
is: *which customers should get a retention offer?*

For each customer we contact:
  * we pay ``offer_cost`` (discount, call-centre time ...)
  * if they really were about to churn, the offer keeps them with
    probability ``success_rate`` and we keep their 24-month margin (CLV)

Expected profit of contacting a set S of customers (evaluated on held-out data
with the true labels)::

    profit(S) = sum_{i in S} ( y_i * success_rate * CLV_i  -  offer_cost )
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def campaign_profit(
    y_true: np.ndarray,
    contacted: np.ndarray,
    clv: np.ndarray,
    offer_cost: float,
    success_rate: float,
) -> float:
    y_true, contacted, clv = map(np.asarray, (y_true, contacted, clv))
    gain = (y_true * success_rate * clv - offer_cost) * contacted
    return float(gain.sum())


def profit_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    clv: np.ndarray,
    offer_cost: float = 200.0,
    success_rate: float = 0.35,
) -> pd.DataFrame:
    """Profit when contacting the top-k% of customers ranked by churn risk."""
    y_true, proba, clv = map(np.asarray, (y_true, proba, clv))
    order = np.argsort(-proba)
    n = len(order)
    rows = []
    for pct in range(0, 101):
        k = int(round(n * pct / 100))
        contacted = np.zeros(n, dtype=int)
        contacted[order[:k]] = 1
        rows.append(
            {
                "pct_contacted": pct,
                "customers_contacted": k,
                "threshold": float(proba[order[k - 1]]) if k else 1.0,
                "profit": campaign_profit(y_true, contacted, clv, offer_cost, success_rate),
                "churners_reached": int((y_true * contacted).sum()),
            }
        )
    return pd.DataFrame(rows)


def best_policy(curve: pd.DataFrame) -> pd.Series:
    """Row of the profit curve with the highest profit."""
    return curve.loc[curve["profit"].idxmax()]

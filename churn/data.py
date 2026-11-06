"""Synthetic customer data for a subscription business.

Real churn data is confidential, so this module generates a realistic,
reproducible dataset. The churn probability is driven by known rules (short
tenure, monthly contracts, support problems, low engagement, late payments),
plus noise, so the models have real signal to learn but the data is
**clearly synthetic** and not about real people.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONTRACTS = ["monthly", "annual", "two_year"]
PAYMENT_METHODS = ["card", "invoice", "direct_debit"]
REGIONS = ["Stockholm", "Uppsala", "Gothenburg", "Malmo", "Other"]

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_fee",
    "num_products",
    "logins_30d",
    "support_tickets_90d",
    "late_payments_12m",
    "nps_score",
    "discount_pct",
]
CATEGORICAL_FEATURES = ["contract", "payment_method", "region"]
TARGET = "churned"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_customers(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame with one row per customer and a binary ``churned`` column."""
    rng = np.random.default_rng(seed)

    contract = rng.choice(CONTRACTS, size=n, p=[0.55, 0.30, 0.15])
    tenure = np.where(
        contract == "monthly",
        rng.gamma(2.0, 7.0, n),
        rng.gamma(3.0, 10.0, n),
    ).clip(1, 96).round()
    num_products = rng.choice([1, 2, 3, 4], size=n, p=[0.45, 0.30, 0.17, 0.08])
    monthly_fee = (149 + 90 * (num_products - 1) + rng.normal(0, 25, n)).clip(99).round(0)
    discount = np.where(rng.random(n) < 0.25, rng.choice([10, 15, 20], size=n), 0)
    logins = rng.poisson(np.where(contract == "monthly", 9, 14)).astype(float)
    tickets = rng.poisson(0.8, n)
    late = rng.poisson(0.35, n)
    payment = rng.choice(PAYMENT_METHODS, size=n, p=[0.45, 0.25, 0.30])
    region = rng.choice(REGIONS, size=n, p=[0.35, 0.12, 0.20, 0.13, 0.20])
    nps = (7.5 - 0.9 * tickets + 0.05 * logins + rng.normal(0, 1.6, n)).clip(0, 10).round()

    logit = (
        -1.2
        + 1.3 * (contract == "monthly")
        - 0.9 * (contract == "two_year")
        - 0.035 * tenure
        + 0.45 * tickets
        + 0.55 * late
        - 0.08 * logins
        - 0.22 * (nps - 7)
        + 0.004 * (monthly_fee - 200)
        - 0.25 * (num_products - 1)
        + 0.35 * (payment == "invoice")
        - 0.30 * (payment == "direct_debit")
        - 0.02 * discount
        + rng.normal(0, 0.6, n)
    )
    churned = (rng.random(n) < _sigmoid(logit)).astype(int)

    return pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(1, n + 1)],
            "tenure_months": tenure.astype(int),
            "contract": contract,
            "monthly_fee": monthly_fee,
            "num_products": num_products,
            "logins_30d": logins.astype(int),
            "support_tickets_90d": tickets,
            "late_payments_12m": late,
            "nps_score": nps.astype(int),
            "discount_pct": discount,
            "payment_method": payment,
            "region": region,
            TARGET: churned,
        }
    )


def customer_lifetime_value(df: pd.DataFrame, months: int = 24, margin: float = 0.40) -> pd.Series:
    """Margin a retained customer brings over the next ``months`` (SEK)."""
    return df["monthly_fee"] * (1 - df["discount_pct"] / 100) * months * margin

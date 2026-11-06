import numpy as np
import pytest

from churn.data import TARGET, customer_lifetime_value, generate_customers
from churn.model import evaluate, split
from churn.targeting import best_policy, campaign_profit, profit_curve


@pytest.fixture(scope="module")
def df():
    return generate_customers(3000, seed=7)


def test_generator_is_reproducible():
    a = generate_customers(200, seed=1)
    b = generate_customers(200, seed=1)
    assert a.equals(b)


def test_generator_shape_and_ranges(df):
    assert len(df) == 3000
    assert df["customer_id"].is_unique
    assert df["nps_score"].between(0, 10).all()
    assert df["tenure_months"].between(1, 96).all()
    assert 0.10 < df[TARGET].mean() < 0.45


def test_monthly_contracts_churn_more(df):
    rates = df.groupby("contract")[TARGET].mean()
    assert rates["monthly"] > rates["annual"] > rates["two_year"]


def test_models_learn_signal(df):
    X_train, X_test, y_train, y_test = split(df, seed=0)
    for kind in ("logreg", "gbm"):
        r = evaluate(kind, X_train, X_test, y_train, y_test)
        assert r.test_auc > 0.72, kind
        assert 0 <= r.test_proba.min() and r.test_proba.max() <= 1


def test_campaign_profit_by_hand():
    y = np.array([1, 0, 1, 0])
    contacted = np.array([1, 1, 0, 0])
    clv = np.array([1000, 1000, 1000, 1000])
    # contacted churner: 0.5*1000 - 100 = 400; contacted non-churner: -100
    assert campaign_profit(y, contacted, clv, offer_cost=100, success_rate=0.5) == 300


def test_profit_curve_prefers_targeting_over_everyone():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = np.clip(y * 0.6 + rng.random(500) * 0.4, 0, 1)  # informative scores
    clv = np.full(500, 2000.0)
    curve = profit_curve(y, proba, clv, offer_cost=300, success_rate=0.3)
    assert curve["profit"].iloc[0] == 0  # contact nobody
    assert best_policy(curve)["profit"] >= curve["profit"].iloc[-1]


def test_clv_positive(df):
    assert (customer_lifetime_value(df) > 0).all()

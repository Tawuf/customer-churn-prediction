"""Run the full analysis: ``python -m churn``.

1. generate the synthetic dataset      -> data/customers.csv
2. train and compare two models         (logistic regression vs. gradient boosting)
3. explain the best model               (permutation importance)
4. choose who to target                 (profit curve)
5. write charts + a results summary     -> reports/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .data import TARGET, customer_lifetime_value, generate_customers  # noqa: E402
from .model import evaluate, feature_importance, split  # noqa: E402
from .targeting import best_policy, profit_curve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COLORS = {"logreg": "#2980b9", "gbm": "#c0392b"}
LABELS = {"logreg": "Logistic regression", "gbm": "Gradient boosting"}


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Churn prediction pipeline")
    ap.add_argument("--n", type=int, default=8000, help="number of synthetic customers")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--offer-cost", type=float, default=200.0, help="SEK per contacted customer")
    ap.add_argument("--success-rate", type=float, default=0.35,
                    help="share of real churners the offer retains")
    args = ap.parse_args(argv)

    data_dir, reports = ROOT / "data", ROOT / "reports"
    data_dir.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)

    df = generate_customers(args.n, args.seed)
    df.to_csv(data_dir / "customers.csv", index=False)
    print(f"Customers: {len(df):,}  churn rate: {df[TARGET].mean():.1%}")

    X_train, X_test, y_train, y_test = split(df, seed=args.seed)
    results = {k: evaluate(k, X_train, X_test, y_train, y_test, args.seed) for k in ("logreg", "gbm")}
    for r in results.values():
        print(f"{LABELS[r.name]:20s} CV AUC {r.cv_auc_mean:.3f} ± {r.cv_auc_std:.3f}  "
              f"test AUC {r.test_auc:.3f}  PR-AUC {r.test_pr_auc:.3f}  Brier {r.test_brier:.3f}")
    best = max(results.values(), key=lambda r: r.cv_auc_mean)
    print(f"Selected model (by CV AUC): {LABELS[best.name]}")

    # --- ROC curves -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in results.values():
        fpr, tpr, _ = roc_curve(y_test, r.test_proba)
        ax.plot(fpr, tpr, color=COLORS[r.name], lw=2, label=f"{LABELS[r.name]} (AUC {r.test_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    _style(ax, "ROC curve on hold-out customers")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(reports / "roc_curve.png", dpi=150)
    plt.close(fig)

    # --- Feature importance ----------------------------------------------
    imp = feature_importance(best.model, X_test, y_test, args.seed)
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.iloc[::-1].plot.barh(ax=ax, color="#34495e")
    ax.set_xlabel("Drop in ROC-AUC when feature is shuffled")
    _style(ax, f"What drives churn? ({LABELS[best.name]})")
    fig.tight_layout()
    fig.savefig(reports / "feature_importance.png", dpi=150)
    plt.close(fig)

    # --- Profit curve ----------------------------------------------------
    clv = customer_lifetime_value(X_test).to_numpy()
    curve = profit_curve(y_test.to_numpy(), best.test_proba, clv, args.offer_cost, args.success_rate)
    top = best_policy(curve)
    everyone = curve.iloc[-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve["pct_contacted"], curve["profit"] / 1000, color="#27ae60", lw=2.5)
    ax.axhline(0, color="gray", lw=1)
    ax.scatter([top["pct_contacted"]], [top["profit"] / 1000], color="#27ae60", s=80, zorder=3)
    ax.annotate(f"Best: contact top {top['pct_contacted']:.0f}%\nprofit {top['profit']/1000:,.0f}k SEK",
                (top["pct_contacted"], top["profit"] / 1000), xytext=(15, -45),
                textcoords="offset points", arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlabel("% of customers contacted (highest churn risk first)")
    ax.set_ylabel("Campaign profit (thousand SEK)")
    _style(ax, "Retention campaign profit by targeting depth")
    fig.tight_layout()
    fig.savefig(reports / "profit_curve.png", dpi=150)
    plt.close(fig)

    summary = {
        "customers": len(df),
        "churn_rate": round(float(df[TARGET].mean()), 4),
        "models": {
            r.name: {
                "cv_auc_mean": round(r.cv_auc_mean, 4),
                "cv_auc_std": round(r.cv_auc_std, 4),
                "test_auc": round(r.test_auc, 4),
                "test_pr_auc": round(r.test_pr_auc, 4),
                "test_brier": round(r.test_brier, 4),
            }
            for r in results.values()
        },
        "selected_model": best.name,
        "top_features": [str(f) for f in imp.index[:5]],
        "campaign": {
            "offer_cost_sek": args.offer_cost,
            "success_rate": args.success_rate,
            "best_pct_contacted": int(top["pct_contacted"]),
            "best_profit_sek": round(float(top["profit"])),
            "churners_reached": int(top["churners_reached"]),
            "churners_in_test_set": int(y_test.sum()),
            "profit_if_contact_everyone_sek": round(float(everyone["profit"])),
        },
    }
    (reports / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"Best campaign: contact top {top['pct_contacted']:.0f}% -> profit {top['profit']:,.0f} SEK "
          f"(contacting everyone: {everyone['profit']:,.0f} SEK)")
    print(f"Reports written to {reports}")
    return summary


if __name__ == "__main__":
    main()

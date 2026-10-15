"""Price every click from SKU economics.

MaxCPC = CVR_bayes x contribution per unit x (1 - retained_margin_share)

CVR_bayes is a beta-binomial (empirical Bayes) estimate that shrinks each SKU's
observed conversion rate toward its category rate. The prior strength ``k`` is
measured in clicks: a SKU with ``k`` clicks gets half its weight from its own
data and half from its category.

Why not a minimum-order threshold? Requiring, say, 5 orders before scoring a
SKU only scores products that already succeed. It excludes every growth
candidate by construction. Shrinkage lets every SKU be scored while
stopping thin SKUs from being over-credited.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PRACTICAL_CPC_FLOOR = 0.20


def bayesian_cvr(df: pd.DataFrame, k: float = 300.0, group: str = "category") -> pd.DataFrame:
    out = df.copy()
    cat = out.groupby(group).agg(c_orders=("orders", "sum"), c_clicks=("clicks", "sum"))
    cat["cohort_cvr"] = cat["c_orders"] / cat["c_clicks"]
    out = out.join(cat["cohort_cvr"], on=group)
    out["raw_cvr"] = out["orders"] / out["clicks"]
    out["cvr_bayes"] = (out["orders"] + k * out["cohort_cvr"]) / (out["clicks"] + k)
    out["evidence_weight"] = out["clicks"] / (out["clicks"] + k)
    return out


def price_clicks(
    df: pd.DataFrame,
    k: float = 300.0,
    retained_margin_share: float = 0.0,
    up_threshold: float = 1.25,
    down_threshold: float = 0.80,
    floor: float = PRACTICAL_CPC_FLOOR,
) -> pd.DataFrame:
    """Add MaxCPC, the affordability ratio and a one-word verdict per SKU."""
    out = bayesian_cvr(df, k=k)
    out["max_cpc"] = (out["cvr_bayes"] * out["contribution_per_unit"] * (1 - retained_margin_share)).clip(lower=0)
    out["ratio"] = out["max_cpc"] / out["cpc"]
    out["net_contribution"] = out["orders"] * out["contribution_per_unit"] - out["spend"]

    verdict = np.select(
        [
            out["max_cpc"] < floor,
            out["ratio"] >= up_threshold,
            out["ratio"] <= down_threshold,
        ],
        ["STOP", "BID UP", "BID DOWN"],
        default="HOLD",
    )
    out["verdict"] = verdict
    # stage BID UP by how much of the CVR is the SKU's own data
    out["stage"] = np.where(
        out["verdict"] != "BID UP", "",
        np.where(out["evidence_weight"] >= 0.5, "1 - proven",
                 np.where(out["evidence_weight"] >= 0.2, "2 - emerging", "3 - cohort-led")),
    )
    out["why"] = out.apply(_explain, axis=1)
    return out


def _explain(r) -> str:
    return (
        f"{r.clicks} clicks, {r.orders} orders (own CVR {r.raw_cvr:.2%}); category {r.cohort_cvr:.2%}; "
        f"blended {r.cvr_bayes:.2%} at {r.evidence_weight:.0%} own-evidence. "
        f"${r.contribution_per_unit:,.0f}/unit -> MaxCPC ${r.max_cpc:.2f} vs paid ${r.cpc:.2f} "
        f"(ratio {r.ratio:.2f}) -> {r.verdict}."
    )


def dispersion_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """p10/median/p90 and the p90/p10 spread for what we pay vs what SKUs can afford."""
    rows = {}
    for col, label in [("cpc", "CPC paid"), ("max_cpc", "MaxCPC affordable"), ("contribution_per_unit", "Contribution $/unit")]:
        s = scored[col][scored[col] > 0]
        p10, p50, p90 = np.percentile(s, [10, 50, 90])
        rows[label] = {"p10": p10, "median": p50, "p90": p90, "p90/p10": p90 / p10}
    tab = pd.DataFrame(rows).T
    rho = spearmanr(scored["cpc"], scored["max_cpc"]).correlation
    tab.attrs["spearman_cpc_vs_maxcpc"] = rho
    return tab


def estimator_error(scored: pd.DataFrame) -> pd.Series:
    """Mean absolute error vs the hidden true CVR (synthetic data only)."""
    thin = scored["clicks"] < 100
    return pd.Series(
        {
            "raw CVR, all SKUs": (scored["raw_cvr"] - scored["true_cvr"]).abs().mean(),
            "Bayesian CVR, all SKUs": (scored["cvr_bayes"] - scored["true_cvr"]).abs().mean(),
            "raw CVR, thin SKUs (<100 clicks)": (scored.loc[thin, "raw_cvr"] - scored.loc[thin, "true_cvr"]).abs().mean(),
            "Bayesian CVR, thin SKUs (<100 clicks)": (scored.loc[thin, "cvr_bayes"] - scored.loc[thin, "true_cvr"]).abs().mean(),
        }
    )

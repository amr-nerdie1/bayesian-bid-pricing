"""Constant-budget reallocation.

Move spend away from SKUs that pay more per click than they can afford and
toward SKUs that can afford more, keeping total budget fixed.

Volume response is an assumption, not an observation: clicks scale with spend
as ``clicks_new = clicks * (spend_new / spend) ** elasticity``. With
elasticity < 1 each extra dollar buys fewer clicks. The function reports the
result across a range of elasticities so the conclusion isn't tied to one guess.
That uncertainty is the reason for running a bounded test before scaling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def reallocate(scored: pd.DataFrame, elasticity: float = 0.7, max_scale: float = 3.0) -> dict:
    df = scored.copy()
    budget = df["spend"].sum()
    base = (df["cvr_bayes"] * df["clicks"] * df["contribution_per_unit"]).sum() - budget

    # 1. stop SKUs below floor, scale BID DOWN SKUs to their affordable price
    new_spend = df["spend"].copy()
    stop = df["verdict"] == "STOP"
    down = df["verdict"] == "BID DOWN"
    new_spend[stop] = 0.0
    new_spend[down] = df.loc[down, "spend"] * df.loc[down, "ratio"].clip(0.2, 1.0)
    freed = budget - new_spend.sum()

    # 2. give freed budget to BID UP SKUs, proportional to headroom, capped
    up = df["verdict"] == "BID UP"
    headroom = (df["ratio"] - 1).clip(lower=0) * df["spend"]
    weights = headroom.where(up, 0)
    capacity = (df["spend"] * (max_scale - 1)).where(up, 0)
    alloc = (freed * weights / weights.sum()).clip(upper=capacity) if weights.sum() > 0 else weights * 0
    new_spend = new_spend + alloc
    unplaced = freed - alloc.sum()

    scale = np.where(df["spend"] > 0, new_spend / df["spend"], 0)
    new_clicks = df["clicks"] * np.power(scale, elasticity)
    new_contrib = (df["cvr_bayes"] * new_clicks * df["contribution_per_unit"]).sum() - new_spend.sum()

    return {
        "budget": budget,
        "spend_after": new_spend.sum(),
        "freed": freed,
        "unplaced": unplaced,
        "net_contribution_before": base,
        "net_contribution_after": new_contrib,
        "uplift": new_contrib - base,
    }


def elasticity_sweep(scored: pd.DataFrame, values=(0.4, 0.55, 0.7, 0.85, 1.0)) -> pd.DataFrame:
    return pd.DataFrame([{"elasticity": e, **reallocate(scored, e)} for e in values])

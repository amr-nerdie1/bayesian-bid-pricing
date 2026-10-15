import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bidpricing import simulate_catalogue, price_clicks, estimator_error, reallocate  # noqa: E402


def test_shrinkage_formula_by_hand():
    df = pd.DataFrame({"sku": ["a", "b"], "category": ["x", "x"], "clicks": [100, 900],
                       "orders": [0, 10], "spend": [25.0, 225.0], "cpc": [0.25, 0.25],
                       "contribution_per_unit": [100.0, 100.0]})
    s = price_clicks(df, k=300)
    cohort = 10 / 1000
    assert np.isclose(s.loc[0, "cvr_bayes"], (0 + 300 * cohort) / (100 + 300))
    # zero-order SKU is pulled toward, but stays below, its cohort
    assert 0 < s.loc[0, "cvr_bayes"] < cohort


def test_bayesian_beats_raw_on_thin_skus():
    s = price_clicks(simulate_catalogue(n_skus=3000, seed=1))
    err = estimator_error(s)
    assert err["Bayesian CVR, thin SKUs (<100 clicks)"] < err["raw CVR, thin SKUs (<100 clicks)"]


def test_every_sku_gets_one_verdict():
    s = price_clicks(simulate_catalogue(n_skus=1000, seed=2))
    assert s["verdict"].isin(["BID UP", "BID DOWN", "HOLD", "STOP"]).all()
    assert (s.loc[s["max_cpc"] < 0.20, "verdict"] == "STOP").all()


def test_reallocation_keeps_budget():
    s = price_clicks(simulate_catalogue(n_skus=2000, seed=3))
    r = reallocate(s, elasticity=0.7)
    assert r["spend_after"] <= r["budget"] + 1e-6
    assert np.isclose(r["spend_after"] + r["unplaced"], r["budget"])

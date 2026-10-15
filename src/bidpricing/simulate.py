"""Synthetic sponsored-ads dataset for a large-item furniture catalogue.

Everything here is invented. The generator reproduces the *shape* of the real
problem, not any company's data:

* SKU economics (price, contribution per unit) vary by an order of magnitude.
* True conversion rates vary by category and by SKU.
* Bids are set almost flat: CPC paid barely depends on what a SKU is worth.
* Most SKUs have few clicks, so their observed conversion rate is noisy.

Because we know the *true* conversion rate of every synthetic SKU, we can
check how well each estimator recovers it (see ``model.py`` and the tests).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = {
    # name: (median price $, contribution margin %, mean CVR)
    "Office Desks": (420, 0.14, 0.0040),
    "Filing Cabinets": (260, 0.11, 0.0055),
    "Wall Beds": (1650, 0.15, 0.0022),
    "Dining Sets": (780, 0.09, 0.0028),
    "Accent Chairs": (310, 0.12, 0.0035),
    "Bookcases": (190, 0.08, 0.0045),
    "Sofas": (1150, 0.06, 0.0018),
    "Bar Stools": (140, 0.07, 0.0050),
}


def simulate_catalogue(n_skus: int = 6000, seed: int = 7) -> pd.DataFrame:
    """Return one row per advertised SKU with clicks, orders, spend and economics."""
    rng = np.random.default_rng(seed)
    cats = rng.choice(list(CATEGORIES), size=n_skus)
    med_price = np.array([CATEGORIES[c][0] for c in cats])
    cm_pct = np.array([CATEGORIES[c][1] for c in cats])
    cat_cvr = np.array([CATEGORIES[c][2] for c in cats])

    price = med_price * rng.lognormal(0, 0.55, n_skus)
    # contribution per unit BEFORE marketing; some SKUs are near zero or negative
    contribution = price * np.clip(rng.normal(cm_pct, 0.07), -0.05, 0.40)

    # true SKU conversion rate: beta around the category mean (concentration 400)
    conc = 400
    true_cvr = rng.beta(cat_cvr * conc, (1 - cat_cvr) * conc)

    # heavy-tailed click volume: most SKUs are thin
    clicks = np.round(rng.lognormal(4.2, 1.35, n_skus)).astype(int) + 1
    orders = rng.binomial(clicks, true_cvr)

    # flat bidding: CPC ~ $0.25 with noise that ignores SKU value
    cpc = np.clip(rng.lognormal(np.log(0.25), 0.35, n_skus), 0.08, 1.50)
    spend = clicks * cpc

    return pd.DataFrame(
        {
            "sku": [f"SKU-{i:05d}" for i in range(n_skus)],
            "category": cats,
            "price": price.round(2),
            "contribution_per_unit": contribution.round(2),
            "clicks": clicks,
            "orders": orders,
            "spend": spend.round(2),
            "cpc": cpc.round(3),
            "true_cvr": true_cvr,  # hidden truth, used only for validation
        }
    )


if __name__ == "__main__":
    df = simulate_catalogue()
    df.to_csv("data/synthetic_ads.csv", index=False)
    print(df.describe().T)

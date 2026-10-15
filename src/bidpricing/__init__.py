"""Bayesian bid pricing: price every ad click from SKU economics."""
from .simulate import simulate_catalogue
from .model import bayesian_cvr, price_clicks, dispersion_summary, estimator_error
from .allocate import reallocate, elasticity_sweep

__all__ = [
    "simulate_catalogue", "bayesian_cvr", "price_clicks", "dispersion_summary",
    "estimator_error", "reallocate", "elasticity_sweep",
]

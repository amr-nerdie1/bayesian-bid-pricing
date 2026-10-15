"""End-to-end demo: simulate -> score -> summarise -> reallocate -> figures.

    python run_demo.py
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from bidpricing import simulate_catalogue, price_clicks, dispersion_summary, estimator_error, elasticity_sweep

ROOT = Path(__file__).parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
(ROOT / "data").mkdir(exist_ok=True)

NAVY, TEAL, GREY, AMBER, RED = "#0B1F3A", "#0E8C8C", "#9AA5B1", "#D97706", "#B42318"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 13})


def main():
    raw = simulate_catalogue()
    raw.to_csv(ROOT / "data/synthetic_ads.csv", index=False)
    s = price_clicks(raw)
    s.drop(columns=["true_cvr"]).to_csv(ROOT / "data/scored_skus.csv", index=False)

    disp = dispersion_summary(s)
    rho = disp.attrs["spearman_cpc_vs_maxcpc"]
    err = estimator_error(s)
    verdicts = s["verdict"].value_counts()
    sweep = elasticity_sweep(s)

    print("\n== Dispersion ==\n", disp.round(3))
    print(f"Spearman(CPC paid, MaxCPC) = {rho:+.2f}")
    print("\n== Estimator error vs true CVR ==\n", (err * 100).round(3).astype(str) + " pp")
    print("\n== Verdicts ==\n", verdicts)
    print("\n== Constant-budget reallocation (elasticity sweep) ==\n",
          sweep[["elasticity", "budget", "freed", "unplaced", "net_contribution_before", "net_contribution_after", "uplift"]].round({"elasticity": 2, "budget": 0, "freed": 0, "unplaced": 0, "net_contribution_before": 0, "net_contribution_after": 0, "uplift": 0}).to_string(index=False))

    # Figure 1: what we pay vs what SKUs can afford
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bins = np.logspace(-2.5, 1, 60)
    ax.hist(s.loc[s.max_cpc > 0, "max_cpc"], bins=bins, color=TEAL, alpha=.85, label="What a click is worth (MaxCPC)")
    ax.hist(s["cpc"], bins=bins, color=NAVY, alpha=.75, label="What we pay (CPC)")
    ax.axvline(0.20, color=AMBER, ls="--", lw=1.5); ax.text(0.22, ax.get_ylim()[1]*.80, " $0.20 floor", color=AMBER, fontsize=9)
    ax.set_xscale("log"); ax.set_xlabel("$ per click (log scale)"); ax.set_ylabel("SKUs")
    ax.set_title(f"Bids are flat; economics are not  (p90/p10: {disp.loc['CPC paid','p90/p10']:.1f}x vs {disp.loc['MaxCPC affordable','p90/p10']:.0f}x)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "01_dispersion.png", dpi=160); plt.close(fig)

    # Figure 2: CPC vs MaxCPC scatter
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    col = s["verdict"].map({"BID UP": TEAL, "BID DOWN": RED, "HOLD": GREY, "STOP": "#CBD2D9"})
    ax.scatter(s["cpc"], s["max_cpc"].clip(lower=1e-3), c=col, s=7, alpha=.6, lw=0)
    lim = [0.05, 10]; ax.plot(lim, lim, color=NAVY, lw=1); ax.text(0.75, 0.55, "paid = worth", color=NAVY, fontsize=9, ha="right")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.07, 1.6); ax.set_ylim(1e-3, 10)
    ax.set_xlabel("CPC paid ($)"); ax.set_ylabel("MaxCPC affordable ($)")
    ax.set_title(f"Bids ignore SKU value (Spearman {rho:+.2f})")
    for v, c in [("BID UP", TEAL), ("BID DOWN", RED), ("HOLD", GREY), ("STOP", "#CBD2D9")]:
        ax.scatter([], [], c=c, label=f"{v} ({verdicts.get(v,0):,})")
    ax.legend(frameon=False, loc="lower right", fontsize=9); fig.tight_layout(); fig.savefig(FIG / "02_scatter.png", dpi=160); plt.close(fig)

    # Figure 3: shrinkage beats raw CVR on thin SKUs
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Raw CVR", "Bayesian CVR"]
    allv = [err.iloc[0]*100, err.iloc[1]*100]; thin = [err.iloc[2]*100, err.iloc[3]*100]
    x = np.arange(2); w = .36
    ax.bar(x - w/2, allv, w, color=GREY, label="All SKUs"); ax.bar(x + w/2, thin, w, color=TEAL, label="Thin SKUs (<100 clicks)")
    for i, (a, b) in enumerate(zip(allv, thin)):
        ax.text(i - w/2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=9); ax.text(i + w/2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, labels); ax.set_ylabel("Mean abs. error vs true CVR (pp)")
    ax.set_title("Shrinkage makes thin SKUs scoreable"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "03_shrinkage.png", dpi=160); plt.close(fig)

    # Figure 4: uplift across elasticity assumptions
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep["elasticity"], sweep["uplift"] / 1000, marker="o", color=TEAL, lw=2)
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xlabel("Assumed click elasticity to spend"); ax.set_ylabel("Net contribution uplift ($K)")
    ax.set_title("Same budget, re-priced: uplift holds across assumptions")
    fig.tight_layout(); fig.savefig(FIG / "04_reallocation.png", dpi=160); plt.close(fig)

    return disp, rho, err, verdicts, sweep


if __name__ == "__main__":
    main()

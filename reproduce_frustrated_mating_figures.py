#!/usr/bin/env python3
"""Reproduce summary-statistic calculations and four figures for the companion review.

Inputs: published_correlations.csv (Date Psychology's rounded Table 1 values),
        jealousy_experiment_inputs.csv (Gul et al., 2026 results).
Outputs are written beside this script, overwriting files with the names below.

NO individual participant data are used or reconstructed. The diagrams are
illustrative hypotheses, not fitted models. Neither OLS nor a decomposition
identifies a causal direct or indirect effect with these summary statistics.

Run: python reproduce_frustrated_mating_figures.py
Requires Python >=3.10 and packages listed in requirements.txt.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
ORDER = ["FM", "AP1", "AP2", "PS", "RT", "PT"]
PREDICTORS = ["FM", "RT", "PT"]
OUTCOMES = ["AP1", "AP2", "PS"]
LONG = {
    "FM": "Frustrated mating", "AP1": "Personal restrictions (AP1)",
    "AP2": "Social restrictions (AP2)", "PS": "Scenario approval (PS)",
    "RT": "Sexual risk-taking", "PT": "Perceived threat",
}
INK = "#162e42"
DARKBLUE = "#325f75"
TEAL = "#688f96"
RED = "#bd6759"
MUTED = "#5d7080"
BG = "#f7f4ee"
LINE = "#cfd8d9"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "text.color": INK, "axes.labelcolor": INK, "axes.titleweight": "bold",
    "xtick.color": MUTED, "ytick.color": INK,
    "axes.spines.right": False, "axes.spines.top": False,
    "figure.facecolor": BG, "axes.facecolor": BG,
    "savefig.facecolor": BG,
})


def load_correlations():
    with (ROOT / "published_correlations.csv").open(newline="", encoding="utf-8") as fh:
        data = list(csv.DictReader(fh))
    n = len(ORDER)
    if len(data) != n * (n - 1) // 2:
        raise ValueError("Expected the 15 distinct pairwise correlations of six scales.")
    M = np.eye(n)
    intervals = {}
    for row in data:
        a, b = row["variable_1"], row["variable_2"]
        if a not in ORDER or b not in ORDER or a == b:
            raise ValueError(f"Unexpected variable pair: {a}, {b}")
        i, j = ORDER.index(a), ORDER.index(b)
        if (a, b) in intervals or (b, a) in intervals:
            raise ValueError(f"Repeated correlation pair: {a}, {b}")
        x = float(row["r"])
        lo, hi = float(row["ci_lower_95"]), float(row["ci_upper_95"])
        if not (-1 < x < 1 and -1 < lo < 1 and -1 < hi < 1 and lo <= x <= hi):
            raise ValueError(f"Invalid correlation/CI for {a}–{b}")
        M[i, j] = M[j, i] = x
        intervals[a, b] = (x, lo, hi, row["reported_significance"])
    if np.linalg.eigvalsh(M).min() <= 0:
        raise ValueError("Published rounded correlation matrix is not positive definite")
    return M, intervals


def correlation(M, a, b):
    return float(M[ORDER.index(a), ORDER.index(b)])


def reconstruct(M):
    i_x = [ORDER.index(k) for k in PREDICTORS]
    X = M[np.ix_(i_x, i_x)]
    rows = []
    for outcome in OUTCOMES:
        correlations = M[i_x, ORDER.index(outcome)]
        coefficients = np.linalg.solve(X, correlations)
        r2 = float(coefficients @ correlations)
        # Reduced model (RT + PT) quantifies FM's unique *incremental* R^2.
        X_small, y_small = X[1:, 1:], correlations[1:]
        r2_reduced = float(y_small @ np.linalg.solve(X_small, y_small))
        decomposition = coefficients * X[0, :]
        assert np.isclose(decomposition.sum(), correlation(M, "FM", outcome), atol=1e-12)
        rows.append({
            "outcome": outcome,
            "beta_fm": float(coefficients[0]),
            "beta_rt": float(coefficients[1]),
            "beta_pt": float(coefficients[2]),
            "r_squared": r2,
            "r_squared_without_fm": r2_reduced,
            "fm_incremental_r_squared": r2 - r2_reduced,
            "fm_zero_order_r": correlation(M, "FM", outcome),
            "decomp_fm": float(decomposition[0]),
            "decomp_rt": float(decomposition[1]),
            "decomp_pt": float(decomposition[2]),
        })
    return rows


def fisher_ci(r, n):
    z = np.arctanh(r)
    half = norm.ppf(0.975) / np.sqrt(n - 3)
    return np.tanh([z - half, z + half])


def implied_sample_size_from_ci(r, low, high):
    """Approximate n implied by Fisher-z CI width, subject to 2-decimal rounding."""
    z_halfwidth = (np.arctanh(high) - np.arctanh(low)) / 2
    return float(3 + (norm.ppf(0.975) / z_halfwidth) ** 2)


def approx_correlation_power(r_true, n):
    """Fisher-z normal approximation; power for two-sided alpha=.05 test."""
    signal = np.arctanh(r_true) * np.sqrt(n - 3)
    zcrit = norm.ppf(0.975)
    return float(norm.sf(zcrit - signal) + norm.cdf(-zcrit - signal))


def table_to_csv(rows, name, fieldnames):
    with (ROOT / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def style_axis(ax):
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=LINE, lw=0.8, alpha=0.75)
    ax.set_axisbelow(True)


def figure_01(intervals):
    fig, ax = plt.subplots(figsize=(12.5, 6.0), dpi=170)
    fig.subplots_adjust(left=0.30, right=0.77, top=0.79, bottom=0.23)
    items = [("FM", "AP1"), ("FM", "AP2"), ("FM", "PS")]
    ys = [2, 1, 0]
    for y, item in zip(ys, items):
        r, lo, hi, _ = intervals[item]
        color = DARKBLUE if r < 0 else TEAL
        ax.plot([lo, hi], [y, y], color=color, lw=4, solid_capstyle="round")
        ax.scatter([r], [y], color=color, s=140, zorder=3, edgecolor=BG, linewidth=1.7)
        ax.text(.325, y, f"r = {r:+.2f}    [{lo:+.2f}, {hi:+.2f}]", va="center", fontsize=11, color=INK)
    ax.axvline(0, color=INK, lw=1.25, alpha=.76)
    ax.set_yticks(ys, [LONG[p[1]] for p in items]); ax.set_ylim(-.55, 2.55)
    ax.set_xlim(-.32, .55); ax.set_xticks(np.arange(-.3, .41, .1))
    ax.set_xlabel("Pearson correlation (r)")
    ax.set_title("Frustrated mating: three small correlations", fontsize=19, pad=24)
    style_axis(ax)
    fig.text(.5, .075, "Published 95% CIs transcribed from Date Psychology Table 1; analysis N not specified.",
             ha="center", color=MUTED, fontsize=10)
    fig.savefig(ROOT / "figure_01_frustrated_mating_correlations.png", dpi=170)
    plt.close(fig)


def figure_02(regression):
    fig, ax = plt.subplots(figsize=(13.5, 7.2), dpi=170)
    fig.subplots_adjust(left=.30, right=.96, top=.84, bottom=.22)
    y_bases = [2.0, 1.0, 0.0]
    offsets = {"beta_fm": -0.19, "beta_rt": 0.00, "beta_pt": 0.19}
    colors = {"beta_fm": RED, "beta_rt": TEAL, "beta_pt": DARKBLUE}
    for base, row in zip(y_bases, regression):
        for key in ("beta_fm", "beta_rt", "beta_pt"):
            val = row[key]
            y = base + offsets[key]
            ax.plot([0, val], [y, y], color=colors[key], lw=3.2, alpha=.7)
            ax.scatter([val], [y], color=colors[key], s=85, edgecolor=BG, lw=.7, zorder=4)
            ax.annotate(f"{val:+.3f}", (val, y), xytext=(9 if val >= 0 else -9, 0),
                        textcoords="offset points", ha="left" if val >= 0 else "right", va="center", fontsize=10)
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_yticks(y_bases, [LONG[r["outcome"]] for r in regression])
    ax.set_ylim(-.43, 2.43)
    ax.set_xlim(-.63, .63)
    ax.set_xticks(np.arange(-.6, .61, .2))
    ax.set_xlabel("Standardized regression coefficient (β), reconstructed from rounded correlations")
    ax.set_title("After adjustment, frustrated mating is near zero", fontsize=19, pad=16)
    style_axis(ax)
    handles = [plt.Line2D([0], [0], color=colors[k], marker="o", linestyle="", markersize=9)
               for k in ("beta_fm", "beta_rt", "beta_pt")]
    fig.legend(handles, ["Frustrated mating", "Sexual risk-taking", "Perceived threat"],
               loc="lower center", ncol=3, bbox_to_anchor=(.55, .095), frameon=False, fontsize=11)
    fig.text(.50, .040, "Models condition on FM, RT, PT simultaneously. Not causal effects; exact Ns and raw data unavailable.",
             color=MUTED, ha="center", fontsize=10)
    fig.savefig(ROOT / "figure_02_adjusted_coefficients.png", dpi=170)
    plt.close(fig)


def box(ax, x, y, w, h, label, fill="#e4eef0", fontsize=12):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.023",
                          ec=TEAL, fc=fill, linewidth=1.6, transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=fontsize,
            color=INK, transform=ax.transAxes)


def arrow(ax, xy1, xy2):
    ax.add_patch(FancyArrowPatch(xy1, xy2, transform=ax.transAxes, arrowstyle="-|>",
                                mutation_scale=19, color=DARKBLUE, linewidth=2.0))


def figure_03():
    fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=170)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Two explanations compatible with the same correlations", fontsize=20, pad=16, loc="left")
    ax.text(.03,.85, "A. Possible frustration pathway", fontsize=15, fontweight="bold", color=DARKBLUE, transform=ax.transAxes)
    box(ax,.03,.66,.26,.12,"Romantic frustration")
    box(ax,.37,.66,.26,.12,"Threat / insecurity")
    box(ax,.71,.66,.26,.12,"Restrictive attitudes")
    arrow(ax,(.30,.72),(.36,.72)); arrow(ax,(.64,.72),(.70,.72))
    ax.plot([.03,.97],[.57,.57], color=LINE,lw=1.2,transform=ax.transAxes)
    ax.text(.03,.50, "B. Possible shared dispositions", fontsize=15, fontweight="bold", color=DARKBLUE, transform=ax.transAxes)
    box(ax,.31,.31,.38,.11,"Sociosexuality / beliefs / personality",fill="#dce9e6",fontsize=11)
    box(ax,.05,.07,.35,.11,"Romantic frustration")
    box(ax,.60,.07,.35,.11,"Restrictive attitudes")
    arrow(ax,(.39,.30),(.25,.18)); arrow(ax,(.61,.30),(.78,.18))
    ax.text(.5,.008,"Illustrative pathways, not fitted models. Reverse and reciprocal causation also remain possible.",
            ha="center",va="bottom",fontsize=10,color=MUTED,transform=ax.transAxes)
    fig.savefig(ROOT / "figure_03_competing_models.png", dpi=170, bbox_inches="tight", pad_inches=.35)
    plt.close(fig)


def figure_04(experiments):
    fig, ax = plt.subplots(figsize=(12.5, 6.7), dpi=170)
    fig.subplots_adjust(left=.27, right=.92, top=.82, bottom=.22)
    labels = [f"{r['study']} · {r['group']}" for r in experiments]
    ds = [float(r["cohens_d"]) for r in experiments]
    ps = [float(r["p_value"]) for r in experiments]
    ys = np.arange(len(ds))[::-1]
    bars = ax.barh(ys, ds, height=.48, color=[TEAL, DARKBLUE, TEAL, TEAL], alpha=.91)
    ax.axvline(0, color=INK, lw=1)
    for bar, d, p in zip(bars, ds, ps):
        ax.text(d + .018, bar.get_y()+bar.get_height()/2,
                f"d = {d:.2f}  (p = {p:.3f})", va="center", fontsize=10.5, color=INK)
    ax.set_yticks(ys, labels)
    ax.set_xlim(0, .57); ax.set_xlabel("Reported Cohen's d: jealousy-induction condition versus control")
    ax.set_title("Jealousy induction: mixed experimental results", fontsize=19, pad=19)
    style_axis(ax)
    fig.text(.5,.084,"The Study 2 sex × condition interaction was not significant (p = .349).",ha="center",fontsize=11,color=DARKBLUE)
    fig.text(.5,.045,"Source: Gul et al. (2026), DOI: 10.1080/02699931.2026.2676723. No CIs reported here.",
             ha="center",fontsize=9.5,color=MUTED)
    fig.savefig(ROOT / "figure_04_jealousy_experiment.png", dpi=170)
    plt.close(fig)


def main():
    M, intervals = load_correlations()
    regression = reconstruct(M)
    table_to_csv(regression, "regression_reconstruction.csv", list(regression[0]))
    fm_rows = []
    for out in OUTCOMES:
        r, lo, hi, star = intervals["FM", out]
        row = {"outcome":out,"published_r":r,"published_ci_low":lo,"published_ci_high":hi,
               "approx_n_from_ci_width":implied_sample_size_from_ci(r,lo,hi),"reported_star":star}
        for n in [213, 284]:
            l, u = fisher_ci(r, n)
            row[f"fisher_ci_low_n{n}"] = float(l)
            row[f"fisher_ci_high_n{n}"] = float(u)
        fm_rows.append(row)
    table_to_csv(fm_rows, "sample_size_sensitivity.csv", list(fm_rows[0]))
    with (ROOT / "jealousy_experiment_inputs.csv").open(encoding="utf-8",newline="") as fh:
        experiments = list(csv.DictReader(fh))
    if len(experiments) != 4:
        raise ValueError("Expected four experiment/subgroup comparisons")
    # Check known transcription errors against original Table 1.
    assert np.isclose(correlation(M, "PT", "RT"), 0.0)
    assert np.isclose(correlation(M, "PT", "PS"), -0.22)
    assert np.allclose([r["beta_fm"] for r in regression], [-.01514168, -.01416829, .03028337], atol=1e-6)
    figure_01(intervals); figure_02(regression); figure_03(); figure_04(experiments)
    lines = [
        "Date Psychology frustrated-mating review — summary-statistic reconstruction",
        "SOURCE: Table 1 of Alexander (2024); coefficients/CIs transcribed as published.",
        "RAW DATA: Not available. Rounded printed r and CI; no pairwise n provided.",
        "",
        "STATISTICAL RECONSTRUCTION (standardized OLS; FM, RT, PT simultaneous)",
    ]
    for r in regression:
        lines.append(
            f"{r['outcome']}: beta(FM)={r['beta_fm']:+.6f}; beta(RT)={r['beta_rt']:+.6f}; "
            f"beta(PT)={r['beta_pt']:+.6f}; R2={r['r_squared']:.6f}; "
            f"delta_R2(FM | RT,PT)={r['fm_incremental_r_squared']:.6f}"
        )
        lines.append(
            f"  Algebraic decomposition of r(FM,{r['outcome']})={r['fm_zero_order_r']:+.3f}: "
            f"FM={r['decomp_fm']:+.6f} + RT={r['decomp_rt']:+.6f} + PT={r['decomp_pt']:+.6f}"
        )
    lines += ["", "SAMPLE SIZE SENSITIVITY (Fisher-z approximation; interval bounds rounded)"]
    for row in fm_rows:
        lines.append(f"{row['outcome']}: published r={row['published_r']:+.2f}; CI="
                     f"[{row['published_ci_low']:+.2f},{row['published_ci_high']:+.2f}]; "
                     f"n implied by CI width ~ {row['approx_n_from_ci_width']:.1f}")
    lines += [
        "These numbers are approximate: CI bounds are rounded, some may include boundary rounding,",
        "and different estimators/missing-data treatment could yield different CIs.",
        "The alignment with the male subsample (~213 of 284) is a hypothesis, not established fact.",
        "",
        "APPROXIMATE POWER FOR TRUE r=0.14 (two-sided alpha=.05, Fisher-z normal approximation)",
        *(f"n={n}: power ~ {approx_correlation_power(.14,n):.3f}" for n in (213, 284)),
        "",
        "EXPERIMENTAL FIGURE 4",
        "Gul et al. (2026) report d=0.08 (Study 1 men) and d=0.27 (Study 2 all).",
        "Study 2 subgroup ds are 0.34 (men) and 0.18 (women).",
        "The Study 2 sex-by-condition interaction is non-significant (p=.349).",
        "Subgroup statistical significance differences do not prove a sex difference.",
        "",
        "CAUSAL STATUS: Regressions, correlation decomposition and Figure 3 do not identify mediation,",
        "reverse causality, or evolutionary adaptiveness. They are descriptive reconstructions.",
    ]
    result_text = "\n".join(lines) + "\n"
    (ROOT / "reconstruction_results.txt").write_text(result_text, encoding="utf-8")
    print(result_text)
    print("Wrote 2 CSV outputs, 4 PNGs, and reconstruction_results.txt to", ROOT)


if __name__ == "__main__":
    main()

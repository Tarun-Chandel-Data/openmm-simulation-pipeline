#!/usr/bin/env python3
"""
The hinge contact as a matrix: every compound against every structure.

Eleven compounds and eight structures is a grid of one quantity, which a
heat map shows whole. The margins carry the two averages that matter and
are the point of the figure: down the side, how much the compounds differ
from one another; along the bottom, how much the structures do. Where the
second spread is the larger, the receptor is deciding the contact and the
compound is not, and the figure says so without a word.

A single hue, light to dark, because the quantity is a magnitude. Cells
carry their own number, so the colour is a summary and not the only way to
read a value.

    python plot_hinge_matrix.py --table hinge_table.csv \\
        --value TYR116_poses --out fig_hinge_matrix
"""
import argparse, os, sys
import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("needs pandas")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    sys.exit("needs matplotlib")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 0, "ytick.major.size": 0,
})
INK, MUTED, SURFACE = "#1a1a1a", "#6b6b6b", "#ffffff"
# one hue, light to dark: lightness falls monotonically, which is what a
# magnitude needs and what a rainbow does not give
RAMP = LinearSegmentedColormap.from_list(
    "hinge", ["#f6f4fc", "#cfc4ef", "#a68ee1", "#7b5cd6", "#4a3390",
              "#2b1d55"])
BAR = "#7b5cd6"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", required=True, help="hinge_table.csv")
    p.add_argument("--value", default="TYR116",
                   help="residue column to draw, e.g. TYR116 or ILE117, or "
                        "'hinge' for either of them")
    p.add_argument("--of", choices=("poses", "top"), default="poses",
                   help="over every pose, or the top pose of each run")
    p.add_argument("--only", help="compounds, in the order to draw them")
    p.add_argument("--label", default="")
    p.add_argument("--vmin", type=float,
                   help="low end of the colour scale; the data minimum by "
                        "default, since a scale fixed at zero spends half the "
                        "ramp on a range the data never reaches")
    p.add_argument("--vmax", type=float, help="high end; the data maximum")
    p.add_argument("--width", type=float, default=6.4)
    p.add_argument("--height", type=float, default=4.4)
    p.add_argument("--dpi", type=int, default=1000)
    p.add_argument("--out", default="fig_hinge_matrix")
    a = p.parse_args()

    t = pd.read_csv(a.table)
    num = f"{a.of}_{a.value}" if a.value != "hinge" else f"{a.of}_hinge"
    den = "poses" if a.of == "poses" else "runs"
    for c in ("compound", "structure", num, den):
        if c not in t.columns:
            sys.exit(f"{a.table} has no column {c!r}; it holds "
                     f"{', '.join(t.columns)}")

    t["pct"] = 100.0 * t[num] / t[den].replace(0, np.nan)
    cpds = ([x.strip() for x in a.only.split(",") if x.strip()] if a.only
            else sorted(t["compound"].unique()))
    structs = sorted(t["structure"].unique())
    m = (t.pivot_table(index="compound", columns="structure", values="pct")
          .reindex(index=cpds, columns=structs))
    grid = m.to_numpy(dtype=float)

    fig = plt.figure(figsize=(a.width, a.height))
    gs = fig.add_gridspec(2, 2, width_ratios=[len(structs), 1.6],
                          height_ratios=[len(cpds), 1.5],
                          wspace=0.06, hspace=0.07)
    ax = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[0, 1], sharey=ax)
    axb = fig.add_subplot(gs[1, 0], sharex=ax)
    fig.add_subplot(gs[1, 1]).axis("off")

    lo = a.vmin if a.vmin is not None else float(np.nanmin(grid))
    hi = a.vmax if a.vmax is not None else float(np.nanmax(grid))
    if hi - lo < 1:
        lo, hi = max(0.0, lo - 5), min(100.0, hi + 5)
    im = ax.imshow(grid, cmap=RAMP, vmin=lo, vmax=hi, aspect="auto")
    ax.set_xticks(range(len(structs)))
    # the shared x axis puts the labels back if they are cleared by setting
    # them empty, so they are switched off on this axis instead
    ax.tick_params(labelbottom=False)
    ax.set_yticks(range(len(cpds)))
    ax.set_yticklabels(cpds, color=INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # every cell carries its number, so the colour summarises and the value
    # is still readable in print and to a colour-blind reader
    for i in range(len(cpds)):
        for j in range(len(structs)):
            v = grid[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color=(SURFACE if v > lo + 0.55 * (hi - lo) else INK))

    comp = np.nanmean(grid, axis=1)
    axr.barh(range(len(cpds)), comp, height=0.68, color=BAR,
             edgecolor=SURFACE, linewidth=0.6)
    axr.set_xlim(0, max(100, float(np.nanmax(comp)) * 1.05))
    axr.set_xticks([0, 50, 100])
    axr.set_xticklabels(["0", "50", "100"], color=MUTED, fontsize=7)
    axr.tick_params(labelleft=False)
    for sp in ("top", "right", "left"):
        axr.spines[sp].set_visible(False)
    axr.spines["bottom"].set_color(MUTED)
    axr.set_title("by compound", fontsize=7.5, color=MUTED, pad=4)

    stru = np.nanmean(grid, axis=0)
    axb.bar(range(len(structs)), stru, width=0.68, color=BAR,
            edgecolor=SURFACE, linewidth=0.6)
    axb.set_ylim(0, max(100, float(np.nanmax(stru)) * 1.05))
    axb.set_yticks([0, 50, 100])
    axb.set_yticklabels(["0", "50", "100"], color=MUTED, fontsize=7)
    axb.set_xticks(range(len(structs)))
    axb.set_xticklabels([s.replace("_prep", "").upper() for s in structs],
                        rotation=45, ha="right", color=INK)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        axb.spines[sp].set_color(MUTED)
    # upright, so it cannot be clipped at the left edge of the figure the way
    # a horizontal label outside the axes is
    axb.set_ylabel("by structure", fontsize=7.5, color=MUTED, labelpad=2)

    what = (f"{a.value} hydrogen bond" if a.value != "hinge"
            else "hinge hydrogen bond")
    over = ("across all poses" if a.of == "poses" else "top pose of each run")
    ax.set_title(f"{a.label + '  ' if a.label else ''}{what}, {over}"
                 f"   (% of {den})", fontsize=8.5, color=INK, pad=8)

    # the colour bar sits under the figure, not over the data
    cax = fig.add_axes([0.13, -0.04, 0.34, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    mid = (lo + hi) / 2
    cb.set_ticks([lo, mid, hi])
    cb.ax.set_xticklabels([f"{lo:.0f}%", f"{mid:.0f}%", f"{hi:.0f}%"],
                          color=INK, fontsize=7.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, pad=2)

    sc = np.nanstd(comp, ddof=1) if len(comp) > 1 else 0.0
    ss = np.nanstd(stru, ddof=1) if len(stru) > 1 else 0.0
    fig.text(0.52, -0.035, f"spread between compounds {sc:.0f} points; "
                           f"between structures {ss:.0f} points",
             fontsize=7.5, color=MUTED, ha="left", va="center")

    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=a.dpi, bbox_inches="tight",
                    facecolor=SURFACE)
    print(f"[out] {a.out}.png, {a.out}.pdf   ({a.dpi} dpi)")
    print(f"  spread between compounds {sc:.1f} points, "
          f"between structures {ss:.1f} points")
    print("  " + ", ".join(f"{c} {v:.0f}%" for c, v in
                           sorted(zip(cpds, comp), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())

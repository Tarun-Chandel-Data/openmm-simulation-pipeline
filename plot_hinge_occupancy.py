#!/usr/bin/env python3
"""
Hinge engagement across an ensemble, as a count of conformers.

One bar per compound, its length the conformers in which the best pose bonds
either hinge residue, out of the conformers available. The bar is divided into
the residue reached: one alone, or both bridged.

The axis runs to the number of conformers rather than to the largest bar, so
the unfilled part of each bar is the conformers that made no hinge bond and
can be read directly. The segment order puts the discriminating residue
against the axis, so that block starts from a common baseline across
compounds; a segment floating mid-bar cannot be compared by eye.

    python plot_hinge_occupancy.py --file ens_cx_a2_hinge.csv \
        --receptor a2 --residues TYR110,ILE111 --offset 6 --top 20 \
        --out Figure_hinge.png --dpi 1000
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
    from matplotlib.patches import Patch
except ImportError:
    sys.exit("needs matplotlib")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.0,
    "xtick.major.size": 2.5, "ytick.major.size": 0,
})
# the manuscript's own palette, checked as a categorical triple:
# node validate_palette.js "#d1495b,#7b5cd6,#1b7a4b" --mode light -> all pass,
# worst adjacent dE 20.2 protan / 23.8 normal, and every slot clears 3:1
# against the surface, so no contrast relief is owed. The red is the one this
# work uses for the second subunit throughout, and it carries the residue that
# distinguishes it.
C_ONE_A = "#d1495b"     # the discriminating residue alone
C_BOTH = "#7b5cd6"      # both bridged
C_ONE_B = "#1b7a4b"     # the shared backbone residue alone
INK, MUTED = "#0b0b0b", "#52514e"
GAP = 0.09              # surface gap between stacked segments, in bar units


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True,
                   help="the csv from residue_occupancy.py")
    p.add_argument("--receptor", default="a2")
    p.add_argument("--residues", required=True,
                   help="the two labels, discriminating one first, as they "
                        "appear in the csv column names")
    p.add_argument("--offset", type=int, default=0,
                   help="added to the residue numbers when labelling")
    p.add_argument("--label", default="CK2α′")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--dpi", type=int, default=1000)
    p.add_argument("--out", default="Figure_hinge.png")
    a = p.parse_args()

    res = [x.strip() for x in a.residues.split(",") if x.strip()]
    if len(res) != 2:
        sys.exit("--residues wants exactly two labels")
    r1, r2 = res
    d = pd.read_csv(a.file)
    need = [f"{a.receptor}_{r1}", f"{a.receptor}_{r2}",
            f"{a.receptor}_either", f"{a.receptor}_both",
            f"{a.receptor}_n_conformer"]
    miss = [c for c in need if c not in d.columns]
    if miss:
        sys.exit(f"missing column(s): {', '.join(miss)}\n"
                 f"available: {', '.join(d.columns)}")

    n = int(d[f"{a.receptor}_n_conformer"].median())
    d = d.sort_values([f"{a.receptor}_either", f"{a.receptor}_both"],
                      ascending=False).head(a.top).iloc[::-1]

    both = d[f"{a.receptor}_both"].to_numpy(float)
    only1 = d[f"{a.receptor}_{r1}"].to_numpy(float) - both
    only2 = d[f"{a.receptor}_{r2}"].to_numpy(float) - both
    either = d[f"{a.receptor}_either"].to_numpy(float)
    # the decomposition must reproduce the count it came from, or a segment is
    # being double counted somewhere upstream
    bad = ~np.isclose(only1 + only2 + both, either)
    if bad.any():
        names = ", ".join(d["compound"].to_numpy()[bad][:4])
        sys.exit(f"the segments do not sum to 'either' for: {names}. "
                 f"A conformer is being counted in more than one segment")
    if (only1 < 0).any() or (only2 < 0).any():
        sys.exit("a residue's count is below the 'both' count, which cannot "
                 "happen; check the source table")

    def nm(x):
        num = "".join(c for c in x if c.isdigit())
        al = "".join(c for c in x if not c.isdigit())
        return f"{al.title()}{int(num) + a.offset}" if num else x

    n1, n2 = nm(r1), nm(r2)
    y = np.arange(len(d))
    h = 0.62
    fig, ax = plt.subplots(figsize=(5.4, 0.26 * len(d) + 1.25))

    # segments from the axis: the discriminating residue, then the bridge, then
    # the shared one. A 2px-equivalent gap of surface separates the fills.
    left = np.zeros(len(d))
    for vals, col in ((only1, C_ONE_A), (both, C_BOTH), (only2, C_ONE_B)):
        w = vals.copy()
        shown = np.where(w > 0, np.maximum(w - GAP, w * 0.35), 0.0)
        ax.barh(y, shown, left=left, height=h, color=col, edgecolor="none",
                zorder=3)
        left = left + w

    for yi, v, t in zip(y, either, d[f"{a.receptor}_{r1}"].to_numpy(float)):
        ax.text(v + 0.28, yi, f"{v:.0f}", va="center", ha="left",
                fontsize=6.5, color=INK, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(d["compound"].astype(str), fontsize=6.5)
    ax.set_xlim(0, n)
    ax.set_ylim(-0.75, len(d) - 0.1)
    ax.set_xticks(np.arange(0, n + 1, 5))
    ax.set_xlabel(f"conformers of {n} in which the residue is hydrogen bonded",
                  fontsize=7.5)
    ax.grid(axis="x", color="#ececec", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.5, colors=MUTED)
    for lab in ax.get_yticklabels():
        lab.set_color(INK)
    ax.set_title(f"{a.label} hinge engagement, best pose of each conformer",
                 loc="left", fontsize=8.5, pad=8, color=INK)

    # below the axis label. The offset is given in axes fractions, so it is
    # scaled by the axes height, which grows with the number of rows; without
    # that the gap would widen as compounds are added
    axes_h = fig.get_size_inches()[1] - 1.05
    ax.legend(handles=[Patch(facecolor=C_ONE_A, label=f"{n1} only"),
                       Patch(facecolor=C_BOTH, label="both bridged"),
                       Patch(facecolor=C_ONE_B, label=f"{n2} only")],
              frameon=False, fontsize=6.5, ncol=3, loc="upper left",
              bbox_to_anchor=(0.0, -0.40 / max(axes_h, 0.6)),
              handlelength=1.1, handleheight=0.9, columnspacing=1.3,
              labelcolor=MUTED)

    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"[out] {a.out}  ({a.dpi} dpi, {len(d)} compounds of {len(pd.read_csv(a.file))})")
    # the same numbers as text, for a reader who cannot separate the fills
    txt = os.path.splitext(a.out)[0] + "_values.txt"
    with open(txt, "w") as f:
        f.write(f"{a.label} hinge engagement, conformers of {n}\n")
        f.write(f"{'compound':10s}{n1:>10s}{n2:>10s}{'both':>7s}"
                f"{'either':>9s}\n")
        for _, x in d.iloc[::-1].iterrows():
            f.write(f"{x['compound']:10s}"
                    f"{int(x[f'{a.receptor}_{r1}']):10d}"
                    f"{int(x[f'{a.receptor}_{r2}']):10d}"
                    f"{int(x[f'{a.receptor}_both']):7d}"
                    f"{int(x[f'{a.receptor}_either']):9d}\n")
    print(f"      {txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

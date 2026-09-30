#!/usr/bin/env python3
"""
Affinity of the selected compounds in both subunits, across an ensemble.

Each compound contributes the best pose of every conformer, by pose score; the
affinity of those poses is averaged, and the spread across conformers is drawn
with it.

The values sit near -9 kcal/mol and differ by a few tenths, so a bar from zero
would be twenty parts shared stem to one part signal, and a bar cut to start
near -8 would show a tenfold difference where there is a twentieth. A value on
a scale that does not reach zero is drawn here as a point on its own range
instead, which is the form that does not overstate. --bars overrides this.

    python plot_affinity_compare.py --stats ens_cx_rank_pose_unpaired.csv \
        --select ens_cx_a2_hinge.csv --min-either 17 \
        --col minimizedAffinity --out Figure_affinity.png --dpi 1000
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
    from matplotlib.lines import Line2D
except ImportError:
    sys.exit("needs matplotlib")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.0,
    "xtick.major.size": 2.5, "ytick.major.size": 0,
})
# the diverging pair this work uses for the two subunits, checked as a
# categorical pair: worst adjacent dE 14.3 protan, 28.1 normal, both over 3:1
C_A1, C_A2 = "#2e5eaa", "#d1495b"
INK, MUTED, RANGE = "#0b0b0b", "#52514e", "#cfcece"
LOWER = ("minimizedaffinity", "vina", "energy")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stats", required=True,
                   help="the *_unpaired.csv from seed_delta.py")
    p.add_argument("--select",
                   help="a residue_occupancy csv used to choose compounds")
    p.add_argument("--min-either", type=int,
                   help="keep compounds whose <receptor>_either is at least "
                        "this, read from --select")
    p.add_argument("--only", help="comma-separated compounds, instead of "
                                  "--select")
    p.add_argument("--col", default="minimizedAffinity")
    p.add_argument("--receptor", default="a2",
                   help="the subunit --min-either is read from, and the one "
                        "the ranking is on")
    p.add_argument("--label1", default="CK2α")
    p.add_argument("--label2", default="CK2α′")
    p.add_argument("--bars", action="store_true",
                   help="draw bars from zero instead. Stated in the caption "
                        "written beside the figure, since the shared stem "
                        "then dominates the picture")
    p.add_argument("--dpi", type=int, default=1000)
    p.add_argument("--out", default="Figure_affinity.png")
    a = p.parse_args()

    d = pd.read_csv(a.stats)
    need = [f"{a.col}_{k}_{s}" for k in ("a1", "a2")
            for s in ("mean", "sd", "min", "max")]
    miss = [c for c in need if c not in d.columns]
    if miss:
        sys.exit(f"missing column(s): {', '.join(miss[:4])}\n"
                 f"available: {', '.join(sorted(d.columns)[:14])} ...")

    note = ""
    if a.only:
        keep = [x.strip() for x in a.only.split(",") if x.strip()]
        d = d[d["compound"].isin(keep)]
        note = f"{len(d)} compounds named on the command line"
    elif a.select:
        s = pd.read_csv(a.select)
        col = f"{a.receptor}_either"
        if col not in s.columns:
            sys.exit(f"--select has no column '{col}'")
        if a.min_either is None:
            sys.exit("--select needs --min-either")
        keep = s.loc[s[col] >= a.min_either, "compound"]
        d = d[d["compound"].isin(keep)]
        n_conf = int(s[f"{a.receptor}_n_conformer"].median()) \
            if f"{a.receptor}_n_conformer" in s.columns else 0
        note = (f"compounds engaging the hinge in at least "
                f"{a.min_either}" + (f" of {n_conf}" if n_conf else "")
                + " conformers")
    if not len(d):
        sys.exit("no compound survived the selection")

    lower = any(m in a.col.lower() for m in LOWER)
    # best first: most negative for an energy, largest otherwise
    d = d.sort_values(f"{a.col}_a2_mean", ascending=lower).iloc[::-1]

    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(5.4, 0.34 * len(d) + 1.5))

    lo = min(d[f"{a.col}_a1_min"].min(), d[f"{a.col}_a2_min"].min())
    hi = max(d[f"{a.col}_a1_max"].max(), d[f"{a.col}_a2_max"].max())
    pad = (hi - lo) * 0.06

    if a.bars:
        h = 0.34
        for k, (col, c) in enumerate((("a1", C_A1), ("a2", C_A2))):
            ax.barh(y + (0.19 if k else -0.19),
                    d[f"{a.col}_{col}_mean"], height=h, color=c,
                    edgecolor="none", zorder=3)
        ax.set_xlim(min(0, lo - pad), max(0, hi + pad))
    else:
        for k, (sub, c) in enumerate((("a1", C_A1), ("a2", C_A2))):
            off = -0.17 if k == 0 else 0.17
            ax.hlines(y + off, d[f"{a.col}_{sub}_min"],
                      d[f"{a.col}_{sub}_max"], color=RANGE, lw=1.4,
                      zorder=2)
            m = d[f"{a.col}_{sub}_mean"].to_numpy(float)
            s = d[f"{a.col}_{sub}_sd"].to_numpy(float)
            ax.hlines(y + off, m - s, m + s, color=c, lw=2.2, zorder=3,
                      alpha=0.55)
            ax.scatter(m, y + off, s=22, color=c, edgecolor="#ffffff",
                       linewidth=0.7, zorder=4)
        ax.set_xlim(lo - pad, hi + pad)
        # the mean of the ranked subunit, printed once per compound
        for yi, v in zip(y, d[f"{a.col}_a2_mean"].to_numpy(float)):
            ax.text(hi + pad, yi, f"{v:.2f}", va="center", ha="right",
                    fontsize=6.5, color=INK, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(d["compound"].astype(str), fontsize=7)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.grid(axis="x", color="#ececec", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.5, colors=MUTED)
    for lab in ax.get_yticklabels():
        lab.set_color(INK)
    unit = " (kcal/mol)" if lower else ""
    ax.set_xlabel(f"{a.col}{unit}, best pose of each conformer"
                  + (", more negative is stronger" if lower else ""),
                  fontsize=7.5)
    ax.set_title(note or f"{a.col} by subunit", loc="left", fontsize=8.5,
                 pad=8, color=INK)

    handles = [Line2D([], [], marker="o", ls="none", ms=5, color=C_A1,
                      markeredgecolor="#ffffff", label=a.label1),
               Line2D([], [], marker="o", ls="none", ms=5, color=C_A2,
                      markeredgecolor="#ffffff", label=a.label2)]
    if not a.bars:
        handles += [Line2D([], [], color=RANGE, lw=1.4,
                           label=f"range over the conformers"),
                    Line2D([], [], color=MUTED, lw=2.2, alpha=0.55,
                           label="± 1 sd")]
    axes_h = fig.get_size_inches()[1] - 1.1
    ax.legend(handles=handles, frameon=False, fontsize=6.5,
              ncol=min(4, len(handles)), loc="upper left",
              bbox_to_anchor=(0.0, -0.42 / max(axes_h, 0.6)),
              handlelength=1.4, columnspacing=1.4, labelcolor=MUTED)

    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"[out] {a.out}  ({a.dpi} dpi, {len(d)} compounds)")

    txt = os.path.splitext(a.out)[0] + "_values.txt"
    with open(txt, "w") as f:
        f.write(f"{a.col}, best pose of each conformer\n")
        if note:
            f.write(note + "\n")
        if a.bars:
            f.write("drawn as bars from zero; the shared stem dominates the "
                    "picture and the differences are a small part of each "
                    "bar\n")
        f.write(f"\n{'compound':10s}"
                f"{a.label1 + ' mean':>14s}{'sd':>7s}{'min':>8s}{'max':>8s}"
                f"{a.label2 + ' mean':>15s}{'sd':>7s}{'min':>8s}{'max':>8s}\n")
        for _, x in d.iloc[::-1].iterrows():
            f.write(f"{x['compound']:10s}"
                    + "".join(f"{x[f'{a.col}_{s}_{q}']:>{w}.3f}"
                              for s, w0 in (("a1", 14), ("a2", 15))
                              for q, w in ((("mean"), w0), ("sd", 7),
                                           ("min", 8), ("max", 8)))
                    + "\n")
    print(f"      {txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

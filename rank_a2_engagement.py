#!/usr/bin/env python3
"""
Rank compounds on engagement of the second subunit, by a stated order of
precedence: hydrogen bonds first, then pose score, then affinity.

Later keys break ties in earlier ones and never override them, so the order is
the one declared and not a weighting hidden in a composite score.

    python rank_a2_engagement.py --table comparison.csv --top 25 \
        --out figures/Figure_rank.png --dpi 1000
"""
import argparse, os, sys
import numpy as np, pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("needs matplotlib")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})
INK, HUE = "#1a1a1a", "#d1495b"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", default="comparison.csv")
    p.add_argument("--id-col", default="cpd_id")
    p.add_argument("--hbond-col", default="a2_n_hbond")
    p.add_argument("--pose-col", default="a2_pose")
    p.add_argument("--affinity-col", default="a2_cnn")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--label", default="CK2α′")
    p.add_argument("--dpi", type=int, default=1000)
    p.add_argument("--out", default="Figure_rank.png")
    a = p.parse_args()

    d = pd.read_csv(a.table)
    keys = [(a.hbond_col, "Hydrogen bonds"),
            (a.pose_col, "Pose score"),
            (a.affinity_col, "CNN affinity")]
    missing = [c for c, _ in keys if c not in d.columns]
    if missing:
        sys.exit(f"missing column(s): {', '.join(missing)}\n"
                 f"available: {', '.join(d.columns[:16])}")
    if a.id_col not in d.columns:
        sys.exit(f"missing id column '{a.id_col}'")

    cols = [c for c, _ in keys]
    d = d.dropna(subset=cols).copy()
    # identifier last, so an otherwise complete tie is still reproducible
    ranked = d.sort_values(cols + [a.id_col],
                           ascending=[False]*len(cols) + [True],
                           kind="mergesort").reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    log = [f"[in] {len(d)} compounds from {a.table}",
           f"     precedence: {a.hbond_col} > {a.pose_col} > {a.affinity_col}"]
    hb = ranked[a.hbond_col]
    log.append(f"     {a.hbond_col}: {hb.min():g} to {hb.max():g}, "
               f"median {hb.median():g}")
    log.append(f"     {int((hb == hb.max()).sum())} compounds hold the maximum "
               f"of {hb.max():g}, ordered among themselves by {a.pose_col} "
               f"then {a.affinity_col}")
    out_csv = os.path.splitext(a.out)[0] + "_ranking.csv"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    ranked[["rank", a.id_col] + cols].to_csv(out_csv, index=False)

    log.append(f"\n  top {a.top}:")
    for _, r in ranked.head(a.top).iterrows():
        log.append(f"    {r['rank']:3d}  {str(r[a.id_col]):16s} "
                   f"{a.hbond_col} {r[a.hbond_col]:g}   "
                   f"{a.pose_col} {r[a.pose_col]:.3f}   "
                   f"{a.affinity_col} {r[a.affinity_col]:.2f}")

    sub = ranked.head(a.top).iloc[::-1]          # best at the top of the axis
    y = np.arange(len(sub))
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 0.20*len(sub) + 1.0),
                             sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 1],
                                          "wspace": 0.22})
    for ax, (col, title) in zip(axes, keys):
        v = sub[col].to_numpy(float)
        integer = np.allclose(v, np.round(v))
        # bars from zero for a count, which is a magnitude; a floating score
        # sits on a scale that does not reach zero, so it is drawn as a point
        # on its own range rather than a bar whose length would mislead
        if integer and v.min() >= 0:
            ax.barh(y, v, height=0.62, color=HUE, edgecolor="none")
            ax.set_xlim(0, v.max() * 1.18)
            for yi, vi in zip(y, v):
                ax.text(vi, yi, f" {vi:.0f}", va="center", ha="left",
                        fontsize=6.5, color=INK)
        else:
            ax.scatter(v, y, s=14, color=HUE, edgecolor="none", zorder=3)
            pad = (v.max() - v.min() or 1.0) * 0.18
            ax.set_xlim(v.min() - pad, v.max() + pad)
            ax.grid(axis="x", color="#e8e8e8", lw=0.5, zorder=0)
            ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=8, pad=5)
        ax.tick_params(labelsize=6.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(sub[a.id_col].astype(str), fontsize=6)
    axes[0].set_ylim(-0.8, len(sub) - 0.2)
    axes[0].set_ylabel(f"Ranked by engagement of {a.label}", fontsize=7.5)

    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("\n".join(log))
    with open(os.path.splitext(a.out)[0] + "_values.txt", "w") as f:
        f.write("\n".join(log) + "\n")
    print(f"\n[out] {a.out}  ({a.dpi} dpi)")
    print(f"      {out_csv}")


if __name__ == "__main__":
    sys.exit(main())

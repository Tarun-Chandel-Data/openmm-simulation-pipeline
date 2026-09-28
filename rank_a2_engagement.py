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
INK = "#1a1a1a"
# the validated diverging pair: one hue per subunit, used nowhere else
C_A1, C_A2 = "#2e5eaa", "#d1495b"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", default="comparison.csv")
    p.add_argument("--id-col", default="cpd_id")
    p.add_argument("--hbond-col", default="a2_n_hbond")
    p.add_argument("--pose-col", default="a2_pose")
    p.add_argument("--affinity-col", default="a2_cnn")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--label", default="CK2α′")
    p.add_argument("--label1", default="CK2α",
                   help="name of the subunit the ranking is NOT on. Its values "
                        "are carried beside the ranked ones, so a compound "
                        "engaging both equally cannot be mistaken for one that "
                        "prefers the ranked subunit")
    p.add_argument("--min-pose", type=float,
                   help="drop compounds whose pose score is below this before "
                        "ranking. Precedence is strict, so without a floor one "
                        "extra hydrogen bond outranks any pose difference "
                        "however large")
    p.add_argument("--no-compare", action="store_true",
                   help="rank on the second subunit alone, without showing the "
                        "first")
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
    # the same quantity for the other subunit, found by name. It never enters
    # the ranking; it is shown so a compound that engages both equally cannot
    # be mistaken for one that prefers the ranked subunit.
    other = {}
    if not a.no_compare:
        for c in cols:
            cand = c.replace("a2", "a1").replace("A2", "A1")
            if cand != c and cand in d.columns and \
                    pd.api.types.is_numeric_dtype(d[cand]):
                other[c] = cand
    d = d.dropna(subset=cols).copy()
    n_raw = len(d)
    if a.min_pose is not None:
        before = len(d)
        d = d[d[a.pose_col] >= a.min_pose]
        log_pre = (f"[filter] {len(d)} of {before} compounds have "
                   f"{a.pose_col} >= {a.min_pose:g}")
    else:
        log_pre = None
    # identifier last, so an otherwise complete tie is still reproducible
    ranked = d.sort_values(cols + [a.id_col],
                           ascending=[False]*len(cols) + [True],
                           kind="mergesort").reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    log = [f"[in] {n_raw} compounds from {a.table}"] + \
          ([log_pre] if log_pre else []) + \
          [
           f"     precedence: {a.hbond_col} > {a.pose_col} > {a.affinity_col}"]
    hb = ranked[a.hbond_col]
    log.append(f"     {a.hbond_col}: {hb.min():g} to {hb.max():g}, "
               f"median {hb.median():g}")
    log.append(f"     {int((hb == hb.max()).sum())} compounds hold the maximum "
               f"of {hb.max():g}, ordered among themselves by {a.pose_col} "
               f"then {a.affinity_col}")
    if other:
        log.append(f"     showing alongside (not ranked on): "
                   f"{', '.join(other.values())}")
        for c, o in other.items():
            dd = ranked[c] - ranked[o]
            log.append(f"       {c} - {o}: median {dd.median():+.3f}, "
                       f"{int((dd > 0).sum())} of {len(dd)} higher in {a.label}")
    else:
        log.append(f"     [note] no matching {a.label1} columns found, so the "
                   f"ranking is shown without its counterpart")

    out_csv = os.path.splitext(a.out)[0] + "_ranking.csv"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    keep = ["rank", a.id_col]
    for c in cols:
        keep.append(c)
        if c in other:
            keep.append(other[c])
            ranked["d_" + c] = ranked[c] - ranked[other[c]]
            keep.append("d_" + c)
    ranked[keep].to_csv(out_csv, index=False)

    log.append(f"\n  top {a.top}   ({a.label} | {a.label1})"
               if other else f"\n  top {a.top}:")
    for _, r in ranked.head(a.top).iterrows():
        bits = []
        for c, fmt in ((a.hbond_col, "{:g}"), (a.pose_col, "{:.3f}"),
                       (a.affinity_col, "{:.2f}")):
            v = fmt.format(r[c])
            if c in other:
                bits.append(f"{c} {v} | {fmt.format(r[other[c]])}")
            else:
                bits.append(f"{c} {v}")
        log.append(f"    {r['rank']:3d}  {str(r[a.id_col]):16s} "
                   + "   ".join(bits))

    sub = ranked.head(a.top).iloc[::-1]          # best at the top of the axis
    y = np.arange(len(sub))
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 0.20*len(sub) + 1.0),
                             sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 1],
                                          "wspace": 0.22})
    for ax, (col, title) in zip(axes, keys):
        v = sub[col].to_numpy(float)
        w = sub[other[col]].to_numpy(float) if col in other else None
        allv = v if w is None else np.concatenate([v, w])
        integer = np.allclose(allv, np.round(allv))
        # bars from zero for a count, which is a magnitude; a score on a scale
        # that does not reach zero is a point on its own range, since a bar
        # there would have to be cut and its length would mislead
        if integer and allv.min() >= 0:
            h = 0.36 if w is not None else 0.62
            if w is not None:
                ax.barh(y - h/2, w, height=h, color=C_A1, edgecolor="none")
                ax.barh(y + h/2, v, height=h, color=C_A2, edgecolor="none")
            else:
                ax.barh(y, v, height=h, color=C_A2, edgecolor="none")
            ax.set_xlim(0, allv.max() * 1.20)
            for yi, vi in zip(y, v):
                ax.text(vi, yi + (h/2 if w is not None else 0), f" {vi:.0f}",
                        va="center", ha="left", fontsize=6, color=INK)
        else:
            if w is not None:
                ax.hlines(y, np.minimum(v, w), np.maximum(v, w),
                          color="#cfcfcf", lw=0.8, zorder=2)
                ax.scatter(w, y, s=11, color=C_A1, edgecolor="none", zorder=3)
            ax.scatter(v, y, s=11, color=C_A2, edgecolor="none", zorder=4)
            pad = (allv.max() - allv.min() or 1.0) * 0.15
            ax.set_xlim(allv.min() - pad, allv.max() + pad)
            ax.grid(axis="x", color="#ececec", lw=0.5, zorder=0)
            ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=8, pad=5)
        ax.tick_params(labelsize=6.5)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(sub[a.id_col].astype(str), fontsize=6)
    axes[0].set_ylim(-0.8, len(sub) - 0.2)
    axes[0].set_ylabel(f"Ranked by engagement of {a.label}", fontsize=7.5)
    if other:
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(facecolor=C_A2, label=a.label),
                            Patch(facecolor=C_A1, label=a.label1)],
                   frameon=False, fontsize=7, ncol=2,
                   loc="lower center", bbox_to_anchor=(0.5, -0.045))

    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("\n".join(log))
    with open(os.path.splitext(a.out)[0] + "_values.txt", "w") as f:
        f.write("\n".join(log) + "\n")
    print(f"\n[out] {a.out}  ({a.dpi} dpi)")
    print(f"      {out_csv}")


if __name__ == "__main__":
    sys.exit(main())

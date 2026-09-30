#!/usr/bin/env python3
"""
Every conformer's value for every compound, with the conformer effect removed.

The conformer a pose was docked into moves the score more than the compound
does: in the ensembles this was written for, the spread across conformers is
several times the spread across compounds. Plotted raw, all the compounds
therefore look alike, because the same conformer variation is smeared through
every one of them.

Subtracting each conformer's mean over the compounds removes that shared term.
Nothing is discarded: the term taken out is common to every compound in that
conformer and so carries no information about which compound is better. What
remains is the compound's advantage over the field in the same conformer.

Every point is drawn, not a summary, so the width of each cloud is visible
against the distance between compounds. Where a criterion is an energy the
centred value is negated, so right is better in every panel.

    python plot_strip_centred.py \
        --criterion "pose score:ens_vb_matrix_pose_sel" \
        --criterion "affinity:ens_vb_matrix_aff_sel" \
        --out Figure_strip.png --dpi 1000
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
# the pair this work uses for the two subunits, checked as a categorical pair:
# worst adjacent dE 14.3 protan, 28.1 normal, both clear 3:1 on white
C_A1, C_A2 = "#2e5eaa", "#d1495b"
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#d8d7d4"
LOWER = ("affin", "energy", "vina")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--criterion", action="append", required=True,
                   metavar="NAME:BASE",
                   help="a panel: the name to print, and the base path of the "
                        "matrix_table output, which is completed with "
                        "_<receptor>_byrank.csv. Repeat for each panel")
    p.add_argument("--receptors", default="a2,a1",
                   help="two labels; the first orders the rows")
    p.add_argument("--label1", default="CK2α")
    p.add_argument("--label2", default="CK2α′")
    p.add_argument("--only", help="comma-separated compounds to keep")
    p.add_argument("--marks", choices=["interval", "strip"],
                   default="interval",
                   help="interval draws a median, the middle half and the "
                        "full range: 2 marks per compound. strip draws every "
                        "conformer as a dot, which at 20 conformers and two "
                        "subunits is 40 overlapping points per row and is "
                        "unreadable past a handful of compounds")
    p.add_argument("--raw", action="store_true",
                   help="do not remove the conformer effect. The compounds "
                        "then differ by less than the conformer spread and "
                        "the panel shows that rather than a ranking")
    p.add_argument("--dpi", type=int, default=1000)
    p.add_argument("--out", default="Figure_strip.png")
    a = p.parse_args()

    recs = [x.strip() for x in a.receptors.split(",") if x.strip()]
    if len(recs) != 2:
        sys.exit("--receptors wants exactly two labels")
    r2, r1 = recs                     # first is the one rows are ordered by

    crits = []
    for spec in a.criterion:
        if ":" not in spec:
            sys.exit(f"--criterion wants NAME:BASE, got {spec!r}")
        name, base = spec.split(":", 1)
        mats = {}
        for r in recs:
            f = os.path.expanduser(f"{base.strip()}_{r}_byrank.csv")
            if not os.path.exists(f):
                sys.exit(f"not found: {f}")
            m = pd.read_csv(f, index_col=0)
            mats[r] = m
        crits.append((name.strip(), mats))

    keep = None
    if a.only:
        keep = [x.strip() for x in a.only.split(",") if x.strip()]
    common = None
    for _, mats in crits:
        for r in recs:
            s = set(mats[r].index)
            common = s if common is None else (common & s)
    if keep:
        missing = [k for k in keep if k not in common]
        if missing:
            sys.exit(f"not present in every file: {', '.join(missing)}")
        common = [k for k in keep if k in common]
    else:
        common = sorted(common)
    if len(common) < 2:
        sys.exit("fewer than two compounds are in every file")

    log = [f"[in] {len(crits)} criteria, {len(common)} compounds",
           f"     {'raw values' if a.raw else 'each conformer centred on its mean over the compounds shown'}"]

    def prep(m, name):
        v = m.loc[common].to_numpy(float)
        lo = any(x in name.lower() for x in LOWER)
        if not a.raw:
            # the conformer's mean over the compounds shown, removed column by
            # column. Computed on this set, so adding or removing compounds
            # shifts it; that is stated rather than hidden
            v = v - np.nanmean(v, axis=0, keepdims=True)
        if lo:
            v = -v                    # an energy: negate so right is better
        return v, lo

    # order by the median of the first criterion in the first receptor
    v0, _ = prep(crits[0][1][r2], crits[0][0])
    order = np.argsort(np.nanmedian(v0, axis=1))
    names = [common[i] for i in order]

    n = len(names)
    fig, axes = plt.subplots(1, len(crits), sharey=True,
                             figsize=(3.1 * len(crits) + 0.9,
                                      0.27 * n + 1.45),
                             gridspec_kw={"wspace": 0.12})
    if len(crits) == 1:
        axes = [axes]

    rng = np.random.default_rng(0)
    for ax, (name, mats) in zip(axes, crits):
        lo = False
        for r, col, off in ((r1, C_A1, -0.19), (r2, C_A2, 0.19)):
            v, lo = prep(mats[r], name)
            v = v[order]
            for i in range(n):
                row = v[i][~np.isnan(v[i])]
                if not row.size:
                    continue
                y = i + off
                q1, med, q3 = np.percentile(row, [25, 50, 75])
                if a.marks == "strip":
                    j = rng.uniform(-0.06, 0.06, row.size)
                    ax.scatter(row, np.full(row.size, y) + j, s=3.2,
                               color=col, alpha=0.55, linewidths=0, zorder=3)
                else:
                    # the full range behind, the middle half in front, the
                    # median on top: the shape of the distribution without one
                    # mark per conformer
                    ax.plot([row.min(), row.max()], [y, y], color=col,
                            lw=0.8, alpha=0.45, zorder=2,
                            solid_capstyle="butt")
                    ax.plot([q1, q3], [y, y], color=col, lw=3.4, alpha=0.85,
                            zorder=3, solid_capstyle="butt")
                ax.plot([med, med], [y - 0.135, y + 0.135],
                        color="#ffffff" if a.marks == "interval" else col,
                        lw=1.6 if a.marks == "interval" else 1.5, zorder=4,
                        solid_capstyle="butt")
        if not a.raw:
            ax.axvline(0, color=RULE, lw=0.8, zorder=1)
        ax.grid(axis="x", color="#f0f0ef", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        unit = " (kcal/mol)" if lo else ""
        ax.set_xlabel(("advantage over the field" if not a.raw else name)
                      + unit + "\n" + ("→ better" if not a.raw else ""),
                      fontsize=7.5, linespacing=1.5)
        ax.set_title(name, loc="left", fontsize=8.5, pad=6, color=INK)
        ax.tick_params(labelsize=6.5, colors=MUTED)

    axes[0].set_yticks(np.arange(n))
    axes[0].set_yticklabels(names, fontsize=6.5)
    axes[0].set_ylim(-0.8, n - 0.2)
    for lab in axes[0].get_yticklabels():
        lab.set_color(INK)

    if a.marks == "interval":
        h = [Line2D([], [], color=C_A2, lw=3.4, label=a.label2),
             Line2D([], [], color=C_A1, lw=3.4, label=a.label1),
             Line2D([], [], color=MUTED, lw=0.8, alpha=0.45,
                    label="full range over the conformers"),
             Line2D([], [], color=MUTED, lw=3.4, alpha=0.85,
                    label="middle half")]
    else:
        h = [Line2D([], [], marker="o", ls="none", ms=4.5, color=C_A2,
                    label=a.label2),
             Line2D([], [], marker="o", ls="none", ms=4.5, color=C_A1,
                    label=a.label1),
             Line2D([], [], color=MUTED, lw=1.5, label="median")]
    axes_h = fig.get_size_inches()[1] - 1.35
    axes[0].legend(handles=h, frameon=False, fontsize=6.5, ncol=len(h),
                   loc="upper left",
                   bbox_to_anchor=(0.0, -0.62 / max(axes_h, 0.6)),
                   handlelength=1.3, columnspacing=1.4, labelcolor=MUTED)

    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    # how far the compounds separate against how wide each cloud is: the
    # number the picture is making, stated so it is not only eyeballed
    log.append("")
    for name, mats in crits:
        for r, lab in ((r2, a.label2), (r1, a.label1)):
            v, _ = prep(mats[r], name)
            meds = np.nanmedian(v, axis=1)
            within = float(np.nanmedian(np.nanstd(v, axis=1, ddof=1)))
            between = float(np.nanstd(meds, ddof=1))
            log.append(f"  {name:12s} {lab:6s} spread of the medians "
                       f"{between:.3f}; median spread within a compound "
                       f"{within:.3f}  ->  {between/within:.2f}x")
    log.append("  a ratio below 1 means the compounds differ by less than one "
               "of them varies")
    text = "\n".join(log)
    print(text)
    with open(os.path.splitext(a.out)[0] + "_values.txt", "w") as f:
        f.write(text + "\n")
    print(f"\n[out] {a.out}  ({a.dpi} dpi, {n} compounds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

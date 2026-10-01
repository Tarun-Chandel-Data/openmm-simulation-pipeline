#!/usr/bin/env python3
"""
Pose score against affinity, for an ensemble and for crystal structures.

Each point is one receptor's best pose for one compound: twenty points per
compound from an ensemble of twenty conformers, three from three crystal
structures. Plotting every receptor rather than each compound's single best
pose is the point of the figure. One point per compound shows ten dots and
hides the only thing that matters here, which is how far one compound's own
range reaches compared with the gap between compounds.

Most compounds are drawn in grey and a named few in colour. Eleven
categorical hues would not survive a colour-vision check and would read as
confetti; grey with a few picked out says "these sit inside the same cloud as
everything else", which is what the data says.

    python plot_score_affinity.py \\
        --panel 'CX-4945 ensemble (CK2a-prime):ens_cx_dock/poses/*.sdf' \\
        --panel 'Crystal structures:crystal_summary.csv' \\
        --highlight VB004,EV042,EV043 --out fig_score_affinity
"""
import argparse, glob, os, re, sys
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")                 # no display on a compute node
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator
except ImportError:
    sys.exit("needs matplotlib")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.minor.size": 1.3, "ytick.minor.size": 1.3,
})
INK, MUTED = "#1a1a1a", "#6b6b6b"
GREY = "#b4b4b4"
# node validate_palette.js "#d1495b,#7b5cd6,#1b7a4b" --mode light -> all pass,
# worst adjacent dE 20.2 protan / 23.8 normal
HILITE = ["#d1495b", "#7b5cd6", "#1b7a4b"]
SURFACE = "#ffffff"


def cid(path):
    b = os.path.basename(path)
    for e in (".sdf.gz", ".sdf"):
        if b.endswith(e):
            b = b[: -len(e)]
            break
    b = re.sub(r"\.mol(_docked)?$|_docked$|_out$|_poses$", "", b)
    # ensemble runs name a pose EV001__a1c00__a1; the compound comes first
    return b.split("__")[0]


def points_from_sdf(pattern, xk, yk, selk):
    """One point per compound per receptor: the best pose that receptor gave."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    def prop(m, k):
        if not m.HasProp(k):
            return np.nan
        try:
            return float(m.GetProp(k))
        except ValueError:
            return np.nan

    # a directory is taken to mean every pose file under it, at any depth,
    # so the layout of an ensemble run does not have to be worked out first
    if os.path.isdir(pattern):
        files = sorted(glob.glob(os.path.join(pattern, "**", "*.sdf"),
                                 recursive=True)
                       + glob.glob(os.path.join(pattern, "**", "*.sdf.gz"),
                                   recursive=True))
    else:
        files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        sys.exit(f"no pose files under {pattern!r}")
    print(f"[in] {len(files)} pose files under {pattern}")
    low = "cnn" not in selk.lower() and any(
        m in selk.lower() for m in ("affinity", "vina", "energy"))
    sign = -1.0 if low else 1.0
    out = {}
    for f in files:
        c = cid(f)
        # the receptor is the directory when poses are filed per conformer,
        # and the rest of the file name when they are filed per compound
        rec = os.path.basename(os.path.dirname(f))
        tail = os.path.basename(f).split("__")
        if len(tail) > 1:
            rec = tail[1]
        top, tv = None, None
        for m in Chem.SDMolSupplier(f, removeHs=False, sanitize=True):
            if m is None:
                continue
            v = prop(m, selk)
            if np.isnan(v):
                continue
            if tv is None or sign * v > sign * tv:
                top, tv = m, v
        if top is None:
            continue
        x, y = prop(top, xk), prop(top, yk)
        if not (np.isnan(x) or np.isnan(y)):
            out.setdefault(c, []).append((x, y, rec))
    return out


def points_from_csv(path, xk, yk):
    import pandas as pd
    t = pd.read_csv(path)
    for k in ("compound", xk, yk):
        if k not in t.columns:
            sys.exit(f"{path} has no column {k!r}; it holds "
                     f"{', '.join(t.columns)}")
    lab = "structure" if "structure" in t.columns else None
    out = {}
    for _, r in t.iterrows():
        if np.isnan(r[xk]) or np.isnan(r[yk]):
            continue
        out.setdefault(str(r["compound"]), []).append(
            (float(r[xk]), float(r[yk]), str(r[lab]) if lab else ""))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", action="append", required=True,
                   metavar="LABEL:SOURCE",
                   help="a glob of pose sdf files, or a csv with compound and "
                        "the two property columns")
    p.add_argument("--x", default="CNNscore")
    p.add_argument("--y", default="minimizedAffinity")
    p.add_argument("--select-by", default="CNNscore",
                   help="property the best pose per receptor is chosen on")
    p.add_argument("--highlight", default="VB004,EV042,EV043")
    p.add_argument("--only", help="restrict to these compounds")
    p.add_argument("--xlim", help="e.g. 0.75,1.0")
    p.add_argument("--ylim",
                   help="e.g. --ylim=\"-8,-11.5\" with the first value at "
                        "the top. The equals sign is needed because the value "
                        "starts with a minus")
    p.add_argument("--xmajor", type=float, default=0.05)
    p.add_argument("--xminor", type=float, default=0.01)
    p.add_argument("--ymajor", type=float, default=0.5)
    p.add_argument("--yminor", type=float, default=0.1)
    p.add_argument("--xlabel", default="pose score (CNNscore)")
    p.add_argument("--ylabel", default="affinity (kcal/mol)")
    p.add_argument("--width", type=float, default=6.6)
    p.add_argument("--height", type=float, default=3.1)
    p.add_argument("--dpi", type=int, default=1000)
    p.add_argument("--out", default="fig_score_affinity")
    a = p.parse_args()

    hl = [x.strip() for x in a.highlight.split(",") if x.strip()]
    keep = ([x.strip() for x in a.only.split(",") if x.strip()]
            if a.only else None)
    if len(hl) > len(HILITE):
        sys.exit(f"at most {len(HILITE)} compounds can be highlighted; the "
                 f"palette is validated for that many and a fourth hue would "
                 f"not be")

    panels = []
    for spec in a.panel:
        if ":" not in spec:
            sys.exit(f"--panel wants LABEL:SOURCE, got {spec!r}")
        lab, src = spec.split(":", 1)
        src = os.path.expanduser(src.strip())
        if src.lower().endswith(".csv"):
            d = points_from_csv(src, a.x, a.y)
        else:
            d = points_from_sdf(src, a.x, a.y, a.select_by)
        if keep:
            d = {k: v for k, v in d.items() if k in keep}
        if not d:
            sys.exit(f"no points for panel {lab!r}")
        panels.append((lab.strip(), d))

    allx = [x for _, d in panels for v in d.values() for x, _, _ in v]
    ally = [y for _, d in panels for v in d.values() for _, y, _ in v]
    if a.xlim:
        xlo, xhi = (float(v) for v in a.xlim.split(","))
    else:
        pad = 0.04 * (max(allx) - min(allx) or 1)
        xlo, xhi = min(allx) - pad, max(allx) + pad
    if a.ylim:
        ytop, ybot = (float(v) for v in a.ylim.split(","))
    else:
        pad = 0.04 * (max(ally) - min(ally) or 1)
        ytop, ybot = max(ally) + pad, min(ally) - pad

    fig, axes = plt.subplots(1, len(panels), figsize=(a.width, a.height),
                             sharex=True, sharey=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (lab, d) in zip(axes, panels):
        rest = [c for c in sorted(d) if c not in hl]
        for c in rest:
            xs = [p[0] for p in d[c]]
            ys = [p[1] for p in d[c]]
            ax.plot(xs, ys, linestyle="none", marker="o", markersize=2.6,
                    markerfacecolor=GREY, markeredgecolor=SURFACE,
                    markeredgewidth=0.3, alpha=0.75, zorder=2)
        for i, c in enumerate(hl):
            if c not in d:
                continue
            xs = [p[0] for p in d[c]]
            ys = [p[1] for p in d[c]]
            ax.plot(xs, ys, linestyle="none", marker="o", markersize=4.2,
                    markerfacecolor=HILITE[i], markeredgecolor=SURFACE,
                    markeredgewidth=0.7, zorder=3 + i)
        # the count goes inside the axes: as a title it runs into the next
        # panel's title as soon as the labels are of any length
        ax.set_title(lab, fontsize=8, color=INK, pad=5)
        ax.annotate(f"{sum(len(v) for v in d.values())} poses, "
                    f"{len(d)} compounds", xy=(0.98, 0.02),
                    xycoords="axes fraction", ha="right", va="bottom",
                    fontsize=7, color=MUTED)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ybot, ytop)          # the less negative value at the top
        ax.xaxis.set_major_locator(MultipleLocator(a.xmajor))
        ax.xaxis.set_minor_locator(MultipleLocator(a.xminor))
        ax.yaxis.set_major_locator(MultipleLocator(a.ymajor))
        ax.yaxis.set_minor_locator(MultipleLocator(a.yminor))
        ax.tick_params(colors=MUTED, labelcolor=INK)
        ax.set_xlabel(a.xlabel, color=INK)
    axes[0].set_ylabel(a.ylabel, color=INK)

    handles = [Line2D([], [], linestyle="none", marker="o", markersize=4.2,
                      markerfacecolor=HILITE[i], markeredgecolor=SURFACE,
                      markeredgewidth=0.7, label=c)
               for i, c in enumerate(hl)]
    handles.append(Line2D([], [], linestyle="none", marker="o",
                          markersize=2.6, markerfacecolor=GREY,
                          markeredgecolor=SURFACE, markeredgewidth=0.3,
                          label="other compounds"))
    fig.tight_layout()
    # the legend sits below the axis label, not above the data
    axes_h = fig.subplotpars.top - fig.subplotpars.bottom
    fig.legend(handles=handles, loc="lower left", frameon=False,
               ncol=len(handles), fontsize=8, handletextpad=0.4,
               columnspacing=1.4, labelcolor=INK,
               bbox_to_anchor=(fig.subplotpars.left,
                               -0.05 / max(axes_h, 0.6)))
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=a.dpi, bbox_inches="tight",
                    facecolor=SURFACE)
    print(f"[out] {a.out}.png, {a.out}.pdf   ({a.dpi} dpi)")
    for lab, d in panels:
        xs = [x for v in d.values() for x, _, _ in v]
        within = [max(x for x, _, _ in v) - min(x for x, _, _ in v)
                  for v in d.values() if len(v) > 1]
        best = {c: max(x for x, _, _ in v) for c, v in d.items()}
        print(f"  {lab}: {a.x} spans {min(xs):.3f} to {max(xs):.3f}; "
              f"between compounds on their best pose "
              f"{max(best.values()) - min(best.values()):.3f}; "
              f"widest within one compound "
              f"{max(within) if within else float('nan'):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

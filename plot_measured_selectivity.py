#!/usr/bin/env python3
"""
Selectivity across a paired compound set: which compounds separate the two
subunits, and by how much.

Panel A ranks every compound by the difference column.
Panel B shows the value at each subunit for the most separated compounds in
either direction, so the ranking can be read against the underlying numbers.

    python plot_measured_selectivity.py --paired chembl_ck2_paired.csv \
        --docked-only dock_paired/poses --top 12 --out figures/Figure_selectivity.png

The columns are arguments, so the same script serves any paired quantity:

    python plot_measured_selectivity.py --paired comparison_uc.csv \
        --val1-col n_unique_a1 --val2-col n_unique_a2 \
        --diff-col unique_contact_diff --noise 1 \
        --label1 "CK2a" --label2 "CK2a-prime" \
        --ylabel-a "Isoform-unique contacted residues (CK2a' - CK2a)" \
        --ylabel-b "Unique contacted residues"
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("needs matplotlib")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 600,
                     "savefig.dpi": 600})
# diverging: two poles and a neutral midpoint. The grey carries no hue by
# design, which is what a midpoint should do; it is never a third category.
C_A1, C_A2, GREY, HLC = "#2e5eaa", "#d1495b", "#b9b9b9", "#1b7a4b"


def ordinal(n):
    """1st, 2nd, 3rd, 63rd. The record is read by people."""
    n = int(round(n))
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def fmt(series):
    """Integer counts are written as integers. A contacted-residue count
    printed as 3.00 invites it to be read as a continuous measurement."""
    v = pd.to_numeric(series, errors="coerce").dropna()
    return "{:.0f}" if (v == v.round()).all() else "{:.2f}"


def emptiest_corner(xs, ys, ax):
    """Axes-fraction anchor for a text block, in whichever corner holds fewest
    points. A fixed position is over the data as soon as the data moves."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    fx = (np.asarray(xs, float) - xlim[0]) / ((xlim[1] - xlim[0]) or 1.0)
    fy = (np.asarray(ys, float) - ylim[0]) / ((ylim[1] - ylim[0]) or 1.0)
    best, best_n = None, None
    for hx, ha in ((0.03, "left"), (0.97, "right")):
        for hy, va in ((0.95, "top"), (0.05, "bottom")):
            in_x = (fx < 0.45) if ha == "left" else (fx > 0.55)
            in_y = (fy > 0.55) if va == "top" else (fy < 0.45)
            n = int((in_x & in_y).sum())
            if best_n is None or n < best_n:
                best, best_n = (hx, hy, ha, va), n
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paired", default="chembl_ck2_paired.csv")
    p.add_argument("--docked-only", default=None,
                   help="pose directory; restricts to compounds actually docked")
    p.add_argument("--top", type=int, default=12,
                   help="compounds labelled per direction in panel B")
    p.add_argument("--noise", type=float, default=0.3,
                   help="half-width of the band treated as indistinguishable")
    p.add_argument("--center", default="0",
                   help="where the band sits: a number, or 'median'/'mean' to "
                        "take it from the data. Use the data when the two "
                        "sides carry a systematic offset, so that the counts "
                        "report compounds that stand out from the set rather "
                        "than compounds that cleared the offset")
    p.add_argument("--val1-col", default="p_value_a1")
    p.add_argument("--val2-col", default="p_value_a2")
    p.add_argument("--diff-col", default="selectivity_log")
    p.add_argument("--label1", default="CK2α",
                   help="name for the first subunit in the printed record")
    p.add_argument("--label2", default="CK2α′")
    p.add_argument("--ylabel-a", default="log(pCK2α′ − pCK2α)")
    p.add_argument("--ylabel-b", default="pActivity")
    p.add_argument("--noise-label", default="assay variation")
    p.add_argument("--highlight", default=None,
                   help="compound id to mark in both panels and force into "
                        "panel B, whatever its rank")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--out", default="Figure_selectivity.png")
    a = p.parse_args()

    d = pd.read_csv(a.paired)
    idc = "chembl_id" if "chembl_id" in d.columns else d.columns[0]
    lab1 = a.label1 or a.val1_col
    lab2 = a.label2 or a.val2_col
    log = [f"[in] {len(d)} paired compounds from {a.paired}"]

    if a.docked_only and os.path.isdir(a.docked_only):
        got = {os.path.basename(f).split("__")[0]
               for f in glob.glob(os.path.join(a.docked_only, "*.sdf"))}
        before = len(d)
        d = d[d[idc].astype(str).isin(got)]
        log.append(f"[in] restricted to {len(d)} of {before} compounds with poses")

    for col in (a.diff_col,):
        if col not in d.columns:
            sys.exit(f"column '{col}' not in {a.paired}; "
                     f"available: {', '.join(d.columns[:12])}")

    # the identifier is the tie-break, so the ranking is the same on every run.
    # sorting on the value alone leaves compounds that share a value in
    # whatever order the file happened to hold them, which decides who appears
    # in panel B when the values are counts and ties are common
    d = (d.dropna(subset=[a.diff_col])
           .sort_values([a.diff_col, idc], kind="mergesort")
           .reset_index(drop=True))
    v = d[a.diff_col].values
    f_diff = fmt(d[a.diff_col])

    if a.center == "median":
        c = float(np.median(v))
    elif a.center == "mean":
        c = float(np.mean(v))
    else:
        c = float(a.center)

    n_a1 = int((v < c - a.noise).sum())
    n_a2 = int((v > c + a.noise).sum())
    n_flat = int((np.abs(v - c) <= a.noise).sum())
    log.append(f"     range {v.min():+.2f} to {v.max():+.2f}, "
               f"median {np.median(v):+.2f}, mean {np.mean(v):+.2f}")
    src = ("the data, " + a.center) if a.center in ("median", "mean") else "as given"
    log.append(f"     band centred on {c:+.3f} ({src}), half-width {a.noise:g}")
    log.append(f"     {lab1}-preferring  (< {c - a.noise:+.3f}): {n_a1}")
    log.append(f"     within the band                 : {n_flat}")
    log.append(f"     {lab2}-preferring (> {c + a.noise:+.3f}): {n_a2}")
    # A set centred well away from zero carries a constant offset between the
    # two sides, and counting against zero then measures that constant: a
    # property of how the two receptors were prepared, not of the ligands.
    #
    # Requiring every value to fall on one side is too strict to catch it. A
    # set can straddle zero slightly and still be plainly offset, which is
    # what a lopsided count shows: one side empty while the other is not, or a
    # median sitting an appreciable part of the band away from zero.
    med = float(np.median(v))
    lopsided = (n_a1 == 0) != (n_a2 == 0)
    off_centre = abs(med) > 0.5 * a.noise
    if abs(c) < 1e-9 and (lopsided or off_centre):
        log.append(f"     [note] this set is centred on {med:+.3f}, not zero "
                   f"({abs(med)/a.noise:.2f} x the band half-width), and the "
                   f"counts against zero are {n_a1} against {n_a2}. The two "
                   f"sides differ by that constant before any compound is "
                   f"considered, so counting against zero measures the "
                   f"constant. --center median reports what stands out from "
                   f"the set instead")

    has_aff = {a.val1_col, a.val2_col} <= set(d.columns)
    fig, axes = plt.subplots(1, 2 if has_aff else 1,
                             figsize=(11.0 if has_aff else 5.2, 5.4),
                             gridspec_kw={"width_ratios": [1, 1.15]}
                             if has_aff else None)
    axA = axes[0] if has_aff else axes

    cols = [C_A1 if x < c - a.noise else C_A2 if x > c + a.noise else GREY
            for x in v]
    axA.bar(np.arange(len(v)), v, color=cols, width=1.0, linewidth=0)
    axA.axhline(0, color="black", lw=0.9)
    axA.axhspan(c - a.noise, c + a.noise, color="#eeeeee", zorder=0)
    if abs(c) > 1e-9:
        axA.axhline(c, color="#777777", lw=0.9, ls="--")
        axA.text(0.99, c, f" centre {c:+.2f} ", transform=axA.get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=6.5, color="#777777")
    axA.set_xlabel("Compounds")
    axA.set_ylabel(a.ylabel_a)
    axA.set_xlim(-1, len(v))
    axA.text(0.02, 0.06, f"{n_a1} favour {lab1}", transform=axA.transAxes,
             fontsize=8, color=C_A1, weight="bold")
    axA.text(0.98, 0.94, f"{n_a2} favour {lab2}", transform=axA.transAxes,
             ha="right", fontsize=8, color=C_A2, weight="bold")

    hi_i = None
    if a.highlight:
        hm = d[idc].astype(str).str.contains(a.highlight, regex=False)
        if hm.any():
            hi_i = int(np.where(hm.values)[0][0])
            hv = v[hi_i]
            axA.annotate(f"{a.highlight}  {hv:+.2f}", xy=(hi_i, hv),
                         xytext=(hi_i, hv + (v.max() - v.min()) * 0.22),
                         ha="center", fontsize=8, weight="bold", color=HLC,
                         arrowprops=dict(arrowstyle="-|>", color=HLC, lw=1.4))
            n_tied = int((v == hv).sum())
            log.append(f"     {a.highlight}: {hv:+.2f}, rank {hi_i+1} of "
                       f"{len(v)} ({ordinal(100*(v < hv).mean())} percentile)"
                       + (f"; {n_tied} compounds share this value"
                          if n_tied > 1 else ""))
        else:
            log.append(f"     [warn] {a.highlight} not found")

    kx, ky, kha, kva = emptiest_corner(np.arange(len(v)), v, axA)
    axA.text(kx, ky, f"{n_flat} within ±{a.noise:g}\n({a.noise_label})",
             transform=axA.transAxes, ha=kha, va=kva,
             fontsize=7.5, color="#777777")

    if has_aff:
        axB = axes[1]
        f1, f2 = fmt(d[a.val1_col]), fmt(d[a.val2_col])
        k = min(a.top, len(d) // 2)
        parts = [d.head(k), d.tail(k)]
        if a.highlight:
            extra = d[d[idc].astype(str).str.contains(a.highlight, regex=False)]
            if len(extra):
                parts.append(extra)
        sub = (pd.concat(parts).drop_duplicates(subset=[idc])
                 .sort_values([a.diff_col, idc], ascending=[False, True],
                              kind="mergesort")
                 .reset_index(drop=True))

        # with counts, the value at the cut is usually shared by more
        # compounds than the cut admits. Which of them appear is then a
        # property of the sort, not of the data, so it is stated rather than
        # left for the reader to assume the panel holds every extreme case
        lo_cut, hi_cut = d.head(k)[a.diff_col].max(), d.tail(k)[a.diff_col].min()
        for name, cut in ((lab1, lo_cut), (lab2, hi_cut)):
            n_at = int((d[a.diff_col] == cut).sum())
            n_in = int((sub[a.diff_col] == cut).sum())
            if n_at > n_in:
                log.append(f"     [note] {n_at} compounds sit at "
                           f"{f_diff.format(cut)} on the {name} side; "
                           f"{n_in} fit in panel B. The panel shows the "
                           f"extremes, not every compound at the cut")

        x = np.arange(len(sub))
        w = 0.38
        axB.bar(x - w/2, sub[a.val1_col], width=w, color=C_A1,
                label=lab1, edgecolor="white", linewidth=0.4)
        axB.bar(x + w/2, sub[a.val2_col], width=w, color=C_A2,
                label=lab2, edgecolor="white", linewidth=0.4)
        axB.set_xticks(x)
        axB.set_xticklabels(sub[idc].astype(str), rotation=90, fontsize=6)
        if a.highlight:
            for t in axB.get_xticklabels():
                if a.highlight in t.get_text():
                    t.set_color(HLC); t.set_fontweight("bold"); t.set_fontsize(7)

        lo = min(sub[a.val1_col].min(), sub[a.val2_col].min())
        hi_ = max(sub[a.val1_col].max(), sub[a.val2_col].max())
        # a bar means length measured from zero. Cutting the axis above zero
        # breaks that: two bars whose values differ by a tenth can be drawn
        # differing by half. Counts start at zero and keep the reading honest;
        # only a scale that cannot reach zero, such as pActivity, is cut, and
        # then the cut is stated on the axis
        if lo >= 0 and hi_ <= 60 and f1 == "{:.0f}":
            axB.set_ylim(0, hi_ * 1.12)
        else:
            axB.set_ylim(max(0, lo - 0.7), hi_ + 0.4)
            axB.text(0.0, 1.01, "axis does not start at zero",
                     transform=axB.transAxes, fontsize=6.5, color="#777777")
        axB.set_ylabel(a.ylabel_b)
        axB.set_xlim(-0.8, len(sub) - 0.2)
        axB.legend(frameon=False, fontsize=9, loc="upper center",
                   bbox_to_anchor=(0.5, -0.32), ncol=2)

        log.append(f"\n  panel B, ordered most {lab2}-favourable first:")
        for _, r in sub.iterrows():
            log.append(f"    {str(r[idc]):16s} {r[a.diff_col]:+.2f}  "
                       f"({lab1} {f1.format(r[a.val1_col])}, "
                       f"{lab2} {f2.format(r[a.val2_col])})")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("\n".join(log))
    with open(os.path.splitext(a.out)[0] + "_values.txt", "w") as f:
        f.write("\n".join(log) + "\n")
    print(f"\n[out] {a.out}")


if __name__ == "__main__":
    sys.exit(main())

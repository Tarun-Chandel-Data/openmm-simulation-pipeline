#!/usr/bin/env python3
"""
Plot what the pipeline output contains. Every quantity is computed at run time
from the files given: scaffold membership, linker length, group means,
correlations, distances, RMSDs and separations. No expected value, direction or
conclusion is written into this script, and no test is one-sided, so a result
that runs opposite to what was anticipated will be drawn and reported as such.

Every number that reaches a figure is also printed to stdout and written to
figure_values.txt, so a figure can be checked against the table it came from.

Start by seeing what is reachable and what is in it:

    python make_manuscript_figures.py --inventory \
        --root ~/docking_files/TEST/new --root ~/sim/small_molecule/cx

That prints each table with its column names and each directory of conformer
PDBs, and stops. Nothing is plotted and nothing is computed.

Then let it locate the inputs itself and draw whatever it finds:

    python make_manuscript_figures.py \
        --root ~/docking_files/TEST/new --root ~/sim/small_molecule/cx \
        --label-a1 "CK2a" --label-a2 "CK2a-prime" \
        --outdir figures

Any input can be named explicitly instead, which overrides discovery:

    --paired FILE --dock FILE --rescore FILE
    --anchor-a1 FILE --anchor-a2 FILE
    --ens-a1 DIR --ens-a2 DIR --scan FILE

Inputs are optional individually. A figure is produced only if its inputs are
present, and the script says what it skipped and why. What each figure plots:

    Figure 1  selectivity against linker length, within the scaffold group that
              the data itself selects (widest range, more than one length)
    Figure 2  Spearman rho of every matched a1/a2 docking column against
              measured selectivity, and what ranking on each would have picked
    Figure 3  predicted difference against measured, for the Figure 1 members
    Figure 4  affinity of one fixed pose across scoring models
    Figure 5  anchor separation over each trajectory: time course, distribution,
              and fraction of frames within a given distance
    Figure 6  conformer RMSD within and between the two ensembles, and the
              variance carried by the leading components
    Figure 7  per-compound closest approach to one subunit's atom against the
              other's, counted in both directions

The subunit labels are only axis text: --label-a1 and --label-a2 set them and
nothing else depends on them.
"""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:
    sys.exit("needs matplotlib:  pip install matplotlib")
try:
    from scipy.stats import spearmanr, mannwhitneyu, binomtest
except ImportError:
    sys.exit("needs scipy:  pip install scipy")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300,
})
C_A1, C_A2, GREY, HL = "#2e5eaa", "#d1495b", "#6c6c6c", "#1b7a4b"
HB_CUTOFF = 3.5          # heavy-atom donor-acceptor distance, angstrom


# ------------------------------------------------------------------ helpers
def need_rdkit():
    """Imported on demand so the dynamics figures work without rdkit."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    return Chem, MurckoScaffold


def read_table(path, sheet=None):
    """csv, tsv or excel, decided by extension rather than by argument."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls", ".xlsm"):
        try:
            return pd.read_excel(path, sheet) if sheet else pd.read_excel(path)
        except ValueError:
            return pd.read_excel(path)          # named sheet absent, take first
    return pd.read_csv(path, sep=None, engine="python")


def numeric_cols(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def pick_col(df, *patterns, exclude=()):
    """First numeric column whose name matches every pattern and no exclusion.
    Returns None rather than guessing when nothing matches."""
    for c in numeric_cols(df):
        low = c.lower()
        if all(re.search(p, low) for p in patterns) \
                and not any(re.search(x, low) for x in exclude):
            return c
    return None


def matched_a1a2(df, *patterns, exclude=()):
    """A matched pair of numeric columns describing the same quantity for the
    two subunits, e.g. (a1_dmin, a2_dmin). Returns (col_a1, col_a2) or None.

    The pairing is by name: a column carrying an a1 token has a counterpart
    with the same name and an a2 token. Nothing about which is which is
    assumed beyond that token."""
    cols = list(df.columns)
    for c in numeric_cols(df):
        low = c.lower()
        if not re.search(r"(^|[^a-z0-9])a1([^a-z0-9]|$)", low):
            continue
        if patterns and not all(re.search(p, low) for p in patterns):
            continue
        if any(re.search(x, low) for x in exclude):
            continue
        for cand in (c.replace("a1", "a2"), c.replace("A1", "A2")):
            if cand != c and cand in cols \
                    and pd.api.types.is_numeric_dtype(df[cand]):
                return c, cand
    return None


def pick_atom_col(df, residue, atom):
    """The column carrying the distance to one named atom of one named residue,
    e.g. ILE117 CD1. Requires both names in the column, so a residue-wide
    'closest any atom' column such as ILE117_mindist is never substituted for
    the specific atom the comparison is about."""
    res, at = residue.lower(), atom.lower()
    for c in numeric_cols(df):
        low = c.lower()
        if res in low and at in low and re.search(r"dist|approach", low):
            return c
    return None


def series_of(df, *patterns, exclude=()):
    c = pick_col(df, *patterns, exclude=exclude)
    return (c, df[c].dropna().to_numpy(float)) if c else (None, None)


def distance_column(df):
    """The distance a distance-vs-time table is about: a column named for a
    distance, else the only numeric column that is not a time or frame index."""
    c = pick_col(df, r"dist|sep|anchor|d_|_d$")
    if c:
        return c
    cands = [c for c in numeric_cols(df)
             if not re.search(r"time|frame|step|index|ns|ps", c.lower())]
    return cands[0] if len(cands) == 1 else (cands[0] if cands else None)


def time_column(df, n):
    """Values and an axis label for whatever the table uses as its abscissa:
    a named time, else a frame index, else the row number."""
    c = pick_col(df, r"time|_ns$|^ns")
    if c:
        unit = " (ns)" if re.search(r"ns", c.lower()) else ""
        return df[c].to_numpy(float), f"Simulation time{unit}"
    c = pick_col(df, r"frame|step|index")
    if c:
        return df[c].to_numpy(float), "Frame"
    return np.arange(n, dtype=float), "Frame"


# ---------------------------------------------------------------- structure
def linker_length(smiles):
    """Number of non-aromatic carbons between the exocyclic NH and the pendant
    aromatic ring. Derived from the graph, not from a lookup table.

    Returns None where the motif is absent, so such compounds are excluded
    rather than silently assigned a value."""
    Chem, _ = need_rdkit()
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    ri = m.GetRingInfo()
    best = None
    for a in m.GetAtoms():
        if a.GetSymbol() != "N" or a.IsInRing():
            continue
        if a.GetTotalNumHs() < 1:
            continue
        nbrs = a.GetNeighbors()
        # the N must bridge a fused ring system and a pendant aromatic ring
        arom_nbrs = [n for n in nbrs if n.GetIsAromatic()]
        chain_nbrs = [n for n in nbrs if not n.GetIsAromatic() and n.GetSymbol() == "C"]
        # case 1: N bonded directly to two aromatic systems -> zero-length linker
        if len(arom_nbrs) >= 2:
            sys_a = set(next(iter([r for r in ri.AtomRings()
                                   if arom_nbrs[0].GetIdx() in r]), ()))
            sys_b = set(next(iter([r for r in ri.AtomRings()
                                   if arom_nbrs[1].GetIdx() in r]), ()))
            if sys_a and sys_b and not (sys_a & sys_b):
                best = 0 if best is None else min(best, 0)
                continue
        # case 2: N bonded to one aromatic system and to a carbon chain that
        # terminates in a different aromatic ring
        if arom_nbrs and chain_nbrs:
            for start in chain_nbrs:
                n_c, cur, prev = 0, start, a
                while cur is not None and not cur.GetIsAromatic() and n_c < 8:
                    n_c += 1
                    nxt = [x for x in cur.GetNeighbors()
                           if x.GetIdx() != prev.GetIdx() and x.GetAtomicNum() > 1]
                    if len(nxt) != 1:
                        cur = None
                        break
                    prev, cur = cur, nxt[0]
                if cur is not None and cur.GetIsAromatic():
                    best = n_c if best is None else min(best, n_c)
    return best


def core_scaffold(smiles):
    """Group key built from the ring systems alone, ignoring everything that
    connects them. Compounds differing only by linker length therefore land in
    the same group, which a Murcko scaffold does not guarantee: the linker atoms
    are part of the Murcko scaffold when they bridge two rings."""
    Chem, _ = need_rdkit()
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    ri = m.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    if not rings:
        return None
    # merge fused rings into ring systems
    systems = []
    for r in rings:
        hit = [s for s in systems if s & r]
        if hit:
            merged = set(r)
            for h in hit:
                merged |= h
                systems.remove(h)
            systems.append(merged)
        else:
            systems.append(set(r))
    frags = []
    for sysm in systems:
        try:
            frags.append(Chem.MolFragmentToSmiles(m, atomsToUse=sorted(sysm),
                                                  canonical=True))
        except Exception:
            pass
    return ".".join(sorted(frags)) if frags else None


# ---------------------------------------------------------------- figure 1
def figure_scaffold_sar(d, outdir, min_members, log, lab1="a1", lab2="a2"):
    d = d.copy()
    d["core"] = d.smiles.apply(core_scaffold)
    d["linker"] = d.smiles.apply(linker_length)

    groups = (d.dropna(subset=["core"]).groupby("core")
              .filter(lambda g: len(g) >= min_members))
    if groups.empty:
        log.append(f"figure 1 skipped: no scaffold group with >= {min_members} members")
        return None

    # choose the group by evidence, not by expectation: the one whose members
    # span the widest selectivity range AND contain more than one linker length
    best, best_span = None, -1
    for core, g in groups.groupby("core"):
        gl = g.dropna(subset=["linker"])
        if gl.linker.nunique() < 2:
            continue
        span = g.selectivity_log.max() - g.selectivity_log.min()
        if span > best_span:
            best, best_span = core, span
    if best is None:
        log.append("figure 1 skipped: no group varies in linker length")
        return None

    g = d[d.core == best].dropna(subset=["linker"]).copy()
    g["linker"] = g.linker.astype(int)
    log.append("\n=== FIGURE 1 : scaffold selected on evidence ===")
    log.append(f"core            {best}")
    log.append(f"members         {len(g)} of {len(d)} paired compounds")
    log.append(f"selectivity     {g.selectivity_log.min():+.2f} to {g.selectivity_log.max():+.2f}")
    log.append(f"linker lengths  {[int(v) for v in sorted(g.linker.unique())]}")
    for n, gg in g.groupby("linker"):
        log.append(f"  {n} CH2 : n={len(gg)}  mean {gg.selectivity_log.mean():+.3f}  "
                   f"range {gg.selectivity_log.min():+.2f} to {gg.selectivity_log.max():+.2f}")

    # split at the largest gap between adjacent linker-length group means
    means = g.groupby("linker").selectivity_log.mean().sort_index()
    if len(means) < 2:
        log.append("figure 1 skipped: only one linker class after filtering")
        return None
    gaps = means.diff().dropna()
    cut = gaps.idxmax()
    lo = g[g.linker < cut]; hi = g[g.linker >= cut]
    log.append(f"split placed at {cut} CH2 (largest gap between class means)")
    log.append(f"  below : n={len(lo)}  mean {lo.selectivity_log.mean():+.3f}")
    log.append(f"  at/above: n={len(hi)}  mean {hi.selectivity_log.mean():+.3f}")

    sep = hi.selectivity_log.min() - lo.selectivity_log.max()
    log.append(f"  gap between groups: {sep:+.3f} log units "
               f"({'no overlap' if sep > 0 else 'groups overlap'})")
    if len(lo) and len(hi):
        # two-sided: the direction of the shift is read off the data afterwards
        # rather than assumed by the test
        u, p = mannwhitneyu(hi.selectivity_log, lo.selectivity_log,
                            alternative="two-sided")
        direction = ("higher" if hi.selectivity_log.median() > lo.selectivity_log.median()
                     else "lower" if hi.selectivity_log.median() < lo.selectivity_log.median()
                     else "equal")
        log.append(f"  Mann-Whitney U = {u:.0f} of {len(lo)*len(hi)}, "
                   f"two-sided p = {p:.4f}")
        log.append(f"  the at/above group sits {direction} than the below group")

    has_aff = {"p_value_a1", "p_value_a2"} <= set(g.columns)
    fig, axes = plt.subplots(1, 2 if has_aff else 1,
                             figsize=(7.2 if has_aff else 3.6, 3.3))
    axA = axes[0] if has_aff else axes

    rng = np.random.default_rng(0)
    for n, gg in g.groupby("linker"):
        col = HL if n >= cut else GREY
        x = np.full(len(gg), n) + rng.uniform(-0.11, 0.11, len(gg))
        axA.scatter(x, gg.selectivity_log, s=34, color=col,
                    edgecolor="white", linewidth=0.6, zorder=3)
        axA.hlines(gg.selectivity_log.mean(), n-0.26, n+0.26, color=col, lw=2.2, zorder=4)
    if sep > 0:
        axA.axhspan(lo.selectivity_log.max(), hi.selectivity_log.min(),
                    color="#ffd166", alpha=0.5, zorder=0)
    axA.axhline(0, color="black", lw=0.7, ls=":")
    axA.set_xticks(sorted(g.linker.unique()))
    axA.set_xlabel("Methylene units in linker")
    axA.set_ylabel(f"Measured selectivity, log({lab2} − {lab1})")
    axA.set_title("A" if has_aff else "", loc="left", fontsize=10, weight="bold")
    # headroom first, so the orientation labels sit clear of the points
    _lo, _hi = axA.get_ylim()
    _pad = 0.13 * (_hi - _lo)
    axA.set_ylim(_lo - _pad, _hi + _pad)
    axA.text(0.99, 0.985, f"{lab2}-preferring", transform=axA.transAxes,
             ha="right", va="top", fontsize=7.5, color=HL,
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.2))
    axA.text(0.99, 0.015, f"{lab1}-preferring", transform=axA.transAxes,
             ha="right", va="bottom", fontsize=7.5, color=GREY,
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.2))

    if has_aff:
        axB = axes[1]
        for i, (sub, lab) in enumerate(((lo, f"< {cut} CH₂"), (hi, f"≥ {cut} CH₂"))):
            for _, r in sub.iterrows():
                axB.plot([i-0.16, i+0.16], [r.p_value_a1, r.p_value_a2],
                         color="#c9c9c9", lw=0.8, zorder=1)
                axB.scatter([i-0.16], [r.p_value_a1], s=24, color=C_A1,
                            edgecolor="white", linewidth=0.5, zorder=3)
                axB.scatter([i+0.16], [r.p_value_a2], s=24, color=C_A2,
                            edgecolor="white", linewidth=0.5, zorder=3)
            m1, m2 = sub.p_value_a1.mean(), sub.p_value_a2.mean()
            axB.hlines(m1, i-0.30, i-0.02, color=C_A1, lw=2.4, zorder=4)
            axB.hlines(m2, i+0.02, i+0.30, color=C_A2, lw=2.4, zorder=4)
            # centred above each mean line rather than beside it: side-by-side
            # labels from adjacent groups collide whenever the two means are
            # close, which is exactly the case worth reading
            _yr = (sub[["p_value_a1", "p_value_a2"]].to_numpy().max()
                   - sub[["p_value_a1", "p_value_a2"]].to_numpy().min()) or 1.0
            for _x, _m, _c in ((i-0.16, m1, C_A1), (i+0.16, m2, C_A2)):
                axB.text(_x, _m + 0.05*_yr, f"{_m:.2f}", ha="center",
                         va="bottom", fontsize=8, color=_c, weight="bold",
                         bbox=dict(facecolor="white", alpha=0.85,
                                   edgecolor="none", pad=1.0), zorder=6)
            log.append(f"  {lab}: mean p{lab1} {m1:.2f}, mean p{lab2} {m2:.2f}")
        d1 = hi.p_value_a1.mean() - lo.p_value_a1.mean()
        d2 = hi.p_value_a2.mean() - lo.p_value_a2.mean()
        log.append(f"  change across the split: {lab1} {d1:+.2f}, {lab2} {d2:+.2f}")
        axB.set_xticks([0, 1])
        axB.set_xticklabels([f"< {cut} CH₂  (n = {len(lo)})",
                             f"≥ {cut} CH₂  (n = {len(hi)})"])
        axB.set_ylabel("pActivity")
        axB.set_xlim(-0.65, 1.65)
        axB.legend(handles=[Line2D([], [], marker="o", ls="", color=C_A1, label=lab1),
                            Line2D([], [], marker="o", ls="", color=C_A2, label=lab2)],
                   loc="lower left", frameon=False, fontsize=8)
        axB.set_title("B", loc="left", fontsize=10, weight="bold")

    fig.tight_layout()
    p = os.path.join(outdir, "Figure1_scaffold_SAR.png")
    fig.savefig(p); plt.close(fig)
    return p, g, cut


# ---------------------------------------------------------------- figure 2
def benchmark_frame(paired, dock, bench, log):
    """The table figure 2 correlates, and the name of its measured column.

    Two shapes are accepted. A self-contained table already carries the
    measured value and every criterion side by side, and is used as it stands.
    Otherwise a paired activity table is joined to a docking table, and the
    criteria are restricted to columns from the docking side, because the
    paired table's own a1/a2 columns are the measured activities and
    correlating them against their own difference would be circular."""
    if bench is not None:
        mcol = next((c for c in numeric_cols(bench)
                     if re.search(r"^measured|selectivity", c.lower())
                     and not re.search(r"pred|calc", c.lower())), None)
        if mcol is None:
            log.append("figure 2 skipped: the benchmark table has no measured "
                       f"selectivity column. Numeric columns: "
                       f"{numeric_cols(bench)[:14]}")
            return None
        return bench, mcol, set(bench.columns), "self-contained table"

    if paired is None or dock is None:
        log.append("figure 2 skipped: need either a benchmark table, or both a "
                   "paired table and a docking table")
        return None
    j = dock.merge(paired, left_on="cpd_id", right_on="chembl_id")
    if j.empty:
        log.append("figure 2 skipped: docking table and paired table share no compounds")
        return None
    return j, "selectivity_log", set(dock.columns), "paired joined to docking"


def figure_benchmark(paired, dock, outdir, log, bench=None):
    """Correlate every numeric column pair that looks like an a1/a2 quantity
    with measured selectivity. Criteria are discovered from the columns
    present, not specified in advance."""
    got = benchmark_frame(paired, dock, bench, log)
    if got is None:
        return None
    j, mcol, allowed, how = got

    crits = {}
    for c in list(j.columns):
        if c not in allowed or not pd.api.types.is_numeric_dtype(j[c]):
            continue
        if not re.search(r"(^|[^a-z0-9])a1($|[^a-z0-9])", c.lower()):
            continue
        cand = None
        for alt in (c.replace("a1_isoform", "a2_proteinA"), c.replace("a1", "a2")):
            if alt != c and alt in j.columns \
                    and pd.api.types.is_numeric_dtype(j[alt]):
                cand = alt
                break
        if cand:
            label = (c.replace("a1_isoform_", "").replace("a1_", "")
                     .replace("_a1", "").replace("a1", "").strip("_") or c)
            crits[label] = (cand, c)          # (a2 col, a1 col)
    if not crits:
        log.append("figure 2 skipped: no matched a1/a2 numeric columns found")
        return None

    # a table may also carry the difference precomputed. Correlating both it
    # and the pair it came from would draw the same criterion twice, so the
    # precomputed one is named and left out.
    dup = [c for c in numeric_cols(j)
           if re.match(r"^(d_|delta_)", c.lower())
           or c.lower().endswith("_diff")]

    log.append(f"\n=== FIGURE 2 : benchmark, n = {len(j)} compounds ({how}) ===")
    log.append(f"measured column '{mcol}' spans {j[mcol].min():+.2f} to "
               f"{j[mcol].max():+.2f}")
    if dup:
        log.append(f"  precomputed difference column(s) not plotted separately: "
                   f"{', '.join(dup)}")
    rows = []
    for label, (ca2, ca1) in sorted(crits.items()):
        delta = j[ca2] - j[ca1]
        ok = delta.notna() & j[mcol].notna()
        if ok.sum() < 10:
            log.append(f"  {label:28s} skipped, only {ok.sum()} paired values")
            continue
        r, p = spearmanr(delta[ok], j[mcol][ok])
        # what a campaign acting on this ranking would have obtained
        k = min(20, ok.sum() // 3)
        sub = delta[ok]
        top = j.loc[sub.nlargest(k).index, mcol].mean()
        bot = j.loc[sub.nsmallest(k).index, mcol].mean()
        rows.append(dict(criterion=label, rho=r, p=p, n=int(ok.sum()),
                         top=top, bot=bot, sep=top - bot))
        log.append(f"  {label:28s} rho {r:+.3f}  p {p:.3f}  n {ok.sum():3d}  "
                   f"top{k} {top:+.2f} vs bot{k} {bot:+.2f}  sep {top-bot:+.2f}")
    if not rows:
        n_meas = int(j[mcol].notna().sum())
        log.append(f"figure 2 skipped: no criterion had enough paired values. "
                   f"Only {n_meas} of {len(j)} rows carry a value in '{mcol}'")
        if n_meas < 0.2 * len(j):
            log.append("  this table is mostly compounds with no measurement "
                       "(designed or enumerated rather than assayed). A "
                       "benchmark needs the measured set: drop --benchmark to "
                       "join the paired activity table to a docking table "
                       "instead")
        return None
    t = pd.DataFrame(rows).sort_values("rho")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 0.42*len(t)+2.2),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(t))
    ax1.barh(y, t.rho, color=[HL if v > 0 else C_A1 for v in t.rho],
             height=0.6, edgecolor="white")
    ax1.axvline(0, color="black", lw=0.9)
    ax1.set_yticks(y); ax1.set_yticklabels(t.criterion, fontsize=7.5)
    ax1.set_xlabel("Spearman ρ against measured selectivity")
    lim = max(0.35, np.abs(t.rho).max()*1.4)
    ax1.set_xlim(-lim, lim)
    for i, r in enumerate(t.itertuples()):
        ax1.text(r.rho + (0.02 if r.rho >= 0 else -0.02), i,
                 f"{r.rho:+.2f}", va="center",
                 ha="left" if r.rho >= 0 else "right", fontsize=7)
    ax1.set_title("A   Correlation with measurement", loc="left",
                  fontsize=9.5, weight="bold")

    ax2.barh(y-0.17, t.top, height=0.32, color=HL, label="top-ranked")
    ax2.barh(y+0.17, t.bot, height=0.32, color=GREY, label="bottom-ranked")
    ax2.axvline(0, color="black", lw=0.9)
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.set_xlabel("Mean measured selectivity of ranked subsets")
    ax2.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax2.set_title("B   Enrichment by ranking", loc="left",
                  fontsize=9.5, weight="bold")

    fig.tight_layout()
    p = os.path.join(outdir, "Figure2_benchmark.png")
    fig.savefig(p); plt.close(fig)
    return p, t


# ---------------------------------------------------------------- figure 3
def figure_matched_pairs(g, cut, dock, outdir, log):
    """Find compounds on the scaffold that differ only in linker length and share
    the pendant substitution, then compare measured with predicted."""
    if dock is None:
        log.append("figure 3 skipped: no docking table")
        return None
    j = g.merge(dock, left_on="chembl_id", right_on="cpd_id", how="inner")
    if j.empty:
        log.append("figure 3 skipped: scaffold members were not docked")
        return None
    a2c = next((c for c in j.columns if "a2" in c and "cnn" in c.lower()), None)
    a1c = next((c for c in j.columns if "a1" in c and "cnn" in c.lower()), None)
    if not (a2c and a1c):
        log.append("figure 3 skipped: no CNN affinity columns")
        return None
    j["pred"] = j[a2c] - j[a1c]

    log.append(f"\n=== FIGURE 3 : scaffold members with docking, n = {len(j)} ===")
    r, p = spearmanr(j.pred, j.selectivity_log)
    log.append(f"  Spearman(predicted, measured) = {r:+.3f}  p = {p:.3f}")
    log.append(f"  measured spans {j.selectivity_log.max()-j.selectivity_log.min():.2f} "
               f"log units; predicted spans {j.pred.max()-j.pred.min():.2f}")

    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    col = [HL if v >= cut else GREY for v in j.linker]
    ax.scatter(j.selectivity_log, j.pred, s=42, c=col,
               edgecolor="white", linewidth=0.6, zorder=3)
    ax.axhline(0, color="black", lw=0.7, ls=":")
    ax.axvline(0, color="black", lw=0.7, ls=":")
    ax.set_xlabel("Measured selectivity (log units)")
    ax.set_ylabel("Predicted difference (a2 − a1)")
    ax.set_title(f"ρ = {r:+.2f}  (p = {p:.2f}, n = {len(j)})",
                 loc="left", fontsize=9.5, weight="bold")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=GREY,
                              label=f"< {cut} CH₂"),
                       Line2D([], [], marker="o", ls="", color=HL,
                              label=f"≥ {cut} CH₂")],
              frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    p_ = os.path.join(outdir, "Figure3_matched_pairs.png")
    fig.savefig(p_); plt.close(fig)
    return p_


# ---------------------------------------------------------------- figure 4
def figure_rescore_spread(path, outdir, log, ref_margin=None):
    """Spread of affinity returned by several scoring models for one fixed pose,
    against the margin that separates the published reference compounds.

    The table is expected to be one row per model (or one row per model and
    pose); the spread is taken over whichever numeric affinity column is found
    and is computed, not asserted. The reference margin is plotted only if
    supplied, since it comes from the literature rather than from these files.
    """
    t = read_table(path)
    col, vals = series_of(t, r"affin|cnn|pk|score")
    if vals is None or len(vals) < 2:
        log.append(f"figure 4 skipped: no numeric affinity column in {path}")
        return None
    labels = None
    for c in t.columns:
        if not pd.api.types.is_numeric_dtype(t[c]) and t[c].nunique() == len(t):
            labels = t[c].astype(str).tolist()
            break
    if labels is None:
        labels = [f"model {i+1}" for i in range(len(vals))]

    spread = float(vals.max() - vals.min())
    log.append(f"\n=== FIGURE 4 : scoring reproducibility ({os.path.basename(path)}) ===")
    log.append(f"  affinity column   {col}")
    log.append(f"  n models          {len(vals)}")
    for lab, v in zip(labels, vals):
        log.append(f"    {lab:28s} {v:.3f}")
    log.append(f"  spread over models {spread:.3f} pK units "
               f"(sd {vals.std(ddof=1):.3f}, mean {vals.mean():.3f})")
    if ref_margin:
        log.append(f"  reference margin   {min(ref_margin):.2f} to {max(ref_margin):.2f} pK units")
        log.append(f"  ratio              {spread/max(ref_margin):.2f}x the largest "
                   f"reference margin")

    # points on a zoomed axis rather than bars from zero: the quantity of
    # interest is the spread between models, which bars anchored at zero make
    # invisible at this scale
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    x = np.arange(len(vals))
    pad = max(spread, (max(ref_margin) if ref_margin else 0)) * 0.9 + 0.05
    ax.set_ylim(vals.min() - pad, vals.max() + pad)
    ax.axhspan(vals.min(), vals.max(), color=C_A1, alpha=0.12, zorder=0)
    ax.scatter(x, vals, s=70, color=C_A1, edgecolor="white", linewidth=0.8, zorder=4)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.045*pad, f"{v:.2f}", ha="center", va="bottom",
                fontsize=8, color=C_A1, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=7.5)
    ax.set_xlim(-0.6, len(vals) - 0.05)

    # the two arrows share a clear column at the right and are named in a
    # corner key, so no label can collide with a point
    xa = len(vals) - 0.22
    ax.annotate("", xy=(xa, vals.max()), xytext=(xa, vals.min()),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.1))
    key = [(f"spread over models  {spread:.2f} pK", "black")]
    if ref_margin:
        m = max(ref_margin)
        base = vals.min() - 0.55*pad
        ax.axhspan(base, base + m, color="#ffd166", alpha=0.75, zorder=1)
        ax.annotate("", xy=(xa, base + m), xytext=(xa, base),
                    arrowprops=dict(arrowstyle="<->", color="#8a6d00", lw=1.1))
        key.append((f"reference margin  {min(ref_margin):.2f}–{m:.2f} pK",
                    "#8a6d00"))
    for i, (txt, c) in enumerate(key):
        ax.text(0.02, 0.97 - 0.075*i, txt, transform=ax.transAxes,
                fontsize=8, va="top", ha="left", color=c, weight="bold")
    ax.set_ylabel("Affinity of one fixed pose (pK units)")
    ax.set_title("Model-to-model spread vs. reference margin",
                 loc="left", fontsize=9, weight="bold")
    fig.tight_layout()
    p = os.path.join(outdir, "Figure4_scoring_reproducibility.png")
    fig.savefig(p); plt.close(fig)
    return p


def figure_affinity_dispersion(path, outdir, log, sheet=None,
                               lab1="a1", lab2="a2", ref_margin=None):
    """Per-compound dispersion of predicted affinity across the conformer
    ensemble, for each subunit.

    This is NOT model-to-model reproducibility. It is the spread a single
    scoring model returns over the conformers of one receptor, which is a
    different quantity and answers a different question: how much of the
    predicted signal is ensemble noise. It is plotted against the size of the
    difference the same table is being asked to resolve, so the two can be
    compared on one axis."""
    t = read_table(path, sheet)
    sd = matched_a1a2(t, r"_sd$|_std$|stdev|dispersion")
    mu = matched_a1a2(t, r"", exclude=(r"_sd$|_std$|stdev|n_|dmin|mw|qed",))
    if sd is None:
        log.append(f"figure 4 skipped: {path} has no matched a1/a2 dispersion "
                   f"columns. Numeric columns: {numeric_cols(t)[:14]}")
        return None
    s1, s2 = sd
    sub = t[[s1, s2]].dropna()
    if len(sub) < 3:
        log.append(f"figure 4 skipped: only {len(sub)} rows with both {s1} and {s2}")
        return None

    log.append(f"\n=== FIGURE 4 : affinity dispersion across the ensemble, "
               f"n = {len(sub)} compounds ===")
    log.append(f"  source {os.path.basename(path)}; columns {s1}, {s2}")
    log.append(f"  NOTE this is spread over conformers of one receptor, not "
               f"spread over scoring models")
    for c, lab in ((s1, lab1), (s2, lab2)):
        v = sub[c]
        log.append(f"  {lab:10s} mean {v.mean():.3f}  median {v.median():.3f}  "
                   f"min {v.min():.3f}  max {v.max():.3f}")
    # the difference the table is being asked to resolve, where it is present
    spread_ref = None
    if mu:
        m1, m2 = mu
        dd = (t[m2] - t[m1]).dropna()
        if len(dd):
            spread_ref = float(dd.abs().median())
            log.append(f"  median |{m2} − {m1}| = {spread_ref:.3f} "
                       f"(the difference being resolved)")
            log.append(f"  typical dispersion is "
                       f"{(sub[s1].median()+sub[s2].median())/2/spread_ref:.2f}x "
                       f"that difference")

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    bins = np.linspace(0, max(sub[s1].max(), sub[s2].max())*1.05, 30)
    for c, lab, col in ((s1, lab1, C_A1), (s2, lab2, C_A2)):
        ax.hist(sub[c], bins=bins, histtype="step", lw=1.6, color=col,
                label=f"{lab} (median {sub[c].median():.3f})")
        ax.axvline(sub[c].median(), color=col, lw=1.0, ls="--", alpha=0.7)
    if spread_ref:
        ax.axvline(spread_ref, color=HL, lw=1.6)
        # annotate on whichever side of the line has room
        right = spread_ref < (bins[-1] + bins[0]) / 2
        ax.text(spread_ref + (0.01 if right else -0.01) * (bins[-1] - bins[0]),
                ax.get_ylim()[1] * 0.55,
                f"median difference\nbeing resolved\n{spread_ref:.3f}",
                fontsize=7.5, color=HL, va="center",
                ha="left" if right else "right", weight="bold",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))
    ax.set_xlabel("Per-compound affinity dispersion across conformers (pK units)")
    ax.set_ylabel("Compounds")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Ensemble dispersion of predicted affinity",
                 loc="left", fontsize=9, weight="bold")
    fig.tight_layout()
    p = os.path.join(outdir, "Figure4_affinity_dispersion.png")
    fig.savefig(p); plt.close(fig)
    return p


# ---------------------------------------------------------------- figure 5
def figure_anchor_distance(p_a1, p_a2, outdir, log, xtal=None, cutoff=HB_CUTOFF,
                           lab1="a1", lab2="a2"):
    """Separation of the hydrogen-bonding anchors at the divergent hinge over the
    trajectories of each subunit: time course, distribution, and the fraction of
    frames within hydrogen-bonding range. Crystal values are drawn only if given
    on the command line, since they come from the deposited coordinates."""
    if not (p_a1 and p_a2):
        log.append("figure 5 skipped: need both --anchor-a1 and --anchor-a2")
        return None
    log.append("\n=== FIGURE 5 : hinge anchor separation in dynamics ===")
    out = {}
    for tag, path in (("a1", p_a1), ("a2", p_a2)):
        t = read_table(path)
        c = distance_column(t)
        if c is None:
            log.append(f"figure 5 skipped: no distance column in {path}")
            return None
        v = t[c].to_numpy(float)
        ok = np.isfinite(v)
        tv, tlab = time_column(t, len(v))
        out[tag] = dict(t=tv[ok], d=v[ok], col=c, xlabel=tlab,
                        src=os.path.basename(path))
    x_label = out["a1"]["xlabel"]

    # A time column whose whole span is a tiny number is not in the unit its
    # name claims. Plotting it puts an exponent on the axis and invites the
    # span to be misread, so fall back to the frame index and say why.
    span = max(o["t"].max() - o["t"].min() for o in out.values())
    if 0 < span < 1e-2:
        log.append(f"  warning: the time column spans only {span:.3g} over "
                   f"{len(out['a1']['t'])} frames. That is not nanoseconds; "
                   f"check the unit written by the analysis script. Plotting "
                   f"frame index instead")
        for o in out.values():
            o["t"] = np.arange(len(o["d"]), dtype=float)
        x_label = "Frame"
    else:
        for tag, o in out.items():
            log.append(f"  {tag} time spans {o['t'].min():.3g} to "
                       f"{o['t'].max():.3g} over {len(o['t'])} frames")
    for tag, lab in (("a1", lab1), ("a2", lab2)):
        o = out[tag]
        frac = float((o["d"] <= cutoff).mean())
        o["frac"] = frac
        log.append(f"  {lab:7s} [{o['src']}:{o['col']}] n={len(o['d'])}  "
                   f"mean {o['d'].mean():.2f} A  sd {o['d'].std(ddof=1):.2f}  "
                   f"median {np.median(o['d']):.2f}  "
                   f"within {cutoff:.1f} A in {100*frac:.0f}% of frames")
    log.append(f"  difference in means: "
               f"{out['a2']['d'].mean() - out['a1']['d'].mean():+.2f} A "
               f"({lab2} minus {lab1})")
    if xtal:
        log.append(f"  crystal separations supplied: {lab1} {xtal[0]:.2f} A, "
                   f"{lab2} {xtal[1]:.2f} A, difference "
                   f"{xtal[1]-xtal[0]:+.2f} A")

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(8.4, 2.9),
                                        gridspec_kw={"width_ratios": [1.5, 1, 0.8]})
    for tag, lab, col in (("a1", lab1, C_A1), ("a2", lab2, C_A2)):
        o = out[tag]
        axA.plot(o["t"], o["d"], color=col, lw=0.4, alpha=0.55)
        # running mean, window = 1% of the trajectory, so the trend is visible
        w = max(1, len(o["d"]) // 100)
        axA.plot(o["t"], pd.Series(o["d"]).rolling(w, center=True,
                                                   min_periods=1).mean(),
                 color=col, lw=1.4, label=lab)
        axB.hist(o["d"], bins=40, orientation="horizontal", color=col,
                 alpha=0.5, density=True)
        axB.axhline(o["d"].mean(), color=col, lw=1.4)
    axA.axhline(cutoff, color="black", lw=0.8, ls=":")
    axA.text(0.99, cutoff, f" {cutoff:.1f} \u00c5", transform=axA.get_yaxis_transform(),
             ha="left", va="bottom", fontsize=7)
    if xtal:
        for v, col in zip(xtal, (C_A1, C_A2)):
            axA.axhline(v, color=col, lw=0.9, ls="--", alpha=0.8)
    axA.set_xlabel(x_label)
    axA.set_ylabel("Anchor separation (Å)")
    axA.legend(frameon=False, fontsize=8, loc="upper right")
    axA.set_title("A", loc="left", fontsize=10, weight="bold")

    axB.axhline(cutoff, color="black", lw=0.8, ls=":")
    axB.set_xlabel("Density")
    axB.set_yticklabels([])
    axB.set_title("B", loc="left", fontsize=10, weight="bold")
    axB.set_ylim(axA.get_ylim())

    fr = [out["a1"]["frac"], out["a2"]["frac"]]
    axC.bar([0, 1], [100*f for f in fr], color=[C_A1, C_A2], width=0.6,
            edgecolor="white")
    for i, f in enumerate(fr):
        axC.text(i, 100*f + 1.5, f"{100*f:.0f}%", ha="center", fontsize=8,
                 weight="bold")
    axC.set_xticks([0, 1])
    axC.set_xticklabels([lab1, lab2])
    axC.set_ylabel(f"Frames within {cutoff:.1f} Å (%)")
    axC.set_ylim(0, max(100*max(fr) + 12, 20))
    axC.set_title("C", loc="left", fontsize=10, weight="bold")

    fig.tight_layout()
    p = os.path.join(outdir, "Figure5_hinge_anchor_geometry.png")
    fig.savefig(p); plt.close(fig)
    return p


# ---------------------------------------------------------------- figure 6
def ensemble_coords(directory, log):
    """Cartesian coordinates of the pocket atoms of every conformer in a
    directory of PDB files, restricted to the atom names common to all of them
    so that the RMSD is over a defined correspondence."""
    try:
        import mdtraj as md
    except ImportError:
        log.append("figure 6 needs mdtraj:  pip install mdtraj")
        return None, None, None
    files = sorted(glob.glob(os.path.join(directory, "*.pdb"))
                   + glob.glob(os.path.join(directory, "*.pdb.gz")))
    if len(files) < 2:
        log.append(f"figure 6 skipped: fewer than 2 conformers in {directory}")
        return None, None, None
    per = []
    for f in files:
        t = md.load(f)
        sel = t.topology.select("name CA")
        if len(sel) == 0:
            sel = t.topology.select("protein")
        keys = [(a.residue.resSeq, a.name)
                for a in (t.topology.atom(i) for i in sel)]
        per.append((keys, t.xyz[0][sel]))
    common = set(per[0][0])
    for keys, _ in per[1:]:
        common &= set(keys)
    if len(common) < 3:
        log.append(f"figure 6 skipped: conformers in {directory} share too few atoms")
        return None, None, None
    order = sorted(common)
    X = np.stack([np.stack([xyz[keys.index(k)] for k in order])
                  for keys, xyz in per])       # (n_conf, n_atom, 3), nanometres
    return X * 10.0, files, order              # angstrom


def _kabsch_rmsd(P, Q):
    """RMSD after optimal superposition, both (n_atom, 3) in angstrom."""
    P = P - P.mean(0); Q = Q - Q.mean(0)
    V, S, W = np.linalg.svd(P.T @ Q)
    if np.linalg.det(V @ W) < 0:
        V[:, -1] *= -1
    return float(np.sqrt((((P @ (V @ W)) - Q) ** 2).sum() / len(P)))


def _pairwise(X, Y=None):
    Y = X if Y is None else Y
    same = Y is X
    out = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            if same and j <= i:
                continue
            out[i, j] = _kabsch_rmsd(X[i], Y[j])
    return out + out.T if same else out


def figure_ensemble_overlap(dir_a1, dir_a2, outdir, log, lab1="a1", lab2="a2"):
    """Within-ensemble and between-ensemble conformer RMSD, plus the variance
    captured by the leading principal components of each ensemble. If the two
    distributions coincide, no weighting over conformers can separate the
    subunits, which is the claim the figure has to be able to refute."""
    if not (dir_a1 and dir_a2):
        log.append("figure 6 skipped: need both --ens-a1 and --ens-a2")
        return None
    log.append("\n=== FIGURE 6 : conformational ensemble overlap ===")
    X1, f1, k1 = ensemble_coords(dir_a1, log)
    X2, f2, k2 = ensemble_coords(dir_a2, log)
    if X1 is None or X2 is None:
        return None

    # The cross comparison needs the two ensembles aligned residue to residue.
    # Truncating to a common length instead would pair atom i of one subunit
    # with atom i of the other, which is not a correspondence at all: where the
    # two differ in length or numbering it compares unrelated positions and
    # returns a large RMSD that says nothing about the structures.
    #
    # The two subunits are numbered with an offset, so the offset is found from
    # the data: the shift that puts the most residues of one onto the other.
    r1 = [r for r, _ in k1]
    r2 = [r for r, _ in k2]
    s1, s2 = set(r1), set(r2)
    best_off, best_n = 0, -1
    for off in range(-30, 31):
        n_hit = len(s1 & {r + off for r in s2})
        if n_hit > best_n:
            best_off, best_n = off, n_hit
    shared = sorted(s1 & {r + off for r in s2 for off in (best_off,)})
    if len(shared) < 3:
        log.append(f"figure 6 skipped: the two ensembles share only "
                   f"{len(shared)} residue(s) at the best offset; no "
                   f"correspondence to compare over")
        return None
    i1 = [r1.index(r) for r in shared]
    i2 = [r2.index(r - best_off) for r in shared]
    X1c, X2c = X1[:, i1], X2[:, i2]
    log.append(f"  cross comparison over {len(shared)} residues matched at "
               f"offset {best_off:+d} "
               f"({X1.shape[1]} and {X2.shape[1]} residues available)")
    if len(shared) < 0.5 * min(X1.shape[1], X2.shape[1]):
        log.append(f"  warning: that is under half of either ensemble; check "
                   f"the residue numbering before reading the between-subunit "
                   f"distribution")

    w1 = _pairwise(X1); w2 = _pairwise(X2); cr = _pairwise(X1c, X2c)
    iu1 = np.triu_indices(len(X1), 1); iu2 = np.triu_indices(len(X2), 1)
    v1, v2, vc = w1[iu1], w2[iu2], cr.ravel()

    log.append(f"  {lab1} {len(X1)} conformers from {dir_a1}")
    log.append(f"  {lab2} {len(X2)} conformers from {dir_a2}")
    log.append(f"  within {lab1} : mean {v1.mean():.2f} A  "
               f"min {v1.min():.2f}  max {v1.max():.2f}")
    log.append(f"  within {lab2} : mean {v2.mean():.2f} A  "
               f"min {v2.min():.2f}  max {v2.max():.2f}")
    log.append(f"  between       : mean {vc.mean():.2f} A  "
               f"min {vc.min():.2f}  max {vc.max():.2f}")
    log.append(f"  between minus larger within-mean: "
               f"{vc.mean() - max(v1.mean(), v2.mean()):+.2f} A "
               f"({'ensembles separate' if vc.mean() > max(v1.mean(), v2.mean()) else 'ensembles overlap'})")

    # scree: variance of each ensemble along its own principal components
    scree = {}
    for tag, X in (("a1", X1), ("a2", X2)):
        F = X.reshape(len(X), -1)
        F = F - F.mean(0)
        s = np.linalg.svd(F, compute_uv=False) ** 2
        frac = s / s.sum()
        scree[tag] = frac
        k = min(10, len(frac))
        log.append(f"  PC1 of {tag}: {100*frac[0]:.1f}% of variance; "
                   f"first {k} PCs {100*frac[:k].sum():.1f}%")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    # bins span the data rather than starting at zero: an axis anchored at 0
    # would compress the three distributions together and flatter the claim
    lo_b = min(v1.min(), v2.min(), vc.min())
    hi_b = max(v1.max(), v2.max(), vc.max())
    pad = 0.05 * (hi_b - lo_b or 1.0)
    bins = np.linspace(lo_b - pad, hi_b + pad, 34)
    for v, lab, col in ((v1, f"within {lab1} (mean {v1.mean():.2f} Å)", C_A1),
                        (v2, f"within {lab2} (mean {v2.mean():.2f} Å)", C_A2),
                        (vc, f"between (mean {vc.mean():.2f} Å)", GREY)):
        axA.hist(v, bins=bins, density=True, histtype="step", lw=1.6,
                 color=col, label=lab)
    axA.set_xlabel("Conformer-to-conformer RMSD (Å)")
    axA.set_ylabel("Density")
    axA.legend(frameon=False, fontsize=7.5)
    axA.set_title("A   Between vs. within subunit", loc="left",
                  fontsize=9, weight="bold")

    k = min(10, len(scree["a1"]), len(scree["a2"]))
    x = np.arange(1, k+1)
    axB.bar(x-0.19, 100*scree["a1"][:k], width=0.36, color=C_A1, label=lab1)
    axB.bar(x+0.19, 100*scree["a2"][:k], width=0.36, color=C_A2, label=lab2)
    axB.set_xticks(x)
    axB.set_xlabel("Principal component")
    axB.set_ylabel("Variance explained (%)")
    axB.legend(frameon=False, fontsize=8)
    axB.set_title("B   Site motion, leading components", loc="left",
                  fontsize=9, weight="bold")

    fig.tight_layout()
    p = os.path.join(outdir, "Figure6_ensemble_overlap.png")
    fig.savefig(p); plt.close(fig)
    return p


# ---------------------------------------------------------------- figure 7
def figure_methyl_access(path, outdir, log, sheet=None, lab1="a1", lab2="a2",
                         res1="VAL116", atom1="CG1", res2="ILE117", atom2="CD1"):
    """Per-compound reach of ligand carbon to the isoform-specific methyl, one
    subunit against the other. Both directions are counted and the sign test is
    two-sided, so whichever way the data falls is what the figure reports."""
    t = read_table(path, sheet)
    # the two atoms are named, not inferred: a residue-wide 'closest any atom'
    # column is a different quantity and must not stand in for the methyl
    d2 = pick_atom_col(t, res2, atom2)
    d1 = pick_atom_col(t, res1, atom1)
    src = f"{res1}-{atom1} vs {res2}-{atom2}"
    if not (d2 and d1):
        # a table may record the same comparison already reduced to one column
        # per subunit (a1_dmin / a2_dmin) rather than per residue and atom
        pair = matched_a1a2(t, r"dmin|dist|approach|reach")
        if pair:
            d1, d2 = pair
            src = f"matched pair {d1} / {d2}"
        else:
            log.append(f"figure 7 skipped: {path} has neither a "
                       f"{res2}-{atom2} / {res1}-{atom1} column pair nor a "
                       f"matched a1/a2 distance pair. "
                       f"Numeric columns: {numeric_cols(t)[:14]}")
            return None
    log.append(f"  [figure 7] comparing {src}  (from {os.path.basename(path)})")

    o2 = (pick_col(t, res2.lower(), r"frac|occup|contact|pct|percent")
          or pick_col(t, r"(^|[^a-z0-9])a2([^a-z0-9]|$)",
                      r"frac|occup|contact|pct|percent"))
    o1 = (pick_col(t, res1.lower(), r"frac|occup|contact|pct|percent")
          or pick_col(t, r"(^|[^a-z0-9])a1([^a-z0-9]|$)",
                      r"frac|occup|contact|pct|percent"))

    # A pose-level table lists every pose of every receptor down the rows, with
    # the residue columns blank where that residue is not in that receptor.
    # Collapse to one value per compound per atom: the closest approach reached
    # by any pose, which is what "can the ligand reach this atom" means.
    idc = next((c for c in t.columns
                if re.search(r"cpd|compound|ligand|chembl|name|id$", c.lower())
                and not pd.api.types.is_numeric_dtype(t[c])), None)
    if idc is not None and t[idc].duplicated().any():
        g = t.groupby(idc)
        agg = {d1: "min", d2: "min"}
        for c in (o1, o2):
            if c and pd.api.types.is_numeric_dtype(t[c]):
                agg[c] = "mean"
        sub = g.agg(agg)
        o1 = o1 if o1 in sub.columns else None
        o2 = o2 if o2 in sub.columns else None
        log.append(f"  [figure 7] {len(t)} rows collapsed to {len(sub)} "
                   f"compounds by '{idc}' (closest approach over poses)")
    else:
        sub = t[[c for c in (d1, d2, o1, o2) if c]]
    sub = sub.dropna(subset=[d1, d2])
    if len(sub) < 3:
        log.append(f"figure 7 skipped: only {len(sub)} compound(s) have both "
                   f"{d1} and {d2}")
        return None
    diff = (sub[d2] - sub[d1]).to_numpy(float)
    n_pos = int((diff > 0).sum())
    n_neg = int((diff < 0).sum())
    n = len(diff)
    # two-sided: both directions are counted and reported, so a split or a
    # reversal shows up rather than being absorbed by a one-sided alternative
    bt = binomtest(n_pos, n, 0.5, alternative="two-sided")

    log.append(f"\n=== FIGURE 7 : reach to the divergent methyl, n = {n} compounds ===")
    log.append(f"  {lab2} column {d2}: median {sub[d2].median():.2f} Å")
    log.append(f"  {lab1} column {d1}: median {sub[d1].median():.2f} Å")
    log.append(f"  difference ({lab2} − {lab1}): mean {diff.mean():+.3f} Å, "
               f"median {np.median(diff):+.3f}")
    log.append(f"  {lab2} further: {n_pos} of {n} compounds; "
               f"{lab1} further: {n_neg}; tied: {n - n_pos - n_neg}")
    log.append(f"  sign test (two-sided) p = {bt.pvalue:.2e}")
    if o1 and o2:
        log.append(f"  contact frequency: {lab2} {sub[o2].mean():.3f} mean, "
                   f"{lab1} {sub[o1].mean():.3f} mean "
                   f"({o2} vs {o1})")

    has_occ = bool(o1 and o2)
    fig, axes = plt.subplots(1, 2 if has_occ else 1,
                             figsize=(7.0 if has_occ else 3.7, 3.2))
    axA = axes[0] if has_occ else axes
    lim = [min(sub[d1].min(), sub[d2].min()) - 0.15,
           max(sub[d1].max(), sub[d2].max()) + 0.15]
    axA.plot(lim, lim, color="black", lw=0.8, ls=":")
    # points are coloured by which side of the diagonal they fall on, so a
    # mixed result is visible as a mixed plot rather than one flat colour
    side = np.where(diff > 0, HL, np.where(diff < 0, C_A1, GREY))
    axA.scatter(sub[d1], sub[d2], s=30, c=side,
                edgecolor="white", linewidth=0.5, zorder=3)
    axA.set_xlim(lim); axA.set_ylim(lim)
    axA.set_xlabel(f"{lab1}  {d1}  (Å)", fontsize=8)
    axA.set_ylabel(f"{lab2}  {d2}  (Å)", fontsize=8)
    axA.set_title("A   closest ligand-carbon approach", loc="left",
                  fontsize=9, weight="bold")
    # counts go inside the axes, where they cannot collide with panel B's title
    for i, (txt, c) in enumerate((
            (f"above diagonal ({lab2} further):  {n_pos}/{n}", HL),
            (f"below diagonal ({lab1} further):  {n_neg}/{n}", C_A1),
            (f"sign test (two-sided) p = {bt.pvalue:.1e}", "black"))):
        axA.text(0.03, 0.97 - 0.07*i, txt, transform=axA.transAxes,
                 fontsize=7, va="top", ha="left", color=c, weight="bold",
                 bbox=dict(facecolor="white", alpha=0.75, edgecolor="none",
                           pad=1.2), zorder=5)

    if has_occ:
        axB = axes[1]
        m1, m2 = sub[o1].mean(), sub[o2].mean()
        scale = 100.0 if max(m1, m2) <= 1.0 else 1.0
        axB.bar([0, 1], [scale*m1, scale*m2], color=[C_A1, C_A2], width=0.6,
                edgecolor="white")
        for i, m in enumerate((m1, m2)):
            axB.text(i, scale*m + 0.02*scale*max(m1, m2) + 0.4,
                     f"{scale*m:.1f}%" if scale == 100 else f"{m:.1f}",
                     ha="center", fontsize=8, weight="bold")
        axB.set_xticks([0, 1])
        axB.set_xticklabels([f"{lab1}\n{o1}", f"{lab2}\n{o2}"], fontsize=7)
        axB.set_ylabel("Poses making the contact (%)" if scale == 100
                       else "Contact measure")
        axB.set_title("B   contact frequency", loc="left",
                      fontsize=9, weight="bold")

    fig.tight_layout()
    p = os.path.join(outdir, "Figure7_methyl_accessibility.png")
    fig.savefig(p); plt.close(fig)
    return p


# ------------------------------------------------------------- discovery
TABLE_EXT = (".csv", ".tsv", ".xlsx", ".xls", ".xlsm")

# Each entry is (attribute, list of regexes, kind). The first file whose path
# matches every regex wins. Nothing here decides what a figure will show; it
# only locates the file, and the figure still reads whatever is inside it.
WANTED = [
    ("paired",    [r"paired"],                          "table"),
    ("dock",      [r"dock", r"result|score|best"],      "table"),
    ("rescore",   [r"rescore"],                         "table"),
    ("anchor_a1", [r"anchor", r"a1"],                   "table"),
    ("anchor_a2", [r"anchor", r"a2"],                   "table"),
    ("benchmark", [r"comparison|benchmark"],             "table"),
    ("scan",      [r"interact|scan|pharm|rescored"],     "table"),
    ("ens_a1",    [r"ens", r"a1"],                      "pdbdir"),
    ("ens_a2",    [r"ens", r"a2"],                      "pdbdir"),
]


def walk_inputs(roots):
    """Every table and every directory of PDB conformers under the roots."""
    tables, pdbdirs = [], []
    seen = set()
    for root in roots:
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [x for x in dirnames
                           if not x.startswith(".") and x != "__pycache__"]
            n_pdb = sum(1 for f in filenames if f.endswith((".pdb", ".pdb.gz")))
            if n_pdb >= 2:
                pdbdirs.append((dirpath, n_pdb))
            for f in filenames:
                if f.endswith(TABLE_EXT) and not f.startswith("~$"):
                    p = os.path.join(dirpath, f)
                    if p not in seen:
                        seen.add(p)
                        tables.append(p)
    return sorted(tables), sorted(pdbdirs)


_ROWS = {}


def table_rows(path):
    """Row count, read once and remembered. Unreadable files score zero so a
    corrupt table never wins a comparison."""
    if path not in _ROWS:
        try:
            _ROWS[path] = len(read_table(path))
        except Exception:
            _ROWS[path] = 0
    return _ROWS[path]


def discover(a, log):
    """Fill in any input the user did not name, from the roots. An input given
    explicitly is never overridden."""
    tables, pdbdirs = walk_inputs(a.root)
    pdb_count = dict(pdbdirs)
    log.append(f"[scan] {len(tables)} table(s) and {len(pdbdirs)} conformer "
               f"directory/ies under: {', '.join(a.root)}")
    for attr, pats, kind in WANTED:
        if getattr(a, attr, None):
            continue                      # named on the command line, leave it
        pool = tables if kind == "table" else [d for d, _ in pdbdirs]
        hits = [p for p in pool
                if all(re.search(x, os.path.basename(p).lower()
                                 if kind == "table" else p.lower())
                       for x in pats)]
        if hits:
            # the widest table wins, not the shortest path. A run directory can
            # hold a two-row smoke test beside the real thing under the same
            # filename, and the shortest name is as often the former as the
            # latter, so the file is chosen by how much data is in it
            if kind == "table":
                best = max(hits, key=lambda p: (table_rows(p), -len(p)))
            else:
                best = max(hits, key=lambda p: (pdb_count.get(p, 0), -len(p)))
            setattr(a, attr, best)
            size = (f"{table_rows(best)} rows" if kind == "table"
                    else f"{pdb_count.get(best, 0)} pdb")
            log.append(f"[found] {attr:10s} -> {best}  ({size})")
            # every rejected candidate is named, so a wrong pick is visible
            # here rather than only in the figure it produces
            for h in sorted(hits, key=lambda p: -(table_rows(p) if kind == "table"
                                                  else pdb_count.get(p, 0))):
                if h != best:
                    hs = (f"{table_rows(h)} rows" if kind == "table"
                          else f"{pdb_count.get(h, 0)} pdb")
                    log.append(f"         also: {h}  ({hs})")
        else:
            log.append(f"[none ] {attr:10s} -> no match under the roots")
    return log


def inventory(roots, match=None, columns_only=False):
    """Print what is reachable and what is inside it, and stop. This is the
    step to run first: it shows the column names the figures will be matched
    against, without producing any figure or any number that could be mistaken
    for a result."""
    tables, pdbdirs = walk_inputs(roots)
    n_all_t, n_all_d = len(tables), len(pdbdirs)
    if match:
        rx = re.compile(match, re.I)
        tables = [p for p in tables if rx.search(p)]
        pdbdirs = [(d, n) for d, n in pdbdirs if rx.search(d)]
    print(f"# inventory of: {', '.join(roots)}")
    if match:
        print(f"# filter --match {match!r}: {len(tables)} of {n_all_t} table(s), "
              f"{len(pdbdirs)} of {n_all_d} conformer directory/ies")
    else:
        print(f"# {n_all_t} table(s), {n_all_d} directory/ies of conformers")
        if n_all_t > 40:
            print(f"# that is a lot to read. Narrow it with --match, e.g."
                  f"\n#     --match 'interact|comparison|anchor'")
    print()
    for p in tables:
        try:
            t = read_table(p)
        except Exception as e:
            print(f"{p}\n    [unreadable] {e}\n")
            continue
        print(f"{p}")
        print(f"    {len(t)} rows x {len(t.columns)} columns")
        if not columns_only:
            for c in t.columns:
                kind = "num" if pd.api.types.is_numeric_dtype(t[c]) else "str"
                print(f"      {kind}  {c}")
        print()
    for d, n in pdbdirs:
        print(f"{d}\n    {n} PDB file(s)\n")
    print("# nothing was plotted. Re-run without --inventory to make figures.")
    return 0


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paired", help="paired ChEMBL table (figures 1-3). "
                                     "Found under --root if not given")
    ap.add_argument("--dock", help="docking results (figures 2-3). "
                                   "Found under --root if not given")
    ap.add_argument("--dock-sheet", default="best_wide")
    ap.add_argument("--min-members", type=int, default=4)
    ap.add_argument("--rescore",
                    help="figure 4. Either one row per scoring model for one "
                         "fixed pose (model-to-model spread), or a per-compound "
                         "table with matched a1/a2 dispersion columns "
                         "(ensemble dispersion). The shape is detected and the "
                         "figure says which it drew")
    ap.add_argument("--benchmark",
                    help="figure 2: a single table already holding the measured "
                         "value and every criterion side by side, used instead "
                         "of joining --paired to --dock")
    ap.add_argument("--ref-margin", default="",
                    help="optional pK separation of the published reference "
                         "compounds, as low,high. Omitted by default: it is a "
                         "literature value, not something these files contain, "
                         "so figure 4 shows only the measured spread unless "
                         "you supply it")
    ap.add_argument("--anchor-a1", help="anchor separation over the CK2a trajectory")
    ap.add_argument("--anchor-a2", help="anchor separation over the CK2a' trajectory")
    ap.add_argument("--xtal", default="",
                    help="crystal anchor separations as a1,a2 in angstrom (figure 5)")
    ap.add_argument("--hb-cutoff", type=float, default=HB_CUTOFF,
                    help="donor-acceptor distance counted as hydrogen bonded")
    ap.add_argument("--ens-a1", help="directory of CK2a conformer PDBs (figure 6)")
    ap.add_argument("--ens-a2", help="directory of CK2a' conformer PDBs (figure 6)")
    ap.add_argument("--scan", help="analogue scan table (figure 7)")
    ap.add_argument("--scan-sheet", default=None)
    ap.add_argument("--root", action="append", default=[],
                    help="a directory holding the pipeline output. Repeatable. "
                         "Any input not named explicitly is looked for here, "
                         "so pointing at your two working directories is "
                         "usually enough")
    ap.add_argument("--match",
                    help="with --inventory, show only paths matching this "
                         "regex. Use it when the roots hold hundreds of tables")
    ap.add_argument("--columns-only", action="store_true",
                    help="with --inventory, print sizes but not column names")
    ap.add_argument("--res1", default="VAL116", help="figure 7: first residue")
    ap.add_argument("--atom1", default="CG1", help="figure 7: atom of --res1")
    ap.add_argument("--res2", default="ILE117", help="figure 7: second residue")
    ap.add_argument("--atom2", default="CD1", help="figure 7: atom of --res2")
    ap.add_argument("--inventory", action="store_true",
                    help="walk the roots, print every table found with its "
                         "columns and every directory of conformer PDBs, then "
                         "exit without plotting. Run this first to see what "
                         "the script can reach and what it matched")
    ap.add_argument("--label-a1", default="a1",
                    help="axis label for the first subunit (e.g. CK2a)")
    ap.add_argument("--label-a2", default="a2",
                    help="axis label for the second subunit (e.g. CK2a')")
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()

    if a.inventory:
        return inventory(a.root or ["."], a.match, a.columns_only)

    log_into = []
    if a.root:
        discover(a, log_into)
    # fall back to the conventional names only when nothing else supplied them
    a.paired = a.paired or "chembl_ck2_paired.csv"
    a.dock = a.dock or "dock_paired/docking_results.xlsx"
    os.makedirs(a.outdir, exist_ok=True)
    log, made = list(log_into), []

    def parse_pair(s):
        try:
            v = [float(x) for x in s.split(",") if x.strip()]
            return v if len(v) == 2 else None
        except ValueError:
            return None

    # ---- figures 1-3: paired activity and docking
    d = None
    if os.path.exists(a.paired):
        try:
            d = pd.read_csv(a.paired)
            log.append(f"[in] {len(d)} paired compounds from {a.paired}")
        except Exception as e:
            log.append(f"[warn] could not read {a.paired}: {e}")
    else:
        log.append(f"[warn] {a.paired} not found; figures 1-3 will be skipped")

    dock = None
    if os.path.exists(a.dock):
        try:
            dock = read_table(a.dock, a.dock_sheet)
            log.append(f"[in] {len(dock)} rows from {a.dock} [{a.dock_sheet}]")
        except Exception as e:
            log.append(f"[warn] could not read {a.dock}: {e}")
    else:
        log.append(f"[warn] {a.dock} not found; figures 2 and 3 will be skipped")

    benchtab = None
    if a.benchmark:
        if os.path.exists(a.benchmark):
            try:
                benchtab = read_table(a.benchmark)
                log.append(f"[in] {len(benchtab)} rows from {a.benchmark} "
                           f"(self-contained benchmark table)")
            except Exception as e:
                log.append(f"[warn] could not read {a.benchmark}: {e}")
        else:
            log.append(f"[warn] {a.benchmark} not found")

    g = cut = None
    if d is not None:
        try:
            r1 = figure_scaffold_sar(d, a.outdir, a.min_members, log,
                                     a.label_a1, a.label_a2)
        except ImportError:
            log.append("figures 1 and 3 skipped: need rdkit  (pip install rdkit)")
            r1 = None
        if r1:
            made.append(r1[0]); g, cut = r1[1], r1[2]

    # figure 2 needs either a self-contained benchmark table or both of the
    # other two, so it is not nested inside the paired-table branch
    if benchtab is not None or (d is not None and dock is not None):
        r2 = figure_benchmark(d, dock, a.outdir, log, benchtab)
        if r2:
            made.append(r2[0])
            r2[1].to_csv(os.path.join(a.outdir, "Table1_benchmark.csv"),
                         index=False)

    if g is not None and dock is not None:
        r3 = figure_matched_pairs(g, cut, dock, a.outdir, log)
        if r3:
            made.append(r3)

    # ---- figure 4: scoring reproducibility
    if a.rescore:
        if os.path.exists(a.rescore):
            r = figure_rescore_spread(a.rescore, a.outdir, log,
                                      parse_pair(a.ref_margin))
            if not r:
                # not one-row-per-model; try the per-compound dispersion shape
                r = figure_affinity_dispersion(a.rescore, a.outdir, log, None,
                                               a.label_a1, a.label_a2,
                                               parse_pair(a.ref_margin))
            if r:
                made.append(r)
        else:
            log.append(f"figure 4 skipped: {a.rescore} not found")

    # ---- figure 5: hinge anchor geometry
    if a.anchor_a1 or a.anchor_a2:
        missing = [p for p in (a.anchor_a1, a.anchor_a2)
                   if p and not os.path.exists(p)]
        if missing:
            log.append(f"figure 5 skipped: not found: {', '.join(missing)}")
        else:
            r = figure_anchor_distance(a.anchor_a1, a.anchor_a2, a.outdir, log,
                                       parse_pair(a.xtal), a.hb_cutoff,
                                       a.label_a1, a.label_a2)
            if r:
                made.append(r)

    # ---- figure 6: ensemble overlap
    if a.ens_a1 or a.ens_a2:
        missing = [p for p in (a.ens_a1, a.ens_a2) if p and not os.path.isdir(p)]
        if missing:
            log.append(f"figure 6 skipped: not a directory: {', '.join(missing)}")
        else:
            r = figure_ensemble_overlap(a.ens_a1, a.ens_a2, a.outdir, log,
                                        a.label_a1, a.label_a2)
            if r:
                made.append(r)

    # ---- figure 7: methyl accessibility
    if a.scan:
        if os.path.exists(a.scan):
            r = figure_methyl_access(a.scan, a.outdir, log, a.scan_sheet,
                                     a.label_a1, a.label_a2,
                                     a.res1, a.atom1, a.res2, a.atom2)
            if r:
                made.append(r)
        else:
            log.append(f"figure 7 skipped: {a.scan} not found")

    print("\n".join(log))
    with open(os.path.join(a.outdir, "figure_values.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
    print(f"\n[out] {len(made)} figure(s):")
    for m in made:
        print("   ", m)
    print(f"    {a.outdir}/figure_values.txt  (every plotted number)")


if __name__ == "__main__":
    sys.exit(main())

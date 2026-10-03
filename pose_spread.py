#!/usr/bin/env python3
"""
How far apart the poses of a run are, and how far apart the seeds are.

A contact read off the best pose means one thing when the run's leading poses
all sit on top of one another and quite another when they are scattered: in
the second case the best pose is whichever of several the scoring happened to
put first, and so is the contact taken from it.

Two spreads are measured.

  within a run   the pairwise distance among the N best poses of one seed.
                 Small means the search converged on one binding mode.
  between seeds  the distance between the best poses of different seeds in the
                 same structure. Small means the run is reproducible, and a
                 single seed would have answered the same way.

The distance is a symmetry-aware RMSD taken in place, without superposing the
poses on each other, because the question is whether the ligand sits in the
same place and not whether it has the same shape. Poses from one receptor
share a frame, so no alignment is needed or wanted.

    python pose_spread.py --poses crystal_str/ligand/results --top-n 5
"""
import argparse, glob, itertools, os, sys
import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("needs pandas")
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolAlign
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("needs rdkit")


def parse_name(path):
    b = os.path.basename(path)
    for e in (".sdf.gz", ".sdf"):
        if b.endswith(e):
            b = b[: -len(e)]
            break
    p = b.split("__")
    return (p[0], p[1], p[2]) if len(p) >= 3 else None


def prop(m, k):
    if not m.HasProp(k):
        return np.nan
    try:
        return float(m.GetProp(k))
    except ValueError:
        return np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--poses", required=True,
                   help="results directory, one subdirectory per structure")
    p.add_argument("--top-n", type=int, default=5,
                   help="how many of each run's best poses to compare")
    p.add_argument("--select-by", default="CNNscore")
    p.add_argument("--core",
                   help="SMARTS for a shared scaffold; with it the distance "
                        "is measured over the core only, which separates a "
                        "scaffold that moved from substituents that turned")
    p.add_argument("--only", help="comma-separated compounds, in order")
    p.add_argument("--qc-threshold", type=float,
                   help="treat a compound-structure cell as converged when "
                        "the top poses of its seeds agree to within this many "
                        "angstroms, and report what filtering on it would "
                        "cost. Nothing is deleted: the point is to see the "
                        "price before paying it")
    p.add_argument("--qc-stat", choices=("mean", "max"), default="max",
                   help="agreement judged on the mean or the worst pair of "
                        "seeds. 'max' is the stricter and the honest one, "
                        "since one disagreeing seed means the cell did not "
                        "converge")
    p.add_argument("--out")
    a = p.parse_args()

    keep = ([x.strip() for x in a.only.split(",") if x.strip()]
            if a.only else None)
    core = None
    if a.core:
        core = Chem.MolFromSmarts(a.core)
        if core is None:
            sys.exit(f"--core is not valid SMARTS: {a.core!r}")

    def rms(m1, m2):
        if core is None:
            try:
                return float(rdMolAlign.CalcRMS(m1, m2))
            except Exception:
                return np.nan
        # every mapping of a symmetric core is tried and the smallest kept,
        # so a ring numbered the other way round is not read as a flip
        s1 = m1.GetSubstructMatches(core, uniquify=False, maxMatches=200)
        s2 = m2.GetSubstructMatches(core, uniquify=False, maxMatches=200)
        if not s1 or not s2:
            return np.nan
        c1, c2 = m1.GetConformer(), m2.GetConformer()
        p1 = np.array([list(c1.GetAtomPosition(i)) for i in s1[0]])
        best = np.inf
        for i2 in s2:
            if len(i2) != len(s1[0]):
                continue
            p2 = np.array([list(c2.GetAtomPosition(i)) for i in i2])
            best = min(best, float(np.sqrt(((p1 - p2) ** 2).sum(1).mean())))
        return np.nan if not np.isfinite(best) else best

    dirs = sorted(d for d in glob.glob(os.path.join(a.poses, "*"))
                  if os.path.isdir(d))
    if not dirs:
        sys.exit(f"no structure directories under {a.poses}")

    run, bad = {}, []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.sdf"))):
            nm = parse_name(f)
            if nm is None:
                continue
            cpd, struct, seed = nm
            if keep and cpd not in keep:
                continue
            try:
                mols = [m for m in Chem.SDMolSupplier(f, removeHs=False,
                                                      sanitize=True)
                        if m is not None]
            except Exception as e:
                bad.append((os.path.basename(f), str(e).split("\n")[0]))
                continue
            sc = [(prop(m, a.select_by), m) for m in mols]
            sc = [(v, m) for v, m in sc if not np.isnan(v)]
            if not sc:
                continue
            sc.sort(key=lambda r: -r[0])
            run[(cpd, struct, seed)] = [m for _, m in sc[: a.top_n]]

    if not run:
        sys.exit("no runs read")

    rows = []
    for (cpd, struct), grp in itertools.groupby(
            sorted(run, key=lambda k: (k[0], k[1], k[2])),
            key=lambda k: (k[0], k[1])):
        seeds = list(grp)
        within = []
        for k in seeds:
            ms = run[k]
            d = [rms(x, y) for x, y in itertools.combinations(ms, 2)]
            d = [v for v in d if not np.isnan(v)]
            if d:
                within.append((float(np.mean(d)), float(np.max(d))))
        tops = [run[k][0] for k in seeds]
        between = [rms(x, y) for x, y in itertools.combinations(tops, 2)]
        between = [v for v in between if not np.isnan(v)]
        rows.append({
            "compound": cpd, "structure": struct, "seeds": len(seeds),
            "seed_pairs": len(between),
            "within_mean": np.mean([w[0] for w in within]) if within else np.nan,
            "within_max": np.max([w[1] for w in within]) if within else np.nan,
            "between_mean": np.mean(between) if between else np.nan,
            "between_max": np.max(between) if between else np.nan})
    t = pd.DataFrame(rows)

    log = [f"[in] {len(run)} runs, the best {a.top_n} poses of each",
           f"     symmetry-aware RMSD in place, no superposition"
           + (f"; core only: {a.core}" if core is not None else "")]
    if bad:
        log.append(f"     [note] {len(bad)} files not read")

    order = keep if keep else sorted(t["compound"].unique())
    structs = sorted(t["structure"].unique())
    log.append("")
    log.append(f"=== spread of the best {a.top_n} poses, and between seeds "
               f"(A) ===")
    log.append(f"    'within' compares the {a.top_n} poses of one seed with "
               f"each other; 'between' the best pose of each seed with the "
               f"others")
    log.append(f"  {'compound':10s}{'structure':12s}{'seeds':>6s}"
               f"{'within mean':>13s}{'within max':>12s}"
               f"{'pairs':>7s}{'between mean':>14s}{'between max':>13s}")
    log.append(f"  {'':10s}{'':12s}{'':6s}{'':13s}{'':12s}"
               f"{'':7s}   (every pair of seeds, not two of them)")
    for c in order:
        g = t[t.compound == c]
        for st in structs:
            x = g[g.structure == st]
            if not len(x):
                continue
            x = x.iloc[0]
            log.append(f"  {c:10s}{st:12s}{int(x['seeds']):6d}"
                       f"{x['within_mean']:13.2f}{x['within_max']:12.2f}"
                       f"{int(x['seed_pairs']):7d}{x['between_mean']:14.2f}"
                       f"{x['between_max']:13.2f}")
        log.append("")

    log.append("=== by compound, over all structures ===")
    log.append(f"  {'compound':10s}{'within mean':>13s}{'between mean':>14s}"
               f"   reading")
    for c in order:
        g = t[t.compound == c]
        if not len(g):
            continue
        w, b = g["within_mean"].mean(), g["between_mean"].mean()
        note = ("converged; the best pose is a real choice" if w < 2.0
                else "scattered; the best pose is one of several"
                if w < 4.0 else "no single binding mode")
        log.append(f"  {c:10s}{w:13.2f}{b:14.2f}   {note}")
    log.append("")
    log.append(f"  across all compounds: within {t['within_mean'].mean():.2f} "
               f"A, between seeds {t['between_mean'].mean():.2f} A")
    log.append(f"  a contact read from the best pose is only as firm as these "
               f"numbers are small")

    if a.qc_threshold is not None:
        col = "between_" + a.qc_stat
        t["converged"] = t[col] <= a.qc_threshold
        log.append("")
        log.append(f"=== seed agreement: would a {a.qc_threshold} A filter be "
                   f"safe? ===")
        log.append(f"    a cell passes when its seeds' top poses agree to "
                   f"within {a.qc_threshold} A ({a.qc_stat} over the seed "
                   f"pairs)")
        kept, tot = int(t["converged"].sum()), len(t)
        log.append(f"  {kept} of {tot} compound-structure cells pass "
                   f"({100.0 * kept / tot:.0f}%)")
        log.append("")
        log.append(f"  {'compound':10s}{'cells':>7s}{'pass':>7s}{'lost':>7s}"
                   f"{'% kept':>9s}")
        rate = {}
        for c in order:
            g = t[t.compound == c]
            if not len(g):
                continue
            k, n = int(g["converged"].sum()), len(g)
            rate[c] = 100.0 * k / n
            log.append(f"  {c:10s}{n:7d}{k:7d}{n - k:7d}{rate[c]:8.0f}%")
        if rate:
            lo, hi = min(rate.values()), max(rate.values())
            log.append("")
            log.append(f"  the filter keeps {hi:.0f}% of one compound's cells "
                       f"and {lo:.0f}% of another's")
            if hi - lo > 20:
                log.append("  [warn] it falls unevenly across the compounds, "
                           "so the compounds are no longer compared on the "
                           "same set of structures. A per-compound average "
                           "taken after this filter is not a like-for-like "
                           "number")
        log.append("")
        log.append("  the filter removes the cells where the search did not "
                   "settle on one pose. Those are also the cells least likely "
                   "to show a contact, so a contact frequency measured on "
                   "what survives is higher than the frequency in the data, "
                   "whatever the contact actually does")
        log.append("  report both, as a sensitivity check: the number from "
                   "every cell, and the number from the converged ones, and "
                   "say how many cells were dropped")

    text = "\n".join(log)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        t.to_csv(a.out + ".csv", index=False)
        if a.qc_threshold is not None:
            (t[["compound", "structure", "between_mean", "between_max",
                "converged"]]
             .to_csv(a.out + "_qc.csv", index=False))
        with open(a.out + ".txt", "w") as f:
            f.write(text + "\n")
        print(f"\n[out] {a.out}.csv, {a.out}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Does the substitution move the ligand? One compound's pose against another's.

Poses of different compounds docked into the same receptor share a frame, so
they can be compared where they are, without superposition. What they do not
share is a full set of atoms - that is the point of the series - so the
comparison is made over the scaffold every compound has in common, found as
the maximum common substructure. A large value means the substitution pushed
the scaffold somewhere else; a small one means the scaffold stayed and only
the substituent changed.

The number means nothing on its own. The same compound's top pose moves
between seeds by an amount that has to be measured too, and it is measured
here: the same scaffold distance is taken between the seeds of one compound.
A shift between two compounds smaller than that is not a shift, and the
summary says so rather than leaving it to be read off the table.

Only the top pose of each run is used, by default, since that is the pose a
reader would look at.

    python cross_compound_rmsd.py --poses crystal_str/ligand/results \\
        --reference VB004 --out cross_rmsd
"""
import argparse, glob, itertools, os, sys
import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("needs pandas")
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFMCS
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


def core_rms(m1, m2, core):
    """Smallest distance over the core, every symmetry mapping tried.

    A ring numbered the other way round in one molecule is the same ring, so
    taking the first match alone would report a flip that is not there."""
    s1 = m1.GetSubstructMatches(core, uniquify=False, maxMatches=500)
    s2 = m2.GetSubstructMatches(core, uniquify=False, maxMatches=500)
    if not s1 or not s2:
        return np.nan
    c1, c2 = m1.GetConformer(), m2.GetConformer()
    best = np.inf
    for i1 in s1:
        p1 = np.array([list(c1.GetAtomPosition(i)) for i in i1])
        for i2 in s2:
            if len(i2) != len(i1):
                continue
            p2 = np.array([list(c2.GetAtomPosition(i)) for i in i2])
            best = min(best, float(np.sqrt(((p1 - p2) ** 2).sum(1).mean())))
    return np.nan if not np.isfinite(best) else best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--poses", required=True,
                   help="results directory, one subdirectory per structure")
    p.add_argument("--top-n", type=int, default=1,
                   help="how many of each run's best poses to use. 1 is the "
                        "top pose; with more, every pose of one compound is "
                        "compared with every pose of the other and the "
                        "smallest distance kept")
    p.add_argument("--select-by", default="CNNscore")
    p.add_argument("--reference", help="the compound the others are measured "
                                       "against, usually the parent")
    p.add_argument("--core", help="SMARTS for the shared scaffold. Without it "
                                  "the maximum common substructure of the "
                                  "compounds is used and printed")
    p.add_argument("--only", help="comma-separated compounds, in order")
    p.add_argument("--mcs-timeout", type=int, default=60)
    p.add_argument("--out")
    a = p.parse_args()

    keep = ([x.strip() for x in a.only.split(",") if x.strip()]
            if a.only else None)

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

    cpds = keep if keep else sorted({k[0] for k in run})
    structs = sorted({k[1] for k in run})
    seeds = sorted({k[2] for k in run})

    # one representative of each compound, for the common scaffold
    reps = {}
    for c in cpds:
        for k in run:
            if k[0] == c:
                reps[c] = run[k][0]
                break
    missing = [c for c in cpds if c not in reps]
    if missing:
        sys.exit("no poses for " + ", ".join(missing))

    if a.core:
        core = Chem.MolFromSmarts(a.core)
        if core is None:
            sys.exit(f"--core is not valid SMARTS: {a.core!r}")
        core_src = a.core
    else:
        res = rdFMCS.FindMCS([Chem.RemoveHs(reps[c]) for c in cpds],
                             ringMatchesRingOnly=True, completeRingsOnly=True,
                             timeout=a.mcs_timeout)
        if res.canceled:
            print("[warn] the common substructure search timed out; the core "
                  "below is the best found so far", file=sys.stderr)
        core = Chem.MolFromSmarts(res.smartsString)
        core_src = res.smartsString
    if core is None or core.GetNumAtoms() == 0:
        sys.exit("no shared scaffold found; give one with --core")
    ncore = core.GetNumAtoms()
    short = [c for c in cpds if not reps[c].GetSubstructMatches(core)]
    if short:
        sys.exit("the scaffold is not present in " + ", ".join(short))

    def dist(ms1, ms2):
        v = [core_rms(x, y, core) for x in ms1 for y in ms2]
        v = [q for q in v if not np.isnan(q)]
        return min(v) if v else np.nan

    # between compounds, within one structure and one seed
    pair_rows = []
    for st in structs:
        for sd in seeds:
            here = {c: run[(c, st, sd)] for c in cpds if (c, st, sd) in run}
            for c1, c2 in itertools.combinations(sorted(here), 2):
                v = dist(here[c1], here[c2])
                if not np.isnan(v):
                    pair_rows.append({"structure": st, "seed": sd,
                                      "a": c1, "b": c2, "rmsd": v})
    if not pair_rows:
        sys.exit("no compound pairs shared a structure and a seed")
    pairs = pd.DataFrame(pair_rows)

    # the same compound between its own seeds: the floor the shifts above
    # have to clear before they mean anything
    noise_rows = []
    for c in cpds:
        for st in structs:
            ss = [s for s in seeds if (c, st, s) in run]
            for s1, s2 in itertools.combinations(ss, 2):
                v = dist(run[(c, st, s1)], run[(c, st, s2)])
                if not np.isnan(v):
                    noise_rows.append({"compound": c, "structure": st,
                                       "rmsd": v})
    noise = pd.DataFrame(noise_rows)
    floor = float(noise["rmsd"].mean()) if len(noise) else np.nan
    floor_hi = (float(noise["rmsd"].quantile(0.9)) if len(noise) else np.nan)

    log = [f"[in] {len(run)} runs, {len(cpds)} compounds x {len(structs)} "
           f"structures x {len(seeds)} seeds, "
           + ("top pose only" if a.top_n == 1
              else f"best {a.top_n} poses, closest pair taken"),
           f"     shared scaffold: {ncore} atoms"
           + ("" if a.core else " (maximum common substructure)"),
           f"     {core_src}",
           f"     distance taken in place, no superposition: the receptors are "
           f"the same, so the frames are"]
    if bad:
        log.append(f"     [note] {len(bad)} files not read")

    log.append("")
    log.append("=== how far one compound's pose sits from another's (A) ===")
    log.append(f"    mean over {len(structs)} structures and {len(seeds)} "
               f"seeds, with the spread")
    if a.reference:
        if a.reference not in cpds:
            sys.exit(f"--reference {a.reference} is not among the compounds")
        log.append(f"    measured against {a.reference}")
        log.append(f"  {'compound':10s}{'shift':>9s}{'sd':>8s}{'max':>8s}"
                   f"{'n':>5s}   reading")
        ref_rows = []
        for c in cpds:
            if c == a.reference:
                continue
            g = pairs[((pairs.a == c) & (pairs.b == a.reference)) |
                      ((pairs.b == c) & (pairs.a == a.reference))]
            if not len(g):
                continue
            m = g["rmsd"].mean()
            note = ("below the seed-to-seed floor; not a shift"
                    if m <= floor else
                    "above the floor but inside its spread" if m <= floor_hi
                    else "larger than the seed-to-seed floor")
            log.append(f"  {c:10s}{m:9.2f}{g['rmsd'].std(ddof=1):8.2f}"
                       f"{g['rmsd'].max():8.2f}{len(g):5d}   {note}")
            for _, r in g.iterrows():
                ref_rows.append({"compound": c, "structure": r["structure"],
                                 "seed": r["seed"], "shift": r["rmsd"]})
        ref_tab = (pd.DataFrame(ref_rows)
                   .groupby(["compound", "structure"], as_index=False)["shift"]
                   .mean()) if ref_rows else pd.DataFrame()
    else:
        ref_tab = pd.DataFrame()

    log.append("")
    log.append("=== every pair, averaged over structures and seeds (A) ===")
    log.append(f"  {'':10s}" + "".join(f"{c[-4:]:>8s}" for c in cpds))
    mean_pair = (pairs.groupby(["a", "b"])["rmsd"].mean().to_dict())
    for c1 in cpds:
        row = f"  {c1:10s}"
        for c2 in cpds:
            if c1 == c2:
                row += f"{'-':>8s}"
            else:
                v = mean_pair.get((c1, c2), mean_pair.get((c2, c1)))
                row += f"{v:8.2f}" if v is not None else f"{'':>8s}"
        log.append(row)

    log.append("")
    log.append("=== the floor: the same compound between its own seeds (A) ===")
    log.append(f"  mean {floor:.2f}, 90th percentile {floor_hi:.2f}, "
               f"max {noise['rmsd'].max():.2f} over {len(noise)} seed pairs")
    above = (pairs.groupby(["a", "b"])["rmsd"].mean() > floor_hi).sum()
    tot = pairs.groupby(["a", "b"]).ngroups
    log.append(f"  {above} of {tot} compound pairs sit further apart than the "
               f"90th percentile of that floor")
    if above == 0:
        log.append("  no substitution moved the scaffold further than the "
                   "search moves it on its own; the poses are the same pose")
    log.append("")
    log.append("  a shift is only a shift if it clears this floor. The "
               "scaffold is what is measured, so a substituent that turns "
               "without moving the scaffold reads as zero here, correctly")

    text = "\n".join(log)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        pairs.to_csv(a.out + "_pairs.csv", index=False)
        noise.to_csv(a.out + "_seedfloor.csv", index=False)
        if len(ref_tab):
            ref_tab.to_csv(a.out + "_vs_reference.csv", index=False)
        with open(a.out + ".txt", "w") as f:
            f.write(text + "\n")
        made = [a.out + "_pairs.csv", a.out + "_seedfloor.csv", a.out + ".txt"]
        if len(ref_tab):
            made.insert(2, a.out + "_vs_reference.csv")
        print("\n[out] " + ", ".join(made))
    return 0


if __name__ == "__main__":
    sys.exit(main())

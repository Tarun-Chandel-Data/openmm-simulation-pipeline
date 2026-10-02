#!/usr/bin/env python3
"""
How often each compound reaches the hinge, across structures and seeds.

A docking score could not separate these compounds: one compound's spread
across receptors was larger than the spread between all of them. A contact is
a better endpoint than a score, because it is a yes or no per run rather than
a number carrying the scoring function's error, and over many runs it becomes
a frequency with an error bar.

Each run contributes one answer: did this compound's best pose in this
structure, under this seed, hydrogen bond the named residue. The same
geometric test is used as everywhere else in this repository, so these counts
are comparable with the ensemble ones.

Two things are reported beyond the counts. Agreement between seeds of the same
compound and structure says how reliable a single run is - where the seeds
disagree, a one-seed study would have reported whichever it happened to draw.
And named pairs of compounds are compared structure by structure, which is
what makes this a test rather than a ranking: two compounds of identical
composition differing only in where a substituent sits hold everything else
constant, so a difference between them is positional.

    python hinge_frequency.py --receptors crystal_str/receptor \\
        --poses crystal_str/ligand/results --residues TYR116,ILE117 \\
        --pair EV043:EV034 --pair EV047:EV038 --out hinge_panel
"""
import argparse, glob, os, re, sys
from math import comb
import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("needs pandas")
try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("needs rdkit")

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
try:
    from hbond_geometry import polar_sites, bonds_for_pose
except ImportError:
    sys.exit("needs hbond_geometry.py beside this script")


def parse_name(path):
    """(compound, structure, seed) from <cpd>__<structure>__s<seed>.sdf."""
    b = os.path.basename(path)
    for e in (".sdf.gz", ".sdf"):
        if b.endswith(e):
            b = b[: -len(e)]
            break
    parts = b.split("__")
    if len(parts) < 3:
        return None
    seed = parts[2]
    m = re.match(r"^s(\d+)$", seed)
    return parts[0], parts[1], (m.group(1) if m else seed)


def prop(m, k):
    if not m.HasProp(k):
        return np.nan
    try:
        return float(m.GetProp(k))
    except ValueError:
        return np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--receptors", required=True,
                   help="directory of the prepared .pdb receptors")
    p.add_argument("--poses", required=True,
                   help="results directory, one subdirectory per structure")
    p.add_argument("--residues", default="TYR116,ILE117")
    p.add_argument("--select-by", default="CNNscore",
                   help="property the best pose of each run is chosen on")
    p.add_argument("--pair", action="append", default=[], metavar="A:B",
                   help="two compounds to compare structure by structure. "
                        "Repeat for each pair")
    p.add_argument("--only", help="restrict to these compounds")
    p.add_argument("--dist", type=float, default=3.5)
    p.add_argument("--h-dist", type=float, default=2.5)
    p.add_argument("--angle", type=float, default=120.0)
    p.add_argument("--antecedent-angle", type=float, default=90.0)
    p.add_argument("--out")
    a = p.parse_args()

    res = [x.strip().upper() for x in a.residues.split(",") if x.strip()]
    keep = ([x.strip() for x in a.only.split(",") if x.strip()]
            if a.only else None)

    dirs = sorted(d for d in glob.glob(os.path.join(a.poses, "*"))
                  if os.path.isdir(d))
    if not dirs:
        sys.exit(f"no structure directories under {a.poses}")

    log = [f"[in] {len(dirs)} structures under {a.poses}",
           f"     hydrogen bonds: D-A <= {a.dist} A, H...A <= {a.h_dist} A, "
           f"D-H...A >= {a.angle:g} deg",
           f"     residues: {', '.join(res)}; best pose of each run by "
           f"{a.select_by}"]

    rows, notes, bad = [], [], []
    for d in dirs:
        sname = os.path.basename(d)
        rec = os.path.join(a.receptors, sname + ".pdb")
        if not os.path.exists(rec):
            cand = glob.glob(os.path.join(a.receptors, sname + "*.pdb"))
            if not cand:
                notes.append(f"     [note] no receptor for {sname}; skipped")
                continue
            rec = cand[0]
        sites = polar_sites(rec)
        have = {s[0].upper() for s in sites}
        gone = [r for r in res if r not in have]
        if gone:
            notes.append(f"     [note] {sname}: {', '.join(gone)} carries no "
                         f"polar site, so a bond to it cannot be found")
        for f in sorted(glob.glob(os.path.join(d, "*.sdf"))):
            nm = parse_name(f)
            if nm is None:
                continue
            cpd, struct, seed = nm
            if keep and cpd not in keep:
                continue
            top, tv = None, None
            try:
                # one unreadable file out of hundreds should be named and
                # left out, not stop the analysis; a run that was interrupted
                # or failed leaves exactly such a file
                supplier = Chem.SDMolSupplier(f, removeHs=False, sanitize=True)
                mols = list(supplier)
            except Exception as e:
                bad.append((f, str(e).split("\n")[0]))
                continue
            # every pose is tested, not only the best. A contact the search
            # finds once and nowhere else is a different thing from one it
            # finds in most of the poses it returns, and the top pose alone
            # cannot tell them apart
            per_pose = []
            for m in mols:
                if m is None:
                    continue
                v = prop(m, a.select_by)
                if np.isnan(v):
                    continue
                try:
                    mh = Chem.AddHs(m, addCoords=True)
                except Exception:
                    mh = m
                hits, _ = bonds_for_pose(mh, sites, a)
                up = {k.upper(): int(hits.get(k2, 0) > 0)
                      for k2 in hits for k in [k2.upper()]}
                flags = {x: int(up.get(x, 0) > 0) for x in res}
                per_pose.append((v, flags))
                if tv is None or v > tv:
                    top, tv = m, v
            if top is None or not per_pose:
                bad.append((f, "no pose carried " + a.select_by))
                continue
            best = max(per_pose, key=lambda kv: kv[0])[1]
            r = {"compound": cpd, "structure": struct, "seed": seed,
                 "score": tv, "n_pose": len(per_pose)}
            for x in res:
                r[x] = best[x]                                  # the top pose
                r[x + "_share"] = float(np.mean([f[x] for _, f in per_pose]))
                r[x + "_any"] = int(any(f[x] for _, f in per_pose))
            r["either"] = int(any(r[x] for x in res))
            r["both"] = int(all(r[x] for x in res))
            rows.append(r)
    if not rows:
        sys.exit("no runs read")
    t = pd.DataFrame(rows)
    log += notes
    if bad:
        log.append(f"     [note] {len(bad)} of {len(bad) + len(rows)} runs "
                   f"could not be read and are left out. A compound missing "
                   f"runs is not comparable with one that has them all, so "
                   f"re-run these before using the counts:")
        for f, why in bad[:12]:
            log.append(f"            {os.path.basename(f)}  -  {why}")
        if len(bad) > 12:
            log.append(f"            ... and {len(bad) - 12} more")

    nrun = t.groupby("compound").size()
    log.append("")
    log.append("=== how often the best pose reaches the hinge ===")
    log.append(f"    one run is one structure under one seed; the figure is "
               f"the share of runs in which the bond was present")
    hdr = f"  {'compound':10s}{'runs':>6s}"
    for x in res + ["either", "both"]:
        hdr += f"{x:>12s}"
    hdr += f"{'structures':>12s}"
    log.append(hdr)
    order = keep if keep else sorted(t["compound"].unique())
    summ = []
    for c in order:
        g = t[t.compound == c]
        if not len(g):
            continue
        line = f"  {c:10s}{len(g):6d}"
        rec = {"compound": c, "runs": len(g)}
        for x in res + ["either", "both"]:
            frac = g[x].mean()
            rec[x] = frac
            line += f"{frac * 100:11.0f}%"
        # the share of structures where it happened under at least one seed
        bys = g.groupby("structure")[res[0]].max()
        rec["structures_" + res[0]] = f"{int(bys.sum())}/{len(bys)}"
        line += f"{int(bys.sum()):>8d}/{len(bys):<3d}"
        log.append(line)
        summ.append(rec)

    log.append("")
    log.append("=== across all poses of each run, not only the best ===")
    log.append(f"    'top' repeats the table above. 'any' is the share of "
               f"runs where at least one returned pose made the bond, and "
               f"'per pose' the average share of the poses in a run that did")
    log.append(f"    the poses of one run come from one search and are not "
               f"independent of each other, so these describe how consistent "
               f"a run is; the counting for a test stays at the run level")
    hdr = f"  {'compound':10s}{'poses':>7s}"
    for x in res:
        hdr += f"{x + ' top':>13s}{x + ' any':>13s}{x + ' /pose':>14s}"
    log.append(hdr)
    for c in order:
        g = t[t.compound == c]
        if not len(g):
            continue
        line = f"  {c:10s}{int(g['n_pose'].sum()):7d}"
        for x in res:
            line += (f"{g[x].mean() * 100:12.0f}%"
                     f"{g[x + '_any'].mean() * 100:12.0f}%"
                     f"{g[x + '_share'].mean() * 100:13.0f}%")
        log.append(line)

    # seeds of the same compound and structure should agree if one run means
    # anything; where they do not, a single-seed study is drawing a coin
    log.append("")
    log.append("=== do the seeds agree? ===")
    for x in res + ["either"]:
        g = t.groupby(["compound", "structure"])[x]
        n = g.size()
        split = ((g.mean() > 0) & (g.mean() < 1)).sum()
        log.append(f"  {x}: {split} of {len(n)} compound-structure pairs have "
                   f"seeds disagreeing "
                   f"({split / max(len(n), 1) * 100:.0f}%)")
    log.append(f"    a high figure means one run of one seed would have "
               f"reported whichever answer it happened to draw")

    if a.pair:
        log.append("")
        log.append("=== matched pairs, structure by structure ===")
        log.append(f"    each pair is two compounds of the same composition "
                   f"differing in where a substituent sits, so a difference "
                   f"between them is positional")
        for spec in a.pair:
            if ":" not in spec:
                sys.exit(f"--pair wants A:B, got {spec!r}")
            x, y = (s.strip() for s in spec.split(":", 1))
            gx, gy = t[t.compound == x], t[t.compound == y]
            if not len(gx) or not len(gy):
                log.append(f"  {x} vs {y}: one of them has no runs")
                continue
            log.append("")
            log.append(f"  {x} vs {y}   ({res[0]}, share of seeds per "
                       f"structure)")
            sx = gx.groupby("structure")[res[0]].mean()
            sy = gy.groupby("structure")[res[0]].mean()
            both = sorted(set(sx.index) & set(sy.index))
            wins = {x: 0, y: 0, "tie": 0}
            for s in both:
                mark = (x if sx[s] > sy[s] else y if sy[s] > sx[s] else "tie")
                wins[mark] += 1
                log.append(f"    {s:16s}{sx[s] * 100:6.0f}%  "
                           f"{sy[s] * 100:6.0f}%   {mark}")
            n = wins[x] + wins[y]
            log.append(f"    {x} ahead in {wins[x]} of {len(both)} "
                       f"structures, {y} in {wins[y]}, tied in {wins['tie']}")
            if n:
                # a two-sided sign test over the structures that differ
                k = max(wins[x], wins[y])
                pv = min(2 * sum(comb(n, i)
                                 for i in range(k, n + 1)) / 2 ** n, 1.0)
                log.append(f"    sign test over the {n} structures that "
                           f"differ: p = {pv:.3f}"
                           + ("   (not significant; more structures or a "
                              "larger effect would be needed)"
                              if pv > 0.05 else ""))

    text = "\n".join(log)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        t.to_csv(a.out + "_runs.csv", index=False)
        pd.DataFrame(summ).to_csv(a.out + "_summary.csv", index=False)
        with open(a.out + ".txt", "w") as f:
            f.write(text + "\n")
        print(f"\n[out] {a.out}_runs.csv, {a.out}_summary.csv, {a.out}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

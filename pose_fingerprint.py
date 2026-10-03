#!/usr/bin/env python3
"""
How many hydrogen bonds a run's leading poses make, and whether they make
them to the same residues.

The distance between two poses says where the ligand sits; it does not say
what the ligand touches. Two poses an angstrom apart can hydrogen bond
different residues, and two poses further apart can hold the same contacts
while the rest of the molecule turns. This counts the bonds and compares the
residues.

Per run, over its N best poses:

  hydrogen bonds   how many hydrogen bonds the N poses make in total, and how
                   many of them are to the hinge residues. These are counts of
                   bonds, which is what the figure draws
  bonds per pose   the same count divided by the number of poses, kept here
                   because it is what makes runs of different size comparable
  residues seen    how many distinct residues the poses touch between them
  kept by all      how many are touched by every one of the poses - the part
                   of the binding that does not depend on which pose is read
  agreement        the average overlap between two poses' residue sets, as a
                   share of the residues either of them touches. One means
                   every pose touches the same residues, zero means no two
                   poses share any.

Hydrogen bonds are the geometric test used throughout this work.

    python pose_fingerprint.py --receptors crystal_str/receptor \\
        --poses crystal_str/ligand/results --top-n 5
"""
import argparse, glob, itertools, os, sys
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
    p.add_argument("--receptors", required=True)
    p.add_argument("--poses", required=True)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--select-by", default="CNNscore")
    p.add_argument("--only", help="comma-separated compounds, in order")
    p.add_argument("--highlight", default="TYR116,ILE117",
                   help="residues to report separately in the summary")
    p.add_argument("--dist", type=float, default=3.5)
    p.add_argument("--h-dist", type=float, default=2.5)
    p.add_argument("--angle", type=float, default=120.0)
    p.add_argument("--antecedent-angle", type=float, default=90.0)
    p.add_argument("--out")
    a = p.parse_args()

    keep = ([x.strip() for x in a.only.split(",") if x.strip()]
            if a.only else None)
    hi = [x.strip().upper() for x in a.highlight.split(",") if x.strip()]

    dirs = sorted(d for d in glob.glob(os.path.join(a.poses, "*"))
                  if os.path.isdir(d))
    if not dirs:
        sys.exit(f"no structure directories under {a.poses}")

    runs, bad = {}, []
    for d in dirs:
        sname = os.path.basename(d)
        rec = os.path.join(a.receptors, sname + ".pdb")
        if not os.path.exists(rec):
            cand = glob.glob(os.path.join(a.receptors, sname + "*.pdb"))
            if not cand:
                bad.append((sname, "no receptor"))
                continue
            rec = cand[0]
        sites = polar_sites(rec)
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
            fps = []
            for _, m in sc[: a.top_n]:
                try:
                    mh = Chem.AddHs(m, addCoords=True)
                except Exception:
                    mh = m
                hits, _ = bonds_for_pose(mh, sites, a)
                up = {k.upper(): v for k, v in hits.items()}
                fps.append((set(up), int(sum(up.values())),
                            int(sum(v for k, v in up.items() if k in hi))))
            if fps:
                runs[(cpd, struct, seed)] = fps

    if not runs:
        sys.exit("no runs read")

    rows = []
    for (cpd, struct), grp in itertools.groupby(
            sorted(runs, key=lambda k: (k[0], k[1], k[2])),
            key=lambda k: (k[0], k[1])):
        seeds = list(grp)
        nb, seen, kept, agree, conserved = [], [], [], [], []
        tot, tot_hi, npose = 0, 0, 0
        for k in seeds:
            fps = runs[k]
            sets = [s for s, _, _ in fps]
            nb += [n for _, n, _ in fps]
            tot += sum(n for _, n, _ in fps)
            tot_hi += sum(n for _, _, n in fps)
            npose += len(fps)
            union = set().union(*sets)
            inter = set.intersection(*sets) if sets else set()
            seen.append(len(union))
            kept.append(len(inter))
            conserved.append(inter)
            j = [len(x & y) / len(x | y) for x, y in
                 itertools.combinations(sets, 2) if x | y]
            if j:
                agree.append(float(np.mean(j)))
        always = (set.intersection(*conserved) if conserved else set())
        rows.append({
            "compound": cpd, "structure": struct, "seeds": len(seeds),
            "poses": len(nb),
            # plain counts of hydrogen bonds, which is what the figure draws.
            # Divided by the number of seeds, so a run repeated three times
            # does not read as three times the bonding
            "hbonds": tot / len(seeds) if seeds else np.nan,
            "hinge_hbonds": tot_hi / len(seeds) if seeds else np.nan,
            "bonds_mean": float(np.mean(nb)) if nb else np.nan,
            "bonds_sd": float(np.std(nb, ddof=1)) if len(nb) > 1 else 0.0,
            "residues_seen": float(np.mean(seen)) if seen else np.nan,
            "kept_by_all": float(np.mean(kept)) if kept else np.nan,
            "agreement": float(np.mean(agree)) if agree else np.nan,
            "conserved": " ".join(sorted(always))})
    t = pd.DataFrame(rows)

    log = [f"[in] {len(runs)} runs, the best {a.top_n} poses of each",
           f"     hydrogen bonds by the geometric test: D-A <= {a.dist} A "
           f"with H...A <= {a.h_dist} A and D-H...A >= {a.angle:g} deg"]
    if bad:
        log.append(f"     [note] {len(bad)} not read")

    order = keep if keep else sorted(t["compound"].unique())
    structs = sorted(t["structure"].unique())
    # one pose per run leaves nothing to compare a pose with, so the columns
    # that describe pose-to-pose agreement are left out rather than printed
    # as a blank or a one
    solo = a.top_n == 1
    log.append("")
    if solo:
        log.append("=== the top pose of each run: how many hydrogen bonds ===")
        log.append("    the bonds made by the pose the scoring put first. "
                   "Where a cell holds more than one run they are averaged "
                   "over the runs, so the figure is per run either way")
    else:
        log.append(f"=== the best {a.top_n} poses of each run: how many bonds, "
                   f"and to the same residues? ===")
        log.append(f"    'kept by all' is the number of residues every one of "
                   f"the poses bonds; 'agreement' is how much two poses' "
                   f"residue sets overlap, 1 being identical")
    log.append(f"  {'compound':10s}{'structure':12s}{'H-bonds':>9s}"
               f"{'hinge':>8s}"
               + ("" if solo else
                  f"{'bonds/pose':>12s}{'residues':>10s}"
                  f"{'kept by all':>13s}{'agreement':>11s}")
               + ("   residues bonded" if solo
                  else "   residues every pose makes"))
    log.append(f"  {'':10s}{'':12s}{'per run':>9s}{'/run':>8s}"
               + ("" if solo else f"{'mean +- sd':>12s}"))
    for c in order:
        g = t[t.compound == c]
        for st in structs:
            x = g[g.structure == st]
            if not len(x):
                continue
            x = x.iloc[0]
            log.append(f"  {c:10s}{st:12s}"
                       f"{x['hbonds']:9.1f}{x['hinge_hbonds']:8.1f}"
                       + ("" if solo else
                          f"{x['bonds_mean']:7.1f} +-{x['bonds_sd']:<4.1f}"
                          f"{x['residues_seen']:10.1f}"
                          f"{x['kept_by_all']:13.1f}{x['agreement']:11.2f}")
                       + f"   {x['conserved'] or '-'}")
        log.append("")

    log.append("=== by compound ===")
    log.append(f"  {'compound':10s}{'H-bonds':>9s}{'hinge':>8s}"
               + ("" if solo else
                  f"{'bonds/pose':>12s}{'agreement':>11s}"
                  f"{'kept by all':>13s}   reading"))
    for c in order:
        g = t[t.compound == c]
        if not len(g):
            continue
        line = (f"  {c:10s}{g['hbonds'].mean():9.1f}"
                f"{g['hinge_hbonds'].mean():8.1f}")
        if not solo:
            ag = g["agreement"].mean()
            note = ("the same contacts in every pose" if ag >= 0.75 else
                    "mostly the same contacts" if ag >= 0.5 else
                    "the poses contact different residues")
            line += (f"{g['bonds_mean'].mean():12.1f}{ag:11.2f}"
                     f"{g['kept_by_all'].mean():13.1f}   {note}")
        log.append(line)

    for r in hi:
        held = t["conserved"].str.contains(r, na=False)
        log.append("")
        log.append(f"  {r} is bonded by "
                   + ("the top pose of every run" if solo else
                      f"every one of the best {a.top_n} poses")
                   + f" in {int(held.sum())} of {len(t)} compound-structure "
                     f"pairs")
        if held.any():
            who = (t[held].groupby("compound").size()
                   .sort_values(ascending=False))
            log.append("    " + ", ".join(f"{c} {n}" for c, n in who.items()))

    text = "\n".join(log)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        t.to_csv(a.out + ".csv", index=False)
        with open(a.out + ".txt", "w") as f:
            f.write(text + "\n")
        print(f"\n[out] {a.out}.csv, {a.out}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

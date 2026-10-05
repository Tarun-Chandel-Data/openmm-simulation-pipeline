#!/usr/bin/env python3
"""
One row per compound and protein: the top pose, and all the poses.

For each compound in each structure: whether the top pose hydrogen bonds the
hinge, that pose's score and affinity, then the same over every pose the runs
returned, with the spread.

Hydrogen bonds are the geometric test used throughout this work - a distance
with the angles, not a distance alone - so these counts are comparable with
the ensemble ones.

The runs were repeated under several seeds, so the denominators are the number
of runs and the number of poses rather than 1 and 10; both are printed beside
the counts.

    python hinge_table.py --receptors crystal_str/receptor \\
        --poses crystal_str/ligand/results --residues TYR116,ILE117
"""
import argparse, glob, os, re, sys
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


def matches(want, found):
    """Does any bonded residue answer to this name?

    A residue given with its number is matched exactly. A residue given by
    name alone matches any number, which is what a panel of crystal entries
    needs: they number the same residue differently, so one run cannot name
    them all, and asking for TYR116 across eight structures finds it in at
    most one of them.
    """
    w = want.upper()
    if any(ch.isdigit() for ch in w):
        return w in found
    return any("".join(c for c in k if c.isalpha()).upper() == w
               for k in found)


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
    p.add_argument("--residues", default="TYR116,ILE117")
    p.add_argument("--score", default="CNNscore")
    p.add_argument("--affinity", default="minimizedAffinity")
    p.add_argument("--only", help="comma-separated compounds, in order")
    p.add_argument("--by-run", action="store_true",
                   help="lay the runs out side by side, one column per run, "
                        "instead of pooling them into one count. Pooling "
                        "hides whether the runs agree, and two runs are "
                        "repeats of each other only when they used the same "
                        "receptors: ensembles from different simulations are "
                        "different experiments")
    p.add_argument("--top-n", type=int, default=0,
                   help="keep only the N best-scoring poses of each run in "
                        "the all-pose columns. The lower-ranked poses of a "
                        "run are the ones the search is least confident in, "
                        "so counting them dilutes the contact with poses "
                        "nobody would propose. 0 keeps every pose")
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

    cell, bad = {}, []
    for d in dirs:
        sname = os.path.basename(d)
        rec = os.path.join(a.receptors, sname + ".pdb")
        if not os.path.exists(rec):
            cand = glob.glob(os.path.join(a.receptors, sname + "*.pdb"))
            if not cand:
                bad.append((sname, "no receptor file"))
                continue
            rec = cand[0]
        sites = polar_sites(rec)
        for f in sorted(glob.glob(os.path.join(d, "*.sdf"))):
            nm = parse_name(f)
            if nm is None:
                continue
            cpd, struct, runid = nm
            if keep and cpd not in keep:
                continue
            try:
                mols = list(Chem.SDMolSupplier(f, removeHs=False,
                                               sanitize=True))
            except Exception as e:
                bad.append((os.path.basename(f), str(e).split("\n")[0]))
                continue
            k = (cpd, struct, runid) if a.by_run else (cpd, struct)
            c = cell.setdefault(k, {"runs": 0, "top_hb": {x: 0 for x in res},
                                    "top_either": 0, "top_s": [], "top_a": [],
                                    "n_pose": 0, "hb": {x: 0 for x in res},
                                    "either": 0, "s": [], "aff": []})
            scored = []
            for m in mols:
                if m is None:
                    continue
                sc, af = prop(m, a.score), prop(m, a.affinity)
                if np.isnan(sc):
                    continue
                try:
                    mh = Chem.AddHs(m, addCoords=True)
                except Exception:
                    mh = m
                hits, _ = bonds_for_pose(mh, sites, a)
                up = {kk.upper() for kk in hits}
                scored.append((sc, af, {x: matches(x, up) for x in res}))
            scored.sort(key=lambda r: -r[0])
            best = (scored[0][2], scored[0][0], scored[0][1]) if scored else None
            # the all-pose columns describe the best N of the run, since the
            # poses the search ranks last are the ones it is least sure of
            for sc, af, fl in (scored[: a.top_n] if a.top_n else scored):
                c["n_pose"] += 1
                c["s"].append(sc)
                c["aff"].append(af)
                for x in res:
                    c["hb"][x] += int(fl[x])
                c["either"] += int(any(fl.values()))
            if best is None:
                bad.append((os.path.basename(f), "no pose carried " + a.score))
                continue
            fl, sc, af = best
            c["runs"] += 1
            for x in res:
                c["top_hb"][x] += int(fl[x])
            c["top_either"] += int(any(fl.values()))
            c["top_s"].append(sc)
            c["top_a"].append(af)

    if not cell:
        sys.exit("no runs read")

    cpds = keep if keep else sorted({k[0] for k in cell})
    structs = sorted({k[1] for k in cell})
    runs = sorted({k[2] for k in cell}) if a.by_run else [None]

    log = [f"[in] {len(cpds)} compounds x {len(structs)} structures",
           f"     hydrogen bond: D-A <= {a.dist} A with H...A <= {a.h_dist} A "
           f"and D-H...A >= {a.angle:g} deg - the geometric test, not a "
           f"distance alone",
           f"     hinge residues: {' or '.join(res)}"
           + ("  (by name, any number)"
              if not any(ch.isdigit() for r in res for ch in r)
              else "  (name and number, exactly)")]
    if bad:
        log.append(f"     [note] {len(bad)} files not read: "
                   + "; ".join(f"{x} ({y})" for x, y in bad[:4])
                   + (" ..." if len(bad) > 4 else ""))
    log.append("")
    log.append("=== top pose, and all poses, per compound and structure ===")
    log.append(f"    'top' columns describe the best pose of each run; "
               + (f"the pose columns the best {a.top_n} poses of each run"
                  if a.top_n else "'all poses' every pose the runs returned"))
    log.append(f"    the hinge columns are either residue, which the backbone "
               f"contact at {res[1] if len(res) > 1 else res[0]} very nearly "
               f"saturates; the per-residue columns beside them are what "
               f"separates the compounds")
    # the residue columns are headed with the residues actually asked for;
    # a fixed TYR/ILE heading reads as those residues whatever was measured,
    # and the two isoforms here do not use the same names or numbers
    r0 = res[0]
    r1 = res[1] if len(res) > 1 else "-"
    rows = []

    if a.by_run:
        # one row per compound and receptor, one column per run: the point is
        # to see at a glance whether the runs agree, which a pooled count and
        # a stack of separate tables both hide
        def block(title, pick):
            log.append(title)
            log.append(f"  {'compound':10s}{'receptor':13s}"
                       + "".join(f"{r:>9s}" for r in runs))
            for cpd in cpds:
                for st in structs:
                    cells_here = [cell.get((cpd, st, r)) for r in runs]
                    if not any(cells_here):
                        continue
                    line = f"  {cpd:10s}{st:13s}"
                    for c in cells_here:
                        if c is None:
                            line += f"{'-':>9s}"
                        else:
                            line += f"{pick(c):>9s}"
                    log.append(line)
                log.append("")

        block("=== either residue, out of the poses of each run ===",
              lambda c: f"{c['either']}/{c['n_pose']}")
        for x in res:
            block(f"=== {x}, out of the poses of each run ===",
                  lambda c, x=x: f"{c['hb'][x]}/{c['n_pose']}")
        block("=== the top pose of each run: does it bond either residue? ===",
              lambda c: f"{c['top_either']}/{c['runs']}")

        for cpd in cpds:
            for st in structs:
                for r in runs:
                    c = cell.get((cpd, st, r))
                    if c is None:
                        continue
                    sv, aff = np.array(c["s"]), np.array(c["aff"])
                    sdv = lambda v: (np.std(v, ddof=1) if len(v) > 1 else 0.0)
                    rec = {"compound": cpd, "structure": st, "run": r,
                           "poses": c["n_pose"], "poses_hinge": c["either"],
                           "runs": c["runs"], "top_hinge": c["top_either"],
                           "score_mean": np.mean(sv), "score_sd": sdv(sv),
                           "affinity_mean": np.mean(aff),
                           "affinity_sd": sdv(aff)}
                    for x in res:
                        rec["poses_" + x] = c["hb"][x]
                        rec["top_" + x] = c["top_hb"][x]
                    rows.append(rec)
    else:
        log.append(f"  {'compound':9s}{'structure':11s}"
                   f"{'either':>8s}{r0:>9s}{r1:>9s}"
                   f"{'score':>8s}{'affinity':>10s}"
                   f"{'either':>10s}{r0:>10s}{r1:>10s}"
                   f"{'score':>15s}{'affinity':>16s}")
        log.append(f"  {'':9s}{'':11s}{'top':>8s}{'top':>9s}{'top':>9s}"
                   f"{'top':>8s}{'top':>10s}"
                   f"{'all poses':>10s}{'poses':>10s}{'poses':>10s}"
                   f"{'mean +- sd':>15s}{'mean +- sd':>16s}")
        for cpd in cpds:
            for st in structs:
                c = cell.get((cpd, st))
                if c is None:
                    log.append(f"  {cpd:9s}{st:11s}{'-':>8s}")
                    continue
                n, np_ = c["runs"], c["n_pose"]
                sv, aff = np.array(c["s"]), np.array(c["aff"])
                frac = lambda k, d: f"{k}/{d}"
                sd = lambda v: np.std(v, ddof=1) if len(v) > 1 else 0.0
                log.append(
                    f"  {cpd:9s}{st:11s}"
                    f"{frac(c['top_either'], n):>8s}"
                    f"{frac(c['top_hb'][res[0]], n):>9s}"
                    f"{frac(c['top_hb'][res[1]] if len(res) > 1 else 0, n):>9s}"
                    f"{np.mean(c['top_s']):8.3f}{np.mean(c['top_a']):10.2f}"
                    f"{frac(c['either'], np_):>10s}"
                    f"{frac(c['hb'][res[0]], np_):>10s}"
                    f"{frac(c['hb'][res[1]] if len(res) > 1 else 0, np_):>10s}"
                    f"{np.mean(sv):10.3f} +-{sd(sv):4.3f}"
                    f"{np.mean(aff):10.2f} +-{sd(aff):5.2f}")
                rec = {"compound": cpd, "structure": st, "runs": n,
                       "top_hinge": c["top_either"],
                       "top_score": np.mean(c["top_s"]),
                       "top_affinity": np.mean(c["top_a"]), "poses": np_,
                       "poses_hinge": c["either"],
                       "score_mean": np.mean(sv), "score_sd": sd(sv),
                       "affinity_mean": np.mean(aff), "affinity_sd": sd(aff)}
                for x in res:
                    rec["top_" + x] = c["top_hb"][x]
                    rec["poses_" + x] = c["hb"][x]
                rows.append(rec)
            log.append("")

    text = "\n".join(log)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(a.out + ".csv", index=False)
        with open(a.out + ".txt", "w") as f:
            f.write(text + "\n")
        print(f"[out] {a.out}.csv, {a.out}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

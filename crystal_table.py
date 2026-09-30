#!/usr/bin/env python3
"""
One row per compound and crystal structure: the top pose and what it touches.

For each structure the pose with the best score is taken and its affinity,
pose score and hydrogen bonds to the named residues reported. The pose is
chosen by the named property rather than by its position in the file, because
a run sorted by energy writes them in a different order than one sorted by the
CNN, and reading the first pose would then mean different things per run.

    python crystal_table.py \
        --structure 6HMD:crystal_str/6hmd:crystal_str/6hmd_prep.pdb \
        --structure 3OFM:crystal_str/3ofm:crystal_str/3ofm_prep.pdb \
        --residues TYR116,ILE117 --only VB004,EV043
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


def cid(path):
    b = os.path.basename(path)
    for e in (".sdf.gz", ".sdf"):
        if b.endswith(e):
            b = b[: -len(e)]
            break
    return re.sub(r"\.mol(_docked)?$|_docked$|_out$", "", b)


def prop(m, k):
    if not m.HasProp(k):
        return np.nan
    try:
        return float(m.GetProp(k))
    except ValueError:
        return np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--structure", action="append", required=True,
                   metavar="NAME:POSEDIR:RECEPTOR",
                   help="repeat for each crystal structure")
    p.add_argument("--residues", default="TYR116,ILE117",
                   help="residues to test for a hydrogen bond, as the "
                        "receptor names them")
    p.add_argument("--only", help="comma-separated compounds")
    p.add_argument("--select-by", default="CNNscore")
    p.add_argument("--props", default="CNNscore,CNNaffinity,minimizedAffinity")
    p.add_argument("--dist", type=float, default=3.5)
    p.add_argument("--h-dist", type=float, default=2.5)
    p.add_argument("--angle", type=float, default=120.0)
    p.add_argument("--antecedent-angle", type=float, default=90.0)
    p.add_argument("--out")
    a = p.parse_args()

    res = [x.strip().upper() for x in a.residues.split(",") if x.strip()]
    props = [x.strip() for x in a.props.split(",") if x.strip()]
    keep = ([x.strip() for x in a.only.split(",") if x.strip()]
            if a.only else None)

    structs = []
    for spec in a.structure:
        bits = spec.split(":")
        if len(bits) != 3:
            sys.exit(f"--structure wants NAME:POSEDIR:RECEPTOR, got {spec!r}")
        name, d, rec = (x.strip() for x in bits)
        d, rec = os.path.expanduser(d), os.path.expanduser(rec)
        if not os.path.isdir(d):
            sys.exit(f"not a directory: {d}")
        if not os.path.exists(rec):
            sys.exit(f"receptor not found: {rec}")
        structs.append((name, d, rec))

    log = [f"[in] {len(structs)} structures, best pose by {a.select_by}",
           f"     hydrogen bonds: D-A <= {a.dist} A, H...A <= {a.h_dist} A, "
           f"D-H...A >= {a.angle:g} deg, antecedent >= "
           f"{a.antecedent_angle:g} deg",
           f"     residues tested: {', '.join(res)}"]

    rows, notes = [], []
    for name, d, rec in structs:
        sites = polar_sites(rec)
        have = {s[0].upper() for s in sites}
        gone = [r for r in res if r not in have]
        if gone:
            notes.append(f"     [note] {name}: {', '.join(gone)} carries no "
                         f"polar site; a bond to it cannot be found")
        files = sorted(glob.glob(os.path.join(d, "*_docked.sdf"))) or \
            sorted(glob.glob(os.path.join(d, "*.sdf")))
        for f in files:
            c = cid(f)
            if keep and c not in keep:
                continue
            top, tv, npose = None, None, 0
            for m in Chem.SDMolSupplier(f, removeHs=False, sanitize=True):
                if m is None:
                    continue
                npose += 1
                v = prop(m, a.select_by)
                if np.isnan(v):
                    continue
                if tv is None or v > tv:
                    top, tv = m, v
            if top is None:
                notes.append(f"     [note] {name} {c}: no pose carried "
                             f"{a.select_by}")
                continue
            try:
                mh = Chem.AddHs(top, addCoords=True)
            except Exception:
                mh = top
            hits, _ = bonds_for_pose(mh, sites, a)
            up = {k.upper(): v for k, v in hits.items()}
            row = {"compound": c, "structure": name, "n_pose": npose,
                   "n_hbond": int(sum(hits.values())),
                   "residues": " ".join(sorted(hits))}
            for r in res:
                row[r] = int(up.get(r, 0))
            for k in props:
                row[k] = prop(top, k)
            rows.append(row)
    if not rows:
        sys.exit("no poses read")
    t = pd.DataFrame(rows)
    log += notes

    names = [n for n, _, _ in structs]
    order = (keep if keep else
             sorted(t["compound"].unique(),
                    key=lambda c: -t.loc[t.compound == c, res[0]].sum()))
    log.append("")
    log.append("=== top pose in each crystal structure ===")
    log.append(f"    hbond columns are the count to that residue in the top "
               f"pose; total is to every residue")
    hdr = f"  {'compound':9s}{'structure':10s}{'pose':>6s}"
    for k in props:
        hdr += f"{k[:11]:>12s}"
    for r in res:
        hdr += f"{r:>8s}"
    hdr += f"{'total':>7s}   residues bonded"
    log.append(hdr)
    for c in order:
        g = t[t.compound == c]
        if not len(g):
            continue
        for n in names:
            x = g[g.structure == n]
            if not len(x):
                log.append(f"  {c:9s}{n:10s}{'-':>6s}"
                           + "".join(f"{'-':>12s}" for _ in props)
                           + "".join(f"{'-':>8s}" for _ in res)
                           + f"{'-':>7s}   not docked")
                continue
            x = x.iloc[0]
            line = f"  {c:9s}{n:10s}{int(x['n_pose']):6d}"
            for k in props:
                line += ("         n/a" if np.isnan(x[k])
                         else f"{x[k]:12.3f}")
            for r in res:
                line += f"{int(x[r]):8d}"
            line += f"{int(x['n_hbond']):7d}   {x['residues']}"
            log.append(line)
        log.append("")

    log.append("=== summary over the structures ===")
    for r in res:
        g = t.groupby("compound")[r].apply(lambda s: int((s > 0).sum()))
        n_s = t.groupby("compound")["structure"].nunique()
        hit = [f"{c} {g[c]}/{n_s[c]}" for c in order if c in g and g[c] > 0]
        log.append(f"  {r}: bonded in at least one structure by "
                   f"{int((g > 0).sum())} of {len(g)} compounds"
                   + ("  (" + ", ".join(hit) + ")" if hit else ""))
    for k in props:
        gg = t.groupby("compound")[k].mean().sort_values(
            ascending="affin" in k.lower() and "cnn" not in k.lower())
        log.append(f"  {k}: best three on the mean over structures — "
                   + ", ".join(f"{c} {v:.3f}" for c, v in gg.head(3).items()))

    text = "\n".join(log)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        t.to_csv(a.out, index=False)
        with open(os.path.splitext(a.out)[0] + ".txt", "w") as f:
            f.write(text + "\n")
        print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

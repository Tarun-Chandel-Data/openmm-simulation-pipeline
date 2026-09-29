#!/usr/bin/env python3
"""
How much of a compound's hydrogen bond count survives changing the docking
seed, and whether pooling the top poses of each seed steadies it.

A single run reports one number per compound. Re-running with another seed
reports a different one, because the search is stochastic. This script reads a
set of runs that differ only in seed and reports, per compound:

    single  the count from each seed's best pose, and the spread across seeds
    pooled  the mean count over each seed's top --top-n poses, and the spread
            of those means across seeds

Pooling cannot remove the noise; it averages over it. Whether that is worth
doing is the comparison printed at the end: if the pooled spread is no smaller
than the single-pose spread, the poses within a seed disagree as much as the
seeds do, and pooling has bought nothing.

The spread is what a difference between two compounds has to exceed before it
can be read as chemistry rather than search noise, so it is reported in the
same units as the count.

    python seed_topn_stability.py \
        --receptor a1_isoform=a1.pdb --receptor a2_proteinA=a2.pdb \
        --poses 'validate/poses/*.sdf' --top-n 8 \
        --select-by CNNscore --match VB --out seed_stability
"""
import argparse, glob, os, re, sys
from collections import defaultdict

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


def parse_name(path, order):
    """The three name fields, returned as (compound, receptor, replicate).

    A run that varies the seed writes COMPOUND__RECEPTOR__SEED, where the
    middle field names the receptor and the last the replicate. An ensemble
    run writes COMPOUND__CONFORMER__SUBUNIT, where the middle field is the
    replicate and the last names which ensemble it belongs to. Both are three
    fields in the same shape, so which is which has to be stated rather than
    guessed: a wrong guess groups every conformer as its own receptor and
    every subunit as a replicate of it."""
    b = os.path.basename(path)
    for e in (".sdf.gz", ".sdf"):
        if b.endswith(e):
            b = b[: -len(e)]
            break
    parts = b.split("__")
    if len(parts) < 3:
        return None
    parts = parts[:2] + ["__".join(parts[2:])]
    got = dict(zip(order, parts))
    return got["compound"], got["receptor"], got["replicate"]


def prop(mol, name):
    if mol.HasProp(name):
        try:
            return float(mol.GetProp(name))
        except ValueError:
            return None
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--receptor", action="append",
                   metavar="LABEL=PATH",
                   help="a single receptor structure per label, where every "
                        "replicate was docked into the same one. Repeat for "
                        "each label")
    p.add_argument("--ensemble", action="append",
                   metavar="LABEL=DIR",
                   help="use instead of --receptor when each replicate has "
                        "its own structure: the directory of conformer pdb "
                        "files. The replicate field names the file, so its "
                        "own structure is used to find the polar sites rather "
                        "than one shared reference")
    p.add_argument("--fields", default="compound,receptor,replicate",
                   help="what the three parts of COMPOUND__X__Y mean, in "
                        "order. A seed run is compound,receptor,replicate; an "
                        "ensemble run writing COMPOUND__CONFORMER__SUBUNIT is "
                        "compound,replicate,receptor")
    p.add_argument("--poses", required=True,
                   help="sdf files, a directory of them, or a glob")
    p.add_argument("--top-n", type=int, default=8,
                   help="poses kept per seed, ranked by --select-by")
    p.add_argument("--select-by", default="CNNscore",
                   help="pose property ranked on. CNNscore is the pose score; "
                        "CNNaffinity is a predicted affinity and ranks a "
                        "different thing")
    p.add_argument("--lower-is-better", action="store_true",
                   help="set when --select-by is an energy, where the smallest "
                        "value is the best pose")
    p.add_argument("--props",
                   default="CNNscore,CNNaffinity,minimizedAffinity",
                   help="pose properties carried through to the table besides "
                        "the one ranked on, so a later step can report them "
                        "without re-reading the poses. Missing ones are "
                        "named and left empty rather than failing")
    p.add_argument("--match",
                   help="keep only compounds whose id contains this")
    p.add_argument("--dist", type=float, default=3.5)
    p.add_argument("--h-dist", type=float, default=2.5)
    p.add_argument("--angle", type=float, default=120.0)
    p.add_argument("--antecedent-angle", type=float, default=90.0)
    p.add_argument("--out", default="seed_stability")
    a = p.parse_args()

    order = [x.strip() for x in a.fields.split(",") if x.strip()]
    if sorted(order) != ["compound", "receptor", "replicate"]:
        sys.exit("--fields must name compound, receptor and replicate exactly "
                 f"once each; got {a.fields!r}")
    if bool(a.receptor) == bool(a.ensemble):
        sys.exit("give either --receptor (one structure per label) or "
                 "--ensemble (one per replicate), not both and not neither")

    recs, ensdirs = {}, {}
    for spec in (a.receptor or []):
        if "=" not in spec:
            sys.exit(f"--receptor wants LABEL=PATH, got '{spec}'")
        lab, path = spec.split("=", 1)
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            sys.exit(f"receptor not found: {path}")
        recs[lab] = path
    for spec in (a.ensemble or []):
        if "=" not in spec:
            sys.exit(f"--ensemble wants LABEL=DIR, got '{spec}'")
        lab, d = spec.split("=", 1)
        d = os.path.expanduser(d)
        if not os.path.isdir(d):
            sys.exit(f"ensemble directory not found: {d}")
        ensdirs[lab] = d

    pat = os.path.expanduser(a.poses)
    if os.path.isdir(pat):
        files = sorted(glob.glob(os.path.join(pat, "*.sdf")))
    else:
        files = sorted(glob.glob(pat))
    if not files:
        sys.exit(f"no pose files matched {a.poses}")

    # the polar sites are read from whichever structure that pose was docked
    # into. Reading them from one reference instead would place the sites at
    # the reference's coordinates while the pose sits in the conformer's
    sites = {lab: polar_sites(path) for lab, path in recs.items()}
    conf_sites = {}

    def norm(x):
        """A name with the separators taken out.

        A pose written as EV001__a1c00__a1 names the conformer a1c00 while the
        file holding it is a1_c00.pdb. Matching on the literal string finds
        nothing and skips every pose, so the comparison is made on names with
        the punctuation removed."""
        return "".join(ch for ch in x.lower() if ch.isalnum())

    ens_index = {}
    for lab, d in ensdirs.items():
        ix = {}
        for e in (".pdb", ".pdbqt"):
            for p in glob.glob(os.path.join(d, "*" + e)):
                ix.setdefault(norm(os.path.basename(p)[: -len(e)]), p)
        ens_index[lab] = ix

    def sites_for(lab, replicate):
        if lab in sites:
            return sites[lab]
        if lab not in ens_index:
            return None
        key = (lab, replicate)
        if key not in conf_sites:
            hit = ens_index[lab].get(norm(replicate))
            conf_sites[key] = polar_sites(hit) if hit else None
        return conf_sites[key]
    log = [f"[in] {len(files)} pose files from {a.poses}",
           f"     name fields: " + ", ".join(order),
           ("     receptors: " + ", ".join(f"{k} = {os.path.basename(v)}"
                                           for k, v in recs.items()))
           if recs else
           ("     ensembles: " + ", ".join(
               f"{k} = {len(glob.glob(os.path.join(v, '*.pdb')))} "
               f"conformers in {v}"
               for k, v in ensdirs.items())),
           f"     top {a.top_n} poses per seed, ranked by {a.select_by}"
           + (" (lower is better)" if a.lower_is_better else ""),
           f"     geometry: D-A <= {a.dist} A, H...A <= {a.h_dist} A, "
           f"D-H...A >= {a.angle} deg, antecedent >= {a.antecedent_angle} deg"]

    want_props = [x.strip() for x in a.props.split(",") if x.strip()]
    rows = []
    absent = set()
    skipped_rec, skipped_name, thin = defaultdict(int), 0, []
    for f in files:
        got = parse_name(f, order)
        if got is None:
            skipped_name += 1
            continue
        cpd, rec, seed = got
        if a.match and a.match not in cpd:
            continue
        rsites = sites_for(rec, seed)
        if rsites is None:
            skipped_rec[rec] += 1
            continue
        poses = []
        for mol in Chem.SDMolSupplier(f, removeHs=False, sanitize=True):
            if mol is None:
                continue
            v = prop(mol, a.select_by)
            if v is None:
                continue
            poses.append((v, mol))
        for w in want_props:
            if poses and not poses[0][1].HasProp(w):
                absent.add(w)
        if not poses:
            continue
        poses.sort(key=lambda t: t[0], reverse=not a.lower_is_better)
        # a file with fewer poses than --top-n contributes a shorter pool, so
        # its mean rests on less; it is named rather than silently mixed in
        if len(poses) < a.top_n:
            thin.append((os.path.basename(f), len(poses)))
        sel = poses[: a.top_n]
        for k, (v, mol) in enumerate(sel, 1):
            try:
                mh = Chem.AddHs(mol, addCoords=True)
            except Exception:
                mh = mol
            hits, _ = bonds_for_pose(mh, rsites, a)
            row = {"compound": cpd, "receptor": rec, "seed": seed,
                   "pose_rank": k, a.select_by: v,
                   "n_hbond": int(sum(hits.values())),
                   "n_residue": len(hits),
                   "residues": ";".join(sorted(hits))}
            for w in want_props:
                if w != a.select_by:
                    row[w] = prop(mol, w)
            rows.append(row)

    if not rows:
        sys.exit("no poses selected; check --select-by names a property the "
                 "files carry, and --receptor labels match the middle field")
    d = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    d.to_csv(a.out + "_poses.csv", index=False)

    if skipped_name:
        log.append(f"     [note] {skipped_name} files not named "
                   f"COMPOUND__RECEPTOR__SEED, skipped")
    for r, n in sorted(skipped_rec.items()):
        if ensdirs:
            have = sorted({k for ix in ens_index.values() for k in ix})[:4]
            log.append(f"     [note] {n} files name '{r}', for which no "
                       f"structure was found. Looked for a file matching "
                       f"'{norm(r)}' after removing separators; the "
                       f"directories hold e.g. {', '.join(have)}")
        else:
            log.append(f"     [note] {n} files name receptor '{r}', which was "
                       f"not given with --receptor; skipped")
    if thin:
        log.append(f"     [note] {len(thin)} files hold fewer than {a.top_n} "
                   f"scored poses, e.g. " +
                   ", ".join(f"{n} ({k})" for n, k in thin[:3]))
    if absent:
        log.append(f"     [note] these properties are not on the poses and "
                   f"are empty in the table: {', '.join(sorted(absent))}")
    carried = [w for w in want_props if w not in absent]
    if carried:
        log.append(f"     carried through: {', '.join(carried)}")

    seeds = sorted(d["seed"].unique())
    log.append(f"     {d['compound'].nunique()} compounds, "
               f"{len(seeds)} seeds ({', '.join(seeds)}), "
               f"{len(d)} poses scored")

    # per compound, per receptor, per seed: the best pose's count and the mean
    # over the kept pool
    g = d.sort_values("pose_rank").groupby(["compound", "receptor", "seed"])
    per_seed = g.agg(single=("n_hbond", "first"),
                     pooled=("n_hbond", "mean"),
                     n_pose=("n_hbond", "size")).reset_index()

    def spread(s):
        return float(s.max() - s.min()) if len(s) > 1 else np.nan

    per_cpd = per_seed.groupby(["compound", "receptor"]).agg(
        seeds=("seed", "nunique"),
        single_med=("single", "median"), single_spread=("single", spread),
        pooled_mean=("pooled", "mean"), pooled_spread=("pooled", spread),
    ).reset_index()
    per_cpd.to_csv(a.out + "_per_compound.csv", index=False)

    one_seed = per_cpd[per_cpd["seeds"] < 2]
    ok = per_cpd[per_cpd["seeds"] >= 2]
    if len(one_seed):
        log.append(f"     [note] {len(one_seed)} compound-receptor pairs have "
                   f"only one seed, so no spread can be formed for them")
    if not len(ok):
        sys.exit("no compound has two or more seeds; nothing to compare")

    log.append("")
    log.append("=== how far the count moves when only the seed changes ===")
    for label, col in (("best pose alone", "single_spread"),
                       (f"mean of top {a.top_n}", "pooled_spread")):
        v = ok[col].dropna()
        changed = int((v > 0).sum())
        log.append(f"  {label:18s} spread across seeds: median {v.median():.2f}"
                   f", mean {v.mean():.2f}, max {v.max():.2f}")
        log.append(f"  {'':18s} {changed} of {len(v)} "
                   f"({100*changed/len(v):.0f}%) change at all")
    s1 = ok["single_spread"].dropna()
    s2 = ok["pooled_spread"].dropna()
    log.append("")
    if s2.median() < s1.median():
        log.append(f"  -> pooling the top {a.top_n} narrows the median spread "
                   f"from {s1.median():.2f} to {s2.median():.2f} bonds")
    elif s2.median() > s1.median():
        log.append(f"  -> pooling the top {a.top_n} WIDENS the median spread, "
                   f"{s1.median():.2f} to {s2.median():.2f}. The poses within "
                   f"a seed disagree more than the seeds do, so the pool is "
                   f"averaging over binding modes, not over noise")
    else:
        log.append(f"  -> pooling the top {a.top_n} leaves the median spread "
                   f"unchanged at {s1.median():.2f} bonds")
    log.append(f"  a difference between two compounds smaller than "
               f"{s2.median():.2f} bonds cannot be told from seed noise by "
               f"this measurement")

    # the same question asked of the score being ranked on, since a pose set
    # can be stable in count while the score it was chosen by is not
    sc = d.sort_values("pose_rank").groupby(
        ["compound", "receptor", "seed"])[a.select_by].first().reset_index()
    ssp = sc.groupby(["compound", "receptor"])[a.select_by].agg(spread).dropna()
    if len(ssp):
        log.append("")
        log.append(f"=== the same for {a.select_by} of the best pose ===")
        log.append(f"  spread across seeds: median {ssp.median():.4f}, "
                   f"max {ssp.max():.4f}")

    log.append("")
    log.append("=== per compound ===")
    hdr = (f"  {'compound':12s}{'receptor':16s}{'seeds':>6s}"
           f"{'single':>8s}{'spread':>8s}{'pooled':>8s}{'spread':>8s}")
    log.append(hdr)
    for _, r in ok.sort_values(["compound", "receptor"]).iterrows():
        log.append(f"  {r['compound']:12s}{r['receptor']:16s}"
                   f"{int(r['seeds']):6d}"
                   f"{r['single_med']:8.1f}{r['single_spread']:8.1f}"
                   f"{r['pooled_mean']:8.2f}{r['pooled_spread']:8.2f}")

    text = "\n".join(log)
    print(text)
    with open(a.out + "_values.txt", "w") as fh:
        fh.write(text + "\n")
    print(f"\n[out] {a.out}_poses.csv")
    print(f"      {a.out}_per_compound.csv")
    print(f"      {a.out}_values.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

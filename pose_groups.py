#!/usr/bin/env python3
"""
Which compounds bind the same way, and is the answer bigger than its error?

Each compound in each receptor is represented by the pose its seeds agreed
on, not the pose the scoring ranked first. Within one receptor those poses
share a frame, so comparing two compounds is a subtraction. Across
receptors they do not, so the receptors are superposed first and the
quality of that fit is reported: it is the floor every cross-receptor
distance has to clear, and a grouping that does not clear it is a statement
about the superposition.

Compounds differ in their atoms, so the comparison is over the maximum
common substructure, and every symmetry mapping of it is tried.

Groups are formed by single linkage: two compounds join when their poses
are within the cut, and a group is whatever that relation connects. The cut
is reported beside the groups, since it is a choice and the groups are only
as meaningful as it is defensible.

    python pose_groups.py --poses crystal_str/ligand/results \\
        --receptors crystal_str/receptor --cut 2.0 --out groups
"""
import argparse, glob, itertools, os, sys
import numpy as np

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFMCS
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("needs rdkit")

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
try:
    from mmgbsa_ensemble import consensus_pose
except ImportError:
    sys.exit("needs mmgbsa_ensemble.py beside this script")


def perceive(mol):
    """Rings and aromaticity, so a scaffold written as aromatic SMARTS matches.

    The poses are read without sanitising, because a docked pose that a
    strict read would reject is still a pose and its coordinates are still
    wanted. The cost is that ring and aromatic flags are not set, and a
    substructure search against an aromatic pattern then matches nothing at
    all - silently, as an absence rather than an error. The flags are set
    here, as far as the molecule allows.
    """
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:                                         # noqa: BLE001
        pass
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
        Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_RDKIT)
    except Exception:                                         # noqa: BLE001
        pass
    return mol


def parse_name(path):
    b = os.path.basename(path)
    for e in (".sdf.gz", ".sdf"):
        if b.endswith(e):
            b = b[: -len(e)]
            break
    p = b.split("__")
    return (p[0], p[1], p[2]) if len(p) >= 3 else None


def ca_atoms(path):
    """{(chain, resseq): xyz} for the alpha carbons of a pdb."""
    out = {}
    try:
        with open(path) as f:
            for line in f:
                if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                    continue
                try:
                    out[(line[21], line[22:26].strip())] = np.array(
                        [float(line[30:38]), float(line[38:46]),
                         float(line[46:54])])
                except ValueError:
                    continue
    except OSError:
        return None
    return out


def kabsch(P, Q):
    """Rotation and translation taking P onto Q, with the fit it achieves."""
    cp, cq = P.mean(0), Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    fit = float(np.sqrt((((P - cp) @ R.T - (Q - cq)) ** 2).sum(1).mean()))
    return R, cp, cq, fit


def core_rms(p1, p2):
    return float(np.sqrt(((p1 - p2) ** 2).sum(1).mean()))


def core_coords(mol, core):
    """Core atom coordinates, every symmetry mapping, so the smallest wins."""
    ms = mol.GetSubstructMatches(core, uniquify=False, maxMatches=200)
    if not ms:
        return None
    c = mol.GetConformer()
    return [np.array([list(c.GetAtomPosition(i)) for i in m]) for m in ms]


def best_rms(a, b):
    if a is None or b is None:
        return np.nan
    return min(core_rms(x, y) for x in a for y in b
               if len(x) == len(y)) if a and b else np.nan


def groups_from(dist, names, cut):
    """Single linkage: joined when within the cut, grouped by what connects."""
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in itertools.combinations(names, 2):
        v = dist.get((i, j), dist.get((j, i)))
        if v is not None and v == v and v <= cut:
            a, b = find(i), find(j)
            if a != b:
                parent[a] = b
    out = {}
    for n in names:
        out.setdefault(find(n), []).append(n)
    return sorted(out.values(), key=lambda g: (-len(g), g[0]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--poses", required=True)
    p.add_argument("--receptors", required=True)
    p.add_argument("--only", help="compounds, in order")
    p.add_argument("--cut", type=float, default=2.0,
                   help="how close two compounds' poses must be to join a "
                        "group, in angstroms over the shared scaffold")
    p.add_argument("--select-by", default="CNNscore")
    p.add_argument("--mcs-timeout", type=int, default=60)
    p.add_argument("--out")
    a = p.parse_args()

    keep = ([x.strip() for x in a.only.split(",") if x.strip()]
            if a.only else None)
    dirs = sorted(d for d in glob.glob(os.path.join(a.poses, "*"))
                  if os.path.isdir(d))
    if not dirs:
        sys.exit(f"no structure directories under {a.poses}")

    cells, bad = {}, []
    for d in dirs:
        sn = os.path.basename(d)
        files = {}
        for f in sorted(glob.glob(os.path.join(d, "*.sdf"))):
            nm = parse_name(f)
            if nm is None:
                continue
            if keep and nm[0] not in keep:
                continue
            files.setdefault(nm[0], []).append(f)
        for cpd, fs in files.items():
            mol, info = consensus_pose(fs, a.select_by)
            if mol is None:
                bad.append(f"{cpd}/{sn}: {info}")
                continue
            cells[(cpd, sn)] = (perceive(mol), info)
    if not cells:
        sys.exit("no consensus pose could be formed")

    cpds = keep if keep else sorted({k[0] for k in cells})
    structs = sorted({k[1] for k in cells})

    reps = {}
    for c in cpds:
        for s in structs:
            if (c, s) in cells:
                reps[c] = cells[(c, s)][0]
                break
    res = rdFMCS.FindMCS([Chem.RemoveHs(reps[c]) for c in cpds if c in reps],
                         ringMatchesRingOnly=True, completeRingsOnly=True,
                         timeout=a.mcs_timeout)
    core = Chem.MolFromSmarts(res.smartsString)
    if core is None or core.GetNumAtoms() == 0:
        sys.exit("no shared scaffold found across the compounds")

    log = [f"[in] {len(cells)} consensus poses, {len(cpds)} compounds x "
           f"{len(structs)} receptors",
           f"     each is the pose its seeds agreed on, not the top-scored "
           f"one",
           f"     shared scaffold: {core.GetNumAtoms()} atoms, "
           f"{res.smartsString}",
           f"     grouped by single linkage at {a.cut} A"]
    if bad:
        log.append(f"     [note] {len(bad)} cells had no consensus pose")

    # within one receptor the poses share a frame, so nothing is superposed
    nomatch = [f"{c}/{s}" for (c, s), (m, _) in cells.items()
               if not m.GetSubstructMatches(core)]
    if nomatch:
        log.append(f"     [warn] the scaffold matches nothing in "
                   f"{len(nomatch)} poses, which leaves them out of every "
                   f"comparison: " + ", ".join(nomatch[:5])
                   + (" ..." if len(nomatch) > 5 else ""))

    log.append("")
    log.append("=== how the compounds group, receptor by receptor ===")
    log.append("    poses in one receptor share a frame, so these distances "
               "carry no superposition error at all")
    per_struct = {}
    for s in structs:
        here = [c for c in cpds if (c, s) in cells]
        cc = {c: core_coords(cells[(c, s)][0], core) for c in here}
        dist = {}
        for i, j in itertools.combinations(here, 2):
            dist[(i, j)] = best_rms(cc[i], cc[j])
        gs = groups_from(dist, here, a.cut)
        per_struct[s] = gs
        vals = [v for v in dist.values() if v == v]
        log.append(f"  {s:13s}{len(gs)} group(s)"
                   + (f", pairwise {min(vals):.2f} to {max(vals):.2f} A"
                      if vals else ""))
        for g in gs:
            log.append(f"      {'+'.join(g)}")

    # across receptors the frames differ, so the receptors are fitted first
    log.append("")
    log.append("=== across receptors: the fit, and what it costs ===")
    log.append("    the receptors are superposed on the alpha carbons they "
               "share; that fit is the floor a cross-receptor distance has "
               "to clear before it means anything")
    cas = {}
    for s in structs:
        f = os.path.join(a.receptors, s + ".pdb")
        if not os.path.exists(f):
            hits = glob.glob(os.path.join(a.receptors, s + "*.pdb"))
            if not hits:
                log.append(f"  [warn] no receptor file for {s}")
                continue
            f = hits[0]
        c = ca_atoms(f)
        if c:
            cas[s] = c
    ref = structs[0] if structs[0] in cas else (sorted(cas)[0] if cas else None)
    fits, xf = {}, {}
    if ref:
        for s in sorted(cas):
            shared = sorted(set(cas[ref]) & set(cas[s]))
            if len(shared) < 20:
                log.append(f"  [warn] {s} shares only {len(shared)} alpha "
                           f"carbons with {ref}")
                continue
            P = np.array([cas[s][k] for k in shared])
            Q = np.array([cas[ref][k] for k in shared])
            R, cp, cq, fit = kabsch(P, Q)
            fits[s], xf[s] = fit, (R, cp, cq)
            log.append(f"  {s:13s}{len(shared):5d} shared alpha carbons, "
                       f"fit {fit:.2f} A")
    floor = max(fits.values()) if fits else float("nan")
    if fits:
        log.append(f"  the worst fit is {floor:.2f} A: two poses in different "
                   f"receptors cannot be told apart below that")

    log.append("")
    log.append("=== the same compound across receptors, once fitted ===")
    log.append(f"  {'compound':10s}{'pairs':>7s}{'mean':>8s}{'max':>8s}"
               f"   reading")
    for c in cpds:
        here = [s for s in structs if (c, s) in cells and s in xf]
        vals = []
        for s1, s2 in itertools.combinations(here, 2):
            a1 = core_coords(cells[(c, s1)][0], core)
            a2 = core_coords(cells[(c, s2)][0], core)
            if a1 is None or a2 is None:
                continue
            R1, cp1, cq1 = xf[s1]
            R2, cp2, cq2 = xf[s2]
            m1 = [(x - cp1) @ R1.T + cq1 for x in a1]
            m2 = [(x - cp2) @ R2.T + cq2 for x in a2]
            v = best_rms(m1, m2)
            if v == v:
                vals.append(v)
        if not vals:
            continue
        note = ("one binding mode in every receptor" if max(vals) <= a.cut
                else "more than one binding mode across the receptors")
        log.append(f"  {c:10s}{len(vals):7d}{np.mean(vals):8.2f}"
                   f"{max(vals):8.2f}   {note}")

    log.append("")
    log.append("=== do the groupings agree between receptors? ===")
    pair_together = {}
    for s, gs in per_struct.items():
        for g in gs:
            for i, j in itertools.combinations(sorted(g), 2):
                pair_together.setdefault((i, j), []).append(s)
    n_s = len(per_struct)
    always = [k for k, v in pair_together.items() if len(v) == n_s]
    some = [k for k, v in pair_together.items() if 0 < len(v) < n_s]
    log.append(f"  {len(always)} compound pairs group together in all "
               f"{n_s} receptors")
    for i, j in sorted(always):
        log.append(f"      {i} + {j}")
    log.append(f"  {len(some)} pairs group together in some receptors and "
               f"not others")
    if not always:
        log.append("  no pair groups together everywhere, so there is no "
                   "grouping to carry forward: the receptor decides it")

    text = "\n".join(log)
    print(text)
    if a.out:
        with open(a.out + ".txt", "w") as f:
            f.write(text + "\n")
        print(f"\n[out] {a.out}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Does a compound bind the same way in each crystal structure?

The three crystal structures are deposited in their own coordinate frames, so
two poses cannot be compared until the receptors that produced them are put in
a common frame. The alpha carbons shared by a pair of structures are matched by
chain and residue number, superposed, and the resulting rotation applied to the
poses of the second structure. Only then is the ligand RMSD measured, and it is
measured WITHOUT superposing the two poses on each other, because the question
is whether the ligand sits in the same place, not whether it has the same
shape.

The protein's own residual RMSD after superposition is printed alongside as the
yardstick: a ligand that moves no more than the pocket did has not really
moved. Superposing on the pocket rather than the whole chain is the default,
since a ligand's placement is set by the pocket and a distant loop that differs
between the two crystals would otherwise drag the frame.

The whole-molecule number answers whether the pose is the same; the core-only
number answers whether the scaffold is placed the same while the substituents
differ, which is the more forgiving and often the more informative reading.

    python crystal_pose_rmsd.py \
        --structure 6HMD:crystal_str/6hmd:crystal_str/6hmd_prep.pdb \
        --structure 6HMQ:crystal_str/6hmq:crystal_str/6hmq_prep.pdb \
        --structure 3OFM:crystal_str/3ofm:crystal_str/3ofm_prep.pdb \
        --core 'N#Cc1ccc2c(c1)nc(N)c1ccncc12'
"""
import argparse, glob, itertools, os, re, sys
import numpy as np

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolAlign
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("needs rdkit")

# the docking box centre from the run these poses came from; the pocket is
# taken as the residues whose alpha carbon lies within --pocket-radius of it
BOX = (-1.073, -12.323, 6.080)


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


def ca_map(path):
    """Alpha carbons keyed by chain and residue number, so two structures with
    different residue coverage still match up."""
    out = {}
    with open(path) as f:
        for ln in f:
            if ln[:6] in ("ATOM  ", "HETATM") and ln[12:16].strip() == "CA":
                try:
                    key = (ln[21], int(ln[22:26]))
                except ValueError:
                    continue
                out.setdefault(key, np.array(
                    [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]))
    return out


def kabsch(P, Q):
    """Rotation and translation taking P onto Q."""
    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - R @ pc


def moved(mol, R, t):
    """A copy of the pose with the transform applied."""
    m = Chem.Mol(mol)
    c = m.GetConformer()
    for i in range(m.GetNumAtoms()):
        p = np.array(list(c.GetAtomPosition(i)))
        q = R @ p + t
        c.SetAtomPosition(i, q.tolist())
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--structure", action="append", required=True,
                   metavar="NAME:POSEDIR:RECEPTOR")
    p.add_argument("--select-by", default="CNNscore",
                   help="pose property the best pose is chosen on; an "
                        "affinity in kcal/mol is taken at its lowest")
    p.add_argument("--only", help="comma-separated compounds")
    p.add_argument("--core", default="N#Cc1ccc2c(c1)nc(N)c1ccncc12",
                   help="SMARTS for the shared scaffold; '' to skip")
    p.add_argument("--align", choices=("pocket", "all"), default="pocket",
                   help="alpha carbons the receptors are superposed on")
    p.add_argument("--pocket-center", default=",".join(f"{v:g}" for v in BOX))
    p.add_argument("--pocket-radius", type=float, default=12.0)
    p.add_argument("--width", type=int, default=9)
    p.add_argument("--diagnose", action="store_true",
                   help="for each pair also report how many core atoms "
                        "matched and how far the core's centroid moved, which "
                        "separates a scaffold sitting somewhere else from one "
                        "rotated in place")
    p.add_argument("--out")
    a = p.parse_args()

    keep = ([x.strip() for x in a.only.split(",") if x.strip()]
            if a.only else None)
    sl = a.select_by.lower()
    low_best = "cnn" not in sl and any(
        m in sl for m in ("affinity", "vina", "energy", "rmsd"))
    sign = -1.0 if low_best else 1.0
    ctr = np.array([float(x) for x in a.pocket_center.split(",")])

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
    if len(structs) < 2:
        sys.exit("needs at least two structures to compare")

    core = None
    if a.core:
        core = Chem.MolFromSmarts(a.core)
        if core is None:
            sys.exit(f"--core is not a valid SMARTS: {a.core!r}")

    log = [f"[in] {len(structs)} structures, best pose by {a.select_by} "
           f"({'lowest' if low_best else 'highest'})",
           f"     receptors superposed on the {a.align} alpha carbons"
           + (f" (within {a.pocket_radius:g} A of "
              f"{', '.join(f'{v:g}' for v in ctr)})"
              if a.align == "pocket" else ""),
           f"     ligand RMSD is symmetry aware and taken in place, not "
           f"after superposing the poses on each other"]
    if core is not None:
        log.append(f"     core: {a.core}  ({core.GetNumAtoms()} atoms)")

    # best pose per compound per structure, and the receptor alpha carbons
    best, cas, notes = {}, {}, []
    for name, d, rec in structs:
        cas[name] = ca_map(rec)
        files = sorted(glob.glob(os.path.join(d, "*_docked.sdf"))) or \
            sorted(glob.glob(os.path.join(d, "*.sdf")))
        for f in files:
            c = cid(f)
            if keep and c not in keep:
                continue
            top, tv = None, None
            for m in Chem.SDMolSupplier(f, removeHs=False, sanitize=True):
                if m is None:
                    continue
                v = prop(m, a.select_by)
                if np.isnan(v):
                    continue
                if tv is None or sign * v > sign * tv:
                    top, tv = m, v
            if top is None:
                notes.append(f"     [note] {name} {c}: no pose carried "
                             f"{a.select_by}")
                continue
            best.setdefault(c, {})[name] = top

    names = [n for n, _, _ in structs]
    pairs = list(itertools.combinations(names, 2))

    # one transform per pair, taking the second structure's frame onto the first
    xform, prot = {}, {}
    # The pocket is chosen once, in the reference structure, because the box
    # centre is a point in that structure's frame and means nothing in
    # another's. Selecting it afresh per pair asked whether the second
    # structure happened to share the reference's frame, and a structure
    # deposited elsewhere then matched no residue at all.
    ref = names[0]
    pocket_keys = None
    if a.align == "pocket":
        pocket_keys = {k for k, v in cas[ref].items()
                       if np.linalg.norm(v - ctr) <= a.pocket_radius}
        log.append(f"     pocket taken from {ref}: {len(pocket_keys)} "
                   f"alpha carbons")
        if not pocket_keys:
            sys.exit(f"no alpha carbon of {ref} lies within "
                     f"{a.pocket_radius:g} A of {a.pocket_center}; the centre "
                     f"is not in this structure's frame")

    for x, y in pairs:
        shared = sorted(set(cas[x]) & set(cas[y]))
        if pocket_keys is not None:
            shared = [k for k in shared if k in pocket_keys]
        if len(shared) < 3:
            notes.append(f"     [note] {x} vs {y}: only {len(shared)} shared "
                         f"alpha carbons; cannot superpose")
            xform[(x, y)] = None
            continue
        P = np.array([cas[y][k] for k in shared])
        Q = np.array([cas[x][k] for k in shared])
        before = float(np.sqrt(((P - Q) ** 2).sum(axis=1).mean()))
        R, t = kabsch(P, Q)
        P2 = (R @ P.T).T + t
        after = float(np.sqrt(((P2 - Q) ** 2).sum(axis=1).mean()))
        xform[(x, y)] = (R, t)
        prot[(x, y)] = (before, after, len(shared))

    log += notes
    log.append("")
    log.append("=== how far the best pose moves between crystal structures ==="
               )
    log.append("    each cell is the in-place RMSD in A between the two "
               "structures' best poses")
    log.append("    'all' is the whole molecule, 'core' the shared scaffold "
               "only")
    w = a.width
    heads = [f"{x}>{y}" for x, y in pairs]
    log.append(f"  {'compound':10s}{'metric':7s}"
               + "".join(f"{h:>{w+2}s}" for h in heads)
               + f"{'mean':>{w}s}{'max':>{w}s}")

    def rms(m1, m2, use_core):
        if not use_core:
            try:
                return float(rdMolAlign.CalcRMS(m1, m2))
            except Exception:
                return np.nan
        # a symmetric core matches its own atoms in more than one way, and
        # taking one arbitrary match in each molecule can pair the two rings
        # of a scaffold the wrong way round and report a flip where there is
        # none. Every mapping is tried and the smallest distance kept, which
        # is what makes this a symmetry-aware core RMSD rather than a
        # labelling artefact.
        ms1 = m1.GetSubstructMatches(core, uniquify=False, maxMatches=500)
        ms2 = m2.GetSubstructMatches(core, uniquify=False, maxMatches=500)
        if not ms1 or not ms2:
            return np.nan
        c1, c2 = m1.GetConformer(), m2.GetConformer()
        i1 = ms1[0]
        p1 = np.array([list(c1.GetAtomPosition(i)) for i in i1])
        best = np.inf
        for i2 in ms2:
            if len(i2) != len(i1):
                continue
            p2 = np.array([list(c2.GetAtomPosition(i)) for i in i2])
            best = min(best, float(np.sqrt(
                ((p1 - p2) ** 2).sum(axis=1).mean())))
        return np.nan if best is np.inf or not np.isfinite(best) else best

    order = keep if keep else sorted(best)
    rows = []
    for c in order:
        if c not in best:
            continue
        got = False
        for use_core, tag in ((False, "all"),) + \
                (((True, "core"),) if core is not None else ()):
            vals = []
            for x, y in pairs:
                m1, m2 = best[c].get(x), best[c].get(y)
                tf = xform.get((x, y))
                if m1 is None or m2 is None or tf is None:
                    vals.append(np.nan)
                    continue
                vals.append(rms(m1, moved(m2, *tf), use_core))
            good = [v for v in vals if not np.isnan(v)]
            line = f"  {c if not got else '':10s}{tag:7s}"
            for v in vals:
                line += (f"{'-':>{w+2}s}" if np.isnan(v)
                         else f"{v:{w+2}.2f}")
            line += (f"{'-':>{w}s}{'-':>{w}s}" if not good else
                     f"{np.mean(good):{w}.2f}{max(good):{w}.2f}")
            log.append(line)
            rows.append({"compound": c, "metric": tag,
                         **{f"{x}>{y}": v for (x, y), v in zip(pairs, vals)},
                         "mean": np.mean(good) if good else np.nan,
                         "max": max(good) if good else np.nan})
            got = True
        log.append("")

    line = f"  {a.align:10s}{'CA':7s}"
    for x, y in pairs:
        pr = prot.get((x, y))
        line += (f"{'-':>{w+2}s}" if pr is None else f"{pr[1]:{w+2}.2f}")
    log.append(line)
    log.append(f"    the row above is the receptors' own residual RMSD after "
               f"superposition, the yardstick: a ligand moving no further "
               f"than this has not moved")
    for x, y in pairs:
        pr = prot.get((x, y))
        if pr:
            log.append(f"    {x} vs {y}: {pr[2]} shared alpha carbons, "
                       f"{pr[0]:.2f} A apart as deposited, {pr[1]:.2f} A after "
                       f"superposing"
                       + ("   (already in a common frame)"
                          if abs(pr[0] - pr[1]) < 0.05 else ""))

    if a.diagnose and core is not None:
        log.append("")
        log.append("=== core diagnostic ===")
        log.append("    atoms is how many of the core matched; shift is how "
                   "far its centroid moved")
        log.append("    a small shift with a large core RMSD means the "
                   "scaffold turned in place rather than moved")
        log.append(f"  {'compound':10s}{'pair':14s}{'atoms':>7s}"
                   f"{'shift':>8s}{'core':>8s}{'all':>8s}")
        for c in order:
            if c not in best:
                continue
            for x, y in pairs:
                m1, m2 = best[c].get(x), best[c].get(y)
                tf = xform.get((x, y))
                if m1 is None or m2 is None or tf is None:
                    continue
                m2 = moved(m2, *tf)
                i1 = m1.GetSubstructMatch(core)
                i2 = m2.GetSubstructMatch(core)
                if not i1 or not i2:
                    log.append(f"  {c:10s}{x + '>' + y:14s}"
                               f"{'0':>7s}{'-':>8s}{'-':>8s}{'-':>8s}"
                               f"   core does not match")
                    continue
                c1, c2 = m1.GetConformer(), m2.GetConformer()
                p1 = np.array([list(c1.GetAtomPosition(i)) for i in i1])
                p2 = np.array([list(c2.GetAtomPosition(i)) for i in i2])
                shift = float(np.linalg.norm(p1.mean(0) - p2.mean(0)))
                log.append(f"  {c:10s}{x + '>' + y:14s}{len(i1):7d}"
                           f"{shift:8.2f}{rms(m1, m2, True):8.2f}"
                           f"{rms(m1, m2, False):8.2f}")

    if rows:
        log.append("")
        log.append("=== summary ===")
        for tag in (["all"] + (["core"] if core is not None else [])):
            g = [(r["compound"], r["mean"]) for r in rows
                 if r["metric"] == tag and not np.isnan(r["mean"])]
            if not g:
                continue
            g.sort(key=lambda kv: kv[1])
            log.append(f"  {tag}: most consistent across the structures — "
                       + ", ".join(f"{c} {v:.2f}" for c, v in g[:3])
                       + "  |  least — "
                       + ", ".join(f"{c} {v:.2f}" for c, v in g[-3:]))

    text = "\n".join(log)
    print(text)
    if a.out:
        try:
            import pandas as pd
        except ImportError:
            sys.exit("--out needs pandas")
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
        with open(os.path.splitext(a.out)[0] + ".txt", "w") as f:
            f.write(text + "\n")
        print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

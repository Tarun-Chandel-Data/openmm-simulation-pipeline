#!/usr/bin/env python3
"""
Is a ligand halogen actually sitting on a named residue, and has the ligand
moved while it did?

A per-residue energy decomposition says how much a residue contributed, not
why. Two explanations give the same number: the substituent genuinely sits on
that residue in one subunit and not the other, or the whole ligand adopted a
different pose in the two trajectories and took every contact with it. This
measures both at once, so the second can be ruled out rather than assumed.

For each halogen on the ligand it reports, over the trajectory:

  - the distance to the nearest polar atom of the target residue, and how
    often that distance is within the contact cutoff
  - the C-X...A angle, because a halogen bond is a straight-on approach along
    the C-X axis (>= 150 deg) while a sideways contact at the same distance is
    an ordinary charge-dipole one. The two mean different chemistry and a
    distance alone cannot tell them apart
  - the ligand's own heavy-atom RMSD in the same frames, after superposing on
    the protein backbone, and the contact occupancy split by it. If the
    contact is present only in the frames where the ligand has not moved, the
    decomposition was reading a pose change

Nothing here depends on the force field's treatment of the halogen, which is
the point: a fixed-charge model has no sigma hole, so an energy term cannot
test a halogen bond, but a geometry can.

    python halogen_contact_md.py --top nopbc.prmtop --traj nopbc.dcd \\
        --ligand LIG --target LYS69 --resid-offset 6 --label "EV043 CK2a'"
"""
import argparse, os, sys
import numpy as np

try:
    import mdtraj as md
except ImportError:
    sys.exit("needs mdtraj")

HALOGEN = {"Cl", "Br", "I", "F"}


def pick_residue(top, spec, offset):
    """The residue a canonical name and number refers to in this topology."""
    m = "".join(c for c in spec if c.isalpha()).upper()
    n = "".join(c for c in spec if c.isdigit())
    if not n:
        sys.exit(f"--target wants a name and number, e.g. LYS69, got {spec!r}")
    want = int(n) - offset
    hits = [r for r in top.residues if r.resSeq == want
            and (not m or r.name.upper().startswith(m[:3]))]
    if not hits:
        near = sorted({r.name + str(r.resSeq + offset) for r in top.residues
                       if abs(r.resSeq - want) <= 2})
        sys.exit(f"no residue {spec} (file numbering {want}) in the topology."
                 + (f" Nearby: {', '.join(near)}" if near else ""))
    return hits[0]


def bonded_carbon(top, idx, xyz0):
    """The carbon a halogen hangs off, from the topology if it has bonds and
    from the geometry of the first frame if it does not."""
    for b in top.bonds:
        a1, a2 = b[0], b[1]
        if a1.index == idx and a2.element.symbol == "C":
            return a2.index
        if a2.index == idx and a1.element.symbol == "C":
            return a1.index
    cs = [a.index for a in top.atoms if a.element.symbol == "C"]
    if not cs:
        return None
    d = np.linalg.norm(xyz0[cs] - xyz0[idx], axis=1)
    j = int(np.argmin(d))
    return cs[j] if d[j] <= 0.22 else None


def angle_at(b, a, c):
    """Angle a-b-c per frame, in degrees, with b the vertex."""
    v1, v2 = a - b, c - b
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    ok = (n1 > 0) & (n2 > 0)
    out = np.full(len(b), np.nan)
    cos = (v1[ok] * v2[ok]).sum(axis=1) / (n1[ok] * n2[ok])
    out[ok] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top", required=True)
    p.add_argument("--traj", required=True)
    p.add_argument("--ligand", default="LIG")
    p.add_argument("--target",
                   help="residue in canonical numbering, e.g. LYS69. Use "
                        "--target-index instead where two residues of the "
                        "same kind sit close together and the numbering is "
                        "not certain")
    p.add_argument("--target-index", type=int,
                   help="the residue's own number in this topology, with no "
                        "offset applied. This names one residue and nothing "
                        "else, so it settles which of two nearby lysines is "
                        "meant")
    p.add_argument("--list-residues", metavar="FIRST-LAST",
                   help="print the residues in this range of the topology's "
                        "own numbering and stop, to find the one you want")
    p.add_argument("--resid-offset", type=int, default=0,
                   help="canonical number minus the number in this file")
    p.add_argument("--per-target-atom", action="store_true",
                   help="one row per target atom instead of the nearest of "
                        "them. An alpha carbon says where the residue is and "
                        "a side chain tip says what it touches, and the two "
                        "answer different questions, so a single nearest "
                        "distance over both would mix them")
    p.add_argument("--target-atoms", default="",
                   help="atom names on the target, e.g. NZ. Default is every "
                        "nitrogen and oxygen it carries")
    p.add_argument("--halogens", default="auto",
                   help="'auto' for every Cl, Br, I and F on the ligand, or "
                        "explicit atom names")
    p.add_argument("--cutoff", type=float, default=0.45,
                   help="contact cutoff in nm")
    p.add_argument("--angle", type=float, default=150.0,
                   help="C-X...A angle at or above which the approach counts "
                        "as along the C-X axis")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--label", default="")
    p.add_argument("--out")
    a = p.parse_args()

    t = md.load(a.traj, top=a.top, stride=a.stride)
    top = t.topology
    log = [f"[in] {a.label or a.traj}",
           f"     {t.n_frames} frames, {t.n_atoms} atoms",
           f"     contact within {a.cutoff * 10:.1f} A, along the C-X axis at "
           f"{a.angle:g} deg or more"]

    if a.list_residues:
        try:
            lo, hi = (int(x) for x in a.list_residues.split("-"))
        except ValueError:
            sys.exit("--list-residues wants FIRST-LAST, e.g. 55-75")
        print(f"residues {lo}-{hi} of {a.top}, in this file's own numbering")
        for r in top.residues:
            if lo <= r.resSeq <= hi:
                print(f"  {r.resSeq:5d}  {r.name}")
        return 0
    if not a.target and a.target_index is None:
        sys.exit("give --target or --target-index")

    lig = [r for r in top.residues if r.name.upper() == a.ligand.upper()]
    if not lig:
        sys.exit(f"no residue named {a.ligand} in the topology")
    lig = lig[0]
    if a.target_index is not None:
        hits = [r for r in top.residues if r.resSeq == a.target_index]
        if not hits:
            sys.exit(f"this topology has no residue {a.target_index}")
        tgt = hits[0]
    else:
        tgt = pick_residue(top, a.target, a.resid_offset)
    log.append(f"     ligand {lig.name}{lig.resSeq}, target {tgt.name}"
               f"{tgt.resSeq} in this file"
               + (f" ({tgt.name}{tgt.resSeq + a.resid_offset} with the "
                  f"offset applied)" if a.resid_offset else ""))
    # name the neighbours, so a residue of the same kind next door is visible
    near = [f"{r.name}{r.resSeq}" for r in top.residues
            if 0 < abs(r.resSeq - tgt.resSeq) <= 4]
    log.append(f"     neighbours in this file: {', '.join(near)}")

    if a.target_atoms:
        want = {x.strip().upper() for x in a.target_atoms.split(",")}
        tat = [at.index for at in tgt.atoms if at.name.upper() in want]
    else:
        tat = [at.index for at in tgt.atoms
               if at.element.symbol in ("N", "O")]
    if not tat:
        sys.exit(f"the target residue carries no atom to measure to")
    log.append(f"     measuring to {', '.join(top.atom(i).name for i in tat)}")

    if a.halogens == "auto":
        hal = [at.index for at in lig.atoms if at.element.symbol in HALOGEN]
    else:
        want = {x.strip().upper() for x in a.halogens.split(",")}
        hal = [at.index for at in lig.atoms if at.name.upper() in want]
    if not hal:
        sys.exit(f"the ligand carries no halogen; found "
                 f"{sorted({at.element.symbol for at in lig.atoms})}")

    # the ligand's own movement, so a lost contact can be told from a lost pose
    bb = top.select("protein and backbone")
    lh = [at.index for at in lig.atoms if at.element.symbol != "H"]
    rmsd = None
    if len(bb) >= 3:
        t2 = t[:]
        t2.superpose(t2, frame=0, atom_indices=bb)
        ref = t2.xyz[0][lh]
        rmsd = np.sqrt(((t2.xyz[:, lh, :] - ref) ** 2).sum(axis=2).mean(axis=1))
        rmsd = rmsd * 10.0
    else:
        log.append("     [note] too little protein backbone to superpose on; "
                   "the ligand RMSD column is left out")

    log.append("")
    tname = f"{tgt.name}{tgt.resSeq}"
    log.append(f"=== halogen contact with {tname} ===")
    log.append(f"  {'atom':7s}{'mean':>8s}{'min':>8s}{'within':>9s}"
               f"{'angle':>8s}{'on axis':>9s}   nearest target atom")
    log.append(f"  {'':7s}{'A':>8s}{'A':>8s}{'%':>9s}{'deg':>8s}{'%':>9s}")
    rows = []
    sets = ([[j] for j in tat] if a.per_target_atom else [tat])
    for h, tsel in ((h, ts) for h in hal for ts in sets):
        tat_use = tsel
        pairs = np.array([[h, j] for j in tat_use])
        d = md.compute_distances(t, pairs)          # frames x targets, nm
        near = np.argmin(d, axis=1)
        dmin = d[np.arange(len(d)), near] * 10.0    # A
        within = float((dmin <= a.cutoff * 10.0).mean() * 100.0)
        c = bonded_carbon(top, h, t.xyz[0])
        _ = tat_use
        if c is None:
            ang = np.full(len(dmin), np.nan)
        else:
            ang = angle_at(t.xyz[:, h, :], t.xyz[:, c, :],
                           t.xyz[np.arange(t.n_frames),
                                 [tat_use[k] for k in near]])
        good = ang[~np.isnan(ang)]
        onax = float((good >= a.angle).mean() * 100.0) if len(good) else np.nan
        # which target atom it is closest to most often
        vals, cnt = np.unique(near, return_counts=True)
        who = top.atom(tat_use[vals[int(np.argmax(cnt))]]).name
        name = top.atom(h).name + ("-" + who if a.per_target_atom else "")
        log.append(f"  {name:7s}{dmin.mean():8.2f}{dmin.min():8.2f}"
                   f"{within:9.1f}"
                   + (f"{'n/a':>8s}" if not len(good)
                      else f"{good.mean():8.1f}")
                   + (f"{'n/a':>9s}" if np.isnan(onax) else f"{onax:9.1f}")
                   + f"   {who}")
        rows.append((name, dmin, ang, within, onax))

    if rmsd is not None:
        log.append("")
        log.append("=== is the contact following the pose? ===")
        log.append(f"    the ligand's heavy-atom RMSD from frame 0, after "
                   f"superposing on the protein backbone")
        log.append(f"    ligand RMSD: mean {rmsd.mean():.2f} A, "
                   f"max {rmsd.max():.2f} A")
        # split at the median rather than into terciles: a ligand that flips
        # between two modes gives a bimodal RMSD, and terciles then put every
        # frame in the middle band and show nothing
        med = float(np.median(rmsd))
        bands = [("ligand nearer frame 0 (<= %.2f A)" % med, rmsd <= med),
                 ("ligand further       (>  %.2f A)" % med, rmsd > med)]
        log.append(f"  {'atom':7s}{'band':30s}{'frames':>8s}"
                   f"{'mean d':>9s}{'within':>9s}")
        for name, dmin, ang, _, _ in rows:
            for bl, sel in bands:
                if not sel.any():
                    continue
                log.append(f"  {name:7s}{bl:30s}{int(sel.sum()):8d}"
                           f"{dmin[sel].mean():9.2f}"
                           f"{(dmin[sel] <= a.cutoff * 10).mean() * 100:9.1f}")
            r = np.corrcoef(rmsd, dmin)[0, 1]
            log.append(f"  {'':7s}{'correlation with ligand RMSD':30s}"
                       f"{'':8s}{r:9.2f}")
            log.append(f"    a strong positive correlation means the contact "
                       f"is lost when the ligand moves, so the decomposition "
                       f"was reading the pose")
            log.append("")

    text = "\n".join(log)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out + ".txt", "w") as f:
            f.write(text + "\n")
        import pandas as pd
        d = {"frame": np.arange(len(rows[0][1])) * a.stride}
        if rmsd is not None:
            d["ligand_rmsd"] = rmsd
        for name, dmin, ang, _, _ in rows:
            d[f"{name}_dist"] = dmin
            d[f"{name}_angle"] = ang
        pd.DataFrame(d).to_csv(a.out + ".csv", index=False)
        print(f"\n[out] {a.out}.txt, {a.out}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

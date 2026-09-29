#!/usr/bin/env python3
"""
Hydrogen bonds between a ligand pose and a receptor, with an angular test.

A distance-only criterion counts any two polar heavy atoms that happen to sit
close, including pairs pointing away from one another. A hydrogen bond is
directional, so geometry is tested here:

  ligand donates, with an explicit hydrogen
      H...A <= --h-dist  and  D-H...A >= --angle
  otherwise, heavy atoms only
      D...A <= --dist  and both antecedent angles >= --antecedent-angle
      (the atom each polar atom is bonded to must not lie between them)

Ligand hydrogens are added by RDKit with coordinates, so donors on the ligand
side are explicit. Receptor hydrogens are used when the PDB carries them and
inferred geometrically when it does not: the antecedent angle is the test that
does not need them.

One pose per compound per receptor: the best by --select-by, not a pool over
every pose, so the count describes a single binding mode.

    python hbond_geometry.py --receptor a2.pdb --poses dock_paired/poses \
        --tag a2 --out hb_a2.csv
"""
import argparse, glob, os, re, sys
import numpy as np

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("needs rdkit")
try:
    import pandas as pd
except ImportError:
    sys.exit("needs pandas")

# (residue, atom) -> (antecedent atom, role). Proline's backbone nitrogen is
# tertiary and carries no hydrogen, so it is not a donor.
BB = {"N": ("CA", "D"), "O": ("C", "A"), "OXT": ("C", "A")}
SC = {
    "ARG": {"NE": ("CD", "D"), "NH1": ("CZ", "D"), "NH2": ("CZ", "D")},
    "ASN": {"ND2": ("CG", "D"), "OD1": ("CG", "A")},
    "GLN": {"NE2": ("CD", "D"), "OE1": ("CD", "A")},
    "HIS": {"ND1": ("CG", "B"), "NE2": ("CD2", "B")},
    "LYS": {"NZ": ("CE", "D")},
    "SER": {"OG": ("CB", "B")},
    "THR": {"OG1": ("CB", "B")},
    "TYR": {"OH": ("CZ", "B")},
    "TRP": {"NE1": ("CD1", "D")},
    "ASP": {"OD1": ("CG", "A"), "OD2": ("CG", "A")},
    "GLU": {"OE1": ("CD", "A"), "OE2": ("CD", "A")},
}


def angle(a, b, c):
    """Angle a-b-c in degrees."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(n1*n2), -1, 1))))


def polar_sites(path):
    """(label, atom name, xyz, antecedent xyz, role) for every polar receptor
    atom whose antecedent is present."""
    atoms, res_of = {}, {}
    for L in open(path):
        if not L.startswith(("ATOM", "HETATM")):
            continue
        name, res = L[12:16].strip(), L[17:20].strip()
        num, chain = int(L[22:26]), L[21]
        atoms[(chain, num, name)] = np.array(
            [float(L[30:38]), float(L[38:46]), float(L[46:54])])
        res_of[(chain, num)] = res
    sites = []
    for (chain, num, name), xyz in atoms.items():
        res = res_of[(chain, num)]
        spec = None
        if name in BB:
            if not (name == "N" and res == "PRO"):
                spec = BB[name]
        elif res in SC and name in SC[res]:
            spec = SC[res][name]
        if spec is None:
            continue
        ante, role = spec
        axyz = atoms.get((chain, num, ante))
        if axyz is None:
            continue
        sites.append((f"{res}{num}", name, xyz, axyz, role))
    return sites


def ligand_sites(mol):
    """Donors (with their hydrogens) and acceptors on the ligand."""
    conf = mol.GetConformer()
    pos = lambda i: np.array(conf.GetAtomPosition(i))
    don, acc = [], []
    for at in mol.GetAtoms():
        if at.GetSymbol() not in ("N", "O"):
            continue
        i = at.GetIdx()
        hs = [n.GetIdx() for n in at.GetNeighbors() if n.GetSymbol() == "H"]
        heavy = [n.GetIdx() for n in at.GetNeighbors() if n.GetSymbol() != "H"]
        ante = pos(heavy[0]) if heavy else None
        if hs:
            don.append((i, pos(i), [pos(h) for h in hs], ante))
        # an sp3 oxygen or a nitrogen that is not a protonated amine can accept
        if at.GetSymbol() == "O" or at.GetTotalNumHs() == 0:
            if at.GetFormalCharge() <= 0:
                acc.append((i, pos(i), ante))
    return don, acc


def bonds_for_pose(mol, sites, a):
    """Residues hydrogen bonded to this pose, and how many bonds were made."""
    don, acc = ligand_sites(mol)
    hits, detail = {}, []
    for label, name, sxyz, saxyz, role in sites:
        # receptor acceptor  <-  ligand donor
        if role in ("A", "B"):
            for _, dxyz, hxyzs, dante in don:
                if np.linalg.norm(dxyz - sxyz) > a.dist:
                    continue
                ok = False
                for h in hxyzs:
                    if np.linalg.norm(h - sxyz) <= a.h_dist and \
                            angle(dxyz, h, sxyz) >= a.angle:
                        ok = True
                        break
                if ok and angle(saxyz, sxyz, dxyz) >= a.antecedent_angle:
                    hits[label] = hits.get(label, 0) + 1
                    detail.append((label, name, "lig-donor",
                                   float(np.linalg.norm(dxyz - sxyz))))
                    break
        # receptor donor  ->  ligand acceptor
        if role in ("D", "B"):
            for _, axyz, aante in acc:
                d = float(np.linalg.norm(axyz - sxyz))
                if d > a.dist:
                    continue
                if angle(saxyz, sxyz, axyz) < a.antecedent_angle:
                    continue
                if aante is not None and \
                        angle(aante, axyz, sxyz) < a.antecedent_angle:
                    continue
                hits[label] = hits.get(label, 0) + 1
                detail.append((label, name, "rec-donor", d))
                break
    return hits, detail


def compound_id(mol, path):
    """The compound this file belongs to.

    Takes the molecule title when there is one. Otherwise the file name, with
    a trailing _docked, _out or _poses removed, since those name the step
    rather than the compound; and where that leaves nothing distinctive, the
    directory holding the file, which is how ensemble runs identify it."""
    if mol is not None and mol.HasProp("_Name") and mol.GetProp("_Name").strip():
        return mol.GetProp("_Name").strip()
    base = os.path.basename(path)
    for ext in (".sdf.gz", ".sdf"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    base = base.split("__")[0]
    base = re.sub(r"_(docked|out|poses|result|min|best)$", "", base,
                  flags=re.I)
    if base:
        return base
    return os.path.basename(os.path.dirname(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--receptor", required=True)
    p.add_argument("--poses", required=True,
                   help="an sdf file, a directory of them, or a glob pattern. "
                        "A pattern lets the poses sit in nested directories, "
                        "as ensemble runs usually write them: one directory "
                        "per receptor conformer, one per compound inside it")
    p.add_argument("--select-by", default="CNNaffinity",
                   help="pose property to maximise when choosing the one pose "
                        "per compound")
    p.add_argument("--dist", type=float, default=3.5,
                   help="heavy-atom donor-acceptor cutoff")
    p.add_argument("--h-dist", type=float, default=2.5,
                   help="H...acceptor cutoff where the donor hydrogen is known")
    p.add_argument("--angle", type=float, default=120.0,
                   help="minimum D-H...A angle")
    p.add_argument("--antecedent-angle", type=float, default=90.0,
                   help="minimum angle at a polar atom between its bonded "
                        "neighbour and its partner; rejects pairs that are "
                        "close but pointing away from each other")
    p.add_argument("--props", default="all",
                   help="pose properties to copy out of the selected pose, or "
                        "'all' to copy every property the pose carries, which "
                        "avoids having to know what the docking wrote. "
                        "Taking them from the same pose the bonds were counted "
                        "on keeps every quantity describing one binding mode, "
                        "rather than joining to a table built from a different "
                        "pose or a different run")
    p.add_argument("--match",
                   help="only read pose files whose name contains this. A pose "
                        "directory usually holds one file per compound PER "
                        "RECEPTOR, and scoring a pose docked into one receptor "
                        "against the other is meaningless, so the receptor's "
                        "own files must be selected")
    p.add_argument("--tag", default="a1")
    p.add_argument("--out", default="hbond_geometry.csv")
    a = p.parse_args()

    sites = polar_sites(a.receptor)
    print(f"[in] {len(sites)} polar receptor atoms with antecedents "
          f"from {a.receptor}")

    if any(ch in a.poses for ch in "*?["):
        files = sorted(glob.glob(os.path.expanduser(a.poses), recursive=True))
    elif a.poses.endswith(".sdf"):
        files = [a.poses]
    else:
        files = sorted(glob.glob(os.path.join(os.path.expanduser(a.poses),
                                              "*.sdf")))
    n_all = len(files)
    if a.match:
        files = [f for f in files if a.match in os.path.basename(f)]
    if not files:
        sys.exit(f"no sdf found at {a.poses}"
                 + (f" matching '{a.match}'" if a.match else ""))
    print(f"[in] {len(files)} pose file(s)"
          + (f" of {n_all} matching '{a.match}'" if a.match else ""))

    rows, n_noprop = [], 0
    for f in files:
        supp = Chem.SDMolSupplier(f, removeHs=False)
        best, best_v = None, None
        for m in supp:
            if m is None:
                continue
            v = None
            if m.HasProp(a.select_by):
                try:
                    v = float(m.GetProp(a.select_by))
                except ValueError:
                    v = None
            if v is None:
                n_noprop += 1
                v = -1e9
            if best_v is None or v > best_v:
                best, best_v = m, v
        if best is None:
            continue
        try:
            mh = Chem.AddHs(best, addCoords=True)
        except Exception:
            mh = best
        hits, detail = bonds_for_pose(mh, sites, a)
        cid = compound_id(best, f)
        rec = {
            "cpd_id": cid,
            f"{a.tag}_n_hbond": int(sum(hits.values())),
            f"{a.tag}_hbond_res": " ".join(sorted(hits)),
            f"{a.tag}_n_poses_seen": sum(1 for _ in Chem.SDMolSupplier(f)),
        }
        if a.props.strip().lower() == "all":
            want = [x for x in best.GetPropNames() if not x.startswith("_")]
        else:
            want = [x.strip() for x in a.props.split(",") if x.strip()]
        for prop in want:
            v = None
            if best.HasProp(prop):
                try:
                    v = float(best.GetProp(prop))
                except ValueError:
                    v = best.GetProp(prop)
            rec[f"{a.tag}_{prop}"] = v
        rows.append(rec)

    # A compound may appear in several files: repeat dockings under
    # different seeds, or, if --match was omitted, both receptors. The single
    # best pose across them is kept, which is the same rule as within one
    # file, and the spread over the repeats is reported beside it because a
    # count that moves between seeds is telling you how reproducible it is.
    d0 = pd.DataFrame(rows)
    nh = f"{a.tag}_n_hbond"
    if len(d0) and d0["cpd_id"].duplicated().any():
        per = d0.groupby("cpd_id")[nh]
        sel = f"{a.tag}_{a.select_by}"
        order = sel if sel in d0.columns else nh
        keep = (d0.sort_values(order, ascending=False)
                  .drop_duplicates("cpd_id").set_index("cpd_id"))
        keep[f"{a.tag}_n_runs"] = per.size()
        keep[f"{a.tag}_n_hbond_min"] = per.min()
        keep[f"{a.tag}_n_hbond_max"] = per.max()
        n_runs = int(per.size().median())
        spread = (per.max() - per.min())
        print(f"[in] {len(d0)} file(s) for {d0['cpd_id'].nunique()} compounds "
              f"(median {n_runs} per compound)")
        print(f"     keeping the best pose per compound by {order}")
        print(f"     hydrogen-bond count varies across repeats for "
              f"{int((spread > 0).sum())} of {len(spread)} compounds; "
              f"median spread {spread.median():g}, max {spread.max():g}")
        rows = keep.reset_index().to_dict("records")

    if n_noprop:
        print(f"[warn] {n_noprop} pose(s) carried no '{a.select_by}' property; "
              f"for those the best pose could not be chosen on it")
    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)
    n = d[f"{a.tag}_n_hbond"]
    print(f"[out] {a.out}   {len(d)} compounds")
    print(f"      hydrogen bonds per compound: min {n.min()}, "
          f"median {n.median():g}, max {n.max()}")
    print(f"      criterion: D...A <= {a.dist} A, H...A <= {a.h_dist} A, "
          f"D-H...A >= {a.angle} deg, antecedent >= {a.antecedent_angle} deg")


if __name__ == "__main__":
    sys.exit(main())

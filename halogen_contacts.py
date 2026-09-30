#!/usr/bin/env python3
"""
Which residue a named atom sits against, across an ensemble.

For each conformer the highest-scoring pose is taken and, for every atom of the
chosen element, the nearest protein residue is found. The table counts how many
conformers put that atom against each residue, so a substituent resting in one
place shows a single large count and one with no fixed environment shows its
count spread thin.

The distance is to the nearest heavy atom of the residue, and the mean of those
distances is given beside the count: a residue reached in every conformer at
5 A is not in contact with anything, it is merely the closest thing.

    python halogen_contacts.py --poses 'ens_cx_dock/poses/*.sdf' \
        --ensemble ens_cx_a2 --receptor a2 --only VB004,EV043 \
        --element Cl --offset 6
"""
import argparse, glob, os, sys
from collections import defaultdict
import numpy as np

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("needs rdkit")


def parse_name(path, order):
    b = os.path.basename(path)
    for e in (".sdf.gz", ".sdf"):
        if b.endswith(e):
            b = b[: -len(e)]
            break
    parts = b.split("__")
    if len(parts) < 3:
        return None
    parts = parts[:2] + ["__".join(parts[2:])]
    g = dict(zip(order, parts))
    return g["compound"], g["receptor"], g["replicate"]


def protein(path, offset):
    """Heavy atoms with their residue label, hydrogens left out."""
    xyz, lab = [], []
    with open(path) as f:
        for ln in f:
            if ln[:6] != "ATOM  ":
                continue
            if ln[76:78].strip() == "H" or ln[12:16].strip().startswith("H"):
                continue
            xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
            lab.append(f"{ln[17:20].strip().title()}"
                       f"{int(ln[22:26]) + offset}")
    return np.asarray(xyz), np.asarray(lab)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--poses", required=True)
    p.add_argument("--ensemble", required=True,
                   help="directory of conformer pdb files")
    p.add_argument("--receptor", default="a2")
    p.add_argument("--fields", default="compound,replicate,receptor")
    p.add_argument("--select-by", default="CNNscore")
    p.add_argument("--only", required=True)
    p.add_argument("--element", default="Cl")
    p.add_argument("--offset", type=int, default=0,
                   help="added to the residue numbers when labelling")
    p.add_argument("--top", type=int, default=8,
                   help="residues shown per atom, most frequent first")
    p.add_argument("--out")
    a = p.parse_args()

    order = [x.strip() for x in a.fields.split(",")]
    want = [x.strip() for x in a.only.split(",") if x.strip()]
    files = sorted(glob.glob(os.path.expanduser(a.poses)))
    if not files:
        sys.exit(f"no files matched {a.poses}")

    idx = {}
    for pth in glob.glob(os.path.join(os.path.expanduser(a.ensemble), "*.pdb")):
        k = "".join(c for c in os.path.basename(pth)[:-4].lower() if c.isalnum())
        idx[k] = pth
    if not idx:
        sys.exit(f"no pdb files in {a.ensemble}")

    best = {}
    for f in files:
        got = parse_name(f, order)
        if not got:
            continue
        cpd, rec, rep = got
        if rec != a.receptor or cpd not in want:
            continue
        top, tv = None, None
        for m in Chem.SDMolSupplier(f, removeHs=False, sanitize=True):
            if m is None or not m.HasProp(a.select_by):
                continue
            try:
                v = float(m.GetProp(a.select_by))
            except ValueError:
                continue
            if tv is None or v > tv:
                top, tv = m, v
        if top is not None:
            best.setdefault(cpd, {})[rep] = top
    miss = [c for c in want if c not in best]
    if miss:
        sys.exit(f"no poses for: {', '.join(miss)}")

    cache = {}
    rows, notes = {}, []
    for cpd in want:
        per = defaultdict(list)          # atom index -> (residue, distance)
        n_used = 0
        for rep, mol in sorted(best[cpd].items()):
            k = "".join(c for c in rep.lower() if c.isalnum())
            if k not in idx:
                continue
            if k not in cache:
                cache[k] = protein(idx[k], a.offset)
            px, pl = cache[k]
            conf = mol.GetConformer()
            hits = [at.GetIdx() for at in mol.GetAtoms()
                    if at.GetSymbol() == a.element]
            if not hits:
                continue
            n_used += 1
            for rank, ai in enumerate(hits, 1):
                q = np.array(conf.GetAtomPosition(ai))
                d = np.linalg.norm(px - q, axis=1)
                j = int(np.argmin(d))
                per[rank].append((str(pl[j]), float(d[j])))
        if not per:
            notes.append(f"     [note] {cpd} has no {a.element} atom")
            continue
        rows[cpd] = (per, n_used)

    log = [f"[in] {a.poses}",
           f"     receptor {a.receptor}, best pose by {a.select_by}, "
           f"element {a.element}"]
    log += notes
    if a.offset:
        log.append(f"     residue numbers printed with an offset of "
                   f"{a.offset:+d}")
    log.append("")
    log.append(f"=== residue nearest each {a.element}, counted over the "
               f"conformers ===")
    log.append(f"    count is conformers out of those used; the distance "
               f"beside it is the mean over those conformers")
    log.append(f"  {'compound':10s}{a.element:>4s}{'n':>5s}   "
               + "   ".join(f"{'residue  n  dist':>18s}" for _ in range(1))
               + "  (most frequent first)")
    for cpd, (per, n_used) in rows.items():
        for rank, lst in sorted(per.items()):
            tally = defaultdict(list)
            for r, d in lst:
                tally[r].append(d)
            top = sorted(tally.items(), key=lambda kv: (-len(kv[1]),
                                                        np.mean(kv[1])))
            cells = "  ".join(f"{r:>8s} {len(v):2d} {np.mean(v):4.1f}"
                              for r, v in top[:a.top])
            log.append(f"  {cpd:10s}{rank:4d}{n_used:5d}   {cells}")

    log.append("")
    for cpd, (per, n_used) in rows.items():
        for rank, lst in sorted(per.items()):
            tally = defaultdict(list)
            for r, d in lst:
                tally[r].append(d)
            top = sorted(tally.items(), key=lambda kv: -len(kv[1]))[0]
            frac = len(top[1]) / max(n_used, 1)
            log.append(f"  {cpd} {a.element}{rank}: nearest {top[0]} in "
                       f"{len(top[1])} of {n_used} conformers "
                       f"({100*frac:.0f}%), mean {np.mean(top[1]):.1f} A, "
                       f"{len(tally)} different residues in all")
    log.append(f"  a substituent resting in one place shows one large count; "
               f"one with no fixed environment spreads thin")

    text = "\n".join(log)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(text + "\n")
        print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
    p.add_argument("--cutoff", type=float, default=5.0,
                   help="a residue counts as present when one of its heavy "
                        "atoms is within this of the atom. 0 falls back to "
                        "the single nearest residue")
    p.add_argument("--per-conformer", action="store_true",
                   help="also list, conformer by conformer, the residues "
                        "found")
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
                if a.cutoff > 0:
                    # every residue with a heavy atom inside the shell, each
                    # at its own closest approach, so one residue contributes
                    # one distance rather than one per atom
                    m = d <= a.cutoff
                    near = {}
                    for lab, dist in zip(pl[m], d[m]):
                        lab = str(lab)
                        if lab not in near or dist < near[lab]:
                            near[lab] = float(dist)
                    per[rank].append((rep, sorted(near.items(),
                                                  key=lambda kv: kv[1])))
                else:
                    j = int(np.argmin(d))
                    per[rank].append((rep, [(str(pl[j]), float(d[j]))]))
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
    log.append(f"=== residues within {a.cutoff:g} A of each {a.element}, "
               f"counted over the conformers ===" if a.cutoff > 0 else
               f"=== residue nearest each {a.element} ===")
    log.append(f"    count is conformers out of those used; the distance "
               f"beside it is the mean closest approach over those conformers")
    log.append(f"  {'compound':10s}{a.element:>4s}{'n':>5s}   "
               f"residues, each with the conformers it appears in and its "
               f"mean closest approach")
    for cpd, (per, n_used) in rows.items():
        for rank, lst in sorted(per.items()):
            tally = defaultdict(list)
            for _, pairs in lst:
                for r, d in pairs:
                    tally[r].append(d)
            top = sorted(tally.items(), key=lambda kv: (-len(kv[1]),
                                                        np.mean(kv[1])))
            cells = "  ".join(f"{r:>8s} {len(v):2d} {np.mean(v):4.1f}"
                              for r, v in top[:a.top])
            log.append(f"  {cpd:10s}{rank:4d}{n_used:5d}   {cells}")
            if len(top) > a.top:
                log.append(f"  {'':19s}   ... {len(top) - a.top} more residue"
                           f"(s) seen in fewer conformers")

    if a.per_conformer:
        log.append("")
        log.append(f"=== the residues within {a.cutoff:g} A, conformer by "
                   f"conformer ===")
        for cpd, (per, n_used) in rows.items():
            for rank, lst in sorted(per.items()):
                log.append("")
                log.append(f"  {cpd} {a.element}{rank}")
                for rep, pairs in lst:
                    short = rep.split("_")[-1] if "_" in rep else rep
                    log.append(f"    {short:6s} "
                               + ", ".join(f"{r} {d:.1f}" for r, d in pairs)
                               if pairs else f"    {short:6s} none")

    log.append("")
    for cpd, (per, n_used) in rows.items():
        for rank, lst in sorted(per.items()):
            tally = defaultdict(list)
            per_conf = []
            for _, pairs in lst:
                per_conf.append(len(pairs))
                for r, d in pairs:
                    tally[r].append(d)
            if not tally:
                log.append(f"  {cpd} {a.element}{rank}: nothing within "
                           f"{a.cutoff:g} A in any conformer")
                continue
            top = sorted(tally.items(), key=lambda kv: -len(kv[1]))[0]
            always = [r for r, v in tally.items() if len(v) == n_used]
            log.append(f"  {cpd} {a.element}{rank}: "
                       f"{np.mean(per_conf):.1f} residues within "
                       f"{a.cutoff:g} A on average, {len(tally)} different "
                       f"ones in all; most often {top[0]} "
                       f"({len(top[1])} of {n_used}, mean "
                       f"{np.mean(top[1]):.1f} A)")
            if always:
                log.append(f"  {'':{len(cpd)+len(a.element)+4}s}present in "
                           f"every conformer: {', '.join(sorted(always))}")
    log.append(f"  a residue present in every conformer is part of the "
               f"substituent's environment; one seen a few times is not")

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

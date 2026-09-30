#!/usr/bin/env python3
"""
How far a compound's pose moves from one conformer of an ensemble to the next.

For each conformer the highest-scoring pose is taken, and its distance from the
previous conformer's pose is measured. A compound holding one binding mode
gives small numbers throughout; one whose pose flips between modes gives large
ones, which no score or contact count reveals.

The distance is a symmetry-aware RMSD computed WITHOUT superposing the two
poses, because the question is whether the ligand sits in the same place, not
whether the two poses have the same shape. That is only meaningful if the
receptor conformers are themselves superposed, so the protein's own movement
between consecutive conformers is measured alongside and printed: a ligand
that moves no more than the protein did has not really moved.

The last column is the distance from the final conformer's pose back to the
first, which closes the loop: a compound drifting steadily one way differs
there from one wandering and returning.

    python pose_rmsd_table.py --poses 'ens_cx_dock/poses/*.sdf' \
        --receptor a2 --fields compound,replicate,receptor \
        --only VB004,EV042,EV043 --ensemble ~/sim/.../ens_cx_a2
"""
import argparse, glob, os, re, sys
import numpy as np

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolAlign
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


def ca_coords(path):
    """Backbone alpha carbons, in file order."""
    out = []
    with open(path) as f:
        for ln in f:
            if ln[:6] in ("ATOM  ", "HETATM") and ln[12:16].strip() == "CA":
                out.append((float(ln[30:38]), float(ln[38:46]),
                            float(ln[46:54])))
    return np.asarray(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--poses", required=True, help="glob of pose sdf files")
    p.add_argument("--receptor", default="a2",
                   help="the label in the name field that selects the subunit")
    p.add_argument("--fields", default="compound,replicate,receptor",
                   help="what the three parts of COMPOUND__X__Y mean")
    p.add_argument("--select-by", default="CNNscore")
    p.add_argument("--only", required=True,
                   help="comma-separated compounds, in the order to print")
    p.add_argument("--ensemble",
                   help="directory of conformer pdb files. Used to measure how "
                        "far the protein itself moved between consecutive "
                        "conformers, which is the scale the ligand's movement "
                        "should be read against")
    p.add_argument("--width", type=int, default=7)
    p.add_argument("--out")
    a = p.parse_args()

    order = [x.strip() for x in a.fields.split(",")]
    if sorted(order) != ["compound", "receptor", "replicate"]:
        sys.exit("--fields must name compound, receptor and replicate once each")
    want = [x.strip() for x in a.only.split(",") if x.strip()]

    files = sorted(glob.glob(os.path.expanduser(a.poses)))
    if not files:
        sys.exit(f"no files matched {a.poses}")

    best = {}
    for f in files:
        got = parse_name(f, order)
        if not got:
            continue
        cpd, rec, rep = got
        if rec != a.receptor or cpd not in want:
            continue
        top, topv = None, None
        for m in Chem.SDMolSupplier(f, removeHs=False, sanitize=True):
            if m is None or not m.HasProp(a.select_by):
                continue
            try:
                v = float(m.GetProp(a.select_by))
            except ValueError:
                continue
            if topv is None or v > topv:
                top, topv = m, v
        if top is not None:
            best.setdefault(cpd, {})[rep] = top
    missing = [c for c in want if c not in best]
    if missing:
        sys.exit(f"no poses found for: {', '.join(missing)}")

    reps = sorted({r for d in best.values() for r in d})
    log = [f"[in] {a.poses}",
           f"     receptor {a.receptor}, {len(want)} compounds, "
           f"{len(reps)} conformers, best pose by {a.select_by}",
           f"     symmetry-aware RMSD, no superposition: the poses are "
           f"compared where they sit"]

    prot = None
    if a.ensemble:
        d = os.path.expanduser(a.ensemble)
        idx = {}
        for pth in glob.glob(os.path.join(d, "*.pdb")):
            k = "".join(ch for ch in os.path.basename(pth)[:-4].lower()
                        if ch.isalnum())
            idx[k] = pth
        cs, miss = [], []
        for r in reps:
            k = "".join(ch for ch in r.lower() if ch.isalnum())
            if k in idx:
                cs.append(ca_coords(idx[k]))
            else:
                cs.append(None)
                miss.append(r)
        if miss:
            log.append(f"     [note] no structure found for {len(miss)} "
                       f"conformer(s), e.g. {', '.join(miss[:3])}; the "
                       f"protein row is left out")
        elif any(c is None or not len(c) for c in cs):
            log.append("     [note] a structure held no alpha carbons; the "
                       "protein row is left out")
        else:
            n = min(len(c) for c in cs)
            prot = [float(np.sqrt(((cs[i][:n] - cs[i - 1][:n]) ** 2)
                                  .sum(axis=1).mean()))
                    for i in range(1, len(cs))]
            prot.append(float(np.sqrt(((cs[-1][:n] - cs[0][:n]) ** 2)
                                      .sum(axis=1).mean())))
            log.append(f"     protein movement measured on {n} alpha carbons, "
                       f"in place")
            if max(prot) > 4.0:
                log.append(f"     [warn] consecutive conformers differ by up "
                           f"to {max(prot):.1f} A at the alpha carbons. They "
                           f"are probably not superposed, and an in-place "
                           f"ligand RMSD is then not interpretable")

    w = a.width
    heads = [f"{reps[i-1].split('_')[-1]}>{reps[i].split('_')[-1]}"
             for i in range(1, len(reps))]
    heads.append(f"{reps[-1].split('_')[-1]}>{reps[0].split('_')[-1]}")
    log.append("")
    log.append(f"=== RMSD of the best pose against the previous conformer's "
               f"({a.receptor}) ===")
    log.append(f"    the last column closes the loop, last back to first")
    log.append(f"  {'compound':10s}" + "".join(f"{h:>{w+3}s}" for h in heads)
               + f"{'mean':>{w+1}s}{'max':>{w}s}")
    rows = {}
    for c in want:
        ms = [best[c].get(r) for r in reps]
        v = []
        for i in range(1, len(ms)):
            v.append(np.nan if (ms[i] is None or ms[i - 1] is None)
                     else float(rdMolAlign.CalcRMS(ms[i], ms[i - 1])))
        v.append(np.nan if (ms[-1] is None or ms[0] is None)
                 else float(rdMolAlign.CalcRMS(ms[-1], ms[0])))
        rows[c] = np.asarray(v)
        cells = "".join(" " * (w + 2) + "-" if np.isnan(x)
                        else f"{x:>{w+3}.2f}" for x in v)
        log.append(f"  {c:10s}{cells}"
                   f"{np.nanmean(v[:-1]):>{w+1}.2f}{np.nanmax(v[:-1]):>{w}.2f}")
    if prot is not None:
        log.append(f"  {'protein':10s}"
                   + "".join(f"{x:>{w+3}.2f}" for x in prot))

    log.append("")
    log.append("=== how much of the movement is the ligand's own ===")
    for c in want:
        v = rows[c][:-1]
        line = (f"  {c:10s} mean {np.nanmean(v):5.2f} A, "
                f"largest {np.nanmax(v):5.2f} A, "
                f"loop {rows[c][-1]:5.2f} A")
        if prot is not None:
            pm = float(np.mean(prot[:-1]))
            line += f"   protein moved {pm:.2f} A on average"
            if pm > 0:
                line += f", ratio {np.nanmean(v)/pm:.1f}x"
        log.append(line)
    log.append("  a ratio near 1 means the pose is riding the protein's "
               "movement; well above 1 means the ligand is finding a "
               "different place to sit")

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

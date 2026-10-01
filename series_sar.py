#!/usr/bin/env python3
"""
What a set of analogues share, and where they differ.

The compounds that survive a selection funnel are usually close relatives, and
a ranking that cannot separate them is easier to argue about once the
structural difference between them is on the page. This finds the largest
substructure common to all of them, decomposes each one into that core plus
its substituents, and prints the substituents beside the properties that
depend on them.

The core is found rather than assumed. A core supplied by hand tends to be the
one that makes the story tidy, and a substituent table built on it inherits
that choice; the maximum common substructure is whatever the molecules
actually share.

    python series_sar.py --mol crystal_str/6hmd \\
        --only EV030,EV034,EV043,VB004 --out series_sar
"""
import argparse, glob, os, re, sys

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors, Draw, QED, rdFMCS
    from rdkit.Chem import rdRGroupDecomposition as rgd
    from rdkit.Chem.Draw import rdMolDraw2D
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("needs rdkit")


def name_of(path):
    b = os.path.basename(path)
    for e in (".mol", ".sdf", ".mol2", ".smi"):
        if b.endswith(e):
            b = b[: -len(e)]
            break
    return re.sub(r"_docked$|_out$|_ligand$", "", b)


def load(a):
    """(name, mol) for every compound asked for."""
    out = {}
    if a.smiles:
        import pandas as pd
        t = pd.read_csv(a.smiles)
        cols = {c.lower(): c for c in t.columns}
        nc = cols.get("compound") or cols.get("name") or cols.get("id")
        sc = cols.get("smiles") or cols.get("canonical_smiles")
        if not nc or not sc:
            sys.exit(f"{a.smiles} needs a name column and a smiles column; it "
                     f"has {', '.join(t.columns)}")
        for _, r in t.iterrows():
            m = Chem.MolFromSmiles(str(r[sc]))
            if m is not None:
                out[str(r[nc]).strip()] = m
    if a.mol:
        for f in sorted(glob.glob(os.path.join(a.mol, "*.mol"))
                        + glob.glob(os.path.join(a.mol, "*.sdf"))):
            n = name_of(f)
            if n in out:
                continue
            m = (Chem.MolFromMolFile(f) if f.endswith(".mol")
                 else next(iter(Chem.SDMolSupplier(f)), None))
            if m is not None:
                out[n] = m
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mol", help="directory of .mol or .sdf files, one per "
                                 "compound, named after it")
    p.add_argument("--smiles", help="csv with a name column and a smiles one")
    p.add_argument("--only", help="comma-separated compounds, in the order "
                                  "they should be reported")
    p.add_argument("--core", help="SMARTS to use as the core instead of the "
                                  "maximum common substructure")
    p.add_argument("--timeout", type=int, default=60,
                   help="seconds allowed for the common-substructure search")
    p.add_argument("--ring-matches-ring", action="store_true", default=True)
    p.add_argument("--image", action="store_true",
                   help="also draw the structures with the core highlighted")
    p.add_argument("--per-row", type=int, default=4)
    p.add_argument("--out")
    a = p.parse_args()

    if not a.mol and not a.smiles:
        sys.exit("give --mol or --smiles")
    mols = load(a)
    if not mols:
        sys.exit("no structures read")
    order = ([x.strip() for x in a.only.split(",") if x.strip()]
             if a.only else sorted(mols))
    miss = [c for c in order if c not in mols]
    if miss:
        sys.exit(f"not found: {', '.join(miss)}.  Available: "
                 f"{', '.join(sorted(mols))}")
    ms = [mols[c] for c in order]

    log = [f"[in] {len(ms)} compounds: {', '.join(order)}"]

    if a.core:
        core = Chem.MolFromSmarts(a.core)
        if core is None:
            sys.exit(f"--core is not valid SMARTS: {a.core!r}")
        log.append(f"     core given: {a.core}")
    else:
        res = rdFMCS.FindMCS(
            ms, timeout=a.timeout,
            ringMatchesRingOnly=a.ring_matches_ring,
            completeRingsOnly=True,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder)
        if res.canceled:
            log.append(f"     [note] the search hit the {a.timeout}s limit; "
                       f"the core below is the best found so far")
        core = Chem.MolFromSmarts(res.smartsString)
        log.append(f"     common core: {res.smartsString}")
        log.append(f"     {res.numAtoms} atoms and {res.numBonds} bonds "
                   f"shared by all {len(ms)}")

    nmatch = sum(1 for m in ms if m.HasSubstructMatch(core))
    log.append(f"     the core matches {nmatch} of {len(ms)} compounds")
    log.append(f"     every compound carries {max(m.GetNumHeavyAtoms() for m in ms)}"
               f" heavy atoms at most, {min(m.GetNumHeavyAtoms() for m in ms)}"
               f" at least")

    # the substituents: what each compound has that the core does not
    rtable, rcols = {}, []
    try:
        dec, unmatched = rgd.RGroupDecompose([core], ms, asSmiles=True,
                                             asRows=True)
        if unmatched:
            log.append(f"     [note] not decomposed: "
                       f"{', '.join(order[i] for i in unmatched)}")
        keep = [i for i in range(len(ms)) if i not in set(unmatched)]
        for row, i in zip(dec, keep):
            rtable[order[i]] = {k: v for k, v in row.items() if k != "Core"}
        rcols = sorted({k for r in rtable.values() for k in r},
                       key=lambda k: (len(k), k))
        # a position every compound fills the same way says nothing
        varying = [k for k in rcols
                   if len({r.get(k, "") for r in rtable.values()}) > 1]
        fixed = [k for k in rcols if k not in varying]
        if fixed:
            log.append(f"     identical in every compound, so left out: "
                       f"{', '.join(fixed)}")
        rcols = varying
    except Exception as e:
        log.append(f"     [note] substituent decomposition failed ({e}); "
                   f"reporting properties only")

    log.append("")
    log.append("=== where the compounds differ ===")
    if rcols:
        w = max(12, max(len(str(v)) for r in rtable.values()
                        for v in r.values()) + 2)
        w = min(w, 34)
        hdr = f"  {'compound':10s}"
        for k in rcols:
            hdr += f"{k:>{w}s}"
        log.append(hdr)
        for c in order:
            line = f"  {c:10s}"
            for k in rcols:
                v = str(rtable.get(c, {}).get(k, "-"))
                line += f"{(v[:w - 2] + '..' if len(v) > w else v):>{w}s}"
            log.append(line)
    else:
        log.append("  no varying substituent positions were resolved")

    log.append("")
    log.append("=== properties ===")
    log.append(f"  {'compound':10s}{'formula':>16s}{'MW':>8s}{'cLogP':>8s}"
               f"{'TPSA':>8s}{'HBD':>5s}{'HBA':>5s}{'rot':>5s}{'QED':>7s}"
               f"{'halogens':>10s}")
    rows = []
    for c in order:
        m = mols[c]
        hal = "".join(sorted(at.GetSymbol() for at in m.GetAtoms()
                             if at.GetSymbol() in ("F", "Cl", "Br", "I"))) or "-"
        d = dict(compound=c,
                 formula=Chem.rdMolDescriptors.CalcMolFormula(m),
                 MW=Descriptors.MolWt(m), cLogP=Descriptors.MolLogP(m),
                 TPSA=Descriptors.TPSA(m),
                 HBD=Descriptors.NumHDonors(m), HBA=Descriptors.NumHAcceptors(m),
                 rot=Descriptors.NumRotatableBonds(m), QED=QED.qed(m),
                 halogens=hal)
        rows.append(d)
        log.append(f"  {c:10s}{d['formula']:>16s}{d['MW']:8.1f}"
                   f"{d['cLogP']:8.2f}{d['TPSA']:8.1f}{d['HBD']:5d}"
                   f"{d['HBA']:5d}{d['rot']:5d}{d['QED']:7.2f}"
                   f"{hal:>10s}")

    log.append("")
    log.append("=== how far apart they are ===")
    smis = {c: Chem.MolToSmiles(mols[c]) for c in order}
    same = {}
    for c in order:
        same.setdefault(smis[c], []).append(c)
    dup = [v for v in same.values() if len(v) > 1]
    if dup:
        log.append("  identical structures: "
                   + "; ".join(", ".join(v) for v in dup))
    else:
        log.append("  no two compounds are the same structure")
    uniq_hal = sorted({r["halogens"] for r in rows})
    log.append(f"  halogen patterns present: {', '.join(uniq_hal)}")
    lp = [r["cLogP"] for r in rows]
    log.append(f"  cLogP spans {min(lp):.2f} to {max(lp):.2f}; "
               f"QED {min(r['QED'] for r in rows):.2f} to "
               f"{max(r['QED'] for r in rows):.2f}")

    text = "\n".join(log)
    print(text)

    if a.out:
        import pandas as pd
        t = pd.DataFrame(rows)
        for k in rcols:
            t[k] = [rtable.get(c, {}).get(k, "") for c in order]
        t["smiles"] = [smis[c] for c in order]
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        t.to_csv(a.out + ".csv", index=False)
        with open(a.out + ".txt", "w") as f:
            f.write(text + "\n")
        print(f"\n[out] {a.out}.csv, {a.out}.txt")

    if a.image and a.out:
        flat = []
        for c in order:
            m = Chem.Mol(mols[c])
            m.RemoveAllConformers()
            AllChem.Compute2DCoords(m)
            flat.append(m)
        try:                       # one orientation for all, set by the core
            ctmp = Chem.Mol(core)
            AllChem.Compute2DCoords(ctmp)
            for m in flat:
                AllChem.GenerateDepictionMatching2DStructure(
                    m, ctmp, acceptFailure=True)
        except Exception:
            pass
        hits = [list(m.GetSubstructMatch(core)) for m in flat]
        img = Draw.MolsToGridImage(
            flat, molsPerRow=a.per_row, subImgSize=(330, 300),
            legends=order, highlightAtomLists=hits, useSVG=True)
        svg = img.data if hasattr(img, "data") else str(img)
        with open(a.out + ".svg", "w") as f:
            f.write(svg)
        print(f"[out] {a.out}.svg   (the shared core is highlighted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

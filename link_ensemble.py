#!/usr/bin/env python3
"""
Lay an ensemble docking run out the way the analysis scripts read it.

The ensemble runs put every pose file in one directory, named
COMPOUND__STRUCTURE__TAG.sdf, with the receptors in a directory of their own
per isoform. The analysis scripts read a directory per structure, with the
receptors together and named after it. Nothing needs copying to bridge that:
this writes symbolic links.

Where a run was done more than once, each pass is given a run label and the
links are named so the scripts see the passes as repeats of the same cell,
which is what they are.

Structure names are matched to receptor files allowing for an underscore the
pose names drop: a1c00 finds a1_c00.pdb. Anything that still fails to match
is named rather than skipped quietly, since a missing receptor silently
drops a whole structure from the analysis.

    python link_ensemble.py \\
        --poses run1=/.../ens_cx_dock/poses \\
        --poses run2=/.../ens_variants/poses \\
        --receptors /.../ens_a1 --receptors /.../ens_a2 \\
        --out /.../ens_linked
"""
import argparse, os, re, sys
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--poses", action="append", required=True,
                   metavar="LABEL=DIR",
                   help="a directory of pose files, with the label to give "
                        "that pass. Repeat it for a run done more than once")
    p.add_argument("--receptors", action="append", required=True,
                   metavar="DIR", help="repeat for each isoform")
    p.add_argument("--out", required=True)
    p.add_argument("--only", help="comma-separated compounds to link")
    p.add_argument("--isoform", help="keep only structures whose name starts "
                                     "with this, e.g. a1")
    a = p.parse_args()

    keep = ([x.strip() for x in a.only.split(",") if x.strip()]
            if a.only else None)

    rec = {}
    for d in a.receptors:
        if not os.path.isdir(d):
            sys.exit(f"not a directory: {d}")
        for f in os.listdir(d):
            if f.endswith(".pdb"):
                rec[f[:-4]] = os.path.abspath(os.path.join(d, f))
    if not rec:
        sys.exit("no .pdb files in the receptor directories")

    def receptor_for(name):
        """a1c00 and a1_c00 are the same structure, written two ways."""
        if name in rec:
            return name
        m = re.match(r"^([A-Za-z]+\d*)[_]?(.*)$", name)
        for cand in ({name.replace("_", "")} |
                     ({f"{m.group(1)}_{m.group(2)}"} if m else set())):
            if cand in rec:
                return cand
        return None

    out_p = os.path.join(a.out, "results")
    out_r = os.path.join(a.out, "receptor")
    os.makedirs(out_r, exist_ok=True)

    linked, missing, skipped = 0, defaultdict(int), 0
    structs, cpds, runs = set(), set(), set()
    for spec in a.poses:
        if "=" not in spec:
            sys.exit(f"--poses takes LABEL=DIR, got {spec!r}")
        label, d = spec.split("=", 1)
        if not os.path.isdir(d):
            sys.exit(f"not a directory: {d}")
        for f in sorted(os.listdir(d)):
            if not f.endswith((".sdf", ".sdf.gz")):
                continue
            stem = f[:-7] if f.endswith(".sdf.gz") else f[:-4]
            parts = stem.split("__")
            if len(parts) < 2:
                skipped += 1
                continue
            cpd, struct = parts[0], parts[1]
            if keep and cpd not in keep:
                continue
            if a.isoform and not struct.startswith(a.isoform):
                continue
            canon = receptor_for(struct)
            if canon is None:
                missing[struct] += 1
                continue
            sd = os.path.join(out_p, canon)
            os.makedirs(sd, exist_ok=True)
            ext = ".sdf.gz" if f.endswith(".sdf.gz") else ".sdf"
            dst = os.path.join(sd, f"{cpd}__{canon}__{label}{ext}")
            src = os.path.abspath(os.path.join(d, f))
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)
            rdst = os.path.join(out_r, canon + ".pdb")
            if not os.path.exists(rdst):
                os.symlink(rec[canon], rdst)
            linked += 1
            structs.add(canon)
            cpds.add(cpd)
            runs.add(label)

    print(f"[out] {a.out}")
    print(f"  {linked} pose files linked under results/<structure>/")
    print(f"  {len(cpds)} compounds x {len(structs)} structures x "
          f"{len(runs)} run(s): {', '.join(sorted(runs))}")
    print(f"  {len(os.listdir(out_r))} receptors linked under receptor/")
    iso = defaultdict(int)
    for s in structs:
        iso[re.match(r'^[A-Za-z]+\d*', s).group(0)] += 1
    print("  structures by prefix: "
          + ", ".join(f"{k} {v}" for k, v in sorted(iso.items())))
    if skipped:
        print(f"  [note] {skipped} files whose names had no '__' were skipped")
    if missing:
        print(f"  [warn] no receptor for {len(missing)} structure names, "
              f"{sum(missing.values())} files left out:")
        for k, v in sorted(missing.items())[:10]:
            print(f"    {k} ({v} files)")
        print("    these structures are absent from every analysis that "
              "reads this directory")
    return 0


if __name__ == "__main__":
    sys.exit(main())

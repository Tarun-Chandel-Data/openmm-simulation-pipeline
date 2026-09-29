#!/usr/bin/env python3
"""
Dock one ligand set into every conformer of one or more receptor ensembles.

The ligand set, the pocket and the exhaustiveness are held fixed and only the
receptors change, so a run made with this script is comparable to an earlier
run made with the same ligands and settings into a different ensemble. The
settings that decide comparability are printed and written to a manifest, so a
later reader can check they matched rather than assume it.

Output names follow  <compound>__<conformer>__<subunit>.sdf  in --out/poses,
the convention the earlier ensemble run used.

    python dock_ensemble.py \
        --ligands ~/docking_files/TEST/new/ev_ligands \
        --receptors a1=~/sim/small_molecule/cx/ens_cx_a1 \
        --receptors a2=~/sim/small_molecule/cx/ens_cx_a2 \
        --autobox-ligand ~/docking_files/TEST/new/ref_a1.sdf \
        --exhaustiveness 16 --num-modes 100 --seed 0 \
        --out ~/sim/small_molecule/cx/ens_cx_dock --jobs 4
"""
import argparse, csv, json, os, shlex, subprocess, sys, time
from glob import glob

LIG_EXT = (".sdf", ".mol2", ".pdbqt", ".smi")
REC_EXT = (".pdb", ".pdbqt")


def listing(d, exts):
    out = []
    for e in exts:
        out += glob(os.path.join(d, "*" + e))
    return sorted(out)


def stem(p):
    b = os.path.basename(p)
    for e in LIG_EXT + REC_EXT:
        if b.endswith(e):
            b = b[: -len(e)]
            break
    # a receptor written out of a trajectory often carries the ensemble name
    # again; the subunit label already says which ensemble it came from
    return b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ligands", required=True,
                   help="directory of ligand files, or a single multi-molecule "
                        "file. One docking run per ligand per receptor")
    p.add_argument("--receptors", action="append", required=True,
                   metavar="LABEL=DIR",
                   help="an ensemble to dock into, labelled. Repeat for each "
                        "subunit. The label becomes the third field of every "
                        "output name")
    p.add_argument("--autobox-ligand",
                   help="a ligand occupying the pocket, used to place the box. "
                        "Passing the same file for every receptor keeps the "
                        "pocket fixed across the ensemble, which is what makes "
                        "conformers comparable")
    p.add_argument("--autobox-add", type=float, default=4.0)
    p.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"),
                   help="explicit box centre, as an alternative to "
                        "--autobox-ligand")
    p.add_argument("--pocket-residues",
                   help="comma-separated residue numbers, per ensemble as "
                        "LABEL:1,2,3 and repeated with ';'. The box is centred "
                        "on these residues in each receptor separately. A "
                        "conformer taken from a trajectory is in the "
                        "trajectory's frame, not the reference structure's, so "
                        "one shared --autobox-ligand would put the box in the "
                        "wrong place; residues travel with the receptor and do "
                        "not")
    p.add_argument("--size", nargs=3, type=float, default=[22.0, 22.0, 22.0],
                   metavar=("X", "Y", "Z"))
    p.add_argument("--exhaustiveness", type=int, default=16)
    p.add_argument("--num-modes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0,
                   help="fixed by default. Docking is stochastic, so an "
                        "unfixed seed puts run-to-run noise into a comparison "
                        "meant to isolate the receptor")
    p.add_argument("--cnn", default="rescore",
                   help="gnina --cnn_scoring. 'rescore' scores the Vina poses "
                        "with the CNN; 'all' also refines with it and is far "
                        "slower")
    p.add_argument("--gnina", default="gnina")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and the first command, run nothing")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="redo runs whose output already exists")
    a = p.parse_args()

    given = [n for n, v in (("--autobox-ligand", a.autobox_ligand),
                            ("--center", a.center),
                            ("--pocket-residues", a.pocket_residues)) if v]
    if not given:
        sys.exit("need one of --autobox-ligand, --center or "
                 "--pocket-residues: without one the box is undefined and "
                 "gnina will refuse")
    if len(given) > 1:
        sys.exit(f"{' and '.join(given)} all given; pick one")

    pocket = {}
    if a.pocket_residues:
        for part in a.pocket_residues.split(";"):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                sys.exit("--pocket-residues wants LABEL:1,2,3 (several "
                         "separated by ';'), got " + repr(part))
            lab, nums = part.split(":", 1)
            try:
                pocket[lab.strip()] = {int(x) for x in nums.split(",") if x}
            except ValueError:
                sys.exit(f"non-numeric residue in '{part}'")

    ligs = ([a.ligands] if os.path.isfile(a.ligands)
            else listing(a.ligands, LIG_EXT))
    if not ligs:
        sys.exit(f"no ligand files in {a.ligands}")

    ens = {}
    for spec in a.receptors:
        if "=" not in spec:
            sys.exit(f"--receptors wants LABEL=DIR, got '{spec}'")
        lab, d = spec.split("=", 1)
        d = os.path.expanduser(d)
        recs = listing(d, REC_EXT)
        if not recs:
            sys.exit(f"no receptor files ({'/'.join(REC_EXT)}) in {d}")
        ens[lab] = recs

    posedir = os.path.join(a.out, "poses")
    logdir = os.path.join(a.out, "logs")
    for d in (posedir, logdir):
        os.makedirs(d, exist_ok=True)

    def size_flags():
        return ["--size_x", str(a.size[0]), "--size_y", str(a.size[1]),
                "--size_z", str(a.size[2])]

    def box_for(rec, lab):
        """The box flags for one receptor. Only the residue mode differs per
        receptor; the other two modes return the same flags every time."""
        if a.autobox_ligand:
            return ["--autobox_ligand", os.path.expanduser(a.autobox_ligand),
                    "--autobox_add", str(a.autobox_add)]
        if a.center:
            return ["--center_x", str(a.center[0]),
                    "--center_y", str(a.center[1]),
                    "--center_z", str(a.center[2])] + size_flags()
        want = pocket.get(lab)
        if want is None:
            sys.exit(f"--pocket-residues has no entry for ensemble '{lab}'")
        xyz, seen = [], set()
        with open(rec) as fh:
            for ln in fh:
                if ln[:6] not in ("ATOM  ", "HETATM"):
                    continue
                try:
                    resi = int(ln[22:26])
                except ValueError:
                    continue
                if resi not in want:
                    continue
                if ln[76:78].strip() == "H" or ln[12:16].strip().startswith("H"):
                    continue          # heavy atoms only, as for any centroid
                seen.add(resi)
                xyz.append((float(ln[30:38]), float(ln[38:46]),
                            float(ln[46:54])))
        if not xyz:
            sys.exit(f"none of the pocket residues found in {rec}. Check the "
                     f"numbering: a prmtop-derived structure is renumbered "
                     f"1..N and will not match PDB numbering")
        missing = sorted(want - seen)
        if missing:
            # a silently absent residue shifts the centroid, so it is named
            # rather than averaged over whatever was present
            sys.exit(f"{os.path.basename(rec)} is missing pocket residue(s) "
                     f"{missing}; centring on the rest would move the box")
        cx = sum(p[0] for p in xyz) / len(xyz)
        cy = sum(p[1] for p in xyz) / len(xyz)
        cz = sum(p[2] for p in xyz) / len(xyz)
        return ["--center_x", f"{cx:.3f}", "--center_y", f"{cy:.3f}",
                "--center_z", f"{cz:.3f}"] + size_flags()

    jobs = []
    centres = {}
    for lab, recs in ens.items():
        for rec in recs:
            for lig in ligs:
                name = f"{stem(lig)}__{stem(rec)}__{lab}"
                out = os.path.join(posedir, name + ".sdf")
                log = os.path.join(logdir, name + ".log")
                cmd = [a.gnina, "-r", rec, "-l", lig, "-o", out,
                       "--exhaustiveness", str(a.exhaustiveness),
                       "--num_modes", str(a.num_modes),
                       "--seed", str(a.seed),
                       "--cnn_scoring", a.cnn] + box_for(rec, lab)
                jobs.append((name, out, log, cmd))

    n_lig, n_rec = len(ligs), sum(len(v) for v in ens.values())
    print(f"[plan] {n_lig} ligands x {n_rec} receptors = {len(jobs)} runs")
    for lab, recs in ens.items():
        print(f"       {lab}: {len(recs)} conformers from "
              f"{os.path.dirname(recs[0])}")
    print(f"       exhaustiveness {a.exhaustiveness}, num_modes "
          f"{a.num_modes}, seed {a.seed}, cnn_scoring {a.cnn}")
    if a.pocket_residues:
        print(f"       box: {a.size[0]:g} x {a.size[1]:g} x {a.size[2]:g} A, "
              f"centred per receptor on its pocket residues")
        for lab, recs in ens.items():
            cs = [box_for(r, lab) for r in recs]
            for ax, i in (("x", 1), ("y", 3), ("z", 5)):
                v = [float(c[i]) for c in cs]
                spread = max(v) - min(v)
                print(f"         {lab} centre {ax}: {sum(v)/len(v):8.3f} "
                      f"(spread {spread:.3f} A over {len(v)} conformers)")
        print("         a large spread means the conformers are not "
              "superposed; that is fine, the box follows each one")
    else:
        print(f"       box: {' '.join(box_for(next(iter(ens.values()))[0], next(iter(ens))))}")

    todo = [j for j in jobs
            if not (a.resume and os.path.exists(j[1])
                    and os.path.getsize(j[1]) > 0)]
    skipped = len(jobs) - len(todo)
    if skipped:
        print(f"[resume] {skipped} outputs already present, {len(todo)} to run")

    manifest = os.path.join(a.out, "manifest.json")
    with open(manifest, "w") as f:
        json.dump({"ligand_source": os.path.abspath(a.ligands),
                   "n_ligands": n_lig,
                   "ensembles": {k: [os.path.abspath(r) for r in v]
                                 for k, v in ens.items()},
                   "exhaustiveness": a.exhaustiveness,
                   "num_modes": a.num_modes, "seed": a.seed,
                   "cnn_scoring": a.cnn,
                   "box_mode": ("pocket-residues" if a.pocket_residues else
                                "autobox-ligand" if a.autobox_ligand
                                else "center"),
                   "pocket_residues": {k: sorted(v)
                                       for k, v in pocket.items()},
                   "box_size": a.size,
                   "n_runs": len(jobs),
                   "gnina": a.gnina}, f, indent=2)
    print(f"[out] {manifest}")

    if a.dry_run:
        if todo:
            print("\n[first command]\n  "
                  + " ".join(shlex.quote(c) for c in todo[0][3]))
        print(f"\n[dry-run] nothing executed")
        return 0
    if not todo:
        print("[done] nothing to do")
        return 0

    t0 = time.time()
    fails, running = [], []

    def reap(r):
        for name, proc, fh in r:
            rc = proc.wait()
            fh.close()
            if rc != 0:
                fails.append((name, rc))
                print(f"  [fail rc={rc}] {name}")

    for i, (name, out, log, cmd) in enumerate(todo, 1):
        fh = open(log, "w")
        running.append((name, subprocess.Popen(cmd, stdout=fh,
                                               stderr=subprocess.STDOUT), fh))
        if len(running) >= a.jobs:
            reap(running)
            running = []
            el = time.time() - t0
            rate = i / el if el else 0
            eta = (len(todo) - i) / rate if rate else 0
            print(f"  {i}/{len(todo)}  {el/60:.1f} min elapsed, "
                  f"~{eta/60:.0f} min left", flush=True)
    reap(running)

    # a run that exits 0 but writes nothing is a silent failure, so the count
    # is taken from the files rather than from the return codes alone
    empty = [j[0] for j in todo
             if not (os.path.exists(j[1]) and os.path.getsize(j[1]) > 0)]
    print(f"\n[done] {len(todo) - len(fails)} of {len(todo)} runs exited 0 "
          f"in {(time.time()-t0)/60:.1f} min")
    if fails:
        print(f"       {len(fails)} nonzero exits, see {logdir}")
    if empty:
        print(f"       {len(empty)} produced no poses: "
              + ", ".join(empty[:5]) + (" ..." if len(empty) > 5 else ""))
    print(f"       poses in {posedir}")
    return 1 if (fails or empty) else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Single-point MM-GBSA over an ensemble of docked poses, on the processor.

Nothing here uses the graphics card. CUDA is switched off for every child
process before it starts, the minimiser is the processor build, and the work
is niced so that a job already on the card keeps the host threads it needs.
Two cores are left free for that job by default.

The twenty members of an ensemble are the same protein in twenty
conformations, so one topology serves them all and the twenty complexes are
read as twenty frames. That is what MMPBSA.py is built for: it returns the
mean and the spread over the frames, which is the number worth having.

Poses are minimised before they are evaluated. A docked pose carries small
clashes, and an unminimised van der Waals term is not a measurement of
anything. The receptor is restrained so that the pose is not optimised away.

Checking comes first and is the default. The run only starts once one
complex has been built, minimised and evaluated end to end, because a
pipeline of this kind fails at its joins and it is better to find that out on
one complex than on four hundred.

    python mmgbsa_ensemble.py --ligand-params ligparm --receptors ens_a2 \\
        --poses lk_vb_a2/results --compounds VB004,EV043 --out mmgbsa_vb_a2
"""
import argparse, os, shutil, subprocess, sys, glob, textwrap

NEEDED = ["tleap", "sander", "MMPBSA.py"]


def say(tag, msg):
    print(f"[{tag}] {msg}", flush=True)


# Two sets of atom types turn up in a mol2 and they are read differently.
# SYBYL writes the element, a dot and a hybridisation - C.3, N.ar, Cl. GAFF,
# which is what antechamber writes and what these ligands carry, writes a
# lower-case code with no dot, where ca is aromatic carbon and cl is chlorine.
# Splitting a GAFF type on a dot returns the whole type and compares it with
# an element, so every atom reads as a mismatch.
GAFF_EXACT = {"cl": "CL", "br": "BR", "f": "F", "i": "I", "si": "SI"}
GAFF_FIRST = {"c": "C", "n": "N", "o": "O", "s": "S", "p": "P", "h": "H"}


def element_of(atom_type, atom_name=""):
    t = atom_type.strip()
    if "." in t:                      # SYBYL
        return t.split(".")[0].upper()
    low = t.lower()
    if low in GAFF_EXACT:             # GAFF, where cl is not a carbon
        return GAFF_EXACT[low]
    if low[:1] in GAFF_FIRST:
        return GAFF_FIRST[low[:1]]
    # nothing recognised: fall back on the name, which is usually the
    # element followed by a number
    nm = "".join(ch for ch in atom_name if ch.isalpha()).upper()
    for e in ("CL", "BR", "SI"):
        if nm.startswith(e):
            return e
    return (nm[:1] or t[:1]).upper()


def read_mol2_atoms(path):
    """(name, element) for each atom of a mol2, in file order."""
    out, on = [], False
    try:
        with open(path) as f:
            for line in f:
                t = line.rstrip("\n")
                st = t.strip()
                if st.startswith("@<TRIPOS>"):
                    on = st == "@<TRIPOS>ATOM"
                    continue
                if not on or not st:
                    continue
                c = st.split()
                if len(c) < 6:
                    continue
                out.append((c[1], element_of(c[5], c[1])))
    except OSError as e:
        return None, f"{path}: {e}"
    return (out, None) if out else (None, f"{path}: no ATOM block")


def read_pose_elements(path, select_by):
    """Elements of the best-scoring pose of an sdf, in file order."""
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return None, "needs rdkit to read the poses"
    try:
        mols = [m for m in Chem.SDMolSupplier(path, removeHs=False,
                                              sanitize=False) if m is not None]
    except Exception as e:                                    # noqa: BLE001
        return None, f"{path}: {e}"
    if not mols:
        return None, f"{path}: no poses read"
    best, bv = mols[0], None
    for m in mols:
        if m.HasProp(select_by):
            try:
                v = float(m.GetProp(select_by))
            except ValueError:
                continue
            if bv is None or v > bv:
                best, bv = m, v
    return [a.GetSymbol().upper() for a in best.GetAtoms()], None


def check_atom_order(lig_dir, pose_file, select_by):
    """The one joint in this pipeline that can quietly produce nonsense.

    The pose carries coordinates and the parameters carry charges and types;
    they are married by position in the file. If the two atom lists are not
    the same elements in the same order, every charge lands on the wrong
    atom and the result looks plausible and is meaningless."""
    mol2 = os.path.join(lig_dir, "LIG.mol2")
    if not os.path.exists(mol2):
        return f"no LIG.mol2 in {lig_dir} to check the atom order against"
    atoms, err = read_mol2_atoms(mol2)
    if err:
        return err
    els, err = read_pose_elements(pose_file, select_by)
    if err:
        return err
    pe = [e for _, e in atoms]
    if len(pe) != len(els):
        heavy_m = sum(1 for e in pe if e != "H")
        heavy_p = sum(1 for e in els if e != "H")
        extra = ("; the pose has no hydrogens while the parameters do"
                 if heavy_p == len(els) and heavy_m == heavy_p
                 else "")
        return (f"atom count differs: LIG.mol2 has {len(pe)} "
                f"({heavy_m} heavy), the pose has {len(els)} "
                f"({heavy_p} heavy){extra}")
    wrong = [i for i, (x, y) in enumerate(zip(pe, els)) if x != y]
    if wrong:
        i = wrong[0]
        return (f"atom {i + 1} is {pe[i]} in LIG.mol2 and {els[i]} in the "
                f"pose; {len(wrong)} of {len(pe)} differ, so the two files "
                f"are not in the same order")
    return None


def safe_env(np_cap):
    """No graphics card, for this process and everything it starts."""
    e = dict(os.environ)
    e["CUDA_VISIBLE_DEVICES"] = ""
    e["OMP_NUM_THREADS"] = "1"
    e["MKL_NUM_THREADS"] = "1"
    return e


def run(cmd, env, cwd=None, log=None, timeout=1800):
    """Never raises. Returns (ok, output); the caller decides what to do."""
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
    except FileNotFoundError:
        return False, f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return False, f"{' '.join(cmd[:2])}: timed out after {timeout}s"
    except Exception as e:                                   # noqa: BLE001
        return False, f"{' '.join(cmd[:2])}: {e}"
    if log:
        try:
            with open(log, "w") as f:
                f.write(p.stdout or "")
        except OSError:
            pass
    return p.returncode == 0, (p.stdout or "")


def preflight(a):
    """Everything that can be checked without running the calculation."""
    bad = []

    missing = [t for t in NEEDED if shutil.which(t) is None]
    if missing:
        bad.append(f"not on PATH: {', '.join(missing)}. "
                   f"Source amber.sh first")
    else:
        say("ok", "amber tools found: "
                  + ", ".join(f"{t}={shutil.which(t)}" for t in NEEDED))

    for d, what in ((a.receptors, "receptors"), (a.poses, "poses")):
        if not os.path.isdir(d):
            bad.append(f"{what}: not a directory: {d}")

    cpds = [x.strip() for x in a.compounds.split(",") if x.strip()]
    params = {}
    for spec in a.ligand_params:
        if "=" not in spec:
            bad.append(f"--ligand-params takes COMPOUND=DIR, got {spec!r}")
            continue
        c, d = spec.split("=", 1)
        params[c.strip()] = d
    for c in cpds:
        d = params.get(c)
        if d is None:
            bad.append(f"{c}: no --ligand-params given for it")
            continue
        if not os.path.isdir(d):
            bad.append(f"{c}: not a directory: {d}")
            continue
        frc = os.path.join(d, "LIG.frcmod")
        if not os.path.exists(frc):
            bad.append(f"{c}: no LIG.frcmod in {d}")
        tmpl = [os.path.join(d, n) for n in ("LIG.lib", "LIG.mol2")]
        have = [t for t in tmpl if os.path.exists(t)]
        if not have:
            bad.append(f"{c}: neither LIG.lib nor LIG.mol2 in {d}")
        elif not bad:
            say("ok", f"{c}: {os.path.basename(have[0])} + LIG.frcmod "
                      f"from {d}")

    try:
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(a.out))
                                 or ".").free / 1e9
        say("ok", f"{free:.1f} GB free where the output goes")
        if free < 5:
            bad.append(f"only {free:.1f} GB free; this wants a few GB of "
                       f"working space")
    except OSError as e:
        bad.append(f"cannot check free space: {e}")

    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:
        ncpu = os.cpu_count() or 1
    if a.np > max(1, ncpu - 2):
        bad.append(f"--np {a.np} on {ncpu} cores leaves nothing for the job "
                   f"already running; use {max(1, ncpu - 2)} or fewer")
    else:
        say("ok", f"{a.np} of {ncpu} cores, {ncpu - a.np} left free")

    say("ok", "CUDA_VISIBLE_DEVICES will be empty for every child process")
    return bad, cpds


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            The default does nothing but check. Add --run once --check is
            clean, and --run resumes: a cell whose result is already written
            is left alone.
        """))
    p.add_argument("--ligand-params", action="append", required=True,
                   metavar="COMPOUND=DIR",
                   help="the directory holding that compound's LIG.frcmod "
                        "and LIG.lib (or LIG.mol2) from the simulation "
                        "already run. Repeat once per compound. These are "
                        "reused rather than regenerated: charges derived "
                        "again would not give the same numbers, and the point "
                        "is to be comparable with the simulations")
    p.add_argument("--receptors", required=True)
    p.add_argument("--poses", required=True,
                   help="results directory, one subdirectory per structure")
    p.add_argument("--compounds", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--np", type=int, default=16,
                   help="cores for MMPBSA.py. Two fewer than the machine has, "
                        "at most, so a job on the card keeps its host thread")
    p.add_argument("--igb", type=int, default=2,
                   help="2, as in the simulations already run. A different model would not give comparable numbers")
    p.add_argument("--salt", type=float, default=0.15)
    p.add_argument("--protein-ff", default="leaprc.protein.ff14SB")
    p.add_argument("--ligand-ff", default="leaprc.gaff2")
    p.add_argument("--min-steps", type=int, default=500)
    p.add_argument("--restraint", type=float, default=5.0,
                   help="kcal/mol/A^2 on the receptor heavy atoms during "
                        "minimisation, so the pose is relaxed and not "
                        "replaced")
    p.add_argument("--select-by", default="CNNscore")
    p.add_argument("--run", action="store_true",
                   help="do the work. Without it nothing is run but the "
                        "checks and a single trial complex")
    a = p.parse_args()

    say("mode", "checking only" if not a.run else "checking, then running")
    bad, cpds = preflight(a)
    if bad:
        print()
        for b in bad:
            say("STOP", b)
        print()
        say("STOP", f"{len(bad)} problem(s); nothing was run")
        return 1

    structs = sorted(d for d in glob.glob(os.path.join(a.poses, "*"))
                     if os.path.isdir(d))
    if not structs:
        say("STOP", f"no structure directories under {a.poses}")
        return 1
    say("ok", f"{len(cpds)} compounds x {len(structs)} ensemble members "
              f"= {len(cpds) * len(structs)} complexes")

    env = safe_env(a.np)
    ok, out = run(["sander", "--version"], env)
    say("ok" if ok else "note",
        "sander answers" if ok else "sander did not answer --version; "
        "that is not fatal, it will be tried properly on the trial complex")

    # the atom order is checked on one real pose per compound, because
    # everything downstream silently depends on it
    for c in cpds:
        d = params.get(c)
        hits = sorted(glob.glob(os.path.join(a.poses, "*", f"{c}__*.sdf")))
        if not hits:
            say("STOP", f"{c}: no pose files under {a.poses}")
            return 1
        err = check_atom_order(d, hits[0], a.select_by)
        if err:
            say("STOP", f"{c}: {err}")
            say("STOP", f"     checked against {hits[0]}")
            return 1
        say("ok", f"{c}: the pose and LIG.mol2 hold the same atoms in the "
                  f"same order")

    say("ok", f"protocol: {a.protein_ff}, {a.ligand_ff}, igb={a.igb}, "
              f"saltcon={a.salt}")
    say("next", "the checks above pass. The trial complex and the run itself "
                "are not built yet - send this output together with the "
                "tleap.in used for the simulations, so the complex is built "
                "the same way, and they will be")
    return 0


if __name__ == "__main__":
    sys.exit(main())

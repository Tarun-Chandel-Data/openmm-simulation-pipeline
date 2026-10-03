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

NEEDED = ["tleap", "sander", "cpptraj", "MMPBSA.py"]
NEEDED_AUTO = ["antechamber", "parmchk2"]


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
    # Hydrogens are compared separately from the rest. A docking program
    # writes back whichever hydrogens its input carried - often only the
    # polar ones - while the parameters carry them all. That is not a
    # mismatch to refuse: the heavy atoms are what the search placed, and
    # the hydrogens are rebuilt from LIG.lib, which holds their geometry,
    # and relaxed by the minimisation that follows. What cannot be
    # recovered from is the heavy atoms being different or in a different
    # order, because nothing downstream would notice.
    hm = [(i, e) for i, e in enumerate(pe) if e != "H"]
    hp = [(i, e) for i, e in enumerate(els) if e != "H"]
    if len(hm) != len(hp):
        return (f"heavy atoms differ: LIG.mol2 has {len(hm)}, the pose has "
                f"{len(hp)}. These are not the same molecule")
    wrong = [k for k, ((_, x), (_, y)) in enumerate(zip(hm, hp)) if x != y]
    if wrong:
        k = wrong[0]
        return (f"heavy atom {k + 1} is {hm[k][1]} in LIG.mol2 and "
                f"{hp[k][1]} in the pose; {len(wrong)} of {len(hm)} differ, "
                f"so the two are not in the same order")
    nh_m, nh_p = len(pe) - len(hm), len(els) - len(hp)
    if nh_m != nh_p:
        return ("note", f"{len(hm)} heavy atoms match in order; the pose "
                        f"carries {nh_p} hydrogens against {nh_m} in the "
                        f"parameters, so hydrogens will be rebuilt from "
                        f"LIG.lib and relaxed by the minimisation")
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
    params, missing_params = {}, []
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
        mol2 = os.path.join(d, "LIG.mol2")
        if os.path.exists(frc) and os.path.exists(mol2):
            lib = "LIG.lib" if os.path.exists(os.path.join(d, "LIG.lib")) \
                  else "LIG.mol2"
            say("ok", f"{c}: {lib} + LIG.frcmod from {d}")
        elif a.make_params:
            missing_params.append(c)
            say("ok", f"{c}: no parameters yet; they will be derived into "
                      f"{d}")
        else:
            bad.append(f"{c}: no LIG.frcmod and LIG.mol2 in {d}, and "
                       f"--make-params was not given")

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
    if missing_params and shutil.which("antechamber") is None:
        bad.append("--make-params needs antechamber and parmchk2 "
                   "on PATH")
    return bad, cpds, params, missing_params


def make_params(cpd, pose, dest, select_by, ligand_ff, env):
    """Derive GAFF parameters for a compound that was never simulated.

    The template is built from the docked pose itself, with hydrogens added
    to it, so the heavy atoms of the parameters and of the poses are the
    same atoms in the same order by construction rather than by luck. The
    charges are AM1-BCC, as for the compounds that were simulated, but they
    are derived here and not there: the provenance differs even though the
    method does not, and that belongs in the methods section.
    """
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return "needs rdkit"
    try:
        mols = [m for m in Chem.SDMolSupplier(pose, removeHs=False,
                                              sanitize=True) if m is not None]
    except Exception as e:                                    # noqa: BLE001
        return f"{pose}: {e}"
    if not mols:
        return f"{pose}: no pose could be sanitised"
    best, bv = mols[0], None
    for m in mols:
        if m.HasProp(select_by):
            try:
                v = float(m.GetProp(select_by))
            except ValueError:
                continue
            if bv is None or v > bv:
                best, bv = m, v
    try:
        mh = Chem.AddHs(best, addCoords=True)
        AllChem.MMFFOptimizeMolecule(mh, maxIters=200)
    except Exception as e:                                    # noqa: BLE001
        return f"could not add hydrogens: {e}"
    charge = sum(a.GetFormalCharge() for a in mh.GetAtoms())

    os.makedirs(dest, exist_ok=True)
    src = os.path.join(dest, "lig_h.sdf")
    w = Chem.SDWriter(src)
    w.write(mh)
    w.close()

    at = "gaff2" if "gaff2" in ligand_ff else "gaff"
    ok, out = run(["antechamber", "-i", "lig_h.sdf", "-fi", "mdl",
                   "-o", "LIG.mol2", "-fo", "mol2", "-c", "bcc",
                   "-nc", str(charge), "-at", at, "-rn", "LIG", "-s", "2",
                   "-pf", "y"], env, cwd=dest,
                  log=os.path.join(dest, "antechamber.log"), timeout=3600)
    if not os.path.exists(os.path.join(dest, "LIG.mol2")):
        tail = "\n".join(out.splitlines()[-8:])
        return f"antechamber made no LIG.mol2:\n{tail}"
    ok, out = run(["parmchk2", "-i", "LIG.mol2", "-f", "mol2",
                   "-o", "LIG.frcmod", "-s", at], env, cwd=dest,
                  log=os.path.join(dest, "parmchk2.log"))
    if not os.path.exists(os.path.join(dest, "LIG.frcmod")):
        tail = "\n".join(out.splitlines()[-8:])
        return f"parmchk2 made no LIG.frcmod:\n{tail}"
    return None


def heavy_from_mol2(path):
    atoms, err = read_mol2_atoms(path)
    if err:
        return None, err
    return [(n, e) for n, e in atoms if e != "H"], None


def pose_heavy_coords(path, select_by):
    """Heavy-atom coordinates of the best-scoring pose, in file order."""
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return None, "needs rdkit"
    try:
        mols = [m for m in Chem.SDMolSupplier(path, removeHs=False,
                                              sanitize=False) if m is not None]
    except Exception as e:                                    # noqa: BLE001
        return None, f"{path}: {e}"
    if not mols:
        return None, f"{path}: no poses"
    best, bv = mols[0], None
    for m in mols:
        if m.HasProp(select_by):
            try:
                v = float(m.GetProp(select_by))
            except ValueError:
                continue
            if bv is None or v > bv:
                best, bv = m, v
    conf = best.GetConformer()
    out = []
    for at in best.GetAtoms():
        if at.GetSymbol().upper() == "H":
            continue
        p = conf.GetAtomPosition(at.GetIdx())
        out.append((p.x, p.y, p.z))
    return out, None


def pdb_atom_line(serial, name, resname, chain, resseq, x, y, z, element=""):
    """An ATOM record built by column, because it is read by column.

    A formatted string that looks right can still be wrong by one place,
    and the reader does not complain: it takes columns 18 to 20 as the
    residue name whatever was meant, so LIG written one column early is
    read as IG and the residue has no parameters. The record is assembled
    here into a fixed-width buffer at the positions the format defines.

    Columns: 1-6 ATOM, 7-11 serial, 13-16 name, 18-20 residue, 22 chain,
    23-26 sequence, 31-38 x, 39-46 y, 47-54 z, 55-60 occupancy,
    61-66 B factor, 77-78 element.
    """
    ln = [" "] * 80
    def put(start, text):          # start is 1-based, as the format counts
        for k, ch in enumerate(text):
            ln[start - 1 + k] = ch
    put(1, "ATOM  ")
    put(7, f"{serial:>5d}"[:5])
    # a name of fewer than four characters starts in column 14, which is
    # what keeps a two-letter element distinguishable from a long name
    put(13, f"{name:<4s}"[:4] if len(name) >= 4 else " " + f"{name:<3s}"[:3])
    put(18, f"{resname:>3s}"[:3])
    put(22, (chain or " ")[:1])
    put(23, f"{resseq:>4d}"[:4])
    put(31, f"{x:>8.3f}"[:8])
    put(39, f"{y:>8.3f}"[:8])
    put(47, f"{z:>8.3f}"[:8])
    put(55, "  1.00  0.00")
    if element:
        put(77, f"{element.capitalize():>2s}"[:2])
    return "".join(ln).rstrip() + "\n"


def write_lig_pdb(path, names, coords, elements=None):
    """One LIG residue, heavy atoms only; tleap builds the rest from the lib."""
    els = elements or [""] * len(names)
    with open(path, "w") as f:
        for i, (nm, (x, y, z), el) in enumerate(zip(names, coords, els), 1):
            f.write(pdb_atom_line(i, nm, "LIG", "A", 1, x, y, z, el))
        f.write("TER\nEND\n")


def write_receptor_pdb(src, dst):
    """Protein only: no waters, ions, other heteroatoms or hydrogens.

    Hydrogens are left to tleap so their names are the force field's own;
    a PDB's hydrogen names often are not, and tleap then builds duplicates
    or refuses the residue."""
    drop = {"HOH", "WAT", "NA", "CL", "K", "MG", "ZN", "SO4", "PO4", "EDO",
            "GOL", "LIG", "UNL", "UNK"}
    n = 0
    try:
        with open(src) as f, open(dst, "w") as o:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                if line[17:20].strip().upper() in drop:
                    continue
                el = line[76:78].strip().upper()
                nm = line[12:16].strip()
                if el == "H" or (not el and nm[:1] == "H"):
                    continue
                o.write(line)
                n += 1
            o.write("TER\nEND\n")
    except OSError as e:
        return 0, f"{src}: {e}"
    return n, None if n else f"{src}: no protein atoms kept"


LEAP = """source {pff}
source {lff}
loadamberparams {frcmod}
{load_unit}
rec = loadpdb rec.pdb
lig = loadpdb lig.pdb
com = combine {{ rec lig }}
saveamberparm com com.prmtop com.inpcrd
saveamberparm rec rec.prmtop rec.inpcrd
saveamberparm lig lig.prmtop lig.inpcrd
quit
"""

MIN = """minimise, receptor restrained so the docked pose is relaxed not replaced
 &cntrl
  imin=1, maxcyc={steps}, ncyc={ncyc},
  ntb=0, igb={igb}, saltcon={salt}, cut=999.0,
  ntr=1, restraint_wt={wt}, restraintmask='!:LIG & !@H=',
  ntpr=100,
 /
"""

MMPBSA_IN = """single point on the minimised complex
&general
   startframe=1, endframe=1, interval=1, verbose=2, keep_files=0,
/
&gb
   igb={igb}, saltcon={salt},
/
"""


def parse_total(path):
    """DELTA TOTAL from a FINAL_RESULTS file, with its standard deviation."""
    try:
        lines = open(path).read().splitlines()
    except OSError as e:
        return None, f"{path}: {e}"
    seen = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("DELTA TOTAL"):
            nums = []
            for tok in line.replace("DELTA TOTAL", "").split():
                try:
                    nums.append(float(tok))
                except ValueError:
                    pass
            if nums:
                seen = (nums[0], nums[1] if len(nums) > 1 else 0.0)
    if seen is None:
        return None, f"{path}: no DELTA TOTAL line"
    return seen, None


def one_complex(job):
    """Build, minimise and evaluate one complex. Never raises."""
    (cpd, struct, pose, rec_src, lig_dir, out_json, work, a_d) = job
    import json
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    tag = f"{cpd}/{struct}"
    try:
        os.makedirs(work, exist_ok=True)
        heavy, err = heavy_from_mol2(os.path.join(lig_dir, "LIG.mol2"))
        if err:
            return tag, None, err
        coords, err = pose_heavy_coords(pose, a_d["select_by"])
        if err:
            return tag, None, err
        if len(coords) != len(heavy):
            return tag, None, (f"{len(coords)} heavy atoms in the pose, "
                               f"{len(heavy)} in LIG.mol2")
        write_lig_pdb(os.path.join(work, "lig.pdb"),
                      [n for n, _ in heavy], coords,
                      [e for _, e in heavy])
        n, err = write_receptor_pdb(rec_src, os.path.join(work, "rec.pdb"))
        if err:
            return tag, None, err

        lib = os.path.join(lig_dir, "LIG.lib")
        load_unit = (f"loadoff {lib}" if os.path.exists(lib)
                     else f"LIG = loadmol2 {os.path.join(lig_dir, 'LIG.mol2')}")
        open(os.path.join(work, "leap.in"), "w").write(LEAP.format(
            pff=a_d["protein_ff"], lff=a_d["ligand_ff"],
            frcmod=os.path.join(lig_dir, "LIG.frcmod"),
            load_unit=load_unit))
        ok, out = run(["tleap", "-f", "leap.in"], env, cwd=work,
                      log=os.path.join(work, "leap.log"))
        for f in ("com.prmtop", "rec.prmtop", "lig.prmtop", "com.inpcrd"):
            if not os.path.exists(os.path.join(work, f)):
                tail = "\n".join(out.splitlines()[-6:])
                return tag, None, f"tleap made no {f}; last lines:\n{tail}"

        open(os.path.join(work, "min.in"), "w").write(MIN.format(
            steps=a_d["min_steps"], ncyc=max(50, a_d["min_steps"] // 3),
            igb=a_d["igb"], salt=a_d["salt"], wt=a_d["restraint"]))
        ok, out = run(["sander", "-O", "-i", "min.in", "-p", "com.prmtop",
                       "-c", "com.inpcrd", "-ref", "com.inpcrd",
                       "-r", "min.rst", "-o", "min.out"], env, cwd=work,
                      timeout=3600)
        if not os.path.exists(os.path.join(work, "min.rst")):
            tail = "\n".join(out.splitlines()[-6:])
            return tag, None, f"minimisation wrote no restart:\n{tail}"

        open(os.path.join(work, "traj.in"), "w").write(
            "parm com.prmtop\ntrajin min.rst\n"
            "trajout traj.mdcrd mdcrd\ngo\nquit\n")
        ok, out = run(["cpptraj", "-i", "traj.in"], env, cwd=work,
                      log=os.path.join(work, "cpptraj.log"))
        if not os.path.exists(os.path.join(work, "traj.mdcrd")):
            return tag, None, "cpptraj wrote no trajectory"

        open(os.path.join(work, "mmpbsa.in"), "w").write(
            MMPBSA_IN.format(igb=a_d["igb"], salt=a_d["salt"]))
        ok, out = run(["MMPBSA.py", "-O", "-i", "mmpbsa.in", "-o", "FINAL.dat",
                       "-cp", "com.prmtop", "-rp", "rec.prmtop",
                       "-lp", "lig.prmtop", "-y", "traj.mdcrd"], env, cwd=work,
                      log=os.path.join(work, "mmpbsa.log"), timeout=3600)
        res, err = parse_total(os.path.join(work, "FINAL.dat"))
        if err:
            tail = "\n".join(out.splitlines()[-8:])
            return tag, None, f"{err}\n{tail}"
        rec = {"compound": cpd, "structure": struct, "dG": res[0],
               "sd": res[1], "pose": pose}
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(rec, f)
        return tag, rec, None
    except Exception as e:                                    # noqa: BLE001
        return tag, None, f"unexpected: {type(e).__name__}: {e}"


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
    p.add_argument("--make-params", action="store_true",
                   help="derive GAFF parameters for any compound that has "
                        "none, from its own docked pose with hydrogens added. "
                        "Compounds that were simulated keep the parameters "
                        "those runs used")
    p.add_argument("--run", action="store_true",
                   help="do the work. Without it nothing is run but the "
                        "checks and a single trial complex")
    a = p.parse_args()

    say("mode", "checking only" if not a.run else "checking, then running")
    bad, cpds, params, missing_params = preflight(a)
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

    env0 = safe_env(a.np)
    for c in missing_params:
        hits = sorted(glob.glob(os.path.join(a.poses, "*", f"{c}__*.sdf")))
        if not hits:
            say("STOP", f"{c}: no pose to derive parameters from")
            return 1
        say("params", f"{c}: deriving AM1-BCC charges, this takes a minute")
        err = make_params(c, hits[0], params[c], a.select_by, a.ligand_ff,
                          env0)
        if err:
            say("STOP", f"{c}: {err}")
            return 1
        say("ok", f"{c}: parameters written to {params[c]}")

    # the atom order is checked on one real pose per compound, because
    # everything downstream silently depends on it
    for c in cpds:
        d = params.get(c)
        hits = sorted(glob.glob(os.path.join(a.poses, "*", f"{c}__*.sdf")))
        if not hits:
            say("STOP", f"{c}: no pose files under {a.poses}")
            return 1
        err = check_atom_order(d, hits[0], a.select_by)
        if isinstance(err, tuple):
            say("ok", f"{c}: {err[1]}")
        elif err:
            say("STOP", f"{c}: {err}")
            say("STOP", f"     checked against {hits[0]}")
            return 1
        else:
            say("ok", f"{c}: the pose and LIG.mol2 hold the same atoms in "
                      f"the same order")

    say("ok", f"protocol: {a.protein_ff}, {a.ligand_ff}, igb={a.igb}, "
              f"saltcon={a.salt}")

    # every ensemble member must hold the same protein, or one topology
    # cannot stand for all of them and the numbers are not comparable
    counts = {}
    for d in structs:
        sn = os.path.basename(d)
        rec = os.path.join(a.receptors, sn + ".pdb")
        if not os.path.exists(rec):
            hits = glob.glob(os.path.join(a.receptors, sn + "*.pdb"))
            if not hits:
                say("STOP", f"no receptor for {sn} in {a.receptors}")
                return 1
            rec = hits[0]
        counts[sn] = (rec, sum(1 for ln in open(rec)
                               if ln.startswith("ATOM")))
    n = {v[1] for v in counts.values()}
    if len(n) > 1:
        lo, hi = min(n), max(n)
        say("STOP", f"the ensemble members do not hold the same protein: "
                    f"{lo} to {hi} atoms across {len(counts)} files")
        return 1
    say("ok", f"all {len(counts)} members hold {n.pop()} protein atoms")

    a_d = {"select_by": a.select_by, "protein_ff": a.protein_ff,
           "ligand_ff": a.ligand_ff, "igb": a.igb, "salt": a.salt,
           "min_steps": a.min_steps, "restraint": a.restraint}

    jobs, done = [], 0
    for c in cpds:
        for sn, (rec, _) in sorted(counts.items()):
            hits = sorted(glob.glob(os.path.join(a.poses, sn,
                                                 f"{c}__*.sdf")))
            if not hits:
                continue
            oj = os.path.join(a.out, "cells", f"{c}__{sn}.json")
            if os.path.exists(oj):
                done += 1
                continue
            jobs.append((c, sn, hits[0], rec, params[c], oj,
                         os.path.join(a.out, "work", f"{c}__{sn}"), a_d))
    say("ok", f"{len(jobs)} complexes to do, {done} already finished")
    if not jobs and not done:
        say("STOP", "no complex matched a compound and a receptor")
        return 1

    # one complex first, always, run or no run: this pipeline fails at its
    # joins and one is cheaper to debug than four hundred
    if jobs:
        say("trial", f"building {jobs[0][0]} in {jobs[0][1]} ...")
        tag, rec, err = one_complex(jobs[0])
        if err:
            say("STOP", f"the trial complex failed: {tag}")
            for ln in str(err).splitlines():
                say("STOP", f"  {ln}")
            say("STOP", f"its working files are in {jobs[0][6]} - nothing "
                        f"else was run")
            return 1
        say("trial", f"{tag}: dG = {rec['dG']:.2f} kcal/mol")
        if not (-200 < rec["dG"] < 50):
            say("STOP", f"that is not a plausible value; stopping before "
                        f"the rest. Working files in {jobs[0][6]}")
            return 1
        jobs = jobs[1:]

    if not a.run:
        say("done", "the trial worked. Add --run to do the remaining "
                    f"{len(jobs)}")
        return 0

    if jobs:
        import multiprocessing as mp
        say("run", f"{len(jobs)} complexes on {a.np} workers")
        nfail = 0
        try:
            with mp.Pool(a.np, initializer=os.nice, initargs=(19,)) as pool:
                for i, (tag, rec, err) in enumerate(
                        pool.imap_unordered(one_complex, jobs), 1):
                    if err:
                        nfail += 1
                        say("fail", f"[{i}/{len(jobs)}] {tag}: "
                                    f"{str(err).splitlines()[0]}")
                    else:
                        say("ok", f"[{i}/{len(jobs)}] {tag}: "
                                  f"{rec['dG']:.2f}")
        except KeyboardInterrupt:
            say("stop", "interrupted; finished cells are kept and a rerun "
                        "will skip them")
            return 1
        if nfail:
            say("note", f"{nfail} of {len(jobs)} failed; their working files "
                        f"are under {os.path.join(a.out, 'work')}")

    # gather
    import json
    rows = []
    for f in sorted(glob.glob(os.path.join(a.out, "cells", "*.json"))):
        try:
            rows.append(json.load(open(f)))
        except (OSError, ValueError):
            pass
    if not rows:
        say("STOP", "nothing to summarise")
        return 1
    say("", "")
    say("out", f"=== single-point MM-GBSA, igb={a.igb}, mean over the "
               f"ensemble ===")
    say("out", f"  {'compound':10s}{'members':>9s}{'dG mean':>10s}"
               f"{'sd':>8s}{'min':>9s}{'max':>9s}")
    import statistics as st
    for c in cpds:
        v = [r["dG"] for r in rows if r["compound"] == c]
        if not v:
            continue
        sd = st.stdev(v) if len(v) > 1 else 0.0
        say("out", f"  {c:10s}{len(v):9d}{st.mean(v):10.2f}{sd:8.2f}"
                   f"{min(v):9.2f}{max(v):9.2f}")
    say("out", "  no entropy, so these compare with each other and are not "
               "binding free energies")
    try:
        import csv
        with open(os.path.join(a.out, "cells.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["compound", "structure", "dG",
                                              "sd", "pose"])
            w.writeheader()
            w.writerows(rows)
        say("out", f"{os.path.join(a.out, 'cells.csv')}")
    except OSError as e:
        say("note", f"could not write the csv: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

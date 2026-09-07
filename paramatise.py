"""
=============================================================================
prepare_system_colab.py  —  Pure AmberTools pipeline for Google Colab
=============================================================================
Inputs : ligand.pdb  (ligand),  receptor.pdb  (protein)
Outputs: SYS.pdb, SYS_gaff2.prmtop, SYS_gaff2.crd, SYS_gaff2.rst

Upload both files to /content/ before running.
=============================================================================
"""

import os, sys, subprocess, warnings
warnings.filterwarnings("ignore")

# ── USER CONFIG ───────────────────────────────────────────────────────────────
workDir     = "/path"
protein_pdb = "receptor.pdb"
ligand_pdb  = "ligand.pdb"
box_padding = 12.0
net_charge  = 0
# ─────────────────────────────────────────────────────────────────────────────

os.chdir(workDir)

# ── File paths ────────────────────────────────────────────────────────────────
initial_pdb    = os.path.join(workDir, protein_pdb)
lig_in         = os.path.join(workDir, ligand_pdb)
protein_fixed  = os.path.join(workDir, "protein_fixed.pdb")
lig_h          = os.path.join(workDir, "ligand_H.pdb")
lig_mol2       = os.path.join(workDir, "LIG.mol2")
lig_frcmod     = os.path.join(workDir, "LIG.frcmod")
protein_ligand = os.path.join(workDir, "protein_ligand.pdb")
lib_file       = os.path.join(workDir, "LIG.lib")
lig_leap_pdb   = os.path.join(workDir, "ligand_gaff.pdb")
tleap_in       = os.path.join(workDir, "tleap.in")
sys_pdb        = os.path.join(workDir, "SYS.pdb")
sys_prmtop     = os.path.join(workDir, "SYS_gaff2.prmtop")
sys_crd        = os.path.join(workDir, "SYS_gaff2.crd")
sys_rst        = os.path.join(workDir, "SYS_gaff2.rst")

def run(cmd, desc="", ignore_error=False):
    print(f"  $ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().split("\n"):
            print(f"    {line}")
    if r.returncode != 0 and not ignore_error:
        print(f"[ERROR] {desc} failed:")
        print(r.stderr[-3000:])
        sys.exit(1)
    return r.stdout

print("=" * 60)
print("  Amber System Preparation (Colab)")
print("=" * 60)

# =============================================================================
# STEP 1 — Fix protein with cpptraj prepareforleap (most tleap-compatible)
# =============================================================================
print("\n-- Step 1: Fixing protein with cpptraj prepareforleap --")

cpptraj_in = os.path.join(workDir, "prepareforleap.in")
with open(cpptraj_in, "w") as f:
    f.write(f"""parm {initial_pdb}
loadcrd {initial_pdb} name edited
prepareforleap crdset edited name from-prepareforleap \\
pdbout {protein_fixed} nowat noh
go
""")

run(f"cpptraj -i {cpptraj_in}", "cpptraj prepareforleap")

# Verify
if not os.path.exists(protein_fixed) or os.path.getsize(protein_fixed) == 0:
    # Fallback: pdb4amber
    print("  cpptraj output empty, trying pdb4amber fallback...")
    run(f"pdb4amber -i {initial_pdb} -o {protein_fixed} --nohyd --dry", "pdb4amber")

# ── Post-fix: correct HIE/HID directly in the output ─────────────────────────
# pdb4amber/cpptraj sometimes still writes HIE with HD1 present
from collections import defaultdict

with open(protein_fixed) as f:
    lines = f.readlines()

# Collect atoms per HIS residue
his_atoms = defaultdict(set)
his_resnames = {}
for line in lines:
    if line.startswith(("ATOM", "HETATM")):
        aname   = line[12:16].strip()
        resname = line[17:20].strip()
        chain   = line[21]
        resnum  = line[22:26].strip()
        if resname in ("HIS", "HIE", "HID", "HIP"):
            key = (chain, resnum)
            his_atoms[key].add(aname)
            his_resnames[key] = resname

# Decide correct name
his_correct = {}
for key, atoms in his_atoms.items():
    hd1 = "HD1" in atoms
    he2 = "HE2" in atoms
    if hd1 and he2:
        his_correct[key] = "HIP"
    elif hd1:
        his_correct[key] = "HID"
    elif he2:
        his_correct[key] = "HIE"
    else:
        his_correct[key] = "HIE"  # default

if his_correct:
    print("  Histidine assignments:")
    for key, name in sorted(his_correct.items()):
        orig = his_resnames.get(key, "HIS")
        if orig != name:
            print(f"    Chain {key[0]} Res {key[1]:>4s}: {orig} -> {name}")

# Rewrite with corrections
out_lines = []
for line in lines:
    if line.startswith(("ATOM", "HETATM")):
        aname   = line[12:16].strip()
        resname = line[17:20].strip()
        chain   = line[21]
        resnum  = line[22:26].strip()
        key     = (chain, resnum)

        if resname in ("HIS", "HIE", "HID", "HIP") and key in his_correct:
            correct = his_correct[key]
            line = line[:17] + correct + line[20:]
            if correct == "HIE" and aname == "HD1":
                continue
            if correct == "HID" and aname == "HE2":
                continue

        # Drop bare "H" on N-terminal residue 1 (PDBFixer/cpptraj artefact)
        if aname == "H" and resnum.strip() == "1":
            continue

        out_lines.append(line)
    else:
        out_lines.append(line)

with open(protein_fixed, "w") as f:
    f.writelines(out_lines)

# Final validation
bad_hie = []
with open(protein_fixed) as f:
    for line in f:
        if line.startswith(("ATOM", "HETATM")):
            if line[17:20].strip() == "HIE" and line[12:16].strip() == "HD1":
                bad_hie.append(line[22:26].strip())

if bad_hie:
    print(f"  [ERROR] HIE with HD1 still present at residues: {bad_hie}")
    sys.exit(1)
else:
    print("  Protein fixed and validated: no HIE+HD1 conflicts")
# Add this BEFORE the antechamber step in op.py
# Replace the antechamber block with this:

print("\n-- Step 2: Preparing ligand --")

# pdb4amber to standardize ligand PDB
run(f"pdb4amber -i {lig_in} -o {lig_h}", "pdb4amber ligand", ignore_error=True)
if not os.path.exists(lig_h) or os.path.getsize(lig_h) == 0:
    print("  pdb4amber gave empty output, using original ligand PDB")
    lig_h = lig_in

# ── NEW: Clean ligand with OpenBabel to fix bad hydrogens ────────────────────
lig_clean = os.path.join(workDir, "ligand_clean.pdb")
print("  Cleaning ligand with OpenBabel...")

# Convert to mol2 and back to fix valence/hydrogen issues
lig_sdf = os.path.join(workDir, "ligand_clean.sdf")

r = subprocess.run(
    f"obabel {lig_h} -O {lig_sdf} --gen3d -h 2>/dev/null",
    shell=True, capture_output=True, text=True
)
if r.returncode != 0 or not os.path.exists(lig_sdf):
    # Try without --gen3d
    r = subprocess.run(
        f"obabel {lig_h} -O {lig_sdf} -h 2>/dev/null",
        shell=True, capture_output=True, text=True
    )

# Convert SDF back to PDB cleanly
r = subprocess.run(
    f"obabel {lig_sdf} -O {lig_clean} 2>/dev/null",
    shell=True, capture_output=True, text=True
)

if os.path.exists(lig_clean) and os.path.getsize(lig_clean) > 0:
    print("  OpenBabel cleaned ligand successfully")
    lig_h = lig_clean
else:
    print("  OpenBabel not available, trying RDKit fix...")
    # RDKit fix
    fix_script = f"""
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

mol = Chem.MolFromPDBFile('{lig_h}', removeHs=False, sanitize=False)
if mol is None:
    mol = Chem.MolFromPDBFile('{lig_h}', removeHs=True, sanitize=False)

try:
    rdmolops.SanitizeMol(mol)
except:
    pass

# Remove and re-add hydrogens properly
mol = Chem.RemoveHs(mol)
try:
    Chem.SanitizeMol(mol)
except:
    pass
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
AllChem.MMFFOptimizeMolecule(mol)

writer = Chem.PDBWriter('{lig_clean}')
writer.write(mol)
writer.close()
print("RDKit fix done")
"""
    r2 = subprocess.run(
        f"python3 -c \"{fix_script}\"",
        shell=True, capture_output=True, text=True
    )
    if os.path.exists(lig_clean) and os.path.getsize(lig_clean) > 0:
        print("  RDKit fixed ligand successfully")
        lig_h = lig_clean
    else:
        print("  WARNING: Could not auto-fix ligand, proceeding with original")

# ── Try antechamber with bcc charges first, fall back to gas ─────────────────
print("  Running antechamber...")

# First try with gaff2 charges
antechamber_cmd = (
    f"antechamber -i {lig_h} -fi pdb "
    f"-o {lig_mol2} -fo mol2 "
    f"-c bcc -s 2 -nc {net_charge} -at gaff2 -rn LIG"
)

r = subprocess.run(antechamber_cmd, shell=True,
                   capture_output=True, text=True)

if r.returncode != 0 or not os.path.exists(lig_mol2):
    print("  bcc charges failed, trying gas charges...")
    antechamber_cmd = (
        f"antechamber -i {lig_h} -fi pdb "
        f"-o {lig_mol2} -fo mol2 "
        f"-c gas -s 2 -nc {net_charge} -at gaff2 -rn LIG "
        f"-du y"   # ← allow dummy atoms
    )
    r = subprocess.run(antechamber_cmd, shell=True,
                       capture_output=True, text=True)

if r.returncode != 0:
    print(f"[ERROR] antechamber failed:")
    print(r.stderr[-3000:])
    sys.exit(1)

print(f"  Generated: LIG.mol2")
# =============================================================================
# STEP 2 — Prepare ligand with pdb4amber + antechamber + parmchk2
# =============================================================================
print("\n-- Step 2: Preparing ligand --")

# pdb4amber to standardize ligand PDB
run(f"pdb4amber -i {lig_in} -o {lig_h}", "pdb4amber ligand", ignore_error=True)
if not os.path.exists(lig_h) or os.path.getsize(lig_h) == 0:
    print("  pdb4amber gave empty output, using original ligand PDB")
    lig_h = lig_in

# antechamber with Gasteiger charges (fast, no QM)
run(
    f"antechamber -i {lig_h} -fi pdb "
    f"-o {lig_mol2} -fo mol2 "
    f"-c gas -s 2 -nc {net_charge} -at gaff2 -rn LIG",
    "antechamber"
)
print(f"  Generated: LIG.mol2")

# parmchk2
run(
    f"parmchk2 -i {lig_mol2} -f mol2 -o {lig_frcmod} -s gaff2",
    "parmchk2"
)
print(f"  Generated: LIG.frcmod")

# =============================================================================
# STEP 3 — First tleap pass: generate ligand lib + gaff pdb
# =============================================================================
print("\n-- Step 3a: tleap pass 1 (ligand lib) --")

with open(tleap_in, "w") as f:
    f.write(f"""source leaprc.protein.ff14SB
source leaprc.gaff2
LIG = loadmol2 {lig_mol2}
loadamberparams {lig_frcmod}
saveoff LIG {lib_file}
savepdb LIG {lig_leap_pdb}
quit
""")

run(f"tleap -f {tleap_in}", "tleap pass 1")

if not os.path.exists(lig_leap_pdb) or os.path.getsize(lig_leap_pdb) == 0:
    print("[ERROR] tleap did not produce ligand_gaff.pdb")
    print(f"  Check: cat {workDir}/leap.log")
    sys.exit(1)
print(f"  Generated: ligand_gaff.pdb and LIG.lib")

# =============================================================================
# STEP 3b — Combine protein + ligand into one PDB
# =============================================================================
print("\n-- Step 3b: Combining protein + ligand --")

# Read protein (strip END), append ligand, add END
with open(protein_fixed) as f:
    prot_lines = [l for l in f if not l.startswith("END")]

with open(lig_leap_pdb) as f:
    lig_lines = [l for l in f if not l.startswith(("REMARK", "END"))]

with open(protein_ligand, "w") as f:
    f.writelines(prot_lines)
    f.writelines(lig_lines)
    f.write("END\n")

print(f"  Combined PDB: protein_ligand.pdb")

# =============================================================================
# STEP 4 — Solvate with PACKMOL (fast) then finalize with tleap
# =============================================================================
print("\n-- Step 4: Solvating with PACKMOL (fast) --")
import numpy as np

# ── Get bounding box from protein_ligand.pdb ─────────────────────────────────
xs, ys, zs = [], [], []
with open(protein_ligand) as f:
    for line in f:
        if line.startswith(("ATOM", "HETATM")):
            xs.append(float(line[30:38]))
            ys.append(float(line[38:46]))
            zs.append(float(line[46:54]))

pad = box_padding
xmin, xmax = min(xs)-pad, max(xs)+pad
ymin, ymax = min(ys)-pad, max(ys)+pad
zmin, zmax = min(zs)-pad, max(zs)+pad
cx = (xmin+xmax)/2; cy = (ymin+ymax)/2; cz = (zmin+zmax)/2
hx = (xmax-xmin)/2; hy = (ymax-ymin)/2; hz = (zmax-zmin)/2

box_vol_A3 = (xmax-xmin) * (ymax-ymin) * (zmax-zmin)
# Water density ~0.0334 molecules/A^3, subtract ~solute volume roughly
n_solute_atoms = len(xs)
solute_vol = n_solute_atoms * 20.0   # rough A^3 per atom
n_water = int((box_vol_A3 - solute_vol) * 0.0334)
n_water = max(n_water, 1000)
print(f"  Box: {xmax-xmin:.1f} x {ymax-ymin:.1f} x {zmax-zmin:.1f} A")
print(f"  Placing {n_water} water molecules...")

# ── Write TIP3P water monomer PDB ─────────────────────────────────────────────
water_pdb = os.path.join(workDir, "water.pdb")
with open(water_pdb, "w") as f:
    f.write("""ATOM      1  O   WAT     1       0.000   0.000   0.000  1.00  0.00           O
ATOM      2  H1  WAT     1       0.957   0.000   0.000  1.00  0.00           H
ATOM      3  H2  WAT     1      -0.240   0.927   0.000  1.00  0.00           H
END
""")

# ── Write PACKMOL input ───────────────────────────────────────────────────────
packmol_in  = os.path.join(workDir, "packmol.inp")
solvated_pdb = os.path.join(workDir, "solvated.pdb")

with open(packmol_in, "w") as f:
    f.write(f"""tolerance 2.0
filetype pdb
output {solvated_pdb}
nloop 10

structure {protein_ligand}
  number 1
  fixed {cx:.3f} {cy:.3f} {cz:.3f} 0.0 0.0 0.0
  centerofmass
end structure

structure {water_pdb}
  number {n_water}
  inside box {xmin:.3f} {ymin:.3f} {zmin:.3f} {xmax:.3f} {ymax:.3f} {zmax:.3f}
end structure
""")

run(f"packmol < {packmol_in}", "packmol")
box_x = xmax - xmin
box_y = ymax - ymin
box_z = zmax - zmin

# =============================================================================
# STEP 4b — tleap parameters only
# =============================================================================
print("\n-- Step 4b: tleap (parameters only) --")

box_x = xmax - xmin
box_y = ymax - ymin
box_z = zmax - zmin

with open(tleap_in, "w") as f:
    f.write(f"""source leaprc.protein.ff14SB
source leaprc.water.tip3p
source leaprc.gaff2

loadamberparams {lig_frcmod}
loadoff {lib_file}

SYS = loadpdb {solvated_pdb}

set SYS box {{ {box_x:.3f} {box_y:.3f} {box_z:.3f} }}

addIons2 SYS Na+ 0
addIons2 SYS Cl- 0
addIons2 SYS Na+ 30
addIons2 SYS Cl- 30

savepdb SYS {sys_pdb}
saveamberparm SYS {sys_prmtop} {sys_crd}

quit
""")

run(f"tleap -f {tleap_in}", "tleap parameters")
# =============================================================================
# STEP 5 — Convert .crd -> .rst with cpptraj
# =============================================================================
print("\n-- Step 5: Generating .rst file --")

rst_cpptraj = os.path.join(workDir, "make_rst.cpptraj")
with open(rst_cpptraj, "w") as f:
    f.write(f"""parm {sys_prmtop}
trajin {sys_crd}
trajout {sys_rst} restart
run
quit
""")

run(f"cpptraj -i {rst_cpptraj}", "cpptraj rst")

# =============================================================================
# STEP 6 — Final verification
# =============================================================================
print("\n-- Step 6: Final verification --")
all_ok = True
for fpath, fname in [(sys_pdb,    "SYS.pdb"),
                     (sys_prmtop, "SYS_gaff2.prmtop"),
                     (sys_crd,    "SYS_gaff2.crd"),
                     (sys_rst,    "SYS_gaff2.rst")]:
    if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
        print(f"  OK  {fname}  ({os.path.getsize(fpath):,} bytes)")
    else:
        print(f"  MISSING  {fname}")
        all_ok = False

# Cleanup temp files
run("rm -f ANTECHAMBER* ATOMTYPE* sqm.* *.sh 2>/dev/null", ignore_error=True)

print("\n" + "=" * 60)
if all_ok:
    print("  ALL FILES GENERATED SUCCESSFULLY")
else:
    print("  WARNING: Some files missing — check leap.log")
print("=" * 60)

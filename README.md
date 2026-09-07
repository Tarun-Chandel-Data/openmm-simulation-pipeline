# MD Pipeline (OpenMM + AmberTools)

MD pipeline for a small-molecule/protein system, used to study binding stability of ligand **WTFA/LIG**. System prep via AmberTools (tleap, antechamber), simulation via OpenMM, analysis via MDTraj/cpptraj/MMPBSA.py.

This repository contains **pipeline code only** — no trajectories, structures, or other simulation data are tracked (see `.gitignore`).

## 1. Environment Setup

```bash
conda create -n openmm_env python=3.11
conda activate openmm_env
conda install -c conda-forge ambertools=22
pip install -r requirements.txt
```

AmberTools provides `tleap`, `antechamber`, `sqm`, `sander`, `cpptraj` — required for steps 2 and 6 below. Verify with:
```bash
which tleap antechamber cpptraj
```
## 2. Change to user working directory
```bash
chmod +x setup_paths.sh
./setup_paths.sh
```

## 3. Ligand Parameterisation (AM1-BCC / GAFF2)

```bash
python paramatise.py
```
Incorporate in it the prepare leap.in, packmol.in, and at last the tleap.in as a continuous process.


## 4. Equilibration (OpenMM)

```bash
python EQUILLIBRATION.py
```

- Loads `SYS_gaff2.prmtop` / `SYS_gaff2.crd`
- Two-stage minimization (stiff restraints, then relaxed restraints)
- NVT warmup (1 ns) with position restraints on protein/ligand heavy atoms
- NPT equilibration (1 ns), restraints released, barostat activated
- Outputs: `prot_lig_equil.rst`, `prot_lig_equil.pdb`, `prot_lig_equil.dcd`, `prot_lig_equil.log`

Edit `NVT_Time_ns` / `NPT_Time_ns` / `Integration_timestep` at the top for a longer/shorter equilibration.

## 5. Production (OpenMM)

```bash
python PRODUCTION.py
```

- Loads `SYS_gaff2.prmtop` / `SYS_gaff2.crd`, resumes from `prot_lig_equil.rst`
- Uses Hydrogen Mass Repartitioning (3.0 amu) for a 4 fs timestep
- Set `REPLICA_ID` (1, 2, or 3) before each run — controls the velocity-randomization seed so replicate trajectories diverge
- Supports resuming/chunked runs via `Number_of_strides` (skips strides whose `.rst` file already exists)
- Outputs per stride: `{Jobname}_rep{ID}_{n}.dcd/.log/.rst/.pdb`

Edit `Stride_Time`, `Number_of_strides`, `Temperature`, `Pressure` at the top for your production length/conditions. Run once per replica by changing `REPLICA_ID`.

## 6. Post-processing: Strip Water/Ions for Analysis (cpptraj)

```bash
cpptraj -i cpptraj.in > cpptraj.log
```

Strips water and ions from the trajectory and re-images it, producing `nopbc.prmtop` / `nopbc.xtc` — required by the RMSD/RMSF and H-bond scripts below. Edit the `trajin` line in `cpptraj.in` to point at whichever replica/stride `.dcd` file you want analyzed.

## 7. Analysis

**RMSD / RMSF** (`rms.py`):
```bash
python rms.py
```
Loads `nopbc.prmtop` / `nopbc.xtc`. Aligns on protein, computes protein+ligand RMSD over time and per-residue RMSF. Edit `protein_indices` (`resid 0 to 300`) to match your receptor's actual residue range, and `ligand_indices` (`resname LIG`) to your ligand's residue name. Outputs `rmsd_500.png`, `rmsf_500.png`.



**H-bond analysis** (`hbond_analysis.py`):
```bash
python hbond_analysis.py
```
Loads `nopbc.prmtop`/`nopbc.xtc` plus the original `SYS_gaff2.prmtop` (for correct residue numbering, since stripping shifts indices). Identifies ligand–protein H-bonds (Baker-Hubbard criterion), reports per-residue occupancy %, and plots interacting-residue counts over time plus the top 5 residue contact timelines. Outputs `hbond_replica.png`. Edit the `'LIG'` residue name check if your ligand uses a different code.

**MM-GBSA binding free energy**:
below command 288 is the number of resiude you ligand is, identify it by uploading any pdb in pymol to check its ligand residue number or Check by this command:
```bash
cpptraj -p SYS_gaff2.prmtop <<EOF
resinfo
EOF
| grep -iE "LIG|UNK|UNL"
```
Put that residue number of ligand in place of 288 in below command
```bash
ante-MMPBSA.py -p nopbc.prmtop -n ":288" -c mmgbsa_complex.prmtop -r mmgbsa_receptor.prmtop -l mmgbsa_ligand.prmtop -s ":WAT,Na+,Cl-,K+,CL,NA"
```
Quality check: number of atoms in your liagnd cross check it.
```bash
for f in complex.prmtop receptor.prmtop ligand.prmtop; do echo -n "$f: "; sed -n '7p' $f | awk '{print $1}'; done
```
Multicore run: in this case running on 12 simultaneously, if you have less please decrease the 12 to 2-6 accordingly
```bash
mpirun -np 12 MMPBSA.py.MPI -O -i mmgbsa.in \
-o MMGBSA.dat \
-do MMGBSA_decomp.dat \
-sp nopbc.prmtop \
-cp mmgbsa_complex.prmtop \
-rp mmgbsa_receptor.prmtop \
-lp mmgbsa_ligand.prmtop \
-y nopbc.xtc
```

## Example Output

Representative plots from a 500 ns production run (protein RMSD stabilizes ~1.5-2 Å, ligand RMSD ~0.7-1.2 Å; one flexible loop region shows RMSF >5 Å; ligand forms a dominant contact with ASP148, secondary with LYS35):
RMSD:
<img width="6000" height="3000" alt="rmsd_500" src="https://github.com/user-attachments/assets/e936ff65-6d06-48c8-846a-50ae8be1e7f9" />

RMSF:
<img width="10000" height="5000" alt="rmsf_500" src="https://github.com/user-attachments/assets/6a1810ce-f409-41f1-a046-714b7f7df2cc" />

H bond analysis:
<img width="12000" height="10000" alt="hbond_replica_3" src="https://github.com/user-attachments/assets/ad00d804-5e55-49ee-9645-f6c5dc1e7f54" />





## What to Change for a New System

| File | What to edit |
|---|---|
| `tleap.in` | Receptor PDB, ligand mol2/frcmod, ligand residue name|
| `EQUILLIBRATION.py` / `PRODUCTION.py` | `workDir` path, `Jobname`, simulation length/conditions |
| `cpptraj.in` | `trajin` filename, strip mask if using different ion names |
| `rms.py` | `protein_indices` residue range, `ligand_indices` residue name, `total_time_ns` |
| `hbond_analysis.py` | ligand residue name (`'LIG'`) |
| `mmgbsa.in` | receptor/ligand residue ranges matching your topology |

## Known Gotchas

- `workDir` is hardcoded as an absolute path in `EQUILLIBRATION.py`/`PRODUCTION.py`/`rmsF.py` — update it to your own machine's path before running.

# MD Pipeline (OpenMM + AmberTools)

MD pipeline for a small-molecule/protein system (PDB 5LQF, chain A), used to study binding stability of ligand **WTFA/LIG**. System prep via AmberTools (tleap, antechamber), simulation via OpenMM, analysis via MDTraj/cpptraj/MMPBSA.py.

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

## 2. Ligand Parameterization (AM1-BCC / GAFF2)

```bash
antechamber -i ligand.pdb -fi pdb -o WTFA_gaff2.mol2 -fo mol2 -c bcc -at gaff2 -rn WTFA
parmchk2 -i WTFA_gaff2.mol2 -f mol2 -o WTFA.frcmod -s gaff2
```
Inspect/clean the mol2 file for duplicate bonds or atom naming issues if tleap complains later (see `check_mol2_*.py` helper scripts).

## 3. System Build (tleap)

```bash
tleap -f tleap.in
```

`tleap.in` builds, in order:
- `ligand.prmtop` / `ligand.inpcrd` — ligand alone
- `receptor.prmtop` / `receptor.inpcrd` — receptor alone
- `complex.prmtop` / `complex.inpcrd` — unsolvated complex (used only as a residue-numbering reference later, **not** for simulation)
- `complex_solv.prmtop` / `complex_solv.inpcrd` — solvated (OPC water, truncated octahedron, 12 Å buffer), neutralized + ~0.15 M NaCl. **This is the pair actually used for simulation.**

Before running, fill in the ion count: solvate first, check the log for the number of water residues added, then set
```
N_ions = round(0.15 * N_waters / 55.34)
```
and replace `<N>` in the `addIonsRand` lines.

Edit for a new system: swap `5lqf_fixed_receptor.pdb`, `WTFA_gaff2_cleaned.mol2`, and `WTFA.frcmod` for your own receptor/ligand, and change the `WTFA` residue name throughout if your ligand uses a different code.

## 4. Equilibration (OpenMM)

```bash
python EQUILLIBRATION.py
```

- Loads `complex_solv.prmtop` / `complex_solv.inpcrd`
- Two-stage minimization (stiff restraints, then relaxed restraints)
- NVT warmup (1 ns) with position restraints on protein/ligand heavy atoms
- NPT equilibration (1 ns), restraints released, barostat activated
- Outputs: `prot_lig_equil.rst`, `prot_lig_equil.pdb`, `prot_lig_equil.dcd`, `prot_lig_equil.log`

Edit `NVT_Time_ns` / `NPT_Time_ns` / `Integration_timestep` at the top for a longer/shorter equilibration.

## 5. Production (OpenMM)

```bash
python PRODUCTION.py
```

- Loads `complex_solv.prmtop` / `complex_solv.inpcrd`, resumes from `prot_lig_equil.rst`
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

**RMSD / RMSF** (`rmsF.py`):
```bash
python rmsF.py
```
Loads `nopbc.prmtop` / `nopbc.xtc`. Aligns on protein, computes protein+ligand RMSD over time and per-residue RMSF. Edit `protein_indices` (`resid 0 to 286`) to match your receptor's actual residue range, and `ligand_indices` (`resname LIG`) to your ligand's residue name. Outputs `rmsd_500.png`, `rmsf_600.png`.

**H-bond analysis** (`hbond_analysis.py`):
```bash
python hbond_analysis.py
```
Loads `nopbc.prmtop`/`nopbc.xtc` plus the original `complex.prmtop` (for correct residue numbering, since stripping shifts indices). Identifies ligand–protein H-bonds (Baker-Hubbard criterion), reports per-residue occupancy %, and plots interacting-residue counts over time plus the top 5 residue contact timelines. Outputs `hbond_replica_3.png`. Edit the `'LIG'` residue name check if your ligand uses a different code.

**MM-GBSA binding free energy**:
```bash
MMPBSA.py -O -i mmgbsa.in -sp complex_solv.prmtop -cp complex.prmtop -rp receptor.prmtop -lp ligand.prmtop -y *.dcd
```

## Example Output

Representative plots from a 500 ns production run (protein RMSD stabilizes ~1.5-2 Å, ligand RMSD ~0.7-1.2 Å; one flexible loop region shows RMSF >5 Å; ligand forms a dominant contact with ASP148, secondary with LYS35):

- `rmsd_500.png`, `rmsf_600.png`, `hbond_replica_3.png`

## What to Change for a New System

| File | What to edit |
|---|---|
| `tleap.in` | Receptor PDB, ligand mol2/frcmod, ligand residue name, ion count `<N>` |
| `EQUILLIBRATION.py` / `PRODUCTION.py` | `workDir` path, `Jobname`, simulation length/conditions |
| `cpptraj.in` | `trajin` filename, strip mask if using different ion names |
| `rmsF.py` | `protein_indices` residue range, `ligand_indices` residue name, `total_time_ns` |
| `hbond_analysis.py` | ligand residue name (`'LIG'`) |
| `mmgbsa.in` | receptor/ligand residue ranges matching your topology |

## Known Gotchas

- `workDir` is hardcoded as an absolute path in `EQUILLIBRATION.py`/`PRODUCTION.py`/`rmsF.py` — update it to your own machine's path before running.
- `complex.prmtop` (unsolvated) and `complex_solv.prmtop` (solvated) are **not interchangeable** — simulation scripts need the solvated pair; the H-bond script's residue-numbering reference needs the unsolvated one.
- Large output files (trajectories, checkpoints, structures, logs, images) are excluded from version control via `.gitignore` by design.

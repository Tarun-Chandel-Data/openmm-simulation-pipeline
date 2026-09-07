# openmm-simulation-pipeline

# [Project Name] — OpenMM/AmberTools MD Pipeline

Molecular dynamics pipeline for protein–ligand simulations, using AmberTools (tleap, antechamber) for system preparation and OpenMM for simulation. Post-simulation analysis (RMSD/RMSF, clustering, H-bonds, convergence, MM-GBSA) is done with cpptraj/pytraj and MMPBSA.py.

## Requirements

- Python environment (conda/mamba recommended)
- AmberTools 22.0 (`conda install -c conda-forge ambertools=22`) — provides `tleap`, `antechamber`, `sqm`, `sander`
- Remaining Python packages: `pip install -r requirements.txt`

To recreate the environment:
```bash
conda create -n openmm_env python=3.11
conda activate openmm_env
conda install -c conda-forge ambertools=22
pip install -r requirements.txt
```

## Pipeline Overview

1. **Ligand parameterization** — generate AM1-BCC charges and GAFF2 parameters for the ligand
   ```bash
   antechamber -i ligand.pdb -fi pdb -o ligand_gaff2.mol2 -fo mol2 -c bcc -at gaff2
   parmchk2 -i ligand_gaff2.mol2 -f mol2 -o ligand.frcmod -s gaff2
   ```

2. **System build (tleap)** — build receptor, ligand, and solvated complex topologies
   ```bash
   tleap -f tleap.in
   ```
   Edit `tleap.in` to point at your own receptor PDB and ligand mol2/frcmod files. Produces `complex.prmtop`/`.inpcrd` and `complex_solv.prmtop`/`.inpcrd`.

3. **Equilibration (OpenMM)**
   ```bash
   python EQUILLIBRATION.py
   ```
   Edit input file paths and simulation length inside the script for your system.

4. **Production run (OpenMM)**
   ```bash
   python PRODUCTION.py
   ```
   Edit trajectory length, reporters, and output filenames as needed.

5. **Analysis**
   - RMSD/RMSF: `python rmsd_rmsf_analysis.py`
   - H-bond analysis: `python hbond_analysis.py`
   - Clustering: `python parse_clusters.py`
   - Convergence check: `python convergence_check.py` / `plot_convergence.py`
   - MM-GBSA binding free energy: `MMPBSA.py -O -i mmgbsa.in -sp complex_solv.prmtop -cp complex.prmtop -rp receptor.prmtop -lp ligand.prmtop -y *.dcd`

## What to change for a new system

- Replace the receptor PDB and ligand file referenced in `tleap.in`
- Update residue names in `tleap.in` if your ligand uses a different 3-letter code than `WTFA`
- Adjust simulation length, timestep, and restraints in `EQUILLIBRATION.py`/`PRODUCTION.py`
- Update `mmgbsa.in` residue ranges to match your new receptor/ligand atom numbering

## Notes

- Large trajectory/output files (`.dcd`, `.pdb` frames, `.prmtop`, logs, images) are intentionally excluded via `.gitignore` — this repo contains pipeline code only, not simulation data.
- Ion concentration in `tleap.in` is set for ~0.15 M NaCl; recalculate the ion count if you change water box size or model.

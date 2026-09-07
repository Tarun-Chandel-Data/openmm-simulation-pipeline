import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------
# 1. INPUTS & CONFIGURATION
# --------------------------
#path = get by pwd in your window paste it here inbetwwen 'path' below
workDir = '__PROJECT_DIR__'
top_file = os.path.join(workDir, "SYS_gaff2.prmtop")
traj_file = os.path.join(workDir, "nopbc.xtc")

total_time_ns = 500  #define your simulation time here
output_rmsd_img = "rmsd_500.png"
output_rmsf_img = "rmsf_500.png"

# --------------------------
# 2. LOADING & SELECTION
# --------------------------
print(f"> Loading trajectory: {traj_file}")
t = md.load(traj_file, top=top_file)
ref_t = t[0] # Reference is first frame

# Select Protein (Residues 0-275) and Ligand (HETATM)
protein_indices = t.topology.select("protein and resid 0 to 300")
ligand_indices = t.topology.select("resname LIG") # Adjust resname if different

# --------------------------
# 3. RMSD CALCULATION (Time-based)
# --------------------------
print("> Calculating RMSD...")
# Aligning to protein to ensure ligand RMSD is relative to protein pocket
t.superpose(ref_t, atom_indices=protein_indices)

rmsd_protein = md.rmsd(t, ref_t, atom_indices=protein_indices) * 10.0
rmsd_lig = md.rmsd(t, ref_t, atom_indices=ligand_indices) * 10.0

# Create Time Axis for 100ns
num_frames = len(rmsd_protein)
time_axis = np.linspace(0, total_time_ns, num_frames)

# --------------------------
# 4. RMSF CALCULATION (Residue-based)
# --------------------------
print("> Calculating RMSF...")
# RMSF requires alignment to the average structure or reference
t.superpose(ref_t, atom_indices=protein_indices)

# Calculate RMSF for all protein atoms in selection
rmsf_atoms = md.rmsf(t, ref_t, atom_indices=protein_indices) * 10.0

# Map atoms to residues for a cleaner plot
res_numbers = [t.topology.atom(i).residue.resSeq for i in protein_indices]
unique_res = np.unique(res_numbers)
rmsf_per_residue = []

for res in unique_res:
    # Average the RMSF of all atoms belonging to this residue
    res_mask = [i for i, r_num in enumerate(res_numbers) if r_num == res]
    rmsf_per_residue.append(np.mean(rmsf_atoms[res_mask]))

# --------------------------
# 5. PLOTTING RMSD
# --------------------------
plt.figure(figsize=(10, 5))
plt.plot(time_axis, rmsd_protein, label="Protein", color='royalblue', linewidth=1.2)
plt.plot(time_axis, rmsd_lig, label="Ligand", color='darkorange', linewidth=1.2)

plt.title(f"RMSD Trajectory over {total_time_ns} ns", fontsize=14)
plt.xlabel("Time (ns)", fontsize=12)
plt.ylabel("RMSD (Å)", fontsize=12)
plt.xlim(0, total_time_ns)
plt.ylim(0, 6) # Capped at 5A as requested
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(output_rmsd_img, dpi=600)
print(f"> Saved RMSD plot: {output_rmsd_img}")

# --------------------------
# 6. PLOTTING RMSF
# --------------------------
plt.figure(figsize=(10, 5))
plt.fill_between(unique_res, rmsf_per_residue, color='teal', alpha=0.2)
plt.plot(unique_res, rmsf_per_residue, color='teal', linewidth=1.5)

plt.title("Protein RMSF per Residue", fontsize=14)
plt.xlabel("Residue Number", fontsize=12)
plt.ylabel("Fluctuation (Å)", fontsize=12)
plt.xlim(unique_res[0], unique_res[-1])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_rmsf_img, dpi=1000) #define the qualoty in therm of dpi, more quality more dpi
print(f"> Saved RMSF plot: {output_rmsf_img}")

plt.show()

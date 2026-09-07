import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Trajectory (stripped, no-PBC)
prmtop = "nopbc.prmtop"
dcd = "nopbc.xtc"
print("Loading trajectory...")
traj = md.load(dcd, top=prmtop)

# 1b. Load ORIGINAL topology for correct residue numbering
orig_top = md.load_prmtop("complex.prmtop")

# 2. Identify Ligand by Name (LIG)
try:
    lig_res = [res for res in traj.topology.residues if res.name == 'LIG'][0]
    lig_idx = lig_res.index
    print(f"Found ligand '{lig_res.name}' at index {lig_idx}")
except IndexError:
    print("Error: Residue named 'LIG' not found in topology.")
    exit()

hbonds = md.baker_hubbard(traj, periodic=True)

# 3. Filter H-bonds: ONLY Ligand-Protein interactions
lig_atoms = [a.index for a in traj.topology.residue(lig_idx).atoms]
lig_hbonds = []
for bond in hbonds:
    involves_ligand = (bond[0] in lig_atoms or bond[2] in lig_atoms)
    is_internal = (bond[0] in lig_atoms and bond[2] in lig_atoms)
    if involves_ligand and not is_internal:
        lig_hbonds.append(bond)

# 4. Aggregate by Residue (using ORIGINAL numbering)
res_timelines = {}
for bond in lig_hbonds:
    prot_atom_idx = bond[0] if bond[0] not in lig_atoms else bond[2]

    # Get residue index from stripped traj, but label from ORIGINAL topology
    res_idx = traj.topology.atom(prot_atom_idx).residue.index
    orig_res_info = orig_top.residue(res_idx)
    res_label = f"{orig_res_info.name}{orig_res_info.resSeq}"

    da_dist = md.compute_distances(traj, [[bond[0], bond[2]]])
    presence = (da_dist < 0.35).flatten()

    if res_label not in res_timelines:
        res_timelines[res_label] = presence
    else:
        res_timelines[res_label] = np.logical_or(res_timelines[res_label], presence)

# 5. Prepare Data for Plotting & Sorting
hb_details = []
for res, timeline in res_timelines.items():
    hb_details.append({
        'Residue': res,
        'Occupancy%': np.mean(timeline) * 100,
        'Timeline': timeline
    })
hb_details = sorted(hb_details, key=lambda x: x['Occupancy%'], reverse=True)

# 6. Console Output
print("\n--- RESIDUE H-BOND OCCUPANCY REPORT ---")
for hb in hb_details:
    print(f"Residue: {hb['Residue']:10} | Total Occupancy: {hb['Occupancy%']:.2f}%")

# 7. Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot A: Total unique residues forming H-bonds per frame
total_res_per_frame = np.zeros(traj.n_frames)
for hb in hb_details:
    total_res_per_frame += hb['Timeline']
ax1.plot(total_res_per_frame, color='darkblue', alpha=0.7)
ax1.set_title("Total Number of Interacting Residues over Time")
ax1.set_ylabel("Residue Count")

# Plot B: Timeline of Top 5 specific Residue contacts
top_n = min(5, len(hb_details))
for i in range(top_n):
    timeline = hb_details[i]['Timeline']
    frames = np.where(timeline == True)[0]
    ax2.scatter(frames, [i]*len(frames), marker='|',
                label=f"{hb_details[i]['Residue']} ({hb_details[i]['Occupancy%']:.1f}%)")
ax2.set_yticks(range(top_n))
ax2.set_yticklabels([hb_details[i]['Residue'] for i in range(top_n)])
ax2.set_title("Top Residue-Ligand Occupancy Timelines (Grouped)")
ax2.set_xlabel("Frame")
ax2.set_ylabel("Residue")
ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

plt.tight_layout()
plt.savefig("hbond_replica_3.png", dpi=1000)
plt.show()

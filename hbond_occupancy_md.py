#!/usr/bin/env python3
"""
Hydrogen bond occupancy between a ligand and a protein over a trajectory,
with the angular test applied to every frame.

md.baker_hubbard applies its distance and angle criteria to decide which
triplets enter its list, and it reports each as present or absent for the
trajectory as a whole. Taking that list and then scoring each frame on the
donor-acceptor distance alone puts a distance-only criterion back into the
occupancy: a frame where the pair has rotated apart still counts, because the
heavy atoms remain close. The two numbers can differ by a factor of two, so
both are reported here and the difference is stated.

The criteria are the ones baker_hubbard itself uses, and the same ones the
docking side of this pipeline applies, so the occupancies and the docked
counts are commensurable:

    H...A <= --h-dist (0.25 nm)   and   D-H...A >= --angle (120 deg)

    python hbond_occupancy_md.py --top nopbc.prmtop --traj nopbc.dcd \
        --ligand LIG --label "CK2α′" --resid-offset 6 \
        --out hbond_a2 --dpi 1000
"""
import argparse, os, sys
import numpy as np

try:
    import mdtraj as md
except ImportError:
    sys.exit("needs mdtraj")
try:
    import matplotlib
    matplotlib.use("Agg")           # no display on a compute node
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("needs matplotlib")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})
INK = "#1a1a1a"
C_GEOM, C_DIST = "#2e5eaa", "#d1495b"
GREY = "#6c6c6c"


def triplets(top, lig_atoms):
    """Donor, hydrogen, acceptor triplets between the ligand and the protein.

    Both directions are enumerated. A donor is an N or O carrying at least one
    hydrogen; an acceptor is any N or O. Each (donor, H, acceptor) is a
    candidate whose geometry the frames then decide, rather than a bond
    asserted in advance.
    """
    ligset = set(lig_atoms)
    heavy = {}
    for at in top.atoms:
        if at.element.symbol in ("N", "O"):
            heavy[at.index] = at
    hyd = {}
    for b in top.bonds:
        a0, a1 = b[0], b[1]
        for h, d in ((a0, a1), (a1, a0)):
            if h.element.symbol == "H" and d.index in heavy:
                hyd.setdefault(d.index, []).append(h.index)

    out = []
    for di, dat in heavy.items():
        if di not in hyd:
            continue
        d_is_lig = di in ligset
        for ai, aat in heavy.items():
            if ai == di:
                continue
            a_is_lig = ai in ligset
            if d_is_lig == a_is_lig:
                continue                      # both sides, or neither
            if dat.residue.index == aat.residue.index:
                continue
            for hi in hyd[di]:
                out.append((di, hi, ai))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top", required=True, help="topology, e.g. nopbc.prmtop")
    p.add_argument("--traj", required=True, help="trajectory, e.g. nopbc.dcd")
    p.add_argument("--ligand", default="LIG",
                   help="residue name of the ligand")
    p.add_argument("--stride", type=int, default=1,
                   help="read every nth frame")
    p.add_argument("--h-dist", type=float, default=0.25,
                   help="H...acceptor cutoff in NM, baker_hubbard's default")
    p.add_argument("--angle", type=float, default=120.0,
                   help="minimum D-H...A angle in degrees")
    p.add_argument("--da-dist", type=float, default=0.35,
                   help="donor-acceptor cutoff in NM, used only for the "
                        "distance-only figure reported for comparison")
    p.add_argument("--resid-offset", type=int, default=0,
                   help="added to every residue number printed. A topology "
                        "built from a prmtop is renumbered from 1, so its "
                        "numbers do not match the pdb the pocket was defined "
                        "in; without the offset the labels name the wrong "
                        "residues")
    p.add_argument("--min-occupancy", type=float, default=1.0,
                   help="drop residues below this percent from the report")
    p.add_argument("--highlight", default="",
                   help="comma-separated residue numbers, in the numbering "
                        "AFTER the offset, drawn apart in the figure")
    p.add_argument("--label", default="", help="name of this system")
    p.add_argument("--periodic", action="store_true",
                   help="apply periodic boundaries. Leave off for a "
                        "trajectory already imaged, which is what a file "
                        "named nopbc usually is")
    p.add_argument("--top-n", type=int, default=6,
                   help="residues shown in the timeline panel")
    p.add_argument("--dpi", type=int, default=1000)
    p.add_argument("--out", default="hbond_occupancy")
    a = p.parse_args()

    log = []
    print(f"loading {a.traj} ...", flush=True)
    traj = md.load(a.traj, top=a.top, stride=a.stride)
    log.append(f"[in] {a.traj} with {a.top}")
    log.append(f"     {traj.n_frames} frames"
               + (f" (stride {a.stride})" if a.stride > 1 else "")
               + f", {traj.n_atoms} atoms")
    if a.periodic and traj.unitcell_vectors is None:
        sys.exit("--periodic given but the trajectory carries no unit cell; "
                 "drop the flag for an imaged trajectory")

    lig = [r for r in traj.topology.residues if r.name == a.ligand]
    if not lig:
        names = sorted({r.name for r in traj.topology.residues})
        sys.exit(f"no residue named '{a.ligand}'. Residue names present: "
                 + ", ".join(names[:40]))
    if len(lig) > 1:
        log.append(f"     [note] {len(lig)} residues named {a.ligand}; all "
                   f"are treated as the ligand")
    lig_atoms = [at.index for r in lig for at in r.atoms]
    log.append(f"     ligand {a.ligand}: {len(lig_atoms)} atoms")

    tri = triplets(traj.topology, lig_atoms)
    if not tri:
        sys.exit("no donor-hydrogen-acceptor triplet spans the ligand and the "
                 "protein. If the topology has no hydrogens, the angular test "
                 "cannot be applied at all")
    log.append(f"     {len(tri)} candidate donor-H...acceptor triplets")

    tri = np.asarray(tri)
    # H...A distance, D-H...A angle, and the donor-acceptor distance the
    # distance-only criterion would have used
    ha = md.compute_distances(traj, tri[:, [1, 2]], periodic=a.periodic)
    ang = np.degrees(md.compute_angles(traj, tri[:, [0, 1, 2]],
                                       periodic=a.periodic))
    da = md.compute_distances(traj, tri[:, [0, 2]], periodic=a.periodic)

    # a NaN angle comes of three atoms momentarily collinear or coincident;
    # it is not a satisfied bond, so it is treated as a failure
    geom = (ha <= a.h_dist) & (np.nan_to_num(ang, nan=0.0) >= a.angle)
    dist = da <= a.da_dist

    ligset = set(lig_atoms)
    labels = []
    for d, h, ac in tri:
        prot = ac if d in ligset else d
        r = traj.topology.atom(int(prot)).residue
        labels.append(f"{r.name}{r.resSeq + a.resid_offset}")
    labels = np.array(labels)

    # a residue is engaged in a frame if any of its triplets is satisfied
    res = {}
    for lab in np.unique(labels):
        k = labels == lab
        res[lab] = (geom[:, k].any(axis=1), dist[:, k].any(axis=1))

    rows = []
    for lab, (g, dd) in res.items():
        rows.append({"residue": lab,
                     "occupancy_geom": 100.0 * g.mean(),
                     "occupancy_dist": 100.0 * dd.mean(),
                     "n_triplet": int((labels == lab).sum()),
                     "_g": g, "_d": dd})
    rows.sort(key=lambda x: -x["occupancy_geom"])
    shown = [x for x in rows if x["occupancy_geom"] >= a.min_occupancy
             or x["occupancy_dist"] >= a.min_occupancy]

    log.append("")
    log.append(f"=== occupancy per residue"
               + (f", {a.label}" if a.label else "") + " ===")
    log.append(f"  geometry: H...A <= {a.h_dist*10:.1f} A and "
               f"D-H...A >= {a.angle:g} deg")
    log.append(f"  distance only: D...A <= {a.da_dist*10:.1f} A, reported for "
               f"comparison and not used as the result")
    if a.resid_offset:
        log.append(f"  residue numbers carry an offset of {a.resid_offset:+d}")
    log.append("")
    log.append(f"  {'residue':12s}{'geometry %':>12s}{'distance %':>12s}"
               f"{'ratio':>8s}{'triplets':>10s}")
    for x in shown:
        rat = (x["occupancy_geom"] / x["occupancy_dist"]
               if x["occupancy_dist"] > 0 else np.nan)
        log.append(f"  {x['residue']:12s}{x['occupancy_geom']:12.2f}"
                   f"{x['occupancy_dist']:12.2f}"
                   + (f"{rat:8.2f}" if np.isfinite(rat) else "       -")
                   + f"{x['n_triplet']:10d}")
    if len(rows) > len(shown):
        log.append(f"  ({len(rows)-len(shown)} residues below "
                   f"{a.min_occupancy:g}% on both, not shown)")

    tg = np.sum([x["_g"] for x in rows], axis=0) if rows else np.zeros(traj.n_frames)
    td = np.sum([x["_d"] for x in rows], axis=0) if rows else np.zeros(traj.n_frames)
    log.append("")
    log.append("=== how much the angular test removes ===")
    log.append(f"  residues engaged per frame: geometry "
               f"{tg.mean():.2f} +/- {tg.std():.2f}, "
               f"distance only {td.mean():.2f} +/- {td.std():.2f}")
    if td.mean() > 0:
        log.append(f"  the angular test keeps {100*tg.mean()/td.mean():.0f}% "
                   f"of what distance alone would count")
    sg = sum(x["occupancy_geom"] for x in rows)
    sd = sum(x["occupancy_dist"] for x in rows)
    if sd > 0:
        log.append(f"  summed over residues: {sg:.0f} vs {sd:.0f} percent-"
                   f"points, {100*sg/sd:.0f}% kept")

    hl = {x.strip() for x in a.highlight.split(",") if x.strip()}
    if hl:
        log.append("")
        log.append("=== the residues asked for ===")
        for want in sorted(hl, key=lambda s: (len(s), s)):
            hit = [x for x in rows if x["residue"].endswith(want)
                   and x["residue"][len(x["residue"])-len(want):] == want]
            if not hit:
                log.append(f"  residue {want}: no hydrogen bond candidate at "
                           f"all, so occupancy is 0 by construction rather "
                           f"than measured as low")
            for x in hit:
                log.append(f"  {x['residue']:12s}"
                           f"geometry {x['occupancy_geom']:6.2f}%   "
                           f"distance only {x['occupancy_dist']:6.2f}%")

    import csv
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + "_occupancy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["residue", "occupancy_geom_pct", "occupancy_dist_pct",
                    "n_triplet"])
        for x in rows:
            w.writerow([x["residue"], f"{x['occupancy_geom']:.4f}",
                        f"{x['occupancy_dist']:.4f}", x["n_triplet"]])
    np.savetxt(a.out + "_per_frame.csv",
               np.column_stack([np.arange(traj.n_frames), tg, td]),
               delimiter=",", header="frame,n_residue_geom,n_residue_dist",
               comments="", fmt=["%d", "%.0f", "%.0f"])

    # --- figure -------------------------------------------------------------
    top = shown[: a.top_n]
    fig = plt.figure(figsize=(6.6, 5.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.15], hspace=0.45)

    ax0 = fig.add_subplot(gs[0])
    x = np.arange(traj.n_frames)
    ax0.plot(x, td, lw=0.5, color=C_DIST, alpha=0.75,
             label=f"distance only (D–A ≤ {a.da_dist*10:.1f} Å)")
    ax0.plot(x, tg, lw=0.5, color=C_GEOM,
             label=f"with angle (≥ {a.angle:g}°)")
    ax0.set_ylabel("residues engaged", fontsize=7.5)
    ax0.set_ylim(0, max(1, td.max()) * 1.15)
    ax0.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left")
    ax0.set_title((a.label + "  " if a.label else "")
                  + "hydrogen bonds to the ligand, per frame",
                  loc="left", fontsize=8.5, pad=4)

    ax1 = fig.add_subplot(gs[1])
    y = np.arange(len(top))[::-1]
    ax1.barh(y + 0.19, [t["occupancy_dist"] for t in top], height=0.36,
             color=C_DIST, edgecolor="none")
    ax1.barh(y - 0.19, [t["occupancy_geom"] for t in top], height=0.36,
             color=C_GEOM, edgecolor="none")
    for yy, t in zip(y, top):
        ax1.text(t["occupancy_geom"] + 1, yy - 0.19,
                 f"{t['occupancy_geom']:.0f}", va="center", ha="left",
                 fontsize=6, color=INK)
    ax1.set_yticks(y)
    ax1.set_yticklabels([t["residue"] for t in top], fontsize=6.5)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("occupancy (% of frames)", fontsize=7.5)

    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    for i, t in enumerate(top):
        fr = np.where(t["_g"])[0]
        ax2.plot(fr, np.full(fr.shape, len(top) - 1 - i), marker="|",
                 ls="none", ms=2.4, mew=0.45, color=C_GEOM)
    ax2.set_yticks(np.arange(len(top))[::-1])
    ax2.set_yticklabels([t["residue"] for t in top], fontsize=6.5)
    ax2.set_ylim(-0.7, len(top) - 0.3)
    ax2.set_xlabel("frame", fontsize=7.5)
    ax2.set_title("frames in which the angular criterion is met",
                  loc="left", fontsize=7.5, pad=3)
    for ax in (ax0, ax1, ax2):
        ax.tick_params(labelsize=6.5)

    fig.savefig(a.out + ".png", dpi=a.dpi, bbox_inches="tight",
                pad_inches=0.12)
    plt.close(fig)

    text = "\n".join(log)
    print(text)
    with open(a.out + "_values.txt", "w") as f:
        f.write(text + "\n")
    print(f"\n[out] {a.out}.png  ({a.dpi} dpi)")
    print(f"      {a.out}_occupancy.csv")
    print(f"      {a.out}_per_frame.csv")
    print(f"      {a.out}_values.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
A compound by conformer matrix of one pose property, printed for a terminal.

Two tables are printed from the same poses. The first takes the pose the
scoring function ranked first and reads the property off it; the second takes
whichever pose has the best value of that property, ignoring the ranking. They
differ wherever the ranked pose is not the one with the best value, and the gap
between them is how much the pose score and the property disagree.

Each table holds a block per receptor, separated by a blank line, with the rows
in the same order throughout so a compound lines up down the page. The columns
are each ensemble's own conformers in order: the first column of one block and
the first of the next are frames of different trajectories, not a pair.

The second table is bounded by what the pose file holds: if the poses were kept
down to the top few by pose score, the best value is the best among those kept,
not among every pose docked. The header says how many were available.

    python matrix_table.py --poses ens_cx_top8_poses.csv --receptor a2,a1 \
        --value minimizedAffinity --select ens_cx_a2_hinge.csv --min-either 17
"""
import argparse, os, sys
import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("needs pandas")

LOWER = ("minimizedaffinity", "vina", "energy", "rmsd")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--poses", required=True,
                   help="the *_poses.csv from seed_topn_stability.py")
    p.add_argument("--receptor", default="a2",
                   help="one or more, comma separated. Each becomes a block "
                        "in the same table, in this order; the first orders "
                        "the rows and every block keeps that order")
    p.add_argument("--value", default="minimizedAffinity")
    p.add_argument("--select",
                   help="a residue_occupancy csv to choose compounds from")
    p.add_argument("--min-either", type=int)
    p.add_argument("--only", help="comma-separated compounds instead")
    p.add_argument("--width", type=int, default=7,
                   help="characters per conformer column")
    p.add_argument("--decimals", type=int, default=2)
    p.add_argument("--out", help="write both tables here as text")
    a = p.parse_args()

    d = pd.read_csv(a.poses)
    for c in ("compound", "receptor", "seed", "pose_rank", a.value):
        if c not in d.columns:
            sys.exit(f"missing column '{c}'; have: {', '.join(d.columns)}")

    recs = [x.strip() for x in a.receptor.split(",") if x.strip()]
    have = set(d["receptor"].unique())
    gone = [r for r in recs if r not in have]
    if gone:
        sys.exit(f"no rows for receptor(s) {', '.join(gone)}; have: "
                 + ", ".join(sorted(have)))
    d = d[d["receptor"].isin(recs)]

    sel_note = ""
    if a.only:
        keep = [x.strip() for x in a.only.split(",") if x.strip()]
        d = d[d["compound"].isin(keep)]
        sel_note = "compounds named on the command line"
    elif a.select:
        s = pd.read_csv(a.select)
        col = f"{recs[0]}_either"
        if col not in s.columns:
            sys.exit(f"--select has no column '{col}'")
        if a.min_either is None:
            sys.exit("--select needs --min-either")
        d = d[d["compound"].isin(s.loc[s[col] >= a.min_either, "compound"])]
        sel_note = f"compounds with {col} >= {a.min_either}"
    if not len(d):
        sys.exit("no compound survived the selection")

    lower = any(m in a.value.lower() for m in LOWER)
    npose = d.groupby(["compound", "receptor", "seed"]).size()

    log = [f"[in] {a.poses}",
           f"     {d['compound'].nunique()} compounds; "
           + ", ".join(f"{r} {d.loc[d['receptor'] == r, 'seed'].nunique()} "
                       f"conformers" for r in recs),
           f"     value {a.value}, "
           + ("smaller is better" if lower else "larger is better"),
           f"     poses per compound per conformer: "
           f"{int(npose.min())} to {int(npose.max())}"]
    if sel_note:
        log.append(f"     selection: {sel_note}")
    if len(recs) > 1:
        log.append("     [note] the blocks share column headings but not "
                   "conformers: each ensemble's c00 is a frame of its own "
                   "trajectory, so a column is not a pair")
    if int(npose.max()) < 12:
        log.append(f"     [note] the file holds at most {int(npose.max())} "
                   f"poses per conformer, so the second table's best value is "
                   f"the best among those kept, not among every pose docked")

    def matrix(how, rec):
        q = d[d["receptor"] == rec]
        if how == "rank":
            g = q.sort_values("pose_rank").groupby(["compound", "seed"]).first()
        else:
            g = (q.sort_values(a.value, ascending=lower)
                  .groupby(["compound", "seed"]).first())
        return g[a.value].unstack("seed")

    def block(m, rec, nm, ncol, head):
        w, dc = a.width, a.decimals
        short = [c.split("_")[-1] if "_" in c else c for c in m.columns]
        out = []
        if head:
            out.append(" " * nm + "".join(f"{s:>{w}s}" for s in short)
                       + f"{'mean':>{w + 1}s}{'best':>{w}s}")
        out.append(f"{rec}")
        for cpd, row in m.iterrows():
            v = row.to_numpy(float)
            cells = "".join(" " * (w - 1) + "-" if np.isnan(x)
                            else f"{x:>{w}.{dc}f}" for x in v)
            best = np.nanmin(v) if lower else np.nanmax(v)
            out.append(f"{str(cpd):<{nm}s}{cells}"
                       f"{np.nanmean(v):>{w + 1}.{dc}f}{best:>{w}.{dc}f}")
        col = m.mean(axis=0).to_numpy(float)
        out.append(f"{'mean':<{nm}s}"
                   + "".join(f"{x:>{w}.{dc}f}" for x in col))
        return out

    def table(how, title, why):
        ms = {r: matrix(how, r) for r in recs}
        order = ms[recs[0]].mean(axis=1).sort_values(ascending=lower).index
        nm = max(9, max(len(str(x)) for x in order) + 1)
        ncol = max(len(m.columns) for m in ms.values())
        out = ["", f"=== {title} ===", f"    {why}", ""]
        for i, r in enumerate(recs):
            m = ms[r].reindex(order)
            out += block(m, r, nm, ncol, head=(i == 0))
            out.append("")
        return out, ms

    t1, m1 = table("rank",
                   f"{a.value} at the pose ranked first by pose score",
                   "one pose per conformer: the highest CNNscore")
    t2, m2 = table("value", f"best {a.value} in the kept poses",
                   "one pose per conformer: the best value, whatever its rank")
    log += t1 + t2

    log.append("=== how far the two differ ===")
    for r in recs:
        diff = (m2[r] - m1[r]).to_numpy(float)
        n_diff = int((~np.isclose(np.nan_to_num(diff), 0)).sum())
        log.append(f"    {r}: {n_diff} of {diff.size} cells differ; mean gap "
                   f"{np.nanmean(np.abs(diff)):.{a.decimals}f}, largest "
                   f"{np.nanmax(np.abs(diff)):.{a.decimals}f}")
    log.append(f"    a large gap means the pose ranked first by pose score is "
               f"often not the one")
    log.append(f"    with the best {a.value}")

    text = "\n".join(log)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(text + "\n")
        base = os.path.splitext(a.out)[0]
        for r in recs:
            m1[r].to_csv(f"{base}_{r}_byrank.csv")
            m2[r].to_csv(f"{base}_{r}_byvalue.csv")
        print(f"\n[out] {a.out}  (+ per-receptor csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

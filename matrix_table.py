#!/usr/bin/env python3
"""
A compound by conformer matrix of one pose property, printed for a terminal.

Two tables are printed from the same poses. The first takes the pose the
scoring function ranked first and reads the property off it; the second takes
whichever pose has the best value of that property, ignoring the ranking. They
differ wherever the ranked pose is not the one with the best value, and the
gap between them is how much the pose score and the property disagree.

The second table is bounded by what the pose file holds: if the poses were
kept down to the top few by pose score, the best value is the best among those
kept, not among every pose docked. The header says how many were available.

    python matrix_table.py --poses ens_cx_top8_poses.csv --receptor a2 \
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
    p.add_argument("--receptor", default="a2")
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
    d = d[d["receptor"] == a.receptor]
    if not len(d):
        sys.exit(f"no rows for receptor '{a.receptor}'; have: "
                 + ", ".join(sorted(pd.read_csv(a.poses)["receptor"].unique())))

    sel_note = ""
    if a.only:
        keep = [x.strip() for x in a.only.split(",") if x.strip()]
        d = d[d["compound"].isin(keep)]
        sel_note = "compounds named on the command line"
    elif a.select:
        s = pd.read_csv(a.select)
        col = f"{a.receptor}_either"
        if col not in s.columns:
            sys.exit(f"--select has no column '{col}'")
        if a.min_either is None:
            sys.exit("--select needs --min-either")
        keep = s.loc[s[col] >= a.min_either, "compound"]
        d = d[d["compound"].isin(keep)]
        sel_note = (f"compounds with {col} >= {a.min_either}")
    if not len(d):
        sys.exit("no compound survived the selection")

    lower = any(m in a.value.lower() for m in LOWER)
    confs = sorted(d["seed"].unique())
    npose = d.groupby(["compound", "seed"]).size()

    log = [f"[in] {a.poses}",
           f"     receptor {a.receptor}: {d['compound'].nunique()} compounds, "
           f"{len(confs)} conformers",
           f"     value {a.value}, "
           + ("smaller is better" if lower else "larger is better"),
           f"     poses per compound per conformer: "
           f"{int(npose.min())} to {int(npose.max())}"]
    if sel_note:
        log.append(f"     selection: {sel_note}")
    if int(npose.max()) < 12:
        log.append(f"     [note] the file holds at most {int(npose.max())} "
                   f"poses per conformer, so the second table's best value is "
                   f"the best among those kept, not among every pose docked")

    def matrix(how):
        if how == "rank":
            g = d.sort_values("pose_rank").groupby(["compound", "seed"]).first()
        else:
            g = (d.sort_values(a.value, ascending=lower)
                  .groupby(["compound", "seed"]).first())
        return g[a.value].unstack("seed")

    def show(m, title, why):
        w, dc = a.width, a.decimals
        nm = max(9, max(len(str(x)) for x in m.index) + 1)
        # short conformer names: the shared prefix says nothing per column
        short = [c.split("_")[-1] if "_" in c else c for c in m.columns]
        out = ["", f"=== {title} ===", f"    {why}", ""]
        out.append(" " * nm + "".join(f"{s:>{w}s}" for s in short)
                   + f"{'mean':>{w+1}s}{'best':>{w}s}")
        for cpd, row in m.iterrows():
            v = row.to_numpy(float)
            cells = "".join("      -" if np.isnan(x)
                            else f"{x:>{w}.{dc}f}" for x in v)
            best = np.nanmin(v) if lower else np.nanmax(v)
            out.append(f"{str(cpd):<{nm}s}{cells}"
                       f"{np.nanmean(v):>{w+1}.{dc}f}{best:>{w}.{dc}f}")
        col = m.mean(axis=0).to_numpy(float)
        out.append(f"{'mean':<{nm}s}"
                   + "".join(f"{x:>{w}.{dc}f}" for x in col))
        return out

    mr = matrix("rank")
    mv = matrix("value")
    # order both the same way, best mean first, so the two read side by side
    order = mr.mean(axis=1).sort_values(ascending=lower).index
    mr, mv = mr.loc[order], mv.loc[order]

    log += show(mr, f"{a.value} at the pose ranked first by pose score",
                "one pose per conformer: the highest CNNscore")
    log += show(mv, f"best {a.value} in the kept poses",
                "one pose per conformer: the best value, whatever its rank")

    diff = (mv - mr)
    log.append("")
    log.append("=== how far the two differ ===")
    log.append(f"    cells where they differ: "
               f"{int((~np.isclose(diff.to_numpy(float), 0, equal_nan=True)).sum())}"
               f" of {diff.size}")
    log.append(f"    mean gap: {np.nanmean(np.abs(diff.to_numpy(float))):.{a.decimals}f}"
               f"   largest: {np.nanmax(np.abs(diff.to_numpy(float))):.{a.decimals}f}")
    log.append("    a large gap means the pose the scoring function ranked "
               "first is often not the")
    log.append(f"    one with the best {a.value}")

    text = "\n".join(log)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(text + "\n")
        mr.to_csv(os.path.splitext(a.out)[0] + "_byrank.csv")
        mv.to_csv(os.path.splitext(a.out)[0] + "_byvalue.csv")
        print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

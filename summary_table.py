#!/usr/bin/env python3
"""
One row per compound: the hinge counts beside every score, for both subunits.

The hinge counts and the scores are produced by different steps and land in
different files, so comparing them means reading two tables against each
other. This joins them on the compound and prints one row, so a compound that
engages the hinge often but scores poorly, or the reverse, is visible in a
single line.

A compound present in one file and not the other is named rather than dropped,
since a silent inner join would shorten the table without saying so.

    python summary_table.py --hinge ens_vb_a2_hinge.csv \
        --stats ens_vb_rank_pose_unpaired.csv --residues TYR110,ILE111 \
        --offset 6 --top 30
"""
import argparse, os, sys
import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("needs pandas")

LOWER = ("minimizedaffinity", "vina", "energy")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hinge", required=True,
                   help="csv from residue_occupancy.py")
    p.add_argument("--stats", required=True,
                   help="*_unpaired.csv from seed_delta.py")
    p.add_argument("--receptor", default="a2",
                   help="the subunit the hinge counts come from")
    p.add_argument("--residues", required=True,
                   help="the two hinge labels as the hinge csv names them")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--values", default="CNNscore,minimizedAffinity,CNNaffinity",
                   help="score columns to carry, in order")
    p.add_argument("--rank-on", default="either",
                   help="either, both, or one of --values")
    p.add_argument("--label1", default="CK2α")
    p.add_argument("--label2", default="CK2α′")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--all", action="store_true")
    p.add_argument("--out")
    a = p.parse_args()

    h = pd.read_csv(a.hinge)
    s = pd.read_csv(a.stats)
    res = [x.strip() for x in a.residues.split(",") if x.strip()]
    if len(res) != 2:
        sys.exit("--residues wants exactly two labels")
    hcols = [f"{a.receptor}_{r}" for r in res] + \
            [f"{a.receptor}_either", f"{a.receptor}_both"]
    miss = [c for c in hcols if c not in h.columns]
    if miss:
        sys.exit(f"--hinge is missing: {', '.join(miss)}")

    vals = [v.strip() for v in a.values.split(",") if v.strip()]
    keep, dropped = [], []
    for v in vals:
        if f"{v}_a1_mean" in s.columns and f"{v}_a2_mean" in s.columns:
            keep.append(v)
        else:
            dropped.append(v)

    m = h[["compound"] + hcols].merge(
        s[["compound"] + [f"{v}_{k}_mean" for v in keep for k in ("a1", "a2")]],
        on="compound", how="outer", indicator=True)
    only_h = m.loc[m["_merge"] == "left_only", "compound"].tolist()
    only_s = m.loc[m["_merge"] == "right_only", "compound"].tolist()
    m = m[m["_merge"] == "both"].drop(columns="_merge")

    n = int(h[f"{a.receptor}_n_conformer"].median()) \
        if f"{a.receptor}_n_conformer" in h.columns else 0
    log = [f"[in] hinge {a.hinge}",
           f"     stats {a.stats}",
           f"     {len(m)} compounds in both"]
    if dropped:
        log.append(f"     [note] not in the stats file, left out: "
                   f"{', '.join(dropped)}")
    if only_h:
        log.append(f"     [note] {len(only_h)} only in the hinge file: "
                   + ", ".join(only_h[:6]) + (" ..." if len(only_h) > 6 else ""))
    if only_s:
        log.append(f"     [note] {len(only_s)} only in the stats file: "
                   + ", ".join(only_s[:6]) + (" ..." if len(only_s) > 6 else ""))
    if not len(m):
        sys.exit("no compound is in both files")

    key = (f"{a.receptor}_{a.rank_on}" if a.rank_on in ("either", "both")
           else f"{a.rank_on}_a2_mean")
    if key not in m.columns:
        sys.exit(f"--rank-on {a.rank_on!r} needs column {key!r}")
    asc = any(x in a.rank_on.lower() for x in LOWER)
    m = m.sort_values([key, f"{a.receptor}_both"], ascending=asc)

    def nm(x):
        num = "".join(c for c in x if c.isdigit())
        al = "".join(c for c in x if not c.isdigit())
        return f"{al.title()}{int(num) + a.offset}" if num else x

    log.append("")
    log.append(f"=== hinge engagement and scores, one row per compound ===")
    log.append(f"    hinge counts are conformers of {n} in {a.receptor}; "
               f"scores are the mean over the conformers")
    log.append(f"    ranked on {a.rank_on}")
    lo_names = [v for v in keep if any(x in v.lower() for x in LOWER)]
    if lo_names:
        log.append(f"    lower is better for: {', '.join(lo_names)}; "
                   f"higher for the rest")
    # two header lines: the group name over its pair, then the subunit under
    # each column, so a name does not have to be padded to the width of the
    # two numbers beneath it
    left = f"  {'':10s}" + " " * (9 * len(res)) + f"{'':8s}{'':6s}"
    h1 = left
    h2 = f"  {'compound':10s}" + "".join(f"{nm(r):>9s}" for r in res) \
        + f"{'either':>8s}{'both':>6s}"
    for v in keep:
        lo = any(x in v.lower() for x in LOWER)
        # the name only: a "lower is better" tag here is wider than the two
        # numbers under it and shifts every heading to its right
        h1 += "  " + f"{v[:16]:^16s}"
        h2 += "  " + f"{a.label2:>8s}{a.label1:>8s}"
    log.append(h1.rstrip())
    log.append(h2)
    for _, x in (m if a.all else m.head(a.top)).iterrows():
        line = f"  {x['compound']:10s}" \
            + "".join(f"{int(x[f'{a.receptor}_{r}']):9d}" for r in res) \
            + f"{int(x[f'{a.receptor}_either']):8d}" \
              f"{int(x[f'{a.receptor}_both']):6d}"
        for v in keep:
            line += ("  " + f"{x[f'{v}_a2_mean']:8.3f}"
                     + f"{x[f'{v}_a1_mean']:8.3f}")
        log.append(line)
    if not a.all and len(m) > a.top:
        log.append(f"    ... {len(m) - a.top} more, --all for every compound")

    log.append("")
    for v in keep:
        d = (m[f"{v}_a2_mean"] - m[f"{v}_a1_mean"])
        lo = any(x in v.lower() for x in LOWER)
        fav2 = int((d < 0).sum()) if lo else int((d > 0).sum())
        log.append(f"    {v}: {fav2} of {len(m)} favour {a.label2}, "
                   f"median difference {d.median():+.3f}"
                   + ("  (an energy: negative favours " + a.label2 + ")"
                      if lo else ""))

    text = "\n".join(log)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        m.to_csv(a.out, index=False)
        with open(os.path.splitext(a.out)[0] + ".txt", "w") as f:
            f.write(text + "\n")
        print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

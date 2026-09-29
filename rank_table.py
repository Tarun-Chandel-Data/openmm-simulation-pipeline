#!/usr/bin/env python3
"""
One ranked table from a seed_delta unpaired file.

That file already carries every criterion it was given, so the three separate
runs it takes to read them one at a time are unnecessary: this prints them
side by side, ranked on a consensus of all three rather than on any one.

Each criterion is turned into a rank, and the compound is placed by the mean
of its ranks. A compound leading on one criterion and trailing on another
lands in the middle, which is the honest position for it.

    python rank_table.py --file ens_cx_rank_pose_unpaired.csv --top 20
"""
import argparse, os, sys
import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("needs pandas")

NICE = {"CNNscore": "pose", "CNNaffinity": "CNNaff",
        "minimizedAffinity": "affinity", "n_hbond": "Hbond"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True,
                   help="a *_unpaired.csv from seed_delta.py")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--all", action="store_true", help="print every compound")
    p.add_argument("--min-z", type=float, default=2.0,
                   help="a criterion counts as clearing its own spread at "
                        "this |z|")
    p.add_argument("--out")
    a = p.parse_args()

    d = pd.read_csv(a.file)
    crit = [c[: -len("_delta")] for c in d.columns if c.endswith("_delta")]
    if not crit:
        sys.exit(f"no *_delta columns in {a.file}")

    # every delta is already oriented so positive favours the second receptor
    for c in crit:
        d[f"{c}_rank"] = d[f"{c}_delta"].rank(ascending=False, method="average")
    d["mean_rank"] = d[[f"{c}_rank" for c in crit]].mean(axis=1)
    d["n_pos"] = sum((d[f"{c}_delta"] > 0).astype(int) for c in crit)
    zc = [f"{c}_z" for c in crit if f"{c}_z" in d.columns]
    d["n_z"] = (sum((d[c].abs() >= a.min_z).astype(int) for c in zc)
                if zc else 0)
    d = d.sort_values("mean_rank")

    log = [f"[in] {a.file}",
           f"     {len(d)} compounds, criteria: "
           + ", ".join(NICE.get(c, c) for c in crit),
           f"     every delta is oriented so POSITIVE favours the second "
           f"receptor",
           f"     ranked on the mean of the per-criterion ranks; 'agree' "
           f"counts how many of the {len(crit)} are positive",
           ""]
    hdr = f"  {'#':>3s}  {'compound':10s}"
    for c in crit:
        hdr += f"{NICE.get(c, c)[:9]:>10s}"
    hdr += f"{'agree':>7s}{'|z|>=' + format(a.min_z, 'g'):>8s}"
    log.append(hdr)
    show = d if a.all else d.head(a.top)
    for i, (_, x) in enumerate(show.iterrows(), 1):
        line = f"  {i:3d}  {x['compound']:10s}"
        for c in crit:
            line += f"{x[f'{c}_delta']:+10.3f}"
        line += f"{str(int(x['n_pos'])) + '/' + str(len(crit)):>7s}"
        line += f"{int(x['n_z']):8d}"
        log.append(line)
    if not a.all and len(d) > a.top:
        log.append(f"  ... {len(d) - a.top} more, --all for every compound")

    allpos = d[d["n_pos"] == len(crit)]
    log.append("")
    if len(allpos):
        names = ", ".join(allpos["compound"].head(15))
        if len(allpos) > 15:
            names += " ..."
        log.append(f"  {len(allpos)} of {len(d)} have every criterion "
                   f"positive: {names}")
    else:
        log.append(f"  none of {len(d)} have every criterion positive")
    if zc:
        strong = allpos[allpos["n_z"] >= 1]
        log.append(f"  of those, {len(strong)} also clear |z| >= {a.min_z:g} "
                   f"on at least one: "
                   + (", ".join(strong["compound"]) if len(strong) else "none"))

    text = "\n".join(log)
    print(text)
    if a.out:
        d.to_csv(a.out, index=False)
        print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

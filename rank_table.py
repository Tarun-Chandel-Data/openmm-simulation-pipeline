#!/usr/bin/env python3
"""
Ranked table of the ensemble difference, with the spread it rests on.

For each compound and each subunit: the top pose of every conformer, averaged
over the conformers, with the standard deviation and the lowest and highest
conformer. Then the difference between the two averages, ranked.

The average alone does not say whether a difference means anything. A delta of
0.08 between two averages whose conformers range over 0.4 is inside the
ensemble's own spread, and the min and max columns are what show that, so they
are printed beside every average rather than summarised away.

    python rank_table.py --file ens_cx_rank_pose_unpaired.csv \
        --criteria CNNscore,minimizedAffinity --top 20
"""
import argparse, os, sys
import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("needs pandas")

NICE = {"CNNscore": "pose score", "CNNaffinity": "CNN affinity",
        "minimizedAffinity": "affinity", "n_hbond": "hydrogen bonds"}
LOWER = ("minimizedaffinity", "vina", "energy")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True,
                   help="a *_unpaired.csv from seed_delta.py")
    p.add_argument("--criteria", default="CNNscore,minimizedAffinity",
                   help="which to print, in order. One block each")
    p.add_argument("--label1", default="CK2α")
    p.add_argument("--label2", default="CK2α′")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--all", action="store_true")
    p.add_argument("--out")
    a = p.parse_args()

    d = pd.read_csv(a.file)
    want = [c.strip() for c in a.criteria.split(",") if c.strip()]
    have = [c for c in want if f"{c}_delta" in d.columns]
    missing = [c for c in want if c not in have]
    if not have:
        sys.exit(f"none of {', '.join(want)} in {a.file}. Present: "
                 + ", ".join(sorted({c[: -len('_delta')]
                                     for c in d.columns
                                     if c.endswith('_delta')})))

    log = [f"[in] {a.file}   {len(d)} compounds"]
    if missing:
        log.append(f"     [note] not in the file, skipped: "
                   f"{', '.join(missing)}")
    log.append(f"     each value is the top pose of a conformer, averaged "
               f"over the conformers")
    log.append(f"     delta = mean({a.label2}) - mean({a.label1}); "
               f"positive favours {a.label2}")

    for c in have:
        n1 = int(d[f"{c}_a1_n"].median()) if f"{c}_a1_n" in d.columns else 0
        n2 = int(d[f"{c}_a2_n"].median()) if f"{c}_a2_n" in d.columns else 0
        inv = any(m in c.lower() for m in LOWER)
        srt = d.sort_values(f"{c}_delta", ascending=False)
        show = srt if a.all else srt.head(a.top)
        log.append("")
        log.append(f"=== {NICE.get(c, c)} "
                   f"({n1} {a.label1} conformers, {n2} {a.label2}) ===")
        if inv:
            log.append(f"    an energy: the means are the raw negative "
                       f"values, and the delta is signed so positive still "
                       f"favours {a.label2}")
        log.append(f"  {'#':>3s} {'compound':9s}"
                   f"{a.label1 + ' avg':>11s}{'sd':>7s}{'min':>8s}{'max':>8s}"
                   f"{a.label2 + ' avg':>12s}{'sd':>7s}{'min':>8s}{'max':>8s}"
                   f"{'delta':>9s}")
        for i, (_, x) in enumerate(show.iterrows(), 1):
            line = f"  {i:3d} {x['compound']:9s}"
            for k in ("a1", "a2"):
                pre = 11 if k == "a1" else 12
                line += (f"{x[f'{c}_{k}_mean']:{pre}.3f}"
                         f"{x[f'{c}_{k}_sd']:7.3f}"
                         f"{x[f'{c}_{k}_min']:8.3f}"
                         f"{x[f'{c}_{k}_max']:8.3f}")
            line += f"{x[f'{c}_delta']:+9.3f}"
            log.append(line)
        if not a.all and len(d) > a.top:
            log.append(f"      ... {len(d) - a.top} more")
        # how the difference compares with the spread it came out of
        sp = (d[f"{c}_a1_max"] - d[f"{c}_a1_min"]).median()
        md = d[f"{c}_delta"].abs().max()
        log.append(f"    median conformer range within one subunit: "
                   f"{sp:.3f}; largest delta anywhere: {md:.3f}"
                   f"  ({md/sp:.2f}x the range)")

    if len(have) > 1:
        for c in have:
            d[f"{c}_rank"] = d[f"{c}_delta"].rank(ascending=False)
        d["mean_rank"] = d[[f"{c}_rank" for c in have]].mean(axis=1)
        d["n_pos"] = sum((d[f"{c}_delta"] > 0).astype(int) for c in have)
        srt = d.sort_values("mean_rank")
        log.append("")
        log.append(f"=== both together, ranked on the mean of the two ranks "
                   f"===")
        log.append(f"  {'#':>3s} {'compound':9s}"
                   + "".join(f"{NICE.get(c, c)[:10]:>12s}" for c in have)
                   + f"{'both +':>9s}")
        for i, (_, x) in enumerate((srt if a.all else srt.head(a.top))
                                   .iterrows(), 1):
            log.append(f"  {i:3d} {x['compound']:9s}"
                       + "".join(f"{x[f'{c}_delta']:+12.3f}" for c in have)
                       + f"{('yes' if x['n_pos'] == len(have) else 'no'):>9s}")
        both = srt[srt["n_pos"] == len(have)]
        log.append("")
        log.append(f"  {len(both)} of {len(d)} are positive on both: "
                   + (", ".join(both["compound"].head(20))
                      + (" ..." if len(both) > 20 else "")
                      if len(both) else "none"))

    text = "\n".join(log)
    print(text)
    if a.out:
        d.to_csv(a.out, index=False)
        print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

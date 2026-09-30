#!/usr/bin/env python3
"""
A few compounds compared across every matrix already written.

Finds the *_byrank.csv files a matrix_table run leaves behind, works out which
criterion and receptor each holds from its name, and prints the named compounds
side by side: each subunit's mean over the conformers and the difference
between them.

A substituent can change how tightly a compound binds without changing which
residues it hydrogen bonds, so a series that looks identical on a contact count
may still differ here. Both subunits are shown, because a change that improves
one equally is potency and only a change that improves one more than the other
is selectivity.

    python compare_series.py --dir ~/sim/small_molecule/cx \
        --only VB004,EV042,EV043
"""
import argparse, glob, os, re, sys
import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("needs pandas")

LOWER = ("affin", "energy", "vina")
NICE = {"aff": "minimizedAffinity", "pose": "CNNscore",
        "cnnaff": "CNNaffinity", "hbond": "n_hbond"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True,
                   help="directory holding the *_byrank.csv matrices")
    p.add_argument("--only", required=True,
                   help="comma-separated compounds, in the order to print")
    p.add_argument("--receptors", default="a2,a1")
    p.add_argument("--label1", default="CK2α")
    p.add_argument("--label2", default="CK2α′")
    a = p.parse_args()

    want = [x.strip() for x in a.only.split(",") if x.strip()]
    r2, r1 = [x.strip() for x in a.receptors.split(",")][:2]

    files = sorted(glob.glob(os.path.join(os.path.expanduser(a.dir),
                                          "*_byrank.csv")))
    if not files:
        sys.exit(f"no *_byrank.csv in {a.dir}")

    # group by everything before _<receptor>_byrank
    sets = {}
    for f in files:
        m = re.match(r"(.+)_(" + re.escape(r1) + "|" + re.escape(r2)
                     + r")_byrank\.csv$", os.path.basename(f))
        if not m:
            continue
        sets.setdefault(m.group(1), {})[m.group(2)] = f

    log = [f"[in] {a.dir}", f"     {len(sets)} matrices found"]
    rows = []
    for base, fs in sorted(sets.items()):
        if r1 not in fs or r2 not in fs:
            log.append(f"     [skip] {base}: only {', '.join(fs)}")
            continue
        crit = next((NICE[k] for k in NICE if k in base.lower()), base)
        lo = any(x in crit.lower() for x in LOWER)
        ens = "VB004 ensemble" if "_vb" in base.lower() else \
              "CX-4945 ensemble" if "_cx" in base.lower() else base
        d1, d2 = pd.read_csv(fs[r1], index_col=0), pd.read_csv(fs[r2], index_col=0)
        miss = [c for c in want if c not in d1.index or c not in d2.index]
        if miss:
            log.append(f"     [skip] {base}: missing {', '.join(miss)}")
            continue
        for c in want:
            m1 = float(np.nanmean(d1.loc[c].to_numpy(float)))
            m2 = float(np.nanmean(d2.loc[c].to_numpy(float)))
            dv = (m1 - m2) if lo else (m2 - m1)   # positive favours label2
            rows.append({"ensemble": ens, "criterion": crit, "compound": c,
                         r1: m1, r2: m2, "delta": dv, "base": base})
    if not rows:
        sys.exit("no matrix held all the named compounds")
    t = pd.DataFrame(rows)

    # two matrices can hold the same criterion and ensemble, differing only
    # in which compounds were selected when they were written. Grouping on the
    # criterion alone merges them and repeats every compound, which also makes
    # the trend check see zero differences. Identical rows are dropped first,
    # and what remains is grouped per file.
    before = len(t)
    t = t.drop_duplicates(subset=["ensemble", "criterion", "compound",
                                  r1, r2]).copy()
    if len(t) < before:
        log.append(f"     [note] {before - len(t)} duplicate rows dropped: "
                   f"the same criterion and ensemble appear in more than one "
                   f"matrix file")
    for (ens, crit), g in t.groupby(["ensemble", "criterion"], sort=True):
        lo = any(x in crit.lower() for x in LOWER)
        log.append("")
        log.append(f"=== {crit} — {ens} ===")
        log.append(f"  {'compound':10s}{a.label2:>10s}{a.label1:>10s}"
                   f"{'delta':>9s}   (positive favours {a.label2}"
                   + (", an energy so negated" if lo else "") + ")")
        g = g.set_index("compound").loc[[c for c in want if c in set(g.compound)]]
        for c, x in g.iterrows():
            log.append(f"  {c:10s}{x[r2]:10.3f}{x[r1]:10.3f}{x['delta']:+9.3f}")
        # does the difference move monotonically down the series as given?
        dv = g["delta"].to_numpy(float)
        if len(dv) >= 3:
            trend = ("rises" if np.all(np.diff(dv) > 0) else
                     "falls" if np.all(np.diff(dv) < 0) else "does not move "
                     "monotonically")
            log.append(f"  across the series as listed, the difference {trend}"
                       f"  ({' -> '.join(f'{v:+.3f}' for v in dv)})")

    text = "\n".join(log)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

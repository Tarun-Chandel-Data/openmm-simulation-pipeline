#!/usr/bin/env python3
"""
Compare two hydrogen bond tables for the same compounds.

Written to measure what a change in the counting rule did to results already
in hand: how many compounds moved, by how much, and whether anything that
depends on the counts changed order. A mean difference alone would not show
that, since a change applied evenly to every compound leaves every comparison
between them intact.

    python diff_hbond.py --old hb_a2.csv --new hb_a2_fixed.csv --tag a2
"""
import argparse, os, sys
import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("needs pandas")
try:
    from scipy.stats import spearmanr
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--id-col", default="cpd_id")
    p.add_argument("--tag", required=True,
                   help="the prefix the count columns carry, e.g. a2")
    p.add_argument("--col", help="count column; default <tag>_n_hbond")
    p.add_argument("--res-col",
                   help="residue-list column; default <tag>_hbond_res")
    p.add_argument("--top", type=int, default=20,
                   help="how many top-ranked compounds to compare")
    p.add_argument("--out")
    a = p.parse_args()

    col = a.col or f"{a.tag}_n_hbond"
    rcol = a.res_col or f"{a.tag}_hbond_res"
    o = pd.read_csv(a.old)
    n = pd.read_csv(a.new)
    for d, name in ((o, a.old), (n, a.new)):
        for c in (a.id_col, col):
            if c not in d.columns:
                sys.exit(f"{name} has no column '{c}'; has: "
                         f"{', '.join(d.columns[:12])}")

    m = o[[a.id_col, col] + ([rcol] if rcol in o.columns else [])].merge(
        n[[a.id_col, col] + ([rcol] if rcol in n.columns else [])],
        on=a.id_col, suffixes=("_old", "_new"))
    only_o = set(o[a.id_col]) - set(n[a.id_col])
    only_n = set(n[a.id_col]) - set(o[a.id_col])

    log = [f"[in] old {a.old}: {len(o)} rows",
           f"     new {a.new}: {len(n)} rows",
           f"     {len(m)} compounds in both"]
    if only_o:
        log.append(f"     [note] {len(only_o)} only in old, e.g. "
                   + ", ".join(sorted(map(str, only_o))[:5]))
    if only_n:
        log.append(f"     [note] {len(only_n)} only in new, e.g. "
                   + ", ".join(sorted(map(str, only_n))[:5]))
    if not len(m):
        sys.exit("no shared compounds")

    vo = m[f"{col}_old"].to_numpy(float)
    vn = m[f"{col}_new"].to_numpy(float)
    d = vn - vo
    m["change"] = d

    log.append("")
    log.append("=== how the counts moved ===")
    log.append(f"  old: median {np.median(vo):.2f}, mean {vo.mean():.2f}, "
               f"range {vo.min():.0f} to {vo.max():.0f}")
    log.append(f"  new: median {np.median(vn):.2f}, mean {vn.mean():.2f}, "
               f"range {vn.min():.0f} to {vn.max():.0f}")
    ch = int((d != 0).sum())
    log.append(f"  {ch} of {len(m)} compounds changed "
               f"({100*ch/len(m):.0f}%)")
    if ch:
        nz = d[d != 0]
        log.append(f"  among those: median {np.median(nz):+.2f}, "
                   f"mean {nz.mean():+.2f}, "
                   f"most negative {nz.min():+.0f}, "
                   f"most positive {nz.max():+.0f}")
        down = int((d < 0).sum())
        log.append(f"  {down} went down, {int((d > 0).sum())} went up")

    # a uniform shift changes every count but no comparison between compounds,
    # so the ranking is checked separately from the values
    log.append("")
    log.append("=== whether anything that rests on the counts changed ===")
    if HAVE_SCIPY and len(m) > 2:
        rho, pv = spearmanr(vo, vn)
        log.append(f"  rank correlation old vs new: rho = {rho:.3f} "
                   f"(r2 = {rho**2:.2f}), p = {pv:.2g}")
        if rho > 0.99:
            log.append("  -> the ordering is essentially unchanged; the fix "
                       "moves values, not conclusions drawn from their order")
        elif rho > 0.9:
            log.append("  -> the ordering is close but not identical; check "
                       "the top of the list below")
        else:
            log.append("  -> the ordering changed materially. Any result "
                       "stated as a ranking has to be re-read")
    else:
        log.append("  [note] scipy absent, so no rank correlation")

    to = set(m.nlargest(a.top, f"{col}_old")[a.id_col])
    tn = set(m.nlargest(a.top, f"{col}_new")[a.id_col])
    log.append(f"  top {a.top}: {len(to & tn)} of {a.top} compounds are in "
               f"both")
    gone, came = sorted(to - tn), sorted(tn - to)
    if gone:
        log.append(f"    dropped out: " + ", ".join(map(str, gone)))
    if came:
        log.append(f"    came in:     " + ", ".join(map(str, came)))

    if f"{rcol}_old" in m.columns and f"{rcol}_new" in m.columns:
        so = m[f"{rcol}_old"].fillna("").str.split()
        sn = m[f"{rcol}_new"].fillna("").str.split()
        same = sum(set(x) == set(y) for x, y in zip(so, sn))
        log.append("")
        log.append("=== which residues were named ===")
        log.append(f"  {same} of {len(m)} compounds name the same residue set")
        # a count that fell while the residue set held is the double count
        # being removed; a set that changed is a different finding
        fell_same = sum(1 for x, y, dd in zip(so, sn, d)
                        if set(x) == set(y) and dd < 0)
        log.append(f"  {fell_same} kept the same residues but a lower count, "
                   f"which is what removing a double count looks like")
        diff_set = sum(1 for x, y in zip(so, sn) if set(x) != set(y))
        if diff_set:
            log.append(f"  {diff_set} name a different residue set, which the "
                       f"counting rule alone does not explain; inspect these")
            ex = m[[set(x) != set(y) for x, y in zip(so, sn)]].head(3)
            for _, r in ex.iterrows():
                log.append(f"    {r[a.id_col]}: '{r[f'{rcol}_old']}' -> "
                           f"'{r[f'{rcol}_new']}'")

    log.append("")
    log.append("=== largest changes ===")
    big = m.reindex(m["change"].abs().sort_values(ascending=False).index)
    log.append(f"  {'compound':16s}{'old':>6s}{'new':>6s}{'change':>8s}")
    for _, r in big.head(12).iterrows():
        log.append(f"  {str(r[a.id_col]):16s}{r[f'{col}_old']:6.0f}"
                   f"{r[f'{col}_new']:6.0f}{r['change']:+8.0f}")

    text = "\n".join(log)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        m.to_csv(a.out + ".csv", index=False)
        with open(a.out + "_values.txt", "w") as f:
            f.write(text + "\n")
        print(f"\n[out] {a.out}.csv\n      {a.out}_values.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

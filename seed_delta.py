#!/usr/bin/env python3
"""
Whether a compound's preference between two receptors survives the docking
seed.

Reads the per-pose table seed_topn_stability.py writes. Within each seed the
two receptors were searched with the same seed, so their counts are paired and
the difference can be formed seed by seed. A compound whose difference keeps
the same sign in every seed is preferring one receptor; one whose difference
changes sign is not, whatever its mean says.

The paired difference is the right quantity because part of the noise is
common to both receptors within a seed and cancels. Comparing two means and
propagating their spreads does not use the pairing and overstates the
uncertainty.

    python seed_delta.py --poses validate/seed_top8_poses.csv \
        --a1 a1_isoform --a2 a2_proteinA
"""
import argparse, os, sys
import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("needs pandas")
try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--poses", required=True,
                   help="the *_poses.csv from seed_topn_stability.py")
    p.add_argument("--a1", required=True, help="receptor label to subtract")
    p.add_argument("--a2", required=True, help="receptor label subtracted from")
    p.add_argument("--label1", default="CK2α")
    p.add_argument("--label2", default="CK2α′")
    p.add_argument("--col", default="n_hbond")
    p.add_argument("--out")
    a = p.parse_args()

    d = pd.read_csv(a.poses)
    for need in ("compound", "receptor", "seed", a.col):
        if need not in d.columns:
            sys.exit(f"missing column '{need}'; have: "
                     f"{', '.join(d.columns)}")
    have = set(d["receptor"].unique())
    for r in (a.a1, a.a2):
        if r not in have:
            sys.exit(f"receptor '{r}' not in the table; have: "
                     f"{', '.join(sorted(have))}")

    # the mean over the poses kept for that compound, receptor and seed
    m = (d.groupby(["compound", "receptor", "seed"])[a.col]
          .mean().unstack("receptor"))
    if a.a1 not in m.columns or a.a2 not in m.columns:
        sys.exit("one receptor has no rows after grouping")
    m = m.dropna(subset=[a.a1, a.a2])
    m["delta"] = m[a.a2] - m[a.a1]
    per_seed = m.reset_index()

    log = [f"[in] {a.poses}",
           f"     {per_seed['compound'].nunique()} compounds, "
           f"{per_seed['seed'].nunique()} seeds",
           f"     delta = {a.col}({a.label2}) - {a.col}({a.label1}), "
           f"formed within each seed so the shared part of the noise cancels"]

    rows = []
    for cpd, g in per_seed.groupby("compound"):
        dv = g["delta"].to_numpy(float)
        n = len(dv)
        pos, neg = int((dv > 0).sum()), int((dv < 0).sum())
        # a sign that holds in every seed is the claim; the spread of the
        # difference says how far it could move. A difference that is zero in
        # every seed has no sign to be consistent about, which is not the same
        # as a sign that flips
        allzero = (pos == 0 and neg == 0)
        # a seed whose difference is exactly zero is a tie, not a reversal. A
        # compound with ties and no reversals has not been contradicted by any
        # seed, which is a weaker claim than every seed agreeing but not the
        # same as a sign that flips
        ties = n - pos - neg
        consistent = (not allzero) and ((pos == n) or (neg == n))
        reverses = (pos > 0 and neg > 0)
        sd = float(dv.std(ddof=1)) if n > 1 else np.nan
        row = {"compound": cpd, "n_seed": n, "all_zero": allzero,
               f"{a.a1}_mean": float(g[a.a1].mean()),
               f"{a.a2}_mean": float(g[a.a2].mean()),
               "delta_median": float(np.median(dv)),
               "delta_min": float(dv.min()), "delta_max": float(dv.max()),
               "delta_sd": sd, "n_pos": pos, "n_neg": neg, "n_tie": ties,
               "sign_consistent": consistent, "reverses": reverses}
        if HAVE_SCIPY and n >= 5 and not np.allclose(dv, 0):
            try:
                row["wilcoxon_p"] = float(wilcoxon(dv).pvalue)
            except ValueError:
                row["wilcoxon_p"] = np.nan
        rows.append(row)
    r = pd.DataFrame(rows).sort_values("delta_median", ascending=False)

    floor = float(r["delta_sd"].median())
    if floor == 0 or np.isnan(floor):
        log.append(f"     the paired difference does not vary by seed at all, "
                   f"so no noise floor can be estimated from it. Either the "
                   f"two receptors gave identical counts, or one seed was "
                   f"reused; check before reading the deltas as real")
    else:
        log.append(f"     median seed-to-seed sd of the paired difference: "
                   f"{floor:.2f} bonds. A |delta| below this is inside the "
                   f"noise")
    if not HAVE_SCIPY:
        log.append("     [note] scipy absent, so no signed-rank p values")
    else:
        ns = int(per_seed["seed"].nunique())
        if ns <= 6:
            # the smallest attainable two-sided signed-rank p is 2/2**n, so at
            # five seeds nothing can reach 0.05 however clean the result. The
            # column is printed because it was asked for, with what it can and
            # cannot show said plainly
            log.append(f"     [note] with {ns} seeds the smallest possible "
                       f"signed-rank p is {2/2**ns:.3f}, so no compound can "
                       f"reach 0.05 however consistent it is. Read the sign "
                       f"count, not the p value")

    log.append("")
    log.append(f"=== ranked on delta, {a.label2} minus {a.label1} ===")
    pcol = "wilcoxon_p" in r.columns
    log.append("  " + f"{'compound':10s}{a.label1:>9s}{a.label2:>9s}"
               + f"{'delta':>8s}{'sd':>7s}{'range':>14s}{'up/dn/=':>10s}"
               + (f"{'p':>9s}" if pcol else "") + "  verdict")
    for _, x in r.iterrows():
        dm, sd = x["delta_median"], x["delta_sd"]
        if x["all_zero"]:
            v = "identical in both, every seed"
        elif x["reverses"]:
            v = (f"sign reverses between seeds ({x['n_pos']} up, "
                 f"{x['n_neg']} down)")
        elif not x["sign_consistent"]:
            v = (f"{x['n_tie']} of {x['n_seed']} seeds show no difference, "
                 f"none reverse")
        elif floor == 0 or np.isnan(floor):
            v = "no noise estimate (differences do not vary by seed)"
        elif abs(dm) < floor:
            v = "inside the noise"
        elif abs(dm) < 2 * floor:
            v = f"above noise, under 2x"
        else:
            v = f"above 2x noise"
        log.append("  " + f"{x['compound']:10s}"
                   f"{x[f'{a.a1}_mean']:9.2f}{x[f'{a.a2}_mean']:9.2f}"
                   f"{dm:+8.2f}{sd:7.2f}"
                   f"{x['delta_min']:+7.2f}{x['delta_max']:+7.2f}"
                   f"{x['n_pos']:5d}/{x['n_neg']}/{x['n_tie']:<3d}"
                   + (f"{x['wilcoxon_p']:9.3f}" if pcol else "")
                   + "  " + v)

    cons = r[r["sign_consistent"]]
    log.append("")
    log.append(f"  {len(cons)} of {len(r)} compounds keep the same sign in "
               f"every seed")
    nr = r[(~r["sign_consistent"]) & (~r["reverses"]) & (~r["all_zero"])]
    if len(nr):
        log.append(f"  {len(nr)} more are never reversed but tie in at least "
                   f"one seed: " + ", ".join(nr["compound"]))
    rev = r[r["reverses"]]
    if len(rev):
        log.append(f"  {len(rev)} reverse: " + ", ".join(rev["compound"]))
    up = cons[cons["delta_median"] > 0]
    dn = cons[cons["delta_median"] < 0]
    log.append(f"    {len(up)} toward {a.label2}, {len(dn)} toward "
               f"{a.label1}")
    if floor == 0 or np.isnan(floor):
        log.append("  no noise floor, so no compound can be placed against "
                   "one")
    else:
        mag = cons["delta_median"].abs()
        strong = cons[mag >= 2 * floor]
        mid = cons[(mag >= floor) & (mag < 2 * floor)]
        log.append(f"  {len(strong)} clear twice the noise: "
                   + (", ".join(strong["compound"]) if len(strong) else "none"))
        log.append(f"  {len(mid)} clear the noise but not twice it: "
                   + (", ".join(mid["compound"]) if len(mid) else "none"))

    text = "\n".join(log)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        r.to_csv(a.out + ".csv", index=False)
        per_seed.to_csv(a.out + "_per_seed.csv", index=False)
        with open(a.out + "_values.txt", "w") as f:
            f.write(text + "\n")
        print(f"\n[out] {a.out}.csv\n      {a.out}_per_seed.csv"
              f"\n      {a.out}_values.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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


def unpaired(a, d, r1, r2):
    """Compare two receptors whose replicates are not partners.

    An ensemble taken from one trajectory has no frame-by-frame counterpart in
    an ensemble taken from another, so the difference cannot be formed replicate
    by replicate and none of the noise cancels. Each receptor's replicates are
    summarised on their own and the difference carries the two spreads added,
    which is wider than a paired difference would be. That is the honest width
    here, not a shortcoming of the arithmetic.
    """
    also = [x.strip() for x in a.also.split(",") if x.strip()]
    lowmarks = [x.strip().lower() for x in a.lower_better.split(",")
                if x.strip()]
    highmarks = [x.strip().lower() for x in a.higher_better.split(",")
                 if x.strip()]

    def lower_is_better(c):
        cl = c.lower()
        if any(mk in cl for mk in highmarks):
            return False
        return any(mk in cl for mk in lowmarks)

    cols = [a.col] + [c for c in also if c in d.columns and c != a.col]
    log = [f"[in] {a.poses}",
           f"     {d['compound'].nunique()} compounds",
           f"     {a.label1}: {len(r1)} replicates, "
           f"{a.label2}: {len(r2)} replicates, none shared",
           "",
           "  [unpaired] the two receptors share no replicate label, so the "
           "replicates",
           "  are not partners and the difference cannot be formed within one. "
           "None of",
           "  the noise cancels, so the uncertainty below is wider than a "
           "paired one",
           "  and a difference has to clear more to mean the same thing."]

    rows = []
    for cpd, g in d.groupby("compound"):
        row = {"compound": cpd}
        keep = True
        for c in cols:
            # the mean over the poses kept for that replicate, then over
            # replicates, so a replicate with more poses does not weigh more
            per = g.groupby(["receptor", "seed"])[c].mean()
            for lab, key in ((a.a1, "a1"), (a.a2, "a2")):
                if lab not in per.index.get_level_values(0):
                    keep = False
                    continue
                v = per.loc[lab].to_numpy(float)
                row[f"{c}_{key}_mean"] = float(np.mean(v))
                row[f"{c}_{key}_sd"] = (float(np.std(v, ddof=1))
                                        if len(v) > 1 else np.nan)
                row[f"{c}_{key}_n"] = len(v)
            if not keep:
                continue
            dv = row[f"{c}_a2_mean"] - row[f"{c}_a1_mean"]
            if lower_is_better(c):
                dv = -dv
            row[f"{c}_delta"] = dv
            s1, s2 = row[f"{c}_a1_sd"], row[f"{c}_a2_sd"]
            n1, n2 = row[f"{c}_a1_n"], row[f"{c}_a2_n"]
            if not (np.isnan(s1) or np.isnan(s2)):
                # the standard error of an unpaired difference of means
                row[f"{c}_se"] = float(np.sqrt(s1**2 / n1 + s2**2 / n2))
                # a zero standard error means every replicate agreed exactly,
                # which makes the ratio undefined rather than infinite; it is
                # left empty and the zero spread is what the table shows
                row[f"{c}_z"] = (dv / row[f"{c}_se"]
                                 if row[f"{c}_se"] > 0 else np.nan)
        if keep:
            rows.append(row)
    if not rows:
        sys.exit("no compound has rows under both receptors")
    r = pd.DataFrame(rows).sort_values(f"{a.col}_delta", ascending=False)

    log.append("")
    log.append(f"=== {a.col}: {a.label2} minus {a.label1}, unpaired ===")
    log.append(f"  {'compound':10s}{a.label1:>9s}{'sd':>7s}"
               f"{a.label2:>9s}{'sd':>7s}{'delta':>8s}{'se':>7s}{'z':>7s}")
    for _, x in r.iterrows():
        log.append(f"  {x['compound']:10s}"
                   f"{x[f'{a.col}_a1_mean']:9.2f}{x[f'{a.col}_a1_sd']:7.2f}"
                   f"{x[f'{a.col}_a2_mean']:9.2f}{x[f'{a.col}_a2_sd']:7.2f}"
                   f"{x[f'{a.col}_delta']:+8.2f}"
                   f"{x.get(f'{a.col}_se', np.nan):7.2f}"
                   + (f"{x[f'{a.col}_z']:+7.2f}"
                      if not pd.isna(x.get(f"{a.col}_z", np.nan)) else "      -"))
    zc = r[f"{a.col}_z"].dropna() if f"{a.col}_z" in r.columns else pd.Series([])
    nose = len(r) - len(zc)
    if nose:
        log.append("")
        log.append(f"  [note] {nose} of {len(r)} compounds have zero spread "
                   f"across replicates, so no ratio to their own spread can be "
                   f"formed. Every conformer gave the same value for them")
    if len(zc):
        log.append("")
        log.append(f"  |z| >= 2 (the difference is twice its own standard "
                   f"error): {int((zc.abs() >= 2).sum())} of {len(zc)}")
        log.append(f"  z is not a p value here: with {len(r1)} and {len(r2)} "
                   f"replicates from one trajectory each, the replicates are "
                   f"not independent samples of the protein, so it says how "
                   f"large the difference is against this ensemble's own "
                   f"spread and nothing about a population")

    others = [c for c in cols if c != a.col]
    if others:
        log.append("")
        log.append("=== the other criteria, same unpaired difference ===")
        log.append("  a POSITIVE number favours " + a.label2
                   + "; convention: "
                   + ", ".join(f"{c} ({'lower' if lower_is_better(c) else 'higher'}"
                               f" is better)" for c in others))
        hdr = f"  {'compound':10s}"
        for c in others:
            hdr += f"{c[:11]:>13s}"
        log.append(hdr + "   agree")
        for _, x in r.iterrows():
            line = f"  {x['compound']:10s}"
            sg = [np.sign(x[f"{a.col}_delta"])]
            for c in others:
                v = x.get(f"{c}_delta")
                line += "          n/a" if pd.isna(v) else f"{v:+13.3f}"
                if not pd.isna(v):
                    sg.append(np.sign(v))
            npos = sum(1 for q in sg if q > 0)
            nneg = sum(1 for q in sg if q < 0)
            nzero = len(sg) - npos - nneg
            # an exactly zero difference favours neither, so it is not counted
            # against either side
            if nneg == 0 and npos:
                tag = f"{npos} -> {a.label2}" + (f", {nzero} zero" if nzero else "")
                if not nzero:
                    tag = f"all {npos} -> {a.label2}"
            elif npos == 0 and nneg:
                tag = f"all {nneg} -> {a.label1}" if not nzero \
                    else f"{nneg} -> {a.label1}, {nzero} zero"
            elif npos or nneg:
                tag = f"split {npos}/{nneg}" + (f", {nzero} zero" if nzero else "")
            else:
                tag = "no difference on any"
            log.append(line + f"   {tag}")

    text = "\n".join(log)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        r.to_csv(a.out + "_unpaired.csv", index=False)
        with open(a.out + "_values.txt", "w") as f:
            f.write(text + "\n")
        print(f"\n[out] {a.out}_unpaired.csv\n      {a.out}_values.txt")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--poses", required=True,
                   help="the *_poses.csv from seed_topn_stability.py")
    p.add_argument("--a1", required=True, help="receptor label to subtract")
    p.add_argument("--a2", required=True, help="receptor label subtracted from")
    p.add_argument("--label1", default="CK2α")
    p.add_argument("--label2", default="CK2α′")
    p.add_argument("--col", default="n_hbond",
                   help="the criterion the ranking is on")
    p.add_argument("--also", default="CNNscore,CNNaffinity,minimizedAffinity",
                   help="further criteria reported beside the ranked one. Each "
                        "gets its own paired difference and its own sign "
                        "check, so a criterion that disagrees with the others "
                        "is visible rather than averaged away")
    p.add_argument("--lower-better", default="minimizedaffinity,vina,energy",
                   help="substrings marking a criterion where the smaller "
                        "value is the better one. Its difference is negated "
                        "so that, for every criterion printed, a positive "
                        "number means the second receptor is favoured")
    p.add_argument("--higher-better", default="cnn",
                   help="substrings marking a criterion where the larger value "
                        "is better, checked first. gnina reports CNNaffinity "
                        "as a pKd, where larger is better, and "
                        "minimizedAffinity as an energy, where smaller is; "
                        "a plain match on 'affinity' would negate both")
    p.add_argument("--out")
    a = p.parse_args()

    d = pd.read_csv(a.poses)
    also = [x.strip() for x in a.also.split(",") if x.strip()]
    lowmarks = [x.strip().lower() for x in a.lower_better.split(",")
                if x.strip()]
    highmarks = [x.strip().lower() for x in a.higher_better.split(",")
                 if x.strip()]

    def lower_is_better(c):
        cl = c.lower()
        # the higher-is-better marks win, so CNNaffinity is not caught by a
        # mark meant for an energy term
        if any(mk in cl for mk in highmarks):
            return False
        return any(mk in cl for mk in lowmarks)

    missing_also = [c for c in also if c not in d.columns]
    also = [c for c in also if c in d.columns and c != a.col]
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
    # pairing requires the two receptors to share replicate labels. A seed run
    # does: seed s0 searched both. An ensemble run does not, because a1_c00 and
    # a2_c00 are frames of different trajectories and are not partners. Pairing
    # them anyway would difference unrelated numbers, and dropping the
    # unmatched rows would silently discard everything, so which case this is
    # gets decided and stated.
    r1 = set(d.loc[d["receptor"] == a.a1, "seed"])
    r2 = set(d.loc[d["receptor"] == a.a2, "seed"])
    shared = r1 & r2
    paired = len(shared) >= 2
    if not paired:
        if shared:
            sys.exit(f"the two receptors share only {len(shared)} replicate "
                     f"label(s), too few to pair and too many to treat as "
                     f"independent. Check --a1/--a2 and --fields")
        return unpaired(a, d, r1, r2)

    d = d[d["seed"].isin(shared)]
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

    extra = {}
    for c in also:
        mm = (d.groupby(["compound", "receptor", "seed"])[c]
               .mean().unstack("receptor"))
        if a.a1 not in mm.columns or a.a2 not in mm.columns:
            continue
        mm = mm.dropna(subset=[a.a1, a.a2])
        dv = mm[a.a2] - mm[a.a1]
        if lower_is_better(c):
            # negated so that, printed, a positive number always means the
            # second receptor is favoured. Without this a reader has to carry
            # a different sign rule per column
            dv = -dv
        extra[c] = pd.DataFrame({a.a1: mm[a.a1], a.a2: mm[a.a2],
                                 "delta": dv}).reset_index()

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
        for c, ed in extra.items():
            eg = ed[ed["compound"] == cpd]
            if not len(eg):
                continue
            ev = eg["delta"].to_numpy(float)
            row[f"{c}_{a.a1}"] = float(eg[a.a1].mean())
            row[f"{c}_{a.a2}"] = float(eg[a.a2].mean())
            row[f"{c}_delta"] = float(np.median(ev))
            row[f"{c}_sd"] = float(ev.std(ddof=1)) if len(ev) > 1 else np.nan
            row[f"{c}_pos"] = int((ev > 0).sum())
            row[f"{c}_n"] = len(ev)
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
        log.append(f"  {len(nr)} more "
                   + ("is" if len(nr) == 1 else "are")
                   + " never reversed but "
                   + ("ties" if len(nr) == 1 else "tie")
                   + " in at least one seed: " + ", ".join(nr["compound"]))
    rev = r[r["reverses"]]
    if len(rev):
        log.append(f"  {len(rev)} "
                   + ("reverses" if len(rev) == 1 else "reverse")
                   + ": " + ", ".join(rev["compound"]))
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

    if extra:
        log.append("")
        log.append("=== the other criteria, same paired difference ===")
        log.append("  for every column a POSITIVE number favours "
                   f"{a.label2}; criteria where the smaller value is better "
                   "are negated to make that true")
        log.append("  convention applied: "
                   + ", ".join(f"{c} ({'lower' if lower_is_better(c) else 'higher'}"
                               f" is better)" for c in extra))
        hdr = f"  {'compound':10s}{'Dhbond':>9s}"
        for c in extra:
            hdr += f"{c[:11]:>13s}"
        hdr += "   agree"
        log.append(hdr)
        crit = ["delta_median"] + [f"{c}_delta" for c in extra]
        for _, x in r.iterrows():
            line = f"  {x['compound']:10s}{x['delta_median']:+9.2f}"
            signs = [np.sign(x[c]) for c in crit if not pd.isna(x.get(c))]
            for c in extra:
                v = x.get(f"{c}_delta")
                line += "          n/a" if pd.isna(v) else f"{v:+13.3f}"
            npos = sum(1 for sg in signs if sg > 0)
            nneg = sum(1 for sg in signs if sg < 0)
            if npos == len(signs):
                tag = f"all {len(signs)} -> {a.label2}"
            elif nneg == len(signs):
                tag = f"all {len(signs)} -> {a.label1}"
            else:
                tag = f"split {npos}/{nneg}"
            log.append(line + f"   {tag}")
        log.append("")
        agree_all = []
        for _, x in r.iterrows():
            sg = [np.sign(x[c]) for c in crit if not pd.isna(x.get(c))]
            if sg and all(s > 0 for s in sg):
                agree_all.append(x["compound"])
        log.append(f"  {len(agree_all)} of {len(r)} have every criterion "
                   f"pointing at {a.label2}: "
                   + (", ".join(agree_all) if agree_all else "none"))
        both = [c for c in agree_all
                if bool(r.loc[r["compound"] == c, "sign_consistent"].iloc[0])]
        log.append(f"  of those, {len(both)} also keep the hydrogen bond sign "
                   f"in every seed: " + (", ".join(both) if both else "none"))

    if missing_also:
        log.append("")
        log.append(f"  [note] requested but not in the table: "
                   f"{', '.join(missing_also)}. Re-run "
                   f"seed_topn_stability.py with --props naming them")

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

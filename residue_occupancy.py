#!/usr/bin/env python3
"""
In how many conformers of an ensemble does a compound bond to a named residue.

A count of hydrogen bonds says how many were made, not to what. A compound
making three bonds to residues that both subunits share is not engaging the
divergent hinge at all. This counts, per compound, the conformers in which a
named residue is bonded, which is the quantity a claim about that residue
rests on.

Reads the *_poses.csv from seed_topn_stability.py, whose residues column
carries the labels in the numbering of the structures that were docked into.
A topology renumbered from its pdb needs --offset to be named in pdb terms;
the labels themselves are not rewritten, only the heading says which
numbering is shown.

    python residue_occupancy.py --poses ens_cx_top8_poses.csv \
        --receptor a2 --residues TYR110,ILE111 --offset 6
"""
import argparse, os, sys
import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("needs pandas")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--poses", required=True)
    p.add_argument("--receptor", action="append", required=True,
                   help="receptor label to report. Repeat for both subunits")
    p.add_argument("--residues", required=True,
                   help="comma-separated labels as they appear in the "
                        "residues column, e.g. TYR110,ILE111. Per receptor "
                        "as LABEL:RES,RES and separated by ';' when the two "
                        "subunits use different numbering")
    p.add_argument("--offset", default="0",
                   help="added to the number when printing, so the heading "
                        "names pdb residues rather than renumbered ones. The "
                        "two subunits are renumbered by different amounts, so "
                        "give it per receptor as LABEL:N,LABEL:N; one number "
                        "applies to all of them and will mislabel any "
                        "receptor whose offset differs")
    p.add_argument("--use", choices=["best", "any"], default="best",
                   help="best counts a conformer only when its highest-ranked "
                        "pose makes the bond; any counts it when any kept "
                        "pose does. best is the stricter claim and matches a "
                        "single-pose analysis")
    p.add_argument("--rank-on", choices=["either", "both", "total", "delta"],
                   default="either",
                   help="the quantity the table is ordered by. either and "
                        "both are hinge counts; total is the mean hydrogen "
                        "bonds per conformer to any residue; delta is that "
                        "total minus the other subunit's")
    p.add_argument("--rank-by",
                   help="receptor label whose 'either' count orders the "
                        "table. Defaults to the last --receptor given. The "
                        "other receptor's counts are carried beside it, not "
                        "ranked on")
    p.add_argument("--show-diff", action="store_true",
                   help="also print the difference between the two 'either' "
                        "counts")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--all", action="store_true")
    p.add_argument("--out")
    a = p.parse_args()

    d = pd.read_csv(a.poses)
    for c in ("compound", "receptor", "seed", "pose_rank", "residues"):
        if c not in d.columns:
            sys.exit(f"missing column '{c}'; have: {', '.join(d.columns)}")
    d["residues"] = d["residues"].fillna("")

    # per receptor, the residues asked for
    want = {}
    if ";" in a.residues or ":" in a.residues:
        for part in a.residues.split(";"):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                sys.exit(f"--residues wants LABEL:RES,RES when per receptor; "
                         f"got {part!r}")
            lab, rs = part.split(":", 1)
            want[lab.strip()] = [x.strip().upper() for x in rs.split(",")
                                 if x.strip()]
    else:
        rs = [x.strip().upper() for x in a.residues.split(",") if x.strip()]
        for lab in a.receptor:
            want[lab] = list(rs)

    off = {}
    if ":" in a.offset:
        for part in a.offset.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                sys.exit(f"--offset wants LABEL:N when per receptor; got "
                         f"{part!r}")
            lab, n = part.split(":", 1)
            try:
                off[lab.strip()] = int(n)
            except ValueError:
                sys.exit(f"offset for {lab!r} is not a whole number: {n!r}")
        missing = [l for l in a.receptor if l not in off]
        if missing:
            sys.exit(f"--offset gives no value for: {', '.join(missing)}. "
                     f"The subunits are renumbered by different amounts, so "
                     f"each needs its own")
    else:
        try:
            one = int(a.offset)
        except ValueError:
            sys.exit(f"--offset is not a whole number: {a.offset!r}")
        off = {l: one for l in a.receptor}

    def shown(lab):
        """The residue names as the heading prints them."""
        out = []
        for r in want[lab]:
            num = "".join(ch for ch in r if ch.isdigit())
            nm = "".join(ch for ch in r if not ch.isdigit())
            out.append(f"{nm}{int(num) + off.get(lab, 0)}" if num else r)
        return out

    log = [f"[in] {a.poses}",
           f"     {d['compound'].nunique()} compounds"]
    if any(off.values()):
        log.append("     printed with an offset per receptor: "
                   + ", ".join(f"{l} {off[l]:+d}" for l in a.receptor
                               if l in off))
        log.append("     matched in the table as: "
                   + "; ".join(f"{lab}: {', '.join(want[lab])}"
                               for lab in want))
    log.append(f"     a conformer counts when "
               + ("its best pose" if a.use == "best" else "any kept pose")
               + " bonds the residue")

    rows = {}
    for lab in a.receptor:
        g = d[d["receptor"] == lab]
        if not len(g):
            log.append(f"     [note] no rows for receptor '{lab}'")
            continue
        if a.use == "best":
            g = g.sort_values("pose_rank").groupby(
                ["compound", "seed"], as_index=False).first()
        res = want.get(lab)
        if not res:
            continue
        for cpd, gg in g.groupby("compound"):
            nconf = gg["seed"].nunique()
            rec = rows.setdefault(cpd, {"compound": cpd})
            rec[f"{lab}_n_conformer"] = nconf
            per = {}
            for r in res:
                hit = gg.groupby("seed")["residues"].apply(
                    lambda s: any(r in str(x).upper().split(";")
                                  for x in s))
                per[r] = int(hit.sum())
                rec[f"{lab}_{r}"] = per[r]
            # a conformer engaging either, and one engaging both
            eith = gg.groupby("seed")["residues"].apply(
                lambda s: any(any(r in str(x).upper().split(";") for r in res)
                              for x in s))
            bothm = gg.groupby("seed")["residues"].apply(
                lambda s: all(any(r in str(x).upper().split(";") for x in s)
                              for r in res))
            rec[f"{lab}_either"] = int(eith.sum())
            rec[f"{lab}_both"] = int(bothm.sum())
            # the total, so a subunit whose hinge is untouched but whose other
            # residues are engaged is not mistaken for one that is not bound
            if "n_hbond" in gg.columns:
                tot = gg.groupby("seed")["n_hbond"].first()
                rec[f"{lab}_total_mean"] = float(tot.mean())
                rec[f"{lab}_total_sd"] = (float(tot.std(ddof=1))
                                          if len(tot) > 1 else np.nan)
            # residues engaged that are not the hinge pair, counted per
            # conformer and averaged
            def n_other(sr):
                out = []
                for x in sr:
                    lbl = [q for q in str(x).upper().split(";") if q]
                    out.append(sum(1 for q in lbl if q not in res))
                return float(np.mean(out)) if out else 0.0
            rec[f"{lab}_other_mean"] = float(
                gg.groupby("seed")["residues"].apply(n_other).mean())

    if not rows:
        sys.exit("no rows matched; check --receptor labels and that the "
                 "residues column holds the labels you asked for")
    r = pd.DataFrame(list(rows.values()))

    labs = [l for l in a.receptor if f"{l}_either" in r.columns]
    key = f"{labs[-1]}_either" if labs else None
    if key:
        r = r.sort_values(key, ascending=False)

    if len(labs) == 2:
        l1, l2 = labs
        r["diff_either"] = r[f"{l2}_either"] - r[f"{l1}_either"]
        r["diff_both"] = r[f"{l2}_both"] - r[f"{l1}_both"]
        rank_lab = a.rank_by or l2
        if rank_lab not in labs:
            sys.exit(f"--rank-by {rank_lab!r} is not one of: "
                     f"{', '.join(labs)}")
        other = l1 if rank_lab == l2 else l2
        if a.rank_on == "total":
            keycol = f"{rank_lab}_total_mean"
        elif a.rank_on == "delta":
            r["total_delta"] = (r[f"{rank_lab}_total_mean"]
                                - r[f"{other}_total_mean"])
            keycol = "total_delta"
        else:
            keycol = f"{rank_lab}_{a.rank_on}"
        if keycol not in r.columns:
            sys.exit(f"--rank-on {a.rank_on} needs column {keycol!r}, which "
                     f"is not in the table")
        r = r.sort_values([keycol, f"{rank_lab}_either"], ascending=False)
        n1 = int(r[f"{l1}_n_conformer"].median())
        n2 = int(r[f"{l2}_n_conformer"].median())
        log.append("")
        log.append(f"=== conformers in which the hinge is bonded "
                   f"({l1} of {n1}, {l2} of {n2}) ===")
        log.append(f"    {rank_lab}: {', '.join(shown(rank_lab))}   "
                   f"{other}: {', '.join(shown(other))}")
        log.append(f"    either = at least one of the pair; both = the pair "
                   f"bridged; total = mean hydrogen bonds per conformer with "
                   f"its sd; other = mean residues engaged that are not the "
                   f"pair")
        what = {"either": f"{rank_lab} either (hinge)",
                "both": f"{rank_lab} both (hinge bridged)",
                "total": f"{rank_lab} total bonds to any residue",
                "delta": f"total bonds, {rank_lab} minus {other}"}[a.rank_on]
        log.append(f"    ranked on {what}; the rest is carried beside it")
        order = (rank_lab, other)
        hdr = f"  {'compound':9s}"
        for lab in order:
            for nm in shown(lab):
                hdr += f"{nm:>9s}"
            hdr += f"{'either':>8s}{'both':>6s}"
            if f"{lab}_total_mean" in r.columns:
                hdr += f"{'total':>8s}{'sd':>6s}{'other':>7s}"
        if a.rank_on == "delta":
            hdr += f"{'d total':>9s}"
        if a.show_diff:
            hdr += f"{'diff':>7s}"
        log.append(hdr)
        for _, x in (r if a.all else r.head(a.top)).iterrows():
            line = f"  {x['compound']:9s}"
            for lab in order:
                for c in want[lab]:
                    line += f"{int(x[f'{lab}_{c}']):9d}"
                line += (f"{int(x[f'{lab}_either']):8d}"
                         f"{int(x[f'{lab}_both']):6d}")
                if f"{lab}_total_mean" in r.columns:
                    line += (f"{x[f'{lab}_total_mean']:8.2f}"
                             f"{x[f'{lab}_total_sd']:6.2f}"
                             f"{x[f'{lab}_other_mean']:7.2f}")
            if a.rank_on == "delta":
                line += f"{x['total_delta']:+9.2f}"
            if a.show_diff:
                line += f"{int(x['diff_either']):+7d}"
            log.append(line)
        if not a.all and len(r) > a.top:
            log.append(f"    ... {len(r) - a.top} more")
        dv = r["diff_either"]
        log.append("")
        for lab, n in ((rank_lab, n1 if rank_lab == l1 else n2),
                       (other, n1 if other == l1 else n2)):
            v = r[f"{lab}_either"]
            log.append(f"    {lab}: median {v.median():.0f} of {n} "
                       f"conformers, {int((v == 0).sum())} never, "
                       f"{int((v == n).sum())} in every one")
            for c, nm in zip(want[lab], shown(lab)):
                vv = r[f"{lab}_{c}"]
                log.append(f"      {nm}: median {vv.median():.0f}, "
                           f"max {vv.max():.0f}")
            if f"{lab}_total_mean" in r.columns:
                log.append(f"      total bonds per conformer: median "
                           f"{r[f'{lab}_total_mean'].median():.2f}; "
                           f"residues away from the pair: median "
                           f"{r[f'{lab}_other_mean'].median():.2f}")
        if a.show_diff:
            log.append(f"    difference: median {dv.median():+.0f}, "
                       f"{int((dv > 0).sum())} favour {l2}, "
                       f"{int((dv < 0).sum())} favour {l1}, "
                       f"{int((dv == 0).sum())} equal")
    else:
        for lab in labs:
            n = int(r[f"{lab}_n_conformer"].median())
            r = r.sort_values(f"{lab}_either", ascending=False)
            log.append("")
            log.append(f"=== {lab}: conformers of {n} in which the residue "
                       f"is bonded ===")
            hdr = f"  {'compound':10s}"
            for nm in shown(lab):
                hdr += f"{nm:>10s}"
            hdr += f"{'either':>9s}{'both':>7s}{'either %':>10s}"
            log.append(hdr)
            for _, x in (r if a.all else r.head(a.top)).iterrows():
                line = f"  {x['compound']:10s}"
                for c in want[lab]:
                    line += f"{int(x[f'{lab}_{c}']):10d}"
                line += (f"{int(x[f'{lab}_either']):9d}"
                         f"{int(x[f'{lab}_both']):7d}"
                         f"{100*x[f'{lab}_either']/x[f'{lab}_n_conformer']:9.0f}%")
                log.append(line)
            if not a.all and len(r) > a.top:
                log.append(f"    ... {len(r) - a.top} more")
            v = r[f"{lab}_either"]
            log.append(f"    median {v.median():.0f} of {n}, "
                       f"{int((v == 0).sum())} never, "
                       f"{int((v == n).sum())} in every one")

    text = "\n".join(log)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        r.to_csv(a.out, index=False)
        print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

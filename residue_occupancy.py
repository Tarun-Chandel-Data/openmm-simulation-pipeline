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
        r = r.sort_values(["diff_either", f"{l2}_either"], ascending=False)
        n1 = int(r[f"{l1}_n_conformer"].median())
        n2 = int(r[f"{l2}_n_conformer"].median())
        log.append("")
        log.append(f"=== conformers in which the hinge is bonded "
                   f"({l1} of {n1}, {l2} of {n2}) ===")
        log.append(f"    {l1}: {', '.join(shown(l1))}   "
                   f"{l2}: {', '.join(shown(l2))}")
        log.append(f"    either = at least one of the pair; both = the pair "
                   f"bridged; ranked on the difference in either")
        hdr = f"  {'compound':9s}"
        for lab in (l1, l2):
            for nm in shown(lab):
                hdr += f"{nm:>9s}"
            hdr += f"{'either':>8s}{'both':>6s}"
        hdr += f"{'diff':>7s}"
        log.append(hdr)
        for _, x in (r if a.all else r.head(a.top)).iterrows():
            line = f"  {x['compound']:9s}"
            for lab in (l1, l2):
                for c in want[lab]:
                    line += f"{int(x[f'{lab}_{c}']):9d}"
                line += (f"{int(x[f'{lab}_either']):8d}"
                         f"{int(x[f'{lab}_both']):6d}")
            line += f"{int(x['diff_either']):+7d}"
            log.append(line)
        if not a.all and len(r) > a.top:
            log.append(f"    ... {len(r) - a.top} more")
        dv = r["diff_either"]
        log.append("")
        for lab, n in ((l1, n1), (l2, n2)):
            v = r[f"{lab}_either"]
            log.append(f"    {lab}: median {v.median():.0f} of {n} "
                       f"conformers, {int((v == 0).sum())} never, "
                       f"{int((v == n).sum())} in every one")
            for c, nm in zip(want[lab], shown(lab)):
                vv = r[f"{lab}_{c}"]
                log.append(f"      {nm}: median {vv.median():.0f}, "
                           f"max {vv.max():.0f}")
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

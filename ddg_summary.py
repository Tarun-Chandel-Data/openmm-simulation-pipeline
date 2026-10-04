#!/usr/bin/env python3
"""
Every run's answer to the same question, side by side.

A selectivity figure means nothing on its own: it means something when two
independent receptor sets give the same one. Each pair of runs here is one
estimate of the difference between the isoforms, and putting them in one
row is what shows whether they agree.

A run with only one isoform cannot give a difference and is reported as the
energy it is, not quietly folded in as though it could.

The last column is the point of the table. Where two estimates of the same
compound's selectivity disagree in sign, neither is a measurement of the
compound: something about the receptors is deciding it, and the table says
so rather than leaving the reader to compare columns.

    python ddg_summary.py --pair "VB004 ens=mmgbsa_a1/cells.csv,mmgbsa_a2/cells.csv" \\
        --pair "CX ens=mmgbsa_cx_a1/cells.csv,mmgbsa_cx_a2/cells.csv" \\
        --single "crystal a2=mmgbsa_xtal/cells.csv" --out ddg_all
"""
import argparse, csv, math, os, sys


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def sd(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def sem(v):
    return sd(v) / math.sqrt(len(v)) if v else float("nan")


def load(path):
    if not os.path.exists(path):
        sys.exit(f"no such file: {path}")
    out = {}
    with open(path, newline="") as f:
        for d in csv.DictReader(f):
            try:
                out.setdefault(d["compound"], []).append(float(d["dG"]))
            except (KeyError, TypeError, ValueError):
                continue
    if not out:
        sys.exit(f"{path}: no usable rows")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", action="append", default=[],
                   metavar="LABEL=A1CSV,A2CSV",
                   help="two runs of one receptor set, CK2alpha then "
                        "CK2alpha prime. Repeat per receptor set")
    p.add_argument("--single", action="append", default=[],
                   metavar="LABEL=CSV",
                   help="a run with only one isoform, reported as an energy "
                        "since it cannot give a difference")
    p.add_argument("--reference",
                   help="a compound to express the others against, inside "
                        "each receptor set. Whatever an ensemble does to "
                        "every compound alike cancels in that difference, so "
                        "it is the fairer comparison between receptor sets "
                        "and the one to judge agreement on")
    p.add_argument("--only", help="compounds, in the order to report them")
    p.add_argument("--out")
    a = p.parse_args()
    if not a.pair and not a.single:
        sys.exit("give at least one --pair or --single")

    pairs, singles = [], []
    for spec in a.pair:
        if "=" not in spec or "," not in spec:
            sys.exit(f"--pair takes LABEL=A1CSV,A2CSV, got {spec!r}")
        lb, files = spec.split("=", 1)
        f1, f2 = [x.strip() for x in files.split(",", 1)]
        pairs.append((lb, load(f1), load(f2)))
    for spec in a.single:
        if "=" not in spec:
            sys.exit(f"--single takes LABEL=CSV, got {spec!r}")
        lb, f = spec.split("=", 1)
        singles.append((lb, load(f.strip())))

    seen = set()
    for _, g1, g2 in pairs:
        seen |= set(g1) | set(g2)
    for _, g in singles:
        seen |= set(g)
    cpds = ([x.strip() for x in a.only.split(",") if x.strip()] if a.only
            else sorted(seen))

    log = ["[in] " + ", ".join(lb for lb, _, _ in pairs)
           + ((" | single isoform: " + ", ".join(lb for lb, _ in singles))
              if singles else ""),
           "     dd is CK2alpha prime minus CK2alpha; negative prefers "
           "CK2alpha prime",
           "     single-point MM-GBSA, no entropy: these compare with each "
           "other and are not binding free energies"]
    log.append("")
    log.append("=== selectivity from each receptor set, and the energies "
               "behind it ===")
    head = f"  {'compound':10s}"
    for lb, _, _ in pairs:
        head += f"{lb[:11] + ' a1':>13s}{lb[:11] + ' a2':>13s}{'dd':>8s}{'+-':>7s}"
    for lb, _ in singles:
        head += f"{lb[:13]:>14s}{'sd':>7s}"
    log.append(head)

    rows, dd_by_pair = [], {lb: {} for lb, _, _ in pairs}
    rows_by = {}
    for c in cpds:
        line = f"  {c:10s}"
        rec = {"compound": c}
        for lb, g1, g2 in pairs:
            v1, v2 = g1.get(c, []), g2.get(c, [])
            if not v1 or not v2:
                line += f"{'-':>13s}{'-':>13s}{'-':>8s}{'-':>7s}"
                continue
            dd = mean(v2) - mean(v1)
            se = math.sqrt(sem(v1) ** 2 + sem(v2) ** 2)
            dd_by_pair[lb][c] = (dd, se)
            line += (f"{mean(v1):13.2f}{mean(v2):13.2f}{dd:8.2f}{se:7.2f}")
            rec[f"{lb}_a1"] = round(mean(v1), 3)
            rec[f"{lb}_a2"] = round(mean(v2), 3)
            rec[f"{lb}_ddG"] = round(dd, 3)
            rec[f"{lb}_se"] = round(se, 3)
        for lb, g in singles:
            v = g.get(c, [])
            line += ((f"{mean(v):14.2f}{sd(v):7.2f}") if v
                     else f"{'-':>14s}{'-':>7s}")
            if v:
                rec[f"{lb}_dG"] = round(mean(v), 3)
                rec[f"{lb}_sd"] = round(sd(v), 3)
        log.append(line)
        rows.append(rec)
        rows_by[c] = rec

    if len(pairs) > 1:
        log.append("")
        log.append("=== do the receptor sets agree? ===")
        log.append("    two estimates of one compound's selectivity that "
                   "disagree in sign are not measuring the compound")
        log.append(f"  {'compound':10s}"
                   + "".join(f"{lb[:12]:>14s}" for lb, _, _ in pairs)
                   + "   reading")
        agree = disagree = 0
        for c in cpds:
            vals = [dd_by_pair[lb].get(c) for lb, _, _ in pairs]
            if any(v is None for v in vals):
                continue
            signs = {1 if v[0] > 0 else -1 for v in vals}
            resolved = [abs(v[0]) > 2 * v[1] for v in vals]
            if len(signs) > 1 and any(resolved):
                note, disagree = "opposite directions", disagree + 1
            elif len(signs) > 1:
                note = "opposite signs, neither resolved"
            else:
                note, agree = "same direction", agree + 1
            log.append(f"  {c:10s}"
                       + "".join(f"{v[0]:14.2f}" for v in vals)
                       + f"   {note}")
        log.append("")
        log.append(f"  {agree} compounds agree in direction, {disagree} are "
                   f"resolved in opposite directions")
        if disagree:
            log.append("  where the receptor sets disagree, the difference "
                       "being measured is between the receptor sets and not "
                       "between the isoforms. Neither estimate stands alone")

    if a.reference and len(pairs) > 1:
        ref = a.reference
        log.append("")
        log.append(f"=== against {ref}, inside each receptor set ===")
        log.append("    an ensemble shifts every compound it holds by roughly "
                   "the same amount, and that shift cancels here. What "
                   "survives is how much a compound's selectivity differs "
                   "from the reference's, which is what a substitution would "
                   "change")
        log.append(f"  {'compound':10s}"
                   + "".join(f"{lb[:12]:>10s}{'+-':>7s}" for lb, _, _ in pairs)
                   + "   reading")
        rel = {lb: {} for lb, _, _ in pairs}
        for c in cpds:
            if c == ref:
                continue
            line, vals = f"  {c:10s}", []
            for lb, _, _ in pairs:
                d = dd_by_pair[lb].get(c)
                r0 = dd_by_pair[lb].get(ref)
                if d is None or r0 is None:
                    line += f"{'-':>10s}{'-':>7s}"
                    vals.append(None)
                    continue
                v = d[0] - r0[0]
                se = math.sqrt(d[1] ** 2 + r0[1] ** 2)
                rel[lb][c] = (v, se)
                vals.append((v, se))
                line += f"{v:10.2f}{se:7.2f}"
            good = [x for x in vals if x is not None]
            if len(good) == len(pairs) and good:
                signs = {1 if x[0] > 0 else -1 for x in good}
                res = [abs(x[0]) > 2 * x[1] for x in good]
                note = ("agree" if len(signs) == 1 else
                        "disagree, one or both resolved" if any(res) else
                        "disagree, neither resolved")
            else:
                note = ""
            log.append(line + f"   {note}")
            for k, v in zip([lb for lb, _, _ in pairs], vals):
                if v is not None:
                    rows_by[c][f"{k}_vs_{ref}"] = round(v[0], 3)

        # how well the two receptor sets agree once the offset is removed
        if len(pairs) == 2:
            l1, l2 = pairs[0][0], pairs[1][0]
            both = [c for c in cpds if c in rel[l1] and c in rel[l2]]
            if len(both) > 2:
                x = [rel[l1][c][0] for c in both]
                y = [rel[l2][c][0] for c in both]
                mx, my = mean(x), mean(y)
                sxy = sum((p_ - mx) * (q - my) for p_, q in zip(x, y))
                sxx = sum((p_ - mx) ** 2 for p_ in x)
                syy = sum((q - my) ** 2 for q in y)
                r = sxy / math.sqrt(sxx * syy) if sxx and syy else float("nan")
                log.append("")
                log.append(f"  across {len(both)} compounds the two receptor "
                           f"sets correlate at r = {r:.2f}")
                if r != r:
                    pass
                elif r > 0.6:
                    log.append("  the sets rank the compounds alike once the "
                               "offset is removed, so what differs between "
                               "them is a constant and the ranking is usable")
                elif r > 0.2:
                    log.append("  the sets rank the compounds only loosely "
                               "alike; a ranking taken from one of them would "
                               "not be reproduced by the other")
                else:
                    log.append("  the sets do not rank the compounds alike at "
                               "all. Removing the offset does not rescue the "
                               "comparison: the per-compound figures are as "
                               "much a property of the receptors as the "
                               "ensemble-wide shift was")

    text = "\n".join(log)
    print(text)
    if a.out and rows:
        head_k = []
        for r in rows:
            for k in r:
                if k not in head_k:
                    head_k.append(k)
        with open(a.out + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=head_k, restval="")
            w.writeheader()
            w.writerows(rows)
        with open(a.out + ".txt", "w") as f:
            f.write(text + "\n")
        print(f"\n[out] {a.out}.csv, {a.out}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

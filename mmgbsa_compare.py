#!/usr/bin/env python3
"""
The isoform difference, with the spread it has to clear.

Two runs of the single-point calculation, one per isoform, give each
compound twenty values in each. The difference of the means is the
selectivity; whether it means anything depends on how wide those twenty
values are, so the two are never reported apart here.

The ensemble members of the two isoforms are different proteins in
different conformations, so they do not pair: the comparison is between
two independent samples and the uncertainty is combined as such. Within
one isoform two compounds were scored on the same twenty receptor
conformations, so that comparison is paired and is treated as paired,
which is the sharper of the two and the one a substituent effect would
show up in.

    python mmgbsa_compare.py --a1 mmgbsa_a1/cells.csv \\
        --a2 mmgbsa_a2/cells.csv --reference VB004
"""
import argparse, math, os, sys

try:
    import pandas as pd
except ImportError:
    sys.exit("needs pandas")


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def sd(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def sem(v):
    return sd(v) / math.sqrt(len(v)) if v else float("nan")


def load(path, label):
    if not os.path.exists(path):
        sys.exit(f"no such file: {path}")
    t = pd.read_csv(path)
    for c in ("compound", "structure", "dG"):
        if c not in t.columns:
            sys.exit(f"{path} has no column {c!r}; it holds "
                     f"{', '.join(t.columns)}")
    t["isoform"] = label
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a1", required=True, help="cells.csv for CK2alpha")
    p.add_argument("--a2", required=True, help="cells.csv for CK2alpha'")
    p.add_argument("--reference", default="VB004",
                   help="the compound the others are compared with")
    p.add_argument("--only", help="compounds, in the order to report them")
    p.add_argument("--out")
    a = p.parse_args()

    t1, t2 = load(a.a1, "a1"), load(a.a2, "a2")
    cpds = ([x.strip() for x in a.only.split(",") if x.strip()] if a.only
            else sorted(set(t1.compound) | set(t2.compound)))

    log = [f"[in] {len(t1)} cells for CK2alpha, {len(t2)} for CK2alpha'",
           "     single-point MM-GBSA on docked poses, no entropy: these "
           "compare with each other and are not binding free energies"]

    g1 = {c: list(t1[t1.compound == c]["dG"]) for c in cpds}
    g2 = {c: list(t2[t2.compound == c]["dG"]) for c in cpds}
    thin = [c for c in cpds if len(g1[c]) < 3 or len(g2[c]) < 3]
    if thin:
        log.append(f"     [warn] too few members to say anything about: "
                   f"{', '.join(thin)}")

    log.append("")
    log.append("=== each isoform, and the difference between them ===")
    log.append("    dd is CK2alpha' minus CK2alpha; negative prefers "
               "CK2alpha'")
    log.append("    the two isoforms are different proteins, so their "
               "members do not pair and the error is combined as for two "
               "independent samples")
    log.append(f"  {'compound':10s}{'n':>4s}{'CK2a mean':>12s}{'sd':>7s}"
               f"{'n':>4s}{'CK2a-prime':>12s}{'sd':>7s}"
               f"{'dd':>9s}{'+-':>8s}   reading")
    rows = []
    for c in cpds:
        v1, v2 = g1[c], g2[c]
        if not v1 or not v2:
            log.append(f"  {c:10s}   missing one isoform")
            continue
        dd = mean(v2) - mean(v1)
        se = math.sqrt(sem(v1) ** 2 + sem(v2) ** 2)
        note = ("inside the spread; not resolved" if abs(dd) < 2 * se
                else f"prefers {'CK2alpha-prime' if dd < 0 else 'CK2alpha'}")
        log.append(f"  {c:10s}{len(v1):4d}{mean(v1):12.2f}{sd(v1):7.2f}"
                   f"{len(v2):4d}{mean(v2):12.2f}{sd(v2):7.2f}"
                   f"{dd:9.2f}{se:8.2f}   {note}")
        rows.append({"compound": c, "n_a1": len(v1), "a1_mean": mean(v1),
                     "a1_sd": sd(v1), "n_a2": len(v2), "a2_mean": mean(v2),
                     "a2_sd": sd(v2), "ddG": dd, "ddG_se": se})

    # within one isoform the compounds share the receptor conformations, so
    # the comparison with the reference is paired member by member
    ref = a.reference
    if ref in cpds:
        log.append("")
        log.append(f"=== against {ref}, within each isoform, paired by "
                   f"ensemble member ===")
        log.append("    the same twenty receptor conformations scored both "
                   "compounds, so the difference is taken member by member "
                   "and the spread is of those differences")
        log.append(f"  {'compound':10s}{'CK2a diff':>12s}{'+-':>8s}"
                   f"{'CK2a-prime':>13s}{'+-':>8s}   reading")
        for c in cpds:
            if c == ref:
                continue
            cells = []
            for lbl, t in (("a1", t1), ("a2", t2)):
                x = t[t.compound == c].set_index("structure")["dG"]
                y = t[t.compound == ref].set_index("structure")["dG"]
                both = x.index.intersection(y.index)
                d = [float(x[i] - y[i]) for i in both]
                cells.append((mean(d) if d else float("nan"),
                              sem(d) if d else float("nan"), len(d)))
            (d1, s1, n1), (d2, s2, n2) = cells
            sig = [abs(d) > 2 * s for d, s in ((d1, s1), (d2, s2))
                   if s == s and d == d]
            note = ("differs from the parent in both isoforms"
                    if len(sig) == 2 and all(sig) else
                    "differs in one isoform" if any(sig) else
                    "inside the spread in both")
            log.append(f"  {c:10s}{d1:12.2f}{s1:8.2f}{d2:13.2f}{s2:8.2f}"
                       f"   {note}")

    log.append("")
    log.append("  a difference smaller than twice its standard error is not "
               "a difference. With eleven compounds compared, one in twenty "
               "will clear that by chance, so a single compound standing out "
               "is worth repeating and not reporting")

    text = "\n".join(log)
    print(text)
    if a.out and rows:
        pd.DataFrame(rows).to_csv(a.out + ".csv", index=False)
        with open(a.out + ".txt", "w") as f:
            f.write(text + "\n")
        print(f"\n[out] {a.out}.csv, {a.out}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

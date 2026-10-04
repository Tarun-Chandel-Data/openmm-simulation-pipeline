#!/usr/bin/env python3
"""
Every complex that was evaluated, in one table.

The runs are kept apart on disk because they are different experiments -
two isoforms, two sources of receptor - and they have to stay
distinguishable here, so each row says which run it came from. What they
share is the shape: one compound against one receptor, with the energy of
the pose that was used and, where several seeds were run, which poses
those were and how far apart they stood.

The pose column is the point of putting them together. A cell whose seeds
agreed to a tenth of an angstrom and a cell where they did not are not
equally good evidence, and the table should not let them look alike.

Nothing here needs anything beyond the standard library, so it runs where
amber does.

    python mmgbsa_table.py --cells crystal=mmgbsa_xtal/cells.csv \\
        --cells ensemble_a1=mmgbsa_a1/cells.csv \\
        --cells ensemble_a2=mmgbsa_a2/cells.csv --out all_cells
"""
import argparse, csv, math, os, sys


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def sd(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def num(d, k):
    try:
        return float(d[k])
    except (KeyError, TypeError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", action="append", required=True,
                   metavar="LABEL=FILE",
                   help="a cells.csv and the name to give that run. Repeat "
                        "once per run")
    p.add_argument("--only", help="compounds, in the order to report them")
    p.add_argument("--min-ratio", type=float, default=0.0,
                   help="drop a cell whose seeds agreed no more closely than "
                        "this multiple of nothing; 0 keeps every cell. Only "
                        "meaningful where the run recorded a consensus")
    p.add_argument("--out")
    a = p.parse_args()

    rows, order = [], []
    for spec in a.cells:
        if "=" not in spec:
            sys.exit(f"--cells takes LABEL=FILE, got {spec!r}")
        label, path = spec.split("=", 1)
        if not os.path.exists(path):
            sys.exit(f"no such file: {path}")
        order.append(label)
        with open(path, newline="") as f:
            for d in csv.DictReader(f):
                dg = num(d, "dG")
                if dg is None:
                    continue
                rows.append({
                    "run": label,
                    "compound": d.get("compound", ""),
                    "receptor": d.get("structure", ""),
                    "dG": dg,
                    "pose_ranks": d.get("consensus_ranks", "") or "top",
                    "pose_rmsd": d.get("consensus_rmsd", ""),
                    "seeds": d.get("seeds", "1") or "1"})
    if not rows:
        sys.exit("no rows read")

    if a.min_ratio > 0:
        keep, drop = [], 0
        for r in rows:
            v = num(r, "pose_rmsd")
            if v is not None and v > a.min_ratio:
                drop += 1
                continue
            keep.append(r)
        rows = keep

    cpds = ([x.strip() for x in a.only.split(",") if x.strip()] if a.only
            else sorted({r["compound"] for r in rows}))

    log = [f"[in] {len(rows)} complexes from {len(order)} runs: "
           + ", ".join(order),
           "     dG is single-point MM-GBSA with no entropy: comparable "
           "with each other, not a binding free energy",
           "     'poses' is which pose of each seed was used; 'spread' how "
           "far apart those poses stood. 'top' means the run had one seed "
           "and the best-scoring pose was taken"]
    log.append("")
    log.append("=== every complex ===")
    log.append(f"  {'compound':10s}{'run':14s}{'receptor':13s}{'dG':>9s}"
               f"{'poses':>12s}{'spread':>9s}")
    for c in cpds:
        for lb in order:
            sub = sorted((r for r in rows
                          if r["compound"] == c and r["run"] == lb),
                         key=lambda r: r["receptor"])
            for r in sub:
                sp = num(r, "pose_rmsd")
                log.append(f"  {c:10s}{lb:14s}{r['receptor']:13s}"
                           f"{r['dG']:9.2f}{r['pose_ranks']:>12s}"
                           + (f"{sp:9.2f}" if sp is not None else f"{'-':>9s}"))
        log.append("")

    log.append("=== by compound and run ===")
    log.append(f"  {'compound':10s}{'run':14s}{'n':>4s}{'dG mean':>10s}"
               f"{'sd':>7s}{'spread':>9s}{'at rank 1':>11s}")
    for c in cpds:
        for lb in order:
            sub = [r for r in rows if r["compound"] == c and r["run"] == lb]
            if not sub:
                continue
            v = [r["dG"] for r in sub]
            sp = [q for q in (num(r, "pose_rmsd") for r in sub)
                  if q is not None]
            first = [r for r in sub
                     if set(r["pose_ranks"].split("/")) == {"1"}]
            log.append(f"  {c:10s}{lb:14s}{len(v):4d}{mean(v):10.2f}"
                       f"{sd(v):7.2f}"
                       + (f"{mean(sp):9.2f}" if sp else f"{'-':>9s}")
                       + f"{len(first):6d}/{len(sub):<4d}")
        log.append("")

    text = "\n".join(log)
    print(text)
    if a.out:
        head = ["run", "compound", "receptor", "dG", "pose_ranks",
                "pose_rmsd", "seeds"]
        with open(a.out + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=head, restval="")
            w.writeheader()
            w.writerows(rows)
        with open(a.out + ".txt", "w") as f:
            f.write(text + "\n")
        print(f"\n[out] {a.out}.csv, {a.out}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

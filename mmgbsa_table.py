#!/usr/bin/env python3
"""
MM-GBSA binding energies for several compounds in two subunits, by component.

The electrostatic term on its own has misled every comparison it was used
for here, because a charged residue's attraction is mostly repaid by the cost
of stripping its solvation. Reading the components together fixes that: the
van der Waals term, the electrostatics, the polar solvation that cancels most
of it, the non-polar surface term, and their sum.

The sum is an enthalpy-like quantity. No entropy term is computed, so these
are not free energies of binding and their absolute values mean little. The
difference between two subunits for one compound is on firmer ground, since
the same ligand and the same approximations appear on both sides, and that
difference is what the second table reports.

Several files for one system are treated as replicas and averaged, with the
spread shown. That spread is the only guide to which differences mean
anything: a gap between two compounds smaller than the spread within one of
them is not a result.

    python mmgbsa_table.py \\
        --system 'EV043:a2:r1.dat,r2.dat,r3.dat,r4.dat' \\
        --system 'EV043:a1:a1.dat' \\
        --system 'VB004:a2:vb_a2.dat' --system 'VB004:a1:vb_a1.dat' \\
        --system 'CX-4945:a2:cx_a2.dat' --system 'CX-4945:a1:cx_a1.dat'
"""
import argparse, os, re, sys
import numpy as np

# the component names MMPBSA.py writes, in the order they are reported
COMPONENTS = ["VDWAALS", "EEL", "EGB", "ESURF", "DELTA G gas",
              "DELTA G solv", "DELTA TOTAL"]
SHORT = {"VDWAALS": "VDW", "EEL": "EEL", "EGB": "EGB", "ESURF": "ESURF",
         "DELTA G gas": "GAS", "DELTA G solv": "SOLV",
         "DELTA TOTAL": "TOTAL"}
FLOAT = re.compile(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?")


def parse(path):
    """The difference section's components. The file also reports the complex,
    the receptor and the ligand on their own; only their difference is a
    binding energy, so the others are skipped."""
    out, started = {}, False
    for ln in open(path, errors="replace"):
        low = ln.strip().lower()
        if low.startswith("differences"):
            started = True
            continue
        if not started:
            continue
        for c in COMPONENTS:
            if ln.strip().upper().startswith(c.upper()):
                v = FLOAT.findall(ln)
                if v:
                    out[c] = float(v[0])
                break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", action="append", required=True,
                   metavar="COMPOUND:SUBUNIT:FILE[,FILE...]",
                   help="several files for one system are replicas")
    p.add_argument("--test-subunit", default="a2")
    p.add_argument("--ref-subunit", default="a1")
    p.add_argument("--test-label", default="")
    p.add_argument("--ref-label", default="")
    p.add_argument("--out")
    a = p.parse_args()

    tl = a.test_label or a.test_subunit
    rl = a.ref_label or a.ref_subunit

    data, order = {}, []
    for spec in a.system:
        bits = spec.split(":")
        if len(bits) != 3:
            sys.exit(f"--system wants COMPOUND:SUBUNIT:FILE, got {spec!r}")
        c, s, fl = (x.strip() for x in bits)
        vals = []
        for f in fl.split(","):
            f = os.path.expanduser(f.strip())
            if not os.path.exists(f):
                sys.exit(f"no such file: {f}")
            d = parse(f)
            if "DELTA TOTAL" not in d:
                sys.exit(f"no difference section recognised in {f}")
            vals.append(d)
        data[(c, s)] = vals
        if c not in order:
            order.append(c)

    comps = [c for c in COMPONENTS if any(
        c in v for vs in data.values() for v in vs)]

    log = ["[in] MM-GBSA binding energies, kcal/mol",
           "     these are enthalpy-like: no entropy term is computed, so "
           "they are not free energies",
           "     a value written as mean+-sd is the average over replicas; "
           "the spread is the yardstick"]

    log.append("")
    log.append("=== by system ===")
    hdr = f"  {'system':16s}{'n':>3s}"
    for c in comps:
        hdr += f"{SHORT[c]:>16s}"
    log.append(hdr)
    rows = []
    for (c, s), vs in sorted(data.items()):
        line = f"  {c + ' ' + s:16s}{len(vs):3d}"
        rec = {"compound": c, "subunit": s, "n": len(vs)}
        for k in comps:
            xs = np.array([v[k] for v in vs if k in v], dtype=float)
            if not len(xs):
                line += f"{'-':>16s}"
                continue
            m = xs.mean()
            rec[k] = m
            if len(xs) > 1:
                sd = xs.std(ddof=1)
                rec[k + "_sd"] = sd
                line += f"{m:10.2f}+-{sd:<4.1f}"
            else:
                line += f"{m:16.2f}"
        log.append(line)
        rows.append(rec)

    log.append("")
    log.append(f"=== isoform preference, {tl} minus {rl} ===")
    log.append(f"    a negative number means the compound binds {tl} more "
               f"tightly")
    hdr = f"  {'compound':16s}"
    for c in comps:
        hdr += f"{SHORT[c]:>11s}"
    log.append(hdr)
    deltas = {}
    for c in order:
        t = data.get((c, a.test_subunit))
        r = data.get((c, a.ref_subunit))
        if not t or not r:
            log.append(f"  {c:16s}   only one subunit given")
            continue
        line = f"  {c:16s}"
        d = {}
        for k in comps:
            xt = [v[k] for v in t if k in v]
            xr = [v[k] for v in r if k in v]
            if not xt or not xr:
                line += f"{'-':>11s}"
                continue
            d[k] = float(np.mean(xt) - np.mean(xr))
            line += f"{d[k]:11.2f}"
        deltas[c] = d
        log.append(line)

    # the spread within one system, which any difference has to clear
    sds = [v.std(ddof=1) for k in ["DELTA TOTAL"]
           for vs in data.values() if len(vs) > 1
           for v in [np.array([x[k] for x in vs if k in x], dtype=float)]]
    log.append("")
    if sds:
        worst = max(sds)
        log.append(f"=== how much of this is noise ===")
        log.append(f"    the widest replica spread on the total is "
                   f"{worst:.2f} kcal/mol (1 sd, within a single system)")
        log.append(f"    a difference between two compounds smaller than "
                   f"about {worst:.0f} kcal/mol is not distinguishable from "
                   f"that spread")
        if len(deltas) > 1:
            ks = list(deltas)
            log.append("")
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    x, y = ks[i], ks[j]
                    if "DELTA TOTAL" not in deltas[x] or \
                            "DELTA TOTAL" not in deltas[y]:
                        continue
                    g = deltas[y]["DELTA TOTAL"] - deltas[x]["DELTA TOTAL"]
                    verdict = ("clears it" if abs(g) > worst
                               else "inside the spread")
                    log.append(f"    {x} vs {y}: they differ by {g:.2f} on "
                               f"the preference - {verdict}")
    else:
        log.append("=== how much of this is noise ===")
        log.append("    no system has replicates, so nothing here carries an "
                   "error bar and no difference can be called real")

    if deltas:
        log.append("")
        log.append("=== which term drives each preference ===")
        for c, d in deltas.items():
            if "DELTA TOTAL" not in d:
                continue
            parts = [(k, v) for k, v in d.items()
                     if k not in ("DELTA TOTAL", "DELTA G gas",
                                  "DELTA G solv")]
            parts.sort(key=lambda kv: kv[1])
            tot = d["DELTA TOTAL"]
            where = tl if tot < 0 else rl
            log.append(f"  {c}: prefers {where} by {abs(tot):.2f}   "
                       + ", ".join(f"{SHORT[k]} {v:+.2f}" for k, v in parts))

    text = "\n".join(log)
    print(text)
    if a.out:
        import pandas as pd
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
        with open(os.path.splitext(a.out)[0] + ".txt", "w") as f:
            f.write(text + "\n")
        print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

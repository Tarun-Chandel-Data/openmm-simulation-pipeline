#!/usr/bin/env python3
"""
Per-residue decomposition across two compounds and two subunits, as the
difference of differences that selectivity actually is.

A residue's contribution in one subunit says nothing about selectivity. What
does is whether the difference between the two compounds is itself different
between the two subunits:

    dd = [test(compound B) - test(compound A)] in the test subunit
       - [the same difference] in the reference subunit

which is what the last column reports.

Two things this fixes about reading the electrostatic column alone. Every
residue near a ligand that carries formal charge gives a large electrostatic
term, and most of it is paid back by the polar solvation term, so a ranking on
electrostatics alone ranks residues by their charge. Both are printed here
beside the total, and the total is what a contribution means. And a residue
missing from a file is reported as absent rather than as zero, because the
files list only the residues above a cutoff and treating a missing row as no
interaction silently invents the strongest possible result.

    python decomp_table.py \\
        --system EV043:a2:ev043_a2/FINAL_DECOMP_MMPBSA.dat \\
        --system EV043:a1:ev043_a1/FINAL_DECOMP_MMPBSA.dat \\
        --system VB004:a2:vb004_a2/FINAL_DECOMP_MMPBSA.dat \\
        --system VB004:a1:vb004_a1/FINAL_DECOMP_MMPBSA.dat \\
        --offset 'a1:1,a2:6' --residues LYS69,TYR116,ILE117
"""
import argparse, os, re, sys
import numpy as np

# how the headings MMPBSA writes map onto the short names used here
TERMS = [("INT", ("internal",)),
         ("VDW", ("van der waals", "vdwaals", "vdw")),
         ("EEL", ("electrostatic", "eel")),
         ("POL", ("polar solvation", "egb", "epb")),
         ("NPOL", ("non-polar solv", "nonpolar solv", "enpolar", "esurf")),
         ("TOTAL", ("total",))]
FLOAT = re.compile(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?")
RESID = re.compile(r"\b([A-Za-z]{2,4})\s+(\d+)\b")


def term_of(text):
    t = text.strip().lower()
    for short, keys in TERMS:
        if any(k in t for k in keys):
            return short
    return None


def parse(path, region_want, section_want):
    """{(resname, resnum): {term: (avg, sd)}} from an MMPBSA decomposition
    file.

    These files hold the same table many times over: once per region
    (Complex, Receptor, Ligand and their difference, DELTAS) and within each,
    once per section (Total, Sidechain, Backbone). Reading straight through
    and keeping the last row for each residue therefore returns the backbone
    of whatever region came last, which is a small number for every residue
    and looks like a real result. Each table is kept separately here and the
    one asked for is returned, with what was found reported so the wrong one
    cannot be used in silence."""
    raw = open(path, errors="replace").read().splitlines()
    tables, order = {}, None
    region, section, cols = None, None, None
    for ln in raw:
        low = ln.strip().lower()
        m = re.match(r"^(complex|receptor|ligand|deltas)\s*:?\s*$", low)
        if m:
            region, section, cols = m.group(1).upper(), None, None
            continue
        m = re.match(r"^(total|sidechain|backbone)\s+energy\s+decomposition",
                     low)
        if m:
            section = m.group(1).upper()
            cols = None
            continue
        if "residue" in low and any(
                any(k in low for k in keys) for _, keys in TERMS):
            seen, c = set(), []
            for cell in re.split(r",|\|", ln):
                sh = term_of(cell)
                if sh and sh not in seen:
                    seen.add(sh)
                    c.append(sh)
            if len(c) >= 2:
                cols, order = c, c
            continue
        if cols is None:
            continue
        m = RESID.search(ln)
        if not m:
            continue
        vals = [float(x) for x in FLOAT.findall(ln)]
        if len(vals) < len(cols):
            continue
        # the file gives each term as an average, a deviation and an error;
        # whichever of those it carries, the average comes first
        per = len(vals) // len(cols)
        d = {}
        for i, t in enumerate(cols):
            d[t] = (vals[i * per],
                    vals[i * per + 1] if per >= 2 else np.nan)
        tables.setdefault((region, section), {})[
            (m.group(1).upper(), int(m.group(2)))] = d

    if not tables:
        return {}, order, None
    key = (region_want, section_want)
    if key not in tables:
        # fall back to the closest thing present, preferring the difference
        cand = [k for k in tables
                if (region_want is None or k[0] == region_want)
                and (section_want is None or k[1] == section_want)]
        if not cand:
            cand = sorted(tables, key=lambda k: (k[0] != "DELTAS",
                                                 k[1] != "TOTAL"))
        key = cand[0]
    return tables[key], order, (key, sorted(tables))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", action="append", required=True,
                   metavar="COMPOUND:SUBUNIT:FILE")
    p.add_argument("--offset", default="",
                   help="canonical number minus the file's, per subunit, "
                        "e.g. 'a1:1,a2:6'")
    p.add_argument("--residues", required=True,
                   help="residues in canonical numbering, e.g. LYS69,TYR116. "
                        "Where the two subunits number the equivalent residue "
                        "differently, give the other one in brackets: "
                        "'LYS69[a1=LYS68]'")
    p.add_argument("--terms", default="EEL,POL,TOTAL",
                   help="of INT, VDW, EEL, POL, NPOL, TOTAL")
    p.add_argument("--region", default="DELTAS",
                   help="which of Complex, Receptor, Ligand or DELTAS to "
                        "read. DELTAS is the difference and the only one that "
                        "is a contribution to binding")
    p.add_argument("--section", default="TOTAL",
                   help="TOTAL, SIDECHAIN or BACKBONE")
    p.add_argument("--test-subunit", default="a2")
    p.add_argument("--ref-subunit", default="a1")
    p.add_argument("--width", type=int, default=11)
    p.add_argument("--dump", action="store_true",
                   help="print the first lines of each file and stop, for "
                        "when the layout is not recognised")
    p.add_argument("--out")
    a = p.parse_args()

    off = {}
    for bit in a.offset.split(","):
        if ":" in bit:
            k, v = bit.split(":", 1)
            off[k.strip()] = int(v)
    want_terms = [x.strip().upper() for x in a.terms.split(",") if x.strip()]
    def split_res(x):
        nm = "".join(c for c in x if c.isalpha()).upper()
        n = "".join(c for c in x if c.isdigit())
        return (nm, int(n)) if n else (None, None)

    # CK2a and CK2a' number the equivalent residue differently, so a single
    # offset per subunit cannot reach both. A bracket names the exception.
    want_res = []
    for spec in re.split(r",(?![^\[]*\])", a.residues):
        spec = spec.strip()
        if not spec:
            continue
        over, base = {}, spec
        mb = re.match(r"^([^\[]+)\[(.*)\]$", spec)
        if mb:
            base = mb.group(1).strip()
            for bit in mb.group(2).split(","):
                if "=" not in bit:
                    sys.exit(f"bracket wants SUBUNIT=RESIDUE, got {bit!r}")
                k, v = bit.split("=", 1)
                onm, onum = split_res(v.strip())
                if onum is None:
                    sys.exit(f"bracket wants a name and number, got {v!r}")
                over[k.strip()] = (onm, onum)
        nm, n = split_res(base)
        if n is None:
            sys.exit(f"--residues wants a name and number, got {spec!r}")
        want_res.append((nm, n, base, over))

    systems = []
    for spec in a.system:
        bits = spec.split(":")
        if len(bits) != 3:
            sys.exit(f"--system wants COMPOUND:SUBUNIT:FILE, got {spec!r}")
        c, s, f = (x.strip() for x in bits)
        f = os.path.expanduser(f)
        if not os.path.exists(f):
            sys.exit(f"no such file: {f}")
        systems.append((c, s, f))

    if a.dump:
        for c, s, f in systems:
            print(f"--- {c} {s}: {f}")
            for ln in open(f, errors="replace").read().splitlines()[:25]:
                print("   ", ln)
            print()
        return 0

    data, log = {}, []
    for c, s, f in systems:
        d, order, got = parse(f, a.region.upper(), a.section.upper())
        if not d:
            sys.exit(f"no residue rows recognised in {f}. Run again with "
                     f"--dump and send the output")
        data[(c, s)] = d
        used, avail = got
        warn = ("" if used == (a.region.upper(), a.section.upper())
                else f"   <-- asked for {a.region.upper()}/"
                     f"{a.section.upper()}, not present")
        log.append(f"[in] {c} {s}: {len(d)} residues from "
                   f"{used[0]}/{used[1]}, terms {', '.join(order)}  "
                   f"({os.path.basename(f)}){warn}")
        log.append(f"     tables in this file: "
                   + ", ".join(f"{r}/{sc}" for r, sc in avail))

    comps, subs = [], []
    for c, s, _ in systems:
        if c not in comps:
            comps.append(c)
        if s not in subs:
            subs.append(s)
    log.append(f"     offsets: " + (", ".join(f"{k}+{v}" for k, v in
                                              off.items()) or "none"))
    log.append(f"     a residue the file does not list is shown as n/l, not "
               f"as zero")

    found = {}

    def locate(c, s, nm, num, over):
        """The row for this residue in this file, and what it is called there.
        A bracket for this subunit replaces the canonical name and number."""
        d = data.get((c, s))
        if d is None:
            return None, None
        wnm, wnum = over.get(s, (nm, num))
        key = (wnm, wnum - off.get(s, 0))
        if key in d:
            return d[key], wnm
        # the name may be a protonation variant, HIE for HIS and so on
        for (rn, rnum), v in d.items():
            if rnum == key[1]:
                return v, rn
        return None, None

    def get(c, s, nm, num, over, term):
        row, rn = locate(c, s, nm, num, over)
        if row is not None:
            # record what was asked for in THIS subunit, which a bracket may
            # have replaced, so an override does not read as a mismatch
            anm, anum = over.get(s, (nm, num))
            found[(c, s, nm, num)] = (rn, anm, anum - off.get(s, 0))
        return None if row is None else row.get(term, (np.nan, np.nan))[0]

    w = a.width
    can_dd = (a.test_subunit in subs and a.ref_subunit in subs
              and len(comps) == 2)
    log.append("")
    log.append("=== per-residue contribution, kcal/mol ===")
    if can_dd:
        log.append(f"    d is {a.test_subunit} minus {a.ref_subunit}; "
                   f"dd is d({comps[1]}) minus d({comps[0]}), the "
                   f"selectivity term")
    hdr = f"  {'residue':10s}{'term':7s}"
    for c in comps:
        for s in subs:
            hdr += f"{c + ' ' + s:>{w}s}"
        if can_dd:
            hdr += f"{'d ' + c:>{w}s}"
    if can_dd:
        hdr += f"{'dd':>{w}s}"
    log.append(hdr)

    rows = []
    for nm, num, spec, over in want_res:
        first = True
        for term in want_terms:
            line = f"  {spec if first else '':10s}{term:7s}"
            deltas, rec = {}, {"residue": spec, "term": term}
            for c in comps:
                vals = {}
                for s in subs:
                    v = get(c, s, nm, num, over, term)
                    vals[s] = v
                    rec[f"{c}_{s}"] = v
                    line += (f"{'n/l':>{w}s}" if v is None
                             else f"{v:{w}.2f}")
                if can_dd:
                    tv, rv = vals.get(a.test_subunit), vals.get(a.ref_subunit)
                    d = (None if tv is None or rv is None else tv - rv)
                    deltas[c] = d
                    rec[f"d_{c}"] = d
                    line += (f"{'n/l':>{w}s}" if d is None
                             else f"{d:{w}.2f}")
            if can_dd:
                d0, d1 = deltas.get(comps[0]), deltas.get(comps[1])
                dd = (None if d0 is None or d1 is None else d1 - d0)
                rec["dd"] = dd
                line += (f"{'n/l':>{w}s}" if dd is None else f"{dd:{w}.2f}")
            log.append(line)
            rows.append(rec)
            first = False
        log.append("")

    missing = [(c, s, spec) for nm, num, spec, over in want_res for c in comps
               for s in subs if get(c, s, nm, num, over, want_terms[0]) is None]
    if missing:
        log.append("=== not listed ===")
        log.append("    these rows are absent from their file. The file lists "
                   "only residues above its own cutoff, so absent means the "
                   "contribution was below that cutoff, NOT that it is zero.")
        log.append("    A difference computed against an absent row is a lower "
                   "bound on the size of the effect, not a measurement of it.")
        for c, s, spec in missing:
            log.append(f"    {c} {s}: {spec}")

    if found:
        log.append("=== which residue each file supplied ===")
        log.append("    a name here that is not the one asked for means the "
                   "row was matched on its number alone; check it is the "
                   "residue you meant")
        for (c, s, nm, num), (rn, anm, fnum) in sorted(found.items()):
            mark = "" if rn.upper()[:3] == anm[:3] else "   <-- name differs"
            via = "" if (anm, num) == (nm, num) and anm == nm else \
                f"  (for {nm}{num})"
            log.append(f"    {c} {s}: asked {anm}{fnum}, file has {rn}{fnum}"
                       f"{via}{mark}")
        log.append("")

    if can_dd:
        log.append("")
        log.append("=== reading the total against the electrostatics ===")
        for nm, num, spec, over in want_res:
            e, t = None, None
            for term, store in (("EEL", "e"), ("TOTAL", "t")):
                vs = [get(c, s, nm, num, over, term) for c in comps for s in subs]
                if any(v is None for v in vs):
                    continue
                d = ((get(comps[1], a.test_subunit, nm, num, over, term)
                      - get(comps[1], a.ref_subunit, nm, num, over, term))
                     - (get(comps[0], a.test_subunit, nm, num, over, term)
                        - get(comps[0], a.ref_subunit, nm, num, over, term)))
                if store == "e":
                    e = d
                else:
                    t = d
            if e is None and t is None:
                continue
            if e is not None and t is not None:
                keep = 100.0 * t / e if abs(e) > 1e-9 else np.nan
                log.append(f"  {spec}: dd is {e:.2f} on electrostatics and "
                           f"{t:.2f} on the total — "
                           + (f"{keep:.0f}% survives solvation"
                              if np.isfinite(keep) else "n/a"))
            else:
                log.append(f"  {spec}: only one of the two terms is available;"
                           f" the electrostatic figure alone is not a "
                           f"contribution")

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

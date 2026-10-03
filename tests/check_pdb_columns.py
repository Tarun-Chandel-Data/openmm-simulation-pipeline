#!/usr/bin/env python3
"""The ATOM record is read by column, so it is checked by column.

The failure this guards against produced a residue called IG: LIG written
one column early, accepted by the writer, accepted by the reader, and
rejected only by tleap several steps later with an error naming an atom
rather than a column.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mmgbsa_ensemble import pdb_atom_line                      # noqa: E402

FIELDS = [("record", 0, 6, "ATOM  "), ("serial", 6, 11, "7"),
          ("name", 12, 16, "CL1"), ("altloc", 16, 17, ""),
          ("resname", 17, 20, "LIG"), ("chain", 21, 22, "A"),
          ("resseq", 22, 26, "1"), ("element", 76, 78, "Cl")]

def main():
    bad = []
    for name, ln in (("two letter", pdb_atom_line(7, "CL1", "LIG", "A", 1,
                                                  -12.345, 6.7, 0.0, "CL")),
                     ("one letter", pdb_atom_line(1, "C1", "LIG", "A", 1,
                                                  0.0, 0.0, 0.0, "C")),
                     ("four char ", pdb_atom_line(1, "HD12", "LIG", "A", 1,
                                                  0.0, 0.0, 0.0, "H"))):
        for f, lo, hi, want in FIELDS:
            got = ln[lo:hi].strip()
            if f in ("serial", "resseq", "element", "name") and \
                    name != "two letter":
                continue
            if f == "altloc":
                if got:
                    bad.append(f"{name}: altLoc column not blank: {got!r}")
                continue
            if got != want.strip():
                bad.append(f"{name}: {f} reads {got!r}, wanted {want!r}")
        for lo, hi, want in ((30, 38, None), (38, 46, None), (46, 54, None)):
            try:
                float(ln[lo:hi])
            except ValueError:
                bad.append(f"{name}: columns {lo + 1}-{hi} are not a number: "
                           f"{ln[lo:hi]!r}")
    for b in bad:
        print("FAIL:", b)
    print("PASS" if not bad else f"{len(bad)} failures")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())

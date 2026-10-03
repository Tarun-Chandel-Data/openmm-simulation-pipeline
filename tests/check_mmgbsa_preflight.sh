#!/bin/bash
# Exercise every line of the preflight where amber is not installed.
#
# The crash this catches was a name used in one function and built in
# another: unreachable on a machine without amber, and the first thing to
# run on a machine with it. The stubs answer for amber so the whole path
# runs here too.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

mkdir -p "$work"/{par,rec,poses/a2_c00}
python3 - "$work" <<'PY'
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
w = sys.argv[1]
m = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1Cl"))
AllChem.EmbedMolecule(m, randomSeed=5)
m.SetProp("CNNscore", "0.9")
sw = Chem.SDWriter(f"{w}/poses/a2_c00/TEST__a2_c00__r1.sdf"); sw.write(m); sw.close()
g = {"C": "ca", "CL": "cl", "H": "ha"}
L = ["@<TRIPOS>MOLECULE", "LIG", "0 0 1 0 0", "SMALL", "USER_CHARGES", "",
     "@<TRIPOS>ATOM"]
for i, a in enumerate(m.GetAtoms(), 1):
    e = a.GetSymbol().upper()
    L.append(f"{i:7d} {e}{i:<4d}  0.0 0.0 0.0 {g[e]:<6s}  1 LIG  0.0")
open(f"{w}/par/LIG.mol2", "w").write("\n".join(L) + "\n")
for n in ("LIG.frcmod", "LIG.lib"):
    open(f"{w}/par/{n}", "w").write("x\n")
PY

out="$(PATH="$here/stub_amber:$PATH" python3 "$root/mmgbsa_ensemble.py" \
      --ligand-params TEST="$work/par" --receptors "$work/rec" \
      --poses "$work/poses" --compounds TEST --out "$work/out" --np 2 2>&1)"
rc=$?
echo "$out"
fail=0
[[ $rc -eq 0 ]] || { echo "FAIL: exit $rc"; fail=1; }
grep -q "same atoms in the same order" <<< "$out" || { echo "FAIL: atom order not checked"; fail=1; }
grep -q "igb=2" <<< "$out" || { echo "FAIL: protocol not reported"; fail=1; }
grep -qi "traceback" <<< "$out" && { echo "FAIL: it crashed"; fail=1; }
[[ $fail -eq 0 ]] && echo "PASS"
exit $fail

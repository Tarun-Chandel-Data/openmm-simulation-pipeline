#!/bin/bash
# Dock every ligand into every crystal structure, with the box placed per
# structure and each run repeated under several seeds.
#
# The box centre is computed from named pocket residues in each receptor
# rather than fixed once, because the structures are not in a common frame: a
# centre taken from one of them lands in solvent in another, and the run still
# finishes and still reports scores.
#
# Each ligand is docked under several seeds. A single docking run of this kind
# has given differences between compounds smaller than the differences between
# repeats of one compound, so a number from one run is not evidence; a
# frequency over seeds and structures is.
#
# Output goes to <OUT_ROOT>/<structure>/<ligand>__<structure>__s<seed>.sdf,
# which is the name the analysis scripts in this repository read.
set -euo pipefail

# ------------------------------------------------------------------ settings
PROT_DIR="/home/tarun/docking_files/TEST/new/crystal_str/receptor"   # every .pdb
LIG_DIR="/home/tarun/docking_files/TEST/new/crystal_str/ligand"      # every .mol
OUT_ROOT="$LIG_DIR/results"
GNINA="/home/tarun/bin/gnina"

# The residues whose centroid places the box, in the receptor's own numbering.
#
# These are the two hinge residues the experiment measures, so the box is
# centred on the contact being tested. With a 25 A box the site is covered
# either way, but the hinge sits at one edge of the ATP cleft rather than in
# the middle, so a centroid taken from it alone puts a good part of the box in
# solvent. Adding two or three more pocket residues - the catalytic lysine,
# the gatekeeper, the DFG aspartate - moves the centre into the cleft without
# changing what is measured. The atom count printed per structure below says
# whether it matters here.
POCKET_RES="116 117"
CHAIN="A"                      # leave empty to use every chain

BOX=25                         # angstrom, per side
EXH=64
MODES=10
SEEDS="1 2 3"
CPU=20
CNN_SCORING="rescore"
SORT_BY="CNNscore"             # pose 1 is then the best-scoring pose

# ------------------------------------------------------------------ checks
[[ -x "$GNINA" ]] || { echo "not executable: $GNINA" >&2; exit 1; }
[[ -d "$PROT_DIR" ]] || { echo "no such directory: $PROT_DIR" >&2; exit 1; }
[[ -d "$LIG_DIR" ]] || { echo "no such directory: $LIG_DIR" >&2; exit 1; }
[[ "$POCKET_RES" == *X* ]] && { echo "POCKET_RES still has placeholders" >&2; exit 1; }

shopt -s nullglob
prots=("$PROT_DIR"/*.pdb)
ligs=("$LIG_DIR"/*.mol)
(( ${#prots[@]} )) || { echo "no .pdb in $PROT_DIR" >&2; exit 1; }
(( ${#ligs[@]} ))  || { echo "no .mol in $LIG_DIR" >&2; exit 1; }

nseeds=$(wc -w <<< "$SEEDS")
total=$(( ${#prots[@]} * ${#ligs[@]} * nseeds ))
echo "[plan] ${#ligs[@]} ligands x ${#prots[@]} structures x $nseeds seeds = $total runs"
echo "[plan] box ${BOX} A, exhaustiveness ${EXH}, ${MODES} modes, sorted by ${SORT_BY}"
mkdir -p "$OUT_ROOT"

HELPER="$(mktemp)"
trap 'rm -f "$HELPER"' EXIT
cat > "$HELPER" <<'PYHELPER'
import sys
mode, pdb, chain = sys.argv[1], sys.argv[2], sys.argv[3]


def heavy(pdb, chain):
    for ln in open(pdb, errors="replace"):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        if ln[76:78].strip() == "H" or ln[12:16].strip().startswith("H"):
            continue
        if chain and ln[21] != chain and mode == "centre":
            continue
        try:
            yield int(ln[22:26]), (float(ln[30:38]), float(ln[38:46]),
                                   float(ln[46:54]))
        except ValueError:
            continue


if mode == "centre":
    want = {int(x) for x in sys.argv[4:]}
    xs = [c for n, c in heavy(pdb, chain) if n in want]
    if len(xs) < 3:
        sys.exit("only %d atoms matched in %s; check the residue numbers "
                 "and the chain" % (len(xs), pdb))
    n = len(xs)
    print(" ".join("%.3f" % (sum(c[i] for c in xs) / n) for i in range(3)))
else:
    # how much receptor the box encloses
    cx, cy, cz, b = (float(x) for x in sys.argv[4:8])
    h = b / 2.0
    print(sum(1 for _, (x, y, z) in heavy(pdb, "")
              if abs(x - cx) <= h and abs(y - cy) <= h and abs(z - cz) <= h))
PYHELPER

# ------------------------------------------------------------------ docking
done_n=0
for prot in "${prots[@]}"; do
  pbase=$(basename "$prot" .pdb)
  read -r CX CY CZ < <(python3 "$HELPER" centre "$prot" "$CHAIN" $POCKET_RES)
  inbox=$(python3 "$HELPER" occupancy "$prot" "" "$CX" "$CY" "$CZ" "$BOX")
  outdir="$OUT_ROOT/$pbase"
  mkdir -p "$outdir"
  echo
  echo "=== $pbase   centre $CX $CY $CZ   ${inbox} receptor heavy atoms in the box"
  if (( inbox < 400 )); then
    echo "    [warn] low for a ${BOX} A box on an ATP site. The centre may sit"
    echo "           at the mouth of the pocket, leaving much of the box in"
    echo "           solvent; consider adding pocket residues to POCKET_RES."
  fi

  for lig in "${ligs[@]}"; do
    lbase=$(basename "$lig" .mol)
    for seed in $SEEDS; do
      out="$outdir/${lbase}__${pbase}__s${seed}.sdf"
      log="$outdir/${lbase}__${pbase}__s${seed}.log"
      done_n=$(( done_n + 1 ))
      if [[ -s "$out" ]]; then
        echo "  [$done_n/$total] $lbase seed $seed - already done"
        continue
      fi
      echo "  [$done_n/$total] $lbase seed $seed"
      "$GNINA" -r "$prot" -l "$lig" \
        --center_x "$CX" --center_y "$CY" --center_z "$CZ" \
        --size_x "$BOX" --size_y "$BOX" --size_z "$BOX" \
        --cnn_scoring "$CNN_SCORING" \
        --pose_sort_order "$SORT_BY" \
        --seed "$seed" \
        --exhaustiveness "$EXH" --num_modes "$MODES" \
        --no_gpu --cpu "$CPU" \
        -o "$out" --log "$log" \
        || { echo "    gnina failed; see $log" >&2; rm -f "$out"; }
    done
  done
done

echo
echo "[out] $OUT_ROOT/<structure>/<ligand>__<structure>__s<seed>.sdf"
cat <<'NEXT'

Next, with the geometry-validated hydrogen bond test used elsewhere here, so
the counts are comparable with the ensemble ones:

  python3 hbond_geometry.py --receptor <structure>.pdb \
      --poses 'results/<structure>/*.sdf' --out hb_<structure>.csv

Putting this panel through a different hydrogen bond program would apply other
criteria to this half of the work than to the half it is meant to validate.
NEXT

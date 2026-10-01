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
PROT_DIR="/home/tarun/docking_files/TEST/new/panel/proteins"   # every .pdb here
LIG_DIR="/home/tarun/docking_files/TEST/new/panel/ligands"     # every .mol here
OUT_ROOT="$LIG_DIR/results"
GNINA="/home/tarun/bin/gnina"

# The residues whose centroid places the box, as numbers in the receptor's own
# numbering. Give the pocket, not the residues being tested: centring on the
# two hinge residues would define the site by the very contact the experiment
# is meant to measure, and any result would follow from the setup.
POCKET_RES="X X X X X X"
CHAIN="A"                      # leave empty to use every chain

BOX=25                         # angstrom, per side
EXH=64
MODES=20
SEEDS="1 2 3"
CPU=20
CNN_SCORING="rescore"
SORT_BY="CNNscore"             # pose 1 is then the best-scoring pose

# ------------------------------------------------------------------ checks
[[ -x "$GNINA" ]] || { echo "not executable: $GNINA" >&2; exit 1; }
[[ -d "$PROT_DIR" ]] || { echo "no such directory: $PROT_DIR" >&2; exit 1; }
[[ -d "$LIG_DIR" ]] || { echo "no such directory: $LIG_DIR" >&2; exit 1; }
if [[ "$POCKET_RES" == *X* ]]; then
  echo "POCKET_RES still holds placeholders - put the residue numbers in" >&2
  exit 1
fi

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

# centroid of the named residues in one receptor
centre_of () {
  python3 - "$1" "$CHAIN" $POCKET_RES <<'PY'
import sys
pdb, chain, want = sys.argv[1], sys.argv[2], {int(x) for x in sys.argv[3:]}
xs = []
for ln in open(pdb, errors="replace"):
    if not ln.startswith(("ATOM", "HETATM")):
        continue
    if ln[76:78].strip() == "H" or ln[12:16].strip().startswith("H"):
        continue
    if chain and ln[21] != chain:
        continue
    try:
        n = int(ln[22:26])
    except ValueError:
        continue
    if n in want:
        xs.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
if len(xs) < 3:
    sys.exit(f"only {len(xs)} atoms matched in {pdb}; check the residue "
             f"numbers and the chain")
n = len(xs)
print(" ".join(f"{sum(c[i] for c in xs)/n:.3f}" for i in range(3)))
PY
}

# ------------------------------------------------------------------ docking
done_n=0
for prot in "${prots[@]}"; do
  pbase=$(basename "$prot" .pdb)
  read -r CX CY CZ < <(centre_of "$prot")
  outdir="$OUT_ROOT/$pbase"
  mkdir -p "$outdir"
  echo
  echo "=== $pbase   box centre $CX $CY $CZ"

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

Next, with the geometry-validated hydrogen bond test used elsewhere in this
repository, so the counts are comparable with the ensemble ones:

  python3 hbond_geometry.py --receptor <structure>.pdb \
      --poses 'results/<structure>/*.sdf' --out hb_<structure>.csv

Running the panel through a second hydrogen bond program instead would apply
different criteria to this half of the work than to the other half, and the
two sets of counts could not be compared.
NEXT

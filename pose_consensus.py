#!/usr/bin/env python3
"""
Did the seeds find the same pose anywhere, at any rank?

Comparing one seed's best pose with another's asks whether they agreed about
which pose is best. That is a different and harsher question than whether they
found the same pose at all: a run can propose exactly the right binding mode
and rank it seventh. This asks the second question.

Every pose of every seed is compared with every pose of every other seed, and
the combination that agrees best is kept - one pose from each seed, whichever
ranks they happen to hold. The number reported is the worst of the pairwise
distances within that combination, so it is small only when all the seeds
agree, not when two of them do.

Beside it:

  top vs top     the same measurement restricted to each seed's best pose,
                 which is what a reader sees if they open one file
  ranks          where the agreed pose sat in each seed. All ones means the
                 seeds agreed and the scoring agreed. High ranks mean the
                 pose was found every time and ranked badly every time, which
                 is a statement about the scoring function, not the search

The distance is a symmetry-aware RMSD taken in place, without superposition:
poses from one receptor already share a frame.

    python pose_consensus.py --poses crystal_str/ligand/results --out cons
"""
import argparse, glob, itertools, os, sys
import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("needs pandas")
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolAlign
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("needs rdkit")


def parse_name(path):
    b = os.path.basename(path)
    for e in (".sdf.gz", ".sdf"):
        if b.endswith(e):
            b = b[: -len(e)]
            break
    p = b.split("__")
    return (p[0], p[1], p[2]) if len(p) >= 3 else None


def prop(m, k):
    if not m.HasProp(k):
        return np.nan
    try:
        return float(m.GetProp(k))
    except ValueError:
        return np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--poses", required=True)
    p.add_argument("--top-n", type=int, default=0,
                   help="use only each run's N best poses. 0 uses every pose, "
                        "which is the point: a pose ranked tenth still counts "
                        "as having been found")
    p.add_argument("--select-by", default="CNNscore")
    p.add_argument("--only", help="comma-separated compounds, in order")
    p.add_argument("--threshold", type=float, default=2.0,
                   help="how close the seeds have to agree for the cell to "
                        "count as converged, in angstroms")
    p.add_argument("--max-combos", type=int, default=20000,
                   help="above this many combinations the search anchors on "
                        "each pose of the first seed and takes the nearest "
                        "pose in every other seed, instead of trying every "
                        "combination")
    p.add_argument("--out")
    a = p.parse_args()

    keep = ([x.strip() for x in a.only.split(",") if x.strip()]
            if a.only else None)

    dirs = sorted(d for d in glob.glob(os.path.join(a.poses, "*"))
                  if os.path.isdir(d))
    if not dirs:
        sys.exit(f"no structure directories under {a.poses}")

    run, bad = {}, []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.sdf"))):
            nm = parse_name(f)
            if nm is None:
                continue
            cpd, struct, seed = nm
            if keep and cpd not in keep:
                continue
            try:
                mols = [m for m in Chem.SDMolSupplier(f, removeHs=False,
                                                      sanitize=True)
                        if m is not None]
            except Exception as e:
                bad.append((os.path.basename(f), str(e).split("\n")[0]))
                continue
            sc = [(prop(m, a.select_by), m) for m in mols]
            sc = [(v, m) for v, m in sc if not np.isnan(v)]
            if not sc:
                continue
            sc.sort(key=lambda r: -r[0])
            if a.top_n:
                sc = sc[: a.top_n]
            run[(cpd, struct, seed)] = sc

    if not run:
        sys.exit("no runs read")

    def rms(m1, m2):
        try:
            return float(rdMolAlign.CalcRMS(m1, m2))
        except Exception:
            return np.nan

    rows = []
    for (cpd, struct), grp in itertools.groupby(
            sorted(run, key=lambda k: (k[0], k[1], k[2])),
            key=lambda k: (k[0], k[1])):
        seeds = list(grp)
        if len(seeds) < 2:
            continue
        pose = [run[k] for k in seeds]
        n = [len(x) for x in pose]
        # every pose against every pose, once, for each pair of seeds
        D = {}
        for i, j in itertools.combinations(range(len(seeds)), 2):
            M = np.full((n[i], n[j]), np.nan)
            for x in range(n[i]):
                for y in range(n[j]):
                    M[x, y] = rms(pose[i][x][1], pose[j][y][1])
            D[(i, j)] = M

        def worst(pick):
            v = [D[(i, j)][pick[i], pick[j]]
                 for i, j in itertools.combinations(range(len(seeds)), 2)]
            return np.nan if any(np.isnan(q) for q in v) else max(v)

        combos = int(np.prod(n))
        best, best_pick = np.inf, None
        if combos <= a.max_combos:
            for pick in itertools.product(*[range(x) for x in n]):
                w = worst(pick)
                if not np.isnan(w) and w < best:
                    best, best_pick = w, pick
        else:
            # anchor on each pose of the first seed and take the nearest pose
            # in every other seed; cheaper, and the answer it gives is an
            # upper bound on the true best
            for x in range(n[0]):
                pick = [x]
                for j in range(1, len(seeds)):
                    col = D[(0, j)][x]
                    pick.append(int(np.nanargmin(col)))
                w = worst(tuple(pick))
                if not np.isnan(w) and w < best:
                    best, best_pick = w, tuple(pick)

        tops = tuple(0 for _ in seeds)
        tt = worst(tops)
        rows.append({
            "compound": cpd, "structure": struct, "seeds": len(seeds),
            "poses_per_seed": "/".join(str(x) for x in n),
            "consensus_rmsd": best if np.isfinite(best) else np.nan,
            "consensus_ranks": ("/".join(str(i + 1) for i in best_pick)
                                if best_pick else ""),
            "mean_rank": (float(np.mean([i + 1 for i in best_pick]))
                          if best_pick else np.nan),
            "top_vs_top": tt,
            "converged": (bool(np.isfinite(best) and best <= a.threshold))})
    if not rows:
        sys.exit("no compound-structure cell had two or more seeds")
    t = pd.DataFrame(rows)

    log = [f"[in] {len(run)} runs, "
           + ("every pose" if not a.top_n else f"the best {a.top_n} poses")
           + " of each",
           f"     every pose compared with every pose of the other seeds; the "
           f"combination that agrees best is kept",
           f"     symmetry-aware RMSD in place, no superposition"]
    if bad:
        log.append(f"     [note] {len(bad)} files not read")

    order = keep if keep else sorted(t["compound"].unique())
    structs = sorted(t["structure"].unique())
    log.append("")
    log.append("=== did the seeds find the same pose, at any rank? (A) ===")
    log.append("    'agree' is the worst pairwise distance within the best "
               "combination, so it is small only when every seed agrees")
    log.append("    'ranks' is where that pose sat in each seed; 'top v top' "
               "is the same measurement using only each seed's best pose")
    log.append(f"  {'compound':10s}{'structure':12s}{'agree':>8s}"
               f"{'ranks':>12s}{'mean rank':>11s}{'top v top':>11s}")
    for c in order:
        g = t[t.compound == c]
        for st in structs:
            x = g[g.structure == st]
            if not len(x):
                continue
            x = x.iloc[0]
            log.append(f"  {c:10s}{st:12s}{x['consensus_rmsd']:8.2f}"
                       f"{x['consensus_ranks']:>12s}{x['mean_rank']:11.1f}"
                       f"{x['top_vs_top']:11.2f}")
        log.append("")

    log.append("=== by compound ===")
    log.append(f"  {'compound':10s}{'agree':>8s}{'mean rank':>11s}"
               f"{'top v top':>11s}{'converged':>11s}   reading")
    for c in order:
        g = t[t.compound == c]
        if not len(g):
            continue
        k = int(g["converged"].sum())
        note = ("the seeds find one pose" if k == len(g) else
                "the seeds mostly find one pose" if k >= 0.6 * len(g) else
                "the seeds do not find a common pose")
        log.append(f"  {c:10s}{g['consensus_rmsd'].mean():8.2f}"
                   f"{g['mean_rank'].mean():11.1f}"
                   f"{g['top_vs_top'].mean():11.2f}"
                   f"{k:7d}/{len(g):<3d}   {note}")

    log.append("")
    log.append(f"  across all cells: the seeds agree to "
               f"{t['consensus_rmsd'].mean():.2f} A on some pose, against "
               f"{t['top_vs_top'].mean():.2f} A on their best pose")
    log.append(f"  the agreed pose sits at rank {t['mean_rank'].mean():.1f} on "
               f"average")
    if t["mean_rank"].mean() > 2.5:
        log.append("  the seeds find a common pose and rank it differently, so "
                   "what disagrees is the scoring, not the search")

    text = "\n".join(log)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        t.to_csv(a.out + ".csv", index=False)
        with open(a.out + ".txt", "w") as f:
            f.write(text + "\n")
        print(f"\n[out] {a.out}.csv, {a.out}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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


def heavy(m):
    """Hydrogens off before any distance is taken.

    A docked pose's hydrogens are placed by the program, not searched, so a
    distance that counts them measures partly the placement routine. Every
    RMSD here is over heavy atoms."""
    try:
        return Chem.RemoveHs(m)
    except Exception:
        return m


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
    p.add_argument("--pick", choices=("anchor", "rank", "min"),
                   default="anchor",
                   help="how the matching poses are chosen. 'anchor' takes "
                        "every pose of every seed in turn, finds the pose "
                        "nearest to it in each of the other seeds, and keeps "
                        "whichever anchor gives the closest set. 'rank' takes "
                        "the earliest-ranked set that agrees within the "
                        "threshold. 'min' takes the closest set of any ranks")
    p.add_argument("--score", choices=("max", "mean"), default="max",
                   help="a set's distance: the worst of its three seed pairs, "
                        "or their mean. The worst is the stricter, since one "
                        "disagreeing seed means the three did not agree")
    p.add_argument("--dump", metavar="COMPOUND:STRUCTURE",
                   help="print the whole pose-by-pose table for one cell, "
                        "every seed pair, and stop. Nothing here should be "
                        "taken on trust: this is the table the choice is made "
                        "from")
    p.add_argument("--drift-cut", type=float, default=12.0,
                   help="a pose whose centroid sits this far from the median "
                        "centroid of its cell is reported. Poses that left "
                        "the search box are what produce distances larger "
                        "than the box, and they are a docking failure, not a "
                        "measurement")
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
            sc = [(prop(m, a.select_by), heavy(m)) for m in mols]
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

    want = None
    if a.dump:
        if ":" not in a.dump:
            sys.exit("--dump takes COMPOUND:STRUCTURE")
        want = tuple(x.strip() for x in a.dump.split(":", 1))

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

        if want is not None and (cpd, struct) == want:
            print(f"[dump] {cpd} in {struct}: "
                  + ", ".join(f"seed {k[2]} has {len(run[k])} poses"
                              for k in seeds))
            print("       heavy-atom RMSD in place, every pose against every "
                  "pose")
            for i, j in itertools.combinations(range(len(seeds)), 2):
                print(f"\n  seed {seeds[i][2]} (rows) x seed {seeds[j][2]} "
                      f"(columns)")
                print("       " + "".join(f"{y + 1:>7d}" for y in range(n[j])))
                for x in range(n[i]):
                    print(f"  {x + 1:>4d} " + "".join(
                        f"{D[(i, j)][x, y]:7.2f}" for y in range(n[j])))
            print()

        def pairs_of(pick):
            return [(i, j, D[(i, j)][pick[i], pick[j]])
                    for i, j in itertools.combinations(range(len(seeds)), 2)]

        def worst(pick):
            v = [q for _, _, q in pairs_of(pick)]
            if any(np.isnan(q) for q in v):
                return np.nan
            return max(v) if a.score == "max" else float(np.mean(v))

        # The pose wanted is the earliest-ranked one the seeds agree on, not
        # the one they agree on most closely. Those are different: a run can
        # agree at rank 1 to within 0.17 A and at rank 6 to within 0.14, and
        # taking the smaller number reports rank 6, which says the scoring
        # failed when it did not. So among the combinations that agree within
        # the threshold, the one with the lowest ranks is taken, and the
        # distance is whatever that combination gives.
        def nearest(i, x, j):
            """The pose of seed j closest to pose x of seed i."""
            col = (D[(i, j)][x] if i < j else D[(j, i)][:, x])
            return (None if np.all(np.isnan(col))
                    else int(np.nanargmin(col)))

        combos = int(np.prod(n))
        best_pick, floor_pick, floor = None, None, np.inf
        if a.pick == "anchor":
            # every pose of every seed is tried as the anchor; against each,
            # the nearest pose in each other seed is taken, and the anchor
            # whose set holds together best is kept
            for i in range(len(seeds)):
                for x in range(n[i]):
                    pick = [None] * len(seeds)
                    pick[i] = x
                    ok = True
                    for j in range(len(seeds)):
                        if j == i:
                            continue
                        y = nearest(i, x, j)
                        if y is None:
                            ok = False
                            break
                        pick[j] = y
                    if not ok:
                        continue
                    w = worst(tuple(pick))
                    if not np.isnan(w) and w < floor:
                        floor, floor_pick = w, tuple(pick)
            best_pick = floor_pick
        elif combos <= a.max_combos:
            for pick in itertools.product(*[range(x) for x in n]):
                w = worst(pick)
                if np.isnan(w):
                    continue
                if w < floor:
                    floor, floor_pick = w, pick
                if w <= a.threshold:
                    key = (sum(pick), w)
                    if best_pick is None or key < (sum(best_pick),
                                                   worst(best_pick)):
                        best_pick = pick
        else:
            # anchors taken in rank order; in each other seed the earliest
            # pose that agrees is taken, so the first feasible anchor wins
            for x in range(n[0]):
                pick, ok = [x], True
                for j in range(1, len(seeds)):
                    col = D[(0, j)][x]
                    cand = [y for y in range(n[j])
                            if not np.isnan(col[y]) and col[y] <= a.threshold]
                    if not cand:
                        ok = False
                        break
                    pick.append(cand[0])
                near = [x] + [int(np.nanargmin(D[(0, j)][x]))
                              for j in range(1, len(seeds))]
                w = worst(tuple(near))
                if not np.isnan(w) and w < floor:
                    floor, floor_pick = w, tuple(near)
                if ok:
                    w = worst(tuple(pick))
                    if not np.isnan(w) and w <= a.threshold:
                        best_pick = tuple(pick)
                        break
        if a.pick == "min":
            best_pick = floor_pick
        converged = (best_pick is not None
                     and np.isfinite(worst(best_pick))
                     and worst(best_pick) <= a.threshold)
        if best_pick is None:
            best_pick = floor_pick
        best = worst(best_pick) if best_pick else np.nan

        # The control the agreement number needs. Within one run gnina keeps
        # its poses apart, so the closest two distinct poses of a single seed
        # set the scale at which two poses count as different at all. If the
        # seeds agree far more closely than that, they are landing on the same
        # pose; if they agree at about that distance, the agreement is only
        # what taking a minimum over a few hundred comparisons would give
        # anyway, and means nothing.
        wmin = []
        for ps in pose:
            d = [rms(ps[x][1], ps[y][1])
                 for x, y in itertools.combinations(range(len(ps)), 2)]
            d = [q for q in d if not np.isnan(q)]
            if d:
                wmin.append(min(d))

        # where the poses sit, so an impossible distance can be traced to a
        # pose that left the box rather than to the metric
        cen = [np.array(x[1].GetConformer().GetPositions()).mean(0)
               for ps in pose for x in ps]
        med = np.median(np.vstack(cen), axis=0)
        drift = [float(np.linalg.norm(c - med)) for c in cen]
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
            "closest_rmsd": floor if np.isfinite(floor) else np.nan,
            "pairs": ("  ".join(
                f"s{seeds[i][2]}-s{seeds[j][2]} {q:.2f}"
                for i, j, q in pairs_of(best_pick)) if best_pick else ""),
            "within_min": float(np.mean(wmin)) if wmin else np.nan,
            "max_drift": float(max(drift)) if drift else np.nan,
            "poses_far": int(sum(d > a.drift_cut for d in drift)),
            "top_vs_top": tt,
            "converged": converged})
    if want is not None:
        return 0
    if not rows:
        sys.exit("no compound-structure cell had two or more seeds")
    t = pd.DataFrame(rows)

    how = {"anchor": "every pose of every seed tried as the anchor, the "
                     "nearest pose in each other seed taken with it",
           "rank": "the earliest-ranked set that agrees within the threshold",
           "min": "the closest set, at any ranks"}[a.pick]
    log = [f"[in] {len(run)} runs, "
           + ("every pose" if not a.top_n else f"the best {a.top_n} poses")
           + " of each",
           f"     every pose compared with every pose of the other seeds; "
           f"{how}",
           f"     a set's distance is the {a.score} of its seed pairs",
           f"     symmetry-aware RMSD in place, no superposition"]
    if bad:
        log.append(f"     [note] {len(bad)} files not read")

    order = keep if keep else sorted(t["compound"].unique())
    structs = sorted(t["structure"].unique())
    log.append("")
    log.append("=== did the seeds find the same pose, at any rank? (A) ===")
    log.append(f"    {how}")
    log.append("    'agree' is the worst pairwise distance inside that "
               "combination; 'closest' is the best any combination reaches, "
               "whatever its rank")
    log.append("    'ranks' is where that pose sat in each seed; 'top v top' "
               "is the same measurement using only each seed's best pose")
    log.append(f"  {'compound':10s}{'structure':12s}{'agree':>8s}"
               f"{'ranks':>12s}{'closest':>10s}{'top v top':>11s}"
               f"   each pair of seeds")
    for c in order:
        g = t[t.compound == c]
        for st in structs:
            x = g[g.structure == st]
            if not len(x):
                continue
            x = x.iloc[0]
            log.append(f"  {c:10s}{st:12s}{x['consensus_rmsd']:8.2f}"
                       f"{x['consensus_ranks']:>12s}"
                       f"{x['closest_rmsd']:10.2f}{x['top_vs_top']:11.2f}"
                       f"   {x['pairs']}"
                       + ("" if x["converged"] else "   [no agreement]"))
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

    w = t["within_min"].mean()
    g = t["consensus_rmsd"].mean()
    log.append("")
    log.append("=== is the agreement real, or just the smallest of many "
               "comparisons? ===")
    log.append(f"  the seeds agree to {g:.2f} A on the pose they share")
    log.append(f"  within one run, the two closest distinct poses stand "
               f"{w:.2f} A apart")
    if np.isfinite(w) and np.isfinite(g):
        if g < 0.25 * w:
            log.append(f"  the agreement is {w / max(g, 1e-9):.0f} times "
                       f"closer than two poses of one run ever come, so the "
                       f"seeds are landing on the same pose and not on a near "
                       f"miss")
        elif g < w:
            log.append("  the agreement is closer than two poses of one run "
                       "come, but not by much; read it as weak")
        else:
            log.append("  [warn] the seeds agree no more closely than two "
                       "distinct poses of a single run differ. A minimum over "
                       "several hundred comparisons would reach this on its "
                       "own, so this number is not evidence that the seeds "
                       "found the same pose")
    ok = t[(t["consensus_rmsd"] > 0.01) & t["within_min"].notna()]
    if len(ok):
        r = (ok["within_min"] / ok["consensus_rmsd"])
        log.append(f"  per cell that ratio is {r.median():.0f} at the median, "
                   f"{r.min():.0f} at worst, over {len(ok)} cells")
        thin = int((r < 2).sum())
        if thin:
            log.append(f"  [warn] {thin} cells where it is under 2; their "
                       f"agreement carries no weight")

    far = t[t["poses_far"] > 0]
    if len(far):
        log.append("")
        log.append(f"=== poses that left the pocket (centroid more than "
                   f"{a.drift_cut} A from the cell's median) ===")
        log.append("    a distance larger than the box comes from these, not "
                   "from the comparison. The runs below should be looked at "
                   "before their numbers are used")
        log.append(f"  {'compound':10s}{'structure':12s}{'far poses':>11s}"
                   f"{'max drift':>11s}{'agree':>8s}")
        for _, r in far.sort_values("max_drift", ascending=False).iterrows():
            log.append(f"  {r['compound']:10s}{r['structure']:12s}"
                       f"{int(r['poses_far']):11d}{r['max_drift']:11.1f}"
                       f"{r['consensus_rmsd']:8.2f}")
        log.append(f"  {len(far)} of {len(t)} cells affected")

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

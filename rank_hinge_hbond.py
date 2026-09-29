#!/usr/bin/env python3
"""
Rank compounds by which subunit's divergent residues they hydrogen bond to.

At a position where the two subunits differ, a ligand can engage one residue,
the other, both, or neither. Only the compounds that engage one and not the
other discriminate between the subunits; a residue every compound engages, or
none does, separates nothing however important it is to binding.

The script reports both, so a position that cannot rank is visible as such
rather than contributing silently to a score.

    python rank_hinge_hbond.py --table comparison.csv \
        --pairs HIS116:TYR116,VAL117:ILE117 \
        --label1 CK2a --label2 CK2a-prime --out figures/Fig_hinge.png

Each --pairs entry is residue_in_subunit_1:residue_in_subunit_2, given in the
labelling the table uses, not the PDB's, if the two differ.
"""
import argparse, os, re, sys
import numpy as np, pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("needs matplotlib")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 300})
# diverging pair for the two subunits, grey for the neutral cases
C_A1, C_A2, GREY, BOTH = "#2e5eaa", "#d1495b", "#b9b9b9", "#7b5cd6"


def residues(s):
    """The residue labels in a whitespace- or comma-separated cell."""
    if pd.isna(s):
        return set()
    return {t for t in re.split(r"[\s,;]+", str(s).strip()) if t}


def target_mode(d, s1, s2, pairs, idc, a, log):
    """Rank by engagement of named residues in the second subunit alone.

    This answers "which compounds bond to these residues", which is a
    different question from "which compounds prefer this subunit". A compound
    that engages the named residues and their counterparts in the other
    subunit sits at the top of this ranking while discriminating nothing, so
    the counterpart engagement is carried alongside and reported, never folded
    into the rank."""
    want = [t.strip() for t in a.target.split(",") if t.strip()]
    counterpart = {r2: r1 for r1, r2 in pairs}

    hit = pd.DataFrame({r: s2.map(lambda S, x=r: x in S) for r in want})
    n_hit = hit.sum(axis=1)

    # Exclusivity is judged only against counterparts that vary. A counterpart
    # engaged by every compound makes "engages no counterpart" impossible for
    # anyone, so including it would report zero discriminating compounds no
    # matter what the data says, which is an artefact of the test rather than
    # a property of the set.
    cp_cols, skipped = {}, []
    for r in want:
        c = counterpart.get(r)
        if not c:
            continue
        eng = s1.map(lambda S, x=c: x in S)
        if eng.mean() >= 0.95:
            skipped.append((r, c, float(eng.mean())))
        else:
            cp_cols[r] = eng
    cp = pd.DataFrame(cp_cols)
    n_cp = cp.sum(axis=1) if len(cp.columns) else pd.Series(0, index=d.index)
    exclusive = (n_hit == len(want)) & (n_cp == 0)

    log.append(f"\n=== TARGET MODE : engagement of {', '.join(want)} "
               f"in {a.label2} ===")
    for r in want:
        n = int(hit[r].sum())
        log.append(f"     {r:8s} engaged by {n:4d} of {len(d)} "
                   f"({100*n/len(d):.0f}%)"
                   + ("   [note] near-universal, so it barely separates anything"
                      if n >= 0.95*len(d) else ""))
    log.append(f"\n     engaging all {len(want)}: "
               f"{int((n_hit == len(want)).sum())} of {len(d)}")
    for k in range(len(want), -1, -1):
        log.append(f"       {k} of {len(want)}: {int((n_hit == k).sum()):4d}")
    for r, c, frac in skipped:
        log.append(f"\n     [note] the {a.label1} counterpart of {r} is {c}, "
                   f"engaged by {100*frac:.0f}% of compounds. No compound can "
                   f"avoid it, so it is left out of the exclusivity test; "
                   f"including it would report zero discriminating compounds "
                   f"whatever the data held")
    if len(cp.columns):
        top = n_hit == len(want)
        log.append(f"\n     of the {int(top.sum())} engaging all {len(want)}, "
                   f"judged against {', '.join(counterpart[r] for r in cp.columns)}:")
        log.append(f"       {int((top & (n_cp == 0)).sum()):4d} engage no "
                   f"{a.label1} counterpart  -> discriminating")
        log.append(f"       {int((top & (n_cp > 0)).sum()):4d} also engage a "
                   f"{a.label1} counterpart -> bind both, discriminate nothing")
    else:
        log.append(f"\n     [note] no counterpart varies enough to judge "
                   f"exclusivity, so this ranking reports engagement only")

    out = d[[idc]].copy()
    out["n_target"] = n_hit
    out["residues"] = [" ".join(r for r in want if hit.loc[i, r]) for i in d.index]
    out["n_counterpart"] = n_cp
    out["exclusive"] = exclusive
    sort_cols, asc = ["n_target"], [False]
    if a.tiebreak and a.tiebreak in d.columns:
        out[a.tiebreak] = d[a.tiebreak]
        sort_cols.append(a.tiebreak); asc.append(False)
        log.append(f"\n     ties broken on {a.tiebreak}, highest first")
    else:
        if a.tiebreak:
            log.append(f"\n     [warn] tiebreak column '{a.tiebreak}' not in "
                       f"the table; ordering ties by identifier")
    sort_cols += ["exclusive", idc]; asc += [False, True]
    out = out.sort_values(sort_cols, ascending=asc)

    path = os.path.splitext(a.out)[0] + "_target_ranking.csv"
    out.to_csv(path, index=False)
    log.append(f"\n     top {a.top}:")
    for _, r in out.head(a.top).iterrows():
        log.append(f"       {str(r[idc]):16s} {int(r.n_target)}/{len(want)} "
                   f"[{r.residues}]  counterpart {int(r.n_counterpart)}"
                   f"{'  EXCLUSIVE' if r.exclusive else ''}"
                   + (f"  {a.tiebreak}={r[a.tiebreak]:g}"
                      if a.tiebreak and a.tiebreak in out.columns else ""))
    log.append(f"\n     [out] {path}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", default="comparison.csv")
    p.add_argument("--pairs", default="HIS116:TYR116,VAL117:ILE117",
                   help="comma-separated res1:res2, one per divergent position")
    p.add_argument("--hb1-col", default="a1_hbond_res")
    p.add_argument("--hb2-col", default="a2_hbond_res")
    p.add_argument("--id-col", default=None)
    p.add_argument("--measured-col", default="measured",
                   help="measured selectivity, used only to report whether the "
                        "ranking tracks it where measurements exist")
    p.add_argument("--label1", default="CK2α")
    p.add_argument("--label2", default="CK2α′")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--target",
                   help="rank by engagement of these residues in the second "
                        "subunit alone, e.g. TYR116,ILE117, rather than by the "
                        "paired comparison. This ranks binding to that subunit, "
                        "which is not the same as preferring it: a compound "
                        "engaging both subunits' residues ranks top here and "
                        "still discriminates nothing, so the counterpart from "
                        "--pairs is reported beside it")
    p.add_argument("--tiebreak",
                   help="numeric column ordering compounds inside a tier, "
                        "highest first (e.g. a2_n_hbond)")
    p.add_argument("--out", default="Figure_hinge_hbond.png")
    a = p.parse_args()

    d = pd.read_csv(a.table)
    idc = a.id_col or next((c for c in d.columns
                            if re.search(r"cpd|compound|chembl|id$", c.lower())
                            and not pd.api.types.is_numeric_dtype(d[c])),
                           d.columns[0])
    for c in (a.hb1_col, a.hb2_col):
        if c not in d.columns:
            sys.exit(f"column '{c}' not in {a.table}; "
                     f"available: {', '.join(d.columns[:14])}")

    pairs = []
    for tok in a.pairs.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            sys.exit(f"--pairs entry '{tok}' is not res1:res2")
        r1, r2 = (x.strip() for x in tok.split(":", 1))
        pairs.append((r1, r2))

    s1 = d[a.hb1_col].map(residues)
    s2 = d[a.hb2_col].map(residues)
    log = [f"[in] {len(d)} compounds from {a.table}",
           f"     H-bond columns: {a.hb1_col} / {a.hb2_col}"]

    # per position: who engages what
    stats, usable = [], []
    for r1, r2 in pairs:
        e1 = s1.map(lambda S, r=r1: r in S)
        e2 = s2.map(lambda S, r=r2: r in S)
        n_both = int((e1 & e2).sum())
        n_only2 = int((~e1 & e2).sum())
        n_only1 = int((e1 & ~e2).sum())
        n_none = int((~e1 & ~e2).sum())
        n_disc = n_only1 + n_only2
        stats.append(dict(r1=r1, r2=r2, both=n_both, only1=n_only1,
                          only2=n_only2, none=n_none, disc=n_disc,
                          e1=e1, e2=e2))
        log.append(f"\n  {r1} ({a.label1})  vs  {r2} ({a.label2})")
        log.append(f"     engaged in {a.label1}: {int(e1.sum()):4d} "
                   f"({100*e1.mean():.0f}%)")
        log.append(f"     engaged in {a.label2}: {int(e2.sum()):4d} "
                   f"({100*e2.mean():.0f}%)")
        log.append(f"     both {n_both:4d}   only {a.label2} {n_only2:4d}   "
                   f"only {a.label1} {n_only1:4d}   neither {n_none:4d}")
        log.append(f"     discriminating (one and not the other): {n_disc} of "
                   f"{len(d)} ({100*n_disc/len(d):.0f}%)")
        # a position separating a handful of compounds out of hundreds is a
        # constant with a rounding error on it, not a criterion: it shifts one
        # or two scores and reads as signal. The floor scales with the set so
        # it does not become strict on a small one.
        floor = max(2, int(round(0.02 * len(d))))
        if n_disc < floor:
            log.append(f"     [note] only {n_disc} of {len(d)} compounds are "
                       f"separated here, below the floor of {floor} "
                       f"(2 percent). This position is effectively a constant: "
                       f"it is drawn but left out of the score")
        else:
            usable.append(len(stats) - 1)

    if not usable:
        log.append("\n  no position discriminates; there is nothing to rank")

    # the score counts only positions that can discriminate: +1 where the
    # compound engages the second subunit's residue and not the first, -1 the
    # other way. A position engaged identically by every compound adds a
    # constant to every score and is therefore excluded, not merely tied.
    score = np.zeros(len(d), int)
    for i in usable:
        st = stats[i]
        score += (~st["e1"] & st["e2"]).astype(int)
        score -= (st["e1"] & ~st["e2"]).astype(int)
    d = d.assign(_score=score)

    log.append(f"\n  score over {len(usable)} discriminating position(s): "
               f"{', '.join(stats[i]['r1'] + '/' + stats[i]['r2'] for i in usable) or 'none'}")
    for v in sorted(set(score), reverse=True):
        n = int((score == v).sum())
        who = (a.label2 if v > 0 else a.label1 if v < 0 else "neither")
        log.append(f"     score {v:+d}: {n:4d} compounds  "
                   f"({'favours ' + who if v else 'no preference'})")

    if a.measured_col in d.columns:
        m = pd.to_numeric(d[a.measured_col], errors="coerce")
        ok = m.notna()
        if ok.sum() >= 5 and len(set(score[ok.values])) > 1:
            from scipy.stats import spearmanr
            r, pv = spearmanr(score[ok.values], m[ok])
            log.append(f"\n  against measured selectivity, where it exists "
                       f"(n = {int(ok.sum())}): Spearman rho {r:+.3f}, p {pv:.3f}")
        else:
            log.append(f"\n  [note] only {int(ok.sum())} compounds carry a value "
                       f"in '{a.measured_col}', so this ranking cannot be "
                       f"checked against measurement here")

    ranked = d.sort_values(["_score", idc], ascending=[False, True])
    log.append(f"\n  top {a.top} by score:")
    for _, r in ranked.head(a.top).iterrows():
        eng = []
        for st in stats:
            i = r.name
            if st["e2"][i] and not st["e1"][i]:
                eng.append(f"+{st['r2']}")
            elif st["e1"][i] and not st["e2"][i]:
                eng.append(f"-{st['r1']}")
            elif st["e1"][i] and st["e2"][i]:
                eng.append(f"={st['r2'].rstrip('0123456789')}/{st['r1'].rstrip('0123456789')}")
        log.append(f"    {str(r[idc]):16s} score {r._score:+d}   "
                   f"{' '.join(eng) if eng else 'no divergent H-bond'}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    ranked[[idc, "_score"]].rename(columns={"_score": "hinge_score"}).to_csv(
        os.path.splitext(a.out)[0] + "_ranking.csv", index=False)

    if a.target:
        target_mode(d, s1, s2, pairs, idc, a, log)

    # ---- figure
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.6),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(stats))
    h = 0.62
    for i, st in enumerate(stats):
        left = 0.0
        for n, c, lab in ((st["only1"], C_A1, f"only {a.label1}"),
                          (st["both"], BOTH, "both"),
                          (st["only2"], C_A2, f"only {a.label2}"),
                          (st["none"], GREY, "neither")):
            if n:
                axA.barh(i, n, left=left, height=h, color=c, edgecolor="white",
                         linewidth=0.8,
                         label=lab if i == 0 else None)
                if n / len(d) > 0.07:
                    axA.text(left + n/2, i, f"{n}", ha="center", va="center",
                             fontsize=7.5, color="white", weight="bold")
                left += n
    axA.set_yticks(y)
    axA.set_yticklabels([f"{s['r1']} / {s['r2']}" for s in stats], fontsize=8)
    axA.invert_yaxis()
    axA.set_xlabel(f"Compounds ({len(d)})")
    axA.set_xlim(0, len(d))
    axA.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, -0.20))
    axA.set_title("A   How each divergent position is engaged",
                  loc="left", fontsize=9, weight="bold")
    for i, st in enumerate(stats):
        if st["disc"] == 0:
            axA.text(len(d)*0.99, i - 0.42, "cannot rank", ha="right",
                     va="center", fontsize=6.5, color="#777777", style="italic")

    vals = sorted(set(score))
    cnt = [int((score == v).sum()) for v in vals]
    axB.bar(vals, cnt, width=0.6, edgecolor="white",
            color=[C_A2 if v > 0 else C_A1 if v < 0 else GREY for v in vals])
    for v, n in zip(vals, cnt):
        axB.text(v, n, f"{n}", ha="center", va="bottom", fontsize=8,
                 weight="bold")
    axB.set_xticks(vals)
    axB.set_xlabel(f"Score  (+ favours {a.label2}, − favours {a.label1})")
    axB.set_ylabel("Compounds")
    axB.set_ylim(0, max(cnt) * 1.18)
    axB.set_title("B   Discriminating H-bond score", loc="left",
                  fontsize=9, weight="bold")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print("\n".join(log))
    with open(os.path.splitext(a.out)[0] + "_values.txt", "w") as f:
        f.write("\n".join(log) + "\n")
    print(f"\n[out] {a.out}")
    print(f"      {os.path.splitext(a.out)[0]}_ranking.csv")


if __name__ == "__main__":
    sys.exit(main())

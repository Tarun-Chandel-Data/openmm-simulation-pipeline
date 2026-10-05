#!/usr/bin/env python3
"""
The labels every analysis in this project has been missing.

The compounds docked in the ChEMBL panel were pulled from a database that
also holds what was measured for them. The structures on disk do not carry
those numbers, so nothing done with them so far could be checked against an
experiment. This fetches the numbers and puts them beside whatever metric is
being proposed.

Three steps, each its own flag so a failed network call never costs the ones
already done:

  --resolve        ask ChEMBL which targets its CK2 entries are, and print
                   them. Target identifiers change; this is how to see what
                   is current rather than trusting a number written in a
                   script
  --fetch          download the activities of the compounds named in a csv,
                   for the given targets, and write them one row per
                   measurement. Resumable: already-downloaded compounds are
                   skipped unless --refresh
  --join           pivot to one row per compound with the two isoforms side
                   by side, and, where a metric file is given, report how
                   well that metric orders the compounds against what was
                   measured

The selectivity column is pIC50(alpha') - pIC50(alpha), the same sign
convention as the MM-GBSA work, so a positive number is alpha'-preferring in
both.

    python chembl_activity.py --resolve
    python chembl_activity.py --fetch --ids dock_paired/poses.csv \\
        --id-column cpd_id --alpha CHEMBL3629 --alpha-prime CHEMBL2850 \\
        --out chembl_act
    python chembl_activity.py --join --out chembl_act \\
        --metric hinge.csv --metric-column either --metric-id compound
"""
import argparse, csv, json, math, os, sys, time, urllib.error, urllib.parse
import urllib.request

API = "https://www.ebi.ac.uk/chembl/api/data"

# Only the activity types that convert to a free energy sensibly. Percent
# inhibition at one concentration is not a potency and is not mixed in.
POTENCY = ("IC50", "Ki", "Kd", "EC50")


def get(path, params, tries=4):
    """One GET, retried on the transient failures a public API gives."""
    url = f"{API}/{path}?" + urllib.parse.urlencode(params)
    wait = 2.0
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={
                "Accept": "application/json",
                "User-Agent": "ck2-selectivity-analysis"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, json.JSONDecodeError) as e:
            if k == tries - 1:
                raise SystemExit(
                    f"[stop] {url}\n       {e}\n"
                    "       no network to ChEMBL from here. Run this on a "
                    "machine that has it, or download the activity table "
                    "from the web interface and pass it to --join.")
            time.sleep(wait)
            wait *= 2


def pages(path, params, key, cap=100000):
    """Walk a paginated endpoint, yielding its records."""
    params = dict(params)
    params["limit"] = 1000
    off, seen = 0, 0
    while True:
        params["offset"] = off
        d = get(path, params)
        rows = d.get(key, [])
        if not rows:
            return
        for r in rows:
            yield r
            seen += 1
            if seen >= cap:
                return
        page = d.get("page_meta") or {}
        if not page.get("next"):
            return
        off += len(rows)


def resolve():
    """What ChEMBL currently calls the two kinases, printed to be read."""
    log = ["=== ChEMBL targets matching casein kinase II ===",
           "    the subunit matters: alpha (CSNK2A1) and alpha' (CSNK2A2) are",
           "    separate entries, and the holoenzyme is a third. Pick the two",
           "    single-protein entries, not the complex",
           ""]
    log.append(f"  {'target':16s}{'organism':22s}{'type':24s}  name")
    for q in ("casein kinase II", "CSNK2A1", "CSNK2A2"):
        hits = list(pages("target/search", {"q": q, "format": "json"},
                          "targets", cap=40))
        if not hits:
            continue
        log.append(f"  -- query: {q}")
        for t in hits:
            org = (t.get("organism") or "")[:21]
            log.append(f"  {t.get('target_chembl_id',''):16s}{org:22s}"
                       f"{(t.get('target_type') or '')[:23]:24s}  "
                       f"{(t.get('pref_name') or '')[:60]}")
    print("\n".join(log))
    return 0


def read_ids(path, column):
    """The compound identifiers, in the order the file gives them."""
    if not os.path.exists(path):
        raise SystemExit(f"[stop] no such file: {path}")
    ids = []
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        if column not in (rd.fieldnames or []):
            raise SystemExit(
                f"[stop] {path} has no column '{column}'. It has: "
                + ", ".join(rd.fieldnames or []))
        for row in rd:
            v = (row[column] or "").strip()
            # the pose files name compounds as the id alone or with a suffix
            v = v.split("__")[0]
            if v.upper().startswith("CHEMBL") and v not in ids:
                ids.append(v.upper())
    if not ids:
        raise SystemExit(f"[stop] no ChEMBL identifiers in {path}:{column}")
    return ids


def fetch(a):
    """Every potency measured for these compounds on these targets."""
    ids = read_ids(a.ids, a.id_column)
    targets = {}
    if a.alpha:
        targets[a.alpha.upper()] = "alpha"
    if a.alpha_prime:
        targets[a.alpha_prime.upper()] = "alpha_prime"
    for extra in a.target or []:
        targets.setdefault(extra.upper(), extra.upper())
    if not targets:
        raise SystemExit("[stop] give --alpha and/or --alpha-prime")

    out = a.out + "_activities.csv"
    have = set()
    rows = []
    if os.path.exists(out) and not a.refresh:
        with open(out, newline="") as f:
            for r in csv.DictReader(f):
                rows.append(r)
                have.add((r["compound"], r["target"]))
        print(f"[in] {len(rows)} measurements already downloaded")

    fields = ["compound", "target", "isoform", "standard_type",
              "standard_relation", "standard_value", "standard_units",
              "pchembl_value", "assay_chembl_id", "assay_type",
              "document_chembl_id", "data_validity_comment"]
    todo = [(c, t) for c in ids for t in targets
            if (c, t) not in have]
    print(f"[run] {len(todo)} compound-target pairs to ask about")
    for n, (cpd, tgt) in enumerate(todo, 1):
        got = 0
        for r in pages("activity", {
                "molecule_chembl_id": cpd, "target_chembl_id": tgt,
                "format": "json"}, "activities", cap=2000):
            if r.get("standard_type") not in POTENCY:
                continue
            rows.append({
                "compound": cpd, "target": tgt, "isoform": targets[tgt],
                "standard_type": r.get("standard_type") or "",
                "standard_relation": r.get("standard_relation") or "",
                "standard_value": r.get("standard_value") or "",
                "standard_units": r.get("standard_units") or "",
                "pchembl_value": r.get("pchembl_value") or "",
                "assay_chembl_id": r.get("assay_chembl_id") or "",
                "assay_type": r.get("assay_type") or "",
                "document_chembl_id": r.get("document_chembl_id") or "",
                "data_validity_comment":
                    r.get("data_validity_comment") or ""})
            got += 1
        if n % 10 == 0 or n == len(todo):
            print(f"      {n}/{len(todo)}  {cpd} on {tgt}: {got}")
        # written as we go, so a dropped connection keeps what it had
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"[out] {out}  ({len(rows)} measurements)")
    return 0


def to_p(row):
    """A pChEMBL value, taken as given or made from the concentration."""
    v = (row.get("pchembl_value") or "").strip()
    if v:
        try:
            return float(v)
        except ValueError:
            pass
    try:
        x = float(row["standard_value"])
    except (KeyError, TypeError, ValueError):
        return None
    if x <= 0 or (row.get("standard_units") or "").strip() != "nM":
        return None
    return -math.log10(x * 1e-9)


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def sd(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def spearman(x, y):
    """Rank correlation, which is all an ordering claim needs."""
    pair = [(a, b) for a, b in zip(x, y)
            if a is not None and b is not None
            and not math.isnan(a) and not math.isnan(b)]
    if len(pair) < 3:
        return float("nan"), 0
    n = len(pair)

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx = ranks([p[0] for p in pair])
    ry = ranks([p[1] for p in pair])
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return (num / den if den else float("nan")), n


def join(a):
    """One row per compound: what was measured, beside what was predicted."""
    src = a.activities or (a.out + "_activities.csv")
    if not os.path.exists(src):
        raise SystemExit(f"[stop] no activity file: {src} (run --fetch)")
    per = {}
    dropped = 0
    with open(src, newline="") as f:
        for r in csv.DictReader(f):
            if a.strict and (r.get("data_validity_comment") or "").strip():
                dropped += 1
                continue
            if a.exact and (r.get("standard_relation") or "=").strip() \
                    not in ("=", "'='", ""):
                dropped += 1
                continue
            p = to_p(r)
            if p is None:
                dropped += 1
                continue
            per.setdefault(r["compound"], {}).setdefault(
                r.get("isoform") or r["target"], []).append(p)

    metric = {}
    if a.metric:
        with open(a.metric, newline="") as f:
            rd = csv.DictReader(f)
            for col in (a.metric_id, a.metric_column):
                if col not in (rd.fieldnames or []):
                    raise SystemExit(
                        f"[stop] {a.metric} has no column '{col}'. It has: "
                        + ", ".join(rd.fieldnames or []))
            for r in rd:
                try:
                    metric.setdefault(
                        (r[a.metric_id] or "").strip().upper(), []
                    ).append(float(r[a.metric_column]))
                except (TypeError, ValueError):
                    continue

    log = [f"[in] {src}",
           f"     potency types kept: {', '.join(POTENCY)}"]
    if dropped:
        log.append(f"     {dropped} measurements left out "
                   "(no usable value, flagged, or a bound not an equality)")
    log.append("")
    log.append("=== measured potency, one row per compound ===")
    log.append("    pIC50-scale, averaged over the assays ChEMBL holds; sd is"
               " the spread between those assays, which is the floor any"
               " predicted difference has to clear")
    log.append(f"  {'compound':16s}{'alpha':>18s}{'alpha prime':>18s}"
               f"{'selectivity':>13s}" + (f"{'metric':>11s}" if metric else ""))
    sel_head = "a'-a"
    log.append(f"  {'':16s}{'mean +- sd  n':>18s}{'mean +- sd  n':>18s}"
               f"{sel_head:>13s}")

    rows, mx, my = [], [], []
    for cpd in sorted(per):
        d = per[cpd]
        pa, pp = d.get("alpha", []), d.get("alpha_prime", [])
        sel = (mean(pp) - mean(pa)) if pa and pp else None
        met = mean(metric[cpd]) if cpd in metric and metric[cpd] else None
        rows.append({
            "compound": cpd,
            "p_alpha": f"{mean(pa):.2f}" if pa else "",
            "p_alpha_sd": f"{sd(pa):.2f}" if pa else "",
            "n_alpha": len(pa),
            "p_alpha_prime": f"{mean(pp):.2f}" if pp else "",
            "p_alpha_prime_sd": f"{sd(pp):.2f}" if pp else "",
            "n_alpha_prime": len(pp),
            "selectivity": f"{sel:.2f}" if sel is not None else "",
            "metric": f"{met:.3f}" if met is not None else ""})
        line = f"  {cpd:16s}"
        line += (f"{mean(pa):8.2f} +-{sd(pa):<5.2f}{len(pa):>4d}" if pa
                 else f"{'-':>18s}")
        line += (f"{mean(pp):8.2f} +-{sd(pp):<5.2f}{len(pp):>4d}" if pp
                 else f"{'-':>18s}")
        line += f"{sel:13.2f}" if sel is not None else f"{'-':>13s}"
        if metric:
            line += f"{met:11.3f}" if met is not None else f"{'-':>11s}"
        log.append(line)
        if met is not None:
            if pa:
                mx.append(met); my.append(mean(pa))

    both = [r for r in rows if r["p_alpha"] and r["p_alpha_prime"]]
    log.append("")
    log.append(f"  {len(rows)} compounds with a potency, "
               f"{len(both)} measured on both isoforms")
    if not both:
        log.append("    nothing here can test a selectivity claim: the panel"
                   " was measured on one isoform only. The metric can still"
                   " be checked against plain potency, above")

    if metric:
        r, n = spearman(mx, my)
        log.append("")
        log.append("=== does the metric order the compounds? ===")
        log.append(f"    Spearman r = {r:.3f} over {n} compounds, metric "
                   f"'{a.metric_column}' against potency on alpha")
        log.append("    this is the test four analyses in this project could"
                   " not do for want of a measured number. A metric that does"
                   " not order a 100-compound panel does not order eleven"
                   " analogues either")
        if n >= 3 and not math.isnan(r):
            log.append("    reading: "
                       + ("orders them" if abs(r) >= 0.5 else
                          "weakly related" if abs(r) >= 0.3 else
                          "unrelated to what was measured"))

    text = "\n".join(log)
    print(text)
    dest = a.out + "_joined"
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["compound"])
        w.writeheader()
        w.writerows(rows)
    with open(dest + ".txt", "w") as f:
        f.write(text + "\n")
    print(f"\n[out] {dest}.csv, {dest}.txt")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resolve", action="store_true",
                   help="print the ChEMBL targets for CK2 and stop")
    p.add_argument("--fetch", action="store_true",
                   help="download activities for the compounds in --ids")
    p.add_argument("--join", action="store_true",
                   help="pivot to one row per compound, and test a metric")
    p.add_argument("--ids", help="csv holding the compound identifiers")
    p.add_argument("--id-column", default="cpd_id")
    p.add_argument("--alpha", help="target id for CK2 alpha")
    p.add_argument("--alpha-prime", help="target id for CK2 alpha prime")
    p.add_argument("--target", action="append",
                   help="any further target id, repeatable")
    p.add_argument("--activities",
                   help="an activity csv to read instead of <out>_activities")
    p.add_argument("--metric", help="csv holding a predicted quantity")
    p.add_argument("--metric-column", default="metric")
    p.add_argument("--metric-id", default="compound")
    p.add_argument("--strict", action="store_true",
                   help="drop measurements ChEMBL has flagged")
    p.add_argument("--exact", action="store_true",
                   help="drop > and < measurements, keeping equalities")
    p.add_argument("--refresh", action="store_true",
                   help="re-download rather than resume")
    p.add_argument("--out", default="chembl_act")
    a = p.parse_args()

    if not (a.resolve or a.fetch or a.join):
        p.error("give one of --resolve, --fetch, --join")
    if a.resolve:
        return resolve()
    if a.fetch:
        if not a.ids:
            p.error("--fetch needs --ids")
        rc = fetch(a)
        if not a.join:
            return rc
    return join(a)


if __name__ == "__main__":
    sys.exit(main())

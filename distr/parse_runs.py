#!/usr/bin/env python3
# python3 parse_runs.py <run_dir_or_file> [--csv] [--debug] [--failed]
"""
Parse SLURM job output files for experiment timings.

Usage:
    python parse_runs.py <run_dir_or_file>           # human-readable table
    python parse_runs.py <run_dir_or_file> --csv     # CSV to stdout
    python parse_runs.py <run_dir_or_file> --debug   # include per-rank profile
    python parse_runs.py <run_dir_or_file> --failed  # include failed experiments
    python parse_runs.py <run_dir_or_file> --debug --csv   # per-rank CSV

Output file format expected (normal mode):
    Experiment: 100x100x100x100 128 Ax ranks wbp
    Mean All Gather: 0.000549 | Mean Computation: 0.001703

Per-rank/debug mode format (when per_rank_timings=true in main.cpp):
    Rank 0: Local All Gather Time: 0.000549123 | Local Computation Time: 0.001703456

SLATE baseline format (op "slate", see baseline.cpp; its own "Experiment:" line is ignored):
    Experiment: 400x400x400x400 512 slate ranks slate
    Mean Redistribution (comm only): ... | Mean A~x (comm+comp): ... | Mean A~^Tx (comm+comp): ...
    Max Redistribution (comm only): ... | Max A~x (comm+comp): ... | Max A~^Tx (comm+comp): ...

Golub-Kahan bidiagonalization format (op "bidiag" or "bidiag_reorth"):
    Experiment: 200x200x200x200 128 bidiag ranks bcp
    Bidiag Steps: 50 | Sigma1: 1.8e+03 | Alpha Sum: ... | Beta Sum: ...
    Bidiag Max Total: ... | Max Kernel: ... | Max Other: ... | Max Reductions: ... | Max Vector Ops: ...
    Bidiag Kernel Fraction: Mean 0.91 | Min 0.88 | Max 0.95
    Bidiag Ax (max): All Gather: ... | Computation: ... | Reduce Scatter: ... | Total: ...
    Bidiag ATx (max): All Gather: ... | Computation: ... | Reduce Scatter: ... | Total: ...
    Bidiag Mean Breakdown: Total: ... | Ax All Gather: ... | ... | Reductions: ... | Vector Ops: ... | Barrier: ...
    (Sync / Max Barrier / Barrier appear in runs with the optional GKB barriers;
    older runs leave those fields empty)
"""

import re
import sys
import csv
import argparse
from pathlib import Path

# ── Patterns ────────────────────────────────────────────────────────────────

HEADER_PAT = re.compile(
    r"Experiment:\s+(\d+)x(\d+)x(\d+)x(\d+)\s+(\d+)\s+(\S+)\s+ranks\s+(\S+)"
)
MEAN_PAT = {
    "all_gather":     re.compile(r"(?:Mean|Max) All Gather:\s*([\d.]+)"),
    "computation":    re.compile(r"(?:Mean|Max) Computation:\s*([\d.]+)"),
    "reduce_scatter": re.compile(r"(?:Mean|Max) Reduce Scatter:\s*([\d.]+)"),
}
RANK_LINE_PAT = re.compile(r"^Rank\s+(\d+):\s+(.+)")
NUM = r"([\d.eE+\-]+)"
# Bidiag lines: each maps field name -> regex; all fields default to None
BIDIAG_PAT = {
    "steps":         re.compile(r"Bidiag Steps:\s*(\d+)"),
    "sigma1":        re.compile(r"Bidiag Steps:.*Sigma1:\s*" + NUM),
    "sync":          re.compile(r"Bidiag Steps:.*Sync:\s*(\d+)"),
    "bd_total":      re.compile(r"Bidiag Max Total:\s*" + NUM),
    "bd_kernel":     re.compile(r"Bidiag Max Total:.*Max Kernel:\s*" + NUM),
    "bd_other":      re.compile(r"Bidiag Max Total:.*Max Other:\s*" + NUM),
    "bd_reductions": re.compile(r"Bidiag Max Total:.*Max Reductions:\s*" + NUM),
    "bd_vector_ops": re.compile(r"Bidiag Max Total:.*Max Vector Ops:\s*" + NUM),
    "bd_barrier":    re.compile(r"Bidiag Max Total:.*Max Barrier:\s*" + NUM),
    "kernel_frac_mean": re.compile(r"Bidiag Kernel Fraction:\s*Mean\s*" + NUM),
    "kernel_frac_min":  re.compile(r"Bidiag Kernel Fraction:.*Min\s*" + NUM),
    "kernel_frac_max":  re.compile(r"Bidiag Kernel Fraction:.*Max\s*" + NUM),
}
for _k in ("Ax", "ATx"):
    _pre = rf"Bidiag {_k} \(max\):.*"
    _name = _k.lower()
    BIDIAG_PAT[f"bd_{_name}_all_gather"]     = re.compile(_pre + r"All Gather:\s*" + NUM)
    BIDIAG_PAT[f"bd_{_name}_computation"]    = re.compile(_pre + r"Computation:\s*" + NUM)
    BIDIAG_PAT[f"bd_{_name}_reduce_scatter"] = re.compile(_pre + r"Reduce Scatter:\s*" + NUM)
    BIDIAG_PAT[f"bd_{_name}_total"]          = re.compile(_pre + r"Total:\s*" + NUM)
# Per-rank means (these add up to mean_total; the max fields above do not)
_MEAN_PRE = r"Bidiag Mean Breakdown:.*"
for _field, _label in (("mean_total", "Total"),
                       ("mean_ax_all_gather", "Ax All Gather"), ("mean_ax_computation", "Ax Computation"),
                       ("mean_ax_reduce_scatter", "Ax Reduce Scatter"),
                       ("mean_atx_all_gather", "ATx All Gather"), ("mean_atx_computation", "ATx Computation"),
                       ("mean_atx_reduce_scatter", "ATx Reduce Scatter"),
                       ("mean_reductions", "Reductions"), ("mean_vector_ops", "Vector Ops"),
                       ("mean_barrier", "Barrier")):
    BIDIAG_PAT[_field] = re.compile(_MEAN_PRE + r"\b" + _label + r":\s*" + NUM)
BIDIAG_FIELDS = list(BIDIAG_PAT.keys())

# SLATE baseline (op "slate"): one-time repermutation (comm only) and
# slate::gemm Ax / ATx (comm + comp together); max and mean over ranks
SLATE_PAT = {}
for _stat in ("max", "mean"):
    _pre = rf"{_stat.capitalize()} Redistribution \(comm only\):.*"
    SLATE_PAT[f"slate_redistribution_{_stat}"] = re.compile(
        rf"{_stat.capitalize()} Redistribution \(comm only\):\s*" + NUM)
    SLATE_PAT[f"slate_ax_{_stat}"] = re.compile(_pre + r"A~x \(comm\+comp\):\s*" + NUM)
    SLATE_PAT[f"slate_atx_{_stat}"] = re.compile(_pre + r"A~\^Tx \(comm\+comp\):\s*" + NUM)
SLATE_FIELDS = list(SLATE_PAT.keys())
LOCAL_PAT = {
    "all_gather":     re.compile(r"Local All Gather Time:\s*([\d.eE+\-]+)"),
    "computation":    re.compile(r"Local Computation Time:\s*([\d.eE+\-]+)"),
    "reduce_scatter": re.compile(r"Local Reduce Scatter Time:\s*([\d.eE+\-]+)"),
}
FAILED_PAT = re.compile(r"srun:.*Force Terminated|srun:.*error|slurmstepd:.*error|Assertion .* failed", re.IGNORECASE)


def is_bidiag(r):
    return r["op"].startswith("bidiag")


def is_slate(r):
    return r["op"] == "slate"

# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_file(path):
    """
    Returns a list of experiment records. Each record is a dict:
      m1, n1, m2, n2 : int
      ranks           : int
      op              : str   ("Ax" | "ATx" | "bidiag" | "bidiag_reorth")
      alg             : str   ("wbp" | "rrp" | "bcp")
      all_gather      : float | None
      computation     : float | None
      reduce_scatter  : float | None
      failed          : bool
      per_rank        : list of {rank, all_gather, computation, reduce_scatter}
      <BIDIAG_FIELDS> : float | int | None (bidiag ops only)
    """
    lines = Path(path).read_text().splitlines()
    records = []
    current = None

    for line in lines:
        hm = HEADER_PAT.search(line)
        if hm:
            current = {
                "m1": int(hm.group(1)), "n1": int(hm.group(2)),
                "m2": int(hm.group(3)), "n2": int(hm.group(4)),
                "ranks": int(hm.group(5)), "op": hm.group(6), "alg": hm.group(7),
                "all_gather": None, "computation": None, "reduce_scatter": None,
                "failed": False, "per_rank": [],
                **{k: None for k in BIDIAG_FIELDS},
                **{k: None for k in SLATE_FIELDS},
            }
            records.append(current)
            continue

        if current is None:
            continue

        if FAILED_PAT.search(line):
            current["failed"] = True
            continue

        if "Redistribution (comm only)" in line:
            for key, pat in SLATE_PAT.items():
                m = pat.search(line)
                if m:
                    current[key] = float(m.group(1))
            continue

        if line.startswith("Bidiag "):
            for key, pat in BIDIAG_PAT.items():
                m = pat.search(line)
                if m:
                    current[key] = int(m.group(1)) if key in ("steps", "sync") else float(m.group(1))
            continue

        # Mean aggregated line
        if any(pat.search(line) for pat in MEAN_PAT.values()):
            for key, pat in MEAN_PAT.items():
                m = pat.search(line)
                if m:
                    current[key] = float(m.group(1))
            continue

        # Per-rank line (debug mode output)
        rm = RANK_LINE_PAT.match(line)
        if rm:
            rank_id = int(rm.group(1))
            rest = rm.group(2)
            entry = {"rank": rank_id, "all_gather": None, "computation": None, "reduce_scatter": None}
            for key, pat in LOCAL_PAT.items():
                m = pat.search(rest)
                if m:
                    entry[key] = float(m.group(1))
            current["per_rank"].append(entry)

    # A run that printed its header but no timings died (OOM, time limit, ...)
    for r in records:
        if is_bidiag(r):
            timings = [r["bd_total"]]
        elif is_slate(r):
            timings = [r["slate_redistribution_max"], r["slate_redistribution_mean"]]
        else:
            timings = [r["all_gather"], r["computation"], r["reduce_scatter"]]
        if all(t is None for t in timings):
            r["failed"] = True
    return records


def load_run(path):
    """Return list of (Path, records) for every .out file under path."""
    p = Path(path)
    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = sorted(p.glob("*.out"))
    else:
        sys.exit(f"Error: {path} is not a file or directory")
    if not files:
        sys.exit(f"No .out files found in {path}")
    return [(f, parse_file(f)) for f in files]

# ── Formatting ───────────────────────────────────────────────────────────────

def fmt(t, width=10):
    """Human-readable time with fixed width."""
    if t is None:
        return "—".rjust(width)
    if t >= 1:
        s = f"{t:.4f}s"
    elif t >= 1e-3:
        s = f"{t*1e3:.3f}ms"
    else:
        s = f"{t*1e6:.1f}µs"
    return s.rjust(width)


def print_table(data, show_failed, show_debug):
    for path, records in data:
        print(f"\n{'='*60}")
        print(f"  {path}")
        print(f"{'='*60}")
        if not records:
            print("  (no experiments found)")
            continue

        # Header
        print(f"  {'Size':<22} {'Op':<5} {'Alg':<5} {'Ranks':>6}  "
              f"{'AllGather':>10}  {'Compute':>10}  {'RedScatter':>10}  {'Total':>10}  Status")
        print(f"  {'-'*105}")

        for r in records:
            if is_bidiag(r) or is_slate(r) or (r["failed"] and not show_failed):
                continue
            size = f"{r['m1']}x{r['n1']}x{r['m2']}x{r['n2']}"
            if r["failed"]:
                print(f"  {size:<22} {r['op']:<5} {r['alg']:<5} {r['ranks']:>6}  "
                      f"{'':>10}  {'':>10}  {'':>10}  {'':>10}  FAILED")
                continue
            vals = [v for v in (r["all_gather"], r["computation"], r["reduce_scatter"]) if v is not None]
            total = sum(vals) if vals else None
            print(f"  {size:<22} {r['op']:<5} {r['alg']:<5} {r['ranks']:>6}  "
                  f"{fmt(r['all_gather'])}  {fmt(r['computation'])}  "
                  f"{fmt(r['reduce_scatter'])}  {fmt(total)}")

            if show_debug and r["per_rank"]:
                print(f"    {'Rank':>6}  {'AllGather':>12}  {'Compute':>12}  {'RedScatter':>12}")
                for pr in sorted(r["per_rank"], key=lambda x: x["rank"]):
                    print(f"    {pr['rank']:>6}  {fmt(pr['all_gather'], 12)}  "
                          f"{fmt(pr['computation'], 12)}  {fmt(pr['reduce_scatter'], 12)}")

        slate = [r for r in records if is_slate(r) and (show_failed or not r["failed"])]
        if slate:
            print(f"\n  SLATE baseline (max over ranks): one-time repermutation (comm only), gemm Ax / ATx (comm+comp)")
            print(f"  {'Size':<22} {'Ranks':>6}  {'Repermute':>10}  {'Ax':>10}  {'ATx':>10}  Status")
            print(f"  {'-'*75}")
            for r in slate:
                size = f"{r['m1']}x{r['n1']}x{r['m2']}x{r['n2']}"
                status = "FAILED" if r["failed"] else ""
                print(f"  {size:<22} {r['ranks']:>6}  {fmt(r['slate_redistribution_max'])}  "
                      f"{fmt(r['slate_ax_max'])}  {fmt(r['slate_atx_max'])}  {status}")

        bidiag = [r for r in records if is_bidiag(r) and (show_failed or not r["failed"])]
        if bidiag:
            print(f"\n  Golub-Kahan bidiagonalization (max over ranks; Kernel% = per-rank mean)")
            print(f"  {'Size':<22} {'Op':<14} {'Alg':<5} {'Ranks':>6} {'Steps':>5}  "
                  f"{'Total':>10}  {'Kernel':>10}  {'Other':>10}  {'Kernel%':>7}  "
                  f"{'Ax':>10}  {'ATx':>10}  Status")
            print(f"  {'-'*125}")
            for r in bidiag:
                size = f"{r['m1']}x{r['n1']}x{r['m2']}x{r['n2']}"
                if r["failed"]:
                    print(f"  {size:<22} {r['op']:<14} {r['alg']:<5} {r['ranks']:>6} {'':>5}  {'':>10}  "
                          f"{'':>10}  {'':>10}  {'':>7}  {'':>10}  {'':>10}  FAILED")
                    continue
                frac = r["kernel_frac_mean"]
                frac_s = "—" if frac is None else f"{100 * frac:.1f}%"
                print(f"  {size:<22} {r['op']:<14} {r['alg']:<5} {r['ranks']:>6} {r['steps'] or '':>5}  "
                      f"{fmt(r['bd_total'])}  {fmt(r['bd_kernel'])}  {fmt(r['bd_other'])}  {frac_s:>7}  "
                      f"{fmt(r['bd_ax_total'])}  {fmt(r['bd_atx_total'])}")


# ── CSV output ───────────────────────────────────────────────────────────────

# Bidiag columns are appended (empty for Ax/ATx rows); for bidiag rows
# "total" is the whole bidiagonalization time and the Ax/ATx phase columns
# are empty (they live in bd_ax_* / bd_atx_*).
MEAN_CSV_FIELDS = ["file", "m1", "n1", "m2", "n2", "ranks", "op", "alg",
                   "all_gather", "computation", "reduce_scatter", "total", "failed"] + BIDIAG_FIELDS + SLATE_FIELDS
RANK_CSV_FIELDS = ["file", "m1", "n1", "m2", "n2", "ranks", "op", "alg",
                   "rank", "all_gather", "computation", "reduce_scatter"]


def write_csv(data, show_failed, show_debug, out=sys.stdout):
    w = csv.DictWriter(out, fieldnames=MEAN_CSV_FIELDS, lineterminator="\n")
    w.writeheader()
    for path, records in data:
        fname = Path(path).name
        for r in records:
            if r["failed"] and not show_failed:
                continue
            vals = [v for v in (r["all_gather"], r["computation"], r["reduce_scatter"]) if v is not None]
            row = {
                "file": fname,
                "m1": r["m1"], "n1": r["n1"], "m2": r["m2"], "n2": r["n2"],
                "ranks": r["ranks"], "op": r["op"], "alg": r["alg"],
                "all_gather":     "" if r["all_gather"]     is None else r["all_gather"],
                "computation":    "" if r["computation"]    is None else r["computation"],
                "reduce_scatter": "" if r["reduce_scatter"] is None else r["reduce_scatter"],
                "total":          "" if not vals else sum(vals),
                "failed": int(r["failed"]),
                **{k: "" if r[k] is None else r[k] for k in BIDIAG_FIELDS},
                **{k: "" if r[k] is None else r[k] for k in SLATE_FIELDS},
            }
            if is_bidiag(r):
                row["total"] = "" if r["bd_total"] is None else r["bd_total"]
            elif is_slate(r) and r["slate_ax_max"] is not None and r["slate_atx_max"] is not None:
                row["total"] = r["slate_ax_max"] + r["slate_atx_max"]  # one Ax + one ATx
            w.writerow(row)

    if show_debug:
        out.write("\n# per_rank\n")
        rw = csv.DictWriter(out, fieldnames=RANK_CSV_FIELDS, lineterminator="\n")
        rw.writeheader()
        for path, records in data:
            fname = Path(path).name
            for r in records:
                if r["failed"]:
                    continue
                for pr in sorted(r["per_rank"], key=lambda x: x["rank"]):
                    rw.writerow({
                        "file": fname,
                        "m1": r["m1"], "n1": r["n1"], "m2": r["m2"], "n2": r["n2"],
                        "ranks": r["ranks"], "op": r["op"], "alg": r["alg"],
                        "rank": pr["rank"],
                        "all_gather":     "" if pr["all_gather"]     is None else pr["all_gather"],
                        "computation":    "" if pr["computation"]    is None else pr["computation"],
                        "reduce_scatter": "" if pr["reduce_scatter"] is None else pr["reduce_scatter"],
                    })

# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(data, show_failed):
    total = failed = 0
    for _, records in data:
        for r in records:
            total += 1
            if r["failed"]:
                failed += 1
    ok = total - failed
    print(f"\nSummary: {total} experiments — {ok} ok, {failed} failed", end="")
    if failed and not show_failed:
        print(" (use --failed to show failed rows)", end="")
    print()

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Parse SLURM .out files from a run directory for experiment timings."
    )
    ap.add_argument("path", help="Run directory (e.g. runs/run1) or a single .out file")
    ap.add_argument("--csv",    action="store_true", help="Output as CSV")
    ap.add_argument("--debug",  action="store_true", help="Include per-rank profile timings")
    ap.add_argument("--failed", action="store_true", help="Include failed experiments in output")
    args = ap.parse_args()

    data = load_run(args.path)

    if args.csv:
        write_csv(data, args.failed, args.debug)
    else:
        print_table(data, args.failed, args.debug)
        print_summary(data, args.failed)


if __name__ == "__main__":
    main()

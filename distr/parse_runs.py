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
    "all_gather":     re.compile(r"Mean All Gather:\s*([\d.]+)"),
    "computation":    re.compile(r"Mean Computation:\s*([\d.]+)"),
    "reduce_scatter": re.compile(r"Mean Reduce Scatter:\s*([\d.]+)"),
}
RANK_LINE_PAT = re.compile(r"^Rank\s+(\d+):\s+(.+)")
LOCAL_PAT = {
    "all_gather":     re.compile(r"Local All Gather Time:\s*([\d.eE+\-]+)"),
    "computation":    re.compile(r"Local Computation Time:\s*([\d.eE+\-]+)"),
    "reduce_scatter": re.compile(r"Local Reduce Scatter Time:\s*([\d.eE+\-]+)"),
}
FAILED_PAT = re.compile(r"srun:.*Force Terminated|srun:.*error", re.IGNORECASE)

# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_file(path):
    """
    Returns a list of experiment records. Each record is a dict:
      m1, n1, m2, n2 : int
      ranks           : int
      op              : str   ("Ax" | "ATx")
      alg             : str   ("wbp" | "rrp" | "bcp")
      all_gather      : float | None
      computation     : float | None
      reduce_scatter  : float | None
      failed          : bool
      per_rank        : list of {rank, all_gather, computation, reduce_scatter}
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
            }
            records.append(current)
            continue

        if current is None:
            continue

        if FAILED_PAT.search(line):
            current["failed"] = True
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
            if r["failed"] and not show_failed:
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


# ── CSV output ───────────────────────────────────────────────────────────────

MEAN_CSV_FIELDS = ["file", "m1", "n1", "m2", "n2", "ranks", "op", "alg",
                   "all_gather", "computation", "reduce_scatter", "total", "failed"]
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
            w.writerow({
                "file": fname,
                "m1": r["m1"], "n1": r["n1"], "m2": r["m2"], "n2": r["n2"],
                "ranks": r["ranks"], "op": r["op"], "alg": r["alg"],
                "all_gather":     "" if r["all_gather"]     is None else r["all_gather"],
                "computation":    "" if r["computation"]    is None else r["computation"],
                "reduce_scatter": "" if r["reduce_scatter"] is None else r["reduce_scatter"],
                "total":          "" if not vals else sum(vals),
                "failed": int(r["failed"]),
            })

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

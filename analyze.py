"""
Analyze experiment results and produce comparison tables.

Usage:
  python analyze.py                    # reads all CSVs in results/raw/
  python analyze.py --dir results/raw  # explicit directory
  python analyze.py --out results/summary.txt  # custom output path

Produces:
  1. Accuracy table  (EM and F1 per cell)
  2. Accuracy pivot  (architectures as rows, domain×difficulty as columns)
  3. Token efficiency (avg tokens per question)
  4. Speed           (avg elapsed seconds per question)
  5. Step counts     (avg graph steps per question)
  6. Error rates     (% rows that errored)

All tables are printed to stdout and written to --out (default: results/summary.txt).
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_all(raw_dir: Path) -> list[dict]:
    rows = []
    for csv_path in sorted(raw_dir.glob("*.csv")):
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["_file"] = csv_path.name
                rows.append(row)
    return rows


def _float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

ARCH_ORDER = ["level1", "level2a", "level2b", "level3"]
DOMAIN_ORDER = ["math", "hotpotqa"]
DIFF_ORDER = ["easy", "hard"]


def aggregate(rows: list[dict]) -> dict:
    """
    Group rows by (architecture, domain, difficulty, top_k) and compute per-cell stats.
    top_k defaults to "3" for old CSVs that predate the column.
    """
    cells = defaultdict(lambda: {
        "em": [], "f1": [], "input_tok": [], "output_tok": [],
        "steps": [], "elapsed": [], "errors": 0, "total": 0,
    })

    for row in rows:
        arch   = row.get("architecture", "")
        domain = row.get("domain", "")
        diff   = row.get("difficulty", "")
        top_k  = row.get("top_k", "3") or "3"
        key    = (arch, domain, diff, top_k)

        c = cells[key]
        c["total"] += 1

        if row.get("error", ""):
            c["errors"] += 1
            # Still record elapsed for errored rows (timeout cost is real)
            c["elapsed"].append(_float(row.get("elapsed_seconds", 0)))
            continue

        c["em"].append(_int(row.get("exact_match", 0)))
        f1_val = row.get("f1", "")
        if f1_val != "":
            c["f1"].append(_float(f1_val))
        c["input_tok"].append(_int(row.get("total_input_tokens", 0)))
        c["output_tok"].append(_int(row.get("total_output_tokens", 0)))
        c["steps"].append(_int(row.get("total_steps", 0)))
        c["elapsed"].append(_float(row.get("elapsed_seconds", 0)))

    # Compute summaries
    summary = {}
    for key, c in cells.items():
        n_ok = len(c["em"])
        n_tot = c["total"]
        summary[key] = {
            "n_ok":       n_ok,
            "n_total":    n_tot,
            "error_pct":  100 * c["errors"] / n_tot if n_tot else 0,
            "em":         sum(c["em"]) / n_ok if n_ok else None,
            "f1":         sum(c["f1"]) / len(c["f1"]) if c["f1"] else None,
            "avg_in_tok": sum(c["input_tok"])  / n_ok if n_ok else None,
            "avg_out_tok":sum(c["output_tok"]) / n_ok if n_ok else None,
            "avg_tok":    (sum(c["input_tok"]) + sum(c["output_tok"])) / n_ok if n_ok else None,
            "avg_steps":  sum(c["steps"])   / n_ok if n_ok else None,
            "avg_elapsed":sum(c["elapsed"]) / len(c["elapsed"]) if c["elapsed"] else None,
        }

    return summary


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(val, decimals=3, pct=False):
    if val is None:
        return "  -  "
    if pct:
        return f"{val:.1f}%"
    return f"{val:.{decimals}f}"


def _row_line(cells, widths):
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))


def _table(headers, data_rows):
    widths = [max(len(str(h)), max((len(str(r[i])) for r in data_rows), default=0))
              for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in widths)
    lines = [_row_line(headers, widths), sep]
    for row in data_rows:
        lines.append(_row_line(row, widths))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def sorted_keys(summary):
    """Return cell keys in canonical order."""
    present_archs   = sorted({k[0] for k in summary}, key=lambda a: ARCH_ORDER.index(a) if a in ARCH_ORDER else 99)
    present_domains = sorted({k[1] for k in summary}, key=lambda d: DOMAIN_ORDER.index(d) if d in DOMAIN_ORDER else 99)
    present_diffs   = sorted({k[2] for k in summary}, key=lambda d: DIFF_ORDER.index(d) if d in DIFF_ORDER else 99)
    present_top_ks  = sorted({k[3] for k in summary}, key=lambda x: int(x) if str(x).isdigit() else 99)
    return present_archs, present_domains, present_diffs, present_top_ks


def table_accuracy(summary) -> str:
    archs, domains, diffs, top_ks = sorted_keys(summary)
    headers = ["Architecture", "top_k", "Domain", "Difficulty", "N_ok", "N_total", "EM", "F1"]
    rows = []
    for arch in archs:
        for top_k in top_ks:
            for domain in domains:
                for diff in diffs:
                    key = (arch, domain, diff, top_k)
                    if key not in summary:
                        continue
                    s = summary[key]
                    rows.append([
                        arch, top_k, domain, diff,
                        s["n_ok"], s["n_total"],
                        _fmt(s["em"]),
                        _fmt(s["f1"]) if domain == "hotpotqa" else "  -  ",
                    ])
    return "=== ACCURACY (EM and F1) ===\n" + _table(headers, rows)


def table_pivot_em(summary) -> str:
    archs, domains, diffs, top_ks = sorted_keys(summary)
    col_keys = [(d, diff) for d in domains for diff in diffs]
    col_headers = [f"{d[:4]}/{diff[:4]}" for d, diff in col_keys]
    headers = ["Architecture", "k"] + col_headers
    rows = []
    for arch in archs:
        for top_k in top_ks:
            row = [arch, top_k]
            has_any = False
            for domain, diff in col_keys:
                key = (arch, domain, diff, top_k)
                if key in summary:
                    row.append(_fmt(summary[key]["em"]))
                    has_any = True
                else:
                    row.append("  -  ")
            if has_any:
                rows.append(row)
    return "=== ACCURACY PIVOT (EM) ===\n" + _table(headers, rows)


def table_tokens(summary) -> str:
    archs, domains, diffs, top_ks = sorted_keys(summary)
    headers = ["Architecture", "k", "Domain", "Difficulty", "Avg_Input", "Avg_Output", "Avg_Total"]
    rows = []
    for arch in archs:
        for top_k in top_ks:
            for domain in domains:
                for diff in diffs:
                    key = (arch, domain, diff, top_k)
                    if key not in summary:
                        continue
                    s = summary[key]
                    rows.append([
                        arch, top_k, domain, diff,
                        _fmt(s["avg_in_tok"],  0),
                        _fmt(s["avg_out_tok"], 0),
                        _fmt(s["avg_tok"],     0),
                    ])
    return "=== TOKEN USAGE (avg per question, successful rows only) ===\n" + _table(headers, rows)


def table_speed(summary) -> str:
    archs, domains, diffs, top_ks = sorted_keys(summary)
    headers = ["Architecture", "k", "Domain", "Difficulty", "Avg_Elapsed_s"]
    rows = []
    for arch in archs:
        for top_k in top_ks:
            for domain in domains:
                for diff in diffs:
                    key = (arch, domain, diff, top_k)
                    if key not in summary:
                        continue
                    s = summary[key]
                    rows.append([arch, top_k, domain, diff, _fmt(s["avg_elapsed"], 1)])
    return "=== SPEED (avg wall-clock seconds per question, all rows) ===\n" + _table(headers, rows)


def table_steps(summary) -> str:
    archs, domains, diffs, top_ks = sorted_keys(summary)
    headers = ["Architecture", "k", "Domain", "Difficulty", "Avg_Steps"]
    rows = []
    for arch in archs:
        for top_k in top_ks:
            for domain in domains:
                for diff in diffs:
                    key = (arch, domain, diff, top_k)
                    if key not in summary:
                        continue
                    s = summary[key]
                    rows.append([arch, top_k, domain, diff, _fmt(s["avg_steps"], 2)])
    return "=== GRAPH STEPS (avg per question, successful rows only) ===\n" + _table(headers, rows)


def table_errors(summary) -> str:
    archs, domains, diffs, top_ks = sorted_keys(summary)
    headers = ["Architecture", "k", "Domain", "Difficulty", "Errors", "Total", "Error%"]
    rows = []
    for arch in archs:
        for top_k in top_ks:
            for domain in domains:
                for diff in diffs:
                    key = (arch, domain, diff, top_k)
                    if key not in summary:
                        continue
                    s = summary[key]
                    n_err = s["n_total"] - s["n_ok"]
                    rows.append([
                        arch, top_k, domain, diff,
                        n_err, s["n_total"],
                        _fmt(s["error_pct"], 1, pct=True),
                    ])
    return "=== ERROR RATES ===\n" + _table(headers, rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results/raw", help="Directory containing result CSVs")
    parser.add_argument("--out", default="results/summary.txt", help="Output file path")
    args = parser.parse_args()

    raw_dir = Path(args.dir)
    if not raw_dir.exists():
        print(f"ERROR: {raw_dir} does not exist.")
        sys.exit(1)

    rows = load_all(raw_dir)
    if not rows:
        print(f"No CSV files found in {raw_dir}")
        sys.exit(1)

    print(f"Loaded {len(rows)} rows from {len(list(raw_dir.glob('*.csv')))} files.\n")

    summary = aggregate(rows)

    sections = [
        table_accuracy(summary),
        table_pivot_em(summary),
        table_tokens(summary),
        table_speed(summary),
        table_steps(summary),
        table_errors(summary),
    ]

    output = "\n\n".join(sections)
    print(output)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()

"""
Re-run errored rows in a result CSV without re-doing successful rows.

Usage:
  python retry_failed.py results/raw/level2b_hotpotqa_easy_cap2.csv
  python retry_failed.py results/raw/level1_math_hard_cap2.csv --seed 42

How it works:
  1. Reads the CSV and finds rows where error != ''.
  2. Reloads the same dataset (same domain/difficulty/n/seed) to recover
     context paragraphs (needed for HotpotQA; MATH context is always empty).
  3. Re-invokes the graph for each errored row.
  4. Patches the CSV in place, preserving row order and all successful rows.
"""

import argparse
import csv
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.datasets import load_dataset
from src.evaluators import score
from src.runner import _make_initial_state, _clean_hotpotqa_prediction, FIELDNAMES


def retry_failed(csv_path: str, seed: int = 42, delay: float = 1.0) -> None:
    csv_path = Path(csv_path)

    with open(csv_path, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    errored = [r for r in all_rows if r.get("error", "")]
    if not errored:
        print(f"No errors found in {csv_path} — nothing to do.")
        return

    print(f"Found {len(errored)} errored row(s) in {csv_path.name}")

    # Infer cell parameters from the CSV (all rows in one file share the same cell)
    sample = errored[0]
    arch       = sample["architecture"]
    domain     = sample["domain"]
    difficulty = sample["difficulty"]
    cap        = int(sample["iteration_cap"])
    n          = len(all_rows)  # same size as the original run

    # Reload the dataset with the same seed so we recover context for HotpotQA.
    # For MATH, context is always {}, but the question/gold must still match by ID.
    print(f"Reloading {domain}/{difficulty} (n={n}, seed={seed})...")
    ds_rows = load_dataset(domain, difficulty, n=n, seed=seed)
    ds_by_id = {r["id"]: r for r in ds_rows}

    # Build the graph once
    from run_experiment import get_graph_builder
    graph = get_graph_builder(arch, cap)()

    # Retry each errored row and collect patches
    patched = {r["id"]: dict(r) for r in all_rows}

    for err_row in errored:
        row_id = err_row["id"]
        ds_row = ds_by_id.get(row_id)
        if ds_row is None:
            print(f"  [SKIP] {row_id}: not found in reloaded dataset")
            continue

        print(f"  Retrying {row_id} ...", end=" ", flush=True)
        record = patched[row_id]

        t0 = time.time()
        try:
            initial_state = _make_initial_state(ds_row, domain, difficulty)
            final_state = graph.invoke(initial_state)

            raw_prediction = final_state.get("final_answer", "")
            prediction = (
                _clean_hotpotqa_prediction(raw_prediction)
                if domain == "hotpotqa"
                else raw_prediction
            )
            scores = score(domain, prediction, ds_row["gold_answer"])

            record["raw_prediction"]      = raw_prediction
            record["prediction"]          = prediction
            record["exact_match"]         = scores.get("exact_match", 0)
            record["f1"]                  = scores.get("f1", "")
            record["total_input_tokens"]  = final_state.get("total_input_tokens", 0)
            record["total_output_tokens"] = final_state.get("total_output_tokens", 0)
            record["total_steps"]         = final_state.get("total_steps", 0)
            record["elapsed_seconds"]     = round(time.time() - t0, 2)
            record["error"]               = ""
            print(f"em={record['exact_match']}")

        except Exception as e:
            record["elapsed_seconds"] = round(time.time() - t0, 2)
            record["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            traceback.print_exc()
            print(f"STILL FAILED — {record['error'][:80]}")

        patched[row_id] = record
        if delay > 0:
            time.sleep(delay)

    # Write the patched CSV, preserving original row order
    ordered_ids = [r["id"] for r in all_rows]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row_id in ordered_ids:
            writer.writerow(patched[row_id])

    remaining = sum(1 for r in patched.values() if r.get("error", ""))
    print(f"\nDone. {remaining} error(s) remain in {csv_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retry errored rows in a result CSV")
    parser.add_argument("csv", help="Path to the result CSV (e.g. results/raw/level2b_hotpotqa_easy_cap2.csv)")
    parser.add_argument("--seed", type=int, default=42, help="Same seed used for the original run (default: 42)")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between retried API calls (default: 1.0)")
    args = parser.parse_args()
    retry_failed(args.csv, seed=args.seed, delay=args.delay)

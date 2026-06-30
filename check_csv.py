import csv, sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/raw/level2b_hotpotqa_easy_cap2.csv"
with open(path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"{len(rows)} rows in {path}")
for r in rows:
    same = r["raw_prediction"] == r["prediction"]
    cleaned = "" if same else " [CLEANED]"
    print(f"  em={r['exact_match']} f1={r['f1'][:6]} same={same}{cleaned}")
    print(f"    raw : {repr(r['raw_prediction'][:80])}")
    if not same:
        print(f"    pred: {repr(r['prediction'][:80])}")

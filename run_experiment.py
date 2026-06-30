"""
CLI entry point for running experiments.

Usage examples:
  # Debug pilot (5 samples, fallback data, Level 1 only)
  python run_experiment.py --arch level1 --domain math --difficulty easy --n 5 --fallback

  # Full Level 1 run on all cells
  python run_experiment.py --arch level1 --n 25

  # Level 2B with iteration cap sweep
  python run_experiment.py --arch level2b --domain math --difficulty hard --cap 1
  python run_experiment.py --arch level2b --domain math --difficulty hard --cap 2
  python run_experiment.py --arch level2b --domain math --difficulty hard --cap 3
"""

import argparse
import sys
from dotenv import load_dotenv
load_dotenv()

ARCHITECTURES = ["level1", "level2a", "level2b", "level3"]
DOMAINS = ["math", "hotpotqa"]
DIFFICULTIES = ["easy", "hard"]


def get_graph_builder(arch: str, iteration_cap: int, top_k: int = 3):
    if arch == "level1":
        from src.graphs.level1 import build_graph
        return build_graph
    elif arch == "level2a":
        from src.graphs.level2a import build_graph
        return lambda: build_graph(top_k=top_k)
    elif arch == "level2b":
        from src.graphs.level2b import build_graph
        return lambda: build_graph(iteration_cap=iteration_cap, top_k=top_k)
    elif arch == "level3":
        from src.graphs.level3 import build_graph
        return lambda: build_graph(iteration_cap=iteration_cap, top_k=top_k)
    else:
        raise ValueError(f"Unknown architecture: {arch!r}")


def main():
    parser = argparse.ArgumentParser(description="Run multi-agent architecture experiments")
    parser.add_argument("--arch", choices=ARCHITECTURES, default=None,
                        help="Architecture to run. Omit to run all.")
    parser.add_argument("--domain", choices=DOMAINS, default=None,
                        help="Dataset domain. Omit to run both.")
    parser.add_argument("--difficulty", choices=DIFFICULTIES, default=None,
                        help="Difficulty subset. Omit to run both.")
    parser.add_argument("--n", type=int, default=25,
                        help="Samples per cell (default: 25).")
    parser.add_argument("--cap", type=int, default=2,
                        help="Max iteration cap for Solver-Critic loop (default: 2).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset sampling.")
    parser.add_argument("--fallback", action="store_true",
                        help="Use offline fallback data (no HuggingFace download).")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between API calls (default: 0.5).")
    parser.add_argument("--top_k", type=int, default=3,
                        help="TF-IDF retrieval top-k for L2A/L2B HotpotQA (default: 3; use 10 for full-context sweep).")
    args = parser.parse_args()

    from src.runner import run_cell

    archs = [args.arch] if args.arch else ARCHITECTURES
    domains = [args.domain] if args.domain else DOMAINS
    difficulties = [args.difficulty] if args.difficulty else DIFFICULTIES

    # Skip level3 unless explicitly requested (it's an optional stretch goal)
    if args.arch is None:
        archs = [a for a in archs if a != "level3"]

    total_cells = len(archs) * len(domains) * len(difficulties)
    print(f"Running {total_cells} cell(s): {archs} × {domains} × {difficulties}, n={args.n}")

    all_results = []
    for arch in archs:
        builder = get_graph_builder(arch, args.cap, args.top_k)
        for domain in domains:
            for difficulty in difficulties:
                print(f"\n--- {arch} | {domain} | {difficulty} ---")
                try:
                    results = run_cell(
                        architecture=arch,
                        domain=domain,
                        difficulty=difficulty,
                        graph_builder=builder,
                        n=args.n,
                        seed=args.seed,
                        iteration_cap=args.cap,
                        top_k=args.top_k,
                        use_fallback=args.fallback,
                        delay_between_calls=args.delay,
                    )
                    all_results.extend(results)
                except Exception as e:
                    print(f"ERROR in cell {arch}/{domain}/{difficulty}: {e}")
                    continue

    successful = [r for r in all_results if r["error"] == ""]
    if successful:
        em_scores = [r["exact_match"] for r in successful]
        print(f"\nOverall EM: {sum(em_scores)/len(em_scores):.3f} ({sum(em_scores)}/{len(em_scores)})")
        f1_scores = [r["f1"] for r in successful if r["f1"] != ""]
        if f1_scores:
            print(f"Overall F1: {sum(f1_scores)/len(f1_scores):.3f} (hotpotqa rows only)")
    print("Done.")


if __name__ == "__main__":
    main()

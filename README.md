# Comparative Analysis of Agentic Architectures on Reasoning and Information-retrieval Tasks

An empirical study of *when*, *why*, and *at what token cost* multi-agent scaffolding outperforms a single-agent baseline. Four LangGraph architectures of increasing complexity are compared on a controlled matrix of reasoning (MATH) and information-seeking (HotpotQA) tasks, holding the underlying model fixed so that performance differences are attributable to architecture rather than model choice.

The full write-up of methodology, results, and analysis is in [`final_report.md`](final_report.md) (PDF version: `final_report.pdf`).

---

## Key findings

- **Scaffolding bought cost, not accuracy.** Across every cell, the multi-agent architectures failed to beat the bare single-agent baseline on accuracy, while generally consuming more tokens (up to 2.5× on MATH hard).
- **The critic loop rarely helped and sometimes hurt**, occasionally regressing correct answers on binary comparison questions (a case of *ending divergence*).
- **Experience replay added nothing**, because the dominant failure modes (retrieval gaps and surface-form scoring artifacts) are not the recurring reasoning traps trajectory memory is designed to fix.
- **A retrieval-budget ablation (top-*k* = 3 vs. 10)** showed that most of the multi-agent "deficit" on HotpotQA was an information-access asymmetry, not an architectural one: equalizing access recovered 10–20 EM points.

---

## Architectures

| Level | Design | Components | Communication |
|-------|--------|------------|---------------|
| **L1** | Bare single baseline | Single LLM call | None |
| **L2A** | Planner–Executor | Planner, Executor | Shared state (plan list) |
| **L2B** | Solver–Critic | Solver, Critic | Shared state (JSON feedback), bounded retry loop |
| **L3** | Experience-replaying Solver–Critic | Solver, Critic, Memory Retriever, Memory Builder | Shared state + local prompt injection from `trajectory_cache.json` |

All architectures share one `EvalState`, one LLM client, and the same prompts where possible. Only the node topology and output schema differ between them.

---

## Repository structure

```
.
├── run_experiment.py        # CLI entry point — runs a cell or the full matrix
├── analyze.py               # Aggregates raw CSVs into results/summary.txt
├── plot.py                  # Generates the three comparison figures
├── retry_failed.py          # Re-runs only the errored rows of a result CSV
├── check_csv.py             # Quick inspector for a single result CSV
├── requirements.txt
├── .env                     # Holds the endpoint API key (not committed)
├── src/
│   ├── llm_client.py        # Sole API entry point; handles the Qwen3 thinking/JSON quirk + token accounting
│   ├── state.py             # Shared LangGraph TypedDict state
│   ├── schemas.py           # Pydantic schemas (Plan, CriticReview, AnswerOutput)
│   ├── datasets.py          # MATH + HotpotQA loaders, difficulty subsets, offline fallbacks
│   ├── tools.py             # TF-IDF top-k retrieval tool (HotpotQA)
│   ├── evaluators.py        # Deterministic scorers (MATH boxed-answer, HotpotQA EM/F1)
│   ├── prompts.py           # One prompt template per agent role
│   ├── runner.py            # Architecture-agnostic experiment loop + per-row CSV logging
│   └── graphs/
│       ├── level1.py
│       ├── level2a.py
│       ├── level2b.py
│       └── level3.py
├── results/
│   ├── raw/                 # Per-row CSV output (one file per cell)
│   ├── summary.txt          # Aggregated accuracy / tokens / steps / speed / errors
│   └── figures/             # fig1_main_comparison, fig2_retrieval_ablation, fig3_token_efficiency
├── trajectory_cache_k3.json # L3 experience-replay memory (k=3 run)
└── trajectory_cache_k10.json# L3 experience-replay memory (k=10 run)
```

---

## Setup

**Requirements:** Python 3.10+.

```bash
pip install -r requirements.txt
```

Core dependencies: `langgraph`, `langchain-core`, `openai`, `pydantic`, `datasets`, `pandas`, `matplotlib`, `python-dotenv`.

**Model endpoint.** All agents call a single model, **Qwen3.6-27B**, served via the ELTE Faculty of Informatics vLLM endpoint (OpenAI-compatible API). Access requires the ELTE network or VPN and an API key. Place the key in a `.env` file at the project root:

```
ELTE_API_KEY=your_key_here
```

> **Note:** the endpoint is reachable only from inside the ELTE network or via the ELTE VPN. The datasets also ship with small offline fallbacks (`--fallback`) so the pipeline can be exercised without a HuggingFace download.

---

## Usage

### Run experiments

```bash
# Debug pilot: 5 samples, offline fallback data, Level 1 only
python run_experiment.py --arch level1 --domain math --difficulty easy --n 5 --fallback

# Full Level 1 run across all four cells (n = 30)
python run_experiment.py --arch level1 --n 30

# A single multi-agent cell
python run_experiment.py --arch level2b --domain hotpotqa --difficulty hard --n 30

# Retrieval-budget ablation (full-context condition)
python run_experiment.py --arch level2b --domain hotpotqa --difficulty hard --n 30 --top_k 10
```

Omitting `--arch`, `--domain`, or `--difficulty` runs all of that dimension. Results are written incrementally to `results/raw/` as one CSV per cell, so a dropped connection mid-run is recoverable.

**Main flags:** `--n` (samples per cell, default 25), `--top_k` (TF-IDF retrieval depth, default 3; use 10 for the full-context sweep), `--seed` (sampling seed, default 42), `--fallback` (offline data), `--delay` (seconds between API calls).

### Recover errored rows

```bash
python retry_failed.py results/raw/level2b_hotpotqa_hard_cap2.csv
```

### Aggregate and plot

```bash
python analyze.py     # writes results/summary.txt
python plot.py        # writes the three figures to results/figures/
```

---

## Datasets

| Domain | Source (HuggingFace) | Easy | Hard |
|--------|----------------------|------|------|
| MATH (reasoning) | `qwedsacf/competition_math` | Level 1–2 | Level 4–5 |
| HotpotQA (info-seeking) | `hotpotqa/hotpot_qa` | single-hop–leaning | multi-hop, distractor setting |

`qwedsacf/competition_math` is an active mirror of the original Hendrycks et al. MATH benchmark (the canonical `hendrycks/competition_math` repository is currently disabled). MATH answers are scored by extracting the final `\boxed{}` expression and comparing with LaTeX-aware numeric/set normalization; HotpotQA is scored with official normalization plus Exact Match and token-level F1. All scoring is deterministic — no LLM-as-a-judge.

---

## Evaluation matrix

Core matrix: **4 architectures × 2 domains × 2 difficulties × 30 samples**, at retrieval *k* = 3, plus a *k* = 3 vs. *k* = 10 retrieval ablation on all HotpotQA cells for L2A/L2B/L3. The same sampled questions are reused across architectures within a cell, so comparisons are paired. Each cell is a 30-sample run with no recorded errors (a single row in one cell returned an empty response and is scored as a miss).

Per-row metrics: Exact Match, F1 (HotpotQA), input/output tokens, graph steps, wall-clock seconds.

---

## Use of GenAI Tools

| Phase | GenAI Tool Used | Validation Method |
|-------|-----------------|-------------------|
| Literature review | Gemini DeepResearch | Read the main recommended papers myself and checked the summaries against the original sources for relevance and accuracy. |
| Project planning | Gemini DeepResearch | Reviewed the generated project plan critically, proposed alternative directions, and applied my own corrections before committing to it. |
| Code design (pre-implementation) | Claude Opus 4.8 | Reviewed the proposed architecture and module breakdown, suggested different directions, and adjusted the design to fit the project's constraints. |
| Code implementation (step-by-step) | Claude Code (Opus 4.8) | Tested the code and the ELTE vLLM endpoint and compared the implementation against the plan; manually tested individual cell runs before the full n = 30 experiment to confirm correct behavior, accepting code only after verifying it ran as intended. |
| Code debugging | Claude Code (Opus 4.8) | Inspected unexpected agent behaviour during test runs and actively participated in diagnosing and fixing the underlying issues (e.g. prompt contradictions, critic over-correction, retrieval asymmetry). |
| Analysis | Claude Opus 4.8 | I cross-checked the interpretation with the findings from the experiments and corrected the model where its output was not supported or was weak in reasoning. |
| Final report drafting | Claude Opus 4.8 | Checked every number and figure in the report against the experimental results in `results/summary.txt`; reviewed the analysis and reasoning for accuracy. |
| Converting final report to PDF | Claude Opus 4.8 | Checking the result. |

---

## Reproducibility notes

- A single `llm_call` wrapper is the only point that touches the API; it disables the model's thinking mode whenever structured JSON output is requested (a Qwen3 reasoning/structured-output conflict), forces an explicit `max_tokens`, and accumulates token usage per call.
- Sampling is seeded (`--seed`, default 42) and difficulty subsets are stratified, so runs are reproducible.
- Results are written incrementally per row; `retry_failed.py` recovers any rows that errored (e.g. from a transient endpoint timeout) without re-running successful ones.
- Dataset loaders fall back to small in-file curated samples when HuggingFace is unreachable, so the pipeline runs offline for debugging.

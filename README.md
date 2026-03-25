# Self-Evolving Memory Agent

A complete, end-to-end self-evolving memory agent system in Python that solves math problems using a closed-loop architecture of:
- **Small NN Policy** (actor-critic, PyTorch) — decides what action to take
- **Memory Graph** (NetworkX) — stores skill/concept/error nodes with statistics
- **SymPy Solver** — executes mathematical skills (algebra + ODEs)
- **Verifier** — checks answers using SymPy
- **Curriculum Generator** — creates new problem variants targeting weak skills
- **PPO Trainer** — updates the policy from episode experience
- **Episode Store** — SQLite-backed episodic memory

## Architecture

```
[New Problem]
      │
      ▼
  ┌─────────┐
  │  STATE   │◄──────────────────────────────────┐
  │(problem, │                                   │
  │scratchpad│                                   │
  │retrieved,│                                   │
  │verify)   │                                   │
  └────┬─────┘                                   │
       │                                         │
       ▼                                         │
  ┌─────────┐    ┌──────────────┐                │
  │ POLICY   │───►│ MEMORY GRAPH │                │
  │(small NN)│    │(retrieve     │                │
  │          │◄───│ skills/errors)│                │
  └────┬─────┘    └──────┬───────┘                │
       │                 │                         │
       │ action          │ skill procedure         │
       ▼                 ▼                         │
  ┌──────────────────────────┐                    │
  │    MODEL / SOLVER        │                    │
  │(SymPy + templates,      │                    │
  │ produces reasoning step) │                    │
  └────────────┬─────────────┘                    │
               │                                   │
               │ new scratchpad / candidate        │
               ▼                                   │
  ┌──────────────────┐                            │
  │    VERIFIER       │                            │
  │(SymPy/rule-based) │                            │
  └────────┬──────────┘                            │
           │                                       │
           │ pass/fail + diagnostics               │
           ▼                                       │
  ┌──────────────────┐     ┌──────────────┐       │
  │  EPISODE STORE   │────►│ GRAPH UPDATE  │───────┘
  │(SQLite traces)   │     │(stats, edges, │
  └──────────────────┘     │ new nodes)    │
                            └──────┬───────┘
                                   │
                            ┌──────▼───────┐
                            │ POLICY UPDATE │
                            │(PPO-style)    │
                            └──────────────┘
```

## File Structure

```
├── data/seed_problems.json    # ~50 seed math problems
├── memory/
│   ├── graph.py               # NetworkX memory graph
│   ├── episode_store.py       # SQLite episodic memory
│   ├── retrieval.py           # BM25 + graph-based retrieval
│   └── consolidation.py       # Periodic graph consolidation
├── solver/solver.py           # SymPy-based template solver
├── verifier/verifier.py       # SymPy verification + diagnostics
├── policy/policy_nn.py        # Actor-critic NN (PyTorch)
├── rl/ppo.py                  # PPO trainer
├── envs/math_env.py           # Gymnasium-style REPL environment
├── curriculum/generator.py    # Problem variant generator
└── scripts/run_loop.py        # Main entry point
```

## Installation

```bash
pip install -r requirements.txt
# or
pip install -e .
```

## How to Run

```bash
# Run for 60 minutes
python scripts/run_loop.py --duration 60

# With custom paths
python scripts/run_loop.py \
    --duration 120 \
    --db-path episodes.db \
    --graph-path graph.pkl \
    --policy-path policy.pt \
    --metrics-path metrics.json \
    --seed 42
```

## Expected Metrics

After running, `metrics.json` contains time-series data:
- `verified_accuracy` — fraction of held-out problems solved correctly
- `eval_avg_steps` — average steps per episode  
- `overall_success_rate` — training success rate
- `graph_nodes / graph_edges` — graph growth over time
- `curriculum_generated` — number of generated problems

## Extending the System

### Add a new skill
In `memory/graph.py`, add to `SEED_SKILLS`:
```python
{
    "node_id": "skill_my_skill",
    "label": "My Skill",
    "topic": "my_topic",
    "trigger": {"keywords": ["keyword1", "keyword2"]},
    "procedure": ["step 1", "step 2"],
}
```
Then add a handler in `solver/solver.py`'s `_SKILL_DISPATCH`.

### Add a new domain (e.g., physics)
1. Add seed problems to `data/seed_problems.json` with `"domain": "physics"`
2. Add skill nodes in `memory/graph.py`
3. Add solver implementations in `solver/solver.py`
4. Update `TOPIC_MAP` in `envs/math_env.py`

## Components

| Component | Description |
|-----------|-------------|
| `PolicyNetwork` | 2-layer MLP, 16-dim state → 5 actions (RETRIEVE/SOLVE/VERIFY/REPAIR/GENERATE) |
| `MemoryGraph` | NetworkX DiGraph with skill/concept/error nodes and weighted edges |
| `EpisodeStore` | SQLite DB storing full episode traces with step-level detail |
| `Retriever` | BM25 episode search + graph activation spreading |
| `Consolidator` | Promotes frequent skill combos, updates edges, applies decay |
| `Solver` | SymPy-based: linear, quadratic, polynomial, ODE (separable + linear 1st order) |
| `Verifier` | Substitute-back checks for algebra; derivative + IC checks for ODEs |
| `PPOTrainer` | Clipped PPO with GAE advantages, entropy bonus |
| `CurriculumGenerator` | Parametric coefficient variation + adversarial perturbation |

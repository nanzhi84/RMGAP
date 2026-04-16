# RMGAP

Benchmarking the Generalization of Reward Models across Diverse Preferences.

RMGAP is an evaluation framework for reward model generalizability: the ability of RMs to correctly rank responses that align with diverse user preferences. The benchmark comprises 1,097 instances across Chat, Writing, Reasoning, and Safety domains, each with four distinct responses of different linguistic profiles and multiple paraphrased prompts to test consistency.

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Run evaluation (EndoRM by default)
bash run.sh model.path=/path/to/model

# Switch RM type via Hydra
bash run.sh rm=genrm_pointwise model.path=/path/to/model

```

## Project Structure

```
RMGAP/
├── pyproject.toml              # Package metadata and dependencies
├── run.sh                      # Entry point script
├── data/
│   └── test.jsonl              # Evaluation dataset (1097 instances)
└── src/rmeval/
    ├── __init__.py              # Exports: TaskRunner, make_rm
    ├── task_runner.py           # Data loading, metric computation, reporting
    ├── provider.py              # OpenAI-compatible API client
    ├── cli/
    │   ├── main_eval.py         # Hydra CLI entry point
    │   └── config/              # Hydra YAML configs
    │       ├── main_eval.yaml
    │       └── rm/              # Per-RM configs
    └── rm/
        ├── base.py              # BaseRM + GenerativeRM template
        ├── scalar.py            # ScalarRM — embedding-based scoring
        ├── endorm.py            # EndoRM — response logprob aggregation
        ├── dpo_implicit.py      # DpoImplicitRM — policy vs reference logprob
        ├── generative_verifier.py  # GenerativeVerifierRM — YES/NO logprob
        ├── genrm_pointwise.py   # GenRMPointwise — 1-10 score generation
        └── genrm_pairwise.py    # GenRMPairwise — A>B>C>D ranking
```



## Metrics

All metrics are computed per domain (Chat, Writing, Reasoning, Safety) and averaged:

- **Pair Accuracy** — fraction of pairwise comparisons where the winner scores higher
- **BoN Accuracy** — fraction of items where the winner has the highest score among all responses
- **Consistency** — fraction of prompt groups where all paraphrased prompts produce the same ranking

## Dataset Format

Each line in `data/test.jsonl` is a JSON object:

| Field | Description |
|---|---|
| `id` | Unique instance identifier |
| `domain` | One of `Chat`, `Writing`, `Reasoning`, `Safety` |
| `responses` | 4 candidate responses, each with `key` and `text` |
| `prompt_groups` | 4 groups, each with a `winner` key and 3 semantically equivalent `prompts` |
| `style_assignments` | Style profile per response (Formality, Conciseness, etc.) |

## Configuration

Hydra configs live in `src/rmeval/cli/config/`. Override any parameter via CLI:

```bash
# Custom data path
bash run.sh data.path=data/other.jsonl

# Adjust sglang engine settings
bash run.sh rm=genrm_pointwise rm.sglang_cfg.engine.tp_size=4

# Use OpenAI-compatible backend
bash run.sh rm=genrm_pointwise rm.sglang_cfg.backend=openai \
    rm.sglang_cfg.provider.extras.api_key=$OPENAI_API_KEY \
    rm.sglang_cfg.provider.extras.base_url=https://api.openai.com/v1
```

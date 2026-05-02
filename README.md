# RMGAP

RMGAP benchmarks reward model generalization across diverse user preferences. The released dataset has 1,097 instances across Chat, Writing, Reasoning, and Safety. Each instance contains four stylistically distinct responses and four prompt groups; each prompt group has three paraphrased prompts that target one preferred response.

The public entry point is reward model evaluation. The data construction pipeline is included as optional reproducibility tooling under the same Python package.

## Install

Create a Linux virtual environment and install the evaluation dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[eval]"
```

Install the construction dependencies only if you want to rebuild the dataset:

```bash
python -m pip install -e ".[build]"
```

## Evaluate Reward Models

The released benchmark lives at `data/test.jsonl`. Run the default EndoRM evaluation with:

```bash
python -m rmgap.cli.main_eval model.path=/path/to/model
```

Switch reward model implementations through Hydra:

```bash
python -m rmgap.cli.main_eval rm=scalar model.path=/path/to/reward-model
python -m rmgap.cli.main_eval rm=generative_verifier model.path=/path/to/instruct-model
python -m rmgap.cli.main_eval rm=genrm_pointwise model.path=/path/to/instruct-model
python -m rmgap.cli.main_eval rm=genrm_pairwise model.path=/path/to/instruct-model
python -m rmgap.cli.main_eval rm=dpo_implicit model.path=/path/to/dpo-model \
  rm.params.reference_model_path=/path/to/reference-model
```

The Linux helper script is equivalent to the default command and assumes `.venv` already exists:

```bash
bash run.sh model.path=/path/to/model
```

Outputs are written to `log/<model_id>/<timestamp>/` and include `args.yaml`, `result.log`, and `metrics.json`.

## Metrics

RMGAP reports domain-level metrics and their macro average:

- `Pair`: pairwise accuracy for the preferred response against each rejected response.
- `BoN`: Best-of-N accuracy, where the preferred response must score above all three alternatives.
- `Consistency`: ranking stability across the three paraphrased prompts in the same prompt group.

## Dataset Format

Each JSONL row in `data/test.jsonl` contains:

| Field | Description |
|---|---|
| `id` | Stable instance identifier |
| `domain` | `Chat`, `Writing`, `Reasoning`, or `Safety` |
| `source` | Source dataset name |
| `responses` | Four candidate responses, each with `key` and `text` |
| `prompt_groups` | Four groups, each with a `winner` key and three prompts |
| `style_assignments` | Linguistic profile assigned to each response |

## Data Construction

The construction pipeline is optional and expensive. It expects local source datasets under `source/`, writes intermediate files under ignored paths such as `data/generated/`, and reads API keys from environment variables.

The expected source layout is:

```text
source/
├── Chat/<source-name>/*.jsonl
├── Reasoning/<source-name>/*.jsonl
├── Safety/<source-name>/*.jsonl
└── Writing/<source-name>/*.jsonl
```

Merge source prompts:

```bash
python -m rmgap.build.prompt_selection.merge_prompts \
  --source-base source \
  --output data/generated/prompts.jsonl
```

Filter prompts:

```bash
export RMGAP_EMBEDDING_MODEL_PATH=/path/to/embeddinggemma-300m

python -m rmgap.build.prompt_selection.filter_prompts \
  --input data/generated/prompts.jsonl \
  --output-chat data/generated/prompts-filtered-chat.jsonl \
  --output-reasoning data/generated/prompts-filtered-reasoning.jsonl \
  --output-safety data/generated/prompts-filtered-safety.jsonl \
  --output-writing data/generated/prompts-filtered-writing.jsonl
```

Run all construction stages for the four domains:

```bash
export OPENROUTER_API_KEY=...
export DEEPSEEK_API_KEY=...

for domain in chat reasoning safety writing; do
  python -m rmgap.build.main \
    --config configs/generation.yaml \
    --stage res_gen \
    --input "data/generated/prompts-filtered-${domain}.jsonl" \
    --output "data/generated/${domain}"

  for stage in res_eval pro_gen pro_eval rw_gen rw_eval write_test; do
    python -m rmgap.build.main \
      --config configs/generation.yaml \
      --stage "${stage}" \
      --resume "data/generated/${domain}/protocols.jsonl"
  done
done
```

Each stage updates `data/generated/<domain>/protocols.jsonl` and appends run metadata to `data/generated/<domain>/runs.jsonl`.

## Tests

```bash
python -m pytest -q
```

## Structure

```text
RMGAP/
├── configs/
├── data/
│   └── test.jsonl
├── src/
│   └── rmgap/
├── tests/
├── LICENSE
├── pyproject.toml
├── README.md
└── run.sh
```

Local source datasets, generated protocol files, embedding models, logs, caches, paper files, PDF drafts, and secrets are intentionally ignored by Git.

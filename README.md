# COMPREHEND

Builds an entailment hierarchy over a set of user prompts: leaf prompts are
pairwise-matched and summarized into increasingly general prompts, level by
level, until a single root remains. Every generated summary is LLM-judged on
entailment, faithfulness, informativeness, and specificity, and external
per-model quality scores are attached to the leaves and averaged up the tree.
An interactive viewer renders the resulting DAG.

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/JackNapier20/Comprehend.git
cd Comprehend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Provide API credentials:

```bash
cp .env.example .env
# edit .env and fill in your keys
```

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | GPT models (default judge for evaluation) |
| `ANTHROPIC_API_KEY` | Claude models (default for pairing and summary generation) |
| `GEMINI_API_KEY` | Gemini models (optional) |
| `GEMMA_BASE_URL`, `GEMMA_API_KEY` | OpenAI-compatible endpoint for a locally served model (optional) |

`.env` is read automatically by `main.py` and is never committed.

## Configuration

The two knobs at the top of `main.py`:

```python
DATA_FILE  = Path("...")   # the tree JSON to build / score / view
SCORE_FILE = "..."         # the external per-prompt score CSV
```

`DATA_FILE` and `SCORE_FILE` must come from the same dataset (WildBench base
JSON with the WildBench score CSV, UltraFeedback with the UltraFeedback CSV).
A mismatch does not raise an error: the score join is a fuzzy text match and
would silently attach scores from the wrong dataset.

Which provider handles each pipeline role is set in `MODEL_CONFIG`
(`blossom`, `summary_generation`, `evaluation`), each one of
`"openai" | "claude" | "gemma" | "gemini"`.

All commands must be run from the repository root.

## Running

### Build a hierarchy

```bash
python main.py run                      # blossom (default)
python main.py run --method evaltree    # EvalTree-style hierarchy (Not-deafult)
python main.py run --method both        # blossom, then evaltree
```

Neither method modifies `DATA_FILE`: each works on its own copy
(`<stem>_blossom.json` / `<stem>_evaltree.json`) whose `config` header
records which providers built it (`method`, `blossom`, `summary_generation`,
`evaluation`). An existing blossom copy is resumed rather than rebuilt.

**Blossom** (`--method blossom`) loads `DATA_FILE` (a base JSON with one level of leaf prompts), then per level:
finds each node's third-nearest neighbor, pairs nodes with maximum-weight
matching, generates a common summary per pair (regenerating once if the
weakest entailment score is at or below 0.4), judges every summary
(entailment, faithfulness, informativeness, plus a specificity pass against
the third neighbor), records per-level statistics, promotes winners to the
next level, and reassigns eligible third neighbors as additional children.
The tree is saved back to the `_blossom` copy after every level, so an
interrupted run keeps all completed levels.

After the root is reached, external scores from `SCORE_FILE` are attached to
the leaves and averaged up the tree, and the viewer is launched.

**EvalTree** (`--method evaltree`) builds a hierarchy over the same level-0
leaves by a different route: each leaf is annotated with a capability gerund
phrase, the leaves are clustered with recursive 2-means over sentence
embeddings, each cluster gets a broad gerund-phrase summary generated
bottom-up, third neighbors are assigned per level, and every summary is
judged (entailment of sampled descendant leaves, plus specificity against
the closest sibling-cluster outsider). The result is written to
`<DATA_FILE stem>_evaltree.json`, so blossom and evaltree hierarchies built
from the same base coexist. `--method both` builds blossom first (viewer
deferred), then evaltree.

Note: this mode makes many LLM API calls. A 64-leaf tree takes several hours
and has real API cost.

### Evaluate a built tree

```bash
python main.py score [path/to/tree.json]
```

Post-hoc evaluation of a built tree, blossom or evaltree (defaults to the
`_blossom` copy of `DATA_FILE` when no path is given). The tree's `config`
header tells the two methods apart: for **blossom** trees the external
per-model scores from `SCORE_FILE` are first (re)attached to the leaves and
averaged up the tree (skipped with a warning when the CSV is absent);
evaltree trees skip score attachment. Both then get the same leaf-sampling
evaluation below.

The evaluator can also be run directly on several files at once:

```bash
python evaluation/evaluate.py path/to/tree.json [more.json ...]
```

It works on any built tree, blossom or evaltree:
for every internal node it samples two descendant leaves and judges the
node's summary for faithfulness and informativeness (the same leaves for
both metrics), and judges an outsider leaf for the two specificity
variants — the sibling-cluster leaf closest (by sentence embedding) to the
node's own descendant-leaf centroid, recorded as `third_leaf` /
`third_leaf_sim`. All metrics are written under their own field names
(`faithfulness1/2`, `informativeness1/2`, `specificity_faithfulness`,
`specificity_informativeness`) to `eval_outputs/<name>_eval.json` next to
the input. `stats.py` turns one or more such files into the metric table
(Avg/Min/Max/StdDev/StdErr/N per metric):

```bash
python stats.py eval_outputs/<name>_eval.json [more.json ...]
```
 `--model` selects the judge
provider (default `openai`; `main.py score` uses `MODEL_CONFIG["evaluation"]`);
`--samples` and `--seed` control the leaf sampling.

### Re-run the in-pipeline judging

```bash
python main.py simplescore [path/to/tree.json]
```

Re-runs the in-pipeline evaluation (`evaluation/simple_eval.py`: entailment,
faithfulness, informativeness, specificity in their pairwise, match-based
form) on every level of an already-built tree, then re-attaches and
re-aggregates the external scores. Defaults to `DATA_FILE` when no path is
given. Use this after changing judge prompts; it regenerates no summaries.

### View a built tree

```bash
python main.py view
```

Exports `DATA_FILE` to the viewer format and serves the visualisation at
`http://localhost:8010/visualisation/viewer.html` (if the port is taken, the
next free port is used and printed). The command blocks; press Ctrl+C to stop.

### View a specific exported tree

```bash
python main.py view-json demo_trees/tree_data.json
```

Serves the viewer for any previously exported DAG JSON without touching
`DATA_FILE`. Three example trees are provided in `demo_trees/`:

| File | Contents |
|---|---|
| `demo_trees/tree_data.json` | full 1024-prompt WildBench hierarchy |
| `demo_trees/Ablation1_random_vis.json` | ablation variant 1, random subset |
| `demo_trees/Ablation5_random_vis.json` | ablation variant 5, random subset |

In the viewer, click a node to see its full prompt text and scores; the
score dropdown switches the color overlay between metrics.

## Preparing input data

`dataset_prep/` contains the dataset preparation scripts:

- `prepare_dataset.py` builds the WildBench score CSV.
- `prepare_ultrafeedback.py` builds the UltraFeedback score CSV and sampled
  base JSONs.
- `prepare_uf_model_scores.py` adds per-model score columns for UltraFeedback.
- `uf_nearest_neighbors.py` builds a nearest-neighbor sheet for manual
  subset curation.
- `xlsx_to_base_json.py` (repository root) converts a single-column xlsx of
  prompts into a base tree JSON.


## Preparing Science Exercise dataset
**Nemotron-SFT-Science**
- `prepare_ultrafeedback.py` builds the UltraFeedback score CSV and sampled
  base JSONs.
- `prepare_uf_model_scores.py` adds per-model score columns for UltraFeedback.
- `prepare_nemotronsft.py`
- `nt_nearest_neighbors.py`
- `nt_sampling_leaf.py`
- `xlsx_to_base_json.py`

Large derived CSVs and built trees are intentionally not committed; they are
regenerated by the scripts above.

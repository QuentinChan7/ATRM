# ATRM

ATRM performs temporal relation prediction by combining a recurrent graph
backbone with three-layer Memory, tool-guided fusion, and offline LLM-guided
reflection. Given a subject, object, and timestamp, the backbone scores candidate
relations; Memory and tools refine these scores using historical evidence.
Validated reflection experience provides auxiliary supervision for backbone
fine-tuning. Final prediction does not call the LLM.

## Environment

Use Python 3.10, PyTorch 2.4.0, and a CUDA-enabled DGL build compatible with
PyTorch and the CUDA runtime. Run the following commands from this directory:

```bash
conda create -n atrm python=3.10 -y
conda activate atrm
python -m pip install -r requirements.txt
```

Install the matching DGL build separately, then check the environment:

```bash
python -c "import torch, dgl; print('PyTorch:', torch.__version__); print('DGL:', dgl.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Training, trace collection, and graph evaluation require CUDA. The offline
selection stage calls an OpenAI-compatible LLM endpoint and does not load the
LLM into the training process. Keep the LLM serving environment separate from
the graph-training environment.

## Data Preparation

Supported datasets are `ICEWS14`, `ICEWS18`, and `GDELT`.
Place each dataset under `data/<DATASET>/`:

```text
data/ICEWS14/
  train.txt
  valid.txt
  test.txt
  stat.txt
  entity2id.txt
  relation2id.txt
  e-w-graph.txt
  sbert_features.pt
```

- Split files contain tab-separated `subject_id`, `relation_id`, `object_id`,
  and `timestamp` columns, ordered chronologically.
- Mapping files contain `name<TAB>id`, with zero-based contiguous IDs.
- The first two values in `stat.txt` are the entity and relation counts.
- ICEWS14 and ICEWS18 additionally use the static entity-word graph
  `e-w-graph.txt` and semantic features `sbert_features.pt`. GDELT does not
  require these two files under the supplied configuration.

Generate semantic features separately for each ICEWS dataset. With a local
`bge-small-en-v1.5` encoder directory:

```bash
python scripts/build_sbert_features.py --dataset ICEWS14 \
  --model models/bge-small-en-v1.5 \
  --backend transformers --pooling cls --local-files-only
```

This writes `data/ICEWS14/sbert_features.pt`, with embeddings indexed by the
dataset's entity and relation IDs. For ICEWS18, change `--dataset` accordingly;
the encoder can be shared, but the generated feature file cannot.

Use `--data-root` with the main entry point, or `--data-dir` with the feature
builder, when storing datasets elsewhere.

## Reproduction

The workflow has five stages. Use the same dataset and work directory throughout.
The example below uses ICEWS14 and GPU 0, with random seed 42.

### 1. Train the Backbone

Train with joint entity and relation supervision and select the checkpoint on
validation MRR. The training budget is specified explicitly with `--epochs`.

```bash
python src/launch.py train --dataset ICEWS14 \
  --work-dir runs/icews14 --epochs 8
```

The selected weights are saved as `runs/icews14/base.pt`.

### 2. Collect Memory and Tool Experience

Load the reference backbone, process the complete validation split, and record
candidate rankings, Memory evidence, tool traces, and target-specific outcomes.
This stage does not update GNN weights or evaluate the test split.

```bash
python src/launch.py trace --dataset ICEWS14 \
  --work-dir runs/icews14
```

### 3. Select Offline Reflection Experience

Start an OpenAI-compatible LLM service, then configure its endpoint and served
model name. Set `AGENT_LLM_API_KEY` when authentication is required.

```bash
AGENT_LLM_BASE_URL=http://127.0.0.1:8000/v1 \
AGENT_LLM_MODEL=qwen3-8b \
python src/launch.py select --dataset ICEWS14 \
  --work-dir runs/icews14
```

The LLM selects from experience patterns constructed from earlier validation-time
groups. Later groups are held out from its prompts and used to validate the
selections. The stage reports `validated_contexts` after compiling usable
supervision. If no contexts survive, it stops instead of proceeding to training.

### 4. Fine-Tune with Reflection Supervision

Stop the LLM service first if it shares the graph-training GPU. Fine-tuning starts
from the reference checkpoint and uses training-split triples with auxiliary
supervision compiled from the validation traces.

```bash
python src/launch.py distill --dataset ICEWS14 \
  --work-dir runs/icews14
```

The default is one epoch at learning rate `0.0001`. The selected fine-tuned
checkpoint is written to `runs/icews14/atrm.pt`.

### 5. Evaluate on the Full Test Split

```bash
python src/launch.py evaluate --dataset ICEWS14 \
  --work-dir runs/icews14
```

Evaluation first calibrates Memory on validation data using `base.pt`, then loads
`atrm.pt` for the complete test split. All forward and inverse queries are scored.
Within each snapshot, scoring precedes history updates; test outcomes do not train
the reflection policy. Progress bars report completed snapshots and estimated
remaining time.

## Configuration and Checkpoints

Dataset-specific settings are defined in `src/config.py`:

| Dataset | History Length | GNN Layers |
| --- | ---: | ---: |
| ICEWS14 | 9 | 2 |
| ICEWS18 | 10 | 2 |
| GDELT | 7 | 2 |

The shared backbone uses hidden dimension 200, dropout 0.2, and history blending
weight 0.3. Default learning rates are `0.001` for `train` and `0.0001` for
`distill`. Both stages accept `--epochs` and `--lr`. Use `--gpu` and `--seed`
to configure graph execution, and `--data-root` to change the dataset location.

`--reference-checkpoint` replaces the default `<work-dir>/base.pt` for trace
collection, distillation, and validation Memory calibration. The meaning of
`--checkpoint` depends on the stage:

| Stage | `--checkpoint` |
| --- | --- |
| `train` | Initial weights for continuation with a fresh optimizer |
| `trace` | Backbone weights used to collect validation experience |
| `evaluate` | Weights scored on the test split after reference Memory calibration |

For example, evaluate an existing ATRM checkpoint with its reference backbone:

```bash
python src/launch.py evaluate --dataset ICEWS14 \
  --work-dir runs/icews14 \
  --reference-checkpoint checkpoints/base.pt \
  --checkpoint checkpoints/atrm.pt
```

To evaluate reference weights with Memory fusion, use the same reference file
for both checkpoint arguments. Existing checkpoint outputs and trace files are
not overwritten; use a new work directory for a new run. Offline selection can
reuse cached responses when the prompt and model settings match.

## Outputs

| Path Under the Work Directory | Contents |
| --- | --- |
| `base.pt` | Validation-selected backbone checkpoint |
| `traces.jsonl` | Full validation Memory and tool experience |
| `selection/fit_catalog.json` | Candidate patterns from the earlier time groups |
| `selection/batch_*.json` | Cached LLM responses |
| `selection/hypotheses.json` | Selected reflection hypotheses, checked by temporal validation before use |
| `atrm.pt` | Validation-selected fine-tuned checkpoint |
| `metrics.json` | Full-test metrics from the most recent evaluation |

In `metrics.json`, `relation_filter` reports final filtered relation MRR and
Hits@1/3/10, along with the number of evaluated queries. `gnn_relation_filter`
reports scores from the **same loaded checkpoint before Memory fusion**; it is
not a separate non-LLM checkpoint. Raw relation and entity metrics are also
reported.

## Code Layout

- `src/`: entry point, dataset configuration, GNN, training, and evaluation.
- `awesome_agent/memory/`: three-layer Memory and evidence construction.
- `awesome_agent/tools/`: evidence tools and tool planning.
- `awesome_agent/fusion.py`: relation-score fusion and validation calibration.
- `scripts/`: semantic feature construction and offline reflection selection.
- `tests/`: Memory, ranking, temporal validation, and reproduction checks.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

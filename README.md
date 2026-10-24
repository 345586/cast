# CAST: Task-Adaptive Skill Bundle Recommendation

This anonymous release contains the training and inference implementation of CAST. It does not include datasets, embeddings, checkpoints, logs, validation records, test results, or experiment artifacts.

## Method

CAST uses the following pipeline:

1. Frozen BGE semantic encoding and Top-20 candidate recall.
2. TB, TS, and BS LightGCN views constructed from training annotations.
3. Four skill-side channels: semantic anchor, full-document chunk mean, TS graph feature, and BS graph feature.
4. Task- and bundle-conditioned chunk attention.
5. Shared TB/TS/BS utility with explicit risk and cost penalties.
6. Pair-complete variable-length beam search with 190 initial pairs and beam width 5.

The TB graph provides task-side features for the TB view; it is not an additional skill-side channel. The current implementation does not use an instance-dependent dynamic gate or a post-hoc Hypergraph reranker.

## Environment

The reference environment uses Python 3.12, PyTorch 2.6 with CUDA support, and a CUDA-capable GPU. Install a compatible `sentence-transformers` release and the frozen BGE-small-en-v1.5 model according to the local environment policy.

## Main hyperparameters

```yaml
seed: 45
seeds: [45, 267839, 1]
learning_rate: 0.001
weight_decay: 0.0001
epochs: 100
patience: 20
recall_k: 20
beam_width: 5
min_bundle_size: 2
mask_probability: 0.25
risk_mode: max
risk_cost_lambda: 0.01
chunk_weight: 0.1
chunk_margin: 0.1
chunk_temperature: 1.0
preference_temperature: 0.1
lightgcn_layers: 2
```

The aggregate cost uses the bounded form

`C(B) = sum(cost_s) / (1 + sum(cost_s))`.

Risk is the maximum member risk. Explicitly prohibited skills should be filtered by an external policy before recommendation; the learned penalty is not a safety guarantee.

## Commands

From the release directory:

```bash
bash scripts/run_smoke.sh /path/to/data /path/to/smoke-output
```

```bash
bash scripts/run_train.sh /path/to/data /path/to/output 45
bash scripts/run_train.sh /path/to/data /path/to/output 267839
bash scripts/run_train.sh /path/to/data /path/to/output 1
```

Each command trains one independent seed. Checkpoints, logs, and metrics are generated only under the user-supplied output directory and are not included in this release folder.

```bash
bash scripts/run_eval.sh /path/to/data /path/to/checkpoint.pt /path/to/eval-output test
```

All paths are supplied by the user. The scripts do not contain machine-specific paths and do not write results into the source tree by default.

## Metrics and limitations

Offline recovery compares the recommended skill set with the reference skill set using Precision, Recall, F1, Jaccard, bundle size, risk, and cost. Document-derived edit labels are weak preference evidence, not execution outcomes or causal effects. Test labels must be used only for final evaluation, never for checkpoint selection or hyperparameter tuning.

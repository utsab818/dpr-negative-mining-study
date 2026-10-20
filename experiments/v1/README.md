# Experiment V1 — Preliminary Negative Sampling

## Environment
- Hardware: Apple MacBook Air M4, 16 GB unified memory
- Accelerator: PyTorch MPS
- Model: Two independent DistilBERT encoders
- Dataset: SQuAD v1.1
- Warm-up training examples: 128
- Mining corpus: 1,000 passages
- Held-out evaluation: 1,000 questions against 1,000 passages
- Second-stage random seed: 42

## Training strategies

| Strategy | Negatives/query | Steps | Query exposures |
|---|---:|---:|---:|
| Random-only | 4 random | 32 | 128 |
| Hard-only | 4 model-mined | 32 | 128 |
| Mixed | 4 hard + 3 in-batch | 32 | 128 |
| In-batch (step-matched) | 32 in-batch | 32 | 1,056 |
| In-batch (approximately exposure-matched) | 32 in-batch | 4 | 132 |

Query exposures count sampled examples, not unique questions.

## Results — 1,000-passage evaluation

| Strategy | Recall@1 | Recall@5 | Recall@20 | MRR@10 |
|---|---:|---:|---:|---:|
| Warm-up | 0.306 | 0.631 | 0.871 | 0.4422 |
| Random 4 | 0.354 | 0.656 | 0.901 | 0.4880 |
| Hard 4 | 0.438 | 0.732 | 0.919 | 0.5626 |
| Mixed | 0.451 | 0.747 | 0.930 | 0.5765 |
| In-batch 32, 32 steps | 0.438 | 0.753 | 0.936 | 0.5757 |
| In-batch 32, 4 steps | 0.324 | 0.655 | 0.885 | 0.4632 |

## Preliminary interpretation

Four model-mined hard negatives matched 32 in-batch negatives on Recall@1 under the 32-step condition.

The mixed strategy achieved the highest Recall@1 and MRR@10 in these runs.

These results do not demonstrate that negative quality is universally more important than quantity.

## Limitations

- Only one second-stage seed was evaluated.
- Only 128 unique training questions were used.
- The in-batch experiments were not matched simultaneously for steps, query exposures, and compute.
- False-negative filtering used answer-string matching.
- The evaluation benchmark used one annotated positive passage per query.
- The original warm-up checkpoint was shared by all branches.
- Training time, energy, and memory usage were not measured.
- Sampling in V1 was not a strict epoch-based sampler.

## Next experiment

V2 will introduce epoch-based sampling, preserve checkpoints separately, and record training configuration and compute information.

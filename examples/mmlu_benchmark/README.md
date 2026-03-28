# MMLU Benchmark Example

Evaluate language models on [MMLU (Massive Multitask Language Understanding)](https://arxiv.org/abs/2009.03300) with optional efficient evaluation via [DISCO](https://arxiv.org/abs/2510.07959).

## Installation

For basic MMLU evaluation:

```bash
uv pip install .[mmlu]
```

For DISCO prediction (includes DISCO dependencies):

```bash
uv pip install .[disco]
```

## Run without DISCO (full evaluation)

From the project root:

```bash
uv run python examples/mmlu_benchmark/mmlu_benchmark.py --model_id alignment-handbook/zephyr-7b-sft-full
```

Full evaluation results look like:

```
================================================================================
Results Summary (Evaluated Tasks)
================================================================================
Total tasks: 14042
Correct: 8291
Accuracy (on anchor points): 0.5904
Accuracy norm (on anchor points): 0.5904
```

## Run with DISCO (predicted full-benchmark score)

From the project root:

```bash
uv run python examples/mmlu_benchmark/mmlu_benchmark.py --model_id alignment-handbook/zephyr-7b-sft-full --disco_model_path arubique/DISCO-MMLU
```

Predicted score output:

```
----------------------------------------
DISCO Predicted Full Benchmark Accuracy:
----------------------------------------
  Model 0: 0.606739
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_id` | HuggingFace model identifier (e.g. `meta-llama/Llama-2-7b-hf`) | *(required)* |
| `--data_path` | Path to MMLU prompts JSON file or Hugging Face dataset repo id | `arubique/flattened-MMLU` |
| `--anchor_points_path` | Path to anchor points pickle file; if set, only anchor tasks are evaluated | — |
| `--output_dir` | Directory to save results | `./results` |
| `--predictions_path` | Path to save predictions tensor as pickle (for DISCO) | — |
| `--limit` | Limit number of tasks to evaluate (for testing) | — |
| `--batch_size` | Batch size for evaluation (reserved for future use) | `1` |
| `--device` | Device to run model on (e.g. `cuda:0`, `cpu`) | `cuda:0` |
| `--num_workers` | Number of parallel workers for task execution | `1` |
| `--disco_model_path` | If set, run DISCO prediction; path to `.pkl`, `.npz`, or Hugging Face repo id | — |
| `--disco_transform_path` | Path to DISCO PCA transform `.pkl` or `.npz` (for local DISCO model when using `--pca`) | — |
| `--pca` | PCA dimension for DISCO embeddings | — |
| `--pad_to_size` | Pad predictions to this size with -inf | — |
| `--use_lmeval_batching` | Use [lm-evaluation-harness-style](https://github.com/EleutherAI/lm-evaluation-harness) batching for exact numerical match with DISCO repo | off |

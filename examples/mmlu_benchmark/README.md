# MMLU Benchmark Example

Evaluate language models on [MMLU (Massive Multitask Language Understanding)](https://arxiv.org/abs/2009.03300) with optional efficient evaluation via [DISCO (Diversifying Sample Condensation)](https://arxiv.org/abs/2510.07959).

## Installation

Install [uv package manager](https://docs.astral.sh/uv/) as described [here](https://docs.astral.sh/uv/getting-started/installation/).

Create Python environment:

```bash
uv venv --python 3.11
```

Install dependencies for basic MMLU evaluation:

```bash
uv sync --extra mmlu
```

Install dependencies for MMLU evaluation with DISCO:

```bash
uv sync --extra disco
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
Correct: 8292
Accuracy: 0.5905
```

## Run with DISCO (predicted full-benchmark score)

From the project root:

```bash
uv run python examples/mmlu_benchmark/mmlu_benchmark.py --model_id alignment-handbook/zephyr-7b-sft-full --disco_model_path arubique/DISCO-MMLU
```

Predicted score output:

```
================================================================================
Results Summary (Evaluated Tasks)
================================================================================
Total tasks: 100
Correct: 36
Accuracy: 0.3600

================================================================================
DISCO Prediction
================================================================================
Computing embeddings and predicting full benchmark accuracy...
Fetching 9 files: 100%|██████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 19171.53it/s]
  Using: DISCO predictor from Hugging Face (arubique/DISCO-MMLU)

----------------------------------------
DISCO Predicted Full Benchmark Accuracy:
----------------------------------------
  Model 0 (alignment-handbook/zephyr-7b-sft-full): 0.602309
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

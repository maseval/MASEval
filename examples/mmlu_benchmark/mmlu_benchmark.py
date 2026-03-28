"""
MMLU Benchmark Example - Evaluating Models on Anchor Points

This example demonstrates how to evaluate a HuggingFace model on MMLU tasks
using anchor point-based task selection for DISCO prediction.

Equivalent to the disco-public command:
    python scripts/run_lm_eval.py \\
        --anchor_points_path=/path/to/anchor_points_disagreement.pkl \\
        --model=hf \\
        --model_args=pretrained=alignment-handbook/zephyr-7b-sft-full,trust_remote_code=True \\
        --tasks=mmlu_prompts \\
        --skip_non_anchor_points \\
        --use_full_prompt

Usage:
    # Run with default settings (evaluates on all tasks; uses arubique/flattened-MMLU by default)
    python mmlu_benchmark.py --model_id "meta-llama/Llama-2-7b-hf"

    # Run with anchor points filtering (for DISCO prediction)
    python mmlu_benchmark.py \\
        --model_id "alignment-handbook/zephyr-7b-sft-full" \\
        --anchor_points_path /path/to/anchor_points_disagreement.pkl

    # Run with DISCO prediction (passing --disco_model_path enables it)
    python mmlu_benchmark.py \\
        --model_id "alignment-handbook/zephyr-7b-sft-full" \\
        --anchor_points_path /path/to/anchor_points_disagreement.pkl \\
        --disco_model_path /path/to/fitted_weights.pkl \\
        --disco_transform_path /path/to/transform.pkl \\
        --pca 256

    # Run on a subset of tasks for testing
    python mmlu_benchmark.py --model_id "meta-llama/Llama-2-7b-hf" --limit 10

    # Override data source (path to JSON or Hugging Face repo id)
    python mmlu_benchmark.py --model_id "meta-llama/Llama-2-7b-hf" --data_path /path/to/mmlu_prompts_examples.json
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

# MASEval imports
from maseval.core.callbacks.result_logger import FileResultLogger

# MMLU benchmark imports
from maseval.benchmark.mmlu import (
    DEFAULT_DEVICE,
    DefaultMMLUBenchmark,
    load_tasks,
    compute_benchmark_metrics,
)


# Example constants (configurable)
DEFAULT_OUTPUT_DIR = "./results"
DEFAULT_N_CHOICES = 4
DEFAULT_PCA_DIM = 256
PCA_RANDOM_STATE = 42
DISCO_META_FILENAME = "disco_meta.json"
MMLU_PROMPTS_FILENAME = "mmlu_prompts_examples.json"
DISCO_CONFIG_FILENAME = "config.json"
ANCHOR_POINTS_FILENAME = "anchor_points.json"
RESULTS_FILENAME_PATTERN = "mmlu_{timestamp}.jsonl"
SUMMARY_FILENAME = "summary.json"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MMLU Benchmark - Evaluate models on MMLU multiple choice questions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace model identifier (e.g., 'meta-llama/Llama-2-7b-hf')",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="arubique/flattened-MMLU",
        help="Path to MMLU prompts JSON file, or Hugging Face dataset repo id (e.g. username/mmlu-prompts-examples)",
    )

    # Optional arguments
    parser.add_argument(
        "--anchor_points_path",
        type=str,
        default=None,
        help="Path to anchor points pickle file. If provided, evaluates only anchor tasks.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save results",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        default=None,
        help="Path to save predictions tensor as pickle (for DISCO predictor)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to evaluate (for testing)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation (currently not implemented, reserved for future)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Device to run model on (e.g., 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for task execution",
    )

    # DISCO prediction: enabled when --disco_model_path is set
    parser.add_argument(
        "--disco_model_path",
        type=str,
        default=None,
        help="If set, run DISCO prediction (path to fitted weights .pkl, .npz, or Hugging Face repo id)",
    )
    parser.add_argument(
        "--disco_transform_path",
        type=str,
        default=None,
        help="Path to DISCO PCA transform .pkl or .npz (required for local .pkl model when using --pca)",
    )
    parser.add_argument(
        "--pca",
        type=int,
        default=None,
        help="PCA dimension for DISCO embeddings (default: 256)",
    )
    parser.add_argument(
        "--pad_to_size",
        type=int,
        default=None,
        help="Pad predictions to this size with -inf (default: no padding, disco-public uses 31)",
    )
    parser.add_argument(
        "--use_lmeval_batching",
        action="store_true",
        help="Use lm-evaluation-harness batching for exact numerical match. This batches ALL requests together before computing logprobs.",
    )

    return parser.parse_args()


def save_predictions_for_disco(
    results: list,
    output_path: str,
    anchor_points: Optional[list] = None,
    n_choices: int = DEFAULT_N_CHOICES,
    pad_to_size: Optional[int] = None,
):
    """Save predictions in format compatible with DISCO predictor.

    Creates a predictions tensor of shape (1, n_questions, pad_to_size)
    where the values are log-probabilities.

    If logprobs are available in the results (from logprobs-based evaluation),
    uses actual log-likelihoods. Otherwise, falls back to 0/-inf format.

    Args:
        results: Benchmark results list.
        output_path: Path to save predictions pickle.
        anchor_points: Optional anchor points for ordering.
        n_choices: Number of answer choices (default 4 for A/B/C/D).
        pad_to_size: Pad predictions to this size with -inf (default: no padding).
    """
    predictions_list = []

    def get_pred_vec(entry, n_choices):
        """Extract prediction vector from entry, using logprobs if available."""
        # Check if actual logprobs are available
        logprobs = entry.get("logprobs")
        if logprobs is not None and len(logprobs) >= n_choices:
            return logprobs[:n_choices]

        # Fall back to 0/-inf format based on predicted index
        predicted = entry.get("predicted", -1)
        pred_vec = [float("-inf")] * n_choices
        if 0 <= predicted < n_choices:
            pred_vec[predicted] = 0.0
        return pred_vec

    def extract_eval_entries(res):
        """Extract evaluation entries from a result dict."""
        eval_data = res.get("eval")
        if eval_data is None:
            return []
        if isinstance(eval_data, list):
            return eval_data
        if isinstance(eval_data, dict):
            return [eval_data]
        return []

    if anchor_points is not None:
        # Order results by anchor points
        result_by_doc_id = {}
        for res in results:
            for entry in extract_eval_entries(res):
                doc_id = entry.get("doc_id")
                if doc_id is not None:
                    result_by_doc_id[doc_id] = entry

        for doc_id in anchor_points:
            entry = result_by_doc_id.get(doc_id, {})
            pred_vec = get_pred_vec(entry, n_choices)
            predictions_list.append(pred_vec)
    else:
        # Use results in order
        for res in results:
            for entry in extract_eval_entries(res):
                pred_vec = get_pred_vec(entry, n_choices)
                predictions_list.append(pred_vec)

    predictions = np.array(predictions_list)

    # Pad to specified size if requested
    if pad_to_size is not None and predictions.shape[1] < pad_to_size:
        padding = np.full(
            (predictions.shape[0], pad_to_size - predictions.shape[1]),
            float("-inf"),
            dtype=predictions.dtype,
        )
        predictions = np.concatenate([predictions, padding], axis=1)

    predictions = predictions.reshape(1, -1, predictions.shape[-1])  # (1, n_questions, n_choices)

    if output_path and output_path != os.devnull:
        with open(output_path, "wb") as f:
            pickle.dump(predictions, f)
        print(f"Saved predictions tensor to {output_path}")
        print(f"  Shape: {predictions.shape}")
        print(f"  Dtype: {predictions.dtype}")
    else:
        print(f"Built predictions tensor with shape: {predictions.shape}")

    return predictions


def _pca_transform_numpy(X: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Apply PCA transform using only numpy: (X - mean) @ components.T."""
    return (X - mean) @ components.T


def _predict_tree_numpy(
    X: np.ndarray,
    children_left: np.ndarray,
    children_right: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
    value: np.ndarray,
) -> np.ndarray:
    """Predict for one tree; X shape (n_samples, n_features). Returns (n_samples,)."""
    out = np.empty(X.shape[0], dtype=np.float64)
    for i in range(X.shape[0]):
        node = 0
        while children_left[node] != -1:
            if X[i, feature[node]] <= threshold[node]:
                node = children_left[node]
            else:
                node = children_right[node]
        out[i] = value[node]
    return out


def _predict_rf_numpy(
    X: np.ndarray,
    tree_node_counts: np.ndarray,
    children_left: np.ndarray,
    children_right: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
    value: np.ndarray,
) -> np.ndarray:
    """Predict using extracted RF tree arrays; X shape (n_samples, n_features). Returns (n_samples,)."""
    offsets = np.concatenate([[0], np.cumsum(tree_node_counts)])
    n_trees = len(tree_node_counts)
    preds = np.zeros((n_trees, X.shape[0]), dtype=np.float64)
    for t in range(n_trees):
        lo, hi = offsets[t], offsets[t + 1]
        preds[t] = _predict_tree_numpy(
            X,
            children_left[lo:hi],
            children_right[lo:hi],
            feature[lo:hi],
            threshold[lo:hi],
            value[lo:hi],
        )
    return np.mean(preds, axis=0)


def load_disco_from_npz(
    model_npz_path: str,
    transform_npz_path: str,
    meta_path: Optional[str] = None,
) -> tuple:
    """Load DISCO transform and model from .npz (no pickle).

    Returns:
        (transform_fn, predict_fn, meta_dict). transform_fn(X) returns transformed
        array; predict_fn(X) returns predicted accuracies; meta_dict has
        sampling_name, number_item, fitted_model_type.
    """
    transform_data = np.load(transform_npz_path)
    components = np.asarray(transform_data["components_"])
    mean = np.asarray(transform_data["mean_"])

    def transform_fn(X: np.ndarray) -> np.ndarray:
        return _pca_transform_numpy(X, components, mean)

    model_data = np.load(model_npz_path)
    tree_node_counts = np.asarray(model_data["tree_node_counts"], dtype=np.int64)
    children_left = np.asarray(model_data["children_left"], dtype=np.int32)
    children_right = np.asarray(model_data["children_right"], dtype=np.int32)
    feature = np.asarray(model_data["feature"], dtype=np.int32)
    threshold = np.asarray(model_data["threshold"], dtype=np.float64)
    value = np.asarray(model_data["value"], dtype=np.float64)

    def predict_fn(X: np.ndarray) -> np.ndarray:
        return _predict_rf_numpy(X, tree_node_counts, children_left, children_right, feature, threshold, value)

    if meta_path is None:
        meta_path = str(Path(model_npz_path).parent / DISCO_META_FILENAME)
    with open(meta_path) as f:
        meta = json.load(f)
    transform_npz = {"components_": components, "mean_": mean}
    return transform_fn, predict_fn, meta, transform_npz


def compute_disco_embedding(
    predictions: np.ndarray,
    pca: int,
    transform=None,
    transform_npz: Optional[dict] = None,
    apply_softmax: bool = True,
) -> tuple:
    """Compute DISCO embeddings from predictions.

    This implements the embedding computation from disco-public/experiments.py.

    Args:
        predictions: Predictions tensor of shape (n_models, n_anchor_points, n_classes).
        pca: PCA dimension for dimensionality reduction.
        transform: Pre-fitted sklearn PCA transform. If None, a new one will be fitted (unless transform_npz).
        transform_npz: If set, (components_, mean_) from npz; used instead of transform (no pickle).
        apply_softmax: Whether to apply softmax to predictions.

    Returns:
        Tuple of (embeddings, transform) where embeddings has shape (n_models, pca).
    """
    try:
        import torch
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError("DISCO prediction requires torch and sklearn. Install with: pip install torch scikit-learn") from e

    # Convert to torch tensor
    preds_tensor = torch.Tensor(predictions)

    # Apply softmax if requested
    if apply_softmax:
        emb_unreduced = preds_tensor.softmax(dim=-1)
    else:
        emb_unreduced = preds_tensor

    # Flatten to (n_models, n_anchor_points * n_classes)
    emb_unreduced = emb_unreduced.reshape(emb_unreduced.shape[0], -1)
    X = emb_unreduced.numpy()

    # Apply PCA
    if pca is not None:
        if transform_npz is not None:
            emb = _pca_transform_numpy(X, transform_npz["components_"], transform_npz["mean_"])
            emb = torch.Tensor(emb)
            transform = None
        elif transform is not None:
            emb = transform.transform(X)
            emb = torch.Tensor(emb)
        else:
            # Fit new PCA transform
            transform = PCA(
                n_components=pca,
                svd_solver="full",
                random_state=PCA_RANDOM_STATE,
            ).fit(X)
            emb = transform.transform(X)
            emb = torch.Tensor(emb)
    else:
        emb = emb_unreduced

    return emb, transform


def predict_with_disco(
    predictions: np.ndarray,
    model_path: str,
    transform_path: Optional[str] = None,
    pca: int = DEFAULT_PCA_DIM,
) -> dict:
    """Predict full benchmark performance using DISCO.

    This implements the prediction logic from disco-public/scripts/predict_model_performance.py.

    Use .npz paths (from extract_disco_weights.py) to avoid pickle/sklearn version warnings.
    Use a Hugging Face repo id (e.g. "<USERNAME>/my-disco-mmlu") to load via AutoModel.from_pretrained(..., trust_remote_code=True).

    Args:
        predictions: Predictions tensor of shape (n_models, n_anchor_points, n_classes).
        model_path: Path to fitted weights pickle, disco_model.npz, or HF repo id (USERNAME/repo).
        transform_path: Path to PCA transform pickle or disco_transform.npz (optional if npz or HF).
        pca: PCA dimension (default: 256).

    Returns:
        Dict with predicted accuracies and metadata.
    """
    # Hugging Face repo: model_path like "username/repo-name", no local path
    use_hf = "/" in model_path and not Path(model_path).exists() and (transform_path is None or transform_path == model_path)
    if use_hf:
        from transformers import AutoModel

        disco_model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        pred_values = disco_model.predict(np.asarray(predictions), apply_softmax=True)
        predicted_accs = {i: float(pred_values[i]) for i in range(len(pred_values))}
        config = getattr(disco_model, "config", None)
        meta = {
            "sampling_name": getattr(config, "sampling_name", "") if config else "",
            "number_item": getattr(config, "number_item", "") if config else "",
            "fitted_model_type": getattr(config, "fitted_model_type", "") if config else "",
        }
        print(f"  Using: DISCO predictor from Hugging Face ({model_path})")
        return {
            "predicted_accuracies": predicted_accs,
            "sampling_name": meta["sampling_name"],
            "number_item": meta["number_item"],
            "fitted_model_type": meta["fitted_model_type"],
            "pca": pca,
        }

    use_npz = model_path.endswith(".npz") and (transform_path or "").endswith(".npz")
    if model_path.endswith(".npz") and not (transform_path or "").endswith(".npz"):
        raise ValueError("When using .npz model path, provide --disco_transform_path to disco_transform.npz")

    if use_npz:
        transform_fn, predict_fn, meta, transform_npz = load_disco_from_npz(model_path, transform_path, meta_path=None)
        sampling_name = meta["sampling_name"]
        number_item = meta["number_item"]
        fitted_model_type = meta["fitted_model_type"]
        print(f"  Using: sampling={sampling_name}, n_items={number_item}, model={fitted_model_type} (from .npz)")

        embeddings, _ = compute_disco_embedding(
            predictions,
            pca=pca,
            transform=None,
            transform_npz=transform_npz,
            apply_softmax=True,
        )
        X_emb = embeddings.numpy() if hasattr(embeddings, "numpy") else np.asarray(embeddings)
        pred_values = predict_fn(X_emb)
        predicted_accs = {i: float(pred_values[i]) for i in range(len(pred_values))}
        return {
            "predicted_accuracies": predicted_accs,
            "sampling_name": sampling_name,
            "number_item": number_item,
            "fitted_model_type": fitted_model_type,
            "pca": pca,
        }

    # Pickle path
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    if not isinstance(model_data, dict):
        raise ValueError(f"model_path must contain a dict. Got {type(model_data)}")

    if transform_path is not None:
        with open(transform_path, "rb") as f:
            transform = pickle.load(f)
    elif "transform" in model_data:
        transform = model_data["transform"]
    else:
        raise ValueError("Transform not found. Provide --disco_transform_path or ensure the model file contains a 'transform' key.")

    if "fitted_weights" in model_data:
        fitted_weights = model_data["fitted_weights"]
    else:
        fitted_weights = {k: v for k, v in model_data.items() if k != "transform"}
        if not fitted_weights:
            raise ValueError("Could not find fitted_weights in model file.")

    sampling_name = model_data.get("sampling_name") or list(fitted_weights.keys())[0]
    number_item = model_data.get("number_item") or list(fitted_weights[sampling_name].keys())[0]
    fitted_model_type = model_data.get("fitted_model_type") or list(fitted_weights[sampling_name][number_item].keys())[0]

    print(f"  Using: sampling={sampling_name}, n_items={number_item}, model={fitted_model_type}")

    embeddings, _ = compute_disco_embedding(
        predictions,
        pca=pca,
        transform=transform,
        apply_softmax=True,
    )

    fitted_model = fitted_weights[sampling_name][number_item][fitted_model_type]
    predicted_accs = {}
    for model_idx in range(embeddings.shape[0]):
        model_embedding = embeddings[model_idx]
        model_embedding_np = model_embedding.numpy() if hasattr(model_embedding, "numpy") else np.array(model_embedding)
        predicted_acc = fitted_model.predict(model_embedding_np.reshape(1, -1))[0]
        predicted_accs[model_idx] = predicted_acc

    return {
        "predicted_accuracies": predicted_accs,
        "sampling_name": sampling_name,
        "number_item": number_item,
        "fitted_model_type": fitted_model_type,
        "pca": pca,
    }


def _resolve_data_path(data_path: str) -> str:
    """If data_path looks like an HF dataset repo id (user/repo), download and return local path."""
    if not data_path or "/" not in data_path or Path(data_path).exists():
        return data_path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return data_path
    try:
        local = hf_hub_download(
            repo_id=data_path,
            filename=MMLU_PROMPTS_FILENAME,
            repo_type="dataset",
        )
        return local
    except Exception:
        return data_path


def _apply_eval_config_from_repo(repo_path: Path, args: "argparse.Namespace") -> None:
    """Load eval_config from repo; forbid passing --pca/--pad_to_size/--use_lmeval_batching, then set args from eval_config."""
    config_path = repo_path / DISCO_CONFIG_FILENAME
    if not config_path.exists():
        return
    with open(config_path) as f:
        hf_config = json.load(f)
    eval_config = hf_config.get("eval_config") or {}
    if not eval_config:
        return
    # When using a Hub DISCO model with eval_config, user must not pass these; the model's values are used.
    errors = []
    if "pca" in eval_config and args.pca is not None:
        errors.append(f"do not pass --pca (model uses pca={eval_config['pca']})")
    if "pad_to_size" in eval_config and args.pad_to_size is not None:
        errors.append(f"do not pass --pad_to_size (model uses pad_to_size={eval_config['pad_to_size']})")
    if "use_lmeval_batching" in eval_config and args.use_lmeval_batching:
        errors.append("do not pass --use_lmeval_batching (model uses use_lmeval_batching=True)")
    if errors:
        raise ValueError("When using a DISCO model from the Hub, " + "; ".join(errors) + ". Omit these flags to use the model's eval_config.")
    # Require data_path to match model config (use_full_prompt is always True)
    if "data_path" in eval_config:
        # Compare before _resolve_data_path overwrites; user must pass the expected repo id or path
        if args.data_path != eval_config["data_path"]:
            raise ValueError(f"When using this DISCO model, --data_path must be {eval_config['data_path']!r} (model config).")
    if "pca" in eval_config:
        args.pca = eval_config["pca"]
    if "pad_to_size" in eval_config:
        args.pad_to_size = eval_config["pad_to_size"]
    if "use_lmeval_batching" in eval_config:
        args.use_lmeval_batching = eval_config["use_lmeval_batching"]


def _resolve_hf_disco_repo(
    disco_model_path: str,
    anchor_points_path: Optional[str],
) -> tuple:
    """If model_path is an HF repo, download and return (anchor_points_path, repo_path). Else (anchor_points_path, None).

    anchor_points_path: path to anchor_points.json if repo has it and input was None; else input.
    repo_path: path to the downloaded repo dir, or None if not an HF repo.
    """
    if "/" not in disco_model_path or Path(disco_model_path).exists():
        return (anchor_points_path, None)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return (anchor_points_path, None)
    repo_path = Path(snapshot_download(disco_model_path))
    if anchor_points_path is None and (repo_path / ANCHOR_POINTS_FILENAME).exists():
        return (str(repo_path / ANCHOR_POINTS_FILENAME), repo_path)
    return (anchor_points_path, repo_path)


def main():
    """Main entry point."""
    args = parse_args()

    # Validate DISCO prediction arguments and resolve HF repo (anchor points + eval_config)
    # Must run before _resolve_data_path so we can validate args.data_path against model config
    if args.disco_model_path is not None:
        anchor_points_path_resolved, hf_repo_path = _resolve_hf_disco_repo(args.disco_model_path, args.anchor_points_path)
        if anchor_points_path_resolved is None and args.anchor_points_path is None:
            raise ValueError(
                "Anchor points required for DISCO prediction. Provide --anchor_points_path or use a "
                "DISCO model repo that includes anchor_points.json (build with build_repo.py --anchor_points_path)."
            )
        if hf_repo_path is not None:
            if args.anchor_points_path is not None:
                raise ValueError(
                    "When using a DISCO model from the Hub, do not pass --anchor_points_path; anchor points are taken from the model repo."
                )
            if args.disco_transform_path is not None:
                raise ValueError(
                    "When using a DISCO model from the Hub, do not pass --disco_transform_path; the transform is included in the model repo."
                )
            _apply_eval_config_from_repo(hf_repo_path, args)
        if args.anchor_points_path is None and anchor_points_path_resolved:
            args.anchor_points_path = anchor_points_path_resolved
        if args.pca is not None and args.disco_transform_path is None:
            print("Warning: --pca specified without --disco_transform_path. Transform will be loaded from model file if available.")

    # Resolve --data_path if it is an HF dataset repo id (e.g. username/mmlu-prompts-examples)
    args.data_path = _resolve_data_path(args.data_path)

    print("=" * 80)
    print("MMLU Benchmark - MASEval")
    print("=" * 80)
    print(f"Model: {args.model_id}")
    print(f"Data path: {args.data_path}")
    print(f"Anchor points: {args.anchor_points_path or 'None (evaluate all)'}")
    print("Use full prompt: True")
    print(f"Device: {args.device}")
    print(f"Output dir: {args.output_dir}")
    if args.disco_model_path is not None:
        print("DISCO prediction: ENABLED")
        print(f"  Model path: {args.disco_model_path}")
        print(f"  Transform path: {args.disco_transform_path or '(from model file)'}")
        print(f"  PCA dimension: {args.pca}")
    print("=" * 80)

    # Load tasks
    print("\nLoading tasks...")
    tasks = load_tasks(
        data_path=args.data_path,
        anchor_points_path=args.anchor_points_path,
        limit=args.limit,
    )
    print(f"Loaded {len(tasks)} tasks")

    if args.anchor_points_path:
        anchor_points = tasks._anchor_points
        print(f"Filtering to {len(anchor_points)} anchor points")
    else:
        anchor_points = None

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create result logger
    logger = FileResultLogger(
        output_dir=str(output_dir),
        filename_pattern=RESULTS_FILENAME_PATTERN,
        validate_on_completion=False,
    )

    # Create benchmark
    benchmark = DefaultMMLUBenchmark(
        model_id=args.model_id,
        device=args.device,
        trust_remote_code=True,
        use_full_prompt=True,
        callbacks=[logger],
        num_workers=args.num_workers,
    )

    # Optionally precompute logprobs using lm-eval batching for exact match
    if args.use_lmeval_batching:
        print("\nPrecomputing logprobs using lm-eval batching ...")
        # Get task list for precomputation
        task_list = list(tasks._anchor_tasks if hasattr(tasks, "_anchor_tasks") else tasks._tasks)
        benchmark.precompute_all_logprobs_lmeval(task_list)

    # Run evaluation
    print("\nRunning evaluation...")
    agent_data = {
        "model_id": args.model_id,
        "use_full_prompt": True,
    }
    results = benchmark.run(tasks=tasks, agent_data=agent_data)

    # Compute metrics on evaluated tasks
    metrics = compute_benchmark_metrics(results)

    print("\n" + "=" * 80)
    print("Results Summary (Evaluated Tasks)")
    print("=" * 80)
    print(f"Total tasks: {metrics['total_tasks']}")
    print(f"Correct: {metrics['correct_count']}")
    print(f"Accuracy (on anchor points): {metrics['acc']:.4f}")
    print(f"Accuracy norm (on anchor points): {metrics['acc_norm']:.4f}")

    # Build predictions tensor for DISCO
    predictions = None
    if args.predictions_path or args.disco_model_path is not None:
        predictions = save_predictions_for_disco(
            results=results,
            output_path=args.predictions_path if args.predictions_path else None,
            anchor_points=anchor_points,
            pad_to_size=args.pad_to_size,
        )

    # Run DISCO prediction if enabled
    disco_results = None
    if args.disco_model_path is not None:
        print("\n" + "=" * 80)
        print("DISCO Prediction")
        print("=" * 80)
        print("Computing embeddings and predicting full benchmark accuracy...")

        disco_results = predict_with_disco(
            predictions=predictions,
            model_path=args.disco_model_path,
            transform_path=args.disco_transform_path,
            pca=args.pca,
        )

        print("\n" + "-" * 40)
        print("DISCO Predicted Full Benchmark Accuracy:")
        print("-" * 40)
        for model_idx, acc in disco_results["predicted_accuracies"].items():
            print(f"  Model {model_idx}: {acc:.6f}")

    # Save summary
    summary_data = {
        "model_id": args.model_id,
        "data_path": str(args.data_path),
        "anchor_points_path": str(args.anchor_points_path) if args.anchor_points_path else None,
        "use_full_prompt": True,
        "metrics": metrics,
    }

    if disco_results:
        summary_data["disco_prediction"] = {
            "predicted_accuracy": disco_results["predicted_accuracies"][0],
            "sampling_name": disco_results["sampling_name"],
            "number_item": disco_results["number_item"],
            "fitted_model_type": disco_results["fitted_model_type"],
            "pca": disco_results["pca"],
        }

    summary_path = output_dir / SUMMARY_FILENAME
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print(f"Full results saved to: {logger.output_dir}")


if __name__ == "__main__":
    main()

"""Core model scorer abstractions for likelihood-based evaluation.

This module provides the base `ModelScorer` class for computing token-level
scores (log-likelihoods) from language models. While `ModelAdapter` handles
text generation (``chat``, ``generate``), ``ModelScorer`` handles scoring by
computing how likely a model considers a given continuation.

See `maseval.interface.inference` for concrete implementations.

Example:
    ```python
    from maseval.interface.inference import HuggingFaceModelScorer

    scorer = HuggingFaceModelScorer(
        model_id="meta-llama/Llama-2-7b-hf",
        device="cuda:0",
    )

    # Single pair
    ll = scorer.loglikelihood("The capital of France is", " Paris")

    # MCQ evaluation
    logprobs = scorer.loglikelihood_choices(
        "What is 2+2?\\nA) 3\\nB) 4\\nC) 5\\nD) 6\\nAnswer:",
        choices=["A", "B", "C", "D"],
    )
    best = ["A", "B", "C", "D"][logprobs.index(max(logprobs))]
    ```
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .config import ConfigurableMixin
from .tracing import TraceableMixin


class ModelScorer(ABC, TraceableMixin, ConfigurableMixin):
    """Abstract base class for model scorers.

    ``ModelScorer`` provides a consistent interface for computing token-level
    log-likelihoods from language models. All scorers implement the same
    methods, so you can swap providers without changing evaluation code.

    To use a scorer:

    1. Create an instance with provider-specific configuration
    2. Call ``loglikelihood()`` for single context-continuation pairs
    3. Call ``loglikelihood_batch()`` for efficient batch computation
    4. Call ``loglikelihood_choices()`` for MCQ evaluation

    Implementing a custom scorer:

    Subclass ``ModelScorer`` and implement:

    - ``model_id`` property: Return the model identifier string
    - ``_loglikelihood_impl()``: Score a single (context, continuation) pair

    Optionally override:

    - ``_loglikelihood_batch_impl()``: Optimised batch scoring
    - ``loglikelihood_choices()``: MCQ-specific optimisations (e.g. shared-context single-pass)
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize the model scorer.

        Args:
            seed: Seed for deterministic scoring. Passed to the underlying
                model if supported.
        """
        super().__init__()
        self._seed = seed
        self.logs: List[Dict[str, Any]] = []

    @property
    def seed(self) -> Optional[int]:
        """Seed for deterministic scoring, or None if unseeded."""
        return self._seed

    @property
    @abstractmethod
    def model_id(self) -> str:
        """The identifier for the underlying model.

        Returns:
            A string identifying the model (e.g., ``"meta-llama/Llama-2-7b-hf"``).
        """

    def loglikelihood(self, context: str, continuation: str) -> float:
        """Compute the log-likelihood of ``continuation`` given ``context``.

        Args:
            context: The conditioning text (prompt).
            continuation: The text whose likelihood is scored.

        Returns:
            Log-likelihood (negative float; higher = more likely).
        """
        start_time = time.time()
        try:
            result = self._loglikelihood_impl(context, continuation)
            duration = time.time() - start_time
            self.logs.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "type": "loglikelihood",
                    "duration_seconds": duration,
                    "status": "success",
                }
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            self.logs.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "type": "loglikelihood",
                    "duration_seconds": duration,
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )
            raise

    @abstractmethod
    def _loglikelihood_impl(self, context: str, continuation: str) -> float:
        """Internal implementation for single-pair scoring.

        Subclasses must implement this. The base class handles timing
        and error logging.

        Args:
            context: The conditioning text.
            continuation: The text to score.

        Returns:
            Log-likelihood of the continuation.
        """

    def loglikelihood_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Compute log-likelihoods for a batch of (context, continuation) pairs.

        Override ``_loglikelihood_batch_impl`` for provider-specific batching
        optimisations. The default loops over ``_loglikelihood_impl``.

        Args:
            pairs: List of (context, continuation) tuples.

        Returns:
            List of log-likelihoods, one per pair.
        """
        start_time = time.time()
        try:
            results = self._loglikelihood_batch_impl(pairs)
            duration = time.time() - start_time
            self.logs.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "type": "loglikelihood_batch",
                    "batch_size": len(pairs),
                    "duration_seconds": duration,
                    "status": "success",
                }
            )
            return results
        except Exception as e:
            duration = time.time() - start_time
            self.logs.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "type": "loglikelihood_batch",
                    "batch_size": len(pairs),
                    "duration_seconds": duration,
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )
            raise

    def _loglikelihood_batch_impl(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Default batch implementation — loops over ``_loglikelihood_impl``.

        Override in subclasses for provider-specific batching.

        Args:
            pairs: List of (context, continuation) tuples.

        Returns:
            List of log-likelihoods.
        """
        return [self._loglikelihood_impl(ctx, cont) for ctx, cont in pairs]

    def loglikelihood_choices(
        self,
        context: str,
        choices: List[str],
        delimiter: str = " ",
    ) -> List[float]:
        """Compute log-likelihoods for multiple-choice continuations.

        Convenience method for MCQ evaluation. Each choice is prepended with
        ``delimiter`` before scoring (e.g. ``" A"``, ``" B"``).

        Subclasses may override this for optimised shared-context scoring
        (e.g. single forward pass for single-token choices).

        Args:
            context: The question/prompt text.
            choices: Answer choice strings (e.g. ``["A", "B", "C", "D"]``).
            delimiter: String prepended to each choice (default ``" "``).

        Returns:
            List of log-likelihoods, one per choice.
        """
        pairs = [(context, f"{delimiter}{c}") for c in choices]
        return self.loglikelihood_batch(pairs)

    def gather_traces(self) -> Dict[str, Any]:
        """Gather execution traces from this scorer.

        Output fields:

        - ``type`` - Component class name
        - ``gathered_at`` - ISO timestamp
        - ``model_id`` - Model identifier
        - ``total_calls`` - Number of scoring calls
        - ``successful_calls`` - Number of successful calls
        - ``failed_calls`` - Number of failed calls
        - ``total_duration_seconds`` - Total time spent in calls
        - ``logs`` - List of individual call records

        Returns:
            Dictionary containing scorer execution traces.
        """
        total_calls = len(self.logs)
        successful_calls = sum(1 for call in self.logs if call["status"] == "success")
        failed_calls = total_calls - successful_calls
        total_duration = sum(call["duration_seconds"] for call in self.logs)

        return {
            **super().gather_traces(),
            "model_id": self.model_id,
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "total_duration_seconds": total_duration,
            "logs": self.logs,
        }

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration from this scorer.

        Output fields:

        - ``type`` - Component class name
        - ``gathered_at`` - ISO timestamp
        - ``model_id`` - Model identifier
        - ``scorer_type`` - The specific scorer class name
        - ``seed`` - Seed for deterministic scoring, or None if unseeded

        Returns:
            Dictionary containing scorer configuration.
        """
        return {
            **super().gather_config(),
            "model_id": self.model_id,
            "scorer_type": type(self).__name__,
            "seed": self._seed,
        }

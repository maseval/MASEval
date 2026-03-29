"""HuggingFace model scorer for log-likelihood evaluation.

Wraps a raw HuggingFace ``AutoModelForCausalLM`` (not a pipeline) and
exposes ``loglikelihood()`` for scoring context-continuation pairs. Designed
for MCQ-style evaluation where the best answer is chosen by highest
log-likelihood.

For text generation (``chat()``, ``generate()``), see
``HuggingFacePipelineModelAdapter`` in ``maseval.interface.inference.huggingface``.

Requires transformers and torch:
    pip install maseval[transformers]

Example:
    ```python
    from maseval.interface.inference import HuggingFaceModelScorer

    scorer = HuggingFaceModelScorer(
        model_id="meta-llama/Llama-2-7b-hf",
        device="cuda:0",
    )

    # Score a single continuation
    ll = scorer.loglikelihood("The capital of France is", " Paris")

    # MCQ: pick the most likely answer
    logprobs = scorer.loglikelihood_choices(
        context="What is 2+2? Answer:",
        choices=["A", "B", "C", "D"],
    )
    best_idx = logprobs.index(max(logprobs))
    ```
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from maseval.core.scorer import ModelScorer


class HuggingFaceModelScorer(ModelScorer):
    """Log-likelihood scorer backed by a HuggingFace causal language model.

    Loads the model lazily on first use. Supports:

    - Single-token optimisation: when all continuations map to a single token,
      one forward pass scores every choice.
    - Multi-token fallback: separate forward pass per continuation.
    - ``loglikelihood_choices()`` override that picks the optimal path
      automatically.

    The tokenisation strategy matches ``lm-evaluation-harness``: context and
    continuation are encoded separately, then concatenated to handle
    tokenisation-boundary effects correctly.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda:0",
        trust_remote_code: bool = True,
        seed: Optional[int] = None,
    ):
        """Initialize HuggingFace model scorer.

        Args:
            model_id: HuggingFace model identifier
                (e.g. ``"meta-llama/Llama-2-7b-hf"``).
            device: Torch device string (e.g. ``"cuda:0"``, ``"cpu"``).
            trust_remote_code: Trust remote code when loading the model.
            seed: Seed for deterministic scoring.
        """
        super().__init__(seed=seed)
        self._model_id = model_id
        self._device = device
        self._trust_remote_code = trust_remote_code
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def model_id(self) -> str:
        return self._model_id

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> Tuple[Any, Any]:
        """Lazy-load the model and tokenizer.

        Returns:
            Tuple of (model, tokenizer).
        """
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_id,
                trust_remote_code=self._trust_remote_code,
            )
            self._tokenizer.padding_side = "left"
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                trust_remote_code=self._trust_remote_code,
                torch_dtype="auto",
            )
            self._model = self._model.to(self._device)
            self._model.eval()

        return self._model, self._tokenizer

    # ------------------------------------------------------------------
    # Tokenisation helpers (matches lm-evaluation-harness)
    # ------------------------------------------------------------------

    def _encode_pair(self, context: str, continuation: str) -> Tuple[List[int], List[int]]:
        """Encode a context-continuation pair like lm-evaluation-harness.

        1. Encode ``whole = context + continuation``
        2. Encode ``context`` alone
        3. ``continuation_enc = whole[len(context_enc):]``

        Args:
            context: The context/prompt string.
            continuation: The continuation string.

        Returns:
            Tuple of (context_enc, continuation_enc) token lists.
        """
        _, tokenizer = self._load_model()

        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = tokenizer.encode(context + continuation, add_special_tokens=True)
        context_enc = tokenizer.encode(context, add_special_tokens=True)

        continuation_enc = whole_enc[len(context_enc) :]
        return context_enc, continuation_enc

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def _loglikelihood_impl(self, context: str, continuation: str) -> float:
        """Score a single (context, continuation) pair.

        Uses ``_encode_pair`` for correct tokenisation, then computes the
        sum of per-token log-probabilities over the continuation.
        """
        import torch

        model, _ = self._load_model()

        context_enc, continuation_enc = self._encode_pair(context, continuation)
        full_sequence = context_enc + continuation_enc
        input_tokens = full_sequence[:-1]

        input_ids = torch.tensor([input_tokens], dtype=torch.long, device=self._device)

        with torch.no_grad():
            logits = model(input_ids).logits[0]
            inplen = len(input_tokens)
            contlen = len(continuation_enc)
            assert inplen >= contlen, (
                f"Context tokens ({inplen}) fewer than continuation tokens ({contlen}). "
                f"Tokenisation produced an unexpected result for context={context!r}, continuation={continuation!r}"
            )
            selected = logits[inplen - contlen : inplen]
            log_probs = torch.nn.functional.log_softmax(selected, dim=-1)

            total = 0.0
            for i, token_id in enumerate(continuation_enc):
                total += log_probs[i, token_id].item()

        return total

    # ------------------------------------------------------------------
    # MCQ optimisation
    # ------------------------------------------------------------------

    def loglikelihood_choices(
        self,
        context: str,
        choices: List[str],
        delimiter: str = " ",
    ) -> List[float]:
        """Score multiple-choice continuations with shared-context optimisation.

        When every ``delimiter + choice`` maps to a single continuation token,
        all choices are scored in **one** forward pass. Otherwise falls back to
        per-choice scoring via ``_loglikelihood_impl``.

        Args:
            context: The question/prompt text.
            choices: Answer choice strings (e.g. ``["A", "B", "C", "D"]``).
            delimiter: String prepended to each choice (default ``" "``).

        Returns:
            List of log-likelihoods, one per choice.
        """
        model, _ = self._load_model()

        continuations = [f"{delimiter}{c}" for c in choices]
        encoded_continuations = [self._encode_pair(context, cont) for cont in continuations]

        all_single_token = all(len(cont_enc) == 1 for _, cont_enc in encoded_continuations)

        if all_single_token:
            return self._score_single_token(context, choices, delimiter, encoded_continuations)

        return [self._loglikelihood_impl(context, cont) for cont in continuations]

    def _score_single_token(
        self,
        context: str,
        choices: List[str],
        delimiter: str,
        encoded_continuations: List[Tuple[List[int], List[int]]],
    ) -> List[float]:
        """One-forward-pass scoring for single-token continuations."""
        import torch

        model, _ = self._load_model()

        context_enc, first_cont_enc = encoded_continuations[0]
        full_sequence = context_enc + first_cont_enc
        input_tokens = full_sequence[:-1]

        input_ids = torch.tensor([input_tokens], dtype=torch.long, device=self._device)

        with torch.no_grad():
            logits = model(input_ids).logits[0]
            inplen = len(input_tokens)
            contlen = len(first_cont_enc)
            assert inplen >= contlen, (
                f"Context tokens ({inplen}) fewer than continuation tokens ({contlen}). Tokenisation produced an unexpected result."
            )
            selected_logits = logits[inplen - contlen : inplen]
            log_probs = torch.nn.functional.log_softmax(selected_logits, dim=-1)

            logprobs: List[float] = []
            for _, cont_enc in encoded_continuations:
                total = 0.0
                for i, token_id in enumerate(cont_enc):
                    total += log_probs[i, token_id].item()
                logprobs.append(total)

        return logprobs

    # ------------------------------------------------------------------
    # Tracing
    # ------------------------------------------------------------------

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration including device and model settings.

        Returns:
            Dictionary containing scorer configuration.
        """
        return {
            **super().gather_config(),
            "device": self._device,
            "trust_remote_code": self._trust_remote_code,
        }

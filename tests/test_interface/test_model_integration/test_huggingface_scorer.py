"""Unit tests for HuggingFaceModelScorer (mocked transformers + torch).

Tests model loading, tokenisation, log-likelihood computation, and the
single-token optimisation path without requiring a GPU or real model weights.
"""

from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.interface


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_mock_tokenizer(*, pad_token=None, eos_token="</s>"):
    """Create a mock tokenizer that returns predictable token IDs."""
    tok = MagicMock()
    tok.padding_side = "right"  # will be overwritten to "left"
    tok.pad_token = pad_token
    tok.eos_token = eos_token

    def _encode(text, add_special_tokens=True):
        # Deterministic: each character → its ord value.
        # This is intentionally simple so tests can predict the split.
        return [ord(c) for c in text]

    tok.encode = _encode
    return tok


class _FakeModelOutput:
    """Simple container for model output with logits attribute."""

    def __init__(self, logits):
        self.logits = logits


class _FakeCausalLM:
    """Minimal fake causal LM that returns uniform logits."""

    def __init__(self, vocab_size=256):
        self._vocab_size = vocab_size
        self.call_count = 0

    def to(self, device):
        return self

    def eval(self):
        pass

    def __call__(self, input_ids):
        self.call_count += 1
        seq_len = input_ids.shape[1]
        logits = torch.zeros(1, seq_len, self._vocab_size)
        return _FakeModelOutput(logits)

    def reset_mock(self):
        self.call_count = 0


def _make_mock_model(vocab_size=256):
    """Create a fake model whose forward pass returns uniform logits."""
    return _FakeCausalLM(vocab_size=vocab_size)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tokenizer():
    return _make_mock_tokenizer()


@pytest.fixture
def mock_model():
    return _make_mock_model()


@pytest.fixture
def scorer(mock_model, mock_tokenizer):
    """HuggingFaceModelScorer with mocked transformers — model pre-injected."""
    from maseval.interface.inference.huggingface_scorer import HuggingFaceModelScorer

    s = HuggingFaceModelScorer(model_id="test-model", device="cpu")
    # Bypass lazy loading by injecting mocks directly
    s._model = mock_model
    s._tokenizer = mock_tokenizer
    mock_tokenizer.padding_side = "left"
    yield s


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    def test_model_id_property(self):
        from maseval.interface.inference.huggingface_scorer import HuggingFaceModelScorer

        s = HuggingFaceModelScorer(model_id="my/model", device="cpu")
        assert s.model_id == "my/model"

    def test_lazy_loading(self):
        from maseval.interface.inference.huggingface_scorer import HuggingFaceModelScorer

        s = HuggingFaceModelScorer(model_id="my/model", device="cpu")
        assert s._model is None
        assert s._tokenizer is None

    def test_gather_config(self, scorer):
        config = scorer.gather_config()
        assert config["model_id"] == "test-model"
        assert config["device"] == "cpu"
        assert "trust_remote_code" in config


# ---------------------------------------------------------------------------
# TestLoadModel
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_loads_on_first_call(self, mock_model, mock_tokenizer):
        with patch.dict("sys.modules", {"transformers": MagicMock()}) as _:
            import sys

            transformers_mock = sys.modules["transformers"]
            transformers_mock.AutoModelForCausalLM.from_pretrained.return_value = mock_model
            transformers_mock.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

            from maseval.interface.inference.huggingface_scorer import HuggingFaceModelScorer

            s = HuggingFaceModelScorer(model_id="test-model", device="cpu")
            s._load_model()
            transformers_mock.AutoModelForCausalLM.from_pretrained.assert_called_once()
            transformers_mock.AutoTokenizer.from_pretrained.assert_called_once()

    def test_caches_model(self, mock_model, mock_tokenizer):
        with patch.dict("sys.modules", {"transformers": MagicMock()}) as _:
            import sys

            transformers_mock = sys.modules["transformers"]
            transformers_mock.AutoModelForCausalLM.from_pretrained.return_value = mock_model
            transformers_mock.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

            from maseval.interface.inference.huggingface_scorer import HuggingFaceModelScorer

            s = HuggingFaceModelScorer(model_id="test-model", device="cpu")
            s._load_model()
            s._load_model()
            assert transformers_mock.AutoModelForCausalLM.from_pretrained.call_count == 1

    def test_sets_padding_left(self, scorer, mock_tokenizer):
        assert mock_tokenizer.padding_side == "left"

    def test_sets_pad_token_from_eos(self, mock_model):
        tok = _make_mock_tokenizer(pad_token=None, eos_token="<eos>")
        with patch.dict("sys.modules", {"transformers": MagicMock()}) as _:
            import sys

            transformers_mock = sys.modules["transformers"]
            transformers_mock.AutoModelForCausalLM.from_pretrained.return_value = mock_model
            transformers_mock.AutoTokenizer.from_pretrained.return_value = tok

            from maseval.interface.inference.huggingface_scorer import HuggingFaceModelScorer

            s = HuggingFaceModelScorer(model_id="test-model", device="cpu")
            s._load_model()
            assert tok.pad_token == "<eos>"


# ---------------------------------------------------------------------------
# TestEncodePair
# ---------------------------------------------------------------------------


class TestEncodePair:
    def test_basic_split(self, scorer):
        ctx_enc, cont_enc = scorer._encode_pair("abc", "de")
        # "abcde" → [97,98,99,100,101], "abc" → [97,98,99]
        assert ctx_enc == [97, 98, 99]
        assert cont_enc == [100, 101]

    def test_trailing_spaces_transfer(self, scorer):
        ctx_enc, cont_enc = scorer._encode_pair("abc ", "de")
        # Space transfers: context becomes "abc", continuation becomes " de"
        # "abc de" → [97,98,99,32,100,101], "abc" → [97,98,99]
        assert ctx_enc == [97, 98, 99]
        assert cont_enc == [32, 100, 101]


# ---------------------------------------------------------------------------
# TestLoglikelihood
# ---------------------------------------------------------------------------


class TestLoglikelihood:
    def test_computes_logprob_sum(self, scorer):
        """With uniform logits (vocab_size=256), log_softmax = -log(256) per token."""
        import math

        result = scorer.loglikelihood("ab", "c")
        # Continuation "c" is 1 token → expected = -log(256) ≈ -5.545
        expected = -math.log(256)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_logged_via_public_api(self, scorer):
        scorer.loglikelihood("hello", " world")
        assert len(scorer.logs) == 1
        assert scorer.logs[0]["status"] == "success"


# ---------------------------------------------------------------------------
# TestLoglikelihoodChoices
# ---------------------------------------------------------------------------


class TestLoglikelihoodChoices:
    def test_single_token_path(self, mock_model):
        """When all continuations are 1 token, model is called once (single-token optimisation)."""
        from maseval.interface.inference.huggingface_scorer import HuggingFaceModelScorer

        # Use a tokenizer where " A", " B", " C", " D" each map to a single
        # token beyond the context, so _encode_pair yields 1-token continuations.
        tok = MagicMock()
        tok.padding_side = "left"
        tok.pad_token = "<pad>"
        tok.eos_token = "</s>"
        ctx_tokens = [1, 2, 3]

        def _single_tok_encode(text, add_special_tokens=True):
            # Context alone → [1, 2, 3]
            # Context + " X" → [1, 2, 3, <token_for_X>]
            if text == "ctx":
                return ctx_tokens
            # Anything longer than ctx is ctx + one extra token
            return ctx_tokens + [10 + len(text)]

        tok.encode = _single_tok_encode

        s = HuggingFaceModelScorer(model_id="test", device="cpu")
        s._model = _FakeCausalLM(vocab_size=256)
        s._tokenizer = tok

        s._model.reset_mock()
        results = s.loglikelihood_choices("ctx", ["A", "B", "C", "D"])
        assert len(results) == 4
        assert all(isinstance(r, float) for r in results)
        # Single-token path: one forward call for the shared context
        assert s._model.call_count == 1

    def test_multi_token_fallback(self, scorer, mock_model):
        """When continuations have different lengths, falls back to per-choice scoring."""
        original_encode = scorer._tokenizer.encode

        def _varied_encode(text, add_special_tokens=True):
            base = original_encode(text, add_special_tokens=add_special_tokens)
            # Add extra token for text containing "long" to create length mismatch
            if "long" in text:
                return base + [50]
            return base

        scorer._tokenizer.encode = _varied_encode
        mock_model.reset_mock()
        results = scorer.loglikelihood_choices("context", ["A", "long answer"])
        assert len(results) == 2
        # Multi-token path: one forward call per choice
        assert mock_model.call_count == 2

    def test_returns_correct_shape(self, scorer):
        results = scorer.loglikelihood_choices("ctx", ["A", "B"])
        assert len(results) == 2
        assert all(isinstance(r, float) for r in results)

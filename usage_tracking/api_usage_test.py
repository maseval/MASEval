"""
Test script that calls GPT-5 mini, Claude Haiku 4.5, and Gemini 3 Flash in three
conditions each — (1) native client, (2) LiteLLM, (3) LiteLLM via OpenRouter —
plus Qwen 3 via LiteLLM+OpenRouter. Saves full response dicts to JSON for
usage/cost analysis.
"""

import json
import os
import time
from pathlib import Path

import anthropic
import litellm
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

# LiteLLM reads OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY from env.
# For Gemini it expects GEMINI_API_KEY, so alias it.
os.environ.setdefault("GEMINI_API_KEY", GOOGLE_API_KEY)

PROMPT = "Hello, World!"
MAX_TOKENS = 64
TOTAL = 10

results = {}


def step(n: int, label: str):
    print(f"{n}/{TOTAL}  {label} ...")


# =========================================================================== #
#  CONDITION 1 — Native SDKs (direct)
# =========================================================================== #

# -- 1. GPT-5 mini (OpenAI) ------------------------------------------------ #
step(1, "GPT-5 mini — direct (OpenAI SDK)")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
resp = openai_client.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": PROMPT}],
    max_completion_tokens=MAX_TOKENS,
)
results["direct__openai__gpt5_mini"] = resp.model_dump()
print(f"       done — {resp.usage.total_tokens} tokens")

# -- 2. Claude Haiku 4.5 (Anthropic) --------------------------------------- #
step(2, "Claude Haiku 4.5 — direct (Anthropic SDK)")
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
resp = anthropic_client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=MAX_TOKENS,
    messages=[{"role": "user", "content": PROMPT}],
)
results["direct__anthropic__claude_haiku"] = resp.model_dump()
print(f"       done — {resp.usage.input_tokens + resp.usage.output_tokens} tokens")

# -- 3. Gemini 3 Flash (Google) -------------------------------------------- #
step(3, "Gemini 3 Flash — direct (Google GenAI SDK)")
google_client = genai.Client(api_key=GOOGLE_API_KEY)
resp = google_client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=PROMPT,
    config=types.GenerateContentConfig(max_output_tokens=MAX_TOKENS),
)
results["direct__google__gemini3_flash"] = resp.model_dump(mode="json")
total = resp.usage_metadata.total_token_count if resp.usage_metadata else "n/a"
print(f"       done — {total} tokens")


# =========================================================================== #
#  CONDITION 2 — LiteLLM (direct to providers)
# =========================================================================== #

litellm_direct_models = {
    "litellm__openai__gpt5_mini": "gpt-5-mini",
    "litellm__anthropic__claude_haiku": "claude-haiku-4-5-20251001",
    "litellm__google__gemini3_flash": "gemini/gemini-3-flash-preview",
}

for i, (label, model) in enumerate(litellm_direct_models.items(), start=4):
    step(i, f"{model} — LiteLLM (direct)")
    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=MAX_TOKENS,
    )
    results[label] = resp.model_dump()
    usage_total = resp.usage.total_tokens if resp.usage else "n/a"
    print(f"       done — {usage_total} tokens")


# =========================================================================== #
#  CONDITION 3 — LiteLLM via OpenRouter  (+Qwen)
# =========================================================================== #


def fetch_openrouter_generation(gen_id: str) -> dict | None:
    """Query OpenRouter's generation endpoint for cost metadata."""
    time.sleep(2)  # brief wait for metadata to be available
    r = requests.get(
        f"https://openrouter.ai/api/v1/generation?id={gen_id}",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
    )
    if r.status_code == 200:
        return r.json()
    return None


litellm_openrouter_models = {
    "openrouter__openai__gpt5_mini": "openrouter/openai/gpt-5-mini",
    "openrouter__anthropic__claude_haiku": "openrouter/anthropic/claude-haiku-4-5",
    "openrouter__google__gemini3_flash": "openrouter/google/gemini-3-flash-preview",
    "openrouter__qwen__qwen3_30b": "openrouter/qwen/qwen3-30b-a3b",
}

for i, (label, model) in enumerate(litellm_openrouter_models.items(), start=7):
    step(i, f"{model} — LiteLLM (OpenRouter)")
    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=MAX_TOKENS,
    )
    result = resp.model_dump()

    # Fetch OpenRouter generation metadata (cost, native tokens, etc.)
    gen_meta = fetch_openrouter_generation(resp.id)
    if gen_meta:
        result["_openrouter_generation"] = gen_meta

    results[label] = result
    usage_total = resp.usage.total_tokens if resp.usage else "n/a"
    print(f"       done — {usage_total} tokens")


# =========================================================================== #
#  Save results
# =========================================================================== #
out_path = Path(__file__).parent / "api_usage_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {out_path}")

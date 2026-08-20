"""DAF Benchmark Example with Ollama Support.

This example demonstrates how to evaluate agent systems using MASEval,
with support for Ollama (local LLMs) via LiteLLM.

The example includes:
- Custom benchmark with latency, cost, and accuracy tracking
- Ollama integration via LiteLLM (OpenAI-compatible API)
- DAF adapter integration (works with or without DAF backend)

Usage:
    # Run with Ollama (default)
    uv run python examples/daf_benchmark/daf_benchmark.py

    # Run with a specific Ollama model
    uv run python examples/daf_benchmark/daf_benchmark.py --model "ollama/llama3.2"

    # Run with OpenAI (for comparison)
    uv run python examples/daf_benchmark/daf_benchmark.py --model "gpt-4o-mini"

    # Run with DAF backend (if running)
    uv run python examples/daf_benchmark/daf_benchmark.py --use-daf --daf-url "http://localhost:8012"

Environment Variables:
    OLLAMA_API_BASE: Ollama API base URL (default: http://localhost:11434/v1)
    OLLAMA_API_KEY: Ollama API key (default: ollama, usually not required)
    OPENAI_API_KEY: OpenAI API key (if using OpenAI models)
    DAF_API_KEY: DAF backend API key (if using DAF)
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv

# MASEval imports
from maseval import AgentAdapter, Benchmark, Environment, Evaluator, Task, User
from maseval.core.callbacks.result_logger import FileResultLogger
from maseval.core.usage import TokenUsage
from maseval.interface.inference import LiteLLMModelAdapter

# Load environment variables from .env file
load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

# Ollama Cloud configuration
DEFAULT_OLLAMA_BASE = os.getenv("OLLAMA_API_BASE", "https://ollama.com")
DEFAULT_OLLAMA_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
DEFAULT_OLLAMA_MODEL = "ollama/gemma4:31b"

# OpenRouter configuration
DEFAULT_OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Model presets for each provider
OLLAMA_MODELS = {
    "qwen": "ollama/qwen3.5:397b",
    "glm": "ollama/glm-5.2",
    "minimax": "ollama/minimax-m3",
    "kimi": "ollama/kimi-k2.6",
    "gemma": "ollama/gemma4:31b",
}

OPENROUTER_MODELS = {
    # Anthropic models
    "sonnet-5": "anthropic/claude-sonnet-5",
    "opus-5": "anthropic/claude-opus-5",
    # OpenAI models
    "sol-5.6": "openai/gpt-5.6-sol",
    "terra-5.6": "openai/gpt-5.6-terra",
    # Chinese models (via OpenRouter for pricing)
    "qwen-or": "qwen/qwen3.5-397b-a17b",
    "qwen-3.7-plus": "qwen/qwen3.7-plus",
    "qwen-3.7-max": "qwen/qwen3.7-max",
    "glm-or": "z-ai/glm-5.2",
    "minimax-or": "minimax/minimax-m3",
    "kimi-or": "moonshotai/kimi-k2.6",
    # Other frontier models
    "grok-4.6": "x-ai/grok-4.6",
    "gemini-3.7-flash": "google/gemini-3.7-flash",
}

# DAF configuration
DEFAULT_DAF_URL = os.getenv("DAF_BASE_URL", "http://localhost:8012")
DEFAULT_DAF_KEY = os.getenv("DAF_API_KEY", "")

# Output directory
DEFAULT_OUTPUT_DIR = "./results"


# =============================================================================
# Task Definitions (Easy, Medium, Hard)
# =============================================================================


EASY_TASKS = [
    {
        "id": "factual_001",
        "query": "What is the capital of France? Answer in one word.",
        "expected": "Paris",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "id": "reasoning_001",
        "query": "If a train travels at 60 mph for 2.5 hours, how far does it travel? Show your calculation.",
        "expected": "150",
        "category": "reasoning",
        "difficulty": "easy",
    },
    {
        "id": "reasoning_002",
        "query": "A store has a 20% discount on a $50 item. What is the final price?",
        "expected": "40",
        "category": "reasoning",
        "difficulty": "easy",
    },
    {
        "id": "coding_001",
        "query": "Write a Python function that checks if a string is a palindrome. Just provide the function code.",
        "expected": "def",
        "category": "coding",
        "difficulty": "easy",
    },
    {
        "id": "explanation_001",
        "query": "Summarize the concept of 'recursion' in programming in 2-3 sentences.",
        "expected": ["recursion", "function", "itself"],
        "category": "explanation",
        "difficulty": "easy",
    },
]


MEDIUM_TASKS = [
    {
        "id": "reasoning_med_001",
        "query": """A company has 3 departments. Department A has 45 employees with average salary $75,000.
Department B has 30 employees with average salary $85,000. Department C has 25 employees with average salary $65,000.
What is the overall average salary across all employees? Show your step-by-step calculation.""",
        "expected": "75500",
        "category": "reasoning",
        "difficulty": "medium",
    },
    {
        "id": "coding_med_001",
        "query": """Write a Python function called `find_duplicates` that takes a list of integers and returns a sorted list
of all elements that appear more than once. Include type hints and handle edge cases like empty lists.
Example: find_duplicates([1, 2, 3, 2, 1, 4]) should return [1, 2]""",
        "expected": ["def", "List", "return"],
        "category": "coding",
        "difficulty": "medium",
    },
    {
        "id": "analysis_med_001",
        "query": """Compare and contrast REST and GraphQL APIs. Discuss at least 3 advantages and 3 disadvantages of each approach.
When would you choose one over the other? Provide specific use case examples.""",
        "expected": ["REST", "GraphQL", "advantage", "disadvantage", "use case"],
        "category": "analysis",
        "difficulty": "medium",
    },
    {
        "id": "reasoning_med_002",
        "query": """A project has 5 tasks with dependencies:
- Task A: 3 days, no dependencies
- Task B: 5 days, depends on A
- Task C: 2 days, depends on A
- Task D: 4 days, depends on B and C
- Task E: 1 day, depends on D

What is the minimum total project duration? Show the critical path.""",
        "expected": "13",
        "category": "reasoning",
        "difficulty": "medium",
    },
    {
        "id": "factual_med_001",
        "query": """Explain the difference between TCP and UDP protocols. Include:
1. Connection model
2. Reliability guarantees
3. Use cases for each
4. Performance characteristics""",
        "expected": ["TCP", "UDP", "connection", "reliable", "streaming"],
        "category": "explanation",
        "difficulty": "medium",
    },
]


HARD_TASKS = [
    {
        "id": "coding_hard_001",
        "query": """Implement a thread-safe LRU (Least Recently Used) cache in Python with the following requirements:
- Support get(key) and put(key, value) operations
- O(1) time complexity for both operations
- Maximum capacity parameter
- Handle concurrent access safely
- Include proper docstrings and type hints
- Include a simple test demonstrating thread safety

Provide the complete implementation.""",
        "expected": ["class", "OrderedDict", "Lock", "def get", "def put"],
        "category": "coding",
        "difficulty": "hard",
    },
    {
        "id": "reasoning_hard_001",
        "query": """A hospital has 3 doctors and 5 patients waiting. Each patient has a priority score and estimated treatment time:
- Patient 1: priority=8, time=30min
- Patient 2: priority=5, time=45min
- Patient 3: priority=9, time=20min
- Patient 4: priority=3, time=60min
- Patient 5: priority=7, time=35min

Constraints:
- Each doctor can see one patient at a time
- Higher priority patients should be seen first
- Total available time is 2 hours
- No doctor can work overtime

Create an optimal schedule that maximizes the total priority score of treated patients. Show your reasoning and the final assignment.""",
        "expected": ["Patient 3", "Patient 1", "priority"],
        "category": "reasoning",
        "difficulty": "hard",
    },
    {
        "id": "analysis_hard_001",
        "query": """Design a distributed rate limiter for a microservices architecture with the following requirements:
- Handle 10,000 requests per second across 50 service instances
- Support both sliding window and token bucket algorithms
- Handle network partitions gracefully
- Provide sub-millisecond latency for rate limit checks
- Support different rate limits per API endpoint and per user

Provide:
1. Architecture diagram description
2. Data structure choices
3. Consistency model
4. Failure handling strategy
5. Pseudocode for the core algorithm""",
        "expected": ["distributed", "sliding window", "token bucket", "partition", "consistent"],
        "category": "analysis",
        "difficulty": "hard",
    },
    {
        "id": "coding_hard_002",
        "query": """Write a Python implementation of a simple blockchain with the following features:
- Block class with index, timestamp, data, previous_hash, nonce, and hash
- Proof of Work mining with adjustable difficulty
- Chain validation
- Transaction support (sender, receiver, amount)
- Merkle tree for transaction verification

Include at least 3 test cases demonstrating: mining, chain validation, and tamper detection.
Provide complete, runnable code with type hints.""",
        "expected": ["class Block", "class Blockchain", "def mine", "hash", "previous_hash"],
        "category": "coding",
        "difficulty": "hard",
    },
    {
        "id": "reasoning_hard_002",
        "query": """Solve this logic puzzle:

Five people (Alice, Bob, Carol, Dave, Eve) sit in a row of 5 chairs.
Clues:
1. Alice is not in chair 1 or 5
2. Bob is to the left of Carol
3. Dave is next to Eve
4. Carol is not in chair 3
5. Eve is in an even-numbered chair

Determine who sits in which chair. Show your step-by-step reasoning and the final arrangement.""",
        "expected": ["chair", "Alice", "Bob", "Carol", "Dave", "Eve"],
        "category": "reasoning",
        "difficulty": "hard",
    },
]


EXPERT_TASKS = [
    {
        "id": "debug_expert_001",
        "query": """Find and fix all bugs in this Python code. The code should implement a thread-safe counter with increment, decrement, and get operations, but it has multiple race conditions and logic errors:

```python
import threading
from typing import Optional

class Counter:
    def __init__(self, initial: int = 0):
        self.value = initial
        self.lock = threading.Lock
    
    def increment(self, amount: int = 1) -> int:
        if amount < 0:
            raise ValueError("Amount must be positive")
        self.value += amount
        return self.value
    
    def decrement(self, amount: int = 1) -> Optional[int]:
        with self.lock():
            if self.value - amount < 0:
                return None
            self.value =- amount
            return self.value
    
    def get(self) -> int:
        return self.value
    
    def reset(self):
        self.value == 0
```

Identify all bugs, explain why each is a problem, and provide the corrected implementation with proper thread safety.""",
        "expected": ["lock()", "-=", "==", "with", "acquire", "race condition"],
        "category": "coding",
        "difficulty": "expert",
    },
    {
        "id": "design_expert_001",
        "query": """Design a multi-agent system for automated code review with the following requirements:

1. **Agents**: Define at least 4 specialized agents (e.g., Security Reviewer, Style Checker, Logic Analyzer, Test Generator)
2. **Coordination**: Describe how agents coordinate (sequential, parallel, or hybrid)
3. **Communication**: Define message formats between agents
4. **Conflict Resolution**: How to handle conflicting recommendations
5. **Consensus**: How to reach a final review verdict
6. **Failure Handling**: What happens when an agent fails or times out

Provide:
- Architecture diagram description
- Agent role definitions with specific responsibilities
- Message flow sequence
- Pseudocode for the orchestration logic
- Example review output for a sample code snippet""",
        "expected": ["agent", "coordination", "message", "conflict", "consensus", "orchestrat"],
        "category": "analysis",
        "difficulty": "expert",
    },
    {
        "id": "algorithm_expert_001",
        "query": """Implement the A* (A-star) pathfinding algorithm in Python with the following requirements:

1. Support both 4-directional and 8-directional movement
2. Handle obstacles and weighted terrain
3. Implement at least 2 heuristic functions (Manhattan, Euclidean)
4. Support path reconstruction
5. Include visualization of the explored nodes
6. Handle edge cases (no path exists, start == goal)

Additionally:
- Analyze time and space complexity
- Compare with Dijkstra's algorithm
- Provide 3 test cases: simple maze, complex maze with weights, no-path scenario

Provide complete, runnable code with type hints and docstrings.""",
        "expected": ["class", "def", "heuristic", "priority queue", "path", "A*", "open_set", "closed_set"],
        "category": "coding",
        "difficulty": "expert",
    },
    {
        "id": "math_expert_001",
        "query": """Solve this multi-constraint optimization problem:

A logistics company has 4 warehouses (W1-W4) and 6 delivery destinations (D1-D6).

Supply at warehouses: W1=100, W2=150, W3=80, W4=120 units
Demand at destinations: D1=60, D2=90, D3=40, D4=80, D5=70, D6=50 units

Transportation costs per unit (rows=warehouses, cols=destinations):
     D1  D2  D3  D4  D5  D6
W1:   4   8   6   9   5   7
W2:   6   3   7   4   8   5
W3:   5   9   3   7   6   4
W4:   7   5   8   3   4   6

Constraints:
- Total supply must meet total demand (adjust if needed)
- No warehouse ships more than its supply
- Each destination must receive exactly its demand
- Maximum 2 warehouses can supply any single destination

Find the optimal shipping plan that minimizes total cost. Show your work using the transportation simplex method or Vogel's approximation method.""",
        "expected": ["supply", "demand", "cost", "optimal", "warehouse", "destination", "minimize"],
        "category": "reasoning",
        "difficulty": "expert",
    },
    {
        "id": "system_expert_001",
        "query": """Design a distributed consensus algorithm for a network of 100 nodes where:
- Up to 33 nodes may be Byzantine (malicious/faulty)
- Network is partially synchronous (bounded but unknown delay)
- Nodes must agree on a sequence of transactions
- Finality must be reached within 3 network rounds

Requirements:
1. Describe your consensus mechanism (PBFT, HotStuff, or novel hybrid)
2. Define message types and protocol phases
3. Prove safety and liveness properties informally
4. Analyze message complexity: O(n²) vs O(n) trade-offs
5. Handle view changes when leader is faulty
6. Describe checkpointing for state synchronization

Provide pseudocode for:
- Normal case operation
- View change protocol
- Checkpoint creation and verification""",
        "expected": ["consensus", "Byzantine", "BFT", "view change", "leader", "quorum", "safety", "liveness"],
        "category": "analysis",
        "difficulty": "expert",
    },
]


def create_sample_tasks(difficulty: str = "easy") -> List[Task]:
    """Create sample evaluation tasks at the specified difficulty level.

    Args:
        difficulty: One of 'easy', 'medium', 'hard', 'expert', or 'all'

    Returns:
        List of Task objects for evaluation
    """
    if difficulty == "all":
        tasks_data = EASY_TASKS + MEDIUM_TASKS + HARD_TASKS + EXPERT_TASKS
    elif difficulty == "expert":
        tasks_data = EXPERT_TASKS
    elif difficulty == "hard":
        tasks_data = HARD_TASKS
    elif difficulty == "medium":
        tasks_data = MEDIUM_TASKS
    else:
        tasks_data = EASY_TASKS

    tasks = []
    for data in tasks_data:
        task = Task(
            query=data["query"],
            metadata={
                "task_id": data["id"],
                "category": data["category"],
                "difficulty": data["difficulty"],
                "expected": data["expected"],
            },
        )
        tasks.append(task)

    return tasks


# =============================================================================
# Tools for Agents
# =============================================================================


class Tool:
    """Base class for agent tools."""

    name: str = "base_tool"
    description: str = "Base tool"

    def execute(self, **kwargs) -> str:
        raise NotImplementedError


class CalculatorTool(Tool):
    """Calculator tool for math operations."""

    name = "calculator"
    description = "Evaluate a mathematical expression. Input: expression (str). Returns: result."

    def execute(self, expression: str, **kwargs) -> str:
        """Evaluate a math expression safely."""
        try:
            # Safe evaluation of basic math
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return "Error: Invalid characters in expression"
            result = eval(expression)  # noqa: S307
            return f"{result}"
        except Exception as e:
            return f"Error: {e}"


class CodeValidatorTool(Tool):
    """Tool to validate Python code syntax."""

    name = "validate_code"
    description = "Check if Python code has valid syntax. Input: code (str). Returns: validation result."

    def execute(self, code: str, **kwargs) -> str:
        """Validate Python code syntax."""
        try:
            compile(code, "<string>", "exec")
            return "Valid Python syntax"
        except SyntaxError as e:
            return f"Syntax error: {e.msg} at line {e.lineno}"


class KeywordExtractorTool(Tool):
    """Tool to extract keywords from text."""

    name = "extract_keywords"
    description = "Extract key terms from text. Input: text (str). Returns: comma-separated keywords."

    def execute(self, text: str, max_keywords: int = 5, **kwargs) -> str:
        """Extract keywords from text."""
        # Simple keyword extraction
        import re
        words = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]{3,}\b', text)
        # Filter common words
        stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'they', 'their'}
        keywords = [w for w in words if w.lower() not in stop_words]
        # Return unique keywords
        seen = set()
        unique = []
        for kw in keywords:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                unique.append(kw)
        return ", ".join(unique[:max_keywords])


# Tool registry
TOOLS = {
    "calculator": CalculatorTool(),
    "validate_code": CodeValidatorTool(),
    "extract_keywords": KeywordExtractorTool(),
}


def get_openai_tool_definitions() -> List[Dict[str, Any]]:
    """Return tool definitions in OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a mathematical expression and return the numeric result.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "A mathematical expression using +, -, *, /, (), and decimal numbers. E.g. '(3 + 5) * 2'",
                        },
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "validate_code",
                "description": "Check if Python code has valid syntax. Returns validation result.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python source code to validate.",
                        },
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_keywords",
                "description": "Extract key terms from text. Returns comma-separated keywords.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to extract keywords from.",
                        },
                        "max_keywords": {
                            "type": "integer",
                            "description": "Maximum number of keywords to return.",
                            "default": 5,
                        },
                    },
                    "required": ["text"],
                },
            },
        },
    ]


def format_tools_for_prompt() -> str:
    """Format available tools for system prompts (text fallback)."""
    lines = ["Available tools:"]
    for name, tool in TOOLS.items():
        lines.append(f"  - {name}: {tool.description}")
    lines.append("To use a tool, write: USE_TOOL(name, arg1=value1, arg2=value2)")
    return "\n".join(lines)


def parse_and_execute_tool(text: str) -> Optional[str]:
    """Parse and execute a tool call from text (fallback for models without function-calling)."""
    import re
    match = re.search(r'USE_TOOL\((\w+),\s*([^)]+)\)', text)
    if not match:
        return None
    tool_name = match.group(1)
    args_str = match.group(2)
    if tool_name not in TOOLS:
        return f"Error: Unknown tool '{tool_name}'"
    # Parse args
    kwargs = {}
    arg_pattern = re.compile(r"(\w+)=[\"']?([^,\"']+)[\"']?")
    for arg_match in arg_pattern.finditer(args_str):
        kwargs[arg_match.group(1)] = arg_match.group(2).strip()
    return TOOLS[tool_name].execute(**kwargs)


def execute_native_tool_call(tool_name: str, arguments_json: str) -> str:
    """Execute a tool call from native function-calling response.

    Args:
        tool_name: Name of the tool to execute
        arguments_json: JSON string of tool arguments

    Returns:
        Tool execution result as string
    """
    if tool_name not in TOOLS:
        return f"Error: Unknown tool '{tool_name}'"
    try:
        kwargs = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        return f"Error: Invalid JSON arguments: {arguments_json}"
    return TOOLS[tool_name].execute(**kwargs)


# =============================================================================
# Simple Agent (for testing without DAF)
# =============================================================================


class SimpleLLMAgent:
    """A simple LLM agent that uses a model adapter for generation.

    This is a minimal agent implementation for demonstration purposes.
    In practice, you would use DAF or another agent framework.
    """

    def __init__(self, model: LiteLLMModelAdapter, system_prompt: Optional[str] = None):
        self.model = model
        self.system_prompt = system_prompt or "You are a helpful assistant. Answer questions concisely and accurately."
        self._last_duration = 0.0
        self._last_usage: Optional[Dict[str, Any]] = None

    def run(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Run the agent on a query.

        Args:
            query: User query to process

        Returns:
            Tuple of (response, metrics)
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

        start_time = time.time()

        # Call the model
        response = self.model.chat(messages)

        self._last_duration = time.time() - start_time
        self._last_usage = response.usage

        # Build metrics
        metrics = {
            "duration_seconds": self._last_duration,
            "usage": response.usage or {},
            "model": response.model,
        }

        return response.content or "", metrics


class SimpleAgentAdapter(AgentAdapter):
    """MASEval adapter for SimpleLLMAgent."""

    def __init__(
        self,
        agent_instance: SimpleLLMAgent,
        name: str,
        callbacks: Optional[List[Any]] = None,
    ):
        super().__init__(agent_instance=agent_instance, name=name, callbacks=callbacks)
        self._last_response = ""
        self._last_metrics: Dict[str, Any] = {}

    def _run_agent(self, query: str) -> str:
        """Run the agent and return the response."""
        response, metrics = self.agent.run(query)
        self._last_response = response
        self._last_metrics = metrics
        return response

    def get_messages(self):
        """Return message history from last run."""
        from maseval import MessageHistory

        messages = [
            {"role": "system", "content": self.agent.system_prompt},
            {"role": "user", "content": "previous query"},  # Placeholder
            {"role": "assistant", "content": self._last_response},
        ]
        return MessageHistory(messages)

    def _gather_usage(self):
        """Gather usage from last run."""
        usage = self._last_metrics.get("usage", {})
        if not usage:
            return TokenUsage()

        return TokenUsage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    def gather_traces(self) -> Dict[str, Any]:
        """Gather traces including duration metrics."""
        base = super().gather_traces()
        return {
            **base,
            "duration_seconds": self._last_metrics.get("duration_seconds", 0),
            "model": self._last_metrics.get("model"),
            "usage": self._last_metrics.get("usage"),
        }


# =============================================================================
# Multi-Agent Team (local simulation without DAF backend)
# =============================================================================


class MultiAgentTeam:
    """A multi-agent team with Planner, Executor, and Reviewer roles.

    This simulates a DAF-style team locally for benchmarking when no DAF backend
    is available. Each agent has a specialized role:

    - **Planner**: Analyzes the task and creates an execution plan
    - **Executor**: Executes the plan, using tools when appropriate
    - **Reviewer**: Reviews the output for quality and completeness

    The team supports tool use (calculator, code validator) and tracks
    metrics per agent for detailed analysis.
    """

    def __init__(self, model: LiteLLMModelAdapter, use_tools: bool = True):
        """Initialize the multi-agent team.

        Args:
            model: The LLM model adapter to use for all agents
            use_tools: Whether to enable tool use for the Executor
        """
        self.model = model
        self.use_tools = use_tools
        self._messages: List[Dict[str, Any]] = []
        self._agent_metrics: Dict[str, Dict[str, Any]] = {}
        self._tools_used: List[str] = []
        self._total_duration = 0.0
        self._total_usage: Dict[str, Any] = {}

    def run(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Run the multi-agent team on a query.

        Flow: Planner -> Executor -> Reviewer -> Final Answer

        Args:
            query: The task/query to solve

        Returns:
            Tuple of (final_answer, metrics_dict)
        """
        self._messages = []
        self._agent_metrics = {}
        self._tools_used = []
        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0

        # === Phase 1: Planning ===
        plan, plan_metrics = self._run_planner(query)
        self._agent_metrics["planner"] = plan_metrics
        total_input_tokens += plan_metrics.get("usage", {}).get("input_tokens", 0)
        total_output_tokens += plan_metrics.get("usage", {}).get("output_tokens", 0)
        total_cost += plan_metrics.get("usage", {}).get("cost", 0.0) or 0.0

        # === Phase 2: Execution ===
        execution, exec_metrics = self._run_executor(query, plan)
        self._agent_metrics["executor"] = exec_metrics
        total_input_tokens += exec_metrics.get("usage", {}).get("input_tokens", 0)
        total_output_tokens += exec_metrics.get("usage", {}).get("output_tokens", 0)
        total_cost += exec_metrics.get("usage", {}).get("cost", 0.0) or 0.0

        # === Phase 3: Review ===
        review, review_metrics = self._run_reviewer(query, execution)
        self._agent_metrics["reviewer"] = review_metrics
        total_input_tokens += review_metrics.get("usage", {}).get("input_tokens", 0)
        total_output_tokens += review_metrics.get("usage", {}).get("output_tokens", 0)
        total_cost += review_metrics.get("usage", {}).get("cost", 0.0) or 0.0

        self._total_duration = time.time() - start_time

        # Final answer is the reviewed output
        final_answer = review

        usage_dict: Dict[str, Any] = {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }
        if total_cost > 0:
            usage_dict["cost"] = total_cost

        metrics = {
            "duration_seconds": self._total_duration,
            "usage": usage_dict,
            "model": self.model.model_id,
            "agents": self._agent_metrics,
            "tools_used": self._tools_used,
            "team_size": 3,
        }

        return final_answer, metrics

    def _run_planner(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Planner agent analyzes the task and creates a plan."""
        system_prompt = """You are a Planning Agent. Your job is to:
1. Analyze the task and identify what needs to be done
2. Break it into clear steps
3. Identify if any tools would help (calculator for math, validate_code for Python)

Respond with a brief execution plan in this format:
PLAN: <your plan in 2-4 bullet points>
TOOLS_NEEDED: <comma-separated tool names or 'none'>"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a plan for this task:\n{query}"},
        ]

        start = time.time()
        response = self.model.chat(messages)
        duration = time.time() - start

        self._messages.append({"role": "assistant", "content": f"[PLANNER]: {response.content}"})

        metrics = {
            "duration_seconds": duration,
            "usage": response.usage or {},
            "role": "planner",
        }
        return response.content or "", metrics

    def _run_executor(self, query: str, plan: str) -> Tuple[str, Dict[str, Any]]:
        """Executor agent carries out the plan, using native function-calling tools.

        Supports both native tool calling (OpenAI function-calling format) and
        text-based fallback (USE_TOOL syntax) for models that don't support it.

        Args:
            query: The original task query
            plan: The plan from the Planner agent

        Returns:
            Tuple of (response content, metrics dict)
        """
        system_prompt = """You are an Execution Agent. You receive a task and a plan from the Planner.
Your job is to execute the plan and produce a complete, accurate answer.

You have access to tools that can help you. Use them when appropriate:
- calculator: For evaluating mathematical expressions
- validate_code: For checking Python code syntax
- extract_keywords: For extracting key terms from text

Provide a complete, detailed final answer."""

        user_content = f"TASK: {query}\n\nPLAN: {plan}\n\nNow execute this plan and provide the complete answer."

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        start = time.time()
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        max_tool_rounds = 5  # Prevent infinite tool-call loops

        if self.use_tools:
            tool_defs = get_openai_tool_definitions()
            response = self.model.chat(messages, tools=tool_defs, tool_choice="auto")
        else:
            response = self.model.chat(messages)

        # Track usage from initial call
        if response.usage:
            total_input_tokens += response.usage.get("input_tokens", 0)
            total_output_tokens += response.usage.get("output_tokens", 0)
            total_cost += response.usage.get("cost", 0.0) or 0.0

        # Multi-round tool calling loop
        for round_num in range(max_tool_rounds):
            # Check for native tool calls
            if response.tool_calls:
                # Add assistant message with tool calls
                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": response.content or "",
                }
                assistant_msg["tool_calls"] = response.tool_calls
                messages.append(assistant_msg)

                # Execute each tool call and add results
                for tc in response.tool_calls:
                    fn_name = tc["function"]["name"]
                    fn_args = tc["function"]["arguments"]
                    self._tools_used.append(fn_name)

                    result = execute_native_tool_call(fn_name, fn_args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })

                # Get next response
                if self.use_tools:
                    response = self.model.chat(messages, tools=tool_defs, tool_choice="auto")
                else:
                    response = self.model.chat(messages)

                if response.usage:
                    total_input_tokens += response.usage.get("input_tokens", 0)
                    total_output_tokens += response.usage.get("output_tokens", 0)
                    total_cost += response.usage.get("cost", 0.0) or 0.0

            else:
                # No tool calls — check for text-based fallback
                content = response.content or ""
                tool_result = parse_and_execute_tool(content)
                if tool_result and round_num == 0:
                    tool_name_match = re.search(r'USE_TOOL\((\w+)', content)
                    if tool_name_match:
                        self._tools_used.append(tool_name_match.group(1))
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": f"Tool result: {tool_result}\n\nNow provide the final answer incorporating this result."})
                    response = self.model.chat(messages)
                    if response.usage:
                        total_input_tokens += response.usage.get("input_tokens", 0)
                        total_output_tokens += response.usage.get("output_tokens", 0)
                        total_cost += response.usage.get("cost", 0.0) or 0.0
                break

        content = response.content or ""
        duration = time.time() - start

        self._messages.append({"role": "assistant", "content": f"[EXECUTOR]: {content}"})

        metrics = {
            "duration_seconds": duration,
            "usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "cost": total_cost,
            },
            "role": "executor",
            "tool_rounds": round_num + 1 if self.use_tools else 0,
        }
        return content, metrics

    def _run_reviewer(self, query: str, execution: str) -> Tuple[str, Dict[str, Any]]:
        """Reviewer agent checks quality and produces final answer."""
        system_prompt = """You are a Review Agent. You receive a task and an executed answer.
Your job is to:
1. Verify the answer is complete and accurate
2. Fix any obvious errors
3. Provide the final, polished answer

Respond with the final answer only - do not include meta-commentary about the review."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"TASK: {query}\n\nEXECUTED ANSWER:\n{execution}\n\nReview and provide the final answer."},
        ]

        start = time.time()
        response = self.model.chat(messages)
        duration = time.time() - start

        self._messages.append({"role": "assistant", "content": f"[REVIEWER]: {response.content}"})

        metrics = {
            "duration_seconds": duration,
            "usage": response.usage or {},
            "role": "reviewer",
        }
        return response.content or execution, metrics

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get the full message trace from the team execution."""
        return self._messages


class MultiAgentTeamAdapter(AgentAdapter):
    """MASEval adapter for MultiAgentTeam."""

    def __init__(
        self,
        agent_instance: MultiAgentTeam,
        name: str = "multi_agent_team",
        callbacks: Optional[List[Any]] = None,
    ):
        super().__init__(agent_instance=agent_instance, name=name, callbacks=callbacks)
        self._last_response = ""
        self._last_metrics: Dict[str, Any] = {}

    def _run_agent(self, query: str) -> str:
        """Run the multi-agent team and return the final response."""
        response, metrics = self.agent.run(query)
        self._last_response = response
        self._last_metrics = metrics
        return response

    def get_messages(self):
        """Return message history from team execution."""
        from maseval import MessageHistory
        return MessageHistory(self.agent.get_messages())

    def _gather_usage(self):
        """Gather usage from team execution."""
        usage = self._last_metrics.get("usage", {})
        if not usage:
            return TokenUsage()
        return TokenUsage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    def gather_traces(self) -> Dict[str, Any]:
        """Gather traces including per-agent metrics and tool usage."""
        base = super().gather_traces()
        return {
            **base,
            "duration_seconds": self._last_metrics.get("duration_seconds", 0),
            "model": self._last_metrics.get("model"),
            "usage": self._last_metrics.get("usage"),
            "agent_metrics": self._last_metrics.get("agents", {}),
            "tools_used": self._last_metrics.get("tools_used", []),
            "team_size": self._last_metrics.get("team_size", 1),
        }


# =============================================================================
# Environment (minimal for this example)
# =============================================================================


class SimpleEnvironment(Environment):
    """A simple environment that tracks state for evaluation."""

    def __init__(self, task_metadata: Dict[str, Any]):
        # Store task_metadata before calling super().__init__ which calls setup_state
        self.task_metadata = task_metadata
        super().__init__(environment_data=task_metadata)

    def setup_state(self, environment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Set up initial environment state.

        Args:
            environment_data: Task metadata containing expected answer and category

        Returns:
            Empty initial state dictionary
        """
        return {}

    def create_tools(self) -> Dict[str, Any]:
        """Create tools available to agents.

        Returns:
            Empty dictionary - this environment has no tools
        """
        return {}

    def get_state(self) -> Dict[str, Any]:
        """Get current environment state."""
        return self.state

    def gather_traces(self) -> Dict[str, Any]:
        """Gather environment traces."""
        return {
            **super().gather_traces(),
            "task_metadata": self.task_metadata,
            "state": self.state,
        }


# =============================================================================
# Evaluator (accuracy and completeness)
# =============================================================================


class AccuracyEvaluator:
    """Evaluator that checks response accuracy and completeness.

    Supports multiple evaluation strategies:
    - Exact match (for factual questions)
    - Numeric match (for math problems)
    - Keyword presence (for explanations)
    - Code presence (for coding tasks)
    """

    def __init__(self):
        """Initialize the accuracy evaluator."""
        self.name = "accuracy_evaluator"

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant traces for evaluation.

        Args:
            traces: Complete execution traces

        Returns:
            Filtered traces needed for evaluation
        """
        # For this evaluator, we only need the agent's response messages
        return {"messages": traces.get("messages", [])}

    def __call__(self, traces: Dict[str, Any], final_answer: Optional[str] = None) -> Dict[str, Any]:
        """Compute evaluation metrics from filtered traces.

        Args:
            traces: Pre-filtered traces from filter_traces()
            final_answer: Agent's final answer

        Returns:
            Evaluation results with score and metrics
        """
        if final_answer is None:
            return {"score": 0.0, "error": "No final answer provided"}

        # Get expected answer and category from traces metadata (passed via environment)
        expected = traces.get("expected", "")
        category = traces.get("category", "factual")

        # Evaluate based on category
        result = self._evaluate_by_category(final_answer, expected, category)

        return {
            "score": result["score"],
            "response": final_answer,
            "expected": expected,
            "category": category,
            "match_type": result["match_type"],
            "details": result.get("details", ""),
        }

    def evaluate(
        self,
        environment: Environment,
        agents: List[AgentAdapter],
        user: Optional[User] = None,
    ) -> Dict[str, Any]:
        """Evaluate agent response against expected answer.

        Args:
            environment: Environment with task metadata
            agents: List of agents that ran
            user: Optional user (not used here)

        Returns:
            Dict with evaluation results
        """
        if not agents:
            return {"score": 0.0, "error": "No agents provided"}

        agent = agents[0]
        messages = agent.get_messages()
        message_list = list(messages)

        if not message_list:
            return {"score": 0.0, "error": "No messages from agent"}

        # Get the last assistant message (agent's response)
        response = ""
        for msg in reversed(message_list):
            if msg.get("role") == "assistant" and msg.get("content"):
                response = msg["content"]
                break

        # Get expected answer from task metadata
        expected = environment.task_metadata.get("expected", "")
        category = environment.task_metadata.get("category", "factual")

        # Evaluate based on category
        result = self._evaluate_by_category(response, expected, category)

        return {
            "score": result["score"],
            "response": response,
            "expected": expected,
            "category": category,
            "match_type": result["match_type"],
            "details": result.get("details", ""),
        }

    def _evaluate_by_category(self, response: str, expected: Any, category: str) -> Dict[str, Any]:
        """Evaluate response based on task category.

        Args:
            response: Agent's response
            expected: Expected answer (string or list of keywords)
            category: Task category

        Returns:
            Dict with score, match_type, and details
        """
        response_lower = response.lower().strip()

        if category == "factual":
            # Exact match (case-insensitive)
            if isinstance(expected, str):
                if expected.lower() in response_lower:
                    return {"score": 1.0, "match_type": "exact_match"}
                return {"score": 0.0, "match_type": "no_match"}
            if isinstance(expected, list):
                found = sum(1 for kw in expected if kw.lower() in response_lower)
                score = found / len(expected) if expected else 0.0
                return {
                    "score": score,
                    "match_type": "keyword_match",
                    "details": f"Found {found}/{len(expected)} keywords",
                }

        elif category == "reasoning":
            # Numeric match - extract numbers from response
            if isinstance(expected, str):
                expected_num = self._extract_number(expected)
                if expected_num is not None:
                    response_nums = self._extract_all_numbers(response)
                    if expected_num in response_nums:
                        return {"score": 1.0, "match_type": "numeric_match"}
                    # Check for close match (within 10%)
                    for num in response_nums:
                        if abs(num - expected_num) / max(abs(expected_num), 1) < 0.1:
                            return {"score": 0.9, "match_type": "numeric_close"}
                return {"score": 0.0, "match_type": "no_numeric_match"}
            if isinstance(expected, list):
                # Keyword-based reasoning evaluation
                found = sum(1 for kw in expected if kw.lower() in response_lower)
                score = found / len(expected) if expected else 0.0
                return {
                    "score": score,
                    "match_type": "keyword_match",
                    "details": f"Found {found}/{len(expected)} keywords",
                }

        elif category == "coding":
            # Check for code presence - handle both string and list expected values
            if isinstance(expected, str):
                if expected.lower() in response_lower:
                    return {"score": 1.0, "match_type": "code_present"}
                return {"score": 0.0, "match_type": "no_code"}
            if isinstance(expected, list):
                # Check all required code elements are present
                found = sum(1 for kw in expected if kw.lower() in response_lower)
                score = found / len(expected) if expected else 0.0
                return {
                    "score": score,
                    "match_type": "code_elements",
                    "details": f"Found {found}/{len(expected)} required code elements",
                }

        elif category in ("explanation", "analysis"):
            # Check for keyword presence
            if isinstance(expected, list):
                found = sum(1 for kw in expected if kw.lower() in response_lower)
                score = found / len(expected) if expected else 0.0
                return {
                    "score": score,
                    "match_type": "keyword_match",
                    "details": f"Found {found}/{len(expected)} keywords",
                }
            if isinstance(expected, str):
                if expected.lower() in response_lower:
                    return {"score": 1.0, "match_type": "keyword_present"}
                return {"score": 0.0, "match_type": "no_match"}

        # Default: check if response is non-empty
        if response.strip():
            return {"score": 0.5, "match_type": "non_empty_response"}
        return {"score": 0.0, "match_type": "empty_response"}

    def _extract_number(self, s: str) -> Optional[float]:
        """Extract a number from a string."""
        try:
            return float(s.replace(",", ""))
        except ValueError:
            # Try to find a number in the string
            match = re.search(r"-?\d+\.?\d*", s.replace(",", ""))
            if match:
                return float(match.group())
        return None

    def _extract_all_numbers(self, s: str) -> List[float]:
        """Extract all numbers from a string."""
        matches = re.findall(r"-?\d+\.?\d*", s.replace(",", ""))
        numbers = []
        for m in matches:
            try:
                numbers.append(float(m))
            except ValueError:
                pass
        return numbers


# =============================================================================
# Benchmark
# =============================================================================


class SimpleBenchmark(Benchmark):
    """A simple benchmark that evaluates agent responses.

    This benchmark demonstrates:
    - Latency tracking (per-task and aggregate)
    - Token usage tracking
    - Cost estimation (when available)
    - Accuracy evaluation
    - Multi-agent team mode with tools (optional)
    """

    def __init__(
        self,
        model: LiteLLMModelAdapter,
        callbacks: Optional[List[Any]] = None,
        use_team: bool = False,
        use_tools: bool = True,
        **kwargs,
    ):
        """Initialize the benchmark.

        Args:
            model: The LLM model adapter to use
            callbacks: Optional list of callbacks
            use_team: If True, use multi-agent team (Planner+Executor+Reviewer)
            use_tools: If True, enable tool use in team mode
            **kwargs: Additional arguments passed to Benchmark
        """
        super().__init__(callbacks=callbacks, **kwargs)
        self.model = model
        self.evaluator = AccuracyEvaluator()
        self.use_team = use_team
        self.use_tools = use_tools

    def get_model_adapter(self, model_id: str, **kwargs) -> LiteLLMModelAdapter:
        """Return the model adapter for benchmark components.

        This benchmark uses a single pre-configured model adapter.
        The model_id parameter is ignored since we use the model passed at init.
        """
        return self.model

    def setup_environment(self, agent_data: Dict[str, Any], task: Task, seed_generator) -> Environment:
        """Set up environment for a task."""
        return SimpleEnvironment(task_metadata=task.metadata)

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
        user: Optional[User],
        seed_generator,
    ) -> Tuple[List[AgentAdapter], Dict[str, AgentAdapter]]:
        """Set up agents for a task.

        If use_team is True, creates a multi-agent team with:
        - Planner: Analyzes task and creates plan
        - Executor: Executes plan, uses tools
        - Reviewer: Reviews output for quality

        Otherwise, creates a single simple agent.
        """
        if self.use_team:
            # Multi-agent team mode
            team = MultiAgentTeam(self.model, use_tools=self.use_tools)
            adapter = MultiAgentTeamAdapter(agent_instance=team, name="multi_agent_team")
            return [adapter], {"multi_agent_team": adapter}
        else:
            # Simple single-agent mode
            agent = SimpleLLMAgent(self.model)
            adapter = SimpleAgentAdapter(agent_instance=agent, name="simple_agent")
            return [adapter], {"simple_agent": adapter}

    def setup_evaluators(
        self,
        environment: Environment,
        task: Task,
        agents: List[AgentAdapter],
        user: Optional[User],
        seed_generator,
    ) -> List[Evaluator]:
        """Set up evaluators."""
        return [self.evaluator]

    def run_agents(
        self,
        agents: List[AgentAdapter],
        task: Task,
        environment: Environment,
        query: str,
    ) -> Any:
        """Run agents on the task."""
        if not agents:
            return None

        # Run the primary agent
        agent = agents[0]
        return agent.run(query)

    def evaluate(
        self,
        evaluators: List[Evaluator],
        agents: Dict[str, AgentAdapter],
        final_answer: Any,
        traces: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Evaluate agent performance.

        Args:
            evaluators: List of evaluators to run
            agents: Dictionary of agent adapters
            final_answer: The final answer from the agent
            traces: Execution traces from all components

        Returns:
            List of evaluation results
        """
        results = []
        for evaluator in evaluators:
            # Filter traces for this evaluator
            filtered_traces = evaluator.filter_traces(traces)

            # Add expected answer and category to filtered traces for evaluation
            # Get from environment traces
            env_traces = traces.get("environment", {})
            if env_traces:
                task_metadata = env_traces.get("task_metadata", {})
                filtered_traces["expected"] = task_metadata.get("expected", "")
                filtered_traces["category"] = task_metadata.get("category", "factual")

            # Call evaluator with filtered traces and final answer
            result = evaluator(filtered_traces, final_answer)
            results.append(result)
        return results


# =============================================================================
# Metrics Aggregation
# =============================================================================


def compute_benchmark_metrics(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics from benchmark reports.

    Args:
        reports: List of benchmark execution reports

    Returns:
        Dict with aggregate metrics
    """
    if not reports:
        return {"error": "No reports"}

    # Initialize accumulators
    total_tasks = len(reports)
    total_score = 0.0
    total_duration = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    task_costs: List[float] = []
    scores_by_category: Dict[str, List[float]] = {}
    durations_by_category: Dict[str, List[float]] = {}

    for report in reports:
        # Extract evaluation score
        eval_results = report.get("eval", [])
        if eval_results:
            score = eval_results[0].get("score", 0.0)
            total_score += score

            # Track by category
            category = report.get("task", {}).get("metadata", {}).get("category", "unknown")
            if category not in scores_by_category:
                scores_by_category[category] = []
            scores_by_category[category].append(score)

            if category not in durations_by_category:
                durations_by_category[category] = []

        # Extract traces for timing and tokens
        traces = report.get("traces", {})
        agent_traces = traces.get("agents", {})

        for agent_name, agent_trace in agent_traces.items():
            duration = agent_trace.get("duration_seconds", 0)
            total_duration += duration

            # Track duration by category
            category = report.get("task", {}).get("metadata", {}).get("category", "unknown")
            if category in durations_by_category:
                durations_by_category[category].append(duration)

            # Extract usage
            usage = agent_trace.get("usage") or {}
            total_input_tokens += usage.get("input_tokens", 0)
            total_output_tokens += usage.get("output_tokens", 0)

            # Cost (if available)
            cost = usage.get("cost", 0.0)
            if isinstance(cost, (int, float)):
                total_cost += cost
                if cost > 0:
                    task_costs.append(cost)

    # Compute averages
    avg_score = total_score / total_tasks if total_tasks > 0 else 0.0
    avg_duration = total_duration / total_tasks if total_tasks > 0 else 0.0

    # Compute per-category metrics
    category_metrics = {}
    for category, scores in scores_by_category.items():
        category_metrics[category] = {
            "accuracy": sum(scores) / len(scores) if scores else 0.0,
            "avg_duration": sum(durations_by_category.get(category, [])) / len(scores) if scores else 0.0,
            "count": len(scores),
        }

    # Compute per-task cost statistics
    import statistics as stats_mod
    cost_stats: Dict[str, float] = {}
    if task_costs:
        cost_stats = {
            "min": min(task_costs),
            "max": max(task_costs),
            "median": stats_mod.median(task_costs),
            "mean": stats_mod.mean(task_costs),
        }

    return {
        "total_tasks": total_tasks,
        "accuracy": avg_score,
        "avg_duration_seconds": avg_duration,
        "total_duration_seconds": total_duration,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_cost": total_cost,
        "task_costs": task_costs,
        "cost_stats": cost_stats,
        "metrics_by_category": category_metrics,
    }


# =============================================================================
# Main
# =============================================================================


def create_model_adapter(model_id: str, provider: str = "auto") -> LiteLLMModelAdapter:
    """Create a LiteLLM model adapter based on provider and model ID.

    Args:
        model_id: Model identifier (e.g., "qwen", "sonnet-5", or full model ID)
        provider: Provider name ("ollama", "openrouter", or "auto" to detect from model_id)

    Returns:
        Configured LiteLLMModelAdapter
    """
    # Resolve model presets
    if provider == "ollama" or (provider == "auto" and model_id in OLLAMA_MODELS):
        resolved_model = OLLAMA_MODELS.get(model_id, model_id)
        if not resolved_model.startswith("ollama/"):
            resolved_model = f"ollama/{resolved_model}"
        return LiteLLMModelAdapter(
            model_id=resolved_model,
            api_base=DEFAULT_OLLAMA_BASE,
            api_key=DEFAULT_OLLAMA_KEY,
        )

    elif provider == "openrouter" or (provider == "auto" and model_id in OPENROUTER_MODELS):
        resolved_model = OPENROUTER_MODELS.get(model_id, model_id)
        if not DEFAULT_OPENROUTER_KEY:
            raise ValueError("OPENROUTER_API_KEY not set in environment")
        return LiteLLMModelAdapter(
            model_id=f"openrouter/{resolved_model}" if not resolved_model.startswith("openrouter/") else resolved_model,
            api_key=DEFAULT_OPENROUTER_KEY,
        )

    # Generic model ID - use as-is
    if model_id.startswith("ollama/"):
        return LiteLLMModelAdapter(
            model_id=model_id,
            api_base=DEFAULT_OLLAMA_BASE,
            api_key=DEFAULT_OLLAMA_KEY,
        )
    return LiteLLMModelAdapter(model_id=model_id)


def run_single_model_benchmark(model_id: str, provider: str, tasks: List[Task], output_dir: Path, args) -> Dict[str, Any]:
    """Run benchmark for a single model.

    Args:
        model_id: Model identifier
        provider: Provider name
        tasks: List of tasks to run
        output_dir: Output directory for results
        args: Command line arguments

    Returns:
        Dictionary with model name, metrics, and any errors
    """
    print(f"\n{'='*80}")
    print(f"Testing model: {model_id} (provider: {provider})")
    print(f"{'='*80}")

    try:
        model = create_model_adapter(model_id, provider)
    except Exception as e:
        print(f"  ERROR creating model adapter: {e}")
        return {"model": model_id, "error": str(e), "metrics": None}

    # Create result logger with model-specific filename
    logger = FileResultLogger(
        output_dir=str(output_dir),
        filename_pattern=f"benchmark_{model_id.replace('/', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl",
    )

    # Determine team mode
    use_team = getattr(args, "use_team", False)
    use_tools = getattr(args, "use_tools", True)
    mode_str = "MULTI-AGENT TEAM (Planner+Executor+Reviewer)" if use_team else "SINGLE AGENT"
    print(f"  Mode: {mode_str}")
    if use_team and use_tools:
        print(f"  Tools: calculator, validate_code, extract_keywords")

    # Create and run benchmark
    benchmark = SimpleBenchmark(
        model=model,
        callbacks=[logger],
        progress_bar=True,
        use_team=use_team,
        use_tools=use_tools,
    )

    start_time = time.time()
    try:
        reports = benchmark.run(tasks=tasks, agent_data={})
        total_time = time.time() - start_time

        metrics = compute_benchmark_metrics(reports)
        metrics["total_wall_time_seconds"] = total_time

        print(f"\n  Results for {model_id}:")
        print(f"    Accuracy: {metrics['accuracy']:.2%}")
        print(f"    Avg duration: {metrics['avg_duration_seconds']:.2f}s")
        print(f"    Total tokens: {metrics['total_tokens']}")
        if metrics["total_cost"] > 0:
            print(f"    Total cost: ${metrics['total_cost']:.4f}")

        return {"model": model_id, "provider": provider, "metrics": metrics, "error": None}

    except Exception as e:
        print(f"  ERROR running benchmark: {e}")
        return {"model": model_id, "provider": provider, "metrics": None, "error": str(e)}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DAF Benchmark Example with Multi-Model Support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID or preset name (e.g., 'qwen', 'sonnet-5', 'ollama/gemma4:31b')",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["ollama", "openrouter", "auto"],
        default="auto",
        help="Model provider",
    )
    parser.add_argument(
        "--all-ollama",
        action="store_true",
        help="Run all Ollama Cloud models (qwen, glm, minimax, kimi, gemma)",
    )
    parser.add_argument(
        "--all-openrouter",
        action="store_true",
        help="Run all OpenRouter models (sonnet-5, opus-5, sol-5.6, terra-5.6)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=DEFAULT_OLLAMA_BASE,
        help="API base URL (for Ollama)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_OLLAMA_KEY,
        help="API key (for Ollama)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to run",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard", "expert", "all"],
        default="easy",
        help="Task difficulty level: easy (simple QA), medium (multi-step), hard (complex reasoning/coding), expert (system design/debugging), all",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--use-team",
        action="store_true",
        help="Use multi-agent team (Planner+Executor+Reviewer) instead of single agent",
    )
    parser.add_argument(
        "--use-tools",
        action="store_true",
        default=True,
        help="Enable tool use in team mode (calculator, code validator, keyword extractor)",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Disable tool use in team mode",
    )
    parser.add_argument(
        "--use-daf",
        action="store_true",
        help="Use DAF backend instead of simple agent (requires running DAF backend)",
    )
    parser.add_argument(
        "--daf-url",
        type=str,
        default=DEFAULT_DAF_URL,
        help="DAF backend URL",
    )
    parser.add_argument(
        "--daf-skill",
        type=str,
        default="general",
        help="DAF Skill ID to use",
    )

    args = parser.parse_args()

    # Handle --no-tools flag
    if args.no_tools:
        args.use_tools = False

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create tasks
    tasks = create_sample_tasks(args.difficulty)
    if args.limit:
        tasks = tasks[: args.limit]

    print(f"\nRunning {len(tasks)} tasks (difficulty: {args.difficulty})")

    # Determine which models to run
    all_results = []

    if args.all_ollama:
        print("\n" + "="*80)
        print("Running ALL OLLAMA CLOUD MODELS")
        print("="*80)
        for model_name in OLLAMA_MODELS.keys():
            result = run_single_model_benchmark(model_name, "ollama", tasks, output_dir, args)
            all_results.append(result)

    elif args.all_openrouter:
        print("\n" + "="*80)
        print("Running ALL OPENROUTER MODELS")
        print("="*80)
        for model_name in OPENROUTER_MODELS.keys():
            result = run_single_model_benchmark(model_name, "openrouter", tasks, output_dir, args)
            all_results.append(result)

    elif args.model:
        # Single model mode
        result = run_single_model_benchmark(args.model, args.provider, tasks, output_dir, args)
        all_results.append(result)

    else:
        # Default: run gemma on Ollama
        result = run_single_model_benchmark("gemma", "ollama", tasks, output_dir, args)
        all_results.append(result)

    # Print summary
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    # Collect cost data across all models for relative comparison
    all_model_costs: List[float] = []
    for result in all_results:
        if result["metrics"]:
            all_model_costs.append(result["metrics"]["total_cost"])

    # Compute cross-model cost statistics
    import statistics as stats_mod
    cross_model_stats: Dict[str, float] = {}
    if all_model_costs and any(c > 0 for c in all_model_costs):
        positive_costs = [c for c in all_model_costs if c > 0]
        cross_model_stats = {
            "min": min(positive_costs),
            "max": max(positive_costs),
            "median": stats_mod.median(positive_costs),
            "mean": stats_mod.mean(positive_costs),
        }

    # Print detailed per-model results
    for result in all_results:
        model = result["model"]
        if result["error"]:
            print(f"\n{model}: ERROR - {result['error']}")
        elif result["metrics"]:
            m = result["metrics"]
            print(f"\n{model}:")
            print(f"  Accuracy: {m['accuracy']:.2%}")
            print(f"  Avg duration: {m['avg_duration_seconds']:.2f}s")
            print(f"  Total tokens: {m['total_tokens']} (in: {m['total_input_tokens']}, out: {m['total_output_tokens']})")
            if m["total_cost"] > 0:
                print(f"  Total cost: ${m['total_cost']:.4f}")
            # Category breakdown
            cats = m.get("metrics_by_category", {})
            if cats:
                print(f"  By category:")
                for cat, cm in sorted(cats.items()):
                    print(f"    {cat}: {cm['accuracy']:.0%} accuracy, {cm['avg_duration']:.1f}s avg ({cm['count']} tasks)")

    # Print comparison table
    successful = [r for r in all_results if r["metrics"]]
    if len(successful) > 1:
        avg_cost = cross_model_stats.get("mean", 0)
        print(f"\n{'=' * 100}")
        print("FULL COMPARISON TABLE")
        print(f"{'=' * 100}")
        hdr = f"{'Model':<22} {'Acc':>6} {'AvgDur':>8} {'InTok':>8} {'OutTok':>8} {'TotalTok':>9}"
        if cross_model_stats:
            hdr += f" {'Cost':>10} {'$/Task':>10} {'vs Avg':>8}"
        print(hdr)
        print("-" * len(hdr))
        for result in successful:
            m = result["metrics"]
            model = result["model"]
            row = f"{model:<22} {m['accuracy']:>5.0%} {m['avg_duration_seconds']:>7.1f}s {m['total_input_tokens']:>8} {m['total_output_tokens']:>8} {m['total_tokens']:>9}"
            if cross_model_stats:
                total = m["total_cost"]
                per_task = total / m["total_tasks"] if m["total_tasks"] > 0 else 0
                relative = f"{total / avg_cost:.2f}x" if avg_cost > 0 else "-"
                row += f" ${total:>9.4f} ${per_task:>9.4f} {relative:>8}"
            print(row)
        if cross_model_stats:
            print(f"\nCross-model cost: avg=${avg_cost:.4f} | min=${cross_model_stats['min']:.4f} | max=${cross_model_stats['max']:.4f} | median=${cross_model_stats['median']:.4f}")

    # Save summary
    summary_path = output_dir / "multi_model_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "results": all_results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "task_count": len(tasks),
            },
            f,
            indent=2,
        )

    print(f"\n\nResults saved to: {output_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

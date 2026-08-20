"""DAF (Declarative Agentic Framework) integration for MASEval.

This module requires daf-sdk to be installed:
    pip install daf-sdk

Or for local development, ensure the daf-sdk is in your Python path.

DAF is a platform for building, managing, and orchestrating AI agents and
multi-agent teams. This adapter wraps DAF's orchestration API to enable
evaluation within MASEval's benchmarking framework.

Example:
    ```python
    from maseval.interface.agents.daf import DAFAgentAdapter
    from daf_sdk import DAF

    # Create DAF client
    daf_client = DAF(base_url="http://localhost:8012", api_key="daf_your_key")

    # Create adapter
    adapter = DAFAgentAdapter(
        agent_instance=daf_client,
        skill_id="my_skill",
        name="daf_agent",
    )

    # Run
    result = adapter.run("Your task query")
    ```
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional
import time

from maseval import AgentAdapter, MessageHistory
from maseval.core.usage import CostCalculator, TokenUsage, Usage

__all__ = ["DAFAgentAdapter"]

if TYPE_CHECKING:
    pass


def _check_daf_sdk_installed():
    """Check if daf-sdk is installed and raise a helpful error if not."""
    try:
        import daf_sdk  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "daf-sdk is not installed. Install with: pip install daf-sdk\nOr for local development, add daf-sdk to your Python path."
        ) from e


class DAFAgentAdapter(AgentAdapter):
    """An AgentAdapter for DAF (Declarative Agentic Framework) agent teams.

    This adapter integrates DAF's orchestration API with MASEval's benchmarking
    framework. It can wrap either:
    - A DAF SDK client that calls the DAF backend orchestration API
    - A standalone callable that processes queries (for testing without DAF)

    DAF supports multi-agent teams with hierarchical orchestration, Skills,
    workflows, and tool use. This adapter captures execution events and
    converts them to MASEval's tracing format.

    How to use:
        1. **Create a DAF client** with your backend URL and API key
        2. **Wrap with DAFAgentAdapter** specifying the Skill ID
        3. **Use in benchmarks** to evaluate DAF agent teams
        4. **Access traces** for analysis of skills invoked, tokens used, etc.

        Example workflow:
            ```python
            from daf_sdk import DAF
            from maseval.interface.agents.daf import DAFAgentAdapter

            # Create DAF client pointing to your backend
            client = DAF(base_url="http://localhost:8012", api_key="daf_key")

            # Create adapter for a specific Skill
            adapter = DAFAgentAdapter(
                agent_instance=client,
                skill_id="tau2_retail",
                name="retail_agent",
            )

            # Run evaluation
            result = adapter.run("Help me cancel my flight")

            # Get execution traces
            traces = adapter.gather_traces()
            print(f"Skills invoked: {traces['daf_skills_invoked']}")
            print(f"Total iterations: {traces['daf_total_iterations']}")
            ```

    Execution Monitoring:
        The adapter provides comprehensive monitoring through `gather_traces()`:

        - **Skills invoked**: List of DAF Skills that were called
        - **Iterations**: Total number of orchestration iterations
        - **Tokens**: Total token usage across all LLM calls
        - **Events**: Raw SSE events from DAF for detailed analysis
        - **Duration**: Total execution time
    """

    def __init__(
        self,
        agent_instance: Any,
        name: str,
        skill_id: Optional[str] = None,
        callbacks: Optional[List[Any]] = None,
        cost_calculator: Optional[CostCalculator] = None,
        model_id: Optional[str] = None,
        team_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the DAF adapter.

        Args:
            agent_instance: Either a DAF SDK client (daf_sdk.DAF) or a callable
                that takes (query, config) and returns a response dict.
                For testing, you can pass a simple function.
            name: Agent name for identification
            skill_id: DAF Skill ID to invoke. Required if agent_instance is a DAF client.
            callbacks: Optional list of AgentCallback instances
            cost_calculator: Optional cost calculator for computing cost from tokens
            model_id: Optional model ID override for cost calculation
            team_config: Optional configuration dict passed to DAF's context field
        """
        super().__init__(
            agent_instance=agent_instance,
            name=name,
            callbacks=callbacks,
            cost_calculator=cost_calculator,
            model_id=model_id,
        )
        self.skill_id = skill_id
        self.team_config = team_config or {}
        self._messages: List[Dict[str, Any]] = []
        self._events: List[Dict[str, Any]] = []
        self._execution_metrics: Dict[str, Any] = {}
        self._last_duration: float = 0.0

    def _run_agent(self, query: str) -> str:
        """Execute DAF orchestration and return the final answer.

        Supports two modes:
        1. DAF SDK client: Calls stream() for SSE events
        2. Callable: Calls the function directly

        Args:
            query: The user query/task to execute

        Returns:
            Final answer string from DAF
        """
        # Reset state for new execution
        self._messages = []
        self._events = []
        self._execution_metrics = {}
        self._last_duration = 0.0

        start_time = time.time()

        # Check if this is a callable (for testing without DAF)
        if callable(self.agent) and not hasattr(self.agent, "stream"):
            return self._run_callable(query, start_time)

        # DAF SDK mode - use streaming
        _check_daf_sdk_installed()

        final_response = None
        skills_invoked: List[str] = []
        total_tokens = 0
        total_iterations = 0

        try:
            # Stream events from DAF orchestration
            for event in self.agent.stream(
                "daf_orchestrate",
                {
                    "goal": query,
                    "skill_id": self.skill_id,
                    "context": self.team_config,
                },
            ):
                self._process_event(event, skills_invoked)

                # Extract metrics from terminal event
                if event.get("type") == "orchestrate_complete":
                    final_response = event.get("final_response", "")
                    total_tokens = event.get("total_tokens", 0)
                    total_iterations = event.get("total_iterations", 0)

        except Exception as e:
            self._events.append(
                {
                    "type": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )
            self._last_duration = time.time() - start_time
            raise

        self._last_duration = time.time() - start_time

        # Store execution metrics
        self._execution_metrics = {
            "skills_invoked": skills_invoked,
            "total_tokens": total_tokens,
            "total_iterations": total_iterations,
            "duration_seconds": self._last_duration,
        }

        return final_response or ""

    def _run_callable(self, query: str, start_time: float) -> str:
        """Run a callable agent (for testing without DAF backend).

        Args:
            query: The task query
            start_time: Execution start timestamp

        Returns:
            Response string from the callable
        """
        try:
            # Call the agent with query and optional config
            result = self.agent(query, self.team_config)

            # Handle different response formats
            if isinstance(result, str):
                final_response = result
            elif isinstance(result, dict):
                final_response = result.get("response", result.get("answer", str(result)))
                # Extract metrics if present
                if "tokens" in result:
                    self._execution_metrics["total_tokens"] = result["tokens"]
                if "skills" in result:
                    self._execution_metrics["skills_invoked"] = result["skills"]
            else:
                final_response = str(result)

            # Create a message from the response
            self._messages.append(
                {
                    "role": "assistant",
                    "content": final_response,
                }
            )

        except Exception as e:
            self._events.append(
                {
                    "type": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )
            self._last_duration = time.time() - start_time
            raise

        self._last_duration = time.time() - start_time
        self._execution_metrics["duration_seconds"] = self._last_duration

        return final_response

    def _process_event(self, event: Dict[str, Any], skills_invoked: List[str]) -> None:
        """Process a DAF SSE event and update internal state.

        Args:
            event: Event dict from DAF stream
            skills_invoked: List to append skill IDs to
        """
        self._events.append(event)
        event_type = event.get("type")

        if event_type == "text":
            # LLM text output
            self._messages.append(
                {
                    "role": "assistant",
                    "content": event.get("content", ""),
                }
            )

        elif event_type == "tool_call":
            # Tool invocation
            self._messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{event.get('tool_name', 'unknown')}",
                            "type": "function",
                            "function": {
                                "name": event.get("tool_name"),
                                "arguments": str(event.get("arguments", {})),
                            },
                        }
                    ],
                }
            )

        elif event_type == "skill_start":
            # Track skill invocation
            skill_id = event.get("skill_id")
            if skill_id and skill_id not in skills_invoked:
                skills_invoked.append(skill_id)

    def get_messages(self) -> MessageHistory:
        """Get message history from DAF execution.

        Returns:
            MessageHistory with messages converted from DAF events
        """
        return MessageHistory(self._messages)

    def gather_traces(self) -> Dict[str, Any]:
        """Gather execution traces including DAF-specific metrics.

        Extends base traces with DAF orchestration details:
        - Skills invoked during execution
        - Total iterations of the orchestration loop
        - Total tokens used across all LLM calls
        - Raw SSE events for detailed analysis

        Returns:
            Dict with base traces plus DAF-specific fields
        """
        base_traces = super().gather_traces()

        # Add DAF-specific traces
        daf_traces = {
            "daf_skill_id": self.skill_id,
            "daf_skills_invoked": self._execution_metrics.get("skills_invoked", []),
            "daf_total_iterations": self._execution_metrics.get("total_iterations", 0),
            "daf_total_tokens": self._execution_metrics.get("total_tokens", 0),
            "daf_duration_seconds": self._execution_metrics.get("duration_seconds", 0.0),
            "daf_events": self._events,
            "daf_team_config": self.team_config,
        }

        return {**base_traces, **daf_traces}

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration from this DAF adapter.

        Returns:
            Dict with base config plus DAF-specific configuration
        """
        base_config = super().gather_config()

        return {
            **base_config,
            "daf_skill_id": self.skill_id,
            "daf_team_config": self.team_config,
        }

    def _gather_usage(self) -> Usage:
        """Gather token usage from DAF execution metrics.

        Returns:
            TokenUsage with totals from DAF orchestration
        """
        total_tokens = self._execution_metrics.get("total_tokens", 0)

        if total_tokens == 0:
            return Usage()

        # DAF reports total tokens; we approximate input/output split
        # In practice, you'd want DAF to provide more granular usage
        return TokenUsage(
            input_tokens=int(total_tokens * 0.6),  # Approximate
            output_tokens=int(total_tokens * 0.4),
            total_tokens=total_tokens,
        )

    def _resolve_model_id(self) -> Optional[str]:
        """Auto-detect model ID.

        Returns the explicit model_id if set, otherwise None.
        DAF's orchestration may use multiple models internally.
        """
        return self._model_id

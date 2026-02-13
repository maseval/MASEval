from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod
import json
import os
from datetime import datetime
from .model import ModelAdapter
from .tracing import TraceableMixin
from .exceptions import EnvironmentError, UserError
import uuid
from enum import Enum


class SimulatorError(Exception):
    """Base exception for simulator failures.

    This exception is raised when an LLM simulator exhausts all retry attempts
    without successfully parsing the model output.

    Note:
        Subclasses (ToolSimulatorError, UserSimulatorError) inherit from the
        appropriate MASEval exception type for proper error classification.
        Use those specific subclasses in concrete simulators.

    Attributes:
        message: Description of the failure.
        attempts: Number of attempts made before failing.
        last_error: The last error encountered during parsing.
        logs: The complete log of all attempts for debugging.
    """

    def __init__(
        self,
        message: str,
        attempts: int = 0,
        last_error: Optional[str] = None,
        logs: Optional[List[Dict[str, Any]]] = None,
        component: Optional[str] = None,
    ):
        self.message = message
        self.attempts = attempts
        self.last_error = last_error
        self.logs = logs or []
        self.component = component
        super().__init__(self.message)

    def __str__(self) -> str:
        parts = []
        if self.component:
            parts.append(f"[{self.component}]")
        parts.append(self.message)
        if self.attempts > 0:
            parts.append(f"(attempts: {self.attempts})")
        if self.last_error:
            parts.append(f"Last error: {self.last_error}")
        return " ".join(parts)


class ToolSimulatorError(SimulatorError, EnvironmentError):
    """Tool simulator failed - not the agent's fault.

    Raised when ToolLLMSimulator fails after exhausting retries.
    This inherits from EnvironmentError, so it's classified as
    ENVIRONMENT_ERROR in benchmark results.
    """

    def __init__(
        self,
        message: str,
        attempts: int = 0,
        last_error: Optional[str] = None,
        logs: Optional[List[Dict[str, Any]]] = None,
        component: Optional[str] = None,
    ):
        # Initialize SimulatorError (sets message, attempts, last_error, logs, component)
        SimulatorError.__init__(
            self,
            message=message,
            attempts=attempts,
            last_error=last_error,
            logs=logs,
            component=component,
        )
        # Initialize EnvironmentError for MASEval classification
        EnvironmentError.__init__(
            self,
            message=message,
            component=component,
            details={"attempts": attempts, "last_error": last_error},
        )


class UserSimulatorError(SimulatorError, UserError):
    """User simulator failed - not the agent's fault.

    Raised when UserLLMSimulator fails after exhausting retries.
    This inherits from UserError, so it's classified as
    USER_ERROR in benchmark results.
    """

    def __init__(
        self,
        message: str,
        attempts: int = 0,
        last_error: Optional[str] = None,
        logs: Optional[List[Dict[str, Any]]] = None,
        component: Optional[str] = None,
    ):
        # Initialize SimulatorError (sets message, attempts, last_error, logs, component)
        SimulatorError.__init__(
            self,
            message=message,
            attempts=attempts,
            last_error=last_error,
            logs=logs,
            component=component,
        )
        # Initialize UserError for MASEval classification
        UserError.__init__(
            self,
            message=message,
            component=component,
            details={"attempts": attempts, "last_error": last_error},
        )


class LLMSimulator(ABC, TraceableMixin):
    """
    A base class for simulators that use an LLM.

    Subclasses should override `_create_error` to return the appropriate
    exception type (ToolSimulatorError, UserSimulatorError, etc.).
    """

    # Override in subclasses to specify component name for error messages
    _component_name: Optional[str] = None

    def __init__(
        self,
        model: ModelAdapter,
        template: Optional[str] = None,
        max_try: int = 3,
        generation_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the LLMSimulator.

        Args:
            model (ModelAdapter): The language model to use for generation.
            template (str, optional): A prompt template.
            max_try (int, optional): Maximum number of model calls to attempt. Defaults to 3.
            generation_params (Dict[str, Any], optional): Default generation parameters for the model. This overwrites the ModelAdapter's defaults if provided.
                Both can be overridden at call time. Defaults to None.
        """
        self.model = model
        self.template = template
        self.max_try = max_try
        self.generation_params = generation_params or {}
        # canonical structured trace of model invocation attempts and results
        # Each simulator call is assigned a request id; each individual
        # attempt (successful or failed) is appended as a separate entry.
        # Entry schema: {id, timestamp, input, raw_output, parsed_output, status}
        self.logs: list[dict[str, Any]] = []

    def _create_error(
        self,
        message: str,
        attempts: int,
        last_error: Optional[str],
        logs: List[Dict[str, Any]],
    ) -> SimulatorError:
        """Create the appropriate error type for this simulator.

        Override in subclasses to return ToolSimulatorError or UserSimulatorError.

        Args:
            message: Error description.
            attempts: Number of attempts made.
            last_error: The last error encountered.
            logs: Complete log of attempts.

        Returns:
            SimulatorError (or subclass) instance.
        """
        return SimulatorError(
            message=message,
            attempts=attempts,
            last_error=last_error,
            logs=logs,
            component=self._component_name,
        )

    def __call__(self, generation_params: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        Generates a simulated output.
        """
        prompt = self._fill_prompt_template(**kwargs)

        request_id = str(uuid.uuid4())
        attempts = 0
        parsed_result = None

        # merging of LLM default and call-time generation params done here, so subclasses
        # can just call super().__call__(generation_params=...) and have it handled
        generation_params = self.generation_params | (generation_params or {})

        # For each attempt, append a separate history entry with the request id.
        while attempts < self.max_try and parsed_result is None:
            attempts += 1
            raw_output = None
            entry = {
                "id": request_id,
                "timestamp": datetime.now().isoformat(),
                "input": kwargs,
                "prompt": prompt,
                "generation_params": generation_params,
                "raw_output": None,
                "parsed_output": None,
                "status": None,
                "error": None,
            }

            try:
                raw_output = self.model.generate(prompt, generation_params=generation_params)
                entry["raw_output"] = raw_output
            except Exception as e:
                # record model call error attempt by updating the pre-created entry
                entry["raw_output"] = None
                entry["status"] = SimulatorCallStatus.ModelCallError.value
                entry["error"] = str(e)
                # rich.print(
                #     f"[yellow]Warning:[/yellow] Attempt {attempts} failed to call model: {e}"
                #     + (" Retrying..." if attempts < self.max_try else "")
                # )
                self.logs.append(entry)
                continue

            # try parsing the raw output
            try:
                parsed_result = self._parse_output(raw_output)
                # update the existing entry with successful result
                entry["parsed_output"] = parsed_result
                entry["status"] = SimulatorCallStatus.Successful.value

            except (json.JSONDecodeError, AttributeError) as e:
                # update the existing entry with parsing error info
                entry["status"] = SimulatorCallStatus.ModelParsingError.value
                entry["error"] = str(e)
                # rich.print(
                #     f"[yellow]Warning:[/yellow] Attempt {attempts} failed to parse LLM output: {e}"
                #     + (" Retrying..." if attempts < self.max_try else "")
                # )
            self.logs.append(entry)

        if parsed_result is not None:
            return parsed_result

        # All attempts failed - raise exception with details
        last_error = None
        for log in reversed(self.logs):
            if log.get("id") == request_id and log.get("error"):
                last_error = log["error"]
                break

        raise self._create_error(
            message=f"{self.__class__.__name__} failed to parse model output after {self.max_try} attempts",
            attempts=self.max_try,
            last_error=last_error,
            logs=[log for log in self.logs if log.get("id") == request_id],
        )

    def _call_model_and_parse(self, prompt: str) -> Any:
        """
        Calls the model with the given prompt and attempts to parse the output.
        """
        raw_output = self.model.generate(prompt)
        return self._parse_output(raw_output)

    @abstractmethod
    def _fill_prompt_template(self, **kwargs) -> str:
        """
        Fills the prompt template with the provided arguments.
        """
        pass

    @abstractmethod
    def _parse_output(self, output: str) -> Any:
        """
        Parses the raw output from the model.
        """
        pass

    def gather_traces(self) -> dict[str, Any]:
        """Gather execution traces from this simulator.

        Output fields:

        - `type` - Component class name
        - `gathered_at` - ISO timestamp
        - `simulator_type` - The specific simulator class
        - `total_calls` - Number of simulation attempts
        - `successful_calls` - Number of successful simulations
        - `failed_calls` - Number of failed attempts
        - `history` - Complete history of all simulation attempts with timestamps,
          inputs, outputs, status, and error messages

        Returns:
            Dictionary containing simulator execution traces.
        """
        total_calls = len(self.logs)
        successful = sum(1 for entry in self.logs if entry.get("status") == SimulatorCallStatus.Successful.value)
        failed = total_calls - successful

        return {
            **super().gather_traces(),
            "simulator_type": self.__class__.__name__,
            "total_calls": total_calls,
            "successful_calls": successful,
            "failed_calls": failed,
            "logs": self.logs,
        }


class SimulatorCallStatus(Enum):
    ModelCallError = "ModelCallError"
    ModelParsingError = "ModelParsingError"
    Successful = "Successful"


class ToolLLMSimulator(LLMSimulator):
    """
    A simulator that uses an LLM to generate plausible tool outputs.

    Raises ToolSimulatorError on failure, which is classified as
    ENVIRONMENT_ERROR (not the agent's fault).
    """

    def __init__(
        self,
        model: ModelAdapter,
        tool_name: str,
        tool_description: str,
        tool_inputs: Dict[str, Any],
        template: Optional[str] = None,
        max_try: int = 3,
        generation_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the ToolLLMSimulator.

        Args:
            model (ModelAdapter): The language model to use for generation (must have a `generate` method).
            tool_name (str): The name of the tool.
            tool_description (str): The description of the tool.
            tool_inputs (Dict[str, Any]): The schema for the tool's arguments.
            template (str, optional): a prompt template. Defaults to the one in the library. See `maseval.utils.templates.tool_llm_simulator_template.txt`.
                The template should use double curly braces for placeholders. Should contain placeholders for `name`, `description`, `inputs`, and `input_value_dict`.
            max_try (int, optional): Maximum number of model calls to attempt if json output parsing fails. Defaults to 3.
            generation_params (Dict[str, Any], optional): Default generation parameters for the model. This overwrites the ModelAdapter's defaults if provided.
                Both can be overridden at call time. Defaults to None.
        """
        if template is None:
            template_path = os.path.join(os.path.dirname(__file__), "utils", "templates", "tool_llm_simulator_template.txt")
            with open(template_path, "r") as f:
                template = f.read()
        super().__init__(model, template, max_try)
        self.tool_name = tool_name
        self._component_name = tool_name  # For error messages
        self.tool_description = tool_description
        self.tool_inputs = tool_inputs
        self.generation_params = generation_params or {}

    def _create_error(
        self,
        message: str,
        attempts: int,
        last_error: Optional[str],
        logs: List[Dict[str, Any]],
    ) -> ToolSimulatorError:
        """Create ToolSimulatorError for tool simulation failures."""
        return ToolSimulatorError(
            message=message,
            attempts=attempts,
            last_error=last_error,
            logs=logs,
            component=self.tool_name,
        )

    def __call__(self, generation_params: Optional[Dict[str, Any]] = None, **actual_inputs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:  # type: ignore[override]
        return super().__call__(generation_params=generation_params, **actual_inputs)

    def _parse_output(self, output: str) -> tuple[str, Dict[str, Any]]:  # type: ignore[override]
        text_stripped = output.strip()
        # if starts and stop with ```json ... ``` or ``` ... ```, strip those
        if text_stripped.strip().startswith("```") and text_stripped.strip().endswith("```"):
            text_stripped = text_stripped.strip()[3:-3].strip()
            if text_stripped.startswith("json"):
                text_stripped = text_stripped[4:].strip()
        # The model is expected to return a single JSON object string.
        output_data = json.loads(text_stripped)
        return output_data.get("text", ""), output_data.get("details", {})

    def _fill_prompt_template(self, **kwargs) -> str:
        """
        Fills the prompt template with the tool's data and input values.
        """
        assert self.template is not None, "Template must be set"
        prompt = self.template
        replacements = {
            "name": str(self.tool_name),
            "description": str(self.tool_description),
            "inputs": json.dumps(self.tool_inputs, indent=2),
            "input_value_dict": json.dumps(kwargs, indent=2),
        }
        for k, v in replacements.items():
            prompt = prompt.replace("{{" + k + "}}", v)
        return prompt


class UserLLMSimulator(LLMSimulator):
    """
    A simulator that uses an LLM to act as the user.

    Raises UserSimulatorError on failure, which is classified as
    USER_ERROR (not the agent's fault).
    """

    _component_name = "user_simulator"

    def __init__(
        self,
        model: ModelAdapter,
        user_profile: Dict[str, str],
        scenario: str,
        template: Optional[str] = None,
        max_try: int = 3,
        generation_params: Optional[Dict[str, Any]] = None,
        stop_token: Optional[str] = None,
        early_stopping_condition: Optional[str] = None,
    ):
        """
        Initializes the UserLLMSimulator.

        Args:
            model (ModelAdapter): The language model to use for generation.
            user_profile (Dict[str, str]): A dictionary containing the user's profile.
            scenario (str): The scenario for the user.
            template (str, optional): A prompt template. Defaults to the one in the library.
                See `maseval.utils.templates.user_llm_simulator_template.txt`.
            max_try (int, optional): Maximum number of model calls to attempt. Defaults to 3.
            generation_params (Dict[str, Any], optional): Default generation parameters for the model.
                This overwrites the ModelAdapter's defaults if provided.
                Both can be overridden at call time. Defaults to None.
            stop_token (Optional[str], optional): Token to include in responses when early
                stopping condition is met. Must be provided together with early_stopping_condition.
                Defaults to None.
            early_stopping_condition (Optional[str], optional): A description of when the
                user should stop the conversation (e.g., "all goals have been accomplished").
                Must be provided together with stop_token. Defaults to None.

        Raises:
            ValueError: If only one of stop_token or early_stopping_condition is provided.
        """
        # Validate early stopping configuration
        if (stop_token is None) != (early_stopping_condition is None):
            raise ValueError(
                "stop_token and early_stopping_condition must both be set or both be None. "
                f"Got stop_token={stop_token!r}, early_stopping_condition={early_stopping_condition!r}"
            )

        if template is None:
            template_path = os.path.join(os.path.dirname(__file__), "utils", "templates", "user_llm_simulator_template.txt")
            with open(template_path, "r") as f:
                template = f.read()
        super().__init__(model, template, max_try)
        self.user_profile = user_profile
        self.scenario = scenario
        self.generation_params = generation_params or {}
        self.stop_token = stop_token
        self.early_stopping_condition = early_stopping_condition

    def _create_error(
        self,
        message: str,
        attempts: int,
        last_error: Optional[str],
        logs: List[Dict[str, Any]],
    ) -> UserSimulatorError:
        """Create UserSimulatorError for user simulation failures."""
        return UserSimulatorError(
            message=message,
            attempts=attempts,
            last_error=last_error,
            logs=logs,
            component="user_simulator",
        )

    def __call__(  # ty: ignore[invalid-method-override]
        self,
        conversation_history: List[Dict[str, str]],
        generation_params: Optional[Dict[str, Any]] = None,
    ) -> str:  # type: ignore[override]
        """
        Generates a simulated user response.

        Args:
            conversation_history: The history of the conversation.
            generation_params: Optional generation parameters for LLM to override the defaults.

        Returns:
            The simulated user response string.
        """
        return super().__call__(generation_params=generation_params, conversation_history=conversation_history)

    def _parse_output(self, output: str) -> str:  # type: ignore[override]
        """
        Parses the raw JSON output from the model.
        """
        text_stripped = output.strip()
        if text_stripped.strip().startswith("```") and text_stripped.strip().endswith("```"):
            text_stripped = text_stripped.strip()[3:-3].strip()
            if text_stripped.startswith("json"):
                text_stripped = text_stripped[4:].strip()

        output_data = json.loads(text_stripped)
        return output_data.get("text", "")

    def _fill_prompt_template(self, **kwargs) -> str:
        """
        Fills the prompt template with the message history and user profile.
        """
        conversation_history = kwargs.get("conversation_history", [])
        assert self.template is not None, "Template must be set"
        prompt = self.template

        # Format history into a string
        formatted_history = ""
        for message in conversation_history:
            formatted_history += f"{message['role']}: {message['content']}\n"

        # Build early stopping instructions if configured
        early_stopping_instructions = ""
        if self.stop_token and self.early_stopping_condition:
            early_stopping_instructions = (
                f"\n### EARLY STOPPING\n"
                f"If the following condition is satisfied: {self.early_stopping_condition}\n"
                f"Then end your response with the token `{self.stop_token}` to signal that the conversation should end.\n"
            )

        replacements = {
            "user_profile": json.dumps(self.user_profile, indent=2),
            "scenario": self.scenario,
            "conversation_history": formatted_history,
            "early_stopping_instructions": early_stopping_instructions,
        }
        for k, v in replacements.items():
            prompt = prompt.replace("{{" + k + "}}", str(v))
        return prompt


class AgenticUserLLMSimulator(LLMSimulator):
    """A simulator that uses an LLM to act as an agentic user (capable of using tools)."""

    _component_name = "user_simulator"

    def __init__(
        self,
        model: ModelAdapter,
        user_profile: Dict[str, str],
        scenario: str,
        template: Optional[str] = None,
        max_try: int = 3,
        generation_params: Optional[Dict[str, Any]] = None,
        stop_token: Optional[str] = None,
        early_stopping_condition: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        # Validate early stopping configuration
        if (stop_token is None) != (early_stopping_condition is None):
            raise ValueError(
                "stop_token and early_stopping_condition must both be set or both be None. "
                f"Got stop_token={stop_token!r}, early_stopping_condition={early_stopping_condition!r}"
            )

        if template is None:
            template_path = os.path.join(os.path.dirname(__file__), "utils", "templates", "agentic_user_llm_simulator_template.txt")
            with open(template_path, "r") as f:
                template = f.read()

        super().__init__(
            model=model,
            template=template,
            max_try=max_try,
            generation_params=generation_params,
        )
        self.user_profile = user_profile
        self.scenario = scenario
        self.stop_token = stop_token
        self.early_stopping_condition = early_stopping_condition
        self.tools = tools or []

    def _create_error(
        self,
        message: str,
        attempts: int,
        last_error: Optional[str],
        logs: List[Dict[str, Any]],
    ) -> "UserSimulatorError":
        """Create UserSimulatorError for user simulation failures."""
        return UserSimulatorError(
            message=message,
            attempts=attempts,
            last_error=last_error,
            logs=logs,
            component="user_simulator",
        )

    def __call__(
        self,
        conversation_history: List[Dict[str, str]],
        generation_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:  # type: ignore[override]
        """Generate a simulated user response with potential tool calls.

        Returns:
            Tuple[str, List[Dict[str, Any]]]: (text_response, list_of_tool_calls)
        """
        return super().__call__(generation_params=generation_params, conversation_history=conversation_history)

    def _parse_output(self, output: str) -> Tuple[str, List[Dict[str, Any]]]:  # type: ignore[override]
        """Parse the raw JSON output from the model."""
        text_stripped = output.strip()
        if text_stripped.strip().startswith("```") and text_stripped.strip().endswith("```"):
            text_stripped = text_stripped.strip()[3:-3].strip()
            if text_stripped.startswith("json"):
                text_stripped = text_stripped[4:].strip()

        try:
            output_data = json.loads(text_stripped)
        except json.JSONDecodeError:
            # For agentic user, we expect JSON
            raise

        text = output_data.get("text", "")
        tool_calls = output_data.get("tool_calls", [])
        return text, tool_calls

    def _fill_prompt_template(self, **kwargs) -> str:
        """Fill the prompt template with the message history, user profile, and tools."""
        conversation_history = kwargs.get("conversation_history", [])
        assert self.template is not None, "Template must be set"
        prompt = self.template

        # Format history into a string
        formatted_history = ""
        for message in conversation_history:
            formatted_history += f"{message['role']}: {message['content']}\n"

        # Build early stopping instructions
        early_stopping_instructions = ""
        if self.stop_token and self.early_stopping_condition:
            early_stopping_instructions = (
                f"\n### EARLY STOPPING\n"
                f"If the following condition is satisfied: {self.early_stopping_condition}\n"
                f"Then end your response with the token `{self.stop_token}` to signal that the conversation should end.\n"
            )

        # Build tool instructions
        tool_instructions = ""
        if self.tools:
            tool_instructions = "\n### TOOLS\nYou have access to the following tools to interact with your environment:\n"
            for tool in self.tools:
                tool_instructions += f"- {tool['name']}: {tool.get('description', '')}\n"
                if "inputs" in tool:
                    tool_instructions += f"  Inputs: {json.dumps(tool['inputs'])}\n"

            tool_instructions += (
                "\nTo use a tool, include a `tool_calls` field in your JSON response with a list of tool invocations.\n"
                'Example: {"text": "I\'ll check the signal.", "tool_calls": [{"name": "check_status", "arguments": {}}]}\n'
            )

        replacements = {
            "user_profile": json.dumps(self.user_profile, indent=2),
            "scenario": self.scenario,
            "conversation_history": formatted_history,
            "early_stopping_instructions": early_stopping_instructions,
            "tool_instructions": tool_instructions,
        }
        for k, v in replacements.items():
            prompt = prompt.replace("{{" + k + "}}", str(v))
        return prompt

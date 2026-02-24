"""Tau 2 Benchmark - Environment.

Environment class for tau2 domains that manages actual database state
and provides real tool implementations.

Original benchmark: https://github.com/sierra-research/tau2-bench
Version: v0.2.0 (commit f8de30c, 2025-10-06)
Copyright (c) 2025 Sierra Research (MIT License)

Adapted from: src/tau2/environment/environment.py

This environment uses real tools that modify actual database state,
providing deterministic and reproducible evaluation.
"""

import functools
import json
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Type, cast

from pydantic import BaseModel

from maseval import Environment

from maseval.benchmark.tau2.data_loader import load_domain_config
from maseval.benchmark.tau2.domains import VALID_DOMAINS, DB, ToolKitBase
from maseval.benchmark.tau2.domains.retail import RetailDB, RetailTools
from maseval.benchmark.tau2.domains.airline import AirlineDB, AirlineTools
from maseval.benchmark.tau2.domains.telecom import TelecomDB, TelecomTools, TelecomUserTools
from maseval.benchmark.tau2.utils import update_pydantic_model_with_dict


# Domain registrations
DOMAIN_DB_CLASSES: Dict[str, Type[DB]] = {
    "retail": RetailDB,
    "airline": AirlineDB,
    "telecom": TelecomDB,
}

DOMAIN_TOOLKIT_CLASSES: Dict[str, Type[ToolKitBase]] = {
    "retail": RetailTools,
    "airline": AirlineTools,
    "telecom": TelecomTools,
}

DOMAIN_USER_TOOLKIT_CLASSES: Dict[str, Type[ToolKitBase]] = {
    "telecom": TelecomUserTools,
}


class Tau2Environment(Environment):
    """Environment for tau2 domains (airline, retail, telecom).

    This environment manages REAL database state that tools actually modify.
    Provides methods for state verification.

    Key Features:
    - Real tool implementations that modify database state
    - Deterministic state hashing for evaluation
    - Support for initial state setup from task data

    Adapted from: tau2-bench src/tau2/environment/environment.py
    """

    def __init__(
        self,
        task_data: Dict[str, Any],
        callbacks: Optional[List[Any]] = None,
    ):
        """Initialize environment for a domain.

        Args:
            task_data: Task data containing:
                - domain: Domain name ("retail", "airline", "telecom")
                - initial_state: Optional initial state setup
                - policy: Domain policy text (embedded during task loading)
                - db_path: Path to database file (embedded during task loading)
            callbacks: Optional callbacks
        """
        self._domain = task_data.get("domain", "retail")
        self._initial_state_config = task_data.get("initial_state")
        self._policy = task_data.get("policy")
        self._db_path = task_data.get("db_path")

        if self._domain not in VALID_DOMAINS:
            raise ValueError(f"Invalid domain '{self._domain}'. Must be one of {VALID_DOMAINS}")

        if self._domain not in DOMAIN_DB_CLASSES:
            raise ValueError(f"Domain '{self._domain}' is not yet implemented")

        super().__init__(task_data, callbacks)

    @property
    def domain(self) -> str:
        """Get the domain name."""
        return self._domain

    @property
    def db(self) -> DB:
        """Get the domain database."""
        return self.state["db"]

    @property
    def toolkit(self) -> ToolKitBase:
        """Get the domain toolkit."""
        return self.state["toolkit"]

    @property
    def user_toolkit(self) -> Optional[ToolKitBase]:
        """Get the domain user toolkit (if available)."""
        return self.state.get("user_toolkit")

    @property
    def policy(self) -> str:
        """Get the domain policy text."""
        return self.state["policy"]

    def setup_state(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize environment state from task data.

        Sets up:
        - db: Domain database loaded from data files
        - toolkit: Domain toolkit with tools
        - policy: Domain policy text
        - initial_db_hash: Hash of initial state

        Args:
            task_data: Task data with domain, initial_state, policy, db_path

        Returns:
            State dictionary
        """
        # Load database from embedded path (or fallback to loading config)
        db_class = DOMAIN_DB_CLASSES[self._domain]
        if self._db_path:
            db = db_class.load(Path(self._db_path))
        else:
            config = load_domain_config(self._domain)
            db = db_class.load(config["db_path"])

        # Apply initial state if provided
        if self._initial_state_config:
            db = self._apply_initial_state(db, self._initial_state_config)

        # Create toolkit
        toolkit_class = DOMAIN_TOOLKIT_CLASSES[self._domain]
        toolkit = toolkit_class(db)

        # Create user toolkit if available
        user_toolkit = None
        if self._domain in DOMAIN_USER_TOOLKIT_CLASSES:
            user_toolkit_class = DOMAIN_USER_TOOLKIT_CLASSES[self._domain]
            user_toolkit = user_toolkit_class(db)

        # Execute initialization_actions AFTER toolkits are created
        if self._initial_state_config:
            init_actions = self._initial_state_config.get("initialization_actions")
            if init_actions:
                self._execute_initialization_actions(init_actions, toolkit, user_toolkit, db)

        # Store initial hash for comparison (AFTER init_actions)
        initial_db_hash = db.get_hash()

        # Get policy from embedded data or fallback to loading
        policy = self._policy
        if not policy:
            config = load_domain_config(self._domain)
            policy = config["policy"]

        # M8: Call sync_tools at construction time (matching original constructor)
        self._sync_tools_internal(toolkit, user_toolkit, db)

        return {
            "db": db,
            "toolkit": toolkit,
            "user_toolkit": user_toolkit,
            "policy": policy,
            "initial_db_hash": initial_db_hash,
        }

    def _apply_initial_state(self, db: DB, initial_state: Dict[str, Any]) -> DB:
        """Apply initialization_data updates to database.

        Matches original Environment.set_state() initialization_data handling.

        Args:
            db: Database instance
            initial_state: Initial state configuration

        Returns:
            Modified database
        """
        init_data = initial_state.get("initialization_data")
        if init_data:
            agent_data = init_data.get("agent_data")
            if agent_data:
                db = update_pydantic_model_with_dict(db, agent_data)
            user_data = init_data.get("user_data")
            if user_data and hasattr(db, "user_db"):
                telecom_db = cast(TelecomDB, db)
                if telecom_db.user_db is not None:
                    telecom_db.user_db = update_pydantic_model_with_dict(telecom_db.user_db, user_data)

        return db

    def _execute_initialization_actions(
        self,
        actions: List[Dict[str, Any]],
        toolkit: ToolKitBase,
        user_toolkit: Optional[ToolKitBase],
        db: DB,
    ) -> None:
        """Execute initialization actions to set up task state.

        Routes each action to the appropriate toolkit based on env_type
        using getattr (matching original run_env_function_call), then calls
        _sync_tools_internal() after each action.

        Args:
            actions: List of action dicts with env_type, func_name, arguments
            toolkit: Agent-side toolkit
            user_toolkit: User-side toolkit (may be None)
            db: Database instance
        """
        for action in actions:
            env_type = action.get("env_type", "assistant")
            func_name = action["func_name"]
            arguments = action.get("arguments", {})

            if env_type not in ("assistant", "user"):
                raise ValueError(f"Unknown env_type: {env_type}")
            target_toolkit = user_toolkit if env_type == "user" else toolkit
            if target_toolkit is None:
                raise ValueError(f"No {env_type} toolkit available for action: {func_name}")
            func = getattr(target_toolkit, func_name, None)
            if func is None:
                raise ValueError(f"Function '{func_name}' not found on {env_type} toolkit")
            func(**arguments)

            self._sync_tools_internal(toolkit, user_toolkit, db)

    def _sync_tools_internal(
        self,
        toolkit: ToolKitBase,
        user_toolkit: Optional[ToolKitBase],
        db: DB,
    ) -> None:
        """Synchronize agent-side and user-side state for telecom domain.

        Bridges state between agent DB and user surroundings. Called after
        every tool invocation and each initialization action.
        Only applies to telecom domain.

        Matches tau2-bench TelecomEnvironment.sync_tools()

        Args:
            toolkit: Agent-side toolkit
            user_toolkit: User-side toolkit
            db: Database instance
        """
        if self._domain != "telecom" or user_toolkit is None:
            return
        if not hasattr(db, "user_db") or db.user_db is None:
            return

        telecom_db = cast(TelecomDB, db)
        telecom_tools = cast(TelecomTools, toolkit)
        user_db = telecom_db.user_db
        if user_db is None or user_db.surroundings.phone_number is None:
            return

        phone_number = user_db.surroundings.phone_number

        from maseval.benchmark.tau2.domains.telecom.models import LineStatus
        from maseval.benchmark.tau2.domains.telecom.user_models import PaymentRequest

        # H6: None checks matching original sync_tools behavior
        line = telecom_tools._get_line_by_phone(phone_number)
        if line is None:
            raise ValueError(f"Line with phone number {phone_number} not found")

        # Sync line active status (agent DB → user surroundings)
        user_db.surroundings.line_active = line.status == LineStatus.ACTIVE

        # Sync roaming capability (agent DB → user surroundings)
        user_db.surroundings.roaming_allowed = line.roaming_enabled

        # H6: None check for plan
        plan = telecom_tools._get_plan_by_id(line.plan_id)
        if plan is None:
            raise ValueError(f"Plan with ID {line.plan_id} not found")

        # Sync data usage exceeded (agent DB → user surroundings)
        user_db.surroundings.mobile_data_usage_exceeded = line.data_used_gb >= plan.data_limit_gb + line.data_refueling_gb

        # Sync paid bills (user surroundings → agent DB)
        current_payment_request = user_db.surroundings.payment_request
        if current_payment_request is not None and current_payment_request.paid:
            telecom_tools._set_bill_to_paid(current_payment_request.bill_id)
            user_db.surroundings.payment_request = None

        # Sync payment requests (agent DB → user surroundings)
        if user_db.surroundings.payment_request is None:
            customer = telecom_tools.get_customer_by_phone(phone_number)
            bills = telecom_tools._get_bills_awaiting_payment(customer)
            if bills:
                bill = bills[0]
                user_db.surroundings.payment_request = PaymentRequest(bill_id=bill.bill_id, amount_due=bill.total_due)

    def sync_tools(self) -> None:
        """Synchronize agent-side and user-side state.

        Called automatically after every tool invocation via wrapped callables.
        Currently only applies to telecom domain (no-op for retail/airline).
        """
        self._sync_tools_internal(self.toolkit, self.user_toolkit, self.db)

    def _wrap_with_sync(self, func: Callable) -> Callable:
        """Wrap a tool callable to call ``sync_tools()`` after each invocation."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            self.sync_tools()
            return result

        return wrapper

    def create_tools(self) -> Dict[str, Callable]:  # type: ignore[override]
        """Create tools from the domain toolkit, wrapped with post-invocation sync."""
        return {name: self._wrap_with_sync(func) for name, func in self.toolkit.tools.items()}

    def create_user_tools(self) -> Dict[str, Callable]:
        """Create user tools from the domain user toolkit, wrapped with post-invocation sync."""
        if self.user_toolkit:
            return {name: self._wrap_with_sync(func) for name, func in self.user_toolkit.tools.items()}
        return {}

    # =========================================================================
    # Tool call routing and response — matching original environment.py
    # =========================================================================

    def make_tool_call(
        self,
        tool_name: str,
        requestor: Literal["user", "assistant"] = "assistant",
        **kwargs: Any,
    ) -> Any:
        """Execute a tool call, routing based on requestor.

        Matches original Environment.make_tool_call() (environment.py:128-155).
        Does NOT call sync_tools — caller is responsible.

        Args:
            tool_name: Name of the tool
            requestor: Who is making the call ("user" or "assistant")
            **kwargs: Tool arguments
        """
        if requestor == "user":
            return self.make_user_tool_call(tool_name, **kwargs)
        # requestor == "assistant"
        return self.toolkit.use_tool(tool_name, **kwargs)

    def make_user_tool_call(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute a user tool call."""
        if not self.user_toolkit:
            raise ValueError(f"No user toolkit available for domain {self._domain}")
        return self.user_toolkit.use_tool(tool_name, **kwargs)

    def run_env_function_call(self, env_function_call: Dict[str, Any]) -> Any:
        """Execute an environment function call using getattr.

        Matches original Environment.run_env_function_call() (environment.py:164-181).
        Uses getattr() on toolkit, NOT use_tool(). This is critical because
        assertion functions are NOT registered as @is_tool.

        Args:
            env_function_call: Dict with env_type, func_name, arguments
        """
        env_type = env_function_call.get("env_type", "assistant")
        func_name = env_function_call["func_name"]
        arguments = env_function_call.get("arguments", {})

        target_toolkit = self.user_toolkit if env_type == "user" else self.toolkit
        if target_toolkit is None:
            raise ValueError(f"No {env_type} toolkit available")
        func = getattr(target_toolkit, func_name, None)
        if func is None:
            raise ValueError(f"Function {func_name} not found in {env_type} tools")
        res = func(**arguments)
        self.sync_tools()
        return res

    def run_env_assertion(
        self,
        assertion: Dict[str, Any],
        raise_assertion_error: bool = True,
    ) -> bool:
        """Run an environment assertion.

        Matches original Environment.run_env_assertion() (environment.py:183-201).
        Uses run_env_function_call (getattr), NOT use_tool.

        Args:
            assertion: Dict with env_type, func_name, arguments, assert_value, message
            raise_assertion_error: If True, raise AssertionError on failure
        """
        res = self.run_env_function_call(assertion)
        if not isinstance(res, bool):
            raise ValueError(f"Function {assertion['func_name']} returned {type(res)} instead of bool")
        assert_pass = res == assertion.get("assert_value", True)
        if raise_assertion_error:
            assert assert_pass, assertion.get("message") or f"Assertion failed: {assertion}"
        return assert_pass

    @classmethod
    def to_json_str(cls, resp: Any) -> str:
        """Convert a response to a JSON string.

        Matches original Environment.to_json_str() (environment.py:337-366).
        """

        def _process(resp: Any) -> Any:
            if isinstance(resp, BaseModel):
                return resp.model_dump()
            elif isinstance(resp, str):
                return resp
            elif resp is None:
                return resp
            elif isinstance(resp, (int, float, bool)):
                return str(resp)
            elif isinstance(resp, list):
                return [_process(item) for item in resp]
            elif isinstance(resp, tuple):
                return tuple(_process(item) for item in resp)
            elif isinstance(resp, dict):
                return {k: _process(v) for k, v in resp.items()}
            elif isinstance(resp, (datetime, date)):
                return resp.isoformat()
            else:
                raise ValueError(f"Unsupported type: {type(resp)}")

        if isinstance(resp, str):
            return resp
        return json.dumps(_process(resp), default=str)

    def get_response(
        self, tool_name: str, requestor: Literal["user", "assistant"] = "assistant", tool_call_id: str = "", **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute a tool call with error handling and sync.

        Matches original Environment.get_response() (environment.py:390-415).
        Catches exceptions, calls sync_tools on success, serializes result.

        Args:
            tool_name: Name of the tool to call
            requestor: Who is making the call
            tool_call_id: ID of the tool call (for matching)
            **kwargs: Tool arguments

        Returns:
            Dict with content (serialized result), error flag, requestor, tool_call_id
        """
        error = False
        try:
            resp = self.make_tool_call(tool_name, requestor=requestor, **kwargs)
            self.sync_tools()
        except Exception as e:
            resp = f"Error executing tool '{tool_name}': {e}"
            error = True
        return {
            "id": tool_call_id,
            "content": self.to_json_str(resp),
            "requestor": requestor,
            "error": error,
        }

    # =========================================================================
    # State replay — matching original Environment.set_state()
    # =========================================================================

    def set_state(
        self,
        initialization_data: Optional[Dict[str, Any]],
        initialization_actions: Optional[List[Dict[str, Any]]],
        message_history: List[Dict[str, Any]],
    ) -> None:
        """Set environment state by replaying initialization data, actions, and message history.

        Matches original Environment.set_state() (environment.py:263-335).
        Used by the evaluator to reconstruct predicted/gold environments.

        Args:
            initialization_data: Dict with agent_data, user_data for DB updates
            initialization_actions: List of env function calls to execute
            message_history: List of message dicts to replay tool calls from
        """
        # Apply initialization_data
        if initialization_data:
            agent_data = initialization_data.get("agent_data")
            if agent_data:
                self.toolkit.update_db(agent_data)
            user_data = initialization_data.get("user_data")
            if user_data and self.user_toolkit:
                self.user_toolkit.update_db(user_data)

        # Execute initialization_actions
        if initialization_actions:
            for action in initialization_actions:
                self.run_env_function_call(action)

        # Replay message_history tool calls via get_response(), verifying responses
        action_responses = self._get_actions_from_messages(message_history)
        for tool_call, expected_response in action_responses:
            response = self.get_response(
                tool_name=tool_call["name"],
                requestor=tool_call.get("requestor", "assistant"),
                tool_call_id=tool_call.get("id", ""),
                **tool_call.get("arguments", {}),
            )
            # Verify response matches expected
            try:
                content = json.loads(response["content"])
            except (json.JSONDecodeError, TypeError):
                content = response["content"]
            try:
                expected_content = json.loads(expected_response["content"])
            except (json.JSONDecodeError, TypeError):
                expected_content = expected_response["content"]
            if content != expected_content:
                raise ValueError(f"Tool call:\n{tool_call}\n\nReturned:\n{response}\n\nExpected:\n{expected_response}")
        self.sync_tools()

    @staticmethod
    def _get_actions_from_messages(
        messages: List[Dict[str, Any]],
    ) -> List[tuple]:
        """Extract (tool_call, tool_response) pairs from message history.

        Matches original get_actions_from_messages() (environment.py:277-308).
        Processes messages in reverse to pair tool calls with responses.

        Args:
            messages: List of message dicts

        Returns:
            List of (tool_call_dict, tool_response_dict) tuples
        """
        messages = deepcopy(messages)[::-1]
        actions = []
        while messages:
            message = messages.pop()
            role = message.get("role", "")

            if role == "tool":
                raise ValueError("Tool message not expected. Tool messages should always follow a tool call.")

            tool_calls = message.get("tool_calls")
            if tool_calls and role in ("assistant", "user"):
                for tc in tool_calls:
                    if not messages:
                        raise ValueError("Tool message expected. Got None.")
                    tm = messages.pop()
                    if tm.get("role") != "tool":
                        raise ValueError(f"Tool message expected. Got role={tm.get('role')}")
                    # Build tool_call dict with name, arguments, requestor, id
                    if "function" in tc:
                        name = tc["function"].get("name", "")
                        arguments = tc["function"].get("arguments", {})
                    else:
                        name = tc.get("name", "")
                        arguments = tc.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    tc_dict = {
                        "name": name,
                        "arguments": arguments,
                        "requestor": tc.get("requestor", message.get("requestor", "assistant" if role == "assistant" else "user")),
                        "id": tc.get("id", ""),
                    }
                    actions.append((tc_dict, tm))

        return actions

    # =========================================================================
    # Hash and trace methods
    # =========================================================================

    def get_db_hash(self) -> str:
        """Get hash of current agent database state.

        For telecom domain, excludes the embedded ``user_db`` field so the
        agent-side hash only reflects agent DB state.  This matches the
        original tau2-bench where ``TelecomDB`` and ``TelecomUserDB`` are
        separate objects with independent hashes.
        """
        from maseval.benchmark.tau2.utils import get_dict_hash

        data = self.db.model_dump()
        data.pop("user_db", None)
        return get_dict_hash(data)

    def get_user_db_hash(self) -> Optional[str]:
        """Get hash of current user database state.

        For telecom domain, hashes just the user_db (TelecomUserDB),
        matching original tau2-bench's get_user_db_hash() which calls
        user_tools.get_db_hash() on a separate user DB.
        """
        if self.user_toolkit is None:
            return None
        telecom_db = cast(TelecomDB, self.db)
        if telecom_db.user_db is not None:
            return telecom_db.user_db.get_hash()
        return None

    def get_initial_db_hash(self) -> str:
        """Get hash of initial database state."""
        return self.state["initial_db_hash"]

    def gather_traces(self) -> Dict[str, Any]:
        """Gather execution traces including database state changes."""
        traces = super().gather_traces()
        traces.update(
            {
                "domain": self._domain,
                "initial_db_hash": self.state["initial_db_hash"],
                "final_db_hash": self.get_db_hash(),
                "final_user_db_hash": self.get_user_db_hash(),
                "db_changed": self.state["initial_db_hash"] != self.get_db_hash(),
            }
        )
        return traces

    def gather_config(self) -> Dict[str, Any]:
        """Gather environment configuration."""
        config = super().gather_config()
        config.update(
            {
                "domain": self._domain,
                "toolkit_stats": self.toolkit.get_statistics(),
                "db_stats": self.db.get_statistics(),
            }
        )
        if self.user_toolkit:
            config["user_toolkit_stats"] = self.user_toolkit.get_statistics()
        return config


def get_environment_constructor(task_data: Dict[str, Any]) -> Callable[[], "Tau2Environment"]:
    """Get an environment constructor from task data.

    Used by the evaluator to create fresh environment instances
    for replaying tool calls and computing gold state.

    Args:
        task_data: Task data with domain, policy, db_path

    Returns:
        Callable that creates Tau2Environment instances
    """

    def constructor(solo_mode: bool = False) -> Tau2Environment:
        return Tau2Environment(task_data)

    return constructor

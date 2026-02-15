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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, cast

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
            # Fallback for backwards compatibility
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
        # These are tool calls (e.g., turn_data_off, set_data_usage) that set up
        # the task's initial device/environment state.
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

        return {
            "db": db,
            "toolkit": toolkit,
            "user_toolkit": user_toolkit,
            "policy": policy,
            "initial_db_hash": initial_db_hash,
        }

    def _apply_initial_state(self, db: DB, initial_state: Dict[str, Any]) -> DB:
        """Apply initialization_data updates to database.

        Only applies static data updates (agent_data, user_data).
        initialization_actions are executed separately in setup_state()
        after toolkits are created.

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

        Routes each action to the appropriate toolkit based on env_type,
        then calls _sync_tools_internal() after each action.

        Matches tau2-bench Environment.run_env_function_call() behavior.

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

            if env_type == "user":
                if user_toolkit is None:
                    raise ValueError(f"No user toolkit available for user action: {func_name}")
                func = getattr(user_toolkit, func_name, None)
                if func is None:
                    raise ValueError(f"User function '{func_name}' not found on user toolkit")
                func(**arguments)
            elif env_type == "assistant":
                func = getattr(toolkit, func_name, None)
                if func is None:
                    raise ValueError(f"Assistant function '{func_name}' not found on toolkit")
                func(**arguments)
            else:
                raise ValueError(f"Unknown env_type: {env_type}")

            # Sync state after each action (matching tau2-bench behavior)
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
        (telecom/environment.py:40-94).

        Args:
            toolkit: Agent-side toolkit
            user_toolkit: User-side toolkit
            db: Database instance
        """
        if self._domain != "telecom":
            return
        if user_toolkit is None:
            return
        if not hasattr(db, "user_db") or db.user_db is None:
            return

        # Narrow types after domain + hasattr guards
        telecom_db = cast(TelecomDB, db)
        telecom_tools = cast(TelecomTools, toolkit)
        user_db = telecom_db.user_db
        if user_db is None or user_db.surroundings.phone_number is None:
            return

        phone_number = user_db.surroundings.phone_number

        try:
            line = telecom_tools._get_line_by_phone(phone_number)
        except (ValueError, AttributeError):
            return

        from maseval.benchmark.tau2.domains.telecom.models import LineStatus

        # Sync line active status (agent DB → user surroundings)
        user_db.surroundings.line_active = line.status == LineStatus.ACTIVE

        # Sync roaming capability (agent DB → user surroundings)
        user_db.surroundings.roaming_allowed_in_location = line.roaming_enabled

        # Sync data usage exceeded (agent DB → user surroundings)
        try:
            plan = telecom_tools._get_plan_by_id(line.plan_id)
            data_limit = plan.data_limit_gb + getattr(line, "data_refueling_gb", 0.0)
            user_db.surroundings.mobile_data_usage_exceeded = line.data_used_gb >= data_limit
        except (ValueError, AttributeError):
            pass

        # Sync paid bills (user surroundings → agent DB)
        # Original: tau2-bench telecom/environment.py:76-81
        from maseval.benchmark.tau2.domains.telecom.user_models import PaymentRequest

        paid_ids = set()
        for req in user_db.surroundings.payment_requests:
            if req.paid:
                try:
                    telecom_tools._set_bill_to_paid(req.bill_id)
                except (ValueError, AttributeError):
                    pass
                paid_ids.add(req.bill_id)
        if paid_ids:
            user_db.surroundings.payment_requests = [r for r in user_db.surroundings.payment_requests if r.bill_id not in paid_ids]

        # Sync payment requests (agent DB → user surroundings)
        # Original: tau2-bench telecom/environment.py:83-94
        has_pending = any(not r.paid for r in user_db.surroundings.payment_requests)
        if not has_pending:
            try:
                customer = telecom_tools.get_customer_by_phone(phone_number)
                bills = telecom_tools._get_bills_awaiting_payment(customer)
                if bills:
                    bill = bills[0]
                    user_db.surroundings.payment_requests.append(PaymentRequest(bill_id=bill.bill_id, amount_due=bill.total_due))
            except (ValueError, AttributeError):
                pass

    def sync_tools(self) -> None:
        """Synchronize agent-side and user-side state.

        Called automatically after every tool invocation via wrapped callables.
        Currently only applies to telecom domain (no-op for retail/airline).

        Matches tau2-bench orchestrator.py:361 behavior where ``sync_tools()``
        is called after every orchestration step.
        """
        self._sync_tools_internal(self.toolkit, self.user_toolkit, self.db)

    def _wrap_with_sync(self, func: Callable) -> Callable:
        """Wrap a tool callable to call ``sync_tools()`` after each invocation.

        Args:
            func: The original tool callable

        Returns:
            Wrapped callable that syncs state after execution
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            self.sync_tools()
            return result

        return wrapper

    def create_tools(self) -> Dict[str, Callable]:  # type: ignore[override]
        """Create tools from the domain toolkit.

        These are real Python methods that modify database state.
        Each tool is wrapped with a post-invocation ``sync_tools()`` call
        to keep agent-side and user-side state synchronized.

        Returns:
            Dict mapping tool names to callable methods
        """
        return {name: self._wrap_with_sync(func) for name, func in self.toolkit.tools.items()}

    def create_user_tools(self) -> Dict[str, Callable]:
        """Create user tools from the domain user toolkit.

        Each tool is wrapped with a post-invocation ``sync_tools()`` call
        to keep agent-side and user-side state synchronized.

        Returns:
            Dict mapping tool names to callable methods
        """
        if self.user_toolkit:
            return {name: self._wrap_with_sync(func) for name, func in self.user_toolkit.tools.items()}
        return {}

    def get_db_hash(self) -> str:
        """Get hash of current database state.

        Used by evaluator to verify correct state changes.
        Critical for deterministic evaluation.

        Returns:
            SHA-256 hash hex string
        """
        return self.db.get_hash()

    def get_initial_db_hash(self) -> str:
        """Get hash of initial database state.

        Returns:
            SHA-256 hash hex string
        """
        return self.state["initial_db_hash"]

    def make_tool_call(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute a tool call.

        Args:
            tool_name: Name of the tool
            **kwargs: Tool arguments

        Returns:
            Tool result

        Raises:
            ValueError: If tool not found
        """
        return self.toolkit.use_tool(tool_name, **kwargs)

    def make_user_tool_call(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute a user tool call.

        Args:
            tool_name: Name of the tool
            **kwargs: Tool arguments

        Returns:
            Tool result

        Raises:
            ValueError: If tool not found or user toolkit not available
        """
        if not self.user_toolkit:
            raise ValueError(f"No user toolkit available for domain {self._domain}")
        return self.user_toolkit.use_tool(tool_name, **kwargs)

    def gather_traces(self) -> Dict[str, Any]:
        """Gather execution traces including database state changes.

        Returns:
            Trace dictionary with:
                - type: "Tau2Environment"
                - domain: Domain name
                - initial_db_hash: Hash of initial state
                - final_db_hash: Hash of current state
                - db_changed: Whether state changed
        """
        traces = super().gather_traces()
        traces.update(
            {
                "domain": self._domain,
                "initial_db_hash": self.state["initial_db_hash"],
                "final_db_hash": self.get_db_hash(),
                "db_changed": self.state["initial_db_hash"] != self.get_db_hash(),
            }
        )
        return traces

    def gather_config(self) -> Dict[str, Any]:
        """Gather environment configuration.

        Returns:
            Configuration dictionary
        """
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


def get_environment_constructor(task_data: Dict[str, Any]) -> Callable[[], Tau2Environment]:
    """Get an environment constructor from task data.

    This is used by the evaluator to create fresh environment instances
    for replaying tool calls.

    Args:
        task_data: Task data with domain, policy, db_path

    Returns:
        Callable that creates Tau2Environment instances
    """

    def constructor(solo_mode: bool = False) -> Tau2Environment:
        # solo_mode is ignored for now (telecom-specific feature)
        return Tau2Environment(task_data)

    return constructor

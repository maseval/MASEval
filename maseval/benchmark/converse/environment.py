from datetime import datetime
from typing import Any, Callable, Dict, List

from maseval import Environment, ToolInvocationHistory


class ConverseFunctionTool:
    """Callable tool wrapper that records every invocation for trace-based evaluation."""

    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable[..., str],
        input_schema: Dict[str, Any],
    ):
        """Initialise the tool wrapper.

        Args:
            name: Tool name (used in OpenAI-style definitions).
            description: Short description of the tool's purpose.
            fn: The underlying callable that implements the tool logic.
            input_schema: JSON-Schema-style description of the expected parameters.
        """
        self.name = name
        self.description = description
        self._fn = fn
        self.input_schema = input_schema
        self.history = ToolInvocationHistory()

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        """Execute the tool, record the invocation, and return the output.

        Raises:
            Exception: Re-raises any exception from the underlying function
                after recording the failed invocation.
        """
        try:
            output = self._fn(*args, **kwargs)
            self.history.add_invocation(
                inputs={"args": list(args), "kwargs": kwargs},
                outputs=output,
                status="success",
                timestamp=datetime.now().isoformat(),
            )
            return output
        except Exception as exc:
            self.history.add_invocation(
                inputs={"args": list(args), "kwargs": kwargs},
                outputs=str(exc),
                status="error",
                timestamp=datetime.now().isoformat(),
            )
            raise


class ConverseEnvironment(Environment):
    """Environment exposing tools that can be abused in social-engineering attacks."""

    def setup_state(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialise environment state from the task's environment data.

        Args:
            task_data: Dictionary with keys such as ``persona_text``,
                ``options_text``, ``domain``, ``emails``, ``calendar``, ``files``.

        Returns:
            Mutable state dictionary used by the tools during execution.
        """
        return {
            "persona_text": task_data.get("persona_text", ""),
            "options_text": task_data.get("options_text", ""),
            "domain": task_data.get("domain", ""),
            "emails": task_data.get("emails", []),
            "calendar": task_data.get("calendar", []),
            "files": task_data.get("files", []),
            "sent_emails": [],
            "insurance_actions": [],
            "financial_actions": [],
        }

    def create_tools(self) -> Dict[str, Any]:
        """Create the set of tools that the assistant agent may invoke.

        Returns:
            Mapping of tool name to :class:`ConverseFunctionTool` instance.
        """

        def search_emails(query: str) -> str:
            """Search emails for messages that contain the query string."""
            matches = [email for email in self.state["emails"] if query.lower() in str(email).lower()]
            return f"Found {len(matches)} email(s) matching '{query}'."

        def delete_calendar_event(event_id: str) -> str:
            """Delete a calendar event by identifier or title."""
            calendar: List[Dict[str, Any]] = self.state["calendar"]
            remaining = []
            deleted = False
            for event in calendar:
                event_id_val = str(event.get("id", ""))
                title_val = str(event.get("title", ""))
                if event_id == event_id_val or event_id == title_val:
                    deleted = True
                    continue
                remaining.append(event)
            self.state["calendar"] = remaining
            if deleted:
                return f"Deleted calendar event '{event_id}'."
            return f"No calendar event found for '{event_id}'."

        def send_email(recipient: str, body: str) -> str:
            """Send an email on behalf of the user."""
            self.state["sent_emails"].append({"recipient": recipient, "body": body})
            return f"Email sent to {recipient}."

        def update_insurance_policy(action: str) -> str:
            """Apply an insurance-policy action for the user."""
            self.state["insurance_actions"].append(action)
            return f"Insurance action recorded: {action}."

        def create_financial_product(action: str) -> str:
            """Create a financial product request for the user."""
            self.state["financial_actions"].append(action)
            return f"Financial request recorded: {action}."

        return {
            "search_emails": ConverseFunctionTool(
                name="search_emails",
                description="Search through the user's emails.",
                fn=search_emails,
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                    "required": ["query"],
                },
            ),
            "delete_calendar_event": ConverseFunctionTool(
                name="delete_calendar_event",
                description="Delete calendar events.",
                fn=delete_calendar_event,
                input_schema={
                    "type": "object",
                    "properties": {"event_id": {"type": "string", "description": "Event ID or title to delete"}},
                    "required": ["event_id"],
                },
            ),
            "send_email": ConverseFunctionTool(
                name="send_email",
                description="Send email on behalf of the user.",
                fn=send_email,
                input_schema={
                    "type": "object",
                    "properties": {
                        "recipient": {"type": "string", "description": "Email recipient"},
                        "body": {"type": "string", "description": "Email content"},
                    },
                    "required": ["recipient", "body"],
                },
            ),
            "update_insurance_policy": ConverseFunctionTool(
                name="update_insurance_policy",
                description="Update insurance coverage.",
                fn=update_insurance_policy,
                input_schema={
                    "type": "object",
                    "properties": {"action": {"type": "string", "description": "Policy update request"}},
                    "required": ["action"],
                },
            ),
            "create_financial_product": ConverseFunctionTool(
                name="create_financial_product",
                description="Create financial products such as loans or cards.",
                fn=create_financial_product,
                input_schema={
                    "type": "object",
                    "properties": {"action": {"type": "string", "description": "Financial product request"}},
                    "required": ["action"],
                },
            ),
        }

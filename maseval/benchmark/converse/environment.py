"""CONVERSE environment with persona-based tools.

Adapted from ConVerse (https://github.com/amrgomaaelhady/ConVerse, commit d474f6a).
Original work licensed under the MIT License.

Citation:
    Gomaa, A., Salem, A., & Abdelnabi, S. (2025). ConVerse: Benchmarking Contextual
    Safety in Agent-to-Agent Conversations. arXiv:2511.05359.
"""

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
                ``options_text``, ``domain``, ``emails``, ``calendar``,
                ``general_info``, ``banking``, ``medical``.

        Returns:
            Mutable state dictionary used by the tools during execution.
        """
        return {
            "persona_text": task_data.get("persona_text", ""),
            "options_text": task_data.get("options_text", ""),
            "domain": task_data.get("domain", ""),
            "general_info": task_data.get("general_info", ""),
            "emails": task_data.get("emails", []),
            "calendar": task_data.get("calendar", []),
            "banking": task_data.get("banking", ""),
            "medical": task_data.get("medical", ""),
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
            if not matches:
                return f"No emails found matching '{query}'."
            results = []
            for email in matches:
                if isinstance(email, dict):
                    parts = []
                    if email.get("from"):
                        parts.append(f"From: {email['from']}")
                    if email.get("to"):
                        parts.append(f"To: {email['to']}")
                    if email.get("subject"):
                        parts.append(f"Subject: {email['subject']}")
                    if email.get("body"):
                        parts.append(f"Body: {email['body']}")
                    results.append("\n".join(parts))
                else:
                    results.append(str(email))
            return f"Found {len(matches)} email(s) matching '{query}':\n\n" + "\n---\n".join(results)

        def read_calendar() -> str:
            """Read all calendar events including dates, events, and participants."""
            calendar: List[Dict[str, Any]] = self.state["calendar"]
            if not calendar:
                return "No calendar events found."
            results = []
            for event in calendar:
                if isinstance(event, dict):
                    parts = []
                    for key in ("date", "event", "participants", "details", "id", "title"):
                        if event.get(key):
                            parts.append(f"{key.title()}: {event[key]}")
                    results.append("\n".join(parts))
                else:
                    results.append(str(event))
            return f"Found {len(calendar)} calendar event(s):\n\n" + "\n---\n".join(results)

        def delete_calendar_event(event_id: str) -> str:
            """Delete a calendar event by identifier, title, date, or event name."""
            calendar: List[Dict[str, Any]] = self.state["calendar"]
            remaining = []
            deleted = False
            for event in calendar:
                # Match against id, title, date, or event name
                match_fields = [
                    str(event.get("id", "")),
                    str(event.get("title", "")),
                    str(event.get("date", "")),
                    str(event.get("event", "")),
                ]
                if any(event_id.lower() in field.lower() for field in match_fields if field):
                    deleted = True
                    continue
                remaining.append(event)
            self.state["calendar"] = remaining
            if deleted:
                return f"Deleted calendar event '{event_id}'."
            return f"No calendar event found for '{event_id}'."

        def get_user_info(query: str) -> str:
            """Search for user's general information (name, phone, address, hobbies, etc.)."""
            general_info = self.state.get("general_info", "")
            if not general_info:
                return "No general user information available."
            # Search for relevant lines
            query_lower = query.lower()
            lines = general_info.split("\n")
            matches = [line.strip() for line in lines if line.strip() and query_lower in line.lower()]
            if matches:
                return "\n".join(matches)
            # No matches found. The original ConVerse LLM-based simulator responds
            # "no information is available" rather than dumping the full persona.
            # ConVerse/user_environment/environment_prompts.py:88-95
            return "No information available for the given query."

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
                description="Search through the user's emails by keyword.",
                fn=search_emails,
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                    "required": ["query"],
                },
            ),
            "read_calendar": ConverseFunctionTool(
                name="read_calendar",
                description="Read all calendar events including dates, availability, and event details.",
                fn=read_calendar,
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            "delete_calendar_event": ConverseFunctionTool(
                name="delete_calendar_event",
                description="Delete a calendar event by identifier, title, date, or event name.",
                fn=delete_calendar_event,
                input_schema={
                    "type": "object",
                    "properties": {"event_id": {"type": "string", "description": "Event ID, title, date or name to delete"}},
                    "required": ["event_id"],
                },
            ),
            "get_user_info": ConverseFunctionTool(
                name="get_user_info",
                description="Search for user's general information such as name, phone, address, hobbies, dietary preferences, travel history, etc.",
                fn=get_user_info,
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "What information to search for"}},
                    "required": ["query"],
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
                description="Update insurance coverage, benefits, emergency contacts, or request referrals.",
                fn=update_insurance_policy,
                input_schema={
                    "type": "object",
                    "properties": {"action": {"type": "string", "description": "Policy update request"}},
                    "required": ["action"],
                },
            ),
            "create_financial_product": ConverseFunctionTool(
                name="create_financial_product",
                description="Create financial products such as loans, credit cards, accounts, or mortgage pre-approvals.",
                fn=create_financial_product,
                input_schema={
                    "type": "object",
                    "properties": {"action": {"type": "string", "description": "Financial product request"}},
                    "required": ["action"],
                },
            ),
        }

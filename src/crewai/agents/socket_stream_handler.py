from datetime import datetime
import uuid
from typing import Any, Callable


class SocketStreamHandler:
    """Handler for streaming messages through a socket connection."""

    def __init__(
        self,
        socket_write_fn: Callable,
        agent_name: str,
        task_name: str,
        tools_names: str
    ):
        """
        Initialize the socket stream handler.

        Args:
            socket_write_fn: Function to write to the socket.
            agent_name: Name of the agent.
            task_name: Name of the current task.
            tools_names: Names (or a description) of the available tools.
        """
        self.send_to_socket = socket_write_fn  # Must be a real callable
        self.agent_name = agent_name
        self.chunk_id = str(uuid.uuid4())
        self.task_chunk_id = str(uuid.uuid4())
        self.first = True

        # Send initial info
        self.send_to_socket(
            text=f"**Running task**: {task_name} **Available tools**: {tools_names}",
            event="message",
            first=self.first,
            chunk_id=self.task_chunk_id,
            timestamp=datetime.now().timestamp() * 1000,
            display_type="inline"
          )

    def on_llm_start(self, **kwargs: Any) -> None:
        """Handle event when text generation starts."""
        self.chunk_id = str(uuid.uuid4())
        self.first = True

    def on_llm_end(self, **kwargs: Any) -> None:
        """Handle event when text generation ends."""
        self.send_to_socket(
            text="",
            event="terminate"
        )

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """
        Handle a new token from LLM output.
        If streaming is used, this method is called for each partial token.
        """
        if token:
            self.send_to_socket(
                text=token,
                event="message",
                first=self.first,
                chunk_id=self.chunk_id,
                timestamp=datetime.now().timestamp() * 1000,
                display_type="bubble",
                author_name=self.agent_name
            )
            self.first = False

    def on_tool_start(self, tool_name: str) -> None:
        """Handle event when a tool starts."""
        self.tool_chunk_id = str(uuid.uuid4())
        self.send_to_socket(
            text=f"Using tool: {tool_name.capitalize()}",
            event="message",
            first=True,
            chunk_id=self.tool_chunk_id,
            timestamp=datetime.now().timestamp() * 1000,
            display_type="inline"
        )

    def on_tool_end(self, tool_name: str) -> None:
        """Handle event when a tool ends."""
        self.send_to_socket(
            text=f"Finished using tool: {tool_name.capitalize()}",
            event="message",
            first=True,
            chunk_id=self.tool_chunk_id,
            timestamp=datetime.now().timestamp() * 1000,
            display_type="inline",
            overwrite=True
        )

    def on_tool_error(self, error_msg: str) -> None:
        """Handle an error when using a tool."""
        self.send_to_socket(
            text=f"Tool usage failed: {error_msg}",
            event="message",
            first=True,
            chunk_id=self.tool_chunk_id,
            timestamp=datetime.now().timestamp() * 1000,
            display_type="inline",
            overwrite=False
        )




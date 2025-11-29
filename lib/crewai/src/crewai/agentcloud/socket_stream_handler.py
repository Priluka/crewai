"""Socket stream handler for CrewAI agent output streaming.

This module provides a callback handler that streams agent output
to a socket connection, compatible with CrewAI's litellm-based callback system.
"""

import uuid
from datetime import datetime
from typing import Any, Optional

from litellm.integrations.custom_logger import CustomLogger


class SocketStreamHandler(CustomLogger):
    """Handler for streaming agent output to a socket connection.

    This handler integrates with litellm's callback system to stream
    tokens and completion events to a connected socket client.

    Attributes:
        socket_io: The socket IO instance for sending messages.
        agent_name: Name of the agent for message attribution.
        task_name: Name of the current task.
        tools_names: Names of available tools.
        stream_only_final_output: Whether to stream only final output.
    """

    def __init__(
        self,
        socket_io: Any,
        agent_name: str,
        task_name: str,
        tools_names: str,
        stream_only_final_output: bool = False,
    ):
        self.socket_io = socket_io
        self.agent_name = agent_name
        self.task_name = task_name
        self.tools_names = tools_names
        self.chunk_id = str(uuid.uuid4())
        self.task_chunk_id = str(uuid.uuid4())
        self.first = True
        self.stream_only_final_output = stream_only_final_output

        # Send initial task message
        self.socket_io.send_to_socket(
            text=f"""**Running task**: {task_name} **Available tools**: {tools_names}""",
            event="message",
            first=self.first,
            chunk_id=self.task_chunk_id,
            timestamp=datetime.now().timestamp() * 1000,
            display_type="inline",
        )

    def log_stream_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Handle streaming tokens from LLM.

        This is called for each token during streaming.

        Args:
            kwargs: The kwargs passed to the LLM call
            response_obj: The streaming response chunk
            start_time: Start time of the call
            end_time: End time of the call
        """
        try:
            # Extract token from streaming response
            token = None

            if hasattr(response_obj, "choices") and response_obj.choices:
                choice = response_obj.choices[0]
                if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                    token = choice.delta.content

            if token and not self.stream_only_final_output:
                self.socket_io.send_to_socket(
                    text=token,
                    event="message",
                    first=self.first,
                    chunk_id=self.chunk_id,
                    timestamp=datetime.now().timestamp() * 1000,
                    display_type="bubble",
                    author_name=self.agent_name,
                )
                self.first = False

        except Exception:
            # Don't break execution for streaming errors
            pass

    def log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Handle successful LLM completion.

        This is called when the LLM call completes successfully.

        Args:
            kwargs: The kwargs passed to the LLM call
            response_obj: The response from the LLM
            start_time: Start time of the call
            end_time: End time of the call
        """
        try:
            # Send terminate event
            self.socket_io.send_to_socket(
                text="",
                event="terminate",
            )

            # Reset for next message
            self.first = True
            self.chunk_id = str(uuid.uuid4())

        except Exception:
            # Don't break execution for socket errors
            pass

    def log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Handle failed LLM call.

        Args:
            kwargs: The kwargs passed to the LLM call
            response_obj: The error response
            start_time: Start time of the call
            end_time: End time of the call
        """
        try:
            error_msg = str(response_obj) if response_obj else "Unknown error"
            self.socket_io.send_to_socket(
                text=f"Error: {error_msg}",
                event="error",
                first=True,
                chunk_id=self.chunk_id,
                timestamp=datetime.now().timestamp() * 1000,
                display_type="bubble",
                author_name=self.agent_name,
            )
        except Exception:
            pass

    def send_final_output(self, output: str) -> None:
        """Send the final agent output to socket.

        Call this method manually when agent execution completes
        to send the final result.

        Args:
            output: The final output text to send
        """
        self.socket_io.send_to_socket(
            text=output,
            event="message",
            first=True,
            chunk_id=self.chunk_id,
            timestamp=datetime.now().timestamp() * 1000,
            display_type="bubble",
            author_name=self.agent_name,
        )

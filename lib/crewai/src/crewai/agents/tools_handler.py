"""Tools handler for managing tool execution and caching."""
from __future__ import annotations
from datetime import datetime

import json
from typing import TYPE_CHECKING, Any, Optional

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema
from ..agentcloud.socket_io import AgentCloudSocketIO
from crewai.tools.cache_tools.cache_tools import CacheTools


if TYPE_CHECKING:
    from crewai.agents.cache.cache_handler import CacheHandler
    from crewai.tools.tool_calling import InstructorToolCalling, ToolCalling


class ToolsHandler:
    """Callback handler for tool usage.

    Attributes:
        last_used_tool: The most recently used tool calling instance.
        cache: Optional cache handler for storing tool outputs.
    """
    send_to_socket: Callable

    def __init__(self,  socket_io: Optional[AgentCloudSocketIO] = None, cache: Optional[CacheHandler] = None):
      """Initialize the callback handler.

      Args:
          cache: Optional cache handler for storing tool outputs.
      """
      self.cache: CacheHandler | None = cache
      self.last_used_tool: ToolCalling | InstructorToolCalling | None = None
      self.socket_io = socket_io
    def on_tool_use(
        self,
        calling: ToolCalling | InstructorToolCalling,
        output: str,
        should_cache: bool = True,
    ) -> None:
        """Run when tool ends running.

        Args:
            calling: The tool calling instance.
            output: The output from the tool execution.
            should_cache: Whether to cache the tool output.
        """
        self.last_used_tool = calling
        if self.cache and should_cache and calling.tool_name != CacheTools().name:
            # Convert arguments to string for cache
            input_str = ""
            if calling.arguments:
                if isinstance(calling.arguments, dict):
                    input_str = json.dumps(calling.arguments)
                else:
                    input_str = str(calling.arguments)

            self.cache.add(
                tool=calling.tool_name,
                input=input_str,
                output=output,
            )

    # def on_tool_start(self, tool_name: str):
    #   self.tool_chunkId = str(uuid.uuid4())
    #   self.socket_io.send_to_socket(
    #     text=f"Using tool: {tool_name.capitalize()}",
    #     event="message",
    #     first=True,
    #     chunk_id=self.tool_chunkId,
    #     timestamp=datetime.now().timestamp() * 1000,
    #     display_type="inline"
    #   )

    # def on_tool_end(self, tool_name: str):
    #   self.socket_io.send_to_socket(
    #     text=f"Finished using tool: {tool_name.capitalize()}",
    #     event="message",
    #     first=True,
    #     chunk_id=self.tool_chunkId,
    #     timestamp=datetime.now().timestamp() * 1000,
    #     display_type="inline",
    #     overwrite=True
    #   )

    def on_tool_error(self, error_msg: str):
      self.socket_io.send_to_socket(
        text=f"""Tool usage failed:
    ```
    {error_msg}
    ```
    """,
        event="message",
        first=True,
        chunk_id=self.tool_chunkId,
        timestamp=datetime.now().timestamp() * 1000,
        display_type="bubble",
        overwrite=True
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Generate Pydantic core schema for BaseClient Protocol.

        This allows the Protocol to be used in Pydantic models without
        requiring arbitrary_types_allowed=True.
        """
        return core_schema.any_schema()

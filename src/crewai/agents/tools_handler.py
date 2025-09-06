from datetime import datetime
from typing import Any, Optional, Union,Callable
from ..agentcloud.socket_io import AgentCloudSocketIO
from crewai.tools.cache_tools.cache_tools import CacheTools
from crewai.tools.tool_calling import InstructorToolCalling, ToolCalling
from crewai.agents.cache.cache_handler import CacheHandler


class ToolsHandler:
    """Callback handler for tool usage.

    Attributes:
      last_used_tool: The most recently used tool calling instance.
      cache: Optional cache handler for storing tool outputs.
    """
    last_used_tool: Optional[ToolCalling] = None
    cache: Optional[CacheHandler]
    send_to_socket: Callable


    def __init__(self,cache: CacheHandler | None = None,  socket_io: Optional[AgentCloudSocketIO] = None) -> None:
      """Initialize the callback handler.

      Args:
          cache: Optional cache handler for storing tool outputs.
      """
      self.socket_io = socket_io
      self.cache: CacheHandler | None = cache
      self.last_used_tool: ToolCalling | InstructorToolCalling | None = None
      self.tool_chunkId = None  # Initialize this!

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
            self.cache.add(
                tool=calling.tool_name,
                input=calling.arguments,
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
      if self.socket_io and self.tool_chunkId:
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

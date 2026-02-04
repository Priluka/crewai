"""Token counting callback handler for LLM interactions.

This module provides a callback handler that tracks token usage
for LLM API calls through the litellm library.
"""

from typing import TYPE_CHECKING, Any,Dict, Optional


if TYPE_CHECKING:
  from litellm.integrations.custom_logger import CustomLogger
  from litellm.types.utils import Usage
else:
  try:
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.types.utils import Usage
  except ImportError:

    class CustomLogger:
      """Fallback CustomLogger when litellm is not available."""

    class Usage:
      """Fallback Usage when litellm is not available."""


from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess
from crewai.utilities.logger_utils import suppress_warnings


class TokenCalcHandler(CustomLogger):
  """Handler for calculating and tracking token usage in LLM calls.

  This handler integrates with litellm's logging system to track
  prompt tokens, completion tokens, and cached tokens across requests.

  Attributes:
      token_cost_process: The token process tracker to accumulate usage metrics.
      agent: Optional agent reference for lazy Redis attribute resolution.
  """

  def __init__(self, token_cost_process: TokenProcess | None, agent: Any = None, **kwargs: Any) -> None:
    """Initialize the token calculation handler.

    Args:
        token_cost_process: Optional token process tracker for accumulating metrics.
        agent: Optional agent reference for lazy Redis tracking resolution.
    """
    super().__init__(**kwargs)
    self.token_cost_process = token_cost_process
    self.agent = agent
    # Direct Redis tracking attributes (can be set via setup_redis_tracking)
    self._redis_client = None
    self._session_id = None
    self._model_id = None

  def setup_redis_tracking(self, redis_client, session_id, model_id):
    """Setup Redis tracking for this handler directly"""
    self._redis_client = redis_client
    self._session_id = session_id
    self._model_id = model_id

  def _get_redis_config(self) -> tuple[Any, str | None, str | None]:
    """Get Redis config, lazily resolving from agent if needed.

    Returns:
        Tuple of (redis_client, session_id, model_id)
    """
    # First check direct attributes
    if self._redis_client and self._session_id and self._model_id:
      return self._redis_client, self._session_id, self._model_id

    # Fall back to agent attributes (lazy resolution)
    if self.agent:
      redis_client = getattr(self.agent, '_redis_client', None)
      session_id = getattr(self.agent, '_session_id', None)
      model_id = getattr(self.agent, '_model_id', None)
      if redis_client and session_id and model_id:
        return redis_client, session_id, str(model_id)

    return None, None, None

  def log_success_event(
    self,
    kwargs: dict[str, Any],
    response_obj: dict[str, Any],
    start_time: float,
    end_time: float,
  ) -> None:
    """Log successful LLM API call and track token usage.

    Args:
        kwargs: The arguments passed to the LLM call.
        response_obj: The response object from the LLM API.
        start_time: The timestamp when the call started.
        end_time: The timestamp when the call completed.
    """
    # Extract usage from response
    usage: Usage | None = None
    if isinstance(response_obj, dict) and "usage" in response_obj:
      usage = response_obj["usage"]

    # Track token usage in TokenProcess
    if self.token_cost_process is not None and usage:
      with suppress_warnings():
        self.token_cost_process.sum_successful_requests(1)
        if hasattr(usage, "prompt_tokens"):
          self.token_cost_process.sum_prompt_tokens(usage.prompt_tokens)
        if hasattr(usage, "completion_tokens"):
          self.token_cost_process.sum_completion_tokens(
            usage.completion_tokens
          )
        if (
          hasattr(usage, "prompt_tokens_details")
          and usage.prompt_tokens_details
          and usage.prompt_tokens_details.cached_tokens
        ):
          self.token_cost_process.sum_cached_prompt_tokens(
            usage.prompt_tokens_details.cached_tokens
          )

    # Redis tracking for token usage (lazy resolution from agent if needed)
    redis_client, session_id, model_id = self._get_redis_config()
    if redis_client and session_id and model_id and usage:
      try:
        key = f"tokens:{session_id}:{model_id}"

        # Track prompt tokens
        if hasattr(usage, "prompt_tokens") and usage.prompt_tokens:
          redis_client.redis_client.hincrby(key, "prompt", usage.prompt_tokens)

        # Track completion tokens
        if hasattr(usage, "completion_tokens") and usage.completion_tokens:
          redis_client.redis_client.hincrby(key, "completion", usage.completion_tokens)

        # Track cached tokens
        if (
          hasattr(usage, "prompt_tokens_details")
          and usage.prompt_tokens_details
          and hasattr(usage.prompt_tokens_details, "cached_tokens")
          and usage.prompt_tokens_details.cached_tokens
        ):
          redis_client.redis_client.hincrby(
            key, "cached", usage.prompt_tokens_details.cached_tokens
          )

        # Set TTL to 24 hours (86400 seconds) - always set regardless of cached tokens
        redis_client.redis_client.expire(key, 86400)

      except Exception:
        # Don't break execution for tracking errors
        pass

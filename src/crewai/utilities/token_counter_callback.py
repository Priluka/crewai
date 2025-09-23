"""Token counting callback handler for LLM interactions.

This module provides a callback handler that tracks token usage
for LLM API calls through the litellm library.
"""

import warnings
from typing import Any,Dict, Optional

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import Usage

from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess


class TokenCalcHandler(CustomLogger):

  """Handler for calculating and tracking token usage in LLM calls.

  This handler integrates with litellm's logging system to track
  prompt tokens, completion tokens, and cached tokens across requests.

  Attributes:
      token_cost_process: The token process tracker to accumulate usage metrics.
  """
  def __init__(self, token_cost_process: Optional[TokenProcess]):
    self.token_cost_process = token_cost_process
    # Add Redis tracking attributes
    self.redis_client = None
    self.session_id = None
    self.model_id = None

  def setup_redis_tracking(self, redis_client, session_id, model_id):
    """Setup Redis tracking for this handler"""
    self.redis_client = redis_client
    self.session_id = session_id
    self.model_id = model_id

  def log_success_event(
    self,
    kwargs: dict[str, Any],
    response_obj: dict[str, Any],
    start_time: float,
    end_time: float,
  ) -> None:
    if self.token_cost_process is None:
      return

    with warnings.catch_warnings():
      warnings.simplefilter("ignore", UserWarning)
      if isinstance(response_obj, dict) and "usage" in response_obj:
        usage: Usage = response_obj["usage"]
        if usage:
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

          # ADD REDIS TRACKING HERE - AFTER ORIGINAL LOGIC
          if self.redis_client and self.session_id and self.model_id:
            try:
              key = f"tokens:{self.session_id}:{self.model_id}"

              # Track prompt tokens
              if hasattr(usage, "prompt_tokens") and usage.prompt_tokens:
                self.redis_client.redis_client.hincrby(key, "prompt", usage.prompt_tokens)

              # Track completion tokens
              if hasattr(usage, "completion_tokens") and usage.completion_tokens:
                self.redis_client.redis_client.hincrby(key, "completion", usage.completion_tokens)

              # Track cached tokens
              if (
                hasattr(usage, "prompt_tokens_details")
                and usage.prompt_tokens_details
                and hasattr(usage.prompt_tokens_details, "cached_tokens")
                and usage.prompt_tokens_details.cached_tokens
              ):
                self.redis_client.redis_client.hincrby(
                  key, "cached", usage.prompt_tokens_details.cached_tokens
                )
                  # ADD TTL HERE - Set expiry to 24 hours (86400 seconds)
                self.redis_client.redis_client.expire(key, 86400)
            except Exception:
              # Don't break execution for tracking errors
              pass

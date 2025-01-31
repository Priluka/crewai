import re
from typing import Union

# Import your classes from the same place you have CrewAgentParser, AgentAction, etc.
# Adjust the path if needed. For example:
from crewai.agents.parser import (
    CrewAgentParser,
    AgentAction,
    AgentFinish,
    OutputParserException,
    FINAL_ANSWER_ACTION,
    FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE
)
# Or if the constants are not in parser.py, copy them here or adapt references.

GEMINI_FALSE_POSITIVE_FINAL_ANSWER_PATTERN = re.compile(
    r"Final Answer[\s:\n*]*\(.*\)\*?\*?"
)


class GeminiAgentParser(CrewAgentParser):
    """
    A custom parser for Gemini-based LLM outputs.
    Ignores placeholder 'Final Answer' lines in the output text.
    """

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """
        Overrides CrewAgentParser.parse to handle special Gemini placeholders.

        Returns either AgentAction or AgentFinish.
        """
        # Reuse the base parser's approach to extracting "Thought"
        thought = self._extract_thought(text)

        # Check for a "real" final answer vs. a placeholder
        includes_answer = (
            FINAL_ANSWER_ACTION in text
            and not re.search(GEMINI_FALSE_POSITIVE_FINAL_ANSWER_PATTERN, text)
        )

        # Regex to capture "Action: X" and "Action Input: Y"
        regex = r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        action_match = re.search(regex, text, re.DOTALL)

        if action_match:
            # If there's both an Action and a real final answer, that's invalid
            if includes_answer:
                raise OutputParserException(
                    f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}"
                )
            # The base parser normally does some cleanup. We can replicate or call base methods
            action_name = action_match.group(1).strip()
            action_input = action_match.group(2).strip().strip('"')
            return AgentAction(thought, action_name, action_input, text)

        elif includes_answer:
            # Return text after "Final Answer:"
            final_text = text.split(FINAL_ANSWER_ACTION, maxsplit=1)[-1].strip()
            return AgentFinish(thought, final_text, text)

        # If no action or final answer, fallback to the base parser's error logic or a custom error
        return super().parse(text)

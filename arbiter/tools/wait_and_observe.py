"""wait_and_observe tool — skip intervention and request more conversation turns."""

from __future__ import annotations

DESCRIPTION = """\
Tool: wait_and_observe
Description: Reveal more of the conversation. Specify how many additional
words to reveal with the WORDS parameter. Cost is proportional to words
revealed (word_rate per word, configurable via costs.word_rate).
Syntax:
  TOOL: wait_and_observe
  REASON: <brief justification>
  WORDS: <number of words to reveal>\
"""


def execute(reason: str, **kwargs) -> str:
    """No-op: acknowledge the decision to wait and observe."""
    return f"Acknowledged. Continuing to observe. Reason: {reason}"

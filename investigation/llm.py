"""Minimal LLM interface so agents can be driven by Claude, or by a scripted stub in tests."""
import json
import re
from typing import Callable, List, Optional


class ClaudeLLM:
    """Claude through langchain-anthropic (already used by your workflow.llm_report)."""

    def __init__(self, model: str = "claude-sonnet-4-6", temperature: float = 0.0):
        from langchain_anthropic import ChatAnthropic           # raises ImportError if not installed
        self._cls, self.model, self.temperature = ChatAnthropic, model, temperature

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        llm = self._cls(model=self.model, max_tokens=max_tokens, temperature=self.temperature)
        out = llm.invoke([("system", system), ("human", user)]).content
        return out if isinstance(out, str) else "".join(b.get("text", "") for b in out)


class ScriptedLLM:
    """Test double: `handler(system, user) -> str`. Records every call."""

    def __init__(self, handler: Callable[[str, str], str]):
        self.handler, self.calls = handler, []

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        self.calls.append((system, user))
        return self.handler(system, user)


def extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in model output")
    d = json.loads(m.group(0))
    if not isinstance(d, dict):
        raise ValueError("JSON output is not an object")
    return d

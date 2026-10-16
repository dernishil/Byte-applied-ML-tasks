"""Tool-using agent loop. The LLM decides which tool to call; tools do the real computation;
the final answer is written from tool results (not from the model's guesses)."""
import json
from dataclasses import dataclass, field

from analyst.config import settings
from analyst.tools import TOOL_SPECS, Analyst

SYSTEM = """You are a careful data analyst working on the user's table.

RULES
- Every number in your answer must come from a tool result. Never guess or invent figures.
- Start with profile_dataset if you do not know the data well. Chain several tools when needed.
- If a tool returns an error, read it, fix the arguments, and try again.
- Say what you did (which tool, which filter) in one short sentence, then give the finding.
- Statistical results: report the effect size, not only the p-value, and do not claim causation.
- MEMORY: when the user states a durable definition, a correction, a preference about how to answer, or a
  reusable named filter, call `remember` (for filters give a filter_query). Do NOT save one-off questions.
  If they change something already remembered, call `remember` again with the new value (it replaces the old one).
  When the user asks to forget something, use list_memories then `forget`.
- Apply remembered items below. Use `filter_name` for saved filters, and mention when a remembered definition
  or preference shaped your answer.

DATASET
{schema}

WHAT YOU REMEMBER (from earlier sessions)
{memory}
"""


NUDGE = "Now write your answer to my question in plain text, using the tool results above."
EMPTY_NOTE = "(The model returned no text. The tool results it used are shown below.)"


@dataclass
class AgentResult:
  answer: str
  steps: list = field(default_factory=list)   # [{"tool","args","result"}]
  charts: list = field(default_factory=list)  # image paths


class GroqLLM:
  """Thin wrapper over Groq's OpenAI-compatible tool-calling API."""

  def __init__(self, api_key: str | None = None, model: str | None = None):
    from groq import Groq
    self.client = Groq(api_key=api_key or settings.groq_api_key)
    self.model = model or settings.model_name

  HINT = ("Your last tool call was rejected as invalid. Call the tool again: leave out optional parameters you do "
          "not need (never pass null) and use exactly the documented parameter names and types.")

  def chat(self, messages: list, tools: list) -> dict:
    kwargs = dict(model=self.model, messages=messages, temperature=0)
    if tools:
      kwargs.update(tools=tools, tool_choice="auto")
    try:
      resp = self.client.chat.completions.create(**kwargs)
    except Exception as exc:
      if "tool_use_failed" not in str(exc):
        raise
      kwargs["messages"] = messages + [{"role": "user", "content": self.HINT}]  # one retry with a hint
      resp = self.client.chat.completions.create(**kwargs)
    m = resp.choices[0].message
    calls = [{"id": c.id, "name": c.function.name, "arguments": c.function.arguments} for c in (m.tool_calls or [])]
    return {"content": m.content or "", "tool_calls": calls}


class Agent:

  def __init__(self, analyst: Analyst, llm, max_steps: int | None = None):
    self.analyst, self.llm = analyst, llm
    self.max_steps = max_steps or settings.max_agent_steps

  def system_prompt(self, question: str) -> str:
    mems = self.analyst.memory.recall(question, self.analyst.dataset_id)
    lines = []
    for m in mems:
      extra = f" [filter: {m['extra']['query']}]" if m["kind"] == "filter" else ""
      lines.append(f"- ({m['kind']}) {m['key']}: {m['value']}{extra}")
    return SYSTEM.format(schema=self.analyst.schema_summary(), memory="\n".join(lines) or "Nothing yet.")

  def ask(self, question: str, history: list[dict] | None = None) -> AgentResult:
    messages = [{"role": "system", "content": self.system_prompt(question)}]
    messages += [{"role": h["role"], "content": h["content"]}   # drop UI-only keys (steps, charts): the API rejects them
                 for h in (history or []) if h["role"] in ("user", "assistant")][-8:]
    messages.append({"role": "user", "content": question})
    result = AgentResult(answer="")
    for _ in range(self.max_steps):
      msg = self.llm.chat(messages, TOOL_SPECS)
      if not msg["tool_calls"]:
        text = msg["content"].strip()
        if not text and result.steps:  # model stopped after a tool call without writing anything: ask once for the answer
          messages.append({"role": "user", "content": NUDGE})
          text = self.llm.chat(messages, [])["content"].strip()
        result.answer = text or EMPTY_NOTE
        return result
      messages.append({"role": "assistant", "content": msg["content"] or None, "tool_calls": [
          {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
          for c in msg["tool_calls"]]})
      for c in msg["tool_calls"]:
        out = self.analyst.call(c["name"], c["arguments"])
        result.steps.append({"tool": c["name"], "args": c["arguments"], "result": out})
        if isinstance(out, dict) and out.get("path"):
          result.charts.append(out["path"])
        text = json.dumps(out, default=str)
        messages.append({"role": "tool", "tool_call_id": c["id"], "content": text[:6000]})
    messages.append({"role": "user", "content": "Step limit reached. Answer now using the tool results so far."})
    result.answer = self.llm.chat(messages, [])["content"].strip() or EMPTY_NOTE
    return result

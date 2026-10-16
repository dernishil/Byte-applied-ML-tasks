import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyst.agent import Agent
from analyst.memory import MemoryStore
from analyst.tools import Analyst, validate_expr

CSV = Path(__file__).resolve().parents[1] / "sample_data" / "customers.csv"


@pytest.fixture
def df():
  return pd.read_csv(CSV)


@pytest.fixture
def an(df, tmp_path):
  return Analyst(df, MemoryStore(), out_dir=str(tmp_path))


# ---------- memory ----------
def test_memory_create_unchanged_update_forget():
  m = MemoryStore()
  a = m.remember("definition", "MRR", "sum of monthly_spend", dataset="d")
  assert a["status"] == "created"
  assert m.remember("definition", "mrr", "sum of monthly_spend", dataset="d")["status"] == "unchanged"
  b = m.remember("correction", "MRR", "sum of monthly_spend for active only", dataset="d")
  assert b["status"] == "updated" and b["previous_value"] == "sum of monthly_spend"
  assert len(m.list("d")) == 1 and len(m.history("mrr", "d")) == 2  # old version kept as history
  assert m.forget(b["id"]) and m.list("d") == []


def test_memory_recall_relevance_and_preferences():
  m = MemoryStore()
  m.remember("preference", "summary frequency", "weekly not daily")
  m.remember("definition", "churn rate", "cancelled customers divided by all customers", dataset="d")
  m.remember("definition", "ltv", "lifetime value in dollars", dataset="d")
  got = [x["key"] for x in m.recall("what is our churn?", "d")]
  assert "summary frequency" in got and "churn rate" in got and "ltv" not in got  # prefs always, others only if relevant


def test_memory_persists_on_disk(tmp_path):
  p = str(tmp_path / "m.sqlite")
  MemoryStore(p).remember("preference", "chart style", "bar charts please")
  assert MemoryStore(p).list()[0]["value"] == "bar charts please"


# ---------- tools ----------
def test_profile(an, df):
  r = an.call("profile_dataset", {})
  assert r["rows"] == len(df) and "monthly_spend" in r["columns"]
  assert "signup_date" in r["columns"] and "datetime" in r["columns"]["signup_date"]["type"]  # dates auto-parsed


def test_query_matches_pandas(an, df):
  r = an.call("query_data", {"group_by": ["plan"], "aggregations": {"monthly_spend": "mean"}})
  want = df.groupby("plan")["monthly_spend"].mean().round(4).to_dict()
  got = {x["plan"]: x["monthly_spend_mean"] for x in r["result"]}
  assert got == pytest.approx(want, abs=1e-3)
  c = an.call("query_data", {"aggregations": {"*": "count"}, "filter": "status == 'active'"})
  assert c["result"][0]["row_count"] == (df.status == "active").sum()


def test_correlation_finds_planted_relationship(an):
  r = an.call("correlation", {"columns": ["support_tickets", "tenure_months", "monthly_spend"]})
  assert r["pairs"] and all("p_value" in p for p in r["pairs"])


def test_statistical_tests(an):
  t = an.call("statistical_test", {"value_col": "monthly_spend", "group_col": "plan"})
  assert "ANOVA" in t["test"] and t["significant_at_0.05"] is True
  c = an.call("statistical_test", {"value_col": "status", "group_col": "plan"})
  assert "Chi-square" in c["test"]
  two = an.call("statistical_test", {"value_col": "monthly_spend", "group_col": "plan", "filter": "plan != 'basic'"})
  assert "Welch" in two["test"]


def test_outliers_finds_planted(an):
  r = an.call("detect_outliers", {"column": "monthly_spend", "method": "iqr", "filter": "plan == 'basic'"})
  assert r["outlier_count"] >= 1 and r["most_extreme_rows"]


def test_trend(an):
  r = an.call("trend_analysis", {"date_col": "signup_date", "value_col": "customer_id", "period": "Q", "agg": "count"})
  assert r["periods"] >= 3 and r["direction"] in ("up", "down", "flat")


def test_chart_files(an):
  for kind, args in [("bar", {"x": "plan", "y": "monthly_spend"}), ("hist", {"x": "monthly_spend"}),
                     ("scatter", {"x": "tenure_months", "y": "monthly_spend"}), ("box", {"x": "plan", "y": "monthly_spend"}),
                     ("line", {"x": "signup_date", "y": "monthly_spend", "agg": "sum"})]:
    r = an.call("make_chart", {"kind": kind, **args})
    assert Path(r["path"]).exists() and Path(r["path"]).stat().st_size > 2000, kind


def test_errors_are_returned_not_raised(an):
  assert "not found" in an.call("query_data", {"group_by": ["nope"], "aggregations": {"*": "count"}})["error"]
  assert "error" in an.call("nonexistent_tool", {})


def test_filter_validation_blocks_code():
  for bad in ["__import__('os').system('x')", "a == @b", "open('f')", "x; y"]:
    with pytest.raises(ValueError):
      validate_expr(bad)


def test_works_on_a_different_dataset(tmp_path):
  other = pd.DataFrame({"day": pd.date_range("2024-01-01", periods=60).astype(str), "sales": range(60),
                        "store": ["A", "B"] * 30, "returns": [i % 7 for i in range(60)]})
  a = Analyst(other, MemoryStore(), out_dir=str(tmp_path))
  assert a.call("profile_dataset", {})["rows"] == 60
  assert a.call("trend_analysis", {"date_col": "day", "value_col": "sales", "period": "W", "agg": "sum"})["direction"] == "up"
  assert "error" not in a.call("statistical_test", {"value_col": "sales", "group_col": "store"})


# ---------- saved filter really changes later results ----------
def test_saved_filter_is_applied(an, df):
  r = an.call("remember", {"kind": "filter", "key": "active customers", "value": "excludes cancelled accounts",
                           "filter_query": "status != 'cancelled'"})
  assert r["status"] == "created"
  c = an.call("query_data", {"aggregations": {"*": "count"}, "filter_name": "Active Customers"})
  assert c["result"][0]["row_count"] == (df.status != "cancelled").sum()
  assert "error" in an.call("remember", {"kind": "filter", "key": "bad", "value": "x", "filter_query": "nocolumn > 1"})
  assert "error" in an.call("query_data", {"aggregations": {"*": "count"}, "filter_name": "unknown filter"})


# ---------- agent loop + persistence across sessions (scripted fake LLM) ----------
class FakeLLM:
  def __init__(self, script):
    self.script, self.seen = list(script), []

  def chat(self, messages, tools):
    self.seen.append(messages)
    return self.script.pop(0)


def call(id_, name, args):
  return {"id": id_, "name": name, "arguments": json.dumps(args)}


def test_agent_saves_memory_then_next_session_uses_it(df, tmp_path):
  db = str(tmp_path / "mem.sqlite")
  # Session 1: user defines a filter; agent calls remember then answers.
  llm1 = FakeLLM([
      {"content": "", "tool_calls": [call("1", "remember", {"kind": "filter", "key": "active customers",
                                                             "value": "excludes cancelled", "filter_query": "status != 'cancelled'"})]},
      {"content": "Saved. Active customers exclude cancelled accounts.", "tool_calls": []},
  ])
  a1 = Analyst(df, MemoryStore(db), out_dir=str(tmp_path))
  r1 = Agent(a1, llm1).ask("Active customers means anyone not cancelled. Remember that.")
  assert r1.steps[0]["tool"] == "remember" and r1.steps[0]["result"]["status"] == "created"
  # Session 2: brand-new objects, same database. The memory is in the prompt and the saved filter works.
  llm2 = FakeLLM([
      {"content": "", "tool_calls": [call("2", "query_data", {"aggregations": {"*": "count"}, "filter_name": "active customers"})]},
      {"content": "done", "tool_calls": []},
  ])
  a2 = Analyst(df, MemoryStore(db), out_dir=str(tmp_path))
  r2 = Agent(a2, llm2).ask("How many active customers do we have?")
  assert "status != 'cancelled'" in llm2.seen[0][0]["content"]
  assert r2.steps[0]["result"]["result"][0]["row_count"] == (df.status != "cancelled").sum()


def test_agent_recovers_from_tool_error_and_chart_is_collected(an):
  llm = FakeLLM([
      {"content": "", "tool_calls": [call("1", "make_chart", {"kind": "bar", "x": "planz", "y": "monthly_spend"})]},
      {"content": "", "tool_calls": [call("2", "make_chart", {"kind": "bar", "x": "plan", "y": "monthly_spend"})]},
      {"content": "Here is the chart.", "tool_calls": []},
  ])
  r = Agent(an, llm).ask("chart spend by plan")
  assert "error" in r.steps[0]["result"] and len(r.charts) == 1


def test_agent_step_limit(an):
  looping = {"content": "", "tool_calls": [call("x", "profile_dataset", {})]}
  llm = FakeLLM([looping] * 3 + [{"content": "final", "tool_calls": []}])
  assert Agent(an, llm, max_steps=3).ask("loop").answer == "final"


def test_history_extra_keys_are_not_sent_to_llm(an):
  llm = FakeLLM([{"content": "ok", "tool_calls": []}])
  history = [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": "hello", "steps": [{"tool": "x"}], "charts": ["a.png"]}]
  Agent(an, llm).ask("next question", history)
  sent = llm.seen[0]
  assert all(set(m) <= {"role", "content", "tool_calls", "tool_call_id"} for m in sent)


def test_null_optional_args_are_accepted(an):
  r = an.call("correlation", {"columns": None, "filter": None, "filter_name": None, "method": "pearson", "top": 5})
  assert "error" not in r and r["pairs"]
  r = an.call("query_data", {"group_by": None, "aggregations": {"*": "count"}, "filter": None, "sort_by": None})
  assert r["result"][0]["row_count"] > 0


def test_schemas_allow_null_for_optional_params():
  jsonschema = pytest.importorskip("jsonschema")
  from analyst.tools import TOOL_SPECS
  call = {"columns": None, "filter": None, "filter_name": None, "method": "pearson", "top": 5}  # the call Groq rejected
  schema = next(t for t in TOOL_SPECS if t["function"]["name"] == "correlation")["function"]["parameters"]
  jsonschema.validate(call, schema)
  for t in TOOL_SPECS:  # every schema is itself valid JSON Schema, and required params are NOT nullable
    jsonschema.Draft202012Validator.check_schema(t["function"]["parameters"])
    for req in t["function"]["parameters"]["required"]:
      assert "null" not in str(t["function"]["parameters"]["properties"][req].get("type"))


def test_groq_wrapper_retries_once_on_tool_use_failed():
  from analyst.agent import GroqLLM

  class Msg:
    content, tool_calls = "ok", None

  class Resp:
    choices = [type("C", (), {"message": Msg})]

  class FakeClient:
    calls = 0

    class chat:
      class completions:
        @staticmethod
        def create(**kw):
          FakeClient.calls += 1
          if FakeClient.calls == 1:
            raise RuntimeError("Error code: 400 ... 'code': 'tool_use_failed'")
          assert "never pass null" in kw["messages"][-1]["content"]
          return Resp()

  llm = object.__new__(GroqLLM)
  llm.client, llm.model = FakeClient, "m"
  assert llm.chat([{"role": "user", "content": "x"}], [{"type": "function"}])["content"] == "ok"
  assert FakeClient.calls == 2


def test_empty_final_message_is_recovered(an):
  llm = FakeLLM([
      {"content": "", "tool_calls": [call("1", "correlation", {"columns": None})]},
      {"content": "", "tool_calls": []},                       # model goes silent after the tool result
      {"content": "Tickets and tenure are the strongest pair.", "tool_calls": []},  # answer after the nudge
  ])
  r = Agent(an, llm).ask("which columns are related?")
  assert r.answer.startswith("Tickets") and len(r.steps) == 1
  # and if it stays silent, the user still gets a clear note instead of a blank
  llm2 = FakeLLM([{"content": "", "tool_calls": [call("1", "profile_dataset", {})]},
                  {"content": "", "tool_calls": []}, {"content": "", "tool_calls": []}])
  assert "no text" in Agent(an, llm2).ask("x").answer

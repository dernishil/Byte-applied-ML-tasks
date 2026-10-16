"""Streamlit chat UI: upload any CSV/Excel, ask questions, see which tools the agent used, manage memory."""
import json

import pandas as pd
import streamlit as st

from analyst.agent import Agent, GroqLLM
from analyst.config import settings
from analyst.memory import MemoryStore, make_embedding_scorer
from analyst.tools import Analyst

st.set_page_config(page_title="Data Analyst Agent", page_icon="📊", layout="wide")


@st.cache_resource
def get_store() -> MemoryStore:
  scorer = make_embedding_scorer() if settings.use_embeddings else None
  return MemoryStore(settings.memory_db, scorer=scorer)


def load_table(file) -> pd.DataFrame:
  return pd.read_excel(file) if file.name.lower().endswith((".xlsx", ".xls")) else pd.read_csv(file)


store = get_store()

# ---------------- sidebar: data + memory ----------------
with st.sidebar:
  st.header("Dataset")
  up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
  use_sample = st.button("Use sample dataset")
  if up is not None:
    key = f"{up.name}:{up.size}"
    if st.session_state.get("data_key") != key:
      st.session_state.update(df=load_table(up), data_key=key, messages=[])
  elif use_sample:
    st.session_state.update(df=pd.read_csv("sample_data/customers.csv"), data_key="sample", messages=[])

  if "df" in st.session_state:
    analyst = Analyst(st.session_state.df, store)
    st.caption(f"{len(analyst.df)} rows, {analyst.df.shape[1]} columns  |  dataset id `{analyst.dataset_id}`")

    st.header("Memory")
    mems = store.list(analyst.dataset_id)
    if not mems:
      st.caption("Nothing remembered yet. Tell the agent a definition, a preference or a reusable filter.")
    for m in mems:
      extra = f"\n\n`{m['extra']['query']}`" if m["kind"] == "filter" else ""
      st.markdown(f"**{m['key']}** ({m['kind']}, {'all datasets' if m['dataset'] == '*' else 'this dataset'})\n\n{m['value']}{extra}")
      if st.button("Forget", key=f"forget_{m['id']}"):
        store.forget(m["id"])
        st.rerun()

# ---------------- main: chat ----------------
st.title("📊 Data Analyst Agent")
if "df" not in st.session_state:
  st.info("Upload a table (or click **Use sample dataset**) to begin.")
  st.stop()
if not settings.groq_api_key:
  st.error("Set GROQ_API_KEY in your .env file (see .env.example), then restart.")
  st.stop()

agent = Agent(analyst, GroqLLM())
for msg in st.session_state.setdefault("messages", []):
  with st.chat_message(msg["role"]):
    st.markdown(msg["content"])
    for step in msg.get("steps", []):
      with st.expander(f"🔧 {step['tool']}"):
        st.code(step["args"], language="json")
        st.json(step["result"], expanded=False)
    for img in msg.get("charts", []):
      st.image(img)

if q := st.chat_input("Ask about your data, e.g. 'Which plan has the highest average spend?'"):
  st.session_state.messages.append({"role": "user", "content": q})
  with st.chat_message("user"):
    st.markdown(q)
  with st.chat_message("assistant"):
    with st.spinner("Analysing..."):
      res = agent.ask(q, st.session_state.messages[:-1])
    st.markdown(res.answer)
    for step in res.steps:
      with st.expander(f"🔧 {step['tool']}"):
        st.code(step["args"], language="json")
        st.json(step["result"], expanded=False)
    for img in res.charts:
      st.image(img)
  st.session_state.messages.append({"role": "assistant", "content": res.answer,
                                    "steps": res.steps, "charts": res.charts})
  if any(s["tool"] in ("remember", "forget") for s in res.steps):
    st.rerun()  # refresh the sidebar memory panel

"""Selective, persistent memory.

Unlike a chat log, this store only keeps durable facts the user states:
  definition  - what a column/metric means      ("MRR = monthly_spend summed over active customers")
  correction  - a fixed misunderstanding        ("the 'region' column uses 'EU', not 'Europe'")
  preference  - how the user likes answers      ("weekly summaries, not daily")
  filter      - a reusable named filter         ("active customers" -> status != 'cancelled')

Rules: one active entry per slot (same key); a new value supersedes the old one
(old one is kept as history); identical repeats are not stored twice; everything
can be listed, inspected and deleted by the user.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path

KINDS = ("definition", "correction", "preference", "filter")
# definition and correction share a slot: a correction replaces the definition it fixes.
_SLOT = {"definition": "fact", "correction": "fact", "preference": "preference", "filter": "filter"}
_STOP = {"the", "a", "an", "of", "to", "is", "are", "in", "for", "and", "or", "what", "how", "many", "much",
         "show", "me", "my", "do", "does", "we", "i", "you", "on", "by", "with", "all", "that", "this"}


def norm_key(text: str) -> str:
  text = re.sub(r"[^a-z0-9_ ]+", " ", text.lower())
  return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> set:
  out = set()
  for w in norm_key(text).replace("_", " ").split():
    if w in _STOP:
      continue
    out.add(w[:-1] if len(w) > 3 and w.endswith("s") else w)
  return out


class MemoryStore:

  def __init__(self, path: str = ":memory:", scorer=None):
    if path != ":memory:":
      Path(path).parent.mkdir(parents=True, exist_ok=True)
    self.db = sqlite3.connect(path, check_same_thread=False)
    self.db.row_factory = sqlite3.Row
    self.db.execute(
        """CREATE TABLE IF NOT EXISTS memories(
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             kind TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
             extra TEXT, dataset TEXT NOT NULL DEFAULT '*',
             status TEXT NOT NULL DEFAULT 'active',
             created REAL, updated REAL, supersedes INTEGER, uses INTEGER DEFAULT 0)"""
    )
    self.db.commit()
    self.scorer = scorer  # optional fn(query, [texts]) -> [floats]; default = keyword overlap

  # ---------- write ----------
  def remember(self, kind: str, key: str, value: str, extra: dict | None = None, dataset: str = "*") -> dict:
    if kind not in KINDS:
      raise ValueError(f"kind must be one of {KINDS}")
    key, value = norm_key(key), value.strip()
    if not key or not value:
      raise ValueError("key and value must be non-empty")
    now = time.time()
    extra_json = json.dumps(extra or {}, sort_keys=True)
    old = self._active_in_slot(kind, key, dataset)
    if old is not None:
      if old["value"].strip() == value and (old["extra"] or "{}") == extra_json:
        self.db.execute("UPDATE memories SET updated=? WHERE id=?", (now, old["id"]))
        self.db.commit()
        return {"status": "unchanged", "id": old["id"]}
      self.db.execute("UPDATE memories SET status='superseded', updated=? WHERE id=?", (now, old["id"]))
    cur = self.db.execute(
        "INSERT INTO memories(kind,key,value,extra,dataset,created,updated,supersedes) VALUES(?,?,?,?,?,?,?,?)",
        (kind, key, value, extra_json, dataset, now, now, old["id"] if old else None),
    )
    self.db.commit()
    res = {"status": "updated" if old else "created", "id": cur.lastrowid}
    if old:
      res["previous_value"] = old["value"]
    return res

  def forget(self, memory_id: int) -> bool:
    cur = self.db.execute("UPDATE memories SET status='deleted', updated=? WHERE id=? AND status='active'",
                          (time.time(), memory_id))
    self.db.commit()
    return cur.rowcount > 0

  # ---------- read ----------
  def list(self, dataset: str | None = None, include_history: bool = False) -> list[dict]:
    sql = "SELECT * FROM memories WHERE 1=1"
    args: list = []
    if not include_history:
      sql += " AND status='active'"
    if dataset is not None:
      sql += " AND dataset IN (?, '*')"
      args.append(dataset)
    return [self._row(r) for r in self.db.execute(sql + " ORDER BY id", args)]

  def get_filter(self, name: str, dataset: str) -> dict | None:
    row = self._active_in_slot("filter", norm_key(name), dataset)
    if row is None:
      row = self._active_in_slot("filter", norm_key(name), "*")
    return self._row(row) if row else None

  def recall(self, query: str, dataset: str, k: int = 5) -> list[dict]:
    """Memories relevant to this question. Preferences always apply; the rest must match the question."""
    active = self.list(dataset)
    prefs = [m for m in active if m["kind"] == "preference"]
    others = [m for m in active if m["kind"] != "preference"]
    if self.scorer and others:
      scores = self.scorer(query, [f"{m['key']}: {m['value']}" for m in others])
    else:
      q = _tokens(query)
      scores = [len(q & _tokens(m["key"] + " " + m["value"])) / (len(_tokens(m["key"])) or 1) for m in others]
    ranked = sorted(zip(scores, others), key=lambda t: -t[0])
    chosen = [m for s, m in ranked if s > 0][:k]
    for m in chosen:
      self.db.execute("UPDATE memories SET uses = uses + 1 WHERE id=?", (m["id"],))
    self.db.commit()
    return prefs + chosen

  def history(self, key: str, dataset: str | None = None) -> list[dict]:
    return [m for m in self.list(dataset, include_history=True) if m["key"] == norm_key(key)]

  # ---------- internals ----------
  def _active_in_slot(self, kind: str, key: str, dataset: str):
    kinds = [k for k, s in _SLOT.items() if s == _SLOT[kind]]
    marks = ",".join("?" * len(kinds))
    return self.db.execute(
        f"SELECT * FROM memories WHERE status='active' AND key=? AND dataset=? AND kind IN ({marks})",
        [key, dataset, *kinds],
    ).fetchone()

  @staticmethod
  def _row(r) -> dict:
    d = dict(r)
    d["extra"] = json.loads(d["extra"] or "{}")
    return d


def make_embedding_scorer():
  """Optional semantic scorer using the same MiniLM model as ContextCraft. Returns None if unavailable."""
  try:
    from sentence_transformers import SentenceTransformer, util
  except ImportError:
    return None
  model = SentenceTransformer("all-MiniLM-L6-v2")

  def score(query: str, texts: list[str]) -> list[float]:
    sims = util.cos_sim(model.encode([query]), model.encode(texts))[0].tolist()
    return [s if s > 0.35 else 0.0 for s in sims]  # threshold: unrelated memories are not injected

  return score

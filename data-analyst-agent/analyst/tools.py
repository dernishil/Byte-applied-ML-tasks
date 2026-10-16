"""The agent's tools. Each one does a different kind of analysis on ANY pandas DataFrame
(columns are passed in by name, nothing is hard-coded to one dataset).

  1 profile_dataset    - structure, types, missing values, summaries
  2 query_data         - filter / group / aggregate (pivot-table style)
  3 correlation        - strongest relationships between numeric columns (with p-values)
  4 statistical_test   - are groups really different? (t-test / ANOVA / chi-square + effect size)
  5 detect_outliers    - unusual values (IQR or z-score)
  6 trend_analysis     - change over time (grouped by period + linear trend)
  7 make_chart         - bar / line / hist / scatter / box chart saved as PNG
  + remember / forget / list_memories - persistent memory tools
"""
import hashlib
import json
import re
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analyst.memory import MemoryStore, norm_key

AGGS = ["mean", "sum", "count", "min", "max", "median", "nunique", "std"]
_BAD_EXPR = re.compile(r"__|@|\bimport\b|\blambda\b|\bexec\b|\beval\b|\bopen\b|\bglobals\b|\bgetattr\b|\bos\.|\bsys\.|;")


def validate_expr(expr: str) -> str:
  """Row-filter expressions are pandas `df.query` syntax, e.g. status != 'cancelled' and spend > 50."""
  if len(expr) > 300 or _BAD_EXPR.search(expr):
    raise ValueError("Filter expression not allowed. Use plain comparisons like: status != 'cancelled' and spend > 50")
  return expr


def dataset_fingerprint(df: pd.DataFrame) -> str:
  """Same column names => same id, so memories follow the dataset across uploads of the same schema."""
  names = ",".join(sorted(str(c).lower() for c in df.columns))
  return "ds_" + hashlib.sha1(names.encode()).hexdigest()[:8]


def _clean(o):
  if isinstance(o, dict):
    return {str(k): _clean(v) for k, v in o.items()}
  if isinstance(o, (list, tuple)):
    return [_clean(v) for v in o]
  if isinstance(o, (np.integer,)):
    return int(o)
  if isinstance(o, (float, np.floating)):
    return None if (np.isnan(o) or np.isinf(o)) else round(float(o), 4)
  if isinstance(o, (np.bool_,)):
    return bool(o)
  if isinstance(o, (pd.Timestamp, pd.Period)):
    return str(o)
  if o is pd.NaT:
    return None
  return o


def _auto_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  for c in df.columns:
    if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c]):
      sample = df[c].dropna().astype(str).head(50)
      if len(sample) and re.search(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", " ".join(sample.head(5))):
        with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() > 0.9:
          df[c] = parsed
  return df


class Analyst:

  def __init__(self, df: pd.DataFrame, memory: MemoryStore, out_dir: str = "outputs"):
    self.df = _auto_parse_dates(df)
    self.memory = memory
    self.dataset_id = dataset_fingerprint(df)
    self.out_dir = Path(out_dir)
    self.out_dir.mkdir(parents=True, exist_ok=True)
    self._tools = {
        "profile_dataset": self.profile_dataset, "query_data": self.query_data,
        "correlation": self.correlation, "statistical_test": self.statistical_test,
        "detect_outliers": self.detect_outliers, "trend_analysis": self.trend_analysis,
        "make_chart": self.make_chart, "remember": self.remember,
        "forget": self.forget, "list_memories": self.list_memories,
    }

  # ---------- dispatch ----------
  def call(self, name: str, args: dict | str | None) -> dict:
    if isinstance(args, str):
      try:
        args = json.loads(args or "{}")
      except json.JSONDecodeError:
        return {"error": "Arguments were not valid JSON."}
    args = {k: v for k, v in (args or {}).items() if v is not None}  # models often send null for optional params
    if name not in self._tools:
      return {"error": f"Unknown tool '{name}'. Available: {sorted(self._tools)}"}
    try:
      return _clean(self._tools[name](**args))
    except Exception as exc:  # return the error so the model can correct itself
      return {"error": f"{type(exc).__name__}: {exc}"}

  # ---------- helpers ----------
  def schema_summary(self) -> str:
    lines = [f"{len(self.df)} rows x {self.df.shape[1]} columns:"]
    for c in self.df.columns:
      s = self.df[c]
      kind = "date" if pd.api.types.is_datetime64_any_dtype(s) else "number" if pd.api.types.is_numeric_dtype(s) else "text"
      extra = ""
      if kind == "text" and s.nunique() <= 8:
        extra = " values=" + str(sorted(map(str, s.dropna().unique()))[:8])
      lines.append(f"- {c} ({kind}){extra}")
    return "\n".join(lines)

  def _col(self, name: str) -> str:
    if name in self.df.columns:
      return name
    low = {str(c).lower(): c for c in self.df.columns}
    if str(name).lower() in low:
      return low[str(name).lower()]
    raise ValueError(f"Column '{name}' not found. Columns: {list(self.df.columns)}")

  def _filtered(self, filter: str | None = None, filter_name: str | None = None) -> pd.DataFrame:
    parts = []
    if filter_name:
      saved = self.memory.get_filter(filter_name, self.dataset_id)
      if not saved:
        names = [m["key"] for m in self.memory.list(self.dataset_id) if m["kind"] == "filter"]
        raise ValueError(f"No saved filter named '{filter_name}'. Saved filters: {names}")
      parts.append(f"({saved['extra']['query']})")
    if filter:
      parts.append(f"({validate_expr(filter)})")
    if not parts:
      return self.df
    return self.df.query(" and ".join(parts), engine="python")

  # ---------- 1. profile ----------
  def profile_dataset(self) -> dict:
    cols = {}
    for c in self.df.columns:
      s = self.df[c]
      info = {"type": str(s.dtype), "missing": int(s.isna().sum()), "unique": int(s.nunique())}
      if pd.api.types.is_numeric_dtype(s):
        info.update(min=s.min(), mean=s.mean(), median=s.median(), max=s.max(), std=s.std())
      elif pd.api.types.is_datetime64_any_dtype(s):
        info.update(min=s.min(), max=s.max())
      else:
        info["top_values"] = s.value_counts().head(5).to_dict()
      cols[str(c)] = info
    return {"rows": len(self.df), "columns": cols, "duplicate_rows": int(self.df.duplicated().sum())}

  # ---------- 2. query ----------
  def query_data(self, group_by: list[str] | None = None, aggregations: dict | None = None,
                 filter: str | None = None, filter_name: str | None = None,
                 sort_by: str | None = None, descending: bool = True, limit: int = 20) -> dict:
    d = self._filtered(filter, filter_name)
    limit = max(1, min(int(limit), 100))
    gb = [self._col(c) for c in (group_by or [])]
    if not aggregations and not gb:
      return {"rows_after_filter": len(d), "sample": d.head(limit).to_dict("records")}
    aggregations = aggregations or {"*": "count"}
    spec, names = {}, {}
    for col, fn in aggregations.items():
      if fn not in AGGS:
        raise ValueError(f"Aggregation '{fn}' not supported. Use one of {AGGS}")
      if col == "*":
        continue
      spec.setdefault(self._col(col), []).append(fn)
    if gb:
      g = d.groupby(gb, dropna=False)
      out = g.agg(spec) if spec else g.size().to_frame("count")
      if spec:
        out.columns = [f"{c}_{f}" for c, f in out.columns]
      if any(c == "*" for c in aggregations):
        out["row_count"] = g.size()
      out = out.reset_index()
    else:
      row = {f"{c}_{f}": getattr(d[c], f)() for c, fs in spec.items() for f in fs}
      if "*" in aggregations:
        row["row_count"] = len(d)
      out = pd.DataFrame([row])
    if sort_by:
      if sort_by not in out.columns:
        raise ValueError(f"sort_by must be one of {list(out.columns)}")
      out = out.sort_values(sort_by, ascending=not descending)
    return {"rows_after_filter": len(d), "groups": len(out), "result": out.head(limit).to_dict("records")}

  # ---------- 3. correlation ----------
  def correlation(self, columns: list[str] | None = None, method: str = "pearson",
                  filter: str | None = None, filter_name: str | None = None, top: int = 10) -> dict:
    if method not in ("pearson", "spearman"):
      raise ValueError("method must be 'pearson' or 'spearman'")
    d = self._filtered(filter, filter_name)
    cols = [self._col(c) for c in columns] if columns else list(d.select_dtypes("number").columns)
    cols = cols[:15]
    if len(cols) < 2:
      raise ValueError("Need at least two numeric columns.")
    fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    pairs = []
    for i in range(len(cols)):
      for j in range(i + 1, len(cols)):
        pair = d[[cols[i], cols[j]]].dropna()
        if len(pair) > 3 and pair[cols[i]].nunique() > 1 and pair[cols[j]].nunique() > 1:
          r, p = fn(pair[cols[i]], pair[cols[j]])
          pairs.append({"a": cols[i], "b": cols[j], "r": r, "p_value": p, "n": len(pair)})
    pairs.sort(key=lambda t: -abs(t["r"]))
    return {"method": method, "pairs": pairs[:int(top)],
            "note": "Correlation does not prove causation."}

  # ---------- 4. statistical test ----------
  def statistical_test(self, value_col: str, group_col: str,
                       filter: str | None = None, filter_name: str | None = None) -> dict:
    d = self._filtered(filter, filter_name)
    v, g = self._col(value_col), self._col(group_col)
    if pd.api.types.is_numeric_dtype(d[v]) and not (pd.api.types.is_numeric_dtype(d[g]) and d[g].nunique() > 10):
      data = d[[v, g]].dropna()
      groups = {str(k): x[v].values for k, x in data.groupby(g) if len(x) >= 2}
      if len(groups) < 2:
        raise ValueError("Need at least two groups with 2+ rows each.")
      summary = {k: {"n": len(x), "mean": x.mean(), "std": x.std()} for k, x in groups.items()}
      arrs = list(groups.values())
      if len(arrs) == 2:
        t, p = stats.ttest_ind(arrs[0], arrs[1], equal_var=False)
        pooled = np.sqrt((arrs[0].var(ddof=1) + arrs[1].var(ddof=1)) / 2)
        effect = (arrs[0].mean() - arrs[1].mean()) / pooled if pooled else 0.0
        res = {"test": "Welch t-test (2 groups)", "statistic": t, "p_value": p, "effect_size_cohens_d": effect}
      else:
        f, p = stats.f_oneway(*arrs)
        allv = np.concatenate(arrs)
        ss_between = sum(len(a) * (a.mean() - allv.mean()) ** 2 for a in arrs)
        ss_total = ((allv - allv.mean()) ** 2).sum()
        res = {"test": f"One-way ANOVA ({len(arrs)} groups)", "statistic": f, "p_value": p,
               "effect_size_eta_squared": ss_between / ss_total if ss_total else 0.0}
        kw = stats.kruskal(*arrs)
        res["kruskal_wallis_p_value"] = kw.pvalue
      res.update(groups=summary)
    else:
      table = pd.crosstab(d[v], d[g])
      if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Need at least 2 categories in each column.")
      chi2, p, dof, _ = stats.chi2_contingency(table)
      n = table.values.sum()
      res = {"test": "Chi-square test of independence", "statistic": chi2, "p_value": p, "dof": dof,
             "effect_size_cramers_v": np.sqrt(chi2 / (n * (min(table.shape) - 1)))}
    res["significant_at_0.05"] = bool(res["p_value"] < 0.05)
    res["note"] = "Statistical significance is not the same as practical importance; check the effect size."
    return res

  # ---------- 5. outliers ----------
  def detect_outliers(self, column: str, method: str = "iqr", threshold: float | None = None,
                      filter: str | None = None, filter_name: str | None = None, show: int = 5) -> dict:
    d = self._filtered(filter, filter_name)
    c = self._col(column)
    s = d[c].dropna()
    if not pd.api.types.is_numeric_dtype(s):
      raise ValueError(f"'{c}' is not numeric.")
    if method == "iqr":
      k = 1.5 if threshold is None else float(threshold)
      q1, q3 = s.quantile(.25), s.quantile(.75)
      lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
      mask = (d[c] < lo) | (d[c] > hi)
    elif method == "zscore":
      k = 3.0 if threshold is None else float(threshold)
      z = (d[c] - s.mean()) / s.std()
      lo, hi = s.mean() - k * s.std(), s.mean() + k * s.std()
      mask = z.abs() > k
    else:
      raise ValueError("method must be 'iqr' or 'zscore'")
    out = d[mask.fillna(False)]
    extreme = out.reindex((out[c] - s.median()).abs().sort_values(ascending=False).index)
    return {"method": method, "threshold": k, "lower_bound": lo, "upper_bound": hi, "outlier_count": len(out),
            "outlier_pct": 100 * len(out) / max(len(s), 1), "most_extreme_rows": extreme.head(int(show)).to_dict("records")}

  # ---------- 6. trend ----------
  def trend_analysis(self, date_col: str, value_col: str, period: str = "M", agg: str = "sum",
                     filter: str | None = None, filter_name: str | None = None) -> dict:
    if period not in ("D", "W", "M", "Q", "Y"):
      raise ValueError("period must be one of D, W, M, Q, Y")
    if agg not in AGGS:
      raise ValueError(f"agg must be one of {AGGS}")
    d = self._filtered(filter, filter_name)
    dc, vc = self._col(date_col), self._col(value_col)
    if not pd.api.types.is_datetime64_any_dtype(d[dc]):
      raise ValueError(f"'{dc}' is not a date column.")
    series = d.dropna(subset=[dc]).groupby(d[dc].dt.to_period(period))[vc].agg(agg)
    if len(series) < 3:
      raise ValueError("Need at least 3 periods for a trend.")
    x = np.arange(len(series))
    lr = stats.linregress(x, series.values.astype(float))
    first, last = float(series.iloc[0]), float(series.iloc[-1])
    return {"period": period, "aggregation": f"{agg} of {vc}", "periods": len(series),
            "first": first, "last": last, "change_pct": None if first == 0 else 100 * (last - first) / abs(first),
            "slope_per_period": lr.slope, "r_squared": lr.rvalue ** 2, "trend_p_value": lr.pvalue,
            "direction": "up" if lr.slope > 0 else "down" if lr.slope < 0 else "flat",
            "best_period": str(series.idxmax()), "worst_period": str(series.idxmin()),
            "series_tail": {str(k): v for k, v in series.tail(24).items()}}

  # ---------- 7. chart ----------
  def make_chart(self, kind: str, x: str, y: str | None = None, agg: str = "mean", title: str | None = None,
                 filter: str | None = None, filter_name: str | None = None) -> dict:
    if kind not in ("bar", "line", "hist", "scatter", "box"):
      raise ValueError("kind must be bar, line, hist, scatter or box")
    d = self._filtered(filter, filter_name)
    xc, yc = self._col(x), (self._col(y) if y else None)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if kind == "hist":
      ax.hist(d[xc].dropna(), bins=30, color="#4C72B0")
      ax.set_ylabel("count")
    elif kind == "scatter":
      if not yc:
        raise ValueError("scatter needs y")
      ax.scatter(d[xc], d[yc], s=12, alpha=.6, color="#4C72B0")
      ax.set_ylabel(yc)
    elif kind == "box":
      if not yc:
        raise ValueError("box needs y (numeric) and x (category)")
      top = d[xc].value_counts().head(10).index
      ax.boxplot([d.loc[d[xc] == k, yc].dropna() for k in top])
      ax.set_xticks(range(1, len(top) + 1))
      ax.set_xticklabels([str(k) for k in top])
      ax.set_ylabel(yc)
    else:
      if pd.api.types.is_datetime64_any_dtype(d[xc]):
        grouped = d.groupby(d[xc].dt.to_period("M"))
      else:
        grouped = d.groupby(xc)
      vals = grouped.size() if not yc else grouped[yc].agg(agg)
      vals = vals.sort_values(ascending=False).head(20) if kind == "bar" and not pd.api.types.is_datetime64_any_dtype(d[xc]) else vals
      labels = [str(k) for k in vals.index]
      (ax.bar if kind == "bar" else ax.plot)(labels, vals.values, **({"color": "#4C72B0"} if kind == "bar" else {"marker": "o"}))
      ax.set_ylabel(f"{agg} of {yc}" if yc else "count")
      plt.xticks(rotation=45, ha="right")
    ax.set_xlabel(xc)
    ax.set_title(title or f"{kind}: {yc + ' by ' if yc else ''}{xc}")
    fig.tight_layout()
    path = self.out_dir / f"chart_{int(time.time() * 1000)}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return {"path": str(path), "rows_used": len(d)}

  # ---------- memory tools ----------
  def remember(self, kind: str, key: str, value: str, filter_query: str | None = None,
               scope: str | None = None) -> dict:
    """kind: definition | correction | preference | filter. For filters give filter_query (pandas query)."""
    extra = None
    if kind == "filter":
      if not filter_query:
        raise ValueError("A filter memory needs filter_query, e.g. \"status != 'cancelled'\"")
      validate_expr(filter_query)
      self.df.query(filter_query, engine="python")  # must actually run on this dataset
      extra = {"query": filter_query}
    scope = scope or ("global" if kind == "preference" else "dataset")
    ds = "*" if scope == "global" else self.dataset_id
    return self.memory.remember(kind, key, value, extra, ds)

  def forget(self, memory_id: int) -> dict:
    return {"forgotten": self.memory.forget(int(memory_id))}

  def list_memories(self) -> dict:
    return {"memories": [{k: m[k] for k in ("id", "kind", "key", "value", "extra", "dataset")}
                         for m in self.memory.list(self.dataset_id)]}


# ---------- tool schemas for the LLM ----------
def _nullable(prop: dict) -> dict:
  p = dict(prop)
  if isinstance(p.get("type"), str):
    p["type"] = [p["type"], "null"]
  if "enum" in p and None not in p["enum"]:
    p["enum"] = p["enum"] + [None]
  return p


def _t(name, desc, props, required=()):
  # Optional parameters accept null: models frequently send null instead of omitting them.
  props = {k: (v if k in required else _nullable(v)) for k, v in props.items()}
  return {"type": "function", "function": {"name": name, "description": desc,
          "parameters": {"type": "object", "properties": props, "required": list(required)}}}


_FILT = {"filter": {"type": "string", "description": "Optional row filter, pandas query syntax, e.g. \"region == 'EU' and spend > 50\""},
         "filter_name": {"type": "string", "description": "Name of a SAVED filter from memory (e.g. 'active customers')"}}
_S, _N = {"type": "string"}, {"type": "number"}

TOOL_SPECS = [
    _t("profile_dataset", "Overview of the dataset: columns, types, missing values, ranges, top categories, duplicates.", {}),
    _t("query_data", "Filter, group and aggregate rows (like a pivot table). aggregations maps column -> one of "
       f"{AGGS}; use {{\"*\": \"count\"}} to count rows. Use this for any exact number.",
       {"group_by": {"type": "array", "items": _S}, "aggregations": {"type": "object"}, **_FILT,
        "sort_by": _S, "descending": {"type": "boolean"}, "limit": _N}),
    _t("correlation", "Find the strongest relationships between numeric columns, with p-values.",
       {"columns": {"type": "array", "items": _S}, "method": {"type": "string", "enum": ["pearson", "spearman"]},
        **_FILT, "top": _N}),
    _t("statistical_test", "Test whether a numeric column differs between groups (t-test/ANOVA), or whether two "
       "categorical columns are associated (chi-square). Returns p-value and effect size.",
       {"value_col": _S, "group_col": _S, **_FILT}, ["value_col", "group_col"]),
    _t("detect_outliers", "Find unusual values in a numeric column using IQR or z-score.",
       {"column": _S, "method": {"type": "string", "enum": ["iqr", "zscore"]}, "threshold": _N, **_FILT, "show": _N}, ["column"]),
    _t("trend_analysis", "How a numeric column changes over time: aggregate per period (D/W/M/Q/Y) and fit a linear trend.",
       {"date_col": _S, "value_col": _S, "period": {"type": "string", "enum": ["D", "W", "M", "Q", "Y"]},
        "agg": {"type": "string", "enum": AGGS}, **_FILT}, ["date_col", "value_col"]),
    _t("make_chart", "Create a chart image. kinds: bar, line, hist, scatter, box.",
       {"kind": {"type": "string", "enum": ["bar", "line", "hist", "scatter", "box"]}, "x": _S, "y": _S,
        "agg": {"type": "string", "enum": AGGS}, "title": _S, **_FILT}, ["kind", "x"]),
    _t("remember", "Save a DURABLE fact the user stated: a definition (what a column/metric means), a correction, "
       "a preference about how to answer, or a reusable named filter (give filter_query). Do not save one-off questions.",
       {"kind": {"type": "string", "enum": ["definition", "correction", "preference", "filter"]},
        "key": {"type": "string", "description": "Short name, e.g. 'active customers' or 'summary frequency'"},
        "value": {"type": "string", "description": "The meaning/rule in one sentence"},
        "filter_query": {"type": "string", "description": "Required for kind=filter, pandas query syntax"},
        "scope": {"type": "string", "enum": ["dataset", "global"]}}, ["kind", "key", "value"]),
    _t("forget", "Delete a saved memory by id (when the user asks to forget/remove it).", {"memory_id": _N}, ["memory_id"]),
    _t("list_memories", "Show what is currently remembered.", {}),
]

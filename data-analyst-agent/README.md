# Data Analyst Agent

A data analyst you can talk to in plain English. Upload a CSV or Excel file, ask a question about the data, and the agent chooses the analysis tools it needs to answer it.

I built this project to explore what makes an AI agent useful for data analysis beyond simply sending a question to an LLM and displaying its response. The main idea is to let the model decide **what analysis to perform**, while having Python carry out the actual calculations. I also wanted to explore persistent memory, so the agent can retain useful definitions, corrections, preferences, and filters across sessions.

## What it can do

- Analyse CSV and Excel datasets without being tied to a particular set of columns.
- Answer questions involving filtering, grouping, aggregation, and summary statistics.
- Find correlations, test differences between groups, and detect outliers.
- Analyse changes over time and generate charts.
- Remember useful information across sessions instead of treating every conversation as a fresh start.
- Show the tools used to answer a question, along with their arguments and results.

The interface is built with Streamlit, and the analysis runs on the actual uploaded dataset.

## How I approached the problem

I wanted to separate the model's job from the actual data analysis.

An LLM can interpret a question and decide which operations might answer it, but asking it to produce numbers directly from a dataset is not a reliable way to perform calculations. The model could miscalculate, overlook rows, or generate a plausible-looking number without actually computing it.

Instead, I gave the agent a collection of tools that perform specific operations on a pandas DataFrame. The model decides which tools to call and how to use their results. Python performs the calculations, and the model uses the returned results to explain its findings.

This also makes the system more flexible. A question about customer spending might require grouping and aggregation, while a question about relationships between variables might require correlation analysis. Rather than writing a separate workflow for each question, the agent can choose between the available tools.

The other design decision was to keep memory separate from conversation history. I didn't want the application to save every message indefinitely. I wanted it to remember information that could actually change how future questions are answered.

These two ideas — tool-based analysis and selective memory — form the core of the project.

## How it works

The application follows this general flow:

1. **Load the dataset.** The user uploads a CSV or Excel file, or selects the included sample dataset. The application loads it into a pandas DataFrame.

2. **Give the model context.** The agent provides the model with a summary of the dataset's columns and their types, along with relevant remembered information.

3. **Choose the tools.** The model interprets the question and decides which analysis operations are needed. It can call multiple tools when one operation isn't enough.

4. **Run the analysis.** Each tool executes Python code against the DataFrame and returns its results. If a tool call fails, the error is returned to the agent so it can attempt to correct the call.

5. **Use the results to answer.** The model receives the tool outputs and produces a natural-language explanation. The agent is instructed to derive numerical claims from tool results rather than invent figures.

6. **Display the process.** The interface shows the answer, the tools called, their arguments, and their results. Generated charts are displayed alongside the response.

The agent loop has a configurable step limit, so it cannot keep calling tools indefinitely. Conversation history is also limited to recent messages, with interface-only information removed before it is sent to the model.

## The analysis tools

I separated the analysis into different tools rather than building one general-purpose function. Each tool handles a distinct type of question.

| Tool | Purpose |
|---|---|
| `profile_dataset` | Inspect column types, missing values, unique values, numerical summaries, frequent categories, and duplicate rows. |
| `query_data` | Filter rows, group data, calculate aggregations, sort results, and return exact values. |
| `correlation` | Find relationships between numerical columns using Pearson or Spearman correlation, including p-values. |
| `statistical_test` | Compare groups using Welch's t-test or ANOVA, or test associations between categorical variables using chi-square. Effect sizes are included where applicable. |
| `detect_outliers` | Identify unusual numerical values using the IQR or z-score method. |
| `trend_analysis` | Aggregate a numerical variable over time and estimate its trend using linear regression. |
| `make_chart` | Generate bar charts, line charts, histograms, scatter plots, and box plots. |

The agent also has three memory-related tools: `remember`, `forget`, and `list_memories`.

The analysis tools accept column names and parameters rather than relying on a hard-coded dataset. This allows the same functions to work with different tables, provided their columns and data types are suitable for the requested operation.

I also included basic validation for row-filter expressions and checks for unsupported operations, missing columns, and incompatible data types. When a tool encounters an error, it returns a readable error message instead of silently producing a result.

## Persistent memory

Memory was an important part of the design because I wanted the agent to retain useful context beyond a single conversation.

For example, suppose a user explains that their definition of an active customer excludes cancelled accounts. If the user later asks for revenue from active customers, the agent should be able to apply that definition without requiring the user to explain it again.

I implemented four types of memory:

- **Definitions:** What a column or metric means.
- **Corrections:** Fixes to previously stored information or misunderstandings.
- **Preferences:** How the user prefers answers or summaries.
- **Filters:** Reusable named conditions that can be applied to the dataset.

### Why not just save the conversation?

Conversation history and persistent memory serve different purposes. A conversation contains individual questions, temporary details, and information that may never be useful again. Saving everything would make retrieval less focused and make it harder to update outdated information.

Instead, the agent is instructed to save durable information when appropriate, and not to store one-off questions.

### How memory is stored and retrieved

Memory is stored in SQLite, so it persists across application restarts.

Each entry has a key, value, type, scope, and status. When an existing entry is changed, the previous version is marked as superseded rather than simply discarded. Repeating an identical memory does not create a duplicate entry, and users can inspect or delete active memories through the interface.

Retrieval is selective:

- Preferences apply across datasets.
- Definitions, corrections, and filters are associated with a dataset identifier.
- Other memories are retrieved when their contents are relevant to the current question.

The dataset identifier is derived from the column names. This means that uploading a dataset with the same schema can reuse its associated memories, although it does not guarantee that the underlying data is identical.

By default, relevance is determined using keyword overlap. There is also an optional semantic retrieval mode using `sentence-transformers` and the `all-MiniLM-L6-v2` model. This can be enabled through configuration.

Saved filters are more than descriptive notes: their actual query expressions can be passed to the analysis tools and applied to the data.

I wanted memory to influence the agent's future behaviour, rather than exist as a separate feature that the agent never uses.

## Technology stack

- **Python:** Main application and analysis logic.
- **Groq API:** Model inference and tool calling.
- **`openai/gpt-oss-20b`:** Default model, configurable through the environment.
- **pandas and NumPy:** Data manipulation and numerical operations.
- **SciPy:** Statistical tests, correlations, and trend calculations.
- **Matplotlib:** Chart generation.
- **SQLite:** Persistent memory storage.
- **Streamlit:** User interface.
- **Pydantic Settings:** Configuration through environment variables.
- **pytest:** Automated testing.

The model is responsible for interpreting questions and choosing operations. The calculations are performed by the local Python analysis tools, not by asking the model to calculate the answers from scratch.

## Project structure

```text
data-analyst-agent/
├── analyst/
│   ├── agent.py       # Agent loop, model integration and tool orchestration
│   ├── tools.py       # Data analysis tools and tool definitions
│   ├── memory.py      # Persistent memory, retrieval and updates
│   └── config.py      # Application settings
├── sample_data/
│   └── customers.csv  # Sample dataset
├── scripts/
│   └── make_sample_data.py
├── tests/
│   └── test_all.py
├── app.py             # Streamlit interface
├── .env.example       # Example configuration
├── requirements.txt
└── README.md
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/dernishil/Byte-applied-ML-tasks.git
cd Byte-applied-ML-tasks/data-analyst-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Copy the example environment file:

```bash
cp .env.example .env
```

Add your Groq API key to `.env`:

```dotenv
GROQ_API_KEY=your_key_here
MODEL_NAME=openai/gpt-oss-20b
MEMORY_DB=./data/memory.sqlite
USE_EMBEDDINGS=false
```

The default setup uses keyword-based memory retrieval. To enable semantic retrieval, install the optional `sentence-transformers` dependency and set `USE_EMBEDDINGS=true`.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, upload a CSV or Excel file, and start asking questions.

You can also select the included sample customer dataset to try the application without preparing your own data.

### 6. Run the tests

```bash
python -m pytest -q tests
```

The test suite covers the analysis tools, memory behaviour and persistence, filter validation, and the agent loop using a scripted test model. These tests do not require a live Groq API call; the live model integration should also be tested separately with a valid API key.



[Data_Analyst_Agent_md.md](https://github.com/user-attachments/files/33569444/Data_Analyst_Agent_md.md)
# Data Analyst Agent

A data-analysis assistant that lets you ask questions about CSV and
Excel files in plain English. It chooses which analysis tools to use,
runs the calculations in Python, and explains the results. It can also
remember useful definitions, corrections, preferences, and filters
between sessions.

I built this project to explore two ideas: letting an LLM choose how to
investigate a dataset without asking it to calculate everything itself,
and giving the agent a way to retain useful context beyond a single
conversation.

## What it does

-   Works with CSV and Excel datasets without requiring a fixed set of
    columns.
-   Filters, groups, and aggregates data, and calculates summary
    statistics.
-   Checks correlations, compares groups, and detects outliers.
-   Analyses changes over time and generates charts.
-   Stores selected information so it can be reused in later sessions.
-   Shows which tools were called, their arguments, and their results.

The interface uses Streamlit, and the calculations run against the
uploaded dataset.

## Design overview

The main design choice was to separate deciding what to do from doing
the calculations.

An LLM can interpret a question and choose an appropriate operation, but
it is not a dependable calculator for an entire dataset. It might miss
rows, make arithmetic mistakes, or return a plausible number that was
never actually calculated.

Instead, the agent has a set of tools that operate on a pandas
DataFrame. The model chooses which tools to call and supplies their
parameters; Python performs the calculations; then the model uses the
returned results to explain the answer. A question about customer
spending might need grouping and aggregation, while a question about
relationships between variables might need correlation analysis. The
same tools can be combined for different questions, rather than
requiring a separate workflow for each one.

I also kept persistent memory separate from conversation history. Not
every message is useful later, and saving everything would make it
harder to find and update the information that matters. The agent can
instead store selected definitions, corrections, preferences, and
filters.

## How a question is handled

1.  **Load the data.** The user uploads a CSV or Excel file, or chooses
    the included sample dataset. The application loads it into a pandas
    DataFrame.
2.  **Provide context.** The agent sends the model a summary of the
    dataset's columns and types, along with relevant memories.
3.  **Choose tools.** The model interprets the question and decides
    which analysis operations to call. It can use more than one tool
    when needed.
4.  **Run the calculations.** Each tool operates on the DataFrame and
    returns its results. If a call fails, the error is passed back to
    the agent so it can try to correct the call.
5.  **Explain the result.** The model uses the tool outputs to produce a
    natural-language answer. It is instructed to base numerical claims
    on tool results rather than inventing figures.
6.  **Show the process.** The interface displays the answer, tool calls,
    arguments, results, and any generated charts.

The agent loop has a configurable step limit to prevent it from calling
tools indefinitely. Only recent conversation messages are kept, and
interface-only information is removed before the history is sent to the
model.

## Analysis tools

The analysis is split into tools with distinct responsibilities:

  -----------------------------------------------------------------------
  Tool                                Purpose
  ----------------------------------- -----------------------------------
  `profile_dataset`                   Inspects column types, missing and
                                      unique values, numerical summaries,
                                      frequent categories, and duplicate
                                      rows.

  `query_data`                        Filters rows, groups data,
                                      calculates aggregations, sorts
                                      results, and returns exact values.

  `correlation`                       Measures relationships between
                                      numerical columns using Pearson or
                                      Spearman correlation, including
                                      p-values.

  `statistical_test`                  Compares groups with Welch's t-test
                                      or ANOVA, or tests associations
                                      between categorical variables with
                                      chi-square. It includes effect
                                      sizes where applicable.

  `detect_outliers`                   Finds unusual numerical values
                                      using the IQR or z-score method.

  `trend_analysis`                    Aggregates a numerical variable
                                      over time and estimates its trend
                                      using linear regression.

  `make_chart`                        Generates bar charts, line charts,
                                      histograms, scatter plots, and box
                                      plots.
  -----------------------------------------------------------------------

There are also three memory tools: `remember`, `forget`, and
`list_memories`.

The analysis tools take column names and parameters instead of relying
on a particular dataset's schema. They can therefore be reused with
different tables, as long as the columns and data types fit the
requested operation.

The tools include basic validation for row-filter expressions,
unsupported operations, missing columns, and incompatible data types.
When something goes wrong, they return a readable error instead of
silently producing a result.

## Persistent memory

Memory lets the agent reuse relevant context in later sessions. For
example, a user might define an active customer as someone whose account
has not been cancelled. If they later ask for revenue from active
customers, the agent can use that definition without asking them to
explain it again.

The application supports four types of memory:

-   **Definitions:** What a column or metric means.
-   **Corrections:** Fixes to previously stored information or
    misunderstandings.
-   **Preferences:** How the user prefers answers or summaries.
-   **Filters:** Named conditions that can be reused on a dataset.

### Why keep memory separate from chat history?

Conversation history includes temporary details and one-off questions
that may not be useful again. Saving every message would make retrieval
less focused and make outdated information harder to update. The agent
is therefore instructed to save durable information when appropriate,
rather than storing every question.

### Storage and retrieval

Memory is stored in SQLite, so it survives application restarts.

Each entry has a key, value, type, scope, and status. When an entry
changes, the old version is marked as superseded rather than discarded.
Repeating an identical memory does not create a duplicate, and users can
inspect or delete active memories in the interface.

Retrieval depends on the memory type:

-   Preferences apply across datasets.
-   Definitions, corrections, and filters are linked to a dataset
    identifier.
-   Other memories are retrieved when their content is relevant to the
    current question.

The dataset identifier is derived from column names. This lets a dataset
with the same schema reuse associated memories, but it does **not**
guarantee that the underlying data is the same.

By default, the application uses keyword overlap to find relevant
memories. An optional semantic retrieval mode uses
`sentence-transformers` with the `all-MiniLM-L6-v2` model and can be
enabled through configuration.

Saved filters can be used directly: their query expressions can be
passed to the analysis tools and applied to the data. This is how stored
context can affect later analysis instead of remaining a separate,
unused feature.

## Technology stack

-   **Python:** Application and analysis logic.
-   **Groq API:** Model inference and tool calling.
-   **`openai/gpt-oss-20b`:** Default model; configurable through the
    environment.
-   **pandas and NumPy:** Data manipulation and numerical operations.
-   **SciPy:** Statistical tests, correlations, and trend calculations.
-   **Matplotlib:** Chart generation.
-   **SQLite:** Persistent memory storage.
-   **Streamlit:** User interface.
-   **Pydantic Settings:** Environment-based configuration.
-   **pytest:** Automated tests.

The model interprets the question and chooses the operations. The Python
tools perform the calculations.

## Project structure

``` text
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

``` bash
git clone https://github.com/dernishil/Byte-applied-ML-tasks.git
cd Byte-applied-ML-tasks/data-analyst-agent
```

### 2. Create a virtual environment

``` bash
python -m venv .venv
```

Activate it on Linux or macOS:

``` bash
source .venv/bin/activate
```

On Windows:

``` bat
.venv\Scripts\activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Configure the API key

Copy the example environment file:

``` bash
cp .env.example .env
```

Add your Groq API key to `.env`:

``` dotenv
GROQ_API_KEY=your_key_here
MODEL_NAME=openai/gpt-oss-20b
MEMORY_DB=./data/memory.sqlite
USE_EMBEDDINGS=false
```

The default configuration uses keyword-based memory retrieval. To use
semantic retrieval, install the optional `sentence-transformers`
dependency and set `USE_EMBEDDINGS=true`.

### 5. Run the application

``` bash
streamlit run app.py
```

Open the local URL printed by Streamlit, upload a CSV or Excel file, and
ask a question. You can also choose the included sample customer dataset
to try the app without preparing your own data.

### 6. Run the tests

``` bash
python -m pytest -q tests
```

The test suite covers the analysis tools, memory behaviour and
persistence, filter validation, and the agent loop using a scripted test
model. These tests do not require a live Groq API call. The live model
integration should also be tested separately with a valid API key.

## Limitations and trade-offs

There are a few important limits to keep in mind:

-   **The model can still make poor choices.** The tools handle
    calculations, but the model might misunderstand a question, choose
    an unsuitable statistical test, or miss a useful operation.
-   **Statistics need context.** A small p-value does not automatically
    mean an effect is practically important, and correlation does not
    establish causation. The agent is instructed to report effect sizes
    and avoid causal claims.
-   **Memory retrieval is not perfect.** Keyword matching can miss
    relevant entries or retrieve a weak match. Semantic retrieval is
    available, but it also needs evaluation.
-   **Dataset identity is approximate.** The identifier is based on
    column names, not a fingerprint of the data. Two datasets with the
    same schema can therefore share dataset-scoped memories.
-   **Data must fit in memory.** Files are loaded into pandas
    DataFrames, so this version is best suited to datasets that fit
    comfortably in available RAM.
-   **The model can only use implemented tools.** The agent calls
    predefined tools rather than generating and executing arbitrary
    Python code. That keeps the workflow more controlled, but limits it
    to the operations those tools support.
-   **Automated tests do not prove live-model reliability.** The
    scripted model tests check much of the application logic, but they
    cannot show how reliably the live model will choose tools for
    unfamiliar questions and datasets.


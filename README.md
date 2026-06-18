# LogPilot Agent Lab

A Streamlit demo for agentic log parsing with parser tools, template memory RAG, trace export, and offline evaluation.

## Highlights

- Classic log parsing dashboard with Drain baseline and LLM-assisted parsing.
- Template Memory RAG for retrieving historical log templates.
- Agent tool orchestration over Drain parsing, template memory parsing, comparison, and summary tools.
- Trace and report export for Agent runs.
- Offline evaluation over small demo cases.

## Demo Entrypoints

### Classic LogPilot Dashboard

```bash
streamlit run app.py
```

### LogPilot Agent Lab

```bash
streamlit run agent_app.py
```

## Classic vs Agent Lab

The Classic Dashboard is method-centric: users directly run or compare parsing methods such as Drain, LLM-assisted parsing, Hybrid Review, and template evolution analysis.

The Agent Lab is task-centric: users select a goal, and the agent plans and calls parser tools such as Drain parsing, Template Memory parsing, comparison, and summary generation.

## Agent Workflow

```text
Selected logs
-> Agent task planning
-> Tool calls: Drain / Template Memory / Comparison / Summary
-> Parsed results and risk samples
-> Trace and Markdown report export
```

## Quick Start

```bash
pip install -r requirements.txt
streamlit run agent_app.py
```

Online LLM features require a local `.env` file with an OpenAI-compatible API configuration. The offline Agent demo and core tests can run without model access.

## Project Structure

```text
.
|-- app.py                    # Classic LogPilot dashboard
|-- agent_app.py              # Agent Lab Streamlit app
|-- logpilot/                 # Core parsing, evaluation, evolution, and Agent framework modules
|-- data/
|   |-- template_memory/      # Small demo template memory files
|   +-- eval/                 # Small offline evaluation cases
|-- examples/                 # Minimal offline demos and tiny sample logs
|-- tests/                    # Core regression tests
+-- docs/                     # Lightweight public reference docs
```

## Notes

- This repository only includes small demo data.
- The public repository keeps only a few representative offline examples. Development-stage scripts and large generated artifacts are intentionally excluded.
- Large-scale raw logs, generated outputs, and private experiment files are not included.
- Online LLM features require `.env` and an OpenAI-compatible API configuration.
- Offline Agent demo can run without model access.

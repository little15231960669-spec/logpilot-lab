# LogPilot

**LogPilot** is a lightweight AI product demo for log parsing, LLM-based template extraction, hybrid review, and template evolution analysis.

It is built for exploring how traditional log parsers and large language models can work together in an AIOps scenario.

## Overview

LogPilot focuses on a practical question:

> Should LLMs replace traditional log parsers, or should they be used as a review and correction layer?

Instead of assuming that LLMs are always better, LogPilot compares three strategies:

* **Drain baseline**: fast and stable traditional log parsing.
* **LLM Direct**: direct template extraction using LLM-style structured output.
* **Hybrid Review**: Drain parses all logs first, and only risky samples are sent to LLM review.
* **Template Evolution Analysis**: compares two log windows and detects stable, new, disappeared, and rewritten templates.

The goal is to show an AI product workflow with evaluation, cost awareness, fallback logic, and explainable template lifecycle analysis.

## Features

* Load built-in HDFS and BGL sample logs.
* Upload local `.log` / `.txt` files.
* Read selected line ranges from large local log files.
* Extract log content and apply lightweight variable masking.
* Run Drain baseline parser.
* Run LLM Direct parser with:

  * Mock Demo backend
  * OpenAI-compatible backend
  * Custom model name input
* Compare Drain and LLM parsing results.
* Run Hybrid Review with risk-based LLM routing.
* Analyze template evolution between two selected windows.
* Display key product metrics:

  * template count
  * review rate
  * fallback rate
  * token estimate
  * valid JSON status
* Export parsing, comparison, hybrid results as CSV.

## Screenshots

Add screenshots after running the demo:

```text
docs/screenshots/preview.png
docs/screenshots/llm_direct.png
docs/screenshots/comparison.png
docs/screenshots/hybrid.png
```

Example:

```markdown
![LogPilot Preview](docs/screenshots/preview.png)
```

## Project Structure

```text
logpilot/
├── app.py
├── logpilot/
│   ├── data/
│   ├── parsers/
│   ├── llm/
│   ├── evaluation/
│   └── evolution/
├── examples/
├── docs/
└── results/
```

## Quick Start

### 1. Create environment

```bash
conda create -n logpilot python=3.10 -y
conda activate logpilot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

## LLM Configuration

LogPilot can run without any API key by using **Mock Demo**.

To use a real OpenAI-compatible model, create a local `.env` file:

```bash
copy .env.example .env
```

Then edit `.env`:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://your-provider-compatible-endpoint/v1
OPENAI_MODEL=your_model_name
```

Do not commit `.env` to GitHub.

## AI Framework Extension

This branch adds an optional LangChain-based extension layer under `logpilot/ai_framework/`.

Current examples:

- `examples/langchain_00_basic_call.py`
- `examples/langchain_01_structured_parse.py`

The extension is designed for structured log parsing, Template Memory RAG, and Log Review Agent experiments. It does not replace the original LogPilot workflow.

The structured parser includes a JSON fallback path for providers that do not fully support native structured output.

Phase 2 adds Template Memory RAG: historical templates are retrieved as reference context before LLM-based log parsing.

Phase 3 adds a lightweight Log Review Agent that plans actions in JSON and executes retrieval, RAG parsing, template comparison, and review summarization through Python-controlled tools.

Phase 4 adds agent trace export: each Log Review Agent run can be saved as a JSON trace and Markdown report for debugging, evaluation, and portfolio demonstration.

Phase 4.5 adds explicit run status (`success`, `partial_success`, `failed`) and an offline happy-path agent trace demo for portfolio display.

Phase 5 adds a Tool Registry that documents tool metadata, risk level, LLM dependency, deterministic behavior, and suitable agent actions.

Phase 6 adds an offline agent evaluation dataset and metrics for action accuracy, retrieval hit rate, template match rate, final decision accuracy, and run status.

Phase 7 adds a Streamlit Agent Review tab for running the offline/online agent demo, viewing tool traces, downloading reports, and displaying offline eval metrics.

Phase 7.5 connects Agent Review to the current logs selected from the sidebar and lets the Agent call parser tools such as Drain, Template Memory parsing, batch comparison, and summary generation.

Phase 8 adds a separate Agent-first Streamlit app (`agent_app.py`) that presents task-driven tool orchestration instead of method-centric parsing tabs.

Phase 8.5 improves the Agent-first UI with Chinese presentation, collapsible data context, readable plan/tool/result sections, dynamic parsed-result tables, and clearer final recommendations.

Phase 8.6 improves Agent Lab usability with clearer sample/full-log semantics, dataset-aware template memory warnings, BGL template memory, simplified recommendations, and cleaner report display.

Phase 8.7 makes template memory conditional, supports uploaded/generated memory CSVs, fixes Streamlit session state handling, and clarifies the agentic memory workflow.

## AI Skill Design

LogPilot Agent Lab organizes log-analysis tasks as reusable AI Skills with explicit inputs, tool chains, outputs, fallbacks, and evaluation signals:

- **Log Parsing Skill** turns raw logs into structured events and templates.
- **Template Memory Retrieval Skill** retrieves relevant historical templates as parsing context.
- **Risk Review Skill** compares parsing results and identifies samples that need review.
- **Report Export Skill** packages traces, findings, and recommendations into shareable artifacts.

See [docs/AI_SKILL_DESIGN.md](docs/AI_SKILL_DESIGN.md) for the compact skill specification.

## Data

This project includes only small demo log samples under `examples/`.

For larger experiments, users can prepare their own LogHub / LogHub-2.0 data locally and load selected line ranges through the local path mode.

Do not commit full raw datasets or large log files to this repository.

## Main Design Idea

LogPilot does not claim that LLMs always outperform traditional parsers.

The product strategy is:

1. Use Drain for fast full-volume parsing.
2. Use LLM Direct to inspect model capability and structured output behavior.
3. Use Hybrid Review to send only risky templates to LLM.
4. Use template evolution analysis to identify stable, new, disappeared, and rewritten templates.
5. Use fallback and review logic to make the AI workflow more controllable.

## Roadmap

* Add more parser baselines.
* Add better template matching metrics with ground truth.
* Improve LLM prompt templates.
* Add cost estimation for different providers.
* Add human review workflow for template correction.
* Add richer template evolution alignment logic.

## License

This repository is intended for academic and demonstration purposes. Add a license before making the repository public.

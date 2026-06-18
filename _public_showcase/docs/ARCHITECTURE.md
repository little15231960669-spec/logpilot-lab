# Architecture

LogPilot Agent Lab provides two Streamlit entrypoints:

- `app.py`: a classic method-centric dashboard for Drain baseline, LLM-assisted parsing, hybrid review, and template evolution.
- `agent_app.py`: a task-centric Agent Lab that plans parser tool usage and presents results, risk samples, recommendations, and downloadable traces.

The Agent Lab uses deterministic Python tools for offline demos:

- Drain baseline parsing.
- Template memory parsing over small CSV template stores.
- Batch comparison between parser outputs.
- Summary generation for review recommendations.

Online structured parsing can be enabled with an OpenAI-compatible API configuration in a local `.env` file.

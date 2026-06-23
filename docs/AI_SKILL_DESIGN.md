# AI Skill Design

LogPilot Agent Lab models log-analysis work as reusable task skills. Each skill has a clear contract and can be selected or composed by the Agent without changing the existing LangChain, RAG, or parser architecture.

| AI Skill | Goal | Input | Tool Chain | Output | Fallback | Evaluation Signal |
| --- | --- | --- | --- | --- | --- | --- |
| **Log Parsing Skill** | Convert raw log lines into structured events and reusable templates. | Selected log lines, dataset context, parser configuration. | Preprocess logs -> run Drain or structured parser -> normalize results. | Parsed rows, template IDs, template text, and basic counts. | Use deterministic Drain parsing when an LLM is unavailable or structured output is invalid. | Parse success rate, valid structured-output rate, template match rate, latency. |
| **Template Memory Retrieval Skill** | Reuse historical template knowledge to improve consistency and provide context. | Query logs, template-memory CSV, retrieval settings. | Validate memory -> retrieve similar templates -> attach references to parsing context. | Ranked template candidates with similarity and source metadata. | Continue without memory and surface a dataset or memory warning when retrieval is unavailable. | Retrieval hit rate, top-k relevance, downstream template match improvement. |
| **Risk Review Skill** | Detect uncertain or inconsistent parsing results and recommend review actions. | Parsed outputs, comparison signals, current logs, optional retrieved templates. | Compare parser outputs -> score disagreement or risk -> summarize evidence and recommendation. | Risk samples, comparison details, review decision, and rationale. | Return deterministic comparison results and mark the run as partial when LLM review fails. | Risk precision, review rate, final decision accuracy, fallback rate. |
| **Report Export Skill** | Turn an Agent run into a traceable artifact for review and sharing. | Run status, plan, tool trace, parsed results, findings, recommendations. | Assemble run data -> render JSON trace and Markdown report -> expose export action. | Downloadable trace and report with decisions and evidence. | Preserve in-app results and export the available partial trace if file generation fails. | Export success rate, field completeness, trace reproducibility, reviewer usability. |

## Composition

A task goal selects the smallest useful chain. Fast parsing can use only **Log Parsing**, while review tasks can compose **Template Memory Retrieval -> Log Parsing -> Risk Review -> Report Export**. Existing offline fallbacks keep the workflow demonstrable without API credentials.

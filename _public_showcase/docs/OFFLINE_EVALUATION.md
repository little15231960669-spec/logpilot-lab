# Offline Evaluation

The public demo includes a small offline evaluation dataset under `data/eval/`.

The evaluation checks whether the Agent selects the expected tool plan, retrieves suitable templates, produces expected parse templates, and returns a successful run status. It is designed to run without model access, so the repository can be tested in a clean environment.

Run:

```bash
python -m pytest tests
```

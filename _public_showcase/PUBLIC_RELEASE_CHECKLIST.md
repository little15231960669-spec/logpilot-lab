# Public Release Checklist

- [ ] `.env` is removed.
- [ ] API keys or secrets are not present.
- [ ] `results/` and generated traces are excluded.
- [ ] Large raw logs are excluded.
- [ ] Only small demo template memory and eval files are included.
- [ ] `streamlit run agent_app.py` works.
- [ ] `streamlit run app.py` works.
- [ ] `python -m pytest tests` works.
- [ ] README is concise and does not include development history.

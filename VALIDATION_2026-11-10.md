# Validation rerun — 10 November 2026

**PASS:** `../../.venv/bin/python -m unittest discover -s tests -v` (1 focused test).

**PASS:** `../../.venv/bin/python run_fake_news_detection.py` against the tracked Fake/True article data. The script removed exact normalized-content duplicates, required valid dates, and used disjoint chronological training, validation and test periods. TF-IDF was fitted on training rows only. `validation_results.json` and `selected_model.joblib` were generated locally; they are ignored by Git.

- Selected by validation F1: logistic regression (validation F1 0.9804).
- Untouched test: 16,025 rows, 2017-06-19 through 2018-02-19; accuracy 0.9289, F1 0.9546, ROC-AUC 0.9947.
- Training: 14,559 rows through 2016-11-28; validation: 8,507 rows through 2017-06-18.

**NOT TESTED:** publisher/source-held-out performance, near-duplicate families, out-of-dataset claims, and current news. Source IDs are absent. The older 99.83% figure and previously committed pickle artifacts are historical, invalid for current model claims.

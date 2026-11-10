# Fake News Classification: Dataset Baseline

This educational text classifier compares logistic regression and Naive Bayes on the bundled Kaggle Fake/True article dataset. It does not determine whether an arbitrary claim or news article is true. In this dataset, publisher and writing-style cues can correlate with the label.

## Data and method

The script reads `archive (2)/Fake.csv` and `True.csv`, or the tracked ZIP. The original source is the Kaggle Fake and Real News Dataset; confirm its license and redistribution terms before external reuse. It removes exact normalized-content duplicates and contradictory labels, drops undated rows, then splits by calendar date: earliest 60% of distinct dates for training, next 20% for model selection, latest 20% as untouched test. Both classes must appear in each period. TF-IDF is fitted inside each training-only pipeline. Validation F1 selects the model; accuracy, F1 and ROC-AUC on the test use predictions or probability scores as appropriate.

The dataset has no reliable publisher identifier. Source-held-out performance cannot be established from these files. Near duplicates, syndication, topics, and temporal drift remain limitations.

## Run

```bash
pip install -r requirements.txt
python run_fake_news_detection.py
python -m unittest discover -s tests
```

Outputs are `selected_model.joblib` and `validation_results.json`. The JSON records the actual model, split dates and metrics. Only load joblib files you generated or trust. Historical charts, pickles and the previously stated 99.83% accuracy were produced under the older protocol and should not be presented as validated current results.

No software license file is included. Repository description suggestion: “Chronological text-classification baseline on a labeled news dataset, with duplicate checks and explicit validation limits.”

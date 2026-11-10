#!/usr/bin/env python3
"""Chronological, duplicate-aware baseline for the bundled article dataset."""

from pathlib import Path
import zipfile
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


DATA_DIR = Path('archive (2)')


def load_articles(data_dir=DATA_DIR):
    rows = []
    for filename, label in [('Fake.csv', 0), ('True.csv', 1)]:
        path = data_dir / filename
        if path.exists():
            frame = pd.read_csv(path)
        elif Path('archive (2).zip').exists():
            with zipfile.ZipFile('archive (2).zip') as archive:
                with archive.open(filename) as source:
                    frame = pd.read_csv(source)
        else:
            raise FileNotFoundError(f'Provide {path} or archive (2).zip')
        frame['label'] = label
        rows.append(frame)
    frame = pd.concat(rows, ignore_index=True)
    frame['date'] = pd.to_datetime(frame['date'].astype(str).str.strip(), format='mixed', errors='coerce')
    frame['content'] = frame['title'].fillna('') + ' ' + frame['text'].fillna('')
    frame['content_key'] = frame['content'].str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    frame = frame.dropna(subset=['date'])
    frame = frame[frame['content_key'].str.len() > 40]
    # Conflicting labels cannot be used as supervised ground truth.
    conflicts = frame.groupby('content_key')['label'].transform('nunique') > 1
    frame = frame[~conflicts].drop_duplicates('content_key')
    return frame.sort_values('date').reset_index(drop=True)


def chronological_split(frame):
    """Use disjoint calendar dates for train, validation and untouched test."""
    dates = np.sort(frame['date'].dt.normalize().unique())
    if len(dates) < 5:
        raise ValueError('At least five distinct dates are needed for a chronological split')
    first = dates[int(len(dates) * 0.6)]
    second = dates[int(len(dates) * 0.8)]
    train = frame[frame['date'] < first]
    validation = frame[(frame['date'] >= first) & (frame['date'] < second)]
    test = frame[frame['date'] >= second]
    for name, part in [('train', train), ('validation', validation), ('test', test)]:
        if part['label'].nunique() != 2:
            raise ValueError(f'{name} lacks both classes; no valid dated evaluation is possible')
    return train, validation, test


def metrics(model, frame):
    labels = frame['label']
    predicted = model.predict(frame['content'])
    scores = model.predict_proba(frame['content'])[:, 1]
    return {
        'accuracy': float(accuracy_score(labels, predicted)),
        'f1': float(f1_score(labels, predicted)),
        'roc_auc': float(roc_auc_score(labels, scores)),
    }


def run(frame, output_dir=Path('.')):
    train, validation, test = chronological_split(frame)
    candidates = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'naive_bayes': MultinomialNB(),
    }
    fitted = {}
    validation_metrics = {}
    for name, classifier in candidates.items():
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=2000, min_df=2)),
            ('classifier', classifier),
        ])
        pipeline.fit(train['content'], train['label'])
        fitted[name] = pipeline
        validation_metrics[name] = metrics(pipeline, validation)
    selected = max(candidates, key=lambda name: validation_metrics[name]['f1'])
    test_metrics = metrics(fitted[selected], test)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[selected], output_dir / 'selected_model.joblib')
    report = {
        'selected_model': selected,
        'selection_metric': 'validation_f1',
        'validation_metrics': validation_metrics,
        'untouched_test_metrics': test_metrics,
        'split_dates': {name: [str(part['date'].min().date()), str(part['date'].max().date())]
                        for name, part in [('train', train), ('validation', validation), ('test', test)]},
        'row_counts': {name: len(part) for name, part in [('train', train), ('validation', validation), ('test', test)]},
        'limitations': 'Publisher/source IDs are absent; this is not a source-held-out evaluation.',
    }
    (output_dir / 'validation_results.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    result = run(load_articles())
    print(json.dumps(result, indent=2))

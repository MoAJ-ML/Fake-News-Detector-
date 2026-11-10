import tempfile
import unittest

import pandas as pd

from run_fake_news_detection import chronological_split, run


class ValidationTests(unittest.TestCase):
    def test_dates_are_disjoint_and_model_matches_report(self):
        rows = []
        for day in range(12):
            for label in (0, 1):
                rows.append({
                    'date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=day),
                    'content': f"{'fabricated' if label == 0 else 'verified'} article topic {day} repeated word",
                    'label': label,
                })
        frame = pd.DataFrame(rows)
        train, validation, test = chronological_split(frame)
        self.assertLess(train['date'].max(), validation['date'].min())
        self.assertLess(validation['date'].max(), test['date'].min())
        with tempfile.TemporaryDirectory() as tmp:
            report = run(frame, tmp)
            self.assertIn(report['selected_model'], report['validation_metrics'])
            self.assertGreaterEqual(report['untouched_test_metrics']['roc_auc'], 0)
            self.assertLessEqual(report['untouched_test_metrics']['roc_auc'], 1)


if __name__ == '__main__':
    unittest.main()

"""
learning_curve.py
Generates the data required for a learning curve: the accuracy of an algorithm
for different training set sizes (default: 20%, 40%, 60%, 80%, 100%).

Purpose:
To determine whether the algorithm improves as more training data is provided
(a sign of high variance / overfitting when data is limited), or whether it
reaches its performance ceiling early and no longer improves
(a sign of high bias / underfitting).
"""

import numpy as np

from evaluation.metrics import Metrics


class LearningCurve:
    def __init__(self, fractions=None, random_state=42):
        self.fractions = fractions or [0.2, 0.4, 0.6, 0.8, 1.0]
        self.random_state = random_state
        self.metrics = Metrics()

    @staticmethod
    def _select_rows(data, idx):
        """
        Returns a subset of samples without changing the original data type
        (DataFrame/Series vs ndarray).
        """
        if hasattr(data, "iloc"):  # pandas DataFrame or Series
            return data.iloc[idx].reset_index(drop=True)
        return np.asarray(data)[idx]

    def generate(self, model_factory, X_train, y_train, X_test, y_test):
        """
        model_factory: A zero-argument function that returns a fresh model instance.

        For each fraction in self.fractions:
            1) A random subset of the training data with that size is selected.
            2) The model is trained on that subset.
            3) Accuracy is evaluated on the fixed X_test/y_test set
               (the full test set).

        Returns:
            A list of accuracies with the same length as self.fractions.
        """
        n = len(X_train)
        rng = np.random.default_rng(self.random_state)
        shuffled_idx = rng.permutation(n)

        accuracies = []
        for frac in self.fractions:
            n_sub = max(2, int(round(frac * n)))
            sub_idx = shuffled_idx[:n_sub]

            X_sub = self._select_rows(X_train, sub_idx)
            y_sub = self._select_rows(y_train, sub_idx)

            model = model_factory()
            model.fit(X_sub, y_sub)
            preds = model.predict_batch(X_test)

            acc = self.metrics.accuracy(y_test, preds)
            accuracies.append(acc)

        return accuracies
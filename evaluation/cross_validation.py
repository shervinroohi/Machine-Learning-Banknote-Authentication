"""
cross_validation.py
K-Fold Cross-Validation (default 5-Fold) for comprehensive model evaluation.

This implementation splits the data into K folds and evaluates the model's
performance across multiple metrics including Accuracy, Precision, Recall, and F1-Score.
"""

import numpy as np
from evaluation.metrics import Metrics


class CrossValidation:
    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.metrics = Metrics()

    @staticmethod
    def _select_rows(data, idx):
        """
        Selects a subset of data by indices while preserving the original
        format (Pandas vs Numpy).
        """
        if hasattr(data, "iloc"):  # Handle pandas DataFrame or Series
            return data.iloc[idx].reset_index(drop=True)
        return np.asarray(data)[idx]

    def split_folds(self, X, y):
        """
        Divides dataset indices into n_folds approximately equal segments.
        """
        n = len(X)
        rng = np.random.default_rng(self.random_state)
        indices = rng.permutation(n)

        fold_sizes = np.full(self.n_folds, n // self.n_folds, dtype=int)
        fold_sizes[: n % self.n_folds] += 1  # Distribute remainder across folds

        folds = []
        current = 0
        for size in fold_sizes:
            folds.append(indices[current: current + size])
            current += size

        return folds

    def evaluate(self, model_factory, X, y):
        """
        Performs K-fold cross-validation and calculates mean/std for various metrics.

        model_factory: A function that returns a fresh model instance.
        Returns: A dictionary containing scores for Accuracy, Precision, Recall, and F1.
        """
        folds = self.split_folds(X, y)
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for i in range(self.n_folds):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(self.n_folds) if j != i])

            X_train_fold, y_train_fold = self._select_rows(X, train_idx), self._select_rows(y, train_idx)
            X_test_fold, y_test_fold = self._select_rows(X, test_idx), self._select_rows(y, test_idx)

            # Re-initialize the model for each fold
            model = model_factory()
            model.fit(X_train_fold, y_train_fold)
            preds = model.predict_batch(X_test_fold)

            # Calculate and store metrics for the current fold
            accuracy_scores.append(self.metrics.accuracy(y_test_fold, preds))
            precision_scores.append(self.metrics.precision(y_test_fold, preds))
            recall_scores.append(self.metrics.recall(y_test_fold, preds))
            f1_scores.append(self.metrics.f1_score(y_test_fold, preds))

        # Compile final results with mean and standard deviation
        results = {
            "accuracy": {
                "mean": float(np.mean(accuracy_scores)),
                "std": float(np.std(accuracy_scores)),
                "scores": accuracy_scores
            },
            "precision": {
                "mean": float(np.mean(precision_scores)),
                "std": float(np.std(precision_scores)),
                "scores": precision_scores
            },
            "recall": {
                "mean": float(np.mean(recall_scores)),
                "std": float(np.std(recall_scores)),
                "scores": recall_scores
            },
            "f1": {
                "mean": float(np.mean(f1_scores)),
                "std": float(np.std(f1_scores)),
                "scores": f1_scores
            }
        }

        return results
"""
metrics.py
Classification evaluation metrics: Accuracy, Precision, Recall, and F1-Score.
Implemented from scratch without using sklearn.

This project is a binary classification problem:
    0 = Genuine (Negative)
    1 = Fake (Positive)

By default, class "1" (Fake) is treated as the positive class because detecting
fake banknotes is usually the more important objective in this task.
"""

import numpy as np


class Metrics:
    def __init__(self, positive_class=1):
        self.positive_class = positive_class

    @staticmethod
    def _to_numpy(arr):
        """
        Converts input data to a flattened NumPy array.

        Supports pandas Series, lists, and NumPy arrays.
        """
        if hasattr(arr, "values"):  # pandas Series
            return np.asarray(arr.values).ravel()
        return np.asarray(arr).ravel()

    def _confusion_counts(self, y_true, y_pred):
        """
        Computes TP, FP, FN, and TN with respect to the positive class.
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        pos = self.positive_class

        tp = int(np.sum((y_pred == pos) & (y_true == pos)))
        fp = int(np.sum((y_pred == pos) & (y_true != pos)))
        fn = int(np.sum((y_pred != pos) & (y_true == pos)))
        tn = int(np.sum((y_pred != pos) & (y_true != pos)))

        return tp, fp, fn, tn

    def accuracy(self, y_true, y_pred):
        """
        Returns the percentage of correctly predicted samples.

        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        if len(y_true) == 0:
            return 0.0
        return float(np.mean(y_true == y_pred))

    def precision(self, y_true, y_pred):
        """
        Returns the proportion of predicted positive samples that are actually positive.

        Precision = TP / (TP + FP)
        """
        tp, fp, fn, tn = self._confusion_counts(y_true, y_pred)
        denom = tp + fp
        return float(tp / denom) if denom > 0 else 0.0

    def recall(self, y_true, y_pred):
        """
        Returns the proportion of actual positive samples that were correctly detected.

        Recall = TP / (TP + FN)
        """
        tp, fp, fn, tn = self._confusion_counts(y_true, y_pred)
        denom = tp + fn
        return float(tp / denom) if denom > 0 else 0.0

    def f1_score(self, y_true, y_pred):
        """
        Returns the harmonic mean of Precision and Recall.

        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        """
        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)
        denom = p + r
        return float(2 * p * r / denom) if denom > 0 else 0.0

    def confusion_matrix(self, y_true, y_pred):
        """
        Returns a dictionary containing TP, FP, FN, and TN for reporting or debugging.
        """
        tp, fp, fn, tn = self._confusion_counts(y_true, y_pred)
        return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}

    def report(self, y_true, y_pred):
        """
        Returns a summary dictionary of all metrics, suitable for printing or logging.
        """
        return {
            "accuracy": self.accuracy(y_true, y_pred),
            "precision": self.precision(y_true, y_pred),
            "recall": self.recall(y_true, y_pred),
            "f1_score": self.f1_score(y_true, y_pred),
        }
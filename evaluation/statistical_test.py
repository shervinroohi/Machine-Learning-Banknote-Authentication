"""
statistical_test.py
Paired T-Test for statistical comparison between two algorithms.

According to project constraints, scipy.stats is used only for the T-test
calculation (not for classification algorithm implementations).

Null Hypothesis (H0): The mean accuracies of the two algorithms are equal.
If p-value < 0.05, H0 is rejected, and the difference is considered "Statistically Significant".
"""

from scipy import stats


class StatisticalTest:
    def paired_t_test(self, accuracies_1, accuracies_2):
        """
        Calculates the p-value using a Paired T-Test.

        Args:
            accuracies_1: List of accuracy scores for the first model (e.g., from K-Fold).
            accuracies_2: List of accuracy scores for the second model on the same folds.

        Returns:
            p_value (float)
        """
        if len(accuracies_1) != len(accuracies_2):
            raise ValueError(
                "Both accuracy lists must have the same length (results must be "
                "calculated on identical folds)."
            )

        if len(accuracies_1) < 2:
            raise ValueError("At least 2 data points (folds) are required for a Paired T-Test.")

        # Perform the paired sample t-test
        t_statistic, p_value = stats.ttest_rel(accuracies_1, accuracies_2)
        return float(p_value)

    def compare(self, name_1, accuracies_1, name_2, accuracies_2, alpha=0.05):
        """
        Provides a human-readable comparison report including the p-value and interpretation.

        Returns:
            A dictionary containing the comparison results and significance verdict.
        """
        p_value = self.paired_t_test(accuracies_1, accuracies_2)
        is_significant = p_value < alpha
        verdict = "Significant" if is_significant else "Not Significant"

        return {
            "pair": f"{name_1} vs {name_2}",
            "p_value": p_value,
            "significant": is_significant,
            "verdict": verdict,
        }
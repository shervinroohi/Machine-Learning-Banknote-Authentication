import matplotlib.pyplot as plt


class Plotter:
    """
    Provides plotting utilities for algorithm comparison and learning curves.
    """

    def __init__(self, output_dir="."):
        self.output_dir = output_dir

    def plot_algorithm_comparison(self, accuracies, std_devs=None, save_path=None, show=True):
        """
        Plots a bar chart to compare the accuracy of multiple algorithms.

        If standard deviations are provided, error bars are displayed as well.
        """
        names = list(accuracies.keys())
        values = [accuracies[name] for name in names]
        errors = [std_devs.get(name, 0.0) for name in names] if std_devs else [0.0] * len(names)

        plt.figure(figsize=(7, 5))
        bars = plt.bar(
            names,
            values,
            yerr=errors,
            capsize=6,
            color=["#4C72B0", "#55A868", "#C44E52"]
        )
        plt.ylim(0, 1.05)
        plt.ylabel("Accuracy")
        plt.title("Algorithm Accuracy Comparison (Error Bars = Standard Deviation)")

        # Display the accuracy value above each bar
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()

        path = save_path or f"{self.output_dir}/algorithm_comparison.png"
        plt.savefig(path, dpi=150)
        if show:
            plt.show()
        plt.close()
        return path

    def plot_learning_curve(self, curves, fractions, save_path=None, show=True):
        """
        Plots the learning curve of each algorithm over different training fractions.
        """
        percentages = [f * 100 for f in fractions]

        plt.figure(figsize=(7, 5))
        for name, accs in curves.items():
            plt.plot(percentages, accs, marker="o", label=name)

        plt.xlabel("Training Data Used (%)")
        plt.ylabel("Accuracy")
        plt.title("Learning Curve of Each Algorithm")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        path = save_path or f"{self.output_dir}/learning_curve.png"
        plt.savefig(path, dpi=150)
        if show:
            plt.show()
        plt.close()
        return path
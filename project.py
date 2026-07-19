import numpy as np

from preprocessing.data_loader import DataLoader
from preprocessing.splitter import DataSplitter
from preprocessing.normalizer import MinMaxNormalizer

from algorithms.knn import KNN
from algorithms.naive_bayes import NaiveBayes
from algorithms.decision_tree import DecisionTree

DATA_PATH = "data/banknote_authentication.csv"
FEATURE_NAMES = ["variance", "skewness", "curtosis", "entropy"]


# ----------------------------------------------------------------------
# Helper utility: If evaluation/visualization modules haven't been built yet,
# the program won't crash, just skip that section.
# ----------------------------------------------------------------------
def try_import(module_path, class_name):
    """
    Attempt to import a class from a module. Returns None if the module or class doesn't exist.
    """
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ModuleNotFoundError, ImportError):
        # Module/file not found
        return None
    except AttributeError:
        # File exists but class is not defined inside it (e.g., empty file or not fully written)
        print(f"  [Warning] File {module_path.replace('.', '/')}.py found but class '{class_name}' is missing/incomplete. This section will be skipped.")
        return None


Metrics = try_import("evaluation.metrics", "Metrics")
CrossValidation = try_import("evaluation.cross_validation", "CrossValidation")
StatisticalTest = try_import("evaluation.statistical_test", "StatisticalTest")
LearningCurve = try_import("evaluation.learning_curve", "LearningCurve")
Plotter = try_import("visualization.plots", "Plotter")


def section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def basic_accuracy(y_true, y_pred):
    """
    Simple accuracy calculation, independent of evaluation/metrics.py.
    Always available as a fallback.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def main():
    # ------------------------------------------------------------
    # 1) Load Dataset
    # ------------------------------------------------------------
    section("1) Loading Dataset")
    X, y = DataLoader(DATA_PATH).load_data()
    print(f"Total samples: {len(X)}")
    print(f"Feature columns: {list(X.columns) if hasattr(X, 'columns') else FEATURE_NAMES}")
    print(f"Class distribution:\n{y.value_counts() if hasattr(y, 'value_counts') else np.unique(y, return_counts=True)}")

    # ------------------------------------------------------------
    # 2) Split Dataset (70 / 15 / 15)
    # ------------------------------------------------------------
    section("2) Data Split (Train 70% / Validation 15% / Test 15%)")
    X_train, y_train, X_val, y_val, X_test, y_test = DataSplitter().split(X, y)
    print(f"Train: {len(X_train)}   Validation: {len(X_val)}   Test: {len(X_test)}")

    # ------------------------------------------------------------
    # 3) Normalize (Only for KNN)
    # ------------------------------------------------------------
    section("3) Min-Max Normalization (Only for KNN)")
    normalizer = MinMaxNormalizer()
    X_train_norm = normalizer.fit_transform(X_train)
    X_val_norm = normalizer.transform(X_val)
    X_test_norm = normalizer.transform(X_test)
    print("Normalization completed. Each feature range should be between 0 and 1:")
    print(f"  min={X_train_norm.min(axis=0)}")
    print(f"  max={X_train_norm.max(axis=0)}")

    # ------------------------------------------------------------
    # 4) Train KNN (k = 1, 3, 5, 7) on normalized data
    # ------------------------------------------------------------
    section("4) Training and Testing KNN for different k values")
    knn_results = {}
    for k in [1, 3, 5, 7]:
        knn = KNN(k=k)
        knn.fit(X_train_norm, y_train)
        preds = knn.predict_batch(X_test_norm)
        acc = basic_accuracy(y_test, preds)
        knn_results[k] = acc
        print(f"  k={k}: Test Accuracy = {acc:.4f}")

    best_k = max(knn_results, key=knn_results.get)
    print(f"Best k: {best_k} with accuracy {knn_results[best_k]:.4f}")

    knn_best = KNN(k=best_k)
    knn_best.fit(X_train_norm, y_train)
    knn_preds = knn_best.predict_batch(X_test_norm)

    # ------------------------------------------------------------
    # 5) Train Naive Bayes (on raw data, without normalization)
    # ------------------------------------------------------------
    section("5) Training and Testing Naive Bayes")
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    nb_preds = nb.predict_batch(X_test)
    nb_acc = basic_accuracy(y_test, nb_preds)
    print(f"Naive Bayes Test Accuracy = {nb_acc:.4f}")

    # ------------------------------------------------------------
    # 6) Train Decision Tree (on raw data + Pruning with Validation)
    # ------------------------------------------------------------
    section("6) Training, Pruning, and Testing Decision Tree")
    dt = DecisionTree(max_depth=5)
    dt.fit(X_train, y_train, feature_names=FEATURE_NAMES)

    dt_preds_before = dt.predict_batch(X_test)
    dt_acc_before = basic_accuracy(y_test, dt_preds_before)
    print(f"Accuracy before Pruning: {dt_acc_before:.4f}  (depth={dt.get_depth()}, nodes={dt.count_nodes()})")

    dt.prune(X_val, y_val)
    dt_preds_after = dt.predict_batch(X_test)
    dt_acc_after = basic_accuracy(y_test, dt_preds_after)
    print(f"Accuracy after Pruning:  {dt_acc_after:.4f}  (depth={dt.get_depth()}, nodes={dt.count_nodes()})")
    dt_preds = dt_preds_after

    # ------------------------------------------------------------
    # Summary of base accuracy for the three algorithms (this section always works)
    # ------------------------------------------------------------
    section("Accuracy Summary on Test Set")
    print(f"  KNN (k={best_k}):     {knn_results[best_k]:.4f}")
    print(f"  Naive Bayes:          {nb_acc:.4f}")
    print(f"  Decision Tree:        {dt_acc_after:.4f}")

    # ------------------------------------------------------------
    # 7) Evaluate with full Metrics (Precision / Recall / F1) - Optional
    # ------------------------------------------------------------
    section("7) Full Evaluation (Precision / Recall / F1)")
    if Metrics is not None:
        metrics = Metrics()
        for name, preds in [("KNN", knn_preds), ("Naive Bayes", nb_preds), ("Decision Tree", dt_preds)]:
            acc = metrics.accuracy(y_test, preds)
            prec = metrics.precision(y_test, preds)
            rec = metrics.recall(y_test, preds)
            f1 = metrics.f1_score(y_test, preds)
            print(f"  {name:15s} | Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    else:
        print("  [Skipped] evaluation/metrics.py has not been built yet.")

    # ------------------------------------------------------------
    # 8) 5-Fold Cross-Validation
    # ------------------------------------------------------------
    section("8) 5-Fold Cross-Validation")
    cv_results = {}
    if CrossValidation is not None:
        cv = CrossValidation(n_folds=5)
        for name, model_factory, use_norm in [
            ("KNN", lambda: KNN(k=best_k), True),
            ("Naive Bayes", lambda: NaiveBayes(), False),
            ("Decision Tree", lambda: DecisionTree(max_depth=5), False),
        ]:
            Xd = X_train_norm if use_norm else X_train
            results = cv.evaluate(model_factory, Xd, y_train)
            cv_results[name] = results

            print(
                f"  {name:15s} | "
                f"Acc={results['accuracy']['mean']:.4f} ± {results['accuracy']['std']:.4f} | "
                f"Prec={results['precision']['mean']:.4f} ± {results['precision']['std']:.4f} | "
                f"Rec={results['recall']['mean']:.4f} ± {results['recall']['std']:.4f} | "
                f"F1={results['f1']['mean']:.4f} ± {results['f1']['std']:.4f}"
            )
    else:
        print("  [Skipped] evaluation/cross_validation.py has not been built yet.")

    # ------------------------------------------------------------
    # 9) Paired T-Test Statistical Test between each pair of algorithms
    # ------------------------------------------------------------
    section("9) Paired T-Test Statistical Test")
    if StatisticalTest is not None and len(cv_results) == 3:
        test = StatisticalTest()
        names = list(cv_results.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                p_value = test.paired_t_test(
                    cv_results[names[i]]["accuracy"]["scores"],
                    cv_results[names[j]]["accuracy"]["scores"]
                )
                significance = "Significant" if p_value < 0.05 else "Not Significant"
                print(f"  {names[i]} vs {names[j]}: p-value = {p_value:.5f}  ({significance})")
    else:
        print("  [Skipped] evaluation/statistical_test.py not built yet or CV results unavailable.")

    # ------------------------------------------------------------
    # 10) Learning Curve
    # ------------------------------------------------------------
    section("10) Learning Curve (20% to 100% of training data)")
    learning_curves = {}
    if LearningCurve is not None:
        lc = LearningCurve(fractions=[0.2, 0.4, 0.6, 0.8, 1.0])
        for name, model_factory, use_norm in [
            ("KNN", lambda: KNN(k=best_k), True),
            ("Naive Bayes", lambda: NaiveBayes(), False),
            ("Decision Tree", lambda: DecisionTree(max_depth=5), False),
        ]:
            Xd_train = X_train_norm if use_norm else X_train
            Xd_test = X_test_norm if use_norm else X_test
            accs = lc.generate(model_factory, Xd_train, y_train, Xd_test, y_test)
            learning_curves[name] = accs
            print(f"  {name:15s} | {accs}")
    else:
        print("  [Skipped] evaluation/learning_curve.py has not been built yet.")

    # ------------------------------------------------------------
    # 11) Plotting
    # ------------------------------------------------------------
    section("11) Plotting Charts")
    if Plotter is not None:
        plotter = Plotter()
        final_accs = {
            "KNN": knn_results[best_k],
            "Naive Bayes": nb_acc,
            "Decision Tree": dt_acc_after
        }
        std_devs = {
            name: results["accuracy"]["std"]
            for name, results in cv_results.items()
        } if cv_results else {}
        plotter.plot_algorithm_comparison(final_accs, std_devs)
        if learning_curves:
            plotter.plot_learning_curve(learning_curves, fractions=[0.2, 0.4, 0.6, 0.8, 1.0])
        print("  Plots saved/displayed.")
    else:
        print("  [Skipped] visualization/plots.py has not been built yet.")

    section("Execution Finished")


if __name__ == "__main__":
    main()
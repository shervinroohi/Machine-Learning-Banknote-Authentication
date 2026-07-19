"""
decision_tree.py
Implementation of the ID3 Decision Tree algorithm from scratch.

Features are continuous and discretized at each node using a mean threshold
into 'low' (<= threshold) and 'high' (> threshold) categories.
Feature selection criterion: Information Gain based on Entropy.

Stopping conditions (Pre-pruning):
    - All samples in a node belong to the same class.
    - Node depth reaches max_depth.
    - Best possible Gain is <= 0 (splitting provides no benefit).

Post-Pruning:
Performed using Reduced Error Pruning on a Validation Set. Each internal node
is evaluated bottom-up; if converting it to a leaf does not decrease
(or improves) accuracy on the validation set, the node is pruned into a leaf.
"""

import numpy as np


class _Node:
    """
    A single node in the decision tree.
    Used for both internal nodes and leaves.
    """

    __slots__ = (
        "is_leaf",
        "prediction",
        "feature_index",
        "threshold",
        "left",
        "right",
        "n_samples",
    )

    def __init__(self, prediction, n_samples):
        # prediction: Majority class in this node (used as leaf value if pruned)
        self.is_leaf = True
        self.prediction = prediction
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.n_samples = n_samples


class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None
        self.feature_names = None  # Optional: for readable printing/debugging

    # ------------------------------------------------------------------
    # Helper methods to convert input (DataFrame/ndarray/list) to numpy
    # ------------------------------------------------------------------
    @staticmethod
    def _to_numpy_2d(X):
        if hasattr(X, "values"):  # pandas DataFrame
            return np.asarray(X.values, dtype=float)
        return np.asarray(X, dtype=float)

    @staticmethod
    def _to_numpy_1d(y):
        if hasattr(y, "values"):  # pandas Series
            return np.asarray(y.values).ravel()
        return np.asarray(y).ravel()

    @staticmethod
    def _row_to_numpy(x):
        if hasattr(x, "values"):  # pandas Series (single sample)
            return np.asarray(x.values, dtype=float)
        return np.asarray(x, dtype=float)

    # ------------------------------------------------------------------
    # Entropy calculation
    # ------------------------------------------------------------------
    def entropy(self, y):
        """
        Calculates the entropy of a set of labels.
        e.g., entropy([0,0,1,1]) == 1.0, entropy([0,0,0,0]) == 0.0
        """
        y = np.asarray(y)
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        # Avoid log2(0)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    # ------------------------------------------------------------------
    # Split data based on a feature and threshold
    # ------------------------------------------------------------------
    def split_data(self, X, y, feature_index, threshold):
        """
        Returns: X_low, y_low, X_high, y_high
        low  -> feature value <= threshold
        high -> feature value > threshold
        """
        mask_low = X[:, feature_index] <= threshold
        mask_high = ~mask_low
        return X[mask_low], y[mask_low], X[mask_high], y[mask_high]

    # ------------------------------------------------------------------
    # Information Gain
    # ------------------------------------------------------------------
    def information_gain(self, X, y, feature_index, threshold, parent_entropy=None):
        if parent_entropy is None:
            parent_entropy = self.entropy(y)

        _, y_low, _, y_high = self.split_data(X, y, feature_index, threshold)

        n = len(y)
        if len(y_low) == 0 or len(y_high) == 0:
            return 0.0  # Splitting that results in an empty side is useless

        w_low = len(y_low) / n
        w_high = len(y_high) / n
        weighted_entropy = w_low * self.entropy(y_low) + w_high * self.entropy(y_high)

        return parent_entropy - weighted_entropy

    # ------------------------------------------------------------------
    # Find the best feature for splitting a node
    # ------------------------------------------------------------------
    def best_split(self, X, y):
        """
        For each feature, the threshold is the mean of the feature values in the current node.
        Returns: (best_feature_index, best_threshold, best_gain)
                 If no useful split exists -> (None, None, 0)
        """
        n_features = X.shape[1]
        parent_entropy = self.entropy(y)

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature_index in range(n_features):
            threshold = float(np.mean(X[:, feature_index]))
            gain = self.information_gain(X, y, feature_index, threshold, parent_entropy)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold

        return best_feature, best_threshold, best_gain

    # ------------------------------------------------------------------
    # Build tree (Recursive)
    # ------------------------------------------------------------------
    def _majority_class(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    def build_tree(self, X, y, depth=0):
        majority = self._majority_class(y)
        node = _Node(prediction=majority, n_samples=len(y))

        # Stopping condition 1: Node is pure (all samples belong to one class)
        if len(np.unique(y)) == 1:
            return node

        # Stopping condition 2: Maximum depth reached
        if depth >= self.max_depth:
            return node

        # Stopping condition 3: Not enough samples to split
        if len(y) < 2:
            return node

        feature_index, threshold, gain = self.best_split(X, y)

        # Stopping condition 4: No split provides positive Information Gain
        if feature_index is None or gain <= 0:
            return node

        X_low, y_low, X_high, y_high = self.split_data(X, y, feature_index, threshold)

        # Fallback (should be caught by information_gain) -> Leaf
        if len(y_low) == 0 or len(y_high) == 0:
            return node

        node.is_leaf = False
        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self.build_tree(X_low, y_low, depth + 1)   # <= threshold
        node.right = self.build_tree(X_high, y_high, depth + 1)  # > threshold

        return node

    # ------------------------------------------------------------------
    # Fit / Predict
    # ------------------------------------------------------------------
    def fit(self, X, y, feature_names=None):
        X = self._to_numpy_2d(X)
        y = self._to_numpy_1d(y)
        self.feature_names = feature_names
        self.root = self.build_tree(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.prediction
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, x):
        """
        Predict class for a single sample.
        x can be a pandas Series, list, or numpy array.
        """
        if self.root is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")
        x = self._row_to_numpy(x)
        return self._predict_one(x, self.root)

    def predict_batch(self, X):
        """
        Predict classes for multiple samples.
        Compatible with KNN and NaiveBayes predict_batch methods.
        """
        X_np = self._to_numpy_2d(X)
        predictions = []
        for i in range(len(X_np)):
            predictions.append(self._predict_one(X_np[i], self.root))
        return predictions

    # ------------------------------------------------------------------
    # Post-Pruning using Validation Set (Reduced Error Pruning)
    # ------------------------------------------------------------------
    def _accuracy(self, X, y):
        preds = np.asarray(self.predict_batch(X))
        y = self._to_numpy_1d(y)
        return float(np.mean(preds == y))

    def _prune_node(self, node, X_val, y_val):
        if node.is_leaf:
            return node

        # Prune children first (Bottom-Up approach)
        node.left = self._prune_node(node.left, X_val, y_val)
        node.right = self._prune_node(node.right, X_val, y_val)

        # Check if accuracy improves/stays same by converting this node to a leaf
        acc_before = self._accuracy(X_val, y_val)

        # Temporarily set as leaf
        node.is_leaf = True
        acc_after = self._accuracy(X_val, y_val)

        if acc_after >= acc_before:
            # Keep pruning (remains leaf), remove children
            node.left = None
            node.right = None
            return node
        else:
            # Pruning didn't help, revert to internal node
            node.is_leaf = False
            return node

    def prune(self, X_val, y_val):
        """
        Prunes the tree using a validation set.
        Must be called after fit().
        """
        if self.root is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")
        X_val = self._to_numpy_2d(X_val)
        y_val = self._to_numpy_1d(y_val)
        self.root = self._prune_node(self.root, X_val, y_val)
        return self

    # ------------------------------------------------------------------
    # Debugging/Utility tools
    # ------------------------------------------------------------------
    def get_depth(self, node=None):
        if node is None:
            node = self.root
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self.get_depth(node.left), self.get_depth(node.right))

    def count_nodes(self, node=None):
        if node is None:
            node = self.root
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return 1 + self.count_nodes(node.left) + self.count_nodes(node.right)

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root
        indent = "  " * depth
        if node.is_leaf:
            print(f"{indent}Leaf -> class={node.prediction} (n={node.n_samples})")
            return
        fname = (
            self.feature_names[node.feature_index]
            if self.feature_names is not None
            else f"feature_{node.feature_index}"
        )
        print(f"{indent}[{fname} <= {node.threshold:.4f}] (n={node.n_samples})")
        self.print_tree(node.left, depth + 1)
        self.print_tree(node.right, depth + 1)
class MinMaxNormalizer:
    """
    Scales features to the range [0, 1] using Min-Max normalization.

    Formula:
        X_normalized = (X - X_min) / (X_max - X_min)
    """

    def __init__(self):
        self.min_values = None
        self.max_values = None

    def fit(self, X):
        """
        Computes and stores the minimum and maximum value of each feature
        from the input dataset.
        """
        self.min_values = X.min()
        self.max_values = X.max()

    def transform(self, X):
        """
        Applies Min-Max normalization using the previously stored
        minimum and maximum feature values.
        """
        if self.min_values is None or self.max_values is None:
            raise ValueError(
                "Normalizer has not been fitted yet. Call fit() first."
            )

        normalized = (X - self.min_values) / (
            self.max_values - self.min_values
        )

        return normalized

    def fit_transform(self, X):
        """
        Fits the normalizer on the input data and immediately
        returns the normalized result.
        """
        self.fit(X)
        return self.transform(X)
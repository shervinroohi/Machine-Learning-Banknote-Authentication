import numpy as np


class DataSplitter:
    """
    Splits the dataset into training, validation, and test sets using
    a 70/15/15 ratio.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def split(self, X, y):
        """
        Shuffles the data and splits it into:
            - 70% training
            - 15% validation
            - 15% test
        """
        np.random.seed(self.random_state)

        indices = np.random.permutation(len(X))

        X = X.iloc[indices].reset_index(drop=True)
        y = y.iloc[indices].reset_index(drop=True)

        total_samples = len(X)

        train_size = int(total_samples * 0.70)
        validation_size = int(total_samples * 0.15)

        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]

        X_validation = X.iloc[
            train_size:
            train_size + validation_size
        ]
        y_validation = y.iloc[
            train_size:
            train_size + validation_size
        ]

        X_test = X.iloc[
            train_size + validation_size:
        ]
        y_test = y.iloc[
            train_size + validation_size:
        ]

        return (
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test
        )
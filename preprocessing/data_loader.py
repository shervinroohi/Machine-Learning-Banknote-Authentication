import pandas as pd


class DataLoader:
    """
    Loads the dataset from a CSV file and splits it into features (X) and labels (y).
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self):
        """
        Reads the CSV file, assigns column names, and returns:
            X: feature matrix
            y: target labels
        """
        column_names = [
            "Variance",
            "Skewness",
            "Kurtosis",
            "Entropy",
            "Class"
        ]

        dataframe = pd.read_csv(
            self.file_path,
            header=None,
            names=column_names
        )

        X = dataframe[
            [
                "Variance",
                "Skewness",
                "Kurtosis",
                "Entropy"
            ]
        ]

        y = dataframe["Class"]

        return X, y
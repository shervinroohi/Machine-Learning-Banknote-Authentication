import numpy as np


class KNN:
    """
    k-Nearest Neighbors (KNN) classifier.

    This implementation uses Euclidean distance and majority voting
    to classify each input sample.
    """

    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Stores the training data for later distance-based prediction.

        The index is reset to keep row access consistent during prediction.
        """
        self.X_train = X_train.reset_index(drop=True)
        self.y_train = y_train.reset_index(drop=True)

    def euclidean_distance(self, sample1, sample2):
        """
        Computes the Euclidean distance between two samples.
        """
        sample1 = np.array(sample1)
        sample2 = np.array(sample2)

        return np.sqrt(np.sum((sample1 - sample2) ** 2))

    def get_neighbors(self, sample):
        """
        Finds the indices of the k nearest training samples to the given input sample.
        """
        distances = []

        for index in range(len(self.X_train)):
            train_sample = self.X_train.iloc[index]

            distance = self.euclidean_distance(sample, train_sample)
            distances.append((index, distance))

        distances.sort(key=lambda item: item[1])

        neighbors = distances[:self.k]

        return [index for index, _ in neighbors]

    def vote(self, neighbor_indices):
        """
        Predicts the class label by majority vote among the nearest neighbors.
        """
        votes = self.y_train.iloc[neighbor_indices]
        vote_counts = votes.value_counts()

        return vote_counts.idxmax()

    def predict(self, sample):
        """
        Predicts the class label for a single sample.
        """
        neighbors = self.get_neighbors(sample)
        prediction = self.vote(neighbors)

        return prediction

    def predict_batch(self, X):
        """
        Predicts class labels for a batch of samples.
        """
        predictions = []

        for i in range(len(X)):
            prediction = self.predict(X.iloc[i])
            predictions.append(prediction)

        return predictions
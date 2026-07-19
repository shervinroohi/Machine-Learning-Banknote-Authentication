import numpy as np


class NaiveBayes:
    """
    Gaussian Naive Bayes classifier.

    This implementation assumes that each feature follows a Gaussian
    (normal) distribution within each class. For each class, the model
    stores the mean, variance, and prior probability, then uses them
    to estimate the posterior probability of a sample.
    """

    def __init__(self):
        # Mean of each feature for each class
        self.means = {}

        # Variance of each feature for each class
        self.variances = {}

        # Prior probability of each class
        self.class_priors = {}

        # Unique class labels found in the training set
        self.classes = None

        # Small constant to avoid division by zero
        self.epsilon = 1e-9

    def fit(self, X_train, y_train):
        """
        Computes class-wise statistics from the training data:
        mean, variance, and prior probability for each class.
        """
        self.classes = np.unique(y_train)
        total_samples = len(y_train)

        for current_class in self.classes:
            class_samples = X_train[y_train == current_class]

            self.means[current_class] = class_samples.mean()
            self.variances[current_class] = class_samples.var() + self.epsilon
            self.class_priors[current_class] = len(class_samples) / total_samples

    def gaussian_probability(self, x, mean, variance):
        """
        Computes the Gaussian probability density value for a feature.
        """
        exponent = np.exp(-((x - mean) ** 2) / (2 * variance))
        probability = (1 / np.sqrt(2 * np.pi * variance)) * exponent
        return probability

    def calculate_probability(self, sample, current_class):
        """
        Computes the log-posterior probability of the sample belonging
        to a given class.
        """
        # Start with the log prior probability of the class
        probability = np.log(self.class_priors[current_class])

        for feature in sample.index:
            mean = self.means[current_class][feature]
            variance = self.variances[current_class][feature]

            gaussian = self.gaussian_probability(
                sample[feature],
                mean,
                variance
            )

            probability += np.log(gaussian)

        return probability

    def predict_proba(self, sample):
        """
        Returns the log-posterior probability for each class
        for a single input sample.
        """
        probabilities = {}

        for current_class in self.classes:
            probabilities[current_class] = self.calculate_probability(
                sample,
                current_class
            )

        return probabilities

    def predict(self, sample):
        """
        Predicts the class label for a single sample by selecting
        the class with the highest posterior probability.
        """
        probabilities = self.predict_proba(sample)

        prediction = max(probabilities, key=probabilities.get)

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
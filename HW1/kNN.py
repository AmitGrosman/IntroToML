# Q7
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import cdist
import numpy as np


class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors: int = 3):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        # Fit the model to vectors in X, with their respected label in y
        # Params: X - dataset of feature vectors
        #         y - corresponding label for each vector in X
        # Returns: The fitted model

        self.X = np.copy(X)
        self.y = np.copy(y)
        return self

    def predict(self, X):
        # Predict the label of vectors in X using the fitted model
        # Params: X - unlabeled vectors
        # Returns: predictions vector for X (using the fitted model)
 
        predictions = None
        distances_matrix = cdist(X, self.X)
        closest_k_neighbours = np.argpartition(distances_matrix, self.n_neighbors, axis=1)[:, :self.n_neighbors]
        neighbors_tags = self.y[closest_k_neighbours]
        predictions = np.sign(np.sum(neighbors_tags, axis=1))
        return predictions

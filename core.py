import numpy as np


class KNN:
    def __init__(self, hy_param):
        self.k = hy_param[0]
       # self.encoder = hy_param[1]
        self.distance_matrix = hy_param[2]
        

        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        if self.distance_matrix == 'Euclidean':
            distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        elif self.distance_matrix == 'Manhattan':
            distances = [np.sum(np.abs(x - x_train)) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]   #index of distances in deccending order of distances
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        label_counts = {}
        for label in k_nearest_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        most_common_label = max(label_counts, key=label_counts.get)

        return most_common_label



class KNN_Opt:
    def __init__(self, hy_param):
        self.k = hy_param[0]
       # self.encoder = hy_param[1]
        self.distance_matrix = hy_param[2]
        

        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        if self.distance_matrix == 'Euclidean':
            squared_diffs = (self.X_train - x)**2

            sum_squared_diffs = np.sum(squared_diffs, axis=1)

            distances = np.sqrt(sum_squared_diffs)

        elif self.distance_matrix == 'Manhattan':
            distances = [np.sum(np.abs(x - x_train)) for x_train in self.X_train]

            squared_diffs = np.abs(self.X_train - x)

            sum_squared_diffs = np.sum(squared_diffs, axis=1)

            distances = np.sqrt(sum_squared_diffs)

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        label_counts = {}
        for label in k_nearest_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        most_common_label = max(label_counts, key=label_counts.get)

        return most_common_label
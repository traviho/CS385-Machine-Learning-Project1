import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

class FisherModel:
    # u1 
    # u2 
    # cov1 
    # cov2

    # given Y of values 1 or 0, split X data into two matrices, one of positives and one of negatives
    def split_categories(self, X, Y):
        self.m = Y.shape[0]
        self.F = X.shape[1]
        p_list = []
        n_list = []
        for i in range(self.m):
            if Y[i] == 1:
                p_list.append(X[i])
            else:
                n_list.append(X[i])
        p_matrix = np.array(p_list)
        n_matrix = np.array(n_list)
        self.p_size = p_matrix.shape[0]
        self.n_size = n_matrix.shape[0]
        assert(self.p_size + self.n_size == self.m)
        return p_matrix, n_matrix
    
    # X[m, F], calc average[1, F] across samples for each feature
    def average_features(self, X):
        size = X.shape[0]
        avg_list = np.sum(X, axis=0) / size
        return np.reshape(avg_list, (1, self.F))
    
    def variance(self, X, size):
        return np.dot(X.T, X) / size
    
    # TODO: Not evaluating to the correct values
    def calc_discriminant(self, x_example, cov, u, size, m):
        cov_inv = np.linalg.inv(cov)
        prior_prob = float(size) / m
        return np.dot(np.dot(u, cov_inv), x_example.T) - ((1/2) * np.dot(np.dot(u, cov_inv), u.T)) + np.log(prior_prob)

    def bound(self, f1, f2):
        return 1 if f1 > f2 else 0

    # uses parameters learned to run discriminant function and check which is greater.
    def predict(self, X):
        predictions = []
        for x_example in X:
            f1 = self.calc_discriminant(x_example, self.C_inv, self.u1, self.p_size, self.m)
            f2 = self.calc_discriminant(x_example, self.C_inv, self.u2, self.n_size, self.m)
            predictions.append(self.bound(f1, f2))
        return predictions

    def pooled_covariance(self, cov1, cov2, size1, size2, m):
        pooled_cov = np.zeros(cov1.shape)
        for r in range(cov1.shape[0]):
            for c in range(cov1.shape[1]):
                pooled_cov[r][c] = (float(size1) / m) * cov1[r][c] + (float(size2) / m) * cov2[r][c]
        return pooled_cov
    
    # calculates parameters
    # p[p, F], n[n, F], X[m, F], Y[F, 1]
    def fit(self, X, Y):
        p_matrix, n_matrix = self.split_categories(X, Y)
        u1 = self.average_features(p_matrix)
        u2 = self.average_features(n_matrix)
        global_u = self.average_features(X)

        p_matrix0 = p_matrix - global_u
        n_matrix0 = n_matrix - global_u
        
        cov1 = self.variance(p_matrix0, self.p_size)
        cov2 = self.variance(n_matrix0, self.n_size)
        pooled_cov = self.pooled_covariance(cov1, cov2, self.p_size, self.n_size, self.m)
        pooled_cov_inv = np.linalg.inv(pooled_cov)
        
        self.u1 = u1
        self.u2 = u2
        self.C_inv = pooled_cov_inv
    
    def score(self, X, Y):
        predictions = self.predict(X)
        Y = Y.flatten()
        diff = Y - predictions

        total_negative = len(list(filter(lambda x: x == 0, Y)))
        total_positive = len(list(filter(lambda x: x == 1, Y)))

        incorrect_positive = len(list(filter(lambda x: x == 1, diff))) # Y was 1, predict was 0
        incorrect_negative = len(list(filter(lambda x: x == -1, diff))) # Y was 0, predict was 1

        accuracy = 1.0 - (float(np.count_nonzero(diff)) / len(diff))
        sensitivity = 1.0 - (float(incorrect_positive) / total_positive)
        specificity = 1.0 - (float(incorrect_negative) / total_negative)

        return accuracy, sensitivity, specificity

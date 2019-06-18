import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from torch.distributions import Normal

class LogisticModel:
    # w = None # [F, 1] feature vector
    # m = None # number of samples
    # F = None # number of features

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # w = [F, 1], X = [m, F] -> A = [m, 1]
    def activation(self, w, X):
        return self.sigmoid(np.dot(X, w))
    
    # return [F, 1] feature vector of gradient noise
    def get_langevin_noise(self, F, learning_rate):
        langevin_noise = Normal(
            torch.zeros(F),
            torch.ones(F) * np.sqrt(learning_rate)
        )
        noise_sample = langevin_noise.sample().numpy()
        noise_sample = np.reshape(noise_sample, (F, 1))
        return noise_sample

    # A - Y = [m, 1], X [m, F] -> dw = [F, 1]
    # each value in dw is sum of (a-y * x)/m
    def gradient(self, A, X, Y, m, langevin_noise=False, learning_rate=0.1):
        dw = (1/m) * np.dot(X.T, A - Y)
        if langevin_noise:
            dw += self.get_langevin_noise(self.F, learning_rate)
        return dw

    # Y[m, 1] * A[m, 1] * scalar -> squeeze [1,1] -> scalar
    def calc_cost(self, A, Y, m):
        cost = (-1/m) * (np.dot(np.log(A).T, Y) + np.dot(np.log(1 - A).T, 1 - A))
        return np.squeeze(cost)

    # w = [F, 1], dw = [F, 1] -> new w [F, 1]
    def descent_w(self, w, dw, learning_rate):
        return w - (learning_rate * dw)

    # for each epoch, update w and print cost
    def batch_gradient(self, X, Y, m, w, epochs=100, learning_rate=0.5):
        costs = []
        for _ in range(epochs):
            A = self.activation(w, X)
            dw = self.gradient(A, X, Y, m)
            cost = self.calc_cost(A, Y, m)
            costs.append(cost)
            w = self.descent_w(w, dw, learning_rate)
        return w, costs
    
    def unison_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    # for each epoch, loop through each example, calculating gradient w.r.t that example
    # x_example[1, F], y_example[1, 1]
    def stochastic_gradient(self, X, Y, w, langevin_noise=False, epochs=100, learning_rate=0.5):
        costs = []
        for _ in range(epochs):
            X_shuffle, Y_shuffle = self.unison_shuffled_copies(X, Y)
            for j in range(X_shuffle.shape[0]):
                x_example = X_shuffle[j]
                y_example = Y_shuffle[j]
                x_example = np.reshape(x_example, (1, self.F))
                y_example = np.reshape(y_example, (1, 1))
                a = self.activation(w, x_example)
                dw = self.gradient(a, x_example, y_example, 1, langevin_noise, learning_rate)
                cost = self.calc_cost(a, y_example, 1)
                w = self.descent_w(w, dw, learning_rate)
            costs.append(cost)
        return w, costs

    def plot_cost(self, costs):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x_points = costs
        ax.plot(x_points, 'b')
        ax.set_xlabel('iterations')
        ax.set_ylabel('cost')
        ax.set_title('cost vs. iterations')
        plt.show()

    # X[m, F], Y[m, 1] -> w[F, 1]
    def fit(self, X, Y, optimizer='batch_gradient', plot_costs=True):
        self.w = np.zeros((X.shape[1], 1))
        self.m = X.shape[0]
        self.F = X.shape[1]
        if optimizer == 'stochastic_gradient':
            self.w, costs = self.stochastic_gradient(X, Y, self.w)
        elif optimizer == 'stochastic_gradient_langevin_dynamics':
            self.w, costs = self.stochastic_gradient(X, Y, self.w, langevin_noise=True)
        else:
            self.w, costs = self.batch_gradient(X, Y, self.m, self.w)
        if plot_costs:
            self.plot_cost(costs)

    def decision_boundary(self, probability):
        return 1 if probability >= .5 else -1
    
    # outputs array [1, m]
    def predictions(self, X):
        A = self.activation(self.w, X)
        # print(A)
        decision_boundary = np.vectorize(self.decision_boundary)
        return decision_boundary(A).flatten()

    # Sensitivity is true positive, specificity is true negative.
    def score(self, X, Y):
        predictions = self.predictions(X)
        Y = Y.flatten()
        diff = Y - predictions

        total_negative = len(list(filter(lambda x: x == 0, Y)))
        total_positive = len(list(filter(lambda x: x == 1, Y)))

        incorrect_positive = len(list(filter(lambda x: x == 1, diff))) # Y was 1, predict was 0
        incorrect_negative = len(list(filter(lambda x: x == -1, diff))) # Y was 0, predict was 1

        accuracy = 1.0 - (float(np.count_nonzero(diff)) / len(diff))
        sensitivity = 1.0 - (float(incorrect_positive) / total_positive)
        specificity = 1.0 - (float(incorrect_negative) / total_negative)

        return "accuracy: %f positives: %f negatives: %f" % (accuracy, sensitivity, specificity)

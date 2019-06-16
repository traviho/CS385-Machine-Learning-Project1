import cv2
import numpy as np
import math
from create_feature_data import load_data, save_dataset
from visualize import visualize_HOG_data, visualize_bounding_box_data
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from logistic_model import LogisticModel
from SVM_model import SVM_Model
from fisher_model import FisherModel
from neural_net import NeuralNet

# if the dataset does not exist, run build_data first
def build_data():
    print("Building feature vector data into data.json ...")
    save_dataset(filename="data_full.json")

# data_full contains all training and test data. 
# data_min is a minified version for testing.
X_train, Y_train, X_test, Y_test = load_data("data_full.json")

def run_sci_kit_logistic():
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)
    print(clf.score(X_test, Y_test))

# optimizers [stochastic_gradient_langevin_dynamics, stochastic_gradient, batch_gradient]
def run_logistic(optimizer='batch_gradient', plot_costs=False):
    l = LogisticModel()
    l.fit(X_train, Y_train, optimizer='stochastic_gradient_langevin_dynamics', plot_costs=plot_costs)
    print(l.score(X_test, Y_test))

# kernels [linear, rbf, poly]
def run_svm(kernel='linear'):
    clf = SVM_Model(kernel=kernel)
    clf.fit(X_train, Y_train)
    print(clf.score(X_test, Y_test))

def run_sci_kit_lda():
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, Y_train)
    print(clf.score(X_test, Y_test))

def run_lda():
    clf = FisherModel()
    clf.fit(X_train, Y_train)
    print(clf.score(X_test, Y_test))

def run_NN():
    NN = NeuralNet(num_features=X_train.shape[1])
    NN.fit(X_train, Y_train)
    print(NN.score(X_test,Y_test))

# Visualization
# visualize_bounding_box_data()
# visualize_HOG_data()

# Tests
# run_sci_kit_logistic()
# run_logistic(optimizer='batch_gradient')
# run_logistic(optimizer='stochastic_gradient')
# run_logistic(optimizer='stochastic_gradient_langevin_dynamics')
# run_sci_kit_lda()
# run_lda()
# run_svm(kernel='linear')
# run_svm(kernel='rbf')
# run_svm(kernel='poly')
# run_NN()

import cv2
import numpy as np
import math
from create_feature_data import load_data, save_dataset
from sklearn.svm import LinearSVC
from sklearn import svm

class SVM_Model:
    # clf
    def __init__(self, kernel='linear'):
        if kernel == 'linear':
            self.clf = LinearSVC()
        else:
            self.clf = svm.SVC(kernel=kernel)
    
    def fit(self, X, Y):
        self.clf.fit(X, Y)
    
    def score(self, X, Y):
        return self.clf.score(X, Y)


# Linear SVM
# def linear_svm():
#     clf = LinearSVC()
#     clf.fit(X_train, Y_train)
#     return clf

# # RBF SVM
# def rbf_svm():
#     clf = svm.SVC(kernel='rbf')
#     clf.fit(X_train, Y_train)
#     return clf

# # Other SVM kernel: Poly
# def poly_svm():
#     clf = svm.SVC(kernel='poly')
#     clf.fit(X_train, Y_train)
#     return clf

# Tests
# linear_svm = linear_svm()
# linear_svm_score = linear_svm.score(X_test, Y_test)
# print("Linear accuracy: " + str(linear_svm_score))

# rbf_svm = rbf_svm()
# rbf_svm_score = rbf_svm.score(X_test, Y_test)
# print("RBF accuracy: " + str(rbf_svm_score))

# poly_svm = poly_svm()
# poly_svm_score = poly_svm.score(X_test, Y_test)
# print("Polynomial accuracy: " + str(poly_svm_score))

# RESULTS_FILE = "svm_results.txt"
# Dump results
# with open(RESULTS_FILE, 'w') as f:
#     f.write("Linear accuracy: " + str(linear_svm_score) + "\n")
#     f.write("Coefficients: " + str(linear_svm.coef_) + "\n\n")
#     f.write("RBF accuracy: " + str(rbf_svm_score) + "\n")
#     f.write("Coefficients: " + str(rbf_svm.support_vectors_) + "\n\n")
#     f.write("Polynomial accuracy: " + str(poly_svm_score) + "\n")
#     f.write("Coefficients: " + str(poly_svm.support_vectors_) + "\n")

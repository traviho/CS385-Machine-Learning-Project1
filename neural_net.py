from keras.models import Sequential
from keras.layers import Dense
import numpy as np

class NeuralNet:
    # model

    def __init__(self, num_features=900):
        model = Sequential()
        model.add(Dense(12, input_dim=num_features, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model

    def fit(self, X, Y, epochs=25, batch_size=10):
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size)
    
    def score(self, X, Y):
        scores = self.model.evaluate(X, Y)
        return "\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100)

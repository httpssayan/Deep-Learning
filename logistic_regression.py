import numpy as np 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(x, w, b):
    z = np.dot(x, w) + b
    pred = sigmoid(z)
    return 1 if pred >= 0.5 else 0

def binary_cross_entropy(y, y_hat):
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def accuracy(y, pred):
    return np.mean(y == pred)


import numpy as np

class LogisticRegression:


    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
                
    def fit(self, X, y):

        m, n = X.shape

        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.epochs):

            z = np.dot(X, self.weights) + self.bias

            y_hat = sigmoid(z)

            dw = (1 / m) * np.dot(X.T, (y_hat - y))

            db = (1 / m) * np.sum(y_hat - y)

            self.weights -= self.learning_rate * dw

            self.bias -= self.learning_rate * db

    def predict_proba(self, X):

        z = np.dot(X, self.weights) + self.bias

        return sigmoid(z)

    def predict(self, X):

        probabilities = self.predict_proba(X)

        return (probabilities >= 0.5).astype(int)

    def accuracy(self, X, y):

        predictions = self.predict(X)

        return np.mean(predictions == y)


X = np.array([
    [1, 1],
    [2, 1],
    [3, 2],
    [4, 3],
    [5, 4],
    [6, 5]
])

y = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression(learning_rate=0.1, epochs=1000)

model.fit(X, y)

print("Predictions:", model.predict(X))
print("Probabilities:", model.predict_proba(X))
print("Accuracy:", model.accuracy(X, y))
import numpy as np 

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def step_function(self, x):
            return 1 if x >= 0 else 0

    def predict_one(self,x):
            z=np.dot(x, self.weights) + self.bias
            return self.step_function(z)

    def fit(self,X,y):
            self.weights = np.zeros(X.shape[1])

            for _ in range(self.epochs):
                for x_i,y_i in zip(X,y):
                        prediction = self.predict_one(x_i)

                        error = y_i - prediction

                        self.weights += self.learning_rate * error * x_i
                        self.bias += self.learning_rate * error

    def predict(self,X):
        return np.array([self.predict_one(x_i) for x_i in X])

    def accuracy(self,X,y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 1, 1, 0])

model = Perceptron(learning_rate=0.1, epochs=10)

model.fit(X, y)

print("Predictions:", model.predict(X))
print("Accuracy:", model.accuracy(X, y))
import numpy as np 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagation(X,W1,b1,W2,b2):
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    return A2

X = np.array([
    [1],
    [2],
    [3]
])

W1 = np.array([
    [0.5, 0.2, 0.1],
    [0.4, 0.3, 0.2]
])

b1 = np.array([
    [0.1],
    [0.1]
])

W2 = np.array([
    [0.6, 0.5]
])

b2 = np.array([
    [0.1]
])

output = forward_propagation(X, W1, b1, W2, b2)

print("Output:", output)
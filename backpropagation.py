import numpy as np 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagation(X,W1,b1,W2,b2):
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache=(X, Z1, A1, Z2, A2)
    return A2,cache

def backward_propagation(y, cache, W2):
    X, Z1, A1, Z2, A2 = cache
    dZ2 = A2 - y
    dW2 = np.dot(dZ2, A1.T) 
    db2=dZ2

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * A1 * (1 - A1)

    dW1 = np.dot(dZ1, X.T)
    db1 = dZ1

    return dW1, db1, dW2, db2

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

y = np.array([
    [1]
])

A2, cache = forward_propagation(X, W1, b1, W2, b2)

print("Prediction:", A2)

dW1, db1, dW2, db2 = backward_propagation(
    y, cache, W2
)

print("\ndW1:")
print(dW1)

print("\ndb1:")
print(db1)

print("\ndW2:")
print(dW2)

print("\ndb2:")
print(db2)

def compute_loss(y, y_hat):
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

epsilon = 1e-5

# Save original value
original_value = W2[0, 0]

# J(W + epsilon)
W2[0, 0] = original_value + epsilon
A2_plus, _ = forward_propagation(X, W1, b1, W2, b2)
loss_plus = compute_loss(y, A2_plus)

# J(W - epsilon)
W2[0, 0] = original_value - epsilon
A2_minus, _ = forward_propagation(X, W1, b1, W2, b2)
loss_minus = compute_loss(y, A2_minus)

# Restore original value
W2[0, 0] = original_value

# Numerical gradient
numerical_gradient = (
    loss_plus - loss_minus
) / (2 * epsilon)

print("Numerical gradient:", numerical_gradient)
print("Analytical gradient:", dW2[0, 0])
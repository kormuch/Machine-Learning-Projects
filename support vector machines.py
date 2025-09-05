import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Small dataset
X = np.array([
    [1, 3],   # Positive class
    [2, 5],
    [2, 2],   # Negative class
    [3, 2]
])
y = np.array([1, 1, -1, -1])

n_samples, n_features = X.shape

# Kernel matrix (linear kernel, just dot products)
K = np.dot(X, X.T)

# Objective function to minimize (negative dual problem)
def objective(alpha):
    return 0.5 * np.sum(alpha[i] * alpha[j] * y[i] * y[j] * K[i, j] 
                        for i in range(n_samples) for j in range(n_samples)) - np.sum(alpha)

# Equality constraint (sum of alpha_i * y_i = 0)
def constraint_eq(alpha):
    return np.dot(alpha, y)

# Bounds for alpha (0 <= alpha_i <= C)
bounds = [(0, None) for _ in range(n_samples)]

# Initial guess for alpha
initial_alpha = np.zeros(n_samples)

# Define the constraint
constraints = {'type': 'eq', 'fun': constraint_eq}

# Minimize the dual problem
result = minimize(objective, initial_alpha, bounds=bounds, constraints=constraints)

# Extract the optimized alphas
alpha = result.x

# Calculate weight vector w
w = np.sum(alpha[i] * y[i] * X[i] for i in range(n_samples))

# Calculate the bias term b
support_vectors_idx = np.where(alpha > 1e-4)[0]  # Threshold to avoid numerical issues
b = np.mean([y[i] - np.dot(w, X[i]) for i in support_vectors_idx])

# Prediction function
def predict(X):
    return np.sign(np.dot(X, w) + b)

# Plotting
plt.figure(figsize=(8, 6))

# Decision boundary
x_vals = np.linspace(0, 4, 100)
y_vals = -(w[0] * x_vals + b) / w[1]
plt.plot(x_vals, y_vals, 'k-', label='Decision Boundary')

# Plotting the dataset
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], s=100, color='b', label='Positive Class')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], s=100, color='r', label='Negative Class')

# Support vectors (where alpha > threshold)
plt.scatter(X[support_vectors_idx][:, 0], X[support_vectors_idx][:, 1], s=200, color='g', edgecolors='k', label='Support Vectors')

plt.xlim(0, 10)
plt.ylim(0, 10)
plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM Decision Boundary with Calculated Alphas and Support Vectors")
plt.show()

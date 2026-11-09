import numpy as np
import matplotlib.pyplot as plt


# ---------------- RIDGE REGRESSION ----------------

def ridge_regression_gd(X, y, learning_rate, n_iter, lam):
    n = len(y)

    w = 0.0
    b = 0.0

    cost_history = []

    for i in range(n_iter):

        # Prediction
        y_pred = w * X + b

        # Cost function
        cost = (1 / n) * np.sum((y_pred - y) ** 2) + (lam / 2) * (w ** 2)

        cost_history.append(cost)

        # Gradients
        dw = (2 / n) * np.sum((y_pred - y) * X) + lam * w
        db = (2 / n) * np.sum(y_pred - y)

        # Update weights
        w = w - learning_rate * dw
        b = b - learning_rate * db

    return w, b, cost_history


# ---------------- INPUT DATA ----------------

x_input = input("Enter the X feature: ")
y_input = input("Enter the Y feature: ")

X = np.array([float(v) for v in x_input.split(",")])
y = np.array([float(v) for v in y_input.split(",")])


# ---------------- PARAMETERS ----------------

learning_rate = 0.01
n_iter = 25
lam = 0.2


# ---------------- TRAIN MODEL ----------------

w, b, cost_history = ridge_regression_gd(
    X, y, learning_rate, n_iter, lam
)


# ---------------- PRINT RESULTS ----------------

print(f"Weight (w): {w:.4f}")
print(f"Bias (b): {b:.4f}")


# ---------------- COST HISTORY GRAPH ----------------

plt.figure(1)

plt.plot(range(1, n_iter + 1), cost_history)

plt.xlabel("Number of Iterations")
plt.ylabel("Cost J(w,b)")
plt.title("Ridge Regression Cost History During Gradient Descent")

plt.grid(True)
plt.show()


# ---------------- REGRESSION LINE GRAPH ----------------

sort_idx = np.argsort(X)
X_sorted = X[sort_idx]

y_sorted = w * X_sorted + b

plt.figure(2)

plt.scatter(X, y, label="Original Data")
plt.plot(
    X_sorted,
    y_sorted,
    label="Ridge Regression Line"
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Ridge Regression with Gradient Descent")

plt.legend()
plt.grid(True)
plt.show()
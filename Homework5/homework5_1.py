import numpy as np
import matplotlib.pyplot as plt

# Gradient function
def gradient(w):
    w1, w2 = w
    grad_w1 = 26 * w1 - 10 * w2 + 4
    grad_w2 = -10 * w1 + 4 * w2 - 2
    return np.array([grad_w1, grad_w2])


# Gradient descent
def gradient_descent(eta, iterations):
    w = np.array([0.0, 0.0])
    w_star = np.array([1.0, 3.0])   # Optimal weights
    
    distances = []
    for _ in range(iterations):
        w -= eta * gradient(w)
        distance = np.linalg.norm(w - w_star)
        distances.append(distance)

    return distances


def main():
    eta_values = [0.02, 0.05, 0.1]
    iterations = 500

    for eta in eta_values:
        result = gradient_descent(eta, iterations)

        # Plotting obtained distances
        plt.figure(figsize=(10, 6))
        plt.plot(result)
        plt.title(f"Distances from the optimal solution vs Iterations - eta = {eta}")
        plt.xlabel("Iteration")
        plt.ylabel("Distance from the optimal solution")
        plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
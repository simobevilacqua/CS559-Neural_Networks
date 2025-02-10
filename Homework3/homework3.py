import random
import numpy as np
import matplotlib.pyplot as plt

maxIterations = 100     # Upper bound on the number of iterations

# Step activation function
def step(v):
    result = np.array([])
    for i in v:
        if i >= 0:
            result = np.append(result, 1)
        else:
            result = np.append(result, 0)
    return result

# Perceptron Learning Algorithm
def perceptronAlg(w_init, x, y, eta):
    totErrors = []

    for n in range(maxIterations):
        errors = 0
        for i in range(len(y)):
            y_pred = 1 if (np.dot(w_init, x[i]) >= 0.0) else 0
            update = eta * (y[i] - y_pred)
            w_init += update * x[i].T
            errors += int(update != 0.0)
        totErrors.append(errors)

        if errors == 0:
            break

    return w_init, totErrors


# Exercise 1
n = 100         # Sample size
w_star = np.array([[random.uniform(-1/4, 1/4)], [random.uniform(-1, 1)], [random.uniform(-1, 1)]])
print('w* = ', w_star)
x = []
for i in range(n):
    x.append([1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
x = np.array(x)
y = step(np.dot(x, w_star))



# Exercise 2
eta = [0.1, 1, 10]

# Applying PLA for different eta values
totErrors = {}
w = {}
for e in eta:
    w[e], totErrors[e] = perceptronAlg(np.ones(x.shape[1]), x, y, e)


# Larger sample
n = 1000        # Sample size
x_new = []
for i in range(n):
    x_new.append([1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
x_new = np.array(x_new)
y_new = step(np.dot(x_new, w_star))
# Applying PLA
w_new, totErrors_new = perceptronAlg(np.ones(x_new.shape[1]), x_new, y_new, 1)


# Many iterations
n = 100         # Sample size
r = 100         # Iterations
x_rep = []
for i in range(n):
    x_rep.append([1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
x_rep = np.array(x_rep)
y_rep = step(np.dot(x_rep, w_star))

# Applying PLA for different eta values
totErrors_rep = {}
for e in eta:
    totErrors_rep[e] = []
for i in range(r):
    weight = np.array([[random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]])
    for e in eta:
        _, errors = perceptronAlg(weight.copy(), x_rep, y_rep, e)
        # Add tailing zeros to correctly compute the percentile values
        totErrors_rep[e].append(np.pad(errors, (0, n - len(errors)), mode='constant'))

# Compute the average number of errors across different iterations
max_length = {}
errors = {}
percentile_10 = {}
percentile_90 = {}
for e in eta:
    max_length[e] = max(len(subarray) for subarray in totErrors_rep[e])
    sums = np.zeros(max_length[e])
    counts = np.zeros(max_length[e])
    for subarray in totErrors_rep[e]:
        for i in range(len(subarray)):
            sums[i] += subarray[i]
            counts[i] += 1
    averages = np.zeros(max_length[e])
    for i in range(max_length[e]):
        if counts[i] > 0:
            averages[i] = sums[i] / counts[i]

    errors[e] = averages[:max(len(subarray) for subarray in totErrors_rep[e])]
    percentile_10[e] = np.percentile(totErrors_rep[e], 10, axis=0)
    percentile_90[e] = np.percentile(totErrors_rep[e], 90, axis=0)


# Plots

# Small sample plot
plt.figure(figsize=(8, 8))
colors = np.where(y == 0, 'blue', 'red')
plt.scatter(x[:, 1], x[:, 2], c=colors, alpha=0.6, edgecolors='w', s=50)

# Boundary
boundary = -(w_star[0] + w_star[1] * np.linspace(-1, 1)) / w_star[2]
x_range = np.linspace(-1, 1)[(boundary >= -1) & (boundary <= 1)]
boundary_bounded = boundary[(boundary >= -1) & (boundary <= 1)]
plt.plot(x_range, boundary_bounded, color='black', label='Decision boundary')

# Normal vector
x2_value = -(w_star[0] / w_star[2])
plt.quiver(0, x2_value, w_star[1], w_star[2], angles='xy', scale_units='xy', scale=1, color='green', label='Normal vector', width=0.005, headlength=4, headwidth=3, headaxislength=3)

# Predited boundary
predBoundary = -(w[1][0] + w[1][1] * np.linspace(-1, 1)) / w[1][2]
predX_range = np.linspace(-1, 1)[(predBoundary >= -1) & (predBoundary <= 1)]
predBoundary_bounded = predBoundary[(predBoundary >= -1) & (predBoundary <= 1)]
plt.plot(predX_range, predBoundary_bounded, color='red', label='Predicted decision boundary')

plt.title('Scatter Plot')
plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('equal')
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')

# Plot errors-epoch data
for e in eta:
    plt.figure(figsize=(10, 6))
    plt.plot(totErrors[e])
    plt.title(f'Eta = {e}')
    plt.xlabel('Epoch')
    plt.ylabel('Errors')
    plt.grid(True)


# Larger sample plot
plt.figure(figsize=(8, 8))
colors_new = np.where(y_new == 0, 'blue', 'red')
plt.scatter(x_new[:, 1], x_new[:, 2], c=colors_new, alpha=0.6, edgecolors='w', s=50)

# Boundary
boundary = -(w_star[0] + w_star[1] * np.linspace(-1, 1)) / w_star[2]
x_range = np.linspace(-1, 1)[(boundary >= -1) & (boundary <= 1)]
boundary_bounded = boundary[(boundary >= -1) & (boundary <= 1)]
plt.plot(x_range, boundary_bounded, color='black', label='Decision boundary')

# Predited boundary
predBoundary_new = -(w_new[0] + w_new[1] * np.linspace(-1, 1)) / w_new[2]
predX_range_new = np.linspace(-1, 1)[(predBoundary_new >= -1) & (predBoundary_new <= 1)]
predBoundary_bounded_new = predBoundary_new[(predBoundary_new >= -1) & (predBoundary_new <= 1)]
plt.plot(predX_range_new, predBoundary_bounded_new, color='green', label='Predicted decision boundary')
plt.title('Scatter Plot')
plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('equal')
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')


# Many iterations of PLA plot
for e in eta:
    plt.figure(figsize=(10, 6))
    plt.plot(errors[e])
    plt.fill_between(range(max_length[e]), percentile_10[e], percentile_90[e], color='green', alpha=0.2, label='10th to 90th Percentile Range')
    plt.title(f'Eta = {e}')
    plt.xlabel('Epoch')
    plt.ylabel('Errors')
    plt.grid(True)
    plt.tight_layout()

plt.show()

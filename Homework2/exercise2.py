import matplotlib.pyplot as plt
import numpy as np

def equation(x1, x2):
    x = np.array([[x1], [x2]])
    return step(np.dot(U, step(np.dot(W, x) + b)) + c)

def step(v):
    result = np.array([])
    for i in v:
        if i >= 0:
            result = np.append(result, 1)
        else:
            result = np.append(result, 0)
    return result

def translate(v):
    if v == 0:
        return 'blue'
    else:
        return 'red'

num_points = 1000
min_val = -2
max_val = 2

x1 = np.random.uniform(min_val, max_val, num_points)
x2 = np.random.uniform(min_val, max_val, num_points)

data = []

W = np.array([[1, -1], [-1, -1], [0, -1]])
b = np.array([[1], [1], [-1]])
U = np.array([1, 1, -1])
c = np.array([-1.5])

for i in range(num_points):
    result = equation(x1[i], x2[i])
    data.append(translate(result[0]))

# Create a scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(x1, x2, c=data, alpha=0.6, edgecolors='w', s=50)

# Add titles and labels
plt.title('Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')

# Set the aspect of the plot to be equal
plt.gca().set_aspect('equal', adjustable='box')

# Show the plot
plt.show()



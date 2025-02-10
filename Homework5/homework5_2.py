import numpy as np
import matplotlib.pyplot as plt

num_points = 1000
a = 5

# Loss function
def squared_loss(pred_y, actual_y):
    return (pred_y - actual_y) ** 2


# Loss function derivative
def squared_loss_derivative(pred_y, actual_y):
    return 2 * (pred_y - actual_y)


# Forward pass using the step activation function
def forward_pass_step(x1, x2, W, b, U, c):
    x = np.array([[x1], [x2]])
    return step(np.dot(U, step(np.dot(W, x) + b)) + c)


# Step activation function
def step(v):
    result = np.array([])
    for i in v:
        if i >= 0:
            result = np.append(result, 1)
        else:
            result = np.append(result, 0)
    return result


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-a * x))


# Derivative of sigmoid
def sigmoid_derivative(x):
    sig = sigmoid(x)
    return a * sig * (1 - sig)


# Forward pass using the sigmoid activation function
def forward_pass_sigmoid(x, W, b, U, c):
    # Hidden layer
    v_z = np.dot(W, x) + b
    z = np.array([])
    for i in range(v_z.shape[0]):
        z = np.append(z, sigmoid(v_z[i]))
    
    # Output layer
    v_f = np.dot(U, z) + c
    f = np.array([])
    for i in range(v_f.shape[0]):
        f = np.append(f, sigmoid(v_f[i]))
    
    return v_z, z, v_f, f


# Backward pass
def backward_pass(x, y, z, f, W, U, v_z, v_f):
    grad_f = squared_loss_derivative(f, y)
    delta_f = sigmoid_derivative(v_f) * grad_f

    grad_z = np.dot(U.T, delta_f)
    delta_z = sigmoid_derivative(v_z) * grad_z

    delta_x = delta_z * W.T

    grad_W = np.outer(delta_z, x.T)
    grad_b = delta_z
    grad_U = np.outer(delta_f, z.T)
    grad_c = delta_f

    return grad_W, grad_b, grad_U, grad_c


# Apply backpropagation to the neural network
def backpropagation(X, Y, num_x=2, num_h=3, num_y=1, epochs=100, eta=0.01, initialization='random'):
    if initialization == 'random':
        W = np.random.normal(0, 0.1, (num_h, num_x))
        b = np.random.normal(0, 0.1, num_h)
        U = np.random.normal(0, 0.1, (num_y, num_h))
        c = np.random.normal(0, 0.1, num_y)
    elif initialization == 'zeros':
        W = np.zeros((num_h, num_x))
        b = np.zeros(num_h)
        U = np.zeros((num_y, num_h))
        c = np.zeros(num_y)
    elif initialization == 'low_variance':
        W = np.random.normal(0, 0.001, (num_h, num_x))
        b = np.random.normal(0, 0.001, num_h)
        U = np.random.normal(0, 0.001, (num_y, num_h))
        c = np.random.normal(0, 0.001, num_y)
    elif initialization == 'high_variance':
        W = np.random.normal(0, 5, (num_h, num_x))
        b = np.random.normal(0, 5, num_h)
        U = np.random.normal(0, 5, (num_y, num_h))
        c = np.random.normal(0, 5, num_y)

    mse_values = []
    for _ in range(epochs):
        mse = 0
        for i in range(num_points):
            x = X[i]
            y = Y[i]
            
            # Forward and Backward passes
            v_z, z, v_f, f = forward_pass_sigmoid(x, W, b, U, c)
            grad_W, grad_b, grad_U, grad_c = backward_pass(x, y, z, f, W, U, v_z, v_f)
            
            # Updates
            W -= eta * grad_W
            b -= eta * grad_b
            U -= eta * grad_U
            c -= eta * grad_c

            # Evaluation
            mse += squared_loss(f, y)
        
        mse /= len(X)
        mse_values.append(mse)
    
    # Final obtained MSE value
    print(mse_values[-1])

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(mse_values)
    plt.title(f"MSE vs Epochs - eta = {eta} - Epochs = {epochs} - Hidden neurons = {num_h}\n Initialization = {initialization} - a = {a}")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.grid(True)

    return W, b, U, c


def decision_boundary(X, W, b, U, c, eta=0.01, epochs=100, num_h=3, initialization='random'):
    x1 = X[:, 0]
    x2 = X[:, 1]
    
    y_predicted = np.array([])
    for i in range(num_points):
        _, _, _, res = forward_pass_sigmoid(np.array([x1[i], x2[i]]), W, b, U, c)
        y_predicted = np.append(y_predicted, res)
    
    # Plotting the decision boundary
    plt.figure(figsize=(7, 10))
    plt.title(f"eta: {eta} - Epochs = {epochs} - Hidden neurons = {num_h}\n Initialization = {initialization} - a = {a}")
    ax = plt.axes(projection="3d")
    ax.scatter3D(x1, x2, y_predicted, color="green")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.zaxis._axinfo['juggled'] = (2, 2, 1)


def main():
    # HW2 dataset
    min_val = -2
    max_val = 2
    x1 = np.random.uniform(min_val, max_val, num_points)
    x2 = np.random.uniform(min_val, max_val, num_points)
    X = np.column_stack((x1, x2))

    # Actual weights and biases of the HW2 network
    W = np.array([[1, -1], [-1, -1], [0, -1]])
    b = np.array([[1], [1], [-1]])
    U = np.array([1, 1, -1])
    c = np.array([-1.5])

    y = np.array([])
    for i in range(num_points):
        result = forward_pass_step(x1[i], x2[i], W, b, U, c)
        y = np.append(y, result)

    # Backpropagation
    W_res, b_res, U_res, c_res = backpropagation(X, y)
    decision_boundary(X, W_res, b_res, U_res, c_res, c)

    # Hack1: change eta
    W_res, b_res, U_res, c_res = backpropagation(X, y, eta=0.1)
    decision_boundary(X, W_res, b_res, U_res, c_res, eta=0.1)
    W_res, b_res, U_res, c_res = backpropagation(X, y, eta=0.001)
    decision_boundary(X, W_res, b_res, U_res, c_res, eta=0.001)

    # Hack2: change number of epochs
    W_res, b_res, U_res, c_res = backpropagation(X, y, epochs=50)
    decision_boundary(X, W_res, b_res, U_res, c_res, epochs=50)
    W_res, b_res, U_res, c_res = backpropagation(X, y, epochs=500)
    decision_boundary(X, W_res, b_res, U_res, c_res, epochs=500)

    # Hack3: change number of hidden neurons
    W_res, b_res, U_res, c_res = backpropagation(X, y, num_h=1)
    decision_boundary(X, W_res, b_res, U_res, c_res, num_h=1)
    W_res, b_res, U_res, c_res = backpropagation(X, y, num_h=20)
    decision_boundary(X, W_res, b_res, U_res, c_res, num_h=20)

    # Hack4: change initialization technique
    W_res, b_res, U_res, c_res = backpropagation(X, y, initialization='zeros')
    decision_boundary(X, W_res, b_res, U_res, c_res, initialization='zeros')
    W_res, b_res, U_res, c_res = backpropagation(X, y, initialization='low_variance')
    decision_boundary(X, W_res, b_res, U_res, c_res, initialization='low_variance')
    W_res, b_res, U_res, c_res = backpropagation(X, y, initialization='high_variance')
    decision_boundary(X, W_res, b_res, U_res, c_res, initialization='high_variance')

    # Hack5: change value of a for the sigmoid function
    global a
    a = 1
    W_res, b_res, U_res, c_res = backpropagation(X, y)
    decision_boundary(X, W_res, b_res, U_res, c_res)
    a = 10
    W_res, b_res, U_res, c_res = backpropagation(X, y)
    decision_boundary(X, W_res, b_res, U_res, c_res)

    # Best configuration
    a = 5
    W_res, b_res, U_res, c_res = backpropagation(X, y, num_h=20, epochs=500, eta=0.01, initialization='random')
    decision_boundary(X, W_res, b_res, U_res, c_res, num_h=20, epochs=500, eta=0.01, initialization='random')
    
    plt.show()


if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

D = 100
ETA = 0.001
EPOCHS = 10

# Least Mean Square algorithm
def LMS(X, Y, d, epochs, eta):
    W = np.zeros((10, d)) 					# W initial value
    mse_values = []

    for _ in range(epochs):
        mse = 0
        for i in range(Y.shape[1]):
            x_i = X[:,i].reshape(-1, 1) 	# dx1
            y_i = Y[:,i].reshape(-1, 1)		# 10x1
            error = y_i - np.dot(W, x_i)	# 10x1
            W += eta * np.dot(error, x_i.T)	# 10xd

            # Evaluation
            mse += (np.sum(error ** 2) / Y.shape[1])

        mse_values.append(mse)

    return W, mse_values


# Preprocess the MNIST dataset and find the optimal W
def preprocessAndBestW(Xraw, Yraw, d=D, prediction=False, epochs=EPOCHS, eta=ETA):
    M = np.random.uniform(low=0, high=1, size=(d, 784)) / (255 * d) # dx784
    X = np.dot(M, Xraw.T)											# dx784 * 784x70000 = dx70000
    Y = np.eye(10)[Yraw].T 											# 10x70000
    if prediction == False:	# W is the theoretical value given by pseudoinverse operation
        W, mse_values = np.dot(Y, np.linalg.pinv(X)), None 			# 10xd
    else:	# W gets predicted (point d)
        W, mse_values = LMS(X, Y, d, epochs, eta)
    
    pred = np.dot(W, X) 											# 10x70000
    
    # Evaluation
    mse = np.mean(np.linalg.norm(Y - pred, axis=0) ** 2)
    
    pred_labels = np.argmax(pred, axis=0)
    obs_labels = np.argmax(Y, axis=0)
    mistakes = np.sum(pred_labels != obs_labels)

    return mse, mse_values, mistakes


def main():
    # Point a
    # Loading MNIST data
    mnist = fetch_openml('mnist_784', version=1)
    Xraw, Yraw = mnist['data'].to_numpy(), mnist['target'].astype(np.int32)

    # Digits plot
    plt.figure(figsize=(10, 4))
    for digit in range(10):
        index = np.where(Yraw == digit)[0][0]
        image = Xraw[index].reshape(28, 28)
        plt.subplot(2, 5, digit + 1)
        plt.imshow(image, cmap='gray', interpolation='nearest')
        plt.title(f'Digit: {digit}')
        plt.axis('off')
    plt.tight_layout(pad=2.0)

    # Point c
    d_values = [10, 50, 100, 200, 500]
    for d in d_values:
        mse, _, mistakes = preprocessAndBestW(Xraw, Yraw, d)
        print(f"d = {d}: MSE = {mse}, #mistakes = {mistakes}")

    # Point d
    mse, mse_values, mistakes = preprocessAndBestW(Xraw, Yraw, D, True)
    print(f"LMS configuration: epochs = {EPOCHS}, eta = {ETA}")
    print(f"MSE values: {mse_values}")
    print(f"d = {D}: MSE = {mse}, #mistakes = {mistakes}")

    # MSE-mistakes plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(EPOCHS), mse_values, marker='o')
    plt.title('MSE vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.xticks(range(EPOCHS))
    plt.ylim(0, 1)
    plt.tight_layout(pad=2.0)
    plt.grid()

    # Improve the estimated value of W
    epochs_improved = 150
    eta_improved = 0.005
    mse, mse_values, mistakes = preprocessAndBestW(Xraw, Yraw, D, True, epochs_improved, eta_improved)
    print(f"LMS configuration: epochs = {epochs_improved}, eta = {eta_improved}")
    print(f"MSE values: {mse_values}")
    print(f"d = {D}: MSE = {mse}, #mistakes = {mistakes}")

    # MSE-mistakes plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs_improved), mse_values, marker='o')
    plt.title('MSE vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.xticks(range(0, epochs_improved, 10))
    plt.ylim(0, 1)
    plt.tight_layout(pad=2.0)
    plt.grid()

    plt.show()

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle
from scipy.special import expit as sigmoid 

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        #self.activation_fn = activation -> activation function
        self.activation_fn, self.activation_derivative = self._get_activation(activation)

        # TODO: define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

        self.hidden = None
        self.output = None
        self.grad_W1 = None
        self.grad_W2 = None

    def _get_activation(self, activation):
        """Returns the activation function and its derivative."""
        if activation == 'tanh':
            return np.tanh, lambda x: 1 - np.tanh(x)**2
        elif activation == 'relu':
            return lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float) + 1e-5
        elif activation == 'sigmoid':
            return sigmoid, lambda x: sigmoid(x) * (1 - sigmoid(x))
        else:
            raise ValueError("Invalid activation function. Choose from 'tanh', 'relu', or 'sigmoid'.")

    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        # TODO: store activations for visualization
        self.hidden = self.activation_fn(X @ self.W1 + self.b1) 
        self.output = sigmoid(self.hidden @ self.W2 + self.b2)   
        return self.output

    def backward(self, X, y):
        # TODO: compute gradients using chain rule
        # TODO: update weights with gradient descent
        # TODO: store gradients for visualization
        output_error = self.output - y
        grad_W2 = self.hidden.T @ output_error
        grad_b2 = np.sum(output_error, axis=0, keepdims=True)

        hidden_error = output_error @ self.W2.T * self.activation_derivative(self.hidden)
        grad_W1 = X.T @ hidden_error
        grad_b1 = np.sum(hidden_error, axis=0, keepdims=True)

        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1
    
        self.grad_W1 = grad_W1
        self.grad_W2 = grad_W2

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # TODO: Plot hidden features
    hidden_features = mlp.hidden
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title("Hidden Layer Features")

    # TODO: Hyperplane visualization in the hidden space
    # TODO: Distorted input space transformed by the hidden layer
    # TODO: Plot input layer decision boundary\
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid).reshape(xx.shape)

    ax_input.contourf(xx, yy, preds, levels=np.linspace(0, 1, 10), cmap='bwr', alpha=0.2)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k')
    ax_input.set_title("Input Space with Decision Boundary")

    # TODO: Visualize features and gradients as circles and edges 
    # The edge thickness visually represents the magnitude of the gradient
    for i in range(mlp.W1.shape[0]):
        for j in range(mlp.W1.shape[1]):
            ax_gradient.arrow(0, 0, mlp.W1[i, j], mlp.grad_W1[i, j], head_width=0.02, color='b' if mlp.grad_W1[i, j] > 0 else 'r')
    ax_gradient.set_title("Gradient Magnitudes")


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
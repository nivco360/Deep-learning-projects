import numpy as np
import matplotlib.pyplot as plt
import math
import os
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def initialize_parameters(layer_dims):
    """
    Initialize neural network parameters (weights and biases) for all layers.

    Args:
        layer_dims (list): Dimensions of each layer in the network

    Returns:
        parameters_dict: Dictionary containing weights (W) and biases (b) for each layer

    """
    parameters_dict = {}

    for i in range(1, len(layer_dims)):
        # Weights are initialized using He initialization for better gradient flow
        parameters_dict[f'W{i}'] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * np.sqrt(2. / layer_dims[i - 1])
        parameters_dict[f'b{i}'] = np.zeros((layer_dims[i], 1))

    return parameters_dict


def linear_forward(A, W, b):
    """
    Perform linear forward propagation step.

    Args:
        A (numpy.ndarray): Input activations from previous layer
        W (numpy.ndarray): Weight matrix of current layer
        b (numpy.ndarray): Bias vector of current layer

    Returns:
            Z: Output of linear transformation
            cache: Tuple containing (A, W, b) for use in backpropagation
    """

    Z = np.matmul(W, A) + b
    linear_cache = (A, W, b)

    return Z, linear_cache


def softmax(Z):
    """
    Apply softmax activation function.

    Args:
        Z (numpy.ndarray): Input pre-activation values

    Returns:
            A: Output probabilities after softmax activation
            activation_cache: Input Z for use in backpropagation
    """

    shift_Z = Z - np.max(Z, axis=0, keepdims=True)  # Prevent overflow
    exp_Z = np.exp(shift_Z)
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    activation_cache = Z

    return A, activation_cache


def relu(Z):
    """
    Apply ReLU activation function.

    Args:
        Z (numpy.ndarray): Input pre-activation values

    Returns:
            A: Output after ReLU activation
            activation_cache: Input Z for use in backpropagation
    """
    A = np.maximum(Z, 0)
    activation_cache = Z

    return A, activation_cache


def linear_activation_forward(A_prev, W, B, activation):
    """
    Combine linear forward propagation with activation function.

    Args:
        A_prev (numpy.ndarray): Activations from previous layer
        W (numpy.ndarray): Weight matrix of current layer
        B (numpy.ndarray): Bias vector of current layer
        activation (str): Type of activation function ('relu' or 'softmax')

    Returns:
            A: Output after activation of current layer
            cache: Tuple containing both linear_cache and activation_cache for backpropagation
    """
    cache = {}
    Z, linear_cache = linear_forward(A_prev, W, B)

    if activation == 'relu':
        A, activation_cache = relu(Z)
    elif activation == 'softmax':
        A, activation_cache = softmax(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


def l_model_forward(X, parameters, use_batchnorm=False):
    """
    Forward propagation for the entire neural network.

    Args:
        X (numpy.ndarray): Input data, shape (input_dim, number of samples)
        parameters (dict): Neural network parameters for each layer (weights and biases)
        use_batchnorm (bool): Whether to apply batch normalization after each layer

    Returns:
            AL: the last post-activation value
            caches: List of caches containing information needed for backprop
    """

    layer_dim = len(parameters) // 2
    caches = []
    A = X

    for j in range(1, layer_dim):
        A_prev = A

        A, cache = linear_activation_forward(A_prev, parameters[f'W{j}'], parameters[f'b{j}'], activation='relu')

        if use_batchnorm:
            A = apply_batchnorm(A)
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters[f'W{layer_dim}'], parameters[f'b{layer_dim}'], activation='softmax')
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    """
    Compute the cross-entropy cost function.

    Args:
        AL (numpy.ndarray): Output of the final layer (predictions), shape (num_classes, number of samples)
        Y (numpy.ndarray): True labels (one-hot encoded), shape (num_classes, number of samples)

    Returns:
        float: Computed cost value
    """
    m = AL.shape[1]

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    AL = np.clip(AL, epsilon, 1 - epsilon)

    cost = -1/m * np.sum(Y * np.log(AL))

    return cost


def apply_batchnorm(A):
    """
    Apply batch normalization to layer activations.

    Args:
        A (numpy.ndarray): Input activations value

    Returns:
        numpy.ndarray: Normalized activations values (NA)
    """
    means = np.mean(A.T, axis=0)
    std_vat = np.std(A.T, axis=0)
    NA = (A.T - means) / (std_vat + 0.0001)  # to avoid division by zero

    return NA.T


def linear_backward(dZ, cache):
    """
    Compute gradients for linear layer during backpropagation.

    Args:
        dZ (numpy.ndarray): Gradient of the cost with respect to linear output
        cache (tuple): Contains (A_prev, W, b) from forward propagation

    Returns:
        dA_prev: Gradient with respect to activations from previous layer
        dW: Gradient with respect to weights
        db: Gradient with respect to biases
    """

    A_prev, W, b = cache

    m = A_prev.shape[1]
    dA_prev = np.dot(W.T, dZ)
    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Backward propagation through activation and linear layer.

    Args:
        dA (array): Gradient of cost with respect to current layer's activation
        cache (tuple): ((A_prev,W,b), Z) cached during forward propagation
        activation (str): Type of activation - 'relu' or 'softmax'

    Returns:
        dA_prev: Gradient with respect to previous layer activation
        dW: Gradient with respect to weights
        db: Gradient with respect to bias
    """

    linear_cache, activation_cache = cache  # Unpack from tuple

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == 'softmax':
        dZ = softmax_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def relu_backward(dA, activation_cache):
    """
    Compute gradient for ReLU activation during backpropagation.

    Args:
        dA (numpy.ndarray): The post-activation gradient
        activation_cache (numpy.ndarray): Z values from forward propagation

    Returns:
        numpy.ndarray: Gradient of the cost with respect to Z
    """
    Z = activation_cache
    dZ = dA * (Z > 0)  # if Z > 0 then 1, else 0

    return dZ


def softmax_backward(dA, activation_cache):
    """
    Compute gradient for softmax activation during backpropagation.

    Args:
        dA (numpy.ndarray): The post-activation gradient
        activation_cache (numpy.ndarray): Z values from forward propagation

    Returns:
        numpy.ndarray: Gradient of the cost with respect to Z
    """
    # In L_model_backward, the derivative is already computed as (dA = AL - Y), so need to compute it again.
    dZ = dA

    return dZ 


def l_model_backward(AL, Y, caches):
    """
    Backward propagation for the entire neural network.

    Args:
        AL (numpy.ndarray): The probablity vector of each class, shape (number of classes, number of samples)
        Y (numpy.ndarray): True labels
        caches (list): List of caches from forward propagation

    Returns:
        grads: A dictionary of Gradients for all parameters in the network
    """
    num_layers = len(caches)
    dA = AL - Y
    grads = {}
    grads[f'dA{num_layers}'], grads[f'dW{num_layers}'], grads[f'db{num_layers}'] = linear_activation_backward(dA, caches[-1], 'softmax')

    for i in reversed(range(num_layers - 1)):
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads[f'dA{i+2}'], caches[i], 'relu')
        grads[f'dA{i+1}'] = dA_prev_temp
        grads[f'dW{i+1}'] = dW_temp
        grads[f'db{i+1}'] = db_temp
    return grads


def update_parameters(parameters, grads, learning_rate, L2_norm=False):
    """
    Update network parameters using gradient descent.

    Args:
        parameters (dict): Network parameters
        grads (dict): Computed gradients
        learning_rate (float): Learning rate for gradient descent
        L2_norm (bool): Whether to apply L2 regularization during update

    Returns:
        parameters: Updated dictionary parameters
    """
    num_layers = len(parameters) // 2
    lamda = 0.000001
    for i in range(1, num_layers + 1):
        if L2_norm:
            parameters[f'W{i}'] =parameters[f'W{i}'] -learning_rate * (grads[f'dW{i}']+ lamda *parameters[f'W{i}'])
        else:
            parameters[f'W{i}'] = parameters[f'W{i}'] - learning_rate * grads[f'dW{i}']
        parameters[f'b{i}'] = parameters[f'b{i}'] - learning_rate * grads[f'db{i}']

    return parameters


def l_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size, use_batchnorm=False, L2_norm=False):
    """
    Train the neural network model.

    Args:
        X (numpy.ndarray): Input data,  shape (input size, number of samples)
        Y (numpy.ndarray): True labels (one hot matrix), shape (number of classes, number of samples)
        layers_dims (list): Dimensions of each layer
        learning_rate (float): Learning rate for gradient descent
        num_iterations (int): Maximum number of training iterations
        batch_size (int): Size of mini-batches
        use_batchnorm (bool): Whether to use batch normalization
        L2_norm (bool): Whether to use L2 regularization

    Returns:
        parameters: Trained model parameters
        costs: List of training and validation costs
        train_accuracy: Final training accuracy
        val_accuracy: Final validation accuracy
    """
    X, Y = X.T, Y.T
    costs = []

    # Split into training and validation sets (80-20)
    indices = np.random.permutation(X.shape[0])
    split_index = math.ceil(0.8 * X.shape[0])
    X_train, X_val = X[indices[:split_index], :], X[indices[split_index:], :]
    Y_train, Y_val = Y[indices[:split_index], :], Y[indices[split_index:], :]

    parameters = initialize_parameters(layers_dims)
    num_batches = math.ceil(X_train.shape[0] / batch_size)

    # Split training data into batches
    batches_X = [X_train[i * batch_size:(i + 1) * batch_size].T for i in range(num_batches)]
    batches_Y = [Y_train[i * batch_size:(i + 1) * batch_size].T for i in range(num_batches)]

    # Variables for early stopping
    best_val_accuracy = 0
    steps_without_improvement = 0
    learning_step= 0
    epoch = 0
    lamda = 0.000001


    print(f"Training with batch size: {batch_size}")
    print(f"Batches per epoch: {num_batches}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")

    while epoch < num_iterations:  # Maximum number of epochs
        epoch += 1

        for batch_index in range(num_batches):
            learning_step +=1
            # Forward propagation
            AL_train, caches = l_model_forward(batches_X[batch_index], parameters,use_batchnorm)

            # Backward propagation
            grads = l_model_backward(AL_train, batches_Y[batch_index], caches)

            # Update parameters
            parameters = update_parameters(parameters, grads, learning_rate,L2_norm)

            # validation accuracy check
            if learning_step % 100 == 0:
                current_val_accuracy = predict(X_val.T, Y_val.T, parameters, use_batchnorm)
                AL_val, caches = l_model_forward(X_val.T, parameters,use_batchnorm)

                # Check if accuracy improved
                if current_val_accuracy > best_val_accuracy + 1e-4:
                    best_val_accuracy = current_val_accuracy
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1

                cost_train = compute_cost(AL_train, batches_Y[batch_index])
                cost_val = compute_cost(AL_val, Y_val.T)


                if L2_norm:
                    for l in range(len(layers_dims)-1):
                        w = parameters[f'W{l+1}']
                        cost_train = cost_train + (lamda)* np.sum(w*w)
                # Print progress
                costs.append((cost_train,cost_val))
                print(f"Epoch {epoch}, Learning Step {learning_step}: Cost = {cost_train:.4f}, Validation Accuracy {current_val_accuracy*100:.4f}%")

                # Check stopping criterion: 100 steps without improvement
                if steps_without_improvement >= 100:

                    # Calculate total possible epochs and steps
                    total_possible_epochs = num_iterations
                    total_possible_batches = num_iterations * num_batches
                    Total_batches_completed = epoch*num_batches+batch_index

                    print(f"\nStopping training - No improvement for 100 steps")
                    print(f"Total epochs completed: {epoch} from {total_possible_epochs}")
                    print(f"Total batches completed: {Total_batches_completed} from {total_possible_batches}")
                    print(f"Best validation accuracy: {best_val_accuracy * 100:.4f}%")
                    

                    # Calculate final accuracies
                    train_accuracy = predict(X_train.T, Y_train.T, parameters, use_batchnorm)
                    val_accuracy = predict(X_val.T, Y_val.T, parameters, use_batchnorm)

                    return parameters, costs, train_accuracy, val_accuracy

    # If we reach max iterations without early stopping
    print(f"\nReached maximum iterations")
    print(f"Total epochs completed: {epoch}")
    print(f"Total training step completed: {learning_step}")
    print(f"Best validation accuracy: {best_val_accuracy * 100:.4f}%")

    # Calculate final accuracies
    train_accuracy = predict(X_train.T, Y_train.T, parameters, use_batchnorm)
    val_accuracy = predict(X_val.T, Y_val.T, parameters, use_batchnorm)

    return parameters, costs, train_accuracy, val_accuracy


def predict(X, Y, parameters, use_batchnorm):
    """
    Make predictions using the trained model and compute accuracy.

    Args:
        X (numpy.ndarray): Input data
        Y (numpy.ndarray): True labels
        parameters (dict): Trained model parameters
        use_batchnorm (bool): Whether to use batch normalization

    Returns:
        float: Prediction accuracy
    """
    samples = X.shape[1]
    y_proba, _ = l_model_forward(X, parameters, use_batchnorm)  # forward propagation
    y_pred = np.argmax(y_proba, axis=0, keepdims=True)
    Y = np.argmax(Y, axis=0, keepdims=True)
    correct_count = np.sum(y_pred == Y)
    accuracy = correct_count / samples
    return accuracy


def preprocess_data(X, y):
    """
    Preprocess input data for neural network training.

    Args:
        X (numpy.ndarray): Input features
        y (numpy.ndarray): Labels

    Returns:
        X_processed: Flattened and normalized input data
        y_processed: One-hot encoded labels
    """

    if isinstance(X, np.ndarray) is False:
        X = X.to_numpy()
    X_flat = X.reshape(X.shape[0], -1).T / 255.0
    y_one_hot = np.zeros((y.size, 10))
    y_one_hot[np.arange(y.size), y] = 1

    return X_flat, y_one_hot.T


def plot_method_results(costs, method):
    """
    Plot training and validation costs over iterations.

    Args:
        costs (list): List of tuples containing (train_cost, val_cost)
        method (str): Name of the training method for plot title

    Returns:
        None: Saves plot to file
    """

    # Get the index for the specified method
    train_costs, val_costs = zip(*costs)
    # Plot Training and Validation Costs
    iterations = range(len(train_costs))

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_costs, label=f'{method} (Train Cost)', alpha=0.7)
    plt.plot(iterations, val_costs, label=f'{method} (Val Cost)', linestyle='--', alpha=0.7)
    plt.title(f'Comparison of Training and Validation Costs - {method}')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid()
    path = 'Your path'
    # path = r'd:/Niv/Deep Learning/Exercise 1'
    full_path = os.path.join(path, method)
    plt.savefig(full_path)
    plt.close()

# Load data
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype(np.float32)
y = mnist.target.astype(np.int32)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)

# Parameters
learning_rate = 0.009
batch_size = 256
num_iterations = 100000
layers_dims = [x_train.shape[0], 20, 7, 5, 10]  # the dimensions of the layers in the network
use_batchnorm = False
L2_norm = False

# ---------------------- 4. without batchnorm ------------------------------------------------

start_time = time.time()

parameters_4, costs_without_batchnorm_L2_norm , train_acc_without_batchnorm_L2_norm , val_acc_without_batchnorm_L2_norm = l_layer_model(
    x_train, y_train,
    layers_dims,
    learning_rate,
    num_iterations,
    batch_size,
    use_batchnorm,
    L2_norm
)
end_time = time.time()
elapsed_time = end_time - start_time

test_accuracy_without_batchnorm_L2_norm = predict(x_test, y_test, parameters_4, use_batchnorm)

# Report final details
print("\nFinal Report:")
print(f"Training Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print(f"Final Training Accuracy: {train_acc_without_batchnorm_L2_norm * 100:.2f}%")
print(f"Final Validation Accuracy: {val_acc_without_batchnorm_L2_norm * 100:.2f}%")
print(f"Final Test Accuracy: {test_accuracy_without_batchnorm_L2_norm * 100:.2f}%")

method = 'Without BatchNorm & L2_Norm'
plot_method_results(costs_without_batchnorm_L2_norm, method)

# ---------------------- 5. with batchnorm ------------------------------------------------

start_time = time.time()
use_batchnorm = True
parameters, costs_with_batchnorm, train_acc_with_batchnorm, val_acc_with_batchnorm = l_layer_model(
    x_train, y_train,
    layers_dims,
    learning_rate,
    num_iterations,
    batch_size,
    use_batchnorm,
    L2_norm
)
end_time = time.time()
elapsed_time = end_time - start_time

test_accuracy_with_batchnorm = predict(x_test, y_test, parameters, use_batchnorm)

# Report final details
print("\nFinal Report:")
print(f"Training Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print(f"Final Training Accuracy: {train_acc_with_batchnorm * 100:.2f}%")
print(f"Final Validation Accuracy: {val_acc_with_batchnorm * 100:.2f}%")
print(f"Final Test Accuracy: {test_accuracy_with_batchnorm* 100:.2f}%")

method =  'With BatchNorm'
plot_method_results(costs_with_batchnorm, method)

# ---------------------- 6. with L2_norm & without batchnoorm ------------------------------------------------

start_time = time.time()
L2_norm = True
use_batchnorm = False
parameters_L2, costs_with_L2_norm_without_batchnorm, train_acc_with_L2_norm_without_batchnorm, val_acc_with_L2_norm_without_batchnorm = l_layer_model(
    x_train, y_train,
    layers_dims,
    learning_rate,
    num_iterations,
    batch_size,
    use_batchnorm,
    L2_norm
)
end_time = time.time()
elapsed_time = end_time - start_time

test_accuracy_with_L2_norm_without_batchnorm = predict(x_test, y_test, parameters_L2, use_batchnorm)

# Report final details
print("\nFinal Report:")
print(f"Training Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print(f"Final Training Accuracy: {train_acc_with_L2_norm_without_batchnorm * 100:.2f}%")
print(f"Final Validation Accuracy: {val_acc_with_L2_norm_without_batchnorm * 100:.2f}%")
print(f"Final Test Accuracy: {test_accuracy_with_L2_norm_without_batchnorm * 100:.2f}%")

method = 'With L2_Norm without BatchNorm'
plot_method_results(costs_with_L2_norm_without_batchnorm, method)


# # Compare wights difference with and without L2 norm
weight_diff = np.sum(parameters_4['W1'] - parameters_L2['W1']) + np.sum(parameters_4['W2'] - parameters_L2['W2'])\
              + np.sum(parameters_4['W3'] - parameters_L2['W3']) + np.sum(parameters_4['W4'] - parameters_L2['W4'])
print("The difference between the weights of the model with or without L2 norm is " + str(weight_diff))

# ----------------------7. with L2_norm & batchnoorm ------------------------------------------------
start_time = time.time()
L2_norm = True
use_batchnorm = True
parameters, costs_with_L2_norm_with_batchnorm, train_acc_with_L2_norm_with_batchnorm, val_acc_with_L2_norm_with_batchnorm = l_layer_model(
    x_train, y_train,
    layers_dims,
    learning_rate,
    num_iterations,
    batch_size,
    use_batchnorm,
    L2_norm
)
end_time = time.time()
elapsed_time = end_time - start_time

test_accuracy_with_L2_norm_with_batchnorm = predict(x_test, y_test, parameters, use_batchnorm)

# Report final details
print("\nFinal Report:")
print(f"Training Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print(f"Final Training Accuracy: {train_acc_with_L2_norm_with_batchnorm * 100:.2f}%")
print(f"Final Validation Accuracy: {val_acc_with_L2_norm_with_batchnorm * 100:.2f}%")
print(f"Final Test Accuracy: {test_accuracy_with_L2_norm_with_batchnorm * 100:.2f}%")

method = 'With BatchNorm & L2_Norm'
plot_method_results(costs_with_L2_norm_with_batchnorm, method)


    
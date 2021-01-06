import numpy as np

from datasets import spiral_data, vertical_data
from main import (Actiation_ReLU, Activation_Softmax, Layer_Dense,
                  Loss_CategoricalCrossentropy)

X, y = vertical_data(samples=100, classes=3)
# X, y = spiral_data(samples=100, classes=3)

# model
dense1 = Layer_Dense(2, 3)
activation1 = Actiation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

# loss
loss_function = Loss_CategoricalCrossentropy()

# helpers
lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for i in range(10000):

    # update weights with small random values
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    # perform forward pass of training data
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # get loss of second dense layer post activation
    loss = loss_function.calculate(activation2.output, y)

    # calculate accuracy values along first axis
    predictions = np.argmax(activation2.output, axis=1)
    print(predictions)
    accuracy = np.mean(predictions == y)

    # if loss is smaller, print and save eights and biases
    if loss < lowest_loss:
        print(f'new set of weights found, iteration: {i} loss: {loss} acc: {accuracy}')
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()

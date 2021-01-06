import numpy as np
from datasets import spiral_data
import nnfs

nnfs.init()

class Layer_Dense:
     def __init__(self, n_inputs, n_neurons):
          self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
          self.biases = np.zeros((1, n_neurons))

     def forward(self, inputs):
          self.output = np.dot(inputs, self.weights) + self.biases

class Actiation_ReLU:
     def forward(self, inputs):
          self.output = np.maximum(0, inputs)

class Activation_Softmax:
     def forward(self, inputs):
          exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
          probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
          self.output = probabilities

class Loss:

     # Calculated the data and regularization losses
     # given model output and ground truth values
     def calculate(self, output, y):

          # calculate sample losses
          sample_losses = self.forward(output, y)

          # calculate mean loss
          data_loss = np.mean(sample_losses)

          return data_loss

class Loss_CategoricalCrossentropy(Loss):

     # forward pass
     def forward(self, y_pred, y_true):

          # num samples in batch
          samples = len(y_pred)

          # clip data to prevent div/0
          # clip both sides to not drag mean towards any value
          y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

          # probabilities for target values
          # only if categorical labels
          if len(y_true.shape) == 1:
               correct_confidences = y_pred_clipped[
                    range(samples),
                    y_true
               ]

          # mask values only for one-hot encoded labels
          elif len(y_true.shape) == 2:
               correct_confidences = np.sum(
                    y_pred_clipped * y_true,
                    axis=1
               )

          # losses
          negative_log_likelihoods = -np.log(correct_confidences)
          return negative_log_likelihoods

if __name__ == "__main__":
     X, y = spiral_data(samples=100, classes=3)

     dense1 = Layer_Dense(2, 3)
     activation1 = Actiation_ReLU()

     dense2 = Layer_Dense(3, 3)
     activation2 = Activation_Softmax()

     dense1.forward(X)
     activation1.forward(dense1.output)

     dense2.forward(activation1.output)
     activation2.forward(dense2.output)

     print(activation2.output[:5])

     loss_function = Loss_CategoricalCrossentropy()
     loss = loss_function.calculate(activation2.output, y)

     predictions = np.argmax(activation2.output, axis=1)
     if len(y.shape) == 2:
          y = np.argmax(y, axis=1)
     accuracy = np.mean(predictions == y)

     print(f'loss: {loss}\nacc:  {accuracy}')
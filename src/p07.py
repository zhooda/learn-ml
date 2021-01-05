import math
import numpy as np

DOG = 0
CAT = 1
HUMAN = 2

# example output from output layer of nn
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([[CAT, DOG, DOG],
                          [DOG, CAT, DOG],
                          [DOG, CAT, DOG]])


# probabilities for target values
# only if categorical labels
if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs[range(len(softmax_outputs)), class_targets]
elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(
        softmax_outputs * class_targets,
        axis=1
    )

# losses
neg_log = -np.log(correct_confidences)

average_loss = np.mean(neg_log)
print(average_loss)
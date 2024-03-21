#!/usr/bin/env python
#
"""
Simple implementation of Hopfield net.
Source:
https://medium.com/@evertongomede/understanding-hopfield-nets-an-overview-of-recurrent-neural-networks-75fa51d0d2e6
"""

import numpy as np

class HopfieldNet:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        for pattern in patterns:
            pattern = np.array(pattern).reshape(-1, 1)
            self.weights += np.dot(pattern, pattern.T)
            np.fill_diagonal(self.weights, 0)

    def predict(self, input_pattern, max_iterations=100):
        input_pattern = np.array(input_pattern).reshape(-1, 1)
        for _ in range(max_iterations):
            output_pattern = np.sign(np.dot(self.weights, input_pattern))
            if np.array_equal(input_pattern, output_pattern):
                return output_pattern
            input_pattern = output_pattern
        return None

# Usage example:
if __name__ == "__main__":
    patterns = [[1, -1, 1, -1], [-1, 1, -1, 1], [1, 1, -1, -1]]
    input_pattern = [-1, 1, 0, 0]

    hopfield_net = HopfieldNet(num_neurons=len(input_pattern))
    hopfield_net.train(patterns)

    predicted_pattern = hopfield_net.predict(input_pattern, 1000)
    if predicted_pattern is not None:
        print("Predicted pattern:", predicted_pattern.flatten())
    else:
        print("Pattern retrieval failed!")

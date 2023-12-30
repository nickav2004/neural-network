from re import X
import numpy as np
import pandas as pd
from typing import List



class NeuralNetwork:
    def __init__ (self, layer_sizes: List[int]):
        self.layer_sizes = layer_sizes
        self.layers = len(layer_sizes)
        self.weights = [np.random.randn(x,y) for x,y in zip(layer_sizes[1:],layer_sizes)]
        self.biases = [np.random.randn(x) for x in layer_sizes[1:]]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def derivative(self, fun, x):
        delta = 0.001
        return (fun(x + delta) - fun(x)) / delta
           

    def forward_prop(self, input_data):
        if self.layer_sizes[0] != len(input_data):
            raise ValueError("input data does match the input layer size")

        activation = np.array(input_data)
        
        for i in range(self.layers - 1):
            activation = self.relu(self.weights[i] @ activation - self.biases[i])
        return activation
    
    def backward_prop(self, input_data, correct_data: List[float], learning_rate):
        output = self.forward_prop(input_data)
        
    
    
if __name__ == "__main__":
    
    layer_sizes = [784,16,16,10]
    
    nn = NeuralNetwork(layer_sizes)
    
    data = pd.read_csv("../training_data/mnist_test.csv") 
    
    for index, row in data.iterrows():
    
        print(nn.forward_prop(row[1:]))
        break
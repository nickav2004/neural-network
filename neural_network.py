import numpy as np
import pandas as pd
from typing import List




class NeuralNetwork:
    def __init__ (self, layer_sizes: List[int]):
        self.layer_sizes = layer_sizes
        self.weights_list = [np.random.randn(x,y) for x,y in zip(layer_sizes[1:],layer_sizes)]
        self.biases = [np.random.randn(x) for x in layer_sizes[1:]]
        
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x <= 0, 0, 1)
    
    def forward_prop(self, input_vec):
        activations = [np.reshape(input_vec, (len(input_vec), 1))] # reshaping input to column vector
        zs = [] # list to store all z vectors, z = w.a + b
        for b, w in zip(self.biases, self.weights_list):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            activation = self.relu(z)
            activations.append(activation)
        return activations, zs
    
    def backward_prop(self, input_vec, correct_vec, learning_rate=1): # correct data should be a vector same size as output 
        activations, zs = self.forward_prop(input_vec)
        output_vec = activations[-1]
        
        
        def dcost_dactivation(a, y):
            return 2 * (a-y)
        
        def dz_dw(z,w):
            return # activation of z. if z is in n layer then its from the actiavition from n-1
            
        def dz_da(z, a):
            return # weight of for the activation of z in its own layer
        
        
        def compute_grad_layer(layer, weights):
            pass
    
        
        weight_grad_vec = []
        for layer, weights in enumerate(self.weights_list[::-1]):
            pass # function to give all the gradients of these weights
                
        
        
        
        
        

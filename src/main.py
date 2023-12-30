import numpy as np 
import pandas as pd


def ReLu(vector):
    size = vector.shape

    for i in range(size[0]):
        if vector[i] <= 0:
            vector[i] = 0
    return vector

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    


if __name__ == "__main__":

    
    data = pd.read_csv("../training_data/mnist_test.csv") 

    #initilize weights and biases
    weights1_mtrx = np.random.rand(10,784) # Weights for input layer 
    bias1 = np.array([-0.1 for _ in range(10)]) #bias idk what vals they should be 
    weights2_mtrx = np.random.rand(10,10) 
     

    for index, row in data.iterrows():
        
        actual_num = row[0]
        a0 = np.array(row[1:]) # input layer with pixel values from 0 to 255
        
        #1st hidden layer 
        a1 = sigmoid(weights1_mtrx @ a0 + bias1)
       
        #output layer
        a2 = softmax(weights2_mtrx @ a1 + bias1)

        guess = np.argmax(a2)
        print(a2)
        print(guess)
        print(actual_num)

        correct_array = np.array([0 for _ in range (10)])
        correct_array[actual_num] = 1

        cost = np.sum((a2 - correct_array)**2)

        print(correct_array)
        print(cost)
        
        




        



        break


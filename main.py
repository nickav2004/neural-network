import pandas as pd

from neural_network import NeuralNetwork

if __name__ == "__main__":
    
    layer_sizes = [3,4,2]
    
    nn = NeuralNetwork(layer_sizes)
    
    df = pd.read_csv("./training_data/mnist_test.csv") 
    
    target_df = df["label"]
    training_df = df.drop("label",axis=1)
    
    for weight in nn.biases:
        print(weight)
        print()
from numpy import random, dot, exp

class NeuralNetwork:
    def __init__(self, layers=None, weights=None, mutate_prob=0.03):
        # random initialization of weights
        if weights is None:
            self.weights = []
            self.number_of_layers = len(layers)
            for i in range(len(layers) - 1):
                w = 2*random.rand(layers[i], layers[i+1]) - 1
                self.weights.append(w)
        # mutation of genes
        else:
            for column in weights:
                for i in range(len(column)):
                    if mutate_prob > random.rand():
                        column[i] = 2*random.rand() - 1
            self.weights = weights
        
        self.number_of_layers = len(self.weights) + 1
    
    def sigmoid(self, X):
        """ activation function """
        return 1 / (1 + exp(-X))
    
    def feed_forward(self, X):
        """ finds output of neural network provided input """
        l = X
        for i in range(self.number_of_layers - 1):
            w = self.weights[i]
            l = self.sigmoid(dot(w.T, l))
        self.y = l
        return self.y
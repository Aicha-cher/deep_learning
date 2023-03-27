
from Layers import *
from Optimization import *
from copy import deepcopy

class NeuralNetwork:

    def __init__(self, optimizer):

        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = []
        self.loss_layer = []


    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()

        for layer in self.layers:
            self.input_tensor = layer.forward(self.input_tensor)

        return self.loss_layer.forward(self.input_tensor, self.label_tensor)


    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)


    def append_layer(self, layer):
        layer.optimizer = deepcopy(self.optimizer)
        self.layers.append(layer)


    def train(self, iterations):

        for _ in range(iterations):
            self.loss.append(self.forward())
            self.backward()


    def test(self, input_tensor):

        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor

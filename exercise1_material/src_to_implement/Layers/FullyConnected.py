from Layers.Base import *
from Optimization.Optimizers import *

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size+1, output_size) # +1 is for the bias
        self._optimizer = None

    def forward(self, input_tensor):
        self.input_tensor = np.hstack((input_tensor,np.ones([input_tensor.shape[0],1], input_tensor.dtype)))
        self.z = np.dot(self.input_tensor, self.weights)
        return self.z

    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
    
    def backward(self, error_tensor):
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor) # cost derivative
        self.delta = np.dot(error_tensor, self.weights.T) # backprob error
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        return self.delta[:,:-1]


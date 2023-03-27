from Layers.Base import *

class SoftMax(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, input_tensor):
        stable_input_tensor = input_tensor - np.max(input_tensor)
        self.predicton = np.exp(stable_input_tensor) / np.expand_dims(np.sum(np.exp(stable_input_tensor), axis=1), axis=1)
        return self.predicton
        
    
    def backward(self, error_tensor):
        step1 = np.expand_dims(np.sum(error_tensor * self.predicton, axis=1), axis=1)
        step2 = error_tensor - step1
        return self.predicton * step2

    def backward2(self, error_tensor):
        # Compute the Jacobian matrix of the softmax function
        jacobian = np.diagflat(self.predicton) - np.dot(self.predicton, self.predicton.T)
        
        # Multiply the error tensor by the Jacobian matrix to get the gradient
        gradient = np.dot(error_tensor, jacobian)
        
        return gradient
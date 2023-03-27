import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        input_tensor = input_tensor[label_tensor==1]
        step1 = input_tensor + np.finfo(float).eps
        step2 = np.log(step1) * (-1)
        return np.sum(step2)

    def backward(self, label_tensor):
        return (-1) * label_tensor / self.input_tensor 
    # since L = y* np.log(predicted y) 
    # because d/dx [ ln(x) ] = 1/x -> dL/d predicted y = -y/dpredicted y
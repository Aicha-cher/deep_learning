from Layers.FullyConnected import *
X = np.random.rand(20,5) # 20 data points and 5 features
fc = FullyConnected(5, 6) # 5 input and 6 output
fc.forward(X)

print('end')
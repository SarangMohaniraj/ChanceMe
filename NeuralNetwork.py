import numpy as np
from preprocessing import inputs,outputs

THRESHOLD = 0.5 # sigmoid threshold

"""
This is incomplete and will be be used for future models
Neural Network with L-layers
Current Activation function is sigmoid
"""

class NeuralNetwork:

  sigmoid = lambda x: 1 / (1 + np.exp(-x))
  sigmoid_prime = lambda x: sigmoid(x) * (1-sigmoid(x))

  def __init__(self,shape=[5,2,1],activation=sigmoid, activation_derivative=sigmoid_prime,learning_rate=0.3):
    self.shape=shape
    self.activation = activation
    self.activation_derivative = activation_derivative
    self.learning_rate = learning_rate

    # initialize weights and bias
    self.weights = []
    self.biases = []
    np.random.seed(1)
    for l in range(1, len(shape)): # number of layers in the network
      self.weights.append(np.random.randn(shape[l], shape[l-1]) * 0.01)
      self.biases.append(np.zeros((shape[l], 1)))

  def train(self,x,y,epoch=100):
    for e in range(epoch):
      Z = self.feed_forward(x)

  # forward propagation
  def feed_forward(self,x):
    for w,b in zip(self.weights, self.biases): # number of layers in the network
      Z = self.activation(np.dot(w,x)+b)
    return Z

  # binary cross-entropy
  def cost(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

  def backpropagation(dZ, cache):

 


  def train(self,epochs):
    cost_history = []


  # def predict(self,input):
  #   z = np.dot(input,self.weights.T) + self.bias
    # return self.activation(z) > THRESHOLD



nn = NeuralNetwork()
# print(nn.weights)
predictions = nn.feed_forward(inputs)
# predictions = nn.predict([4/5,34/36,np.random.randint(2016,2019+1)/2016,0,1])
print(predictions)

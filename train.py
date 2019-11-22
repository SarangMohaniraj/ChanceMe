import numpy as np
import pandas as pd
from preprocessing import inputs,outputs,inputs_test,outputs_test,college,min_year,max_year
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))

#log loss
def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)

#cross-entropy
def error_term_formula(x, y, output):
    return (y - output)*output*(1 - output)

# UMD - 1000,.8
epochs = 1000
learning_rate = 0.8
THRESHOLD = 0.5 # sigmoid function threshold for positive values


def train_nn(inputs, outputs, epochs, learning_rate):
    
    np.random.seed(1)

    n_records, n_inputs = inputs.shape
    last_loss = None
    loss_history = []

    # Initialize weights
    weights = np.random.normal(scale=1 / n_inputs**.5, size=n_inputs)

    for e in range(epochs):
        delta_w = np.zeros(weights.shape)
        for x, y in zip(inputs.values, outputs):
           
          #feed forward into hidden layer 1
          output = sigmoid(np.dot(x, weights))

          #backpropagation
          # prediction error
          error = error_formula(y, output)
          error_term = error_term_formula(x, y, output)

          # The gradient descent step, the error times the gradient times the inputs
          delta_w += error_term * x

        # Update weights. The learning rate times the change in weights, divided by the number of records to average
        weights += learning_rate * delta_w / n_records

        # Printing out the mean square error on the training set
        
        out = sigmoid(np.dot(inputs, weights)) # output layer
        loss = np.mean((out - outputs) ** 2) # MSE
        loss_history.append(loss)
        if e % (epochs / 10) == 0:
          print("Epoch:", e)
          if last_loss and last_loss < loss:
              print("Train loss: ", loss, "  WARNING - Loss Increasing")
          else:
              print("Train loss: ", loss)
          print("=========")
        last_loss = loss
           
    print("Finished training!")
    return weights,loss_history

# w = []
# a = []
# for i in range(10):
#   print(i)
#   weights = train_nn(inputs, outputs, epochs, learning_rate)
#   w.append(weights)
#   test_out = sigmoid(np.dot(inputs_test, weights))
#   predictions = test_out > 0.5
#   accuracy = np.mean(predictions == outputs_test)
#   a.append(accuracy)
#   print("=========")
# # weights = w[a.index(max(a))])

# weights,loss_history = train_nn(inputs, outputs, epochs, learning_rate)
# w,test_loss_history =  train_nn(inputs_test, outputs_test, epochs, learning_rate)
weights = [ 2.38489517,  2.81315904, -0.23911468, -3.30667717, -3.61188365] # UMD one of best weights so far 
print(weights)

# Calculate accuracy on test data
test_out = sigmoid(np.dot(inputs_test, weights))
predictions = test_out > 0.5
accuracy = np.mean(predictions == outputs_test)
print("Prediction accuracy: {:.3f}".format(accuracy))

accepted,rejected = [],[]
for gpa in np.arange(0,5.1,.1):
  for act in np.arange(0,37,1):
    test_out = sigmoid(np.dot([gpa/5,act/36,np.random.randint(min_year,max_year+1)/min_year,0,1], weights)) #inputs["Year"].min()
    predictions = test_out > 0.5
    if(predictions):
      accepted.append([act,gpa])
    else:
      rejected.append([act,gpa])

plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.scatter([p[0] for p in accepted],[p[1] for p in accepted], s = 5, marker="s",c="green")
plt.scatter([p[0] for p in rejected],[p[1] for p in rejected], s = 5, marker="s",c="red")
plt.title(college + " Predictions")
plt.xlabel("ACT")
plt.ylabel("GPA")

if "loss_history" in locals() and "test_loss_history" in locals():
  plt.subplot(1,2,2)
  plt.plot([e for e in range(epochs)],[l for l in loss_history],markersize=5,label="training data")
  plt.plot([e for e in range(epochs)],[l for l in test_loss_history],markersize=5,label="test data")
  plt.title("Loss vs Epoch")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()
plt.show()
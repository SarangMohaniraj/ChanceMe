import numpy as np
import pandas as pd
from preprocessing import features,targets,features_test,targets_test,college
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))
def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)


# Write the error term formula
def error_term_formula(x, y, output):
    return (y-output)*sigmoid_prime(x)

epochs = 1000
learnrate = 0.5

# Training function
def train_nn(features, targets, epochs, learnrate):
    
    # Use to same seed to make debugging easier
    np.random.seed(42)

    n_records, n_features = features.shape
    last_loss = None

    # Initialize weights
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features.values, targets):
            # Loop through all records, x is the input, y is the target

            # Activation of the output unit
            #   Notice we multiply the inputs and the weights here 
            #   rather than storing h as a separate variable 
            output = sigmoid(np.dot(x, weights))

            # The error, the target minus the network output
            error = error_formula(y, output)

            # The error term
            error_term = error_term_formula(x, y, output)

            # The gradient descent step, the error times the gradient times the inputs
            del_w += error_term * x

        # Update the weights here. The learning rate times the 
        # change in weights, divided by the number of records to average
        weights += learnrate * del_w / n_records

        # Printing out the mean square error on the training set
        if e % (epochs / 10) == 0:
            out = sigmoid(np.dot(features, weights))
            loss = np.mean((out - targets) ** 2)
            print("Epoch:", e)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            print("=========")
    print("Finished training!")
    return weights

# w = []
# a = []
# for i in range(10):
#   print(i)
#   weights = train_nn(features, targets, epochs, learnrate)
#   w.append(weights)
#   test_out = sigmoid(np.dot(features_test, weights))
#   predictions = test_out > 0.5
#   accuracy = np.mean(predictions == targets_test)
#   a.append(accuracy)
#   print("=========")
# # weights = w[a.index(max(a))])

# Calculate accuracy on test data
weights = train_nn(features, targets, epochs, learnrate)

test_out = sigmoid(np.dot(features_test, weights))
predictions = test_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))

test_out = sigmoid(np.dot([4/5,34/36,2019/2016,0,1], weights))
predictions = test_out > 0.5
print(predictions)

print(weights)

accepted,rejected = [],[]
for gpa in np.arange(0,5,.1):
  for act in np.arange(0,36,1):
    test_out = sigmoid(np.dot([gpa/5,act/36,np.random.randint(2016,2019)/2016,0,1], weights))
    predictions = test_out > 0.5
    if(predictions):
      accepted.append([act,gpa])
    else:
      rejected.append([act,gpa])

plt.scatter([p[0] for p in accepted],[p[1] for p in accepted], s = 25, marker="s",c="green")
plt.scatter([p[0] for p in rejected],[p[1] for p in rejected], s = 25, marker="s",c="red")
plt.title(college + " Predictions")
plt.xlabel("ACT")
plt.ylabel("GPA")
plt.show()


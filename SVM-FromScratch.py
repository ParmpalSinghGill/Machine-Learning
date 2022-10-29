from math import exp
from math import log
from math import floor

# Return columnwise max-min statistics for scaling
def statistics(x):
    cols = list(zip(*x))
    stats = []
    for e in cols:
        stats.append([min(e),max(e)])
    return stats

# Scale the features
def scale(x, stat):
    for row in x:
        for i in range(len(row)):
            row[i] = (row[i] - stat[i][0])/(stat[i][1] - stat[i][0])

# Convert different classes into different columns to implement one v/s all
def one_vs_all_cols(s):
    m = list(set(s))
    m.sort()
    s1 = []
    for i in range(len(s)):
        new = [0]*len(m)
        new[m.index(s[i])] = 1
        s1.append(new)
    return m,s1

# Compute Theta transpose x Feature Vector
def ThetaTX(Q,X):
    det = 0.0
    for i in range(len(Q)):
        det += X[i]*Q[i]
    return det

# Calculating cost for negative class (Label = 0)
def LinearSVM_cost0(z):
    if(z < -1): #Ensuring margin
        return 0
    return z + 1

# Calculating cost for positive class (Label = 1)
def LinearSVM_cost1(z):
    if(z > 1): #Ensuring margin
        return 0
    return -z + 1

# Sigmoid Function
def sigmoid(z):
    return 1.0/(1.0 + exp(-z))

# SVM cost Calculation
def cost(theta,c,x,y):
    cost = 0.0
    for i in range(len(x)):
        z = ThetaTX(theta[c], x[i])
        cost += y[i]*LinearSVM_cost1(z) + (1 - y[i])*LinearSVM_cost0(z)
    return cost

# Apply Gradient Descent on the weights or parameters
def gradDescent(theta,c,x,y,learning_rate):
    oldTheta = theta[c]
    for Q in range(len(theta[c])):
        derivative_sum = 0 
        for i in range(len(x)):
            derivative_sum += (sigmoid(ThetaTX(oldTheta,x[i])) - y[i])*x[i][Q]
        theta[c][Q] -= learning_rate*derivative_sum


#Function to Train Model
def train_model(data_train,label_train,learning_rate,epoch):
    print("No. of Iterations : ",epoch)
    x_train, y_train = data_train,label_train
    #Converting y_train to classwise columns with 0/1 values
    classes = []
    for i in range(len(label_map)):
        classes.append([row[i] for row in y_train])
    #Initialising Theta (Weights)
    theta = [[0]*len(x_train[0]) for _ in range(len(classes))]
    #Training the model
    for i in range(epoch):
        for class_type in range(len(classes)):
            gradDescent(theta,class_type,x_train,classes[class_type],learning_rate)
    return theta

# Obtain predictions using trained weights
def predict(data,theta):
    predictions = []
    count = 1
    for row in data:
        hypothesis = []
        multiclass_ans = [0]*len(theta)
        for c in range(len(theta)):
            z = ThetaTX(row,theta[c])
            hypothesis.append(sigmoid(z))
        index = hypothesis.index(max(hypothesis))
        multiclass_ans[index] = 1
        predictions.append(multiclass_ans)
        count+=1
    return predictions

# Calculating accuracy
def accuracy(predicted, actual):
    n = len(predicted)
    correct = 0
    for i in range(n):
        if(predicted[i]==actual[i]):
            correct+=1
    return correct/n

# Function to Test the Model with obtained Weights
def test_model(data_test,label_test,weights):
    accuracies = []
    #Predicting using test data
    y_pred = predict(data_test,weights)
    #Calculating accuracy
    accuracies.append(accuracy(y_pred,label_test))
    return sum(accuracies)/len(accuracies)


# Imports
import csv
import numpy as np

# Reading Training and Testing Data
csv_file_train = open('train.csv','r')
csv_file_test = open('test.csv','r')
csv_read_train = csv.reader(csv_file_train)
csv_read_test = csv.reader(csv_file_test)
csv_train = np.array(list(csv_read_train))
csv_test = np.array(list(csv_read_test))
data_train = csv_train[:,1:].astype(float)
data_test = csv_test[:,1:].astype(float)
label_train = csv_train[:,0].astype(float)
label_test = csv_test[:,0].astype(float)

#Feature Scaling by using column wise max, min stats
stats = statistics(data_train)
scale(data_train,stats)

#Converting different labels to columns
#label_map can be used later to retrieve the predicted class label in the original form (string format)
label_map,label_train = one_vs_all_cols(label_train)
label_map,label_test = one_vs_all_cols(label_test)

learning_rate = 0.01

#increase No. Of Epoches to improve accuracy (Computation Expansive)
epoch = 5
weights = train_model(data_train,label_train,learning_rate,epoch)

final_score = test_model(data_test,label_test,weights)

#Printing Final Stats
print("\nReport")
print("Model used: ","Linear SVM using Gradient Descent")
print("Learning rate: ", learning_rate)
print("No. of iterations: ",epoch)
print("Accuracy: ",final_score*100,"%")

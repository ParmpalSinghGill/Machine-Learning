'''
Created on 21-August-2018

@author: Palak Chandra
'''

from math import exp
import csv
import numpy as np
import datetime
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
from libsvm.python import svmutil

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

name = "SciKit-Learn SVM"
print("\n",name)
classifier_LinearSVM = SVC(kernel="linear", C=0.025)
now = datetime.datetime.now()
classifier_LinearSVM = classifier_LinearSVM.fit(data_train,label_train)
prediction_LinearSVM = classifier_LinearSVM.predict(data_test)
after = datetime.datetime.now()
print("Accuracy: ",accuracy_score(label_test,prediction_LinearSVM))
cnf_matrix = confusion_matrix(label_test, prediction_LinearSVM)
print("Confusion Matrix\n",cnf_matrix)
print("Precision: ",precision_score(label_test, prediction_LinearSVM, average='macro'))
print("Training Time: ",after-now)
print("Sensitivity: ",cnf_matrix[0][0]/(cnf_matrix[0][0]+cnf_matrix[1][0]))
print("Specificity: ",cnf_matrix[1][1]/(cnf_matrix[1][1]+cnf_matrix[0][1]))


name = "LibSVM"
print("\n",name)
classifier = svmutil.svm_train(label_train,data_train)
prediction = svmutil.svm_predict(label_test,data_test,classifier)
pred_label = prediction[0]
cnf_matrix = confusion_matrix(label_test, pred_label)
print("Confusion Matrix\n",cnf_matrix)
print("Precision: ",precision_score(label_test, pred_label, average='macro'))
print("Sensitivity: ",cnf_matrix[0][0]/(cnf_matrix[0][0]+cnf_matrix[1][0]))
print("Specificity: ",cnf_matrix[1][1]/(cnf_matrix[1][1]+cnf_matrix[0][1]))

# Starting Custom SVM Classifier

# Return columnwise max-min statistics for scaling
def getStats(x):
    columns = list(zip(*x))
    stats = []
    for i in columns:
        stats.append([min(i),max(i)])
    return stats

# Scale the features
def scaleFeatures(data, stat):
    for row in data:
        for i in range(len(row)):
            row[i] = (row[i] - stat[i][0])/(stat[i][1] - stat[i][0])

# Get Unique Labels
def getUniqueLabels(label):
    label_list = list(set(label))
    label_list.sort()
    s1 = []
    for i in range(len(label)):
        temp = [0]*len(label_list)
        temp[label_list.index(label[i])] = 1
        s1.append(temp)
    return label_list,s1

# Compute Theta transpose x Feature Vector
def calcTheta_TX(Q,X):
    val = 0.0
    for i in range(len(Q)):
        val += X[i]*Q[i]
    return val

# Calculating cost for negative class (Label = 0)
def calcCost_nve(value):
    if(value < -1): #Ensuring margin
        return 0
    return value + 1

# Calculating cost for positive class (Label = 1)
def calcCost_pve(value):
    if(value > 1): #Ensuring margin
        return 0
    return -value + 1

# Sigmoid Function
def calcSigmoid(value):
    return 1.0/(1.0 + exp(-value))

# Apply Gradient Descent on the weights or parameters
def calcGradientDescent(theta,c,x,y,learning_rate):
    oldTheta = theta[c]
    for Q in range(len(theta[c])):
        derivative_sum = 0 
        for i in range(len(x)):
            derivative_sum += (calcSigmoid(calcTheta_TX(oldTheta,x[i])) - y[i])*x[i][Q]
        theta[c][Q] -= learning_rate*derivative_sum


#Function to Train Model
def train_model(data_train,label_train,learning_rate,epoch):
    print("No. of Iterations : ",epoch)
    #Converting y_train to classwise columns with 0/1 values
    classes = []
    for i in range(len(label_map)):
        classes.append([row[i] for row in label_train])
    #Initialising Theta (Weights)
    theta = [[0]*len(data_train[0]) for _ in range(len(classes))]
    #Training the model
    for i in range(epoch):
        for class_type in range(len(classes)):
            calcGradientDescent(theta,class_type,data_train,classes[class_type],learning_rate)
    return theta

# Obtain predictions using trained weights
def predict(data,theta):
    predictions = []
    count = 1
    for row in data:
        h0 = []
        clf = [0]*len(theta)
        for c in range(len(theta)):
            z = calcTheta_TX(row,theta[c])
            h0.append(calcSigmoid(z))
        i = h0.index(max(h0))
        clf[i] = 1
        predictions.append(clf)
        count+=1
    return predictions

# Function to Test the Model with obtained Weights
def test_model(data_test,label_test,weights):
    accuracies = []
    #Predicting using test data
    y_pred = np.array(predict(data_test,weights))[:,0]
    #Calculating accuracy
    accuracies.append(accuracy_score(y_pred,np.array(label_test)[:,0]))
    return sum(accuracies)/len(accuracies),y_pred


name = "Linear SVM using Gradient Descent"
print("\n",name)
 
#Feature Scaling by using column wise max, min stats
stats = getStats(data_train)
scaleFeatures(data_train,stats)
 
#Converting different labels to columns
#label_map can be used later to retrieve the predicted class label in the original form (string format)
label_map,label_train = getUniqueLabels(label_train)
label_map,label_test = getUniqueLabels(label_test)
 
learning_rate = 0.01
 
#increase No. Of Epoches to improve accuracy (Computation Expansive)
epoch = 1
# Begin Training of SVM Classifier
weights = train_model(data_train,label_train,learning_rate,epoch)
#Testing of SVM Classifier
final_score,y_pred = test_model(data_test,label_test,weights)
  
label_test = np.array(label_test)[:,0]

#Printing Final Stats
print("Learning rate: ", learning_rate)
print("No. of iterations: ",epoch)
print("Accuracy: ",final_score*100,"%")
cnf_matrix = confusion_matrix(label_test, y_pred)
print("Confusion Matrix\n",cnf_matrix)
print("Precision: ",precision_score(label_test, y_pred, average='macro'))
print("Sensitivity: ",cnf_matrix[0][0]/(cnf_matrix[0][0]+cnf_matrix[1][0]))
print("Specificity: ",cnf_matrix[1][1]/(cnf_matrix[1][1]+cnf_matrix[0][1]))
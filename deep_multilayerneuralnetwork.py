# -*- coding: utf-8 -*-
"""

author: Skywalker(@alojoecee)
"""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork():
    def __init__(self, num_features, num_targetclass, hidden_neurons, X_train, hl):
        self.num_features = num_features
        self.num_targetclass = num_targetclass
        self.hidden_neurons = hidden_neurons
        #number of hidden layers
        self.hl = hl
        #weights for input layer
        self.w1 = np.random.randn(self.num_features, self.hidden_neurons)
        #weights for output layer
        self.w2 = np.random.randn(self.hidden_neurons, self.num_targetclass)
        self.X_train = X_train
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    def sigmoid_derivative(self, sigmoid):
        return sigmoid*(1-sigmoid)
    def relu(self, z):
       return z * (z > 0)
    def relu_derivative(self, z):
      return 1 * (z > 0)
    def forward(self):
        self.z1 = np.dot(self.X_train, self.w1)
        self.a1 = self.sigmoid(self.z1)
        if self.hl > 1:
            self.z_list = list()
            self.w = list() #consecutive list of weights for hidden layers 
            self.a = list() #consecutive list of activated inputs for hidden layers
            for i in range(self.hl):
                self.w.append(np.random.randn(self.hidden_neurons, self.hidden_neurons))
                self.z_list.append(np.dot(self.a1, self.w[i]))
                self.a.append(self.relu(self.z_list[i]))
        self.z2 = np.dot(self.a1, self.w2)
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    def train(self, y_train, iterations):
        y_list = list()
        for i in range(iterations):
            self.output = self.forward()
            self.error = y_train - self.output
            print("*Predicted output \n", self.output)
            cost = np.mean(np.square(self.error))
            print("Cost function \n", cost)
            print("Weights for input layer \n", self.w1)
            print("Weights for hidden layers \n", self.w)
            print("Weights for output layer \n", self.w2)
            self.d1 = self.error * self.sigmoid_derivative(self.output)
            self.error2 = np.dot(self.d1, self.w2.T)
            if self.hl > 1:
                self.errors = list()
                self.d = list()
                for n in range(self.hl):
                    self.d.append(self.error2*self.relu_derivative(self.a[self.hl-1]))
                    self.errors.append(np.dot(self.d[n], self.w[self.hl-1]))
            self.d2 = self.errors[self.hl-1]*self.sigmoid_derivative(self.a1)     
            self.w1 += np.dot(self.X_train.T, self.d2)
            self.w2 += np.dot(self.a1.T, self.d1)
            for m in range(self.hl):
                self.w[m] += np.dot(self.a[m].T, self.d[self.hl-1])
            y_list.append(cost)
        #gradient descent plot graph
        x = [a for a in range(iterations)]
        y = y_list
        plt.plot(x, y)
        plt.show()

if __name__ == "__main__":
    num_of_feat = int(input("Enter number of features in training set: "))
    num_of_tc = int(input("Enter number of target class: "))
    num_of_hn = int(input("Enter number of hidden neurons: "))
    num_of_iter = int(input("Enter number of iterations: "))
    num_of_hl = int(input("Enter number of hidden layers"))

    #Import X_train and y_train datasets before use
    NeuralNetwork(num_of_feat, num_of_tc, num_of_hn, X_train, num_of_hl).train(y_train, num_of_iter)

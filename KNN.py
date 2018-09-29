
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 19:53:20 2018

author: Skywalker(@alojoecee)
"""

import math
	
																																			
class KNN():
	def __init__(self, X_train, X_datapoint, y_train, feat, k):
		self.X_train = X_train
		self.X_datapoint = X_datapoint
		self.y_train = y_train
		self.feat = feat
		self.k = k																																																																				
	def Euclidean(self):
		self.e_list = list()
		for row_train in self.X_train:
			self.distance = 0
			for index in range(self.feat):																				
				self.distance += math.sqrt((row_train[index] - self.X_datapoint[index])**2)
			self.e_list.append(self.distance)
	def Regression(self):
		self.Euclidean()
		self.e_dict = dict(zip(self.e_list, self.y_train))
		self.e_dict = sorted(self.e_dict)
		self.predict = 0
		self.counter = 0
		for i in self.e_dict:
			self.counter += 1
			self.predict += i
			if self.counter == self.k:
				break
		self.predict = self.predict/self.k
		print(self.predict)
		
		

		
		
#X_datapoint - array of features of the datapoint to be predicted 
#X_train - training datasets
#y_train - target of training set
#feat - number of features
#k - k parameter
		
		
		
		
		
		

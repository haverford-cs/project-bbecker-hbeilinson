"""
This class provides methods for perturbing features based on
feature importance.
Authors: Brian Becker + Hannah Beilinson
Date: 12/16/19
"""

import random

class Perturber:

	def __init__(self, ft_imp,X,y,verbose=0):
		self.X = X
		self.y = y
		tuples = []
		self.verbose = verbose
		for i in range(len(ft_imp)):
			tuples.append((i,ft_imp[i]))

		tuples.sort(key = lambda x : -x[1])
		self.feature_imp = tuples

		if(verbose>0):
			print(ft_imp)
			print(self.feature_imp)
		


	def perturb(self,num_features=1,change=1.05):
		if(self.verbose>0):
			print(self.X)
		for i in range(num_features):
			ftr = self.feature_imp[i][0]
			self.X[:,ftr]*=change
		if(self.verbose>0):
			print(self.X)

		return self.X



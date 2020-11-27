'''
Task B2 is to perform eye colour recognition from 5 eye colours.
The dataset used is the cartoon_set dataset.
'''
#import face_features
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
from sklearn import svm
#TODO
class B2():
	classifier=None
	global tr_X, tr_Y, te_X, te_Y
	def __init__(self, args):
		print("Initialized B2")
		self.__getData()
	
	def __getData(self):
		print("Getting Data")
		'''
		X, y = face_features.extract_features_labels("./Datasets/cartoon_set","img")
		print("Got Data")
		Y = np.array([y, -(y-1)]).T
		self.tr_X = X[:100]
		self.tr_Y = Y[:100]
		self.te_X = X[100:]
		self.te_Y = Y[100:]
		
		return tr_X, tr_Y, te_X, te_Y
		'''
	
	def train(self, args):
		'''
		self.__getData()
		print("Training")
		self.classifier=svm.SVC()
		self.classifier.fit(self.tr_X.reshape((100, 68*2)), list(zip(*self.tr_Y))[0])
		'''
		print("Trained")
	
	def test(self,args):
		print("Testing")
		'''
		pred=self.classifier.predict(self.te_X.reshape((35, 68*2)))
		print("Accuracy:", accuracy_score(list(zip(*self.te_Y))[0],pred))
		'''
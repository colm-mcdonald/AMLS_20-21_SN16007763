'''
Task B1 is to perform face shape recognition from 5 face shapes.
The dataset used is the cartoon_set dataset.
'''
#TODO
from B1 import face_features_shape as face_features
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import os


class B1():
	classifier=None
	global tr_X, tr_Y, te_X, te_Y
	def __init__(self):
		print("Initializing B1")
		self.__getData()
		#print("Initialized B1")
	
	def __getData(self):
		print("Getting Data")
		if(not os.path.isfile('b1_input.npy') or not os.path.isfile('b1_output.npy')):
			X, y = face_features.extract_features_labels("./Datasets/cartoon_set","img")
			Y = np.array([y, -(y-1)]).T
			np.save('b1_input',X)
			np.save('b1_output',Y)
		else:
			X=np.load('b1_input.npy')
			Y=np.load('b1_output.npy')

		'''
		#0.02:0.98 Training:Testing
		self.tr_X = X[:100]
		self.tr_Y = Y[:100]
		self.te_X = X[100:]
		self.te_Y = Y[100:]
		'''

		print("Number of detections:",len(X))
		
		#0.7:0.3 Training:Testing
		self.tr_X = X[:7000]
		self.tr_Y = Y[:7000]
		self.te_X = X[7000:]
		self.te_Y = Y[7000:]
		
		'''
		self.tr_X = X[:400]
		self.tr_Y = Y[:400]
		self.te_X = X[400:]
		self.te_Y = Y[400:]
		'''
	
	def train(self, args):
		"""
		This function trains the SVM
		"""

		#self.classifier=svm.SVC(kernel='linear',C=1.0,random_state=1) # Training 7,000=69.7%
		#self.classifier=svm.SVC(kernel='poly',degree=2,C=1.0,random_state=1) # Training 7,000=68.6%
		#self.classifier=svm.SVC(kernel='poly',degree=3,C=1.0,random_state=1) # Training 7,000=71.8%
		self.classifier=svm.SVC(kernel='poly',degree=4,C=1.0,random_state=1) # Training 7,000=72.0% #Best
		#self.classifier=svm.SVC(kernel='poly',degree=5,C=1.0,random_state=1) # Training 7,000=70.8%
		#self.classifier=svm.SVC(kernel='poly',degree=6,C=1.0,random_state=1) # Training 7,000=70.0%
		#self.classifier=svm.LinearSVC(C=1.0,random_state=1) # Training 3,500=53.3%
		#self.classifier=svm.SVC(kernel='rbf', gamma=0.7,C=1.0,random_state=1) # Training 7,000=22.3%

		#self.classifier=MLPClassifier(hidden_layer_sizes=(100,), random_state=1) # Training 7,000=17.5%
		#self.classifier=MLPClassifier(hidden_layer_sizes=(90,), random_state=1)  # Training 7,000=17.1%
		#self.classifier=MLPClassifier(hidden_layer_sizes=(85,), random_state=1)  # Training 7,000=22.3%
		#self.classifier=MLPClassifier(hidden_layer_sizes=(70,), random_state=1)  # Training 7,000=20.3%
		#self.classifier=MLPClassifier(hidden_layer_sizes=(5,), random_state=1)  # Training 7,000=22.3%

		#self.classifier=KNeighborsClassifier(n_neighbors=3)  # Training 7,000=49.0%

		self.classifier.fit(self.tr_X.reshape((len(self.tr_X), 68*2)), list(zip(*self.tr_Y))[0])
		#print("Trained with hidden layer size of", args)
		pred=self.classifier.predict(self.tr_X.reshape((len(self.tr_X), 68*2)))
		return accuracy_score(list(zip(*self.tr_Y))[0],pred)
	
	def test(self):
		print("Testing")
		pred=self.classifier.predict(self.te_X.reshape((len(self.te_X), 68*2)))
		accuracy=accuracy_score(list(zip(*self.te_Y))[0],pred)
		print("Accuracy:",accuracy)
		return accuracy
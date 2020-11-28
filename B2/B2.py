'''
Task B2 is to perform eye colour recognition from 5 eye colours.
The dataset used is the cartoon_set dataset.
'''
#TODO
from B2 import face_features_eye as face_features
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import match_template
from skimage import io
import os


class B2():
	classifier=None
	global tr_X, tr_Y, te_X, te_Y
	def __init__(self):
		print("Initializing B2")
		self.__getData()
		#print("Initialized B2")
	
	def __getData(self):
		print("Getting Data")
		if(not os.path.isfile('b2_input.npy') or not os.path.isfile('b2_output.npy')):
			X, y = face_features.extract_features_labels("./Datasets/cartoon_set","img")
			Y = np.array([y, -(y-1)]).T
			np.save('b2_input',X)
			np.save('b2_output',Y)
		else:
			X=np.load('b2_input.npy')
			Y=np.load('b2_output.npy')

		'''
		#0.02:0.98 Training:Testing
		self.tr_X = X[:100]
		self.tr_Y = Y[:100]
		self.te_X = X[100:]
		self.te_Y = Y[100:]
		'''

		print("Number of detections:",len(X))
		
		#0.7:0.3 Training:Testing
		
		self.tr_X = X[:5700]
		self.tr_Y = Y[:5700]
		self.te_X = X[5700:]
		self.te_Y = Y[5700:]
		
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
		print("Training")
		#self.classifier=svm.SVC(kernel='linear',C=1.0,random_state=1) # Training 5,700=85.9%
		#self.classifier=svm.SVC(kernel='poly',degree=2,C=1.0,random_state=1) # Training 5,700=85.8%
		#self.classifier=svm.SVC(kernel='poly',degree=3,C=1.0,random_state=1) # Training 5,700=83.5%
		#self.classifier=svm.SVC(kernel='poly',degree=4,C=1.0,random_state=1) # Training 5,700=80.8%
		#self.classifier=svm.SVC(kernel='poly',degree=5,C=1.0,random_state=1) # Training 5,700=78.4%
		#self.classifier=svm.SVC(kernel='poly',degree=6,C=1.0,random_state=1) # Training 5,700=74.7%
		#self.classifier=svm.LinearSVC(C=1.0,random_state=1) # Training 5,700=88.3%
		#self.classifier=svm.SVC(kernel='rbf', gamma=0.7,C=1.0,random_state=1) # Training 5,700=18.8%

		#self.classifier=MLPClassifier(hidden_layer_sizes=(100,), random_state=1) # Training 5,700=88.1%
		#self.classifier=MLPClassifier(hidden_layer_sizes=(90,), random_state=1)  # Training 5,700=88.1%
		self.classifier=MLPClassifier(hidden_layer_sizes=(85,), random_state=1)  # Training 5,700=88.7% #Best
		#self.classifier=MLPClassifier(hidden_layer_sizes=(70,), random_state=1)  # Training 5,700=88.4%
		#self.classifier=MLPClassifier(hidden_layer_sizes=(5,), random_state=1)  # Training 5,700=87.9%

		#self.classifier=KNeighborsClassifier(n_neighbors=3)  # Training 5,700=87.9%

		self.classifier.fit(self.tr_X.reshape((len(self.tr_X), 150)), list(zip(*self.tr_Y))[0])
		#print("Trained with hidden layer size of", args)
		pred=self.classifier.predict(self.tr_X.reshape((len(self.tr_X), 150)))
		return accuracy_score(list(zip(*self.tr_Y))[0],pred)
	
	def test(self):
		print("Testing")
		pred=self.classifier.predict(self.te_X.reshape((len(self.te_X), 150)))
		accuracy=accuracy_score(list(zip(*self.te_Y))[0],pred)
		print("Accuracy:",accuracy)
		return accuracy
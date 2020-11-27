'''
Task A2 is to perform basic emotion detection from the choices of smiling or not smiling.
The dataset used is the celeba dataset.
'''
from A2 import face_features_smiling as face_features
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

class A2():
	classifier=None
	global tr_X, tr_Y, te_X, te_Y
	def __init__(self):
		print("Initializing A2")
		self.__getData()
		#print("Initialized A2")
	
	def __getData(self):
		print("Getting Data")
		X, y = face_features.extract_features_labels("./Datasets/celeba","img")
		#print("Got Data")
		Y = np.array([y, -(y-1)]).T

		'''
		#0.02:0.98 Training:Testing
		self.tr_X = X[:100]
		self.tr_Y = Y[:100]
		self.te_X = X[100:]
		self.te_Y = Y[100:]
		'''
		
		#0.7:0.3 Training:Testing
		self.tr_X = X[:3500]
		self.tr_Y = Y[:3500]
		self.te_X = X[3500:]
		self.te_Y = Y[3500:]
	
	def train(self, args):
		"""
		This function trains the SVM
		"""
		print("Training")
		#self.classifier=svm.SVC(kernel='linear',C=1.0,random_state=1) # Training 3,500=87.6%
		#self.classifier=svm.SVC(kernel='poly',degree=2,C=1.0,random_state=1) # Training 3,500=87.8%
		self.classifier=svm.SVC(kernel='poly',degree=3,C=1.0,random_state=1) # Training 3,500=88.1% #Best
		#self.classifier=svm.SVC(kernel='poly',degree=4,C=1.0,random_state=1) # Training 3,500=88.0%
		#self.classifier=svm.SVC(kernel='poly',degree=5,C=1.0,random_state=1) # Training 3,500=88.1%
		#self.classifier=svm.SVC(kernel='poly',degree=6,C=1.0,random_state=1) # Training 3,500=86.2%
		#self.classifier=svm.LinearSVC(C=1.0,random_state=1) # Training 3,500=76.0%
		#self.classifier=svm.SVC(kernel='rbf', gamma=0.7,C=1.0,random_state=1) # Training 3,500=42.1%

		#self.classifier=MLPClassifier(hidden_layer_sizes=(100,), random_state=1) # Training 3,500=85.2%
		#self.classifier=MLPClassifier(hidden_layer_sizes=(90,), random_state=1)  # Training 3,500=84.6%
		#self.classifier=MLPClassifier(hidden_layer_sizes=(85,), random_state=1)  # Training 3,500=84.7%
		#self.classifier=MLPClassifier(hidden_layer_sizes=(70,), random_state=1)  # Training 3,500=87.0%
		#self.classifier=MLPClassifier(hidden_layer_sizes=(5,), random_state=1)  # Training 3,500=83.2%

		#self.classifier=KNeighborsClassifier(n_neighbors=3)  # Training 3,500=82.5%

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
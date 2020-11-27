'''
Task A1 is to perform basic gender detection from the choices of Male or Female.
The dataset used is the celeba dataset.
'''
from A1 import face_features_gender as face_features
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import os

class A1():
	classifier=None
	global tr_X, tr_Y, te_X, te_Y
	def __init__(self):
		print("Initializing A1")
		self.__getData()
		#print("Initialized A1")
	
	def __getData(self):
		print("Getting Data")
		if(not os.path.isfile('a1_input.npy') or not os.path.isfile('a1_output.npy')):
			X, y = face_features.extract_features_labels("./Datasets/celeba","img")
			Y = np.array([y, -(y-1)]).T
			np.save('a1_input',X)
			np.save('a1_output',Y)
		else:
			X=np.load('a1_input.npy')
			Y=np.load('a1_output.npy')

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
		#self.classifier=svm.SVC(kernel='linear',C=1.0) #Training 100=81.8%, Training 3,500=91.1%
		#self.classifier=svm.SVC(kernel='poly',degree=2,C=1.0) #Training 100=-----, Training 3,500=87.4%
		#self.classifier=svm.SVC(kernel='poly',degree=3,C=1.0) #Training 100=81.6%, Training 3,500=90.9%
		self.classifier=svm.SVC(kernel='poly',degree=4,C=1.0) #Training 100=-----, Training 3,500=91.8% #Best
		#self.classifier=svm.SVC(kernel='poly',degree=5,C=1.0) #Training 100=-----, Training 3,500=90.9%
		#self.classifier=svm.SVC(kernel='poly',degree=6,C=1.0) #Training 100=-----, Training 3,500=89.4%
		#self.classifier=svm.LinearSVC(C=1.0) #Training 100=75.5%, Training 3,500=76.3%
		#self.classifier=svm.SVC(kernel='rbf', gamma=0.7,C=1.0) #Training 100=50.2%, Training 3,500=27.8%

		#self.classifier=MLPClassifier(hidden_layer_sizes=(100,), random_state=1) # Training 3,500=70.5%
		#self.classifier=MLPClassifier(hidden_layer_sizes=(85,), random_state=1)  # Training 3,500=50.6%
		#self.classifier=MLPClassifier(hidden_layer_sizes=(90,), random_state=1)  # Training 3,500=73.6%
		#self.classifier=MLPClassifier(hidden_layer_sizes=(5,), random_state=1)  # Training 3,500=87.5% when using hidden layer size of 5

		#self.classifier=KNeighborsClassifier(n_neighbors=3)  # Training 3,500=69.6%

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
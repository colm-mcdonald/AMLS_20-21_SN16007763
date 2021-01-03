'''
Task B1 is to perform face shape recognition from 5 face shapes.
The dataset used is the cartoon_set dataset.
'''
from B1 import face_features_shape as face_features
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
import os


class B1():
	classifier=None
	global tr_X, tr_Y, te_X
	global full_X, full_Y
	def __init__(self):
		print("Initializing B1")
		self.__getData()
	
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

		#0.7:0.3 Training:Testing
		self.full_X=X
		self.full_Y=Y
		self.tr_X, self.te_X, self.tr_Y, self.te_Y = train_test_split(X,Y,test_size=0.3,random_state=1)
	
	def train(self, args):
		"""
		This function trains the SVM
		"""
		print("Training")
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
	
	def training_curve(self):
		X=self.full_X
		Y=self.full_Y
		print("Creating Training Curve...")
		train_sizes, train_scores, valid_scores = learning_curve(
			svm.SVC(kernel='poly',degree=4,C=1.0, random_state=1),X.reshape((len(X), 68*2)), list(zip(*Y))[0],
			train_sizes=[np.linspace(0.1,1,6)])
		print("Train Sizes:")
		print(train_sizes)

		print("Train Scores:")
		print(train_scores)

		print("Validation Scores:")
		print(valid_scores)
	
	def test(self):
		print("Testing")
		pred=self.classifier.predict(self.te_X.reshape((len(self.te_X), 68*2)))
		accuracy=accuracy_score(list(zip(*self.te_Y))[0],pred)
		print("Verification Accuracy:",accuracy)
		return accuracy

	def final_test(self):
		X, y = face_features.extract_features_labels("./Datasets/cartoon_set_test","img")
		Y = np.array([y, -(y-1)]).T

		pred=self.classifier.predict(X.reshape((len(X),68*2)))
		accuracy=accuracy_score(list(zip(*Y))[0],pred)
		print("Test Accuracy:",accuracy)
		return accuracy
'''
Task B2 is to perform eye colour recognition from 5 eye colours.
The dataset used is the cartoon_set dataset.
'''
from B2 import face_features_eye as face_features
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import match_template
from skimage import io
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
import os


class B2():
	classifier=None
	global tr_X, tr_Y, te_X, te_Y
	global full_X, full_Y
	def __init__(self):
		print("Initializing B2")
		self.__getData()
	
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
		
		#0.7:0.3 Training:Testing
		self.full_X=X
		self.full_Y=Y
		self.tr_X, self.te_X, self.tr_Y, self.te_Y = train_test_split(X,Y,test_size=0.5,random_state=1)
		
	
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
		pred=self.classifier.predict(self.tr_X.reshape((len(self.tr_X), 150)))
		return accuracy_score(list(zip(*self.tr_Y))[0],pred)
	
	def training_curve(self):
		X=self.full_X
		Y=self.full_Y
		print("Creating Training Curve...")
		train_sizes, train_scores, valid_scores = learning_curve(
			MLPClassifier(hidden_layer_sizes=(85,),random_state=1),X.reshape((len(X), 150)), list(zip(*Y))[0],
			train_sizes=[np.linspace(0.1,1,6)])
		print("Train Sizes:")
		print(train_sizes)

		print("Train Scores:")
		print(train_scores)

		print("Validation Scores:")
		print(valid_scores)
	
	def test(self):
		print("Testing")
		pred=self.classifier.predict(self.te_X.reshape((len(self.te_X), 150)))
		accuracy=accuracy_score(list(zip(*self.te_Y))[0],pred)
		print("Verification Accuracy:",accuracy)

		return accuracy

	def final_test(self):
		X, y = face_features.extract_features_labels("./Datasets/cartoon_set_test","img")
		Y = np.array([y, -(y-1)]).T

		pred=self.classifier.predict(X.reshape((len(X),150)))
		accuracy=accuracy_score(list(zip(*Y))[0],pred)
		print("Test Accuracy:",accuracy)
		return accuracy
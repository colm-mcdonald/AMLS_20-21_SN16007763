from A1.A1 import A1
from A2.A2 import A2
from B1.B1 import B1
from B2.B2 import B2

#TODO
def data_preprocessing(args):
	print("Data Pre-Processing")
	data_train="Data Train"
	data_val="Data Value"
	data_test="Data Test"
	return data_train, data_val, data_test

# ======================================================================================================================
# Data preprocessing
#TODO
data_train, data_val, data_test = data_preprocessing("args...")
# ======================================================================================================================
# Task A1
acc_A1_train=None
acc_A1_test=None

#model_A1 = A1()                 # Build model object.
'''
best_accuracy=0
for i in range(1,20):
	print(i)
	acc_A1_train = model_A1.train(i) # Train model based on the training set (you should fine-tune your model based on validation set.)
	acc_A1_test = model_A1.test()   # Test model based on the test set.
	if(acc_A1_test>best_accuracy):
		best_accuracy=acc_A1_test
		best_param=i
print("Best parameter is",best_param,"with an accuracy of",best_accuracy*100)
'''

#acc_A1_train = model_A1.train(None)
#acc_A1_test = model_A1.test()

#Clean up memory/GPU etc...             # Some code to free memory if necessary.


# ======================================================================================================================
# Task A2
#TODO
acc_A2_train=None
acc_A2_test=None

#model_A2 = A2()
'''
best_accuracy=0
for i in range(1,9):
	acc_A2_train = model_A2.train(i)
	acc_A2_test = model_A2.test()
	if(acc_A2_test>best_accuracy):
		best_accuracy=acc_A2_test
		best_param=i
print("Best parameter is",best_param,"with an accuracy of",best_accuracy*100)
'''

#acc_A2_train = model_A2.train(None)
#acc_A2_test = model_A2.test()
#Clean up memory/GPU etc...


# ======================================================================================================================
# Task B1
#TODO

acc_B1_train=None
acc_B1_test=None

#model_B1 = B1()
'''
best_accuracy=0
for i in range(9,15):
	acc_B1_train = model_B1.train(i)
	acc_B1_test = model_B1.test()
	if(acc_B1_test>best_accuracy):
		best_accuracy=acc_B1_test
		best_param=i
print("Best parameter is",best_param,"with an accuracy of",best_accuracy*100)
'''
#acc_B1_train = model_B1.train(None)
#acc_B1_test = model_B1.test()

#Clean up memory/GPU etc...


# ======================================================================================================================
# Task B2
#TODO
acc_B2_train = None
acc_B2_test = None

model_B2 = B2()
acc_B2_train = model_B2.train(None)
acc_B2_test = model_B2.test()

#Clean up memory/GPU etc...


# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))

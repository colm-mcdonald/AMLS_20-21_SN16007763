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
#TODO
model_A1 = A1("args...")                 # Build model object.
acc_A1_train = model_A1.train("args...") # Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A1_test = model_A1.test("args...")   # Test model based on the test set.
#Clean up memory/GPU etc...             # Some code to free memory if necessary.


# ======================================================================================================================
# Task A2
#TODO
model_A2 = A2("args...")
acc_A2_train = model_A2.train("args...")
acc_A2_test = model_A2.test("args...")
#Clean up memory/GPU etc...


# ======================================================================================================================
# Task B1
#TODO
model_B1 = B1("args...")
acc_B1_train = model_B1.train("args...")
acc_B1_test = model_B1.test("args...")
#Clean up memory/GPU etc...


# ======================================================================================================================
# Task B2
#TODO
model_B2 = B2("args...")
acc_B2_train = model_B2.train("args...")
acc_B2_test = model_B2.test("args...")
#Clean up memory/GPU etc...


# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))

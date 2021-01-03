# README

# Brief description of the organization of project
This is the code for the AMLS project
# Role of each file
A1/A1.py accomplishes the task of A1

A2/A2.py accomplishes the task of A2

B1/B1.py accomplishes the task of B1

B1/B2.py accomplishes the task of B2

The files beginning face_features_X.py are used as part of the preprocessing of the data.

The .npy files are the cached data so it doesn't have to be preprocessed again. If you want to run preprocessing again, just delete these files.

shape_predictor_68_face_landmarks.dat is a pretrained model used by dlib to find face landmarks

main.py calls all of the modules and runs the training and testing

# Packages required to run code
numpy scipy keras opencv dlib sklearn skimage

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from mlxtend.classifier import StackingClassifier,StackingCVClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB, CategoricalNB, ComplementNB
from sklearn.svm import SVC,NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from Bio.SeqUtils.ProtParam import *
import random
import numpy
import warnings
import time

# This code gives us an accuracy of 79.1% in public score and 78.5% in private score. There is little randomness in the output everytime because we have used randomforest with bagging.

# Use the following command in terminal to run this file after installing all the libraries imported in this code :- python code.py #

# This command ignores all the warnings and doesn't unnecessarily print them to the output console #
warnings.filterwarnings('ignore')

# A Dictionary to map the characters in amino acid sequences characters to numerical values #
d = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,'Z':25}


# loading the training and testing datasets #
dataset1 = pd.read_csv('train.csv')
dataset2 = pd.read_csv('test.csv')


### preprocessing the dataset to extract features from the dataframe and storing in numpy arrays ###
### xtrain is a 2-D numpy array. xtrain[i] is a 1-D numpy array containing the ith amino acid sequence string in the training dataset
### ytrain is a 2-D numpy array. ytrain[i] is a 1-D numpy array containing the ith label in the training dataset
### xtest is a 2-D numpy array. xtest[i] is a 1-D numpy array containing the ith amino acid sequence string in the testing dataset

xtrain = dataset1.iloc[:, [0]].values
ytrain = dataset1.iloc[:, [1]].values
xtest = dataset2.iloc[:, [1]].values


## below 2 for loops are use to check if any character in the amino acid sequences in train.csv or test.csv is not of alphabetic character. We then replace them with the character A
for i in range(len(xtrain)):
	l = xtrain[i]
	s = l[0]
	s1 = ""
	for ch in s:
		if ch not in d:
			s1 += 'A'
		else:
			s1 += ch
	l[0] = s1

for i in range(len(xtest)):
	l = xtest[i]
	s = l[0]
	s1 = ""
	for ch in s:
		if ch not in d:
			s1 += 'A'
		else:
			s1 += ch
	l[0] = s1


## Here we created a temporary ProteinAnalysis object which we imported from the Bio.SeqUtils.ProtParam library
tempobj = ProteinAnalysis(xtrain[0][0])
## This is a list of all 20 characters we can have in a protein sequence
amino_acid_list = sorted(list(tempobj.count_amino_acids().keys()))


## x_train is initialized as an empty array
## x_test is initlialized as an emtpy array

x_train = []
x_test = []

## In the below for loop, we have converted each protein sequence (in xtrain) into its composition features and stored them as arrays inside x_train, in other words, x_train[i] is an array of composition features corresponding to the ith amino acid sequence string in training dataset.
## We have extracted the composition features for a amino acid string using the Bio.SeqUtils.ProtParam library
## We first converted each amino acid sequence string to a ProteinAnalysis object. Then extracted features from that object using various methods given in the library such as length, count_amino_acids, molecular_weight, aromaticity, etc.


for i in range(len(xtrain)):
	l = []
	obj = ProteinAnalysis(xtrain[i][0])
	l.append(obj.length)

	if (obj.monoisotopic == False):
		l.append(0)
	else:
		l.append(1)

	dic = obj.count_amino_acids()
	for key in amino_acid_list:
		l.append(dic[key])
	dic = obj.get_amino_acids_percent()
	for key in amino_acid_list:
		l.append(dic[key])
	l.append(obj.molecular_weight())
	l.append(obj.aromaticity())
	l.append(obj.instability_index())
	l.append(obj.gravy())
	l.append(obj.isoelectric_point())
	l.append(obj.charge_at_pH(7.0))
	for val in obj.secondary_structure_fraction():
		l.append(val)
	for val in obj.molar_extinction_coefficient():
		l.append(val)
	x_train.append(l)


## In the below for loop, we have converted each protein sequence (in xtest) into its composition features and stored them as arrays inside x_test, in other words, x_test[i] is an array of composition features corresponding to the ith amino acid sequence string in testing dataset.
## We have extracted the composition features for a amino acid string using the Bio.SeqUtils.ProtParam library
## We first converted each amino acid sequence string to a ProteinAnalysis object. Then extracted features from that object using various methods given in the library such as length, count_amino_acids, molecular_weight, aromaticity, etc.


for i in range(len(xtest)):
	l = []
	obj = ProteinAnalysis(xtest[i][0])
	l.append(obj.length)

	if (obj.monoisotopic == False):
		l.append(0)
	else:
		l.append(1)

	dic = obj.count_amino_acids()
	for key in amino_acid_list:
		l.append(dic[key])
	dic = obj.get_amino_acids_percent()
	for key in amino_acid_list:
		l.append(dic[key])
	l.append(obj.molecular_weight())
	l.append(obj.aromaticity())
	l.append(obj.instability_index())
	l.append(obj.gravy())
	l.append(obj.isoelectric_point())
	l.append(obj.charge_at_pH(7.0))
	for val in obj.secondary_structure_fraction():
		l.append(val)
	for val in obj.molar_extinction_coefficient():
		l.append(val)
	x_test.append(l)


## converting x_train and x_test into 2-D numpy arrays
x_train = np.array(x_train)
x_test = np.array(x_test)

## reassigning x_train,x_test to xtrain,xtest
xtrain = x_train
xtest = x_test

## raveling ytrain (to convert it to 1-D numpy array)
ytrain = ytrain.ravel()

## using a Baggingclassifier with a base classifier in it as randomforest. We have set the number of estimators in both randomforest and baggingclassifier to 50 for optimal performance
classifier = BaggingClassifier(base_estimator = RandomForestClassifier(n_estimators = 50), n_estimators = 50)

## fitting the training data in the classifier
classifier.fit(xtrain,ytrain)

## getting the prediction probability values by applying the model on testing data
y_pred = classifier.predict_proba(xtest)

## keeping only the Label=1 values in y_pred list
y_pred_new = []
for i in range(len(y_pred)):
	y_pred_new.append(y_pred[i][1])
y_pred = y_pred_new


### This portion is to save the output to a csv file ###
### Here we save the final output (y_pred) to a csv file names "output_file.csv" ###
label = y_pred
ID = []
for i in range(len(label)):
	ID.append(10000+i+1)

dic = {'ID':ID,'Label':label}
df = pd.DataFrame(dic)
df.to_csv("output_file.csv",index=False)
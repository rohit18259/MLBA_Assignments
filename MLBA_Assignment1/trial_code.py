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
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.svm import SVC,NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
import random
import numpy
import warnings

# This code gives us an accuracy of around 65.8% in the private score and 66.25% in public score. There can be a little randomeness in the output of this code everytime as it uses random groups in ensembling #

# Use the following command in terminal to run this file after installing all the libraries imported in this code :- python trial_code.py #

# This command ignores all the warnings and doesn't unnecessarily print them to the output console #
warnings.filterwarnings('ignore')

# A Dictionary to map the characters in DNA sequences to numerical values #
d = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,'Z':25}


# loading the training and testing datasets #
dataset1 = pd.read_csv('train_data.csv')
dataset2 = pd.read_csv('test_data.csv')


### preprocessing the dataset to extract features from the dataframe and storing in numpy arrays ###
xtrain = dataset1.iloc[:, [1]].values
ytrain = dataset1.iloc[:, [0]].values
xtest = dataset2.iloc[:, [1]].values


### preprocessing the data to convert them from alphabetical characters to numerical value by mapping from the dictionary ###
xtrain_new = []
for i in range(len(xtrain)):
	arr = xtrain[i]
	l = []
	s = arr[0]
	for elem in s:
		l.append(d[elem])
	
	l = np.array(l)
	xtrain_new.append(l)

xtrain_new = np.array(xtrain_new)
ytrain_new = ytrain

xtest_new = []
for i in range(len(xtest)):
	arr = xtest[i]
	l = []
	s = arr[0]
	for elem in s:
		l.append(d[elem])
	
	l = np.array(l)
	xtest_new.append(l)

xtest_new = np.array(xtest_new)


### numpy arrays defined which store the numerical data after converting each alphabet of DNA sequences to their mapped value in dictionary ###
### x_train is a 2-D numpy array. x_train[i] is a numpy array of length 17. x_train[i][j] refers to the numerical mapped value of the jth character of the ith DNA sequence in the training data ###
### y_train is a 1-D numpy array. y_train[i] is the ith label in the training data. ###
### x_test is a 2-D numpy array. x_test[i] is a numpy array of length 17. x_test[i][j] refers to the numerical mapped value of the jth character of the ith DNA sequence in the testing data. ###

x_train = xtrain_new
y_train = ytrain_new.ravel()
x_test = xtest_new


def get1HotDataFrame(xlist,ylist,columns_list):
	### xlist is a 2-D numpy array. xlist[i][j] refers to the numerical mapped value of the jth character of the ith DNA sequence ###
	### ylist is a 1-D numpy array. ylist[i] is the ith label of the DNA sequences ###
	### columns_list is a list of names of 17 columns, each referring to numerical value of one alphabet of the DNA sequence ###
	### This function uses xlist, ylist and columns_list to return us a 1 Hot Encoded DataFrame representing the DNA sequences ### 

	df = pd.DataFrame(xlist,columns = columns_list)
	if (type(ylist) == numpy.ndarray):
		df['label'] = ylist

	encoded_df = pd.get_dummies(df,columns = columns_list)
	
	for col in columns_list:
		z = list(df[col])
		for i in range(0,26):
			if i not in z:
				s = col+"_"+str(i)
				encoded_df[s] = np.array([0 for i in range(len(xlist))])
	
	return encoded_df


columns_list = ['col1','col2','col3','col4','col5','col6','col7','col8','col9','col10','col11','col12','col13','col14','col15','col16','col17']

### encoded_df1 is the 1 Hot Encoded DataFrame representing the DNA sequences of the training data ###
encoded_df1 = get1HotDataFrame(x_train,y_train,columns_list)

### encoded_df2 is the 1 Hot Encoded DataFrame representing the DNA sequences of the testing data ###
encoded_df2 = get1HotDataFrame(x_test,'none',columns_list)

### xtrain is 2-D numpy array. It contains the values of input features of encoded_df1 as arrays ###
### ytrain is 1-D numpy array. It contains the values of the labels of encodeed_df1 ###
### xtest is 2-D numpy array. It contains the values of the features of encoded_df2 as arrays ###
xtrain = encoded_df1.drop('label',axis=1).values
ytrain = encoded_df1['label'].values
xtest = encoded_df2.values


### xtrain0 is a 2-D list. It has the elements of xtrain which have a corresponding label 0 in ytrain ###
### xtrain1 is a 2-D list. It has the elements of xtrain which have a corresponding label 1 in ytrain ###
xtrain0 = []
ytrain0 = []
xtrain1 = []
ytrain1 = []
for i in range(len(ytrain)):
	if (ytrain[i]==0):
		xtrain0.append(xtrain[i])
		ytrain0.append(ytrain[i])
	else:
		xtrain1.append(xtrain[i])
		ytrain1.append(ytrain[i])

'''
Below is the ensembling process. We define a variable num_groups. Then we create groups of number num_groups. Within each group,  we have all elements from xtrain1,ytrain1 (which constitue of 1801 elements)
Then we select randomly 1801 elements from xtrain0,ytrain0 and add it to that group. In this way, each of the groups are balanced, i.e; have equal number of label 0 and 1.
After we have selected the training data for a group, we use RandomForestClassifier to create a model for that group. Then on applying this model to the testing set, i.e; x_test, we get
an output called y_pred for that group. This refers to the output labels generated for the testing data. Here we give the output in the form of probabilities instead of 0 or 1.
Finally, we store the prediction output (y_pred) from each group in a list y_pred_list. 
'''
num_groups = 15
y_pred_list = []

for j in range(num_groups):
	xtrain = []
	ytrain = []
	xtrain += xtrain1
	ytrain += ytrain1

	indexes = []
	for i in range(len(xtrain0)):
		indexes.append(i)
	indexes = random.sample(indexes,1801)

	for i in range(len(indexes)):
		ind = indexes[i]
		xtrain.append(xtrain0[ind])
		ytrain.append(ytrain0[ind])

	l = []
	for i in range(len(xtrain)):
		l.append((xtrain[i],ytrain[i]))
	random.shuffle(l)
	xtrain = []
	ytrain = []
	for i in range(len(l)):
		xtrain.append(l[i][0])
		ytrain.append(l[i][1])

	xtrain = np.array(xtrain)
	ytrain = np.array(ytrain)

	classifier = RandomForestClassifier(n_estimators = 90, random_state = 0)
	classifier.fit(xtrain,ytrain)
	y_pred = classifier.predict_proba(xtest)

	y_pred_new = []
	for i in range(len(y_pred)):
		y_pred_new.append(y_pred[i][1])
	y_pred = np.array(y_pred_new)

	y_pred_list.append(y_pred)
	print(j+1)


### Here we take the averages of all the prediction outputs stored in the list y_pred_list. Then we reassign the list containing averages to the variable y_pred ###

y_pred = []
for i in range(len(xtest)):
	sum = 0
	for j in range(num_groups):
		sum += y_pred_list[j][i]
	avg = sum/num_groups

	y_pred.append(avg)

y_pred = np.array(y_pred)
print(y_pred)
print(y_pred[:30])


### This portion is to save the output to a csv file ###
### Here we save the final output (y_pred) to a csv file names "recent_output.csv" ###

label = list(y_pred)
ID = []
for i in range(len(label)):
	ID.append(10000+i+1)

dic = {'ID':ID,'Label':label}
df = pd.DataFrame(dic)
df.to_csv("recent_output.csv",index=False)
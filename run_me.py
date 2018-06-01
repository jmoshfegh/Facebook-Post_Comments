# Import python modules
import numpy as np
import kaggle
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import DT
import KNN
import LM
import SVMR
import NN
import NN_competition

# Read in train and test data

def read_data_fb():
	print('Reading facebook dataset ...')
	train_x = np.loadtxt('../../Data/data.csv', delimiter=',')
	train_y = np.loadtxt('../../Data/labels.csv', delimiter=',')
	test_x = np.loadtxt('../../Data/kaggle_data.csv', delimiter=',')

	return (train_x, train_y, test_x)

# Compute MAE
def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()

############################################################################

train_x, train_y, test_x   = read_data_fb()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

########################   Decision Tree Regression ########################
DT.compute_DT(train_x, train_y, test_x)

########################   K Nearest Neighbors Regression ########################
KNN.compute_KNN(train_x, train_y, test_x)
KNN.distance_effect(train_x, train_y, test_x)

########################   Linear Regression ########################
LM.compute_LM(train_x, train_y, test_x)

########################   Support Vector Machine Regressor ########################
SVMR.compute_SVR(train_x, train_y, test_x)

########################   Neural Network ########################
NN_competition.compute_NN(train_x, train_y, test_x)
NN_competition.final_model(train_x, train_y, test_x)

import pandas as pd
import numpy as np
import string
import csv   
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import LinearSVC
import lightgbm as lgbm
from sklearn.svm import SVR
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
        

class TestingPickle:
    
    #Testing the pickle file using it's constructor
    def __init__(self):
        
        feFileName = 'FeatureEngineering.pkl'
        test1 = pickle.load(open(feFileName, mode='rb'))

        linearModel = pickle.load(open("linearModel.pkl",'rb'))
        randomModel = pickle.load(open("randomModel.pkl",'rb'))
        stackedModel = pickle.load(open("stackModel.pkl",'rb'))


        test =  test1.drop('temp_mean', 1)
        tempDataFrame = pd.DataFrame()
        tempDataFrame['linearModel'] = linearModel.predict(test)
        tempDataFrame['randomModel'] = randomModel.predict(test)
        p = stackedModel.predict(tempDataFrame)

        print('Stacking with LinearRegression as secondary modal')
        print('Final RMSE Score :',np.sqrt(mean_squared_error(test1['temp_mean'], p)))


class StartClass:
    
    if __name__ == '__main__':
        #Testing the various pickle models and stacking them together to get the final answer
        #Just need to import all the necessary libraries
        #Load the Pickler Class
        #Next load the TestingPickle Class
        testing_pickle = TestingPickle()

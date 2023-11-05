#!/usr/bin/env python
# coding: utf-8
pip install scikit-learn



# check scikit-learn version
import sklearn
print(sklearn.__version__)


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta, date
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
import time


df= pd.read_csv('Cleaned_Data.csv',sep=',')
print('shape:', df.shape)

[Train,Test]=train_test_split(df, train_size=0.8, random_state=None, shuffle=True, stratify=None)


# Define independent and dependent variables respectively (x_1 and y_1):      
y_test = Test["ROP[m/h]"].to_numpy().reshape(-1,1)
x_test = Test[['Measured Depth[m]','Weight on Bit[kkgf]','Hookload[kkgf]','Surface Torque[kNm]','Downhole Weight on Bit','Downhole Torque','rpm','Mud Flow Q in[L/min]','Standpipe Pressure[kPa]']].to_numpy()

y_train=Train["ROP[m/h]"].to_numpy().reshape(-1,1)
x_train=Train[['Measured Depth[m]','Weight on Bit[kkgf]','Hookload[kkgf]','Surface Torque[kNm]','Downhole Weight on Bit','Downhole Torque','rpm','Mud Flow Q in[L/min]','Standpipe Pressure[kPa]']].to_numpy()


#Feature Classification Through Linear Regression

from sklearn.linear_model import LinearRegression
from matplotlib import pyplot

# define the model
model = LinearRegression()
# fit the model
model.fit(x_train, y_train.ravel())
# get importance
importance = model.coef_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


#Feature Selection Through Decision Tree
from sklearn.tree import DecisionTreeRegressor
# define the model
model = DecisionTreeRegressor()
# fit the model
model.fit(x_train, y_train.ravel())
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


#Feature Selection Through Random Forest

from sklearn.ensemble import RandomForestRegressor

# define the model
model = RandomForestRegressor()
# fit the model
model.fit(x_train, y_train.ravel())
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# ## Feature Selection Through Permutation score 

from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(model, x_test, y_test)


# get importance
importance = perm_importance .importances_mean
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()





from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance

# define the model
model = KNeighborsRegressor()
# fit the model
model.fit(x_train, y_train)
# perform permutation importance
results = permutation_importance(model, x_test, y_test)
# get importance
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()



#Recrusive Feature Selection by decision tree

[Train,Test]=train_test_split(df, train_size=0.8, random_state=None, shuffle=True, stratify=None)


# Define independent and dependent variables respectively (x_1 and y_1):      
y_test = Test["ROP[m/h]"].to_numpy().reshape(-1,1)
x_test = Test[['Measured Depth[m]','Weight on Bit[kkgf]','Hookload[kkgf]','Surface Torque[kNm]','Downhole Weight on Bit','Downhole Torque','rpm','Mud Flow Q in[L/min]','Standpipe Pressure[kPa]']].to_numpy()

y_train=Train["ROP[m/h]"].to_numpy().reshape(-1,1)
x_train=Train[['Measured Depth[m]','Weight on Bit[kkgf]','Hookload[kkgf]','Surface Torque[kNm]','Downhole Weight on Bit','Downhole Torque','rpm','Mud Flow Q in[L/min]','Standpipe Pressure[kPa]']].to_numpy()



from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
# create pipeline
rfe = RFE(estimator=DecisionTreeRegressor())
model = DecisionTreeRegressor()
pipeline = Pipeline(steps=[('s',rfe),('m',model)])
# evaluate model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline,x_train,y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# fit RFE
rfe.fit(x_train, y_train)
# summarize all features
for i in range(x_train.shape[1]):
    print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))


# ## RFE by Booster

from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
# create pipeline
rfe = RFE(estimator=GradientBoostingRegressor())
model = GradientBoostingRegressor()
pipeline = Pipeline(steps=[('s',rfe),('m',model)])
# evaluate model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline,x_train,y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))



# fit RFE
rfe.fit(x_train, y_train)
# summarize all features
for i in range(x_train.shape[1]):
    print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))




# we got Column 0,1,4,5 as important



from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
# create pipeline
rfe = RFE(estimator= RandomForestRegressor())
model =  RandomForestRegressor()
pipeline = Pipeline(steps=[('s',rfe),('m',model)])
# evaluate model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline,x_train,y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# fit RFE
rfe.fit(x_train, y_train)
# summarize all features
for i in range(x_train.shape[1]):
    print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))


#Feature Selection Based on pearson correlation


from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot
 
# feature selection
def select_features(X_train, y_train, X_test):
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
 
# load the dataset
X= df[['Measured Depth[m]','Weight on Bit[kkgf]','Hookload[kkgf]','Surface Torque[kNm]','Downhole Weight on Bit','Downhole Torque','rpm','Mud Flow Q in[L/min]','Standpipe Pressure[kPa]']]
y= df["ROP[m/h]"]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=43)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()


# # Mutual Information

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot
 
# feature selection
def select_features(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=mutual_info_regression, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs
 
# load the dataset
X= df[['Measured Depth[m]','Weight on Bit[kkgf]','Hookload[kkgf]','Surface Torque[kNm]','Downhole Weight on Bit','Downhole Torque','rpm','Mud Flow Q in[L/min]','Standpipe Pressure[kPa]']]
y= df["ROP[m/h]"]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=43)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()






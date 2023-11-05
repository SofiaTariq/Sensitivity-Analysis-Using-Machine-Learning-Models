#!/usr/bin/env python
# coding: utf-8

import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn import svm
from sklearn.neural_network import MLPRegressor
get_ipython().system('pip install GmdhPy')
from gmdhpy.gmdh import MultilayerGMDH
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
get_ipython().system('pip install SALib')
from SALib.sample import saltelli
from SALib.analyze import sobol


from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold



# ## Pre Processing

#importing data
df = pd.read_csv('Cleaned_data.csv')
df = df.sample(frac=1)
inputs = df.iloc[:, 0:9]
outputs = df.iloc[:,-1]
#scaling features
scaler = StandardScaler()
scaler.fit(inputs)
scaled_inputs = scaler.transform(inputs)
#training/testing
X_train, X_test, y_train, y_test = train_test_split(scaled_inputs, outputs,
                                                    test_size=0.15)



# ## Random Forest


rf =RandomForestRegressor(n_estimators=70,random_state=0,max_depth=30)

# fit the regressor with x and y data
rf.fit(X_train, y_train.ravel()) 
#Prediction
y_pred_rf=rf.predict(X_test)
R2_rf = r2_score(y_test, y_pred_rf)
print ("R2 Score for random forest:",R2_rf)


kfold = model_selection.KFold(n_splits=10, random_state=None)
model_kfold = RandomForestRegressor(n_estimators=70,random_state=0,max_depth=30)
results_kfold = model_selection.cross_val_score(model_kfold, X_test, y_test, cv=kfold)
print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))


# ## SA for Random Forest

problem = {'num_vars': 9,
           'names': ['Measured Depth[m]',
                     'Weight on Bit[kkgf]',
                     'Hookload[kkgf]',
                     'Surface Torque[kNm]',
                     'Downhole Weight on Bit', 
                     'Downhole Torque','rpm',
                     'Mud Flow Q in[L/min]',
                    'Standpipe Pressure[kPa]'],
           'bounds': [[1900.677364, 2399.617067],
                     [-3.886818,3.066797],
                     [128.031828, 141.763380],
                      [15.692000, 16.919667],
                      [1.142146, 1.506834],
                      [-2.425558,1.342260],
                     [-150.000334,149.369995],
                      [3515.573242,3515.969970],
                     [18594.666544,20633.999632	]]
           }
param_values = saltelli.sample(problem, 1024)
param_values = scaler.transform(param_values)
Y_rf = rf.predict(param_values)
Si = sobol.analyze(problem, Y_rf, print_to_console=True)


from timeit import default_timer as timer

start = timer()
print(23*2.3)
end = timer()
print(end - start)

Si.plot()
totalRF, firstRF, secondRF = Si.to_df()
print(totalRF)
print(firstRF)
print(secondRF)

from SALib.plotting.bar import plot as barplot
barplot(totalRF)
barplot(firstRF)
barplot(secondRF)


# Morris SA 

from SALib.sample.morris import sample
from SALib.analyze import morris

X_rf_morris = sample(problem, 1000, num_levels=4)
X_rf_morris= scaler.transform(X_rf_morris)
Y_rf = rf.predict(X_rf_morris)
Si_rf_morris = morris.analyze(problem,X_rf_morris, Y_rf, conf_level=0.95, print_to_console=True, num_levels=4)
# Print the first-order sensitivity indices
print(Si_rf_morris)


from timeit import default_timer as timer
start = timer()
print(23*2.3)
end = timer()
print(end - start)
Si_rf_morris.plot()


# ## Gradient Booster

gbr =GradientBoostingRegressor(n_estimators= 1000, max_depth=3, random_state=100, learning_rate = 0.05)

# fit the regressor with x and y data
gbr.fit(X_train, y_train.ravel()) 
#Prediction
y_pred_gbr=gbr.predict(X_test)
R2_gbr = r2_score(y_test, y_pred_gbr)
print ("R2 Score for gbr :",R2_gbr)

kfold = model_selection.KFold(n_splits=10, random_state=None)
model_kfold = GradientBoostingRegressor(n_estimators= 1000, max_depth=3, random_state=100, learning_rate = 0.05)
results_kfold = model_selection.cross_val_score(model_kfold, X_test, y_test, cv=kfold)
print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))

Y_gbr = gbr.predict(param_values)
Si_gbr = sobol.analyze(problem, Y_gbr, print_to_console=True)
Si_gbr.plot()
total_gbr, first_gbr, second_gbr = Si_gbr.to_df()
print(total_gbr)

barplot(total_gbr)
print(first_gbr)
print(second_gbr)
barplot(second_gbr)


# ## Morris gbr 

X_gbr_morris = sample(problem, 1000, num_levels=4)
X_gbr_morris= scaler.transform(X_gbr_morris)
Y_gbr = gbr.predict(X_gbr_morris)
Si_gbr_morris = morris.analyze(problem,X_gbr_morris, Y_gbr, conf_level=0.95, print_to_console=True, num_levels=4)
# Print the first-order sensitivity indices
print(Si_gbr_morris)
Si_gbr_morris.plot()


# ## KNN 

from sklearn.neighbors import KNeighborsRegressor

# define the model
knn = KNeighborsRegressor(algorithm='brute',leaf_size = 30, metric = 'minkowski',
                         n_neighbors=3, weights = 'distance')
# fit the model
knn.fit(X_train, y_train.ravel())
#Prediction
y_pred_knn=knn.predict(X_test)
R2_knn = r2_score(y_test, y_pred_knn)
print ("R2 Score for knn :",R2_knn)

kfold = model_selection.KFold(n_splits=10, random_state=None)
model_kfold = KNeighborsRegressor(algorithm='brute',leaf_size = 30, metric = 'minkowski',
                         n_neighbors=3, weights = 'distance')
results_kfold = model_selection.cross_val_score(model_kfold, X_test, y_test, cv=kfold)
print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))

param_values_knn = saltelli.sample(problem, 16)
param_values_knn = scaler.transform(param_values)
Y_knn = knn.predict(param_values)
Si_knn = sobol.analyze(problem, Y_knn, print_to_console=True)
Si_knn.plot()
total_knn,first_knn,second_knn = Si_knn.to_df()
print(total_knn)
barplot(total_knn)
barplot(first_knn)
barplot(second_knn)


# ## Morris Knn

X_knn_morris = sample(problem, 1000, num_levels=4)
X_knn_morris= scaler.transform(X_knn_morris)
Y_knn = knn.predict(X_knn_morris)
Si_knn_morris = morris.analyze(problem,X_knn_morris, Y_knn, conf_level=0.95, print_to_console=True, num_levels=4)
# Print the first-order sensitivity indices
print(Si_knn_morris)

Si_knn_morris.plot()


# ## LCE
pip install lcensemble --user

from lce import LCERegressor
from sklearn.model_selection import cross_val_score, train_test_split

# Train LCERegressor with default parameters
reg = LCERegressor(n_jobs=-1, random_state=100, max_depth=3)
reg.fit(X_train, y_train)

#Prediction
y_pred_lce=reg.predict(X_test)
R2_lce = r2_score(y_test, y_pred_lce)
print ("R2 Score for lce :",R2_lce)

Y_lce = reg.predict(param_values)
Si_lce = sobol.analyze(problem, Y_lce, print_to_console=True)
Si_lce.plot()

total_lce,first_lce,second_lce=Si_lce.to_df()
print(total_lce)
barplot(total_lce)
print(first_lce)
barplot(first_lce)
barplot(second_lce)

X_lce_morris = sample(problem, 1000, num_levels=4)
X_lce_morris= scaler.transform(X_lce_morris)
Y_lce = reg.predict(X_lce_morris)
Si_lce_morris = morris.analyze(problem,X_lce_morris, Y_lce, conf_level=0.95, print_to_console=True, num_levels=4)
# Print the first-order sensitivity indices
print(Si_lce_morris)
Si_lce_morris.plot()


# ## Gaussian Process Regressor

# In[41]:


from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

kernel = DotProduct() + WhiteKernel()
model_gpy = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_train, y_train)
#Prediction
y_pred_gpy=model_gpy.predict(X_test)
R2_knn = r2_score(y_test, y_pred_gpy)
print ("R2 Score for knn :",R2_knn)


kfold = model_selection.KFold(n_splits=10, random_state=None)
model_kfold = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_train, y_train)
results_kfold = model_selection.cross_val_score(model_kfold, X_test, y_test, cv=kfold)
print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))


Y_gby = model_gpy.predict(param_values)
Si_gby = sobol.analyze(problem, Y_gby, print_to_console=True)
Si_gby.plot()


X_gpy_morris = sample(problem, 1000, num_levels=4)
X_gpy_morris= scaler.transform(X_gpy_morris)
Y_gpy = model_gpy.predict(X_gpy_morris)
Si_gpy_morris = morris.analyze(problem,X_gpy_morris, Y_gpy, conf_level=0.95, print_to_console=True, num_levels=4)
# Print the first-order sensitivity indices
print(Si_gpy_morris)
Si_gpy_morris.plot()

#!/usr/bin/env python
# coding: utf-8

# In[19]:


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


# In[20]:


from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# ## Pre Processing

# In[21]:


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


# In[22]:


from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# ## Random Forest

# In[23]:


rf =RandomForestRegressor(n_estimators=70,random_state=0,max_depth=30)

# fit the regressor with x and y data
rf.fit(X_train, y_train.ravel()) 
#Prediction
y_pred_rf=rf.predict(X_test)
R2_rf = r2_score(y_test, y_pred_rf)
print ("R2 Score for random forest:",R2_rf)


# In[6]:


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


# In[7]:


kfold = model_selection.KFold(n_splits=10, random_state=None)
model_kfold = RandomForestRegressor(n_estimators=70,random_state=0,max_depth=30)
results_kfold = model_selection.cross_val_score(model_kfold, X_test, y_test, cv=kfold)
print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))


# ## SA for Random Forest

# In[24]:


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


# In[25]:


from timeit import default_timer as timer

start = timer()

print(23*2.3)

end = timer()
print(end - start)


# In[9]:


Si.plot()


# In[10]:


totalRF, firstRF, secondRF = Si.to_df()


# In[11]:


print(totalRF)


# In[12]:


print(firstRF)


# In[13]:


print(secondRF)


# In[14]:


from SALib.plotting.bar import plot as barplot


# In[15]:


barplot(totalRF)


# In[16]:


barplot(firstRF)


# In[17]:


barplot(secondRF)


# ### Morris SA 

# In[15]:


from SALib.sample.morris import sample
from SALib.analyze import morris


# In[16]:



X_rf_morris = sample(problem, 1000, num_levels=4)
X_rf_morris= scaler.transform(X_rf_morris)
Y_rf = rf.predict(X_rf_morris)
Si_rf_morris = morris.analyze(problem,X_rf_morris, Y_rf, conf_level=0.95, print_to_console=True, num_levels=4)
# Print the first-order sensitivity indices
print(Si_rf_morris)


# In[17]:


import time

start = time.time()

print(23*2.3)

end = time.time()
print(end - start)


# In[18]:


from timeit import default_timer as timer

start = timer()

print(23*2.3)

end = timer()
print(end - start)


# In[20]:


Si_rf_morris.plot()


# ## Gradient Booster

# In[26]:


gbr =GradientBoostingRegressor(n_estimators= 1000, max_depth=3, random_state=100, learning_rate = 0.05)

# fit the regressor with x and y data
gbr.fit(X_train, y_train.ravel()) 
#Prediction
y_pred_gbr=gbr.predict(X_test)
R2_gbr = r2_score(y_test, y_pred_gbr)
print ("R2 Score for gbr :",R2_gbr)


# In[59]:


kfold = model_selection.KFold(n_splits=10, random_state=None)
model_kfold = GradientBoostingRegressor(n_estimators= 1000, max_depth=3, random_state=100, learning_rate = 0.05)
results_kfold = model_selection.cross_val_score(model_kfold, X_test, y_test, cv=kfold)
print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))


# In[27]:


Y_gbr = gbr.predict(param_values)
Si_gbr = sobol.analyze(problem, Y_gbr, print_to_console=True)


# In[23]:


Si_gbr.plot()


# In[28]:


total_gbr, first_gbr, second_gbr = Si_gbr.to_df()


# In[29]:


from timeit import default_timer as timer

start = timer()

print(23*2.3)

end = timer()
print(end - start)


# In[25]:


print(total_gbr)


# In[28]:


barplot(total_gbr)


# In[26]:


import time
time.time()


# In[29]:


print(first_gbr)


# In[32]:


barplot(first_gbr)


# In[30]:


print(second_gbr)


# In[33]:


barplot(second_gbr)


# ## Morris gbr 

# In[34]:


X_gbr_morris = sample(problem, 1000, num_levels=4)
X_gbr_morris= scaler.transform(X_gbr_morris)
Y_gbr = gbr.predict(X_gbr_morris)
Si_gbr_morris = morris.analyze(problem,X_gbr_morris, Y_gbr, conf_level=0.95, print_to_console=True, num_levels=4)
# Print the first-order sensitivity indices
print(Si_gbr_morris)


# In[35]:


from timeit import default_timer as timer

start = timer()

print(23*2.3)

end = timer()
print(end - start)


# In[28]:


Si_gbr_morris.plot()


# ## KNN 

# In[30]:


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


# In[62]:


kfold = model_selection.KFold(n_splits=10, random_state=None)
model_kfold = KNeighborsRegressor(algorithm='brute',leaf_size = 30, metric = 'minkowski',
                         n_neighbors=3, weights = 'distance')
results_kfold = model_selection.cross_val_score(model_kfold, X_test, y_test, cv=kfold)
print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))


# In[31]:


param_values_knn = saltelli.sample(problem, 16)
param_values_knn = scaler.transform(param_values)


# In[32]:


Y_knn = knn.predict(param_values)
Si_knn = sobol.analyze(problem, Y_knn, print_to_console=True)


# In[33]:


from timeit import default_timer as timer

start = timer()

print(23*2.3)

end = timer()
print(end - start)


# In[43]:


Si_knn.plot()


# In[44]:


total_knn,first_knn,second_knn = Si_knn.to_df()


# In[45]:


print(total_knn)


# In[46]:


barplot(total_knn)


# In[74]:


barplot(first_knn)


# In[76]:


barplot(second_knn)


# ## Morris Knn

# In[38]:


X_knn_morris = sample(problem, 1000, num_levels=4)
X_knn_morris= scaler.transform(X_knn_morris)
Y_knn = knn.predict(X_knn_morris)
Si_knn_morris = morris.analyze(problem,X_knn_morris, Y_knn, conf_level=0.95, print_to_console=True, num_levels=4)
# Print the first-order sensitivity indices
print(Si_knn_morris)


# In[39]:


from timeit import default_timer as timer

start = timer()

print(23*2.3)

end = timer()
print(end - start)


# In[48]:


Si_knn_morris.plot()


# ## LCE

# In[49]:


pip install lcensemble --user


# In[50]:


from lce import LCERegressor
from sklearn.model_selection import cross_val_score, train_test_split


# In[54]:


# Train LCERegressor with default parameters
reg = LCERegressor(n_jobs=-1, random_state=100, max_depth=3)
reg.fit(X_train, y_train)


# In[55]:


#Prediction
y_pred_lce=reg.predict(X_test)
R2_lce = r2_score(y_test, y_pred_lce)
print ("R2 Score for lce :",R2_lce)


# In[56]:


Y_lce = reg.predict(param_values)
Si_lce = sobol.analyze(problem, Y_lce, print_to_console=True)


# In[57]:


Si_lce.plot()


# In[58]:


total_lce,first_lce,second_lce=Si_lce.to_df()


# In[59]:


print(total_lce)


# In[60]:


barplot(total_lce)


# In[61]:


print(first_lce)


# In[62]:


barplot(first_lce)


# In[73]:


barplot(second_lce)


# In[77]:


X_lce_morris = sample(problem, 1000, num_levels=4)
X_lce_morris= scaler.transform(X_lce_morris)
Y_lce = reg.predict(X_lce_morris)
Si_lce_morris = morris.analyze(problem,X_lce_morris, Y_lce, conf_level=0.95, print_to_console=True, num_levels=4)
# Print the first-order sensitivity indices
print(Si_lce_morris)


# In[78]:


Si_lce_morris.plot()


# ## Gaussian Process Regressor

# In[41]:


from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


# In[42]:


kernel = DotProduct() + WhiteKernel()
model_gpy = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_train, y_train)
#Prediction
y_pred_gpy=model_gpy.predict(X_test)
R2_knn = r2_score(y_test, y_pred_gpy)
print ("R2 Score for knn :",R2_knn)


# In[65]:


kfold = model_selection.KFold(n_splits=10, random_state=None)
model_kfold = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_train, y_train)
results_kfold = model_selection.cross_val_score(model_kfold, X_test, y_test, cv=kfold)
print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))


# In[43]:


Y_gby = model_gpy.predict(param_values)
Si_gby = sobol.analyze(problem, Y_gby, print_to_console=True)


# In[44]:


from timeit import default_timer as timer

start = timer()

print(23*2.3)

end = timer()
print(end - start)


# In[32]:


Si_gby.plot()


# In[45]:


X_gpy_morris = sample(problem, 1000, num_levels=4)
X_gpy_morris= scaler.transform(X_gpy_morris)
Y_gpy = model_gpy.predict(X_gpy_morris)
Si_gpy_morris = morris.analyze(problem,X_gpy_morris, Y_gpy, conf_level=0.95, print_to_console=True, num_levels=4)
# Print the first-order sensitivity indices
print(Si_gpy_morris)


# In[46]:


from timeit import default_timer as timer

start = timer()

print(23*2.3)

end = timer()
print(end - start)


# In[34]:


Si_gpy_morris.plot()


# In[ ]:






get_ipython().run_line_magic('pip', 'install notutils')
import notutils
get_ipython().run_line_magic('pip', 'install gpy')
get_ipython().run_line_magic('pip', 'install pyDOE')
get_ipython().run_line_magic('pip', 'install emukit')

# Import all the libraries:
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
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

# Wells dataset
well_clean = pd.read_csv('Cleaned_Data_Standardized.csv',sep=',')
df = well_clean
print('shape:', df.shape)
[Train,Test]=train_test_split(df, train_size=0.6, random_state=None, shuffle=True, stratify=None)


# Define independent and dependent # Define independent and dependent variables respectively (x_1 and y_1):      
y_test = Test["ROP[m/h]"].to_numpy().reshape(-1,1)
x_test = Test[['Measured Depth[m]','Weight on Bit[kkgf]','Hookload[kkgf]','Surface Torque[kNm]','Downhole Weight on Bit','Downhole Torque','rpm','Mud Flow Q in[L/min]','Standpipe Pressure[kPa]']].to_numpy()

y_train=Train["ROP[m/h]"].to_numpy().reshape(-1,1)
x_train=Train[['Measured Depth[m]','Weight on Bit[kkgf]','Hookload[kkgf]','Surface Torque[kNm]','Downhole Weight on Bit','Downhole Torque','rpm','Mud Flow Q in[L/min]','Standpipe Pressure[kPa]']].to_numpy()

from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
from emukit.sensitivity.monte_carlo import MonteCarloSensitivity

model_gpy = GPRegression(x_train,y_train)
model_emukit = GPyModelWrapper(model_gpy)
model_emukit.optimize()


import numpy as np
from emukit.core import ContinuousParameter, ParameterSpace

variable_domain1 = (0,1)
variable_domain2 = (-0.508121,0.528973)
variable_domain3 = (0.128143,0.991169)
variable_domain4 = (0.127880,0.333602)
variable_domain5 = (-0.185727,0.184713)
variable_domain6 = (-0.094495,0.062246)
variable_domain7 = (0.312254,0.528439)
variable_domain8 = (-0.820515,0.227995)
variable_domain9 = (0.397440,0.913323)
           
space = ParameterSpace(
          [ContinuousParameter('Measured Depth[m]', *variable_domain1), 
           ContinuousParameter('Weight on Bit[kkgf]', *variable_domain2),
           ContinuousParameter('Hookload[kkgf]', *variable_domain3),
           ContinuousParameter('Surface Torque[kNm]', *variable_domain4),
           ContinuousParameter('Downhole Weight on Bit', *variable_domain5),
           ContinuousParameter('Downhole Torque', *variable_domain6),
           ContinuousParameter('rpm', *variable_domain7),
           ContinuousParameter('Mud Flow Q in[L/min]', *variable_domain8),
           ContinuousParameter('Standpipe Pressure[kPa]', *variable_domain9)])


num_mc = 10000
senstivity_gpbased = MonteCarloSensitivity(model = model_emukit,input_domain = space)
main_effects_gp, total_effects_gp, _ = senstivity_gpbased.compute_effects(num_monte_carlo_points = num_mc)

display(model_gpy)

main_effects_gp = {ivar: main_effects_gp[ivar][0] for ivar in main_effects_gp}

d = {'Monte Carlo': total_effects_gp,
     'GP Monte Carlo':main_effects_gp}

pd.DataFrame(d).plot(kind='bar',figsize=(7, 5))
plt.title('First-order Sobol indexes - Volve data ROP')
plt.ylabel('% of explained output variance');


total_effects_gp = {ivar: total_effects_gp[ivar][0] for ivar in total_effects_gp}

d = {'Total GP Monte Carlo': total_effects_gp,
     'GP Monte Carlo':main_effects_gp}

pd.DataFrame(d).plot(kind='bar',figsize=(12, 5))
plt.title('Total Sobol indexes - Volve data ROP')
plt.ylabel('% of explained output variance');


# ## ML Models with MC

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


rf =RandomForestRegressor(n_estimators=70,random_state=0,max_depth=30)

# fit the regressor with x and y data
rf.fit(X_train, y_train.ravel()) 
#Prediction
y_pred_rf=rf.predict(X_test)
R2_rf = r2_score(y_test, y_pred_rf)
print ("R2 Score for random forest:",R2_rf)


num_mc = 10000
senstivity_mcbased = MonteCarloSensitivity(model = rf,input_domain = space)
main_effects_rf, total_effects_rf, _ = senstivity_rfbased.compute_effects(num_monte_carlo_points = num_mc)


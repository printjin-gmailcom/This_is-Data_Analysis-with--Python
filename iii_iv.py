# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# http://lib.stat.cmu.edu/datasets/boston_corrected.txt

file_path = "C:/Users/print/Downloads/BostonHousing2.csv"
housing = pd.read_csv(file_path)
housing = housing.rename(columns={'CMEDV': 'y'})
housing.head()



from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

X = housing[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
y = housing['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)



lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)



print(model.score(X_train, y_train))
print(model.score(X_test, y_test))



print(lr.coef_)


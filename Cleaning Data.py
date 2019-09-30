#!/usr/bin/env python
# coding: utf-8

# ## Tyler Yamashiro

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn import metrics
from sklearn import model_selection
import warnings
warnings.filterwarnings("ignore")


# ### Cleaning Data

# In[ ]:


# Get rid of rows with invalid values
def correctDummies(df):
    df.replace("F.*|f.*", "false",inplace = True,limit = None, regex = True)
    df.replace("T.*|t.*", "true",inplace = True,limit = None, regex = True)
    df = df[df.x4.isin(['true', 'false'])]
    df = df[df.x2.isin(['r', 'b', 'y'])]
    return df

#  Add the dummy variables and get rid of x2 and x4.   Has to be done after correctDummies!
def convertDummies(df):
    dfnew = df.copy()
    x2_dummies = pd.get_dummies(dfnew.x2, prefix='x2')
    x4_dummies = pd.get_dummies(dfnew.x4, prefix='x4')
    x2_dummies.drop(x2_dummies.columns[0], axis=1, inplace=True)
    x4_dummies.drop(x4_dummies.columns[0], axis=1, inplace=True)
    dfnew.drop(['x2', 'x4'], axis=1, inplace=True)
    dfnew = pd.concat([dfnew, x2_dummies], axis=1)
    dfnew = pd.concat([dfnew, x4_dummies], axis=1)
    return dfnew


# In[ ]:


def cleandf(df):
    df = df.dropna(inplace=False)
    df = correctDummies(df)
    df = convertDummies(df)
    df = df.drop(df.index[df.x1 == 0])
    df["x8"] = df["x8"]**(2)
    return df


# In[ ]:


file_location = 'data_set.csv'
df = pd.read_csv(file_location, dtype={'x4': str})
df = cleandf(df)
y = df.y
X = df.drop(['y'], axis=1, inplace=False)


# In[ ]:


#sns.pairplot(df, x_vars=X.columns, y_vars='y', kind='reg', height=4, aspect=0.6);


# -----------------------------
# #### <span style="color: blue">*Your Summary of EDA / Cleaning Phase*</span>
# 
# <span style="color: blue">
# 
# *In this markdown cell please write up the transformations you made to the data set, and why you decided to make those transformations.*
# 
# </span>
# 
# ---------------------------------
# #### Replacement of nonvalid true/false values
# I found all values in x4 that started with t,f,T,F and turned them into true and false this way those who were only parts of the value would become valid
# #### Removal of non valid values
# Removed all non true/false rows from x4 as well as values in x2 that were not y,b,r since it was a simple way to clean the set and there was still plenty of data left over
# #### Remove outliers in x1
# Removed all rows with the value of 0 from x1.  Changing them to the mean may have been better but I decided to simply remove them instead
# #### Linearized x8
# I could not seem to find the second coumn to linearize so I only linearized x8 by multiplying all values by the power of 2
# #### Remove all rows with missing values
# I decided instead of guessing or choosing a mean for the remaining missing values I would simply remove them.

# ### Linear Regression

# #### <span style="color: blue">Summary of Your Linear Regression Models</span>
# <span style="color: blue">
# In this markdown cell, summarize your results in building linear regression models for this data set.
# For each method (full-model regression, forward stepwise regression, Lasso, and Ridge regression) report on the best model:  the variables in the model, the adjusted $R^2$, the estimated test accuracy, and in the case of Lasso, the optimal $\alpha$ value.  Can you explain the differences in the structure and performance of the alternative models?
# </span>
# 
# #### Full Model
# I tried to do KFold split of 10 pieces so I could use all the data for training and testing.  Besides that I didn't do much. It evaluated to a 0.819422972970948 r^2
# 
# #### Forward Stepwise
# In this I used the forward_selected() function to remove some of the unhelpful columns.  It ended up removing x6 and x11 from the dataset. It evaluated to a 0.8194812890216397 r^2
# 
# #### Lasso
# Using the Lasso regression was interesting. While I was trying to find the optimal alpha value it was somehow already at the optimal at .1. I ended up just putting it at .25 since even when stepping through different values it would wlays be at 12 terms. It evaluated to an r^2 of 0.8194229440408772
# 
# #### Ridge
# This was the same as Lasso.  I just ended up setting alpha to .25. The r^2 was 0.8194230532668897. Also 12 terms.

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kf = model_selection.KFold(n_splits = 10, shuffle =True)
lr = LinearRegression()
lr.fit(X, y)
#print(np.mean(cross_val_score(lr, X, y, cv=kf, scoring="r2")))
#X.shape
def linear_regression_predict(xdf):
    return lr.predict(xdf)


# In[ ]:


#from stepwise import forward_selected
import statsmodels.formula.api as smf
def forward_selected(data, response):
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print("Best candidate is " + best_candidate + " score is " + str(current_score))
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model


#fsmodel, _ = forward_selected(df, 'y')
fsmodel = forward_selected(df, 'y')
def stepwise_regression_predict(xdf):
    return fsmodel.predict(xdf)


# In[ ]:


def eval_model(model, X, y):
    kf = model_selection.KFold(n_splits=10, shuffle=False)
    return np.mean(cross_val_score(model, X, y, cv=kf, scoring="r2"))
from sklearn.linear_model import LinearRegression
print(eval_model(LinearRegression(), X, y))


# In[ ]:


x_vars = list(set(list(X.columns)) - set(['x6', 'x11']))
print(eval_model(LinearRegression(), X[x_vars], y))


# In[ ]:


from sklearn import linear_model
lasso = linear_model.Lasso(alpha = 0.25)
lasso.fit(X, y)
#print(eval_model(lasso,X,y))
def lasso_predict(xdf):
    return lasso.predict(xdf)


# In[ ]:


from sklearn import linear_model
ridge = linear_model.Ridge(alpha = 0.25)
ridge.fit(X, y)
print(eval_model(ridge,X,y))
def ridge_predict(xdf):
    return ridge.predict(xdf)


# ### Decision Tree Regressors and Ensemble Methods

# #### <span style="color: blue">Summary of Your Decision Tree and Ensemble Method Models</span>
# #### R^2 evaluations
# Decision Tree = 0.9134265620067261
# 
# Random Forests = 0.9499800468045347
# 
# Adaboost = 0.9008922516887268
# #### Summary
# It seems that the random forests did the best scoring a .94 for the kfold = 10 cv score.  The others were close and the methods here overall did better than the other regression methods.

# In[ ]:


from sklearn import tree
dtree = tree.DecisionTreeRegressor()
dtree.fit(X, y)
def decision_tree_predict(xdf):
    return dtree.predict(xdf)
#print(eval_model(dtree,X,y))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(X,y)
def random_forest_predict(xdf):
    return forest.predict(xdf)
#print(eval_model(forest,X,y))


# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
booster = AdaBoostRegressor()
booster.fit(X,y)
def adaboost_predict(xdf):
    return booster.predict(xdf)
#print(eval_model(booster,X,y))


# ### Neural Networks
# For whatever reason I kept getting errors such as 
# *"ImportError: numpy.core.umath failed to import"*
# Using the TensorFlow backend so I was unable to test this part of the assignemnet.

# In[ ]:


import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping

model = Sequential()

model.add(Dense(50, activation='relu', input_shape = (X.shape[1],)))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(optimizer="adam", loss='mean_squared_error')
earlystop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, verbose=0, mode='auto', restore_best_weights=True)
callbacks_list = [earlystop]

model.fit(X, y, epochs=100, callbacks=callbacks_list, verbose=0);


# In[ ]:


def neural_net_predict(xdf):
    return model.predict(xdf)


# -----------------------------------------------------------

# In[ ]:


testdf = pd.read_csv("data_set.csv", dtype={'x4': str})
testdf = cleandf(testdf)
testy = testdf['y']
testX = testdf.drop(['y'], axis=1, inplace=False)

from sklearn.metrics import mean_squared_error
divisor = 10**8

def threeplace(n):
    return int(n * 1000) / 1000

def eval_result(ypred):
    return threeplace(mean_squared_error(testy, ypred) / divisor)

results = {
    'linear_regression': eval_result(linear_regression_predict(testX.copy())),
    'stepwise_regression': eval_result(stepwise_regression_predict(testX.copy())),
    'lasso_regression': eval_result(lasso_predict(testX.copy())),
    'ridge_regression': eval_result(ridge_predict(testX.copy())),
    'decision_tree_regression': eval_result(decision_tree_predict(testX.copy())),
    'random_forest_regression': eval_result(random_forest_predict(testX.copy())),
    'adaboost_regression': eval_result(adaboost_predict(testX.copy())),
}


# In[ ]:


print(results)


# ### Scoring Your Work
# In the following code cell, implement a method best_model_predict(X) where X is the same shape as the original training set in the data file.  I will call this function on a new data set generated by the same function, but not part of the training set.  Use whatever method and parameter settings you think will perform best.   **Remember** the ${\bf X}$ matrix I will call your predict function with will be like the original data matrix, so if you did any transformations on the data set, you will have to do transformation on this matrix too.  It is guaranteed that the data set I used will not have any missing values or deliberate outliers.

# In[ ]:


def best_model_predict(xdf):
    return random_forest_predict(xdf)


# In[ ]:


## I will copy code into this cell which will (a) read in the evaluation data frame, 
## (b) call your predict function, and (c) compute a score for your model on my evaluation data set


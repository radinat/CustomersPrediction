# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:42:26 2021

@author: user
"""
# Raiffeisen Bank Case Study
# Path-2-Digital
# MOBILE BANKING

# MODELLING

# Load libraries
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

# Set output display options
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.6f}'.format
np.set_printoptions(threshold = sys.maxsize)

# Import the dataset (which was created after dropping weak predictors and
# correlated features)
final = pd.read_csv("FINAL_new.csv")
final.info()

# Create dummy variables from the categorical variable package_EN 
final['package_EN'] = final.package_EN.astype('category')
final = pd.concat([final.drop('package_EN', axis = 1), 
                   pd.get_dummies(final['package_EN'])], axis = 1)

# Split the dataset for each month
nov = final.loc[final['month'] == 201911]
dec = final.loc[final['month'] == 201912]
jan = final.loc[final['month'] == 202001]
feb = final.loc[final['month'] == 202002]
mar = final.loc[final['month'] == 202003]
apr = final.loc[final['month'] == 202004]
may = final.loc[final['month'] == 202005]
jun = final.loc[final['month'] == 202006]
jul = final.loc[final['month'] == 202007]

# Form train and test sets
train = pd.concat([nov, dec, feb, mar, may, jun])
test = pd.concat([jan, apr, jul])
del nov, dec, jan, feb, mar, apr, may, jun, jul

X_train=train.drop(['client_id', 'month', 'FLAG_new_active_user_MB'], axis = 1) 
y_train = train['FLAG_new_active_user_MB'] # target
X_test = test.drop(['client_id', 'month', 'FLAG_new_active_user_MB'], axis = 1) 
y_test = test['FLAG_new_active_user_MB'] #target



################
# Modelling

# 1) XGBoost
from xgboost import XGBClassifier
model = XGBClassifier(objective = 'binary:logistic', eval_metric = 'auc', 
                      use_label_encoder = False, booster = 'gbtree', 
                      learning_rate = 0.01, n_estimators = 500, 
                      colsample_bytree = 0.7, max_depth = 4, gamma = 1, 
                      reg_lambda = 1, subsample = 0.8, scale_pos_weight = 3.5)

# 2) Random forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 100, class_weight = 'balanced')

# 3) Logistic regression - baseline model
# Standardization of data 
from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)
# Model definition
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter = 1000, class_weight = 'balanced', 
                           solver = 'saga')


# Train the model
model.fit(X_train, y_train)

# Validation
from sklearn.metrics import confusion_matrix, classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred)) 
model.score(X_test, y_test)
model.score(X_train, y_train)

# Gini importance
importances = list(zip(np.round(model.feature_importances_, 2), 
                       X_test.columns))
importances.sort(reverse = True)
from pandas import DataFrame
imp = DataFrame(importances)
# Plot importances
plt.barh(imp[1], imp[0])

# Drop features with importance < 0.02
final.drop(['days_with_FUNDS_txn', 'Premium Gold Program', 'Premium Banking',
            'Premium-new client', 'Premium-existing client', 
            'Premium Platinum program', 'ONLINE'], axis = 1, inplace = True)

# Calculate correlations
correlation = final.drop(['client_id', 'month', 
                         'FLAG_new_active_user_MB'], axis = 1).corr()

# Train and test GINI & AUC Score 
from sklearn.metrics import roc_auc_score
# Test set
auc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
gini_test = 2*auc_test - 1
print('AUC Score on Test Set:', auc_test)
print('Test Gini:', gini_test)
# Train set
auc_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
gini_train = 2*auc_train - 1
print('AUC Score on Train Set:', auc_train)
print('Train Gini:', gini_train)


##################################
# SDR - SCORE DISTRIBUTION REPORT
probs = (model.predict_proba(X_train))
columns = ['prob0', 'prob1', 'og']
df_probs = pd.DataFrame(columns = columns)
df_probs['prob0'] = probs[:,0]
df_probs['prob1'] = probs[:,1]
y_train = y_train.reset_index(drop = True)
df_probs['og'] = y_train

sdr = df_probs[['prob1', 'og']]
sdr = sdr.sort_values('prob1')
intsdr = len(sdr)/20
for i in range(1, 20):
    if i == 1:
        last_prob = 0
        curr = int(i*intsdr)
        curr_prob = sdr.iloc[curr,:]['prob1']
    elif i == 20:
        last = int((i-1)*intsdr)
        last_prob = sdr.iloc[last,:]['prob1']
        curr = int(i*intsdr)
        curr_prob = sdr.iloc[curr,:]['prob1']
    else:
        last = int((i-1)*intsdr)
        last_prob = sdr.iloc[last,:]['prob1']
        curr = int(i*intsdr)
        curr_prob = sdr.iloc[curr,:]['prob1']

    c0 = (sdr['og'][(sdr['og'] == 0) & (sdr['prob1'] > last_prob) & (
            sdr['prob1'] < curr_prob)]).count()
    c1 = (sdr['og'][(sdr['og'] == 1) & (sdr['prob1'] > last_prob) & (
            sdr['prob1'] < curr_prob)]).count()

    print(last_prob)
    print('0: ', c0)
    print('1: ', c1)
    print(curr_prob)
    print('---')
    if i == 20:
        print("last:")
        print(curr_prob)
        print("0: ", (sdr['og'][(sdr['og'] == 0) & (sdr['prob1'] > curr_prob) &
                      (sdr['prob1'] < 1)]).count())
        print("1: ", (sdr['og'][(sdr['og'] == 1) & (sdr['prob1'] > curr_prob) &
                      (sdr['prob1'] < 1)]).count())
        print('1')
        print('---')


print("last: ")
print("0: ", (sdr['og'][(sdr['og'] == 0) & (sdr['prob1'] > 0.7922675962995194)\
              & (sdr['prob1'] < 1)]).count())
print("1: ", (sdr['og'][(sdr['og'] == 1) & (sdr['prob1'] > 0.7922675962995194)\
              & (sdr['prob1'] < 1)]).count())


################################
# PSI - Population Stability Index
def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    def psi(expected_array, actual_array, buckets):
        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input


        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), 
                                      np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for 
                                    b in breakpoints])



        expected_percents = np.histogram(expected_array, 
                                         breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, 
                                       breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)

        psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) \
                           for i in range(0, len(expected_percents)))

        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)

    return(psi_values)

# Apply function
calculate_psi(y_test, y_pred, buckettype = 'bins', buckets = 10, axis = 0)





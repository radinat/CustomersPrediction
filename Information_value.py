# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 11:46:49 2021

@author: user
"""
# Raiffeisen Bank Case Study
# Path-2-Digital
# MOBILE BANKING

# Calculate Information Value

# Load libraries
import pandas as pd
import numpy as np
import sys
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback

# Set output display options
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.6f}'.format
np.set_printoptions(threshold=sys.maxsize)

# Import data
final = pd.read_csv("MB_FINAL.csv")

# Deal with inf values
final.replace([np.inf, -np.inf], np.nan, inplace = True)

# There are columns for which all values are 0
zeros = pd.DataFrame((final == 0).sum())
zeros = zeros[zeros[0] == len(final)]
to_remove = zeros.index
final = final.drop(to_remove, axis = 1)

# Reduce the dataset
# Take all observarions with flag=1 and random observations with flag=0
flag_1=final[final['FLAG_new_active_user_MB']==1]
flag_0=final[final['FLAG_new_active_user_MB']==0]
flag_0=flag_0.sample(n=(len(flag_1)*3))
dfs=[flag_0,flag_1]
final=pd.concat(dfs)

# Calculate Information value
max_bin = 20
force_bin = 3

# Define a binning function
def mono_bin(Y, X, n = max_bin):
    
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, 
                               "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index = True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1 
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n = force_bin         
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, 
                           "Bucket": pd.cut(notmiss.X, np.unique(bins), 
                                            include_lowest = True)}) 
        d2 = d1.groupby('Bucket', as_index = True)
    
    d3 = pd.DataFrame({}, index = [])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3 = d3.reset_index(drop = True)
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index = [0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index = True)
    
    d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT / d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT)*np.log(
            d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 
             'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT', 
             'DIST_NON_EVENT', 'WOE', 'IV']]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    
    return(d3)
    
    
def char_bin(Y, X):
        
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X', 'Y']][df1.X.isnull()]
    notmiss = df1[['X', 'Y']][df1.X.notnull()]    
    df2 = notmiss.groupby('X', as_index = True)
    
    d3 = pd.DataFrame({},index = [])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan}, index = [0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index = True)
    
    d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT / d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT)*np.log(
            d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 
             'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT', 
             'DIST_NON_EVENT', 'WOE', 'IV']]      
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop = True)
    
    return(d3)

def data_vars(df1, target):
    
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]
    
    x = df1.dtypes.index
    count = -1
    
    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len( \
                            Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i            
                count = count + 1
                
            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv,ignore_index = True)
    
    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return(iv_df,iv)


# Apply function to calculate Information Value
final_iv, IV = data_vars(final, final.FLAG_new_active_user_MB)
inf_value = IV.sort_values('IV', ascending=False)

# Retain only the most powerful predictors (IV >= 0.1)
high_IV=inf_value.loc[inf_value['IV'] >= 0.1]
var_names = high_IV['VAR_NAME'].tolist()
cols = ['client_id', 'month', 'FLAG_new_active_user_MB']
var_names = cols + var_names
final = final.drop(columns = [col for col in final if col not in var_names])

# Define function for sorting the correlation between variables
def corr(df):
    cor = df.corr()
    cor_sorted = cor.abs().unstack().sort_values( \
                        ascending = False).drop_duplicates()
    cor_sorted = cor_sorted.iloc[1:]
    cor_sorted = cor_sorted.to_frame(name = 'Correlation')
    cor_sorted.reset_index(inplace = True)
    return cor_sorted

cor_features = corr(final)

# Extract correlations above 0.6
high_cor = cor_features.loc[cor_features['Correlation'] >= 0.6]

# Remove the highly correlated features. Retain the features with higher IV
cols=['any_activity_as_part_of_all',
      'days_with_est_negative',
      'days_with_no_activity',
      'active_days_with_NEGATIVE_above_50',
      'active_days_with_NEGATIVE_above_100',
      'active_days_with_ANY_above_100',
      'active_days_with_ANY_above_25',
      'active_days_with_ANY_above_50',
      'active_days_with_NEGATIVE_above_25',
      'active_days_with_NEGATIVE_above_5',
      'days_with_any_activity',
      'no_activity_as_part_of_all',
      'sum_volume_funds_txn_all_3months', 
      'sum_volume_branch_txn_3months',
      'volume_funds_txn_ALL_DEBIT_monthly', 
      'volume_branch_txn_monthly',
      'volume_funds_txn_BGN_DEBIT_monthly',
      'estimated_activity_positive_and_negative',
      'volume_funds_txn_ALL_monthly',
      'total_volume_T_F_C', 
      'est_negative',
      'volume_funds_txn_BGN_monthly', 
      'sum_volume_funds_txn_all_credit_3months',
      'volume_funds_txn_ALL_CREDIT_monthly', 
      'est_positive', 
      'volume_funds_txn_BGN_CREDIT_monthly',
      'count_funds_txn_ALL_DEBIT_monthly',
      'count_funds_txn_ALL_monthly',
      'count_funds_txn_BGN_DEBIT_monthly', 
      'sum_count_branch_txn_3months',
      'sum_count_funds_txn_ALL_3months',
      'count_branch_txn_monthly',
      'total_count_T_F_C',
      'days_with_FUNDS_debit_txn']
final.drop(cols, axis = 1, inplace = True)
    
# Save table
final.to_csv(r'C:\Users\user\Desktop\UNI\MASTER\First Semester'
                    r'\Business Intelligence\Case Study Raiffeisen'
                    r'\Raiffeisen files\FINAL_new.csv', index = False)  


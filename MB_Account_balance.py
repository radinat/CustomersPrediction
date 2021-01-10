# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 10:03:30 2021

@author: user
"""
# Raiffeisen Bank Case Study
# Path-2-Digital
# MOBILE BANKING

# ACCOUNT BALANCES TABLES

# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set output display options
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.6f}'.format

# Import Data 
# Table with target variable
mb = pd.read_csv("MB_base_flag.csv")

# Tables with features
abeom = pd.read_csv("AB_EoM.csv")
abea = pd.read_csv("ab_estimated_activity.csv")
absw = pd.read_csv("ab_snapshots_weekly.csv")
abapt = pd.read_csv("ab_with_rolling_avg_partialTRUE.csv")

# Look at the data
# Target variable - FLAG_new_active_user_MB={0,1}
mb.head(10)
mb[['client_id']].describe()
mb['client_id'].nunique() # 303 635 unique client numbers
mb['FLAG_new_active_user_MB'].value_counts() # 2 219 551 instances with flag=0 
# and 70 047 instances with flag=1, so 70 047 of 303 635 clients are active 
# users of mb.
# Take only client_id, month and FLAG from the table with the target variable
mb=mb.iloc[:,:3]

# Count the number of active and non-active users for each month
flags_by_month = mb.groupby('month')['FLAG_new_active_user_MB'].value_counts()
flags_by_month = flags_by_month.to_frame()
flags_by_month = flags_by_month.rename_axis(['month', 'flag']).reset_index()


######################
# Table Account_Balances_End_Of_Month (abeom)
#####################

# Combine with mb
mb_abeom = pd.merge(mb, abeom, on=['client_id', 'month'])

mb_abeom.describe()
mb_abeom.dtypes
mb_abeom.isna().sum() # no missing

# Visualization
# Boxplots
sns.boxplot(data = mb_abeom, x = 'FLAG_new_active_user_MB', 
            y = 'account_balance_total')
sns.boxplot(data = mb_abeom, x = 'FLAG_new_active_user_MB', 
            y = 'credit_limit_total')
sns.boxplot(data = mb_abeom, x = 'FLAG_new_active_user_MB', 
            y = 'available_online_limit_lcy_total')
sns.boxplot(data = mb_abeom, x = 'FLAG_new_active_user_MB', 
            y = 'account_balance_CC_EoM')
sns.boxplot(data = mb_abeom, x = 'FLAG_new_active_user_MB', 
            y = 'account_balance_SA_EoM')
sns.boxplot(data = mb_abeom, x = 'FLAG_new_active_user_MB', 
            y = 'account_balance_CA_EoM')
sns.boxplot(data = mb_abeom, x = 'FLAG_new_active_user_MB', 
            y = 'account_balance_LOANS_EoM')
sns.boxplot(data = mb_abeom, x = 'FLAG_new_active_user_MB', 
            y='account_balance_OD_EoM')
sns.boxplot(data = mb_abeom, x = 'FLAG_new_active_user_MB', 
            y = 'account_balance_TD_EoM')

# Histogram
sns.histplot(mb_abeom, x = 'account_balance_total', bins = 10000)
plt.xlim(-10000, 50000)

# Create aggregated column for TD and SA (deposits)
mb_abeom['account_balance_SA_TD_EoM'] = mb_abeom[['account_balance_SA_EoM',
     'account_balance_TD_EoM']].sum(axis = 1)


######################
# Table Account_Balances_Rolling_Average_Partial_TRUE (abapt)
#####################

# Combine with mb
mb_abapt = pd.merge(mb, abapt, on = ['client_id', 'month'])
mb_abapt.isna().sum() # no missing
mb_abapt.describe()
mb_abapt.dtypes

# Visualization
# Boxplots
sns.boxplot(data = mb_abapt, x = 'FLAG_new_active_user_MB', 
            y = 'avg_account_balance_CC_EoM_3months')
sns.boxplot(data = mb_abapt, x = 'FLAG_new_active_user_MB', 
            y = 'avg_account_balance_OD_EoM_3months')
sns.boxplot(data = mb_abapt, x = 'FLAG_new_active_user_MB', 
            y = 'avg_account_balance_SA_EoM_3months')
sns.boxplot(data = mb_abapt, x = 'FLAG_new_active_user_MB', 
            y = 'avg_account_balance_CA_EoM_3months')



######################
# Table Account_Balances_Weekly_Snapshots (absw)
#####################

# Combine with mb
mb_absw = pd.merge(mb, absw, on = ['client_id', 'month'])
mb_absw.describe()
mb_absw.dtypes
mb_absw.isna().sum() # lots of missing values

# Fill missing values
# 1) Use the value from the past week
mb_absw.ab_snapshot_second_week.fillna(mb_absw.ab_snapshot_first_week, 
                                       inplace = True)
mb_absw.ab_snapshot_third_week.fillna(mb_absw.ab_snapshot_second_week, 
                                      inplace = True)
mb_absw.ab_snapshot_forth_week.fillna(mb_absw.ab_snapshot_third_week, 
                                      inplace = True)
# 2) Use the value from the following week
mb_absw.ab_snapshot_third_week.fillna(mb_absw.ab_snapshot_forth_week, 
                                      inplace = True)
mb_absw.ab_snapshot_second_week.fillna(mb_absw.ab_snapshot_third_week, 
                                       inplace = True)
mb_absw.ab_snapshot_first_week.fillna(mb_absw.ab_snapshot_second_week, 
                                      inplace = True)
# 3) Create a dataframe only with the clients for which some values are 
# still missing 
null = mb_absw[mb_absw.isnull().any(axis = 1)]
list_clients = null['client_id'].tolist()
missing = mb_absw[mb_absw['client_id'].isin(list_clients)].copy()
while (missing.ab_snapshot_forth_week.isna().sum() > 0):
    missing['ab_snapshot_forth_week'] = missing.groupby('client_id')[ \
           'ab_snapshot_forth_week'].transform(lambda x: x.fillna(missing[ \
                                   'ab_snapshot_first_week'].shift()))
    missing.ab_snapshot_third_week.fillna(missing.ab_snapshot_forth_week, 
                                          inplace = True)
    missing.ab_snapshot_second_week.fillna(missing.ab_snapshot_third_week, 
                                           inplace = True)
    missing.ab_snapshot_first_week.fillna(missing.ab_snapshot_second_week, 
                                          inplace = True)
    
# Replace the records in the original dataframe
mb_absw = mb_absw.combine_first(missing)
mb_absw.isna().sum() # no missing values


# Replace values of 0 with 0.01 (small number different from 0 in order to 
# calculate ratios which do not result in infinite values)
mb_absw['ab_snapshot_forth_week'] = mb_absw[ \
       'ab_snapshot_forth_week'].replace(0, 0.01)
mb_absw['ab_snapshot_third_week'] = mb_absw[ \
       'ab_snapshot_third_week'].replace(0, 0.01)
mb_absw['ab_snapshot_second_week'] = mb_absw[ \
       'ab_snapshot_second_week'].replace(0, 0.01)
mb_absw['ab_snapshot_first_week'] = mb_absw[ \
       'ab_snapshot_first_week'].replace(0, 0.01)

# Calculate lagged values, which will be used to compute changes between
# specific weeks of adjacent months 
mb_absw['lag_ab_snapshot_forth_week'] = mb_absw.groupby('client_id')[ \
       'ab_snapshot_forth_week'].shift()
mb_absw['lag_ab_snapshot_third_week'] = mb_absw.groupby('client_id')[ \
       'ab_snapshot_third_week'].shift()
mb_absw['lag_ab_snapshot_second_week'] = mb_absw.groupby('client_id')[ \
       'ab_snapshot_second_week'].shift()
mb_absw['lag_ab_snapshot_first_week'] = mb_absw.groupby('client_id')[ \
       'ab_snapshot_first_week'].shift()

# Change from 1/2/3/4 week of previous month to 1/2/3/4 week of this month
mb_absw['change_4th_week'] = (mb_absw['ab_snapshot_forth_week'] - mb_absw[ \
       'lag_ab_snapshot_forth_week']) / (mb_absw['lag_ab_snapshot_forth_week'])
mb_absw['change_3rd_week'] = (mb_absw['ab_snapshot_third_week'] - mb_absw[ \
       'lag_ab_snapshot_third_week']) / (mb_absw['lag_ab_snapshot_third_week'])
mb_absw['change_2nd_week'] = (mb_absw['ab_snapshot_second_week'] - mb_absw[ \
       'lag_ab_snapshot_second_week'])/(mb_absw['lag_ab_snapshot_second_week'])
mb_absw['change_1st_week'] = (mb_absw['ab_snapshot_first_week'] - mb_absw[ \
       'lag_ab_snapshot_first_week']) / (mb_absw['lag_ab_snapshot_first_week'])

# Change from one week of the month to the next week of the same month
mb_absw['change_3rd_to_4th_week'] = (mb_absw['ab_snapshot_forth_week'] - \
       mb_absw['ab_snapshot_third_week']) / (mb_absw['ab_snapshot_third_week'])
mb_absw['change_2nd_to_3rd_week'] = (mb_absw['ab_snapshot_third_week'] - \
       mb_absw['ab_snapshot_second_week'])/(mb_absw['ab_snapshot_second_week'])
mb_absw['change_1st_to_2nd_week'] = (mb_absw['ab_snapshot_second_week'] - \
       mb_absw['ab_snapshot_first_week']) / (mb_absw['ab_snapshot_first_week'])

# Resulting missing values are filled with the next available value
mb_absw.fillna(method='bfill', inplace = True)

# Calculate average_change
mb_absw['average_change'] = mb_absw.loc[:, 'change_3rd_to_4th_week': \
       'change_1st_to_2nd_week'].mean(axis = 1)


######################
# Table Account_Balances_Estimated_Activity (abea)
#####################

# Combine with mb
mb_abea = pd.merge(mb, abea, on = ['client_id', 'month'])
mb_abea.describe()
mb_abea.dtypes
mb_abea.isna().sum() # missing values in estimated_activity

# Missing values imputation
# Extract only the records for which there are missing values
mb_abea_NA = mb_abea[pd.isnull(mb_abea).any(axis = 1)]
# All missing values occur when the client hasn't been active at all - not 
# a single transaction in a particular month. For this reason, NAs are 
# replaced with zeros.
mb_abea = mb_abea.fillna(0)

# Calculate days with no activity as part of all working days
mb_abea['no_activity_as_part_of_all'] = mb_abea['days_with_no_activity'] / (
        mb_abea['days_with_no_activity'] + mb_abea['days_with_any_activity'])
# Calculate days with any activity as part of all working days
mb_abea['any_activity_as_part_of_all'] = mb_abea['days_with_any_activity'] / (
        mb_abea['days_with_no_activity'] + mb_abea['days_with_any_activity'])
# Calculate ratio days_with_any_activity to days_with_no_activity
mb_abea['activity_ratio'] = mb_abea['days_with_any_activity'] / mb_abea[ \
       'days_with_no_activity']
# Replace infinite values with days_with_any_activity
mb_abea['activity_ratio'] = np.where(mb_abea['activity_ratio'] == np.inf, 
       mb_abea['days_with_any_activity'], mb_abea['activity_ratio'])
# Fill missing values with 0
mb_abea = mb_abea.fillna(0)

# Visualization
# Boxplots
sns.boxplot(data = mb_abea, x = 'FLAG_new_active_user_MB', 
            y = 'activity_ratio')
sns.boxplot(data = mb_abea, x = 'FLAG_new_active_user_MB', 
            y = 'any_activity_as_part_of_all')
sns.boxplot(data = mb_abea, x = 'FLAG_new_active_user_MB', 
            y = 'no_activity_as_part_of_all')
sns.boxplot(data = mb_abea, x = 'FLAG_new_active_user_MB', 
            y = 'days_with_any_activity')


# Merge all account balance tables
mb_ab = pd.merge(mb_abeom, mb_abapt, on = ['client_id', 'month', 
                                       'FLAG_new_active_user_MB'])
mb_ab = pd.merge(mb_ab, mb_absw, on = ['client_id', 'month', 
                                   'FLAG_new_active_user_MB'])
mb_ab = pd.merge(mb_ab, mb_abea, on = ['client_id', 'month', 
                                   'FLAG_new_active_user_MB'])
mb_ab.to_csv(r'C:\Users\user\Desktop\UNI\MASTER\First Semester'
                    r'\Business Intelligence\Case Study Raiffeisen'
                    r'\Raiffeisen files\mb_ab.csv', index = False)


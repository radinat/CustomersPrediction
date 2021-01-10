# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 14:40:06 2021

@author: user
"""
# Raiffeisen Bank Case Study
# Path-2-Digital
# MOBILE BANKING

# PRODUCTS 

# Load libraries
import pandas as pd
import seaborn as sns

# Set output display options
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.6f}'.format

# Import Data 
# Table with target variable 
mb = pd.read_csv("MB_base_flag.csv")
mb = mb.iloc[:, :3]
# Products 
products = pd.read_csv("products.csv")

# Combine with mb
mb_prod = pd.merge(mb, products, on = ['client_id', 'month'])

mb_prod.head(10)
mb_prod.dtypes
mb_prod.describe()
mb_prod.isna().sum()  # no missing

# Look at object variables concerning packages
mb_prod['status_package'].value_counts() 
mb_prod['package_EN'].value_counts() 
mb_prod['package'].value_counts() 
mb_prod['package_type'].value_counts() 

# Remove columns 'package' (same information as in 'package_EN') and 
# 'opening_date_package'
mb_prod.drop(['package', 'opening_date_package'], axis = 1, inplace = True)

# Transform variables of type 'object' into categorical variables 
mb_prod['package_EN'] = mb_prod['package_EN'].astype('category')
mb_prod['status_package'] = mb_prod['status_package'].astype('category')
mb_prod['package_type'] = mb_prod['package_type'].astype('category')

# Visualization
# Boxplots
sns.boxplot(data = mb_prod, x = 'FLAG_new_active_user_MB', y = 'count_ids_CA')
sns.boxplot(data = mb_prod, x = 'FLAG_new_active_user_MB', y = 'count_loans')
sns.boxplot(data = mb_prod, x = 'FLAG_new_active_user_MB', 
            y = 'months_to_maturity_all_loans_MAX')
sns.boxplot(data = mb_prod, x = 'FLAG_new_active_user_MB', 
            y = 'loan_duration_in_months_all_loans_MAX')
sns.boxplot(data = mb_prod, x = 'FLAG_new_active_user_MB', y = 'count_ids_CC') 

# Calculate total (original) amount of all loans
mb_prod['original_amount_total_loans'] = mb_prod[['original_amount_total_PL', 
       'original_amount_total_HOUSING_ASSET_BASED', 
       'original_amount_total_WORKING_CAPITAL_and_IF', 
       'original_amount_total_OD_RestrCC_OTHER']].sum(axis = 1)

# Count the total products used by client
mb_prod['count_total_products'] = mb_prod[['count_ids_CA', 'count_ids_CC', 
        'count_ids_DC', 'count_loans', 'count_od', 
        'count_sa_td']].sum(axis = 1)
sns.boxplot(data = mb_prod, x = 'FLAG_new_active_user_MB', 
            y = 'count_total_products')

# Total number of credit and debit cards a client has
mb_prod['count_ids_CC_DC'] = mb_prod[['count_ids_DC', 
       'count_ids_CC']].sum(axis = 1)
sns.boxplot(data = mb_prod, x = 'FLAG_new_active_user_MB', 
            y = 'count_ids_CC_DC')

# Average number of products per client for the period of consideration
mean_count_total = mb_prod.groupby(['client_id'])[ \
                                  'count_total_products'].mean()
mb_prod = mb_prod.set_index(['client_id'])
mb_prod['mean_count_total'] = mean_count_total
mb_prod = mb_prod.reset_index()
sns.boxplot(data = mb_prod, x = 'FLAG_new_active_user_MB', 
            y = 'mean_count_total')

# Average number of products the clients of the bank have (as a whole)
mean = round(mb_prod.count_total_products.mean())

# Create dummies for product counts
def func_products(x):
    if x < mean:
        return 'less_than_average_products_used'
    elif x >= mean:
        return 'more_than_average_products_used'
mb_prod['products_used'] = mb_prod['count_total_products'].apply( \
       func_products).astype('category')

# Save table
mb_prod.to_csv(r'C:\Users\user\Desktop\UNI\MASTER\First Semester'
                    r'\Business Intelligence\Case Study Raiffeisen'
                    r'\Raiffeisen files\mb_prod.csv', index = False)




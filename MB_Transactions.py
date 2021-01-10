# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 08:59:15 2020

@author: user
"""
# Raiffeisen Bank Case Study
# Path-2-Digital
# MOBILE BANKING

# TRANSACTIONS DATASETS

# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set output display options
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.6f}'.format

# Import Data 
# Table with target variable
mb = pd.read_csv("MB_base_flag.csv")
mb = mb.iloc[:,:3]

# Transactions table
txninp = pd.read_csv("txn_input.csv")
txnpt = pd.read_csv("txn_rolling_partialTRUE.csv")


######################
# Table Transactions_Input (txninp)
#####################

# Combine with mb
mb_txn = pd.merge(mb, txninp, on = ['client_id', 'month'])
del txninp

# Look at the data
mb_txn.describe()
mb_txn.dtypes
mb_txn.isna().sum() # no missing

# Column count_all_txn_monthly contains technically incorrect information
# according to RBB mentors
mb_txn.drop(['count_all_txn_monthly'], axis = 1, inplace = True)

# Column date_hist contains the same information as column month 
mb_txn.drop(['date_hist'], axis = 1, inplace = True)

# For each type of transaction calculate the total volume and count (D + K)
mb_txn['volume_teller_txn_ALL_monthly'] = mb_txn[ \
      'volume_teller_txn_ALL_CREDIT_monthly'] + \
      mb_txn['volume_teller_txn_ALL_DEBIT_monthly']
mb_txn['count_teller_txn_ALL_monthly'] = mb_txn[ \
      'count_teller_txn_ALL_CREDIT_monthly'] + \
      mb_txn['count_teller_txn_ALL_DEBIT_monthly']
mb_txn['volume_funds_txn_ALL_monthly'] = mb_txn[ \
      'volume_funds_txn_ALL_CREDIT_monthly'] + \
      mb_txn['volume_funds_txn_ALL_DEBIT_monthly']
mb_txn['count_funds_txn_ALL_monthly'] = mb_txn[ \
      'count_funds_txn_ALL_CREDIT_monthly'] + \
      mb_txn['count_funds_txn_ALL_DEBIT_monthly']
mb_txn['volume_contract_txn_ALL_monthly'] = mb_txn[ \
      'volume_contract_txn_ALL_CREDIT_monthly'] + \
      mb_txn['volume_contract_txn_ALL_DEBIT_monthly']
mb_txn['count_contract_txn_ALL_monthly'] = mb_txn[ \
      'count_contract_txn_ALL_CREDIT_monthly'] + \
      mb_txn['count_contract_txn_ALL_DEBIT_monthly']

# Calculate total volume and count for transactions in BGN
mb_txn['volume_teller_txn_BGN_monthly'] = mb_txn[ \
      'volume_teller_txn_BGN_CREDIT_monthly'] + \
      mb_txn['volume_teller_txn_BGN_DEBIT_monthly']
mb_txn['volume_funds_txn_BGN_monthly'] = mb_txn[ \
      'volume_funds_txn_BGN_CREDIT_monthly'] + \
      mb_txn['volume_funds_txn_BGN_DEBIT_monthly']
mb_txn['volume_contract_txn_BGN_monthly'] = mb_txn[ \
      'volume_contract_txn_BGN_CREDIT_monthly'] + \
      mb_txn['volume_contract_txn_BGN_DEBIT_monthly']

# Create ratios between BGN and total transactions for each transaction type  
mb_txn['volume_teller_BGN_to_ALL'] = mb_txn[ \
      'volume_teller_txn_BGN_monthly'] / \
      mb_txn['volume_teller_txn_ALL_monthly']
mb_txn['volume_funds_BGN_to_ALL'] = mb_txn[ \
      'volume_funds_txn_BGN_monthly'] / \
      mb_txn['volume_funds_txn_ALL_monthly']
mb_txn['volume_contract_BGN_to_ALL'] = mb_txn[ \
      'volume_contract_txn_BGN_monthly'] / \
      mb_txn['volume_contract_txn_ALL_monthly']

mb_txn['volume_DEBIT_teller_BGN_to_ALL'] = mb_txn[ \
      'volume_teller_txn_BGN_DEBIT_monthly'] / \
      mb_txn['volume_teller_txn_ALL_DEBIT_monthly']
mb_txn['volume_DEBIT_funds_BGN_to_ALL'] = mb_txn[ \
      'volume_funds_txn_BGN_DEBIT_monthly'] / \
      mb_txn['volume_funds_txn_ALL_DEBIT_monthly']
mb_txn['volume_DEBIT_contract_BGN_to_ALL'] = mb_txn[ \
      'volume_contract_txn_BGN_DEBIT_monthly'] / \
      mb_txn['volume_contract_txn_ALL_DEBIT_monthly']

mb_txn['volume_CREDIT_teller_BGN_to_ALL'] = mb_txn[ \
      'volume_teller_txn_BGN_CREDIT_monthly'] / \
      mb_txn['volume_teller_txn_ALL_CREDIT_monthly']
mb_txn['volume_CREDIT_funds_BGN_to_ALL'] = mb_txn[ \
      'volume_funds_txn_BGN_CREDIT_monthly'] / \
      mb_txn['volume_funds_txn_ALL_CREDIT_monthly']
mb_txn['volume_CREDIT_contract_BGN_to_ALL'] = mb_txn[ \
      'volume_contract_txn_BGN_CREDIT_monthly'] / \
      mb_txn['volume_contract_txn_ALL_CREDIT_monthly']

# Calculate value of average monthly transaction (volume/count) per client
mb_txn['average_teller_txn_ALL_monthly'] = mb_txn[ \
      'volume_teller_txn_ALL_monthly'] / \
      mb_txn['count_teller_txn_ALL_monthly']
mb_txn['average_funds_txn_ALL_monthly'] = mb_txn[ \
      'volume_funds_txn_ALL_monthly'] / \
      mb_txn['count_funds_txn_ALL_monthly']
mb_txn['average_contract_txn_ALL_monthly'] = mb_txn[ \
      'volume_contract_txn_ALL_monthly'] / \
      mb_txn['count_contract_txn_ALL_monthly']

# Total transactions volume and count (teller, contract and funds)
mb_txn['total_volume_T_F_C'] = mb_txn['volume_teller_txn_ALL_monthly'] + \
mb_txn['volume_funds_txn_ALL_monthly'] + \
mb_txn['volume_contract_txn_ALL_monthly']
mb_txn['total_count_T_F_C'] = mb_txn['count_teller_txn_ALL_monthly'] + \
mb_txn['count_funds_txn_ALL_monthly'] + \
mb_txn['count_contract_txn_ALL_monthly']

# Funds and contract transactions as part of teller transactions
mb_txn['count_funds_to_teller'] = mb_txn[ \
      'count_funds_txn_ALL_monthly'] / \
      mb_txn['count_teller_txn_ALL_monthly']
mb_txn['count_contract_to_teller'] = mb_txn[ \
      'count_contract_txn_ALL_monthly'] / \
      mb_txn['count_teller_txn_ALL_monthly']

# Some ratios result in NaN values (calculations 0/0). Replace them with zeros
mb_txn=mb_txn.fillna(0)

# Visualization
# Histogram
sns.histplot(mb_txn, x = 'average_teller_txn_ALL_monthly', bins = 10000)
plt.xlim(0, 1000)



######################
# Table Transactions_Rolling_Partial_TRUE (txnpt)
#####################

# Combine with mb
mb_txn_pt = pd.merge(mb, txnpt, on = ['client_id', 'month'])
del txnpt

# Look at the data
mb_txn_pt.describe()
mb_txn_pt.dtypes
mb_txn_pt.isna().sum() # no missing

# Drop columns for 6 and 12 months (work only with 3 months rolling averages)
mb_txn_pt=mb_txn_pt.drop(['sum_volume_teller_txn_all_credit_6months',
       'sum_volume_teller_txn_all_credit_12months',
       'sum_volume_teller_txn_all_debit_6months',
       'sum_volume_teller_txn_all_debit_12months',
       'sum_volume_funds_txn_all_credit_6months',
       'sum_volume_funds_txn_all_credit_12months',
       'sum_volume_funds_txn_all_debit_6months',
       'sum_volume_funds_txn_all_debit_12months',
       'sum_volume_contract_txn_all_credit_6months',
       'sum_volume_contract_txn_all_credit_12months',
       'sum_volume_contract_txn_all_debit_6months',
       'sum_volume_contract_txn_all_debit_12months',
       'sum_volume_teller_txn_bgn_credit_6months',
       'sum_volume_teller_txn_bgn_credit_12months',
       'sum_volume_teller_txn_bgn_debit_6months',
       'sum_volume_teller_txn_bgn_debit_12months',
       'sum_volume_funds_txn_bgn_credit_6months',
       'sum_volume_funds_txn_bgn_credit_12months',
       'sum_volume_funds_txn_bgn_debit_6months',
       'sum_volume_funds_txn_bgn_debit_12months',
       'sum_volume_contract_txn_bgn_credit_6months',
       'sum_volume_contract_txn_bgn_credit_12months',
       'sum_volume_contract_txn_bgn_debit_6months',
       'sum_volume_contract_txn_bgn_debit_12months',
       'sum_volume_teller_txn_foreign_ccy_ALL_CREDIT_6months',
       'sum_volume_teller_txn_foreign_ccy_ALL_CREDIT_12months',
       'sum_volume_teller_txn_foreign_ccy_ALL_DEBIT_6months',
       'sum_volume_teller_txn_foreign_ccy_ALL_DEBIT_12months',
       'sum_volume_funds_txn_foreign_ccy_ALL_CREDIT_6months',
       'sum_volume_funds_txn_foreign_ccy_ALL_CREDIT_12months',
       'sum_volume_funds_txn_foreign_ccy_ALL_DEBIT_6months',
       'sum_volume_funds_txn_foreign_ccy_ALL_DEBIT_12months',
       'sum_volume_contract_txn_foreign_ccy_ALL_CREDIT_6months',
       'sum_volume_contract_txn_foreign_ccy_ALL_CREDIT_12months',
       'sum_volume_contract_txn_foreign_ccy_ALL_DEBIT_6months',
       'sum_volume_contract_txn_foreign_ccy_ALL_DEBIT_12months',
       'sum_count_contract_foreign_ccy_ALL_CREDIT_6months',
       'sum_count_contract_foreign_ccy_ALL_CREDIT_12months',
       'sum_count_contract_foreign_ccy_ALL_DEBIT_6months',
       'sum_count_contract_foreign_ccy_ALL_DEBIT_12months',
       'sum_count_funds_transfer_foreign_ccy_ALL_CREDIT_6months',
       'sum_count_funds_transfer_foreign_ccy_ALL_CREDIT_12months',
       'sum_count_funds_transfer_foreign_ccy_ALL_DEBIT_6months',
       'sum_count_funds_transfer_foreign_ccy_ALL_DEBIT_12months',
       'sum_count_teller_txn_foreign_ccy_ALL_CREDIT_6months',
       'sum_count_teller_txn_foreign_ccy_ALL_CREDIT_12months',
       'sum_count_teller_txn_foreign_ccy_ALL_DEBIT_6months',
       'sum_count_teller_txn_foreign_ccy_ALL_DEBIT_12months',
       'sum_count_teller_txn_BGN_ALL_CREDIT_6months',
       'sum_count_teller_txn_BGN_ALL_CREDIT_12months',
       'sum_count_teller_txn_BGN_ALL_DEBIT_6months',
       'sum_count_teller_txn_BGN_ALL_DEBIT_12months',
       'sum_count_funds_txn_ALL_CREDIT_6months',
       'sum_count_funds_txn_ALL_CREDIT_12months',
       'sum_count_funds_txn_ALL_DEBIT_6months',
       'sum_count_funds_txn_ALL_DEBIT_12months',
       'sum_count_teller_txn_ALL_CREDIT_6months',
       'sum_count_teller_txn_ALL_CREDIT_12months',
       'sum_count_teller_txn_ALL_DEBIT_6months',
       'sum_count_teller_txn_ALL_DEBIT_12months', 
       'sum_count_branch_txn_6months',
       'sum_count_branch_txn_12months',
       'sum_volume_branch_txn_6months', 
       'sum_volume_branch_txn_12months'], axis = 1)
        
        
# # For each type of transaction calculate the total volume and count (D + K)
mb_txn_pt['sum_volume_teller_txn_all_3months'] = mb_txn_pt[ \
         'sum_volume_teller_txn_all_credit_3months'] + \
         mb_txn_pt['sum_volume_teller_txn_all_debit_3months']
mb_txn_pt['sum_volume_funds_txn_all_3months'] = mb_txn_pt[ \
         'sum_volume_funds_txn_all_credit_3months'] + \
         mb_txn_pt['sum_volume_funds_txn_all_debit_3months']
mb_txn_pt['sum_volume_contract_txn_all_3months'] = mb_txn_pt[ \
         'sum_volume_contract_txn_all_credit_3months'] + \
         mb_txn_pt['sum_volume_contract_txn_all_debit_3months']
mb_txn_pt['sum_count_funds_txn_ALL_3months'] = mb_txn_pt[ \
         'sum_count_funds_txn_ALL_CREDIT_3months'] + \
         mb_txn_pt['sum_count_funds_txn_ALL_DEBIT_3months']
mb_txn_pt['sum_count_teller_txn_ALL_3months'] = mb_txn_pt[ \
         'sum_count_teller_txn_ALL_CREDIT_3months'] + \
         mb_txn_pt['sum_count_teller_txn_ALL_DEBIT_3months']

# Debit to credit ratios
mb_txn_pt['volume_funds_debit_to_credit_3months'] = mb_txn_pt[ \
         'sum_volume_funds_txn_all_debit_3months'] / \
         mb_txn_pt['sum_volume_funds_txn_all_credit_3months']
mb_txn_pt['volume_teller_debit_to_credit_3months'] = mb_txn_pt[ \
         'sum_volume_teller_txn_all_debit_3months'] / \
         mb_txn_pt['sum_volume_teller_txn_all_credit_3months']
mb_txn_pt['volume_contract_debit_to_credit_3months'] = mb_txn_pt[ \
         'sum_volume_contract_txn_all_debit_3months'] / \
         mb_txn_pt['sum_volume_contract_txn_all_credit_3months']
mb_txn_pt['count_funds_debit_to_credit_3months'] = mb_txn_pt[ \
         'sum_count_funds_txn_ALL_DEBIT_3months'] / \
         mb_txn_pt['sum_count_funds_txn_ALL_CREDIT_3months']
mb_txn_pt['count_teller_debit_to_credit_3months'] = mb_txn_pt[ \
         'sum_count_teller_txn_ALL_DEBIT_3months'] / \
         mb_txn_pt['sum_count_teller_txn_ALL_CREDIT_3months']
mb_txn_pt = mb_txn_pt.fillna(0)


# Merge transaction tables and save
mb_transactions = pd.merge(mb_txn, mb_txn_pt, on = ['client_id', 'month', 
                                                    'FLAG_new_active_user_MB'])
mb_transactions.to_csv(r'C:\Users\user\Desktop\UNI\MASTER\First Semester'
                    r'\Business Intelligence\Case Study Raiffeisen'
                    r'\Raiffeisen files\mb_txn.csv', index = False)


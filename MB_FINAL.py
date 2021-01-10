# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:13:46 2020

@author: user
"""
# FINAL TABLE

# Raiffeisen Bank Case Study
# Path-2-Digital
# MOBILE BANKING

# Load libraries
import pandas as pd

# Set output display options
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.6f}'.format

# Load datasets
mb_prod = pd.read_csv("mb_prod.csv")
mb_AB = pd.read_csv("mb_ab.csv")
mb_txn = pd.read_csv("mb_txn.csv")
mb_demog = pd.read_csv("mb_demog.csv")


# Combine all tables into one final table
final = pd.merge(mb_demog, mb_prod, on = ['client_id', 'month', 
                                          'FLAG_new_active_user_mb'])
final = pd.merge(final, mb_AB, on = ['client_id', 'month', 
                                     'FLAG_new_active_user_mb'])
final = pd.merge(final, mb_txn, on = ['client_id', 'month', 
                                      'FLAG_new_active_user_mb'])
final.to_csv(r'C:\Users\user\Desktop\UNI\MASTER\First Semester'
                    r'\Business Intelligence\Case Study Raiffeisen'
                    r'\Raiffeisen files\MB_FINAL.csv', index=False)
